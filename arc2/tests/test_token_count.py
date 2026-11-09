"""``Backend.token_count`` for its remaining consumers, and the duck's
window invariant end-to-end.

The duck's compaction decision calls ``token_count`` only in token-native
mode (every vLLM preset), where the answer is the id-ledger's exact
weight; elsewhere it runs on the engine's billed usage plus a
conservative bound (see ``packages/duck/tests/test_context_accounting.py``
for the unit tier). The count API also stays on the backend for other
consumers, and its policy-awareness contract is pinned here. The tier-1
tests run the duck through the real solver so the invariant is proven
live, not merely correct in isolation: an engine billing within the window
runs clean, and an engine billing past it fails the game loudly instead of
handing an oversized sequence downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
from duck import DuckAgent


@dataclass
class _CountingStub(ullm.backends.stub.StubBackend):
    """Stub that counts exactly: one token per character of rendered text,
    plus ``reasoning_chars`` per retained assistant reasoning block, so a
    test can show the count moving with ``send_back_reasoning`` — the
    property that motivates counting backend-side at all."""

    reasoning_chars: int = 0
    counted: list[int] = field(default_factory=lambda: list[int]())

    async def _token_count(self, sampling_params: Any, conversation: ullm.conv.Conversation) -> int:
        send_back = self.resolve_send_back_reasoning(sampling_params)
        total = len(conversation.system_prompt.content.to_string()) if conversation.system_prompt else 0
        turn = conversation.last_turn
        for element in conversation.elements:
            content = getattr(element, "content", None)
            if content is not None:
                total += len(content.to_string())
            if isinstance(element, ullm.conv.ResponseTurn):
                retained = send_back == "all" or (send_back == "current_turn" and element is turn)
                if retained:
                    total += self.reasoning_chars * sum(
                        1 for e in element.elements if isinstance(e, ullm.conv.LLMResponse)
                    )
        self.counted.append(total)
        return total


async def test_count_follows_the_reasoning_passback_policy() -> None:
    """Whether past reasoning is re-sent changes the input size, and the duck
    defaults to sending it all. A count blind to that policy would be wrong
    by exactly the retained reasoning — the reason this lives on the backend
    next to ``_prepare_input`` rather than on the conversation object."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys")),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hello")),
            ullm.conv.ResponseTurn(elements=[ullm.conv.LLMResponse()]),
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="again")),
            ullm.conv.ResponseTurn(elements=[ullm.conv.LLMResponse()]),
        ],
        status="intermediate",
    )
    backend = _CountingStub(reasoning_chars=100)
    all_back = await backend.token_count(
        ullm.backend.GenericSamplingParameters(send_back_reasoning="all"), conversation
    )
    none_back = await backend.token_count(
        ullm.backend.GenericSamplingParameters(send_back_reasoning="none"), conversation
    )
    current_only = await backend.token_count(
        ullm.backend.GenericSamplingParameters(send_back_reasoning="current_turn"), conversation
    )
    assert none_back is not None  # a backend that cannot count fails this test, loudly
    assert all_back == none_back + 200  # two retained turns
    assert current_only == none_back + 100  # only the last turn retained


# ---------------------------------------------------------------------------
# Tier 1: end-to-end through the real solver, so the check runs where it
# runs in production — after a turn the manager assembled, on usage the
# backend stamped. A green unit suite over a dead check is exactly what
# these catch.
# ---------------------------------------------------------------------------


@dataclass
class _BillingStub(ullm.backends.stub.StubBackend):
    """Stub whose responses carry engine-shaped usage: what a vLLM server
    reports, at sizes the test chooses."""

    bill_prompt: int = 0
    bill_completion: int = 0

    def _process_output(self, sampling_params, llm_output, conversation, turn, element) -> None:  # type: ignore[no-untyped-def]
        """Stamp the configured bill onto the response usage."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        assert element.token_usage is not None
        element.token_usage.prompt_tokens = self.bill_prompt
        element.token_usage.completion_tokens = self.bill_completion
        element.token_usage.cached_tokens = 0


def _answering_script() -> list[ullm.conv.LLMResponse]:
    """One kernel call that writes the answer, then a plain reply: the
    shortest complete game."""
    call = ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.ToolCallContent(
                    tool_name="python", arguments={"code": "test_out[0][0] = np.fliplr(test_in[0])\n"}
                )
            ]
        )
    )
    done = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="Done."))
    return [call, done]


async def _run_duck(backend: ullm.backend.Backend, *, context_window_tokens: int) -> Any:
    import ullm.factory as _factory

    import taaf.benchmark
    from platypus import LLMAgentSolver, LLMConfig, RunConfig
    from taaf.game_examples import flip_game

    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=64,
        context_window_tokens=context_window_tokens,
        compact_to_tokens=2048,
        llm_config=LLMConfig(
            backend_factory=_factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=99, on_agent_error="crash_game", process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game("tc_game")],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
    )
    await benchmark.run()
    (run,) = benchmark.game_runs
    return run


async def test_engine_billing_within_the_window_runs_clean() -> None:
    """The positive half: honest, in-window usage anchors the decision path
    and trips nothing."""
    backend = _BillingStub(responses=_answering_script(), on_exhausted="loop", bill_prompt=5000, bill_completion=10)
    run = await _run_duck(backend, context_window_tokens=32768)
    assert run.state != "crashed", f"in-window billing must not trip the invariant: {run.solver_note}"


async def test_an_over_window_bill_fails_the_game_loudly() -> None:
    """The invariant, live: an engine account of prompt+completion above
    the window is an oversized sequence in the making. It must fail the ONE
    game at the moment of the overrun, with the cause in the note, rather
    than hand the history downstream."""
    backend = _BillingStub(responses=_answering_script(), on_exhausted="loop", bill_prompt=29000, bill_completion=10)
    run = await _run_duck(backend, context_window_tokens=28672)
    assert run.state == "crashed"
    assert "context window invariant violated" in str(run.solver_note)


if __name__ == "__main__":
    pytest.main([__file__])
