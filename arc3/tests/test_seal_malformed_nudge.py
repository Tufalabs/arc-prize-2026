"""``malformed_nudges``: a play reply refused on every re-sample is dropped and nudged, not crashed.

ullm re-samples a reply with no text and no tool call up to ``malformed_retries``
times and then raises, which ends the game. In practice such replies loop inside
their reasoning, and a context that loops four times in a row loops on. With
nudges left, the seal drops the refused draws as the retries did, keeps the
turn's completed rounds, and sends a nudge that says the reply was discarded --
the model never sees it. Played in a real game through a scripted stub on ullm's
own retry path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.prompts import SealPrompts

LOOP = "Let me try to think about the rules differently.\n\nOK.\n"
RETRIES = 3


def _looping() -> ullm.conv.LLMResponse:
    """A reply that went in circles inside its reasoning and stopped there."""
    return ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=LOOP * 20),
        final_content=ullm.conv.EmptyContent(),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=40),
    )


def _text(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=3),
    )


def _python(code: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})]
        ),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


class _Script(ullm.backends.stub.StubBackend):
    """Scripted replies, empty ones allowed. Records, per request, the last
    user message and the conversation's text; class-level, since the game
    plays on a copy of the backend the test built."""

    last_user: ClassVar[list[str]] = []
    texts: ClassVar[list[str]] = []

    def _check_constraints(self) -> None:
        """Allow empty script entries to reach the runtime rejection policy."""
        ullm.backend.Backend._check_constraints(self)

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        """Record what the request carries, then pick the scripted reply."""
        users = [e.content.to_string() for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        type(self).last_user.append(users[-1] if users else "")
        type(self).texts.append(conversation.to_string())
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


async def _play(tmp_path: Path, replies: list[ullm.conv.LLMResponse], *, nudges: int) -> Any:
    """One game on the scripted replies; returns its run. Refused draws are not
    billed, so the game budget ends it only once replies are accepted."""
    _Script.last_user = []
    _Script.texts = []
    backend = _Script(responses=replies, on_exhausted="repeat_last")
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        max_tokens_per_game=20,
        max_tokens_per_level=None,
        uncap_when_queue_drained=False,
        malformed_retries=RETRIES,
        malformed_nudges=nudges,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    (run,) = benchmark.game_runs
    return run


NUDGE = SealPrompts().malformed_nudge()


async def test_a_reply_refused_on_every_retry_is_dropped_and_nudged(tmp_path: Path) -> None:
    run = await _play(tmp_path, [_looping()] * (RETRIES + 1) + [_text("Done.")], nudges=1)

    # The refused draws, then the request the nudge closes; the game goes on
    # past it and ends on its budget, not on the refusals.
    assert run.state == "gave_up"
    assert _Script.last_user[RETRIES + 1] == NUDGE
    assert not any(LOOP in text for text in _Script.texts), "a refused draw reached a request"


async def test_completed_rounds_of_the_turn_stay(tmp_path: Path) -> None:
    replies = [_python("print('a round')"), *[_looping()] * (RETRIES + 1), _text("Done.")]
    run = await _play(tmp_path, replies, nudges=1)

    assert run.state == "gave_up"
    nudged = 1 + RETRIES + 1
    assert _Script.last_user[nudged] == NUDGE
    # The round before the refusals ran, and its result is still in the
    # request the nudge closes.
    assert "a round" in _Script.texts[nudged]
    assert not any(LOOP in text for text in _Script.texts)


@pytest.mark.parametrize("nudges", [0, 1])
async def test_past_the_count_the_refusal_crashes_the_game(tmp_path: Path, nudges: int) -> None:
    draws = (RETRIES + 1) * (nudges + 1)
    run = await _play(tmp_path, [_looping()] * draws + [_text("unused")], nudges=nudges)

    assert run.state == "crashed"
    assert len(_Script.last_user) == draws
    assert NUDGE not in _Script.last_user[: RETRIES + 1]


def test_only_the_refusals_the_nudge_describes_are_nudged() -> None:
    agent = SealAgent(malformed_nudges=1)
    empty = ullm.backend.MalformedOutputError("Model produced no final content (no text response, no tool call).")
    assert agent._nudges_past(empty)  # pyright: ignore[reportPrivateUsage]
    assert agent._nudges_past(ullm.backend.DegenerateOutputError("repetition"))  # pyright: ignore[reportPrivateUsage]
    unparsed = ullm.backend.MalformedOutputError("could not parse the tool call")
    assert not agent._nudges_past(unparsed), "a parser failure still crashes"  # pyright: ignore[reportPrivateUsage]
    agent._malformed_nudged = 1  # pyright: ignore[reportPrivateUsage]
    assert not agent._nudges_past(empty), "past the count"  # pyright: ignore[reportPrivateUsage]


def test_a_retained_draw_cannot_be_nudged() -> None:
    agent = SealAgent(preserve_failed_draws=True, malformed_retries=0, malformed_nudges=1)
    with pytest.raises(ValueError, match="malformed_nudges=0"):
        agent.prepare_for_backend()


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="malformed_nudges"):
        SealAgent(malformed_nudges=-1).prepare_for_backend()
