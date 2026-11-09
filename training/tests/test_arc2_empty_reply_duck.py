"""The pinned ARC-2 Duck after a retained empty reply, played with the training
worker's settings (harness_support/arc2/worker.py): every draw is kept
(``accept_malformed_output``, ``malformed_retries=0``,
``preserve_failed_draws``), the Duck's malformed-reply nudge is off
(``malformed_nudges=0``), tool calls are classified, and the processed-response
observer journals each draw.

An empty reply reaches the Duck as an engine returns it, a natural stop with
no text and no tool call; ullm's backend keeps it as malformed. With
``terminate_on_empty_reply`` off (the Flash recipe since 2026-10-10) it is a
journaled step and the game goes on: ullm's turn manager feeds back "Malformed
reply: no text and no tool call." and the Duck's analyzer entry follows that
with its ordinary nudge, so the next request extends the last one rather than
repeating it. ``malformed_nudges=0`` does not suppress this: that nudge is for
replies the backend refuses, and training refuses none. On, the first empty
reply gives the game up. Each empty reply is one malformed call, charged
``training.malformed_call_penalty``, and the episode at most
``training.max_episode_penalty`` in all.

Runs where the ARC-2 Duck is installed (the ARC-2 worker environment; the
harness-environment test runs on kander use it); CI has no Duck.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, ClassVar

import pytest

duck_agent = pytest.importorskip("duck.agent")
if not hasattr(duck_agent.DuckAgent, "terminate_on_empty_reply") or not hasattr(
    duck_agent.DuckAgent, "_uses_context_summary"
):
    pytest.skip("requires the ARC-2 Duck environment", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import taaf.benchmark  # noqa: E402 - requires the ARC-2 environment
import ullm.backend  # noqa: E402
import ullm.backends.stub  # noqa: E402
import ullm.conv  # noqa: E402
import ullm.factory  # noqa: E402
from platypus import LLMAgentSolver, LLMConfig, RunConfig  # noqa: E402
from taaf.game_examples import flip_game, flip_task  # noqa: E402

from harness_support.arc2.contract import ProcessedResponseObserver  # noqa: E402
from harness_support.arc2.results import enrich_arc2_result  # noqa: E402

DRAW_TOKENS = 100
FEEDBACK = "Malformed reply: no text and no tool call."
GAME = "emptyreply"


def _empty() -> ullm.conv.LLMResponse:
    """What the engine returns for a reply that ends at once. Empty content,
    not empty text: the stub then serves the Duck's prompt injection as the
    reasoning, where a thinking template puts it."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.EmptyContent(),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=50, completion_tokens=DRAW_TOKENS, num_requests=1
        ),
    )


def _solve() -> ullm.conv.LLMResponse:
    """A python call that writes the flip task's answer."""
    code = "import numpy as np\ntest_out[0][0] = np.fliplr(np.array(test_in[0]))\n"
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})]
        ),
        stop_reason="tool_use",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5, num_requests=1),
    )


def _done() -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="Done."),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=3, num_requests=1),
    )


def _transcript(conversation: ullm.conv.Conversation) -> list[str]:
    """A request after its system prompt: each user message's text, each reply
    as ``<reply STOP_REASON>``."""
    out = []
    for element in conversation.elements:
        parts = element.elements if isinstance(element, ullm.conv.ResponseTurn) else [element]
        for part in parts:
            if isinstance(part, ullm.conv.LLMResponse):
                out.append(f"<reply {part.stop_reason}>")
            elif isinstance(part, ullm.conv.UserPrompt):
                out.append(part.content.to_string())
            elif isinstance(part, ullm.conv.ToolResult):
                out.append("<tool result>")
    return out


class _Engine(ullm.backends.stub.StubBackend):
    """Scripted draws as an engine returns them (empty ones allowed), stamped
    with the run#nonce identity the journal keys on; records every request."""

    requests: ClassVar[list[list[str]]] = []

    def _check_constraints(self) -> None:
        ullm.backend.Backend._check_constraints(self)

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        type(self).requests.append(_transcript(conversation))
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _process_output(self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        element.generation_request_id = f"run0#draw{self._call_index}"


def _play(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: list[Any], *, terminate: bool):
    """One flip game on SCRIPT; the run and its processed-response journal."""
    _Engine.requests = []
    import ullm.tools.python

    if not ullm.tools.python._unshare_usable():
        # A development host without user namespaces (macOS): the game's
        # kernel runs without network isolation, which this test never uses.
        ensure_repl = duck_agent.DuckAgent._ensure_repl

        def unisolated(self, ctx=None):
            repl = ensure_repl(self, ctx)
            repl.isolate_network = False
            return repl

        monkeypatch.setattr(duck_agent.DuckAgent, "_ensure_repl", unisolated)
    journal_dir = tmp_path / "processed_responses"
    monkeypatch.setenv("ARC2_TRAINING_RESPONSE_DIR", str(journal_dir))
    backend = _Engine(responses=script, on_exhausted="repeat_last", accept_malformed_output=True)
    agent = duck_agent.DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        # Ends a game that only ever replies empty after three draws.
        max_tokens_per_game=3 * DRAW_TOKENS,
        images="off",
        show_changed_answers=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
        # The training worker's retention settings.
        malformed_retries=0,
        perturb_retry_prompt=False,
        preserve_failed_draws=True,
        malformed_nudges=0,
        classify_tool_calls=True,
        response_observer=ProcessedResponseObserver(),
        terminate_on_empty_reply=terminate,
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game(GAME)], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    asyncio.run(asyncio.wait_for(benchmark.run(), timeout=120))
    (run,) = benchmark.game_runs
    path = journal_dir / "run0.jsonl"
    journal = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return run, journal


def _scored(run: Any, journal: list[dict], *, penalty: float, cap: float | None = None) -> dict:
    """The worker's scoring of the run (harness_support/arc2/results.py)."""
    report = {"state": run.state, "steps": [dict(record) for record in journal]}
    return enrich_arc2_result(
        report,
        benchmark_run=run.to_json_dict(),
        test_output_count=len(flip_task(GAME).to_answer),
        malformed_call_penalty=penalty,
        max_episode_penalty=cap,
    )


def _empty_record(record: dict) -> bool:
    """The journal of an empty reply as ullm and the Duck leave it."""
    return record == {
        "nonce": record["nonce"],
        "malformed_call_count": 1,
        "truncation_cap_source": None,
        "stop_reason": "malformed",
        "has_malformed_error": True,
        "has_tool_calls": False,
        "empty_reply": True,
    }


def test_off_every_empty_reply_is_a_step_and_the_game_goes_on(tmp_path, monkeypatch):
    run, journal = _play(tmp_path, monkeypatch, [_empty()], terminate=False)
    # The game spent its whole budget on empty replies: three draws, each
    # journaled as an empty reply, then the budget gave it up.
    assert run.state == "gave_up"
    assert len(_Engine.requests) == len(journal) == 3
    assert all(_empty_record(record) for record in journal)
    # Each request extends the one before with the empty reply, ullm's
    # feedback and the Duck's nudge: never the same turn again.
    for before, after in zip(_Engine.requests, _Engine.requests[1:]):
        assert after[: len(before)] == before
        reply, feedback, nudge = after[len(before):]
        assert reply == "<reply malformed>"
        assert feedback.startswith(FEEDBACK)
        assert nudge and FEEDBACK not in nudge
    # Each is one malformed call, and counted as an empty reply.
    scored = _scored(run, journal, penalty=0.2)
    assert scored["arc2_outcome"]["malformed_call_count"] == 3
    assert scored["arc2_outcome"]["empty_reply_count"] == 3
    assert scored["arc2_outcome"]["penalty_capped"] is False
    assert scored["terminal_reward"] == pytest.approx(0.0 - 3 * 0.2)
    # Under a cap below the three charges the episode pays the cap.
    capped = _scored(run, journal, penalty=0.2, cap=0.5)
    assert capped["arc2_outcome"]["malformed_penalty"] == pytest.approx(3 * 0.2)
    assert capped["arc2_outcome"]["penalty_capped"] is True
    assert capped["terminal_reward"] == pytest.approx(-0.5)


def test_off_a_model_that_recovers_after_an_empty_reply_still_wins(tmp_path, monkeypatch):
    run, journal = _play(tmp_path, monkeypatch, [_empty(), _solve(), _done()], terminate=False)
    assert run.state == "won"
    assert _empty_record(journal[0])
    assert not any(record["empty_reply"] for record in journal[1:])
    second = _Engine.requests[1]
    assert second[-3] == "<reply malformed>" and second[-2].startswith(FEEDBACK)
    scored = _scored(run, journal, penalty=0.2)
    assert scored["arc2_outcome"]["correctness"] == 1.0
    assert scored["arc2_outcome"]["malformed_call_count"] == 1
    assert scored["arc2_outcome"]["empty_reply_count"] == 1
    assert scored["terminal_reward"] == pytest.approx(1.0 - 0.2)


def test_on_the_first_empty_reply_gives_the_game_up(tmp_path, monkeypatch):
    run, journal = _play(tmp_path, monkeypatch, [_empty(), _solve(), _done()], terminate=True)
    assert run.state == "gave_up"
    assert len(_Engine.requests) == len(journal) == 1
    assert _empty_record(journal[0])
    scored = _scored(run, journal, penalty=0.2)
    assert scored["terminal_reward"] == pytest.approx(0.0 - 0.2)
