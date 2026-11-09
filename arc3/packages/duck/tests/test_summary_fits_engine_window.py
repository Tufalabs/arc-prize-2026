"""A summary request never asks the engine for more than its window holds.

The summary's cap -- twice its target, six times with thinking -- is lowered on
each request to what the engine window has left over the request's measured
prompt, so a configuration only has to leave the summary
``MIN_SUMMARY_ROOM_TOKENS`` in the worst case it owns (2026-09-23).
Played through a scripted stub that records the summary request's
``max_tokens``, with an engine ceiling tight enough that the clamp binds; and
the ceiling a supplied engine settles when it connects.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import arcengine
import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

import platypus.diagnostics
from duck import DuckAgent
from duck.agent import MIN_SUMMARY_ROOM_TOKENS
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.solver import AgentSolver

WINDOW = 40_000
REPLY = 2_000
SUMMARY_TEXT = "SUMMARY: UP advances; keep pressing UP."

#: What the stub saw, for the test to read back: the process runs in-process.
SEEN: dict[str, list[int | None]] = {"summary_max_tokens": []}


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running one snippet."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})]
        )
    )


@dataclass
class _RecordingSummaryStub(ullm.backends.stub.StubBackend):
    """Plays the script, and answers a summary request with a summary after
    noting the ``max_tokens`` it was sent with."""

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        users = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        last = users[-1].content.to_string() if users else ""
        if last.startswith("Your context window is nearly full"):
            SEEN["summary_max_tokens"].append(sampling_params.max_tokens)
            return ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=SUMMARY_TEXT))
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        if llm_output.final_content.to_string() == SUMMARY_TEXT:
            # Not a scripted response: fill it without advancing the script.
            self._fill_into(llm_output, element, conversation)
            return
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


def _agent(ceiling: int | None, script: list[ullm.conv.LLMResponse]) -> DuckAgent:
    return DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=_RecordingSummaryStub(responses=script)),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        context_window_tokens=WINDOW,
        reply_cap_tokens=REPLY,
        engine_ceiling_tokens=ceiling,
        compaction_strategy="summarize",
        summary_thinking=True,
        max_tokens_per_level=None,
    )


async def test_the_summary_request_is_capped_at_what_the_engine_window_has_left(tmp_path: Path) -> None:
    SEEN["summary_max_tokens"].clear()
    # The tightest ceiling the configuration may state: the window, the reply
    # reserve and the room the summary is owed.
    ceiling = WINDOW + REPLY + MIN_SUMMARY_ROOM_TOKENS
    script = [_python(f"print('look {i} ' * 1500)\nsubmit_action(['UP'])\n") for i in range(16)]
    agent = _agent(ceiling, script)
    thinking_cap = 6 * (WINDOW // agent.compaction_ratio)
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 16], label="summary_cap")],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=tmp_path,
    )
    await benchmark.run()

    (run,) = benchmark.game_runs
    assert run.state == "won", run.state
    assert SEEN["summary_max_tokens"], "no summary fired: the window is too wide for the script"
    # Every summary's record carries the installed text's own length.
    raw = platypus.diagnostics.read_job_file(tmp_path, f"games/{run.run_id}/spans.jsonl")
    assert raw is not None
    records = [json.loads(line)["values"] for line in raw.decode().splitlines() if '"compacted":true' in line]
    assert records and all(record["summary_text_chars"] == len(SUMMARY_TEXT) for record in records), records
    # A summary fires once the prompt passes the budget, so every request's
    # prompt is at least that, and its cap at most what the ceiling leaves.
    budget = agent.context_budget_tokens()
    for sent in SEEN["summary_max_tokens"]:
        assert sent is not None
        assert sent <= ceiling - budget < thinking_cap, (sent, ceiling - budget, thinking_cap)


def test_a_configuration_that_leaves_the_summary_too_little_room_is_refused() -> None:
    with pytest.raises(ValueError, match="leave less than"):
        _agent(WINDOW + REPLY + MIN_SUMMARY_ROOM_TOKENS - 1, [])._derived_engine_window()
    assert _agent(WINDOW + REPLY + MIN_SUMMARY_ROOM_TOKENS, [])._derived_engine_window() == (
        WINDOW + REPLY + MIN_SUMMARY_ROOM_TOKENS
    )


@dataclass
class _ConnectedEngineBackend(ullm.backends.stub.StubBackend):
    """A supplied backend over an engine that reports its window once connected."""

    window: int = 60_000
    engine: Any = field(default_factory=lambda: SimpleNamespace(model_config=None))

    async def probe_continuation(self) -> None:
        self.engine.model_config = SimpleNamespace(max_model_len=self.window)


@pytest.mark.parametrize(("stated", "settled"), [(None, 60_000), (262_144, 60_000), (55_000, 55_000)])
async def test_a_supplied_engine_settles_the_ceiling_the_summary_cap_follows(
    monkeypatch: pytest.MonkeyPatch, stated: int | None, settled: int
) -> None:
    """A supplied 60,000-token engine: a ceiling the agent does not state
    becomes the engine's, a stated one wider than the engine is cut to it, and
    a narrower one stands. Each game's copy of the agent carries it, so a
    summary with a 54,000-token prompt asks for what that ceiling leaves."""
    games: list[DuckAgent] = []

    async def run_games(self: AgentSolver, _games: Any) -> None:
        games.append(cast(DuckAgent, copy.deepcopy(self.agent)))  # as each (game, pass) copy is taken

    monkeypatch.setattr(AgentSolver, "_run_games", run_games)
    agent = _agent(stated, [])
    agent.llm_config.backend_factory = ullm.factory.PrebuiltBackend(backend=_ConnectedEngineBackend())
    await LLMAgentSolver(agent=agent)._run_games([])

    (game,) = games
    assert game.engine_ceiling_tokens == settled
    assert game._summary_completion_cap(10_000, 54_000) == settled - 54_000
