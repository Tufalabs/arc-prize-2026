"""``LLMAgentSolver`` under TAAF's ``startup_stealth`` (TAAF R2.66): the
engine starts, is gated and is served before any game is opened, once; a
start-up whose gate fails tears the engine down before the run raises; and
without the flag the engine starts inside ``run_games``, after the games open,
as it always has. On a stub backend: what is under test is the order, not the
engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent

ALLOWANCE = 100


class LoggedGame(ExampleGame):
    """An example game that says when it is opened."""

    def _start_game(self, session: Any) -> Any:
        EVENTS.append("open")
        return super()._start_game(session)


#: The run's engine and game events in order; reset by each test.
EVENTS: list[str] = []


@pytest.fixture(autouse=True)
def logged_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every factory start and close goes on ``EVENTS``."""
    EVENTS.clear()
    start = ullm.factory.BackendFactory.start
    close = ullm.factory.BackendFactory.close

    async def logged_start(self: ullm.factory.BackendFactory) -> ullm.backend.Backend:
        EVENTS.append("engine start")
        return await start(self)

    async def logged_close(self: ullm.factory.BackendFactory) -> None:
        if self.started:
            EVENTS.append("engine close")
        await close(self)

    monkeypatch.setattr(ullm.factory.BackendFactory, "start", logged_start)
    monkeypatch.setattr(ullm.factory.BackendFactory, "close", logged_close)


def _benchmark(tmp_path: Path, *, stealth: bool, traces: bool = False) -> taaf.benchmark.Benchmark:
    """One game of a seal that gives up on its first level's budget, served by
    a stub; ``traces`` asks for a gate the stub cannot pass."""
    spent = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="World model: thinking about it"),
        stop_reason="max_tokens",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=ALLOWANCE),
    )
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        use_world_model="none",
        level_reflection="none",
        compaction_strategy="prune",
        max_tokens_per_level=(ALLOWANCE,),
        max_llm_calls_per_move=0,
        uncap_when_queue_drained=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=[spent], on_exhausted="loop")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
            traces=traces,
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    return taaf.benchmark.Benchmark(
        games=[LoggedGame()],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=tmp_path,
        startup_stealth=stealth,
    )


async def test_under_startup_stealth_the_engine_is_up_before_any_game_opens(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, stealth=True)
    await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["engine start", "open", "engine close"]
    (run,) = benchmark.game_runs
    assert run.state == "gave_up"


async def test_without_startup_stealth_the_games_open_before_the_engine_starts(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, stealth=False)
    await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["open", "engine start", "engine close"]


async def test_a_start_up_whose_gate_fails_closes_the_engine_and_opens_no_game(tmp_path: Path) -> None:
    """The trace gate refuses the stub: the started engine is closed before
    the run raises, and no game was opened."""
    benchmark = _benchmark(tmp_path, stealth=True, traces=True)
    with pytest.raises(RuntimeError, match="cannot produce R5-grade traces"):
        await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["engine start", "engine close"]
    assert benchmark.game_runs == []
