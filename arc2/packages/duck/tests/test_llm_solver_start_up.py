"""``LLMAgentSolver`` under TAAF's start-up step: the engine starts, is gated
and is served in ``start_up()``, once, before the solver plays; a start-up
whose gate fails closes the engine and is a solver error like any other; and
a solver whose ``start_up()`` never ran still starts the engine inside
``run_games``. On a stub backend: what is under test is the order, not the
engine."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.factory
from duck.agent import DuckAgent

import taaf.benchmark
from platypus import LLMAgentSolver, LLMConfig
from platypus.solver import AgentSolver
from taaf.game_examples import flip_game

#: The run's engine and play events in order; reset by each test.
EVENTS: list[str] = []


@pytest.fixture(autouse=True)
def logged_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every factory start and close, the solver's start-up returning, and
    the agent's play go on ``EVENTS``."""
    EVENTS.clear()
    start = ullm.factory.BackendFactory.start
    close = ullm.factory.BackendFactory.close
    start_up = LLMAgentSolver._start_up

    async def logged_start(self: ullm.factory.BackendFactory) -> ullm.backend.Backend:
        EVENTS.append("engine start")
        return await start(self)

    async def logged_close(self: ullm.factory.BackendFactory) -> None:
        if self.started:
            EVENTS.append("engine close")
        await close(self)

    async def logged_start_up(self: LLMAgentSolver) -> None:
        await start_up(self)
        EVENTS.append("up")

    async def play(self: AgentSolver, games: Any) -> None:
        EVENTS.append("play")
        for game in games:
            game.finish_game()

    monkeypatch.setattr(ullm.factory.BackendFactory, "start", logged_start)
    monkeypatch.setattr(ullm.factory.BackendFactory, "close", logged_close)
    monkeypatch.setattr(LLMAgentSolver, "_start_up", logged_start_up)
    monkeypatch.setattr(AgentSolver, "_run_games", play)


def _solver(*, traces: bool = False) -> LLMAgentSolver:
    """A duck on a stub engine; ``traces`` asks for a gate the stub cannot pass."""
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        context_window_tokens=32768,
        compact_to_tokens=8192,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.StubBackend(), close_backend=True),
            traces=traces,
            verify_backend=False,
        ),
    )
    return LLMAgentSolver(agent=agent)


async def test_the_engine_is_up_once_before_the_solver_plays() -> None:
    benchmark = taaf.benchmark.Benchmark(games=[flip_game("g")], solver=_solver(), n_passes=1)
    await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["engine start", "up", "play", "engine close"]


async def test_a_start_up_whose_gate_fails_closes_the_engine_and_is_a_solver_error() -> None:
    """The trace gate refuses the stub: the started engine is closed, the game
    is never played, and the run raises only under ``raise_solver_errors``."""
    benchmark = taaf.benchmark.Benchmark(games=[flip_game("g")], solver=_solver(traces=True), n_passes=1)
    await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["engine start", "engine close"]
    assert [run.state for run in benchmark.game_runs] == ["crashed"]

    EVENTS.clear()
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game("g")], solver=_solver(traces=True), n_passes=1, raise_solver_errors=True
    )
    with pytest.raises(RuntimeError, match="cannot produce R5-grade traces"):
        await asyncio.wait_for(benchmark.run(), timeout=120)
    assert EVENTS == ["engine start", "engine close"]


async def test_without_start_up_run_games_starts_the_engine_itself() -> None:
    solver = _solver()
    solver.setup()
    await solver.run_games([])
    assert EVENTS == ["engine start", "play", "engine close"]
