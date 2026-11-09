"""on_agent_error policies (I12-part), contract validation, and
cancellation semantics (I5)."""

from __future__ import annotations

import asyncio

import pytest
from infra_fixtures import BadReturnAgent, CrashingAgent, ProbeAgent, SlowAgent, UnavailableActionAgent
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig


async def test_agent_crash_policy_crash_game() -> None:
    ProbeAgent.reset_log()
    agent = CrashingAgent(script=("ACTION1",))  # default policy: crash_game
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "deliberate agent crash" in run.solver_note
    # on_finish still ran, best-effort, with the crashed result (I5).
    (_, result, history) = ProbeAgent.finishes[0]
    assert result.state == "crashed"
    assert len(history) == 1


async def test_agent_crash_policy_raise() -> None:
    ProbeAgent.reset_log()
    agent = CrashingAgent(script=(), run_config=RunConfig(on_agent_error="raise", process_mode="inprocess"))
    solver = AgentSolver(agent=agent)
    solver.setup()
    games: list[Game] = [ExampleGame(label=f"game{i}") for i in range(2)]
    for game in games:
        game.start_game()
    # The original exception propagates out of run_games — a debugger /
    # pytest sees the real raise site, and the whole run is sacrificed.
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games(games)


async def test_crash_in_one_game_does_not_stop_others() -> None:
    ProbeAgent.reset_log()
    agent = CrashingAgent(script=())
    games: list[Game] = [ExampleGame(label="crasher"), ExampleGame(label="bystander")]
    # The bystander can't win with an empty script either — but it must
    # still get its own crash, not a cancellation from the crasher.
    benchmark = Benchmark(games=games, solver=AgentSolver(agent=agent))
    await benchmark.run()
    assert {run.game_id: run.state for run in benchmark.game_runs} == {
        "crasher": "crashed",
        "bystander": "crashed",
    }
    assert len(ProbeAgent.finishes) == 2


@pytest.mark.parametrize("agent_type", [BadReturnAgent, UnavailableActionAgent])
async def test_contract_violations_hit_error_policy(agent_type: type[ProbeAgent]) -> None:
    ProbeAgent.reset_log()
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent_type()))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "ContractViolationError" in run.solver_note


async def test_cancellation_lands_mid_move_and_on_finish_runs() -> None:
    ProbeAgent.reset_log()
    solver = AgentSolver(agent=SlowAgent())
    solver.setup()
    games: list[Game] = [ExampleGame(label=f"game{i}") for i in range(2)]
    for game in games:
        game.start_game()
    task = asyncio.create_task(solver.run_games(games))
    # Both games in their move: starts are paced (``RunConfig.start_gap_s``),
    # so a fixed number of loop ticks no longer gets the second one there.
    while len(ProbeAgent.seen) < len(games):
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for game in games:
        assert game.game_run is not None
        assert game.game_run.state == "cancelled"
    # on_finish ran shielded for every game, seeing the cancelled result.
    assert sorted(game_id for game_id, _, _ in ProbeAgent.finishes) == ["game0", "game1"]
    assert all(result.state == "cancelled" for _, result, _ in ProbeAgent.finishes)


async def test_checkpointing_config_validated_at_setup() -> None:
    agent = ProbeAgent(run_config=RunConfig(checkpointing="move"))
    solver = AgentSolver(agent=agent)
    with pytest.raises(NotImplementedError):
        solver.setup()
