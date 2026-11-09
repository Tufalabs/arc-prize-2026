"""RunConfig.max_runtime_s_per_game: the infra-enforced per-game wall
budget (scored as-is via gave_up; agents still never read clocks)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import arcengine
import taaf.benchmark
from taaf.game_examples import ExampleGame

from platypus import Action, AgentSolver, GiveUp, MoveContext, Observation, RunConfig
from platypus.agent import Agent


@dataclass(kw_only=True)
class FirstTargetAgent(Agent):
    """Plays ExampleGame's level-1 target action forever."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        return Action("ACTION1")


@dataclass(kw_only=True)
class SlowMoveAgent(Agent):
    """A single move that blocks far longer than the budget. Mirrors the duck's
    unbounded-rounds move: if the cap were only checked between moves this would
    run to completion and record an action; the cap must cancel it in flight."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        await asyncio.sleep(30)
        return Action("ACTION1")


async def test_zero_budget_gives_up_before_first_move() -> None:
    agent = FirstTargetAgent(run_config=RunConfig(process_mode="inprocess", max_runtime_s_per_game=0.0))
    benchmark = taaf.benchmark.Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert len(run.history) == 0
    assert run.solver_note is not None and "runtime limit" in run.solver_note


async def test_budget_cancels_in_flight_move() -> None:
    # The move blocks 30s but the budget is 0.2s: the cap must cancel the
    # in-flight do_move (its await raises CancelledError) rather than let it
    # finish. A between-moves-only check would run the full 30s and record
    # one move; here the move is cancelled before it produces an action.
    agent = SlowMoveAgent(run_config=RunConfig(process_mode="inprocess", max_runtime_s_per_game=0.2))
    benchmark = taaf.benchmark.Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert len(run.history) == 0  # cancelled mid-await: no completed move
    assert run.solver_note is not None and "runtime limit" in run.solver_note


@dataclass(kw_only=True)
class SlowLevelWinAgent(Agent):
    """Blocks inside the DELIVERY-ONLY call a level win produces. An agent may
    do real work there (the whale relearns its model on a level win), so the
    budget has to cover that call too — it is the one do_move the loop does not
    reach through its own decision path."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        if obs.outcome == "level_won":
            await asyncio.sleep(30)
        return Action("ACTION1")


async def test_budget_cancels_a_delivery_only_move() -> None:
    agent = SlowLevelWinAgent(run_config=RunConfig(process_mode="inprocess", max_runtime_s_per_game=0.2))
    game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1], [arcengine.GameAction.ACTION1]])
    benchmark = taaf.benchmark.Benchmark(games=[game], solver=AgentSolver(agent=agent))
    start = time.monotonic()
    await benchmark.run()
    elapsed = time.monotonic() - start
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert run.solver_note is not None and "runtime limit" in run.solver_note
    # The level-winning action stands; the delivery that followed was cut.
    assert len(run.history) == 1
    assert elapsed < 10.0, elapsed  # the unbounded delivery would take 30s


async def test_unlimited_budget_plays_to_completion() -> None:
    agent = FirstTargetAgent(run_config=RunConfig(process_mode="inprocess", max_runtime_s_per_game=None, rng_seed=0))
    game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1] * 2])
    benchmark = taaf.benchmark.Benchmark(games=[game], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "won"
    assert len(run.history) == 2
