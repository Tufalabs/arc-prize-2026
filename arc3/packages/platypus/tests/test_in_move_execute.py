"""In-move execution (ctx.execute): the agent executes actions itself
during do_move and returns None; the loop records the same history as the
returned-action style, in both process modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import arcengine
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import Action, Agent, AgentSolver, GameResult, GiveUp, RunConfig
from platypus.context import MoveContext
from platypus.contract import ContractViolationError, Observation


@dataclass(kw_only=True)
class InMoveScriptAgent(Agent):
    """Plays a fixed engine-action script through ctx.execute, up to
    ``per_move`` actions per do_move, and returns None."""

    script: list[str] = field(default_factory=lambda: list[str]())
    per_move: int = 2
    cursor: int = 0
    seen_outcomes: list[str] = field(default_factory=lambda: list[str]())

    #: What each per-game copy saw come back from ``ctx.execute``, published
    #: at on_finish. The solver deepcopies the template per (game, pass), so
    #: the copy's own ``seen_outcomes`` is the only record of that and it is
    #: unreachable from the test otherwise. In-process hosting only.
    executed_outcomes: ClassVar[list[list[str]]] = []

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        if obs.outcome != "playing":
            return GiveUp("unexpected delivery")  # in-move results never re-deliver extras
        assert ctx.execute is not None
        executed = 0
        while executed < self.per_move and self.cursor < len(self.script):
            observations = await ctx.execute(Action(self.script[self.cursor]))
            self.cursor += 1
            executed += 1
            self.seen_outcomes.extend(o.outcome for o in observations)
            if observations[-1].outcome != "playing":
                break
        if executed == 0:
            return GiveUp("script exhausted")
        return None

    async def _on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        InMoveScriptAgent.executed_outcomes.append(list(self.seen_outcomes))


def _games() -> list[Game]:
    action1, action2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2
    return [ExampleGame(target_sequence_per_level=[[action1], [action2, action1]], label="three_press")]


async def _run(mode: str) -> tuple[list[tuple[str, int, list[str]]], InMoveScriptAgent]:
    agent = InMoveScriptAgent(
        script=["ACTION1", "ACTION2", "ACTION1"],
        run_config=RunConfig(rng_seed=3, process_mode=mode, on_agent_error="raise"),  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()
    rows = [
        (run.state, run.levels_completed, [record.action.id.name for record in run.history])
        for run in benchmark.game_runs
    ]
    return rows, agent


async def test_in_move_execute_wins_and_records_history() -> None:
    rows, _ = await _run("inprocess")
    assert rows == [("won", 2, ["ACTION1", "ACTION2", "ACTION1"])]


async def test_in_move_execute_observationally_equivalent_across_modes() -> None:
    in_process, _ = await _run("inprocess")
    subprocess_mode, _ = await _run("subprocess")
    assert in_process == subprocess_mode


async def test_in_move_execute_returns_level_split_to_agent() -> None:
    """A level transition's observations come back from ctx.execute itself
    (level_won + next initial) — not through extra do_move deliveries."""
    InMoveScriptAgent.executed_outcomes.clear()
    agent = InMoveScriptAgent(
        script=["ACTION1", "ACTION2", "ACTION1"],
        per_move=1,
        run_config=RunConfig(rng_seed=3, process_mode="inprocess", on_agent_error="raise"),  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()
    (run,) = benchmark.game_runs
    assert run.state == "won"

    # The level-winning ACTION1 hands BOTH of its observations back to the
    # caller of ctx.execute, in partition order. A loop that returned only
    # the settled one — or fused the two — reads as a single "playing" here.
    (seen,) = InMoveScriptAgent.executed_outcomes
    assert seen == ["level_won", "playing", "playing", "game_won"]
    # And they were not ALSO re-delivered: a delivery-only do_move call would
    # hit the agent's non-playing guard and give up, so winning is the proof.
    assert [record.action.id.name for record in run.history] == ["ACTION1", "ACTION2", "ACTION1"]
    assert agent.cursor == 0  # A3: template survives untouched


@dataclass(kw_only=True)
class NoneWithoutExecuteAgent(Agent):
    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        return None  # contract violation: no in-move execute happened


async def test_none_without_execute_is_a_contract_violation() -> None:
    # Benchmark.run swallows solver exceptions (R2.15); the violation
    # surfaces as the game crashing with the error in solver_note.
    agent = NoneWithoutExecuteAgent(
        run_config=RunConfig(process_mode="inprocess", on_agent_error="crash_game")  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert "ContractViolationError" in (run.solver_note or "")
    assert "without executing any action" in (run.solver_note or "")


@dataclass(kw_only=True)
class ExecuteOutsideMoveAgent(Agent):
    grabbed: list[object] = field(default_factory=lambda: list[object]())

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        self.grabbed.append(ctx.execute)
        return GiveUp("done probing")

    async def _on_finish(self, ctx: MoveContext, result: object) -> None:  # type: ignore[override]
        # Outside do_move the executor is disarmed: calling it must refuse.
        execute = self.grabbed[0]
        assert execute is not None
        try:
            await execute(Action("ACTION1"))  # type: ignore[operator]
        except ContractViolationError:
            return
        raise AssertionError("ctx.execute must refuse outside do_move")


async def test_execute_refused_outside_do_move() -> None:
    agent = ExecuteOutsideMoveAgent(
        run_config=RunConfig(process_mode="inprocess", on_agent_error="raise")  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()  # the assertion lives in on_finish; a failure surfaces as solver_note
    run = benchmark.game_runs[0]
    assert "on_finish failed" not in (run.solver_note or "")
