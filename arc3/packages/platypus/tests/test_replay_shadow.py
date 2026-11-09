"""Replays answered from the record (``platypus.replay_shadow``): the agent sees
what it would have without the shadow, the game is sent only what could teach
the agent something, and a game that does not reproduce a replay is caught."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import arcengine
import numpy as np
import pytest
from taaf.benchmark import Benchmark
from taaf.game import GameState
from taaf.game_examples import ExampleGame

from platypus import Action, Agent, AgentSolver, GiveUp, RunConfig
from platypus.context import MoveContext
from platypus.contract import Observation

A1, A2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2


@dataclass(kw_only=True)
class ScriptAgent(Agent):
    """Plays a fixed script, one action a move, and publishes the frames it was shown."""

    script: list[str] = field(default_factory=lambda: list[str]())
    cursor: int = 0
    shown: list[int] = field(default_factory=lambda: list[int]())

    #: What each per-game copy was shown, published as it finishes: the solver
    #: plays copies of the template, so the copy's own record is unreachable otherwise.
    published: ClassVar[list[list[int]]] = []

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        self.shown.append(int(obs.frame.data[0, 0]))
        if obs.outcome != "playing":
            return None  # a delivery: its return is discarded, so it takes no script entry
        if self.cursor >= len(self.script):
            ScriptAgent.published.append(list(self.shown))
            return GiveUp("script done")
        self.cursor += 1
        return Action(self.script[self.cursor - 1])


class DriftingResetGame(ExampleGame):
    """An ExampleGame whose RESET does not return the same board twice: each one
    shifts the board's colour, which no replay can reproduce."""

    _resets: int = 0

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        state = super()._execute_action(action)
        if action.id == arcengine.GameAction.RESET:
            self._resets += 1
            state.raw.frame = [np.full((64, 64), self._resets % 16, dtype=np.int8)]
        return state


async def _play(script: list[str], *, shadow: bool, game: ExampleGame | None = None, mismatch: str = "crash"):
    """Run ``script`` on one game; return the run and the frames the agent was shown."""
    ScriptAgent.published = []
    agent = ScriptAgent(
        script=script,
        run_config=RunConfig(
            rng_seed=1,
            process_mode="inprocess",
            on_agent_error="crash_game",
            replay_shadow=shadow,
            replay_shadow_mismatch=mismatch,  # type: ignore[arg-type]
        ),
    )
    game = game or ExampleGame(target_sequence_per_level=[[A1, A2, A1], [A2, A2]], label="shadowed")
    benchmark = Benchmark(games=[game], solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()
    (run,) = benchmark.game_runs
    return run, (ScriptAgent.published[0] if ScriptAgent.published else None)


# Level 1 (A1 A2 A1): a RESET at the level's start and an attempt A1 A2 from
# it, two RESETs and a replay of the attempt, a RESET and A1, all answered from
# the record; then A1 off the recorded paths: the game is caught up -- RESET,
# A1 -- and the rest is played for real to the win. Level 2 (A2 A2) opens a
# fresh record: A2, a RESET and its replay (answered), then A2 off the path.
SCRIPT = [
    "RESET", "ACTION1", "ACTION2", "RESET", "RESET", "ACTION1", "ACTION2", "RESET", "ACTION1", "ACTION1",
    "ACTION1", "ACTION2", "ACTION1", "ACTION2", "RESET", "ACTION2", "ACTION2",
]  # fmt: skip


async def test_the_agent_sees_what_it_would_without_the_shadow() -> None:
    shadowed, shown = await _play(SCRIPT, shadow=True)
    plain, plain_shown = await _play(SCRIPT, shadow=False)
    assert shown == plain_shown
    assert shadowed.state == plain.state == "won"


async def test_the_game_is_sent_only_what_could_teach_the_agent_something() -> None:
    shadowed, _ = await _play(SCRIPT, shadow=True)
    plain, _ = await _play(SCRIPT, shadow=False)
    assert [r.action.id.name for r in plain.history] == ["RESET" if a == "RESET" else a for a in SCRIPT]
    # The RESETs before a replay and the replays never reach the game; the
    # last attempt's catch-up does (RESET, A1), then the new A1 and what follows.
    assert [r.action.id.name for r in shadowed.history] == [
        "ACTION1", "ACTION2",
        "RESET", "ACTION1", "ACTION1", "ACTION1", "ACTION2", "ACTION1",
        "ACTION2", "RESET", "ACTION2", "ACTION2",
    ]  # fmt: skip
    assert shadowed.actions_per_level[0] < plain.actions_per_level[0]


async def test_a_game_that_does_not_reproduce_a_replay_crashes_it_by_default() -> None:
    # The catch-up's RESET, ahead of A2, returns a board the level did not open on.
    script = ["ACTION1", "RESET", "ACTION1", "ACTION2", "ACTION1"]
    run, _ = await _play(script, shadow=True, game=DriftingResetGame(target_sequence_per_level=[[A1, A2, A1]]))
    assert run.state == "crashed"


async def test_continue_plays_on_from_the_games_real_state() -> None:
    script = ["ACTION1", "RESET", "ACTION1", "ACTION2", "ACTION1", "RESET", "ACTION1"]
    game = DriftingResetGame(target_sequence_per_level=[[A1, A2, A1]], label="drifting")
    run, shown = await _play(script, shadow=True, game=game, mismatch="continue")
    assert run.state != "crashed"
    # After the failed catch-up the shadow is off: the later RESET reaches the game too.
    assert [r.action.id.name for r in run.history].count("RESET") == 2
    assert shown is not None


@pytest.mark.parametrize("shadow", [True, False])
async def test_a_game_with_no_replays_is_played_exactly_as_before(shadow: bool) -> None:
    run, _ = await _play(["ACTION1", "ACTION2", "ACTION1", "ACTION2", "ACTION2"], shadow=shadow)
    assert run.state == "won"
    assert [r.action.id.name for r in run.history] == ["ACTION1", "ACTION2", "ACTION1", "ACTION2", "ACTION2"]
