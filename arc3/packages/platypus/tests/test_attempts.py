"""A game stuck on a level ends (``RunConfig.max_attempt_repeats``,
``max_actions_per_level``): the same attempt played too often on one level,
consecutively or not, each attempt counted on its own, or too many actions
there."""

from __future__ import annotations

import arcengine
import pytest
from infra_fixtures import ProbeAgent
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame

from platypus import Action, AgentSolver, RunConfig
from platypus.attempts import Attempts
from platypus.solver import STUCK_REASON

A1, A2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2


def _counts(actions: list[Action], level: int = 0) -> list[int]:
    attempts = Attempts()
    return [attempts.record(action, level) for action in actions]


def test_each_attempt_keeps_its_own_count_and_a_reset_alone_is_one() -> None:
    up, down, reset = Action("ACTION1"), Action("ACTION2"), Action("RESET")
    # A RESET at the level's start, one straight after it, two attempts taking
    # turns, and a click whose coordinates make it another move.
    actions = [reset, reset, up, reset, down, reset, up, reset, down, reset, up, reset]
    assert _counts(actions) == [1, 2, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3]
    click = [Action("ACTION6", row=1, col=2), reset, Action("ACTION6", row=1, col=3), reset]
    assert _counts(click) == [0, 1, 0, 1]


def test_a_new_level_starts_the_counts_over() -> None:
    attempts = Attempts()
    reset = Action("RESET")
    assert [attempts.record(reset, 0), attempts.record(reset, 0), attempts.record(reset, 1)] == [1, 2, 1]
    assert attempts.actions == 1


async def _play(script: list[str], **limits: int | None) -> tuple[str, str | None, int]:
    """Play ``script``; return the game's state, its note and how many moves the agent made."""
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=tuple(script), run_config=RunConfig(process_mode="inprocess", **limits))  # type: ignore[arg-type]
    game = ExampleGame(target_sequence_per_level=[[A1, A2, A1], [A2, A2]], label="stuck")
    benchmark = Benchmark(games=[game], solver=AgentSolver(agent=agent), n_passes=1)
    await benchmark.run()
    (run,) = benchmark.game_runs
    ((_, _, moves),) = ProbeAgent.finishes
    return run.state, run.solver_note, len(moves)


async def test_the_same_attempt_played_too_often_ends_the_game_where_it_reaches_the_limit() -> None:
    # Two attempts taking turns: the first reaches three on the fifth RESET.
    script = ["ACTION2", "RESET", "ACTION1", "RESET"] * 3
    state, note, moves = await _play(script, max_attempt_repeats=3)
    assert state == "gave_up" and STUCK_REASON in (note or "")
    assert moves == 10


async def test_resets_alone_count_from_the_level_start() -> None:
    state, note, moves = await _play(["RESET"] * 5, max_attempt_repeats=3)
    assert state == "gave_up" and STUCK_REASON in (note or "") and moves == 3


async def test_too_many_actions_on_a_level_end_the_game() -> None:
    state, note, moves = await _play(["ACTION2"] * 6, max_actions_per_level=4)
    assert state == "gave_up" and STUCK_REASON in (note or "")
    assert moves == 4


async def test_the_action_that_wins_a_level_at_the_cap_plays_on() -> None:
    # Level 1 is won on its third action, the cap; level 2 takes two more.
    state, note, moves = await _play(["ACTION1", "ACTION2", "ACTION1", "ACTION2", "ACTION2"], max_actions_per_level=3)
    assert STUCK_REASON not in (note or "")
    assert state == "won" and moves == 5


@pytest.mark.parametrize("limits", [{"max_attempt_repeats": None}, {"max_attempt_repeats": 3}])
async def test_counts_do_not_carry_over_a_level_and_off_is_off(limits: dict[str, int | None]) -> None:
    # Twice on each level, under a limit of three; or many times with the limit off.
    script = ["ACTION2", "RESET"] * 2 + ["ACTION1", "ACTION2", "ACTION1"] + ["ACTION1", "RESET"] * 2
    if limits["max_attempt_repeats"] is None:
        script += ["ACTION1", "RESET"] * 5
    _, note, moves = await _play(script, **limits)
    assert STUCK_REASON not in (note or "")
    assert moves == len(script)
