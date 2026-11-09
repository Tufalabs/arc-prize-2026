"""Tests for taaf.solver and taaf.solver_examples — the solver contract and
the reference agent's behaviour."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

import taaf.game_examples
import taaf.solver_examples
from taaf.game import ArcGame, Game, RunSession
from taaf.grid import Grid
from taaf.solver import Solver


def test_setup_and_teardown_run_once_and_in_order():
    """The framework owns the lifecycle; the hooks are the subclass's half."""
    calls: list[str] = []

    class _Recording(Solver):
        """Records its lifecycle calls."""

        def _setup(self) -> None:
            calls.append("setup")

        def _teardown(self) -> None:
            calls.append("teardown")

        async def _run_games(self, games: list[Game]) -> None:
            calls.append("run")

    solver = _Recording()
    solver.setup()
    asyncio.run(solver.run_games([]))
    solver.teardown()
    assert calls == ["setup", "run", "teardown"]


def test_run_games_before_setup_is_refused():
    """A solver that plays before setup has skipped whatever setup was for."""
    solver = taaf.solver_examples.SolverFlipLR()
    with pytest.raises(AssertionError, match="setup"):
        asyncio.run(solver.run_games([]))


def test_flip_lr_mirrors_and_keeps_the_question_as_its_second_attempt():
    """Attempt 2 is deliberately the unmodified question, so the two-attempt
    machinery is exercised rather than duplicated past."""
    spec = taaf.game_examples.flip_task("f", n_to_answer=2, size=3)
    game = ArcGame(spec_data=spec)
    view = game.start_game()
    submission = taaf.solver_examples.flip_lr(view)
    assert len(submission.attempts) == 2
    for question, (first, second) in zip(view.questions, submission.attempts, strict=True):
        assert first == Grid(np.fliplr(question.data))
        assert second == question


async def test_flip_lr_solves_flip_tasks_and_not_constant_ones():
    """The reference agent's whole contract: exact on the transformation it
    implements, zero elsewhere."""
    flip = ArcGame(spec_data=taaf.game_examples.flip_task("f"))
    const = ArcGame(spec_data=taaf.game_examples.constant_task("c"))
    for game in (flip, const):
        game.start_game()
    solver = taaf.solver_examples.SolverFlipLR()
    solver.setup()
    await solver.run_games([flip, const])
    assert flip.game_run is not None and flip.game_run.final_score == 100.0
    assert const.game_run is not None and const.game_run.final_score == 0.0


async def test_the_demo_solver_replaces_its_answer_and_is_scored_on_the_last():
    """The schedule ends on the mirror, and the mirror is what scores — even
    though a wrong answer stood at three of the four points along the way."""
    game = ArcGame(spec_data=taaf.game_examples.flip_task("f"))
    view = game.start_game()
    solver = taaf.solver_examples.SolverFlipLR()
    solver.setup()
    await solver.run_games([game])
    run = game.game_run
    assert run is not None
    assert len(run.history) == len(taaf.solver_examples.DEMO_ANSWERS)
    assert run.standing_submission == taaf.solver_examples.flip_lr(view)
    assert run.history[0].submission == taaf.solver_examples.all_black(view)
    assert run.final_score == 100.0


async def test_a_game_cancelled_mid_schedule_is_scored_on_what_stood():
    """The point of letting a solver answer more than once: a run cut off
    part-way submits its best so far rather than nothing."""
    game = ArcGame(spec_data=taaf.game_examples.flip_task("f"))
    game.start_game()
    solver = taaf.solver_examples.SolverFlipLR(delay_s=0.02)
    solver.setup()
    task = asyncio.create_task(solver.run_games([game]))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = game.game_run
    assert run is not None
    assert run.state == "cancelled"
    assert 0 < len(run.history) < len(taaf.solver_examples.DEMO_ANSWERS)
    assert run.final_score == (100.0 if len(run.history) % 2 == 0 else 0.0)


async def test_cancellation_finishes_every_still_playing_game():
    """On cancel the solver wraps up, so games land `cancelled`
    rather than being stamped `crashed` at teardown."""
    games: list[Game] = [ArcGame(spec_data=taaf.game_examples.flip_task(f"g{i}")) for i in range(3)]
    for game in games:
        game.start_game()
    solver = taaf.solver_examples.SolverFlipLR(delay_s=10.0)
    solver.setup()
    task = asyncio.create_task(solver.run_games(games))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(g.game_run is not None and g.game_run.state == "cancelled" for g in games)


async def test_analysing_solver_writes_notes_and_its_analysis_page(tmp_path: Path):
    """The solver-owned diagnostics seams have a user, so they stay wired."""
    game = ArcGame(spec_data=taaf.game_examples.flip_task("f"))
    game.start_game()
    solver = taaf.solver_examples.SolverAnalysing()
    solver.job_dir = tmp_path
    solver.setup()
    await solver.run_games([game])
    solver.teardown()
    assert game.game_run is not None
    assert game.game_run.solver_note == "settled after 4 answer(s)"
    assert (tmp_path / "solver_analysis" / "flip_lr.html").is_file()


async def test_tracing_solver_registers_a_trace(tmp_path: Path):
    """The R5 trace path needs a user outside its own tests."""
    game = ArcGame(spec_data=taaf.game_examples.flip_task("f"))
    game.start_game(RunSession(trace_root=tmp_path / "traces"))
    solver = taaf.solver_examples.SolverTracing()
    solver.setup()
    await solver.run_games([game])
    assert game.game_run is not None
    folder = tmp_path / "traces" / game.game_run.run_id
    assert (folder / "_SUCCESS").is_file()
