# type: ignore
"""Tests for taaf.solver (R12.01–R12.02) and taaf.solver_examples (R12.03 + SolverSequence)."""

import asyncio
import pickle
from dataclasses import dataclass, field

import arcengine
import pytest

import taaf


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


# --- Solver base class ------------------------------------------------------


def test_solver_picklable_before_setup():
    """R12.02: Solver must be picklable until setup() is called."""
    solver = taaf.solver_examples.SolverRandom()
    blob = pickle.dumps(solver)
    restored = pickle.loads(blob)
    assert isinstance(restored, taaf.solver_examples.SolverRandom)
    assert restored._setup_called is False


# --- SolverRandom -----------------------------------------------------------


async def test_random_solver_runs_example_game():
    solver = taaf.solver_examples.SolverRandom(max_actions_per_game=200)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    await solver.run_games([game])
    assert game.game_run is not None
    # Bounded by max_actions_per_game; either won or gave_up by the cap.
    assert game.game_run.state in ("won", "gave_up")
    assert len(game.game_run.history) <= 200


async def _action_sequence(seed: int, max_actions: int = 30) -> list[str]:
    solver = taaf.solver_examples.SolverRandom(seed=seed, max_actions_per_game=max_actions)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    await solver.run_games([game])
    assert game.game_run is not None
    return [rec.action.id.name for rec in game.game_run.history]


async def test_random_solver_seed_is_reproducible():
    """Same seed → same action sequence."""
    a = await _action_sequence(seed=7)
    b = await _action_sequence(seed=7)
    assert a == b


async def test_random_solver_different_seeds_diverge():
    """Different seeds produce different sequences (within ~30 actions)."""
    a = await _action_sequence(seed=7)
    b = await _action_sequence(seed=8)
    assert a != b


async def test_random_solver_per_game_seed_offset():
    """Each game in `run_games` gets seed + i — multi-pass copies don't share a sequence."""
    solver = taaf.solver_examples.SolverRandom(seed=42, max_actions_per_game=30)
    solver.setup()
    games = [taaf.game_examples.ExampleGame() for _ in range(4)]
    for g in games:
        g.start_game()
    await solver.run_games(games)
    sequences = [tuple(rec.action.id.name for rec in g.game_run.history) for g in games]
    assert len(set(sequences)) == len(sequences), f"expected distinct sequences across games, got {sequences}"


async def test_random_solver_responds_to_cancel():
    """Cancel mid-run; finish_game called on every still-playing game; state cancelled."""
    # Use a max actions cap that's effectively infinite, so the solver is
    # guaranteed still running when we cancel it.
    solver = taaf.solver_examples.SolverRandom(max_actions_per_game=10**9)
    solver.setup()
    games = [taaf.game_examples.ExampleGame() for _ in range(3)]
    for g in games:
        g.start_game()
    task = asyncio.create_task(solver.run_games(games))
    # Let the solver actually take some actions.
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for g in games:
        assert g.game_run is not None
        # Either fully won by chance during the warm-up (unlikely but possible),
        # or marked cancelled by finish_game seeing the cancelling task.
        assert g.game_run.state in ("won", "cancelled")


async def test_random_solver_allows_no_action_cap():
    solver = taaf.solver_examples.SolverRandom(max_actions_per_game=None)
    solver.setup()
    game = taaf.game_examples.ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]])
    game.start_game()
    await solver.run_games([game])

    assert game.game_run is not None
    assert game.game_run.state == "won"


def test_random_solver_resets_on_engine_game_over():
    """Engine GAME_OVER → SolverRandom forces RESET on the next pick.

    Without this, the walker would waste actions on non-RESET picks while the
    engine is stuck (the lose-state is recoverable via RESET only).
    """
    import random as _random

    import numpy as np

    game = taaf.game_examples.ExampleGame()
    game.start_game()
    # Synthesize a GAME_OVER state and inject it as current_state. The engine
    # may still list non-RESET actions on game-over (varies by game/version);
    # we explicitly populate available_actions here to verify the solver
    # short-circuits regardless.
    raw = arcengine.FrameDataRaw(
        state=arcengine.GameState.GAME_OVER,
        levels_completed=0,
        win_levels=2,
        available_actions=[1, 2],
    )
    raw.frame = [np.zeros((64, 64), dtype=np.int8)]
    game._current_state = taaf.game.GameState(raw=raw)

    solver = taaf.solver_examples.SolverRandom()
    action = solver._pick_action(game, _random.Random(0))
    assert action.id == arcengine.GameAction.RESET


def test_random_solver_click_on_color_targets_present_colors():
    """click_on_color weights present colors equally, so a single-pixel color
    is reachable — uniform pixel clicking would essentially never land on it.
    """
    import random as _random

    import numpy as np

    frame = np.full((64, 64), 3, dtype=np.int8)
    frame[10, 20] = 9  # lone rare-color pixel at (x=20, y=10)
    raw = arcengine.FrameDataRaw(
        state=arcengine.GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=2,
        available_actions=[6],  # CLICK only → always chosen
    )
    raw.frame = [frame]
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    game._current_state = taaf.game.GameState(raw=raw)

    solver = taaf.solver_examples.SolverRandom(click_on_color=True)
    rng = _random.Random(0)
    picks = [solver._pick_action(game, rng) for _ in range(50)]

    targeted = {int(frame[a.data["y"], a.data["x"]]) for a in picks}
    assert targeted <= {3, 9}, "only colors present in the frame may be targeted"
    assert 9 in targeted, "the single-pixel color must be reachable via equal color weighting"


# --- SolverSequence ---------------------------------------------------------


async def test_sequence_solver_wins_example_game():
    """Deterministic winning sequence → state=won."""
    actions = [_act(1)] * 3 + [_act(2)] * 3
    solver = taaf.solver_examples.SolverSequence(actions=actions)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    await solver.run_games([game])
    assert game.game_run is not None
    assert game.game_run.state == "won"
    assert game.game_run.levels_completed == 2
    assert game.game_run.final_score == pytest.approx(100.0)


async def test_sequence_solver_truncated_calls_finish():
    """Sequence ends before the win; solver calls finish_game → gave_up."""
    actions = [_act(1)] * 3  # only enough to win level 0
    solver = taaf.solver_examples.SolverSequence(actions=actions)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    await solver.run_games([game])
    assert game.game_run is not None
    assert game.game_run.state == "gave_up"
    assert game.game_run.levels_completed == 1


async def test_sequence_solver_skips_invalid_actions():
    """Actions not in available_actions are skipped, not raised."""
    # ACTION3 (LEFT) is not in ExampleGame's available_actions; should be skipped.
    actions = [_act(3), _act(1), _act(1), _act(1), _act(2), _act(2), _act(2)]
    solver = taaf.solver_examples.SolverSequence(actions=actions)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    await solver.run_games([game])
    assert game.game_run is not None
    assert game.game_run.state == "won"


# --- 5-second yield contract ------------------------------------------------


async def test_solver_yields_promptly():
    """SolverRandom should yield often; we don't expect to wait long for it.

    R12.02 mandates yielding ≥ every 5s. SolverRandom yields once per action
    (`await asyncio.sleep(0)`), so we can verify by interleaving and counting
    the yields the test sees before the solver finishes naturally.
    """
    # Tight cap so the solver finishes quickly even if it wins by random luck.
    solver = taaf.solver_examples.SolverRandom(max_actions_per_game=200)
    solver.setup()
    game = taaf.game_examples.ExampleGame()
    game.start_game()

    task = asyncio.create_task(solver.run_games([game]))

    yields_seen = 0
    deadline = asyncio.get_event_loop().time() + 5.0
    while not task.done() and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0)
        yields_seen += 1

    # Solver should finish within the 5s deadline AND we should have observed
    # the loop yielding control back (R12.02 contract).
    assert task.done(), "Solver did not finish within 5s"
    assert yields_seen > 10, f"Solver yielded only {yields_seen} times in 5s"
    await task  # surface any solver-side exception


# --- stop_on_error mode ----------------------------------------------------


@dataclass
class _ErroringGame(taaf.game_examples.ExampleGame):
    """ExampleGame that raises on its Nth ``_execute_action`` call. Used to
    simulate per-game errors for ``SolverRandom.stop_on_error`` tests."""

    fail_at: int = 1  # raise on the (1-indexed) Nth execute_action call
    _n_calls: int = field(default=0, init=False, repr=False)

    def _execute_action(self, action):
        self._n_calls += 1
        if self._n_calls == self.fail_at:
            raise RuntimeError("boom: simulated per-game error")
        return super()._execute_action(action)


async def test_random_solver_swallows_per_game_error_by_default():
    """Default ``stop_on_error=False``: an erroring game ends in
    ``gave_up`` with the error noted in ``solver_note``; other games keep
    playing without disruption."""
    ok_game = taaf.game_examples.ExampleGame(label="ok")
    bad_game = _ErroringGame(label="bad", fail_at=2)
    ok_game.start_game()
    bad_game.start_game()
    solver = taaf.solver_examples.SolverRandom(seed=0, max_actions_per_game=50)
    solver.setup()
    # Must not raise.
    await solver.run_games([ok_game, bad_game])
    # Bad game ends in a terminal state (gave_up) with a solver_note.
    assert bad_game.game_run.state == "gave_up"
    assert bad_game.game_run.final_score is not None
    assert "errored" in (bad_game.game_run.solver_note or "")
    assert "RuntimeError" in (bad_game.game_run.solver_note or "")
    # Other game is unaffected — runs to its own terminal state.
    assert ok_game.game_run.state in ("won", "gave_up")


async def test_random_solver_reraises_when_stop_on_error_true():
    """``stop_on_error=True``: the per-game exception propagates out of
    ``run_games`` immediately with no ``finish_game`` cleanup on the
    offending game (original framework R2.15 path — the benchmark's
    teardown handles cleanup)."""
    bad_game = _ErroringGame(label="bad", fail_at=1)
    bad_game.start_game()
    solver = taaf.solver_examples.SolverRandom(seed=0, max_actions_per_game=50, stop_on_error=True)
    solver.setup()
    with pytest.raises(RuntimeError, match=r"boom"):
        await solver.run_games([bad_game])
    # No finish_game was called → game still 'playing'.
    assert bad_game.game_run.state == "playing"
    assert bad_game.game_run.final_score is None
