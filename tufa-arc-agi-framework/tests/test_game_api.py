# type: ignore
"""Tests for taaf.game_api (R11.11 + R11.12).

Use the default ``ArcadeSpec()`` — OFFLINE mode pointed at re-arc-3's
bundled env files. No network, no API key, no slow marker. Pinned to
the ``ls20`` env for speed.
"""

import copy
import pickle
import warnings
from pathlib import Path

import arcengine
import numpy as np
import pytest

import taaf
from taaf.game_api import ArcadeSpec, GameAPI

# Pinned env id; small game, fast to interact with.
ENV_NAME = "ls20"

# 13-move solution to ls20 level 0. Wins at least the first level —
# used as the "non-trivial score" probe in the smoke test.
LS20_LEVEL0_WINNING_MOVES = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


def _ls20_winning_actions() -> list[arcengine.ActionInput]:
    return [_act(i) for i in LS20_LEVEL0_WINNING_MOVES]


# --- Construction & lifecycle ----------------------------------------------


def test_game_api_construction_no_network():
    """Constructing a GameAPI must not hit the network. R11.11 picklability hinges on this."""
    game = GameAPI(env_name=ENV_NAME)
    assert game.game_run is None
    assert game._arcade is None
    assert game._scorecard_id is None
    assert game.env is None
    assert game.arcade_spec == ArcadeSpec()  # default


async def test_game_api_full_play_through(tmp_path: Path):
    """End-to-end: GameAPI driven by SolverSequence playing the known ls20 win."""
    bm = taaf.benchmark.Benchmark(
        label="game_api_smoke",
        games=[GameAPI(env_name=ENV_NAME)],
        solver=taaf.solver_examples.SolverSequence(actions=_ls20_winning_actions()),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()
    assert len(bm.game_runs) == 1
    run = bm.game_runs[0]
    assert run.state in ("won", "gave_up")
    # The 13-move solution clears at least level 0 of ls20. Score must
    # be non-trivial — scoring is weighted by level number and capped at
    # 100, so winning level 0 alone yields a positive score.
    assert run.final_score is not None
    assert run.final_score > 0, f"expected non-trivial score, got {run.final_score}"
    assert run.levels_completed >= 1, f"expected ≥1 level cleared, got {run.levels_completed}"


# --- Pickle / deepcopy contract --------------------------------------------


def test_game_api_pickle_before_start_round_trip():
    """Unstarted GameAPI must pickle cleanly. The deploy scenario simulates this."""
    game = GameAPI(env_name=ENV_NAME, arcade_spec=ArcadeSpec())
    blob = pickle.dumps(game)
    restored = pickle.loads(blob)
    assert isinstance(restored, GameAPI)
    assert restored._arcade is None
    assert restored._scorecard_id is None
    assert restored.env is None
    assert restored.arcade_spec == game.arcade_spec
    assert restored.env_name == ENV_NAME


def test_game_api_deepcopy_before_start():
    """Deepcopy of unstarted GameAPI shares the (frozen) arcade_spec; transient fields stay None."""
    game = GameAPI(env_name=ENV_NAME)
    new = copy.deepcopy(game)
    assert new is not game
    assert new.arcade_spec == game.arcade_spec  # frozen → equal
    assert new._arcade is None
    assert new.env is None


# --- RunSession caching ----------------------------------------------------


async def test_game_api_session_caches_arcade():
    """Two GameAPIs sharing one spec + one session share the same Arcade."""
    session = taaf.game.RunSession()
    g1 = GameAPI(env_name=ENV_NAME)
    g2 = GameAPI(env_name=ENV_NAME)
    g1.start_game(session)
    g2.start_game(session)
    assert g1._arcade is g2._arcade  # same Arcade by cache identity
    # Different specs → different Arcades. arc_base_url isn't actually hit in
    # OFFLINE mode, so changing it stays a pure cache-key difference for the test.
    g3 = GameAPI(env_name=ENV_NAME, arcade_spec=ArcadeSpec(arc_base_url="https://other.example"))
    g3.start_game(session)
    assert g3._arcade is not g1._arcade


# --- R11.12 score reconciliation -------------------------------------------


async def test_game_api_score_matches_engine_scorecard(tmp_path: Path):
    """No mismatch warning fires on a clean play-through.

    Uses the known winning sequence so both the framework and engine
    produce a positive score; ``0 == 0`` would be a degenerate match.
    """
    bm = taaf.benchmark.Benchmark(
        label="r11_12_smoke",
        games=[GameAPI(env_name=ENV_NAME)],
        solver=taaf.solver_examples.SolverSequence(actions=_ls20_winning_actions()),
        n_passes=1,
        job_dir=tmp_path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await bm.run()
    mismatch = [w for w in caught if "R11.12" in str(w.message) and "mismatch" in str(w.message)]
    assert not mismatch, f"unexpected R11.12 mismatch warning(s): {[str(w.message) for w in mismatch]}"
    assert bm.game_runs[0].final_score is not None
    assert bm.game_runs[0].final_score > 0, "framework score should be non-trivial"


async def test_baseline_multiplier_scales_baselines_and_skips_reconciliation(tmp_path: Path):
    """R11.15: the run carries tripled baselines, scores against them, and
    does not warn — the engine scored the same play by its own baselines,
    so R11.12 has nothing comparable to check.

    The level is won the slow way, after wasted RESETs, so that the play
    takes more actions than its baseline and the two scores can differ at
    all: a play already within the baseline scores the cap either way."""
    slow = [_act(0)] * 20 + _ls20_winning_actions()
    bm = taaf.benchmark.Benchmark(
        label="r11_15_smoke",
        games=[GameAPI(env_name=ENV_NAME, baseline_multiplier=3)],
        solver=taaf.solver_examples.SolverSequence(actions=slow),
        n_passes=1,
        job_dir=tmp_path,
    )
    plain = taaf.benchmark.Benchmark(
        label="r11_15_reference",
        games=[GameAPI(env_name=ENV_NAME)],
        solver=taaf.solver_examples.SolverSequence(actions=slow),
        n_passes=1,
        job_dir=tmp_path / "plain",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await bm.run()
        await plain.run()
    assert not [w for w in caught if "R11.12" in str(w.message)], [str(w.message) for w in caught]

    run, reference = bm.game_runs[0], plain.game_runs[0]
    assert run.base_actions_per_level == [3 * n for n in reference.base_actions_per_level]
    assert run.baseline_multiplier == 3.0
    # Same play, easier baselines: a strictly better score, and the stored
    # one is what recomputing off the run gives.
    assert run.final_score > reference.final_score > 0
    assert run.final_score == pytest.approx(run._compute_final_score())


async def test_game_api_score_matches_engine_scorecard_with_random_walk(tmp_path: Path):
    """R11.12 holds for a random walk that hits GAME_OVER → RESET cycles.

    SolverRandom on ls20 dies into ``GameState.GAME_OVER`` quickly; the
    auto-reset shortcut in ``SolverRandom._pick_action`` issues RESET on
    GAME_OVER so the engine never gets a non-RESET action it would refuse.
    (If a future solver misses that check, ``GameAPI._execute_action``
    raises on the empty frame rather than synthesising a no-op.) This test
    catches regressions in that interaction surface.
    """
    bm = taaf.benchmark.Benchmark(
        label="r11_12_random",
        games=[GameAPI(env_name=ENV_NAME)],
        solver=taaf.solver_examples.SolverRandom(seed=42, max_actions_per_game=200),
        n_passes=1,
        job_dir=tmp_path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await bm.run()
    mismatch = [w for w in caught if "R11.12" in str(w.message) and "mismatch" in str(w.message)]
    assert not mismatch, f"unexpected R11.12 mismatch warning(s): {[str(w.message) for w in mismatch]}"


# --- Keepalive RESET (R2.63) ----------------------------------------------


def test_game_api_keepalive_reset_leaves_play_identical():
    """On the real engine a keepalive RESET on an untouched level -- at the
    start and again right after the level-0 clear -- changes no frame the
    solver sees and no framework score."""

    def play(keepalive: bool) -> tuple[list[np.ndarray], taaf.game.GameRun]:
        game = GameAPI(env_name=ENV_NAME)
        frames = [game.start_game().frame.data.copy()]
        try:
            if keepalive:
                assert game.keepalive_reset()
            for action in _ls20_winning_actions():
                frames.append(game.execute_action(action).frame.data.copy())
            if keepalive:
                assert game.keepalive_reset()
            frames.append(game.execute_action(_act(1)).frame.data.copy())
        finally:
            game.finish_game()
        return frames, game.game_run

    plain_frames, plain_run = play(keepalive=False)
    kept_frames, kept_run = play(keepalive=True)
    assert len(kept_frames) == len(plain_frames)
    assert all(np.array_equal(a, b) for a, b in zip(plain_frames, kept_frames))
    assert kept_run.keepalive_resets == 2
    assert kept_run.actions_per_level == plain_run.actions_per_level
    assert kept_run.final_score == plain_run.final_score


# --- Level window: start_at_level / stop_at_level (R11.07) -----------------


def test_game_api_start_at_level_jumps_engine():
    """start_at_level fast-forwards the live engine: the initial state reports
    the requested level and renders a different frame than a level-0 start."""
    game = GameAPI(env_name=ENV_NAME, start_at_level=3)
    initial = game.start_game()
    try:
        assert game.number_of_levels == 7  # ls20
        assert initial.levels_completed == 3
        assert game.game_run.levels_completed == 3

        baseline = GameAPI(env_name=ENV_NAME)
        b0 = baseline.start_game()
        try:
            assert not np.array_equal(initial.frame.data, b0.frame.data)
        finally:
            baseline.finish_game()
    finally:
        game.finish_game()


async def test_game_api_stop_at_level_wins_and_reconciles(tmp_path: Path):
    """stop_at_level=1 turns the level-0 clear into a win. WIN is forged only
    on the returned GameState, after the real step is recorded, so R11.12
    reconciliation still runs and matches the engine scorecard (no warning)."""
    bm = taaf.benchmark.Benchmark(
        label="stop_at_level",
        games=[GameAPI(env_name=ENV_NAME, stop_at_level=1)],
        solver=taaf.solver_examples.SolverSequence(actions=_ls20_winning_actions()),
        n_passes=1,
        job_dir=tmp_path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await bm.run()
    run = bm.game_runs[0]
    assert run.state == "won"
    assert run.levels_completed == 1
    assert run.final_score is not None and 0 < run.final_score < 100
    assert not [w for w in caught if "R11.12" in str(w.message)]


# --- Pickle-after-start gate (inherited from Game ABC) ---------------------


async def test_game_api_pickle_after_start_raises():
    """ABC R11.05 contract: pickling a started GameAPI raises RuntimeError."""
    session = taaf.game.RunSession()
    game = GameAPI(env_name=ENV_NAME)
    game.start_game(session)
    try:
        with pytest.raises(RuntimeError, match="after start_game"):
            pickle.dumps(game)
    finally:
        game.finish_game()
