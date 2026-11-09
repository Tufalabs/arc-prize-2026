# type: ignore
"""Tests for taaf.game (R11.01–R11.05) and taaf.game_examples (R11.06)."""

import asyncio
import copy
import json
import pickle
import threading
import time

import arcengine
import matplotlib
import numpy as np
import pytest

import taaf

matplotlib.use("Agg")  # headless, no display required


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


# --- Lifecycle / state machine ----------------------------------------------


def test_example_game_full_play_through():
    """Full happy path: not_started → playing → won across two levels."""
    game = taaf.game_examples.ExampleGame()
    assert game.game_run is None

    initial = game.start_game()
    assert game.game_run is not None
    assert game.game_run.state == "playing"
    assert game.number_of_levels == 2
    assert game.base_actions_per_level == [3, 3]
    assert initial.levels_completed == 0
    assert initial.previous_action is None

    # Level 0: ACTION1 x 3
    for _ in range(3):
        s = game.execute_action(_act(1), generated_tokens=2, uncached_input_tokens=5)

    # After third ACTION1: just_won_level + on level 1
    assert s.just_won_level is True
    assert s.levels_completed == 1
    assert game.game_run.state == "playing"

    # ACTION1 on level 1 doesn't help — resets level 1's progress (no-op since 0)
    s = game.execute_action(_act(1))
    assert s.levels_completed == 1
    assert game.game_run.state == "playing"

    # Level 1: ACTION2 x 3 → win
    for _ in range(3):
        s = game.execute_action(_act(2))
    assert s.won is True
    assert s.levels_completed == 2
    assert game.game_run.state == "won"


def test_finish_without_winning_marks_gave_up():
    """Sync caller (no asyncio task) → finish_game sets gave_up."""
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    game.execute_action(_act(1))  # 1/3 toward level 0
    game.finish_game()
    assert game.game_run is not None
    assert game.game_run.state == "gave_up"
    assert game.game_run.final_score is not None  # computed even though 0


def test_finish_under_cancellation_marks_cancelled():
    """When the current asyncio task is cancelling, finish_game sets cancelled."""

    async def play_then_get_cancelled():
        game = taaf.game_examples.ExampleGame()
        game.start_game()
        game.execute_action(_act(1))
        try:
            # An infinite sleep that we'll cancel from outside.
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Solver-side equivalent: call finish_game on the cancelling task.
            game.finish_game()
            raise
        return game

    async def driver():
        task = asyncio.create_task(play_then_get_cancelled())
        # Let the inner task start and reach the sleep.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Inspect the game by re-running the inner workflow ourselves to a
        # ``cancelled`` marker — the task that raised CancelledError dropped
        # its return value, but we know it called finish_game in the except.

    asyncio.run(driver())

    # Now do it again, but this time capture the game object for assertion.
    captured: list[taaf.game.Game] = []

    async def play_capture():
        game = taaf.game_examples.ExampleGame()
        captured.append(game)
        game.start_game()
        game.execute_action(_act(1))
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            game.finish_game()
            raise

    async def driver2():
        task = asyncio.create_task(play_capture())
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(driver2())
    g = captured[0]
    assert g.game_run is not None
    assert g.game_run.state == "cancelled"


def test_token_logging():
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    game.execute_action(_act(1), generated_tokens=42, uncached_input_tokens=7)
    rec = game.game_run.history[0]
    assert rec.generated_tokens == 42
    assert rec.uncached_input_tokens == 7
    assert rec.wallclock_seconds >= 0


def test_finish_game_records_no_move_tokens():
    """A solver can report tokens spent on its final no-move turn (give-up /
    cancel) via finish_game(...) (R11.03). They land on the GameRun final_*
    fields and round-trip through JSON, but add no action — and a second
    (idempotent) finish_game call must not overwrite or double-count them."""
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    game.execute_action(_act(1), generated_tokens=42, uncached_input_tokens=7)
    game.finish_game(generated_tokens=100, uncached_input_tokens=30)
    run = game.game_run
    assert run.final_generated_tokens == 100
    assert run.final_uncached_input_tokens == 30
    # No phantom action: the no-move turn doesn't touch history / counters.
    assert len(run.history) == 1
    assert sum(run.actions_per_level) == 1
    # Idempotent: the early-return guard means a second call is a no-op.
    game.finish_game(generated_tokens=999, uncached_input_tokens=999)
    assert run.final_generated_tokens == 100
    assert run.final_uncached_input_tokens == 30
    # Finish time stamped on the same monotonic axis as action wallclocks,
    # so it's not before the last move.
    assert run.final_wallclock_seconds >= run.history[-1].wallclock_seconds
    # Round-trips through JSON.
    restored = taaf.game.GameRun.from_json_dict(run.to_json_dict())
    assert restored.final_generated_tokens == 100
    assert restored.final_uncached_input_tokens == 30
    assert restored.final_wallclock_seconds == run.final_wallclock_seconds


def test_start_game_rejects_unpopulated_subclass():
    """A subclass whose `_start_game()` forgets to populate basic info must trip an assert."""
    raw = arcengine.FrameDataRaw(state=arcengine.GameState.NOT_FINISHED, available_actions=[1])
    raw.frame = [np.zeros((4, 4), dtype=np.int8)]
    initial = taaf.game.GameState(raw=raw)

    class MissingId(taaf.game.Game):
        def _start_game(self, session):
            self.number_of_levels = 1
            self.base_actions_per_level = [3]
            return initial  # game_id intentionally left empty

    class MissingLevels(taaf.game.Game):
        def _start_game(self, session):
            self.game_id = "x"
            self.base_actions_per_level = [3]
            return initial  # number_of_levels intentionally left at default 0

    class MismatchedBaseline(taaf.game.Game):
        def _start_game(self, session):
            self.game_id = "x"
            self.number_of_levels = 2
            self.base_actions_per_level = [3]  # wrong length
            return initial

    for cls in (MissingId, MissingLevels, MismatchedBaseline):
        with pytest.raises(AssertionError):
            cls().start_game()

    # `hint` is optional — leaving it unset must NOT trip the assertion.
    class NoHint(taaf.game.Game):
        def _start_game(self, session):
            self.game_id = "x"
            self.number_of_levels = 1
            self.base_actions_per_level = [3]
            return initial

    NoHint().start_game()  # no exception


def test_disallowed_action_raises():
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    # ACTION3 (LEFT) is not in available_actions for ExampleGame — only RESET, ACTION1, ACTION2.
    with pytest.raises(ValueError, match=r"not in available_actions"):
        game.execute_action(_act(3))


# --- R11.05 picklability / deepcopyability ----------------------------------


def test_pickle_round_trip_before_start():
    game = taaf.game_examples.ExampleGame()
    blob = pickle.dumps(game)
    restored = pickle.loads(blob)
    assert restored.game_run is None
    restored.start_game()
    assert restored.game_run is not None


def test_deepcopy_before_start():
    game = taaf.game_examples.ExampleGame()
    copy.deepcopy(game).start_game()  # no exception
    # Original is independent.
    assert game.game_run is None


def test_pickle_after_start_raises():
    """R11.05 ABC contract: pickle is forbidden post-start (no opt-in)."""
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    with pytest.raises(RuntimeError, match="after start_game"):
        pickle.dumps(game)


def test_deepcopy_after_start_raises():
    """R11.05 ABC contract: deepcopy is refused post-start unless allow_deepcopy=True."""
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    with pytest.raises(RuntimeError, match="allow_deepcopy"):
        copy.deepcopy(game)


def test_arcengine_enums_pickle():
    """Verifies the relocated arcengine pickle fix in game.py still applies."""
    a = arcengine.GameAction.ACTION1
    assert pickle.loads(pickle.dumps(a)) is a
    s = arcengine.GameState.WIN
    assert pickle.loads(pickle.dumps(s)) is s
    ai = arcengine.ActionInput(id=arcengine.GameAction.ACTION1, data={})
    restored = pickle.loads(pickle.dumps(ai))
    assert restored.id is arcengine.GameAction.ACTION1


# --- GameState properties ---------------------------------------------------


def test_game_state_properties():
    game = taaf.game_examples.ExampleGame()
    state = game.start_game()
    assert isinstance(state.frame, taaf.game.Frame)
    assert state.frame.data.shape == (64, 64)
    # animation_frames is empty since ExampleGame returns one-frame states.
    assert state.animation_frames == []
    assert state.all_frames == [state.frame]
    # RESET (0) is always in available_actions even though ExampleGame's engine omits it.
    assert 0 in state.available_actions
    assert 1 in state.available_actions
    assert 2 in state.available_actions
    assert state.game_over is False
    assert state.won is False


def test_game_over_true_on_win_and_engine_game_over():
    """R11.01: game_over is true when the game ended, won OR lost; won only
    on WIN. arcengine's GAME_OVER (a lost attempt) is recoverable via RESET
    but still an ended game."""
    for engine_state, game_over, won in (
        (arcengine.GameState.NOT_FINISHED, False, False),
        (arcengine.GameState.WIN, True, True),
        (arcengine.GameState.GAME_OVER, True, False),
    ):
        raw = arcengine.FrameDataRaw(state=engine_state, available_actions=[1])
        raw.frame = [np.zeros((4, 4), dtype=np.int8)]
        state = taaf.game.GameState(raw=raw)
        assert state.game_over is game_over
        assert state.won is won


def test_invalid_window_start_failure_runs_subclass_teardown():
    """R11.07: a window validation failure after _start_game must run the
    subclass's _finish_game hook — game_run doesn't exist yet, so nothing
    else can release resources the subclass opened (e.g. GameAPI's
    scorecard / competition run slot)."""
    calls: list[str] = []

    class _TrackedGame(taaf.game_examples.ExampleGame):
        def _finish_game(self):
            calls.append("finish")
            super()._finish_game()

    game = _TrackedGame(stop_at_level=99)
    with pytest.raises(ValueError, match="stop_at_level"):
        game.start_game()
    assert calls == ["finish"]
    assert game.game_run is None


def test_frame_draw_returns_axes():
    f = taaf.game.Frame(data=np.zeros((4, 4), dtype=np.int8))
    ax = f.draw()
    import matplotlib.axes

    assert isinstance(ax, matplotlib.axes.Axes)


def test_frame_copies_the_buffer_it_is_built_on():
    """arcengine keeps writing into the array it hands back, so a Frame that
    aliased it would change under everyone holding it. ``np.asarray`` would
    not be enough here — on an already-int8 array it returns the same object.
    """
    buf = np.zeros((3, 3), dtype=np.int8)
    f = taaf.game.Frame(data=buf)
    buf[0, 0] = 7
    assert f.data[0, 0] == 0
    assert f.data is not buf


def test_frame_equality_is_by_content():
    """``Frame`` defines ``__eq__`` by hand: the dataclass-generated one
    compares the arrays with ``==``, which yields an array, not a bool.
    """
    a = taaf.game.Frame(data=np.zeros((2, 2), dtype=np.int8))
    b = taaf.game.Frame(data=np.zeros((2, 2), dtype=np.int8))
    c = taaf.game.Frame(data=np.ones((2, 2), dtype=np.int8))
    assert a == b
    assert a != c
    # Different shape, same values — array_equal is False, and must not raise.
    assert a != taaf.game.Frame(data=np.zeros((2, 3), dtype=np.int8))
    # Non-Frame: NotImplemented at the dunder, so python falls back to False.
    assert taaf.game.Frame.__eq__(a, 5) is NotImplemented
    assert not (a == 5)
    assert a != 5


def test_frame_is_unhashable():
    """``eq=False`` stops the dataclass generating ``__eq__``/``__hash__``,
    but the hand-written ``__eq__`` in the class body makes python set
    ``__hash__ = None`` — so ``Frame`` stays unhashable, as it was when the
    generated ``__hash__`` tried to hash an ndarray. Pinned because a
    refactor that dropped the custom ``__eq__`` would silently restore
    identity hashing, and frames would dedupe by object rather than content.
    """
    f = taaf.game.Frame(data=np.zeros((2, 2), dtype=np.int8))
    assert taaf.game.Frame.__hash__ is None
    with pytest.raises(TypeError):
        hash(f)


# --- GameRun JSON round-trip -------------------------------------------------


def test_game_run_to_from_json():
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    for _ in range(3):
        game.execute_action(_act(1), generated_tokens=10, uncached_input_tokens=2)
    game.finish_game()
    run = game.game_run
    assert run is not None

    d = run.to_json_dict()
    # intermediate_states is intentionally NOT in the JSON (R11.02).
    assert "intermediate_states" not in d
    assert d["state"] == "gave_up"
    assert len(d["history"]) == 3
    assert d["history"][0]["action"]["id"] == "ACTION1"

    restored = taaf.game.GameRun.from_json_dict(d)
    assert restored.state == "gave_up"
    assert len(restored.history) == 3
    assert restored.history[0].generated_tokens == 10
    assert restored.intermediate_states == []  # not reconstructed
    assert restored.actions_per_level == run.actions_per_level
    assert restored.levels_completed == run.levels_completed


# --- Intermediate states, held compressed ----------------------------------


def test_intermediate_states_come_back_as_recorded():
    """Every step's state is held compressed and rebuilt on access, with
    the same frames and bookkeeping it was recorded with, and in less
    space than the frames themselves."""
    game = taaf.game_examples.ExampleGame()
    first = game.start_game()
    states = [first] + [game.execute_action(_act(1 + i % 2)) for i in range(6)]
    run = game.game_run
    assert run is not None
    assert len(run.intermediate_states) == len(states)
    for recorded, original in zip(run.intermediate_states, states, strict=True):
        assert [f.data.tolist() for f in recorded.all_frames] == [f.data.tolist() for f in original.all_frames]
        assert recorded.levels_completed == original.levels_completed
        assert recorded.previous_action == original.previous_action
    assert [s.levels_completed for s in run.intermediate_states[1:3]] == [s.levels_completed for s in states[1:3]]
    frame_bytes = sum(f.nbytes for s in states for f in s.raw.frame)
    assert sum(map(len, run.intermediate_states._blobs)) < frame_bytes
    # Two rebuilt copies of a GameState cannot compare (its arrays); the
    # record still equals a copy of itself.
    assert copy.deepcopy(run.intermediate_states) == run.intermediate_states


def test_intermediate_states_are_a_snapshot():
    """A frame buffer written into after the step is recorded does not
    change the record: the engine reuses the buffers it hands back."""
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    state = game.execute_action(_act(1))
    run = game.game_run
    assert run is not None
    before = run.intermediate_states[-1].raw.frame[-1].copy()
    state.raw.frame[-1][...] = (before + 1) % 16
    assert np.array_equal(run.intermediate_states[-1].raw.frame[-1], before)


# --- Score formula ----------------------------------------------------------


def _score(
    *,
    n_levels: int,
    base: list[int],
    actions: list[int],
    levels_completed: int,
) -> float:
    run = taaf.game.GameRun(
        game_id="x",
        number_of_levels=n_levels,
        base_actions_per_level=base,
        actions_per_level=list(actions),
        levels_completed=levels_completed,
    )
    return run._compute_final_score()


def test_score_formula_at_baseline():
    """Two levels, completed exactly at baseline → 100."""
    s = _score(n_levels=2, base=[3, 3], actions=[3, 3], levels_completed=2)
    assert s == pytest.approx(100.0)


def test_score_formula_faster_than_baseline_caps_at_115():
    """Both levels at half baseline (4x) — per-level capped at 115; final cap at 100."""
    s = _score(n_levels=2, base=[3, 3], actions=[1, 1], levels_completed=2)
    # Each level: (3/1)^2*100 = 900 → capped at 115.
    # total_score = 115*1 + 115*2 = 345; total_weights=3; score=115.
    # max_score = 3/3*100 = 100; final = min(115, 100) = 100.
    assert s == pytest.approx(100.0)


def test_score_formula_partial_completion_uses_max_cap():
    """Three levels, only level 0 completed at baseline — capped by max_weights."""
    s = _score(n_levels=3, base=[3, 3, 3], actions=[3, 0, 0], levels_completed=1)
    # level_scores=[100, 0, 0]; total_score=100*1=100; total_weights=1+2+3=6; score=100/6≈16.67
    # max_weights=1 (only level 0 has score>0); max_score=1/6*100≈16.67; final=16.67.
    assert s == pytest.approx(100.0 / 6, abs=1e-6)


def test_score_formula_partial_blazing_capped_by_max_score():
    """Level 0 completed at 5x baseline — per-level cap kicks in but max_score still caps total."""
    s = _score(n_levels=3, base=[5, 5, 5], actions=[1, 0, 0], levels_completed=1)
    # level 0: (5/1)^2*100 = 2500 → capped at 115.
    # total_score = 115*1 = 115; total_weights = 6; score = 115/6 ≈ 19.17
    # max_weights = 1; max_score = 100/6 ≈ 16.67; final = 16.67.
    assert s == pytest.approx(100.0 / 6, abs=1e-6)


def test_score_formula_zero_when_no_levels_completed():
    s = _score(n_levels=2, base=[3, 3], actions=[0, 0], levels_completed=0)
    assert s == 0.0


def test_score_formula_handles_no_baseline():
    """No baseline (e.g. submission mode) → score is 0 (we don't surface None here)."""
    run = taaf.game.GameRun(
        game_id="x",
        number_of_levels=2,
        base_actions_per_level=None,
        actions_per_level=[3, 3],
        levels_completed=2,
    )
    assert run._compute_final_score() == 0.0


# --- Level window: start_at_level / stop_at_level (R11.07) -------------------


def _multi_level_game(n_levels: int, **kwargs) -> "taaf.game_examples.ExampleGame":
    """ExampleGame with ``n_levels`` levels, each cleared by a single ACTION1
    (baseline 1/level) — minimal scaffolding for window tests."""
    seqs = [[arcengine.GameAction.ACTION1] for _ in range(n_levels)]
    return taaf.game_examples.ExampleGame(target_sequence_per_level=seqs, **kwargs)


def test_stop_at_level_forces_early_win_and_caps_score():
    """stop_at_level=2 (half-open) → play indices 0,1 then auto-win; the
    unreached levels 2,3 cap the score below 100."""
    game = _multi_level_game(4, stop_at_level=2)
    game.start_game()

    s = game.execute_action(_act(1))  # complete level 0
    assert not s.won and s.levels_completed == 1
    s = game.execute_action(_act(1))  # complete level 1 → levels_completed hits stop_at_level
    assert s.won and s.levels_completed == 2
    assert game.game_run.state == "won"

    game.finish_game()
    # Levels 0,1 cleared at baseline (weights 1,2); levels 2,3 (weights 3,4)
    # never reached. max_score = (1+2)/(1+2+3+4)*100 = 30.
    assert game.game_run.final_score == pytest.approx(30.0)


# --- R11.15 baseline multiplier ---------------------------------------------


def test_baseline_multiplier_scales_what_every_score_is_read_against():
    """The same play, scored against the baselines the game reported and
    against tripled ones: a level taking twice its baseline scores 25, and
    against the tripled baseline the same play is well past full marks,
    which the game's own score caps at 100."""

    def play(**kwargs):
        # Each level: a RESET (costs an action, zeroes in-level progress),
        # then the action that clears it — two against a baseline of one.
        game = _multi_level_game(2, **kwargs)
        game.start_game()
        for _ in range(2):
            game.execute_action(_act(0))
            game.execute_action(_act(1))
        game.finish_game()
        return game.game_run

    plain = play()
    assert plain.base_actions_per_level == [1, 1]
    assert plain.baseline_multiplier == 1.0
    assert plain.actions_per_level == [2, 2]
    assert plain.final_score == pytest.approx(25.0)  # (1/2)^2

    scaled = play(baseline_multiplier=3)
    assert scaled.base_actions_per_level == [3, 3]
    assert scaled.baseline_multiplier == 3.0
    assert scaled.actions_per_level == [2, 2]
    assert scaled.final_score == pytest.approx(100.0)  # (3/2)^2, capped
    # The stored score is what a reader recomputes off the stored baselines.
    assert scaled.final_score == pytest.approx(scaled._compute_final_score())


def test_baseline_multiplier_rounds_to_whole_actions_and_survives_json():
    """The scaled baselines are action counts, and the factor rides along
    through the run's JSON so a reader knows what it is looking at."""
    game = taaf.game_examples.ExampleGame(baseline_multiplier=2.5)
    game.start_game()
    assert game.game_run.base_actions_per_level == [8, 8]  # 3 * 2.5 = 7.5

    restored = taaf.game.GameRun.from_json_dict(json.loads(json.dumps(game.game_run.to_json_dict())))
    assert restored.base_actions_per_level == [8, 8]
    assert restored.baseline_multiplier == 2.5


def test_baseline_multiplier_keeps_a_positive_baseline_positive():
    """A factor below 1 cannot round a baseline away: a one-action level at
    half its baseline still has one, and winning it in one action scores."""
    game = _multi_level_game(1, baseline_multiplier=0.5)
    game.start_game()
    assert game.game_run.base_actions_per_level == [1]  # round(0.5) is 0
    game.execute_action(_act(1))
    game.finish_game()
    assert game.game_run.state == "won"
    assert game.game_run.final_score == pytest.approx(100.0)


def test_baseline_multiplier_must_be_positive():
    """Caught before the game is started rather than at scoring time."""
    game = _multi_level_game(2, baseline_multiplier=0)
    with pytest.raises(ValueError, match="baseline_multiplier must be positive"):
        game.start_game()


def test_start_at_level_skips_levels_and_seeds_counter():
    """start_at_level=2 → initial state is level index 2, actions land there,
    and only the played levels contribute to the score."""
    game = _multi_level_game(4, start_at_level=2)
    initial = game.start_game()
    assert initial.levels_completed == 2
    assert game.game_run.levels_completed == 2  # seeded from the initial frame

    s = game.execute_action(_act(1))  # complete level 2
    assert s.levels_completed == 3
    assert game.game_run.actions_per_level == [0, 0, 1, 0]
    s = game.execute_action(_act(1))  # complete level 3 → real engine win
    assert s.won and s.levels_completed == 4

    game.finish_game()
    # Only levels 2,3 (weights 3,4) score; max = (3+4)/10*100 = 70.
    assert game.game_run.final_score == pytest.approx(70.0)


def test_start_and_stop_window_plays_only_inner_levels():
    """[start=1, stop=3) → play exactly indices 1 and 2."""
    game = _multi_level_game(5, start_at_level=1, stop_at_level=3)
    initial = game.start_game()
    assert initial.levels_completed == 1

    s = game.execute_action(_act(1))  # complete level 1
    assert not s.won and s.levels_completed == 2
    s = game.execute_action(_act(1))  # complete level 2 → reach stop=3
    assert s.won and s.levels_completed == 3
    assert game.game_run.actions_per_level == [0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(start_at_level=4), "start_at_level=4 out of range"),
        (dict(stop_at_level=5), "stop_at_level=5 out of range"),
        (dict(start_at_level=2, stop_at_level=2), "must be <"),
        (dict(start_at_level=3, stop_at_level=1), "must be <"),
    ],
)
def test_level_window_validation(kwargs, match):
    game = _multi_level_game(4, **kwargs)  # 4 levels → valid range is [0, 4) / (0, 4]
    with pytest.raises(ValueError, match=match):
        game.start_game()


def test_start_at_level_unhonored_by_subclass_raises():
    """A subclass whose _start_game ignores start_at_level (initial state not
    at the requested level) must trip the base contract check."""

    class IgnoresStart(taaf.game.Game):
        def _start_game(self, session):
            del session
            self.game_id = "ig"
            self.number_of_levels = 3
            self.base_actions_per_level = [1, 1, 1]
            self.grid_size = (64, 64)
            raw = arcengine.FrameDataRaw(
                game_id="ig",
                state=arcengine.GameState.NOT_FINISHED,
                levels_completed=0,  # ignores start_at_level
                win_levels=3,
                available_actions=[1],
            )
            raw.frame = [np.zeros((64, 64), dtype=np.int8)]
            return taaf.game.GameState(raw=raw)

        def _execute_action(self, action):
            raise NotImplementedError

    with pytest.raises(RuntimeError, match="ignored start_at_level"):
        IgnoresStart(start_at_level=1).start_game()


# --- Keepalive RESET (R2.63) -------------------------------------------------


def test_keepalive_reset_goes_unseen_on_an_untouched_level():
    """ExampleGame's RESET zeros in-level progress, so a RESET that landed
    mid-level would break the winning sequence below."""
    game = taaf.game_examples.ExampleGame()
    initial = game.start_game()
    assert game.keepalive_reset()
    run = game.game_run
    assert run.keepalive_resets == 1
    assert run.history == [] and run.actions_per_level == [0, 0]
    assert game.current_state is initial

    game.execute_action(_act(1))
    assert not game.keepalive_reset(), "a touched level must not be reset"
    game.execute_action(_act(1))
    game.execute_action(_act(1))
    assert game.current_state.levels_completed == 1

    assert game.keepalive_reset(), "the new level is untouched again"
    for _ in range(3):
        game.execute_action(_act(2))
    assert run.state == "won"
    assert run.keepalive_resets == 2
    assert len(run.history) == 6
    assert not game.keepalive_reset(), "a finished game is left alone"


def test_keepalive_resets_round_trip_through_json():
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    game.keepalive_reset()
    restored = taaf.game.GameRun.from_json_dict(json.loads(json.dumps(game.game_run.to_json_dict())))
    assert restored.keepalive_resets == 1


def test_finish_game_is_bounded_while_a_keepalive_reset_is_stuck(monkeypatch):
    """A RESET stuck inside its engine call holds the action lock; finishing
    the game must not wait on it forever (R2.63), and the RESET's return
    afterwards touches nothing but its counter."""
    release = threading.Event()
    entered = threading.Event()

    class StuckReset(taaf.game_examples.ExampleGame):
        def _execute_action(self, action):
            if action.id == arcengine.GameAction.RESET:
                entered.set()
                release.wait()
            return super()._execute_action(action)

    monkeypatch.setattr(taaf.game, "FINISH_LOCK_TIMEOUT_S", 0.1)
    game = StuckReset()
    game.start_game()
    keepalive = threading.Thread(target=game.keepalive_reset, daemon=True)
    keepalive.start()
    assert entered.wait(timeout=5.0)

    started = time.monotonic()
    game.finish_game()
    assert time.monotonic() - started < 2.0, "finish waited on the stuck RESET"
    assert game.game_run.state == "gave_up"

    release.set()
    keepalive.join(timeout=5.0)
    assert not keepalive.is_alive()
    assert game.game_run.keepalive_resets == 1
    assert game.game_run.state == "gave_up"
