"""ARC-3 scoring has one copy (harness_support/arc3/scoring.py), used by the
worker's callback and the trainer, and it scores games as TAAF does."""

from types import SimpleNamespace

import pytest

from harness_support.arc3 import results, scoring

# (number_of_levels, base_actions_per_level, levels_completed, actions_per_level)
GAMES = {
    "no level completed": (3, [10, 20, 30], 0, [7]),
    "one level at its baseline": (3, [10, 20, 30], 1, [10, 4]),
    "the first level over-efficient (capped at 115)": (3, [10, 20, 30], 1, [2]),
    "two levels, one slow": (4, [10, 20, 30, 40], 2, [5, 80, 1]),
    "every level, all over-efficient (game cap)": (3, [10, 20, 30], 3, [1, 1, 1]),
    "every level at its baseline": (2, [4, 6], 2, [4, 6]),
    "a zero baseline scores nothing": (3, [0, 20, 30], 2, [5, 20]),
    "a level of zero actions scores nothing": (2, [4, 6], 2, [0, 6]),
    "single level": (1, [12], 1, [24]),
    "no levels": (0, [], 0, []),
    "hidden baselines": (2, None, 2, [3, 4]),
}


def _final(game):
    levels, baselines, completed, actions = game
    cumulative = scoring._cumulative_arc_scores(completed, actions, baselines, levels)
    return cumulative[-1] if cumulative else 0.0


@pytest.mark.parametrize("name", GAMES)
def test_the_game_score_is_the_official_weighted_and_capped_mean(name):
    levels, baselines, completed, actions = GAMES[name]
    per_level = scoring._per_level_arc_scores(completed, actions, baselines)
    expected = scoring.compute_game_score(
        [(level, per_level[level - 1] if level <= completed else 0.0)
         for level in range(1, max(levels, completed, 1) + 1)]
    ) if completed else 0.0
    assert _final(GAMES[name]) == pytest.approx(expected)
    assert all(0.0 <= score <= scoring.PER_LEVEL_SCORE_CAP for score in per_level)
    assert 0.0 <= _final(GAMES[name]) <= scoring.GAME_SCORE_CAP


def test_fixture_values():
    """Hand-computed: weights 1..n, per level min(115, (base/actions)^2 * 100)."""
    assert _final(GAMES["one level at its baseline"]) == pytest.approx(100 / 6)
    # 115 on level 1 alone is capped at its weight share: 1/6 * 100.
    assert _final(GAMES["the first level over-efficient (capped at 115)"]) == pytest.approx(100 / 6)
    # (1 * 115 + 2 * (20/80)^2 * 100) / 10, under the cap of 3/10 * 100.
    assert _final(GAMES["two levels, one slow"]) == pytest.approx((115 + 2 * 6.25) / 10)
    assert _final(GAMES["every level, all over-efficient (game cap)"]) == pytest.approx(100.0)
    assert _final(GAMES["every level at its baseline"]) == pytest.approx(100.0)
    assert _final(GAMES["a zero baseline scores nothing"]) == pytest.approx(2 * 100 / 6)
    assert _final(GAMES["hidden baselines"]) == 0.0
    assert scoring.arc_game_score_for_completed_levels({1: 100.0}, total_levels=2.5) == (
        pytest.approx(100 / 6)
    )


def test_a_replayed_prefix_earns_nothing():
    cumulative = scoring._cumulative_arc_scores(3, [5, 10, 15], [5, 10, 15], 3)
    assert scoring._earned_arc_score(cumulative, start_level=0) == pytest.approx(100.0)
    assert scoring._earned_arc_score(cumulative, start_level=1) == pytest.approx(
        100.0 - 100 / 6
    )
    assert scoring._earned_arc_score(cumulative, start_level=3) == 0.0
    # Boundaries count the run's own actions from its start level.
    assert scoring._level_boundaries([5, 10, 15], 3) == [4, 14, 29]
    assert scoring._level_boundaries([5, 10, 15], 3, start_level=1) == [9, 24]


def test_stage_features_read_the_start_level_and_the_decision_clock():
    run = SimpleNamespace(
        start_level=1,
        actions_per_level=[5, 4, 6],
        base_actions_per_level=[5, 8, 6],
        number_of_levels=3,
        action_wallclock_seconds=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    boundaries = scoring._level_boundaries(run.actions_per_level, 2, start_level=1)
    stage, levels_before = scoring._stage_features(
        run, cursor=2, boundaries=boundaries, max_runtime_seconds=60.0
    )
    assert levels_before == 1 and stage["level"] == 2
    assert stage["actions_on_current_level"] == 2
    assert stage["potential_score_now"] == pytest.approx(scoring.PER_LEVEL_SCORE_CAP)
    assert stage["run_elapsed_seconds"] == 2.0
    assert stage["time_remaining_seconds"] == 58.0
    stage, _ = scoring._stage_features(
        run, cursor=5, boundaries=boundaries, max_runtime_seconds=None,
        decision_elapsed_seconds=9.5,
    )
    assert stage["level"] == 3 and stage["actions_on_current_level"] == 1
    assert stage["run_elapsed_seconds"] == 9.5 and stage["time_remaining_seconds"] is None
    # A run without a start level (the trainer's RunResult) starts at level 1.
    del run.start_level
    stage, levels_before = scoring._stage_features(
        run, cursor=0, boundaries=[4], max_runtime_seconds=None
    )
    assert (levels_before, stage["level"], stage["run_elapsed_seconds"]) == (0, 1, 0.0)


def test_the_worker_and_the_trainer_score_with_the_same_functions():
    for name in (
        "_cumulative_arc_scores",
        "_per_level_arc_scores",
        "_level_boundaries",
        "_action_index_ranges",
        "_stage_features",
        "_earned_arc_score",
    ):
        assert getattr(results, name) is getattr(scoring, name), name
    from harness_support.arc3 import trainer as arc3_support
    from training import platypus_rollout

    for name in ("_cumulative_arc_scores", "_level_boundaries", "_per_level_arc_scores", "_stage_features"):
        assert getattr(arc3_support, name) is getattr(scoring, name), name
    assert platypus_rollout._action_index_ranges is scoring._action_index_ranges
    assert platypus_rollout._cumulative_arc_scores is scoring._cumulative_arc_scores


def _taaf_game_run():
    """TAAF's GameRun at the ARC-3 harness pin, or a skip without it.

    The ARC-2 harness ships a package of the same name (``taaf``) with another
    GameRun; only the ARC-3 framework's scores levels.
    """
    game = pytest.importorskip("taaf.game")
    game_run = getattr(game, "GameRun", None)
    if game_run is None or not hasattr(game_run, "_compute_final_score"):
        pytest.skip("requires the ARC-3 TAAF (tufa-arc-agi-framework)")
    return game_run


@pytest.mark.parametrize("name", GAMES)
def test_every_prefix_scores_as_taaf_scores_the_game(name):
    """``GameRun._compute_final_score`` on the same evidence, for the game and
    for the game stopped after each completed level."""
    game_run = _taaf_game_run()
    levels, baselines, completed, actions = GAMES[name]

    def taaf_score(done):
        return game_run(
            game_id="fixture",
            number_of_levels=levels,
            base_actions_per_level=baselines,
            # TAAF reads the actions of completed levels only.
            actions_per_level=list(actions),
            levels_completed=done,
        )._compute_final_score()

    cumulative = scoring._cumulative_arc_scores(completed, actions, baselines, levels)
    assert (cumulative[-1] if cumulative else 0.0) == pytest.approx(taaf_score(completed))
    for done in range(1, completed + 1):
        assert cumulative[done - 1] == pytest.approx(taaf_score(done)), done
