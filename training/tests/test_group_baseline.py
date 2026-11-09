from __future__ import annotations

import copy
import itertools
import math

import pytest

from training.contracts import RolloutStep, RolloutTrajectory
from training.group_baseline import build_group_mean_signals, rollout_group_key


_IDS = itertools.count()


def _rollout(
    score: float,
    *,
    game: str = "ar25",
    level: int = 0,
    budget: float | None = 900.0,
    policy: int = -1,
    prompt: list[int] | None = None,
    steps: int = 1,
) -> RolloutTrajectory:
    return RolloutTrajectory(
        trajectory_id=f"rollout-{next(_IDS)}",
        task_id=game,
        steps=[
            RolloutStep(
                step_index=index,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=score if index == steps - 1 else 0.0,
                done=index == steps - 1,
                prompt_token_ids=list(prompt or [10, 20, 30]),
                completion_token_ids=[40] * (index + 1),
            )
            for index in range(steps)
        ],
        done=True,
        trajectory_reward=score,
        metadata={
            "requested_game": game,
            "start_level": level,
            "max_runtime_seconds": budget,
            "rollout_policy_iteration": policy,
            "policy_iteration": policy,
            "final_arc_score": score,
        },
    )


def _advantages(rollouts: list[RolloutTrajectory]) -> list[float]:
    return [
        steps[0].advantage
        for steps in build_group_mean_signals(rollouts).per_rollout_signals
    ]


def test_arithmetic_rollout_mean_is_scaled_once_without_std_whitening() -> None:
    rollouts = [_rollout(0), _rollout(10), _rollout(50)]
    batch = build_group_mean_signals(rollouts)
    assert _advantages(rollouts) == pytest.approx([-0.2, -0.1, 0.3])
    assert batch.mean_value == batch.mean_target == 20.0
    assert batch.mean_raw_advantage == 0.0
    assert batch.stats.std_advantage == pytest.approx(math.sqrt(0.14 / 3))
    assert batch.stats.std_advantage != pytest.approx(1.0)
    assert batch.diagnostics["group_count"] == 1.0
    assert batch.diagnostics["group_size_min"] == 3.0


def test_harder_games_are_centered_against_their_own_group() -> None:
    easy = [_rollout(80, game="easy"), _rollout(100, game="easy")]
    hard = [_rollout(0, game="hard"), _rollout(10, game="hard")]
    assert _advantages(easy + hard) == pytest.approx([-0.1, 0.1, -0.05, 0.05])
    batch = build_group_mean_signals(easy + hard)
    assert batch.diagnostics["group_count"] == 2.0
    # The harder game's successful attempt has positive credit. Its smaller
    # score difference is preserved, so game-specific variance is not erased.
    assert batch.per_rollout_signals[-1][0].advantage > 0


@pytest.mark.parametrize(
    "difference",
    [
        {"budget": 60.0},
        {"budget": None},
        {"policy": 0},
    ],
)
def test_different_starts_budgets_and_policies_do_not_share_a_mean(difference) -> None:
    rollouts = [
        _rollout(0),
        _rollout(20),
        _rollout(80, **difference),
        _rollout(100, **difference),
    ]
    assert _advantages(rollouts) == pytest.approx([-0.1, 0.1, -0.1, 0.1])
    batch = build_group_mean_signals(rollouts)
    assert batch.diagnostics["group_count"] == 2.0
    assert batch.diagnostics["singleton_episode_fraction"] == 0.0


def test_singleton_group_fails_even_when_other_games_have_enough_attempts() -> None:
    with pytest.raises(ValueError, match="at least two rollouts"):
        build_group_mean_signals(
            [_rollout(0), _rollout(100), _rollout(0, game="other")]
        )


def test_compaction_and_generated_length_do_not_change_rollout_votes() -> None:
    win = _rollout(100, steps=4)
    lose = _rollout(0)
    before = build_group_mean_signals([win, lose])
    win.steps[1].metadata.update({"compaction": True, "summary": True})
    win.steps[2].metadata["segment_start"] = True
    win.steps[2].prompt_token_ids = [111, 222]
    win.steps[3].completion_token_ids = list(range(5000))
    after = build_group_mean_signals([win, lose])
    assert before == after
    assert [signal.advantage for signal in after.per_rollout_signals[0]] == [0.5] * 4
    assert [signal.return_to_go for signal in after.per_rollout_signals[0]] == [100] * 4
    assert after.per_rollout_signals[1][0].advantage == -0.5


def test_only_this_window_contributes_no_historical_state() -> None:
    before = _advantages([_rollout(0), _rollout(20)])
    build_group_mean_signals([_rollout(100), _rollout(100)])
    after = _advantages([_rollout(0), _rollout(20)])
    assert before == after == [-0.1, 0.1]


def test_uses_terminal_arc_score_in_percentage_points_not_per_level_rewards() -> None:
    win, lose = _rollout(100), _rollout(0)
    win.metadata["level_scores"] = {"1": 115.0, "2": 115.0, "3": 115.0}
    batch = build_group_mean_signals([win, lose])
    assert [steps[0].advantage for steps in batch.per_rollout_signals] == [0.5, -0.5]
    assert batch.stats.mean_reward == 50.0
    assert batch.per_rollout_signals[0][0].return_to_go == 100.0
    assert batch.per_rollout_signals[0][0].reward == 100.0
    assert batch.diagnostics["mean_absolute_error"] == 50.0


def test_terminal_arc_score_must_match_the_gateway_reconciled_level1_return() -> None:
    rollout = _rollout(100)
    rollout.trajectory_reward = 345.0
    with pytest.raises(ValueError, match="gateway-reconciled trajectory_reward"):
        build_group_mean_signals([rollout])


def test_tiny_score_roundoff_does_not_break_reward_reconciliation() -> None:
    rollout = _rollout(10)
    rollout.trajectory_reward = 10.0 + 1e-8
    batch = build_group_mean_signals([rollout, _rollout(10)])
    assert batch.stats.mean_reward == 10.0


@pytest.mark.parametrize("value", [True, "100", float("nan"), float("inf")])
def test_malformed_gateway_reconciled_return_is_rejected(value) -> None:
    rollout = _rollout(100)
    rollout.trajectory_reward = value
    with pytest.raises(ValueError, match="trajectory_reward"):
        build_group_mean_signals([rollout])


@pytest.mark.parametrize("scores", [[0, 0], [50, 50], [100, 100]])
def test_zero_advantage_rollouts_remain_in_batch(scores) -> None:
    rollouts = [_rollout(score) for score in scores]
    batch = build_group_mean_signals(rollouts)
    assert batch.rollouts == rollouts
    assert len(batch.per_rollout_signals) == len(scores)
    assert batch.stats.rollout_count == len(scores)
    assert _advantages(rollouts) == [0.0] * len(scores)
    assert batch.diagnostics["zero_advantage_episode_fraction"] == 1.0


def test_empty_window_and_rollout_without_outputs_are_safe() -> None:
    empty = build_group_mean_signals([])
    assert empty.rollouts == empty.per_rollout_signals == []
    assert empty.stats.rollout_count == 0
    assert all(math.isfinite(value) for value in empty.diagnostics.values())
    no_output = _rollout(0, steps=0)
    other = _rollout(0, steps=0)
    batch = build_group_mean_signals([no_output, other])
    assert batch.rollouts == [no_output, other]
    assert batch.per_rollout_signals == [[], []]


def test_different_random_draw_seeds_and_run_ids_do_not_split_a_group() -> None:
    left, right = _rollout(0), _rollout(100)
    left.metadata.update({"run_id": "a", "seed": 1, "game_id": "ar25#L0#s0"})
    right.metadata.update({"run_id": "b", "seed": 2, "game_id": "ar25#L0#s1"})
    assert rollout_group_key(left) == rollout_group_key(right)
    assert _advantages([left, right]) == [-0.5, 0.5]


def test_initial_prompt_differences_do_not_split_repeated_attempts() -> None:
    left, right = _rollout(0), _rollout(100, prompt=[999])
    assert rollout_group_key(left) == rollout_group_key(right)
    assert _advantages([left, right]) == [-0.5, 0.5]


def test_attempt_without_generated_tokens_still_lowers_its_game_mean() -> None:
    no_output = _rollout(0, steps=0)
    solved = _rollout(100)
    batch = build_group_mean_signals([no_output, solved])
    assert batch.mean_value == batch.mean_target == 50.0
    assert batch.per_rollout_signals == [[], batch.per_rollout_signals[1]]
    assert batch.per_rollout_signals[1][0].advantage == 0.5
    assert batch.stats.rollout_count == 2
    assert batch.diagnostics["group_count"] == 1.0
    assert batch.diagnostics["group_size_min"] == 2.0
    assert batch.diagnostics["singleton_episode_fraction"] == 0.0


def test_requested_game_prevents_aliases_from_merging_distinct_environments() -> None:
    left, right = _rollout(0, game="ar25-aaa"), _rollout(100, game="ar25-bbb")
    left.task_id = right.task_id = "ar25"
    with pytest.raises(ValueError, match="at least two rollouts"):
        build_group_mean_signals([left, right])


@pytest.mark.parametrize(
    "score", [None, True, "50", float("nan"), float("inf"), -1, 101]
)
def test_malformed_terminal_scores_fail_loudly(score) -> None:
    rollout = _rollout(0)
    rollout.metadata["final_arc_score"] = score
    with pytest.raises(ValueError, match="final_arc_score"):
        build_group_mean_signals([rollout])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_game", "", "game ID"),
        ("start_level", True, "start_level"),
        ("start_level", -1, "start_level"),
        ("start_level", 1, "start_level"),
        ("max_runtime_seconds", float("nan"), "max_runtime_seconds"),
        ("max_runtime_seconds", 0, "runtime budget"),
        ("rollout_policy_iteration", True, "policy iteration"),
        ("rollout_policy_iteration", -2, "policy iteration"),
        ("rollout_policy_iteration", 0, "conflicting policy"),
    ],
)
def test_malformed_grouping_metadata_is_not_silently_defaulted(
    field, value, message
) -> None:
    rollout = _rollout(0)
    rollout.metadata[field] = value
    with pytest.raises(ValueError, match=message):
        build_group_mean_signals([rollout])


@pytest.mark.parametrize(
    "field", ["start_level", "max_runtime_seconds", "final_arc_score"]
)
def test_missing_required_callback_facts_fail(field) -> None:
    rollout = _rollout(0)
    del rollout.metadata[field]
    with pytest.raises(ValueError, match=field):
        build_group_mean_signals([rollout])


def test_nonfinite_step_reward_is_rejected() -> None:
    rollout = _rollout(0)
    rollout.steps[0].reward = float("nan")
    with pytest.raises(ValueError, match="step 0 reward"):
        build_group_mean_signals([rollout, _rollout(0)])


def test_duplicate_rollout_is_not_counted_twice_after_compaction() -> None:
    rollout = _rollout(100)
    duplicate = copy.deepcopy(rollout)
    duplicate.steps[0].metadata["segment_start"] = True
    with pytest.raises(ValueError, match="duplicate rollout"):
        build_group_mean_signals([rollout, duplicate])
