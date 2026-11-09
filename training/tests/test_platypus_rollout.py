"""Tests for gateway-window assembly: reward assignment from level
boundaries, segment detection from token-prefix continuity, capture
verification (crc/length), task-id resolution, and the old-logprob
alignment contract build_samples enforces downstream."""

import pytest

from training.dataset import _target_aligned_logprobs
from training.platypus_rollout import (
    _cumulative_arc_scores,
    build_window_trajectories,
)
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)


def _window() -> WindowCapture:
    """One game run, 4 actions over 2 completed levels, 3 model steps.

    Step 0 authorizes actions 0-1 (completes level 0 at action 1),
    step 1 is inspect-only, step 2 authorizes actions 2-3 (completes
    level 1 at action 3) and its prompt does NOT extend step 1 (compaction).
    """
    captures = [
        StepCapture(
            run_id="run-abc",
            nonce=f"n{i}",
            prompt_token_ids=prompt,
            completion_token_ids=out,
            completion_logprobs=logprobs,
            finish_reason="stop",
            order=i,
            native_token_prompt=True,
        )
        for i, (prompt, out, logprobs) in enumerate(
            [
                ([1, 2], [3, 4], [-0.1, -0.2]),
                ([1, 2, 3, 4, 5], [6], [-0.3]),
                # Compaction: prompt no longer extends step 1's prompt+output.
                ([9, 9, 9], [7, 8], [-0.4, -0.5]),
            ]
        )
    ]
    result = RunResult(
        run_id="run-abc",
        game_id="pathfinder-0001",
        state="won",
        levels_completed=2,
        number_of_levels=2,
        actions_per_level=[2, 2],
        base_actions_per_level=[2, 4],
        final_wallclock_seconds=200.0,
        action_wallclock_seconds=[10.0, 20.0, 30.0, 40.0],
        steps=[
            {
                "nonce": capture.nonce,
                "action_index": action_index,
                "output_len": len(capture.completion_token_ids),
                "output_crc32": token_ids_crc32(capture.completion_token_ids),
                # Step 1 continued step 0's tokens; step 2 was rendered afresh.
                "continued": continued,
                "history_epoch": epoch,
                "exact_history": True,
            }
            for capture, action_index, continued, epoch in zip(
                captures, [0, None, 2], [False, True, False], [0, 0, 1]
            )
        ],
    )
    return WindowCapture(
        captures={(c.run_id, c.nonce): c for c in captures},
        results={result.run_id: result},
    )


def _build(window: WindowCapture):
    return build_window_trajectories(
        window,
        policy_iteration=7,
        max_runtime_seconds=300.0,
        game_ids=["pathfinder"],
    )


def test_rewards_land_on_the_authorizing_steps() -> None:
    (trajectory,) = _build(_window())
    scores = _cumulative_arc_scores(2, [2, 2], [2, 4], 2)
    deltas = [scores[0], scores[1] - scores[0]]
    rewards = [step.reward for step in trajectory.steps]
    # Level 0 completes at action 1 (step 0's range), level 1 at action 3
    # (step 2's range); the inspect-only step earns nothing.
    assert rewards == pytest.approx([deltas[0], 0.0, deltas[1]])
    assert trajectory.trajectory_reward == pytest.approx(scores[-1])
    assert trajectory.metadata["status"] == "COMPLETED_RUN"
    assert trajectory.metadata["rollout_policy_iteration"] == 7


def test_a_crashed_run_is_a_crashed_terminal_with_its_rewards_unchanged() -> None:
    """A game whose agent raised (on_agent_error="crash_game") keeps the level
    rewards it earned; only its status and the note say how it ended."""
    (won,) = _build(_window())
    window = _window()
    (run,) = window.results.values()
    run.state = "crashed"
    run.solver_note = "agent error:\n" + "x" * 400 + "RuntimeError: boom\n"
    (crashed,) = _build(window)
    assert [step.reward for step in crashed.steps] == [step.reward for step in won.steps]
    assert crashed.trajectory_reward == won.trajectory_reward
    assert crashed.termination_reason == crashed.metadata["status"] == "CRASHED"
    assert crashed.metadata["crash_note"] == run.solver_note[:300]
    assert crashed.metadata["crash_note_tail"].endswith("RuntimeError: boom\n")
    assert won.metadata["crash_note"] is None


def test_segment_and_stage_features() -> None:
    (trajectory,) = _build(_window())
    segments = [step.metadata["segment_start"] for step in trajectory.steps]
    # Step 1 extends step 0's tokens; step 2 broke the prefix (compaction).
    assert segments == [True, False, True]
    stages = [step.metadata["stage"] for step in trajectory.steps]
    # Decision-time state = everything executed BEFORE the call. Step 0 sees
    # a fresh game; steps 1 (inspect-only) and 2 both come after actions 0-1
    # completed level 0 at wallclock 20s — the inspect-only step must NOT
    # look like the start of the game.
    assert [s["levels_completed_before_decision"] for s in stages] == [0, 1, 1]
    assert [s["actions_before_decision"] for s in stages] == [0, 2, 2]
    assert [s["run_elapsed_seconds"] for s in stages] == pytest.approx(
        [0.0, 20.0, 20.0]
    )
    assert stages[2]["time_remaining_seconds"] == pytest.approx(280.0)
    # Efficiency features for the CURRENT level: level 1's baseline is 4
    # actions and none are spent yet -> full potential, zero ratio.
    assert stages[2]["potential_score_now"] == pytest.approx(115.0)
    assert stages[2]["action_efficiency_ratio"] == pytest.approx(0.0)
    assert stages[0]["potential_score_now"] == pytest.approx(115.0)
    assert trajectory.steps[-1].done is True


def test_efficiency_features_track_actions_spent_on_the_level() -> None:
    """An acting step that arrives after 2 of the current level's actions
    were spent sees the degraded potential and a 2/baseline ratio."""
    window = _window()
    run = window.results["run-abc"]
    # Re-map: step 0 authorizes actions 0-1 (level 0 done at action 1),
    # step 1 authorizes action 2, step 2 authorizes action 3. Level 1 has
    # baseline 4, so at step 2 one action (index 2) is spent on it.
    run.steps[1]["action_index"] = 2
    run.steps[2]["action_index"] = 3
    (trajectory,) = _build(window)
    stage = trajectory.steps[2].metadata["stage"]
    assert stage["actions_on_current_level"] == 1
    assert stage["action_efficiency_ratio"] == pytest.approx(0.25)
    assert stage["potential_score_now"] == pytest.approx(115.0)  # (4/1)^2*100 capped
    assert stage["run_elapsed_seconds"] == pytest.approx(30.0)


def test_task_id_resolves_to_configured_game() -> None:
    """TAAF's instance suffix must not leak into task_id: the trainer keys
    per-game return histories by the CONFIGURED name."""
    (trajectory,) = _build(_window())
    assert trajectory.task_id == "pathfinder"
    # The raw TAAF id stays available for joins/diagnostics.
    assert trajectory.metadata["taaf_game_id"] == "pathfinder-0001"


def test_logprobs_align_for_ppo_consumption() -> None:
    (trajectory,) = _build(_window())
    step = trajectory.steps[0]
    aligned = _target_aligned_logprobs(
        prompt_token_count=len(step.prompt_token_ids),
        completion_token_ids=step.completion_token_ids,
        completion_logprobs=step.completion_logprobs,
    )
    # Causal targets: one entry per input position except the first; the
    # completion tail carries the captured logprobs.
    assert aligned[-2:] == pytest.approx([-0.1, -0.2])


def test_unverified_capture_is_refused() -> None:
    window = _window()
    capture = window.captures[("run-abc", "n0")]
    capture.completion_token_ids = [3, 5]  # retry race kept the wrong attempt
    with pytest.raises(GatewayError, match="does not match"):
        _build(window)


def test_missing_capture_is_refused() -> None:
    window = _window()
    del window.captures[("run-abc", "n1")]
    with pytest.raises(GatewayError, match="no gateway capture"):
        _build(window)


def test_continued_step_that_does_not_nest_is_corrupted_evidence() -> None:
    window = _window()
    (result,) = window.results.values()
    result.steps[2]["continued"] = (
        True  # harness claims continuation, tokens say otherwise
    )
    with pytest.raises(GatewayError, match="does not extend"):
        _build(window)


def test_fresh_render_opens_a_segment_even_when_the_tokens_happen_to_nest() -> None:
    window = _window()
    (result,) = window.results.values()
    result.steps[1]["continued"] = False  # rendered afresh (e.g. compaction) but nests
    result.steps[1]["history_epoch"] = 1
    result.steps[2]["history_epoch"] = 2
    (trajectory,) = _build(window)
    assert [step.metadata["segment_start"] for step in trajectory.steps] == [
        True,
        True,
        True,
    ]


@pytest.mark.parametrize("field", ["continued", "history_epoch", "exact_history"])
def test_missing_continuation_evidence_is_refused(field: str) -> None:
    window = _window()
    del window.results["run-abc"].steps[1][field]
    with pytest.raises(GatewayError, match=field):
        _build(window)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("continued", "false"),
        ("continued", 1),
        ("history_epoch", True),
        ("history_epoch", 0.0),
        ("history_epoch", -1),
        ("exact_history", False),
        ("exact_history", 1),
    ],
)
def test_invalid_continuation_evidence_is_refused(field: str, value: object) -> None:
    window = _window()
    window.results["run-abc"].steps[1][field] = value
    with pytest.raises(GatewayError, match=field):
        _build(window)


def test_false_continuation_flag_without_rewrite_is_refused_even_when_tokens_nest() -> (
    None
):
    window = _window()
    window.results["run-abc"].steps[1]["continued"] = False
    with pytest.raises(GatewayError, match="no history rewrite was declared"):
        _build(window)


def test_missing_prefix_cannot_be_hidden_by_a_false_continuation_flag() -> None:
    window = _window()
    window.results["run-abc"].steps[2]["history_epoch"] = 0
    with pytest.raises(GatewayError, match="no history rewrite was declared"):
        _build(window)


def test_continuation_flag_cannot_cross_a_declared_rewrite() -> None:
    window = _window()
    window.results["run-abc"].steps[1]["history_epoch"] = 1
    with pytest.raises(GatewayError, match="contradicts the declared history rewrite"):
        _build(window)


def test_history_epoch_cannot_decrease() -> None:
    window = _window()
    for report, epoch in zip(window.results["run-abc"].steps, [2, 2, 1], strict=True):
        report["history_epoch"] = epoch
    with pytest.raises(GatewayError, match="history_epoch decreased"):
        _build(window)


def test_first_call_cannot_claim_continuation() -> None:
    window = _window()
    window.results["run-abc"].steps[0]["continued"] = True
    with pytest.raises(GatewayError, match="does not extend"):
        _build(window)


def test_continued_call_requires_native_wire_prompt_even_when_ids_nest() -> None:
    window = _window()
    window.captures[("run-abc", "n1")].native_token_prompt = False
    with pytest.raises(GatewayError, match="requires a native token-id prompt"):
        _build(window)


def test_fresh_chat_calls_are_valid_at_initial_and_declared_boundaries() -> None:
    window = _window()
    window.captures[("run-abc", "n0")].native_token_prompt = False
    window.captures[("run-abc", "n2")].native_token_prompt = False
    # More than one rewrite can happen between calls (e.g. a compaction
    # rebuild followed by an eval level transition).
    window.results["run-abc"].steps[2]["history_epoch"] = 3
    (trajectory,) = _build(window)
    assert [step.metadata["history_epoch"] for step in trajectory.steps] == [0, 0, 3]
    assert [step.metadata["segment_start"] for step in trajectory.steps] == [
        True,
        False,
        True,
    ]


def test_cumulative_scores_respect_the_official_game_cap() -> None:
    # One ultra-efficient level of a 2-level game caps at its
    # completed-weight share of GAME_SCORE_CAP (1/3 * 100), never raw 115.
    assert _cumulative_arc_scores(1, [1], [10], 2) == pytest.approx([100.0 / 3.0])
    # Full completion caps at GAME_SCORE_CAP even when every per-level
    # term hits PER_LEVEL_SCORE_CAP.
    assert _cumulative_arc_scores(2, [1, 1], [10, 10], 2)[-1] == pytest.approx(100.0)


def test_metadata_feeds_official_eval_cutoff_stats() -> None:
    from training.official_eval import _stats_at_minutes

    (trajectory,) = _build(_window())
    metadata = trajectory.metadata
    assert metadata["level_scores"] == pytest.approx({"1": 100.0, "2": 115.0})
    assert metadata["level_completion_elapsed_seconds"] == pytest.approx(
        {"1": 20.0, "2": 40.0}
    )
    # At a 30 s cutoff only level 1 (completed at 20 s) survives, and the
    # official completed-weight cap applies to the recomputed score.
    stats = _stats_at_minutes([trajectory], minutes=0.5)
    assert stats["avg_levels_completed"] == pytest.approx(1.0)
    assert stats["avg_arc_score"] == pytest.approx(100.0 / 3.0)


def _window_with_rejected_attempt() -> WindowCapture:
    """The _window() run plus one gateway capture no reported step claims:
    an attempt the harness rejected as malformed between steps 0 and 1
    (completion order 1, between step 0 at order 0 and step 1 at order 2)."""
    window = _window()
    for capture, order in zip(window.captures.values(), [0, 2, 3], strict=True):
        capture.order = order
    window.captures[("run-abc", "void1")] = StepCapture(
        run_id="run-abc",
        nonce="void1",
        prompt_token_ids=[1, 2, 3, 4, 5],
        completion_token_ids=[42, 43, 44],
        completion_logprobs=[-1.0, -1.1, -1.2],
        finish_reason="stop",
        order=1,
    )
    return window


def test_rejected_attempts_are_counted_but_not_trained_by_default() -> None:
    (trajectory,) = _build(_window_with_rejected_attempt())
    assert trajectory.metadata["void_attempts"] == 1
    assert trajectory.metadata["void_attempt_tokens"] == 3
    assert len(trajectory.steps) == 3
    assert trajectory.metadata["run_total_steps"] == 3
    assert trajectory.steps[-1].done is True


def test_unreported_attempts_are_retained_without_inventing_format_errors() -> None:
    (trajectory,) = build_window_trajectories(
        _window_with_rejected_attempt(),
        policy_iteration=7,
        max_runtime_seconds=300.0,
        game_ids=["pathfinder"],
        include_void_attempts=True,
    )
    assert len(trajectory.steps) == 4
    void = trajectory.steps[-1]
    assert void.metadata["void_attempt"] is True
    assert void.metadata["format_break"] is False
    assert void.metadata["segment_start"] is True
    assert void.reward == 0.0
    assert void.done is False
    assert void.completion_token_ids == [42, 43, 44]
    assert [entry["logprob"] for entry in void.completion_logprobs] == [
        -1.0,
        -1.1,
        -1.2,
    ]
    # It was sampled at the state of the reported step that followed it
    # (step 1: after actions 0-1), and it executed nothing.
    assert void.metadata["stage"] == trajectory.steps[1].metadata["stage"]
    assert void.metadata["executed_action_range"] == [2, 2]
    # The reported chain is untouched: its last step is still the terminal.
    assert trajectory.steps[2].done is True
    assert trajectory.metadata["run_total_steps"] == 3
    assert trajectory.metadata["void_attempts"] == 1


def test_harness_format_break_reaches_the_step_metadata() -> None:

    window = _window()
    (result,) = window.results.values()
    result.steps[1]["format_break"] = True
    (trajectory,) = _build(window)
    assert [step.metadata["format_break"] for step in trajectory.steps] == [
        False,
        True,
        False,
    ]
    # Malformed output is evidence, not an extra reward penalty.
    assert sum(step.reward for step in trajectory.steps) == pytest.approx(
        trajectory.trajectory_reward
    )


def test_critic_uses_captured_request_time_not_previous_action_time():
    window = _window()
    run = window.results["run-abc"]
    for step, clock in zip(run.steps, [2.0, 24.0, 28.0], strict=True):
        capture = window.captures[(run.run_id, step["nonce"])]
        capture.progress = {"data": {"decision_elapsed_seconds": clock}}
        step["decision_elapsed_seconds"] = clock
    (trajectory,) = _build(window)
    assert [
        step.metadata["stage"]["run_elapsed_seconds"] for step in trajectory.steps
    ] == [2.0, 24.0, 28.0]
    assert trajectory.steps[-1].metadata["stage"]["time_remaining_seconds"] == 272.0


@pytest.mark.parametrize("clock", [19.0, 201.0, float("nan"), float("inf"), True])
def test_critic_rejects_impossible_request_time(clock):
    window = _window()
    capture = window.captures[("run-abc", "n1")]
    capture.progress = {"data": {"decision_elapsed_seconds": clock}}
    window.results["run-abc"].steps[1]["decision_elapsed_seconds"] = clock
    with pytest.raises(GatewayError):
        _build(window)


def test_critic_rejects_reported_time_different_from_gateway_capture():
    window = _window()
    window.captures[("run-abc", "n1")].progress = {
        "data": {"decision_elapsed_seconds": 24.0}
    }
    window.results["run-abc"].steps[1]["decision_elapsed_seconds"] = 25.0
    with pytest.raises(GatewayError, match="decision"):
        _build(window)
