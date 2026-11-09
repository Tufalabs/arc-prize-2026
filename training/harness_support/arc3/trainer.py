"""ARC-3 translations for the shared trainer process.

Pure Python: this runs inside the trainer and imports no harness package.
The worker-side adapter (task construction, harness API calls) is
``harness_support/arc3/worker.py``; the scoring rules, one copy shared with
the worker callback (``harness_support/arc3/results.py``), live in
``harness_support/arc3/scoring.py``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from harness_support.arc3.scoring import (
    _action_index_ranges,
    _cumulative_arc_scores,
    _level_boundaries,
    _per_level_arc_scores,
    _stage_features,
)
from training.contracts import RolloutStep, RolloutTrajectory
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    WindowCapture,
)

# The default remains text-only; an explicit Duck image mode enables vision.
MEDIA: dict[str, bool] = {"images": False}
# The worker does not echo a behavior-policy snapshot; adapter identity is
# verified per capture at the gateway.
REQUIRES_BEHAVIOR_POLICY_SNAPSHOT = False
# Run states that count as a deadline stop when the run lasted the full budget.
DEADLINE_TERMINAL_STATES = frozenset({"gave_up", "cancelled"})


def media_requirements(inference: Any) -> dict[str, bool]:
    """Pin image processing whenever the configured Duck sends grid images."""
    return {**MEDIA, "images": MEDIA["images"] or bool(inference.agent_multimodal_context)}


def evaluation_context_overrides(inference: Any, evaluation: Any) -> dict[str, int]:
    """Resize summary caps and retain the configured serving headroom.

    The summary cap and reset target follow the evaluation window as on
    ARC-2. The served ceiling is the larger of ARC-2's rule (the configured
    room above window plus summary cap) and ARC-3's own: the configured room
    above window plus reply reserve, never under the 8192 tokens validation
    requires there. ARC-2's rule alone can fall below ARC-3's minimum when an
    evaluation shrinks the window (65536 -> 32768 with a 77824 ceiling gave
    40960 < 45056)."""
    if not inference.agent_max_summary_tokens:
        return {}
    from harness_support.arc2.trainer import _resized_limits as resize
    from training.config import resolve_context_contract

    overrides = resize(inference, evaluation)
    if not overrides:
        return overrides
    context = resolve_context_contract(inference)
    room = max(
        8192,
        context.max_served_sequence_tokens
        - context.context_window_tokens
        - context.max_completion_tokens,
    )
    overrides["agent_max_served_sequence_tokens"] = max(
        overrides["agent_max_served_sequence_tokens"],
        evaluation.context_window_tokens + context.max_completion_tokens + room,
    )
    return overrides


def validate_config(config: Any) -> None:
    """ARC-3 selects games directly; datasets are an ARC-2 capability."""
    if config.inference.dataset is not None:
        raise ValueError("inference.dataset is supported only for ARC-2.")
    # ARC-3 rewards are per-level ARC-score deltas with no reply penalty: an
    # empty reply is a retained step reported as a format_break, a diagnostic
    # only (training.format_break_penalty must be 0). The ARC-2 cap is not
    # read, so a value would promise a cap ARC-3 never applies.
    if config.training.max_episode_penalty is not None:
        raise ValueError(
            "training.max_episode_penalty is an ARC-2 reward cap; ARC-3 charges "
            "no reply penalties (an empty reply is a format_break diagnostic), "
            "so leave it unset."
        )
    if config.official_eval.dataset is not None:
        raise ValueError("official_eval.dataset is supported only for ARC-2.")
    if config.inference.agent_multimodal_context and not config.training.multimodal:
        raise ValueError("ARC-3 image observations require training.multimodal=true.")
    if (
        config.inference.agent_multimodal_context
        and config.training.value_baseline == "hidden_critic"
    ):
        raise ValueError(
            "ARC-3 image observations do not support value_baseline='hidden_critic': "
            "critic feature extraction does not replay image inputs. "
            "Use training.value_baseline='group_mean' with advantage_clip=0."
        )
    from training.config import evaluation_inference_config, resolve_context_contract

    for inference in (config.inference, evaluation_inference_config(config)):
        if not inference.agent_max_summary_tokens:
            continue
        context = resolve_context_contract(inference)
        if context.max_summary_completion_tokens != 2 * (
            context.context_window_tokens // 16
        ):
            raise ValueError("ARC-3 summary cap must be twice the window/16 target")
        if context.reset_prompt_ceiling_tokens != context.compaction_trigger_tokens:
            raise ValueError("ARC-3 summaries reset to the ordinary prompt budget")
        if context.summary_forbidden_token_ids != (248058,):
            raise ValueError(
                "ARC-3 summaries require the Qwen tool-call token ban [248058]"
            )
        minimum = context.context_window_tokens + context.max_completion_tokens + 8192
        if context.max_served_sequence_tokens < minimum:
            raise ValueError(
                "ARC-3 served ceiling must fit the window, reply reserve and 8192 summary tokens"
            )


def resolve_datasets(config: Any) -> dict[str, Any] | None:
    del config
    return None


def training_identity(config: Any, manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bind image runs to their observation schedule; preserve text run identity."""
    del manifest
    if config.inference.agent_multimodal_context:
        return {
            "environment": "arc3",
            "advantage_estimator": config.training.advantage_estimator,
            "games": sorted(config.inference.games or [config.inference.game]),
            "multimodal_context": config.inference.agent_multimodal_context,
        }
    return None


def validate_run_result(run: RunResult) -> None:
    """Rewards are ordered per action; level evidence must cover completions."""
    if run.terminal_reward != 0:
        raise ValueError("ARC-3 uses ordered action rewards")
    for name in ("actions_per_level", "base_actions_per_level"):
        if len(getattr(run, name)) < run.levels_completed:
            raise ValueError(f"{name} lacks completed-level evidence")
    if run.agent_settings is not None:
        # The Duck's per-level completion-token budget the run played under:
        # null (none) or one allowance per level.
        budget = run.agent_settings.get("max_tokens_per_level", ())
        if budget is not None and (
            not isinstance(budget, list)
            or not budget
            or any(type(tokens) is not int or tokens <= 0 for tokens in budget)
        ):
            raise ValueError("ARC-3 agent_settings.max_tokens_per_level is malformed")


def rollout_identity(rollouts: Sequence[Any]) -> None:
    """ARC-3 pins no reported harness settings in its checkpoint identity; the
    rollouts still carry them (``agent_settings`` in their metadata)."""
    del rollouts
    return None


def run_to_trajectory(
    window: WindowCapture,
    run: RunResult,
    *,
    policy_iteration: int,
    max_runtime_seconds: float | None,
    game_ids: Sequence[str] | None = None,
    include_void_attempts: bool = False,
) -> RolloutTrajectory:
    """Assign each level completion's ARC-score delta to the authorizing step."""
    from training.platypus_rollout import (
        _apply_decision_time,
        _resolve_step_capture,
        _resolve_task_id,
        _status_from_state,
        _verified_segment_starts,
        _void_captures,
        crash_notes,
    )

    total_actions = len(run.action_wallclock_seconds)

    cumulative_scores = _cumulative_arc_scores(
        run.levels_completed,
        run.actions_per_level,
        run.base_actions_per_level,
        run.number_of_levels,
    )
    final_arc_score = cumulative_scores[-1] if cumulative_scores else 0.0
    per_level_scores = _per_level_arc_scores(
        run.levels_completed, run.actions_per_level, run.base_actions_per_level
    )
    level_deltas = [
        score - (cumulative_scores[index - 1] if index else 0.0)
        for index, score in enumerate(cumulative_scores)
    ]
    boundaries = _level_boundaries(run.actions_per_level, run.levels_completed)
    ranges = _action_index_ranges(run.steps, total_actions)
    captures = [
        _resolve_step_capture(window, run, step_report) for step_report in run.steps
    ]

    segment_starts = _verified_segment_starts(run, captures)

    steps: list[RolloutStep] = []
    # The decision-time state of a step is everything executed BEFORE it:
    # `cursor` counts the actions already taken when the model was called.
    # Inspect-only steps (empty action range) must see the same cursor as
    # the acting step that follows them, not "start of game".
    cursor = 0
    cursors: list[int] = []
    for position, (capture, (range_start, range_end)) in enumerate(
        zip(captures, ranges, strict=True)
    ):
        # Reward: sum of level-completion deltas whose completing action
        # falls inside this step's authored action range.
        reward = sum(
            delta
            for boundary, delta in zip(boundaries, level_deltas)
            if range_start <= boundary < range_end
        )
        if run.episode_return is not None:
            reported = run.steps[position]["reward"]
            if not math.isclose(reported, reward, rel_tol=1e-9, abs_tol=1e-7):
                raise GatewayError(
                    f"run {run.run_id}: callback reward differs from action evidence"
                )
            reward = reported
        stage, _levels_before = _stage_features(
            run,
            cursor=cursor,
            boundaries=boundaries,
            max_runtime_seconds=max_runtime_seconds,
        )
        _apply_decision_time(stage, capture, run, max_runtime_seconds)
        cursors.append(cursor)
        steps.append(
            RolloutStep(
                step_index=position,
                prompt_messages=list(capture.request_messages),
                tool_schemas=[],
                assistant_message=dict(capture.assistant_message),
                reward=float(reward),
                done=False,
                prompt_token_ids=list(capture.prompt_token_ids),
                completion_token_ids=list(capture.completion_token_ids),
                completion_topk_ids=capture.completion_topk_ids,
                completion_topk_logprobs=capture.completion_topk_logprobs,
                completion_logprobs=[
                    {"token_id": token_id, "logprob": logprob}
                    for token_id, logprob in zip(
                        capture.completion_token_ids,
                        capture.completion_logprobs,
                        strict=True,
                    )
                ],
                metadata={
                    # One-based: extract_state_critic_observations treats the
                    # contract as counting decisions from 1 (its fallback is
                    # step_index + 1 and it floors at 1).
                    "analysis_step": position + 1,
                    "segment_start": segment_starts[position],
                    "history_epoch": run.steps[position]["history_epoch"],
                    "finish_reason": capture.finish_reason,
                    "logit_bias": dict(capture.logit_bias),
                    "executed_action_range": [range_start, range_end],
                    "stage": stage,
                    # The harness's word on the reply (a kept malformed turn):
                    # the fact format_break_penalty reads.
                    "format_break": bool(
                        run.steps[position].get("format_break", False)
                    ),
                },
                media=capture.media,
                media_evidence=capture.media_evidence,
            )
        )
        if range_end > range_start:
            cursor = range_end
    if steps:
        steps[-1].done = True
    reported_steps = len(steps)

    void_captures = _void_captures(window, run)
    void_attempt_tokens = sum(
        len(capture.completion_token_ids) for capture in void_captures
    )
    if include_void_attempts:
        # A final unreported deadline draw remains in the finite-horizon
        # decision chain with zero immediate reward. It is not evidence of
        # a malformed reply; that diagnostic requires a harness report.
        accepted_orders = [capture.order for capture in captures]
        for void_index, capture in enumerate(void_captures):
            following = next(
                (
                    position
                    for position, order in enumerate(accepted_orders)
                    if order > capture.order
                ),
                None,
            )
            if following is not None:
                void_cursor = cursors[following]
            elif cursors:
                void_cursor = cursor
            else:
                void_cursor = 0
            stage, _levels_before = _stage_features(
                run,
                cursor=void_cursor,
                boundaries=boundaries,
                max_runtime_seconds=max_runtime_seconds,
            )
            _apply_decision_time(stage, capture, run, max_runtime_seconds)
            step_index = reported_steps + void_index
            steps.append(
                RolloutStep(
                    step_index=step_index,
                    prompt_messages=list(capture.request_messages),
                    tool_schemas=[],
                    assistant_message=dict(capture.assistant_message),
                    reward=0.0,
                    done=False,
                    prompt_token_ids=list(capture.prompt_token_ids),
                    completion_token_ids=list(capture.completion_token_ids),
                    completion_topk_ids=capture.completion_topk_ids,
                    completion_topk_logprobs=capture.completion_topk_logprobs,
                    completion_logprobs=[
                        {"token_id": token_id, "logprob": logprob}
                        for token_id, logprob in zip(
                            capture.completion_token_ids,
                            capture.completion_logprobs,
                            strict=True,
                        )
                    ],
                    metadata={
                        "analysis_step": step_index + 1,
                        "segment_start": True,
                        "finish_reason": capture.finish_reason,
                        "logit_bias": dict(capture.logit_bias),
                        "executed_action_range": [void_cursor, void_cursor],
                        "stage": stage,
                        "void_attempt": True,
                        "format_break": False,
                    },
                    media=capture.media,
                    media_evidence=capture.media_evidence,
                )
            )

    metadata: dict[str, Any] = {
        "game_id": run.game_id,
        "requested_game": run.game_id,
        "run_id": run.run_id,
        "trace_run_id": run.run_id,
        "status": _status_from_state(run.state),
        "levels_completed": run.levels_completed,
        "game_total_levels": run.number_of_levels,
        "final_arc_score": final_arc_score,
        "start_level": 0,
        "arc_score_deltas": {
            str(index + 1): delta for index, delta in enumerate(level_deltas)
        },
        # Per-level raw scores and completion wallclocks: official_eval's
        # _stats_at_minutes recomputes the game score at each configured
        # cutoff from exactly these two maps.
        "level_scores": {
            str(level): score for level, score in enumerate(per_level_scores, start=1)
        },
        "level_completion_elapsed_seconds": {
            str(level): (
                run.action_wallclock_seconds[boundary]
                if boundary < len(run.action_wallclock_seconds)
                else run.final_wallclock_seconds
            )
            for level, boundary in enumerate(boundaries, start=1)
        },
        "run_total_actions": total_actions,
        "run_total_steps": reported_steps,
        # Rejected attempts the harness rolled back (see _void_captures):
        # always counted, trained on only under include_void_attempts.
        "void_attempts": len(void_captures),
        "void_attempt_tokens": void_attempt_tokens,
        "max_runtime_seconds": max_runtime_seconds,
        "run_duration_seconds": run.final_wallclock_seconds,
        "policy_iteration": policy_iteration,
        # The lockstep freshness validator reads this legacy-named stamp.
        "rollout_policy_iteration": policy_iteration,
        "taaf_game_id": run.game_id,
        # What the worker's agent played under, as the harness resolved it
        # (harness_support/agent_settings.py).
        "agent_settings": run.agent_settings,
        # The frame color permutation of the whole run, if augmented
        # (harness_support/augmentation.py); level scoring ignores colors.
        "augmentation": run.augmentation,
        # A crashed run (status CRASHED, its level rewards as earned before
        # its agent raised): the harness note's head and tail; else None.
        **crash_notes(run),
    }
    task_id = _resolve_task_id(run.game_id, game_ids)
    return RolloutTrajectory(
        trajectory_id=f"{run.run_id}-{run.game_id}",
        task_id=task_id,
        steps=steps,
        done=True,
        trajectory_reward=final_arc_score,
        termination_reason=_status_from_state(run.state),
        metadata=metadata,
    )
