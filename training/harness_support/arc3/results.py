"""ARC-3 rewards and diagnostics produced on the harness side of the callback.

This adapter imports no training modules: the score rules are
``harness_support/arc3/scoring.py``, the one copy the trainer re-derives the
same numbers with. Raw task rewards exclude optimizer shaping penalties; the
trainer consumes the reported numbers and verifies token linkage.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from harness_support.arc3.scoring import (
    _action_index_ranges,
    _cumulative_arc_scores,
    _earned_arc_score,
    _level_boundaries,
    _per_level_arc_scores,
    _stage_features,
)


def progress_snapshot(run: Any, *, start_level: int) -> dict[str, Any]:
    """Report live task facts using the same score definition as the callback.

    This runs in the harness host after completed actions, outside the agent's
    model context. Replayed levels contribute no reward to this episode.
    """
    cumulative = _cumulative_arc_scores(
        run.levels_completed,
        run.actions_per_level,
        run.base_actions_per_level,
        run.number_of_levels,
    )
    return {
        "version": 1,
        "score": _earned_arc_score(cumulative, start_level=start_level),
        "stage": int(run.levels_completed),
        "data": {"arc_score": cumulative[-1] if cumulative else 0.0},
    }


def enrich_arc3_result(payload: dict, *, max_runtime_seconds: float | None) -> dict:
    """Attach nonce-keyed step rewards, episode return, and ARC diagnostics.

    Rewards are gains in the cumulative official game score, excluding credit
    from the initial replayed prefix. Raw level scores remain diagnostics.
    A completing action credits the response that authorized it; inspect-only
    and memory responses earn zero.
    Stages describe the state before each response, including inspect-only calls.
    """
    result = dict(payload)
    result["steps"] = [dict(step) for step in payload["steps"]]
    run = SimpleNamespace(**result)
    start_level = int(result.get("start_level", 0))
    run.start_level = start_level
    total_actions = len(run.action_wallclock_seconds)
    scores = _per_level_arc_scores(
        run.levels_completed, run.actions_per_level, run.base_actions_per_level
    )
    cumulative = _cumulative_arc_scores(
        run.levels_completed,
        run.actions_per_level,
        run.base_actions_per_level,
        run.number_of_levels,
    )
    rewards = [
        score - (cumulative[index - 1] if index else 0.0)
        for index, score in enumerate(cumulative)
        if index >= start_level
    ]
    boundaries = _level_boundaries(
        run.actions_per_level, run.levels_completed, start_level=start_level
    )
    cursor = 0
    for step, (start, end) in zip(
        run.steps, _action_index_ranges(run.steps, total_actions), strict=True
    ):
        decision_elapsed = step.get("decision_elapsed_seconds")
        if "decision_elapsed_seconds" in step and (
            type(decision_elapsed) not in (int, float)
            or not math.isfinite(decision_elapsed)
            or decision_elapsed < 0
        ):
            raise ValueError(
                "step requires a finite nonnegative decision_elapsed_seconds"
            )
        step["reward"] = float(
            sum(
                score
                for boundary, score in zip(boundaries, rewards, strict=True)
                if start <= boundary < end
            )
        )
        step["stage"], _ = _stage_features(
            run,
            cursor=cursor,
            boundaries=boundaries,
            max_runtime_seconds=max_runtime_seconds,
            decision_elapsed_seconds=decision_elapsed,
        )
        if end > start:
            cursor = end
    result["episode_return"] = _earned_arc_score(cumulative, start_level=start_level)
    result["final_arc_score"] = cumulative[-1] if cumulative else 0.0
    result["level_rewards"] = {
        str(start_level + index + 1): float(score)
        for index, score in enumerate(rewards)
    }
    result["level_scores"] = {
        str(level): float(score) for level, score in enumerate(scores, start=1)
    }
    result["level_completion_elapsed_seconds"] = {
        str(level): (
            run.action_wallclock_seconds[boundary]
            if boundary < total_actions
            else run.final_wallclock_seconds
        )
        for level, boundary in enumerate(boundaries, start=start_level + 1)
    }
    result["final_stage"], _ = _stage_features(
        run,
        cursor=cursor,
        boundaries=boundaries,
        max_runtime_seconds=max_runtime_seconds,
        decision_elapsed_seconds=run.final_wallclock_seconds,
    )
    return result
