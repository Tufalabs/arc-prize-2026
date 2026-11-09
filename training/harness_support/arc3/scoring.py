"""ARC-3 scoring: the official game score and its attribution to steps.

The one copy both sides use. The rollout worker's callback
(``harness_support/arc3/results.py``, in the harness environment) computes
step rewards, live progress and stage features with it; the trainer
(``harness_support/arc3/trainer.py``, ``training/platypus_rollout.py``,
``training/official_eval.py``) re-derives the same numbers from the reported
evidence and refuses a disagreement. Standard library only, so it loads in
either environment. The game score mirrors TAAF's
``GameRun._compute_final_score`` (arc_agi's ``EnvironmentScoreCalculator``
v0.9.8); ``tests/test_arc3_scoring.py`` compares the two in the harness
environment. Per-prefix scores, ``start_level`` crediting and stage features
are training's own.
"""

from __future__ import annotations

import math
from typing import Any

PER_LEVEL_SCORE_CAP = 115.0
GAME_SCORE_CAP = 100.0


def compute_game_score(level_scores: list[tuple[int, float]]) -> float:
    """Official weighted score of ``(level, score)`` pairs, levels 1-based.

    The level-weighted mean, capped at the completed levels' share of the
    total weight times ``GAME_SCORE_CAP`` (a level counts as completed when
    it scored above zero).
    """
    total_weight = sum(level for level, _score in level_scores)
    if total_weight <= 0:
        return 0.0
    weighted_score = sum(level * score for level, score in level_scores) / total_weight
    completed_weight = sum(level for level, score in level_scores if score > 0.0)
    return min(weighted_score, completed_weight / total_weight * GAME_SCORE_CAP)


def arc_game_score_for_completed_levels(
    completed_level_scores: dict[int, float],
    *,
    total_levels: float,
) -> float:
    """Official score of a game with these levels' raw scores, the rest zero."""
    max_level = max(0, int(math.ceil(total_levels)))
    return compute_game_score(
        [
            (level, float(completed_level_scores.get(level, 0.0)))
            for level in range(1, max_level + 1)
        ]
    )


def _per_level_arc_scores(
    levels_completed: int,
    actions_per_level: list[int],
    baselines: list[int] | None,
) -> list[float]:
    """Raw v0.9.8 per-level scores for completed levels.

    Per completed level: ``min(PER_LEVEL_SCORE_CAP, (baseline/actions)^2 *
    100)``, unweighted; the game-level weighting and cap live in
    ``compute_game_score``.
    """
    scores: list[float] = []
    for index in range(max(0, levels_completed)):
        actions = actions_per_level[index] if index < len(actions_per_level) else 0
        baseline = (
            baselines[index] if baselines is not None and index < len(baselines) else 0
        )
        if actions > 0 and baseline > 0:
            scores.append(min(PER_LEVEL_SCORE_CAP, (baseline / actions) ** 2 * 100.0))
        else:
            scores.append(0.0)
    return scores


def _cumulative_arc_scores(
    levels_completed: int,
    actions_per_level: list[int],
    baselines: list[int] | None,
    number_of_levels: int,
) -> list[float]:
    """Official ARC score after each completed level.

    Every prefix is scored by ``compute_game_score``, so the completed-weight
    cap applies to each cumulative value: an over-efficient level contributes
    at most its weight share of ``GAME_SCORE_CAP``, never a raw
    ``PER_LEVEL_SCORE_CAP``.
    """
    per_level = _per_level_arc_scores(levels_completed, actions_per_level, baselines)
    if not per_level:
        return []
    total_levels = max(number_of_levels, levels_completed, 1)
    return [
        compute_game_score(
            [
                (level, per_level[level - 1] if level <= prefix else 0.0)
                for level in range(1, total_levels + 1)
            ]
        )
        for prefix in range(1, len(per_level) + 1)
    ]


def _earned_arc_score(cumulative: list[float], *, start_level: int) -> float:
    """Official score gained after the initial replayed prefix."""
    if len(cumulative) <= start_level:
        return 0.0
    initial = cumulative[start_level - 1] if start_level else 0.0
    return float(cumulative[-1] - initial)


def _level_boundaries(
    actions_per_level: list[int], levels_completed: int, *, start_level: int = 0
) -> list[int]:
    """Index of the action that completed each level the run played
    (levels ``start_level`` .. ``levels_completed - 1``; 0-based, cumulative
    over the run's own actions)."""
    boundaries: list[int] = []
    running = 0
    for index in range(max(0, start_level), levels_completed):
        count = actions_per_level[index] if index < len(actions_per_level) else 0
        running += count
        boundaries.append(max(0, running - 1))
    return boundaries


def _action_index_ranges(
    steps: list[dict[str, Any]], total_actions: int
) -> list[tuple[int, int]]:
    """Half-open executed-action range [start, end) authored by each step.

    Steps carry the index of the FIRST action they authorized (or None
    for inspect-only calls); a step's batch extends to the next step's
    first action.
    """
    starts = [step.get("action_index") for step in steps]
    ranges: list[tuple[int, int]] = []
    for position, start in enumerate(starts):
        if start is None:
            ranges.append((0, 0))
            continue
        end = total_actions
        for later in starts[position + 1 :]:
            if later is not None:
                end = int(later)
                break
        ranges.append((int(start), max(int(start), end)))
    return ranges


def _stage_features(
    run: Any,
    *,
    cursor: int,
    boundaries: list[int],
    max_runtime_seconds: float | None,
    decision_elapsed_seconds: float | None = None,
) -> tuple[dict[str, Any], int]:
    """Decision-time state after ``cursor`` executed actions (critic stage
    features), plus the number of levels completed before that point.

    Levels are absolute: a run that started at level k (``run.start_level``,
    0 when the run has none) stands on level k + 1 (1-based) before it
    completes anything. ``decision_elapsed_seconds`` is the generation-time
    clock a step reports; without it, the wallclock of the last action
    executed before the decision stands in (traces that predate it).
    """
    start_level = max(0, int(getattr(run, "start_level", 0)))
    levels_before = start_level + sum(1 for boundary in boundaries if boundary < cursor)
    actions_before_level = sum(run.actions_per_level[start_level:levels_before])
    actions_on_current_level = max(0, cursor - actions_before_level)
    current_baseline = (
        run.base_actions_per_level[levels_before]
        if levels_before < len(run.base_actions_per_level)
        else 0
    )
    if current_baseline > 0:
        # Per-level score if the current level completed right now
        # (v0.9.8 per-level term, uncapped by weighting), and how far
        # past the baseline the level already is: the critic's
        # efficiency features.
        potential_score_now = min(
            PER_LEVEL_SCORE_CAP,
            (current_baseline / max(1, actions_on_current_level)) ** 2 * 100.0,
        )
        action_efficiency_ratio = actions_on_current_level / current_baseline
    else:
        potential_score_now = 0.0
        action_efficiency_ratio = 0.0
    elapsed = (
        decision_elapsed_seconds
        if decision_elapsed_seconds is not None
        else run.action_wallclock_seconds[cursor - 1]
        if cursor > 0
        else 0.0
    )
    remaining = (
        max(0.0, float(max_runtime_seconds) - elapsed)
        if max_runtime_seconds is not None
        else None
    )
    stage = {
        "level": levels_before + 1,
        "levels_completed_before_decision": levels_before,
        "levels_remaining": max(0, run.number_of_levels - levels_before),
        "actions_before_decision": cursor,
        "actions_on_current_level": actions_on_current_level,
        "potential_score_now": potential_score_now,
        "action_efficiency_ratio": action_efficiency_ratio,
        "run_elapsed_seconds": elapsed,
        "time_remaining_seconds": remaining,
    }
    return stage, levels_before
