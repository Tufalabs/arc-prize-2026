"""Mean-only ARC-score advantages over equivalent starts in one rollout window.

Every original rollout contributes one score, regardless of its generated token
count or compaction count. The same centered score is attached to every generated
step, across ordinary compaction boundaries. This baseline has no state:
there is no critic, historical EMA, standard-deviation division or clipping.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from training.contracts import RolloutTrajectory
from training.baseline_batch import BaselineSignalBatch
from training.rewards import AdvantageBatchStats, StepSignal


ARC_SCORE_SCALE = 100.0


@dataclass(frozen=True, slots=True)
class RolloutGroupKey:
    """Only equivalent initial problems and collection conditions share a mean."""

    game_id: str
    start_level: int
    max_runtime_seconds: float | None
    policy_iteration: int


def _number(value: Any, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"group baseline {label} must be a finite number")
    return float(value)


def rollout_group_key(rollout: RolloutTrajectory) -> RolloutGroupKey:
    """Read grouping facts already recorded by the trainer's capture adapter.

    This recipe requires level-1 starts with the same harness settings and fixed
    budget across a window. Sampling seeds and initial prompt tokens are absent:
    different attempts at one game, including attempts without any output, all
    contribute to its score mean. Incidental rendering differences must not
    silently turn the window into singleton groups with no learning signal.
    """
    metadata = rollout.metadata
    label = f"rollout {rollout.trajectory_id!r}"
    if not isinstance(metadata, dict):
        raise ValueError(f"group baseline {label} metadata must be an object")
    game = metadata.get("requested_game", rollout.task_id)
    if not isinstance(game, str) or not game.strip():
        raise ValueError(f"group baseline {label} requires a game ID")
    level = metadata.get("start_level")
    if type(level) is not int or level != 0:
        raise ValueError(
            f"group baseline {label} requires start_level=0 (level-1 starts)"
        )
    if "max_runtime_seconds" not in metadata:
        raise ValueError(f"group baseline {label} requires max_runtime_seconds")
    budget = metadata["max_runtime_seconds"]
    if budget is not None:
        budget = _number(budget, label=f"{label} max_runtime_seconds")
        if budget <= 0.0:
            raise ValueError(f"group baseline {label} runtime budget must be positive")
    policy = metadata.get("rollout_policy_iteration", metadata.get("policy_iteration"))
    if type(policy) is not int or policy < -1:
        raise ValueError(f"group baseline {label} requires a policy iteration >= -1")
    if "policy_iteration" in metadata and (
        type(metadata["policy_iteration"]) is not int
        or metadata["policy_iteration"] != policy
    ):
        raise ValueError(f"group baseline {label} has conflicting policy iterations")
    return RolloutGroupKey(game.strip(), level, budget, policy)


def _moments(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def build_group_mean_signals(
    rollouts: list[RolloutTrajectory],
) -> BaselineSignalBatch:
    """Build ``(terminal ARC score - group mean) / 100`` once before packing.

    Scores are the final callback's ``final_arc_score`` in percentage points;
    ``trajectory_reward`` and sums of per-level scores are not substitutes. An
    empty window is valid. Every nonempty group requires at least two original
    attempts; constant-score groups have zero advantage. Tokenization/packing
    must not remove attempts before this mean is computed.
    """
    groups: dict[RolloutGroupKey, list[int]] = defaultdict(list)
    scores: list[float] = []
    seen: set[str] = set()
    for index, rollout in enumerate(rollouts):
        if rollout.trajectory_id in seen:
            raise ValueError(
                f"duplicate rollout {rollout.trajectory_id!r} in group baseline window; "
                "each original rollout must contribute exactly once"
            )
        seen.add(rollout.trajectory_id)
        key = rollout_group_key(rollout)
        score = _number(
            rollout.metadata.get("final_arc_score"),
            label=f"rollout {rollout.trajectory_id!r} final_arc_score",
        )
        if not 0.0 <= score <= ARC_SCORE_SCALE:
            raise ValueError("group baseline final_arc_score must be in [0, 100]")
        if key.start_level == 0:
            # The gateway reconciles trajectory_reward with live score feedback
            # and the nonce-linked step rewards. At level 1 its value must be
            # the final game ARC score, so a diagnostic cannot silently replace
            # the score that passed the gateway's reward reconciliation.
            reported_return = _number(
                rollout.trajectory_reward,
                label=f"rollout {rollout.trajectory_id!r} trajectory_reward",
            )
            if not math.isclose(score, reported_return, rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(
                    f"group baseline rollout {rollout.trajectory_id!r} final_arc_score "
                    "does not match the gateway-reconciled trajectory_reward at level 1"
                )
        scores.append(score)
        groups[key].append(index)

    baselines = [0.0] * len(rollouts)
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(
                "group baseline requires at least two rollouts per game/start/budget/policy "
                f"group; {key.game_id!r} has {len(indices)}"
            )
        baseline = math.fsum(scores[index] for index in indices) / len(indices)
        for index in indices:
            baselines[index] = baseline
    raw_advantages = [
        score - baseline for score, baseline in zip(scores, baselines, strict=True)
    ]
    advantages = [value / ARC_SCORE_SCALE for value in raw_advantages]
    signals: list[list[StepSignal]] = []
    for rollout, score, advantage in zip(rollouts, scores, advantages, strict=True):
        signals.append(
            [
                StepSignal(
                    reward=_number(
                        step.reward,
                        label=f"rollout {rollout.trajectory_id!r} step {step.step_index} reward",
                    ),
                    return_to_go=score,
                    advantage=advantage,
                )
                for step in rollout.steps
            ]
        )
    mean_score, std_score = _moments(scores)
    mean_raw, std_raw = _moments(raw_advantages)
    mean_advantage, std_advantage = _moments(advantages)
    group_sizes = [len(indices) for indices in groups.values()]
    count = len(rollouts)
    return BaselineSignalBatch(
        per_rollout_signals=signals,
        stats=AdvantageBatchStats(
            rollout_count=count,
            mean_reward=mean_score,
            std_reward=std_score,
            min_reward=min(scores, default=0.0),
            max_reward=max(scores, default=0.0),
            mean_advantage=mean_advantage,
            std_advantage=std_advantage,
        ),
        rollouts=list(rollouts),
        mean_value=_moments(baselines)[0],
        mean_target=mean_score,
        mean_raw_advantage=mean_raw,
        std_raw_advantage=std_raw,
        explained_variance=1.0 - (std_raw / std_score) ** 2 if std_score else 0.0,
        diagnostics={
            "episode_count": float(count),
            "group_count": float(len(groups)),
            "group_size_mean": _moments([float(size) for size in group_sizes])[0],
            "group_size_min": float(min(group_sizes, default=0)),
            "group_size_max": float(max(group_sizes, default=0)),
            "singleton_episode_fraction": sum(size == 1 for size in group_sizes) / count
            if count
            else 0.0,
            "zero_advantage_episode_fraction": sum(value == 0.0 for value in advantages)
            / count
            if count
            else 0.0,
            "nonzero_advantage_episode_count": float(
                sum(value != 0.0 for value in advantages)
            ),
            "raw_mean": mean_raw,
            "raw_std": std_raw,
            "mean_absolute_error": math.fsum(abs(value) for value in raw_advantages)
            / count
            if count
            else 0.0,
            "advantage_mean": mean_advantage,
            "advantage_std": std_advantage,
            "score_scale": ARC_SCORE_SCALE,
        },
    )
