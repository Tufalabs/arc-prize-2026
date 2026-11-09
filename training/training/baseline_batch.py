"""Shared whole-game baseline batch records and validation."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from training.contracts import RolloutTrajectory
from training.rewards import AdvantageBatchStats, StepSignal


def _require_unique_rollouts(rollouts: Iterable[RolloutTrajectory]) -> None:
    seen: set[str] = set()
    for rollout in rollouts:
        if rollout.trajectory_id in seen:
            raise ValueError(
                f"duplicate rollout {rollout.trajectory_id!r} in level baseline window; "
                "each episode must contribute exactly once"
            )
        seen.add(rollout.trajectory_id)


@dataclass(slots=True)
class BaselineSignalBatch:
    """Frozen signals and diagnostics for one complete rollout window."""

    per_rollout_signals: list[list[StepSignal]]
    stats: AdvantageBatchStats
    rollouts: list[RolloutTrajectory]
    mean_value: float
    mean_target: float
    mean_raw_advantage: float
    std_raw_advantage: float
    explained_variance: float
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineUpdateStats:
    loss: float = 0.0
    update_steps: int = 0
    mean_prediction: float = 0.0
    mean_target: float = 0.0
    mean_absolute_error: float = 0.0
    explained_variance: float = 0.0
