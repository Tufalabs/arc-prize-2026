from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StepSignal:
    reward: float
    return_to_go: float
    advantage: float


@dataclass(slots=True)
class AdvantageBatchStats:
    rollout_count: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_advantage: float
    std_advantage: float
