from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class RolloutStep:
    step_index: int
    prompt_messages: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    reward: float
    done: bool
    prompt_token_ids: list[int] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[Any] = field(default_factory=list)
    # Only generated rows are stored; collation scatters them to causal targets.
    completion_topk_ids: np.ndarray | None = None
    completion_topk_logprobs: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    media: dict[str, Any] | None = None
    media_evidence: dict[str, Any] | None = None


@dataclass(slots=True)
class RolloutTrajectory:
    trajectory_id: str
    task_id: str
    steps: list[RolloutStep]
    done: bool = False
    trajectory_reward: float = 0.0
    termination_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
