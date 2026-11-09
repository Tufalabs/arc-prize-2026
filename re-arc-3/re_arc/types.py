from collections.abc import Callable, Mapping
from typing import SupportsIndex, SupportsInt, TypedDict

import numpy as np
from arc_agi import EnvironmentWrapper  # type: ignore[import-untyped]
from arcengine import FrameDataRaw, GameAction

type Action = GameAction
type Observation = FrameDataRaw
type Environment = EnvironmentWrapper
type FrameLayer = np.ndarray
type Frame = FrameLayer | list[FrameLayer]

type ActionValue = int | float | str | bool | None
type ActionData = Mapping[str, ActionValue]
type IntCoercible = str | bytes | bytearray | SupportsInt | SupportsIndex
type ActionRawValue = IntCoercible | tuple[IntCoercible, type]


class StepInfo(TypedDict):
    game_id: str
    level_transition: bool
    level_reset: bool
    reset_ignored: bool
    current_level: int
    num_levels: int
    actions_in_level: int
    episode_reward: float


type StepResult = tuple[Observation, float, bool, StepInfo]
type InnerStepResult = tuple[Observation, float, bool, Mapping[str, ActionValue]]
type Renderer = Callable[[int, Observation], None]
