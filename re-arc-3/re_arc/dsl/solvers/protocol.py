from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")

Transition = tuple[ActionT, StateT, float]
ExpandFn = Callable[[StateT], Iterable[Transition[ActionT, StateT]]]
GoalFn = Callable[[StateT], bool]
HeuristicFn = Callable[[StateT], float]


class SolverSpec:
    """Small protocol-like container for reusable graph search inputs."""

    def __init__(
        self,
        start_state: StateT,
        is_goal: GoalFn[StateT],
        expand: ExpandFn[ActionT, StateT],
        heuristic: HeuristicFn[StateT] | None = None,
    ):
        self.start_state = start_state
        self.is_goal = is_goal
        self.expand = expand
        self.heuristic = heuristic
