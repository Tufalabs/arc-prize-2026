from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5

DIRECTION_ORDER = (UP, LEFT, DOWN, RIGHT)
DIRECTIONS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


def _cross(row: int, col: int, horizontal: int, vertical: int) -> list[tuple[str, int]]:
    return [("R", row)] * horizontal + [("D", col)] * vertical + [("L", row)] * horizontal + [("U", col)] * vertical


LEVEL_SHIFTS = (
    [("R", 1)],
    [("U", 0), ("D", 2)],
    _cross(1, 1, 1, 1),
    [("R", 3), *_cross(1, 1, 2, 2), *_cross(3, 1, 2, 1), *_cross(1, 2, 1, 2), *_cross(2, 2, 2, 1), ("L", 1)],
    _cross(0, 1, 1, 2)
    + _cross(3, 1, 2, 1)
    + _cross(3, 3, 1, 2)
    + _cross(1, 3, 2, 1)
    + _cross(1, 2, 1, 1)
    + _cross(4, 2, 2, 2)
    + _cross(4, 4, 1, 1)
    + _cross(0, 4, 2, 1),
    _cross(2, 0, 2, 1)
    + _cross(0, 0, 1, 2)
    + _cross(0, 3, 2, 1)
    + _cross(4, 3, 1, 2)
    + _cross(4, 1, 2, 2)
    + _cross(1, 1, 1, 1)
    + _cross(1, 4, 2, 1)
    + _cross(2, 4, 1, 2)
    + [("L", 2), ("L", 2)],
    _cross(0, 0, 2, 1)
    + _cross(2, 0, 1, 2)
    + _cross(2, 3, 2, 1)
    + _cross(4, 3, 1, 2)
    + _cross(4, 1, 2, 1)
    + _cross(1, 1, 1, 2)
    + _cross(1, 4, 2, 1)
    + _cross(3, 4, 1, 2)
    + _cross(3, 2, 2, 2)
    + _cross(0, 2, 1, 1),
)

SIZES = (3, 3, 3, 4, 5, 5, 5)
INITIAL_FOCUS = (("R", 1), ("U", 0), ("R", 1), ("R", 3), ("R", 0), ("R", 2), ("R", 0))


def _handles(size: int) -> tuple[tuple[str, int], ...]:
    return tuple(
        [("L", row) for row in range(size)]
        + [("U", col) for col in range(size)]
        + [("R", row) for row in range(size)]
        + [("D", col) for col in range(size)]
    )


def _center(size: int, handle: tuple[str, int]) -> tuple[int, int]:
    origin = (64 - 10 * size) // 2
    kind, index = handle
    if kind == "L":
        return origin - 4, origin + 10 * index + 5
    if kind == "R":
        return origin + 10 * size + 3, origin + 10 * index + 5
    if kind == "U":
        return origin + 10 * index + 5, origin - 4
    return origin + 10 * index + 5, origin + 10 * size + 3


def _neighbor(size: int, focus: tuple[str, int], action: int) -> tuple[str, int] | None:
    handles = _handles(size)
    px, py = _center(size, focus)
    dx, dy = DIRECTIONS[action]
    candidates = []
    for rank, handle in enumerate(handles):
        if handle == focus:
            continue
        qx, qy = _center(size, handle)
        forward = (qx - px) * dx + (qy - py) * dy
        if forward > 0:
            side = abs((qx - px) * -dy + (qy - py) * dx)
            candidates.append((side, forward, rank, handle))
    return min(candidates)[3] if candidates else None


def _shortest_path(size: int, start: tuple[str, int], goal: tuple[str, int]) -> list[int]:
    queue = deque([start])
    previous: dict[tuple[str, int], tuple[tuple[str, int], int] | None] = {start: None}
    while queue:
        focus = queue.popleft()
        if focus == goal:
            break
        for action in DIRECTION_ORDER:
            neighbor = _neighbor(size, focus, action)
            if neighbor is not None and neighbor not in previous:
                previous[neighbor] = (focus, action)
                queue.append(neighbor)
    actions = []
    cursor = goal
    while previous[cursor] is not None:
        parent, action = previous[cursor]
        actions.append(action)
        cursor = parent
    return list(reversed(actions))


def _program(level_index: int) -> list[int]:
    size = SIZES[level_index]
    focus = INITIAL_FOCUS[level_index]
    actions: list[int] = []
    for target in LEVEL_SHIFTS[level_index]:
        actions.extend(_shortest_path(size, focus, target))
        actions.append(SPACE)
        focus = target
    return actions


LEVEL_PROGRAMS = tuple(_program(index) for index in range(7))


class ShiftMosaicDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if level_idx != self._current_level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing levels_completed in Shift Mosaic observation")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Shift Mosaic program exhausted before reaching WIN")
        action = program[self._action_idx]
        self._action_idx += 1
        return action, {}


AGENT_CLASS = ShiftMosaicDslAgent
