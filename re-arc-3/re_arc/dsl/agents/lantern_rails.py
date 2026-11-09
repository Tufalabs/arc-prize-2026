from __future__ import annotations

from ..core import DslAgent, observation_level_index

SPACE = 5
CLICK = 6


def _click_cell(x: int, y: int) -> tuple[int, dict[str, int]]:
    return CLICK, {"x": 4 + x * 5, "y": 4 + y * 5}


def _a(action_id: int, count: int = 1) -> list[tuple[int, dict[str, int]]]:
    return [(action_id, {}) for _ in range(count)]


LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [_click_cell(0, 4), *_a(SPACE, 3), *_a(2), *_a(SPACE), *_a(2), *_a(SPACE)],
    [*_a(SPACE, 3), _click_cell(4, 0), *_a(4), *_a(SPACE, 3), _click_cell(0, 3), *_a(2, 3), *_a(SPACE, 2)],
    [_click_cell(5, 0), *_a(3), *_a(SPACE, 5), *_a(4), *_a(SPACE, 9)],
    [
        *_a(SPACE, 2),
        _click_cell(3, 0),
        *_a(4),
        *_a(SPACE, 2),
        _click_cell(0, 5),
        *_a(1, 2),
        *_a(SPACE, 2),
        _click_cell(4, 0),
        *_a(3, 2),
        *_a(SPACE, 2),
        _click_cell(0, 3),
        *_a(2, 2),
        *_a(SPACE, 4),
        _click_cell(2, 0),
        *_a(4, 4),
        *_a(SPACE, 2),
    ],
    [
        *_a(SPACE),
        *_a(1),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 4),
        _click_cell(4, 0),
        *_a(4, 4),
        *_a(SPACE, 3),
        _click_cell(0, 3),
        *_a(2, 3),
        *_a(SPACE, 2),
        *_a(1),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE, 3),
        _click_cell(8, 0),
        *_a(3),
        *_a(SPACE, 2),
        _click_cell(0, 7),
        *_a(2, 2),
        *_a(SPACE, 3),
    ],
    [
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 4),
        *_a(1),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 2),
        *_a(1),
        *_a(SPACE, 3),
        *_a(2, 4),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE, 5),
        *_a(2),
        *_a(SPACE, 3),
        *_a(1, 3),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 6),
        *_a(1),
        *_a(SPACE, 2),
    ],
    [
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 4),
        *_a(1),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 4),
        *_a(2, 3),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE, 4),
        _click_cell(6, 0),
        *_a(4),
        *_a(SPACE, 2),
        _click_cell(0, 8),
        *_a(2, 2),
        *_a(SPACE, 3),
        *_a(1, 4),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 6),
        _click_cell(7, 0),
        *_a(4, 2),
        *_a(SPACE, 2),
    ],
    [
        _click_cell(6, 0),
        *_a(3),
        *_a(SPACE),
        _click_cell(0, 6),
        *_a(1),
        *_a(SPACE, 4),
        *_a(1),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 4),
        *_a(2, 3),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE),
        *_a(2),
        *_a(SPACE, 4),
        _click_cell(5, 0),
        *_a(4, 2),
        *_a(SPACE, 2),
        _click_cell(0, 8),
        *_a(2, 2),
        *_a(SPACE, 3),
        _click_cell(7, 0),
        *_a(3),
        *_a(SPACE, 9),
        _click_cell(0, 10),
        *_a(1, 4),
        *_a(SPACE),
        *_a(1),
        *_a(SPACE, 6),
        _click_cell(6, 0),
        *_a(4, 3),
        *_a(SPACE, 2),
    ],
]


class LanternRailsAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, env: object, observation: object) -> tuple[int, dict[str, int]]:
        del env
        level_idx = observation_level_index(observation, len(LEVEL_PROGRAMS))
        if level_idx is None:
            raise RuntimeError("Missing level index in Lantern Rails observation.")
        if level_idx != self._level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
        program = LEVEL_PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Lantern Rails DSL exhausted on level {level_idx}.")
        action = program[self._action_idx]
        self._action_idx += 1
        return action


AGENT_CLASS = LanternRailsAgent
