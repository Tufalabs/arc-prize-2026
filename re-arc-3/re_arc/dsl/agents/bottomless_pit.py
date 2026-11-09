from __future__ import annotations

from ..core import CachedProgramDslAgent

TILE = 5
ORIGIN_X = 4
ORIGIN_Y = 4

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6


def _click(x: int, y: int, camera_top: int) -> tuple[int, dict[str, int]]:
    return CLICK, {"x": ORIGIN_X + x * TILE + 2, "y": ORIGIN_Y + (y - camera_top) * TILE + 2}


def _simple(action_id: int) -> tuple[int, dict[str, int]]:
    return action_id, {}


LEVEL_PROGRAMS = [
    [_simple(RIGHT) for _ in range(8)],
    [_click(5, 2, 0), _click(5, 6, 0), *[_simple(RIGHT) for _ in range(3)]],
    [_simple(RIGHT), _simple(RIGHT), _click(4, 4, 0), *[_simple(RIGHT) for _ in range(4)]],
    [_click(5, 2, 0), _click(7, 9, 0), *[_simple(RIGHT) for _ in range(4)]],
    [
        _simple(DOWN),
        _simple(DOWN),
        _click(4, 14, 6),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(6, 13, 8),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
    ],
    [_click(5, 7, 0), *[_simple(RIGHT) for _ in range(9)]],
    [
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(6, 7, 4),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(8, 6, 3),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(10, 5, 0),
        _simple(DOWN),
        _simple(DOWN),
        _click(9, 15, 6),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(LEFT),
    ],
    [
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(5, 10, 4),
        _click(6, 10, 4),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(9, 7, 4),
        _simple(DOWN),
        _click(9, 15, 7),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(10, 8, 3),
        _simple(RIGHT),
        _simple(LEFT),
    ],
    [
        _click(4, 5, 0),
        _click(4, 9, 0),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(6, 7, 3),
        _click(7, 9, 3),
        _click(9, 6, 3),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _simple(RIGHT),
        _click(10, 7, 2),
        _simple(DOWN),
        _simple(DOWN),
        _simple(DOWN),
        _click(9, 21, 11),
        _click(6, 21, 11),
        _click(6, 19, 11),
        _simple(SPACE),
        _simple(RIGHT),
        _simple(LEFT),
        _simple(LEFT),
        _simple(LEFT),
        _simple(LEFT),
        _simple(LEFT),
    ],
]


class BottomlessPitDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        level_idx = int(self._current_level_idx or 0)
        return list(LEVEL_PROGRAMS[level_idx])


AGENT_CLASS = BottomlessPitDslAgent
