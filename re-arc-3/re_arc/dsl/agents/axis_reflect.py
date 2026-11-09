from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5

LEVEL_PROGRAMS = [
    [RIGHT, RIGHT, RIGHT, DOWN],
    [RIGHT, RIGHT, DOWN, DOWN],
    [RIGHT, RIGHT, DOWN, SPACE, RIGHT, RIGHT],
    [RIGHT, RIGHT, DOWN, DOWN, SPACE, LEFT, LEFT, LEFT, UP, UP],
    [RIGHT, RIGHT, DOWN, DOWN, SPACE, UP, UP, SPACE, DOWN, DOWN],
    [RIGHT, RIGHT, RIGHT, RIGHT, UP, UP, UP, UP],
    [RIGHT, RIGHT, UP, UP, UP, SPACE, LEFT, LEFT, UP, UP, UP, SPACE, RIGHT, RIGHT, SPACE, UP, UP, UP],
    [
        LEFT,
        LEFT,
        LEFT,
        UP,
        UP,
        UP,
        UP,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        UP,
        UP,
        UP,
        UP,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        SPACE,
        DOWN,
        DOWN,
        DOWN,
    ],
]


class AxisReflectDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        level_idx = int(env._game.level_index)
        return [(action, {}) for action in LEVEL_PROGRAMS[level_idx]]
