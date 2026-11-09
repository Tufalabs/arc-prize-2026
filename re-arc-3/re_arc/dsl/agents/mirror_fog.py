from __future__ import annotations

from ..core import DslAgent, observation_level_index

CELL = 5
BOARD_ORIGIN = 2

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6


def _click_data(cell_x: int, cell_y: int) -> dict[str, int]:
    return {"x": BOARD_ORIGIN + cell_x * CELL + 2, "y": BOARD_ORIGIN + cell_y * CELL + 2}


LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(action, {}) for action in [SPACE, LEFT, UP, RIGHT, RIGHT, SPACE, RIGHT, SPACE]],
    [
        (action, {})
        for action in [
            SPACE,
            DOWN,
            SPACE,
            UP,
            LEFT,
            UP,
            RIGHT,
            RIGHT,
            RIGHT,
            SPACE,
            UP,
            RIGHT,
            RIGHT,
            DOWN,
            LEFT,
            LEFT,
            LEFT,
        ]
    ],
    [
        (action, {})
        for action in [
            UP,
            UP,
            UP,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            LEFT,
            DOWN,
            DOWN,
            RIGHT,
            RIGHT,
            RIGHT,
            RIGHT,
            SPACE,
            UP,
            UP,
            UP,
            RIGHT,
            SPACE,
        ]
    ],
    [
        (action, {})
        for action in [
            UP,
            RIGHT,
            DOWN,
            RIGHT,
            UP,
            LEFT,
            UP,
            RIGHT,
            RIGHT,
            RIGHT,
            RIGHT,
            DOWN,
            DOWN,
            DOWN,
            RIGHT,
            SPACE,
            UP,
            UP,
            UP,
            LEFT,
            LEFT,
            DOWN,
            LEFT,
            LEFT,
            LEFT,
            UP,
        ]
    ],
    [
        (CLICK, _click_data(7, 5)),
        *[
            (action, {})
            for action in [
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                DOWN,
                DOWN,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                SPACE,
                UP,
                RIGHT,
                RIGHT,
                RIGHT,
                DOWN,
                RIGHT,
                SPACE,
                DOWN,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                DOWN,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                SPACE,
                UP,
                LEFT,
                LEFT,
                LEFT,
                UP,
                UP,
                UP,
                UP,
                RIGHT,
            ]
        ],
        (CLICK, _click_data(7, 5)),
        (SPACE, {}),
    ],
    [
        (CLICK, _click_data(7, 5)),
        (CLICK, _click_data(8, 9)),
        *[
            (action, {})
            for action in [
                DOWN,
                DOWN,
                DOWN,
                RIGHT,
                DOWN,
                RIGHT,
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
                LEFT,
                UP,
                LEFT,
                LEFT,
                DOWN,
                DOWN,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                SPACE,
                UP,
                UP,
                UP,
                LEFT,
                UP,
                UP,
                RIGHT,
            ]
        ],
        (CLICK, _click_data(7, 5)),
        (SPACE, {}),
    ],
    [
        (CLICK, _click_data(7, 5)),
        (UP, {}),
        (CLICK, _click_data(8, 9)),
        *[
            (action, {})
            for action in [
                RIGHT,
                RIGHT,
                RIGHT,
                DOWN,
                DOWN,
                DOWN,
                DOWN,
                RIGHT,
                UP,
                UP,
                UP,
                SPACE,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                DOWN,
                DOWN,
                DOWN,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                SPACE,
                UP,
                UP,
                UP,
                LEFT,
                UP,
                UP,
                RIGHT,
            ]
        ],
        (CLICK, _click_data(7, 5)),
        *[(action, {}) for action in [SPACE, DOWN, LEFT, LEFT, UP, DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, UP]],
    ],
    [
        (CLICK, _click_data(7, 5)),
        (UP, {}),
        (CLICK, _click_data(8, 9)),
        *[
            (action, {})
            for action in [
                RIGHT,
                RIGHT,
                RIGHT,
                DOWN,
                DOWN,
                DOWN,
                DOWN,
                RIGHT,
                UP,
                UP,
                UP,
                SPACE,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                LEFT,
                DOWN,
                DOWN,
                DOWN,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                RIGHT,
                SPACE,
                UP,
                LEFT,
                LEFT,
                UP,
                LEFT,
                DOWN,
                SPACE,
                RIGHT,
                RIGHT,
                RIGHT,
                UP,
                UP,
                LEFT,
                UP,
                UP,
                RIGHT,
            ]
        ],
        (CLICK, _click_data(7, 5)),
        *[(action, {}) for action in [SPACE, DOWN, LEFT, LEFT, UP, DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, UP]],
    ],
]


class MirrorFogDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
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
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Mirror Fog observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Mirror Fog DSL program exhausted before WIN.")
        action_id, data = program[self._action_idx]
        self._action_idx += 1
        return action_id, data
