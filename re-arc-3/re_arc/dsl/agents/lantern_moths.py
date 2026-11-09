from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6


def click_xy(x: int, y: int) -> tuple[str, int, int]:
    return ("click", x, y)


LEVEL_PROGRAMS = [
    [UP, SPACE],
    [("S",), UP, SPACE, SPACE, SPACE],
    [SPACE, SPACE, ("L",), RIGHT, RIGHT, RIGHT, RIGHT, UP, ("S",), DOWN, SPACE, SPACE, SPACE],
    [("E",), DOWN, DOWN, SPACE, SPACE, ("D",), LEFT, SPACE, SPACE],
    [
        click_xy(31, 38),
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        click_xy(29, 54),
        RIGHT,
        click_xy(28, 33),
        RIGHT,
        click_xy(22, 17),
        RIGHT,
        RIGHT,
        RIGHT,
        click_xy(55, 46),
        UP,
        click_xy(36, 31),
        RIGHT,
        click_xy(28, 18),
        LEFT,
        LEFT,
        LEFT,
    ],
    [
        click_xy(26, 24),
        click_xy(21, 39),
        UP,
        RIGHT,
        RIGHT,
        RIGHT,
        click_xy(26, 24),
        DOWN,
        LEFT,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        UP,
        UP,
        click_xy(34, 34),
        RIGHT,
        RIGHT,
        RIGHT,
        click_xy(21, 33),
        RIGHT,
        RIGHT,
        click_xy(16, 25),
        UP,
        click_xy(31, 33),
        RIGHT,
        RIGHT,
        RIGHT,
        UP,
        UP,
        RIGHT,
        click_xy(16, 20),
        RIGHT,
        LEFT,
        RIGHT,
        LEFT,
        LEFT,
    ],
    [
        SPACE,
        SPACE,
        SPACE,
        ("M",),
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        DOWN,
        ("U",),
        UP,
        SPACE,
        SPACE,
        SPACE,
        ("M",),
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        LEFT,
        LEFT,
        LEFT,
        SPACE,
        SPACE,
        RIGHT,
        RIGHT,
        RIGHT,
        UP,
        ("D",),
        DOWN,
        SPACE,
        SPACE,
        ("M",),
        UP,
        UP,
        LEFT,
        SPACE,
        UP,
        UP,
        LEFT,
    ],
]

LEVEL_PROGRAMS = [*LEVEL_PROGRAMS[:4], LEVEL_PROGRAMS[6], *LEVEL_PROGRAMS[4:6]]


class LanternMothsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Lantern Moths observation.")

        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Lantern Moths DSL program exhausted before reaching WIN.")

        token = program[self._action_idx]
        self._action_idx += 1
        if isinstance(token, tuple):
            if len(token) == 3 and token[0] == "click":
                return CLICK, {"x": int(token[1]), "y": int(token[2])}
            game = env._game
            item = game.movables[token[0]]
            x, y = game.cell_center_display(tuple(item["pos"]))
            return CLICK, {"x": int(x), "y": int(y)}
        return int(token), {}
