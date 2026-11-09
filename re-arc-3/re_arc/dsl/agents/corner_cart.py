from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6


def _expand(*parts: tuple[int, int] | int) -> list[int]:
    result: list[int] = []
    for part in parts:
        if isinstance(part, tuple):
            action, count = part
            result.extend([action] * count)
        else:
            result.append(part)
    return result


LEVEL_PROGRAMS = (
    _expand(LEFT, SPACE, (UP, 3), SPACE, (RIGHT, 2)),
    _expand(CLICK, (DOWN, 4), (RIGHT, 4), (UP, 2), SPACE, (UP, 2)),
    _expand(UP, (LEFT, 4), (DOWN, 2), CLICK, (UP, 3), (RIGHT, 4), UP, SPACE, UP, RIGHT, CLICK),
    _expand(
        (UP, 2),
        (RIGHT, 3),
        SPACE,
        (DOWN, 2),
        SPACE,
        (UP, 2),
        CLICK,
        (LEFT, 4),
        (DOWN, 2),
        LEFT,
        CLICK,
        (DOWN, 4),
        CLICK,
        RIGHT,
        LEFT,
        (UP, 5),
        RIGHT,
        SPACE,
    ),
    _expand(
        (UP, 2),
        (RIGHT, 4),
        CLICK,
        RIGHT,
        (LEFT, 4),
        (DOWN, 4),
        (LEFT, 3),
        (DOWN, 4),
        (RIGHT, 3),
        SPACE,
        (UP, 2),
        (DOWN, 2),
        (LEFT, 3),
        CLICK,
        (UP, 4),
        RIGHT,
        (UP, 2),
        RIGHT,
        (SPACE, 2),
    ),
    _expand(
        (DOWN, 2),
        (LEFT, 2),
        CLICK,
        (DOWN, 3),
        CLICK,
        (RIGHT, 4),
        (UP, 3),
        CLICK,
        (DOWN, 3),
        (LEFT, 4),
        SPACE,
        (UP, 3),
        SPACE,
        RIGHT,
        (UP, 4),
        (RIGHT, 3),
        UP,
        (RIGHT, 3),
        CLICK,
        (LEFT, 4),
        (DOWN, 3),
        LEFT,
        CLICK,
    ),
    _expand(
        (UP, 3),
        LEFT,
        (SPACE, 2),
        (DOWN, 5),
        RIGHT,
        (DOWN, 3),
        UP,
        (RIGHT, 2),
        SPACE,
        (RIGHT, 5),
        (UP, 6),
        RIGHT,
        (UP, 4),
        CLICK,
        LEFT,
        CLICK,
        RIGHT,
        SPACE,
        (DOWN, 4),
        SPACE,
        LEFT,
        (DOWN, 6),
        (LEFT, 5),
        (UP, 4),
        (LEFT, 2),
    ),
)


class CornerCartDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is None:
            raise RuntimeError("Corner Cart observation has no level progress")
        self.mark_levels_solved(level_index)
        if level_index != self._level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError(f"Corner Cart program exhausted on level {level_index + 1}")
        action = program[self._action_index]
        self._action_index += 1
        if action != CLICK:
            return action, {}
        game = env._game
        row, col = game.elbow
        x = game.origin_x + (col - 1) * game.tile_size + game.tile_size // 2
        y = game.origin_y + (row - 1) * game.tile_size + game.tile_size // 2
        return CLICK, {"x": int(x), "y": int(y)}


AGENT_CLASS = CornerCartDslAgent
