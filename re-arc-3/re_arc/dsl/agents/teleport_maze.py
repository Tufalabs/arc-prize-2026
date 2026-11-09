from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
RIGHT = 4
SPACE = 5


class TeleportMazeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _repeat(self, action_id: int, count: int) -> list[tuple[int, dict[str, int]]]:
        return [(action_id, {}) for _ in range(count)]

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.level_index

        if level == 0:
            return self._repeat(RIGHT, 6)

        if level == 1:
            return [
                (SPACE, {}),
                *self._repeat(UP, 4),
                (RIGHT, {}),
                (SPACE, {}),
                *self._repeat(DOWN, 8),
                (RIGHT, {}),
                (RIGHT, {}),
                *self._repeat(UP, 4),
            ]

        if level == 2:
            # blue→purple(1), purple→red(2), down+right to red tp→C, red→blue(1), down to blue tp→D, right to goal
            return [
                (SPACE, {}),
                (SPACE, {}),
                *self._repeat(DOWN, 4),
                *self._repeat(RIGHT, 2),
                (SPACE, {}),
                *self._repeat(DOWN, 2),
                *self._repeat(RIGHT, 3),
            ]

        raise RuntimeError(f"Unsupported teleport_maze level: {level}")


AGENT_CLASS = TeleportMazeDslAgent
