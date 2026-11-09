from __future__ import annotations

import re

from ..core import CachedProgramDslAgent
from ..solvers.grid import grid_to_display_click

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
FIRE = 5
CLICK = 6


class GridArcheryDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _repeat(self, action_id: int, count: int) -> list[tuple[int, dict[str, int]]]:
        return [(action_id, {}) for _ in range(max(0, int(count)))]

    def _wait(self, count: int) -> list[tuple[int, dict[str, int]]]:
        return [(CLICK, {"x": -1, "y": -1}) for _ in range(max(0, int(count)))]

    def _level_no(self, env) -> int:
        name = str(getattr(env._game.current_level, "name", ""))
        match = re.search(r"Level\s+(\d+)", name)
        if not match:
            raise RuntimeError(f"Could not parse grid_archery level from name={name!r}.")
        return int(match.group(1))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_no = self._level_no(env)

        program: list[tuple[int, dict[str, int]]] = []

        if level_no == 1:
            program += self._repeat(UP, 7)
            program += self._repeat(RIGHT, 9)
            program += [(FIRE, {})]
            program += self._repeat(DOWN, 2)
            program += self._repeat(RIGHT, 12)
            program += self._repeat(UP, 2)
            return program

        if level_no == 2:
            program += self._repeat(RIGHT, 10)
            program += self._repeat(UP, 6)
            program += [(FIRE, {})]
            program += self._wait(1)
            program += self._repeat(DOWN, 5)
            program += self._repeat(RIGHT, 9)
            return program

        if level_no == 3:
            program += self._repeat(UP, 7)
            program += self._repeat(RIGHT, 3)
            program += self._repeat(UP, 1)
            program += [(FIRE, {})]
            program += self._wait(2)
            program += self._repeat(DOWN, 4)
            program += self._repeat(RIGHT, 7)
            program += [(FIRE, {})]
            program += self._repeat(RIGHT, 3)
            program += self._repeat(UP, 1)
            program += self._repeat(RIGHT, 3)
            program += [(FIRE, {})]
            program += self._repeat(DOWN, 4)
            program += self._repeat(RIGHT, 6)
            return program

        if level_no == 4:
            mirror_click = grid_to_display_click(game.camera, (12, 14))
            program += [(CLICK, mirror_click)]
            program += [(FIRE, {})]
            program += self._wait(10)
            program += self._repeat(RIGHT, 19)
            program += self._repeat(DOWN, 3)
            return program

        if level_no == 5:
            program += self._repeat(UP, 8)
            program += self._repeat(RIGHT, 3)
            program += self._repeat(UP, 1)
            program += [(FIRE, {})]
            program += self._repeat(DOWN, 3)
            program += self._repeat(DOWN, 1)
            program += self._wait(2)
            program += self._repeat(RIGHT, 11)
            program += self._repeat(UP, 6)
            program += self._repeat(RIGHT, 1)
            program += [(FIRE, {})]
            program += self._wait(1)
            program += self._repeat(LEFT, 1)
            program += self._repeat(DOWN, 10)
            program += self._repeat(RIGHT, 8)
            return program

        if level_no == 6:
            mirror_click = grid_to_display_click(game.camera, (24, 14))
            program += self._repeat(UP, 6)
            program += self._repeat(RIGHT, 3)
            program += self._repeat(UP, 1)
            program += [(FIRE, {})]
            program += self._wait(1)
            program += self._repeat(DOWN, 2)
            program += self._repeat(RIGHT, 6)
            program += [(FIRE, {})]
            program += self._repeat(RIGHT, 4)
            program += self._repeat(UP, 4)
            program += self._repeat(LEFT, 1)
            program += self._repeat(RIGHT, 1)
            program += [(FIRE, {})]
            program += self._wait(1)
            program += self._repeat(DOWN, 7)
            program += self._repeat(RIGHT, 2)
            program += [(CLICK, mirror_click)]
            program += [(FIRE, {})]
            program += self._wait(3)
            program += self._repeat(DOWN, 3)
            program += self._repeat(RIGHT, 6)
            return program

        raise RuntimeError(f"Unsupported grid_archery level: {level_no}")


AGENT_CLASS = GridArcheryDslAgent
