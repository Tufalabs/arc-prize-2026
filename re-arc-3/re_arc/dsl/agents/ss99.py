from __future__ import annotations

from ..core import CachedProgramDslAgent

CLICK = 6
SPACE = 5


class Ss99DslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=4)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level = game.level_index

        if level == 0:
            return [(CLICK, {"x": 50, "y": 31}), (SPACE, {})]

        if level == 1:
            return [(CLICK, {"x": 45, "y": 20}), (SPACE, {}), (CLICK, {"x": 50, "y": 45}), (SPACE, {})]

        if level == 2:
            return [(CLICK, {"x": 36, "y": 2}), (SPACE, {})]

        if level == 3:
            return [(CLICK, {"x": 23, "y": 62}), (SPACE, {}), (CLICK, {"x": 23, "y": 1}), (SPACE, {})]

        raise RuntimeError(f"Unsupported ss99 level: {level}")


AGENT_CLASS = Ss99DslAgent
