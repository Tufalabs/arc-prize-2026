from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


class GravityFlipDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.level_index

        if level == 0:
            return [(LEFT, {}), (UP, {}), (RIGHT, {}), (UP, {})]

        if level == 1:
            return [
                (LEFT, {}),
                (DOWN, {}),
                (RIGHT, {}),
                (DOWN, {}),
                (RIGHT, {}),
                (DOWN, {}),
                (LEFT, {}),
                (DOWN, {}),
                (LEFT, {}),
            ]

        if level == 2:
            return [(DOWN, {}), (RIGHT, {}), (DOWN, {}), (LEFT, {}), (DOWN, {}), (LEFT, {})]

        raise RuntimeError(f"Unsupported gravity_flip level: {level}")


AGENT_CLASS = GravityFlipDslAgent
