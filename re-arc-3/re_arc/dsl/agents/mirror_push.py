from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


class MirrorPushDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.level_index

        if level == 0:
            # L-shape from (3,10)→(4,8): UP*2 push, LEFT walk, UP walk, RIGHT push
            return [(UP, {}), (UP, {}), (LEFT, {}), (UP, {}), (RIGHT, {})]

        if level == 1:
            # Push yellow (4,7)→(4,5): UP walk, UP*2 push
            # Reposition: LEFT*2, UP*2 walk
            # Push red (3,5)→(4,5): RIGHT push
            return [(UP, {}), (UP, {}), (UP, {}), (LEFT, {}), (LEFT, {}), (UP, {}), (UP, {}), (RIGHT, {})]

        if level == 2:
            # Push F down 3: DOWN*3
            # Walk to B: RIGHT*4, DOWN*3
            # Push B left: LEFT
            # Walk to K: DOWN*5, LEFT*2
            # Push K up 2: UP*2
            return [
                *[(DOWN, {})] * 3,
                *[(RIGHT, {})] * 4,
                *[(DOWN, {})] * 3,
                (LEFT, {}),
                *[(DOWN, {})] * 5,
                *[(LEFT, {})] * 2,
                *[(UP, {})] * 2,
            ]

        raise RuntimeError(f"Unsupported mirror_push level: {level}")


AGENT_CLASS = MirrorPushDslAgent
