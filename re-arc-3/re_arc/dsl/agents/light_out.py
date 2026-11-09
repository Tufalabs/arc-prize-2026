from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5

# SPACE triggers flash, then FLASH_FRAMES=2 extra step() calls consume actions.
# Pad with SPACE after to fill those frames (they are no-ops during flash).
_FLASH = [(SPACE, {}), (SPACE, {}), (SPACE, {})]


class LightOutDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.level_index

        if level == 0:
            # (3,12)->(12,3): flash, right 9, up 9
            return _FLASH + [(RIGHT, {})] * 9 + [(UP, {})] * 9

        if level == 1:
            # (3,3)->(7,13): flash, right 4, down 10
            return _FLASH + [(RIGHT, {})] * 4 + [(DOWN, {})] * 10

        if level == 2:
            # (2,2)->(11,12): flash then BFS path
            return [
                *_FLASH,
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (UP, {}),
                (UP, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (RIGHT, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (DOWN, {}),
                (RIGHT, {}),
            ]

        raise RuntimeError(f"Unsupported light_out level: {level}")


AGENT_CLASS = LightOutDslAgent
