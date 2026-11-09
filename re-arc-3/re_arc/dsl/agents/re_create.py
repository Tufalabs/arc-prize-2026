from __future__ import annotations

from arcengine import GameAction

from ..core import CachedProgramDslAgent

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)
SPACE = int(GameAction.ACTION5.value)


class ReCreateDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = getattr(env, "_game", None)
        if game is None:
            raise RuntimeError("re_create DSL expects TransitionRewardEnv with _game")

        level = getattr(game, "current_level", None)
        if level is None:
            raise RuntimeError("re_create DSL missing current level")

        pattern = [list(row) for row in (level.get_data("pattern") or [[]])]
        rows = len(pattern)
        cols = len(pattern[0]) if pattern else 0

        program: list[tuple[int, dict[str, int]]] = []
        cx, cy = 0, 0

        # Snake-order traversal for shorter cursor paths
        for row in range(rows):
            col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
            for col in col_range:
                target = pattern[row][col]
                if target == 0:
                    continue

                # Navigate cursor to (col, row)
                while cx < col:
                    program.append((RIGHT, {}))
                    cx += 1
                while cx > col:
                    program.append((LEFT, {}))
                    cx -= 1
                while cy < row:
                    program.append((DOWN, {}))
                    cy += 1
                while cy > row:
                    program.append((UP, {}))
                    cy -= 1

                # Cycle color: press space `target` times (0 -> 1 -> ... -> target)
                for _ in range(target):
                    program.append((SPACE, {}))

        return program


AGENT_CLASS = ReCreateDslAgent
