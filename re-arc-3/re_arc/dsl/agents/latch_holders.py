from __future__ import annotations

from ..core import CachedProgramDslAgent

BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 8
CELL_SIZE = 4


def _click(cell: tuple[int, int]) -> tuple[int, dict[str, int]]:
    return (6, {"x": BOARD_ORIGIN_X + (cell[0] * CELL_SIZE) + 1, "y": BOARD_ORIGIN_Y + (cell[1] * CELL_SIZE) + 1})


LEVEL_PROGRAMS: dict[int, list[tuple[int, dict[str, int]]]] = {
    0: [
        _click((2, 8)),
        _click((2, 7)),
        _click((2, 6)),
        _click((2, 5)),
        _click((4, 5)),
        _click((5, 5)),
        _click((6, 5)),
        _click((7, 5)),
        _click((8, 5)),
        _click((9, 5)),
        _click((9, 5)),
    ],
    1: [
        _click((2, 9)),
        _click((2, 8)),
        _click((2, 3)),
        _click((3, 3)),
        _click((4, 3)),
        _click((5, 3)),
        _click((6, 3)),
        _click((7, 3)),
        _click((2, 8)),
        _click((3, 8)),
        _click((4, 8)),
        _click((5, 8)),
        _click((6, 8)),
        _click((7, 8)),
        _click((8, 8)),
        _click((9, 8)),
        _click((9, 9)),
        _click((9, 9)),
    ],
    2: [
        _click((2, 9)),
        _click((2, 8)),
        _click((2, 2)),
        _click((3, 2)),
        _click((4, 2)),
        _click((5, 2)),
        _click((6, 2)),
        _click((2, 8)),
        _click((3, 8)),
        _click((4, 8)),
        _click((5, 8)),
        _click((6, 8)),
        _click((7, 8)),
        _click((8, 8)),
        _click((9, 8)),
        _click((9, 7)),
        _click((6, 2)),
        _click((6, 3)),
        _click((6, 4)),
        _click((7, 4)),
        _click((8, 4)),
        _click((9, 7)),
        _click((10, 7)),
        _click((10, 6)),
        _click((10, 5)),
        _click((10, 4)),
        _click((10, 3)),
        _click((10, 2)),
        _click((8, 4)),
        _click((9, 4)),
        _click((10, 4)),
        _click((10, 5)),
        _click((10, 6)),
        _click((10, 7)),
        _click((10, 8)),
        _click((10, 9)),
        _click((10, 9)),
    ],
}


class LatchHoldersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env):
        game = getattr(getattr(env, "_env", None), "_game", None)
        if game is None:
            raise RuntimeError("Latch Holders DSL could not access the loaded game.")
        return list(LEVEL_PROGRAMS[int(game.level_index)])


AGENT_CLASS = LatchHoldersDslAgent
