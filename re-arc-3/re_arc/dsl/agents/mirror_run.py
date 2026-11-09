from __future__ import annotations

from ..core import CachedProgramDslAgent, camera_grid_to_display

BOARD_ORIGIN = 6
CELL_SIZE = 4

SOLUTION_CELLS = (
    ((5, 6),),
    ((4, 10), (4, 4)),
    ((4, 8),),
    ((11, 2), (9, 8), (3, 8), (3, 2), (5, 2)),
    ((10, 9), (10, 2), (7, 2), (7, 5), (7, 7)),
    ((1, 10), (3, 10), (3, 1), (6, 1), (6, 5), (6, 8)),
    ((8, 2), (10, 2), (10, 9), (5, 3), (5, 9), (2, 3), (8, 9), (7, 9)),
)


class MirrorRunDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(SOLUTION_CELLS))

    def _build_level_program(self, env):
        if self._current_level_idx is None:
            raise RuntimeError("Mirror Run DSL has no active level")
        program = []
        for cell_x, cell_y in SOLUTION_CELLS[self._current_level_idx]:
            pixel_x = BOARD_ORIGIN + CELL_SIZE * cell_x + 1
            pixel_y = BOARD_ORIGIN + CELL_SIZE * cell_y + 1
            display_x, display_y = camera_grid_to_display(env._game.camera, pixel_x, pixel_y)
            program.append((6, {"x": display_x, "y": display_y}))
        return program
