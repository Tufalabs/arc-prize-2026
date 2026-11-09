from __future__ import annotations

from ..core import CachedProgramDslAgent

CELL_SIZE = 5
CELL_PITCH = 6


class CrossPressDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        game = env._game
        spec = game.current_level.get_data("spec")
        content_width = CELL_PITCH * spec.width - 1
        content_height = CELL_PITCH * spec.height - 1
        x0 = (64 - content_width) // 2
        y0 = (64 - content_height) // 2

        program: list[tuple[int, dict[str, int]]] = []
        press_cells = spec.replay_order or tuple(
            (row, col)
            for row, mask_row in enumerate(spec.solution)
            for col, value in enumerate(mask_row)
            if value == "X"
        )
        for row, col in press_cells:
            program.append(
                (6, {"x": x0 + CELL_PITCH * col + CELL_SIZE // 2, "y": y0 + CELL_PITCH * row + CELL_SIZE // 2})
            )
        return program


AGENT_CLASS = CrossPressDslAgent
