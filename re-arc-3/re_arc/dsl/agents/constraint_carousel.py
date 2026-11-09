from __future__ import annotations

from ..core import CachedProgramDslAgent, camera_grid_to_display


class ConstraintCarouselDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        game = env._game
        program = []
        for row, col in game.spec.solution:
            grid_x = game.x0 + 10 * col + 6
            grid_y = game.y0 + 10 * row + 6
            display_x, display_y = camera_grid_to_display(game.camera, grid_x, grid_y)
            program.append((6, {"x": int(display_x), "y": int(display_y)}))
        return program


AGENT_CLASS = ConstraintCarouselDslAgent
