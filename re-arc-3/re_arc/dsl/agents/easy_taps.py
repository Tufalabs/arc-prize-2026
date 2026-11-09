from __future__ import annotations

from ..core import CachedProgramDslAgent, camera_grid_to_display


class EasyTapsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=10)

    def _build_level_program(self, env):
        game = env._game
        level = game.current_level
        target = level.get_data("target")
        if target is None or len(target) != 2:
            raise RuntimeError("Missing target data in easy_taps level.")
        dx, dy = camera_grid_to_display(game.camera, int(target[0]), int(target[1]))
        return [(6, {"x": int(dx), "y": int(dy)})]


AGENT_CLASS = EasyTapsDslAgent
