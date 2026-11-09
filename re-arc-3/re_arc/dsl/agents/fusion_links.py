from __future__ import annotations

from ..core import CachedProgramDslAgent, camera_grid_to_display


class FusionLinksDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id, total_levels=7)

    def _build_level_program(self, env):
        game = env._game
        spec = game.current_level.get_data("spec")
        markers = {sprite.name: sprite for sprite in game.current_level.get_sprites_by_tag("fusion_bead")}
        program = []
        for link_id in spec.solution:
            marker = markers[link_id]
            x, y = camera_grid_to_display(game.camera, int(marker.x), int(marker.y))
            program.append((6, {"x": x, "y": y}))
        return program


AGENT_CLASS = FusionLinksDslAgent
