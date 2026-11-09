from __future__ import annotations

from ..core import CachedProgramDslAgent, camera_grid_to_display


class PickAndPlaceDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env):
        game = env._game
        level = game.current_level
        blocks = level.get_data("blocks") or []
        targets = level.get_data("targets") or []

        # Build color → target index mapping
        color_to_target: dict[int, int] = {}
        for tidx, t in enumerate(targets):
            color_to_target[t["color"]] = tidx

        program: list[tuple[int, dict[str, int]]] = []
        for b in blocks:
            # Click block (first cell) to select
            bx, by = b["home"][0] + b["cells"][0][0], b["home"][1] + b["cells"][0][1]
            dx, dy = camera_grid_to_display(game.camera, bx, by)
            program.append((6, {"x": int(dx), "y": int(dy)}))

            # Click matching target (first cell) to place
            tidx = color_to_target[b["real_color"]]
            t = targets[tidx]
            tx, ty = t["origin"][0] + t["cells"][0][0], t["origin"][1] + t["cells"][0][1]
            dx, dy = camera_grid_to_display(game.camera, tx, ty)
            program.append((6, {"x": int(dx), "y": int(dy)}))

        return program


AGENT_CLASS = PickAndPlaceDslAgent
