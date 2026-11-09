from __future__ import annotations

import sys

from ..core import CachedProgramDslAgent, camera_grid_to_display


class BallBounceDslAgent(CachedProgramDslAgent):
    """Replays each level's authored solution: click object, click target, press space."""

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=None)

    def _build_level_program(self, env):
        game = env._game
        mod = sys.modules[type(game).__module__]  # the game module, for grid helpers
        level = game.current_level
        objects = level.get_data("objects") or []
        solution = level.get_data("solution") or []

        def placement_click(obj_idx, tx, ty, w, h):
            # Reuse the game's own placement-cell math (raises on odd sizes), then
            # pick a click pixel that the game would actually accept: not on an
            # inert lattice line, not on a wall, and not over a different movable
            # piece (which would re-select instead of place). Fail loud rather than
            # emit a click that silently misfires.
            for cx, cy in mod.cell_click_pixels(tx, ty, w, h):
                if mod.on_grid_line(cx, cy):
                    continue
                if game._wall_pixel_at(cx, cy):
                    continue
                other = game._object_at(cx, cy)
                if other is not None and other != obj_idx:
                    continue
                return cx, cy
            raise RuntimeError(f"no placeable click pixel for object {obj_idx} at ({tx}, {ty})")

        positions = {idx: (int(obj["x"]), int(obj["y"])) for idx, obj in enumerate(objects)}
        program: list[tuple[int, dict[str, int]]] = []
        for step in solution:
            idx = int(step["object"])
            obj = objects[idx]
            w, h = int(obj["w"]), int(obj["h"])

            # Click the object's current center to select it.
            ox, oy = positions[idx]
            dx, dy = camera_grid_to_display(game.camera, ox + w // 2, oy + h // 2)
            program.append((6, {"x": int(dx), "y": int(dy)}))

            # Click inside the target cell (avoiding the inert lattice lines).
            tx, ty = int(step["pos"][0]), int(step["pos"][1])
            cx, cy = placement_click(idx, tx, ty, w, h)
            dx, dy = camera_grid_to_display(game.camera, cx, cy)
            program.append((6, {"x": int(dx), "y": int(dy)}))
            positions[idx] = (tx, ty)

        # Launch the ball.
        program.append((5, {}))
        return program


AGENT_CLASS = BallBounceDslAgent
