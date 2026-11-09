from __future__ import annotations

import math

from ..core import CachedProgramDslAgent

CLICK = 6


def _walk_clicks(
    start_x: int, start_y: int, target_x: int, target_y: int, step: int = 3
) -> list[tuple[int, dict[str, int]]]:
    clicks: list[tuple[int, dict[str, int]]] = []
    cx, cy = float(start_x), float(start_y)
    while True:
        dx = target_x - cx
        dy = target_y - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 2.0:
            break
        if dist <= step:
            clicks.append((CLICK, {"x": target_x, "y": target_y}))
            break
        scale = step / dist
        nx = round(cx + dx * scale)
        ny = round(cy + dy * scale)
        clicks.append((CLICK, {"x": nx, "y": ny}))
        cx, cy = float(nx), float(ny)
    return clicks


class HomeFindingDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._moved_agents: set[int] = set()

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        agents = level.get_data("agents") or []
        circles = level.get_data("circles") or []

        pairs: list[tuple[dict, dict]] = []
        used_circles: set[int] = set()
        for agent in agents:
            for ci, circle in enumerate(circles):
                if ci not in used_circles and circle["color"] == agent["color"]:
                    pairs.append((agent, circle))
                    used_circles.add(ci)
                    break

        program: list[tuple[int, dict[str, int]]] = []
        for agent, circle in pairs:
            ax, ay = int(agent["x"]) + 1, int(agent["y"]) + 1
            tx, ty = int(circle["cx"]), int(circle["cy"])
            program.extend(_walk_clicks(ax, ay, tx, ty))
        return program


AGENT_CLASS = HomeFindingDslAgent
