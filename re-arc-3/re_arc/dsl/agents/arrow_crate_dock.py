from __future__ import annotations

from ..core import CachedProgramDslAgent


class ArrowCrateDockDslAgent(CachedProgramDslAgent):
    LEVELS = (
        {"crate_start": (3, 5), "dock_pos": (5, 4)},
        {"crate_start": (4, 5), "dock_pos": (1, 2)},
        {"crate_start": (1, 6), "dock_pos": (5, 3)},
    )

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env):
        level_idx = len(self._programs)
        level = self.LEVELS[level_idx]
        start_x, start_y = level["crate_start"]
        dock_x, dock_y = level["dock_pos"]
        actions: list[tuple[int, dict[str, int]]] = []

        while start_x < dock_x:
            actions.append((4, {}))
            start_x += 1
        while start_x > dock_x:
            actions.append((3, {}))
            start_x -= 1
        while start_y < dock_y:
            actions.append((2, {}))
            start_y += 1
        while start_y > dock_y:
            actions.append((1, {}))
            start_y -= 1
        return actions


AGENT_CLASS = ArrowCrateDockDslAgent
