from __future__ import annotations

from ..core import CachedProgramDslAgent

RIGHT = {"x": 56, "y": 56}
LEFT = {"x": 8, "y": 56}
DOWN = {"x": 40, "y": 56}
PANEL_A_L1 = {"x": 27, "y": 11}
PANEL_A_L2 = {"x": 19, "y": 11}
PANEL_B_L2 = {"x": 35, "y": 27}
PANEL_A_L3 = {"x": 27, "y": 11}
PANEL_B_L3 = {"x": 35, "y": 19}

PROGRAMS = [
    [(6, PANEL_A_L1), (6, RIGHT), (6, RIGHT), (6, RIGHT), (6, RIGHT), (6, RIGHT)],
    [(6, PANEL_A_L2), (6, RIGHT), (6, PANEL_B_L2), (6, RIGHT), (6, RIGHT), (6, RIGHT)],
    [(6, PANEL_A_L3), (6, RIGHT), (6, PANEL_B_L3), (6, DOWN), (6, PANEL_A_L3), (6, DOWN), (6, LEFT), (6, LEFT)],
]


class PanelDoorDockDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env):
        level_idx = 0 if self._current_level_idx is None else int(self._current_level_idx)
        return list(PROGRAMS[level_idx])


AGENT_CLASS = PanelDoorDockDslAgent
