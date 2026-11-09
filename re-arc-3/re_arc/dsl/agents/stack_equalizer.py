from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

# Button pixel coordinates (center of 4x4 buttons at y=57..60)
_LEFT_BTN = {"x": 16, "y": 59}  # click left  button → move block right→left
_RIGHT_BTN = {"x": 48, "y": 59}  # click right button → move block left→right

# Level 1:  7 vs  3 → need 2 blocks right  (left→right x2)
# Level 2: 15 vs  1 → need 7 blocks right  (left→right x7)
# Level 3:  3 vs 19 → need 8 blocks left   (right→left x8)
_LEVEL_PROGRAMS: list[list[tuple[int, dict]]] = [[(6, _RIGHT_BTN)] * 2, [(6, _RIGHT_BTN)] * 7, [(6, _LEFT_BTN)] * 8]


class StackEqualizerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))

    def _build_level_program(self, _env) -> list[tuple[int, dict]]:
        idx = self._current_level_idx or 0
        return _LEVEL_PROGRAMS[idx]


AGENT_CLASS = StackEqualizerDslAgent
