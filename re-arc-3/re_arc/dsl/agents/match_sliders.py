from __future__ import annotations

from ..core import CachedProgramDslAgent

# Per-level action programs derived from _LEVELS in matchsliders.py:
#   level 0: init_offset = -3*SCALE (-12 px) → 3 * ACTION4 (move right)
#   level 1: init_offset = +3*SCALE (+12 px) → 3 * ACTION3 (move left)
#   level 2: init_offset = -3*SCALE (-12 px) → 3 * ACTION4 (move right)
_PROGRAMS: list[list[tuple[int, dict]]] = [
    [(4, {}), (4, {}), (4, {})],
    [(3, {}), (3, {}), (3, {})],
    [(4, {}), (4, {}), (4, {})],
]


class MatchSlidersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_PROGRAMS))

    def _build_level_program(self, _env) -> list[tuple[int, dict]]:
        idx = self._current_level_idx or 0
        return list(_PROGRAMS[idx])


AGENT_CLASS = MatchSlidersDslAgent
