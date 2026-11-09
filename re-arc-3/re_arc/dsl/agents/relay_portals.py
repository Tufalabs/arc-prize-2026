from __future__ import annotations

from typing import ClassVar

from ..core import CachedProgramDslAgent


class RelayPortalsDslAgent(CachedProgramDslAgent):
    PROGRAMS: ClassVar[dict[int, list[int]]] = {
        0: [4, 4, 4, 4, 4, 4, 1, 2],
        1: [4, 4, 4, 4, 4, 4, 1, 2],
        2: [3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2],
    }

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env):
        level_idx = int(getattr(self, "_current_level_idx", 0) or 0)
        solution = list(self.PROGRAMS[level_idx])
        return [(int(action_id), {}) for action_id in solution]


AGENT_CLASS = RelayPortalsDslAgent
