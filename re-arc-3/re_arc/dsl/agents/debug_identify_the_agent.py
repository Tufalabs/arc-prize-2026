from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

LEVEL_PROGRAMS = ([1, 1, 4, 4], [1] * 12 + [4] * 12)


class DebugIdentifyTheAgentAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(self._current_level_idx or 0)
        return [(action_id, {}) for action_id in LEVEL_PROGRAMS[level_idx]]
