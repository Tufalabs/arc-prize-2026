from __future__ import annotations

from ..core import CachedProgramDslAgent

TRACE_STRINGS = (
    "UUR",
    "UULUR",
    "RRDRRUDRU",
    "RRDRRRUDRUULLLLLLUURRDRRU",
    "RLDDDRDRRRRUUUULLLLLUURURRRDDDDRDDDRUURULDDLLLUUDLU",
    "RDRRRRRRULUULLLLLLURRRRURRDDDDDLDDDLLUUULUUUUUURDDDLDR",
    "UULLLURDRRRRRRRUUULLLLLLLURDDRRRRRRDLUULUURDLLLLLUURLDDDDDDLLDRUUUUUUURRDDDDDLDRRDRRU",
)

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}


class StoneSkatersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(TRACE_STRINGS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Stone Skaters level index is unavailable.")
        return [(ACTION_BY_LETTER[letter], {}) for letter in TRACE_STRINGS[self._current_level_idx]]


AGENT_CLASS = StoneSkatersDslAgent
