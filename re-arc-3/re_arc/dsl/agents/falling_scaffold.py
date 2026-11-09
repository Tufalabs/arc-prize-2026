from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = (
    "RU",
    "SRU",
    "RRSRRRRR",
    "SRRUUULLSLLLL",
    "SRRRLLLSLURRRUUULLLLLRSRR",
    "LLLLLRSLSLUUUUUURRRRRSRRRR",
    "SRRRRRRUUULLLSLUULLLLRSLLLLSLUUUUUURRRRRRSSRRRDLSLL",
)

ACTION_IDS = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}


class FallingScaffoldDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id, total_levels=len(PROGRAMS))

    def _build_level_program(self, env):
        del env
        assert self._current_level_idx is not None
        return [(ACTION_IDS[token], {}) for token in PROGRAMS[self._current_level_idx]]


AGENT_CLASS = FallingScaffoldDslAgent
