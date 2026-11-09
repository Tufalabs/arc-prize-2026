from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAM_TEXT = (
    "RRR",
    "URRRRRUUUL",
    "LRRUU",
    "UUURRRRRULUURRRDDDLLDDRUUU",
    "UULLLUUUURRRDDLLLLRUULULLDDDDRRRRRULURUUUUDDD",
    "DDLLLDDDDRRRUULLLLRDDLLDLUUUURRRRRLDDDRRRLLDDRDDDD",
    "UUUUUUDRRRLDDRURRDRRURRUULDDDDRDDDLDDLULUUUULURRDDDDDD",
)

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}


class TwinShadowDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(PROGRAM_TEXT))

    def _build_level_program(self, env):
        del env
        assert self._current_level_idx is not None
        return [(ACTION_BY_LETTER[letter], {}) for letter in PROGRAM_TEXT[self._current_level_idx]]


AGENT_CLASS = TwinShadowDslAgent
