from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = (
    "RR",
    "LLRR",
    "RUURRRR",
    "DDDLLRRRDDDDRRR",
    "ULUULLLDDRRRDDDDDDRRRRR",
    "RRULLDUUUURRRRRRDLLLLLDDDLLDDDD",
    "RRRRDLDDDLLUURRULLUDDRDDDDLDDLLLURDRRRRR",
)

ACTION = {"U": 1, "D": 2, "L": 3, "R": 4}


class RugShiftDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(PROGRAMS))

    def _build_level_program(self, _env):
        assert self._current_level_idx is not None
        return [(ACTION[letter], {}) for letter in PROGRAMS[self._current_level_idx]]


AGENT_CLASS = RugShiftDslAgent
