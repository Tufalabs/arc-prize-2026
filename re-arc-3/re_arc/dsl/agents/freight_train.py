from __future__ import annotations

from ..core import CachedProgramDslAgent

ROUTES = (
    "RR",
    "RRDRRU",
    "UURRDRU",
    "RUUULURRRRDRULUURDRRDLULDDDD",
    "LURRRRDRUUUURRRULLLUURRDDDDDRDLLLLLLLULD",
    "L UUUU L U RRRRR | LLLL DDDDD RRR UUU | UU | L U RRR | L UU RR D | DDDDDDD",
    "UUUU | U | L U RRRR | U R DDDDDDD | UUUUUUU L | L | U LL DD | RR D | LLLLL | U L DDDDDD",
)

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}


class FreightTrainDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        del env
        assert self._current_level_idx is not None
        route = ROUTES[self._current_level_idx].replace(" ", "").replace("|", "")
        return [(ACTION_BY_LETTER[letter], {}) for letter in route]


AGENT_CLASS = FreightTrainDslAgent
