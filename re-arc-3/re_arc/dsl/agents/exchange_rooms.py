from __future__ import annotations

from ..core import CachedProgramDslAgent

TRACE_GROUPS = (
    ("R",),
    ("R", "ULD", "RUULD"),
    ("R", "ULLD", "R"),
    ("R", "DDRRR", "UURRD", "LUR", "ULD", "LUUULL", "DDLLD"),
    ("R", "UUUURRR", "DRU", "RDDRDD", "DDRR", "ULD", "LUR", "ULLDLLL", "UULLLD"),
    ("R", "RRRRRRR", "ULD", "LUR", "DLDD", "DDDRRR", "ULD", "LUR", "ULD", "LUR", "URRUU", "UULLLD", "DLLLLLL"),
    (
        "RRRDDRR",
        "URRRRR",
        "ULLDR",
        "ULLDR",
        "DLLDD",
        "DDDRRRR",
        "ULD",
        "LUR",
        "DLLLLLL",
        "LLLLL",
        "URD",
        "RUL",
        "URD",
        "RUL",
        "UUU",
        "LUUURD",
        "RRRRR",
        "DRR",
        "DLU",
    ),
)

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}
LEVEL_PROGRAMS = [[ACTION_BY_LETTER[letter] for letter in "".join(groups)] for groups in TRACE_GROUPS]


class ExchangeRoomsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Exchange Rooms DSL has no active level.")
        return [(action, {}) for action in LEVEL_PROGRAMS[self._current_level_idx]]


AGENT_CLASS = ExchangeRoomsDslAgent
