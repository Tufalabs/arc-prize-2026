from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = [
    "SPACE L D R R U R R",
    "D R R SPACE U L L L L L D R D R R U U U U",
    "SPACE U L L L L L D R D R R U U R D L U U U",
    "R R R D R R SPACE U L L L L L D R D R R U U U R R R D U U",
    "R R R D R R SPACE D L L U U L L L D R D R R U U R D D D R U L U L U U R R R D U U R",
    ("R R R D R R SPACE D L L U U L L L D R D R R U U R D D D R U L U L U U R R R D U U L SPACE R R"),
    ("L D D L SPACE R R U R R R SPACE U L L D D U U L L L D R D R R U U R D D D R U L U L U U R R R D U U L SPACE R R"),
    (
        "L D D L SPACE R R U R R R SPACE U L L D D U U L L L D R D R R U U "
        "R D D D R R R U L L L U L U U R R R D U R R D U U L SPACE R R"
    ),
]

ACTION_IDS = {"U": 1, "D": 2, "L": 3, "R": 4, "SPACE": 5}


class ClayFootprintsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(PROGRAMS))

    def _build_level_program(self, env):
        del env
        level_idx = int(self._current_level_idx or 0)
        return [(ACTION_IDS[token], {}) for token in PROGRAMS[level_idx].split()]


AGENT_CLASS = ClayFootprintsDslAgent
