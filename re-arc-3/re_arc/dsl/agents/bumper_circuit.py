from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = (
    "RD",
    "RDLUL",
    "ULDDRULD",
    "DLULULDLURDRURDLULURDULDRLDLDR",
    "LURDRULDLDLRULDLURRULULDLDLULDRURDRULDLUR",
    "LURDRULDLDRULURDRULDRDLURUURDRULDLDLULDRURDRULDLUR",
    "LURDRULDLDRUDLURDRULDRDLURULDLDLURDRULDRDLURUURDRULDLDLULDRDLUR",
)

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}


class BumperCircuitDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id, total_levels=len(PROGRAMS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None or not 0 <= self._current_level_idx < len(PROGRAMS):
            raise RuntimeError("Bumper Circuit DSL received an invalid level index.")
        return [(ACTION_BY_LETTER[letter], {}) for letter in PROGRAMS[self._current_level_idx]]


AGENT_CLASS = BumperCircuitDslAgent
