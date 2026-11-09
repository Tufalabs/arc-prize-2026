from __future__ import annotations

from ..core import CachedProgramDslAgent

PETAL_CENTERS = {"B": (56, 3), "Y": (61, 8), "G": (59, 16), "O": (51, 16), "P": (49, 8)}

LEVEL_PROGRAMS = (
    "YB",
    "YBGB",
    "YBGOP",
    "YGBOBPGYBGOPYGBYOPG",
    "BYGBOPBYBGOPYGBPOYPGBOPYB",
    "BOYGBPOYPBGB OPOYPBGBOPBYBGOPBYGBO".replace(" ", ""),
    "BOPYBGOPBYPBGOYPBYGBPOYPGBPOYPGBOYPBGOY",
)


class FloodPaintDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Flood Paint DSL has no current level.")
        sequence = LEVEL_PROGRAMS[self._current_level_idx]
        return [(6, {"x": PETAL_CENTERS[color][0], "y": PETAL_CENTERS[color][1]}) for color in sequence]


AGENT_CLASS = FloodPaintDslAgent
