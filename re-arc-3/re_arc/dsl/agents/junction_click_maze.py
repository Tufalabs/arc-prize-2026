from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

LEVEL_PROGRAMS = {
    0: [(14, 30), (30, 30), (30, 46), (14, 30)],
    1: [(14, 34), (30, 34), (46, 34), (46, 50), (14, 34)],
    2: [(14, 18), (14, 34), (30, 34), (30, 24)],
}


class JunctionClickMazeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _on_new_level(self, _env, level_idx: int):
        self._programs[level_idx] = self._build_level_program(level_idx)

    def _build_level_program(self, level_idx: int) -> list[tuple[int, dict[str, int]]]:
        clicks = LEVEL_PROGRAMS[int(level_idx)]
        return [(6, {"x": int(x), "y": int(y)}) for x, y in clicks]


AGENT_CLASS = JunctionClickMazeDslAgent
