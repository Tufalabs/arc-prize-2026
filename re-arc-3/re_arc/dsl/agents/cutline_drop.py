from __future__ import annotations

from ..core import CachedProgramDslAgent

LEVEL_PROGRAMS: dict[int, list[tuple[int, dict[str, int]]]] = {
    0: [(6, {"x": 20, "y": 8})] + [(4, {}) for _ in range(7)],
    1: [(6, {"x": 20, "y": 8})] + [(4, {}) for _ in range(8)],
    2: [(6, {"x": 16, "y": 8}), (6, {"x": 28, "y": 8})] + [(4, {}) for _ in range(7)],
}


class CutlineDropDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "cutline_drop-0001") -> None:
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(env._game.level_index)
        return list(LEVEL_PROGRAMS[level_idx])


AGENT_CLASS = CutlineDropDslAgent
