from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

_LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(RIGHT, {})] * 5 + [(UP, {})] * 4,
    [(LEFT, {})] + [(DOWN, {})] * 10,
    [(UP, {})] * 5 + [(RIGHT, {})],
]


class DodgeBallDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))

    def _build_level_program(self, _env: object) -> list[tuple[int, dict[str, int]]]:
        idx = self._current_level_idx or 0
        return _LEVEL_PROGRAMS[idx]


AGENT_CLASS = DodgeBallDslAgent
