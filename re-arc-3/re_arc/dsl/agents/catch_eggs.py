from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

DOWN = 2
LEFT = 3
RIGHT = 4

_LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(LEFT, {})] * 3 + [(DOWN, {})] + [(RIGHT, {})] * 5 + [(DOWN, {})] + [(LEFT, {})] * 2,
    [(DOWN, {})] * 5 + [(RIGHT, {})] * 2 + [(DOWN, {})] * 6,
    [(LEFT, {})] + [(DOWN, {})] * 5 + [(RIGHT, {})] * 2 + [(DOWN, {})] * 2 + [(RIGHT, {})] + [(DOWN, {})] * 2,
]


class CatchEggsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))

    def _build_level_program(self, _env: object) -> list[tuple[int, dict[str, int]]]:
        idx = self._current_level_idx or 0
        return _LEVEL_PROGRAMS[idx]


AGENT_CLASS = CatchEggsDslAgent
