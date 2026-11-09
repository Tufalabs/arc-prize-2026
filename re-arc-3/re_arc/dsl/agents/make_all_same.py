from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

# Cell centers matching the game layout (grid coords = display coords for 64x64)
_CELL_CENTERS = [(12, 12), (32, 12), (52, 12), (12, 32), (52, 32), (12, 52), (32, 52), (52, 52)]

_LEVELS = [
    (3, [0, 4], [1, 5]),
    (4, [0, 2, 5, 7], [2, 6, 1, 5]),
    (3, [0, 1, 2, 3, 4, 5, 6, 7], [1, 5, 2, 6, 4, 1, 5, 2]),
]

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_CLICK = 6


def _build_program(target: int, pos_indices: list[int], sizes: list[int]) -> list[tuple[int, dict]]:
    program: list[tuple[int, dict]] = []
    for i, idx in enumerate(pos_indices):
        cx, cy = _CELL_CENTERS[idx]
        delta = target - sizes[i]
        if delta == 0:
            continue
        program.append((ACTION_CLICK, {"x": cx, "y": cy}))
        if delta > 0:
            program.extend([(ACTION_UP, {})] * delta)
        else:
            program.extend([(ACTION_DOWN, {})] * (-delta))
    return program


class MakeAllSameDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVELS))

    def _build_level_program(self, _env) -> list[tuple[int, dict]]:
        idx = self._current_level_idx or 0
        target, pos_indices, sizes = _LEVELS[idx]
        return _build_program(target, pos_indices, sizes)


AGENT_CLASS = MakeAllSameDslAgent
