from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent

ACTION_CLICK = 6
ACTION_SPACE = 5

_LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(ACTION_CLICK, {"x": 26, "y": 30}), (ACTION_SPACE, {})],
    [
        (ACTION_CLICK, {"x": 14, "y": 54}),
        (ACTION_CLICK, {"x": 22, "y": 50}),
        (ACTION_CLICK, {"x": 22, "y": 34}),
        (ACTION_SPACE, {}),
    ],
    [
        (ACTION_CLICK, {"x": 22, "y": 10}),
        (ACTION_CLICK, {"x": 42, "y": 10}),
        (ACTION_CLICK, {"x": 50, "y": 22}),
        (ACTION_CLICK, {"x": 50, "y": 42}),
        (ACTION_CLICK, {"x": 42, "y": 50}),
        (ACTION_CLICK, {"x": 22, "y": 50}),
        (ACTION_SPACE, {}),
    ],
]


class ScaleGateFloodDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))

    def _build_level_program(self, _env: object) -> list[tuple[int, dict[str, int]]]:
        idx = self._current_level_idx or 0
        return _LEVEL_PROGRAMS[idx]


AGENT_CLASS = ScaleGateFloodDslAgent
