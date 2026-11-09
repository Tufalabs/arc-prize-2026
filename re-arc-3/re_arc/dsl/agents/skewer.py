from __future__ import annotations

from ..core import CachedProgramDslAgent

LEVEL_PROGRAMS: list[list[int]] = [
    [4, 4, 4],
    [4, 1, 4, 4, 4, 3, 3, 3, 2, 4, 4, 4],
    [4, 4, 4, 3, 3, 3, 1, 4, 4, 4],
    [1, 4, 2, 2, 4, 4, 4, 4, 3, 1, 3, 3, 2, 4, 4],
    [4, 1, 4, 1, 4, 4, 3, 2, 4],
    [4, 4, 4, 4, 3, 3, 3, 3, 5, 2, 4, 1, 4, 4, 4],
    [2, 4, 4, 4, 4, 3, 3, 3, 3, 5, 2, 2, 4, 1, 1, 4, 4, 5, 1, 4, 4, 4, 4],
    [
        1,
        2,
        4,
        4,
        4,
        1,
        2,
        3,
        4,
        1,
        3,
        5,
        1,
        3,
        5,
        3,
        2,
        1,
        3,
        3,
        5,
        2,
        2,
        1,
        1,
        2,
        1,
        3,
        3,
        4,
        4,
        4,
        5,
        3,
        3,
        2,
        1,
        4,
        4,
        4,
        3,
        3,
        3,
        4,
        5,
        2,
        2,
        2,
        5,
        1,
        5,
        1,
        1,
        1,
        2,
        2,
        1,
        2,
        1,
        1,
        5,
        4,
        4,
        4,
        4,
    ],
]


class SkewerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        level_idx = int(self._current_level_idx or 0)
        return [(action_id, {}) for action_id in LEVEL_PROGRAMS[level_idx]]


AGENT_CLASS = SkewerDslAgent
