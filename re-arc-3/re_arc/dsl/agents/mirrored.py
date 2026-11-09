from __future__ import annotations

from ..core import CachedProgramDslAgent

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6

LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(ACTION_RIGHT, {}), (ACTION_RIGHT, {}), (ACTION_RIGHT, {})],
    [(ACTION_UP, {}), (ACTION_RIGHT, {}), (ACTION_RIGHT, {})],
    [(ACTION_RIGHT, {}), (ACTION_UP, {}), (ACTION_RIGHT, {}), (ACTION_UP, {}), (ACTION_RIGHT, {})],
    [
        (ACTION_UP, {}),
        (ACTION_LEFT, {}),
        (ACTION_LEFT, {}),
        (ACTION_DOWN, {}),
        (ACTION_DOWN, {}),
        (ACTION_RIGHT, {}),
        (ACTION_CLICK, {"x": 30, "y": 41}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_DOWN, {}),
    ],
    [
        (ACTION_UP, {}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_UP, {}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
    ],
    [
        (ACTION_UP, {}),
        (ACTION_RIGHT, {}),
        (ACTION_UP, {}),
        (ACTION_UP, {}),
        (ACTION_RIGHT, {}),
        (ACTION_UP, {}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_UP, {}),
        (ACTION_CLICK, {"x": 32, "y": 22}),
        (ACTION_RIGHT, {}),
        (ACTION_RIGHT, {}),
        (ACTION_CLICK, {"x": 26, "y": 28}),
        (ACTION_UP, {}),
    ],
]


class MirroredDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Mirrored DSL cannot build a program before level sync.")
        return list(LEVEL_PROGRAMS[self._current_level_idx])


AGENT_CLASS = MirroredDslAgent
