from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
CLICK = 6


def click_cell(x: int, y: int) -> tuple[int, dict[str, int]]:
    return CLICK, {"x": 2 + x * 6 + 2, "y": 2 + y * 6 + 2}


LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(UP, {}), (UP, {}), (RIGHT, {}), (RIGHT, {})],
    [(LEFT, {}), (LEFT, {}), (LEFT, {}), (LEFT, {}), (UP, {}), (RIGHT, {})],
    [
        (CLICK, {"x": 42, "y": 29}),
        (UP, {}),
        (RIGHT, {}),
        (DOWN, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (DOWN, {}),
        (RIGHT, {}),
        (UP, {}),
    ],
    [
        (CLICK, {"x": 35, "y": 29}),
        (CLICK, {"x": 35, "y": 29}),
        (CLICK, {"x": 35, "y": 29}),
        (CLICK, {"x": 35, "y": 29}),
        (CLICK, {"x": 35, "y": 29}),
        (DOWN, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (UP, {}),
        (RIGHT, {}),
        (UP, {}),
        (UP, {}),
        (LEFT, {}),
        (UP, {}),
        (RIGHT, {}),
    ],
    [
        (UP, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (UP, {}),
        (UP, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (RIGHT, {}),
        (DOWN, {}),
        (DOWN, {}),
        (DOWN, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (DOWN, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (CLICK, {"x": 34, "y": 42}),
        (UP, {}),
        (UP, {}),
        (UP, {}),
        (RIGHT, {}),
        (UP, {}),
        (RIGHT, {}),
        (DOWN, {}),
    ],
    [
        (RIGHT, {}),
        (RIGHT, {}),
        (DOWN, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (UP, {}),
        (UP, {}),
        (RIGHT, {}),
        (DOWN, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (CLICK, {"x": 55, "y": 35}),
        (CLICK, {"x": 54, "y": 35}),
        (CLICK, {"x": 54, "y": 35}),
        (CLICK, {"x": 52, "y": 33}),
        (CLICK, {"x": 53, "y": 21}),
        (CLICK, {"x": 53, "y": 22}),
        (RIGHT, {}),
    ],
    [
        (CLICK, {"x": 22, "y": 40}),
        (CLICK, {"x": 22, "y": 40}),
        (CLICK, {"x": 22, "y": 40}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (LEFT, {}),
        (CLICK, {"x": 46, "y": 10}),
        (CLICK, {"x": 49, "y": 9}),
        (CLICK, {"x": 47, "y": 11}),
        (CLICK, {"x": 17, "y": 11}),
        (RIGHT, {}),
        (RIGHT, {}),
        (RIGHT, {}),
        (UP, {}),
        (UP, {}),
        (UP, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (LEFT, {}),
        (DOWN, {}),
        (DOWN, {}),
        (RIGHT, {}),
    ],
]

LEVEL_PROGRAMS = [
    LEVEL_PROGRAMS[0],
    LEVEL_PROGRAMS[1],
    LEVEL_PROGRAMS[2],
    LEVEL_PROGRAMS[4],
    LEVEL_PROGRAMS[3],
    LEVEL_PROGRAMS[5],
    LEVEL_PROGRAMS[6],
]


class GlassOrchardDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Glass Orchard observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Glass Orchard DSL program exhausted before reaching WIN.")
        action_id, data = program[self._action_idx]
        self._action_idx += 1
        return action_id, data
