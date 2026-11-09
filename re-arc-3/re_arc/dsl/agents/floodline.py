from __future__ import annotations

from ..core import DslAgent, observation_level_index

U, D, L, R, S = 1, 2, 3, 4, 5

LEVEL_PROGRAMS = [
    [R] * 3,
    [S, R] + [D] * 4 + [L] * 2 + [U],
    [S, R] + [U] * 2 + [L] * 2,
    [S] + [U] * 4 + [R] * 11 + [D] * 5 + [L] * 4,
    [R] * 6
    + [S]
    + [R] * 2
    + [D] * 2
    + [R] * 2
    + [D] * 2
    + [R] * 2
    + [S]
    + [U] * 2
    + [R] * 4
    + [D] * 4
    + [R] * 2
    + [D] * 4
    + [L] * 6
    + [S]
    + [D] * 4
    + [L] * 6,
    [D] * 4
    + [L] * 6
    + [D] * 4
    + [S]
    + [R] * 2
    + [U] * 2
    + [R] * 2
    + [U] * 4
    + [R] * 8
    + [D] * 6
    + [R] * 6
    + [R] * 2
    + [S]
    + [D] * 2
    + [L] * 4
    + [D] * 4
    + [R] * 4,
    [U] * 4
    + [R] * 2
    + [S]
    + [R] * 4
    + [U] * 6
    + [R] * 4
    + [D] * 2
    + [R] * 2
    + [S]
    + [R] * 4
    + [D] * 4
    + [R] * 2
    + [U] * 2
    + [S]
    + [R] * 4
    + [D] * 6
    + [L] * 6
    + [U] * 4
    + [L] * 6
    + [U] * 8
    + [R] * 6,
]


class FloodlineDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, _env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is not None:
            self.mark_levels_solved(level_index)
            if level_index != self._level_index:
                self._level_index = level_index
                self._action_index = 0
        if self._level_index is None:
            raise RuntimeError("Floodline observation is missing level progress")
        program = LEVEL_PROGRAMS[self._level_index]
        if self._action_index >= len(program):
            raise RuntimeError("Floodline DSL program exhausted before victory")
        action = program[self._action_index]
        self._action_index += 1
        return action, {}
