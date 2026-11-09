from __future__ import annotations

from ..core import DslAgent, observation_level_index

U, D, L, R, S = 1, 2, 3, 4, 5

LEVEL_PROGRAMS = [
    [R, R, R],
    [U, D, R, R, R, R],
    [S, R, L, R, U, R, R],
    [U, U, R, U, R, U, R, U, R, R, S, D, D, D, D, R, R, R, U, L, U, R, U, U, U, L, L],
    [U, U, U, R, S, U, R, U, U, U, U, R, R, R, R, S, L, L, L, L, D, D, D, D, L, D, L, D, D, D, R, R, R, D],
    [S, D, L, U, R, S, U, R, U, U, U, U, R, U, R, R, R, S, U, R, R, R, R] + [D] * 11 + [L] * 4 + [U, U, R],
    [
        S,
        U,
        L,
        D,
        R,
        S,
        U,
        R,
        U,
        U,
        U,
        U,
        R,
        R,
        R,
        S,
        D,
        L,
        U,
        R,
        S,
        U,
        R,
        U,
        R,
        R,
        D,
        D,
        D,
        R,
        S,
        L,
        U,
        U,
        U,
        D,
        D,
        D,
        L,
        D,
        D,
        L,
        L,
        L,
        D,
        L,
        L,
        L,
        D,
        D,
        R,
        D,
    ],
]


class SandChannelsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level = None
        self._index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level = None
        self._index = 0

    def next_action(self, _env, observation):
        level = observation_level_index(observation, self.total_levels)
        if level is None:
            raise RuntimeError("Missing levels_completed in Sand Channels observation")
        self.mark_levels_solved(level)
        if level != self._level:
            self._level, self._index = level, 0
        program = LEVEL_PROGRAMS[level]
        if self._index >= len(program):
            raise RuntimeError("Sand Channels DSL program exhausted before WIN")
        action = program[self._index]
        self._index += 1
        return action, {}
