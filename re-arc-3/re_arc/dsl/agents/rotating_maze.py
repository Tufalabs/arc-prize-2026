from __future__ import annotations

from ..core import DslAgent, observation_level_index

U, D, L, R, S = 1, 2, 3, 4, 5


def _expand(*parts: int | tuple[int, int]) -> list[int]:
    actions: list[int] = []
    for part in parts:
        if isinstance(part, tuple):
            action, count = part
            actions.extend([action] * count)
        else:
            actions.append(part)
    return actions


LEVEL_PROGRAMS = [
    _expand(R, U, U),
    _expand(U, (S, 3), U),
    _expand(R, L, (S, 3), L),
    _expand((D, 4), (R, 3), S, (L, 3), (U, 5), (R, 2), D, R, S, L, U, (L, 2), D, R, L, (D, 5)),
    _expand(
        (U, 2),
        (R, 6),
        (D, 3),
        S,
        (U, 3),
        (L, 6),
        (D, 2),
        R,
        L,
        (D, 2),
        (R, 4),
        S,
        (L, 4),
        (U, 4),
        (R, 6),
        (D, 2),
        (U, 2),
        (R, 2),
        (D, 6),
    ),
    _expand(
        (D, 4),
        (R, 4),
        (U, 4),
        L,
        R,
        (S, 3),
        R,
        (S, 3),
        L,
        (S, 2),
        L,
        L,
        R,
        S,
        (D, 4),
        (L, 4),
        (U, 4),
        R,
        (R, 3),
        S,
        (D, 4),
        R,
        U,
        (U, 3),
        R,
        (U, 3),
        R,
    ),
    _expand(
        U,
        (R, 2),
        D,
        L,
        R,
        (S, 3),
        R,
        (R, 4),
        (S, 3),
        (L, 4),
        (S, 2),
        L,
        (L, 2),
        (D, 3),
        (R, 3),
        U,
        (S, 3),
        U,
        L,
        R,
        (S, 2),
        (L, 2),
        (D, 2),
        R,
        L,
        D,
        (R, 3),
        U,
        U,
        L,
        (S, 3),
        L,
        U,
    ),
]


class RotatingMazeDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, _env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is None:
            raise RuntimeError("Missing levels_completed in Rotating Maze observation.")
        self.mark_levels_solved(level_index)
        if level_index != self._level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError(f"Rotating Maze program exhausted on level {level_index + 1}.")
        action = program[self._action_index]
        self._action_index += 1
        return action, {}


AGENT_CLASS = RotatingMazeDslAgent
