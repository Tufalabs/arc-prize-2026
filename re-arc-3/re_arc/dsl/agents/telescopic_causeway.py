from __future__ import annotations

from ..core import DslAgent, observation_level_index

U, D, L, R = 1, 2, 3, 4


def _moves(*parts: int | tuple[int, int]) -> list[int]:
    program: list[int] = []
    for part in parts:
        if isinstance(part, tuple):
            action, count = part
            program.extend([action] * count)
        else:
            program.append(part)
    return program


LEVEL_PROGRAMS = [
    _moves(R, U, R, D, (R, 6)),
    _moves(U, (R, 7), (L, 2), U, L, D, (L, 7)),
    _moves(U, (R, 3), D, (R, 3), U, (R, 5), L, U, L, D, (L, 4), L, D, L, U, (L, 5), U),
    _moves((R, 2), D, R, U, (R, 4), D, (R, 9), L, U, L, D, (L, 3), U, (L, 4), D, L, U, (L, 8)),
    _moves(U, (R, 5), U, (R, 2), (U, 6), R, U, (R, 7), L, U, L, D, (L, 6), D, R, D, L, (D, 6), L, U, L, D, (L, 8)),
    _moves(
        (R, 2), D, R, U, (R, 8), (L, 4), (U, 3), L, (U, 5), R, U, (R, 9), L, U, L, D, (L, 8), D, R, D, L, (D, 7), (L, 8)
    ),
    _moves(
        (R, 2),
        D,
        R,
        U,
        (R, 4),
        D,
        (R, 5),
        U,
        (R, 4),
        L,
        U,
        L,
        D,
        (L, 3),
        (L, 4),
        (U, 3),
        L,
        (U, 5),
        R,
        U,
        (R, 7),
        L,
        U,
        L,
        D,
        (L, 6),
        D,
        R,
        D,
        L,
        (D, 7),
        (L, 8),
    ),
]


class TelescopicCausewayDslAgent(DslAgent):
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
            if level_idx != self._current_level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing level progress in Telescopic Causeway observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Telescopic Causeway DSL program exhausted before level completion.")
        action = program[self._action_idx]
        self._action_idx += 1
        return action, {}
