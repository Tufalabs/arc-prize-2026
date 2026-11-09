from __future__ import annotations

from ..core import CachedProgramDslAgent

U = 1
D = 2
L = 3
R = 4
S = 5


def simple(action_id: int) -> tuple[int, dict[str, int]]:
    return action_id, {}


LEVEL_PROGRAMS: tuple[tuple[tuple[int, dict[str, int]], ...], ...] = (
    tuple(simple(action) for action in (R, R, U, U, R, R, D, D, R, R, D, D, L, L, L, U)),
    tuple(simple(action) for action in (S, R, R, R, R, D, D, L, S, D, D, D, D, D, S, L, L)),
    tuple(
        simple(action)
        for action in (S, R, S, S, D, R, D, R, R, U, R, R, R, S, U, L, L, L, L, L, L, U, L, S, D, D, D, D, D, D)
    ),
    tuple(
        simple(action)
        for action in (S, S, R, R, S, S, R, R, R, R, D, D, L, S, R, R, R, L, S, S, L, L, S, S, D, D, D, D)
    ),
    tuple(simple(action) for action in (D, S, R, S, S, U, S, U, U, S, D, D, D, D, S, U, R, R, R, U)),
    tuple(
        simple(action)
        for action in (R, R, S, S, L, L, D, L, L, S, R, R, R, D, D, D, L, S, D, D, D, S, S, D, L, L, L, L, S, D)
    ),
    tuple(
        simple(action)
        for action in (
            S,
            S,
            R,
            R,
            D,
            S,
            S,
            D,
            D,
            R,
            D,
            S,
            D,
            D,
            D,
            D,
            D,
            D,
            D,
            S,
            S,
            S,
            D,
            L,
            D,
            L,
            S,
            S,
            S,
            D,
            D,
            D,
            D,
            L,
            L,
            L,
            L,
            L,
            S,
            U,
            D,
            L,
            L,
            S,
            S,
            L,
            D,
            D,
        )
    ),
)

LEVEL_PROGRAMS = (
    LEVEL_PROGRAMS[0],
    LEVEL_PROGRAMS[1],
    LEVEL_PROGRAMS[2],
    LEVEL_PROGRAMS[3],
    LEVEL_PROGRAMS[4],
    LEVEL_PROGRAMS[5],
    LEVEL_PROGRAMS[6],
)


class PaintSnakesAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str) -> None:
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        if self._current_level_idx is None:
            raise RuntimeError("paint_snakes DSL agent has no current level.")
        return list(LEVEL_PROGRAMS[self._current_level_idx])


AGENT_CLASS = PaintSnakesAgent
