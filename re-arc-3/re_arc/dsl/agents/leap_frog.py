from __future__ import annotations

from ..core import CachedProgramDslAgent

BOARD_ORIGIN = 5
CELL_PITCH = 6


def _click(cell_x: int, cell_y: int) -> tuple[int, dict[str, int]]:
    return 6, {"x": BOARD_ORIGIN + cell_x * CELL_PITCH + 2, "y": BOARD_ORIGIN + cell_y * CELL_PITCH + 2}


LEVEL_PROGRAMS: tuple[tuple[tuple[int, dict[str, int]], ...], ...] = (
    (_click(2, 4), _click(4, 4), _click(5, 4), _click(3, 4)),
    ((3, {}), _click(2, 4), _click(4, 4), _click(5, 4), _click(3, 4)),
    ((3, {}), _click(2, 4), _click(4, 4), _click(5, 4), _click(3, 4)),
    (_click(4, 2), _click(4, 4), _click(3, 4), _click(5, 4)),
    ((3, {}), _click(4, 2), _click(4, 4), _click(3, 4), _click(5, 4)),
    (_click(5, 4), _click(5, 2), _click(3, 4), _click(5, 4), _click(5, 3), _click(5, 5)),
    (_click(4, 2), _click(4, 4), _click(3, 4), _click(5, 4)),
    (
        (3, {}),
        _click(2, 2),
        _click(4, 2),
        _click(4, 1),
        _click(4, 3),
        _click(3, 3),
        _click(5, 3),
        _click(5, 4),
        _click(5, 2),
    ),
    (
        (3, {}),
        _click(3, 2),
        _click(5, 2),
        _click(5, 1),
        _click(5, 3),
        _click(6, 2),
        _click(6, 0),
        _click(4, 3),
        _click(6, 3),
        _click(6, 4),
        _click(6, 2),
        _click(6, 1),
        _click(6, 3),
        _click(7, 3),
        _click(5, 3),
    ),
    (
        (3, {}),
        (3, {}),
        _click(2, 3),
        _click(4, 3),
        _click(3, 4),
        _click(5, 4),
        _click(5, 5),
        _click(5, 3),
        _click(6, 2),
        _click(6, 0),
        _click(4, 3),
        _click(6, 3),
        _click(6, 4),
        _click(6, 2),
        _click(6, 1),
        _click(6, 3),
        _click(7, 3),
        _click(5, 3),
        (1, {}),
        (1, {}),
        (1, {}),
        _click(5, 4),
        _click(5, 2),
        _click(5, 1),
        _click(5, 3),
    ),
)


class LeapFrogDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Leap Frog DSL has no active level.")
        return list(LEVEL_PROGRAMS[self._current_level_idx])


AGENT_CLASS = LeapFrogDslAgent
