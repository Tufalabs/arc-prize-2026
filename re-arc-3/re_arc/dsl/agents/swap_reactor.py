from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_SIZES = ((2, 3), (5, 3), (5, 3), (5, 5), (6, 6), (5, 6), (6, 6))

# Connector notation is (kind, one-indexed row, one-indexed column).
LEVEL_PROGRAMS = (
    (("V", 1, 2),),
    (("H", 3, 1), ("V", 4, 2)),
    (("H", 3, 1),),
    (("H", 1, 2), ("H", 2, 2), ("H", 4, 2), ("H", 4, 3)),
    (("H", 2, 2), ("H", 2, 5), ("H", 6, 2), ("V", 5, 5)),
    (("H", 1, 2), ("H", 1, 3), ("H", 4, 3), ("H", 2, 2)),
    (
        ("H", 2, 2),
        ("H", 2, 5),
        ("H", 6, 2),
        ("V", 5, 5),
        ("H", 6, 4),
        ("H", 6, 3),
        ("V", 5, 1),
        ("H", 6, 1),
        ("V", 5, 2),
        ("H", 3, 1),
        ("H", 3, 2),
        ("V", 3, 3),
        ("V", 3, 1),
        ("V", 4, 1),
        ("H", 6, 2),
        ("V", 5, 3),
    ),
)


def _board_origin(height: int, width: int) -> tuple[int, int]:
    return (64 - (9 * width - 3)) // 2, (56 - (9 * height - 3)) // 2 + 1


def _connector_center(level_index: int, connector: tuple[str, int, int]) -> tuple[int, int]:
    height, width = LEVEL_SIZES[level_index]
    x0, y0 = _board_origin(height, width)
    kind, row_one, col_one = connector
    row = row_one - 1
    col = col_one - 1
    if kind == "H":
        return x0 + col * 9 + 7, y0 + row * 9 + 2
    return x0 + col * 9 + 2, y0 + row * 9 + 7


class SwapReactorDslAgent(DslAgent):
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
        if level_idx is None:
            raise RuntimeError("Missing levels_completed in Swap Reactor observation.")
        self.mark_levels_solved(level_idx)
        if self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        program = LEVEL_PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Swap Reactor DSL program exhausted before reaching WIN.")
        x, y = _connector_center(level_idx, program[self._action_idx])
        self._action_idx += 1
        return 6, {"x": x, "y": y}
