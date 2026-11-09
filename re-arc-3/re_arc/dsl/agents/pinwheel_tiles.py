from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_PROGRAMS = (
    (2, 3, (("P", 2, 0, 1),), ("P",)),
    (2, 3, (("L", 2, 0, 0), ("R", 2, 0, 1)), ("L", "R")),
    (2, 3, (("L", 2, 0, 0), ("R", 2, 0, 1)), ("R", "L")),
    (
        4,
        4,
        (("M", 3, 0, 0), ("T", 2, 0, 2), ("R", 2, 1, 2), ("Q", 2, 2, 1), ("B", 2, 2, 2), ("L", 2, 2, 0)),
        ("M", "T", "R", "Q", "B", "L"),
    ),
    (
        5,
        5,
        (
            ("C", 2, 0, 0),
            ("M", 3, 0, 0),
            ("T", 2, 0, 2),
            ("R", 2, 0, 3),
            ("L", 2, 2, 0),
            ("E", 2, 2, 1),
            ("Q", 3, 2, 2),
            ("Z", 2, 3, 0),
        ),
        ("C", "M", "R", "E", "Z", "Q", "T", "L"),
    ),
    (
        5,
        5,
        (
            ("A", 3, 0, 0),
            ("B", 3, 1, 1),
            ("C", 3, 2, 2),
            ("N", 2, 0, 3),
            ("W", 2, 2, 0),
            ("S", 2, 3, 0),
            ("E", 2, 1, 3),
            ("D", 2, 3, 3),
            ("X", 2, 0, 2),
            ("Y", 2, 2, 1),
        ),
        ("X", "A", "N", "B", "C", "W", "E", "D", "Y", "S"),
    ),
    (
        6,
        6,
        (
            ("NW", 3, 0, 0),
            ("NE", 3, 0, 3),
            ("SW", 3, 3, 0),
            ("SE", 3, 3, 3),
            ("M", 3, 1, 1),
            ("T", 2, 0, 2),
            ("L", 2, 2, 0),
            ("R", 2, 2, 4),
            ("B", 2, 4, 2),
            ("C", 2, 2, 2),
            ("X", 2, 1, 3),
            ("Y", 2, 3, 1),
        ),
        ("T", "NW", "C", "M", "SW", "X", "NE", "R", "SE", "B", "Y", "L"),
    ),
)


class PinwheelTilesDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Missing levels_completed in Pinwheel Tiles observation")
        self.mark_levels_solved(level_idx)
        if level_idx != self._level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
        height, width, pivot_specs, program = LEVEL_PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Pinwheel Tiles program exhausted on level {level_idx + 1}")
        pivot_id = program[self._action_idx]
        pivot = next(spec for spec in pivot_specs if spec[0] == pivot_id)
        self._action_idx += 1
        x0, y0, x1, y1 = game_hitbox(height, width, pivot)
        return 6, {"x": (x0 + x1) // 2, "y": (y0 + y1) // 2}


def game_hitbox(height, width, pivot):
    _id, size, row, col = pivot
    ox = (64 - (9 * width - 2)) // 2
    oy = (64 - (9 * height - 2)) // 2
    if size == 2:
        x = ox + 9 * col + 6
        y = oy + 9 * row + 6
        return x, y, x + 3, y + 3
    x = ox + 9 * (col + 1) + 1
    y = oy + 9 * (row + 1) + 1
    return x, y, x + 4, y + 4


AGENT_CLASS = PinwheelTilesDslAgent
