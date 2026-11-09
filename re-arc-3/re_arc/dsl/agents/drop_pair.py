from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_DIMS = ((4, 2), (2, 6), (3, 6), (8, 8), (12, 10), (12, 10), (12, 11))
LEVEL_CLICKS = (
    ((0, 0), (3, 0)),
    ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
    ((1, 4), (2, 0), (2, 1), (1, 0), (0, 1), (1, 0), (0, 0)),
    (
        (2, 3),
        (2, 2),
        (1, 2),
        (1, 0),
        (2, 1),
        (4, 2),
        (0, 1),
        (0, 0),
        (4, 1),
        (3, 1),
        (0, 1),
        (4, 1),
        (5, 1),
        (0, 0),
        (0, 0),
        (6, 0),
        (1, 0),
        (5, 2),
        (5, 1),
        (2, 0),
        (5, 1),
        (3, 0),
        (4, 0),
        (6, 0),
        (6, 0),
    ),
    (
        (7, 2),
        (7, 1),
        (7, 0),
        (9, 3),
        (9, 2),
        (8, 2),
        (7, 0),
        (1, 3),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 1),
        (1, 0),
        (0, 0),
        (6, 0),
        (5, 1),
        (5, 0),
        (5, 2),
        (4, 2),
        (3, 2),
        (4, 1),
        (9, 2),
        (10, 1),
        (10, 0),
        (10, 0),
        (9, 1),
        (10, 0),
        (9, 0),
        (8, 0),
        (1, 0),
        (0, 1),
        (0, 0),
        (2, 0),
        (3, 1),
        (4, 0),
        (2, 0),
        (1, 0),
        (4, 0),
        (5, 0),
        (6, 0),
    ),
    (
        (3, 0),
        (9, 3),
        (3, 4),
        (1, 3),
        (4, 4),
        (9, 2),
        (5, 5),
        (0, 3),
        (6, 5),
        (7, 2),
        (0, 2),
        (0, 1),
        (7, 1),
        (7, 0),
        (0, 1),
        (1, 0),
        (0, 0),
        (8, 2),
        (7, 0),
        (6, 0),
        (5, 1),
        (5, 0),
        (5, 2),
        (4, 2),
        (3, 2),
        (4, 1),
        (9, 2),
        (10, 1),
        (10, 0),
        (10, 0),
        (9, 1),
        (10, 0),
        (9, 0),
        (8, 0),
        (1, 0),
        (0, 1),
        (0, 0),
        (2, 0),
        (3, 1),
        (4, 0),
        (2, 0),
        (1, 0),
        (4, 0),
        (5, 0),
        (6, 0),
    ),
    (
        (8, 1),
        (3, 1),
        (0, 1),
        (8, 1),
        (1, 3),
        (9, 4),
        (3, 0),
        (9, 3),
        (3, 4),
        (9, 2),
        (4, 4),
        (0, 3),
        (5, 5),
        (6, 5),
        (7, 2),
        (0, 2),
        (0, 1),
        (7, 1),
        (7, 0),
        (0, 1),
        (1, 0),
        (0, 0),
        (8, 2),
        (7, 0),
        (6, 0),
        (5, 1),
        (5, 0),
        (5, 2),
        (4, 2),
        (3, 2),
        (4, 1),
        (9, 2),
        (10, 1),
        (10, 0),
        (10, 0),
        (9, 1),
        (10, 0),
        (9, 0),
        (8, 0),
        (1, 0),
        (0, 1),
        (0, 0),
        (2, 0),
        (3, 1),
        (4, 0),
        (2, 0),
        (1, 0),
        (4, 0),
        (5, 0),
        (6, 0),
    ),
)


class DropPairDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_CLICKS))
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if level_idx != self._level_idx:
                self._level_idx = level_idx
                self._action_idx = 0
        if self._level_idx is None:
            raise RuntimeError("Drop & Pair observation has no level progress")
        program = LEVEL_CLICKS[self._level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Drop & Pair DSL witness exhausted before WIN")
        x, y = program[self._action_idx]
        self._action_idx += 1
        width, height = LEVEL_DIMS[self._level_idx]
        origin_x = (64 - 5 * width) // 2
        origin_y = (62 - 5 * height) // 2
        return 6, {"x": origin_x + 5 * x + 1, "y": origin_y + 5 * (height - 1 - y) + 1}
