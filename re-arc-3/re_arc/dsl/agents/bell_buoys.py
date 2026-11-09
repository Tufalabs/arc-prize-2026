from __future__ import annotations

from ..core import DslAgent, observation_level_index

C = {
    "A": {"x": 4, "y": 34},
    "M": {"x": 19, "y": 59},
    "M_L3": {"x": 4, "y": 59},
    "X": {"x": 9, "y": 4},
    "Q": {"x": 59, "y": 59},
    "K": {"x": 59, "y": 59},
    "K_CANCEL_L5": {"x": 4, "y": 49},
    "K_CANCEL": {"x": 29, "y": 54},
    "N": {"x": 39, "y": 59},
    "N_L3": {"x": 59, "y": 34},
    "P": {"x": 9, "y": 59},
    "R": {"x": 39, "y": 9},
    "T": {"x": 57, "y": 7},
    "T_LEFT7": {"x": 39, "y": 9},
    "T_LEFT6": {"x": 34, "y": 9},
    "L": {"x": 19, "y": 39},
    "L8": {"x": 34, "y": 59},
    "W": {"x": 9, "y": 59},
    "F": {"x": 29, "y": 59},
    "F_RIGHT": {"x": 54, "y": 59},
}


def click(name: str) -> tuple[int, dict[str, int]]:
    return 6, C[name]


def rep(action: int, count: int) -> list[tuple[int, dict[str, int]]]:
    return [(action, {}) for _ in range(count)]


LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    rep(5, 10),
    rep(4, 4) + rep(5, 10),
    [
        click("M_L3"),
        *rep(4, 1),
        *rep(1, 5),
        *rep(5, 1),
        *rep(2, 1),
        *rep(1, 1),
        *rep(2, 1),
        *rep(1, 1),
        *rep(2, 1),
        click("N_L3"),
        *rep(5, 2),
    ],
    rep(1, 4) + rep(5, 5) + [click("X")] + rep(5, 1) + [click("Q")] + rep(3, 5) + rep(5, 10),
    rep(1, 1)
    + rep(3, 6)
    + [click("M")]
    + rep(1, 5)
    + rep(5, 5)
    + [click("K_CANCEL_L5")]
    + rep(5, 1)
    + rep(4, 6)
    + rep(2, 1)
    + rep(5, 4)
    + [click("N")]
    + rep(4, 4)
    + rep(5, 10),
    rep(3, 6)
    + rep(1, 1)
    + [click("P")]
    + rep(1, 5)
    + rep(5, 8)
    + [click("K_CANCEL")]
    + rep(5, 1)
    + rep(2, 1)
    + rep(4, 6)
    + rep(5, 10)
    + rep(5, 5)
    + [click("R")]
    + rep(5, 6),
    [
        *rep(3, 4),
        click("L"),
        *rep(4, 4),
        *rep(5, 4),
        click("W"),
        *rep(1, 5),
        *rep(5, 9),
        *rep(5, 3),
        click("T_LEFT7"),
        *rep(5, 6),
        click("F"),
        *rep(4, 5),
        click("T_LEFT7"),
        *rep(5, 3),
        click("F_RIGHT"),
        *rep(5, 10),
    ],
]


class BellBuoysDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx = None
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
            raise RuntimeError("Missing `levels_completed` in Bell Buoys observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Bell Buoys DSL program exhausted before reaching WIN.")
        action_id, data = program[self._action_idx]
        self._action_idx += 1
        return action_id, data
