from __future__ import annotations

from ..core import DslAgent, observation_level_index

ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}
SOLUTION_STRINGS = (
    "RRUU",
    "DURRRUUU",
    "RRUURRRUURRR",
    "UURUUUUURRRRRDRRDDRDDDLLLDLLL",
    "UUUUUURUURRRDDDRRUURRRDDDDLLLUULLLDDDDRRRRRRD",
    "UUUURRUUUURRRRUUUURRDDRRDDLLLLDDRRDDRRRRDDRRUUUUUULLUUUU",
    "UUUURRUUUURRUUUURRRRDDRRDDDDDDLLLLUULLUURRRRRRRRUUUURRDDDDDDDDDDDDLLLLLL",
)
LEVEL_PROGRAMS = [[ACTION_BY_LETTER[letter] for letter in solution] for solution in SOLUTION_STRINGS]


class ColorFerryDslAgent(DslAgent):
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
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Color Ferry observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Color Ferry DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
