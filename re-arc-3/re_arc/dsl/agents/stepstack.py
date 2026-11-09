from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

LEVEL_PROGRAMS = (
    [RIGHT] * 6 + [UP] * 2 + [RIGHT],
    [RIGHT] * 4 + [UP] * 2 + [RIGHT] * 2 + [UP] * 2 + [RIGHT],
    [RIGHT] + [LEFT] * 4 + [UP, DOWN, RIGHT] + [UP] * 4 + [RIGHT, LEFT] + [DOWN] * 4 + [RIGHT] * 4 + [UP] * 3,
    [LEFT] * 3
    + [UP, DOWN]
    + [RIGHT] * 2
    + [UP] * 4
    + [RIGHT] * 2
    + [LEFT]
    + [DOWN] * 4
    + [RIGHT] * 4
    + [UP] * 3
    + [RIGHT],
    [RIGHT] * 3
    + [LEFT] * 7
    + [UP, DOWN, RIGHT]
    + [UP] * 5
    + [UP, DOWN, RIGHT, LEFT, UP, DOWN]
    + [RIGHT] * 3
    + [LEFT] * 3
    + [DOWN] * 5
    + [RIGHT] * 6
    + [UP] * 4,
    [LEFT, UP, DOWN]
    + [RIGHT] * 9
    + [LEFT] * 9
    + [UP, DOWN]
    + [RIGHT] * 3
    + [UP] * 5
    + [UP, DOWN, RIGHT, LEFT, UP, DOWN]
    + [RIGHT] * 3
    + [LEFT] * 3
    + [DOWN] * 5
    + [RIGHT] * 4
    + [UP] * 3
    + [RIGHT]
    + [UP] * 2,
    [LEFT, UP, DOWN]
    + [RIGHT] * 10
    + [LEFT] * 10
    + [UP, DOWN]
    + [RIGHT] * 2
    + [UP] * 7
    + [UP, DOWN, RIGHT, LEFT, UP, DOWN]
    + [RIGHT] * 3
    + [LEFT] * 3
    + [DOWN] * 7
    + [RIGHT] * 5
    + [UP] * 4
    + [RIGHT] * 2
    + [UP] * 2,
)


class StepstackDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
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
            raise RuntimeError("Missing `levels_completed` in Stepstack observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Stepstack DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = StepstackDslAgent
