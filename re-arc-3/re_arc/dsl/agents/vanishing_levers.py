from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_PROGRAMS = (
    (5,),
    (5, 2, 5),
    (5, 4, 5, 3, 5),
    (5, 3, 5, 2, 5, 4, 5),
    (5, 4, 5, 3, 5, 4, 4, 5, 3, 5, 2, 5, 4, 5),
    (5, 1, 5, 4, 5, 3, 5, 1, 5, 4, 1, 5, 3, 5, 2, 5, 4, 5),
    (5, 1, 1, 4, 5, 3, 3, 5, 2, 2, 5, 4, 5, 1, 5, 1, 1, 4, 5, 2, 5, 3, 5, 1, 5, 3, 3, 5),
)


class VanishingLeversDslAgent(DslAgent):
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
            raise RuntimeError("Missing `levels_completed` in Vanishing Levers observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Vanishing Levers DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
