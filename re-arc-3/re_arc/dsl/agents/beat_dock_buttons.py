from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP_CLICK = {"x": 31, "y": 4}
RIGHT_CLICK = {"x": 58, "y": 31}
DOWN_CLICK = {"x": 31, "y": 58}
LEFT_CLICK = {"x": 4, "y": 31}

PROGRAMS = {
    0: [RIGHT_CLICK, RIGHT_CLICK, RIGHT_CLICK],
    1: [UP_CLICK, RIGHT_CLICK, RIGHT_CLICK, DOWN_CLICK, RIGHT_CLICK],
    2: [RIGHT_CLICK, UP_CLICK, RIGHT_CLICK, DOWN_CLICK, LEFT_CLICK, RIGHT_CLICK, RIGHT_CLICK],
}


class BeatDockButtonsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Missing levels_completed in beat_dock_buttons observation.")

        self.mark_levels_solved(level_idx)
        if self._current_level_idx is None or self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0

        program = PROGRAMS.get(level_idx)
        if program is None:
            raise RuntimeError(f"No DSL program for level {level_idx}.")
        if self._action_idx >= len(program):
            raise RuntimeError(f"beat_dock_buttons DSL program exhausted on level {level_idx}.")

        payload = dict(program[self._action_idx])
        self._action_idx += 1
        return 6, payload


AGENT_CLASS = BeatDockButtonsDslAgent
