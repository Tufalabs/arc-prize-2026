from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_PROGRAMS = {0: [3, 3, 3, 3, 3, 3, 3], 1: [3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1], 2: [3, 3, 1, 1, 1, 3]}


class CommandTwinsDslAgent(DslAgent):
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
            raise RuntimeError("command_twins observation is missing levels_completed.")

        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif reset_level and self._action_idx > 0:
            self._action_idx = 0

        program = LEVEL_PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"command_twins DSL program exhausted on level {level_idx}.")

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = CommandTwinsDslAgent
