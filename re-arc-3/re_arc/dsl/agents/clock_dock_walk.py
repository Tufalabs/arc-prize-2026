from __future__ import annotations

from ..core import DslAgent, observation_level_index


class ClockDockWalkDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._current_level_idx: int | None = None
        self._action_idx = 0
        self._programs = {0: [4, 4, 4, 5, 4], 1: [1, 1, 4, 4, 4, 1, 4, 5, 5, 4], 2: [4, 4, 4, 2, 4, 4, 1, 1, 4]}

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("clock_dock_walk observation is missing levels_completed.")

        self.mark_levels_solved(level_idx)
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif bool(getattr(observation, "full_reset", False)) and self._action_idx > 0:
            self._action_idx = 0

        program = self._programs[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"clock_dock_walk DSL program exhausted before level advance level={level_idx} steps={len(program)}"
            )

        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = ClockDockWalkDslAgent
