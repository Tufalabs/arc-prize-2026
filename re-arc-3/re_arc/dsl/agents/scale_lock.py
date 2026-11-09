from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent, observation_level_index

_scale_lock_module = import_module("re_arc.environment_files.scale_lock.0001.scalelock")
SOLVER_PROGRAMS = _scale_lock_module.SOLVER_PROGRAMS


class ScaleLockDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "scale_lock-0001"):
        super().__init__(game_id=game_id, total_levels=len(SOLVER_PROGRAMS))

    def _level_index(self, observation):
        return observation_level_index(observation, self.total_levels)

    def _build_level_program(self, env):
        del env
        level_idx = self._current_level_idx
        if level_idx is None:
            raise RuntimeError("Scale Lock DSL requires a current level index.")
        return [
            (action_id, {"x": 0, "y": 0} if int(action_id) == 6 else {}) for action_id in SOLVER_PROGRAMS[level_idx]
        ]


AGENT_CLASS = ScaleLockDslAgent
