from __future__ import annotations

from ..core import CachedProgramDslAgent


class ShapeMoldGptDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=4)

    def _build_level_program(self, env):
        del env
        programs = {
            0: [4, 4, 4, 4, 4, 4, 4, 4],
            1: [4, 4, 4, 4, 4, 1, 1, 1, 1],
            2: [4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4],
            3: [4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 3],
        }
        level_idx = self._current_level_idx
        if level_idx is None or level_idx not in programs:
            raise RuntimeError(f"Missing Shape Mold GPT program for level {level_idx}.")
        action_ids = [*programs[level_idx], 4]
        return [(action_id, {}) for action_id in action_ids]


AGENT_CLASS = ShapeMoldGptDslAgent
