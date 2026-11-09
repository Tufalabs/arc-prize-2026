from __future__ import annotations

import gc

from ..core import CachedProgramDslAgent

ACTION_IDS = {"U": 1, "D": 2, "L": 3, "R": 4, "X": 5}

LEVEL_PROGRAMS = (
    "X R R R",
    "X R X D X R X R R R",
    "R R R R L X R X R R R",
    "D X R X L X R X R R X L L L U X D R X L U X D R X L U U U X U U",
    "U L X U X L L U X D R R U X U U U U L X D D D R X L U L X D R X R L X R R U X U U",
    "R R R R L X R U X L X R U X R L X U U R X L D D L X U U R X "
    "R R X U X L X U X U U X R X U D X R R U X L L X R R U X U U",
    "U L L U U U D X R X D X R X R R U X D X U U L X D D D R X "
    "L U U U R X D D D R X R L X R R U X U R X U X R X U X U D X "
    "U U R X L X R X R U",
)


class CrateCapsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _on_new_level(self, env, level_idx):
        # Multi-frame interactions allocate several observations. Collect between
        # levels so cyclic-GC work never lands unpredictably inside a game step.
        gc.collect()
        super()._on_new_level(env, level_idx)

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            return []
        return [(ACTION_IDS[token], {}) for token in LEVEL_PROGRAMS[self._current_level_idx].split()]


AGENT_CLASS = CrateCapsDslAgent
