from __future__ import annotations

import importlib

from ..core import CachedProgramDslAgent

_frogger_mod = importlib.import_module("re_arc.environment_files.frogger.0001.frogger")
LEVEL_DEFS = _frogger_mod.LEVEL_DEFS
solve_frogger_level = _frogger_mod.solve_frogger_level
WAIT_ACTION_ID = _frogger_mod.WAIT_ACTION_ID


class FroggerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_DEFS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(getattr(game, "_score", 0))
        level_idx = max(0, min(level_idx, len(LEVEL_DEFS) - 1))

        plan = solve_frogger_level(LEVEL_DEFS[level_idx], max_expansions=900_000)
        if plan is None:
            raise RuntimeError(f"Frogger DSL could not solve level {level_idx}.")
        plan = [*plan, int(WAIT_ACTION_ID)]
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = FroggerDslAgent
