from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_env_mod = import_module("re_arc.environment_files.dash_puzzle.0001.dashpuzzle")
compute_level_plan = _env_mod.compute_level_plan
TOTAL_LEVELS = int(getattr(_env_mod, "TOTAL_LEVELS", 6))


class DashPuzzleDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=TOTAL_LEVELS)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(env._game.current_level.get_data("level_index") or 0)
        actions = list(compute_level_plan(level_idx))
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = DashPuzzleDslAgent
