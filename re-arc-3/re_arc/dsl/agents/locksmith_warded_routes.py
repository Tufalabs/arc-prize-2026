from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_env_mod = import_module("re_arc.environment_files.locksmith_warded_routes.0001.locksmithwardedroutes")


class LocksmithWardedRoutesDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "locksmith_warded_routes-0001") -> None:
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        solution = game.current_level.get_data("solution")
        if solution is None:
            level_idx = int(game.level_index)
            solution = _env_mod.LEVEL_SPECS[level_idx]["solution"]
        return [(int(action_id), {}) for action_id in solution]


AGENT_CLASS = LocksmithWardedRoutesDslAgent
