from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_ENV_MOD = import_module("re_arc.environment_files.one_enemy_duel.0001.oneenemyduel")

_deserialize_model = _ENV_MOD._deserialize_model
build_solver_program = _ENV_MOD.build_solver_program


class OneEnemyDuelDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        return build_solver_program(model)


AGENT_CLASS = OneEnemyDuelDslAgent
