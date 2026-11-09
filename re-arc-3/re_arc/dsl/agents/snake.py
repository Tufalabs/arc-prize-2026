from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_ENV_MOD = import_module("re_arc.environment_files.snake.0001.snake")

search_plan_for_model = _ENV_MOD.search_plan_for_model


class SnakeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = dict(level.get_data("model") or {})
        plan = search_plan_for_model(model)
        if plan is None:
            level_idx = int(level.get_data("level_index") or 0)
            raise RuntimeError(f"snake DSL could not solve level {level_idx}")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = SnakeDslAgent
