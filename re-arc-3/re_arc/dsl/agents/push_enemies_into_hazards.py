from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.environment_files.push_enemies_into_hazards.common import LEVEL_SPECS, build_level_models, find_plan


class PushEnemiesIntoHazardsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))
        self._models = build_level_models()

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(getattr(game, "level_index", 0))
        model = self._models[level_idx]
        plan = find_plan(model)
        if plan is None:
            raise RuntimeError(f"push_enemies_into_hazards DSL could not solve level {level_idx}.")

        # The game keeps a 6-step CLEAR animation before level transition.
        actions = list(plan) + [5] * 6
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = PushEnemiesIntoHazardsDslAgent
