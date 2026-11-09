from __future__ import annotations

from ..core import CachedProgramDslAgent


class LightShadowDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        planner = getattr(game, "compute_solver_plan_for_current_level", None)
        if not callable(planner):
            raise RuntimeError("light_shadow environment does not expose a solver planner")

        actions = list(planner())
        if not actions:
            raise RuntimeError("light_shadow solver returned an empty plan")
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = LightShadowDslAgent
