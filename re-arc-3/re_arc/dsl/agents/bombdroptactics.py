from __future__ import annotations

from ..core import CachedProgramDslAgent


class BombDropTacticsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        plan_fn = getattr(game, "plan_current_level", None)
        if not callable(plan_fn):
            raise RuntimeError("bombdroptactics game is missing `plan_current_level`")
        action_ids = [int(action_id) for action_id in plan_fn()]
        # Keep stepping while the win animation runs so level transition is observed.
        action_ids.extend([1, 1, 1, 1])
        if not action_ids:
            raise RuntimeError("bombdroptactics planner produced an empty program.")
        return [(int(action_id), {}) for action_id in action_ids]


AGENT_CLASS = BombDropTacticsDslAgent
