from __future__ import annotations

from ..core import MOVE_ACTION_BY_DELTA, CachedProgramDslAgent, find_shortest_action_plan


class CornerHallMatchDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _game(self, env):
        wrapped = getattr(env, "_env", env)
        game = getattr(wrapped, "_game", None)
        if game is None:
            raise RuntimeError("Could not resolve underlying game instance for corner_hall_match.")
        return game

    def _build_level_program(self, env):
        level = self._game(env).current_level
        walkable = {tuple(cell) for cell in level.get_data("walkable")}
        start = tuple(level.get_data("start"))
        goal = tuple(level.get_data("goal"))

        def is_goal(state):
            return state == goal

        def expand(state):
            x, y = state
            for delta, action_id in MOVE_ACTION_BY_DELTA.items():
                next_state = (x + delta[0], y + delta[1])
                if next_state in walkable:
                    yield action_id, next_state

        plan = find_shortest_action_plan(
            start, is_goal, expand, dominance_key=lambda state: state, dominance_score=lambda _state: 0
        )
        if plan is None:
            raise RuntimeError(f"No route found for level {level.name!r}.")
        return [(int(action_id), {}) for action_id in plan]
