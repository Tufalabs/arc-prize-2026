from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.pathfinder.0001.pathfinder")

apply_action_transition = _ENV_MOD.apply_action_transition
current_goal_from_model = _ENV_MOD.current_goal_from_model
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class PathfinderDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=5)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        model = env._game.current_level_model()
        start_state = initial_search_state_from_model(model)
        goal = current_goal_from_model(model)

        def is_goal(state: tuple[int, int]) -> bool:
            return (int(state[0]), int(state[1])) == goal

        def expand(state: tuple[int, int]):
            for action_id in (1, 2, 3, 4):
                next_state, _won = apply_action_transition(model, state, int(action_id))
                if next_state is None:
                    continue
                yield int(action_id), next_state, 1.0

        plan = bfs_plan(start_state, is_goal, expand)
        if plan is None:
            raise RuntimeError("Pathfinder DSL could not solve the current fixed level.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = PathfinderDslAgent
