from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import astar_plan

_ENV_MOD = import_module("re_arc.environment_files.boulder_push_roll.0001.boulderpushroll")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class BoulderPushRollDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        model = _deserialize_model(env._game.current_level)
        start_state = initial_search_state_from_model(model)
        goals = set(model["goals"])

        def is_goal(state: tuple[int, int, int, int, int, tuple[tuple[int, int, int], ...]]) -> bool:
            return (int(state[0]), int(state[1])) in goals

        def expand(state: tuple[int, int, int, int, int, tuple[tuple[int, int, int], ...]]):
            for action_id in (1, 2, 3, 4, 5):
                nxt, _won = apply_action_transition(model, state, action_id)
                if nxt is None:
                    continue
                yield action_id, nxt, 1.0

        def heuristic(state: tuple[int, int, int, int, int, tuple[tuple[int, int, int], ...]]) -> float:
            px = int(state[0])
            py = int(state[1])
            return float(min(abs(px - gx) + abs(py - gy) for gx, gy in goals))

        plan = astar_plan(start_state, is_goal, expand, heuristic)
        if plan is None:
            raise RuntimeError("boulder_push_roll DSL could not solve the current level")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = BoulderPushRollDslAgent
