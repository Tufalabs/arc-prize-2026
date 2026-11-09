from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.shortest_route_challenge.0001.shortestroutechallenge")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class ShortestRouteChallengeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        start_state = initial_search_state_from_model(model)

        exit_cells = set(model["exit_cells"])

        def is_goal(state: tuple[int, ...]) -> bool:
            return (int(state[0]), int(state[1])) in exit_cells

        def expand(state: tuple[int, ...]):
            for action_id in (1, 2, 3, 4, 5):
                next_state, won, failed = apply_action_transition(model, state, action_id)
                if next_state is None or failed:
                    continue
                yield int(action_id), next_state, 1.0
                if won:
                    continue

        enemy_tail = tuple(int(v) for v in start_state[13:])

        def dominance_key(state: tuple[int, ...]) -> tuple:
            return (
                int(state[0]),
                int(state[1]),
                int(state[2]),
                int(state[3]),
                int(state[4]),
                int(state[5]),
                int(state[6]),
                int(state[7]),
                int(state[8]),
                int(state[9]),
                int(state[11]),
                tuple(int(v) for v in state[13:]) if len(state) > 13 else enemy_tail,
            )

        def dominance_score(state: tuple[int, ...]) -> float:
            return float(state[10])

        plan = bfs_plan(start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("shortest_route_challenge DSL could not solve the current level")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = ShortestRouteChallengeDslAgent
