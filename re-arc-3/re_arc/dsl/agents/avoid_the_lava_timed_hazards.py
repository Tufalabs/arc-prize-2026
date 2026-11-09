from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.avoid_the_lava_timed_hazards.0001.avoidthelavatimedhazards")

LEVEL_MODELS = _ENV_MOD.LEVEL_MODELS
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class AvoidTheLavaTimedHazardsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(getattr(env._game, "level_index", 0))
        model = LEVEL_MODELS[level_idx]
        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple[int, int, int, tuple[int, ...], tuple[int, ...]]) -> bool:
            return (int(state[0]), int(state[1])) in model.exits

        def expand(state: tuple[int, int, int, tuple[int, ...], tuple[int, ...]]):
            for action_id in (1, 2, 3, 4, 5):
                next_state, _won = apply_action_transition(model, state, int(action_id))
                if next_state is None:
                    continue
                yield int(action_id), next_state, 1.0

        plan = bfs_plan(start_state, is_goal, expand)
        if plan is None:
            raise RuntimeError("avoid_the_lava_timed_hazards DSL could not solve this level.")
        # Winning a level enters a one-step flash state; one extra wait advances to next
        # level.
        return [(int(action_id), {}) for action_id in plan] + [(5, {})]


AGENT_CLASS = AvoidTheLavaTimedHazardsDslAgent
