from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.hookshot.0001.hookshot")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class HookshotDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        exits = {tuple(cell) for cell in model["exits"]}

        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple) -> bool:
            return (int(state[0]), int(state[1])) in exits

        def expand(state: tuple):
            for action_id in (1, 2, 3, 4, 5):
                next_state, won, restarted = apply_action_transition(model, state, action_id)
                if restarted:
                    continue
                if won:
                    yield action_id, next_state, 1.0
                    continue
                yield action_id, next_state, 1.0

        def dominance_key(state: tuple) -> tuple:
            px, py, facing, stun, saw_state, door_state, switch_mask, _time_left = state
            return (
                int(px),
                int(py),
                int(facing),
                int(stun),
                tuple((int(item[0]), int(item[1])) for item in saw_state),
                tuple(int(v) for v in door_state),
                int(switch_mask),
            )

        def dominance_score(state: tuple) -> float:
            return float(state[7])

        plan = bfs_plan(start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("hookshot DSL could not solve the current level")

        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = HookshotDslAgent
