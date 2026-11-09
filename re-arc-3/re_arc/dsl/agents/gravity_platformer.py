from __future__ import annotations

import importlib

from ..core import CachedProgramDslAgent
from ..solvers import astar_plan

_mod = importlib.import_module("re_arc.environment_files.gravity_platformer.0001.gravityplatformer")

ACTION_IDLE = _mod.ACTION_IDLE
ACTION_INTERACT = _mod.ACTION_INTERACT
ACTION_LEFT = _mod.ACTION_LEFT
ACTION_RIGHT = _mod.ACTION_RIGHT
ACTION_UP = _mod.ACTION_UP
GAME_ID = _mod.GAME_ID
LEVEL_MODELS = _mod.LEVEL_MODELS
SimState = _mod.SimState
advance_state = _mod.advance_state
initial_state = _mod.initial_state

SEARCH_ACTIONS = (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_INTERACT, ACTION_IDLE)


class GravityPlatformerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_MODELS))

    @property
    def _agent_tag(self) -> str:
        return "gravity_platformer"

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        level_idx = int(level.get_data("level_index") or 0)
        model = LEVEL_MODELS[level_idx]
        start = initial_state(model)

        def is_goal(state: SimState) -> bool:
            return (state.x, state.y) in model.exits

        def expand(state: SimState):
            for action_id in SEARCH_ACTIONS:
                nxt, won, dead = advance_state(model, state, action_id)
                if dead:
                    continue
                if nxt.time_left <= 0 and not won:
                    continue
                yield action_id, nxt, 1.0

        def heuristic(state: SimState) -> float:
            if not model.exits:
                return 0.0
            return float(min(abs(state.x - gx) + abs(state.y - gy) for gx, gy in model.exits))

        plan = astar_plan(start, is_goal, expand, heuristic)
        if plan is None:
            raise RuntimeError(f"{GAME_ID} DSL could not find a valid plan for level {level_idx + 1}.")

        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = GravityPlatformerDslAgent
