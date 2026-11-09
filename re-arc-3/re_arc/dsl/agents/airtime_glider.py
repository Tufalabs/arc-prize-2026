from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.dsl.solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.airtime_glider.0001.airtimeglider")

ACTION_UP = _ENV_MOD.ACTION_UP
ACTION_DOWN = _ENV_MOD.ACTION_DOWN
ACTION_IDLE = _ENV_MOD.ACTION_IDLE
TIMEBAR_STEPS = _ENV_MOD.TIMEBAR_STEPS

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class AirtimeGliderDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple[int, int, int, int, int, tuple[tuple, ...]]) -> bool:
            return int(state[0]) >= TIMEBAR_STEPS

        def expand(state: tuple[int, int, int, int, int, tuple[tuple, ...]]):
            for action_id in (ACTION_UP, ACTION_DOWN, ACTION_IDLE):
                next_state, won = apply_action_transition(model, state, action_id)
                if next_state is None:
                    continue
                yield int(action_id), next_state, 1.0
                if won:
                    continue

        def dominance_key(state: tuple[int, int, int, int, int, tuple[tuple, ...]]) -> tuple:
            tick, lane, _health, _shield, pending, entities = state
            return (int(tick), int(lane), int(pending), tuple(entities))

        def dominance_score(state: tuple[int, int, int, int, int, tuple[tuple, ...]]) -> float:
            _tick, _lane, health, shield, _pending, _entities = state
            return float(int(health) * 1000 + int(shield))

        plan = bfs_plan(start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("airtime_glider DSL could not find a survival plan for this level.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = AirtimeGliderDslAgent
