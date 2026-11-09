from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.dsl.solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.key_door.0001.keydoor")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model
is_win_state = _ENV_MOD.is_win_state


class KeyDoorDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple) -> bool:
            return bool(is_win_state(model, state))

        def expand(state: tuple):
            for action_id in (1, 2, 3, 4, 5):
                next_state, won = apply_action_transition(model, state, action_id)
                if next_state is None:
                    continue
                yield int(action_id), next_state, 1.0
                if won:
                    continue

        def dominance_key(state: tuple) -> tuple:
            (
                px,
                py,
                has_key,
                door_phase,
                key_collected,
                laser_phases,
                guard_x,
                guard_y,
                guard_target_idx,
                _time_left,
            ) = state
            return (
                int(px),
                int(py),
                int(has_key),
                int(door_phase),
                int(key_collected),
                tuple(int(v) for v in laser_phases),
                int(guard_x),
                int(guard_y),
                int(guard_target_idx),
            )

        def dominance_score(state: tuple) -> float:
            return float(int(state[9]))

        plan = bfs_plan(start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("key_door DSL could not find a solution for the current level")

        # The environment enforces a 6-step win freeze before level transition.
        padded = list(int(action_id) for action_id in plan)
        padded.extend([5] * 6)
        return [(action_id, {}) for action_id in padded]


AGENT_CLASS = KeyDoorDslAgent
