from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.one_screen_precision_jumps.0001.onescreenprecisionjumps")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model
MOVE_LEFT = _ENV_MOD.MOVE_LEFT
MOVE_RIGHT = _ENV_MOD.MOVE_RIGHT
JUMP = _ENV_MOD.JUMP


class OneScreenPrecisionJumpsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level.get_data("model") or {})

        start_state = initial_search_state_from_model(model)
        start = (start_state, 0)

        def is_goal(node: tuple[tuple[int, ...], int]) -> bool:
            return int(node[1]) == 1

        def expand(node: tuple[tuple[int, ...], int]):
            state = node[0]
            for action_id in (MOVE_LEFT, MOVE_RIGHT, JUMP):
                next_state, won = apply_action_transition(model, state, int(action_id))
                yield int(action_id), (next_state, 1 if won else 0), 1.0

        def dominance_key(node: tuple[tuple[int, ...], int]) -> tuple:
            state, won = node
            if won:
                return ("won",)
            (
                px,
                py,
                vy,
                _time_left,
                checkpoint_mask,
                respawn_cp,
                toggle_tick,
                platform_x,
                platform_dir,
                _pulse,
                *crumble,
            ) = state
            if model.crumble is None:
                crumble_phase = ()
            else:
                crumble_phase = []
                crack = int(model.crumble.crack_delay)
                fall = int(model.crumble.fall_delay)
                for age in crumble:
                    value = int(age)
                    if value < 0:
                        crumble_phase.append(-1)
                    elif value >= fall:
                        crumble_phase.append(2)
                    elif value >= crack:
                        crumble_phase.append(1)
                    else:
                        crumble_phase.append(0)
            return (
                int(px),
                int(py),
                int(vy),
                int(checkpoint_mask),
                int(respawn_cp),
                int(toggle_tick),
                int(platform_x),
                int(platform_dir),
                *(int(v) for v in crumble_phase),
            )

        def dominance_score(node: tuple[tuple[int, ...], int]) -> float:
            state, won = node
            if won:
                return 10_000.0
            return float(state[3])

        plan = bfs_plan(start, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("one_screen_precision_jumps DSL could not solve the current level.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = OneScreenPrecisionJumpsDslAgent
