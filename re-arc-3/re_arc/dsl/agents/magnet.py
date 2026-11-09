from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import astar_plan

_ENV_MOD = import_module("re_arc.environment_files.magnet.0001.magnet")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class MagnetDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        all_lit_mask = (1 << len(model["pads"])) - 1

        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple) -> bool:
            return int(state[9]) == all_lit_mask

        def expand(state: tuple):
            for action_id in (1, 2, 3, 4, 5):
                next_state, won = apply_action_transition(model, state, action_id)
                if next_state is None:
                    continue
                yield int(action_id), next_state, 1.0
                if won:
                    continue

        def heuristic(state: tuple) -> float:
            (_px, _py, _magnet_on, _time_left, _tick, metals, _doors, _plates, clamped_mask, pad_mask) = state
            remaining = 0
            for pad_idx, (tx, ty) in enumerate(model["pads"]):
                if int(pad_mask) & (1 << pad_idx):
                    continue
                best = 999
                for metal_idx, (mx, my) in enumerate(metals):
                    if int(mx) < 0:
                        continue
                    if int(clamped_mask) & (1 << metal_idx):
                        continue
                    distance = abs(int(mx) - int(tx)) + abs(int(my) - int(ty))
                    if distance < best:
                        best = distance
                if best < 999:
                    remaining += best
            remaining += (len(model["pads"]) - int(pad_mask).bit_count()) * 2
            return float(remaining)

        def dominance_key(state: tuple) -> tuple:
            (
                px,
                py,
                magnet_on,
                _time_left,
                _tick,
                metals,
                door_open_mask,
                plate_pressed_mask,
                clamped_mask,
                pad_mask,
            ) = state
            return (
                int(px),
                int(py),
                int(magnet_on),
                tuple((int(mx), int(my)) for mx, my in metals),
                int(door_open_mask),
                int(plate_pressed_mask),
                int(clamped_mask),
                int(pad_mask),
            )

        def dominance_score(state: tuple) -> float:
            return float(int(state[3]))

        plan = astar_plan(start_state, is_goal, expand, heuristic)
        if plan is None:
            raise RuntimeError("magnet DSL could not solve the current level.")
        # Environment advances after an 8-step win animation.
        plan = list(plan) + [1] * 8
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = MagnetDslAgent
