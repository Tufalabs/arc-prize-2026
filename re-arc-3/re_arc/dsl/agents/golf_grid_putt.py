from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent, camera_grid_to_display
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.golf_grid_putt.0001.golfgridputt")

_deserialize_model = _ENV_MOD._deserialize_model
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model
simulate_decision_transition = _ENV_MOD.simulate_decision_transition
STATE_TO_CODE = _ENV_MOD.STATE_TO_CODE
DIR8 = _ENV_MOD.DIR8
GRID_WIDTH = _ENV_MOD.GRID_WIDTH
GRID_HEIGHT = _ENV_MOD.GRID_HEIGHT


class GolfGridPuttDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        model = _deserialize_model(env._game.current_level)
        start_state = initial_search_state_from_model(model)
        goal_code = int(STATE_TO_CODE["LEVEL_WIN"])
        gate_period = 6 if model.get("gates") else 1

        def is_goal(state: tuple[int, ...]) -> bool:
            return int(state[0]) == goal_code

        def expand(state: tuple[int, ...]):
            bx, by = int(state[1]), int(state[2])

            next_state, _won, _prims = simulate_decision_transition(model, state, ("wait",))
            if next_state is not None:
                yield ("wait",), next_state, 1.0

            for dx, dy in DIR8:
                for power in range(1, int(model["max_power"]) + 1):
                    tx = int(bx + dx * power)
                    ty = int(by + dy * power)
                    if not (0 <= tx < GRID_WIDTH and 0 <= ty < GRID_HEIGHT):
                        break
                    command = ("shot", int(dx), int(dy), int(power))
                    next_state, _won, _prims = simulate_decision_transition(model, state, command)
                    if next_state is None:
                        continue
                    yield command, next_state, 1.0

        def dominance_key(state: tuple[int, ...]) -> tuple:
            return (
                int(state[1]),
                int(state[2]),
                int(state[11]),
                int(state[12]) % gate_period,
                int(state[16]),
                int(state[17]),
            )

        def dominance_score(state: tuple[int, ...]) -> float:
            return float(int(state[10]))

        macro_plan = bfs_plan(
            start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score
        )
        if macro_plan is None:
            raise RuntimeError("golf_grid_putt DSL could not find a valid solution.")

        primitives: list[tuple[int, dict[str, int]]] = []
        state = start_state
        for command in macro_plan:
            next_state, won, actions = simulate_decision_transition(model, state, command)
            if next_state is None:
                raise RuntimeError(f"golf_grid_putt DSL replay failed for command={command}.")
            for action_id, payload in actions:
                if int(action_id) == 6:
                    gx, gy = int(payload["x"]), int(payload["y"])
                    dx, dy = camera_grid_to_display(env._game.camera, gx, gy)
                    primitives.append((6, {"x": int(dx), "y": int(dy)}))
                else:
                    primitives.append((5, {}))
            state = next_state
            if won:
                break

        primitives.append((5, {}))
        return primitives


AGENT_CLASS = GolfGridPuttDslAgent
