from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.grid import grid_to_display_click
from ..solvers.search import dijkstra_plan

_ENV_MOD = import_module("re_arc.environment_files.traffic_jam_rush_hour_mini.0001.trafficjamrushhourmini")

WIN_CELEBRATION_STEPS = int(_ENV_MOD.WIN_CELEBRATION_STEPS)
_deserialize_level_model = _ENV_MOD._deserialize_level_model
_unflatten_positions = _ENV_MOD._unflatten_positions
_cells_for = _ENV_MOD._cells_for
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model
iter_search_moves = _ENV_MOD.iter_search_moves
is_goal_state = _ENV_MOD.is_goal_state


class TrafficJamRushHourMiniDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_level_model(level)
        start_state = initial_search_state_from_model(model)

        plan = dijkstra_plan(
            start_state, lambda state: is_goal_state(model, state), lambda state: iter_search_moves(model, state)
        )
        if plan is None:
            raise RuntimeError("traffic_jam_rush_hour_mini DSL could not solve the current level.")

        camera = env._game.camera
        wait_click = grid_to_display_click(camera, (0, 0))
        program: list[tuple[int, dict[str, int]]] = []
        state = start_state

        car_specs = tuple(model["car_specs"])
        for move in plan:
            car_idx, dx, dy, distance = move
            positions = _unflatten_positions(state)
            spec = car_specs[int(car_idx)]
            x, y = positions[int(car_idx)]

            if spec.axis == "h":
                indicator = (int(x - 1), int(y)) if int(dx) < 0 else (int(x + spec.length), int(y))
            else:
                indicator = (int(x), int(y - 1)) if int(dy) < 0 else (int(x), int(y + spec.length))

            car_cells = _cells_for(spec, (int(x), int(y)))
            car_click = car_cells[0]

            program.append((6, grid_to_display_click(camera, car_click)))
            program.append((6, grid_to_display_click(camera, indicator)))
            for _ in range(int(distance)):
                program.append((6, dict(wait_click)))

            next_state = None
            for candidate_move, candidate_state, _cost in iter_search_moves(model, state):
                if tuple(candidate_move) == tuple(move):
                    next_state = candidate_state
                    break
            if next_state is None:
                raise RuntimeError("traffic_jam_rush_hour_mini DSL state reconstruction failed.")
            state = next_state

        for _ in range(WIN_CELEBRATION_STEPS):
            program.append((6, dict(wait_click)))

        return program


AGENT_CLASS = TrafficJamRushHourMiniDslAgent
