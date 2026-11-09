from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.dsl.solvers.search import bfs_plan

MOVE_ACTIONS = ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0)))

WIN_ANIMATION_STEPS = 3


class SwitchDoorsToggleWallsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level = game.current_level

        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        time_limit = int(level.get_data("time_limit"))
        start = tuple(int(v) for v in level.get_data("start"))

        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        hazards = {tuple(int(v) for v in item) for item in (level.get_data("hazards") or [])}
        exit_cells = {tuple(int(v) for v in item) for item in (level.get_data("exit_cells") or [])}

        door_a_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_a_positions") or [])}
        door_b_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_b_positions") or [])}
        door_a_initial_closed = {
            tuple(int(v) for v in item) for item in (level.get_data("door_a_initial_closed") or [])
        }
        door_b_initial_closed = {
            tuple(int(v) for v in item) for item in (level.get_data("door_b_initial_closed") or [])
        }

        switch_specs = list(level.get_data("switches") or [])
        switch_cells: dict[tuple[int, int], tuple[int, str]] = {}
        for idx, switch in enumerate(switch_specs):
            kind = str(switch.get("kind", "")).upper()
            if kind not in {"A", "B"}:
                continue
            for x, y in switch.get("cells") or []:
                switch_cells[(int(x), int(y))] = (idx, kind)

        def door_closed(cell: tuple[int, int], *, initial_closed: set[tuple[int, int]], parity: int) -> bool:
            closed = cell in initial_closed
            if parity % 2 == 1:
                closed = not closed
            return closed

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        start_last_switch = -1
        if start in switch_cells:
            start_last_switch = int(switch_cells[start][0])

        start_state = (int(start[0]), int(start[1]), 0, 0, start_last_switch)

        def is_goal(state: tuple[int, int, int, int, int]) -> bool:
            x, y, _a, _b, _s = state
            return (x, y) in exit_cells

        def expand(state: tuple[int, int, int, int, int]):
            x, y, parity_a, parity_b, prev_switch = state
            out = []

            for action_id, (dx, dy) in MOVE_ACTIONS:
                nx, ny = x + dx, y + dy
                tx, ty = x, y

                if in_bounds(nx, ny) and ny > 0 and (nx, ny) not in walls:
                    blocked = False
                    if (nx, ny) in door_a_positions and door_closed(
                        (nx, ny), initial_closed=door_a_initial_closed, parity=parity_a
                    ):
                        blocked = True
                    if (nx, ny) in door_b_positions and door_closed(
                        (nx, ny), initial_closed=door_b_initial_closed, parity=parity_b
                    ):
                        blocked = True
                    if not blocked:
                        tx, ty = nx, ny

                if (tx, ty) in hazards:
                    continue

                n_parity_a = parity_a
                n_parity_b = parity_b
                switch_entry = switch_cells.get((tx, ty))
                switch_id = -1
                if switch_entry is not None:
                    switch_id, kind = switch_entry
                    if switch_id != prev_switch:
                        if kind == "A":
                            n_parity_a ^= 1
                        else:
                            n_parity_b ^= 1

                out.append((action_id, (tx, ty, n_parity_a, n_parity_b, switch_id), 1.0))

            return out

        actions = bfs_plan(start_state, is_goal, expand)
        if actions is None:
            raise RuntimeError("switch_doors_toggle_walls DSL could not find a path to the exit.")
        if len(actions) > time_limit:
            raise RuntimeError(
                f"switch_doors_toggle_walls DSL path exceeds time limit steps={len(actions)} limit={time_limit}."
            )

        program = [(int(action_id), {}) for action_id in actions]
        for _ in range(WIN_ANIMATION_STEPS):
            program.append((1, {}))
        return program


AGENT_CLASS = SwitchDoorsToggleWallsDslAgent
