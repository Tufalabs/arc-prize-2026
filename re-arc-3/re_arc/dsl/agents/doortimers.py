from __future__ import annotations

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.dsl.solvers import bfs_plan

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
WAIT_ACTION = 5
PHASE_CLOSED = 0


class DoortimersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        model = game._model

        player = game._player
        if player is None:
            raise RuntimeError("Doortimers DSL missing player sprite.")

        start_x, start_y = int(player.x), int(player.y)
        time_limit = int(game._time_left)
        exits = set(model.exits)
        spikes = set(model.spikes)
        terrain = model.terrain
        door_cell_to_group = dict(model.door_cell_to_group)
        initial_phases = tuple(int(p) for p in game._door_phases)

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < len(terrain[0]) and 0 <= y < len(terrain)

        def phase_for_cell(cell: tuple[int, int], steps_used: int) -> int | None:
            group_idx = door_cell_to_group.get(cell)
            if group_idx is None:
                return None
            return (initial_phases[group_idx] + int(steps_used)) % 3

        goal_state = (-1, -1, -1)
        start_state = (int(start_x), int(start_y), 0)

        def is_goal(state: tuple[int, int, int]) -> bool:
            return state == goal_state

        def expand(state: tuple[int, int, int]):
            if state == goal_state:
                return []

            x, y, steps_used = state
            next_steps = int(steps_used) + 1

            transitions: list[tuple[int, tuple[int, int, int], float]] = []
            for action_id in (1, 2, 3, 4, WAIT_ACTION):
                nx = int(x)
                ny = int(y)
                delta = MOVE_DELTAS.get(action_id)
                if delta is not None:
                    tx = nx + int(delta[0])
                    ty = ny + int(delta[1])
                    if in_bounds(tx, ty):
                        target = (tx, ty)
                        if target in exits:
                            nx, ny = tx, ty
                        else:
                            phase_before = phase_for_cell(target, steps_used)
                            if phase_before is not None:
                                if phase_before != PHASE_CLOSED:
                                    nx, ny = tx, ty
                            elif terrain[ty][tx] in {".", "^"}:
                                nx, ny = tx, ty

                cell = (nx, ny)

                if cell in spikes:
                    continue

                if cell in exits:
                    transitions.append((action_id, goal_state, 1.0))
                    continue

                phase_after = phase_for_cell(cell, next_steps)
                if phase_after == PHASE_CLOSED:
                    continue

                if next_steps >= time_limit:
                    continue

                transitions.append((action_id, (nx, ny, next_steps), 1.0))

            return transitions

        plan = bfs_plan(start_state, is_goal, expand)
        if not plan:
            raise RuntimeError("Doortimers DSL could not find a valid plan.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = DoortimersDslAgent
