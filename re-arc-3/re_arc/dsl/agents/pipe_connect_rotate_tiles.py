from __future__ import annotations

import math
import random
from importlib import import_module

from ..core import CachedProgramDslAgent

_ENV = import_module("re_arc.environment_files.pipe_connect_rotate_tiles.0001.pipeconnectrotatetiles")

_deserialize_level_model = _ENV._deserialize_level_model
_action_click_for_cell = _ENV._action_click_for_cell
can_supply_without_pre_sink_leak = _ENV.can_supply_without_pre_sink_leak
compute_distances = _ENV.compute_distances
leak_cells_from_filled = _ENV.leak_cells_from_filled
supplied_sinks = _ENV.supplied_sinks


def _rotation_cycle(mask: int) -> int:
    if mask == (1 | 4) or mask == (2 | 8):
        return 2
    if mask == (1 | 2 | 4 | 8):
        return 1
    return 4


class PipeConnectRotateTilesDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_level_model(level)
        geo = model["geo"]
        rot_cells = tuple(tuple(int(v) for v in cell) for cell in model["rotatable_cells"])

        cycles = tuple(_rotation_cycle(int(geo["rotatable_base_masks"][idx])) for idx in range(len(rot_cells)))

        target_state = self._search_orientation(geo, cycles)
        if target_state is None:
            raise RuntimeError("pipe_connect_rotate_tiles DSL could not find a solvable rotation plan.")

        program: list[tuple[int, dict[str, int]]] = []
        program.append((6, {"x": -1, "y": -1}))
        for idx, turns in enumerate(target_state):
            for _ in range(int(turns)):
                gx, gy = rot_cells[int(idx)]
                program.append((6, _action_click_for_cell(level, gx, gy)))

        # Apply the queued final click and allow water wave to reach sinks.
        wait_budget = 96
        for _ in range(wait_budget):
            program.append((6, {"x": -1, "y": -1}))

        return program

    @staticmethod
    def _state_eval(geo: dict, state: tuple[int, ...]) -> tuple[int, bool]:
        dist = compute_distances(geo, state)
        if not dist:
            return -10_000, False

        need = set(range(len(geo["sink_objects"])))
        max_dist = max(dist.values())
        for radius in range(max_dist + 1):
            filled = {cell for cell, d in dist.items() if d <= radius}
            leaks = leak_cells_from_filled(geo, state, filled)
            got = supplied_sinks(geo, state, filled)
            if leaks:
                return len(got) * 1000 + radius * 5 - len(leaks) * 20, False
            if need.issubset(got):
                return 100_000 + (max_dist - radius), True
        return 0, False

    def _search_orientation(self, geo: dict, cycles: tuple[int, ...]) -> tuple[int, ...] | None:
        start = tuple(0 for _ in cycles)
        if can_supply_without_pre_sink_leak(geo, start):
            return start

        best_state = start
        best_score, solved = self._state_eval(geo, best_state)
        if solved:
            return best_state

        rng = random.Random(2026 + len(cycles) * 17)
        restarts = 120
        max_steps = 5000

        for _restart in range(restarts):
            current = [0 for _ in cycles]
            for idx, cycle in enumerate(cycles):
                if cycle <= 1:
                    continue
                if rng.random() < 0.2:
                    current[idx] = rng.randrange(int(cycle))

            state = tuple(current)
            score, solved = self._state_eval(geo, state)
            if solved:
                return state
            if score > best_score:
                best_state = state
                best_score = score

            temperature = 3.0
            for _ in range(max_steps):
                index = rng.randrange(len(cycles))
                cycle = int(cycles[index])
                if cycle <= 1:
                    continue

                nxt = list(state)
                step = 1 if rng.random() < 0.5 else (cycle - 1)
                nxt[index] = (int(nxt[index]) + int(step)) % cycle
                nxt_state = tuple(nxt)
                nxt_score, nxt_solved = self._state_eval(geo, nxt_state)
                if nxt_solved:
                    return nxt_state

                if nxt_score > score:
                    state, score = nxt_state, nxt_score
                else:
                    delta = float(nxt_score - score)
                    accept = rng.random() < math.exp(delta / max(0.05, temperature))
                    if accept:
                        state, score = nxt_state, nxt_score
                if score > best_score:
                    best_state = state
                    best_score = score

                temperature *= 0.9995

        if can_supply_without_pre_sink_leak(geo, best_state):
            return best_state
        return None


AGENT_CLASS = PipeConnectRotateTilesDslAgent
