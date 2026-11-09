from __future__ import annotations

from collections.abc import Iterable

from ..core import CachedProgramDslAgent, find_shortest_action_plan

LEVEL_MODELS = (
    {
        "water": (12, 8),
        "valves_open": (False,),
        "valves": ((0, 1),),
        "targets": {1: 10},
        "guards": {},
        "budget": 6,
        "actions": ((5, {}), (6, {"x": 31, "y": 43})),
    },
    {
        "water": (16, 8, 2),
        "valves_open": (False, False),
        "valves": ((0, 1), (1, 2)),
        "targets": {1: 11},
        "guards": {},
        "budget": 10,
        "actions": ((5, {}), (6, {"x": 18, "y": 43}), (6, {"x": 36, "y": 43})),
    },
    {
        "water": (18, 10, 2),
        "valves_open": (False, False),
        "valves": ((0, 1), (1, 2)),
        "targets": {2: 8},
        "guards": {0: 14},
        "budget": 36,
        "actions": ((5, {}), (6, {"x": 18, "y": 43}), (6, {"x": 36, "y": 43})),
    },
)


def _tick(
    water: tuple[int, ...], valves_open: tuple[bool, ...], valves: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    deltas = [0] * len(water)
    for is_open, (left, right) in zip(valves_open, valves, strict=True):
        if not is_open:
            continue
        diff = water[left] - water[right]
        if diff >= 2:
            deltas[left] -= 1
            deltas[right] += 1
        elif diff <= -2:
            deltas[left] += 1
            deltas[right] -= 1
    return tuple(max(0, min(24, level + delta)) for level, delta in zip(water, deltas, strict=True))


def _apply_action(
    water: tuple[int, ...], valves_open: tuple[bool, ...], action_idx: int | None, model: dict[str, object]
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    next_open = list(valves_open)
    if action_idx is not None:
        next_open[action_idx] = not next_open[action_idx]
    next_open_tuple = tuple(next_open)
    next_water = _tick(water, next_open_tuple, model["valves"])
    return next_water, next_open_tuple


class ValveEqualizerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "valve_equalizer-0001"):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(env._game.level_index)
        model = LEVEL_MODELS[level_idx]
        start_state = (model["water"], model["valves_open"], int(model["budget"]))

        def is_goal(state: tuple[tuple[int, ...], tuple[bool, ...], int]) -> bool:
            water, _open, _budget = state
            for tank_idx, target in model["targets"].items():
                if water[int(tank_idx)] < int(target):
                    return False
            for tank_idx, minimum in model["guards"].items():
                if water[int(tank_idx)] < int(minimum):
                    return False
            return True

        def expand(
            state: tuple[tuple[int, ...], tuple[bool, ...], int],
        ) -> Iterable[tuple[int, tuple[tuple[int, ...], tuple[bool, ...], int] | None]]:
            water, open_state, budget = state
            if budget <= 0:
                return

            wait_water, wait_open = _apply_action(water, open_state, None, model)
            yield 0, (wait_water, wait_open, budget - 1)

            for valve_idx in range(len(model["valves"])):
                next_water, next_open = _apply_action(water, open_state, valve_idx, model)
                yield valve_idx + 1, (next_water, next_open, budget - 1)

        def dominance_key(
            state: tuple[tuple[int, ...], tuple[bool, ...], int],
        ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
            water, open_state, _budget = state
            return water, open_state

        def dominance_score(state: tuple[tuple[int, ...], tuple[bool, ...], int]) -> int:
            return int(state[2])

        plan = find_shortest_action_plan(start_state, is_goal, expand, dominance_key, dominance_score)
        if plan is None:
            raise RuntimeError(f"Valve Equalizer DSL could not solve level {level_idx}.")

        program: list[tuple[int, dict[str, int]]] = []
        for symbolic_action in plan:
            action_id, payload = model["actions"][symbolic_action]
            program.append((int(action_id), dict(payload)))
        return program
