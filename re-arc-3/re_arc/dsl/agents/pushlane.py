from __future__ import annotations

from collections import deque
from importlib import import_module

from ..core import CachedProgramDslAgent

_env_mod = import_module("re_arc.environment_files.pushlane.0001.pushlane")

ACTION_TO_DELTA = {
    int(action_id): (int(delta[0]), int(delta[1])) for action_id, delta in _env_mod.ACTION_TO_DELTA.items()
}
CELL = int(_env_mod.CELL)
LAYOUT = [str(row) for row in _env_mod.LAYOUT]
TIME_LIMIT = int(_env_mod.TIME_LIMIT)


def _parse_layout():
    walls: set[tuple[int, int]] = set()
    player = (0, 0)
    crate = (0, 0)
    switch = (0, 0)
    gate = (0, 0)
    collects: set[tuple[int, int]] = set()

    for y, row in enumerate(LAYOUT):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                player = (x, y)
            elif ch == "C":
                crate = (x, y)
            elif ch == "S":
                switch = (x, y)
            elif ch == "g":
                gate = (x, y)
            elif ch in {"a", "b", "c"}:
                collects.add((x, y))

    return walls, player, crate, switch, gate, tuple(sorted(collects))


_WALLS, _PLAYER_START, _CRATE_START, _SWITCH_CELL, _GATE_CELL, _COLLECTS = _parse_layout()
_COLLECT_INDEX = {pos: idx for idx, pos in enumerate(_COLLECTS)}
_GOAL_MASK = 0
_START_COLLECT_MASK = (1 << len(_COLLECTS)) - 1


class PushlaneDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        _ = env
        start = (
            int(_PLAYER_START[0]),
            int(_PLAYER_START[1]),
            int(_CRATE_START[0]),
            int(_CRATE_START[1]),
            0,
            int(_START_COLLECT_MASK),
        )

        queue: deque[tuple[int, int, int, int, int, int]] = deque([start])
        previous: dict[tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int] | None] = {start: None}
        previous_action: dict[tuple[int, int, int, int, int, int], int] = {}
        goal_state: tuple[int, int, int, int, int, int] | None = None

        action_order = (4, 2, 3, 1, 5)

        while queue:
            px, py, cx, cy, gate_open, collect_mask = queue.popleft()
            if collect_mask == _GOAL_MASK:
                goal_state = (px, py, cx, cy, gate_open, collect_mask)
                break

            for action_id in action_order:
                npx, npy = px, py
                ncx, ncy = cx, cy
                ngate = gate_open

                if action_id in ACTION_TO_DELTA:
                    dx, dy = ACTION_TO_DELTA[action_id]
                    tx, ty = px + dx, py + dy

                    def blocked(x: int, y: int, crate_x: int, crate_y: int, gate_is_open: int) -> bool:
                        if (x, y) in _WALLS:
                            return True
                        if (x, y) == _GATE_CELL and not bool(gate_is_open):
                            return True
                        return (x, y) == (crate_x, crate_y)

                    if (tx, ty) == (cx, cy):
                        bx, by = cx + dx, cy + dy
                        if not blocked(bx, by, -999, -999, ngate):
                            ncx, ncy = bx, by
                            npx, npy = tx, ty
                    else:
                        if not blocked(tx, ty, cx, cy, ngate):
                            npx, npy = tx, ty

                elif action_id == 5 and (px, py) == _SWITCH_CELL:
                    ngate = 1

                nmask = int(collect_mask)
                collect_idx = _COLLECT_INDEX.get((npx, npy))
                if collect_idx is not None:
                    nmask &= ~(1 << int(collect_idx))

                next_state = (int(npx), int(npy), int(ncx), int(ncy), int(ngate), int(nmask))
                if next_state in previous:
                    continue
                previous[next_state] = (px, py, cx, cy, gate_open, collect_mask)
                previous_action[next_state] = int(action_id)
                queue.append(next_state)

        if goal_state is None:
            raise RuntimeError("pushlane DSL failed to find a valid level program.")

        plan: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            plan.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[assignment]
        plan.reverse()

        if len(plan) >= TIME_LIMIT:
            raise RuntimeError(f"pushlane DSL found a plan of length {len(plan)}, exceeding time limit {TIME_LIMIT}.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = PushlaneDslAgent
