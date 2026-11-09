from __future__ import annotations

from collections import deque
from typing import Final

from ..core import CachedProgramDslAgent

VOID: Final[str] = "void"
FLOOR: Final[str] = "floor"
ARROW_N: Final[str] = "arrow_n"
ARROW_E: Final[str] = "arrow_e"
ARROW_S: Final[str] = "arrow_s"
ARROW_W: Final[str] = "arrow_w"
BEACON_NONE: Final[str] = "beacon_none"
BEACON_E: Final[str] = "beacon_e"

RED: Final[str] = "red"
BLUE: Final[str] = "blue"

PASSABLE_TILES: Final[set[str]] = {FLOOR, ARROW_N, ARROW_E, ARROW_S, ARROW_W, BEACON_NONE, BEACON_E}
DIRS: Final[tuple[tuple[int, int], ...]] = ((0, -1), (0, 1), (-1, 0), (1, 0))
ARROW_DELTAS: Final[dict[str, tuple[int, int]]] = {
    ARROW_N: (0, -1),
    ARROW_E: (1, 0),
    ARROW_S: (0, 1),
    ARROW_W: (-1, 0),
    BEACON_E: (1, 0),
}

LEVEL_SPECS: Final[list[dict[str, object]]] = [
    {
        "budget": 7,
        "active": RED,
        "red_start": (3, 3),
        "blue_start": (4, 5),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, ARROW_E, BEACON_NONE, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
        ),
    },
    {
        "budget": 8,
        "active": RED,
        "red_start": (3, 3),
        "blue_start": (4, 6),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, ARROW_E, BEACON_E, ARROW_W, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
        ),
    },
    {
        "budget": 16,
        "active": RED,
        "red_start": (1, 1),
        "blue_start": (6, 7),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, FLOOR, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, FLOOR, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, ARROW_E, ARROW_E, ARROW_E, BEACON_E, ARROW_W, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, FLOOR, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, FLOOR, FLOOR, VOID),
        ),
    },
]


def _click_data(cell: tuple[int, int]) -> dict[str, int]:
    return {"x": int(cell[0] * 8 + 4), "y": int(cell[1] * 8 + 4)}


def _continue_click() -> dict[str, int]:
    return {"x": 0, "y": 0}


class AlternatingHelpersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        observation = getattr(env, "observation_space", None)
        level_idx = int(getattr(observation, "levels_completed", 0) or 0)
        action_cells = _shortest_solution(level_idx)
        program = [(6, _click_data(cell)) for cell in action_cells]
        if level_idx < len(LEVEL_SPECS) - 1:
            program.append((6, _continue_click()))
        return program


AGENT_CLASS = AlternatingHelpersDslAgent


def _shortest_solution(level_idx: int) -> list[tuple[int, int]]:
    spec = LEVEL_SPECS[level_idx]
    rows = spec["rows"]
    red_start = tuple(spec["red_start"])  # type: ignore[arg-type]
    blue_start = tuple(spec["blue_start"])  # type: ignore[arg-type]
    active = str(spec["active"])
    remaining = int(spec["budget"])
    beacon = _find_beacon(rows)
    is_last = level_idx == len(LEVEL_SPECS) - 1

    start = (red_start, blue_start, active, remaining)
    queue = deque([start])
    previous: dict[tuple[object, ...], tuple[tuple[object, ...] | None, tuple[int, int] | None]] = {start: (None, None)}

    while queue:
        state = queue.popleft()
        for target_cell, next_state, outcome in _expand_state(rows=rows, state=state, beacon=beacon, is_last=is_last):
            if outcome == "solved":
                path = [target_cell]
                cursor = state
                while previous[cursor][0] is not None:
                    parent, action = previous[cursor]
                    path.append(action)  # type: ignore[arg-type]
                    cursor = parent  # type: ignore[assignment]
                path.reverse()
                return path
            if next_state in previous:
                continue
            previous[next_state] = (state, target_cell)
            queue.append(next_state)

    raise RuntimeError(f"No solution found for alternating_helpers level {level_idx}.")


def _expand_state(
    *,
    rows: tuple[tuple[str, ...], ...],
    state: tuple[tuple[int, int], tuple[int, int], str, int],
    beacon: tuple[int, int],
    is_last: bool,
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int], str, int] | None, str]]:
    red_pos, blue_pos, active, moves_left = state
    if moves_left <= 0:
        return []

    active_pos = red_pos if active == RED else blue_pos
    other_pos = blue_pos if active == RED else red_pos
    actions: list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int], str, int] | None, str]] = []

    clickable_cells = [other_pos]

    for dx, dy in DIRS:
        target = (active_pos[0] + dx, active_pos[1] + dy)
        if not _is_manual_move_valid(rows, target, active_pos, other_pos, beacon):
            continue
        clickable_cells.append(target)

    seen: set[tuple[int, int]] = set()
    for target in clickable_cells:
        if target in seen:
            continue
        seen.add(target)
        outcome, next_state = _resolve_click(
            rows=rows,
            red_pos=red_pos,
            blue_pos=blue_pos,
            active=active,
            moves_left=moves_left,
            click_cell=target,
            beacon=beacon,
            is_last=is_last,
        )
        if outcome == "noop":
            continue
        actions.append((target, next_state, outcome))

    return actions


def _resolve_click(
    *,
    rows: tuple[tuple[str, ...], ...],
    red_pos: tuple[int, int],
    blue_pos: tuple[int, int],
    active: str,
    moves_left: int,
    click_cell: tuple[int, int],
    beacon: tuple[int, int],
    is_last: bool,
) -> tuple[str, tuple[tuple[int, int], tuple[int, int], str, int] | None]:
    active_pos = red_pos if active == RED else blue_pos
    other_pos = blue_pos if active == RED else red_pos

    if _is_manual_move_valid(rows, click_cell, active_pos, other_pos, beacon):
        if active == RED:
            red_pos = click_cell
        else:
            blue_pos = click_cell
    elif click_cell == other_pos:
        active = BLUE if active == RED else RED
    else:
        return "noop", None

    return _resolve_success(
        rows=rows,
        red_pos=red_pos,
        blue_pos=blue_pos,
        active=active,
        moves_left=moves_left - 1,
        beacon=beacon,
        is_last=is_last,
    )


def _resolve_success(
    *,
    rows: tuple[tuple[str, ...], ...],
    red_pos: tuple[int, int],
    blue_pos: tuple[int, int],
    active: str,
    moves_left: int,
    beacon: tuple[int, int],
    _is_last: bool,
) -> tuple[str, tuple[tuple[int, int], tuple[int, int], str, int] | None]:
    if red_pos == blue_pos == beacon:
        return "solved", None

    helper_active = BLUE if active == RED else RED
    if helper_active == BLUE:
        blue_pos = _helper_move(rows, blue_pos, red_pos, beacon)
    else:
        red_pos = _helper_move(rows, red_pos, blue_pos, beacon)

    if moves_left <= 0:
        return "lost", None
    return "playing", (red_pos, blue_pos, active, moves_left)


def _helper_move(
    rows: tuple[tuple[str, ...], ...], helper_pos: tuple[int, int], active_pos: tuple[int, int], beacon: tuple[int, int]
) -> tuple[int, int]:
    delta = ARROW_DELTAS.get(rows[helper_pos[1]][helper_pos[0]])
    if delta is None:
        return helper_pos
    target = (helper_pos[0] + delta[0], helper_pos[1] + delta[1])
    if not _is_passable(rows, target):
        return helper_pos
    if target == active_pos and target != beacon:
        return helper_pos
    return target


def _is_manual_move_valid(
    rows: tuple[tuple[str, ...], ...],
    target: tuple[int, int],
    active_pos: tuple[int, int],
    other_pos: tuple[int, int],
    beacon: tuple[int, int],
) -> bool:
    if not _is_passable(rows, target):
        return False
    if abs(target[0] - active_pos[0]) + abs(target[1] - active_pos[1]) != 1:
        return False
    if target == other_pos and target != beacon:
        return False
    return True


def _is_passable(rows: tuple[tuple[str, ...], ...], pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < 8 and 0 <= pos[1] < 8 and rows[pos[1]][pos[0]] in PASSABLE_TILES


def _find_beacon(rows: tuple[tuple[str, ...], ...]) -> tuple[int, int]:
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            if tile in {BEACON_NONE, BEACON_E}:
                return (x, y)
    raise ValueError("Beacon missing from level.")
