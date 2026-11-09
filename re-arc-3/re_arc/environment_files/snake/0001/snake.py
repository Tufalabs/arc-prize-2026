from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "snake-0001"
WIDTH = 28
HEIGHT = 17
MAX_TIME = 28

COLOR_FLOOR = 0
COLOR_WALL = 1
COLOR_HEAD = 2
COLOR_BODY = 3
COLOR_FOOD = 4
COLOR_GLOW = 5
COLOR_POISON = 6
COLOR_TIME_FILL = 7
COLOR_TIME_EMPTY = 8
COLOR_GATE_CLOSED = 9
COLOR_GATE_OPEN = 10
COLOR_PORTAL_A = 11
COLOR_PORTAL_B = 12
COLOR_HAZARD = 13
COLOR_EXIT_ACTIVE = 14
COLOR_EXIT_INACTIVE = 15

MOVE_BY_ACTION = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_ORDER = (1, 4, 2, 3, 5)


class SnakeState(NamedTuple):
    snake: tuple[tuple[int, int], ...]
    direction: int
    growth_pending: int
    time_left: int
    foods_mask: int
    poisons_mask: int
    gate_phase: int
    blink_phase: int
    hazard_left: int
    hazard_dir: int


class _FailStep(RuntimeError):
    pass


def _normalize_row(row: str) -> str:
    out = row.replace(" ", ".")
    if len(out) != WIDTH:
        raise ValueError(f"Expected row width {WIDTH}, got {len(out)} for row {row!r}")
    return out


def _build_level_specs() -> list[dict]:
    specs = [
        {
            "name": "Level 1",
            "start_time": 26,
            "food_time_gain": 8,
            "rows": [
                "++++++++++++++++++++++++++--",
                "############################",
                "#..........................#",
                "#..........................#",
                "#..................$$......#",
                "#..................$$......#",
                "#..........................#",
                "#..........................#",
                "#..........................#",
                "#..OO>.....................#",
                "#..........................#",
                "#..........................#",
                "#......................:::.#",
                "#......................:::.#",
                "#......................:::.#",
                "#..........................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
        {
            "name": "Level 2",
            "start_time": 24,
            "food_time_gain": 8,
            "rows": [
                "++++++++++++++++++++++++----",
                "############################",
                "#..........................#",
                "#....######................#",
                "#....#....#.........$$.....#",
                "#....#....#.........$$.....#",
                "#....#....#...........:::..#",
                "#....######....#######:::..#",
                "#..............#......:::..#",
                "#..$$..........#...........#",
                "#..$$..........#...........#",
                "#..............#...........#",
                "#..............#######.....#",
                "#.....................$$...#",
                "#.....................$$...#",
                "#..OOO>....................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
        {
            "name": "Level 3",
            "start_time": 28,
            "food_time_gain": 12,
            "rows": [
                "++++++++++++++++++++++++++++",
                "############################",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#..........................#",
                "#..........:::.............#",
                "#..........:::.............#",
                "#..........:::.............#",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#..........................#",
                "#..........................#",
                "#..........................#",
                "#..OOOO>...................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
        {
            "name": "Level 4",
            "start_time": 28,
            "food_time_gain": 14,
            "rows": [
                "++++++++++++++++++++++++++++",
                "############################",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#..........................#",
                "#..........:::.............#",
                "#..........:::.............#",
                "#..........:::.............#",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#..........................#",
                "#..........................#",
                "#..........................#",
                "#..OOOO>...................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
        {
            "name": "Level 5",
            "start_time": 28,
            "food_time_gain": 14,
            "rows": [
                "++++++++++++++++++++++++++++",
                "############################",
                "#..........................#",
                "#....$$....!!....$$........#",
                "#....$$....!!....$$........#",
                "#..........................#",
                "#..........:::.............#",
                "#..........:::....$$.......#",
                "#..........:::....$$.......#",
                "#..........................#",
                "#....$$........!!...$$.....#",
                "#....$$........!!...$$.....#",
                "#..........................#",
                "#..........................#",
                "#..........................#",
                "#..OOOOO>..................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
        {
            "name": "Level 6",
            "start_time": 28,
            "food_time_gain": 16,
            "rows": [
                "++++++++++++++++++++++++++++",
                "############################",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#......................!!..#",
                "#..........:::.........!!..#",
                "#..........:::.............#",
                "#..........:::.............#",
                "#..........................#",
                "#....$$............$$......#",
                "#....$$............$$......#",
                "#..........................#",
                "#..........................#",
                "#..........................#",
                "#..OOOOO>..................#",
                "############################",
            ],
            "gate_cycle": (0, 0),
        },
    ]

    for spec in specs:
        spec["rows"] = [_normalize_row(row) for row in spec["rows"]]
        if len(spec["rows"]) != HEIGHT:
            raise ValueError(f"Level requires {HEIGHT} rows.")
    return specs


def _collect_blocks(rows: list[str], mark: str, block_w: int, block_h: int) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if rows[y][x] != mark or (x, y) in seen:
                continue
            cells: list[tuple[int, int]] = []
            for dy in range(block_h):
                for dx in range(block_w):
                    cx = x + dx
                    cy = y + dy
                    if cx >= WIDTH or cy >= HEIGHT or rows[cy][cx] != mark:
                        raise ValueError(f"Malformed {mark} block at {(x, y)}")
                    cells.append((cx, cy))
                    seen.add((cx, cy))
            out.append(tuple(cells))
    return out


def _collect_horizontal_block(rows: list[str], mark: str, run_len: int) -> tuple[tuple[int, int], ...] | None:
    for y in range(HEIGHT):
        for x in range(WIDTH - run_len + 1):
            if all(rows[y][x + i] == mark for i in range(run_len)):
                return tuple((x + i, y) for i in range(run_len))
    return None


def _parse_snake(rows: list[str]) -> tuple[tuple[tuple[int, int], ...], int]:
    heads: list[tuple[int, int, str]] = []
    body: set[tuple[int, int]] = set()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ch = rows[y][x]
            if ch in "^v<>":
                heads.append((x, y, ch))
            elif ch == "O":
                body.add((x, y))
    if len(heads) != 1:
        raise ValueError("Each level must have exactly one snake head.")

    hx, hy, hch = heads[0]
    dir_by_head = {"^": 0, "v": 1, "<": 2, ">": 3}
    direction = dir_by_head[hch]

    snake: list[tuple[int, int]] = [(hx, hy)]
    previous = (hx, hy)
    current_candidates = [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)]
    neck = None
    for candidate in current_candidates:
        if candidate in body:
            neck = candidate
            break

    if neck is None and body:
        raise ValueError("Snake body is disconnected from head.")

    while neck is not None:
        snake.append(neck)
        body.remove(neck)
        nx, ny = neck
        options = [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)]
        nxt = None
        for candidate in options:
            if candidate == previous:
                continue
            if candidate in body:
                nxt = candidate
                break
        previous, neck = neck, nxt

    if body:
        raise ValueError("Snake body contains disconnected cells.")

    return tuple(snake), direction


def _build_portal_map(
    portal_a: tuple[tuple[int, int], ...], portal_b: tuple[tuple[int, int], ...]
) -> dict[tuple[int, int], tuple[int, int]]:
    map_out: dict[tuple[int, int], tuple[int, int]] = {}
    if not portal_a or not portal_b:
        return map_out

    min_ax = min(x for x, _ in portal_a)
    min_ay = min(y for _, y in portal_a)
    min_bx = min(x for x, _ in portal_b)
    min_by = min(y for _, y in portal_b)

    b_offsets: dict[tuple[int, int], tuple[int, int]] = {}
    for bx, by in portal_b:
        b_offsets[(bx - min_bx, by - min_by)] = (bx, by)

    for ax, ay in portal_a:
        offset = (ax - min_ax, ay - min_ay)
        pair = b_offsets[offset]
        map_out[(ax, ay)] = pair
        map_out[pair] = (ax, ay)

    return map_out


def _parse_hazard(rows: list[str]) -> dict[str, int]:
    hazard = _collect_horizontal_block(rows, "=", 3)
    if not hazard:
        return {}

    y = hazard[0][1]
    left = min(x for x, _ in hazard)

    posts = [x for x in range(1, WIDTH - 1) if rows[y][x] == "#"]
    if len(posts) < 2:
        raise ValueError("Hazard track requires two interior posts.")
    left_post = min(posts)
    right_post = max(posts)

    min_left = left_post + 1
    max_left = right_post - 3
    if not (min_left <= left <= max_left):
        raise ValueError("Hazard initial position is outside track bounds.")

    return {"row": y, "left": left, "min_left": min_left, "max_left": max_left, "dir": 1}


def _parse_level(spec: dict) -> dict:
    rows = spec["rows"]
    timebar = rows[0]
    start_time = int(spec["start_time"])
    if timebar.count("+") != start_time:
        raise ValueError(f"Timebar mismatch: expected {start_time} but found {timebar.count('+')}")

    snake, direction = _parse_snake(rows)

    walls = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if rows[y][x] == "#"}
    foods = _collect_blocks(rows, "$", 2, 2)
    poisons = _collect_blocks(rows, "!", 2, 2)
    exits = _collect_blocks(rows, ":", 3, 3)
    exit_cells = frozenset(exits[0]) if exits else frozenset()

    gate = _collect_horizontal_block(rows, "|", 3)
    gate_cells = frozenset(gate or ())

    portals_a = _collect_blocks(rows, "(", 2, 2)
    portals_b = _collect_blocks(rows, ")", 2, 2)
    portal_a = portals_a[0] if portals_a else tuple()
    portal_b = portals_b[0] if portals_b else tuple()
    portal_map = _build_portal_map(portal_a, portal_b)

    hazard = _parse_hazard(rows)

    return {
        "name": spec["name"],
        "width": WIDTH,
        "height": HEIGHT,
        "start_time": start_time,
        "food_time_gain": int(spec["food_time_gain"]),
        "walls": tuple(sorted(walls)),
        "snake": tuple(snake),
        "direction": int(direction),
        "foods": tuple(tuple(block) for block in foods),
        "poisons": tuple(tuple(block) for block in poisons),
        "exit_cells": tuple(sorted(exit_cells)),
        "gate_cells": tuple(sorted(gate_cells)),
        "gate_closed_steps": int(spec["gate_cycle"][0]),
        "gate_open_steps": int(spec["gate_cycle"][1]),
        "portal_a": tuple(portal_a),
        "portal_b": tuple(portal_b),
        "portal_map": tuple((k[0], k[1], v[0], v[1]) for k, v in sorted(portal_map.items())),
        "hazard": hazard,
    }


LEVEL_MODELS: tuple[dict, ...] = tuple(_parse_level(spec) for spec in _build_level_specs())


def _portal_map_from_model(model: dict) -> dict[tuple[int, int], tuple[int, int]]:
    return {(int(ax), int(ay)): (int(bx), int(by)) for ax, ay, bx, by in (model.get("portal_map") or ())}


def hazard_cells(left: int, row: int) -> tuple[tuple[int, int], ...]:
    return ((left, row), (left + 1, row), (left + 2, row))


def gate_is_closed(model: dict, gate_phase: int) -> bool:
    closed = int(model.get("gate_closed_steps", 0) or 0)
    opened = int(model.get("gate_open_steps", 0) or 0)
    if closed <= 0 and opened <= 0:
        return False
    cycle = max(1, closed + opened)
    phase = gate_phase % cycle
    return phase < closed


def initial_search_state_from_model(model: dict) -> SnakeState:
    foods_count = len(model.get("foods") or ())
    poisons_count = len(model.get("poisons") or ())
    hazard = model.get("hazard") or {}
    return SnakeState(
        snake=tuple((int(x), int(y)) for x, y in (model.get("snake") or ())),
        direction=int(model.get("direction", 3)),
        growth_pending=0,
        time_left=int(model.get("start_time", 20)),
        foods_mask=(1 << foods_count) - 1,
        poisons_mask=(1 << poisons_count) - 1,
        gate_phase=0,
        blink_phase=0,
        hazard_left=int(hazard.get("left", -1)),
        hazard_dir=int(hazard.get("dir", 1)),
    )


def _food_cells_set(model: dict, foods_mask: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, block in enumerate(model.get("foods") or ()):
        if foods_mask & (1 << idx):
            out.update((int(x), int(y)) for x, y in block)
    return out


def _poison_cells_set(model: dict, poisons_mask: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, block in enumerate(model.get("poisons") or ()):
        if poisons_mask & (1 << idx):
            out.update((int(x), int(y)) for x, y in block)
    return out


def _transition_or_fail(model: dict, state: SnakeState, action_id: int) -> tuple[SnakeState, bool]:
    if state.time_left <= 0:
        raise _FailStep("timeout")

    direction = int(state.direction)
    if action_id in MOVE_BY_ACTION:
        requested = {1: 0, 2: 1, 3: 2, 4: 3}[action_id]
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}[direction]
        if requested != opposite:
            direction = requested

    head_x, head_y = state.snake[0]
    dx, dy = DIRS[direction]
    new_head = (head_x + dx, head_y + dy)

    walls = set((int(x), int(y)) for x, y in (model.get("walls") or ()))
    body = set(state.snake[1:])

    if new_head in walls:
        raise _FailStep("wall")
    if new_head in body:
        raise _FailStep("body")

    gate_cells = set((int(x), int(y)) for x, y in (model.get("gate_cells") or ()))
    if gate_cells and gate_is_closed(model, state.gate_phase) and new_head in gate_cells:
        raise _FailStep("gate")

    hazard = model.get("hazard") or {}
    if hazard:
        row = int(hazard["row"])
        for hx, hy in hazard_cells(state.hazard_left, row):
            if new_head == (hx, hy):
                raise _FailStep("hazard")

    foods_mask = int(state.foods_mask)
    poisons_mask = int(state.poisons_mask)
    growth_pending = int(state.growth_pending)
    time_delta = 0
    poison_shrink = 0

    foods = model.get("foods") or ()
    for idx, block in enumerate(foods):
        if not (foods_mask & (1 << idx)):
            continue
        if new_head in set((int(x), int(y)) for x, y in block):
            foods_mask &= ~(1 << idx)
            growth_pending += 2
            time_delta += int(model.get("food_time_gain", 0))
            break

    poisons = model.get("poisons") or ()
    for idx, block in enumerate(poisons):
        if not (poisons_mask & (1 << idx)):
            continue
        if new_head in set((int(x), int(y)) for x, y in block):
            poisons_mask &= ~(1 << idx)
            poison_shrink = 3
            time_delta -= 6
            break

    portal_map = _portal_map_from_model(model)
    if new_head in portal_map:
        new_head = portal_map[new_head]
        if new_head in walls:
            raise _FailStep("portal_wall")
        if new_head in body:
            raise _FailStep("portal_body")
        if gate_cells and gate_is_closed(model, state.gate_phase) and new_head in gate_cells:
            raise _FailStep("portal_gate")
        if hazard:
            row = int(hazard["row"])
            for hx, hy in hazard_cells(state.hazard_left, row):
                if new_head == (hx, hy):
                    raise _FailStep("portal_hazard")

    exit_cells = set((int(x), int(y)) for x, y in (model.get("exit_cells") or ()))
    won_immediate = (foods_mask == 0) and (new_head in exit_cells)

    snake = [new_head]
    snake.extend(state.snake)

    if growth_pending > 0:
        growth_pending -= 1
    else:
        snake.pop()

    if poison_shrink > 0:
        if len(snake) - poison_shrink < 2:
            raise _FailStep("poison")
        for _ in range(poison_shrink):
            snake.pop()

    if won_immediate:
        return (
            SnakeState(
                snake=tuple(snake),
                direction=direction,
                growth_pending=growth_pending,
                time_left=int(state.time_left),
                foods_mask=foods_mask,
                poisons_mask=poisons_mask,
                gate_phase=int(state.gate_phase),
                blink_phase=int(state.blink_phase),
                hazard_left=int(state.hazard_left),
                hazard_dir=int(state.hazard_dir),
            ),
            True,
        )

    time_left = int(state.time_left) - 1
    if time_left <= 0:
        raise _FailStep("timeout")
    time_left = max(0, min(MAX_TIME, time_left + time_delta))
    if time_left <= 0:
        raise _FailStep("timeout")

    gate_phase = int(state.gate_phase)
    cycle = int(model.get("gate_closed_steps", 0) or 0) + int(model.get("gate_open_steps", 0) or 0)
    if cycle > 0:
        gate_phase = (gate_phase + 1) % cycle

    hazard_left = int(state.hazard_left)
    hazard_dir = int(state.hazard_dir)
    if hazard:
        min_left = int(hazard["min_left"])
        max_left = int(hazard["max_left"])
        proposed = hazard_left + hazard_dir
        if proposed < min_left or proposed > max_left:
            hazard_dir *= -1
            proposed = hazard_left + hazard_dir
        hazard_left = proposed
        row = int(hazard["row"])
        hazard_after = set(hazard_cells(hazard_left, row))
        for cell in snake:
            if cell in hazard_after:
                raise _FailStep("hazard_move")

    blink_phase = 1 - int(state.blink_phase)

    won = False

    next_state = SnakeState(
        snake=tuple(snake),
        direction=direction,
        growth_pending=growth_pending,
        time_left=time_left,
        foods_mask=foods_mask,
        poisons_mask=poisons_mask,
        gate_phase=gate_phase,
        blink_phase=blink_phase,
        hazard_left=hazard_left,
        hazard_dir=hazard_dir,
    )
    return next_state, won


def apply_action_transition(model: dict, state: SnakeState, action_id: int) -> tuple[SnakeState | None, bool]:
    try:
        return _transition_or_fail(model, state, int(action_id))
    except _FailStep:
        return None, False


def search_plan_for_model(model: dict) -> list[int] | None:
    start = initial_search_state_from_model(model)

    def bfs_segment(
        start_state: SnakeState, goal_fn, *, max_nodes: int = 300000, max_depth: int = 180
    ) -> tuple[list[int], SnakeState] | None:
        queue = deque([start_state])
        prev: dict[SnakeState, SnakeState | None] = {start_state: None}
        prev_action: dict[SnakeState, int] = {}
        depth: dict[SnakeState, int] = {start_state: 0}
        visited_nodes = 0

        while queue:
            state = queue.popleft()
            visited_nodes += 1
            if visited_nodes > max_nodes:
                return None

            if goal_fn(state, False):
                actions: list[int] = []
                cursor = state
                while prev[cursor] is not None:
                    actions.append(prev_action[cursor])
                    cursor = prev[cursor]  # type: ignore[index]
                actions.reverse()
                return actions, state

            d = depth[state]
            if d >= max_depth:
                continue

            for action_id in ACTION_ORDER:
                next_state, won = apply_action_transition(model, state, action_id)
                if next_state is None:
                    continue
                if goal_fn(next_state, won):
                    prev[next_state] = state
                    prev_action[next_state] = action_id
                    actions: list[int] = []
                    cursor = next_state
                    while prev[cursor] is not None:
                        actions.append(prev_action[cursor])
                        cursor = prev[cursor]  # type: ignore[index]
                    actions.reverse()
                    return actions, next_state
                if next_state in prev:
                    continue
                prev[next_state] = state
                prev_action[next_state] = action_id
                depth[next_state] = d + 1
                queue.append(next_state)
        return None

    foods = model.get("foods") or ()
    state = start
    all_actions: list[int] = []

    while state.foods_mask != 0:
        remaining = [idx for idx in range(len(foods)) if state.foods_mask & (1 << idx)]
        if not remaining:
            break

        head = state.snake[0]
        target_idx = min(
            remaining, key=lambda idx: min(abs(head[0] - int(x)) + abs(head[1] - int(y)) for x, y in foods[idx])
        )
        target_bit = 1 << target_idx

        segment = bfs_segment(
            state, lambda s, _won, target_bit=target_bit: (s.foods_mask & target_bit) == 0, max_depth=120
        )
        if segment is None:
            return None
        seg_actions, state = segment
        all_actions.extend(seg_actions)

    final_segment = bfs_segment(state, lambda _s, won: bool(won), max_depth=180)
    if final_segment is not None:
        seg_actions, _ = final_segment
        all_actions.extend(seg_actions)
        return all_actions

    # Fallback: full-state A* when greedy staging cannot route around timing dynamics.
    def heuristic(s: SnakeState) -> int:
        head = s.snake[0]
        remaining_foods: list[tuple[int, int]] = []
        for idx, block in enumerate(model.get("foods") or ()):
            if s.foods_mask & (1 << idx):
                remaining_foods.extend((int(x), int(y)) for x, y in block)
        if remaining_foods:
            nearest = min(abs(head[0] - fx) + abs(head[1] - fy) for fx, fy in remaining_foods)
            return nearest + 2 * bin(s.foods_mask).count("1")
        exits = [(int(x), int(y)) for x, y in (model.get("exit_cells") or ())]
        if not exits:
            return 0
        return min(abs(head[0] - ex) + abs(head[1] - ey) for ex, ey in exits)

    heap: list[tuple[int, int, int, SnakeState]] = [(heuristic(start), 0, 0, start)]
    best_cost: dict[SnakeState, int] = {start: 0}
    previous: dict[SnakeState, SnakeState | None] = {start: None}
    prev_action: dict[SnakeState, int] = {}
    counter = 0
    expanded = 0
    max_expanded = 600000

    while heap and expanded < max_expanded:
        _f, g, _n, current = heappop(heap)
        expanded += 1
        if g != best_cost.get(current, -1):
            continue

        for action_id in ACTION_ORDER:
            next_state, won = apply_action_transition(model, current, action_id)
            if next_state is None:
                continue
            ng = g + 1
            old = best_cost.get(next_state)
            if old is not None and old <= ng:
                continue
            best_cost[next_state] = ng
            previous[next_state] = current
            prev_action[next_state] = action_id
            if won:
                actions: list[int] = []
                cursor = next_state
                while previous[cursor] is not None:
                    actions.append(prev_action[cursor])
                    cursor = previous[cursor]  # type: ignore[index]
                actions.reverse()
                return actions
            counter += 1
            heappush(heap, (ng + heuristic(next_state), ng, counter, next_state))

    return None


def _render_grid(model: dict, state: SnakeState) -> np.ndarray:
    grid = np.full((HEIGHT, WIDTH), COLOR_FLOOR, dtype=np.int8)

    for x, y in model.get("walls") or ():
        grid[int(y), int(x)] = np.int8(COLOR_WALL)

    filled = max(0, min(WIDTH, int(state.time_left)))
    grid[0, :] = np.int8(COLOR_TIME_EMPTY)
    if filled > 0:
        grid[0, :filled] = np.int8(COLOR_TIME_FILL)

    for idx, block in enumerate(model.get("foods") or ()):
        if not (state.foods_mask & (1 << idx)):
            continue
        color = COLOR_FOOD if state.blink_phase == 0 else COLOR_GLOW
        for x, y in block:
            grid[int(y), int(x)] = np.int8(color)

    for idx, block in enumerate(model.get("poisons") or ()):
        if not (state.poisons_mask & (1 << idx)):
            continue
        color = COLOR_POISON if state.blink_phase == 0 else COLOR_GLOW
        for x, y in block:
            grid[int(y), int(x)] = np.int8(color)

    gate_cells = model.get("gate_cells") or ()
    if gate_cells:
        gate_color = COLOR_GATE_CLOSED if gate_is_closed(model, state.gate_phase) else COLOR_GATE_OPEN
        for x, y in gate_cells:
            grid[int(y), int(x)] = np.int8(gate_color)

    portal_a = model.get("portal_a") or ()
    portal_b = model.get("portal_b") or ()
    color_a = COLOR_PORTAL_A if state.blink_phase == 0 else COLOR_GLOW
    color_b = COLOR_PORTAL_B if state.blink_phase == 0 else COLOR_GLOW
    for x, y in portal_a:
        grid[int(y), int(x)] = np.int8(color_a)
    for x, y in portal_b:
        grid[int(y), int(x)] = np.int8(color_b)

    hazard = model.get("hazard") or {}
    if hazard:
        for hx, hy in hazard_cells(state.hazard_left, int(hazard["row"])):
            grid[int(hy), int(hx)] = np.int8(COLOR_HAZARD)

    exit_cells = model.get("exit_cells") or ()
    exit_active = state.foods_mask == 0
    if exit_active:
        exit_color = COLOR_EXIT_ACTIVE if state.blink_phase == 0 else COLOR_EXIT_INACTIVE
    else:
        exit_color = COLOR_EXIT_INACTIVE
    for x, y in exit_cells:
        grid[int(y), int(x)] = np.int8(exit_color)

    for x, y in reversed(state.snake[1:]):
        grid[int(y), int(x)] = np.int8(COLOR_BODY)

    hx, hy = state.snake[0]
    grid[int(hy), int(hx)] = np.int8(COLOR_HEAD)

    return grid


def _build_level(level_index: int, model: dict) -> Level:
    grid = np.full((HEIGHT, WIDTH), COLOR_FLOOR, dtype=np.int8)
    return Level(
        name=model["name"],
        grid_size=(WIDTH, HEIGHT),
        sprites=[Sprite(pixels=grid, name="canvas", x=0, y=0, layer=0, tags=["canvas"], collidable=False)],
        data={"level_index": int(level_index), "model": model},
    )


class Snake(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._snake_state: SnakeState | None = None
        self._model: dict | None = None
        self._canvas: Sprite | None = None
        self._snake_score = 0

        levels = [_build_level(i, dict(model)) for i, model in enumerate(LEVEL_MODELS)]
        camera = Camera(width=WIDTH, height=HEIGHT, background=COLOR_FLOOR)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._model = dict(level.get_data("model") or {})
        self._snake_state = initial_search_state_from_model(self._model)
        self._canvas = self.current_level.get_sprites_by_name("canvas")[0]
        self._snake_score = 0
        self._redraw()

    def _redraw(self) -> None:
        if self._canvas is None or self._snake_state is None or self._model is None:
            return
        self._canvas.pixels = _render_grid(self._model, self._snake_state)

    def step(self) -> None:
        if self._snake_state is None or self._model is None:
            self.lose()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id not in (1, 2, 3, 4, 5):
            self._redraw()
            self.complete_action()
            return

        next_state, won = apply_action_transition(self._model, self._snake_state, action_id)

        if next_state is None:
            self.lose()
            self.complete_action()
            return

        self._snake_state = next_state
        self._snake_score = len(self._model.get("foods") or ()) - bin(self._snake_state.foods_mask).count("1")
        self._redraw()

        if won:
            self.next_level()

        self.complete_action()
