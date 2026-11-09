from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

COLOR_EMPTY = 0
COLOR_WALL = 1
COLOR_PADDLE = 2
COLOR_BALL = 3
COLOR_BALL_FAST = 4
COLOR_TRAIL = 5
COLOR_TIME_FULL = 6
COLOR_TIME_PARTIAL = 7
COLOR_TIME_EMPTY = 8
COLOR_DRAIN = 9
COLOR_BUMPER = 10
COLOR_BUMPER_FLASH = 11
COLOR_MIRROR = 12
COLOR_GATE = 13
COLOR_SPEED = 14
COLOR_ALERT = 15

PHASE_READY = 0
PHASE_PLAYING = 1
PHASE_FAIL = 2

ACTION_LEFT = int(GameAction.ACTION3.value)
ACTION_RIGHT = int(GameAction.ACTION4.value)
ACTION_SPACE = int(GameAction.ACTION5.value)

BALL = tuple[int, int, int, int, int]  # x, y, dx, dy, fast_timer
STATE = tuple[
    int,  # phase
    int,  # paddle_left
    int,  # serve_index
    int,  # time_left
    tuple[int, ...],  # gate pattern indices
    int,  # splitter_used
    int,  # splitter_flash
    tuple[int, ...],  # bumper flash timers aligned with bumper_cells
    tuple[BALL, ...],
]


LEVEL_GRIDS: list[list[str]] = [
    [
        "#========================#",
        "##########################",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#           o            #",
        "#       [-------]        #",
        "#!!!!!!!!!!!!!!!!!!!!!!!!#",
    ],
    [
        "#========================#",
        "##########################",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#     #            #     #",
        "#     #            #     #",
        "#     #            #     #",
        "#     #            #     #",
        "#                  #     #",
        "#     #            #     #",
        "#     #            #     #",
        "#     #                  #",
        "#     #            #     #",
        "#     #            #     #",
        "#                        #",
        "#           o            #",
        "#        [-----]         #",
        "#!!!!!!!!!!!!!!!!!!!!!!!!#",
    ],
    [
        "#========================#",
        "##########################",
        "#                        #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#     #            #     #",
        "#     #            #     #",
        "#     #            #     #",
        "#     #     **     #     #",
        "#           **     #     #",
        "#     #            #     #",
        "#  ** #            # **  #",
        "#  ** #              **  #",
        "#     #            #     #",
        "#     #            #     #",
        "#                        #",
        "#           o            #",
        "#        [-----]         #",
        "#!!!!!!!!!!!!!!!!!!!!!!!!#",
    ],
    [
        "#========================#",
        "##########################",
        "#                        #",
        "#                        #",
        "#                        #",
        "#        /\\    /\\        #",
        "#        \\/    \\/        #",
        "#                        #",
        "#           **           #",
        "#           **           #",
        "#                        #",
        "#           @            #",
        "#           @            #",
        "#           @            #",
        "#                        #",
        "#                        #",
        "#                        #",
        "#           o            #",
        "#         [---]          #",
        "#!!!!!!!!!!!!!!!!!!!!!!!!#",
    ],
    [
        "#####================#####",
        "##########################",
        "#####                #####",
        "#####                #####",
        "#####     >>>        #####",
        "#####                #####",
        "#####   <<<          #####",
        "#####                #####",
        "#####      /\\        #####",
        "#####      \\/        #####",
        "#####                #####",
        "#####                #####",
        "#####       @        #####",
        "#####       @        #####",
        "#####       @        #####",
        "#####                #####",
        "#####                #####",
        "#####       o        #####",
        "#####     [---]      #####",
        "#####!!!!!!!!!!!!!!!!#####",
    ],
    [
        "#####================#####",
        "##########################",
        "#####  /\\      /\\    #####",
        "#####  \\/      \\/    #####",
        "#####   >>>      <<< #####",
        "#####                #####",
        "##### **          ** #####",
        "##### **    %%    ** #####",
        "#####       %%       #####",
        "#####                #####",
        "#####           @    #####",
        "#####           @    #####",
        "#####    @      @    #####",
        "#####    @           #####",
        "#####    @           #####",
        "#####                #####",
        "#####                #####",
        "#####        o       #####",
        "#####     [---]      #####",
        "#####!!!!!!!!!!!!!!!!#####",
    ],
]

TIME_LIMITS = [100, 120, 135, 155, 180, 210]
SERVE_CYCLES = [
    [(0, -1)],
    [(1, -1), (-1, -1)],
    [(1, -1), (0, -1), (-1, -1)],
    [(1, -1), (-1, -1)],
    [(0, -1), (1, -1), (-1, -1)],
    [(1, -1), (-1, -1)],
]


@dataclass(frozen=True)
class GateDef:
    base_x: int
    ys: tuple[int, ...]
    pattern: tuple[int, ...]


def _normalize_grid(lines: list[str]) -> list[str]:
    width = max(len(row) for row in lines)
    return [row + (" " * (width - len(row))) for row in lines]


def _connected_components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    stack.append((nx, ny))
                    comp.append((nx, ny))
        components.append(comp)
    return components


def _parse_level(idx: int, lines_raw: list[str]) -> dict:
    lines = _normalize_grid(lines_raw)
    height = len(lines)
    width = len(lines[0]) if height else 0
    if width <= 0 or height <= 0:
        raise ValueError("pong_solo_training_walls level grid must be non-empty")

    walls: set[tuple[int, int]] = set()
    drains: set[tuple[int, int]] = set()
    bumpers: set[tuple[int, int]] = set()
    mirrors: dict[tuple[int, int], str] = {}
    speed_pads: set[tuple[int, int]] = set()
    splitter: set[tuple[int, int]] = set()
    gate_cells: set[tuple[int, int]] = set()

    ball_start: tuple[int, int] | None = None
    paddle_row: int | None = None
    paddle_left: int | None = None
    paddle_len: int | None = None

    for y, row in enumerate(lines):
        if len(row) != width:
            raise ValueError("pong_solo_training_walls rows must have equal width")
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "!":
                drains.add((x, y))
            elif ch == "*":
                bumpers.add((x, y))
            elif ch in {"/", "\\"}:
                mirrors[(x, y)] = ch
            elif ch in {">", "<", "{", "}"}:
                speed_pads.add((x, y))
            elif ch == "%":
                splitter.add((x, y))
            elif ch == "@":
                gate_cells.add((x, y))
            elif ch == "o":
                ball_start = (x, y)

        if "[" in row and "]" in row:
            left = row.index("[")
            right = row.index("]")
            paddle_row = y
            paddle_left = left
            paddle_len = right - left + 1

    if ball_start is None or paddle_row is None or paddle_left is None or paddle_len is None:
        raise ValueError("pong_solo_training_walls level missing ball/paddle")

    gate_defs: list[GateDef] = []
    if gate_cells:
        for comp in sorted(_connected_components(gate_cells), key=lambda c: min(x for x, _ in c)):
            xs = sorted({x for x, _ in comp})
            ys = tuple(sorted(y for _, y in comp))
            if len(xs) != 1:
                raise ValueError("moving gate components must be vertical columns")
            base_x = xs[0]
            gate_defs.append(GateDef(base_x=base_x, ys=ys, pattern=(base_x - 1, base_x, base_x + 1, base_x)))

    # Level-specific gate pacing/phasing for readability.
    if idx == 3 and gate_defs:
        gate_defs[0] = GateDef(base_x=gate_defs[0].base_x, ys=gate_defs[0].ys, pattern=(11, 12, 13, 12))
    if idx == 4 and gate_defs:
        gate_defs[0] = GateDef(base_x=gate_defs[0].base_x, ys=gate_defs[0].ys, pattern=(11, 12, 13, 12))
    if idx == 5 and len(gate_defs) >= 2:
        left_gate = gate_defs[0] if gate_defs[0].base_x <= gate_defs[1].base_x else gate_defs[1]
        right_gate = gate_defs[1] if left_gate is gate_defs[0] else gate_defs[0]
        left_gate = GateDef(
            base_x=left_gate.base_x,
            ys=left_gate.ys,
            pattern=(left_gate.base_x - 1, left_gate.base_x, left_gate.base_x + 1, left_gate.base_x),
        )
        right_gate = GateDef(
            base_x=right_gate.base_x,
            ys=right_gate.ys,
            pattern=(right_gate.base_x + 1, right_gate.base_x, right_gate.base_x - 1, right_gate.base_x),
        )
        gate_defs = [left_gate, right_gate]

    # Paddle movement bounds are computed from hard walls on paddle row.
    valid_lefts: list[int] = []
    for left in range(width - paddle_len + 1):
        if all((left + dx, paddle_row) not in walls for dx in range(paddle_len)):
            valid_lefts.append(left)
    if not valid_lefts:
        raise ValueError("pong_solo_training_walls level has no valid paddle range")

    return {
        "name": f"Level {idx + 1}",
        "width": width,
        "height": height,
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "drains": sorted((int(x), int(y)) for x, y in drains),
        "bumpers": sorted((int(x), int(y)) for x, y in bumpers),
        "mirrors": [
            [int(x), int(y), str(mtype)]
            for (x, y), mtype in sorted(mirrors.items(), key=lambda item: (item[0][1], item[0][0]))
        ],
        "speed_pads": sorted((int(x), int(y)) for x, y in speed_pads),
        "splitter": sorted((int(x), int(y)) for x, y in splitter),
        "gates": [
            {"base_x": int(g.base_x), "ys": [int(y) for y in g.ys], "pattern": [int(px) for px in g.pattern]}
            for g in gate_defs
        ],
        "ball_start": (int(ball_start[0]), int(ball_start[1])),
        "paddle_row": int(paddle_row),
        "paddle_left": int(paddle_left),
        "paddle_len": int(paddle_len),
        "paddle_min_left": int(min(valid_lefts)),
        "paddle_max_left": int(max(valid_lefts)),
        "time_limit": int(TIME_LIMITS[idx]),
        "serve_cycle": [[int(dx), int(dy)] for (dx, dy) in SERVE_CYCLES[idx]],
    }


def _gate_cells(model: dict, gate_indices: tuple[int, ...]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for gidx, gate in enumerate(model["gates"]):
        pattern = gate["pattern"]
        px = int(pattern[gate_indices[gidx] % len(pattern)])
        for y in gate["ys"]:
            out.add((px, int(y)))
    return out


def _ball_key(ball: BALL) -> tuple[int, int, int, int, int]:
    return (int(ball[0]), int(ball[1]), int(ball[2]), int(ball[3]), int(ball[4]))


def initial_search_state_from_model(model: dict) -> STATE:
    bx, by = model["ball_start"]
    return (
        PHASE_READY,
        int(model["paddle_left"]),
        0,
        int(model["time_limit"]),
        tuple(0 for _ in model["gates"]),
        0,
        0,
        tuple(0 for _ in model["bumper_cells"]),
        ((int(bx), int(by), 0, 0, 0),),
    )


def _normalize_model(raw: dict) -> dict:
    mirrors = {(int(x), int(y)): str(kind) for x, y, kind in raw.get("mirrors", [])}
    return {
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "walls": {tuple(int(v) for v in cell) for cell in raw.get("walls", [])},
        "drains": {tuple(int(v) for v in cell) for cell in raw.get("drains", [])},
        "bumper_cells": tuple(tuple(int(v) for v in cell) for cell in raw.get("bumpers", [])),
        "bumper_set": {tuple(int(v) for v in cell) for cell in raw.get("bumpers", [])},
        "mirrors": mirrors,
        "speed_pads": {tuple(int(v) for v in cell) for cell in raw.get("speed_pads", [])},
        "splitter_cells": tuple(tuple(int(v) for v in cell) for cell in raw.get("splitter", [])),
        "splitter_set": {tuple(int(v) for v in cell) for cell in raw.get("splitter", [])},
        "gates": [
            {
                "base_x": int(g["base_x"]),
                "ys": tuple(int(y) for y in g["ys"]),
                "pattern": tuple(int(px) for px in g["pattern"]),
            }
            for g in raw.get("gates", [])
        ],
        "ball_start": tuple(int(v) for v in raw["ball_start"]),
        "paddle_row": int(raw["paddle_row"]),
        "paddle_left": int(raw["paddle_left"]),
        "paddle_len": int(raw["paddle_len"]),
        "paddle_min_left": int(raw["paddle_min_left"]),
        "paddle_max_left": int(raw["paddle_max_left"]),
        "time_limit": int(raw["time_limit"]),
        "serve_cycle": [tuple(int(v) for v in pair) for pair in raw.get("serve_cycle", [])],
    }


def _deserialize_model(level: Level) -> dict:
    raw = dict(level.get_data("model") or {})
    if not raw:
        raise RuntimeError("pong_solo_training_walls level missing model data")
    return _normalize_model(raw)


def _move_paddle(model: dict, paddle_left: int, action_id: int) -> int:
    if action_id == ACTION_LEFT:
        paddle_left -= 1
    elif action_id == ACTION_RIGHT:
        paddle_left += 1
    return max(int(model["paddle_min_left"]), min(int(model["paddle_max_left"]), int(paddle_left)))


def _is_paddle_cell(model: dict, paddle_left: int, x: int, y: int) -> bool:
    return int(y) == int(model["paddle_row"]) and int(paddle_left) <= int(x) < int(paddle_left) + int(
        model["paddle_len"]
    )


def _axis_solid(model: dict, gates: set[tuple[int, int]], splitter_used: int, paddle_left: int, x: int, y: int) -> bool:
    cell = (int(x), int(y))
    if cell in model["walls"] or cell in gates or cell in model["bumper_set"]:
        return True
    if cell in model["mirrors"]:
        return True
    if splitter_used == 0 and cell in model["splitter_set"]:
        return True
    return _is_paddle_cell(model, paddle_left, cell[0], cell[1])


def _next_gate_indices(model: dict, gate_indices: tuple[int, ...]) -> tuple[int, ...]:
    if not model["gates"]:
        return gate_indices

    current = list(int(v) for v in gate_indices)
    occupied = _gate_cells(model, tuple(current))
    blocked_static = (
        set(model["walls"]) | set(model["bumper_set"]) | set(model["mirrors"].keys()) | set(model["splitter_set"])
    )

    for gidx, gate in enumerate(model["gates"]):
        pattern = gate["pattern"]
        cur_idx = current[gidx]
        nxt_idx = (cur_idx + 1) % len(pattern)

        cur_cells = {(int(pattern[cur_idx]), int(y)) for y in gate["ys"]}
        nxt_cells = {(int(pattern[nxt_idx]), int(y)) for y in gate["ys"]}
        occupied -= cur_cells

        can_move = True
        for cx, cy in nxt_cells:
            if cx < 0 or cy < 0 or cx >= int(model["width"]) or cy >= int(model["height"]):
                can_move = False
                break
            if (cx, cy) in occupied or (cx, cy) in blocked_static:
                can_move = False
                break

        if can_move:
            current[gidx] = nxt_idx
            occupied |= nxt_cells
        else:
            occupied |= cur_cells

    return tuple(current)


def _spawn_split_ball(
    model: dict,
    x: int,
    y: int,
    dx: int,
    dy: int,
    fast_timer: int,
    gates: set[tuple[int, int]],
    splitter_used: int,
    paddle_left: int,
    occupied_cells: set[tuple[int, int]],
) -> BALL | None:
    if splitter_used != 0:
        return None

    candidates = [(x + 1, y), (x - 1, y), (x, y - 1)]
    for cx, cy in candidates:
        if cx < 0 or cy < 0 or cx >= int(model["width"]) or cy >= int(model["height"]):
            continue
        if (cx, cy) in occupied_cells:
            continue
        if (cx, cy) in model["drains"]:
            continue
        if _axis_solid(model, gates, 1, paddle_left, cx, cy):
            continue
        new_dx = -int(dx)
        if new_dx == 0:
            new_dx = 1
        return (int(cx), int(cy), int(new_dx), int(dy), int(fast_timer))
    return None


def _advance_ball(
    model: dict,
    ball: BALL,
    paddle_left: int,
    gate_cells: set[tuple[int, int]],
    splitter_used: int,
    splitter_flash: int,
    bumper_timers: list[int],
    all_balls_before_step: tuple[BALL, ...],
) -> tuple[BALL, list[BALL], int, int, bool]:
    x, y, dx, dy, fast_timer = (int(v) for v in ball)
    spawned: list[BALL] = []
    fail_now = False

    micro_steps = 2 if fast_timer > 0 else 1

    for _ in range(micro_steps):
        nx = x + dx
        ny = y + dy

        if (nx, ny) in model["drains"]:
            x, y = nx, ny
            fail_now = True
            break

        if dy > 0 and _is_paddle_cell(model, paddle_left, nx, ny):
            dy = -1
            hit = nx - paddle_left
            if hit <= 0:
                dx = -1
            elif hit >= int(model["paddle_len"]) - 1:
                dx = 1
            else:
                dx = 0
            continue

        mirror = model["mirrors"].get((nx, ny))
        if mirror == "/":
            dx, dy = -dy, -dx
            continue
        if mirror == "\\":
            dx, dy = dy, dx
            continue

        blocked_x = dx != 0 and _axis_solid(model, gate_cells, splitter_used, paddle_left, x + dx, y)
        blocked_y = dy != 0 and _axis_solid(model, gate_cells, splitter_used, paddle_left, x, y + dy)
        blocked_target = _axis_solid(model, gate_cells, splitter_used, paddle_left, nx, ny)

        if blocked_x or blocked_y or blocked_target:
            if blocked_x and blocked_y:
                dx = -dx
                dy = -dy
            elif blocked_x:
                dx = -dx
            elif blocked_y:
                dy = -dy
            else:
                dx = -dx
                dy = -dy

            hit_cells = {(nx, ny), (x + dx, y), (x, y + dy)}
            for bidx, bcell in enumerate(model["bumper_cells"]):
                if bcell in hit_cells:
                    bumper_timers[bidx] = 2
            if dx == 0 and any(cell in model["bumper_set"] for cell in hit_cells):
                dx = -1 if x >= (int(model["width"]) // 2) else 1
            continue

        x, y = nx, ny

        if (x, y) in model["speed_pads"]:
            fast_timer = 20

        if splitter_used == 0 and (x, y) in model["splitter_set"] and len(all_balls_before_step) == 1:
            occupied = {(int(b[0]), int(b[1])) for b in all_balls_before_step}
            occupied.add((x, y))
            spawned_ball = _spawn_split_ball(
                model=model,
                x=x,
                y=y,
                dx=dx,
                dy=dy,
                fast_timer=fast_timer,
                gates=gate_cells,
                splitter_used=splitter_used,
                paddle_left=paddle_left,
                occupied_cells=occupied,
            )
            if spawned_ball is not None:
                spawned.append(spawned_ball)
                splitter_used = 1
                splitter_flash = 2

    if fast_timer > 0:
        fast_timer -= 1

    return (x, y, dx, dy, fast_timer), spawned, splitter_used, splitter_flash, fail_now


def apply_action_transition(model: dict, state: STATE, action_id: int) -> tuple[STATE | None, bool]:
    (phase, paddle_left, serve_index, time_left, gate_indices, splitter_used, splitter_flash, bumper_timers, balls) = (
        state
    )
    action_id = int(action_id)
    won = False

    if phase == PHASE_FAIL:
        return state, False

    if phase == PHASE_READY:
        new_paddle = _move_paddle(model, paddle_left, action_id)
        bx, by, _, _, _ = balls[0]
        bx += new_paddle - paddle_left
        new_balls = ((int(bx), int(by), 0, 0, 0),)
        if action_id == ACTION_SPACE:
            serve = model["serve_cycle"][serve_index % len(model["serve_cycle"])]
            dx, dy = int(serve[0]), int(serve[1])
            new_balls = ((int(bx), int(by), dx, dy, 0),)
            return (
                PHASE_PLAYING,
                int(new_paddle),
                int(serve_index + 1),
                int(time_left),
                tuple(gate_indices),
                int(splitter_used),
                int(splitter_flash),
                tuple(int(v) for v in bumper_timers),
                new_balls,
            ), False

        return (
            PHASE_READY,
            int(new_paddle),
            int(serve_index),
            int(time_left),
            tuple(gate_indices),
            int(splitter_used),
            int(splitter_flash),
            tuple(int(v) for v in bumper_timers),
            new_balls,
        ), False

    new_paddle = _move_paddle(model, paddle_left, action_id)
    new_gate_indices = _next_gate_indices(model, gate_indices)
    gate_cells = _gate_cells(model, new_gate_indices)

    bumper_list = [int(v) for v in bumper_timers]
    new_splitter_used = int(splitter_used)
    new_splitter_flash = int(splitter_flash)

    next_balls: list[BALL] = []
    fail_now = False

    balls_before_step = tuple(_ball_key(ball) for ball in balls)
    for ball in balls:
        updated, spawned, new_splitter_used, new_splitter_flash, ball_failed = _advance_ball(
            model=model,
            ball=ball,
            paddle_left=new_paddle,
            gate_cells=gate_cells,
            splitter_used=new_splitter_used,
            splitter_flash=new_splitter_flash,
            bumper_timers=bumper_list,
            all_balls_before_step=balls_before_step,
        )
        next_balls.append(updated)
        next_balls.extend(spawned)
        if ball_failed:
            fail_now = True
            break

    if new_splitter_flash > 0:
        new_splitter_flash -= 1
    bumper_list = [max(0, int(timer) - 1) if timer > 0 else 0 for timer in bumper_list]

    if fail_now:
        return (
            PHASE_FAIL,
            int(new_paddle),
            int(serve_index),
            int(time_left),
            tuple(new_gate_indices),
            int(new_splitter_used),
            int(new_splitter_flash),
            tuple(bumper_list),
            tuple(_ball_key(ball) for ball in next_balls),
        ), False

    new_time = int(time_left) - 1
    if new_time <= 0:
        won = True
        new_time = 0

    return (
        PHASE_PLAYING,
        int(new_paddle),
        int(serve_index),
        int(new_time),
        tuple(new_gate_indices),
        int(new_splitter_used),
        int(new_splitter_flash),
        tuple(bumper_list),
        tuple(_ball_key(ball) for ball in next_balls),
    ), won


def choose_policy_action(model: dict, state: STATE) -> int:
    (
        phase,
        paddle_left,
        _serve_index,
        _time_left,
        _gate_indices,
        _splitter_used,
        _splitter_flash,
        _bumper_timers,
        balls,
    ) = state

    if phase == PHASE_FAIL:
        return ACTION_SPACE
    if phase == PHASE_READY:
        return ACTION_SPACE

    # Track the most threatening descending ball and align paddle center to it.
    descending = [ball for ball in balls if int(ball[3]) > 0]
    if descending:
        target_ball = max(
            descending, key=lambda b: (int(b[1]), abs(int(b[0]) - (int(paddle_left) + (int(model["paddle_len"]) // 2))))
        )
    else:
        target_ball = max(balls, key=lambda b: int(b[1]))

    target_x = int(target_ball[0])
    center = int(paddle_left) + (int(model["paddle_len"]) // 2)
    if target_x < center:
        return ACTION_LEFT
    if target_x > center:
        return ACTION_RIGHT
    return ACTION_SPACE


def _build_level(model: dict) -> Level:
    width = int(model["width"])
    height = int(model["height"])

    pixels = np.full((height, width), COLOR_EMPTY, dtype=np.int8)
    sprite = Sprite(pixels=pixels, name="board", collidable=False, layer=1, tags=["board"])

    return Level(name=str(model["name"]), sprites=[sprite], grid_size=(width, height), data={"model": model})


def _validate_level_solvable(model: dict) -> None:
    state = initial_search_state_from_model(model)
    max_steps = int(model["time_limit"]) * 4
    for _ in range(max_steps):
        action_id = choose_policy_action(model, state)
        next_state, won = apply_action_transition(model, state, action_id)
        if next_state is None:
            break
        state = next_state
        if won:
            return
    raise ValueError(f"pong_solo_training_walls level `{model['name']}` is not solvable by DSL policy")


class PongSoloTrainingWalls(ARCBaseGame):
    def __init__(self, seed: int = 0):
        models = [_parse_level(idx, lines) for idx, lines in enumerate(LEVEL_GRIDS)]
        for model in models:
            _validate_level_solvable(_normalize_model(model))

        levels = [_build_level(model) for model in models]
        w0, h0 = levels[0].grid_size or (26, 20)
        camera = Camera(width=w0, height=h0, background=COLOR_EMPTY)
        super().__init__(
            game_id="pong_solo_training_walls-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE],
            seed=seed,
        )

        self._model: dict | None = None
        self._runtime_state: STATE | None = None
        self._board: Sprite | None = None
        self._trail_cells: set[tuple[int, int]] = set()
        self._flash_tick = 0
        self._route_score = 0

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._runtime_state = initial_search_state_from_model(self._model)
        boards = level.get_sprites_by_name("board")
        if not boards:
            raise RuntimeError("pong_solo_training_walls missing board sprite")
        self._board = boards[0]
        self._trail_cells = set()
        self._flash_tick = 0
        self._route_score = 0
        self._render()

    def _render(self) -> None:
        if self._board is None or self._model is None or self._runtime_state is None:
            return

        model = self._model
        (
            phase,
            paddle_left,
            _serve_index,
            time_left,
            gate_indices,
            splitter_used,
            _splitter_flash,
            bumper_timers,
            balls,
        ) = self._runtime_state
        width = int(model["width"])
        height = int(model["height"])

        pixels = np.full((height, width), COLOR_EMPTY, dtype=np.int8)

        for x, y in model["walls"]:
            pixels[y][x] = COLOR_WALL
        for x, y in model["drains"]:
            pixels[y][x] = COLOR_DRAIN
        for x, y in model["speed_pads"]:
            pixels[y][x] = COLOR_SPEED
        for (x, y), _mirror_type in model["mirrors"].items():
            pixels[y][x] = COLOR_MIRROR
        for bidx, (x, y) in enumerate(model["bumper_cells"]):
            pixels[y][x] = COLOR_BUMPER_FLASH if bumper_timers[bidx] > 0 else COLOR_BUMPER

        if splitter_used == 0:
            splitter_color = COLOR_ALERT
            for x, y in model["splitter_cells"]:
                pixels[y][x] = splitter_color

        for x, y in _gate_cells(model, gate_indices):
            pixels[y][x] = COLOR_GATE

        for x, y in self._trail_cells:
            if 0 <= x < width and 0 <= y < height and pixels[y][x] == COLOR_EMPTY:
                pixels[y][x] = COLOR_TRAIL

        for x in range(paddle_left, paddle_left + int(model["paddle_len"])):
            if 0 <= x < width:
                pixels[int(model["paddle_row"])][x] = COLOR_PADDLE

        for x, y, _dx, _dy, fast_timer in balls:
            if 0 <= x < width and 0 <= y < height:
                pixels[y][x] = COLOR_BALL_FAST if int(fast_timer) > 0 else COLOR_BALL

        inner = width - 2
        filled = round((max(0, int(time_left)) / float(max(1, int(model["time_limit"])))) * inner)
        for ix in range(inner):
            color = COLOR_TIME_FULL if ix < filled else COLOR_TIME_EMPTY
            pixels[0][ix + 1] = color
        pixels[0][0] = COLOR_WALL
        pixels[0][width - 1] = COLOR_WALL

        if phase == PHASE_FAIL and (self._flash_tick % 2 == 0):
            for x, y in model["drains"]:
                pixels[y][x] = COLOR_ALERT

        self._board.pixels = pixels

    def step(self) -> None:
        if self._model is None or self._runtime_state is None:
            self.complete_action()
            return

        action_id = _to_action_id(self.action.id)
        model = self._model
        (
            phase,
            _paddle_left,
            _serve_index,
            _time_left,
            _gate_indices,
            _splitter_used,
            _splitter_flash,
            _bumper_timers,
            balls,
        ) = self._runtime_state

        if phase == PHASE_FAIL:
            self.lose()
            self.complete_action()
            return

        if phase == PHASE_PLAYING:
            prior_cells = {(int(x), int(y)) for x, y, _dx, _dy, _ft in balls}
            next_state, won = apply_action_transition(model, self._runtime_state, action_id)
            if next_state is None:
                self.complete_action()
                return
            self._runtime_state = next_state
            _, _, _, _, _, _, _, _, next_balls = self._runtime_state
            next_cells = {(int(x), int(y)) for x, y, _dx, _dy, _ft in next_balls}
            self._trail_cells = prior_cells.union(next_cells - prior_cells)
            self._route_score += 1

            if self._runtime_state[0] == PHASE_FAIL:
                self.lose()
                self.complete_action()
                return

            if won:
                self.next_level()
                self.complete_action()
                return
        else:
            next_state, _won = apply_action_transition(model, self._runtime_state, action_id)
            if next_state is not None:
                self._runtime_state = next_state
            self._trail_cells = set()

        self._flash_tick += 1
        self._render()
        self.complete_action()


def _to_action_id(action_obj) -> int:
    value = getattr(action_obj, "value", action_obj)
    return int(value)
