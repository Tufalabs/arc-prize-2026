from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "golf_grid_putt-0001"

GRID_WIDTH = 24
GRID_HEIGHT = 18
HUD_ROWS = 2
SEPARATOR_ROW = 2
TIMEBAR_LEN = 18

COLOR_HUD_BG = 0
COLOR_WATER = 1
COLOR_GRASS = 2
COLOR_SAND = 3
COLOR_ICE = 4
COLOR_WALL = 5
COLOR_HOLE_DARK = 6
COLOR_HOLE_PULSE = 7
COLOR_TIME_EMPTY = 8
COLOR_TIME_FULL = 9
COLOR_STROKE_FULL = 10
COLOR_STROKE_EMPTY = 11
COLOR_BALL_A = 12
COLOR_BALL_B = 13
COLOR_AIM = 14
COLOR_ACTIVE = 15

ACTION_SPACE_ID = int(GameAction.ACTION5.value)
ACTION_CLICK_ID = int(GameAction.ACTION6.value)

STATE_IDLE = "IDLE"
STATE_AIM = "AIM_PREVIEW"
STATE_MOVING = "MOVING"
STATE_SPLASH = "SPLASH"
STATE_LEVEL_WIN = "LEVEL_WIN"
STATE_LEVEL_FAIL = "LEVEL_FAIL"

STATE_TO_CODE = {STATE_IDLE: 0, STATE_AIM: 1, STATE_MOVING: 2, STATE_SPLASH: 3, STATE_LEVEL_WIN: 4, STATE_LEVEL_FAIL: 5}
CODE_TO_STATE = {v: k for k, v in STATE_TO_CODE.items()}

DIR8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


@dataclass(frozen=True)
class LevelSpec:
    stroke_limit: int
    time_max: int
    max_power: int
    rows: tuple[str, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        stroke_limit=4,
        time_max=120,
        max_power=10,
        rows=(
            "==================______",
            "++++____________________",
            "########################",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#................%%....#",
            "#................%%....#",
            "#......................#",
            "#......................#",
            "#...@..................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        stroke_limit=5,
        time_max=140,
        max_power=11,
        rows=(
            "==================______",
            "+++++___________________",
            "########################",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...######..#",
            "#..........#...#....#..#",
            "#..........#........#..#",
            "#..........#...#.%%.#..#",
            "#..........#...#.%%.#..#",
            "#..........#...######..#",
            "#......................#",
            "#..........#...........#",
            "#..........#...........#",
            "#...@......#...........#",
            "#..........#...........#",
            "########################",
        ),
    ),
    LevelSpec(
        stroke_limit=6,
        time_max=160,
        max_power=12,
        rows=(
            "==================______",
            "++++++__________________",
            "########################",
            "#......................#",
            "#............######....#",
            "#............#....#%%..#",
            "#....~~......#....#%%..#",
            "#....~~......######....#",
            "#............::::::....#",
            "#............::::::....#",
            "#......######::::::....#",
            "#......#....#..........#",
            "#......#....#....~~~~..#",
            "#......######..........#",
            "#...@..................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        stroke_limit=7,
        time_max=180,
        max_power=12,
        rows=(
            "==================______",
            "+++++++_________________",
            "########################",
            "#......................#",
            "#...........>>>>.......#",
            "#...........>>>>..%%...#",
            "#.....,,,,,,,,,,,,.%%..#",
            "#.....,,,,,,,,,,,,~~~~.#",
            "#.....,,,,,,,,,,,,~~~~.#",
            "#.....,,,,,,,,,,,,.....#",
            "#............::::......#",
            "#...@........::::......#",
            "#............::::......#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        stroke_limit=8,
        time_max=220,
        max_power=12,
        rows=(
            "==================______",
            "++++++++________________",
            "########################",
            "#......................#",
            "#..............%%......#",
            "#..............%%..~~~~#",
            "#......................#",
            "#.........00...........#",
            "#.........00...........#",
            "#......................#",
            "##########||||##########",
            "#......................#",
            "#...@..........00......#",
            "#..............00......#",
            "#..................::::#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        stroke_limit=9,
        time_max=260,
        max_power=12,
        rows=(
            "==================______",
            "+++++++++_______________",
            "########################",
            "#......................#",
            "#..................%%..#",
            "#..................%%..#",
            "#......>>>>..||||......#",
            "#......((.....,,,,,....#",
            "#......((.....,,,,,....#",
            "#............:::::.....#",
            "#............:::::.....#",
            "#......................#",
            "#....:::::.............#",
            "#....:::::..((.........#",
            "#..........~~((........#",
            "#...@..............00..#",
            "#..................00..#",
            "########################",
        ),
    ),
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _parse_level_model(level_index: int) -> dict:
    spec = LEVEL_SPECS[level_index]
    rows = spec.rows
    if len(rows) != GRID_HEIGHT:
        raise ValueError(f"Invalid level height at index {level_index}")

    tee = None
    holes: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    sands: set[tuple[int, int]] = set()
    ices: set[tuple[int, int]] = set()
    waters: set[tuple[int, int]] = set()
    winds: dict[tuple[int, int], tuple[int, int]] = {}
    gates: set[tuple[int, int]] = set()
    bumpers: set[tuple[int, int]] = set()
    portals: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        if len(row) != GRID_WIDTH:
            raise ValueError(f"Invalid level width at index {level_index} row {y}")
        for x, token in enumerate(row):
            if token == "@":
                tee = (x, y)
            elif token == "%":
                holes.add((x, y))
            elif token == "#":
                walls.add((x, y))
            elif token == "~":
                sands.add((x, y))
            elif token == ",":
                ices.add((x, y))
            elif token == ":":
                waters.add((x, y))
            elif token == ">":
                winds[(x, y)] = (1, 0)
            elif token == "<":
                winds[(x, y)] = (-1, 0)
            elif token == "^":
                winds[(x, y)] = (0, -1)
            elif token == "v":
                winds[(x, y)] = (0, 1)
            elif token == "|":
                gates.add((x, y))
            elif token == "0":
                bumpers.add((x, y))
            elif token == "(":
                portals.add((x, y))

    if tee is None:
        raise ValueError(f"Missing tee in level {level_index}")
    if len(holes) != 4:
        raise ValueError(f"Hole must be 2x2 in level {level_index}")

    portal_pads: list[list[tuple[int, int]]] = []
    if portals:
        remaining = set(portals)
        while remaining:
            start = next(iter(remaining))
            stack = [start]
            comp = set([start])
            remaining.remove(start)
            while stack:
                cx, cy = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in remaining:
                        remaining.remove((nx, ny))
                        comp.add((nx, ny))
                        stack.append((nx, ny))
            ordered = sorted(comp)
            if len(ordered) != 4:
                raise ValueError(f"Portal pad must be 2x2 in level {level_index}")
            portal_pads.append(ordered)

    if portal_pads and len(portal_pads) != 2:
        raise ValueError(f"Level {level_index} must have exactly two portal pads")

    steps_per_segment = math.ceil(spec.time_max / TIMEBAR_LEN)

    return {
        "level_index": int(level_index),
        "stroke_limit": int(spec.stroke_limit),
        "time_max": int(spec.time_max),
        "max_power": int(spec.max_power),
        "steps_per_segment": int(steps_per_segment),
        "tee": (int(tee[0]), int(tee[1])),
        "holes": frozenset((int(x), int(y)) for x, y in holes),
        "walls": frozenset((int(x), int(y)) for x, y in walls),
        "sands": frozenset((int(x), int(y)) for x, y in sands),
        "ices": frozenset((int(x), int(y)) for x, y in ices),
        "waters": frozenset((int(x), int(y)) for x, y in waters),
        "winds": {tuple(k): tuple(v) for k, v in winds.items()},
        "gates": frozenset((int(x), int(y)) for x, y in gates),
        "bumpers": frozenset((int(x), int(y)) for x, y in bumpers),
        "portal_pads": tuple(tuple(c for c in pad) for pad in portal_pads),
        "gate_period": 6,
        "gate_closed_steps": 4,
    }


def _initial_runtime_from_model(model: dict) -> dict:
    tee_x, tee_y = model["tee"]
    return {
        "state": STATE_IDLE,
        "ball": (int(tee_x), int(tee_y)),
        "dir": (0, 0),
        "power": 0,
        "preview_dir": (0, 0),
        "preview_power": 0,
        "preview_timer": 0,
        "time_remaining": int(model["time_max"]),
        "strokes_used": 0,
        "tick": 0,
        "splash_timer": 0,
        "splash_pos": (int(tee_x), int(tee_y)),
        "pending_tp": None,
        "portal_flash_timer": 0,
        "portal_flash_pos": None,
        "bumper_flash": {},
    }


def _gate_is_open(model: dict, tick: int) -> bool:
    if not model["gates"]:
        return True
    phase = (int(tick) - 1) % int(model["gate_period"])
    return phase >= int(model["gate_closed_steps"])


def _cell_solid(model: dict, x: int, y: int, tick: int) -> bool:
    if not _in_bounds(x, y):
        return True
    pos = (x, y)
    if pos in model["walls"]:
        return True
    if pos in model["bumpers"]:
        return True
    return bool(pos in model["gates"] and not _gate_is_open(model, tick))


def _terrain_token(model: dict, x: int, y: int, tick: int) -> str:
    pos = (x, y)
    if pos in model["walls"]:
        return "#"
    if pos in model["gates"] and not _gate_is_open(model, tick):
        return "|"
    if pos in model["bumpers"]:
        return "0"
    if pos in model["holes"]:
        return "%"
    if pos in model["waters"]:
        return ":"
    if pos in model["sands"]:
        return "~"
    if pos in model["ices"]:
        return ","
    if pos in model["winds"]:
        dx, dy = model["winds"][pos]
        if dx == 1:
            return ">"
        if dx == -1:
            return "<"
        if dy == -1:
            return "^"
        return "v"
    if any(pos in set(pad) for pad in model["portal_pads"]):
        return "("
    return "."


def _portal_destination(model: dict, x: int, y: int) -> tuple[int, int] | None:
    if not model["portal_pads"]:
        return None
    for idx, pad in enumerate(model["portal_pads"]):
        if (x, y) not in pad:
            continue
        other = model["portal_pads"][1 - idx]
        min_x = min(px for px, _ in pad)
        min_y = min(py for _, py in pad)
        off_x = x - min_x
        off_y = y - min_y
        other_min_x = min(px for px, _ in other)
        other_min_y = min(py for _, py in other)
        return int(other_min_x + off_x), int(other_min_y + off_y)
    return None


def _decrement_effect_timers(runtime: dict) -> None:
    if int(runtime["portal_flash_timer"]) > 0:
        runtime["portal_flash_timer"] = int(runtime["portal_flash_timer"]) - 1
        if int(runtime["portal_flash_timer"]) <= 0:
            runtime["portal_flash_pos"] = None

    flashes = {(int(x), int(y)): int(t) - 1 for (x, y), t in runtime["bumper_flash"].items() if int(t) - 1 > 0}
    runtime["bumper_flash"] = flashes


def _reflect_dir(model: dict, runtime: dict, forced_push: bool = False) -> tuple[bool, tuple[int, int], int]:
    bx, by = runtime["ball"]
    dx, dy = runtime["dir"]
    if dx == 0 and dy == 0:
        return False, (dx, dy), 0

    nx, ny = bx + dx, by + dy
    if not _cell_solid(model, nx, ny, int(runtime["tick"])):
        return False, (dx, dy), 0

    hx_blocked = dx != 0 and _cell_solid(model, bx + dx, by, int(runtime["tick"]))
    vy_blocked = dy != 0 and _cell_solid(model, bx, by + dy, int(runtime["tick"]))

    ndx, ndy = dx, dy
    if dx != 0 and dy != 0:
        if hx_blocked:
            ndx = -ndx
        if vy_blocked:
            ndy = -ndy
        if not hx_blocked and not vy_blocked:
            ndx = -ndx
            ndy = -ndy
    elif dx != 0:
        ndx = -ndx
    elif dy != 0:
        ndy = -ndy

    runtime["dir"] = (int(ndx), int(ndy))

    bounce_penalty = 0
    if forced_push:
        bounce_penalty = 1
    else:
        terrain = _terrain_token(model, bx, by, int(runtime["tick"]))
        if terrain == ",":
            bounce_penalty = 1

    if (nx, ny) in model["bumpers"]:
        runtime["power"] = min(int(model["max_power"]), int(runtime["power"]) + 1)
        runtime["bumper_flash"][(int(nx), int(ny))] = 2

    if bounce_penalty > 0:
        runtime["power"] = max(0, int(runtime["power"]) - int(bounce_penalty))

    return True, (int(ndx), int(ndy)), int(bounce_penalty)


def _apply_forced_wind_push(model: dict, runtime: dict) -> None:
    bx, by = runtime["ball"]
    wind = model["winds"].get((bx, by))
    if wind is None:
        return
    wdx, wdy = int(wind[0]), int(wind[1])

    original_dir = runtime["dir"]
    runtime["dir"] = (wdx, wdy)

    bounced, _, _ = _reflect_dir(model, runtime, forced_push=True)
    if not bounced:
        nx, ny = bx + wdx, by + wdy
        if not _cell_solid(model, nx, ny, int(runtime["tick"])):
            runtime["ball"] = (int(nx), int(ny))

    runtime["dir"] = original_dir


def _handle_landing(model: dict, runtime: dict) -> None:
    bx, by = runtime["ball"]
    pos = (bx, by)

    if pos in model["waters"]:
        runtime["state"] = STATE_SPLASH
        runtime["splash_timer"] = 2
        runtime["splash_pos"] = (int(bx), int(by))
        runtime["dir"] = (0, 0)
        runtime["power"] = 0
        runtime["preview_timer"] = 0
        return

    portal_dest = _portal_destination(model, bx, by)
    if portal_dest is not None:
        runtime["pending_tp"] = (int(portal_dest[0]), int(portal_dest[1]))


def _perform_movement_step(model: dict, runtime: dict) -> None:
    if int(runtime["power"]) <= 0:
        runtime["power"] = 0
        runtime["state"] = STATE_IDLE
        runtime["dir"] = (0, 0)
        return

    bounced, _, _ = _reflect_dir(model, runtime, forced_push=False)
    if bounced:
        if int(runtime["power"]) <= 0:
            runtime["power"] = 0
            runtime["state"] = STATE_IDLE
            runtime["dir"] = (0, 0)
        return

    bx, by = runtime["ball"]
    dx, dy = runtime["dir"]
    nx, ny = bx + dx, by + dy
    runtime["ball"] = (int(nx), int(ny))

    token = _terrain_token(model, nx, ny, int(runtime["tick"]))
    if token == "~":
        runtime["power"] = max(0, int(runtime["power"]) - 2)
    elif token == ",":
        pass
    else:
        runtime["power"] = max(0, int(runtime["power"]) - 1)

    _handle_landing(model, runtime)
    if runtime["state"] == STATE_SPLASH:
        return

    if runtime["state"] == STATE_MOVING and (nx, ny) in model["winds"]:
        _apply_forced_wind_push(model, runtime)
        _handle_landing(model, runtime)
        if runtime["state"] == STATE_SPLASH:
            return

    if int(runtime["power"]) <= 0:
        runtime["power"] = 0
        runtime["dir"] = (0, 0)
        runtime["state"] = STATE_IDLE


def _apply_portal_if_pending(runtime: dict) -> None:
    pending = runtime.get("pending_tp")
    if pending is None:
        return
    px, py = pending
    runtime["ball"] = (int(px), int(py))
    runtime["pending_tp"] = None
    runtime["portal_flash_timer"] = 1
    runtime["portal_flash_pos"] = (int(px), int(py))


def _resolve_click_shot(runtime: dict, click: tuple[int, int] | None, shot: tuple[int, int, int] | None, model: dict):
    if shot is not None:
        dx, dy, power = shot
        dx = int(_sign(int(dx)))
        dy = int(_sign(int(dy)))
        power = max(0, min(int(model["max_power"]), int(power)))
        if (dx, dy) == (0, 0) or power <= 0:
            return None
        return dx, dy, power

    if click is None:
        return None

    cx, cy = int(click[0]), int(click[1])
    bx, by = runtime["ball"]
    dx_raw = cx - bx
    dy_raw = cy - by
    dx = _sign(dx_raw)
    dy = _sign(dy_raw)
    if dx == 0 and dy == 0:
        return None
    power = max(abs(dx_raw), abs(dy_raw))
    power = max(1, min(int(model["max_power"]), int(power)))
    return int(dx), int(dy), int(power)


def apply_step_transition(
    model: dict,
    runtime_in: dict,
    action_id: int,
    *,
    click: tuple[int, int] | None = None,
    shot: tuple[int, int, int] | None = None,
) -> tuple[dict, str | None]:
    runtime = {
        **runtime_in,
        "ball": tuple(runtime_in["ball"]),
        "dir": tuple(runtime_in["dir"]),
        "preview_dir": tuple(runtime_in["preview_dir"]),
        "splash_pos": tuple(runtime_in["splash_pos"]),
        "bumper_flash": dict(runtime_in["bumper_flash"]),
    }

    if runtime_in.get("pending_tp") is not None:
        runtime["pending_tp"] = tuple(runtime_in["pending_tp"])
    if runtime_in.get("portal_flash_pos") is not None:
        runtime["portal_flash_pos"] = tuple(runtime_in["portal_flash_pos"])

    aid = int(action_id)

    if runtime["state"] in {STATE_LEVEL_WIN, STATE_LEVEL_FAIL}:
        runtime["tick"] = int(runtime["tick"]) + 1
        _decrement_effect_timers(runtime)
        if aid in {ACTION_SPACE_ID, ACTION_CLICK_ID}:
            if runtime["state"] == STATE_LEVEL_WIN:
                return runtime, "advance"
            return runtime, "lose"
        return runtime, None

    runtime["tick"] = int(runtime["tick"]) + 1
    _decrement_effect_timers(runtime)

    if runtime["state"] in {STATE_IDLE, STATE_AIM, STATE_MOVING}:
        _apply_portal_if_pending(runtime)

    if runtime["state"] == STATE_IDLE and aid == ACTION_CLICK_ID:
        resolved = _resolve_click_shot(runtime, click, shot, model)
        if resolved is not None:
            dx, dy, power = resolved
            runtime["preview_dir"] = (int(dx), int(dy))
            runtime["preview_power"] = int(power)
            runtime["preview_timer"] = 2
            runtime["state"] = STATE_AIM

    if runtime["state"] == STATE_AIM:
        runtime["preview_timer"] = int(runtime["preview_timer"]) - 1
        if int(runtime["preview_timer"]) <= 0:
            runtime["strokes_used"] = int(runtime["strokes_used"]) + 1
            runtime["dir"] = tuple(runtime["preview_dir"])
            runtime["power"] = int(runtime["preview_power"])
            runtime["state"] = STATE_MOVING

    elif runtime["state"] == STATE_MOVING:
        _perform_movement_step(model, runtime)

    elif runtime["state"] == STATE_SPLASH:
        runtime["splash_timer"] = int(runtime["splash_timer"]) - 1
        if int(runtime["splash_timer"]) <= 0:
            runtime["ball"] = tuple(model["tee"])
            runtime["dir"] = (0, 0)
            runtime["power"] = 0
            runtime["preview_timer"] = 0
            runtime["pending_tp"] = None
            runtime["strokes_used"] = int(runtime["strokes_used"]) + 1
            runtime["state"] = STATE_IDLE

    runtime["time_remaining"] = int(runtime["time_remaining"]) - 1

    if tuple(runtime["ball"]) in model["holes"]:
        runtime["state"] = STATE_LEVEL_WIN
    elif int(runtime["time_remaining"]) <= 0 or int(runtime["strokes_used"]) > int(model["stroke_limit"]):
        runtime["state"] = STATE_LEVEL_FAIL
        return runtime, "lose"

    return runtime, None


def _pack_runtime(runtime: dict) -> tuple[int, ...]:
    bx, by = runtime["ball"]
    dx, dy = runtime["dir"]
    pdx, pdy = runtime["preview_dir"]
    sx, sy = runtime["splash_pos"]
    ptx, pty = (-1, -1) if runtime.get("pending_tp") is None else runtime["pending_tp"]
    code = int(STATE_TO_CODE[runtime["state"]])
    return (
        code,
        int(bx),
        int(by),
        int(dx),
        int(dy),
        int(runtime["power"]),
        int(pdx),
        int(pdy),
        int(runtime["preview_power"]),
        int(runtime["preview_timer"]),
        int(runtime["time_remaining"]),
        int(runtime["strokes_used"]),
        int(runtime["tick"]),
        int(runtime["splash_timer"]),
        int(sx),
        int(sy),
        int(ptx),
        int(pty),
    )


def _unpack_runtime(state: tuple[int, ...]) -> dict:
    (
        code,
        bx,
        by,
        dx,
        dy,
        power,
        pdx,
        pdy,
        preview_power,
        preview_timer,
        time_remaining,
        strokes_used,
        tick,
        splash_timer,
        sx,
        sy,
        ptx,
        pty,
    ) = state
    return {
        "state": CODE_TO_STATE[int(code)],
        "ball": (int(bx), int(by)),
        "dir": (int(dx), int(dy)),
        "power": int(power),
        "preview_dir": (int(pdx), int(pdy)),
        "preview_power": int(preview_power),
        "preview_timer": int(preview_timer),
        "time_remaining": int(time_remaining),
        "strokes_used": int(strokes_used),
        "tick": int(tick),
        "splash_timer": int(splash_timer),
        "splash_pos": (int(sx), int(sy)),
        "pending_tp": None if int(ptx) < 0 else (int(ptx), int(pty)),
        "portal_flash_timer": 0,
        "portal_flash_pos": None,
        "bumper_flash": {},
    }


def initial_search_state_from_model(model: dict) -> tuple[int, ...]:
    return _pack_runtime(_initial_runtime_from_model(model))


def apply_action_transition(
    model: dict,
    packed_state: tuple[int, ...],
    action_id: int,
    *,
    click: tuple[int, int] | None = None,
    shot: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, ...], bool]:
    runtime = _unpack_runtime(packed_state)
    next_runtime, _ = apply_step_transition(model, runtime, int(action_id), click=click, shot=shot)
    won = next_runtime["state"] == STATE_LEVEL_WIN
    return _pack_runtime(next_runtime), bool(won)


def simulate_decision_transition(
    model: dict, packed_state: tuple[int, ...], command: tuple
) -> tuple[tuple[int, ...] | None, bool, list[tuple[int, dict[str, int]]]]:
    runtime = _unpack_runtime(packed_state)
    primitives: list[tuple[int, dict[str, int]]] = []

    state_name = runtime["state"]
    if state_name not in {STATE_IDLE, STATE_LEVEL_WIN, STATE_LEVEL_FAIL}:
        return None, False, primitives

    if command[0] == "wait":
        runtime, _ = apply_step_transition(model, runtime, ACTION_SPACE_ID)
        primitives.append((ACTION_SPACE_ID, {}))
    elif command[0] == "shot":
        _, dx, dy, power = command
        dx = int(dx)
        dy = int(dy)
        power = int(power)
        bx, by = runtime["ball"]
        target = (int(bx + (dx * power)), int(by + (dy * power)))
        runtime, _ = apply_step_transition(model, runtime, ACTION_CLICK_ID, click=target, shot=(dx, dy, power))
        primitives.append((ACTION_CLICK_ID, {"x": int(target[0]), "y": int(target[1])}))
    else:
        return None, False, primitives

    while runtime["state"] not in {STATE_IDLE, STATE_LEVEL_WIN, STATE_LEVEL_FAIL}:
        runtime, _ = apply_step_transition(model, runtime, ACTION_SPACE_ID)
        primitives.append((ACTION_SPACE_ID, {}))

    if runtime["state"] == STATE_LEVEL_FAIL:
        return None, False, primitives

    return _pack_runtime(runtime), bool(runtime["state"] == STATE_LEVEL_WIN), primitives


def _render_level_pixels(model: dict, runtime: dict) -> np.ndarray:
    pixels = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_HUD_BG, dtype=np.int8)
    tick = int(runtime["tick"])
    parity = tick % 2

    for y in range(SEPARATOR_ROW, GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            pos = (x, y)
            if pos in model["walls"]:
                pixels[y, x] = COLOR_WALL
            elif pos in model["gates"]:
                if _gate_is_open(model, tick):
                    pixels[y, x] = COLOR_GRASS
                else:
                    pixels[y, x] = COLOR_ACTIVE if parity == 0 else COLOR_STROKE_EMPTY
            elif pos in model["holes"]:
                pixels[y, x] = COLOR_HOLE_DARK if parity == 0 else COLOR_HOLE_PULSE
            elif pos in model["waters"]:
                pixels[y, x] = COLOR_WATER if ((x + y + parity) % 2 == 0) else COLOR_ACTIVE
            elif pos in model["sands"]:
                pixels[y, x] = COLOR_SAND
            elif pos in model["ices"]:
                pixels[y, x] = COLOR_ICE
            elif pos in model["winds"]:
                pixels[y, x] = COLOR_ACTIVE if parity == 0 else COLOR_HOLE_PULSE
            elif pos in model["bumpers"]:
                if runtime["bumper_flash"].get((x, y), 0) > 0:
                    pixels[y, x] = COLOR_ACTIVE
                else:
                    pixels[y, x] = COLOR_STROKE_EMPTY
            elif any((x, y) in set(pad) for pad in model["portal_pads"]):
                pixels[y, x] = COLOR_ACTIVE if parity == 0 else COLOR_HOLE_PULSE
            else:
                pixels[y, x] = COLOR_GRASS

    steps_per_segment = int(model["steps_per_segment"])
    fill_segments = max(0, min(TIMEBAR_LEN, math.ceil(max(0, runtime["time_remaining"]) / max(1, steps_per_segment))))
    for x in range(TIMEBAR_LEN):
        pixels[0, x] = COLOR_TIME_FULL if x < fill_segments else COLOR_TIME_EMPTY

    remaining_strokes = max(0, int(model["stroke_limit"]) - int(runtime["strokes_used"]))
    for x in range(int(model["stroke_limit"])):
        pixels[1, x] = COLOR_STROKE_FULL if x < remaining_strokes else COLOR_STROKE_EMPTY

    if runtime["state"] == STATE_AIM:
        bx, by = runtime["ball"]
        pdx, pdy = runtime["preview_dir"]
        pwr = int(runtime["preview_power"])
        trail: list[tuple[int, int]] = []
        for i in range(1, pwr + 1):
            tx, ty = bx + pdx * i, by + pdy * i
            if not _in_bounds(tx, ty):
                break
            trail.append((tx, ty))
        for tx, ty in trail[:-1]:
            pixels[ty, tx] = COLOR_AIM
        if trail:
            hx, hy = trail[-1]
            pixels[hy, hx] = COLOR_AIM

    if runtime["state"] == STATE_SPLASH:
        sx, sy = runtime["splash_pos"]
        if _in_bounds(sx, sy) and parity == 0:
            pixels[sy, sx] = COLOR_ACTIVE
    else:
        bx, by = runtime["ball"]
        if _in_bounds(bx, by):
            moving = runtime["state"] == STATE_MOVING
            pixels[by, bx] = COLOR_BALL_A if (not moving or parity == 0) else COLOR_BALL_B

    if runtime["portal_flash_pos"] is not None and int(runtime["portal_flash_timer"]) > 0:
        fx, fy = runtime["portal_flash_pos"]
        if _in_bounds(fx, fy):
            pixels[fy, fx] = COLOR_ACTIVE

    if runtime["state"] in {STATE_LEVEL_WIN, STATE_LEVEL_FAIL} and parity == 1:
        for y in range(HUD_ROWS):
            for x in range(TIMEBAR_LEN):
                if pixels[y, x] != COLOR_HUD_BG:
                    pixels[y, x] = COLOR_ACTIVE

    return pixels


def _build_level(index: int) -> Level:
    return Level(
        name=f"Golf Grid Putt {index + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[
            Sprite(
                pixels=_solid(GRID_WIDTH, GRID_HEIGHT, COLOR_HUD_BG),
                name="board",
                x=0,
                y=0,
                layer=0,
                tags=["board"],
                collidable=False,
            )
        ],
        data={"level_index": int(index)},
    )


def _deserialize_model(level: Level) -> dict:
    idx = int(level.get_data("level_index") or 0)
    return _parse_level_model(idx)


class GolfGridPutt(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._model: dict | None = None
        self._runtime: dict | None = None
        self._level_index = 0
        levels = [_build_level(i) for i in range(len(LEVEL_SPECS))]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_HUD_BG)
        super().__init__(
            game_id=GAME_ID, levels=levels, camera=camera, win_score=len(levels), available_actions=[5, 6], seed=seed
        )

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._level_index = int(level.get_data("level_index") or 0)
        self._runtime = _initial_runtime_from_model(self._model)
        self._sync_pixels()

    def _sync_pixels(self) -> None:
        if self._model is None or self._runtime is None:
            return
        board = self.current_level.get_sprites_by_name("board")
        if not board:
            return
        board[0].pixels = _render_level_pixels(self._model, self._runtime)

    def _parse_click_grid(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if not payload:
            return None
        try:
            display_x = int(payload.get("x", -1))
            display_y = int(payload.get("y", -1))
        except (TypeError, ValueError):
            return None

        grid = self.camera.display_to_grid(display_x, display_y)
        if grid is None:
            return None
        gx, gy = int(grid[0]), int(grid[1])
        if not _in_bounds(gx, gy):
            return None
        return gx, gy

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        if self._model is None or self._runtime is None:
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        click_grid = self._parse_click_grid() if action_id == ACTION_CLICK_ID else None

        self._runtime, event = apply_step_transition(self._model, self._runtime, action_id, click=click_grid)

        if event == "lose":
            self.lose()
            self.complete_action()
            return
        if event == "advance":
            if int(self._level_index) < len(LEVEL_SPECS) - 1:
                self.next_level()
                self.complete_action()
                return
            self.next_level()

        self._sync_pixels()
        self.complete_action()


__all__ = [
    "GAME_ID",
    "GolfGridPutt",
    "_deserialize_model",
    "apply_action_transition",
    "initial_search_state_from_model",
    "simulate_decision_transition",
]
