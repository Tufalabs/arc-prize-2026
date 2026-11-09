from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "moving_platforms_manual_animation-0001"

WIDTH = 32
HEIGHT = 18
PLAY_MIN_Y = 1
PLAY_MAX_Y = HEIGHT - 1

C_VOID = 0
C_WALL = 1
C_GROUND = 2
C_PLAYER_A = 3
C_PLAYER_B = 4
C_EXIT_A = 5
C_EXIT_B = 6
C_PLATFORM = 7
C_PLATFORM_EDGE = 8
C_SPIKE_A = 9
C_SPIKE_B = 10
C_BUTTON_OFF = 11
C_BUTTON_ON = 12
C_GATE_CLOSED = 13
C_GATE_ANIM = 14
C_TIME_FILL = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION = int(GameAction.ACTION5.value)
CLICK_ACTION = int(GameAction.ACTION6.value)

T_WALL = 1
T_GROUND = 2
T_VOID = 3
T_SPIKE = 4
T_GATE = 5

GATE_CLOSED = 0
GATE_OPENING = 1
GATE_OPEN = 2
GATE_CLOSING = 3


@dataclass(frozen=True)
class PlatformModel:
    mode: str
    x: int
    y: int
    dir_sign: int = 1
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0
    loop_index: int = 0
    loop_points: tuple[tuple[int, int], ...] = ()
    route_id: int = 0
    route_ranges: tuple[tuple[int, int], ...] = ()
    pending_route_id: int = -1
    reverse_pending: int = 0
    last_dx: int = 0
    last_dy: int = 0


@dataclass(frozen=True)
class LevelRuntimeState:
    px: int
    py: int
    time_left: int
    platforms: tuple[PlatformModel, ...]
    gate_timer: int
    gate_phase: int
    lever_on: int


@dataclass(frozen=True)
class ActionToken:
    action_id: int
    click_x: int = -1
    click_y: int = -1


@dataclass(frozen=True)
class LevelModel:
    name: str
    time_limit: int
    terrain: tuple[tuple[int, ...], ...]
    start: tuple[int, int]
    exits: tuple[tuple[int, int], ...]
    buttons: tuple[tuple[int, int], ...]
    levers: tuple[tuple[int, int], ...]
    gate_cells: tuple[tuple[int, int], ...]
    gate_duration: int
    platforms: tuple[PlatformModel, ...]
    has_spikes: int


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _make_loop_points(min_x: int, max_x: int, min_y: int, max_y: int) -> tuple[tuple[int, int], ...]:
    points: list[tuple[int, int]] = []
    for x in range(min_x, max_x + 1):
        points.append((x, min_y))
    for y in range(min_y + 1, max_y + 1):
        points.append((max_x, y))
    for x in range(max_x - 1, min_x - 1, -1):
        points.append((x, max_y))
    for y in range(max_y - 1, min_y, -1):
        points.append((min_x, y))
    return tuple(points)


def _normalize_row(row: str) -> str:
    out = row.rstrip("\n")
    if len(out) != WIDTH:
        raise ValueError(f"row has width={len(out)} expected={WIDTH}: {out!r}")
    return out


def _parse_layout(raw_rows: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    if len(raw_rows) != HEIGHT:
        raise ValueError(f"layout must have {HEIGHT} rows")

    parsed: list[str] = []
    platform_starts: list[tuple[int, int]] = []

    for y, source in enumerate(raw_rows):
        src = source.rstrip("\n")
        built: list[str] = []
        i = 0
        x = 0
        while i < len(src):
            if src.startswith("[-]", i):
                platform_starts.append((x, y))
                built.extend([".", ".", "."])
                i += 3
                x += 3
                continue
            built.append(src[i])
            i += 1
            x += 1
        row = _normalize_row("".join(built))
        parsed.append(row)

    return parsed, platform_starts


def _build_level_model(
    *, name: str, time_limit: int, rows: list[str], platform_factories: list, gate_duration: int = 0
) -> LevelModel:
    parsed_rows, starts = _parse_layout(rows)

    terrain: list[list[int]] = [[T_VOID for _ in range(WIDTH)] for _ in range(HEIGHT)]
    start = (-1, -1)
    exits: list[tuple[int, int]] = []
    buttons: list[tuple[int, int]] = []
    levers: list[tuple[int, int]] = []
    gates: list[tuple[int, int]] = []
    has_spikes = 0

    for y, row in enumerate(parsed_rows):
        for x, ch in enumerate(row):
            if y == 0:
                terrain[y][x] = T_VOID
                continue
            if ch == "#":
                terrain[y][x] = T_WALL
            elif ch == "=":
                terrain[y][x] = T_GROUND
            elif ch == ".":
                terrain[y][x] = T_VOID
            elif ch == "^":
                terrain[y][x] = T_SPIKE
                has_spikes = 1
            elif ch == "@":
                terrain[y][x] = T_GROUND
                start = (x, y)
            elif ch == "X":
                terrain[y][x] = T_GROUND
                exits.append((x, y))
            elif ch == "o":
                terrain[y][x] = T_GROUND
                buttons.append((x, y))
            elif ch == "!":
                terrain[y][x] = T_GROUND
                levers.append((x, y))
            elif ch == "|":
                terrain[y][x] = T_GATE
                gates.append((x, y))
            elif ch == "t":
                terrain[y][x] = T_VOID
            else:
                raise ValueError(f"unsupported layout char {ch!r}")

    if start[0] < 0:
        raise ValueError(f"{name}: missing start")
    if not exits:
        raise ValueError(f"{name}: missing exit")

    if len(starts) != len(platform_factories):
        raise ValueError(f"{name}: platform markers={len(starts)} factories={len(platform_factories)}")

    platforms: list[PlatformModel] = []
    for idx, factory in enumerate(platform_factories):
        sx, sy = starts[idx]
        platforms.append(factory(sx, sy))

    return LevelModel(
        name=name,
        time_limit=int(time_limit),
        terrain=tuple(tuple(int(v) for v in row) for row in terrain),
        start=(int(start[0]), int(start[1])),
        exits=tuple((int(x), int(y)) for x, y in exits),
        buttons=tuple((int(x), int(y)) for x, y in buttons),
        levers=tuple((int(x), int(y)) for x, y in levers),
        gate_cells=tuple((int(x), int(y)) for x, y in gates),
        gate_duration=int(gate_duration),
        platforms=tuple(platforms),
        has_spikes=int(has_spikes),
    )


def _platform_cells(p: PlatformModel) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return ((p.x, p.y), (p.x + 1, p.y), (p.x + 2, p.y))


def _advance_platform(p: PlatformModel) -> PlatformModel:
    if p.mode == "bounce_x":
        direction = int(p.dir_sign)
        if p.reverse_pending:
            direction = -direction
        nx = int(p.x) + direction
        if nx < int(p.min_x) or nx > int(p.max_x):
            direction = -direction
            nx = int(p.x) + direction
        return PlatformModel(
            mode=p.mode,
            x=int(nx),
            y=int(p.y),
            dir_sign=int(direction),
            min_x=int(p.min_x),
            max_x=int(p.max_x),
            min_y=int(p.min_y),
            max_y=int(p.max_y),
            loop_index=int(p.loop_index),
            loop_points=p.loop_points,
            route_id=int(p.route_id),
            route_ranges=p.route_ranges,
            pending_route_id=int(p.pending_route_id),
            reverse_pending=0,
            last_dx=int(nx - p.x),
            last_dy=0,
        )

    if p.mode == "bounce_y":
        direction = int(p.dir_sign)
        ny = int(p.y) + direction
        if ny < int(p.min_y) or ny > int(p.max_y):
            direction = -direction
            ny = int(p.y) + direction
        return PlatformModel(
            mode=p.mode,
            x=int(p.x),
            y=int(ny),
            dir_sign=int(direction),
            min_x=int(p.min_x),
            max_x=int(p.max_x),
            min_y=int(p.min_y),
            max_y=int(p.max_y),
            loop_index=int(p.loop_index),
            loop_points=p.loop_points,
            route_id=int(p.route_id),
            route_ranges=p.route_ranges,
            pending_route_id=int(p.pending_route_id),
            reverse_pending=0,
            last_dx=0,
            last_dy=int(ny - p.y),
        )

    if p.mode == "loop":
        if not p.loop_points:
            return p
        old_x = int(p.x)
        old_y = int(p.y)
        next_idx = (int(p.loop_index) + 1) % len(p.loop_points)
        nx, ny = p.loop_points[next_idx]
        return PlatformModel(
            mode=p.mode,
            x=int(nx),
            y=int(ny),
            dir_sign=int(p.dir_sign),
            min_x=int(p.min_x),
            max_x=int(p.max_x),
            min_y=int(p.min_y),
            max_y=int(p.max_y),
            loop_index=int(next_idx),
            loop_points=p.loop_points,
            route_id=int(p.route_id),
            route_ranges=p.route_ranges,
            pending_route_id=int(p.pending_route_id),
            reverse_pending=0,
            last_dx=int(nx - old_x),
            last_dy=int(ny - old_y),
        )

    if p.mode == "route_bounce_x":
        route_id = int(p.route_id)
        ranges = p.route_ranges
        min_x, max_x = ranges[route_id]
        direction = int(p.dir_sign)
        nx = int(p.x) + direction
        if nx < int(min_x) or nx > int(max_x):
            direction = -direction
            nx = int(p.x) + direction

        pending = int(p.pending_route_id)
        next_route = route_id
        if pending >= 0:
            next_route = pending
            target_min, target_max = ranges[next_route]
            nx = max(int(target_min), min(int(target_max), int(nx)))
            pending = -1

        return PlatformModel(
            mode=p.mode,
            x=int(nx),
            y=int(p.y),
            dir_sign=int(direction),
            min_x=int(p.min_x),
            max_x=int(p.max_x),
            min_y=int(p.min_y),
            max_y=int(p.max_y),
            loop_index=int(p.loop_index),
            loop_points=p.loop_points,
            route_id=int(next_route),
            route_ranges=p.route_ranges,
            pending_route_id=int(pending),
            reverse_pending=0,
            last_dx=int(nx - p.x),
            last_dy=0,
        )

    raise ValueError(f"unsupported platform mode {p.mode}")


def _gate_blocks(phase: int) -> bool:
    return int(phase) != GATE_OPEN


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= int(x) < WIDTH and PLAY_MIN_Y <= int(y) <= PLAY_MAX_Y


def _platform_occupancy(platforms: tuple[PlatformModel, ...]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for p in platforms:
        occupied.update(_platform_cells(p))
    return occupied


def _is_passable(model: LevelModel, state: LevelRuntimeState, x: int, y: int) -> bool:
    if not _in_bounds(x, y):
        return False

    occupied = _platform_occupancy(state.platforms)
    if (x, y) in occupied:
        return True

    tile = int(model.terrain[y][x])
    if tile == T_WALL:
        return False
    if tile == T_SPIKE:
        return False
    if tile == T_VOID:
        return False
    if tile == T_GATE:
        return not _gate_blocks(state.gate_phase)
    return True


def _parse_click(action: ActionToken, camera: Camera | None = None) -> tuple[int, int] | None:
    x = int(action.click_x)
    y = int(action.click_y)
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        return x, y
    if camera is None:
        return None
    try:
        grid_pos = camera.display_to_grid(x, y)
    except Exception:
        return None
    if grid_pos is None:
        return None
    gx = int(grid_pos[0])
    gy = int(grid_pos[1])
    if 0 <= gx < WIDTH and 0 <= gy < HEIGHT:
        return gx, gy
    return None


def _apply_gate_state_machine(current_phase: int, should_open: bool) -> int:
    phase = int(current_phase)
    if should_open:
        if phase == GATE_OPEN:
            return GATE_OPEN
        return GATE_OPENING if phase != GATE_OPENING else GATE_OPEN

    if phase == GATE_CLOSED:
        return GATE_CLOSED
    return GATE_CLOSING if phase != GATE_CLOSING else GATE_CLOSED


def transition_level_state(
    model: LevelModel, state: LevelRuntimeState, action: ActionToken, *, camera: Camera | None = None
) -> tuple[LevelRuntimeState | None, bool]:
    px = int(state.px)
    py = int(state.py)
    platforms = tuple(state.platforms)
    gate_phase = int(state.gate_phase)
    gate_timer = int(state.gate_timer)
    lever_on = int(state.lever_on)

    # A) Apply action in current state.
    aid = int(action.action_id)
    if aid in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[aid]
        tx = px + dx
        ty = py + dy
        if _is_passable(model, state, tx, ty):
            px, py = tx, ty
    elif aid == CLICK_ACTION:
        clicked = _parse_click(action, camera=camera)
        if clicked is not None and model.levers:
            cx, cy = clicked
            adjacent = abs(cx - px) + abs(cy - py) == 1
            if adjacent and (cx, cy) in set(model.levers):
                lever_on = 0 if lever_on else 1
                next_platforms = list(platforms)
                for idx, p in enumerate(next_platforms):
                    if p.mode == "route_bounce_x":
                        target_route = 1 if lever_on else 0
                        next_platforms[idx] = PlatformModel(
                            mode=p.mode,
                            x=int(p.x),
                            y=int(p.y),
                            dir_sign=int(p.dir_sign),
                            min_x=int(p.min_x),
                            max_x=int(p.max_x),
                            min_y=int(p.min_y),
                            max_y=int(p.max_y),
                            loop_index=int(p.loop_index),
                            loop_points=p.loop_points,
                            route_id=int(p.route_id),
                            route_ranges=p.route_ranges,
                            pending_route_id=int(target_route),
                            reverse_pending=int(p.reverse_pending),
                            last_dx=int(p.last_dx),
                            last_dy=int(p.last_dy),
                        )
                    elif p.mode == "bounce_x" and idx == len(next_platforms) - 1 and model.name.endswith("L6"):
                        next_platforms[idx] = PlatformModel(
                            mode=p.mode,
                            x=int(p.x),
                            y=int(p.y),
                            dir_sign=int(p.dir_sign),
                            min_x=int(p.min_x),
                            max_x=int(p.max_x),
                            min_y=int(p.min_y),
                            max_y=int(p.max_y),
                            loop_index=int(p.loop_index),
                            loop_points=p.loop_points,
                            route_id=int(p.route_id),
                            route_ranges=p.route_ranges,
                            pending_route_id=int(p.pending_route_id),
                            reverse_pending=1,
                            last_dx=int(p.last_dx),
                            last_dy=int(p.last_dy),
                        )
                platforms = tuple(next_platforms)

    # B1) Move all platforms.
    old_platforms = tuple(platforms)
    _platform_occupancy(old_platforms)

    moved_platforms = tuple(_advance_platform(p) for p in old_platforms)
    new_occupancy = _platform_occupancy(moved_platforms)

    # B2/B3) Ride and crush check.
    carry_dx = 0
    carry_dy = 0
    for old, new in zip(old_platforms, moved_platforms, strict=False):
        if (px, py) in set(_platform_cells(old)):
            carry_dx = int(new.x - old.x)
            carry_dy = int(new.y - old.y)
            break

    if carry_dx != 0 or carry_dy != 0:
        target_x = px + carry_dx
        target_y = py + carry_dy
        if not _in_bounds(target_x, target_y):
            return None, False
        tile = int(model.terrain[target_y][target_x])
        if tile == T_WALL:
            return None, False
        if tile == T_GATE and _gate_blocks(gate_phase):
            return None, False
        px, py = target_x, target_y

    # B4) Hazard check.
    tile = int(model.terrain[py][px])
    if tile == T_SPIKE:
        return None, False
    if tile == T_VOID and (px, py) not in new_occupancy:
        return None, False

    # B5) Buttons and gates.
    on_button = (px, py) in set(model.buttons)
    if model.gate_duration > 0:
        if on_button:
            gate_timer = int(model.gate_duration)
        elif gate_timer > 0:
            gate_timer -= 1
        gate_phase = _apply_gate_state_machine(gate_phase, gate_timer > 0)

    # B6) Animation is visual-only.

    # Win before timeout decrement.
    requires_lever = model.name.endswith("L6")
    if (px, py) in set(model.exits) and (not requires_lever or int(lever_on) == 1):
        return LevelRuntimeState(
            px=int(px),
            py=int(py),
            time_left=int(state.time_left),
            platforms=moved_platforms,
            gate_timer=int(gate_timer),
            gate_phase=int(gate_phase),
            lever_on=int(lever_on),
        ), True

    # B7) Timebar.
    time_left = int(state.time_left) - 1
    if time_left <= 0:
        return None, False

    return (
        LevelRuntimeState(
            px=int(px),
            py=int(py),
            time_left=int(time_left),
            platforms=moved_platforms,
            gate_timer=int(gate_timer),
            gate_phase=int(gate_phase),
            lever_on=int(lever_on),
        ),
        False,
    )


def initial_level_state(model: LevelModel) -> LevelRuntimeState:
    return LevelRuntimeState(
        px=int(model.start[0]),
        py=int(model.start[1]),
        time_left=int(model.time_limit),
        platforms=tuple(model.platforms),
        gate_timer=0,
        gate_phase=GATE_CLOSED if model.gate_cells else GATE_OPEN,
        lever_on=0,
    )


def _level_specs() -> list[LevelModel]:
    levels: list[LevelModel] = []

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L1",
            time_limit=80,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#=========............====XX===#",
                "#====@====..[-].......====XX===#",
                "#=========............=========#",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=8, max_x=20)
            ],
        )
    )

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L2",
            time_limit=95,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#======........====......XX====#",
                "#==@====.[-]....====.[-]..XX===#",
                "#======........====......======#",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=8, max_x=15),
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=-1, min_x=19, max_x=26),
            ],
        )
    )

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L3",
            time_limit=110,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#######################...==XX=#",
                "#######################...==XX=#",
                "#######################...######",
                "#######################...######",
                "#######################[-]######",
                "#######################...######",
                "#######################...######",
                "##========............====######",
                "##==@======..[-].......=^^=#####",
                "##========............====######",
                "################################",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(mode="bounce_y", x=sx, y=sy, dir_sign=-1, min_y=6, max_y=12),
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=8, max_x=20),
            ],
        )
    )

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L4",
            time_limit=130,
            gate_duration=40,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#######################...|=XX=#",
                "#######################...|=XX=#",
                "#######################...######",
                "#######################...######",
                "#######################[-]######",
                "#######################...######",
                "#######################...######",
                "##========............====######",
                "##==@==o==..[-].......=^^=######",
                "##========............====######",
                "################################",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(mode="bounce_y", x=sx, y=sy, dir_sign=-1, min_y=6, max_y=12),
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=8, max_x=20),
            ],
        )
    )

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L5",
            time_limit=150,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#=XX=................==#...#===#",
                "#=XX=.......[-]......===..!#===#",
                "#====................===...#===#",
                "########################...#####",
                "########################...#####",
                "########################[-]#####",
                "########################...#####",
                "##========............====######",
                "##==@======..[-].......====#####",
                "##========............====######",
                "################################",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(
                    mode="route_bounce_x", x=sx, y=sy, dir_sign=1, route_id=0, route_ranges=((6, 16), (4, 20))
                ),
                lambda sx, sy: PlatformModel(mode="bounce_y", x=sx, y=sy, dir_sign=-1, min_y=7, max_y=13),
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=8, max_x=22),
            ],
        )
    )

    loop_points = _make_loop_points(5, 11, 10, 13)

    levels.append(
        _build_level_model(
            name="moving_platforms_manual_animation_L6",
            time_limit=180,
            gate_duration=80,
            rows=[
                "tttttttttttttttttttttttttttttttt",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#=XX|................==#...#===#",
                "#=XX|......[-].......===..!#===#",
                "#====................===...#===#",
                "########################...#####",
                "#====.[-].....=========#...#===#",
                "#====....o....=========#...#===#",
                "#====.........=========#...#===#",
                "#====.........==========...#===#",
                "#======............====#...#===#",
                "#==@====.[-]........=====...#==#",
                "#======............====#[-]#===#",
                "################################",
            ],
            platform_factories=[
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=5, max_x=18),
                lambda sx, sy: PlatformModel(mode="loop", x=sx, y=sy, loop_index=0, loop_points=loop_points),
                lambda sx, sy: PlatformModel(mode="bounce_x", x=sx, y=sy, dir_sign=1, min_x=7, max_x=16),
                lambda sx, sy: PlatformModel(mode="bounce_y", x=sx, y=sy, dir_sign=-1, min_y=7, max_y=16),
            ],
        )
    )

    return levels


LEVEL_MODELS = _level_specs()


def _level_to_arc_level(model: LevelModel) -> Level:
    return Level(
        name=model.name,
        grid_size=(WIDTH, HEIGHT),
        sprites=[
            Sprite(_solid(WIDTH, HEIGHT, C_VOID), name="canvas", x=0, y=0, layer=0, tags=["canvas"], collidable=False)
        ],
        data={"time_limit": model.time_limit},
    )


class MovingPlatformsManualAnimation(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_models = LEVEL_MODELS
        self._model: LevelModel | None = None
        self._runtime_state: LevelRuntimeState | None = None
        self._anim_phase = 0

        levels = [_level_to_arc_level(model) for model in self._level_models]
        camera = Camera(width=WIDTH, height=HEIGHT, background=C_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = 0
        for candidate_idx, candidate_level in enumerate(getattr(self, "_levels", [])):
            if candidate_level is level:
                idx = candidate_idx
                break
        self._model = self._level_models[idx]
        self._runtime_state = initial_level_state(self._model)
        self._anim_phase = 0
        self._render()

    def _canvas(self) -> Sprite:
        sprites = self.current_level.get_sprites_by_name("canvas")
        if not sprites:
            raise RuntimeError("canvas sprite missing")
        return sprites[0]

    def _parse_action_token(self) -> ActionToken:
        aid = int(getattr(self.action.id, "value", self.action.id))
        if aid != CLICK_ACTION:
            return ActionToken(action_id=aid)

        payload = self.action.data if isinstance(self.action.data, dict) else {}
        try:
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
        except (TypeError, ValueError):
            x, y = -1, -1
        return ActionToken(action_id=aid, click_x=int(x), click_y=int(y))

    def _lose_level(self) -> None:
        self.lose()

    def _platform_leading_idx(self, p: PlatformModel) -> int:
        if p.last_dx > 0:
            return 2
        if p.last_dx < 0:
            return 0
        return 1

    def _render(self) -> None:
        model = self._model
        state = self._runtime_state
        if model is None or state is None:
            return

        frame = np.full((HEIGHT, WIDTH), C_VOID, dtype=np.int8)

        for y in range(1, HEIGHT):
            for x in range(WIDTH):
                tile = int(model.terrain[y][x])
                if tile == T_WALL:
                    frame[y, x] = C_WALL
                elif tile == T_GROUND:
                    frame[y, x] = C_GROUND
                elif tile == T_VOID:
                    frame[y, x] = C_VOID
                elif tile == T_SPIKE:
                    frame[y, x] = C_SPIKE_A if (self._anim_phase % 2 == 0) else C_SPIKE_B
                elif tile == T_GATE:
                    phase = int(state.gate_phase)
                    if phase == GATE_OPEN:
                        frame[y, x] = C_GROUND
                    elif phase in {GATE_OPENING, GATE_CLOSING}:
                        frame[y, x] = C_GATE_ANIM
                    else:
                        frame[y, x] = C_GATE_CLOSED

        for x, y in model.exits:
            frame[y, x] = C_EXIT_A if (self._anim_phase % 2 == 0) else C_EXIT_B

        button_set = set(model.buttons)
        lever_set = set(model.levers)
        for x, y in button_set:
            active = (state.px, state.py) == (x, y)
            frame[y, x] = C_BUTTON_ON if active else C_BUTTON_OFF

        for x, y in lever_set:
            frame[y, x] = C_BUTTON_ON if state.lever_on else C_BUTTON_OFF

        for p in state.platforms:
            cells = _platform_cells(p)
            lead = self._platform_leading_idx(p)
            for idx, (x, y) in enumerate(cells):
                if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
                    continue
                color = C_PLATFORM
                if idx == lead or (self._anim_phase % 2 == 1 and idx == 1):
                    color = C_PLATFORM_EDGE
                frame[y, x] = color

        frame[state.py, state.px] = C_PLAYER_A if (self._anim_phase % 2 == 0) else C_PLAYER_B

        fill = max(0, min(WIDTH, round((state.time_left / max(1, model.time_limit)) * WIDTH)))
        frame[0, :fill] = C_TIME_FILL
        frame[0, fill:] = C_VOID

        self._canvas().pixels = frame

    def step(self) -> None:
        model = self._model
        state = self._runtime_state
        if model is None or state is None:
            self.complete_action()
            return

        token = self._parse_action_token()
        if token.action_id not in {1, 2, 3, 4, 5, 6}:
            self.complete_action()
            return
        next_state, won = transition_level_state(model, state, token, camera=self.camera)

        if next_state is None:
            self._lose_level()
            self.complete_action()
            return

        self._runtime_state = next_state
        self._anim_phase += 1
        self._render()

        if won:
            self.next_level()
            self.complete_action()
            return

        self.complete_action()


__all__ = [
    "GAME_ID",
    "LEVEL_MODELS",
    "ActionToken",
    "LevelModel",
    "LevelRuntimeState",
    "MovingPlatformsManualAnimation",
    "initial_level_state",
    "transition_level_state",
]
