from __future__ import annotations

from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "deliver_the_package-0001"
GRID_W = 24
GRID_H = 16
WORLD_Y_MIN = 1

C_EMPTY = 0
C_FLOOR = 1
C_WALL = 2
C_PLAYER_BODY = 3
C_PLAYER_FACE = 4
C_PACKAGE = 5
C_HIGHLIGHT = 6
C_BAY_A = 7
C_BAY_B = 8
C_HAZARD_ON = 9
C_HAZARD_OFF = 10
C_DOOR_CLOSED = 11
C_DOOR_TRANSITION = 12
C_PLATE_OFF = 13
C_PLATE_ON = 14
C_BAY_C = 15

TARGET_BAY_COLOR = {"+": C_BAY_A, "&": C_BAY_B, "$": C_BAY_C}

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

FACE_OFFSETS = {"U": (0, 0), "R": (1, 0), "D": (1, 1), "L": (0, 1)}

MARKER_TO_FACE = {"^": "U", ">": "R", "v": "D", "<": "L"}
FACE_TO_MARKER = {v: k for k, v in MARKER_TO_FACE.items()}

DOOR_CLOSED = 0
DOOR_OPENING = 1
DOOR_OPEN = 2
DOOR_CLOSING = 3


class DroneRoute(NamedTuple):
    y: int
    x_min: int
    x_max: int


class LevelSpec(NamedTuple):
    name: str
    time_max_steps: int
    target_bay: str
    rows: tuple[str, ...]
    wrong_bays: tuple[str, ...] = ()
    drone_route: DroneRoute | None = None


class CompiledLevel(NamedTuple):
    name: str
    time_max_steps: int
    target_bay: str
    wrong_bays: tuple[str, ...]
    walls: frozenset[tuple[int, int]]
    conveyors: frozenset[tuple[int, int]]
    laser_cells: frozenset[tuple[int, int]]
    plate_cells: tuple[tuple[int, int], ...]
    door_cells: tuple[tuple[int, int], ...]
    bay_cells: dict[str, frozenset[tuple[int, int]]]
    player_anchor: tuple[int, int]
    player_facing: str
    package_pos: tuple[int, int]
    drone_route: DroneRoute | None
    drone_start_x: int | None
    drone_dir: int


class SimState(NamedTuple):
    player_x: int
    player_y: int
    facing: str
    carrying: bool
    package_x: int
    package_y: int
    door_state: int
    drone_x: int
    drone_dir: int
    time_remaining: int
    anim_phase: int


class StepResult(NamedTuple):
    state: SimState
    won: bool
    failed: bool
    fail_reason: str | None


def _rows(*values: str) -> tuple[str, ...]:
    if len(values) != GRID_H:
        raise ValueError(f"Each level needs {GRID_H} rows.")
    for row in values:
        if len(row) != GRID_W:
            raise ValueError(f"Row length must be {GRID_W}; got {len(row)} for {row!r}.")
    return tuple(values)


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1 - Pickup and Deliver",
        time_max_steps=90,
        target_bay="+",
        rows=_rows(
            "========================",
            "########################",
            "#......................#",
            "#..@>..................#",
            "#..@@..................#",
            "#......o...............#",
            "#......................#",
            "#..................++..#",
            "#..................++..#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 2 - Carrying Around Walls",
        time_max_steps=120,
        target_bay="+",
        rows=_rows(
            "========================",
            "########################",
            "#......................#",
            "#..@>..................#",
            "#..@@......##########..#",
            "#..........#........#..#",
            "#....o.....#..++....#..#",
            "#..........#..++....#..#",
            "#..........#........#..#",
            "#..........####..####..#",
            "#......................#",
            "#..##########..........#",
            "#..#........#..........#",
            "#..#........#..........#",
            "#..##########..........#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 3 - Color Match",
        time_max_steps=150,
        target_bay="&",
        wrong_bays=("+", "$"),
        rows=_rows(
            "========================",
            "########################",
            "#..@>..................#",
            "#..@@..................#",
            "#..........##..##......#",
            "#.....o....##..##..++..#",
            "#..........##..##..++..#",
            "#..........##..##......#",
            "#..........##..##..&&..#",
            "#..........##..##..&&..#",
            "#..........##..##..$$..#",
            "#..........##..##..$$..#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 4 - Plate and Door",
        time_max_steps=170,
        target_bay="+",
        rows=_rows(
            "========================",
            "########################",
            "#..@>......##..........#",
            "#..@@......##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#.....o....##..........#",
            "#........~~||....++....#",
            "#........~~||....++....#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 5 - Drone Timing",
        time_max_steps=180,
        target_bay="+",
        drone_route=DroneRoute(y=8, x_min=1, x_max=21),
        rows=_rows(
            "========================",
            "########################",
            "#..@>....######........#",
            "#..@@....######........#",
            "#........######..++....#",
            "#........######..++....#",
            "#...o....######........#",
            "#........######........#",
            "#..........}{..........#",
            "#..........][..........#",
            "#........######........#",
            "#........######........#",
            "#........######........#",
            "#........######........#",
            "#........######........#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 6 - Combined",
        time_max_steps=220,
        target_bay="&",
        wrong_bays=("+",),
        drone_route=DroneRoute(y=9, x_min=19, x_max=20),
        rows=_rows(
            "========================",
            "########################",
            "#..@>..................#",
            "#..@@.....#####........#",
            "#.....o...#####........#",
            "#.........#####........#",
            "#.........##~~##.......#",
            "#.........##~~##...++..#",
            "#....>>>>>..;;..||.++..#",
            "#....>>>>>..;;..||.}{..#",
            "#..............##..][..#",
            "#..............##..&&..#",
            "#..............##..&&..#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
)


def _validate_spec(spec: LevelSpec) -> CompiledLevel:
    rows = spec.rows

    marker_cells: list[tuple[int, int, str]] = []
    body_cells: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    conveyors: set[tuple[int, int]] = set()
    laser_cells: set[tuple[int, int]] = set()
    plate_cells: list[tuple[int, int]] = []
    door_cells: list[tuple[int, int]] = []
    bay_cells = {"+": set(), "&": set(), "$": set()}
    package_pos: tuple[int, int] | None = None
    drone_anchor: tuple[int, int] | None = None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "@":
                body_cells.add((x, y))
            elif ch in MARKER_TO_FACE:
                marker_cells.append((x, y, ch))
            elif ch == "o":
                if package_pos is not None:
                    raise ValueError(f"{spec.name}: exactly one package is required.")
                package_pos = (x, y)
            elif ch in {"+", "&", "$"}:
                bay_cells[ch].add((x, y))
            elif ch == "~":
                plate_cells.append((x, y))
            elif ch == "|":
                door_cells.append((x, y))
            elif ch == ";":
                laser_cells.add((x, y))
            elif ch == ">":
                conveyors.add((x, y))

    player_marker: tuple[int, int, str] | None = None
    player_anchor: tuple[int, int] | None = None
    for mx, my, marker in marker_cells:
        facing = MARKER_TO_FACE[marker]
        fx, fy = FACE_OFFSETS[facing]
        anchor = (mx - fx, my - fy)
        player_cells = {
            (anchor[0], anchor[1]),
            (anchor[0] + 1, anchor[1]),
            (anchor[0], anchor[1] + 1),
            (anchor[0] + 1, anchor[1] + 1),
        }
        if player_cells.issubset(body_cells | {(mx, my)}):
            if player_marker is not None:
                raise ValueError(f"{spec.name}: multiple valid player markers found.")
            player_marker = (mx, my, marker)
            player_anchor = anchor

    if player_marker is None or player_anchor is None:
        raise ValueError(f"{spec.name}: could not infer a valid 2x2 player sprite.")

    mx, my, marker = player_marker
    facing = MARKER_TO_FACE[marker]

    if package_pos is None:
        raise ValueError(f"{spec.name}: package spawn is required.")

    if player_anchor[1] < WORLD_Y_MIN:
        raise ValueError(f"{spec.name}: player must not overlap row 0.")

    if not bay_cells[spec.target_bay]:
        raise ValueError(f"{spec.name}: target bay {spec.target_bay!r} has no cells.")

    for bay in spec.wrong_bays:
        if bay == spec.target_bay:
            raise ValueError(f"{spec.name}: wrong bay cannot equal target bay.")
        if not bay_cells[bay]:
            raise ValueError(f"{spec.name}: wrong bay {bay!r} has no cells.")

    if door_cells and not plate_cells:
        raise ValueError(f"{spec.name}: door exists but no pressure plate exists.")

    if plate_cells and not door_cells:
        raise ValueError(f"{spec.name}: pressure plate exists but no door exists.")

    if door_cells and len(door_cells) != 4:
        raise ValueError(f"{spec.name}: expected one 2x2 door.")
    if plate_cells and len(plate_cells) != 4:
        raise ValueError(f"{spec.name}: expected one 2x2 pressure plate.")

    if spec.drone_route is not None:
        found = False
        for y in range(GRID_H - 1):
            for x in range(GRID_W - 1):
                if rows[y][x] == "}" and rows[y][x + 1] == "{" and rows[y + 1][x] == "]" and rows[y + 1][x + 1] == "[":
                    drone_anchor = (x, y)
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"{spec.name}: drone route configured but sprite not found in map.")

    return CompiledLevel(
        name=spec.name,
        time_max_steps=int(spec.time_max_steps),
        target_bay=spec.target_bay,
        wrong_bays=tuple(spec.wrong_bays),
        walls=frozenset(walls),
        conveyors=frozenset(conveyors),
        laser_cells=frozenset(laser_cells),
        plate_cells=tuple(sorted(plate_cells)),
        door_cells=tuple(sorted(door_cells)),
        bay_cells={k: frozenset(v) for k, v in bay_cells.items()},
        player_anchor=player_anchor,
        player_facing=facing,
        package_pos=package_pos,
        drone_route=spec.drone_route,
        drone_start_x=(drone_anchor[0] if drone_anchor is not None else None),
        drone_dir=1,
    )


COMPILED_LEVELS: tuple[CompiledLevel, ...] = tuple(_validate_spec(spec) for spec in LEVEL_SPECS)


def initial_sim_state(level_index: int) -> SimState:
    level = COMPILED_LEVELS[level_index]
    return SimState(
        player_x=level.player_anchor[0],
        player_y=level.player_anchor[1],
        facing=level.player_facing,
        carrying=False,
        package_x=level.package_pos[0],
        package_y=level.package_pos[1],
        door_state=DOOR_CLOSED if level.door_cells else DOOR_OPEN,
        drone_x=(level.drone_start_x if level.drone_start_x is not None else -1),
        drone_dir=int(level.drone_dir),
        time_remaining=int(level.time_max_steps),
        anim_phase=0,
    )


def _player_cells(state: SimState) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    x = state.player_x
    y = state.player_y
    return ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))


def _marker_cell(state: SimState) -> tuple[int, int]:
    ox, oy = FACE_OFFSETS[state.facing]
    return state.player_x + ox, state.player_y + oy


def front_cell(state: SimState) -> tuple[int, int]:
    mx, my = _marker_cell(state)
    if state.facing == "U":
        return mx, my - 1
    if state.facing == "R":
        return mx + 1, my
    if state.facing == "D":
        return mx, my + 1
    return mx - 1, my


def carried_package_cell(state: SimState) -> tuple[int, int] | None:
    if not state.carrying:
        return None
    return front_cell(state)


def _in_world(x: int, y: int) -> bool:
    return 0 <= x < GRID_W and WORLD_Y_MIN <= y < GRID_H


def _is_solid(level: CompiledLevel, state: SimState, x: int, y: int) -> bool:
    if not _in_world(x, y):
        return True
    if (x, y) in level.walls:
        return True
    return bool(state.door_state == DOOR_CLOSED and (x, y) in level.door_cells)


def _plate_pressed(level: CompiledLevel, state: SimState) -> bool:
    if not level.plate_cells:
        return False
    plate_set = set(level.plate_cells)
    for cell in _player_cells(state):
        if cell in plate_set:
            return True
    if state.carrying:
        carried = carried_package_cell(state)
        if carried is not None and carried in plate_set:
            return True
    else:
        if (state.package_x, state.package_y) in plate_set:
            return True
    return False


def _advance_door(level: CompiledLevel, state: SimState) -> int:
    if not level.door_cells:
        return DOOR_OPEN
    pressed = _plate_pressed(level, state)
    if pressed:
        if state.door_state == DOOR_CLOSED:
            return DOOR_OPENING
        if state.door_state in {DOOR_OPENING, DOOR_CLOSING}:
            return DOOR_OPEN
        return DOOR_OPEN

    if state.door_state == DOOR_OPEN:
        return DOOR_CLOSING
    if state.door_state == DOOR_CLOSING:
        return DOOR_CLOSED
    if state.door_state == DOOR_OPENING:
        return DOOR_CLOSED
    return DOOR_CLOSED


def _attempt_move(level: CompiledLevel, state: SimState, dx: int, dy: int, *, update_facing: bool) -> SimState:
    nx = state.player_x + dx
    ny = state.player_y + dy

    for px, py in ((nx, ny), (nx + 1, ny), (nx, ny + 1), (nx + 1, ny + 1)):
        if _is_solid(level, state, px, py):
            return state

    facing = state.facing
    if update_facing:
        facing = {(0, -1): "U", (1, 0): "R", (0, 1): "D", (-1, 0): "L"}[(dx, dy)]

    moved = SimState(
        player_x=nx,
        player_y=ny,
        facing=facing,
        carrying=state.carrying,
        package_x=state.package_x,
        package_y=state.package_y,
        door_state=state.door_state,
        drone_x=state.drone_x,
        drone_dir=state.drone_dir,
        time_remaining=state.time_remaining,
        anim_phase=state.anim_phase,
    )

    if moved.carrying:
        cx, cy = front_cell(moved)
        if _is_solid(level, moved, cx, cy):
            return state

    return moved


def _drop_or_pick(level: CompiledLevel, state: SimState) -> SimState:
    fx, fy = front_cell(state)
    if not _in_world(fx, fy):
        return state

    if not state.carrying:
        if (state.package_x, state.package_y) == (fx, fy):
            return SimState(
                player_x=state.player_x,
                player_y=state.player_y,
                facing=state.facing,
                carrying=True,
                package_x=state.package_x,
                package_y=state.package_y,
                door_state=state.door_state,
                drone_x=state.drone_x,
                drone_dir=state.drone_dir,
                time_remaining=state.time_remaining,
                anim_phase=state.anim_phase,
            )
        return state

    if _is_solid(level, state, fx, fy):
        return state

    return SimState(
        player_x=state.player_x,
        player_y=state.player_y,
        facing=state.facing,
        carrying=False,
        package_x=fx,
        package_y=fy,
        door_state=state.door_state,
        drone_x=state.drone_x,
        drone_dir=state.drone_dir,
        time_remaining=state.time_remaining,
        anim_phase=state.anim_phase,
    )


def _apply_conveyor(level: CompiledLevel, state: SimState) -> SimState:
    if not level.conveyors:
        return state
    out = state
    for _ in range(GRID_W):
        if not any(cell in level.conveyors for cell in _player_cells(out)):
            break
        moved = _attempt_move(level, out, 1, 0, update_facing=False)
        if moved == out:
            break
        out = moved
    return out


def _advance_drone(level: CompiledLevel, state: SimState) -> tuple[int, int]:
    if level.drone_route is None:
        return -1, 0

    x = state.drone_x
    d = state.drone_dir
    if x < level.drone_route.x_min:
        x = level.drone_route.x_min
    if x > level.drone_route.x_max:
        x = level.drone_route.x_max
    nx = x + d
    nd = d
    if nx < level.drone_route.x_min or nx > level.drone_route.x_max:
        nd = -d
        nx = x + nd
    nx = max(level.drone_route.x_min, min(level.drone_route.x_max, nx))
    return nx, nd


def _laser_is_on(state: SimState) -> bool:
    return int(state.anim_phase) == 0


def _drone_cells(level: CompiledLevel, state: SimState) -> set[tuple[int, int]]:
    if level.drone_route is None:
        return set()
    x = state.drone_x
    y = level.drone_route.y
    return {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}


def package_cell(_level: CompiledLevel, state: SimState) -> tuple[int, int]:
    if state.carrying:
        carried = carried_package_cell(state)
        if carried is None:
            return state.package_x, state.package_y
        return carried
    return state.package_x, state.package_y


def simulate_step(level: CompiledLevel, state: SimState, action_id: int) -> StepResult:
    out = state

    if action_id in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[action_id]
        out = _attempt_move(level, out, dx, dy, update_facing=True)
    elif action_id == int(GameAction.ACTION5.value):
        out = _drop_or_pick(level, out)

    out = _apply_conveyor(level, out)
    out = SimState(
        player_x=out.player_x,
        player_y=out.player_y,
        facing=out.facing,
        carrying=out.carrying,
        package_x=out.package_x,
        package_y=out.package_y,
        door_state=_advance_door(level, out),
        drone_x=out.drone_x,
        drone_dir=out.drone_dir,
        time_remaining=out.time_remaining,
        anim_phase=out.anim_phase,
    )

    nx, nd = _advance_drone(level, out)
    out = SimState(
        player_x=out.player_x,
        player_y=out.player_y,
        facing=out.facing,
        carrying=out.carrying,
        package_x=out.package_x,
        package_y=out.package_y,
        door_state=out.door_state,
        drone_x=nx,
        drone_dir=nd,
        time_remaining=max(0, out.time_remaining - 1),
        anim_phase=1 - int(out.anim_phase),
    )

    pkg = package_cell(level, out)

    for wrong in level.wrong_bays:
        if pkg in level.bay_cells[wrong]:
            return StepResult(state=out, won=False, failed=True, fail_reason="wrong_bay")

    if _laser_is_on(out):
        for cell in _player_cells(out):
            if cell in level.laser_cells:
                return StepResult(state=out, won=False, failed=True, fail_reason="laser")
        carried = carried_package_cell(out)
        if carried is not None and carried in level.laser_cells:
            return StepResult(state=out, won=False, failed=True, fail_reason="laser")

    drone_cells = _drone_cells(level, out)
    if drone_cells:
        for cell in _player_cells(out):
            if cell in drone_cells:
                return StepResult(state=out, won=False, failed=True, fail_reason="drone")
        carried = carried_package_cell(out)
        if carried is not None and carried in drone_cells:
            return StepResult(state=out, won=False, failed=True, fail_reason="drone")

    if out.time_remaining <= 0:
        return StepResult(state=out, won=False, failed=True, fail_reason="time")

    if pkg in level.bay_cells[level.target_bay]:
        return StepResult(state=out, won=True, failed=False, fail_reason=None)

    return StepResult(state=out, won=False, failed=False, fail_reason=None)


def _build_level_sprite() -> Level:
    canvas = Sprite(
        name="canvas",
        pixels=np.zeros((GRID_H, GRID_W), dtype=np.int8),
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["frame"],
    )
    return Level(grid_size=(GRID_W, GRID_H), sprites=[canvas], data={})


class DeliverThePackage(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level_sprite() for _ in COMPILED_LEVELS]
        camera = Camera(width=GRID_W, height=GRID_H, background=C_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._level_index = 0
        self._sim_state = initial_sim_state(0)
        self._fail_freeze_steps = 0
        self._fail_reason: str | None = None

    def on_set_level(self, _level: Level) -> None:
        self._level_index = int(getattr(self, "current_level_index", self._score) or 0)
        self._level_index = max(0, min(self._level_index, len(COMPILED_LEVELS) - 1))
        self._sim_state = initial_sim_state(self._level_index)
        self._fail_freeze_steps = 0
        self._fail_reason = None
        self._redraw(force_hazard_flash=False)

    def _current(self) -> CompiledLevel:
        return COMPILED_LEVELS[self._level_index]

    def _canvas(self) -> Sprite:
        sprites = self.current_level.get_sprites_by_name("canvas")
        if not sprites:
            raise RuntimeError("deliver_the_package missing canvas sprite")
        return sprites[0]

    def _fill_background(self, grid: np.ndarray, level: CompiledLevel) -> None:
        grid[:, :] = C_FLOOR
        grid[0, :] = C_EMPTY
        for x, y in level.walls:
            grid[y, x] = C_WALL

        for x, y in level.plate_cells:
            if _plate_pressed(level, self._sim_state):
                grid[y, x] = C_PLATE_ON
            else:
                grid[y, x] = C_PLATE_OFF

        TARGET_BAY_COLOR[level.target_bay]
        for bay_type, cells in level.bay_cells.items():
            base_color = TARGET_BAY_COLOR[bay_type]
            for x, y in cells:
                if bay_type == level.target_bay and self._sim_state.anim_phase == 1:
                    grid[y, x] = C_HIGHLIGHT
                else:
                    grid[y, x] = base_color

        if level.door_cells:
            if self._sim_state.door_state == DOOR_CLOSED:
                door_color = C_DOOR_CLOSED
            elif self._sim_state.door_state in {DOOR_OPENING, DOOR_CLOSING}:
                door_color = C_DOOR_TRANSITION
            else:
                door_color = C_FLOOR
            for x, y in level.door_cells:
                grid[y, x] = door_color

        for x, y in level.conveyors:
            grid[y, x] = C_BAY_C

        if level.laser_cells:
            laser_color = C_HAZARD_ON if _laser_is_on(self._sim_state) else C_HAZARD_OFF
            for x, y in level.laser_cells:
                grid[y, x] = laser_color

        if level.drone_route is not None:
            drone_color = C_HAZARD_ON if self._sim_state.anim_phase == 0 else C_HAZARD_OFF
            for x, y in _drone_cells(level, self._sim_state):
                if _in_world(x, y):
                    grid[y, x] = drone_color

        pkg_color = C_PACKAGE
        if self._sim_state.anim_phase == 1:
            pkg_color = TARGET_BAY_COLOR[level.target_bay]

        px, py = package_cell(level, self._sim_state)
        if _in_world(px, py):
            grid[py, px] = pkg_color

        for x, y in _player_cells(self._sim_state):
            if _in_world(x, y):
                grid[y, x] = C_PLAYER_BODY

        mx, my = _marker_cell(self._sim_state)
        if _in_world(mx, my):
            grid[my, mx] = C_PLAYER_FACE

        filled = int((GRID_W * self._sim_state.time_remaining) // max(1, level.time_max_steps))
        filled = max(0, min(GRID_W, filled))
        if filled > 0:
            grid[0, :filled] = C_HIGHLIGHT
        if filled < GRID_W:
            grid[0, filled:] = C_EMPTY

    def _redraw(self, *, force_hazard_flash: bool) -> None:
        level = self._current()
        grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)
        self._fill_background(grid, level)

        if force_hazard_flash:
            for x, y in level.laser_cells:
                grid[y, x] = C_HAZARD_ON
            for x, y in _drone_cells(level, self._sim_state):
                if _in_world(x, y):
                    grid[y, x] = C_HAZARD_ON

        self._canvas().pixels = grid

    def _trigger_fail(self, reason: str) -> None:
        if self._fail_freeze_steps > 0:
            return
        self._fail_reason = reason
        self._fail_freeze_steps = 6

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

        if self._fail_freeze_steps > 0:
            self._sim_state = SimState(
                player_x=self._sim_state.player_x,
                player_y=self._sim_state.player_y,
                facing=self._sim_state.facing,
                carrying=self._sim_state.carrying,
                package_x=self._sim_state.package_x,
                package_y=self._sim_state.package_y,
                door_state=self._sim_state.door_state,
                drone_x=self._sim_state.drone_x,
                drone_dir=self._sim_state.drone_dir,
                time_remaining=self._sim_state.time_remaining,
                anim_phase=1 - int(self._sim_state.anim_phase),
            )
            self._fail_freeze_steps -= 1
            self._redraw(force_hazard_flash=True)
            if self._fail_freeze_steps <= 0:
                self.lose()
            self.complete_action()
            return

        action_raw = self.action.id
        action_id = int(getattr(action_raw, "value", action_raw))
        result = simulate_step(self._current(), self._sim_state, action_id)
        self._sim_state = result.state

        if result.won:
            self._redraw(force_hazard_flash=False)
            self.next_level()
            self.complete_action()
            return

        if result.failed:
            self._trigger_fail(result.fail_reason or "fail")
            self._redraw(force_hazard_flash=True)
            self.complete_action()
            return

        self._redraw(force_hazard_flash=False)
        self.complete_action()


__all__ = [
    "COMPILED_LEVELS",
    "GAME_ID",
    "DeliverThePackage",
    "SimState",
    "StepResult",
    "initial_sim_state",
    "package_cell",
    "simulate_step",
]
