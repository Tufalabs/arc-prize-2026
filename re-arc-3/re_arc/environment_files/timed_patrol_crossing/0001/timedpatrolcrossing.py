from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "timed_patrol_crossing-0001"

GRID_W = 25
GRID_H = 17

COLOR_VOID = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_GRASS = 3
COLOR_TIME_FILLED = 4
COLOR_TIME_EMPTY = 5
COLOR_PLAYER_VISIBLE = 6
COLOR_PLAYER_HIDDEN = 7
COLOR_GOAL = 8
COLOR_DOOR = 9
COLOR_LEVER = 10
COLOR_GUARD_A = 11
COLOR_GUARD_B = 12
COLOR_VISION = 13
COLOR_DANGER = 14

MOVE_DELTAS: dict[int, tuple[int, int]] = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION = int(GameAction.ACTION5.value)
CLICK_ACTION = int(GameAction.ACTION6.value)

DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3

DIR_TO_DELTA = {DIR_UP: (0, -1), DIR_DOWN: (0, 1), DIR_LEFT: (-1, 0), DIR_RIGHT: (1, 0)}
REVERSE_DIR = {DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP, DIR_LEFT: DIR_RIGHT, DIR_RIGHT: DIR_LEFT}

MODE_NORMAL = "NORMAL"
MODE_WIN_FLASH = "WIN_FLASH"


@dataclass(frozen=True)
class GuardSpec:
    start_x: int
    start_y: int
    start_dir: int
    patrol_min: int
    patrol_max: int
    horizontal: bool


@dataclass(frozen=True)
class DoorSpec:
    x: int
    y: int
    starts_closed: bool = True


@dataclass(frozen=True)
class LeverSpec:
    x: int
    y: int
    linked_doors: tuple[int, ...]
    starts_on: bool = False


@dataclass(frozen=True)
class LevelSpec:
    name: str
    layout: tuple[str, ...]
    player_start: tuple[int, int]
    goal_top_left: tuple[int, int]
    guards: tuple[GuardSpec, ...]
    doors: tuple[DoorSpec, ...]
    levers: tuple[LeverSpec, ...]
    controls_click: bool
    vision_on: bool
    vision_range: int
    time_per_segment: int

    @property
    def max_segments(self) -> int:
        return GRID_W - 2

    @property
    def max_time(self) -> int:
        return self.max_segments * int(self.time_per_segment)


@dataclass(frozen=True)
class SearchModel:
    level_idx: int
    walkable: frozenset[tuple[int, int]]
    grass: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]
    goal_tiles: frozenset[tuple[int, int]]
    guards: tuple[GuardSpec, ...]
    doors: tuple[DoorSpec, ...]
    levers: tuple[LeverSpec, ...]
    controls_click: bool
    vision_on: bool
    vision_range: int


@dataclass(frozen=True)
class SearchState:
    player_x: int
    player_y: int
    time_remaining: int
    door_mask: int
    lever_mask: int
    guards: tuple[tuple[int, int, int], ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1 — Bump timing",
        layout=(
            "#=======================#",
            "#########################",
            "#.................$$....#",
            "#.................$$....#",
            "#.......................#",
            "#.......................#",
            "###########..############",
            "#......&&...............#",
            "#......&&...............#",
            "###########..############",
            "#.......................#",
            "#.......................#",
            "#.......................#",
            "#..@....................#",
            "#.......................#",
            "#.......................#",
            "#########################",
        ),
        player_start=(3, 13),
        goal_top_left=(18, 2),
        guards=(GuardSpec(start_x=7, start_y=7, start_dir=DIR_RIGHT, patrol_min=2, patrol_max=20, horizontal=True),),
        doors=(),
        levers=(),
        controls_click=False,
        vision_on=False,
        vision_range=7,
        time_per_segment=3,
    ),
    LevelSpec(
        name="Level 2 — Sightline",
        layout=(
            "#=======================#",
            "#########################",
            "#.................$$....#",
            "#.................$$....#",
            "#.......................#",
            "#.......................#",
            "#######..################",
            "#......:::::::&&........#",
            "#......:::::::&&........#",
            "#######..################",
            "#.......................#",
            "#.......................#",
            "#.......................#",
            "#..@....................#",
            "#.......................#",
            "#.......................#",
            "#########################",
        ),
        player_start=(3, 13),
        goal_top_left=(18, 2),
        guards=(GuardSpec(start_x=14, start_y=7, start_dir=DIR_LEFT, patrol_min=2, patrol_max=20, horizontal=True),),
        doors=(),
        levers=(),
        controls_click=False,
        vision_on=True,
        vision_range=7,
        time_per_segment=3,
    ),
    LevelSpec(
        name="Level 3 — Hide in grass",
        layout=(
            "#=======================#",
            "#########################",
            "###########$$############",
            "###########$$############",
            "###########..############",
            "###########..############",
            "####.&&:::::,,.......####",
            "####.&&:::::,,.......####",
            "###########..############",
            "###########..############",
            "####.......,,:::::&&.####",
            "####.......,,:::::&&.####",
            "###########..############",
            "###########..############",
            "###########@.############",
            "###########..############",
            "#########################",
        ),
        player_start=(11, 14),
        goal_top_left=(11, 2),
        guards=(
            GuardSpec(start_x=5, start_y=6, start_dir=DIR_RIGHT, patrol_min=4, patrol_max=16, horizontal=True),
            GuardSpec(start_x=18, start_y=10, start_dir=DIR_LEFT, patrol_min=6, patrol_max=18, horizontal=True),
        ),
        doors=(),
        levers=(),
        controls_click=False,
        vision_on=True,
        vision_range=7,
        time_per_segment=3,
    ),
    LevelSpec(
        name="Level 4 — Door control",
        layout=(
            "#=======================#",
            "#########################",
            "#..........$$...........#",
            "#..........$$...........#",
            "#.......................#",
            "###########,,############",
            "###########,,############",
            "###########,,############",
            "###########,,############",
            "###########,,############",
            "###########,,############",
            "###########,,,,##########",
            "###########,,,,##########",
            "#....!..................#",
            "#.......................#",
            "#..@....................#",
            "#########################",
        ),
        player_start=(3, 15),
        goal_top_left=(11, 2),
        guards=(GuardSpec(start_x=11, start_y=6, start_dir=DIR_DOWN, patrol_min=5, patrol_max=12, horizontal=False),),
        doors=(DoorSpec(x=11, y=9, starts_closed=True),),
        levers=(LeverSpec(x=5, y=13, linked_doors=(0,), starts_on=False),),
        controls_click=True,
        vision_on=True,
        vision_range=7,
        time_per_segment=3,
    ),
    LevelSpec(
        name="Level 5 — Two gates",
        layout=(
            "#=======================#",
            "#########################",
            "#.................$$....#",
            "#.................$$....#",
            "#........&&::::.........#",
            "#........&&::::.........#",
            "###########||############",
            "###########||############",
            "#..&&::::,,,............#",
            "#..&&::::,,,..!.........#",
            "#...........,,,.........#",
            "#####||##################",
            "#####||##################",
            "#.................!.....#",
            "#.......................#",
            "#..@....................#",
            "#########################",
        ),
        player_start=(3, 15),
        goal_top_left=(18, 2),
        guards=(
            GuardSpec(start_x=3, start_y=8, start_dir=DIR_RIGHT, patrol_min=2, patrol_max=10, horizontal=True),
            GuardSpec(start_x=9, start_y=4, start_dir=DIR_RIGHT, patrol_min=2, patrol_max=18, horizontal=True),
        ),
        doors=(DoorSpec(x=5, y=11, starts_closed=True), DoorSpec(x=11, y=6, starts_closed=True)),
        levers=(
            LeverSpec(x=18, y=13, linked_doors=(0,), starts_on=False),
            LeverSpec(x=14, y=9, linked_doors=(1,), starts_on=False),
        ),
        controls_click=True,
        vision_on=True,
        vision_range=7,
        time_per_segment=3,
    ),
    LevelSpec(
        name="Level 6 — Full crossing",
        layout=(
            "#=======================#",
            "#########################",
            "#.................$$....#",
            "#.................$$....#",
            "#.....&&:::::...........#",
            "#.....&&:::::...........#",
            "#####||##################",
            "#####||##################",
            "#..,,!......&&:::::.....#",
            "#..,,.&&::::&&:::::.....#",
            "#.....&&::::............#",
            "#####..##########||######",
            "#####..##########||######",
            "#.........&&::,,!.......#",
            "#.........&&::,,........#",
            "#..@....................#",
            "#########################",
        ),
        player_start=(3, 15),
        goal_top_left=(18, 2),
        guards=(
            GuardSpec(start_x=6, start_y=4, start_dir=DIR_RIGHT, patrol_min=2, patrol_max=10, horizontal=True),
            GuardSpec(start_x=12, start_y=8, start_dir=DIR_RIGHT, patrol_min=10, patrol_max=18, horizontal=True),
            GuardSpec(start_x=6, start_y=9, start_dir=DIR_RIGHT, patrol_min=5, patrol_max=10, horizontal=True),
            GuardSpec(start_x=10, start_y=13, start_dir=DIR_RIGHT, patrol_min=8, patrol_max=12, horizontal=True),
        ),
        doors=(DoorSpec(x=17, y=11, starts_closed=True), DoorSpec(x=5, y=6, starts_closed=True)),
        levers=(
            LeverSpec(x=16, y=13, linked_doors=(0,), starts_on=False),
            LeverSpec(x=5, y=8, linked_doors=(1,), starts_on=False),
        ),
        controls_click=True,
        vision_on=True,
        vision_range=7,
        time_per_segment=2,
    ),
)


def _rect_tiles(x: int, y: int, w: int, h: int) -> tuple[tuple[int, int], ...]:
    return tuple((x + dx, y + dy) for dy in range(h) for dx in range(w))


def _build_level(index: int, spec: LevelSpec) -> Level:
    board = np.zeros((GRID_H, GRID_W), dtype=np.int8)
    sprite = Sprite(pixels=board, name="board", x=0, y=0, layer=0, tags=["board", "sys_static"], collidable=False)
    return Level(name=spec.name, grid_size=(GRID_W, GRID_H), sprites=[sprite], data={"level_index": int(index)})


def _terrain_sets(spec: LevelSpec) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    walkable: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    grass: set[tuple[int, int]] = set()
    for y, row in enumerate(spec.layout):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
                continue
            walkable.add((x, y))
            if ch == ",":
                grass.add((x, y))
    return walkable, walls, grass


def _goal_tiles(spec: LevelSpec) -> frozenset[tuple[int, int]]:
    gx, gy = spec.goal_top_left
    return frozenset(_rect_tiles(gx, gy, 2, 2))


def _initial_masks(spec: LevelSpec) -> tuple[int, int]:
    door_mask = 0
    lever_mask = 0
    for idx, door in enumerate(spec.doors):
        if door.starts_closed:
            door_mask |= 1 << idx
    for idx, lever in enumerate(spec.levers):
        if lever.starts_on:
            lever_mask |= 1 << idx
    return door_mask, lever_mask


def build_search_model(level_idx: int) -> SearchModel:
    spec = LEVEL_SPECS[int(level_idx)]
    walkable, walls, grass = _terrain_sets(spec)
    return SearchModel(
        level_idx=int(level_idx),
        walkable=frozenset(walkable),
        grass=frozenset(grass),
        walls=frozenset(walls),
        goal_tiles=_goal_tiles(spec),
        guards=spec.guards,
        doors=spec.doors,
        levers=spec.levers,
        controls_click=bool(spec.controls_click),
        vision_on=bool(spec.vision_on),
        vision_range=int(spec.vision_range),
    )


def initial_search_state(level_idx: int) -> SearchState:
    spec = LEVEL_SPECS[int(level_idx)]
    door_mask, lever_mask = _initial_masks(spec)
    guards = tuple((guard.start_x, guard.start_y, guard.start_dir) for guard in spec.guards)
    return SearchState(
        player_x=int(spec.player_start[0]),
        player_y=int(spec.player_start[1]),
        time_remaining=int(spec.max_time),
        door_mask=int(door_mask),
        lever_mask=int(lever_mask),
        guards=guards,
    )


def _door_closed(door_mask: int, idx: int) -> bool:
    return bool((int(door_mask) >> int(idx)) & 1)


def _set_door_closed(door_mask: int, idx: int, closed: bool) -> int:
    if closed:
        return int(door_mask) | (1 << int(idx))
    return int(door_mask) & ~(1 << int(idx))


def _toggle_door(door_mask: int, idx: int) -> int:
    return int(door_mask) ^ (1 << int(idx))


def _toggle_lever(lever_mask: int, idx: int) -> int:
    return int(lever_mask) ^ (1 << int(idx))


def _tile_has_closed_door(model: SearchModel, door_mask: int, x: int, y: int) -> bool:
    for idx, door in enumerate(model.doors):
        if not _door_closed(door_mask, idx):
            continue
        if door.x <= x < door.x + 2 and door.y <= y < door.y + 2:
            return True
    return False


def _guard_blocked_by_solid(model: SearchModel, door_mask: int, x: int, y: int) -> bool:
    for tx, ty in _rect_tiles(x, y, 2, 2):
        if tx < 0 or ty < 1 or tx >= GRID_W or ty >= GRID_H:
            return True
        if (tx, ty) in model.walls:
            return True
        if _tile_has_closed_door(model, door_mask, tx, ty):
            return True
    return False


def _move_guard(
    model: SearchModel, door_mask: int, guard_idx: int, state: tuple[int, int, int]
) -> tuple[int, int, int]:
    x, y, facing = int(state[0]), int(state[1]), int(state[2])
    spec = model.guards[int(guard_idx)]
    dx, dy = DIR_TO_DELTA[facing]
    nx, ny = x + dx, y + dy

    blocked = False
    if spec.horizontal:
        if nx < spec.patrol_min or nx > spec.patrol_max:
            blocked = True
    else:
        if ny < spec.patrol_min or ny > spec.patrol_max:
            blocked = True
    if not blocked and _guard_blocked_by_solid(model, door_mask, nx, ny):
        blocked = True

    if blocked:
        return x, y, REVERSE_DIR[facing]
    return nx, ny, facing


def _vision_tiles(model: SearchModel, state: SearchState) -> set[tuple[int, int]]:
    if not model.vision_on:
        return set()

    visible: set[tuple[int, int]] = set()

    lever_tiles = {(int(lever.x), int(lever.y)) for lever in model.levers}

    def _is_door_tile(x: int, y: int) -> bool:
        for door in model.doors:
            if int(door.x) <= x < int(door.x) + 2 and int(door.y) <= y < int(door.y) + 2:
                return True
        return False

    for _guard_idx, guard_state in enumerate(state.guards):
        gx, gy, facing = int(guard_state[0]), int(guard_state[1]), int(guard_state[2])
        door_mask = int(state.door_mask)

        rays: list[tuple[int, int, int, int]] = []
        if facing == DIR_RIGHT:
            rays = [(gx + 1, gy, 1, 0), (gx + 1, gy + 1, 1, 0)]
        elif facing == DIR_LEFT:
            rays = [(gx, gy, -1, 0), (gx, gy + 1, -1, 0)]
        elif facing == DIR_DOWN:
            rays = [(gx, gy + 1, 0, 1), (gx + 1, gy + 1, 0, 1)]
        elif facing == DIR_UP:
            rays = [(gx, gy, 0, -1), (gx + 1, gy, 0, -1)]

        for sx, sy, dx, dy in rays:
            for step in range(1, int(model.vision_range) + 1):
                tx = sx + (dx * step)
                ty = sy + (dy * step)
                if tx < 0 or ty < 1 or tx >= GRID_W or ty >= GRID_H:
                    break
                if (tx, ty) in model.walls:
                    break
                if _tile_has_closed_door(model, door_mask, tx, ty):
                    break
                if (tx, ty) in model.grass:
                    break
                if (tx, ty) in model.goal_tiles:
                    continue
                if (tx, ty) in lever_tiles:
                    continue
                if _is_door_tile(tx, ty):
                    continue
                visible.add((tx, ty))

    return visible


def _player_on_grass(model: SearchModel, px: int, py: int) -> bool:
    return (int(px), int(py)) in model.grass


def _player_collides_guard(px: int, py: int, guards: tuple[tuple[int, int, int], ...]) -> bool:
    for gx, gy, _facing in guards:
        if int(gx) <= int(px) < int(gx) + 2 and int(gy) <= int(py) < int(gy) + 2:
            return True
    return False


def _is_walkable(model: SearchModel, door_mask: int, x: int, y: int) -> bool:
    if x < 0 or y < 1 or x >= GRID_W or y >= GRID_H:
        return False
    if (x, y) not in model.walkable:
        return False
    return not _tile_has_closed_door(model, door_mask, x, y)


def _consume_action(
    model: SearchModel, state: SearchState, action_id: int, click_pos: tuple[int, int] | None
) -> SearchState:
    px = int(state.player_x)
    py = int(state.player_y)
    door_mask = int(state.door_mask)
    lever_mask = int(state.lever_mask)

    if action_id in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[int(action_id)]
        nx, ny = px + dx, py + dy
        if _is_walkable(model, door_mask, nx, ny):
            px, py = nx, ny
    elif action_id == CLICK_ACTION and model.controls_click and click_pos is not None:
        cx, cy = int(click_pos[0]), int(click_pos[1])
        for lever_idx, lever in enumerate(model.levers):
            if (cx, cy) != (int(lever.x), int(lever.y)):
                continue
            if abs(px - cx) + abs(py - cy) != 1:
                continue
            lever_mask = _toggle_lever(lever_mask, lever_idx)
            for door_idx in lever.linked_doors:
                door_mask = _toggle_door(door_mask, int(door_idx))
            break

    return SearchState(
        player_x=px,
        player_y=py,
        time_remaining=int(state.time_remaining),
        door_mask=door_mask,
        lever_mask=lever_mask,
        guards=tuple(state.guards),
    )


def apply_action_transition(
    model: SearchModel, state: SearchState, action_id: int, click_pos: tuple[int, int] | None = None
) -> tuple[SearchState | None, str]:
    action_id = int(action_id)

    after_action = _consume_action(model, state, action_id, click_pos)

    moved_guards: list[tuple[int, int, int]] = []
    for guard_idx, guard_state in enumerate(after_action.guards):
        moved_guards.append(_move_guard(model, int(after_action.door_mask), guard_idx, guard_state))

    stepped = SearchState(
        player_x=int(after_action.player_x),
        player_y=int(after_action.player_y),
        time_remaining=int(after_action.time_remaining),
        door_mask=int(after_action.door_mask),
        lever_mask=int(after_action.lever_mask),
        guards=tuple(moved_guards),
    )

    px, py = int(stepped.player_x), int(stepped.player_y)

    if _player_collides_guard(px, py, stepped.guards):
        return stepped, "fail"

    visible = _vision_tiles(model, stepped)
    if (px, py) in visible and not _player_on_grass(model, px, py):
        return stepped, "fail"

    if (px, py) in model.goal_tiles:
        return stepped, "win"

    next_time = int(stepped.time_remaining) - 1
    if next_time <= 0:
        expired = SearchState(
            player_x=px,
            player_y=py,
            time_remaining=0,
            door_mask=int(stepped.door_mask),
            lever_mask=int(stepped.lever_mask),
            guards=tuple(stepped.guards),
        )
        return expired, "fail"

    next_state = SearchState(
        player_x=px,
        player_y=py,
        time_remaining=next_time,
        door_mask=int(stepped.door_mask),
        lever_mask=int(stepped.lever_mask),
        guards=tuple(stepped.guards),
    )
    return next_state, "ok"


class TimedPatrolCrossing(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(idx, spec) for idx, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._level_idx = 0
        self._model: SearchModel = build_search_model(0)
        self._runtime_state: SearchState = initial_search_state(0)
        self._mode = MODE_NORMAL
        self._anim_tick = 0
        self._board: Sprite | None = None
        self._vision_cache: set[tuple[int, int]] = set()
        self._route_score = 0

    def on_set_level(self, level: Level) -> None:
        idx = int(level.get_data("level_index") or 0)
        self._level_idx = idx
        self._model = build_search_model(idx)
        self._runtime_state = initial_search_state(idx)
        self._mode = MODE_NORMAL
        self._anim_tick = 0
        self._board = next(iter(level.get_sprites_by_name("board")), None)
        self._vision_cache = _vision_tiles(self._model, self._runtime_state)
        self._render_board(flash=False)

    def export_search_model(self) -> SearchModel:
        return self._model

    def export_initial_search_state(self) -> SearchState:
        return initial_search_state(self._level_idx)

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
        if gx < 0 or gy < 0 or gx >= GRID_W or gy >= GRID_H:
            return None
        return gx, gy

    def _timebar_filled_segments(self) -> int:
        spec = LEVEL_SPECS[self._level_idx]
        tps = int(spec.time_per_segment)
        if self._runtime_state.time_remaining <= 0:
            return 0
        return max(0, min(spec.max_segments, (int(self._runtime_state.time_remaining) + tps - 1) // tps))

    def _guard_tiles(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for gx, gy, _dir in self._runtime_state.guards:
            out.update(_rect_tiles(int(gx), int(gy), 2, 2))
        return out

    def _door_tiles(self, include_open: bool = False) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for idx, door in enumerate(self._model.doors):
            if not include_open and not _door_closed(self._runtime_state.door_mask, idx):
                continue
            out.update(_rect_tiles(int(door.x), int(door.y), 2, 2))
        return out

    def _goal_tiles(self) -> set[tuple[int, int]]:
        return set(self._model.goal_tiles)

    def _lever_tiles(self) -> set[tuple[int, int]]:
        return {(int(lever.x), int(lever.y)) for lever in self._model.levers}

    def _render_board(self, flash: bool) -> None:
        board = np.full((GRID_H, GRID_W), int(COLOR_VOID), dtype=np.int8)

        for y in range(1, GRID_H):
            for x in range(GRID_W):
                if (x, y) in self._model.walls:
                    board[y, x] = COLOR_WALL
                elif (x, y) in self._model.grass:
                    board[y, x] = COLOR_GRASS
                elif (x, y) in self._model.walkable:
                    board[y, x] = COLOR_FLOOR

        for idx, door in enumerate(self._model.doors):
            if not _door_closed(self._runtime_state.door_mask, idx):
                continue
            for x, y in _rect_tiles(int(door.x), int(door.y), 2, 2):
                board[y, x] = COLOR_DOOR

        for x, y in self._goal_tiles():
            board[y, x] = COLOR_GOAL

        for x, y in self._lever_tiles():
            board[y, x] = COLOR_LEVER

        guard_tiles = self._guard_tiles()
        closed_door_tiles = self._door_tiles(include_open=False)
        entity_tiles = set(guard_tiles) | self._lever_tiles() | self._goal_tiles() | closed_door_tiles

        if self._model.vision_on:
            for vx, vy in self._vision_cache:
                if (vx, vy) in entity_tiles:
                    continue
                if (vx, vy) not in self._model.walkable:
                    continue
                if (vx, vy) in self._model.grass:
                    continue
                board[vy, vx] = COLOR_VISION

        guard_color = COLOR_GUARD_A if (self._anim_tick % 2 == 0) else COLOR_GUARD_B
        for gx, gy, _dir in self._runtime_state.guards:
            for tx, ty in _rect_tiles(int(gx), int(gy), 2, 2):
                board[ty, tx] = guard_color

        px, py = int(self._runtime_state.player_x), int(self._runtime_state.player_y)
        player_color = COLOR_PLAYER_HIDDEN if _player_on_grass(self._model, px, py) else COLOR_PLAYER_VISIBLE
        board[py, px] = player_color

        filled_segments = self._timebar_filled_segments()
        low_time = filled_segments <= 5
        blink_fill = COLOR_DANGER if (low_time and (self._anim_tick % 2 == 1)) else COLOR_TIME_FILLED
        board[0, 0] = COLOR_WALL
        board[0, GRID_W - 1] = COLOR_WALL
        for x in range(1, GRID_W - 1):
            segment = x - 1
            board[0, x] = blink_fill if segment < filled_segments else COLOR_TIME_EMPTY

        if flash:
            for gx, gy, _ in self._runtime_state.guards:
                for tx, ty in _rect_tiles(int(gx), int(gy), 2, 2):
                    board[ty, tx] = COLOR_DANGER
            for vx, vy in self._vision_cache:
                if board[vy, vx] == COLOR_VISION:
                    board[vy, vx] = COLOR_DANGER
            board[py, px] = COLOR_DANGER

        if self._board is not None:
            self._board.pixels = board

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

        if self._mode == MODE_WIN_FLASH:
            self.next_level()
            self._route_score += 1
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == CLICK_ACTION and not self._model.controls_click:
            action_id = WAIT_ACTION

        click_pos = self._parse_click_grid() if action_id == CLICK_ACTION else None
        next_state, outcome = apply_action_transition(self._model, self._runtime_state, action_id, click_pos)

        if next_state is not None:
            self._runtime_state = next_state
        self._anim_tick += 1
        self._vision_cache = _vision_tiles(self._model, self._runtime_state)

        if outcome == "fail":
            self._mode = MODE_NORMAL
            self._render_board(flash=True)
            self.lose()
            self.complete_action()
            return

        if outcome == "win":
            self._mode = MODE_WIN_FLASH
            self._render_board(flash=True)
            self.complete_action()
            return

        self._mode = MODE_NORMAL
        self._render_board(flash=False)
        self.complete_action()
