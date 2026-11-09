from __future__ import annotations

from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "tag-0001"
GRID_WIDTH = 32
GRID_HEIGHT = 18
TIMEBAR_ROWS = 2

COLOR_VOID = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER_BODY = 3
COLOR_PLAYER_ARROW_READY = 4
COLOR_TARGET_BODY = 5
COLOR_TARGET_LEGS = 6
COLOR_TIME_FILL = 7
COLOR_TIME_EMPTY = 8
COLOR_DOOR_CLOSED = 9
COLOR_OPEN = 10
COLOR_PLATE = 11
COLOR_MUD = 12
COLOR_TELEPORT = 13
COLOR_MOVING_GATE_CLOSED = 14
COLOR_FLASH = 15

ACTION_WAIT = int(GameAction.ACTION6.value)
ACTION_DASH = int(GameAction.ACTION5.value)

DIR_UP = 0
DIR_LEFT = 1
DIR_DOWN = 2
DIR_RIGHT = 3
DIR_STAY = 4

MOVE_BY_DIR = {DIR_UP: (0, -1), DIR_LEFT: (-1, 0), DIR_DOWN: (0, 1), DIR_RIGHT: (1, 0), DIR_STAY: (0, 0)}

ACTION_TO_DIR = {
    int(GameAction.ACTION1.value): DIR_UP,
    int(GameAction.ACTION2.value): DIR_DOWN,
    int(GameAction.ACTION3.value): DIR_LEFT,
    int(GameAction.ACTION4.value): DIR_RIGHT,
}

TARGET_DIR_ORDER = [DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT, DIR_STAY]


class Rect2(NamedTuple):
    x: int
    y: int


class TagLevelModel(NamedTuple):
    index: int
    name: str
    time_limit: int
    walls: frozenset[tuple[int, int]]
    mud: frozenset[tuple[int, int]]
    plates: tuple[tuple[int, int], ...]
    doors: tuple[Rect2, ...]
    moving_gate: Rect2 | None
    teleporter_pairs: tuple[tuple[Rect2, Rect2], ...]
    player_start: tuple[int, int]
    player_dir: int
    target_start: tuple[int, int]

    @property
    def dash_enabled(self) -> bool:
        return self.index >= 4


class TagSimState(NamedTuple):
    player_x: int
    player_y: int
    player_dir: int
    target_x: int
    target_y: int
    target_prev_dir: int
    door_open: bool
    moving_gate_open: bool
    exhausted: bool
    time_left: int
    won: bool = False


class TagStepEffects(NamedTuple):
    dashed: bool
    trail_cells: frozenset[tuple[int, int]]
    plate_flash_cells: frozenset[tuple[int, int]]
    teleporter_flash_indices: frozenset[int]
    lose: bool


LEVEL_LAYOUTS: list[tuple[str, int, list[str]]] = [
    (
        "Level 1",
        64,
        [
            "================================",
            "================================",
            "################################",
            "#..............................#",
            "#......................##......#",
            "#......................##..@@..#",
            "#..........................@@..#",
            "#..............####............#",
            "#..............#..#............#",
            "#..............####............#",
            "#..............................#",
            "#...........##.................#",
            "#...........##.................#",
            "#..............................#",
            "#..o>..........................#",
            "#..o>..........................#",
            "#..............................#",
            "################################",
        ],
    ),
    (
        "Level 2",
        56,
        [
            "================================",
            "========================--------",
            "################################",
            "#..............................#",
            "#.........................@@...#",
            "#.........................@@...#",
            "#..............####............#",
            "#..............####............#",
            "#....####......##########......#",
            "#....####......##########......#",
            "#......##......##########......#",
            "#......##......................#",
            "#......##......######..........#",
            "#..............######..........#",
            "#....................####......#",
            "#..o>..............####........#",
            "#..o>..............####........#",
            "################################",
        ],
    ),
    (
        "Level 3",
        52,
        [
            "================================",
            "====================------------",
            "################################",
            "#..............##..............#",
            "#...######.....##........@@....#",
            "#...#....#.....##........@@....#",
            "#...#....#.....##..............#",
            "#...######.....##..#####.......#",
            "#..............##..#...#.......#",
            "#......&.......##..#...#.......#",
            "#..............||..............#",
            "#..............||....#####.....#",
            "#..............##....#...#.....#",
            "#..####........##....#...#.....#",
            "#..#..#........##....#####.....#",
            "#..o>..........##..............#",
            "#..o>..........##..............#",
            "################################",
        ],
    ),
    (
        "Level 4",
        48,
        [
            "================================",
            "================----------------",
            "################################",
            "#..............##..............#",
            "#...######.....##........@@....#",
            "#...#....#.....##........@@....#",
            "#...######.....##..#####.......#",
            "#..............:::::::.........#",
            "#..............:::::::.........#",
            "#..####........##....#####.....#",
            "#..#..#........##....#...#.....#",
            "#..#..#........##....#.&.#.....#",
            "#..####........##....#####.....#",
            "#..............||....#...#.....#",
            "#..............||....#####.....#",
            "#..o>..........##..............#",
            "#..o>..........##..............#",
            "################################",
        ],
    ),
    (
        "Level 5",
        44,
        [
            "================================",
            "============--------------------",
            "################################",
            "#..............................#",
            "#......................@@......#",
            "#............::::......@@......#",
            "#............::::..............#",
            "#......##################......#",
            "#......##################......#",
            "#......########||########......#",
            "#......########||########......#",
            "#......##################......#",
            "#......##################......#",
            "#..............................#",
            "#......&.......................#",
            "#..o>..........................#",
            "#..o>..........................#",
            "################################",
        ],
    ),
    (
        "Level 6",
        40,
        [
            "================================",
            "========------------------------",
            "################################",
            "#..............##..............#",
            "#..OO..........##........@@....#",
            "#..OO..::::....##........@@....#",
            "#......::::....##....#####.....#",
            "#..............XX....#...#.....#",
            "#..............XX....#...#.....#",
            "#...######.....##....#.&.#.....#",
            "#...#....#.....##....#####.....#",
            "#...######.....##..............#",
            "#..............||....#####.....#",
            "#..............||....#...#.....#",
            "#..............##..OO..........#",
            "#..o>..........##..OO..........#",
            "#..o>..####....##..............#",
            "################################",
        ],
    ),
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _footprint(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))


def _rect_cells(rect: Rect2) -> tuple[tuple[int, int], ...]:
    return _footprint(rect.x, rect.y)


def _overlaps_rect(entity_x: int, entity_y: int, rect: Rect2) -> bool:
    ex0, ey0 = entity_x, entity_y
    ex1, ey1 = entity_x + 1, entity_y + 1
    rx0, ry0 = rect.x, rect.y
    rx1, ry1 = rect.x + 1, rect.y + 1
    return not (ex1 < rx0 or ex0 > rx1 or ey1 < ry0 or ey0 > ry1)


def _initial_state(model: TagLevelModel) -> TagSimState:
    return TagSimState(
        player_x=model.player_start[0],
        player_y=model.player_start[1],
        player_dir=model.player_dir,
        target_x=model.target_start[0],
        target_y=model.target_start[1],
        target_prev_dir=DIR_STAY,
        door_open=False,
        moving_gate_open=False,
        exhausted=False,
        time_left=model.time_limit,
        won=False,
    )


def _direction_from_player_cells(cells: list[tuple[int, int, str]]) -> int:
    arrows = [char for _, _, char in cells if char in {"^", "<", "v", ">"}]
    if not arrows:
        return DIR_RIGHT
    arrow = arrows[0]
    if arrow == "^":
        return DIR_UP
    if arrow == "<":
        return DIR_LEFT
    if arrow == "v":
        return DIR_DOWN
    return DIR_RIGHT


def _find_compact_rects(grid: list[str], marker: str) -> tuple[Rect2, ...]:
    found: list[Rect2] = []
    h = len(grid)
    w = len(grid[0]) if h else 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] != marker:
                continue
            if x > 0 and grid[y][x - 1] == marker:
                continue
            if y > 0 and grid[y - 1][x] == marker:
                continue
            if x + 1 >= w or y + 1 >= h:
                raise ValueError(f"{marker} at {(x, y)} must be a 2x2 block")
            if not (grid[y][x + 1] == marker and grid[y + 1][x] == marker and grid[y + 1][x + 1] == marker):
                raise ValueError(f"{marker} at {(x, y)} must be a 2x2 block")
            found.append(Rect2(x, y))
    return tuple(found)


def _build_models() -> list[TagLevelModel]:
    models: list[TagLevelModel] = []
    for idx, (name, time_limit, rows) in enumerate(LEVEL_LAYOUTS):
        if len(rows) != GRID_HEIGHT:
            raise ValueError(f"{name} must have {GRID_HEIGHT} rows")
        if any(len(row) != GRID_WIDTH for row in rows):
            raise ValueError(f"{name} must have width {GRID_WIDTH}")

        walls: set[tuple[int, int]] = set()
        mud: set[tuple[int, int]] = set()
        plates: list[tuple[int, int]] = []
        player_cells: list[tuple[int, int, str]] = []
        target_cells: list[tuple[int, int, str]] = []

        for y, row in enumerate(rows):
            for x, char in enumerate(row):
                if char == "#":
                    walls.add((x, y))
                elif char == ":":
                    mud.add((x, y))
                elif char == "&":
                    plates.append((x, y))
                elif char in {"o", "^", "<", "v", ">"}:
                    player_cells.append((x, y, char))
                elif char in {"@", "~"}:
                    target_cells.append((x, y, char))

        if len(player_cells) != 4:
            raise ValueError(f"{name} must define a 2x2 player")
        if len(target_cells) != 4:
            raise ValueError(f"{name} must define a 2x2 target")

        player_x = min(x for x, _, _ in player_cells)
        player_y = min(y for _, y, _ in player_cells)
        target_x = min(x for x, _, _ in target_cells)
        target_y = min(y for _, y, _ in target_cells)

        doors = _find_compact_rects(rows, "|")
        gates = _find_compact_rects(rows, "X")
        teleports = _find_compact_rects(rows, "O")
        teleporter_pairs: list[tuple[Rect2, Rect2]] = []
        if teleports:
            if len(teleports) % 2 != 0:
                raise ValueError(f"{name} teleporter blocks must come in pairs")
            for pair_idx in range(0, len(teleports), 2):
                teleporter_pairs.append((teleports[pair_idx], teleports[pair_idx + 1]))

        models.append(
            TagLevelModel(
                index=idx,
                name=name,
                time_limit=int(time_limit),
                walls=frozenset(walls),
                mud=frozenset(mud),
                plates=tuple(sorted(plates)),
                doors=doors,
                moving_gate=gates[0] if gates else None,
                teleporter_pairs=tuple(teleporter_pairs),
                player_start=(player_x, player_y),
                player_dir=_direction_from_player_cells(player_cells),
                target_start=(target_x, target_y),
            )
        )

    return models


TAG_LEVEL_MODELS = _build_models()


def _closed_blockers(model: TagLevelModel, state: TagSimState) -> set[tuple[int, int]]:
    blocked = set(model.walls)
    if model.doors and not state.door_open:
        for door in model.doors:
            blocked.update(_rect_cells(door))
    if model.moving_gate is not None and not state.moving_gate_open:
        blocked.update(_rect_cells(model.moving_gate))
    return blocked


def _in_play_bounds(x: int, y: int) -> bool:
    return 0 <= x < GRID_WIDTH and TIMEBAR_ROWS <= y < GRID_HEIGHT


def _passable_entity(model: TagLevelModel, state: TagSimState, x: int, y: int) -> bool:
    blocked = _closed_blockers(model, state)
    for cx, cy in _footprint(x, y):
        if not _in_play_bounds(cx, cy):
            return False
        if (cx, cy) in blocked:
            return False
    return True


def _entity_on_mud(model: TagLevelModel, x: int, y: int) -> bool:
    return any((cx, cy) in model.mud for cx, cy in _footprint(x, y))


def _apply_teleport(model: TagLevelModel, x: int, y: int) -> tuple[int, int, frozenset[int]]:
    flash: set[int] = set()
    for pair_idx, (a_rect, b_rect) in enumerate(model.teleporter_pairs):
        if _overlaps_rect(x, y, a_rect):
            flash.add(pair_idx)
            return (b_rect.x + (x - a_rect.x), b_rect.y + (y - a_rect.y), frozenset(flash))
        if _overlaps_rect(x, y, b_rect):
            flash.add(pair_idx)
            return (a_rect.x + (x - b_rect.x), a_rect.y + (y - b_rect.y), frozenset(flash))
    return x, y, frozenset()


def _overlap_entities(ax: int, ay: int, bx: int, by: int) -> bool:
    a = set(_footprint(ax, ay))
    b = set(_footprint(bx, by))
    return not a.isdisjoint(b)


def advance_tag_state(model: TagLevelModel, state: TagSimState, action_id: int) -> tuple[TagSimState, TagStepEffects]:
    if state.won:
        return state, TagStepEffects(False, frozenset(), frozenset(), frozenset(), False)

    next_player_dir = state.player_dir
    player_x = state.player_x
    player_y = state.player_y
    dashed = False
    dash_extra_cost = 0
    trail_cells: set[tuple[int, int]] = set()

    if action_id in ACTION_TO_DIR:
        next_player_dir = ACTION_TO_DIR[action_id]
        dx, dy = MOVE_BY_DIR[next_player_dir]
        nx, ny = player_x + dx, player_y + dy
        candidate = TagSimState(
            player_x=player_x,
            player_y=player_y,
            player_dir=next_player_dir,
            target_x=state.target_x,
            target_y=state.target_y,
            target_prev_dir=state.target_prev_dir,
            door_open=state.door_open,
            moving_gate_open=state.moving_gate_open,
            exhausted=state.exhausted,
            time_left=state.time_left,
            won=False,
        )
        if _passable_entity(model, candidate, nx, ny):
            player_x, player_y = nx, ny

    elif action_id == ACTION_DASH and model.dash_enabled and not state.exhausted:
        dx, dy = MOVE_BY_DIR[next_player_dir]
        mid_x, mid_y = player_x + dx, player_y + dy
        end_x, end_y = player_x + (2 * dx), player_y + (2 * dy)
        candidate = TagSimState(
            player_x=player_x,
            player_y=player_y,
            player_dir=next_player_dir,
            target_x=state.target_x,
            target_y=state.target_y,
            target_prev_dir=state.target_prev_dir,
            door_open=state.door_open,
            moving_gate_open=state.moving_gate_open,
            exhausted=state.exhausted,
            time_left=state.time_left,
            won=False,
        )
        if _passable_entity(model, candidate, mid_x, mid_y) and _passable_entity(model, candidate, end_x, end_y):
            dashed = True
            dash_extra_cost = 1
            trail_cells.update(_footprint(mid_x, mid_y))
            player_x, player_y = end_x, end_y

    player_x, player_y, player_tp_flash = _apply_teleport(model, player_x, player_y)

    door_open = state.door_open
    plate_flash_cells: set[tuple[int, int]] = set()
    if model.doors and any((px, py) in set(model.plates) for px, py in _footprint(player_x, player_y)):
        door_open = not door_open
        for plate in model.plates:
            if plate in _footprint(player_x, player_y):
                plate_flash_cells.add(plate)

    moving_gate_open = state.moving_gate_open
    if model.moving_gate is not None:
        moving_gate_open = not moving_gate_open

    target_x = state.target_x
    target_y = state.target_y
    target_prev_dir = state.target_prev_dir

    path_state = TagSimState(
        player_x=player_x,
        player_y=player_y,
        player_dir=next_player_dir,
        target_x=target_x,
        target_y=target_y,
        target_prev_dir=target_prev_dir,
        door_open=door_open,
        moving_gate_open=moving_gate_open,
        exhausted=state.exhausted,
        time_left=state.time_left,
        won=False,
    )

    best_choice: tuple[int, int, int, int, bool, int] | None = None
    for order_idx, dir_idx in enumerate(TARGET_DIR_ORDER):
        dx, dy = MOVE_BY_DIR[dir_idx]
        nx, ny = target_x + dx, target_y + dy
        if not _passable_entity(model, path_state, nx, ny):
            continue
        distance = abs(nx - player_x) + abs(ny - player_y)
        enters_mud = _entity_on_mud(model, nx, ny)
        same_dir = dir_idx == target_prev_dir
        candidate_choice = (distance, 1 if not enters_mud else 0, 1 if same_dir else 0, -order_idx, enters_mud, dir_idx)
        if best_choice is None or candidate_choice > best_choice:
            best_choice = candidate_choice
            target_x, target_y = nx, ny
            target_prev_dir = dir_idx

    target_x, target_y, target_tp_flash = _apply_teleport(model, target_x, target_y)

    won = _overlap_entities(player_x, player_y, target_x, target_y)
    mud_extra_cost = 1 if _entity_on_mud(model, player_x, player_y) else 0
    time_left = state.time_left - (1 + mud_extra_cost + dash_extra_cost)
    lose = (not won) and (time_left <= 0)

    exhausted_next = dashed

    next_state = TagSimState(
        player_x=player_x,
        player_y=player_y,
        player_dir=next_player_dir,
        target_x=target_x,
        target_y=target_y,
        target_prev_dir=target_prev_dir,
        door_open=door_open,
        moving_gate_open=moving_gate_open,
        exhausted=exhausted_next,
        time_left=time_left,
        won=won,
    )
    return (
        next_state,
        TagStepEffects(
            dashed=dashed,
            trail_cells=frozenset(trail_cells),
            plate_flash_cells=frozenset(plate_flash_cells),
            teleporter_flash_indices=frozenset(set(player_tp_flash) | set(target_tp_flash)),
            lose=lose,
        ),
    )


def _player_pixels(direction: int, exhausted: bool) -> np.ndarray:
    pixels = np.full((2, 2), COLOR_PLAYER_BODY, dtype=np.int8)
    arrow_color = COLOR_TIME_EMPTY if exhausted else COLOR_PLAYER_ARROW_READY
    if direction == DIR_UP:
        pixels[0, 0] = arrow_color
        pixels[0, 1] = arrow_color
    elif direction == DIR_LEFT:
        pixels[0, 0] = arrow_color
        pixels[1, 0] = arrow_color
    elif direction == DIR_DOWN:
        pixels[1, 0] = arrow_color
        pixels[1, 1] = arrow_color
    else:
        pixels[0, 1] = arrow_color
        pixels[1, 1] = arrow_color
    return pixels


def _target_pixels() -> np.ndarray:
    pixels = np.full((2, 2), COLOR_TARGET_BODY, dtype=np.int8)
    pixels[1, 0] = COLOR_TARGET_LEGS
    pixels[1, 1] = COLOR_TARGET_LEGS
    return pixels


def _build_level(model: TagLevelModel) -> Level:
    sprites: list[Sprite] = [
        Sprite(
            pixels=_solid(GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR),
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8),
            name="walls",
            x=0,
            y=0,
            layer=1,
            tags=["wall", "blocker"],
            collidable=True,
        ),
        Sprite(
            pixels=np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8),
            name="mud",
            x=0,
            y=0,
            layer=2,
            tags=["mud"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8),
            name="trail",
            x=0,
            y=0,
            layer=6,
            tags=["trail"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((TIMEBAR_ROWS, GRID_WIDTH), COLOR_TIME_FILL, dtype=np.int8),
            name="timebar",
            x=0,
            y=0,
            layer=8,
            tags=["hud", "timer"],
            collidable=False,
        ),
        Sprite(
            pixels=_player_pixels(model.player_dir, exhausted=False),
            name="player",
            x=model.player_start[0],
            y=model.player_start[1],
            layer=7,
            tags=["player"],
            collidable=True,
        ),
        Sprite(
            pixels=_target_pixels(),
            name="target",
            x=model.target_start[0],
            y=model.target_start[1],
            layer=5,
            tags=["target"],
            collidable=False,
        ),
    ]

    wall_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
    for x, y in model.walls:
        wall_pixels[y, x] = COLOR_WALL
    sprites[1].pixels = wall_pixels

    mud_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
    for x, y in model.mud:
        mud_pixels[y, x] = COLOR_MUD
    sprites[2].pixels = mud_pixels

    for plate_idx, (px, py) in enumerate(model.plates):
        sprites.append(
            Sprite(
                pixels=np.array([[COLOR_PLATE]], dtype=np.int8),
                name=f"plate_{plate_idx}",
                x=px,
                y=py,
                layer=4,
                tags=["plate"],
                collidable=False,
            )
        )

    for door_idx, door in enumerate(model.doors):
        sprites.append(
            Sprite(
                pixels=np.full((2, 2), COLOR_DOOR_CLOSED, dtype=np.int8),
                name=f"door_{door_idx}",
                x=door.x,
                y=door.y,
                layer=3,
                tags=["door", "blocker"],
                collidable=True,
            )
        )

    if model.moving_gate is not None:
        sprites.append(
            Sprite(
                pixels=np.full((2, 2), COLOR_MOVING_GATE_CLOSED, dtype=np.int8),
                name="moving_gate",
                x=model.moving_gate.x,
                y=model.moving_gate.y,
                layer=3,
                tags=["moving_gate", "blocker"],
                collidable=True,
            )
        )

    for pair_idx, (a_rect, b_rect) in enumerate(model.teleporter_pairs):
        sprites.append(
            Sprite(
                pixels=np.full((2, 2), COLOR_TELEPORT, dtype=np.int8),
                name=f"teleporter_{pair_idx}_a",
                x=a_rect.x,
                y=a_rect.y,
                layer=3,
                tags=["teleporter", f"teleporter_pair_{pair_idx}"],
                collidable=False,
            )
        )
        sprites.append(
            Sprite(
                pixels=np.full((2, 2), COLOR_TELEPORT, dtype=np.int8),
                name=f"teleporter_{pair_idx}_b",
                x=b_rect.x,
                y=b_rect.y,
                layer=3,
                tags=["teleporter", f"teleporter_pair_{pair_idx}"],
                collidable=False,
            )
        )

    return Level(
        name=model.name,
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"level_idx": model.index, "time_limit": model.time_limit},
    )


class Tag(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._models = TAG_LEVEL_MODELS
        self._sim_state = _initial_state(self._models[0])
        self._mud_shimmer = False
        self._teleporter_pulse = False

        self._trail_cells: set[tuple[int, int]] = set()
        self._plate_flash_cells: set[tuple[int, int]] = set()
        self._teleporter_flash_indices: set[int] = set()

        self._player: Sprite | None = None
        self._target: Sprite | None = None
        self._mud_sprite: Sprite | None = None
        self._trail_sprite: Sprite | None = None
        self._timebar: Sprite | None = None
        self._plates: list[Sprite] = []
        self._doors: list[Sprite] = []
        self._moving_gate: Sprite | None = None
        self._teleporter_sprites: dict[int, tuple[Sprite, Sprite]] = {}

        levels = [_build_level(model) for model in self._models]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def _current_model(self) -> TagLevelModel:
        level_idx = int(self.current_level.get_data("level_idx") or 0)
        return self._models[level_idx]

    def on_set_level(self, level: Level) -> None:
        model = self._models[int(level.get_data("level_idx") or 0)]
        self._sim_state = _initial_state(model)
        self._mud_shimmer = False
        self._teleporter_pulse = False
        self._trail_cells.clear()
        self._plate_flash_cells.clear()
        self._teleporter_flash_indices.clear()

        self._player = next(iter(level.get_sprites_by_name("player")), None)
        self._target = next(iter(level.get_sprites_by_name("target")), None)
        self._mud_sprite = next(iter(level.get_sprites_by_name("mud")), None)
        self._trail_sprite = next(iter(level.get_sprites_by_name("trail")), None)
        self._timebar = next(iter(level.get_sprites_by_name("timebar")), None)
        self._plates = sorted(level.get_sprites_by_tag("plate"), key=lambda sprite: sprite.name)
        self._doors = sorted(level.get_sprites_by_tag("door"), key=lambda sprite: sprite.name)
        gate = level.get_sprites_by_name("moving_gate")
        self._moving_gate = gate[0] if gate else None

        self._teleporter_sprites = {}
        for pair_idx in range(len(model.teleporter_pairs)):
            a = next(iter(level.get_sprites_by_name(f"teleporter_{pair_idx}_a")), None)
            b = next(iter(level.get_sprites_by_name(f"teleporter_{pair_idx}_b")), None)
            if a is not None and b is not None:
                self._teleporter_sprites[pair_idx] = (a, b)

        self._sync_visuals(just_dashed=False)

    def _sync_visuals(self, *, just_dashed: bool) -> None:
        model = self._current_model()

        if self._player is not None:
            exhausted_visual = self._sim_state.exhausted or just_dashed
            self._player.set_position(self._sim_state.player_x, self._sim_state.player_y)
            self._player.pixels = _player_pixels(self._sim_state.player_dir, exhausted=exhausted_visual)

        if self._target is not None:
            self._target.set_position(self._sim_state.target_x, self._sim_state.target_y)
            self._target.pixels = _target_pixels()

        if self._mud_sprite is not None:
            mud_color = COLOR_FLASH if self._mud_shimmer else COLOR_MUD
            mud_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
            for x, y in model.mud:
                mud_pixels[y, x] = mud_color
            self._mud_sprite.pixels = mud_pixels

        if self._trail_sprite is not None:
            trail_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
            for x, y in self._trail_cells:
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    trail_pixels[y, x] = COLOR_FLASH
            self._trail_sprite.pixels = trail_pixels

        for plate in self._plates:
            color = COLOR_FLASH if (int(plate.x), int(plate.y)) in self._plate_flash_cells else COLOR_PLATE
            plate.pixels = np.array([[color]], dtype=np.int8)

        for door in self._doors:
            color = COLOR_OPEN if self._sim_state.door_open else COLOR_DOOR_CLOSED
            door.pixels = np.full((2, 2), color, dtype=np.int8)
            door.set_collidable(not self._sim_state.door_open)

        if self._moving_gate is not None:
            if self._sim_state.moving_gate_open:
                gate_color = COLOR_OPEN
            else:
                gate_color = COLOR_MOVING_GATE_CLOSED
            self._moving_gate.pixels = np.full((2, 2), gate_color, dtype=np.int8)
            self._moving_gate.set_collidable(not self._sim_state.moving_gate_open)

        for pair_idx, sprites in self._teleporter_sprites.items():
            flash = pair_idx in self._teleporter_flash_indices
            base_color = COLOR_FLASH if flash else (COLOR_FLASH if self._teleporter_pulse else COLOR_TELEPORT)
            for sprite in sprites:
                sprite.pixels = np.full((2, 2), base_color, dtype=np.int8)

        if self._timebar is not None:
            ticks = max(0, min(64, int(self._sim_state.time_left)))
            row1_fill = min(32, ticks)
            row0_fill = max(0, ticks - 32)
            bar = np.full((2, 32), COLOR_TIME_EMPTY, dtype=np.int8)
            if row0_fill > 0:
                bar[0, :row0_fill] = COLOR_TIME_FILL
            if row1_fill > 0:
                bar[1, :row1_fill] = COLOR_TIME_FILL
            self._timebar.pixels = bar

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

        action_id = int(self.action.id.value)
        if action_id not in {1, 2, 3, 4, 5, 6}:
            action_id = ACTION_WAIT

        self._trail_cells.clear()
        self._plate_flash_cells.clear()
        self._teleporter_flash_indices.clear()

        model = self._current_model()
        next_state, effects = advance_tag_state(model, self._sim_state, action_id)

        self._trail_cells = set(effects.trail_cells)
        self._plate_flash_cells = set(effects.plate_flash_cells)
        self._teleporter_flash_indices = set(effects.teleporter_flash_indices)

        if effects.lose:
            self.lose()
            self.complete_action()
            return

        self._sim_state = next_state

        self._mud_shimmer = not self._mud_shimmer
        self._teleporter_pulse = not self._teleporter_pulse

        self._sync_visuals(just_dashed=effects.dashed)

        if self._sim_state.won:
            self.next_level()

        self.complete_action()
