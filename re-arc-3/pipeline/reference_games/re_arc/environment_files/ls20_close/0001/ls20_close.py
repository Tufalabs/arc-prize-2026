from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
CELL_SIZE = 5
BOARD_LEFT = 4
BOARD_TOP = 0

MAX_STEPS = 42
STARTING_LIVES = 3

COLOR_BLACK = 0
COLOR_WHITE = 1
COLOR_RED = 2
COLOR_BACKGROUND = 3
COLOR_ENERGY = 4
COLOR_FLOOR = 5
COLOR_LOCK_FRAME = 6
COLOR_RECHARGE = 7
COLOR_BLUE = 9
COLOR_ORANGE = 10
COLOR_GREEN = 11
COLOR_FOG = 0

KEY_COLORS = [COLOR_ORANGE, COLOR_BLUE, COLOR_GREEN, COLOR_RED]
SHAPE_COUNT = 6
PUSHER_DIRECTIONS = {">": (1, 0), "<": (-1, 0), "^": (0, 1), "v": (0, -1)}

KEY_SHAPE_INDEX = 5
KEY_COLOR = COLOR_BLUE

KEY_SHAPE_MASKS = (
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int8),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=np.int8),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int8),
    np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1]], dtype=np.int8),
)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start_key: KeyState
    target_keys: tuple[KeyState, ...]
    fog: bool = False
    rotation_contact_degrees: int = 90
    step_budget: int = MAX_STEPS


@dataclass
class KeyState:
    shape_index: int
    color: int
    rotation: int

    def rotate_clockwise(self) -> None:
        self.rotation = (self.rotation + 90) % 360

    def advance_color(self) -> None:
        color_index = KEY_COLORS.index(self.color)
        self.color = KEY_COLORS[(color_index + 1) % len(KEY_COLORS)]

    def advance_shape(self) -> None:
        self.shape_index = (self.shape_index + 1) % SHAPE_COUNT

    def matches(self, other: KeyState) -> bool:
        return self.shape_index == other.shape_index and self.color == other.color and self.rotation == other.rotation


LEVEL_SPECS = (
    LevelSpec(
        name="Rotation Tutorial",
        rows=(
            "############",
            "############",
            "######L#####",
            "######.#####",
            "######.#####",
            "##........##",
            "##.R.#....##",
            "##...#....##",
            "###.##....##",
            "###...P...##",
            "############",
            "############",
        ),
        start_key=KeyState(KEY_SHAPE_INDEX, KEY_COLOR, 270),
        target_keys=(KeyState(KEY_SHAPE_INDEX, KEY_COLOR, 0),),
    ),
    LevelSpec(
        name="Recharge Detour",
        rows=(
            "############",
            "###.......##",
            "#.........##",
            "#.E.#..#..##",
            "#...#..##..#",
            "##.###..#..#",
            "##.###..#.##",
            "##.##..##.##",
            "##L##P.#...#",
            "########.R.#",
            "#######E...#",
            "############",
        ),
        start_key=KeyState(KEY_SHAPE_INDEX, KEY_COLOR, 0),
        target_keys=(KeyState(KEY_SHAPE_INDEX, KEY_COLOR, 270),),
    ),
    LevelSpec(
        name="Color And Pushers",
        rows=(
            "############",
            "#>.....#..^#",
            "#.###.##.R.#",
            "#.#...E#...#",
            "#.#....###.#",
            "#.#........#",
            "#.#E.......#",
            "#.###.####.#",
            "#..#...###.#",
            "#P.#.C.###.#",
            "####...###L#",
            "############",
        ),
        start_key=KeyState(KEY_SHAPE_INDEX, COLOR_ORANGE, 0),
        target_keys=(KeyState(KEY_SHAPE_INDEX, COLOR_BLUE, 180),),
    ),
    LevelSpec(
        name="Shape Color Pushers",
        rows=(
            "############",
            "#L...##...P#",
            "####.#..####",
            "###E....#..#",
            "##...#>.^..#",
            "#..###.v<#.#",
            "#..#S#C#...#",
            "#>.^.###...#",
            "#...<#..<#.#",
            "##.....#...#",
            "####..E##..#",
            "############",
        ),
        start_key=KeyState(4, COLOR_GREEN, 0),
        target_keys=(KeyState(5, COLOR_BLUE, 0),),
    ),
    LevelSpec(
        name="Moving Rotation Setup",
        rows=(
            "############",
            "##...#^.E#L#",
            "#E.S.#..##.#",
            "##.......#.#",
            "###.##>..#.#",
            "#..<#C.v.#.#",
            "#.######.^<#",
            "#.R..#.....#",
            "#......#>P.#",
            "##E..#####.#",
            "###.......v#",
            "############",
        ),
        start_key=KeyState(4, COLOR_ORANGE, 0),
        target_keys=(KeyState(0, COLOR_RED, 180),),
        rotation_contact_degrees=180,
    ),
    LevelSpec(
        name="Two Locks",
        rows=(
            "############",
            "#E.....E#^.#",
            "#.S.....#..#",
            "#.#####...##",
            "#.#...#..<##",
            "#.......#..#",
            "#.#.C.#.##.#",
            "#.#####.##L#",
            "#.....R.##.#",
            "#E......##.#",
            "##..P..###L#",
            "############",
        ),
        start_key=KeyState(0, COLOR_GREEN, 0),
        target_keys=(KeyState(5, COLOR_BLUE, 90), KeyState(0, COLOR_RED, 180)),
        step_budget=120,
    ),
    LevelSpec(
        name="Fog Locksmith",
        rows=(
            "############",
            "#E....#E#E.#",
            "#...#...#.R#",
            "#..P##.##..#",
            "#.###E.^#..#",
            "#.......#..#",
            "#.###.v<#..#",
            "#...#.#....#",
            "#C.S#.#....#",
            "##E##.###.##",
            "#####L##..E#",
            "############",
        ),
        start_key=KeyState(1, COLOR_ORANGE, 0),
        target_keys=(KeyState(0, COLOR_RED, 180),),
        fog=True,
        step_budget=80,
    ),
)


class LocksmithCloseView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Ls20Close | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        self._draw_lock_hint(frame, game)
        self._draw_current_key_panel(frame, game.current_key)
        self._draw_step_bar(frame, game.remaining_steps, game.step_budget)
        self._draw_lives(frame, game.lives)
        if game.fog_enabled:
            self._draw_fog(frame, game)
        return frame

    def _draw_lock_hint(self, frame: np.ndarray, game: Ls20Close) -> None:
        for lock in game.locks:
            if lock["open"] or not game.current_key.matches(lock["target_key"]):
                continue
            x0, y0 = _cell_top_left(*lock["cell"])
            frame[y0 - 1, x0 - 1 : x0 + CELL_SIZE + 1] = COLOR_WHITE

    def _draw_current_key_panel(self, frame: np.ndarray, key: KeyState) -> None:
        frame[52:64, 0:12] = COLOR_BLACK
        frame[52, 0:12] = COLOR_LOCK_FRAME
        frame[52:64, 11] = COLOR_LOCK_FRAME
        self._draw_key_icon(frame, 3, 55, key, scale=2)

    def _draw_step_bar(self, frame: np.ndarray, remaining_steps: int, step_budget: int) -> None:
        bar_x = 14
        bar_y = 61
        bar_width = 42
        frame[bar_y : bar_y + 2, bar_x : bar_x + bar_width] = COLOR_BLACK
        filled = max(0, min(bar_width, int(bar_width * remaining_steps / max(1, step_budget))))
        if filled:
            frame[bar_y : bar_y + 2, bar_x : bar_x + filled] = COLOR_ENERGY

    def _draw_lives(self, frame: np.ndarray, lives: int) -> None:
        for idx in range(STARTING_LIVES):
            x0 = 57 + idx * 2
            color = COLOR_RED if idx < lives else COLOR_BLACK
            frame[61:64, x0 : x0 + 1] = color

    def _draw_key_icon(self, frame: np.ndarray, x: int, y: int, key: KeyState, *, scale: int) -> None:
        mask = _key_mask(key)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row, col] == 0:
                    continue
                frame[y + row * scale : y + (row + 1) * scale, x + col * scale : x + (col + 1) * scale] = key.color

    def _draw_fog(self, frame: np.ndarray, game: Ls20Close) -> None:
        player_x, player_y = _cell_top_left(*game._player_cell())
        center_x = player_x + CELL_SIZE // 2
        center_y = player_y + CELL_SIZE // 2
        yy, xx = np.ogrid[:VIEW_SIZE, :VIEW_SIZE]
        visible = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 20**2
        hud = yy >= 52
        frame[~(visible | hud)] = COLOR_FOG


def _cell_top_left(cell_x: int, cell_y: int) -> tuple[int, int]:
    return BOARD_LEFT + cell_x * CELL_SIZE, BOARD_TOP + cell_y * CELL_SIZE


def _cell_from_top_left(pixel_x: int, pixel_y: int) -> tuple[int, int]:
    return (pixel_x - BOARD_LEFT) // CELL_SIZE, (pixel_y - BOARD_TOP) // CELL_SIZE


def _rotated_shape_mask(rotation: int) -> np.ndarray:
    turns_clockwise = (rotation // 90) % 4
    return np.rot90(KEY_SHAPE_MASKS[KEY_SHAPE_INDEX], -turns_clockwise)


def _key_mask(key: KeyState) -> np.ndarray:
    turns_clockwise = (key.rotation // 90) % 4
    return np.rot90(KEY_SHAPE_MASKS[key.shape_index], -turns_clockwise)


def _solid_pixels(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _floor_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    return Sprite(
        _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR),
        name=f"floor_{cell_x}_{cell_y}",
        x=x,
        y=y,
        layer=-10,
        collidable=False,
        tags=["floor"],
    )


def _wall_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    return Sprite(
        _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_BACKGROUND),
        name=f"wall_{cell_x}_{cell_y}",
        x=x,
        y=y,
        layer=-5,
        collidable=True,
        tags=["wall"],
    )


def _player_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = np.array(
        [
            [COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
        ],
        dtype=np.int8,
    )
    return Sprite(pixels, name="player", x=x, y=y, layer=20, collidable=True, tags=["player"])


def _rotation_tile_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR)
    pixels[2, 1:4] = COLOR_WHITE
    pixels[1:4, 2] = COLOR_WHITE
    return Sprite(pixels, name="rotation_tile", x=x, y=y, layer=5, collidable=False, tags=["rotation_tile"])


def _recharge_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR)
    pixels[1:4, 1:4] = COLOR_RECHARGE
    return Sprite(pixels, name=f"recharge_{cell_x}_{cell_y}", x=x, y=y, layer=6, collidable=False, tags=["recharge"])


def _color_tile_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = np.array(
        [
            [COLOR_ORANGE, COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_ORANGE, COLOR_ORANGE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE],
            [COLOR_GREEN, COLOR_GREEN, COLOR_WHITE, COLOR_RED, COLOR_RED],
            [COLOR_GREEN, COLOR_GREEN, COLOR_RED, COLOR_RED, COLOR_RED],
            [COLOR_GREEN, COLOR_GREEN, COLOR_RED, COLOR_RED, COLOR_RED],
        ],
        dtype=np.int8,
    )
    return Sprite(pixels, name="color_tile", x=x, y=y, layer=5, collidable=False, tags=["color_tile"])


def _shape_tile_sprite(cell_x: int, cell_y: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR)
    pixels[1, 2] = COLOR_WHITE
    pixels[2, 1:4] = COLOR_WHITE
    pixels[3, 1] = COLOR_WHITE
    pixels[3, 3] = COLOR_WHITE
    return Sprite(pixels, name="shape_tile", x=x, y=y, layer=5, collidable=False, tags=["shape_tile"])


def _pusher_sprite(cell_x: int, cell_y: int, symbol: str) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR)
    if symbol in {">", "<"}:
        pixels[1:4, :] = COLOR_WHITE
    else:
        pixels[:, 1:4] = COLOR_WHITE
    return Sprite(pixels, name=f"pusher_{cell_x}_{cell_y}", x=x, y=y, layer=9, collidable=False, tags=["pusher"])


def _lock_frame_sprite(cell_x: int, cell_y: int, lock_index: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = _solid_pixels(CELL_SIZE + 2, CELL_SIZE + 2, COLOR_LOCK_FRAME)
    pixels[1:-1, 1:-1] = COLOR_BLACK
    return Sprite(
        pixels,
        name=f"lock_frame_{lock_index}",
        x=x - 1,
        y=y - 1,
        layer=8,
        collidable=False,
        tags=["lock_visual", f"lock_{lock_index}"],
    )


def _lock_blocker_sprite(cell_x: int, cell_y: int, lock_index: int) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    return Sprite(
        _solid_pixels(CELL_SIZE, CELL_SIZE, COLOR_BLACK),
        name=f"lock_blocker_{lock_index}",
        x=x,
        y=y,
        layer=7,
        visible=False,
        collidable=True,
        tags=["lock_blocker", f"lock_{lock_index}"],
    )


def _key_icon_sprite(name: str, cell_x: int, cell_y: int, key: KeyState) -> Sprite:
    x, y = _cell_top_left(cell_x, cell_y)
    pixels = np.full((3, 3), -1, dtype=np.int8)
    pixels[_key_mask(key) == 1] = key.color
    return Sprite(pixels, name=name, x=x + 1, y=y + 1, layer=12, collidable=False, tags=["key_icon"])


def _build_level(spec: LevelSpec) -> Level:
    sprites: list[Sprite] = [
        Sprite(_solid_pixels(VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND), name="background", layer=-50, collidable=False)
    ]
    start_cell: tuple[int, int] | None = None
    lock_cells: list[tuple[int, int]] = []
    rotation_cells: list[tuple[int, int]] = []
    color_cells: list[tuple[int, int]] = []
    shape_cells: list[tuple[int, int]] = []
    recharge_cells: list[tuple[int, int]] = []
    wall_cells: list[tuple[int, int]] = []
    pusher_specs: list[dict[str, object]] = []

    for y, row in enumerate(spec.rows):
        for x, cell in enumerate(row):
            if cell == "#":
                wall_cells.append((x, y))
                sprites.append(_wall_sprite(x, y))
                continue
            sprites.append(_floor_sprite(x, y))
            if cell == "P":
                start_cell = (x, y)
                sprites.append(_player_sprite(x, y))
            elif cell == "R":
                rotation_cells.append((x, y))
                sprites.append(_rotation_tile_sprite(x, y))
            elif cell == "E":
                recharge_cells.append((x, y))
                sprites.append(_recharge_sprite(x, y))
            elif cell == "C":
                color_cells.append((x, y))
                sprites.append(_color_tile_sprite(x, y))
            elif cell == "S":
                shape_cells.append((x, y))
                sprites.append(_shape_tile_sprite(x, y))
            elif cell in PUSHER_DIRECTIONS:
                pusher_specs.append({"cell": (x, y), "direction": PUSHER_DIRECTIONS[cell]})
                sprites.append(_pusher_sprite(x, y, cell))
            elif cell == "L":
                lock_index = len(lock_cells)
                target_key = spec.target_keys[min(lock_index, len(spec.target_keys) - 1)]
                lock_cells.append((x, y))
                sprites.extend(
                    [
                        _lock_frame_sprite(x, y, lock_index),
                        _lock_blocker_sprite(x, y, lock_index),
                        _key_icon_sprite(f"lock_target_key_{lock_index}", x, y, target_key),
                    ]
                )

    if start_cell is None or not lock_cells:
        raise ValueError(f"{spec.name} must include player and lock cells.")

    return Level(
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        sprites=sprites,
        name=spec.name,
        data={
            "start_cell": start_cell,
            "lock_cells": lock_cells,
            "rotation_cells": rotation_cells,
            "color_cells": color_cells,
            "shape_cells": shape_cells,
            "recharge_cells": recharge_cells,
            "wall_cells": wall_cells,
            "pusher_specs": pusher_specs,
            "start_key": spec.start_key,
            "target_keys": spec.target_keys,
            "fog": spec.fog,
            "rotation_contact_degrees": spec.rotation_contact_degrees,
            "step_budget": spec.step_budget,
        },
    )


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class Ls20Close(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = LocksmithCloseView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "ls20_close",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT],
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        lock_blockers = self.current_level.get_sprites_by_tag("lock_blocker")
        lock_cells = self.current_level.get_data("lock_cells")
        target_keys = self.current_level.get_data("target_keys")
        self.locks = [
            {
                "cell": tuple(cell),
                "target_key": target_keys[index],
                "blocker": lock_blockers[index],
                "visuals": [
                    sprite
                    for sprite in self.current_level.get_sprites_by_tag(f"lock_{index}")
                    if "lock_visual" in sprite.tags
                ],
                "key_icons": self.current_level.get_sprites_by_name(f"lock_target_key_{index}"),
                "open": False,
            }
            for index, cell in enumerate(lock_cells)
        ]
        self.recharge_sprites = self.current_level.get_sprites_by_tag("recharge")
        self.pusher_sprites = self.current_level.get_sprites_by_tag("pusher")
        self.start_cell = self.current_level.get_data("start_cell")
        self.lock_cell = self.locks[0]["cell"]
        self.rotation_cells = set(self.current_level.get_data("rotation_cells") or [])
        self.color_cells = set(self.current_level.get_data("color_cells") or [])
        self.shape_cells = set(self.current_level.get_data("shape_cells") or [])
        self.wall_cells = set(self.current_level.get_data("wall_cells") or [])
        self.pusher_by_cell = {
            tuple(spec["cell"]): {
                "direction": tuple(spec["direction"]),
                "sprite": self.pusher_sprites[index],
                "start_pixel": _cell_top_left(*spec["cell"]),
            }
            for index, spec in enumerate(self.current_level.get_data("pusher_specs") or [])
        }
        self.animation_frames: list[tuple[Sprite, tuple[int, int]]] = []
        self.start_key = self.current_level.get_data("start_key")
        self.target_key = self.locks[0]["target_key"]
        self.fog_enabled = bool(self.current_level.get_data("fog"))
        self.rotation_contact_degrees = int(self.current_level.get_data("rotation_contact_degrees") or 90)
        self.step_budget = int(self.current_level.get_data("step_budget") or MAX_STEPS)
        self._reset_attempt()
        self.lives = STARTING_LIVES
        self.lock_open = False

    def _reset_attempt(self) -> None:
        self.player.set_position(*_cell_top_left(*self.start_cell))
        self.current_key = KeyState(self.start_key.shape_index, self.start_key.color, self.start_key.rotation)
        self.remaining_steps = self.step_budget
        for lock in self.locks:
            lock["open"] = False
            for sprite in lock["visuals"]:
                sprite.set_visible(True)
            lock["blocker"].set_collidable(True)
        for recharge in self.recharge_sprites:
            recharge.set_visible(True)
        for spec in self.pusher_by_cell.values():
            spec["sprite"].set_position(*spec["start_pixel"])
        self.animation_frames.clear()

    def _player_cell(self) -> tuple[int, int]:
        return _cell_from_top_left(int(self.player.x), int(self.player.y))

    def _try_move(self, dx: int, dy: int) -> None:
        target_x, target_y = self._player_cell()
        target_cell = (target_x + dx, target_y + dy)
        target_lock = self._lock_at(target_cell)
        for lock in self.locks:
            should_block = not lock["open"] and not (
                lock is target_lock and self.current_key.matches(lock["target_key"])
            )
            lock["blocker"].set_collidable(should_block)

        previous_cell = self._player_cell()
        self.try_move_sprite(self.player, dx * CELL_SIZE, dy * CELL_SIZE)
        recharged = False
        if self._player_cell() != previous_cell:
            recharged = self._apply_cell_effect()
        self._spend_step(recharged=recharged)

    def _apply_cell_effect(self) -> bool:
        player_cell = self._player_cell()
        if player_cell in self.rotation_cells:
            for _ in range(self.rotation_contact_degrees // 90):
                self.current_key.rotate_clockwise()
            return False
        if player_cell in self.color_cells:
            self.current_key.advance_color()
            return False
        if player_cell in self.shape_cells:
            self.current_key.advance_shape()
            return False
        if player_cell in self.pusher_by_cell:
            self._queue_pusher_animation(player_cell)
            return False
        lock = self._lock_at(player_cell)
        if lock is not None and self.current_key.matches(lock["target_key"]):
            self._open_lock(lock)
            return False
        for recharge in self.recharge_sprites:
            if recharge.is_visible and _cell_from_top_left(int(recharge.x), int(recharge.y)) == player_cell:
                recharge.set_visible(False)
                self.remaining_steps = self.step_budget
                return True
        return False

    def _lock_at(self, cell: tuple[int, int]) -> dict[str, object] | None:
        for lock in self.locks:
            if not lock["open"] and lock["cell"] == cell:
                return lock
        return None

    def _open_lock(self, lock: dict[str, object]) -> None:
        lock["open"] = True
        for sprite in [lock["blocker"], *lock["visuals"], *lock["key_icons"]]:
            sprite.set_visible(False)
            sprite.set_collidable(False)
        self.lock_open = all(lock_state["open"] for lock_state in self.locks)
        if self.lock_open:
            self.next_level()

    def _queue_pusher_animation(self, pusher_cell: tuple[int, int]) -> None:
        spec = self.pusher_by_cell[pusher_cell]
        direction = spec["direction"]
        pusher = spec["sprite"]
        carry_cells = self._pusher_carry_cells(pusher_cell, direction)
        self.animation_frames = []
        for cell in carry_cells:
            pixel = _cell_top_left(*cell)
            self.animation_frames.append((self.player, pixel))
            self.animation_frames.append((pusher, pixel))
        for cell in reversed(carry_cells[:-1]):
            self.animation_frames.append((pusher, _cell_top_left(*cell)))
        self.animation_frames.append((pusher, spec["start_pixel"]))

    def _pusher_carry_cells(self, start_cell: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
        cells = [start_cell]
        x, y = start_cell
        dx, dy = direction
        while True:
            next_cell = (x + dx, y + dy)
            if self._blocks_pusher(next_cell):
                return cells
            cells.append(next_cell)
            x, y = next_cell

    def _blocks_pusher(self, cell: tuple[int, int]) -> bool:
        return cell in self.wall_cells or self._lock_at(cell) is not None

    def _play_animation_frame(self) -> None:
        sprite, pixel = self.animation_frames.pop(0)
        sprite.set_position(*pixel)
        if not self.animation_frames:
            self.complete_action()

    def _spend_step(self, *, recharged: bool) -> None:
        if self.lock_open or recharged:
            return

        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return

        self.lives -= 1
        if self.lives <= 0:
            self.lose()
            return
        self._reset_attempt()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action in MOVE_DELTAS and not self.lock_open:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)
        if not self.animation_frames:
            self.complete_action()
