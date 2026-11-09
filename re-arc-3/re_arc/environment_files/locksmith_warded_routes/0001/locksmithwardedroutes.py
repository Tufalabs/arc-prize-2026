from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "locksmith_warded_routes"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

DELTA_BY_ACTION = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
PUSH_DELTA = {">": (1, 0), "<": (-1, 0), "^": (0, -1), "v": (0, 1)}

COLOR_VOID = 5
COLOR_FLOOR = 1
COLOR_WALL = 3
COLOR_AVATAR = 15
COLOR_WHITE = 0
COLOR_YELLOW = 11
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_GREEN = 14
COLOR_MAGENTA = 6
COLOR_LOCK = 4
COLOR_EXIT = 11
COLOR_FOG = 2
COLOR_RAIL = 3
COLOR_OPEN = 2
COLOR_HUD = 10

KEY_COLOR_ID = {"red": COLOR_RED, "blue": COLOR_BLUE, "green": COLOR_GREEN, "magenta": COLOR_MAGENTA}
COLOR_TILE = {"R": "red", "B": "blue", "G": "green", "M": "magenta"}
SHAPE_TILE = {"h": "Hook", "f": "Fork", "o": "Loop"}
ROTATIONS = (0, 90, 180, 270)

TILE = 4
BOARD_Y = 10
ICON_SIZE = 9
TOKEN_SIZE = 8


@dataclass(frozen=True)
class KeyState:
    shape: str
    color: str
    rotation: int


LEVEL_SPECS: tuple[dict[str, object], ...] = (
    {
        "name": "level_1",
        "rows": ("#######", "#P.AX##", "#+#...#", "#...###", "#######"),
        "key": ("Hook", "red", 0),
        "energy": 7,
        "max_energy": 7,
        "locks": {"A": ("Hook", "red", 90)},
        "moving": (),
        "solution": (2, 1, 4, 4, 4),
    },
    {
        "name": "level_2",
        "rows": ("#########", "#P.A##X##", "#.#.##C##", "#.#Y#...#", "#...#.B.#", "###.+...#", "#########"),
        "key": ("Hook", "red", 90),
        "energy": 8,
        "max_energy": 14,
        "locks": {"A": ("Hook", "red", 90), "C": ("Hook", "blue", 180)},
        "moving": (),
        "solution": (4, 4, 2, 2, 2, 2, 4, 4, 4, 1, 1, 1, 1),
    },
    {
        "name": "level_3",
        "rows": ("##########", "#P..B#..X#", "#.#.##A###", "#.#Y#....#", "#...#....#", "###>>>+###", "##########"),
        "key": ("Hook", "red", 0),
        "energy": 8,
        "max_energy": 12,
        "locks": {"A": ("Hook", "blue", 90)},
        "moving": (),
        "solution": (4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4),
    },
    {
        "name": "level_4",
        "rows": (
            "###########",
            "#P.A..#D.X#",
            "#.#.#.#C###",
            "#.#f#.+...#",
            "#Y..#.#hG.#",
            "###.#.....#",
            "###########",
        ),
        "key": ("Hook", "blue", 90),
        "energy": 18,
        "max_energy": 24,
        "locks": {"A": ("Hook", "blue", 90), "C": ("Fork", "blue", 180), "D": ("Hook", "green", 180)},
        "moving": (),
        "solution": (4, 4, 2, 2, 2, 3, 3, 4, 4, 1, 1, 1, 4, 4, 2, 2, 4, 4, 1, 2, 2, 4, 3, 1, 1, 1, 4, 4),
    },
    {
        "name": "level_5",
        "rows": ("##########", "#P..#..X##", "#.#.#A####", "#.#1:+...#", "#Y#.#.##.#", "#...#....#", "##########"),
        "key": ("Hook", "red", 90),
        "energy": 8,
        "max_energy": 14,
        "locks": {"A": ("Hook", "blue", 180)},
        "moving": ({"token": "1", "effect": ("color", "blue"), "path": ((3, 3), (4, 3)), "start": (3, 3), "dir": 1},),
        "solution": (2, 2, 2, 1, 1, 1, 4, 4, 2, 5, 2, 4, 4, 1, 1, 4, 4),
    },
    {
        "name": "level_6",
        "rows": (
            "#############",
            "#P.A..#.....#",
            "#.#.#.#...DX#",
            "#.#Y#1>>>:C.#",
            "#...#:#..2M.#",
            "###.#.####+.#",
            "#############",
        ),
        "key": ("Hook", "green", 0),
        "energy": 14,
        "max_energy": 20,
        "locks": {"A": ("Hook", "green", 0), "C": ("Fork", "green", 90), "D": ("Fork", "magenta", 180)},
        "moving": (
            {"token": "1", "effect": ("shape", "Fork"), "path": ((5, 3), (5, 4)), "start": (5, 3), "dir": 1},
            {"token": "2", "effect": ("rotate", 90), "path": ((9, 3), (9, 4)), "start": (9, 4), "dir": -1},
        ),
        "solution": (4, 4, 2, 2, 1, 1, 4, 4, 2, 5, 2, 4, 4, 2, 2, 1, 1, 1, 4),
    },
    {
        "name": "level_7",
        "rows": (
            "###############",
            "#P.A..#########",
            "#.#.#.#####D.##",
            "#.#Y#1>>>:CB.##",
            "#...#:###2#+.##",
            "#####...####..#",
            "#######..EG.h.#",
            "#########.#####",
            "#########oM+FX#",
            "###############",
            "###############",
        ),
        "key": ("Hook", "red", 0),
        "energy": 16,
        "max_energy": 34,
        "locks": {
            "A": ("Hook", "red", 0),
            "C": ("Fork", "red", 90),
            "D": ("Fork", "blue", 180),
            "E": ("Hook", "green", 180),
            "F": ("Loop", "magenta", 270),
        },
        "moving": (
            {"token": "1", "effect": ("shape", "Fork"), "path": ((5, 3), (5, 4)), "start": (5, 3), "dir": 1},
            {"token": "2", "effect": ("rotate", 90), "path": ((9, 3), (9, 4)), "start": (9, 4), "dir": -1},
        ),
        "fog": {"x0": 5, "x1": 13, "y0": 2, "y1": 8},
        "solution": (4, 4, 2, 2, 1, 1, 4, 4, 2, 5, 2, 4, 4, 4, 2, 1, 1, 4, 2, 2, 2, 2, 3, 3, 3, 2, 2, 4, 4, 4, 4),
    },
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int16)


def _rotate_cells(cells: set[tuple[int, int]], rotation: int, size: int) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range((rotation % 360) // 90):
        out = {(size - 1 - y, x) for x, y in out}
    return out


def _key_cells(shape: str, size: int) -> set[tuple[int, int]]:
    if size < ICON_SIZE:
        return _scaled_cells(_key_cells(shape, ICON_SIZE), ICON_SIZE, size)
    if shape == "Fork":
        return {
            (2, 0),
            (4, 0),
            (6, 0),
            (2, 1),
            (4, 1),
            (6, 1),
            (3, 2),
            (4, 2),
            (5, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            (4, 7),
        }
    if shape == "Loop":
        return {
            (3, 0),
            (4, 0),
            (5, 0),
            (2, 1),
            (6, 1),
            (1, 2),
            (7, 2),
            (1, 3),
            (7, 3),
            (1, 4),
            (7, 4),
            (2, 5),
            (6, 5),
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 7),
        }
    return {
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (5, 1),
        (5, 2),
        (4, 2),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 6),
        (4, 7),
    }


def _scaled_cells(cells: set[tuple[int, int]], old_size: int, new_size: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for x, y in cells:
        out.add(
            (
                min(new_size - 1, round(x * (new_size - 1) / (old_size - 1))),
                min(new_size - 1, round(y * (new_size - 1) / (old_size - 1))),
            )
        )
    return out


def _draw_key_icon(
    frame: np.ndarray,
    px: int,
    py: int,
    shape: str,
    color: str | int,
    rotation: int,
    *,
    size: int = ICON_SIZE,
    plate: int | None = None,
    outline: int = COLOR_WHITE,
) -> None:
    if plate is not None:
        frame[py : py + size, px : px + size] = plate
    cells = _rotate_cells(_key_cells(shape, size), rotation, size)
    key_color = KEY_COLOR_ID[color] if isinstance(color, str) else int(color)
    for x, y in cells:
        if 0 <= py + y < frame.shape[0] and 0 <= px + x < frame.shape[1]:
            frame[py + y, px + x] = key_color
    for x, y in cells:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in cells:
                frame[py + ny, px + nx] = outline


def _draw_box(frame: np.ndarray, px: int, py: int, size: int, fill: int, outline: int = COLOR_WHITE) -> None:
    frame[py : py + size, px : px + size] = fill
    frame[py, px : px + size] = outline
    frame[py + size - 1, px : px + size] = outline
    frame[py : py + size, px] = outline
    frame[py : py + size, px + size - 1] = outline


def _draw_centered_box(
    frame: np.ndarray, cx: int, cy: int, size: int, fill: int, outline: int = COLOR_WHITE
) -> tuple[int, int]:
    px = cx - size // 2
    py = cy - size // 2
    _draw_box(frame, px, py, size, fill, outline)
    return px, py


def _draw_arrow(frame: np.ndarray, px: int, py: int, token: str, color: int = COLOR_WHITE) -> None:
    masks = {
        ">": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)),
        "<": ((3, 1), (2, 1), (1, 0), (1, 1), (1, 2), (0, 1)),
        "^": ((1, 3), (1, 2), (0, 1), (1, 1), (2, 1), (1, 0)),
        "v": ((1, 0), (1, 1), (0, 2), (1, 2), (2, 2), (1, 3)),
    }
    for x, y in masks[token]:
        frame[py + y, px + x] = color


def _draw_big_arrow(frame: np.ndarray, px: int, py: int, token: str, color: int = COLOR_WHITE) -> None:
    masks = {
        ">": ((0, 3), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4), (4, 1), (4, 3), (4, 5), (5, 3)),
        "<": ((5, 3), (4, 3), (3, 3), (2, 2), (2, 3), (2, 4), (1, 1), (1, 3), (1, 5), (0, 3)),
        "^": ((3, 5), (3, 4), (3, 3), (2, 2), (3, 2), (4, 2), (1, 1), (3, 1), (5, 1), (3, 0)),
        "v": ((3, 0), (3, 1), (3, 2), (2, 3), (3, 3), (4, 3), (1, 4), (3, 4), (5, 4), (3, 5)),
    }
    for x, y in masks[token]:
        if 0 <= py + y < frame.shape[0] and 0 <= px + x < frame.shape[1]:
            frame[py + y, px + x] = color


def _draw_rotate_mark(frame: np.ndarray, px: int, py: int, size: int, color: int) -> None:
    coords = ((2, 1), (3, 1), (4, 1), (5, 2), (5, 3), (4, 4), (3, 4), (2, 4), (1, 3), (1, 2), (4, 0), (5, 1), (6, 1))
    for x, y in coords:
        sx = px + round(x * (size - 1) / 7)
        sy = py + round(y * (size - 1) / 7)
        if 0 <= sy < frame.shape[0] and 0 <= sx < frame.shape[1]:
            frame[sy, sx] = color


def _copy_level_data(spec: dict[str, object]) -> dict[str, object]:
    return deepcopy(spec)


def _build_level(spec: dict[str, object]) -> Level:
    frame = _solid(64, 64, COLOR_VOID)
    sprite = Sprite(frame, name="scene", x=0, y=0, layer=0, collidable=False, tags=["scene"])
    return Level(sprites=[sprite], grid_size=(64, 64), name=str(spec["name"]), data=_copy_level_data(spec))


LEVELS = [_build_level(spec) for spec in LEVEL_SPECS]


class LocksmithWardedRoutes(ARCBaseGame):
    def __init__(self) -> None:
        self._scene: Sprite | None = None
        self._route_score = 0
        self._failed = False
        super().__init__(
            GAME_ID,
            LEVELS,
            Camera(0, 0, 64, 64, COLOR_VOID, COLOR_VOID, []),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE],
            win_score=len(LEVELS),
        )

    def on_set_level(self, level: Level) -> None:
        self._scene = level.get_sprites_by_tag("scene")[0]
        self.rows = [str(row) for row in level.get_data("rows")]
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        self.board_x = 2 + ((15 - self.width) * TILE) // 2
        self.player = self._find("P")
        shape, color, rotation = level.get_data("key")
        self.key = KeyState(str(shape), str(color), int(rotation))
        self.energy = int(level.get_data("energy"))
        self.max_energy = int(level.get_data("max_energy"))
        self.locks = {str(k): KeyState(*v) for k, v in dict(level.get_data("locks")).items()}
        self.lock_positions = {self._find(token): token for token in self.locks}
        self.opened: set[tuple[int, int]] = set()
        self.used_pickups: set[tuple[int, int]] = set()
        self.moving = []
        for raw in level.get_data("moving") or ():
            item = dict(raw)
            path = [tuple(pos) for pos in item["path"]]
            self.moving.append(
                {
                    "effect": tuple(item["effect"]),
                    "path": path,
                    "index": path.index(tuple(item["start"])),
                    "dir": int(item["dir"]),
                }
            )
        self.fog = level.get_data("fog")
        self.revealed: set[tuple[int, int]] = set()
        self._failed = False
        self._reveal_near_player()
        self._sync_scene()

    def _find(self, token: str) -> tuple[int, int]:
        for y, row in enumerate(self.rows):
            for x, ch in enumerate(row):
                if ch == token:
                    return (x, y)
        raise ValueError(f"Level is missing token {token!r}.")

    def _tile(self, x: int, y: int) -> str:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "#"
        return self.rows[y][x]

    def _moving_at(self, pos: tuple[int, int]) -> dict[str, object] | None:
        for mover in self.moving:
            if mover["path"][int(mover["index"])] == pos:
                return mover
        return None

    def _key_matches(self, required: KeyState) -> bool:
        return self.key == required

    def _try_enter(self, x: int, y: int) -> bool:
        tile = self._tile(x, y)
        pos = (x, y)
        if tile == "#":
            return False
        lock_token = self.lock_positions.get(pos)
        if lock_token is not None and pos not in self.opened:
            if not self._key_matches(self.locks[lock_token]):
                return False
            self.opened.add(pos)
        self.player = pos
        self._trigger_entered_cell(pos)
        return True

    def _trigger_entered_cell(self, pos: tuple[int, int]) -> None:
        x, y = pos
        tile = self._tile(x, y)
        if tile == "+":
            self._apply_effect(("rotate", 90))
        elif tile in COLOR_TILE and pos not in self.lock_positions:
            self._apply_effect(("color", COLOR_TILE[tile]))
        elif tile in SHAPE_TILE:
            self._apply_effect(("shape", SHAPE_TILE[tile]))
        elif tile == "Y" and pos not in self.used_pickups:
            self.energy = self.max_energy
            self.used_pickups.add(pos)
        mover = self._moving_at(pos)
        if mover is not None:
            self._apply_effect(tuple(mover["effect"]))

    def _apply_effect(self, effect: tuple[object, object]) -> None:
        kind, value = effect
        if kind == "rotate":
            rotation = ROTATIONS[(ROTATIONS.index(self.key.rotation) + 1) % len(ROTATIONS)]
            self.key = KeyState(self.key.shape, self.key.color, rotation)
        elif kind == "color":
            self.key = KeyState(self.key.shape, str(value), self.key.rotation)
        elif kind == "shape":
            self.key = KeyState(str(value), self.key.color, self.key.rotation)

    def _resolve_pushers(self) -> None:
        guard = 0
        while self._tile(*self.player) in PUSH_DELTA and guard < 64:
            guard += 1
            dx, dy = PUSH_DELTA[self._tile(*self.player)]
            nx, ny = self.player[0] + dx, self.player[1] + dy
            if not self._try_enter(nx, ny):
                break

    def _advance_movers(self) -> None:
        for mover in self.moving:
            path = mover["path"]
            index = int(mover["index"])
            direction = int(mover["dir"])
            next_index = index + direction
            if not (0 <= next_index < len(path)):
                direction *= -1
                next_index = index + direction
            mover["dir"] = direction
            mover["index"] = next_index

    def _reveal_near_player(self) -> None:
        if not self.fog:
            return
        px, py = self.player
        for y in range(self.height):
            for x in range(self.width):
                if abs(px - x) + abs(py - y) <= 2:
                    self.revealed.add((x, y))

    def _is_exit(self) -> bool:
        return self._tile(*self.player) == "X"

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == int(GameAction.RESET.value):
            self.on_set_level(self.current_level)
            self.complete_action()
            return

        if self._failed:
            self.complete_action()
            return

        self.energy = max(0, self.energy - 1)

        if action_id in DELTA_BY_ACTION:
            dx, dy = DELTA_BY_ACTION[action_id]
            self._try_enter(self.player[0] + dx, self.player[1] + dy)

        self._resolve_pushers()
        won = self._is_exit()
        if not won:
            self._advance_movers()
            self._reveal_near_player()

        if won:
            self._route_score += 1
            self._sync_scene()
            self.next_level()
            self.complete_action()
            return

        if self.energy <= 0:
            self._failed = True
            self._sync_scene()
            self.lose()
            self.complete_action()
            return

        self._sync_scene()
        self.complete_action()

    def _sync_scene(self) -> None:
        if self._scene is None:
            return
        frame = _solid(64, 64, COLOR_VOID)
        self._draw_hud(frame)
        for y, row in enumerate(self.rows):
            for x, tile in enumerate(row):
                self._draw_tile(frame, x, y, tile)
        for mover in self.moving:
            x, y = mover["path"][int(mover["index"])]
            self._draw_moving(frame, x, y, tuple(mover["effect"]))
        self._draw_avatar(frame)
        self._draw_fog(frame)
        self._scene.pixels = frame

    def _pixel(self, x: int, y: int) -> tuple[int, int]:
        return self.board_x + x * TILE, BOARD_Y + y * TILE

    def _draw_hud(self, frame: np.ndarray) -> None:
        frame[0:10, :] = COLOR_VOID
        _draw_box(frame, 1, 0, 11, COLOR_LOCK, COLOR_WHITE)
        _draw_key_icon(frame, 2, 1, self.key.shape, self.key.color, self.key.rotation, size=ICON_SIZE, plate=None)
        capacity = max(1, self.max_energy)
        fill = min(21, round(21 * self.energy / capacity))
        frame[2:4, 41:62] = COLOR_OPEN
        frame[2:4, 41 : 41 + fill] = COLOR_YELLOW

    def _draw_tile(self, frame: np.ndarray, x: int, y: int, tile: str) -> None:
        px, py = self._pixel(x, y)
        pos = (x, y)
        base = COLOR_WALL if tile == "#" else COLOR_FLOOR
        if tile == ":" or tile.isdigit():
            base = COLOR_RAIL
        frame[py : py + TILE, px : px + TILE] = base
        if tile == ":" or tile.isdigit():
            frame[py + 1, px : px + TILE] = COLOR_LOCK
            frame[py + 2, px : px + TILE] = COLOR_OPEN
        if pos in self.opened:
            frame[py : py + TILE, px : px + TILE] = COLOR_FLOOR
            frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_OPEN
            return
        cx = px + TILE // 2
        cy = py + TILE // 2
        if tile == "X":
            frame[py : py + TILE, px : px + TILE] = COLOR_EXIT
            frame[py, px : px + TILE] = COLOR_WHITE
            frame[py + TILE - 1, px : px + TILE] = COLOR_LOCK
        elif pos in self.lock_positions:
            target = self.locks[self.lock_positions[pos]]
            if self._tile(x + 1, y) == "X":
                cx -= 2
            elif self._tile(x - 1, y) == "X":
                cx += 2
            if self._tile(x, y + 1) == "X":
                cy -= 2
            elif self._tile(x, y - 1) == "X":
                cy += 2
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, COLOR_LOCK, COLOR_WHITE)
            _draw_key_icon(frame, ox, oy, target.shape, target.color, target.rotation, size=TOKEN_SIZE, plate=None)
        elif tile == "+":
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, COLOR_WHITE, COLOR_LOCK)
            _draw_rotate_mark(frame, ox, oy, TOKEN_SIZE, COLOR_LOCK)
        elif tile in COLOR_TILE:
            frame[py : py + TILE, px : px + TILE] = KEY_COLOR_ID[COLOR_TILE[tile]]
            frame[py, px : px + TILE] = COLOR_WHITE
            frame[py + TILE - 1, px : px + TILE] = COLOR_WHITE
            frame[py + 1, px + 1 : px + 3] = COLOR_LOCK
            frame[py + 2, px + 1 : px + 3] = COLOR_WHITE
        elif tile in SHAPE_TILE:
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, COLOR_OPEN, COLOR_LOCK)
            _draw_key_icon(
                frame, ox, oy, SHAPE_TILE[tile], COLOR_WHITE, 0, size=TOKEN_SIZE, plate=None, outline=COLOR_LOCK
            )
        elif tile == "Y" and pos not in self.used_pickups:
            frame[py : py + TILE, px : px + TILE] = COLOR_FLOOR
            frame[py, px + 1 : px + 3] = COLOR_YELLOW
            frame[py + 1 : py + 3, px : px + TILE] = COLOR_YELLOW
            frame[py + 3, px + 1 : px + 3] = COLOR_YELLOW
            frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_WHITE
        elif tile in PUSH_DELTA:
            frame[py + 1 : py + 3, px : px + TILE] = COLOR_WHITE
            _draw_arrow(frame, px, py, tile, COLOR_LOCK)

    def _draw_moving(self, frame: np.ndarray, x: int, y: int, effect: tuple[object, object]) -> None:
        px, py = self._pixel(x, y)
        cx = px + TILE // 2
        cy = py + TILE // 2
        kind, value = effect
        if kind == "color":
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, KEY_COLOR_ID[str(value)], COLOR_WHITE)
            frame[oy + 2 : oy + 6, ox + 2 : ox + 6] = KEY_COLOR_ID[str(value)]
            frame[oy + 3, ox + 3 : ox + 5] = COLOR_WHITE
        elif kind == "shape":
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, COLOR_OPEN, COLOR_WHITE)
            _draw_key_icon(frame, ox, oy, str(value), COLOR_WHITE, 0, size=TOKEN_SIZE, plate=None, outline=COLOR_LOCK)
        elif kind == "rotate":
            ox, oy = _draw_centered_box(frame, cx, cy, TOKEN_SIZE, COLOR_WHITE, COLOR_LOCK)
            _draw_rotate_mark(frame, ox, oy, TOKEN_SIZE, COLOR_LOCK)

    def _draw_avatar(self, frame: np.ndarray) -> None:
        px, py = self._pixel(*self.player)
        frame[py : py + TILE, px : px + TILE] = COLOR_AVATAR
        frame[py, px] = COLOR_WHITE
        frame[py, px + TILE - 1] = COLOR_WHITE
        frame[py + TILE - 1, px] = COLOR_LOCK
        frame[py + TILE - 1, px + TILE - 1] = COLOR_LOCK
        frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_HUD

    def _draw_fog(self, frame: np.ndarray) -> None:
        if not self.fog:
            return
        fog = dict(self.fog)
        for y in range(int(fog["y0"]), int(fog["y1"]) + 1):
            for x in range(int(fog["x0"]), int(fog["x1"]) + 1):
                if self._tile(x, y) == "#" or (x, y) in self.revealed:
                    continue
                px, py = self._pixel(x, y)
                if self._tile(x, y) == "X":
                    frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_EXIT
                    continue
                for yy in range(TILE):
                    for xx in range(TILE):
                        frame[py + yy, px + xx] = COLOR_OPEN if (xx + yy) % 2 == 0 else COLOR_FOG

    def _get_hidden_state(self) -> np.ndarray:
        return np.array(
            [
                int(self.level_index),
                self.player[0],
                self.player[1],
                ("Hook", "Fork", "Loop").index(self.key.shape),
                ("red", "blue", "green", "magenta").index(self.key.color),
                ROTATIONS.index(self.key.rotation),
                self.energy,
                len(self.opened),
                int(self._failed),
            ],
            dtype=np.int16,
        )


class LocksmithWardedRoutes0001(LocksmithWardedRoutes):
    pass
