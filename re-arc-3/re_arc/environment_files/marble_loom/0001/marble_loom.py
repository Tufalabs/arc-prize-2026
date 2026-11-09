from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64
BOARD_SIZE = 8
CELL = 6
BOARD_X = 8
BOARD_Y = 8
GRAVITY_BUTTON_X = 55
GRAVITY_BUTTON_Y = 0
GRAVITY_BUTTON_SIZE = 9
GRAVITY_INTRO_FLASH_TICKS = 8

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

COLOR_WHITE = 0
COLOR_H_THREAD = 1
COLOR_GRAY = 2
COLOR_AIR = 3
COLOR_BG = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_WARN = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_H_ACCENT = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_WRONG = 13
COLOR_GREEN = 14
COLOR_V_THREAD = 15

COLOR_IDS = {
    "red": COLOR_RED,
    "blue": COLOR_BLUE,
    "yellow": COLOR_YELLOW,
    "orange": COLOR_ORANGE,
    "green": COLOR_GREEN,
    "magenta": COLOR_MAGENTA,
}
COLOR_ORDER = ("red", "blue", "yellow", "orange", "green", "magenta")
DIR_DELTA = {"D": (0, 1), "R": (1, 0), "L": (-1, 0)}


@dataclass(frozen=True)
class MarbleStart:
    color: str
    x: int
    y: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    horizontal: dict[int, str]
    vertical: dict[int, str]
    arrows: dict[tuple[int, int], str]
    solution: tuple[str, ...]
    gravity: dict[str, str] | None = None
    selectable_marbles: bool = True
    side_gravity_colors: tuple[str, ...] = ()
    side_gravity_direction: str = "R"
    initial_selected_color: str | None = None
    step_budget: int | None = None
    cup_directions: dict[str, str] | None = None


@dataclass
class Marble:
    color: str
    x: int
    y: int


def _expanded(*parts: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in parts:
        if "x" in part:
            move, count = part.split("x")
            out.extend([move] * int(count))
        else:
            out.append(part)
    return tuple(out)


LEVEL_SPECS = (
    LevelSpec(
        name="The First Dropping Gap",
        rows=("...R....", "........", "........", "........", "........", "........", "........", "...r...."),
        horizontal={
            0: "##.#####",
            1: "####.###",
            2: "##.#####",
            3: "####.###",
            4: "##.#####",
            5: "####.###",
            6: "##.#####",
        },
        vertical={},
        arrows={},
        solution=("H0R", "H1L", "H2R", "H3L", "H4R", "H5L", "H6R"),
        selectable_marbles=True,
    ),
    LevelSpec(
        name="One Drops, One Waits",
        rows=("...R.B..", "........", "........", "........", "........", "........", "........", "..r....b"),
        horizontal={
            0: "##.###.#",
            1: "##.####.",
            2: ".#.#####",
            3: "#.####.#",
            4: "#.####.#",
            5: "#.####.#",
            6: "#.####.#",
        },
        vertical={},
        arrows={},
        solution=("H0R", "H0L", "H1R", "H1L", "H2L", "H2R", "H3R", "H4R", "H5R", "H6R"),
        initial_selected_color="blue",
    ),
    LevelSpec(
        name="The Side Lock",
        rows=(".R......", "........", "........", "........", "........", "........", "........", "....r..."),
        horizontal={
            0: "##.#####",
            1: "#####.##",
            2: "#####.##",
            3: "#####.##",
            4: "#####.##",
            5: "#####.##",
            6: "#####.##",
        },
        vertical={2: "########", 6: "#######."},
        arrows={},
        solution=("H0R", "H0R", "H0L", "V3D", "H1R", "H1R", "H1L", "H2R", "H3R", "H4R", "H5R", "H6R"),
        step_budget=50,
    ),
    LevelSpec(
        name="Two Side Rescues, One Shared Loom",
        rows=(".R......", "......O.", ".......b", ".B......", ".G......", "........", ".......g", "r....o.."),
        horizontal={0: "##.#####", 1: "#.######", 3: "######.#", 6: "#####.##"},
        vertical={0: "###..###", 1: "#####.##", 3: "####.###", 5: "#####.##", 6: "###.####"},
        arrows={},
        gravity={"blue": "R", "green": "R"},
        step_budget=55,
        solution=_expanded(
            "H3R",
            "H3L",
            "H4R",
            "H5R",
            "H6R",
            "H3Lx4",
            "H4Lx4",
            "H5Lx4",
            "H6Lx4",
            "H0R",
            "H0L",
            "H1R",
            "H2R",
            "H2Rx2",
            "V5D",
            "V5U",
            "H3Rx3",
            "H4Rx3",
            "H5Rx3",
            "H6Rx3",
            "H2R",
            "V3D",
            "V3U",
            "H5L",
            "H6L",
            "H4L",
        ),
    ),
    LevelSpec(
        name="The Left Pocket Is Last",
        rows=(".R......", "......O.", ".M......", "...B.G..", "o......m", "........", "........", "..r.b.g."),
        horizontal={0: "##.#####", 1: "#.######", 3: "######.#", 4: "#####.##", 6: "#####.##"},
        vertical={5: "##.#####", 3: "####.###", 0: "###.####"},
        arrows={},
        side_gravity_colors=("orange", "magenta"),
        side_gravity_direction="L",
        step_budget=60,
        cup_directions={"orange": "L", "magenta": "R"},
        solution=_expanded(
            "H3R",
            "H3L",
            "H4R",
            "H5R",
            "H6R",
            "H3Lx4",
            "H4Lx4",
            "H5Lx4",
            "H6Lx4",
            "H0R",
            "H0L",
            "H1R",
            "H2R",
            "H2Rx2",
            "V5D",
            "V5U",
            "H3Rx3",
            "H4Rx3",
            "H5Rx3",
            "H6Rx3",
            "H2R",
            "V3D",
            "V3U",
            "H5L",
            "H6L",
            "H4L",
            "H3Rx3",
            "H4Lx3",
            "H5Lx3",
            "H6Lx3",
            "V0D",
            "V0U",
            "H3R",
        ),
    ),
    LevelSpec(
        name="The Away-From-Target Rescue",
        rows=(".R......", "......B.", ".M......", ".....G..", ".......O", ".......m", ".o......", "b.r...g."),
        horizontal={0: "##.#####", 1: "#.######", 3: "######.#", 4: "#####.##", 5: "#####.##"},
        vertical={5: "##.#####", 3: "####.###", 0: "########", 6: "#####.##"},
        arrows={},
        side_gravity_colors=("magenta", "orange"),
        side_gravity_direction="L",
        step_budget=65,
        cup_directions={"magenta": "R", "orange": "L"},
        solution=_expanded(
            "H3R",
            "H3L",
            "H4R",
            "H5R",
            "H6R",
            "H3Lx4",
            "H4Lx4",
            "H5Lx4",
            "H6Lx4",
            "H0R",
            "H0L",
            "H1R",
            "H2R",
            "H2Rx2",
            "V5D",
            "V5U",
            "H3Rx3",
            "H4Rx3",
            "H5Rx3",
            "H6Rx3",
            "H2R",
            "V3D",
            "V3U",
            "H5L",
            "H6L",
            "H4L",
            "H3Rx3",
            "H4Lx3",
            "H5Lx3",
            "H6Lx3",
            "V0D",
            "V0U",
            "H3R",
            "H5Rx4",
            "H6Rx2",
            "V6D",
            "V6U",
            "H5Lx3",
            "H5R",
        ),
    ),
)


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


def _solid_sprite() -> Sprite:
    return Sprite(np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BG, dtype=np.int8), name="canvas", layer=-10, collidable=False)


def _handle_sprite(name: str, x: int, y: int, pixels: np.ndarray) -> Sprite:
    return Sprite(pixels, name=name, x=x, y=y, layer=2, collidable=False, tags=["sys_click"])


def _triangle_pixels(direction: str, color: int = COLOR_YELLOW) -> np.ndarray:
    right = np.full((6, 6), -1, dtype=np.int8)
    right[1, 1:4] = color
    right[2, 1:6] = color
    right[3, 1:4] = color
    if direction == "R":
        return right
    if direction == "L":
        return np.fliplr(right)
    if direction == "U":
        return np.rot90(right)
    return np.rot90(right, -1)


def _build_level(spec: LevelSpec) -> Level:
    sprites = [_solid_sprite()]
    for y in spec.horizontal:
        sprites.append(_handle_sprite(f"H{y}L", 1, BOARD_Y + y * CELL + 3, _triangle_pixels("L", COLOR_YELLOW)))
        sprites.append(_handle_sprite(f"H{y}R", 57, BOARD_Y + y * CELL + 3, _triangle_pixels("R", COLOR_YELLOW)))
    for x in spec.vertical:
        sprites.append(_handle_sprite(f"V{x}U", BOARD_X + x * CELL + 3, 1, _triangle_pixels("U", COLOR_YELLOW)))
        sprites.append(_handle_sprite(f"V{x}D", BOARD_X + x * CELL + 3, 57, _triangle_pixels("D", COLOR_YELLOW)))
    if spec.side_gravity_colors:
        sprites.append(
            _handle_sprite(
                "GT",
                GRAVITY_BUTTON_X,
                GRAVITY_BUTTON_Y,
                np.full((GRAVITY_BUTTON_SIZE, GRAVITY_BUTTON_SIZE), COLOR_ORANGE, dtype=np.int8),
            )
        )

    marbles: list[tuple[str, int, int]] = []
    cups: dict[tuple[int, int], str] = {}
    color_by_char = {"R": "red", "B": "blue", "Y": "yellow", "O": "orange", "G": "green", "M": "magenta"}
    for y, row in enumerate(spec.rows):
        if len(row) != BOARD_SIZE:
            raise ValueError(f"{spec.name}: row width must be 8")
        for x, char in enumerate(row):
            if char in color_by_char:
                marbles.append((color_by_char[char], x, y))
            elif char.upper() in color_by_char and char.islower():
                cups[(x, y)] = color_by_char[char.upper()]

    return Level(
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        sprites=sprites,
        name=spec.name,
        data={
            "rows": spec.rows,
            "marbles": tuple(marbles),
            "cups": cups,
            "horizontal": spec.horizontal,
            "vertical": spec.vertical,
            "arrows": spec.arrows,
            "gravity": spec.gravity or {},
            "side_gravity_colors": spec.side_gravity_colors,
            "side_gravity_direction": spec.side_gravity_direction,
            "cup_directions": spec.cup_directions or {},
            "selectable_marbles": spec.selectable_marbles,
            "initial_selected_color": spec.initial_selected_color,
            "solution": spec.solution,
            "step_budget": spec.step_budget or len(spec.solution) * 6,
        },
    )


class LoomDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: MarbleLoom | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BG
        self._draw_board(frame, game)
        self._draw_handles(frame, game)
        self._draw_step_bar(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: MarbleLoom) -> None:
        frame[BOARD_Y : BOARD_Y + BOARD_SIZE * CELL, BOARD_X : BOARD_X + BOARD_SIZE * CELL] = COLOR_AIR
        for (x, y), color in game.cups.items():
            self._draw_cup(frame, x, y, color)
        self._draw_arrows(frame, game)
        self._draw_gates(frame, game)
        for marble in game._visual_marbles():
            self._draw_marble(
                frame, marble["x"], marble["y"], marble["color"], marble.get("dx", 0), marble.get("dy", 0)
            )
            if marble.get("selected", False):
                self._draw_selection_dots(frame, marble["x"], marble["y"], marble.get("dx", 0), marble.get("dy", 0))
        if game.flash_cell is not None:
            x, y = game.flash_cell
            x0, y0 = BOARD_X + x * CELL, BOARD_Y + y * CELL
            frame[y0 : y0 + CELL, x0 : x0 + CELL] = np.where(
                frame[y0 : y0 + CELL, x0 : x0 + CELL] == COLOR_AIR, COLOR_WARN, frame[y0 : y0 + CELL, x0 : x0 + CELL]
            )
        if game.completion_pulse:
            for x, y in game.cups:
                x0, y0 = BOARD_X + x * CELL + 2, BOARD_Y + y * CELL + 1
                frame[y0 : y0 + 2, x0 : x0 + 2] = COLOR_WHITE

    def _draw_arrows(self, frame: np.ndarray, game: MarbleLoom) -> None:
        for (x, y), direction in game.arrows.items():
            cx = BOARD_X + x * CELL
            cy = BOARD_Y + y * CELL
            if direction == "R":
                frame[cy + 2, cx + 1 : cx + 5] = COLOR_YELLOW
                frame[cy + 1, cx + 4] = COLOR_YELLOW
                frame[cy + 3, cx + 4] = COLOR_YELLOW
            elif direction == "L":
                frame[cy + 2, cx + 1 : cx + 5] = COLOR_YELLOW
                frame[cy + 1, cx + 1] = COLOR_YELLOW
                frame[cy + 3, cx + 1] = COLOR_YELLOW

    def _draw_gates(self, frame: np.ndarray, game: MarbleLoom) -> None:
        for y, pattern in game.horizontal.items():
            offset = game.gate_offset if game.anim_gate == ("H", y) else 0
            pattern = (
                game.anim_gate_pattern if game.anim_gate == ("H", y) and game.anim_gate_pattern is not None else pattern
            )
            gy = BOARD_Y + (y + 1) * CELL - 1
            for x, char in enumerate(pattern):
                if char != "#":
                    continue
                sx = BOARD_X + x * CELL + offset
                for wrapped in (sx, sx - BOARD_SIZE * CELL, sx + BOARD_SIZE * CELL):
                    if BOARD_X - CELL <= wrapped < BOARD_X + BOARD_SIZE * CELL:
                        frame[gy : gy + 2, max(BOARD_X, wrapped) : min(BOARD_X + BOARD_SIZE * CELL, wrapped + CELL)] = (
                            COLOR_H_THREAD
                        )
                        frame[gy, max(BOARD_X, wrapped + 1) : min(BOARD_X + BOARD_SIZE * CELL, wrapped + CELL - 1)] = (
                            COLOR_H_ACCENT
                        )
        for x, pattern in game.vertical.items():
            offset = game.gate_offset if game.anim_gate == ("V", x) else 0
            pattern = (
                game.anim_gate_pattern if game.anim_gate == ("V", x) and game.anim_gate_pattern is not None else pattern
            )
            gx = BOARD_X + (x + 1) * CELL - 1
            for y, char in enumerate(pattern):
                if char != "#":
                    continue
                sy = BOARD_Y + y * CELL + offset
                for wrapped in (sy, sy - BOARD_SIZE * CELL, sy + BOARD_SIZE * CELL):
                    if BOARD_Y - CELL <= wrapped < BOARD_Y + BOARD_SIZE * CELL:
                        frame[max(BOARD_Y, wrapped) : min(BOARD_Y + BOARD_SIZE * CELL, wrapped + CELL), gx : gx + 2] = (
                            COLOR_V_THREAD
                        )
                        frame[
                            max(BOARD_Y, wrapped + 1) : min(BOARD_Y + BOARD_SIZE * CELL, wrapped + CELL - 1), gx + 1
                        ] = COLOR_GRAY

    def _draw_handles(self, frame: np.ndarray, game: MarbleLoom) -> None:
        for y in game.horizontal:
            cy = BOARD_Y + y * CELL + 3
            self._draw_handle_triangle(frame, 1, cy, "L", game.flash_handle == f"H{y}L")
            self._draw_handle_triangle(frame, 57, cy, "R", game.flash_handle == f"H{y}R")
        for x in game.vertical:
            cx = BOARD_X + x * CELL + 3
            self._draw_handle_triangle(frame, cx, 1, "U", game.flash_handle == f"V{x}U")
            self._draw_handle_triangle(frame, cx, 57, "D", game.flash_handle == f"V{x}D")
        if game.side_gravity_colors:
            intro_flash = game.gravity_intro_flash_ticks > 0 and game.gravity_intro_flash_ticks % 4 in (1, 2)
            self._draw_side_gravity_button(frame, game.side_gravity_direction, game.flash_handle == "GT" or intro_flash)

    def _draw_handle_triangle(self, frame: np.ndarray, x: int, y: int, direction: str, flash: bool) -> None:
        pixels = _triangle_pixels(direction, COLOR_WARN if flash else COLOR_YELLOW)
        frame[y : y + 6, x : x + 6] = np.where(pixels >= 0, pixels, frame[y : y + 6, x : x + 6])

    def _draw_side_gravity_button(self, frame: np.ndarray, direction: str, flash: bool) -> None:
        x0 = GRAVITY_BUTTON_X
        y0 = GRAVITY_BUTTON_Y
        frame[y0 : y0 + GRAVITY_BUTTON_SIZE, x0 : x0 + GRAVITY_BUTTON_SIZE] = COLOR_BLACK
        frame[y0 + 1 : y0 + 8, x0 + 1 : x0 + 8] = COLOR_ORANGE
        frame[y0 + 2 : y0 + 7, x0 + 2 : x0 + 7] = COLOR_BLACK
        frame[y0 + 3 : y0 + 6, x0 + 3 : x0 + 6] = COLOR_ORANGE
        if flash:
            frame[y0, x0 + 1 : x0 + 8] = COLOR_WHITE
            frame[y0 + 8, x0 + 1 : x0 + 8] = COLOR_WHITE
            frame[y0 + 1 : y0 + 8, x0] = COLOR_WHITE
            frame[y0 + 1 : y0 + 8, x0 + 8] = COLOR_WHITE
        if direction == "R":
            frame[y0 + 4, x0 + 2 : x0 + 7] = COLOR_WHITE
            frame[y0 + 3, x0 + 6] = COLOR_WHITE
            frame[y0 + 5, x0 + 6] = COLOR_WHITE
        else:
            frame[y0 + 4, x0 + 2 : x0 + 7] = COLOR_WHITE
            frame[y0 + 3, x0 + 2] = COLOR_WHITE
            frame[y0 + 5, x0 + 2] = COLOR_WHITE

    def _draw_cup(self, frame: np.ndarray, x: int, y: int, color: str) -> None:
        x0 = BOARD_X + x * CELL
        y0 = BOARD_Y + y * CELL
        c = COLOR_IDS[color]
        direction = self._cup_gravity_direction(color)
        if direction == "R":
            frame[y0 + 1, x0 + 1 : x0 + 5] = c
            frame[y0 + 4, x0 + 1 : x0 + 5] = c
            frame[y0 + 1 : y0 + 5, x0 + 4] = c
        elif direction == "L":
            frame[y0 + 1, x0 + 1 : x0 + 5] = c
            frame[y0 + 4, x0 + 1 : x0 + 5] = c
            frame[y0 + 1 : y0 + 5, x0 + 1] = c
        elif direction == "U":
            frame[y0 + 1 : y0 + 5, x0 + 1] = c
            frame[y0 + 1 : y0 + 5, x0 + 4] = c
            frame[y0 + 1, x0 + 1 : x0 + 5] = c
        else:
            frame[y0 + 1 : y0 + 5, x0 + 1] = c
            frame[y0 + 1 : y0 + 5, x0 + 4] = c
            frame[y0 + 4, x0 + 1 : x0 + 5] = c
        frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = COLOR_AIR

    def _cup_gravity_direction(self, color: str) -> str:
        game = self.game
        if game is None:
            return "D"
        if color in game.cup_directions:
            return game.cup_directions[color]
        if color in game.side_gravity_colors:
            return game.side_gravity_direction
        return game.marble_gravity.get(color, "D")

    def _draw_marble(self, frame: np.ndarray, x: int, y: int, color: str, dx: int, dy: int) -> None:
        x0 = BOARD_X + x * CELL + 1 + dx
        y0 = BOARD_Y + y * CELL + 1 + dy
        c = COLOR_IDS[color]
        for px, py in ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)):
            xx, yy = x0 + px, y0 + py
            if 0 <= xx < VIEW_SIZE and 0 <= yy < VIEW_SIZE:
                frame[yy, xx] = c

        game = self.game
        direction = "D"
        if game is not None and color in game.side_gravity_colors:
            direction = game.side_gravity_direction
        elif game is not None:
            direction = game.marble_gravity.get(color) or game.arrows.get((x, y), "D")
        if direction in ("D", "L", "R", "U"):
            marks = {
                "D": ((1, 2), (2, 3), (3, 2)),
                "L": ((1, 0), (0, 1), (1, 2)),
                "R": ((2, 0), (3, 1), (2, 2)),
                "U": ((1, 1), (2, 0), (3, 1)),
            }[direction]
            for px, py in marks:
                xx, yy = x0 + px, y0 + py
                if 0 <= xx < VIEW_SIZE and 0 <= yy < VIEW_SIZE:
                    frame[yy, xx] = COLOR_WHITE

    def _draw_selection_dots(self, frame: np.ndarray, x: int, y: int, dx: int, dy: int) -> None:
        x0 = BOARD_X + x * CELL + 1 + dx
        y0 = BOARD_Y + y * CELL + 1 + dy
        for px, py in ((x0, y0), (x0 + 3, y0), (x0, y0 + 3), (x0 + 3, y0 + 3)):
            if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                frame[py, px] = COLOR_YELLOW

    def _draw_step_bar(self, frame: np.ndarray, game: MarbleLoom) -> None:
        frame[63:64, 8:56] = COLOR_BLACK
        filled = int(48 * game.remaining_steps / max(1, game.step_budget))
        if filled > 0:
            frame[63:64, 8 : 8 + filled] = COLOR_GREEN if filled > 12 else COLOR_YELLOW


class MarbleLoom(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = LoomDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "marble_loom-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BG, COLOR_BG, [self._view]),
            False,
            len(levels),
            [ACTION_CLICK],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.horizontal = dict(level.get_data("horizontal"))
        self.vertical = dict(level.get_data("vertical"))
        self.arrows = dict(level.get_data("arrows"))
        self.marble_gravity = dict(level.get_data("gravity") or {})
        self.side_gravity_colors = tuple(level.get_data("side_gravity_colors") or ())
        self.side_gravity_direction = str(level.get_data("side_gravity_direction") or "R")
        self.cup_directions = dict(level.get_data("cup_directions") or {})
        self.selectable_marbles = bool(level.get_data("selectable_marbles") or False)
        self.cups = dict(level.get_data("cups"))
        self.marbles = [Marble(color, x, y) for color, x, y in level.get_data("marbles")]
        self.step_budget = int(level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.animation_frames: list[dict[str, object]] = []
        self.anim_gate: tuple[str, int] | None = None
        self.anim_gate_pattern: str | None = None
        self.gate_offset = 0
        self.marble_offsets: dict[int, tuple[int, int]] = {}
        self.flash_handle: str | None = None
        self.flash_cell: tuple[int, int] | None = None
        self.gravity_intro_flash_ticks = GRAVITY_INTRO_FLASH_TICKS if self.side_gravity_colors else 0
        self.selected_marble_idx: int | None = None
        initial_selected_color = level.get_data("initial_selected_color")
        if self.selectable_marbles and initial_selected_color:
            self.selected_marble_idx = next(
                (idx for idx, marble in enumerate(self.marbles) if marble.color == initial_selected_color), None
            )
        self.completion_pulse = False
        self._pending_win = False
        self._sync_click_sprites()

    def _sync_click_sprites(self) -> None:
        active = {f"H{y}{side}" for y in self.horizontal for side in ("L", "R")}
        active.update({f"V{x}{side}" for x in self.vertical for side in ("U", "D")})
        if self.side_gravity_colors:
            active.add("GT")
        for sprite in self.current_level.get_sprites_by_tag("sys_click"):
            sprite.set_visible(sprite.name in active)

    def _visual_marbles(self) -> list[dict[str, object]]:
        out = []
        for idx, marble in enumerate(self.marbles):
            dx, dy = self.marble_offsets.get(idx, (0, 0))
            out.append(
                {
                    "x": marble.x,
                    "y": marble.y,
                    "color": marble.color,
                    "dx": dx,
                    "dy": dy,
                    "selected": self.selectable_marbles and idx == self.selected_marble_idx,
                }
            )
        return out

    def _handle_at(self, px: int, py: int) -> str | None:
        if (
            self.side_gravity_colors
            and GRAVITY_BUTTON_X <= px < GRAVITY_BUTTON_X + GRAVITY_BUTTON_SIZE
            and GRAVITY_BUTTON_Y <= py < GRAVITY_BUTTON_Y + GRAVITY_BUTTON_SIZE
        ):
            return "GT"
        for y in self.horizontal:
            cy = BOARD_Y + y * CELL + 3
            if 1 <= px <= 6 and cy <= py <= cy + 5:
                return f"H{y}L"
            if 57 <= px <= 62 and cy <= py <= cy + 5:
                return f"H{y}R"
        for x in self.vertical:
            cx = BOARD_X + x * CELL + 3
            if cx <= px <= cx + 5 and 1 <= py <= 6:
                return f"V{x}U"
            if cx <= px <= cx + 5 and 57 <= py <= 62:
                return f"V{x}D"
        return None

    def _marble_at(self, px: int, py: int) -> int | None:
        if not (BOARD_X <= px < BOARD_X + BOARD_SIZE * CELL and BOARD_Y <= py < BOARD_Y + BOARD_SIZE * CELL):
            return None
        x = (px - BOARD_X) // CELL
        y = (py - BOARD_Y) // CELL
        for idx in range(len(self.marbles) - 1, -1, -1):
            marble = self.marbles[idx]
            if marble.x == x and marble.y == y:
                return idx
        return None

    def _direction_at(self, x: int, y: int) -> str | None:
        if (x, y) in self.cups:
            return None
        return self.arrows.get((x, y), "D")

    def _direction_for(self, marble: Marble) -> str | None:
        if self.cups.get((marble.x, marble.y)) == marble.color:
            return None
        if marble.color in self.side_gravity_colors:
            return self.side_gravity_direction
        if (marble.x, marble.y) in self.cups:
            return None
        return self.marble_gravity.get(marble.color) or self.arrows.get((marble.x, marble.y), "D")

    def _toggle_side_gravity(self) -> None:
        old_positions = [(m.x, m.y) for m in self.marbles]
        self.side_gravity_direction = "L" if self.side_gravity_direction == "R" else "R"
        self.flash_handle = "GT"
        settle_moves = self._settle()
        self._queue_settle_animation(old_positions, settle_moves)

    def _gate_blocks(self, direction: str, x: int, y: int) -> bool:
        if direction == "D":
            return self.horizontal.get(y, "........")[x] == "#"
        if direction == "R":
            return self.vertical.get(x, "........")[y] == "#"
        if direction == "L":
            return self.vertical.get(x - 1, "........")[y] == "#"
        return True

    def _vertical_blocks_slide(self, x: int, y: int, dx: int) -> bool:
        if dx > 0:
            return self.vertical.get(x, "........")[y] == "#"
        if dx < 0:
            return self.vertical.get(x - 1, "........")[y] == "#"
        return False

    def _horizontal_blocks_slide(self, x: int, y: int, dy: int) -> bool:
        if dy > 0:
            return self.horizontal.get(y, "........")[x] == "#"
        if dy < 0:
            return self.horizontal.get(y - 1, "........")[x] == "#"
        return False

    def _can_move(self, marble: Marble, occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
        direction = self._direction_for(marble)
        if direction is None or self._gate_blocks(direction, marble.x, marble.y):
            return None
        dx, dy = DIR_DELTA[direction]
        nx, ny = marble.x + dx, marble.y + dy
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or (nx, ny) in occupied:
            return None
        return nx, ny

    def _settle(self) -> list[list[tuple[int, tuple[int, int], tuple[int, int]]]]:
        microticks: list[list[tuple[int, tuple[int, int], tuple[int, int]]]] = []
        while True:
            occupied = {(m.x, m.y) for m in self.marbles}
            moves: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
            order = sorted(
                range(len(self.marbles)),
                key=lambda idx: (-self.marbles[idx].y, self.marbles[idx].x, COLOR_ORDER.index(self.marbles[idx].color)),
            )
            for idx in order:
                marble = self.marbles[idx]
                target = self._can_move(marble, occupied)
                if target is None:
                    continue
                old = (marble.x, marble.y)
                occupied.remove(old)
                occupied.add(target)
                marble.x, marble.y = target
                moves.append((idx, old, target))
            if not moves:
                return microticks
            microticks.append(moves)

    def _is_win(self) -> bool:
        if len(self.marbles) != len(self.cups):
            return False
        seen: set[tuple[int, int]] = set()
        for marble in self.marbles:
            cell = (marble.x, marble.y)
            if cell in seen or self.cups.get(cell) != marble.color:
                return False
            seen.add(cell)
        return seen == set(self.cups)

    def _pressed_on_shifted_gate(self, marble: Marble, gate_kind: str, gate_index: int, old_pattern: str) -> bool:
        direction = self._direction_for(marble)
        if gate_kind == "H":
            return direction == "D" and marble.y == gate_index and old_pattern[marble.x] == "#"
        if direction == "R":
            return marble.x == gate_index and old_pattern[marble.y] == "#"
        if direction == "L":
            return marble.x - 1 == gate_index and old_pattern[marble.y] == "#"
        return False

    def _marble_is_on_slider(self, marble: Marble, gate_kind: str, gate_index: int) -> bool:
        direction = self._direction_for(marble)
        if gate_kind == "H":
            return direction == "D" and marble.y == gate_index
        if direction == "R":
            return marble.x == gate_index
        if direction == "L":
            return marble.x - 1 == gate_index
        return False

    def _handle_moves_selected_slider(self, handle: str) -> bool:
        if self.selected_marble_idx is None:
            return False
        marble = self.marbles[self.selected_marble_idx]
        gate_kind = handle[0]
        gate_index = int(handle[1])
        return self._marble_is_on_slider(marble, gate_kind, gate_index)

    def _try_shift(self, handle: str) -> bool:
        gate_kind = handle[0]
        gate_index = int(handle[1])
        direction = handle[2]
        if gate_kind == "H":
            old_pattern = self.horizontal[gate_index]
            new_pattern = old_pattern[-1] + old_pattern[:-1] if direction == "R" else old_pattern[1:] + old_pattern[0]
            dx, dy = (1, 0) if direction == "R" else (-1, 0)
        else:
            old_pattern = self.vertical[gate_index]
            new_pattern = old_pattern[-1] + old_pattern[:-1] if direction == "D" else old_pattern[1:] + old_pattern[0]
            dx, dy = (0, 1) if direction == "D" else (0, -1)

        carried: dict[int, tuple[int, int]] = {}
        for idx, marble in enumerate(self.marbles):
            selected_marble = self.selectable_marbles and idx == self.selected_marble_idx
            if self.selectable_marbles:
                if not selected_marble:
                    continue
                if not self._marble_is_on_slider(marble, gate_kind, gate_index):
                    continue
            else:
                if not self._pressed_on_shifted_gate(marble, gate_kind, gate_index, old_pattern):
                    continue
                contact_index = marble.x if gate_kind == "H" else marble.y
                if new_pattern[contact_index] == ".":
                    continue
            target = (marble.x + dx, marble.y + dy)
            if not (0 <= target[0] < BOARD_SIZE and 0 <= target[1] < BOARD_SIZE):
                if selected_marble:
                    target = (target[0] % BOARD_SIZE, target[1] % BOARD_SIZE)
                else:
                    continue
            if gate_kind == "H" and self._vertical_blocks_slide(marble.x, marble.y, dx):
                continue
            if gate_kind == "V" and self._horizontal_blocks_slide(marble.x, marble.y, dy):
                continue
            carried[idx] = target

        original_occupied = {(m.x, m.y) for m in self.marbles}
        carried_sources = {(self.marbles[idx].x, self.marbles[idx].y) for idx in carried}
        stationary = original_occupied - carried_sources
        carried_targets: set[tuple[int, int]] = set()
        for target in carried.values():
            if target in stationary or target in carried_targets:
                return False
            carried_targets.add(target)

        old_positions = [(m.x, m.y) for m in self.marbles]
        if gate_kind == "H":
            self.horizontal[gate_index] = new_pattern
        else:
            self.vertical[gate_index] = new_pattern
        for idx, (nx, ny) in carried.items():
            self.marbles[idx].x = nx
            self.marbles[idx].y = ny
        settle_moves = self._settle()
        self._queue_valid_animation(
            gate_kind, gate_index, direction, old_pattern, new_pattern, old_positions, carried, settle_moves
        )
        return True

    def _queue_settle_animation(
        self,
        old_positions: list[tuple[int, int]],
        settle_moves: list[list[tuple[int, tuple[int, int], tuple[int, int]]]],
    ) -> None:
        final_positions = [(m.x, m.y) for m in self.marbles]
        self.animation_frames = []
        current_positions = list(old_positions)
        for microtick in settle_moves:
            next_positions = list(current_positions)
            for idx, _old, new in microtick:
                next_positions[idx] = new
            self.animation_frames.append({"positions": next_positions, "offsets": {}})
            current_positions = next_positions
        if self.animation_frames:
            self.animation_frames.append({"positions": final_positions, "offsets": {}, "pulse": self._is_win()})
            self._apply_frame(self.animation_frames.pop(0))

    def _queue_valid_animation(
        self,
        gate_kind: str,
        gate_index: int,
        direction: str,
        old_pattern: str,
        new_pattern: str,
        old_positions: list[tuple[int, int]],
        carried: dict[int, tuple[int, int]],
        settle_moves: list[list[tuple[int, tuple[int, int], tuple[int, int]]]],
    ) -> None:
        final_positions = [(m.x, m.y) for m in self.marbles]
        self.animation_frames = []
        gate_dir = {"R": 1, "L": -1, "D": 1, "U": -1}[direction]
        for offset in (0, gate_dir * 2, gate_dir * 4):
            self.animation_frames.append(
                {
                    "positions": old_positions,
                    "gate": (gate_kind, gate_index),
                    "gate_offset": offset,
                    "gate_pattern": old_pattern,
                    "offsets": {
                        idx: (offset if gate_kind == "H" else 0, offset if gate_kind == "V" else 0) for idx in carried
                    },
                }
            )
        self.animation_frames.append(
            {
                "positions": [(carried.get(idx) or old_positions[idx]) for idx in range(len(old_positions))],
                "gate": (gate_kind, gate_index),
                "gate_offset": 0,
                "gate_pattern": new_pattern,
                "offsets": {},
            }
        )

        current_positions = [(carried.get(idx) or old_positions[idx]) for idx in range(len(old_positions))]
        for microtick in settle_moves:
            next_positions = list(current_positions)
            for idx, _old, new in microtick:
                next_positions[idx] = new
            self.animation_frames.append({"positions": next_positions, "offsets": {}})
            current_positions = next_positions
        self.animation_frames.append({"positions": final_positions, "offsets": {}, "pulse": self._is_win()})
        # The logical move has already been resolved before frames are queued.
        # Prime the visual state now so the first post-click render starts at
        # the beginning of the motion instead of briefly showing the final state.
        self._apply_frame(self.animation_frames.pop(0))

    def _apply_frame(self, frame_data: dict[str, object]) -> None:
        positions = frame_data.get("positions")
        if positions is not None:
            for marble, (x, y) in zip(self.marbles, positions, strict=True):
                marble.x = int(x)
                marble.y = int(y)
        self.anim_gate = frame_data.get("gate")  # type: ignore[assignment]
        self.anim_gate_pattern = frame_data.get("gate_pattern")  # type: ignore[assignment]
        self.gate_offset = int(frame_data.get("gate_offset", 0))
        self.marble_offsets = dict(frame_data.get("offsets", {}))  # type: ignore[arg-type]
        self.completion_pulse = bool(frame_data.get("pulse", False))

    def _play_animation_frame(self) -> None:
        frame_data = self.animation_frames.pop(0)
        self._apply_frame(frame_data)
        if self.animation_frames:
            return
        self.anim_gate = None
        self.anim_gate_pattern = None
        self.gate_offset = 0
        self.marble_offsets = {}
        self.flash_handle = None
        self.flash_cell = None
        if self.selected_marble_idx is not None and self.selected_marble_idx >= len(self.marbles):
            self.selected_marble_idx = None
        if self._is_win():
            self.next_level()
        self.complete_action()

    def _invalid_feedback(self, handle: str | None, px: int, py: int) -> None:
        self.flash_handle = handle
        if BOARD_X <= px < BOARD_X + BOARD_SIZE * CELL and BOARD_Y <= py < BOARD_Y + BOARD_SIZE * CELL:
            self.flash_cell = ((px - BOARD_X) // CELL, (py - BOARD_Y) // CELL)
        self.animation_frames = [{"positions": [(m.x, m.y) for m in self.marbles], "offsets": {}}]

    def _spend_step(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return True
        self.lose()
        self.complete_action()
        return False

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        if self.gravity_intro_flash_ticks > 0:
            self.gravity_intro_flash_ticks -= 1
        self.flash_handle = None
        self.flash_cell = None
        self.completion_pulse = False
        action = _action_id(self.action.id)
        if action == _action_id(GameAction.RESET):
            self.complete_action()
            return
        if action != ACTION_CLICK:
            self.complete_action()
            return

        data = self.action.data or {}
        px = int(data.get("x", 0))
        py = int(data.get("y", 0))
        handle = self._handle_at(px, py)
        if handle is None:
            if self.selectable_marbles:
                clicked_marble_idx = self._marble_at(px, py)
                if self.selected_marble_idx is None and clicked_marble_idx is not None:
                    self.selected_marble_idx = clicked_marble_idx
                else:
                    self.selected_marble_idx = None
                self.complete_action()
            else:
                self._invalid_feedback(None, px, py)
            return
        if handle == "GT":
            self._toggle_side_gravity()
            if not self.animation_frames:
                self.complete_action()
            return
        if (
            self.selectable_marbles
            and self.selected_marble_idx is not None
            and not self._handle_moves_selected_slider(handle)
        ):
            self.selected_marble_idx = None
        if not self._spend_step():
            return
        if not self._try_shift(handle):
            self._invalid_feedback(handle, px, py)
            return
        if not self.animation_frames:
            self.complete_action()
