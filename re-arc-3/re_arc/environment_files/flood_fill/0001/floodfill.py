from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "flood_fill-0001"

BOARD_W = 12
BOARD_H = 10
CELL = 4
PUZZLE_X = 8
PUZZLE_Y = 10
PUZZLE_X_MAX = PUZZLE_X + BOARD_W * CELL - 1
PUZZLE_Y_MAX = PUZZLE_Y + BOARD_H * CELL - 1
FRAME_X0 = 7
FRAME_Y0 = 9
FRAME_X1 = 56
FRAME_Y1 = 50
SWATCH_Y = 54
SWATCH_SIZE = 8
SWATCH_XS = [8, 18, 28, 38, 48]

WHITE = 0
LIGHT_GRAY = 1
GRAY = 2
DARK_GRAY = 3
VERY_DARK_GRAY = 4
RED = 8
BLUE = 9
YELLOW = 11
GREEN = 14
PURPLE = 15

COLOR_BY_TOKEN = {"R": RED, "B": BLUE, "G": GREEN, "Y": YELLOW, "P": PURPLE}
TOKEN_BY_COLOR = {value: key for key, value in COLOR_BY_TOKEN.items()}
SWATCH_COLORS = [RED, BLUE, GREEN, YELLOW, PURPLE]

PLAYING = "playing"
LEVEL_WON = "level_won"
LEVEL_FAILED = "level_failed"
GAME_COMPLETE = "game_complete"


class LevelSpec(NamedTuple):
    enabled_colors: tuple[int, ...]
    default_active_color: int
    budget: int
    rows: tuple[str, ...]


LEVEL_SPECS = [
    LevelSpec(
        enabled_colors=(RED, BLUE),
        default_active_color=RED,
        budget=6,
        rows=(
            "RRRRRRBBBBBB",
            "RRRRRRBBBBBB",
            "RRRRRRBBBBBB",
            "RRRRRRBBBBBB",
            "RRRRRRBBBBBB",
            "BBBBBBRRRRRR",
            "BBBBBBRRRRRR",
            "BBBBBBRRRRRR",
            "BBBBBBRRRRRR",
            "BBBBBBRRRRRR",
        ),
    ),
    LevelSpec(
        enabled_colors=(RED, BLUE, GREEN),
        default_active_color=RED,
        budget=9,
        rows=(
            "BBBRRBBGGRRR",
            "BBBRRBBGGRRR",
            "BBBRRBBGGRRR",
            "GGGGBBBBRRRR",
            "BBBBBBBBBBBB",
            "BBBBBBBBBBBB",
            "RRRRBBBBGGGG",
            "RRRGGBBRRBBB",
            "RRRGGBBRRBBB",
            "RRRGGBBRRBBB",
        ),
    ),
    LevelSpec(
        enabled_colors=(RED, BLUE, GREEN, YELLOW, PURPLE),
        default_active_color=RED,
        budget=15,
        rows=(
            "RRBBYYGGPPRR",
            "RRBBYYGGPPRR",
            "RBBBYYGGPPPR",
            "RBBYYYGGGPRR",
            "YYBBYPPGGGRR",
            "YYBPPPPGGRRR",
            "GYYPPRRBBBRR",
            "GGYPPRRBBBYY",
            "GGPPRRBBYYPP",
            "GGPPRRBBYYPP",
        ),
    ),
]


def _screen_sprite() -> Sprite:
    return Sprite(
        pixels=np.zeros((64, 64), dtype=np.int8), name="screen", x=0, y=0, layer=0, visible=True, collidable=False
    )


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for index, spec in enumerate(LEVEL_SPECS):
        levels.append(
            Level(
                sprites=[_screen_sprite()],
                grid_size=(64, 64),
                data={
                    "enabled_colors": list(spec.enabled_colors),
                    "default_active_color": int(spec.default_active_color),
                    "budget": int(spec.budget),
                    "rows": list(spec.rows),
                },
                name=f"level_{index + 1}",
            )
        )
    return levels


def rows_to_board(rows: tuple[str, ...] | list[str]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(COLOR_BY_TOKEN[ch] for ch in row) for row in rows)


def board_to_rows(board: tuple[tuple[int, ...], ...]) -> tuple[str, ...]:
    return tuple("".join(TOKEN_BY_COLOR[cell] for cell in row) for row in board)


def find_component(board: tuple[tuple[int, ...], ...], start_x: int, start_y: int) -> tuple[tuple[int, int], ...]:
    target = board[start_y][start_x]
    queue = deque([(start_x, start_y)])
    seen = {(start_x, start_y)}
    cells: list[tuple[int, int]] = []
    while queue:
        x, y = queue.popleft()
        cells.append((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < BOARD_W and 0 <= ny < BOARD_H and (nx, ny) not in seen and board[ny][nx] == target:
                seen.add((nx, ny))
                queue.append((nx, ny))
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    return tuple(cells)


def iter_components(board: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    components: list[tuple[tuple[int, int], ...]] = []
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            if (x, y) in seen:
                continue
            component = find_component(board, x, y)
            seen.update(component)
            components.append(component)
    return components


def recolor_component(
    board: tuple[tuple[int, ...], ...], component: tuple[tuple[int, int], ...], color: int
) -> tuple[tuple[int, ...], ...]:
    mutable = [list(row) for row in board]
    for x, y in component:
        mutable[y][x] = color
    return tuple(tuple(row) for row in mutable)


def is_uniform(board: tuple[tuple[int, ...], ...]) -> bool:
    target = board[0][0]
    return all(cell == target for row in board for cell in row)


class FloodFill(ARCBaseGame):
    def __init__(self) -> None:
        self.phase = PLAYING
        self.active_color = RED
        self.remaining_budget = 0
        self.board: tuple[tuple[int, ...], ...] = tuple()
        super().__init__(
            "flood_fill", _build_levels(), Camera(0, 0, 64, 64, WHITE, WHITE), False, len(LEVEL_SPECS), [5, 6]
        )
        self._game_id = GAME_ID

    def on_set_level(self, level: Level) -> None:
        self.phase = PLAYING
        self.active_color = int(level.get_data("default_active_color"))
        self.remaining_budget = int(level.get_data("budget"))
        self.board = rows_to_board(level.get_data("rows"))
        self._render_screen()

    def _current_spec(self) -> LevelSpec:
        return LEVEL_SPECS[self.level_index]

    def _screen(self) -> Sprite:
        return self.current_level.get_sprites_by_name("screen")[0]

    def _click_xy(self) -> tuple[int, int] | None:
        if self.action.id != GameAction.ACTION6:
            return None
        try:
            return int(self.action.data.get("x", -1)), int(self.action.data.get("y", -1))
        except (TypeError, ValueError):
            return None

    def _swatch_at(self, x: int, y: int) -> int | None:
        if not (SWATCH_Y <= y < SWATCH_Y + SWATCH_SIZE):
            return None
        for index, left in enumerate(SWATCH_XS):
            if left <= x < left + SWATCH_SIZE:
                return index
        return None

    def _puzzle_cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        if not (PUZZLE_X <= x <= PUZZLE_X_MAX and PUZZLE_Y <= y <= PUZZLE_Y_MAX):
            return None
        return ((x - PUZZLE_X) // CELL, (y - PUZZLE_Y) // CELL)

    def _reset_level(self) -> None:
        self.level_reset()

    def _advance_after_win(self) -> None:
        if self.is_last_level():
            self.phase = GAME_COMPLETE
            self._render_screen()
            self.next_level()
            return
        self.next_level()

    def _handle_playing_click(self, x: int, y: int) -> None:
        swatch_index = self._swatch_at(x, y)
        spec = self._current_spec()
        if swatch_index is not None:
            color = SWATCH_COLORS[swatch_index]
            if color in spec.enabled_colors:
                self.active_color = color
            return

        cell = self._puzzle_cell_at(x, y)
        if cell is None:
            return

        cx, cy = cell
        component = find_component(self.board, cx, cy)
        current_color = self.board[cy][cx]
        if current_color == self.active_color:
            return

        self.board = recolor_component(self.board, component, self.active_color)
        self.remaining_budget -= 1
        if is_uniform(self.board):
            self.phase = LEVEL_WON
            return
        if self.remaining_budget <= 0:
            self.phase = LEVEL_FAILED
            self.lose()

    def _draw_rect_fill(self, pixels: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        pixels[y0 : y1 + 1, x0 : x1 + 1] = color

    def _draw_rect_border(self, pixels: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        pixels[y0, x0 : x1 + 1] = color
        pixels[y1, x0 : x1 + 1] = color
        pixels[y0 : y1 + 1, x0] = color
        pixels[y0 : y1 + 1, x1] = color

    def _draw_budget(self, pixels: np.ndarray) -> None:
        total = self._current_spec().budget
        total_width = total * 4
        start_x = (64 - total_width) // 2
        for index in range(total):
            left = start_x + index * 4
            fill = YELLOW if index < self.remaining_budget else DARK_GRAY
            self._draw_rect_border(pixels, left, 1, left + 3, 4, VERY_DARK_GRAY)
            self._draw_rect_fill(pixels, left + 1, 2, left + 2, 3, fill)

    def _draw_status_icon(self, pixels: np.ndarray) -> None:
        if self.phase not in {LEVEL_WON, LEVEL_FAILED, GAME_COMPLETE}:
            return
        cx = 31
        cy = 3
        if self.phase in {LEVEL_WON, GAME_COMPLETE}:
            color = GREEN
            points = [
                (0, -2),
                (-1, -1),
                (0, -1),
                (1, -1),
                (-2, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (2, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
                (0, 2),
            ]
        else:
            color = RED
            points = [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (2, -2), (1, -1), (-1, 1), (-2, 2)]
        for dx, dy in points:
            px = cx + dx
            py = cy + dy
            if 0 <= px < 64 and 0 <= py < 64:
                pixels[py, px] = color

    def _draw_swatches(self, pixels: np.ndarray) -> None:
        enabled = set(self._current_spec().enabled_colors)
        for index, color in enumerate(SWATCH_COLORS):
            left = SWATCH_XS[index]
            fill = color if color in enabled else GRAY
            self._draw_rect_border(
                pixels, left, SWATCH_Y, left + SWATCH_SIZE - 1, SWATCH_Y + SWATCH_SIZE - 1, VERY_DARK_GRAY
            )
            self._draw_rect_fill(
                pixels, left + 1, SWATCH_Y + 1, left + SWATCH_SIZE - 2, SWATCH_Y + SWATCH_SIZE - 2, fill
            )
            if color == self.active_color and color in enabled:
                pixels[SWATCH_Y, left] = WHITE
                pixels[SWATCH_Y, left + SWATCH_SIZE - 1] = WHITE
                pixels[SWATCH_Y + SWATCH_SIZE - 1, left] = WHITE
                pixels[SWATCH_Y + SWATCH_SIZE - 1, left + SWATCH_SIZE - 1] = WHITE
                pixels[SWATCH_Y + 1 : SWATCH_Y + SWATCH_SIZE - 1, left + 1] = LIGHT_GRAY
                pixels[SWATCH_Y + 1 : SWATCH_Y + SWATCH_SIZE - 1, left + SWATCH_SIZE - 2] = LIGHT_GRAY
                pixels[SWATCH_Y + 1, left + 1 : left + SWATCH_SIZE - 1] = LIGHT_GRAY
                pixels[SWATCH_Y + SWATCH_SIZE - 2, left + 1 : left + SWATCH_SIZE - 1] = LIGHT_GRAY

    def _render_screen(self) -> None:
        pixels = np.full((64, 64), WHITE, dtype=np.int8)
        frame_color = VERY_DARK_GRAY
        if self.phase == LEVEL_FAILED:
            frame_color = RED
        elif self.phase in {LEVEL_WON, GAME_COMPLETE}:
            frame_color = GREEN

        self._draw_rect_border(pixels, FRAME_X0, FRAME_Y0, FRAME_X1, FRAME_Y1, frame_color)
        self._draw_budget(pixels)
        self._draw_status_icon(pixels)
        self._draw_swatches(pixels)

        for y, row in enumerate(self.board):
            py = PUZZLE_Y + y * CELL
            for x, color in enumerate(row):
                px = PUZZLE_X + x * CELL
                pixels[py : py + CELL, px : px + CELL] = color

        self._screen().pixels = pixels

    def step(self) -> None:
        if self.action.id == GameAction.ACTION5:
            if self.phase == PLAYING:
                self._reset_level()
            elif self.phase == LEVEL_WON:
                self._advance_after_win()
            elif self.phase == LEVEL_FAILED:
                self.lose()
            elif self.phase == GAME_COMPLETE:
                self.full_reset()
            self.complete_action()
            return

        if self.action.id in {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}:
            self._render_screen()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            click_xy = self._click_xy()
            if click_xy is not None:
                x, y = click_xy
                if self.phase == PLAYING:
                    self._handle_playing_click(x, y)
                elif self.phase == LEVEL_WON:
                    self._advance_after_win()
                elif self.phase == LEVEL_FAILED:
                    self.lose()
                elif self.phase == GAME_COMPLETE:
                    self.full_reset()
            self._render_screen()
            self.complete_action()
            return

        self._render_screen()
        self.complete_action()
