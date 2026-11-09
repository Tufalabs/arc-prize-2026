from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 10
CELL_SIZE = 4
BOARD_ORIGIN = (2, 2)

COLOR_WHITE = 0
COLOR_LIGHT_GRAY = 1
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_LIGHT_MAGENTA = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14

ACTION_TO_DELTA = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

BOT_A_PIXELS = np.array(
    [
        [COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_WHITE],
        [COLOR_BLUE, COLOR_LIGHT_BLUE, COLOR_LIGHT_BLUE, COLOR_BLUE],
        [COLOR_BLUE, COLOR_LIGHT_BLUE, COLOR_LIGHT_BLUE, COLOR_BLUE],
        [COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_WHITE],
    ],
    dtype=np.int8,
)
BOT_B_PIXELS = np.array(
    [
        [COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_WHITE],
        [COLOR_MAGENTA, COLOR_LIGHT_MAGENTA, COLOR_LIGHT_MAGENTA, COLOR_MAGENTA],
        [COLOR_MAGENTA, COLOR_LIGHT_MAGENTA, COLOR_LIGHT_MAGENTA, COLOR_MAGENTA],
        [COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_WHITE],
    ],
    dtype=np.int8,
)
COMBINED_PIXELS = np.array(
    [
        [COLOR_BLUE, COLOR_BLUE, COLOR_MAGENTA, COLOR_MAGENTA],
        [COLOR_LIGHT_BLUE, COLOR_LIGHT_BLUE, COLOR_LIGHT_MAGENTA, COLOR_LIGHT_MAGENTA],
        [COLOR_LIGHT_BLUE, COLOR_LIGHT_BLUE, COLOR_LIGHT_MAGENTA, COLOR_LIGHT_MAGENTA],
        [COLOR_BLUE, COLOR_BLUE, COLOR_MAGENTA, COLOR_MAGENTA],
    ],
    dtype=np.int8,
)
BEACON_PIXELS = np.array(
    [
        [COLOR_WHITE, COLOR_YELLOW, COLOR_YELLOW, COLOR_WHITE],
        [COLOR_YELLOW, COLOR_ORANGE, COLOR_ORANGE, COLOR_YELLOW],
        [COLOR_YELLOW, COLOR_ORANGE, COLOR_ORANGE, COLOR_YELLOW],
        [COLOR_WHITE, COLOR_YELLOW, COLOR_YELLOW, COLOR_WHITE],
    ],
    dtype=np.int8,
)
WALL_PIXELS = np.array(
    [
        [COLOR_BLACK, COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_BLACK],
        [COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_VERY_DARK],
        [COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_VERY_DARK],
        [COLOR_BLACK, COLOR_VERY_DARK, COLOR_VERY_DARK, COLOR_BLACK],
    ],
    dtype=np.int8,
)

LEVEL_SPECS = (
    {"name": "Level 1", "beacon": (0, 4), "bot_a": (2, 4), "bot_b": (7, 4), "walls": (), "budget": 10},
    {"name": "Level 2", "beacon": (0, 0), "bot_a": (2, 2), "bot_b": (7, 5), "walls": (), "budget": 16},
    {
        "name": "Level 3",
        "beacon": (5, 5),
        "bot_a": (7, 5),
        "bot_b": (8, 8),
        "walls": ((4, 4), (5, 4), (4, 5), (5, 6)),
        "budget": 10,
    },
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


class CommandTwins(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._move_budget = 0
        self._bot_a = (0, 0)
        self._bot_b = (0, 0)
        self._beacon = (0, 0)
        self._walls: set[tuple[int, int]] = set()
        self._failed = False
        self._final_win = False
        self._canvas_sprite: Sprite | None = None

        levels = [self._build_level(idx, spec) for idx, spec in enumerate(LEVEL_SPECS)]
        super().__init__(
            game_id="command_twins-0001",
            levels=levels,
            camera=Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_WHITE),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _build_level(self, index: int, spec: dict[str, object]) -> Level:
        sprite = Sprite(
            _solid(GRID_SIZE, GRID_SIZE, COLOR_WHITE),
            name="canvas",
            x=0,
            y=0,
            layer=0,
            tags=["canvas"],
            collidable=False,
        )
        return Level(
            name=str(spec["name"]),
            grid_size=(GRID_SIZE, GRID_SIZE),
            sprites=[sprite],
            data={
                "level_index": index,
                "beacon": tuple(spec["beacon"]),
                "bot_a": tuple(spec["bot_a"]),
                "bot_b": tuple(spec["bot_b"]),
                "walls": [list(wall) for wall in spec["walls"]],
                "budget": int(spec["budget"]),
            },
        )

    def on_set_level(self, level: Level) -> None:
        self._beacon = tuple(level.get_data("beacon"))
        self._bot_a = tuple(level.get_data("bot_a"))
        self._bot_b = tuple(level.get_data("bot_b"))
        self._walls = {tuple(wall) for wall in level.get_data("walls")}
        self._move_budget = int(level.get_data("budget"))
        self._failed = False
        self._final_win = False
        canvases = self.current_level.get_sprites_by_name("canvas")
        self._canvas_sprite = canvases[0] if canvases else None
        self._render()

    def _cell_bounds(self, x: int, y: int) -> tuple[int, int, int, int]:
        px = BOARD_ORIGIN[0] + (CELL_SIZE * int(x))
        py = BOARD_ORIGIN[1] + (CELL_SIZE * int(y))
        return px, py, px + CELL_SIZE, py + CELL_SIZE

    def _blit(self, canvas: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
        height, width = pixels.shape
        canvas[y : y + height, x : x + width] = pixels

    def _draw_pips(self, canvas: np.ndarray) -> None:
        max_budget = int(self.current_level.get_data("budget"))
        for idx in range(max_budget):
            cx = idx % 4
            cy = idx // 4
            x = 46 + (4 * cx)
            y = 40 + (4 * cy)
            color = COLOR_GREEN if idx < self._move_budget else COLOR_MAROON
            canvas[y : y + 3, x : x + 3] = np.int8(color)

    def _render(self) -> None:
        if self._canvas_sprite is None:
            return

        canvas = _solid(GRID_SIZE, GRID_SIZE, COLOR_WHITE)
        border_color = COLOR_DARK_GRAY
        if self._failed:
            border_color = COLOR_RED
            canvas[0, :] = np.int8(COLOR_RED)
            canvas[-1, :] = np.int8(COLOR_RED)
            canvas[:, 0] = np.int8(COLOR_RED)
            canvas[:, -1] = np.int8(COLOR_RED)
        elif self._final_win:
            border_color = COLOR_GREEN
            canvas[0, :] = np.int8(COLOR_GREEN)
            canvas[-1, :] = np.int8(COLOR_GREEN)
            canvas[:, 0] = np.int8(COLOR_GREEN)
            canvas[:, -1] = np.int8(COLOR_GREEN)

        canvas[1:43, 1] = np.int8(border_color)
        canvas[1:43, 42] = np.int8(border_color)
        canvas[1, 1:43] = np.int8(border_color)
        canvas[42, 1:43] = np.int8(border_color)

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                x0, y0, x1, y1 = self._cell_bounds(x, y)
                canvas[y0:y1, x0:x1] = np.int8(COLOR_WHITE if (x + y) % 2 == 0 else COLOR_LIGHT_GRAY)

        bx, by = self._beacon
        x0, y0, _, _ = self._cell_bounds(bx, by)
        self._blit(canvas, BEACON_PIXELS, x0, y0)

        for wall_x, wall_y in self._walls:
            x0, y0, _, _ = self._cell_bounds(wall_x, wall_y)
            self._blit(canvas, WALL_PIXELS, x0, y0)

        if self._bot_a == self._bot_b:
            x0, y0, _, _ = self._cell_bounds(*self._bot_a)
            self._blit(canvas, COMBINED_PIXELS, x0, y0)
        else:
            ax, ay, _, _ = self._cell_bounds(*self._bot_a)
            bx, by, _, _ = self._cell_bounds(*self._bot_b)
            self._blit(canvas, BOT_A_PIXELS, ax, ay)
            self._blit(canvas, BOT_B_PIXELS, bx, by)

        self._draw_pips(canvas)
        self._canvas_sprite.pixels = canvas

    def _try_move(self, position: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        next_x = int(position[0]) + int(delta[0])
        next_y = int(position[1]) + int(delta[1])
        candidate = (next_x, next_y)
        if not (0 <= next_x < BOARD_SIZE and 0 <= next_y < BOARD_SIZE):
            return position
        if candidate in self._walls:
            return position
        return candidate

    def step(self) -> None:
        action_id = int(self.action.id.value)
        delta = ACTION_TO_DELTA.get(action_id)
        if delta is None:
            self.complete_action()
            return

        self._move_budget -= 1
        self._bot_a = self._try_move(self._bot_a, delta)
        self._bot_b = self._try_move(self._bot_b, delta)

        solved = self._bot_a == self._beacon and self._bot_b == self._beacon
        if solved:
            if self.is_last_level():
                self._final_win = True
            self._render()
            self.next_level()
            self.complete_action()
            return

        if self._move_budget == 0:
            self._failed = True
            self._render()
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()
