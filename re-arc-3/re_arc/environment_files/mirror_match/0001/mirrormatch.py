from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "mirror_match-0001"

GRID_SIZE = 64
HUD_HEIGHT = 10
BOARD_TOP = 14
BOARD_HEIGHT = 40
BOARD_SLOT = 5
CELL_FILL = 4
BOARD_ROWS = 8
BOARD_COLS = 6
LEFT_ORIGIN = (1, 14)
RIGHT_ORIGIN = (33, 14)

COLOR_WHITE = 0
COLOR_PANEL = 1
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK_GRAY = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

PLAY = "PLAY"
LEVEL_CLEAR = "LEVEL_CLEAR"
LEVEL_FAIL = "LEVEL_FAIL"


class LevelSpec:
    def __init__(
        self, level_index: int, colors: tuple[int, ...], budget: int, left_cells: tuple[tuple[int, int, int], ...]
    ) -> None:
        self.level_index = level_index
        self.colors = colors
        self.budget = budget
        self.left_cells = left_cells

    @property
    def required_total(self) -> int:
        return len(self.left_cells)


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        level_index=0,
        colors=(COLOR_BLUE,),
        budget=9,
        left_cells=((1, 1, COLOR_BLUE), (1, 2, COLOR_BLUE), (2, 2, COLOR_BLUE)),
    ),
    LevelSpec(
        level_index=1,
        colors=(COLOR_BLUE, COLOR_ORANGE),
        budget=14,
        left_cells=(
            (1, 1, COLOR_BLUE),
            (2, 1, COLOR_ORANGE),
            (2, 2, COLOR_ORANGE),
            (3, 2, COLOR_BLUE),
            (2, 3, COLOR_BLUE),
            (3, 3, COLOR_ORANGE),
            (4, 3, COLOR_BLUE),
        ),
    ),
    LevelSpec(
        level_index=2,
        colors=(COLOR_BLUE, COLOR_ORANGE, COLOR_PURPLE),
        budget=18,
        left_cells=(
            (1, 1, COLOR_PURPLE),
            (2, 1, COLOR_BLUE),
            (2, 2, COLOR_ORANGE),
            (3, 2, COLOR_PURPLE),
            (4, 2, COLOR_BLUE),
            (4, 3, COLOR_ORANGE),
            (3, 4, COLOR_BLUE),
            (4, 4, COLOR_PURPLE),
            (3, 5, COLOR_BLUE),
            (4, 5, COLOR_ORANGE),
        ),
    ),
)

PALETTE_POSITIONS: tuple[tuple[int, int], ...] = ((2, 2), (9, 2), (16, 2))


def _solid(size: int, color: int) -> np.ndarray:
    return np.full((size, size), np.int8(color), dtype=np.int8)


class MirrorMatch(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._canvas: Sprite | None = None
        self._static_frame: np.ndarray | None = None
        self._level_index = 0
        self._mode = PLAY
        self._selected_color = COLOR_BLUE
        self._remaining_budget = 0
        self._match_count = 0
        self._matched_right_cells: set[tuple[int, int]] = set()
        self._required_targets: dict[tuple[int, int], int] = {}
        self._flash_cell: tuple[int, int] | None = None
        self._mirror_score = 0

        levels = [self._build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_WHITE, COLOR_WHITE),
            False,
            len(levels),
            [1, 2, 3, 4, 6],
            seed=seed,
        )

    def _build_level(self, spec: LevelSpec) -> Level:
        canvas = Sprite(_solid(GRID_SIZE, COLOR_WHITE), name="canvas", x=0, y=0, layer=0, collidable=False)
        return Level(
            sprites=[canvas],
            grid_size=(GRID_SIZE, GRID_SIZE),
            data={"level_spec": spec},
            name=f"Mirror Match {spec.level_index + 1}",
        )

    def on_set_level(self, level: Level) -> None:
        self._canvas = level.get_sprites_by_name("canvas")[0]
        spec = self._level_spec(level)
        self._level_index = spec.level_index
        self._reset_level_state(spec)

    def _level_spec(self, level: Level | None = None) -> LevelSpec:
        target_level = self.current_level if level is None else level
        return target_level.get_data("level_spec")

    def _reset_level_state(self, spec: LevelSpec) -> None:
        self._mode = PLAY
        self._selected_color = spec.colors[0]
        self._remaining_budget = spec.budget
        self._matched_right_cells.clear()
        self._match_count = 0
        self._mirror_score = 0
        self._flash_cell = None
        self._required_targets = {(BOARD_COLS - 1 - c, r): color for c, r, color in spec.left_cells}
        self._static_frame = self._render_static_frame(spec)
        self._sync_canvas()

    def _render_static_frame(self, spec: LevelSpec) -> np.ndarray:
        frame = np.full((GRID_SIZE, GRID_SIZE), np.int8(COLOR_WHITE), dtype=np.int8)
        frame[0:HUD_HEIGHT, :] = np.int8(COLOR_PANEL)

        self._fill_board_panel(frame, LEFT_ORIGIN)
        self._fill_board_panel(frame, RIGHT_ORIGIN)

        for y in range(BOARD_TOP, BOARD_TOP + BOARD_HEIGHT):
            stripe = COLOR_DARK_GRAY if (y - BOARD_TOP) % 2 == 0 else COLOR_VERY_DARK_GRAY
            frame[y, 31:33] = np.int8(stripe)

        for c, r, color in spec.left_cells:
            self._paint_cell(frame, LEFT_ORIGIN, c, r, color)

        return frame

    def _fill_board_panel(self, frame: np.ndarray, origin: tuple[int, int]) -> None:
        ox, oy = origin
        frame[oy : oy + BOARD_HEIGHT, ox : ox + BOARD_COLS * BOARD_SLOT] = np.int8(COLOR_PANEL)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                self._paint_cell(frame, origin, c, r, COLOR_WHITE)

    def _paint_cell(self, frame: np.ndarray, origin: tuple[int, int], c: int, r: int, color: int) -> None:
        ox, oy = origin
        x0 = ox + (c * BOARD_SLOT)
        y0 = oy + (r * BOARD_SLOT)
        frame[y0 : y0 + CELL_FILL, x0 : x0 + CELL_FILL] = np.int8(color)

    def _draw_pip(self, frame: np.ndarray, x: int, y: int, color: int) -> None:
        frame[y : y + 2, x : x + 2] = np.int8(color)

    def _sync_canvas(self) -> None:
        spec = self._level_spec()
        frame = np.array(self._static_frame, copy=True)

        for idx, color in enumerate(spec.colors):
            swatch_x, swatch_y = PALETTE_POSITIONS[idx]
            border = COLOR_BLACK if color == self._selected_color else COLOR_DARK_GRAY
            frame[swatch_y : swatch_y + 5, swatch_x : swatch_x + 5] = np.int8(border)
            frame[swatch_y + 1 : swatch_y + 4, swatch_x + 1 : swatch_x + 4] = np.int8(color)

        for idx in range(spec.required_total):
            pip_x = 22 + (idx % 5) * 3
            pip_y = 2 + (idx // 5) * 3
            pip_color = COLOR_GREEN if idx < self._match_count else COLOR_DARK_GRAY
            self._draw_pip(frame, pip_x, pip_y, pip_color)

        spent = spec.budget - self._remaining_budget
        for idx in range(spec.budget):
            pip_x = 38 + (idx % 9) * 3
            pip_y = 2 + (idx // 9) * 3
            pip_color = COLOR_DARK_GRAY if idx < spent else COLOR_YELLOW
            self._draw_pip(frame, pip_x, pip_y, pip_color)

        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if (c, r) in self._matched_right_cells:
                    color = COLOR_GREEN
                elif self._flash_cell == (c, r):
                    color = COLOR_RED
                else:
                    color = COLOR_WHITE
                self._paint_cell(frame, RIGHT_ORIGIN, c, r, color)

        if self._mode == LEVEL_CLEAR:
            self._draw_border(frame, COLOR_GREEN)
        elif self._mode == LEVEL_FAIL:
            self._draw_border(frame, COLOR_RED)

        if self._canvas is not None:
            self._canvas.pixels = frame

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[0, :] = np.int8(color)
        frame[-1, :] = np.int8(color)
        frame[:, 0] = np.int8(color)
        frame[:, -1] = np.int8(color)

    def _palette_hit(self, x: int, y: int, spec: LevelSpec) -> int | None:
        for idx, (sx, sy) in enumerate(PALETTE_POSITIONS[: len(spec.colors)]):
            if sx <= x <= sx + 4 and sy <= y <= sy + 4:
                return idx
        return None

    def _right_cell_hit(self, x: int, y: int) -> tuple[int, int] | None:
        ox, oy = RIGHT_ORIGIN
        if not (ox <= x <= ox + (BOARD_COLS * BOARD_SLOT) - 1 and oy <= y <= oy + BOARD_HEIGHT - 1):
            return None
        return ((x - ox) // BOARD_SLOT, (y - oy) // BOARD_SLOT)

    def _handle_click(self, x: int, y: int, spec: LevelSpec) -> None:
        palette_index = self._palette_hit(x, y, spec)
        if palette_index is not None:
            self._selected_color = spec.colors[palette_index]
            return

        cell = self._right_cell_hit(x, y)
        if cell is None:
            return
        if cell in self._matched_right_cells:
            return

        target_color = self._required_targets.get(cell)
        if target_color is not None and target_color == self._selected_color:
            self._matched_right_cells.add(cell)
            self._match_count = len(self._matched_right_cells)
            self._mirror_score = self._match_count
            return

        self._flash_cell = cell

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

        spec = self._level_spec()
        action_id = int(self.action.id.value)

        if self._mode == LEVEL_CLEAR:
            self.next_level()
            self.complete_action()
            return

        if self._mode == LEVEL_FAIL:
            self.lose()
            self.complete_action()
            return

        if self._flash_cell is not None:
            self._flash_cell = None

        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data or {}
            click_x = int(payload.get("x", -1))
            click_y = int(payload.get("y", -1))
            if 0 <= click_x < GRID_SIZE and 0 <= click_y < GRID_SIZE:
                self._handle_click(click_x, click_y, spec)

        self._remaining_budget = max(0, self._remaining_budget - 1)

        if self._match_count == spec.required_total:
            self._mode = LEVEL_CLEAR
        elif self._remaining_budget == 0:
            self._mode = LEVEL_FAIL
            self.lose()

        self._sync_canvas()
        self.complete_action()
