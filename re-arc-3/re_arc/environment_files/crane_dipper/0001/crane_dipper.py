from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "crane_dipper-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
GRID_SIZE = 10
REF_X = 2
REF_Y = 2
REF_CELL = 2
CANVAS_X = 30
CANVAS_Y = 18
CANVAS_CELL = 3

COLOR_BG = 4
COLOR_FRAME_REF = 3
COLOR_FRAME_CANVAS = 2
COLOR_ACTIVE_INNER = 0
COLOR_SMALL = 10
COLOR_INVALID = 8
COLOR_WIN = 14
COLOR_BAR_EMPTY = 3
COLOR_BAR_FULL = 10
COLOR_PAINT_FLASH = 0

VALID_STATIONS = {
    (-1, -1): "NW",
    (-1, 0): "N",
    (-1, 1): "NE",
    (0, -1): "W",
    (0, 1): "E",
    (1, -1): "SW",
    (1, 0): "S",
    (1, 1): "SE",
}
STATION_CENTERS = {
    (-1, -1): (27, 15),
    (-1, 0): (44, 15),
    (-1, 1): (62, 15),
    (0, -1): (27, 32),
    (0, 1): (62, 32),
    (1, -1): (27, 50),
    (1, 0): (44, 50),
    (1, 1): (62, 50),
}
MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}
SWATCHES = ((2, 7, 25, 30), (10, 15, 25, 30), (18, 23, 25, 30), (2, 7, 33, 38), (10, 15, 33, 38), (18, 23, 33, 38))
SMALL_BUTTON = (2, 23, 48, 60)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    reference: tuple[tuple[int, ...], ...]
    palette: tuple[int, ...]
    start_station: tuple[int, int]
    start_color: int
    small_available: bool
    step_budget: int


LEVEL_SPECS = (
    LevelSpec(
        name="Right half after top half",
        reference=(
            (9, 9, 9, 9, 9, 11, 11, 11, 11, 11),
            (9, 9, 9, 9, 9, 11, 11, 11, 11, 11),
            (9, 9, 9, 9, 9, 11, 11, 11, 11, 11),
            (9, 9, 9, 9, 9, 11, 11, 11, 11, 11),
            (9, 9, 9, 9, 9, 11, 11, 11, 11, 11),
            (0, 0, 0, 0, 0, 11, 11, 11, 11, 11),
            (0, 0, 0, 0, 0, 11, 11, 11, 11, 11),
            (0, 0, 0, 0, 0, 11, 11, 11, 11, 11),
            (0, 0, 0, 0, 0, 11, 11, 11, 11, 11),
            (0, 0, 0, 0, 0, 11, 11, 11, 11, 11),
        ),
        palette=(9, 11),
        start_station=(-1, 0),
        start_color=9,
        small_available=False,
        step_budget=30,
    ),
    LevelSpec(
        name="The green seam is not real",
        reference=(
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 12),
            (9, 9, 9, 9, 9, 9, 9, 9, 12, 12),
            (9, 9, 9, 9, 9, 9, 9, 12, 12, 12),
            (9, 9, 9, 9, 9, 9, 12, 12, 12, 12),
            (9, 9, 9, 9, 9, 12, 12, 12, 12, 12),
            (9, 9, 9, 9, 12, 12, 12, 12, 12, 12),
            (9, 9, 9, 12, 12, 12, 12, 12, 12, 12),
            (9, 9, 12, 12, 12, 12, 12, 12, 12, 12),
            (9, 12, 12, 12, 12, 12, 12, 12, 12, 12),
            (12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
        ),
        palette=(9, 12),
        start_station=(-1, -1),
        start_color=9,
        small_available=False,
        step_budget=42,
    ),
    LevelSpec(
        name="The inset row",
        reference=(
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 9),
            (11, 11, 11, 11, 11, 11, 11, 11, 11, 11),
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 9),
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 9),
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 9),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
        palette=(9, 11),
        start_station=(-1, 0),
        start_color=9,
        small_available=True,
        step_budget=24,
    ),
    LevelSpec(
        name="A strip that must be cropped",
        reference=(
            (9, 9, 9, 9, 9, 9, 9, 9, 9, 12),
            (11, 11, 11, 11, 11, 11, 11, 11, 12, 12),
            (9, 9, 9, 9, 9, 9, 9, 12, 12, 12),
            (9, 9, 9, 9, 9, 9, 12, 12, 12, 12),
            (9, 9, 9, 9, 9, 12, 12, 12, 12, 12),
            (0, 0, 0, 0, 12, 12, 12, 12, 12, 12),
            (0, 0, 0, 12, 12, 12, 12, 12, 12, 12),
            (0, 0, 12, 12, 12, 12, 12, 12, 12, 12),
            (0, 12, 12, 12, 12, 12, 12, 12, 12, 12),
            (12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
        ),
        palette=(9, 11, 12),
        start_station=(-1, 0),
        start_color=9,
        small_available=True,
        step_budget=60,
    ),
    LevelSpec(
        name="Crossing strips under a purple wedge",
        reference=(
            (15, 9, 9, 9, 9, 12, 12, 12, 14, 12),
            (15, 15, 11, 11, 11, 11, 11, 11, 14, 11),
            (15, 15, 15, 9, 9, 12, 12, 12, 14, 12),
            (15, 15, 15, 15, 9, 12, 12, 12, 14, 12),
            (15, 15, 15, 15, 15, 12, 12, 12, 14, 12),
            (15, 15, 15, 15, 15, 15, 12, 12, 14, 12),
            (15, 15, 15, 15, 15, 15, 15, 12, 14, 12),
            (15, 15, 15, 15, 15, 15, 15, 15, 14, 12),
            (15, 15, 15, 15, 15, 15, 15, 15, 15, 12),
            (15, 15, 15, 15, 15, 15, 15, 15, 15, 15),
        ),
        palette=(9, 12, 11, 14, 15),
        start_station=(-1, 0),
        start_color=9,
        small_available=True,
        step_budget=120,
    ),
    LevelSpec(
        name="Repair after the wedge",
        reference=(
            (15, 9, 8, 8, 8, 12, 12, 12, 14, 12),
            (15, 9, 11, 11, 11, 11, 11, 11, 14, 11),
            (15, 9, 15, 8, 8, 12, 12, 12, 14, 12),
            (15, 9, 15, 15, 8, 12, 12, 12, 14, 12),
            (15, 9, 15, 15, 15, 12, 12, 12, 14, 12),
            (15, 9, 15, 15, 15, 15, 12, 12, 14, 12),
            (15, 9, 15, 15, 15, 15, 15, 12, 14, 12),
            (15, 9, 15, 15, 15, 15, 15, 15, 14, 12),
            (15, 9, 15, 15, 15, 15, 15, 15, 15, 12),
            (15, 9, 15, 15, 15, 15, 15, 15, 15, 15),
        ),
        palette=(8, 12, 11, 14, 15, 9),
        start_station=(-1, -1),
        start_color=8,
        small_available=True,
        step_budget=150,
    ),
)


class CraneDipperDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: CraneDipper | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BG
        self._draw_reference(frame, game.reference)
        self._draw_canvas(frame, game.canvas, game.level_solved)
        self._draw_paint_flash(frame, game)
        self._draw_stations(frame, game)
        self._draw_toolbar(frame, game)
        self._draw_step_bar(frame, game.remaining_steps, game.step_budget)
        return frame

    def _draw_reference(self, frame: np.ndarray, reference: np.ndarray) -> None:
        frame[REF_Y - 1 : REF_Y + GRID_SIZE * REF_CELL + 1, REF_X - 1] = COLOR_FRAME_REF
        frame[REF_Y - 1 : REF_Y + GRID_SIZE * REF_CELL + 1, REF_X + GRID_SIZE * REF_CELL] = COLOR_FRAME_REF
        frame[REF_Y - 1, REF_X - 1 : REF_X + GRID_SIZE * REF_CELL + 1] = COLOR_FRAME_REF
        frame[REF_Y + GRID_SIZE * REF_CELL, REF_X - 1 : REF_X + GRID_SIZE * REF_CELL + 1] = COLOR_FRAME_REF
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0 = REF_X + c * REF_CELL
                y0 = REF_Y + r * REF_CELL
                frame[y0 : y0 + REF_CELL, x0 : x0 + REF_CELL] = int(reference[r, c])

    def _draw_canvas(self, frame: np.ndarray, canvas: np.ndarray, solved: bool) -> None:
        frame_color = COLOR_WIN if solved else COLOR_FRAME_CANVAS
        x1 = CANVAS_X + GRID_SIZE * CANVAS_CELL
        y1 = CANVAS_Y + GRID_SIZE * CANVAS_CELL
        frame[CANVAS_Y - 1 : y1 + 1, CANVAS_X - 1] = frame_color
        frame[CANVAS_Y - 1 : y1 + 1, x1] = frame_color
        frame[CANVAS_Y - 1, CANVAS_X - 1 : x1 + 1] = frame_color
        frame[y1, CANVAS_X - 1 : x1 + 1] = frame_color
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0 = CANVAS_X + c * CANVAS_CELL
                y0 = CANVAS_Y + r * CANVAS_CELL
                frame[y0 : y0 + CANVAS_CELL, x0 : x0 + CANVAS_CELL] = int(canvas[r, c])

    def _draw_paint_flash(self, frame: np.ndarray, game: CraneDipper) -> None:
        if game.paint_flash_frames <= 0:
            return
        for r, c in game.paint_flash_cells:
            x0 = CANVAS_X + c * CANVAS_CELL
            y0 = CANVAS_Y + r * CANVAS_CELL
            frame[y0 : y0 + CANVAS_CELL, x0 : x0 + CANVAS_CELL] = game.paint_flash_color
            frame[y0, x0 : x0 + CANVAS_CELL] = COLOR_PAINT_FLASH
            frame[y0 + CANVAS_CELL - 1, x0 : x0 + CANVAS_CELL] = COLOR_PAINT_FLASH
            frame[y0 : y0 + CANVAS_CELL, x0] = COLOR_PAINT_FLASH
            frame[y0 : y0 + CANVAS_CELL, x0 + CANVAS_CELL - 1] = COLOR_PAINT_FLASH

    def _draw_stations(self, frame: np.ndarray, game: CraneDipper) -> None:
        for station, (cx, cy) in STATION_CENTERS.items():
            color = COLOR_FRAME_REF
            if station == game.station:
                color = COLOR_INVALID if game.invalid_flash > 0 else game.selected_color
            x0 = max(0, cx - 1)
            x1 = min(VIEW_SIZE, cx + 2)
            y0 = max(0, cy - 1)
            y1 = min(VIEW_SIZE, cy + 2)
            frame[y0:y1, x0:x1] = color
            if station == game.station and 0 <= cx < VIEW_SIZE and 0 <= cy < VIEW_SIZE:
                frame[cy, cx] = COLOR_SMALL if game.small_active else COLOR_ACTIVE_INNER

    def _draw_toolbar(self, frame: np.ndarray, game: CraneDipper) -> None:
        for idx, color in enumerate(game.palette):
            x0, x1, y0, y1 = SWATCHES[idx]
            frame[y0 : y1 + 1, x0 : x1 + 1] = color
            border = COLOR_ACTIVE_INNER if color == game.selected_color else COLOR_FRAME_REF
            frame[y0 : y1 + 1, x0] = border
            frame[y0 : y1 + 1, x1] = border
            frame[y0, x0 : x1 + 1] = border
            frame[y1, x0 : x1 + 1] = border
        if game.small_available:
            x0, x1, y0, y1 = SMALL_BUTTON
            border = COLOR_SMALL if game.small_active else COLOR_FRAME_REF
            frame[y0 : y1 + 1, x0] = border
            frame[y0 : y1 + 1, x1] = border
            frame[y0, x0 : x1 + 1] = border
            frame[y1, x0 : x1 + 1] = border
            frame[y0 + 3 : y1 - 2, x0 + 8 : x1 - 7] = COLOR_FRAME_CANVAS
            frame[y0 + 5 : y1 - 4, x0 + 10 : x1 - 9] = COLOR_SMALL
            frame[y0 + 2 : y0 + 5, x0 + 12 : x0 + 14] = COLOR_SMALL

    def _draw_step_bar(self, frame: np.ndarray, remaining_steps: int, step_budget: int) -> None:
        x0 = 2
        y0 = 62
        width = 60
        frame[y0:64, x0 : x0 + width] = COLOR_BAR_EMPTY
        filled = max(0, min(width, round(width * remaining_steps / max(1, step_budget))))
        if filled:
            frame[y0:64, x0 : x0 + filled] = COLOR_BAR_FULL


class CraneDipper(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = CraneDipperDisplay()
        levels = [Level(grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(
            0, 0, VIEW_SIZE, VIEW_SIZE, background=COLOR_BG, letter_box=COLOR_BG, interfaces=[self._display]
        )
        self.reference = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.palette: tuple[int, ...] = ()
        self.station = (-1, 0)
        self.selected_color = 0
        self.small_available = False
        self.small_active = False
        self.invalid_flash = 0
        self.paint_flash_cells: list[tuple[int, int]] = []
        self.paint_flash_color = 0
        self.paint_flash_frames = 0
        self.level_solved = False
        self.remaining_steps = 1
        self.step_budget = 1
        self._display.game = self
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6], seed=seed)

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.reference = np.array(spec.reference, dtype=np.int8)
        self.canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.palette = tuple(spec.palette)
        self.station = tuple(spec.start_station)
        self.selected_color = int(spec.start_color)
        self.small_available = bool(spec.small_available)
        self.small_active = False
        self.invalid_flash = 0
        self.paint_flash_cells = []
        self.paint_flash_color = 0
        self.paint_flash_frames = 0
        self.level_solved = False
        self.remaining_steps = int(spec.step_budget)
        self.step_budget = int(spec.step_budget)

    def step(self) -> None:
        if self.paint_flash_frames > 0:
            self.paint_flash_frames -= 1
            if self.paint_flash_frames == 0:
                self.paint_flash_cells = []
                self._finish_resolved_action()
            return

        if self.invalid_flash > 0:
            self.invalid_flash -= 1
            if self.invalid_flash == 0:
                self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.level_solved = False
        handled = self._resolve_action()
        if handled:
            self.remaining_steps = max(0, self.remaining_steps - 1)

        if self.paint_flash_frames > 0:
            return

        self._finish_resolved_action()

    def _finish_resolved_action(self) -> None:
        if self._is_solved():
            self.level_solved = True
            self.complete_action()
            self.next_level()
            return

        if self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
            return

        if self.invalid_flash > 0:
            return
        self.complete_action()

    def _resolve_action(self) -> bool:
        action_id = int(self.action.id.value)
        if action_id in MOVE_DELTAS:
            self._move_station(action_id)
            return True
        if action_id == ACTION_SPACE:
            self._paint()
            return True
        if action_id == ACTION_CLICK:
            data = self.action.data or {}
            self._click(int(data.get("x", 0)), int(data.get("y", 0)))
            return True
        self._mark_invalid()
        return True

    def _move_station(self, action_id: int) -> None:
        dr, dc = MOVE_DELTAS[action_id]
        candidate = (self.station[0] + dr, self.station[1] + dc)
        if candidate in VALID_STATIONS:
            self.station = candidate
        else:
            self._mark_invalid()

    def _click(self, x: int, y: int) -> None:
        for idx, (x0, x1, y0, y1) in enumerate(SWATCHES):
            if idx < len(self.palette) and x0 <= x <= x1 and y0 <= y <= y1:
                self.selected_color = self.palette[idx]
                return
        x0, x1, y0, y1 = SMALL_BUTTON
        if self.small_available and x0 <= x <= x1 and y0 <= y <= y1:
            self.small_active = not self.small_active

    def _paint(self) -> None:
        sr, sc = self.station
        if self.small_active:
            if sr != 0 and sc != 0:
                self._mark_invalid()
                return
            cells = self._small_cells()
        else:
            cells = self._broad_cells()
        for r, c in cells:
            self.canvas[r, c] = self.selected_color
        self.paint_flash_cells = list(cells)
        self.paint_flash_color = self.selected_color
        self.paint_flash_frames = 1

    def _small_cells(self) -> list[tuple[int, int]]:
        sr, sc = self.station
        if (sr, sc) == (-1, 0):
            return [(1, c) for c in range(GRID_SIZE)]
        if (sr, sc) == (1, 0):
            return [(8, c) for c in range(GRID_SIZE)]
        if (sr, sc) == (0, -1):
            return [(r, 1) for r in range(GRID_SIZE)]
        if (sr, sc) == (0, 1):
            return [(r, 8) for r in range(GRID_SIZE)]
        return []

    def _broad_cells(self) -> list[tuple[int, int]]:
        sr, sc = self.station
        cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self._broad_contains(sr, sc, r, c):
                    cells.append((r, c))
        return cells

    def _broad_contains(self, sr: int, sc: int, r: int, c: int) -> bool:
        if (sr, sc) == (-1, 0):
            return r <= 4
        if (sr, sc) == (1, 0):
            return r >= 5
        if (sr, sc) == (0, -1):
            return c <= 4
        if (sr, sc) == (0, 1):
            return c >= 5
        if (sr, sc) == (-1, -1):
            return r + c <= 9
        if (sr, sc) == (1, 1):
            return r + c >= 9
        if (sr, sc) == (-1, 1):
            return c >= r
        if (sr, sc) == (1, -1):
            return r >= c
        return False

    def _is_solved(self) -> bool:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if r == c or r + c == GRID_SIZE - 1:
                    continue
                if int(self.canvas[r, c]) != int(self.reference[r, c]):
                    return False
        return True

    def _mark_invalid(self) -> None:
        self.invalid_flash = 2
