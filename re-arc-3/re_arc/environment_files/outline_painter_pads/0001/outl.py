from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 7
CELL_SIZE = 6
BOARD_ORIGIN_X = 11
BOARD_ORIGIN_Y = 11

COLOR_BG = 0
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK_GRAY = 4
COLOR_BLACK = 5
COLOR_PINK = 7
COLOR_RED = 8
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

PAD_REGIONS = {UP: (27, 35, 1, 9), DOWN: (27, 35, 54, 62), LEFT: (1, 9, 27, 35), RIGHT: (54, 62, 27, 35)}

PIP_CENTERS = (2, 6, 10, 14, 18, 22, 26, 38, 42, 46, 50, 54, 58, 62)
LEVEL_SPECS = (
    {
        "name": "Level 1",
        "start_pos": (1, 3),
        "target_set": frozenset({(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)}),
        "decoy_set": frozenset(),
        "move_budget": 12,
        "optimal_moves": 4,
    },
    {
        "name": "Level 2",
        "start_pos": (1, 5),
        "target_set": frozenset({(1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (4, 3)}),
        "decoy_set": frozenset(),
        "move_budget": 14,
        "optimal_moves": 5,
    },
    {
        "name": "Level 3",
        "start_pos": (1, 5),
        "target_set": frozenset({(1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (4, 3), (5, 3)}),
        "decoy_set": frozenset({(2, 4), (2, 3)}),
        "move_budget": 14,
        "optimal_moves": 6,
    },
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), np.int8(color), dtype=np.int8)


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    cx, cy = cell
    return BOARD_ORIGIN_X + (cx * CELL_SIZE), BOARD_ORIGIN_Y + (cy * CELL_SIZE)


def _in_bounds(cell: tuple[int, int]) -> bool:
    cx, cy = cell
    return 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE


def _frame_sprite() -> Sprite:
    return Sprite(_solid(GRID_SIZE, GRID_SIZE, COLOR_BG), name="frame", x=0, y=0, layer=0)


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(LEVEL_SPECS, start=1):
        levels.append(
            Level(
                name=f"Level {idx}",
                sprites=[_frame_sprite()],
                grid_size=(GRID_SIZE, GRID_SIZE),
                data={
                    "spec": dict(spec),
                    "start_pos": spec["start_pos"],
                    "target_set": tuple(sorted(spec["target_set"])),
                    "decoy_set": tuple(sorted(spec["decoy_set"])),
                    "move_budget": spec["move_budget"],
                    "optimal_moves": spec["optimal_moves"],
                },
            )
        )
    return levels


class OutlinePainterPads(ARCBaseGame):
    def __init__(self) -> None:
        self._route_score = 0
        self._roller_pos = (0, 0)
        self._painted_set: set[tuple[int, int]] = set()
        self._remaining_moves = 0
        self._budget_max = 0
        self._target_set: frozenset[tuple[int, int]] = frozenset()
        self._decoy_set: frozenset[tuple[int, int]] = frozenset()
        self._last_dir = RIGHT
        self._won = False
        self._lost = False
        super().__init__(
            "outline_painter_pads",
            _build_levels(),
            Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BG, COLOR_BG),
            False,
            len(LEVEL_SPECS),
            [6],
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        assert isinstance(spec, dict)
        self._route_score = 0
        self._roller_pos = tuple(spec["start_pos"])
        self._painted_set = {tuple(spec["start_pos"])}
        self._remaining_moves = int(spec["move_budget"])
        self._budget_max = int(spec["move_budget"])
        self._target_set = frozenset(spec["target_set"])
        self._decoy_set = frozenset(spec["decoy_set"])
        self._last_dir = RIGHT
        self._won = False
        self._lost = False
        self._refresh_frame()

    def step(self) -> None:
        click_x = int(self.action.data.get("x", 0))
        click_y = int(self.action.data.get("y", 0))
        direction = self._pad_direction(click_x, click_y)
        if direction is None:
            self._refresh_frame()
            self.complete_action()
            return

        next_cell = (self._roller_pos[0] + direction[0], self._roller_pos[1] + direction[1])
        if not _in_bounds(next_cell):
            self._refresh_frame()
            self.complete_action()
            return

        self._roller_pos = next_cell
        self._last_dir = direction
        self._remaining_moves -= 1
        self._painted_set.add(next_cell)

        if self._painted_set == self._target_set:
            self._won = True
            self._refresh_frame()
            self.next_level()
            self.complete_action()
            return

        if self._is_complete_failure():
            self._lost = True
            self._refresh_frame()
            self.lose()
            self.complete_action()
            return

        self._refresh_frame()
        self.complete_action()

    def _pad_direction(self, x: int, y: int) -> tuple[int, int] | None:
        for direction, (x0, x1, y0, y1) in PAD_REGIONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return direction
        return None

    def _is_complete_failure(self) -> bool:
        if not self._painted_set.issubset(self._target_set):
            return True
        unpainted_targets = len(self._target_set - self._painted_set)
        return self._remaining_moves < unpainted_targets

    def _refresh_frame(self) -> None:
        frame = _solid(GRID_SIZE, GRID_SIZE, COLOR_BG)
        self._draw_board_frame(frame)
        self._draw_outline_cells(frame, self._target_set, decoy=False)
        self._draw_outline_cells(frame, self._decoy_set, decoy=True)
        self._draw_painted_cells(frame)
        self._draw_roller(frame)
        self._draw_pads(frame)
        self._draw_budget(frame)
        self.current_level.get_sprites_by_name("frame")[0].pixels = frame

    def _draw_board_frame(self, frame: np.ndarray) -> None:
        frame[10, 10:54] = COLOR_DARK_GRAY
        frame[53, 10:54] = COLOR_DARK_GRAY
        frame[10:54, 10] = COLOR_DARK_GRAY
        frame[10:54, 53] = COLOR_DARK_GRAY
        accent = COLOR_GREEN if self._won else COLOR_RED if self._lost else None
        if accent is not None:
            frame[9, 9:55] = accent
            frame[54, 9:55] = accent
            frame[9:55, 9] = accent
            frame[9:55, 54] = accent

    def _draw_outline_cells(self, frame: np.ndarray, cells: frozenset[tuple[int, int]], *, decoy: bool) -> None:
        for cell in cells:
            ox, oy = _cell_origin(cell)
            if decoy:
                self._draw_decoy_outline(frame, ox, oy)
            else:
                self._draw_real_outline(frame, ox, oy)

    def _draw_real_outline(self, frame: np.ndarray, ox: int, oy: int) -> None:
        frame[oy : oy + CELL_SIZE, ox : ox + CELL_SIZE] = COLOR_LIGHT_BLUE
        frame[oy + 1 : oy + 5, ox + 1 : ox + 5] = COLOR_BG

    def _draw_decoy_outline(self, frame: np.ndarray, ox: int, oy: int) -> None:
        for dx in (0, 1, 4, 5):
            frame[oy, ox + dx] = COLOR_PINK
            frame[oy + 5, ox + dx] = COLOR_PINK
        for dy in (0, 1, 4, 5):
            frame[oy + dy, ox] = COLOR_PINK
            frame[oy + dy, ox + 5] = COLOR_PINK
        frame[oy + 1 : oy + 5, ox + 1 : ox + 5] = COLOR_BG

    def _draw_painted_cells(self, frame: np.ndarray) -> None:
        for cell in self._painted_set:
            ox, oy = _cell_origin(cell)
            if cell in self._target_set:
                frame[oy : oy + CELL_SIZE, ox : ox + CELL_SIZE] = COLOR_LIGHT_BLUE
                frame[oy + 1 : oy + 5, ox + 1 : ox + 5] = COLOR_GREEN
            else:
                frame[oy : oy + CELL_SIZE, ox : ox + CELL_SIZE] = COLOR_MAROON
                frame[oy + 1 : oy + 5, ox + 1 : ox + 5] = COLOR_RED

    def _draw_roller(self, frame: np.ndarray) -> None:
        ox, oy = _cell_origin(self._roller_pos)
        frame[oy : oy + CELL_SIZE, ox : ox + CELL_SIZE] = COLOR_BLACK
        frame[oy + 1 : oy + 5, ox + 1 : ox + 5] = COLOR_GREEN
        frame[oy + 2 : oy + 4, ox + 2 : ox + 4] = COLOR_YELLOW
        if self._last_dir == RIGHT:
            frame[oy + 1 : oy + 5, ox + 4] = COLOR_ORANGE
        elif self._last_dir == LEFT:
            frame[oy + 1 : oy + 5, ox + 1] = COLOR_ORANGE
        elif self._last_dir == UP:
            frame[oy + 1, ox + 1 : ox + 5] = COLOR_ORANGE
        else:
            frame[oy + 4, ox + 1 : ox + 5] = COLOR_ORANGE

    def _draw_pads(self, frame: np.ndarray) -> None:
        self._draw_pad(frame, 27, 1, 0)
        self._draw_pad(frame, 27, 54, 2)
        self._draw_pad(frame, 1, 27, 1)
        self._draw_pad(frame, 54, 27, 3)

    def _draw_pad(self, frame: np.ndarray, x: int, y: int, rotation_k: int) -> None:
        frame[y : y + 9, x : x + 9] = COLOR_VERY_DARK_GRAY
        frame[y, x : x + 9] = COLOR_BLACK
        frame[y + 8, x : x + 9] = COLOR_BLACK
        frame[y : y + 9, x] = COLOR_BLACK
        frame[y : y + 9, x + 8] = COLOR_BLACK

        arrow = np.full((9, 9), -1, dtype=np.int8)
        for px, py in (
            (4, 1),
            (3, 2),
            (4, 2),
            (5, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
        ):
            arrow[py, px] = COLOR_YELLOW
        arrow = np.rot90(arrow, k=rotation_k)
        mask = arrow >= 0
        frame[y : y + 9, x : x + 9][mask] = arrow[mask]

    def _draw_budget(self, frame: np.ndarray) -> None:
        shown_budget = min(self._budget_max, len(PIP_CENTERS))
        spent = shown_budget - min(self._remaining_moves, shown_budget)
        for idx in range(shown_budget):
            center_x = PIP_CENTERS[idx]
            color = COLOR_GRAY if idx < spent else COLOR_GREEN
            frame[3:6, center_x - 1 : center_x + 2] = color


class Outl(OutlinePainterPads):
    pass
