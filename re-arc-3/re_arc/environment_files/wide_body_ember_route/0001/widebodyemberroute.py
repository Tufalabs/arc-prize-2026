from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_SPENT = 3
COLOR_WALL = 4
COLOR_WALL_EDGE = 5
COLOR_FAIL = 8
COLOR_AVATAR_OUTLINE = 9
COLOR_AVATAR_FILL = 10
COLOR_BUDGET_HIGH = 11
COLOR_EMBER = 12
COLOR_DOCK = 14

GRID_WIDTH = 16
GRID_HEIGHT = 14
CELL_SIZE = 4
PLAYFIELD_Y = 8
HUD_BAR_X = 2
HUD_BAR_Y = 2
HUD_BAR_W = 60
HUD_BAR_H = 4


LEVEL_SPECS: tuple[dict[str, object], ...] = (
    {"start": (2, 5), "dock": (12, 5), "budget": 18, "ember_rectangles": ((5, 4, 13, 4), (5, 7, 13, 7))},
    {"start": (2, 4), "dock": (12, 3), "budget": 24, "ember_rectangles": ((8, 2, 8, 4), (8, 6, 8, 7), (8, 10, 8, 11))},
    {
        "start": (2, 10),
        "dock": (12, 5),
        "budget": 28,
        "ember_rectangles": (
            (3, 2, 5, 4),
            (7, 3, 9, 5),
            (4, 8, 6, 10),
            (10, 8, 11, 10),
            (12, 4, 14, 4),
            (12, 7, 14, 7),
        ),
    },
)

MOVE_DELTAS: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

AVATAR_PIXELS = np.array(
    [
        [9, 9, 9, 9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 10, 10, 9, 9, 10, 10, 9],
        [9, 10, 10, 9, 9, 10, 10, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 9, 9, 9, 9, 9, 9, 9],
    ],
    dtype=np.int8,
)
FAIL_AVATAR_PIXELS = np.where(AVATAR_PIXELS >= 0, COLOR_FAIL, AVATAR_PIXELS).astype(np.int8)
EMBER_PIXELS = np.array([[11, 12, 12, 11], [12, 8, 8, 12], [12, 8, 8, 12], [11, 12, 12, 11]], dtype=np.int8)


def _pixel_rect_for_cell(cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
    px = cell_x * CELL_SIZE
    py = PLAYFIELD_Y + (cell_y * CELL_SIZE)
    return px, py, px + CELL_SIZE, py + CELL_SIZE


def _expand_embers(rectangles: tuple[tuple[int, int, int, int], ...]) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for x1, y1, x2, y2 in rectangles:
        for cell_x in range(x1, x2 + 1):
            for cell_y in range(y1, y2 + 1):
                cells.add((cell_x, cell_y))
    return frozenset(cells)


class WideBodyHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: WideBodyEmberRoute | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[:, :] = COLOR_WHITE
        frame[PLAYFIELD_Y:, :] = COLOR_FLOOR

        self._draw_budget_bar(frame, game)
        self._draw_walls(frame)
        self._draw_dock(frame, game.dock_x, game.dock_y)
        self._draw_embers(frame, game)
        self._draw_avatar(frame, game)
        return frame

    def _draw_budget_bar(self, frame: np.ndarray, game: WideBodyEmberRoute) -> None:
        frame[HUD_BAR_Y : HUD_BAR_Y + HUD_BAR_H, HUD_BAR_X : HUD_BAR_X + HUD_BAR_W] = COLOR_SPENT
        if game.phase == "fail" and game.fail_reason == "budget":
            fill_color = COLOR_FAIL
            fill_width = HUD_BAR_W
        else:
            fill_width = int((HUD_BAR_W * max(0, game.remaining_moves)) // game.level_budget)
            fill_color = COLOR_BUDGET_HIGH if game.remaining_moves * 3 > game.level_budget else COLOR_EMBER
        if fill_width > 0:
            frame[HUD_BAR_Y : HUD_BAR_Y + HUD_BAR_H, HUD_BAR_X : HUD_BAR_X + fill_width] = fill_color

    def _draw_walls(self, frame: np.ndarray) -> None:
        for cell_x in range(GRID_WIDTH):
            self._draw_wall_cell(frame, cell_x, 0)
            self._draw_wall_cell(frame, cell_x, GRID_HEIGHT - 1)
        for cell_y in range(1, GRID_HEIGHT - 1):
            self._draw_wall_cell(frame, 0, cell_y)
            self._draw_wall_cell(frame, GRID_WIDTH - 1, cell_y)

    def _draw_wall_cell(self, frame: np.ndarray, cell_x: int, cell_y: int) -> None:
        x0, y0, x1, y1 = _pixel_rect_for_cell(cell_x, cell_y)
        frame[y0:y1, x0:x1] = COLOR_WALL
        frame[y0, x0:x1] = COLOR_WALL_EDGE
        frame[y1 - 1, x0:x1] = COLOR_WALL_EDGE
        frame[y0:y1, x0] = COLOR_WALL_EDGE
        frame[y0:y1, x1 - 1] = COLOR_WALL_EDGE

    def _draw_dock(self, frame: np.ndarray, dock_x: int, dock_y: int) -> None:
        x0 = dock_x * CELL_SIZE
        y0 = PLAYFIELD_Y + (dock_y * CELL_SIZE)
        frame[y0 : y0 + 8, x0 : x0 + 8] = COLOR_DOCK
        frame[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = COLOR_FLOOR
        frame[y0 : y0 + 8, x0 : x0 + 8 : 7] = COLOR_DOCK
        frame[y0 : y0 + 8 : 7, x0 : x0 + 8] = COLOR_DOCK

    def _draw_embers(self, frame: np.ndarray, game: WideBodyEmberRoute) -> None:
        highlighted = game.fail_overlap_cells() if game.phase == "fail" and game.fail_reason == "ember" else set()
        for cell_x, cell_y in game.ember_cells:
            x0, y0, _, _ = _pixel_rect_for_cell(cell_x, cell_y)
            pixels = EMBER_PIXELS.copy()
            if (cell_x, cell_y) in highlighted:
                pixels[0, 1:3] = COLOR_BUDGET_HIGH
                pixels[1:3, 0] = COLOR_BUDGET_HIGH
                pixels[1:3, 3] = COLOR_BUDGET_HIGH
                pixels[3, 1:3] = COLOR_BUDGET_HIGH
            frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = pixels

    def _draw_avatar(self, frame: np.ndarray, game: WideBodyEmberRoute) -> None:
        if game.phase == "fail":
            avatar_x = game.fail_flash_x if game.fail_reason == "ember" else game.player_x
            avatar_y = game.fail_flash_y if game.fail_reason == "ember" else game.player_y
            avatar_pixels = FAIL_AVATAR_PIXELS
        else:
            avatar_x = game.player_x
            avatar_y = game.player_y
            avatar_pixels = AVATAR_PIXELS
        px = avatar_x * CELL_SIZE
        py = PLAYFIELD_Y + (avatar_y * CELL_SIZE)
        frame[py : py + 8, px : px + 8] = avatar_pixels


class WideBodyEmberRoute(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = WideBodyHud()
        self._specs = LEVEL_SPECS
        levels = [
            Level(grid_size=(64, 64), data={"spec": spec}, name=f"Level {level_index + 1}")
            for level_index, spec in enumerate(self._specs)
        ]
        super().__init__(
            "wide_body_ember_route",
            levels,
            Camera(0, 0, 64, 64, COLOR_WHITE, COLOR_WHITE, [self._hud]),
            False,
            len(self._specs),
            [1, 2, 3, 4, 5, 6],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        if not isinstance(spec, dict):
            raise TypeError("Level spec missing from level data.")
        self._level_spec = spec
        self.level_budget = int(spec["budget"])
        self.player_x, self.player_y = spec["start"]
        self.dock_x, self.dock_y = spec["dock"]
        self.remaining_moves = int(spec["budget"])
        self.phase = "play"
        self.fail_reason: str | None = None
        self.fail_flash_x = self.player_x
        self.fail_flash_y = self.player_y
        self.ember_cells = _expand_embers(spec["ember_rectangles"])

    def _attempt_delta(self) -> tuple[int, int]:
        return MOVE_DELTAS.get(int(self.action.id.value), (0, 0))

    def _occupied_cells(self, top_left_x: int, top_left_y: int) -> tuple[tuple[int, int], ...]:
        return (
            (top_left_x, top_left_y),
            (top_left_x + 1, top_left_y),
            (top_left_x, top_left_y + 1),
            (top_left_x + 1, top_left_y + 1),
        )

    def _blocked_by_wall(self, top_left_x: int, top_left_y: int) -> bool:
        for cell_x, cell_y in self._occupied_cells(top_left_x, top_left_y):
            if cell_x < 0 or cell_x >= GRID_WIDTH or cell_y < 0 or cell_y >= GRID_HEIGHT:
                return True
            if cell_x == 0 or cell_x == GRID_WIDTH - 1 or cell_y == 0 or cell_y == GRID_HEIGHT - 1:
                return True
        return False

    def _hits_ember(self, top_left_x: int, top_left_y: int) -> bool:
        return any(cell in self.ember_cells for cell in self._occupied_cells(top_left_x, top_left_y))

    def fail_overlap_cells(self) -> set[tuple[int, int]]:
        if self.fail_reason != "ember":
            return set()
        return {cell for cell in self._occupied_cells(self.fail_flash_x, self.fail_flash_y) if cell in self.ember_cells}

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

        if self.phase == "fail":
            self.lose()
            self.complete_action()
            return

        dx, dy = self._attempt_delta()
        attempt_x = self.player_x + dx
        attempt_y = self.player_y + dy
        self.remaining_moves -= 1

        if not self._blocked_by_wall(attempt_x, attempt_y):
            if self._hits_ember(attempt_x, attempt_y):
                self.phase = "fail"
                self.fail_reason = "ember"
                self.fail_flash_x = attempt_x
                self.fail_flash_y = attempt_y
                self.complete_action()
                return
            self.player_x = attempt_x
            self.player_y = attempt_y

        if self.player_x == self.dock_x and self.player_y == self.dock_y:
            if self.is_last_level():
                self.phase = "win"
                self.next_level()
            else:
                self.next_level()
            self.complete_action()
            return

        if self.remaining_moves <= 0:
            self.phase = "fail"
            self.fail_reason = "budget"
            self.fail_flash_x = self.player_x
            self.fail_flash_y = self.player_y

        self.complete_action()
