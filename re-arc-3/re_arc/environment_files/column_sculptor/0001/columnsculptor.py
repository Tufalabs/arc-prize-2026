from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
HUD_MAX_Y = 7
PLAYFIELD_MIN_Y = 8
FLOOR_MIN_Y = 58
COLUMN_BOTTOM_Y = 57

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_OUTLINE = 4
COLOR_MAGENTA = 6
COLOR_MAGENTA_HI = 7
COLOR_FAIL = 8
COLOR_BLUE = 9
COLOR_BLUE_HI = 10
COLOR_BUDGET = 11
COLOR_LAUNCH = 12
COLOR_WALL = 13
COLOR_TERRAIN = 14

MODE_EDIT = "EDIT"
MODE_RUNNING = "RUNNING"
MODE_SUCCESS = "SUCCESS"
MODE_FAIL = "FAIL"

COLUMN_X_SPANS: tuple[tuple[int, int], ...] = ((6, 14), (17, 25), (28, 36), (39, 47), (50, 58))
GAP_X_SPANS: tuple[tuple[int, int], ...] = ((15, 16), (26, 27), (37, 38), (48, 49))
HEIGHT_TO_TOP_Y = {1: 50, 2: 42, 3: 34, 4: 26}
LAUNCH_RECT = (54, 63, 0, 7)


def _blank_pixels(width: int, height: int, color: int = COLOR_BG) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int16)


sprites = {
    "board": Sprite(
        pixels=_blank_pixels(GRID_SIZE, GRID_SIZE),
        name="board",
        x=0,
        y=0,
        visible=True,
        collidable=False,
        tags=["board"],
    )
}


@dataclass(frozen=True)
class BallSpec:
    color: Literal["blue", "magenta"]
    start_index: int


@dataclass(frozen=True)
class BasinSpec:
    color: Literal["blue", "magenta"]


@dataclass(frozen=True)
class WallSpec:
    left_column: int
    height: int


@dataclass(frozen=True)
class LevelSpec:
    key: str
    budget: int
    columns: tuple[int, int, int, int, int]
    balls: tuple[BallSpec, ...]
    left_basin: BasinSpec | None
    right_basin: BasinSpec | None
    walls: tuple[WallSpec, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        key="teach",
        budget=5,
        columns=(4, 4, 3, 2, 4),
        balls=(BallSpec("blue", 1),),
        left_basin=None,
        right_basin=BasinSpec("blue"),
        walls=(),
    ),
    LevelSpec(
        key="shared",
        budget=8,
        columns=(4, 3, 4, 3, 4),
        balls=(BallSpec("blue", 1), BallSpec("magenta", 3)),
        left_basin=BasinSpec("blue"),
        right_basin=BasinSpec("magenta"),
        walls=(),
    ),
    LevelSpec(
        key="wall",
        budget=7,
        columns=(4, 4, 2, 2, 1),
        balls=(BallSpec("blue", 1),),
        left_basin=None,
        right_basin=BasinSpec("blue"),
        walls=(WallSpec(left_column=1, height=2),),
    ),
)


def _make_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(LEVEL_SPECS):
        board = sprites["board"].clone()
        levels.append(Level([board], grid_size=(GRID_SIZE, GRID_SIZE), data={"spec_index": idx}, name=spec.key))
    return levels


levels = _make_levels()


class ColumnSculptor(ARCBaseGame):
    def __init__(self) -> None:
        super().__init__(
            "column_sculptor",
            levels,
            Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BG, COLOR_BG, []),
            available_actions=[1, 2, 3, 4, 5, 6],
        )
        self._mode = MODE_EDIT
        self._remaining_budget = 0
        self._columns: list[int] = []
        self._ball_positions: list[int | str] = []
        self._ball_captured: list[bool] = []
        self._failed = False

    def on_set_level(self, _level: Level) -> None:
        self._load_level_state(self.level_index)

    def _level_spec(self, level_index: int | None = None) -> LevelSpec:
        idx = self.level_index if level_index is None else int(level_index)
        return LEVEL_SPECS[idx]

    def _board_sprite(self) -> Sprite:
        board = self.current_level.get_sprites_by_tag("board")
        if not board:
            raise RuntimeError("Column Sculptor board sprite missing.")
        return board[0]

    def _load_level_state(self, level_index: int) -> None:
        spec = self._level_spec(level_index)
        self._mode = MODE_EDIT
        self._remaining_budget = int(spec.budget)
        self._columns = list(spec.columns)
        self._ball_positions = [ball.start_index for ball in spec.balls]
        self._ball_captured = [False for _ in spec.balls]
        self._failed = False
        self._sync_board()

    def _lose_level(self) -> None:
        self._mode = MODE_FAIL
        self._failed = True
        self._sync_board()
        self.lose()

    def _sync_board(self) -> None:
        frame = _blank_pixels(GRID_SIZE, GRID_SIZE)
        frame[FLOOR_MIN_Y:, :] = COLOR_FLOOR

        self._draw_budget(frame)
        self._draw_launch_button(frame)
        self._draw_basins(frame)
        self._draw_columns(frame)
        self._draw_walls(frame)
        self._draw_balls(frame)

        if self._mode == MODE_SUCCESS or self._failed:
            border_color = COLOR_TERRAIN if self._mode == MODE_SUCCESS else COLOR_FAIL
            frame[0, :] = border_color
            frame[-1, :] = border_color
            frame[:, 0] = border_color
            frame[:, -1] = border_color

        self._board_sprite().pixels = frame

    def _draw_budget(self, frame: np.ndarray) -> None:
        total = self._level_spec().budget
        for idx in range(total):
            x0 = 4 + idx * 6
            x1 = x0 + 4
            color = COLOR_BUDGET if idx < self._remaining_budget else COLOR_OUTLINE
            frame[2:6, x0:x1] = color

    def _draw_launch_button(self, frame: np.ndarray) -> None:
        x0, x1, y0, y1 = LAUNCH_RECT
        frame[y0 : y1 + 1, x0 : x1 + 1] = COLOR_OUTLINE
        frame[y0 + 1 : y1, x0 + 1 : x1] = COLOR_LAUNCH
        frame[y0 + 2 : y1 - 1, x0 + 2 : x1 - 1] = COLOR_BUDGET
        frame[y0 + 1 : y1, x0 + 3 : x1 - 2] = COLOR_LAUNCH

    def _draw_columns(self, frame: np.ndarray) -> None:
        for idx, height in enumerate(self._columns):
            x0, x1 = COLUMN_X_SPANS[idx]
            top_y = HEIGHT_TO_TOP_Y[height]
            frame[top_y : COLUMN_BOTTOM_Y + 1, x0 : x1 + 1] = COLOR_OUTLINE
            frame[top_y + 1 : COLUMN_BOTTOM_Y, x0 + 1 : x1] = COLOR_TERRAIN
            frame[top_y + 1, x0 + 1 : x1] = COLOR_BLUE_HI

    def _draw_balls(self, frame: np.ndarray) -> None:
        for idx, ball in enumerate(self._level_spec().balls):
            position = self._ball_positions[idx]
            if isinstance(position, str):
                if position == "left_basin":
                    self._draw_ball_token(frame, 0, 50, ball.color)
                elif position == "right_basin":
                    self._draw_ball_token(frame, 59, 50, ball.color)
                continue

            x0, x1 = COLUMN_X_SPANS[position]
            center_x = (x0 + x1) // 2
            top_y = HEIGHT_TO_TOP_Y[self._columns[position]] - 5
            self._draw_ball_token(frame, center_x - 2, top_y, ball.color)

    def _draw_ball_token(self, frame: np.ndarray, x0: int, y0: int, color: str) -> None:
        fill = COLOR_BLUE if color == "blue" else COLOR_MAGENTA
        hi = COLOR_BLUE_HI if color == "blue" else COLOR_MAGENTA_HI
        token = (
            (-1, -1, hi, -1, -1),
            (-1, hi, fill, hi, -1),
            (hi, fill, fill, fill, hi),
            (-1, hi, fill, hi, -1),
            (-1, -1, hi, -1, -1),
        )
        for dy, row in enumerate(token):
            for dx, cell in enumerate(row):
                if cell < 0:
                    continue
                px = x0 + dx
                py = y0 + dy
                if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                    frame[py, px] = cell

    def _draw_basins(self, frame: np.ndarray) -> None:
        spec = self._level_spec()
        if spec.left_basin is not None:
            self._draw_basin(frame, "left", spec.left_basin.color)
        if spec.right_basin is not None:
            self._draw_basin(frame, "right", spec.right_basin.color)

    def _draw_basin(self, frame: np.ndarray, side: str, color: str) -> None:
        fill = COLOR_BLUE if color == "blue" else COLOR_MAGENTA
        x0 = 0 if side == "left" else 59
        x1 = x0 + 4
        y0 = 48
        y1 = 57
        frame[y0:y1, x0] = fill
        frame[y0:y1, x1] = fill
        frame[y1, x0 : x1 + 1] = fill
        frame[y0 + 1 : y1, x0 + 1 : x1] = COLOR_FLOOR

    def _draw_walls(self, frame: np.ndarray) -> None:
        for wall in self._level_spec().walls:
            x0, x1 = GAP_X_SPANS[wall.left_column]
            top_y = HEIGHT_TO_TOP_Y[wall.height]
            frame[top_y : COLUMN_BOTTOM_Y + 1, x0] = COLOR_OUTLINE
            frame[top_y : COLUMN_BOTTOM_Y + 1, x1] = COLOR_WALL

    def _column_index_at(self, x: int, y: int) -> int | None:
        if not (PLAYFIELD_MIN_Y <= y <= COLUMN_BOTTOM_Y):
            return None
        for idx, (x0, x1) in enumerate(COLUMN_X_SPANS):
            if x0 <= x <= x1:
                return idx
        return None

    def _clicked_launch(self, x: int, y: int) -> bool:
        x0, x1, y0, y1 = LAUNCH_RECT
        return x0 <= x <= x1 and y0 <= y <= y1

    def _spend_budget(self) -> bool:
        if self._remaining_budget <= 0:
            return False
        self._remaining_budget -= 1
        return True

    def _wall_between(self, left_index: int, right_index: int) -> WallSpec | None:
        low = min(left_index, right_index)
        for wall in self._level_spec().walls:
            if wall.left_column == low:
                return wall
        return None

    def _can_cross(self, from_index: int, to_index: int) -> bool:
        wall = self._wall_between(from_index, to_index)
        if wall is None:
            return True
        return self._columns[from_index] > wall.height and self._columns[to_index] > wall.height

    def _intended_destination(self, ball_idx: int) -> int | str | None:
        position = self._ball_positions[ball_idx]
        if isinstance(position, str):
            return None

        candidates: list[tuple[int, int | str]] = []
        current_height = self._columns[position]

        left_index = position - 1
        right_index = position + 1
        if left_index >= 0 and self._columns[left_index] < current_height and self._can_cross(position, left_index):
            candidates.append((self._columns[left_index], left_index))
        if (
            right_index < len(self._columns)
            and self._columns[right_index] < current_height
            and self._can_cross(position, right_index)
        ):
            candidates.append((self._columns[right_index], right_index))

        spec = self._level_spec()
        if position == 0 and spec.left_basin is not None:
            candidates.append((0, "left_basin"))
        if position == len(self._columns) - 1 and spec.right_basin is not None:
            candidates.append((0, "right_basin"))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], 0 if item[1] == "left_basin" or item[1] == 0 else 1))
        return candidates[0][1]

    def _basin_matches(self, side: str, ball_idx: int) -> bool:
        spec = self._level_spec()
        basin = spec.left_basin if side == "left" else spec.right_basin
        if basin is None:
            return False
        return basin.color == spec.balls[ball_idx].color

    def _advance_running_tick(self) -> None:
        planned = [self._intended_destination(ball_idx) for ball_idx in range(len(self._ball_positions))]
        moved_any = False

        for ball_idx, destination in enumerate(planned):
            if destination is None:
                continue
            moved_any = True
            self._ball_positions[ball_idx] = destination
            if destination == "left_basin":
                if not self._basin_matches("left", ball_idx):
                    self._lose_level()
                    return
                self._ball_captured[ball_idx] = True
            elif destination == "right_basin":
                if not self._basin_matches("right", ball_idx):
                    self._lose_level()
                    return
                self._ball_captured[ball_idx] = True

        if all(self._ball_captured):
            self._mode = MODE_SUCCESS
            self._sync_board()
            return

        if not moved_any:
            self._lose_level()
            return

        self._sync_board()

    def _handle_edit_action(self) -> None:
        action_id = int(self.action.id.value)
        if action_id in {1, 2, 3, 4}:
            self.complete_action()
            return

        if action_id == 5:
            if not self._spend_budget():
                self._lose_level()
                self.complete_action()
                return
            self._mode = MODE_RUNNING
            self._sync_board()
            self.complete_action()
            return

        if action_id != 6:
            self.complete_action()
            return

        click_x = int(self.action.data.get("x", -1))
        click_y = int(self.action.data.get("y", -1))
        if self._clicked_launch(click_x, click_y):
            if not self._spend_budget():
                self._lose_level()
                self.complete_action()
                return
            self._mode = MODE_RUNNING
            self._sync_board()
            self.complete_action()
            return

        column_index = self._column_index_at(click_x, click_y)
        if column_index is None:
            self.complete_action()
            return

        if not self._spend_budget():
            self._lose_level()
            self.complete_action()
            return

        self._columns[column_index] = (self._columns[column_index] % 4) + 1
        if self._remaining_budget == 0:
            self._lose_level()
            self.complete_action()
            return
        self._sync_board()
        self.complete_action()

    def step(self) -> None:
        if self._mode == MODE_EDIT:
            self._handle_edit_action()
            return

        if self._mode == MODE_RUNNING:
            self._advance_running_tick()
            self.complete_action()
            return

        if self._mode == MODE_SUCCESS:
            self.next_level()
            self.complete_action()
            return

        self.complete_action()

    def _get_valid_actions(self) -> list[ActionInput]:
        if self._mode == MODE_EDIT:
            actions = [ActionInput(id=GameAction.ACTION1), ActionInput(id=GameAction.ACTION5)]
            for x0, x1 in COLUMN_X_SPANS:
                actions.append(ActionInput(id=GameAction.ACTION6, data={"x": (x0 + x1) // 2, "y": 30}))
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": 58, "y": 4}))
            return actions
        return [ActionInput(id=GameAction.ACTION1)]
