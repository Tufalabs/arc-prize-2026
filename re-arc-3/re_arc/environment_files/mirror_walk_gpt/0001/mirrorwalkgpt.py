from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "mirror_walk_gpt-0001"
GRID_WIDTH = 64
GRID_HEIGHT = 64
BOARD_ROWS = 15
BOARD_COLS = 16
CELL_SIZE = 4
BOARD_TOP = 4

FLOOR = 0
FLOOR_ACCENT = 1
SPENT_BUDGET = 2
WALL = 4
WALL_ACCENT = 5
RIGHT_AVATAR = 6
RIGHT_AVATAR_ACCENT = 7
EXIT_CLOSED = 8
LEFT_AVATAR = 9
LEFT_AVATAR_ACCENT = 10
BUTTON = 11
FRAME = 12
DIVIDER = 13
OPEN = 14

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)
SPACE = int(GameAction.ACTION5.value)
CLICK = int(GameAction.ACTION6.value)

LEFT_ROOM_COLS = frozenset(range(1, 7))
RIGHT_ROOM_COLS = frozenset(range(9, 15))

LEFT_AVATAR_TILE = np.array([[0, 9, 9, 0], [9, 10, 10, 9], [9, 9, 9, 9], [0, 9, 0, 9]], dtype=np.int8)
RIGHT_AVATAR_TILE = np.array([[0, 6, 6, 0], [6, 7, 7, 6], [6, 6, 6, 6], [6, 0, 6, 0]], dtype=np.int8)
BUTTON_TILE = np.array([[0, 11, 11, 0], [11, 12, 12, 11], [11, 12, 12, 11], [0, 11, 11, 0]], dtype=np.int8)
BUTTON_PRESSED_TILE = np.array([[0, 14, 14, 0], [14, 11, 11, 14], [14, 11, 11, 14], [0, 14, 14, 0]], dtype=np.int8)
EXIT_CLOSED_TILE = np.array([[12, 12, 12, 12], [12, 8, 8, 12], [12, 8, 8, 12], [12, 12, 12, 12]], dtype=np.int8)
EXIT_OPEN_TILE = np.array([[12, 14, 14, 12], [12, 0, 0, 12], [12, 0, 0, 12], [12, 14, 14, 12]], dtype=np.int8)
DIVIDER_TILE_LEFT = np.array([[13, 4, 13, 4], [13, 4, 13, 4], [13, 4, 13, 4], [13, 4, 13, 4]], dtype=np.int8)
DIVIDER_TILE_RIGHT = np.array([[4, 13, 4, 13], [4, 13, 4, 13], [4, 13, 4, 13], [4, 13, 4, 13]], dtype=np.int8)

MOVE_DELTAS = {UP: ((-1, 0), (-1, 0)), DOWN: ((1, 0), (1, 0)), LEFT: ((0, -1), (0, 1)), RIGHT: ((0, 1), (0, -1))}


@dataclass(frozen=True)
class LevelSpec:
    left_start: tuple[int, int]
    right_start: tuple[int, int]
    exit_cell: tuple[int, int]
    budget: int
    walls: frozenset[tuple[int, int]]
    button_cell: tuple[int, int] | None = None


def _wall_tile(base: int = WALL, accent: int = WALL_ACCENT) -> np.ndarray:
    tile = np.full((CELL_SIZE, CELL_SIZE), int(base), dtype=np.int8)
    tile[1:3, 1:3] = np.int8(accent)
    return tile


def _floor_tile(row: int, col: int) -> np.ndarray:
    tile = np.full((CELL_SIZE, CELL_SIZE), FLOOR, dtype=np.int8)
    if (row + col) % 2 == 0:
        tile[3, 3] = np.int8(FLOOR_ACCENT)
    else:
        tile[0, 0] = np.int8(FLOOR_ACCENT)
    return tile


def _build_level_specs() -> list[LevelSpec]:
    border = {(row, col) for row in range(BOARD_ROWS) for col in range(BOARD_COLS) if row in {0, 14} or col in {0, 15}}
    divider = {(row, col) for row in range(BOARD_ROWS) for col in (7, 8)}

    return [
        LevelSpec(
            left_start=(12, 2),
            right_start=(12, 13),
            exit_cell=(5, 5),
            budget=30,
            walls=frozenset(border | divider | {(9, 12), (10, 12), (11, 12), (12, 12), (13, 12)}),
        ),
        LevelSpec(
            left_start=(12, 2),
            right_start=(12, 13),
            exit_cell=(6, 6),
            button_cell=(10, 11),
            budget=30,
            walls=frozenset(border | divider | {(10, 10), (9, 11)}),
        ),
        LevelSpec(
            left_start=(12, 3),
            right_start=(12, 12),
            exit_cell=(4, 4),
            button_cell=(8, 10),
            budget=30,
            walls=frozenset(border | divider | {(9, 5), (10, 5), (11, 5), (12, 5), (8, 9), (7, 10)}),
        ),
    ]


LEVEL_SPECS = _build_level_specs()


def _board_sprite() -> Sprite:
    return Sprite(
        np.full((GRID_HEIGHT, GRID_WIDTH), FLOOR, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )


def _level_from_spec(index: int, spec: LevelSpec) -> Level:
    return Level(
        name=f"Mirror Walk {index + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[_board_sprite()],
        data={
            "index": int(index),
            "left_start": tuple(spec.left_start),
            "right_start": tuple(spec.right_start),
            "exit_cell": tuple(spec.exit_cell),
            "button_cell": None if spec.button_cell is None else tuple(spec.button_cell),
            "budget": int(spec.budget),
            "walls": [tuple(cell) for cell in sorted(spec.walls)],
        },
    )


class MirrorWalkGpt(ARCBaseGame):
    _board: Sprite
    _left_pos: tuple[int, int]
    _right_pos: tuple[int, int]
    _exit_open: bool
    _failed: bool
    _button_pressed: bool
    _budget_remaining: int
    _budget_max: int
    _level_spec: LevelSpec

    def __init__(self) -> None:
        levels = [_level_from_spec(index, spec) for index, spec in enumerate(LEVEL_SPECS)]
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, FLOOR, FLOOR),
            win_score=len(levels),
            available_actions=[UP, DOWN, LEFT, RIGHT, SPACE, CLICK],
        )

    def on_set_level(self, level: Level) -> None:
        index = int(level.get_data("index"))
        self._board = self.current_level.get_sprites_by_name("board")[0]
        self._level_spec = LEVEL_SPECS[index]
        self._reset_current_level_state()

    def _reset_current_level_state(self) -> None:
        self._left_pos = tuple(self._level_spec.left_start)
        self._right_pos = tuple(self._level_spec.right_start)
        self._budget_max = int(self._level_spec.budget)
        self._budget_remaining = int(self._level_spec.budget)
        self._button_pressed = False
        self._failed = False
        self._exit_open = self._level_spec.button_cell is None
        self._render_board()

    def _logical_to_pixels(self, row: int, col: int) -> tuple[int, int]:
        return BOARD_TOP + row * CELL_SIZE, col * CELL_SIZE

    def _paint_tile(self, frame: np.ndarray, row: int, col: int, tile: np.ndarray) -> None:
        py, px = self._logical_to_pixels(row, col)
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = tile

    def _is_border(self, row: int, col: int) -> bool:
        return row in {0, BOARD_ROWS - 1} or col in {0, BOARD_COLS - 1}

    def _is_divider(self, row: int, col: int) -> bool:
        del row
        return col in {7, 8}

    def _is_highlight_border(self, row: int, col: int) -> bool:
        del row, col
        return self._failed or (str(getattr(getattr(self, "_state", None), "name", "")).upper() == "WIN")

    def _border_tile(self) -> np.ndarray:
        if self._failed:
            return _wall_tile(EXIT_CLOSED, EXIT_CLOSED)
        if str(getattr(getattr(self, "_state", None), "name", "")).upper() == "WIN":
            return _wall_tile(OPEN, OPEN)
        return _wall_tile()

    def _render_board(self) -> None:
        frame = np.full((GRID_HEIGHT, GRID_WIDTH), FLOOR, dtype=np.int8)
        frame[0:4, :] = np.int8(FLOOR)

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self._is_border(row, col):
                    self._paint_tile(frame, row, col, self._border_tile())
                    continue
                if self._is_divider(row, col):
                    divider_tile = DIVIDER_TILE_LEFT if col == 7 else DIVIDER_TILE_RIGHT
                    self._paint_tile(frame, row, col, divider_tile)
                    continue
                if (row, col) in self._level_spec.walls:
                    self._paint_tile(frame, row, col, _wall_tile())
                    continue
                self._paint_tile(frame, row, col, _floor_tile(row, col))

        if self._level_spec.button_cell is not None:
            button_tile = BUTTON_PRESSED_TILE if self._button_pressed else BUTTON_TILE
            self._paint_tile(frame, self._level_spec.button_cell[0], self._level_spec.button_cell[1], button_tile)

        exit_tile = EXIT_OPEN_TILE if self._exit_open else EXIT_CLOSED_TILE
        self._paint_tile(frame, self._level_spec.exit_cell[0], self._level_spec.exit_cell[1], exit_tile)

        self._paint_tile(frame, self._left_pos[0], self._left_pos[1], LEFT_AVATAR_TILE)
        self._paint_tile(frame, self._right_pos[0], self._right_pos[1], RIGHT_AVATAR_TILE)

        for pip_idx in range(self._budget_max):
            x0 = 2 * pip_idx + 4
            color = OPEN if pip_idx < self._budget_remaining else SPENT_BUDGET
            frame[1:3, x0 : x0 + 2] = np.int8(color)

        self._board.pixels = frame

    def _cell_passable(self, cell: tuple[int, int], *, exit_open: bool) -> bool:
        row, col = cell
        if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
            return False
        if (row, col) in self._level_spec.walls:
            return False
        if cell == self._level_spec.exit_cell and not exit_open:
            return False
        return True

    def _step_positions(self, action_id: int) -> None:
        left_delta, right_delta = MOVE_DELTAS[action_id]
        prior_exit_open = bool(self._exit_open)

        left_target = (self._left_pos[0] + left_delta[0], self._left_pos[1] + left_delta[1])
        right_target = (self._right_pos[0] + right_delta[0], self._right_pos[1] + right_delta[1])

        if self._cell_passable(left_target, exit_open=prior_exit_open):
            self._left_pos = left_target
        if self._cell_passable(right_target, exit_open=prior_exit_open):
            self._right_pos = right_target

        self._button_pressed = (
            self._level_spec.button_cell is not None and self._right_pos == self._level_spec.button_cell
        )
        self._exit_open = self._level_spec.button_cell is None or self._button_pressed

    def step(self) -> None:
        state_name = str(getattr(getattr(self, "_state", None), "name", "")).upper()
        action_id = int(self.action.id.value)

        if state_name in {"WIN", "LOSE"}:
            self.complete_action()
            return

        if action_id == CLICK:
            self.complete_action()
            return

        if action_id == SPACE:
            self._reset_current_level_state()
            self.complete_action()
            return

        if action_id not in MOVE_DELTAS:
            self.complete_action()
            return

        self._budget_remaining -= 1
        self._step_positions(action_id)

        if self._left_pos == self._level_spec.exit_cell and self._exit_open:
            if self.level_index < len(LEVEL_SPECS) - 1:
                self.next_level()
            else:
                self.next_level()
                self._render_board()
            self.complete_action()
            return

        if self._budget_remaining <= 0:
            self._failed = True
            self.lose()
            self._render_board()
            self.complete_action()
            return

        self._render_board()
        self.complete_action()
