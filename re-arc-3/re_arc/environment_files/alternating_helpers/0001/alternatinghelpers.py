from __future__ import annotations

from typing import Final

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BACKGROUND_COLOR: Final[int] = 0
CELL_SIZE: Final[int] = 8
BOARD_SIZE: Final[int] = 8
FRAME_SIZE: Final[int] = CELL_SIZE * BOARD_SIZE

VOID: Final[str] = "void"
FLOOR: Final[str] = "floor"
ARROW_N: Final[str] = "arrow_n"
ARROW_E: Final[str] = "arrow_e"
ARROW_S: Final[str] = "arrow_s"
ARROW_W: Final[str] = "arrow_w"
BEACON_NONE: Final[str] = "beacon_none"
BEACON_E: Final[str] = "beacon_e"

RED: Final[str] = "red"
BLUE: Final[str] = "blue"

LEVEL_SPECS: Final[list[dict[str, object]]] = [
    {
        "name": "Level 1",
        "budget": 7,
        "active": RED,
        "red_start": (3, 3),
        "blue_start": (4, 5),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, ARROW_E, BEACON_NONE, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
        ),
    },
    {
        "name": "Level 2",
        "budget": 8,
        "active": RED,
        "red_start": (3, 3),
        "blue_start": (4, 6),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, VOID, VOID, ARROW_E, BEACON_E, ARROW_W, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
        ),
    },
    {
        "name": "Level 3",
        "budget": 16,
        "active": RED,
        "red_start": (1, 1),
        "blue_start": (6, 7),
        "rows": (
            (VOID, VOID, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, FLOOR, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, FLOOR, VOID, VOID, VOID, VOID, VOID, VOID),
            (VOID, ARROW_E, ARROW_E, ARROW_E, BEACON_E, ARROW_W, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, FLOOR, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, VOID, VOID, VOID),
            (VOID, VOID, VOID, VOID, ARROW_N, FLOOR, FLOOR, VOID),
        ),
    },
]

PASSABLE_TILES: Final[set[str]] = {FLOOR, ARROW_N, ARROW_E, ARROW_S, ARROW_W, BEACON_NONE, BEACON_E}
ARROW_DELTAS: Final[dict[str, tuple[int, int]]] = {
    ARROW_N: (0, -1),
    ARROW_E: (1, 0),
    ARROW_S: (0, 1),
    ARROW_W: (-1, 0),
    BEACON_E: (1, 0),
}
DIRS: Final[tuple[tuple[int, int], ...]] = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _blank_frame() -> np.ndarray:
    return np.full((FRAME_SIZE, FRAME_SIZE), BACKGROUND_COLOR, dtype=np.int8)


def _level_rows(level_index: int) -> tuple[tuple[str, ...], ...]:
    rows = LEVEL_SPECS[level_index]["rows"]
    return rows if isinstance(rows, tuple) else tuple(rows)


def _make_level(level_index: int) -> Level:
    frame_sprite = Sprite(_blank_frame(), name="frame", x=0, y=0, layer=0)
    spec = LEVEL_SPECS[level_index]
    return Level(
        sprites=[frame_sprite],
        grid_size=(FRAME_SIZE, FRAME_SIZE),
        data={"level_index": level_index},
        name=str(spec["name"]),
    )


levels = [_make_level(level_index) for level_index in range(len(LEVEL_SPECS))]


class AlternatingHelpers(ARCBaseGame):
    _frame_sprite: Sprite
    _level_idx: int
    _rows: tuple[tuple[str, ...], ...]
    _beacon: tuple[int, int]
    _red_pos: tuple[int, int]
    _blue_pos: tuple[int, int]
    _active: str
    _remaining_moves: int
    _pending_level_continue: bool
    _display_state: str

    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            "alternating_helpers",
            levels,
            Camera(0, 0, FRAME_SIZE, FRAME_SIZE, BACKGROUND_COLOR, BACKGROUND_COLOR),
            False,
            len(levels),
            [6],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._frame_sprite = level.get_sprites_by_name("frame")[0]
        raw_level_idx = level.get_data("level_index")
        self._level_idx = int(raw_level_idx if raw_level_idx is not None else 0)
        spec = LEVEL_SPECS[self._level_idx]
        self._rows = _level_rows(self._level_idx)
        self._beacon = self._find_beacon()
        self._red_pos = tuple(spec["red_start"])  # type: ignore[arg-type]
        self._blue_pos = tuple(spec["blue_start"])  # type: ignore[arg-type]
        self._active = str(spec["active"])
        self._remaining_moves = int(spec["budget"])
        self._pending_level_continue = False
        self._display_state = "playing"
        self._refresh_frame()

    def _find_beacon(self) -> tuple[int, int]:
        for y, row in enumerate(self._rows):
            for x, tile in enumerate(row):
                if tile in {BEACON_NONE, BEACON_E}:
                    return (x, y)
        raise ValueError("Beacon missing from level definition.")

    def _refresh_frame(self) -> None:
        self._frame_sprite.pixels = self._render_board()

    def _render_board(self) -> np.ndarray:
        frame = _blank_frame()
        self._draw_board(frame)
        self._draw_move_pips(frame)
        self._draw_valid_move_markers(frame)
        self._draw_pawns(frame)
        if self._display_state == "won":
            self._draw_win_pulse(frame)
        elif self._display_state == "lost":
            self._draw_loss_accent(frame)
        return frame

    def _draw_board(self, frame: np.ndarray) -> None:
        for cy, row in enumerate(self._rows):
            for cx, tile in enumerate(row):
                if tile == VOID:
                    continue
                self._draw_tile_base(frame, cx, cy)
                if tile in {ARROW_N, ARROW_E, ARROW_S, ARROW_W}:
                    self._draw_arrow(frame, cx, cy, tile)
                elif tile in {BEACON_NONE, BEACON_E}:
                    self._draw_beacon(frame, cx, cy, with_outflow=(tile == BEACON_E))

    def _draw_tile_base(self, frame: np.ndarray, cx: int, cy: int) -> None:
        left = cx * CELL_SIZE
        top = cy * CELL_SIZE
        frame[top : top + CELL_SIZE, left : left + CELL_SIZE] = 1
        for dy in range(CELL_SIZE):
            for dx in range(CELL_SIZE):
                at_edge = dx in {0, CELL_SIZE - 1} or dy in {0, CELL_SIZE - 1}
                if not at_edge:
                    continue
                nx = cx + (-1 if dx == 0 else 1 if dx == CELL_SIZE - 1 else 0)
                ny = cy + (-1 if dy == 0 else 1 if dy == CELL_SIZE - 1 else 0)
                if dx in {0, CELL_SIZE - 1} and dy in {0, CELL_SIZE - 1}:
                    nx = cx + (-1 if dx == 0 else 1)
                    ny = cy + (-1 if dy == 0 else 1)
                if not self._in_bounds(nx, ny) or self._rows[ny][nx] == VOID:
                    frame[top + dy, left + dx] = 3

    def _draw_arrow(self, frame: np.ndarray, cx: int, cy: int, tile: str) -> None:
        points_by_tile = {
            ARROW_N: ((3, 1), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (3, 4), (3, 5)),
            ARROW_S: ((3, 1), (3, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (2, 4), (3, 4), (4, 4), (3, 5)),
            ARROW_E: ((1, 3), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (5, 3)),
            ARROW_W: ((1, 3), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 3), (5, 3)),
        }
        left = cx * CELL_SIZE
        top = cy * CELL_SIZE
        for dx, dy in points_by_tile[tile]:
            frame[top + dy, left + dx] = 3

    def _draw_beacon(self, frame: np.ndarray, cx: int, cy: int, *, with_outflow: bool) -> None:
        left = cx * CELL_SIZE
        top = cy * CELL_SIZE
        ring_color = 4 if self._display_state == "lost" else 12
        core_color = 4 if self._display_state == "lost" else 11
        ring_points = (
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (1, 1),
            (6, 1),
            (0, 2),
            (7, 2),
            (0, 3),
            (7, 3),
            (0, 4),
            (7, 4),
            (1, 5),
            (6, 5),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
        )
        core_points = (
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
        )
        for dx, dy in ring_points:
            frame[top + dy, left + dx] = ring_color
        for dx, dy in core_points:
            frame[top + dy, left + dx] = core_color
        if with_outflow:
            for dx, dy in ((6, 2), (7, 2), (6, 3), (7, 3), (6, 4), (7, 4), (5, 3)):
                frame[top + dy, left + dx] = ring_color

    def _draw_move_pips(self, frame: np.ndarray) -> None:
        total_slots = 16
        for idx in range(total_slots):
            left = 1 + idx * 4
            top = 1
            if self._display_state == "lost":
                color = 8
            elif idx < self._remaining_moves:
                color = 12 if self._remaining_moves <= 2 else 14
            else:
                color = 4
            frame[top : top + 3, left : left + 3] = color

    def _draw_valid_move_markers(self, frame: np.ndarray) -> None:
        if self._display_state != "playing":
            return
        active_pos = self._red_pos if self._active == RED else self._blue_pos
        other_pos = self._blue_pos if self._active == RED else self._red_pos
        for dx, dy in DIRS:
            target = (active_pos[0] + dx, active_pos[1] + dy)
            if not self._is_manual_move_valid(target, active_pos, other_pos):
                continue
            left = target[0] * CELL_SIZE
            top = target[1] * CELL_SIZE
            for offset_x, offset_y in ((0, 0), (6, 0), (0, 6), (6, 6)):
                frame[top + offset_y : top + offset_y + 2, left + offset_x : left + offset_x + 2] = 14

    def _draw_pawns(self, frame: np.ndarray) -> None:
        if self._red_pos == self._blue_pos == self._beacon:
            self._draw_combined_beacon_pawn(frame, self._beacon)
            return
        self._draw_single_pawn(frame, self._red_pos, RED, active=(self._active == RED))
        self._draw_single_pawn(frame, self._blue_pos, BLUE, active=(self._active == BLUE))

    def _draw_single_pawn(self, frame: np.ndarray, pos: tuple[int, int], pawn: str, *, active: bool) -> None:
        left = pos[0] * CELL_SIZE
        top = pos[1] * CELL_SIZE
        fill = 8 if pawn == RED else 9
        outline = 13 if pawn == RED else 10
        outline_points = (
            (2, 1),
            (3, 1),
            (1, 2),
            (4, 2),
            (0, 3),
            (5, 3),
            (0, 4),
            (5, 4),
            (1, 5),
            (4, 5),
            (2, 6),
            (3, 6),
        )
        fill_points = ((2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (4, 3), (1, 4), (2, 4), (3, 4), (4, 4), (2, 5), (3, 5))
        for dx, dy in outline_points:
            frame[top + dy, left + dx] = outline
        for dx, dy in fill_points:
            frame[top + dy, left + dx] = fill
        if active:
            for dx, dy in (
                (0, 0),
                (1, 0),
                (0, 1),
                (6, 0),
                (7, 0),
                (7, 1),
                (0, 6),
                (0, 7),
                (1, 7),
                (6, 7),
                (7, 6),
                (7, 7),
            ):
                frame[top + dy, left + dx] = 15

    def _draw_combined_beacon_pawn(self, frame: np.ndarray, pos: tuple[int, int]) -> None:
        left = pos[0] * CELL_SIZE
        top = pos[1] * CELL_SIZE
        self._draw_beacon(frame, pos[0], pos[1], with_outflow=(self._rows[pos[1]][pos[0]] == BEACON_E))
        for dy in range(1, 7):
            for dx in range(1, 7):
                if dx <= 3:
                    frame[top + dy, left + dx] = 8 if 1 < dx < 3 and 2 <= dy <= 5 else 13
                else:
                    frame[top + dy, left + dx] = 9 if 4 < dx < 6 and 2 <= dy <= 5 else 10

    def _draw_win_pulse(self, frame: np.ndarray) -> None:
        left = self._beacon[0] * CELL_SIZE
        top = self._beacon[1] * CELL_SIZE
        for dx in range(CELL_SIZE):
            frame[top, left + dx] = 14
            frame[top + CELL_SIZE - 1, left + dx] = 14
        for dy in range(CELL_SIZE):
            frame[top + dy, left] = 14
            frame[top + dy, left + CELL_SIZE - 1] = 14

    def _draw_loss_accent(self, frame: np.ndarray) -> None:
        bx, by = self._beacon
        left = bx * CELL_SIZE
        top = by * CELL_SIZE
        for offset in range(CELL_SIZE):
            frame[top + offset, left + offset] = 8
            frame[top + offset, left + (CELL_SIZE - 1 - offset)] = 8

    def _in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE

    def _tile_at(self, pos: tuple[int, int]) -> str:
        return self._rows[pos[1]][pos[0]]

    def _is_passable(self, pos: tuple[int, int]) -> bool:
        return self._in_bounds(pos[0], pos[1]) and self._tile_at(pos) in PASSABLE_TILES

    def _is_manual_move_valid(
        self, target: tuple[int, int], active_pos: tuple[int, int], other_pos: tuple[int, int]
    ) -> bool:
        if not self._is_passable(target):
            return False
        if abs(target[0] - active_pos[0]) + abs(target[1] - active_pos[1]) != 1:
            return False
        if target == other_pos and target != self._beacon:
            return False
        return True

    def _action_click_cell(self) -> tuple[int, int] | None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id != int(GameAction.ACTION6.value):
            return None
        x = self.action.data.get("x")
        y = self.action.data.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        if not (0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE):
            return None
        return (x // CELL_SIZE, y // CELL_SIZE)

    def _active_positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._active == RED:
            return self._red_pos, self._blue_pos
        return self._blue_pos, self._red_pos

    def _set_active_position(self, pos: tuple[int, int]) -> None:
        if self._active == RED:
            self._red_pos = pos
        else:
            self._blue_pos = pos

    def _helper_step(self) -> None:
        helper = BLUE if self._active == RED else RED
        helper_pos = self._blue_pos if helper == BLUE else self._red_pos
        tile = self._tile_at(helper_pos)
        delta = ARROW_DELTAS.get(tile)
        if delta is None:
            return
        target = (helper_pos[0] + delta[0], helper_pos[1] + delta[1])
        if not self._is_passable(target):
            return
        active_pos = self._red_pos if self._active == RED else self._blue_pos
        if target == active_pos and target != self._beacon:
            return
        if helper == BLUE:
            self._blue_pos = target
        else:
            self._red_pos = target

    def _handle_pending_screen(self) -> bool:
        if self._pending_level_continue:
            self.next_level()
            self.complete_action()
            return True
        return False

    def step(self) -> None:
        if self._handle_pending_screen():
            return

        clicked_cell = self._action_click_cell()
        if clicked_cell is None:
            self._refresh_frame()
            self.complete_action()
            return

        active_pos, other_pos = self._active_positions()
        successful_action = False

        if self._is_manual_move_valid(clicked_cell, active_pos, other_pos):
            self._set_active_position(clicked_cell)
            successful_action = True
        elif clicked_cell == other_pos:
            self._active = BLUE if self._active == RED else RED
            successful_action = True

        if not successful_action:
            self._refresh_frame()
            self.complete_action()
            return

        self._remaining_moves -= 1

        if self._red_pos == self._blue_pos == self._beacon:
            self._display_state = "won"
            if self.is_last_level():
                self.next_level()
            else:
                self._pending_level_continue = True
            self._refresh_frame()
            self.complete_action()
            return

        self._helper_step()

        if self._remaining_moves <= 0:
            self._display_state = "lost"
            self._refresh_frame()
            self.lose()
        else:
            self._display_state = "playing"
            self._refresh_frame()
        self.complete_action()
