from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64
CELL_SIZE = 5
CELL_PITCH = 6

COLOR_CELL = 2
COLOR_FRAME = 3
COLOR_BACKGROUND = 4
COLOR_DARK = 5
COLOR_AFFECTED = 10
COLOR_LIT = 11
COLOR_PRESSED = 12
COLOR_COMPLETE = 14

ACTION_CLICK = 6
ACTION_UNDO = 7


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    solution: tuple[str, ...]
    replay_order: tuple[tuple[int, int], ...] = ()

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def baseline_actions(self) -> int:
        return sum(row.count("X") for row in self.solution)

    @property
    def action_budget(self) -> int:
        return 6 * self.baseline_actions


LEVEL_SPECS = (
    LevelSpec("Single Impression", ("010", "111", "010"), ("...", ".X.", "...")),
    LevelSpec(
        "Diagonal Echo",
        ("0100000", "1100000", "0010000", "0001000", "0000100", "0000011", "0000010"),
        (".......", ".X.....", "..X....", "...X...", "....X..", ".....X.", "......."),
    ),
    LevelSpec(
        "Dark Hub",
        ("0000000", "0001000", "0000000", "0100010", "0011100", "0000000", "0000000"),
        (".......", ".......", "...X...", "..XXX..", ".......", ".......", "......."),
        ((3, 3), (2, 3), (3, 2), (3, 4)),
    ),
    LevelSpec(
        "Broken Frame",
        ("0000000", "0011110", "0100010", "0100010", "0100010", "0111110", "0000000"),
        (".XXXXX.", "X.XXX.X", "XX.X..X", "XXX....", "XX...XX", "X...XX.", ".XX.X.."),
    ),
    LevelSpec(
        "Opposed C-Frames",
        ("11101110", "10000000", "10111111", "10000001", "10000001", "10111111", "10000000", "11111110"),
        ("XXXX..XX", "X.....X.", "X.XX.X..", "X.XXX.XX", "X.X..X.X", "X....XXX", "XX.X....", "...XXXXX"),
    ),
    LevelSpec(
        "Hooked Chamber",
        ("11111100", "11100000", "10111110", "10100010", "10100010", "10100010", "00111110", "00000010"),
        ("XXX.XXXX", "X.XXX.X.", "X..XXX..", "XXX.XXX.", ".XXXX.XX", "XXXXXX..", "X.X...XX", ".XXX.XX."),
    ),
    LevelSpec(
        "Crossed Hooks",
        (
            "1111111100",
            "1000000000",
            "1011111111",
            "1010000001",
            "1000000001",
            "1111110001",
            "0000000001",
            "0011111111",
        ),
        (
            "XXX.XXXXXX",
            "X.XXX...X.",
            "XXXXX.X...",
            ".XXX.X...X",
            "XXXXXX..X.",
            "X...XXXXXX",
            "XX.X.XXX.X",
            "X..XXX.XXX",
        ),
    ),
)


def _blank_board_sprite(name: str) -> Sprite:
    return Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8),
        name=name,
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )


class CrossPress(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(
                sprites=[_blank_board_sprite(f"board_{index}")],
                grid_size=(VIEW_SIZE, VIEW_SIZE),
                data={"spec": spec},
                name=spec.name,
            )
            for index, spec in enumerate(LEVEL_SPECS)
        ]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, background=COLOR_BACKGROUND, letter_box=COLOR_BACKGROUND)
        super().__init__("cross_press-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.lit = [[value == "1" for value in row] for row in self.spec.rows]
        self.solved = False
        self.selected_cell = None
        self.last_pressed: tuple[int, int] | None = None
        self.last_affected: set[tuple[int, int]] = set()
        self.remaining_steps = self.spec.action_budget
        self.step_budget = self.spec.action_budget
        self._undo_history: list[tuple[tuple[bool, ...], ...]] = []
        self._board = level.get_sprites_by_tag("board")[0]
        self._sync_visuals()

    def _board_origin(self) -> tuple[int, int]:
        content_width = CELL_PITCH * self.spec.width - 1
        content_height = CELL_PITCH * self.spec.height - 1
        return (VIEW_SIZE - content_width) // 2, (VIEW_SIZE - content_height) // 2

    def _cell_from_click(self, x: int, y: int) -> tuple[int, int] | None:
        x0, y0 = self._board_origin()
        local_x = x - x0
        local_y = y - y0
        if local_x < 0 or local_y < 0:
            return None
        col, within_x = divmod(local_x, CELL_PITCH)
        row, within_y = divmod(local_y, CELL_PITCH)
        if row >= self.spec.height or col >= self.spec.width:
            return None
        if within_x >= CELL_SIZE or within_y >= CELL_SIZE:
            return None
        return row, col

    def _affected(self, row: int, col: int) -> set[tuple[int, int]]:
        candidates = ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        return {
            (affected_row, affected_col)
            for affected_row, affected_col in candidates
            if 0 <= affected_row < self.spec.height and 0 <= affected_col < self.spec.width
        }

    def _snapshot(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(tuple(row) for row in self.lit)

    def _restore(self, snapshot: tuple[tuple[bool, ...], ...]) -> None:
        self.lit = [list(row) for row in snapshot]
        self.solved = not any(any(row) for row in self.lit)

    def _press(self, row: int, col: int) -> None:
        self._undo_history.append(self._snapshot())
        affected = self._affected(row, col)
        previous = self._snapshot()
        for affected_row, affected_col in affected:
            self.lit[affected_row][affected_col] = not previous[affected_row][affected_col]
        self.last_pressed = (row, col)
        self.last_affected = affected
        self.solved = not any(any(board_row) for board_row in self.lit)

    def _draw_step_bar(self, pixels: np.ndarray) -> None:
        left, right = 10, 54
        top, bottom = 60, 63
        pixels[top:bottom, left:right] = COLOR_CELL
        interior_width = right - left - 2
        pixels[top + 1, left + 1 : right - 1] = COLOR_DARK
        filled = int(interior_width * self.remaining_steps / max(1, self.step_budget))
        if filled:
            pixels[top + 1, left + 1 : left + 1 + filled] = COLOR_PRESSED

    def _sync_visuals(self) -> None:
        pixels = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)
        x0, y0 = self._board_origin()
        content_width = CELL_PITCH * self.spec.width - 1
        content_height = CELL_PITCH * self.spec.height - 1
        frame_color = COLOR_COMPLETE if self.solved else COLOR_FRAME
        pixels[y0 - 1, x0 - 1 : x0 + content_width + 1] = frame_color
        pixels[y0 + content_height, x0 - 1 : x0 + content_width + 1] = frame_color
        pixels[y0 - 1 : y0 + content_height + 1, x0 - 1] = frame_color
        pixels[y0 - 1 : y0 + content_height + 1, x0 + content_width] = frame_color

        for row in range(self.spec.height):
            for col in range(self.spec.width):
                cell_x = x0 + CELL_PITCH * col
                cell_y = y0 + CELL_PITCH * row
                perimeter = COLOR_COMPLETE if self.solved else COLOR_CELL
                if not self.solved and (row, col) == self.last_pressed:
                    perimeter = COLOR_PRESSED
                elif not self.solved and (row, col) in self.last_affected:
                    perimeter = COLOR_AFFECTED
                pixels[cell_y : cell_y + CELL_SIZE, cell_x : cell_x + CELL_SIZE] = perimeter
                center = COLOR_LIT if self.lit[row][col] else COLOR_DARK
                pixels[cell_y + 1 : cell_y + 4, cell_x + 1 : cell_x + 4] = center

        self._draw_step_bar(pixels)
        self._board.pixels = pixels

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        self.last_pressed = None
        self.last_affected = set()
        self.remaining_steps = max(0, self.remaining_steps - 1)

        if self.action.id == GameAction.ACTION7:
            if self._undo_history:
                self._restore(self._undo_history.pop())
        elif self.action.id == GameAction.ACTION6:
            data = self.action.data
            cell = self._cell_from_click(int(data.get("x", 0)), int(data.get("y", 0)))
            if cell is not None:
                self._press(*cell)

        self._sync_visuals()
        if self.solved:
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps == 0:
            self.lose()
        self.complete_action()
