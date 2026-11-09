from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64

COLOR_BACKGROUND = 4
COLOR_BACKING = 3
COLOR_BRACKET_OFF = 2
COLOR_ZERO = 9
COLOR_ZERO_HIGHLIGHT = 10
COLOR_ONE = 12
COLOR_ONE_HIGHLIGHT = 11
COLOR_MATCH = 14
COLOR_FAILURE = 8

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    current_rows: tuple[str, ...]
    target_rows: tuple[str, ...]
    origin: tuple[int, int]
    module_size: int
    gap: int
    baseline_actions: int

    @property
    def step_budget(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec("Lone Dissenter", ("01", "11"), ("11", "11"), (10, 10), 20, 4, 1),
    LevelSpec("The Cell Does Not Vote", ("1101011", "1100011"), ("1111111", "1111111"), (4, 24), 7, 1, 5),
    LevelSpec("Split Center", ("101", "101", "101"), ("111", "111", "111"), (10, 10), 13, 2, 3),
    LevelSpec(
        "One-Way Front",
        ("01010101", "10101010", "01010101", "10101010", "01010101", "10101010", "01010101", "10101010"),
        ("10101011", "10101000", "00101011", "11101010", "00101011", "10101000", "01101011", "10101010"),
        (4, 4),
        6,
        1,
        27,
    ),
    LevelSpec(
        "Junctions First",
        (
            "101010101",
            "010101010",
            "101010101",
            "010101010",
            "101010101",
            "010101010",
            "101010101",
            "010101010",
            "101010101",
        ),
        (
            "010101011",
            "010001010",
            "010101010",
            "010101110",
            "010101010",
            "011101010",
            "010101010",
            "010100010",
            "110101011",
        ),
        (5, 5),
        5,
        1,
        46,
    ),
    LevelSpec(
        "Keep a Spark",
        (
            "010101010",
            "101010101",
            "010101010",
            "101010101",
            "010101010",
            "101010101",
            "010101010",
            "101010101",
            "010101010",
        ),
        (
            "101010100",
            "000000000",
            "100010001",
            "000000000",
            "101110101",
            "000000000",
            "100010001",
            "000000000",
            "001010100",
        ),
        (5, 5),
        5,
        1,
        57,
    ),
    LevelSpec(
        "The Living Tree",
        (
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
        ),
        (
            "1000100011",
            "0000000000",
            "1010101010",
            "0000000000",
            "1000001000",
            "0001000001",
            "1000000000",
            "0001010101",
            "1000000000",
            "1101000100",
        ),
        (2, 2),
        5,
        1,
        69,
    ),
)


def _matrix(rows: tuple[str, ...]) -> list[list[int]]:
    return [[int(value) for value in row] for row in rows]


@cache
def _cell_pixels(scale: int, state: int, target: int, matched: bool, highlighted: bool) -> np.ndarray:
    pixels = np.full((5, 5), COLOR_BACKGROUND, dtype=np.int8)
    pixels[0, 2:5] = COLOR_BACKING
    pixels[1:3, 0:5] = COLOR_BACKING
    pixels[3:5, 0:4] = COLOR_BACKING

    state_color = COLOR_ONE if state else COLOR_ZERO
    if highlighted:
        state_color = COLOR_ONE_HIGHLIGHT if state else COLOR_ZERO_HIGHLIGHT
    pixels[2:5, 0:3] = state_color
    if state == 0:
        pixels[3, 1] = COLOR_BACKGROUND

    if target:
        pixels[0:2, 3:5] = COLOR_ONE
    else:
        pixels[0, 3:5] = COLOR_ZERO
        pixels[1, 3] = COLOR_ZERO
        pixels[1, 4] = COLOR_BACKGROUND

    bracket = COLOR_MATCH if matched else COLOR_BRACKET_OFF
    pixels[0, 2] = bracket
    pixels[1, 2] = bracket
    pixels[2, 3:5] = bracket
    return np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)


class MajorityCellsDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: MajorityCells | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        spec = game.level_spec
        feedback = game.feedback
        feedback_neighbors = set(feedback.get("neighbors", ())) if feedback else set()
        snapshot = feedback.get("snapshot") if feedback else None

        for row in range(game.rows):
            for col in range(game.cols):
                state = game.current[row][col]
                highlight = False
                special = None
                if feedback and (row, col) in feedback_neighbors:
                    state = snapshot[row][col]
                    highlight = True
                if feedback and (row, col) == feedback.get("cell"):
                    special = str(feedback["kind"])
                    if special in {"tie", "changed"}:
                        state = snapshot[row][col]
                self._draw_cell(frame, game, row, col, state, highlight, special)

        if feedback and feedback.get("kind") == "changed":
            self._draw_vote_flows(frame, game, feedback)
            self._draw_change_fill(frame, game, feedback)
        if game.board_pulse is not None:
            self._draw_board_pulse(frame, game, game.board_pulse)
        self._draw_step_bar(frame, game.remaining_steps, spec.step_budget)
        return frame

    def _module_top_left(self, spec: LevelSpec, row: int, col: int) -> tuple[int, int]:
        stride = spec.module_size + spec.gap
        return spec.origin[0] + col * stride, spec.origin[1] + row * stride

    def _draw_cell(
        self,
        frame: np.ndarray,
        game: MajorityCells,
        row: int,
        col: int,
        state: int,
        highlighted: bool,
        special: str | None,
    ) -> None:
        spec = game.level_spec
        module = spec.module_size
        scale = max(1, module // 5)
        sprite_size = 5 * scale
        module_x, module_y = self._module_top_left(spec, row, col)
        x0 = module_x + (module - sprite_size) // 2
        y0 = module_y + (module - sprite_size) // 2

        target = game.target[row][col]
        matched = game.current[row][col] == target
        frame[y0 : y0 + sprite_size, x0 : x0 + sprite_size] = _cell_pixels(scale, state, target, matched, highlighted)
        if not matched and special is None:
            frame[y0, x0 : x0 + sprite_size] = COLOR_ONE_HIGHLIGHT
            frame[y0 + sprite_size - 1, x0 : x0 + sprite_size] = COLOR_ONE_HIGHLIGHT
            frame[y0 : y0 + sprite_size, x0] = COLOR_ONE_HIGHLIGHT
            frame[y0 : y0 + sprite_size, x0 + sprite_size - 1] = COLOR_ONE_HIGHLIGHT

        sx = x0
        sy = y0 + 2 * scale
        sw = 3 * scale
        sh = 3 * scale
        if special == "tie":
            split = sx + sw // 2
            frame[sy : sy + sh, sx:split] = COLOR_ZERO
            frame[sy : sy + sh, split : sx + sw] = COLOR_ONE
        elif special == "same":
            frame[sy, sx : sx + sw] = COLOR_MATCH
            frame[sy : sy + sh, sx] = COLOR_MATCH
            frame[sy + sh - 1, sx : sx + sw] = COLOR_MATCH
            frame[sy : sy + sh, sx + sw - 1] = COLOR_MATCH

    def _draw_vote_flows(self, frame: np.ndarray, game: MajorityCells, feedback: dict[str, object]) -> None:
        spec = game.level_spec
        row, col = feedback["cell"]
        winner = int(feedback["winner"])
        color = COLOR_ONE_HIGHLIGHT if winner else COLOR_ZERO_HIGHLIGHT
        cx, cy = self._module_top_left(spec, row, col)
        module = spec.module_size
        for nr, nc in feedback["winning_neighbors"]:
            nx, ny = self._module_top_left(spec, nr, nc)
            if nr == row:
                start = min(nx + module, cx + module)
                end = max(nx, cx)
                y = cy + module // 2
                frame[y, start:end] = color
            else:
                start = min(ny + module, cy + module)
                end = max(ny, cy)
                x = cx + module // 2
                frame[start:end, x] = color

    def _draw_change_fill(self, frame: np.ndarray, game: MajorityCells, feedback: dict[str, object]) -> None:
        spec = game.level_spec
        row, col = feedback["cell"]
        winner = int(feedback["winner"])
        color = COLOR_ONE_HIGHLIGHT if winner else COLOR_ZERO_HIGHLIGHT
        module_x, module_y = self._module_top_left(spec, row, col)
        scale = max(1, spec.module_size // 5)
        sprite_size = 5 * scale
        x0 = module_x + (spec.module_size - sprite_size) // 2
        y0 = module_y + (spec.module_size - sprite_size) // 2
        sx = x0
        sy = y0 + 2 * scale
        size = 3 * scale
        for nr, nc in feedback["winning_neighbors"]:
            if nr < row:
                frame[sy : sy + scale, sx : sx + size] = color
            elif nr > row:
                frame[sy + size - scale : sy + size, sx : sx + size] = color
            elif nc < col:
                frame[sy : sy + size, sx : sx + scale] = color
            else:
                frame[sy : sy + size, sx + size - scale : sx + size] = color

    def _draw_board_pulse(self, frame: np.ndarray, game: MajorityCells, color: int) -> None:
        spec = game.level_spec
        width = game.cols * spec.module_size + (game.cols - 1) * spec.gap
        height = game.rows * spec.module_size + (game.rows - 1) * spec.gap
        left = max(0, spec.origin[0] - 1)
        top = max(0, spec.origin[1] - 1)
        right = min(63, spec.origin[0] + width)
        bottom = min(61, spec.origin[1] + height)
        frame[top, left : right + 1] = color
        frame[bottom, left : right + 1] = color
        frame[top : bottom + 1, left] = color
        frame[top : bottom + 1, right] = color

    def _draw_step_bar(self, frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[62:64, :] = COLOR_BACKING
        filled = int(VIEW_SIZE * max(0, remaining) / max(1, budget))
        if filled:
            frame[62:64, :filled] = COLOR_ONE_HIGHLIGHT


class MajorityCells(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        display = MajorityCellsDisplay()
        levels = [
            Level(sprites=[], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)
            for spec in LEVEL_SPECS
        ]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [display])
        self.level_spec = LEVEL_SPECS[0]
        self.current: list[list[int]] = []
        self.target: list[list[int]] = []
        self.rows = 0
        self.cols = 0
        self.remaining_steps = 0
        self.undo_history: list[list[list[int]]] = []
        self.feedback: dict[str, object] | None = None
        self.board_pulse: int | None = None
        self._pending_resolution = False
        super().__init__("majority_cells-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        display.game = self

    def on_set_level(self, level: Level) -> None:
        self.level_spec = level.get_data("spec")
        self.current = _matrix(self.level_spec.current_rows)
        self.target = _matrix(self.level_spec.target_rows)
        self.rows = len(self.current)
        self.cols = len(self.current[0])
        self.remaining_steps = self.level_spec.step_budget
        self.undo_history = []
        self.feedback = None
        self.board_pulse = None
        self._pending_resolution = False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.feedback = None
            self.board_pulse = None
            self.complete_action()
            return

        if self._pending_resolution:
            self.feedback = None
            self._pending_resolution = False
            if self._is_solved():
                self.board_pulse = COLOR_MATCH
                self.next_level()
            elif self.remaining_steps <= 0:
                self.board_pulse = COLOR_FAILURE
                self.lose()
            self.complete_action()
            return

        self.board_pulse = None
        self.remaining_steps -= 1

        if self.action.id == GameAction.ACTION7:
            if self.undo_history:
                self.current = self.undo_history.pop()
            self.feedback = None
            self._finish_or_fail()
            return

        if self.action.id != GameAction.ACTION6:
            self.feedback = None
            self._finish_or_fail()
            return

        cell = self._cell_from_click(self.action.data)
        if cell is None:
            self.feedback = None
            self._finish_or_fail()
            return

        self.undo_history.append([row[:] for row in self.current])
        self._apply_click(*cell)
        self._pending_resolution = True

    def _finish_or_fail(self) -> None:
        if self._is_solved():
            self.board_pulse = COLOR_MATCH
            self.next_level()
        elif self.remaining_steps <= 0:
            self.board_pulse = COLOR_FAILURE
            self.lose()
        self.complete_action()

    def _cell_from_click(self, data: dict[str, object]) -> tuple[int, int] | None:
        try:
            x = int(data["x"])
            y = int(data["y"])
        except (KeyError, TypeError, ValueError):
            return None
        spec = self.level_spec
        local_x = x - spec.origin[0]
        local_y = y - spec.origin[1]
        if local_x < 0 or local_y < 0:
            return None
        stride = spec.module_size + spec.gap
        col, inside_x = divmod(local_x, stride)
        row, inside_y = divmod(local_y, stride)
        if row >= self.rows or col >= self.cols:
            return None
        if inside_x >= spec.module_size or inside_y >= spec.module_size:
            return None
        return row, col

    def _apply_click(self, row: int, col: int) -> None:
        snapshot = [line[:] for line in self.current]
        neighbors = []
        for dr, dc in ORTHOGONAL:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        ones = sum(snapshot[nr][nc] for nr, nc in neighbors)
        zeros = len(neighbors) - ones
        old = snapshot[row][col]

        if ones == zeros:
            kind = "tie"
            winner = old
        else:
            winner = int(ones > zeros)
            kind = "changed" if winner != old else "same"

        equal_before = self._equal_adjacencies(snapshot)
        self.current[row][col] = winner
        if winner != old:
            assert self._equal_adjacencies(self.current) > equal_before

        self.feedback = {
            "cell": (row, col),
            "neighbors": tuple(neighbors),
            "winning_neighbors": tuple((nr, nc) for nr, nc in neighbors if snapshot[nr][nc] == winner),
            "snapshot": snapshot,
            "kind": kind,
            "winner": winner,
        }

    def _equal_adjacencies(self, matrix: list[list[int]]) -> int:
        total = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if row + 1 < self.rows:
                    total += matrix[row][col] == matrix[row + 1][col]
                if col + 1 < self.cols:
                    total += matrix[row][col] == matrix[row][col + 1]
        return total

    def _is_solved(self) -> bool:
        return self.current == self.target


__all__ = ["MajorityCells"]
