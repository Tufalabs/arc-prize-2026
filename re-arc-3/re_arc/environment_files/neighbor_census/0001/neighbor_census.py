from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
GAME_ID = "neighbor_census-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_TOGGLE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
AVAILABLE_ACTIONS = [1, 2, 3, 4, 5, 6, 7]
MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

BACKGROUND = 5
CELL_OFF = 2
CELL_ON = 9
NEIGHBOR_OFF = 3
NEIGHBOR_ON = 10
TARGET_EMPTY = 4
TARGET_POSITIVE = 11
TARGET_ZERO = 13
HALO_SHORT = 11
HALO_EXACT = 14
HALO_OVER = 8
SELECTION = 6
NUDGE = 7


@dataclass(frozen=True)
class LevelSpec:
    name: str
    target_rows: tuple[str, ...]
    state_rows: tuple[str, ...]
    cursor: tuple[int, int]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec("One Lit Spoke", ("1", "."), ("0", "0"), (1, 0), 1),
    LevelSpec("Zero Means None", (".0.",), ("101",), (0, 0), 4),
    LevelSpec("The Observer Switches", ("121",), ("000",), (0, 1), 6),
    LevelSpec("Four-Way Hinge", ("02..", "1233", "13..", "..21"), ("0110", "1000", "0010", "1111"), (1, 1), 25),
    LevelSpec(
        "Preserve and Reroute",
        ("22.3.", "...3.", ".42.3", "322.2", "..0.."),
        ("10010", "00101", "00000", "10111", "00100"),
        (2, 3),
        50,
    ),
    LevelSpec(
        "The Helpful Cell Must Go",
        ("...01.", "100...", "12.141", "..3...", ".34.40", ".22.0."),
        ("000110", "101001", "011000", "110010", "101010", "100001"),
        (2, 2),
        55,
    ),
    LevelSpec(
        "Double Census Exchange",
        (".03...", ".2.210", "1322.1", "....30", "...3..", "21.33."),
        ("001111", "110111", "010111", "010010", "001010", "001001"),
        (2, 2),
        64,
    ),
)


def _build_level(spec: LevelSpec) -> Level:
    height = len(spec.target_rows)
    width = len(spec.target_rows[0])
    if len(spec.state_rows) != height or any(len(row) != width for row in (*spec.target_rows, *spec.state_rows)):
        raise ValueError(f"Malformed rectangular level: {spec.name}")
    return Level(
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        sprites=[],
        name=spec.name,
        data={
            "target_rows": spec.target_rows,
            "state_rows": spec.state_rows,
            "cursor": spec.cursor,
            "baseline_actions": spec.baseline_actions,
            "step_budget": spec.baseline_actions * 6,
        },
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class NeighborCensusView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: NeighborCensus | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        scale = min(3, 60 // (10 * max(game.height, game.width)))
        origin_x = (64 - 10 * scale * game.width) // 2
        origin_y = (64 - 10 * scale * game.height) // 2

        for row in range(game.height):
            for col in range(game.width):
                base = self._cell_pixels(game, row, col)
                scaled = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)
                x0 = origin_x + 10 * scale * col
                y0 = origin_y + 10 * scale * row
                frame[y0 : y0 + 10 * scale, x0 : x0 + 10 * scale] = scaled

        self._draw_improving_cells(frame, game, scale, origin_x, origin_y)
        self._draw_selection(frame, game, scale, origin_x, origin_y)
        self._draw_budget(frame, game.remaining_steps, game.step_budget)
        if game.completion_visible:
            frame[0:2, 0:2] = HALO_EXACT
            frame[0:2, 62:64] = HALO_EXACT
            frame[62:64, 0:2] = HALO_EXACT
            frame[62:64, 62:64] = HALO_EXACT
        return frame

    def _draw_improving_cells(
        self, frame: np.ndarray, game: NeighborCensus, scale: int, origin_x: int, origin_y: int
    ) -> None:
        current_error = game.total_error()
        for row in range(game.height):
            for col in range(game.width):
                game.on[row][col] = not game.on[row][col]
                improved = game.total_error() < current_error
                game.on[row][col] = not game.on[row][col]
                if not improved:
                    continue
                x0 = origin_x + 10 * scale * col
                y0 = origin_y + 10 * scale * row
                frame[y0 : y0 + scale, x0 : x0 + 10 * scale] = HALO_EXACT
                frame[y0 + 9 * scale : y0 + 10 * scale, x0 : x0 + 10 * scale] = HALO_EXACT

    def _cell_pixels(self, game: NeighborCensus, row: int, col: int) -> np.ndarray:
        pixels = np.full((10, 10), BACKGROUND, dtype=np.int8)
        pixels[2:8, 2:8] = CELL_ON if game.on[row][col] else CELL_OFF
        target = game.targets[row][col]
        if target is None:
            return pixels

        count = game.live_count(row, col)
        slots = TARGET_ZERO if target == 0 else TARGET_EMPTY
        pixels[4:6, 3:7] = slots
        if target > 0:
            pixels[4:6, 3 : 3 + target] = TARGET_POSITIVE

        for dr, dc, ys, xs in (
            (-1, 0, slice(0, 2), slice(4, 6)),
            (0, 1, slice(4, 6), slice(8, 10)),
            (1, 0, slice(8, 10), slice(4, 6)),
            (0, -1, slice(4, 6), slice(0, 2)),
        ):
            nr, nc = row + dr, col + dc
            if 0 <= nr < game.height and 0 <= nc < game.width:
                pixels[ys, xs] = NEIGHBOR_ON if game.on[nr][nc] else NEIGHBOR_OFF

        # The semantic halo is layered over the lamps so an exact or overfull
        # perimeter remains visibly continuous at every cardinal crossing.
        if count == target:
            pixels[1, 1:9] = HALO_EXACT
            pixels[8, 1:9] = HALO_EXACT
            pixels[1:9, 1] = HALO_EXACT
            pixels[1:9, 8] = HALO_EXACT
        elif count > target:
            pixels[1, 1:9] = HALO_OVER
            pixels[8, 1:9] = HALO_OVER
            pixels[1:9, 1] = HALO_OVER
            pixels[1:9, 8] = HALO_OVER
        else:
            pixels[1, 1:3] = HALO_SHORT
            pixels[1:3, 1] = HALO_SHORT
            pixels[1, 7:9] = HALO_SHORT
            pixels[1:3, 8] = HALO_SHORT
            pixels[8, 1:3] = HALO_SHORT
            pixels[7:9, 1] = HALO_SHORT
            pixels[8, 7:9] = HALO_SHORT
            pixels[7:9, 8] = HALO_SHORT
        return pixels

    def _draw_selection(
        self, frame: np.ndarray, game: NeighborCensus, scale: int, origin_x: int, origin_y: int
    ) -> None:
        x0 = origin_x + 10 * scale * game.cursor_c
        y0 = origin_y + 10 * scale * game.cursor_r
        color = SELECTION
        shift_x = shift_y = 0
        if game.nudge_direction is not None:
            dr, dc = MOVE_DELTAS[game.nudge_direction]
            shift_x, shift_y = dc * scale, dr * scale
            color = NUDGE
        corners = ((0, 0), (1, 0), (0, 1), (9, 0), (8, 0), (9, 1), (0, 9), (1, 9), (0, 8), (9, 9), (8, 9), (9, 8))
        for px, py in corners:
            sx = x0 + px * scale + shift_x
            sy = y0 + py * scale + shift_y
            xa, xb = max(0, sx), min(64, sx + scale)
            ya, yb = max(0, sy), min(64, sy + scale)
            if xa < xb and ya < yb:
                frame[ya:yb, xa:xb] = color

    def _draw_budget(self, frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[63, 2:62] = NEIGHBOR_OFF
        filled = max(0, min(60, (60 * remaining + budget - 1) // budget))
        if filled:
            frame[63, 2 : 2 + filled] = NEIGHBOR_ON


class NeighborCensus(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = NeighborCensusView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND, [self._view]),
            False,
            len(levels),
            AVAILABLE_ACTIONS,
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        target_rows = tuple(self.current_level.get_data("target_rows"))
        state_rows = tuple(self.current_level.get_data("state_rows"))
        self.height = len(target_rows)
        self.width = len(target_rows[0])
        self.targets = [[None if cell == "." else int(cell) for cell in row] for row in target_rows]
        self.on = [[cell == "1" for cell in row] for row in state_rows]
        self.cursor_r, self.cursor_c = self.current_level.get_data("cursor")
        self.step_budget = int(self.current_level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.nudge_direction: int | None = None
        self.completion_visible = False
        self.undo_history: list[tuple[list[list[bool]], int, int]] = []

    def live_count(self, row: int, col: int) -> int:
        return sum(
            int(self.on[nr][nc])
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if 0 <= nr < self.height and 0 <= nc < self.width
        )

    def total_error(self) -> int:
        return sum(
            abs(self.live_count(row, col) - target)
            for row, targets in enumerate(self.targets)
            for col, target in enumerate(targets)
            if target is not None
        )

    def _is_solved(self) -> bool:
        return all(
            target is None or self.live_count(row, col) == target
            for row, targets in enumerate(self.targets)
            for col, target in enumerate(targets)
        )

    def _snapshot(self) -> None:
        self.undo_history.append(([row.copy() for row in self.on], self.cursor_r, self.cursor_c))

    def _spend_step(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        return self.remaining_steps == 0

    def step(self) -> None:
        action = _action_id(self.action.id)
        self.nudge_direction = None
        self.completion_visible = False

        if action == GameAction.RESET.value:
            self.complete_action()
            return

        expired = self._spend_step()
        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            nr, nc = self.cursor_r + dr, self.cursor_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                self._snapshot()
                self.cursor_r, self.cursor_c = nr, nc
            else:
                self.nudge_direction = action
        elif action == ACTION_TOGGLE:
            self._snapshot()
            self.on[self.cursor_r][self.cursor_c] = not self.on[self.cursor_r][self.cursor_c]
        elif action == ACTION_UNDO and self.undo_history:
            self.on, self.cursor_r, self.cursor_c = self.undo_history.pop()
            self.on = [row.copy() for row in self.on]
        # ACTION6 is intentionally a visible-state no-op.

        if self._is_solved():
            self.completion_visible = True
            self.next_level()
        elif expired:
            self.lose()
        self.complete_action()
