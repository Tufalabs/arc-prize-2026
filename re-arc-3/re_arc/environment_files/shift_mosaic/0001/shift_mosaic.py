from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 10

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

COLOR_FITTED = 0
COLOR_PLAQUE = 1
COLOR_OUTLINE = 2
COLOR_RAIL = 3
COLOR_BOARD = 4
COLOR_BACKGROUND = 5
COLOR_MAGENTA = 6
COLOR_FOCUS = 7
COLOR_BUMP = 8
COLOR_BLUE = 9
COLOR_SHIFT = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_COMPLETE = 15

ROW_HUES = (COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE, COLOR_MAGENTA)
MOTIFS = (
    (".##.", ".##.", ".##.", ".##."),
    ("....", "####", "####", "...."),
    ("##..", "##..", "##..", "####"),
    (".##.", "####", "####", ".##."),
    ("##..", "###.", ".###", "..##"),
)

ARROW_RIGHT = np.array(
    [[-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1], [1, 1, 1, 1, 1], [-1, -1, -1, 1, -1], [-1, -1, 1, -1, -1]], dtype=np.int8
)


@dataclass(frozen=True)
class LevelSpec:
    title: str
    rows: tuple[str, ...]
    initial_focus: tuple[str, int]
    baseline_actions: int

    @property
    def size(self) -> int:
        return len(self.rows)


LEVEL_SPECS = (
    LevelSpec("One Turn", ("A1 A2 A3", "B2 B3 B1", "C1 C2 C3"), ("R", 1), 1),
    LevelSpec("Two Axes", ("C1 A2 B3", "A1 B2 C3", "B1 C2 A3"), ("U", 0), 5),
    LevelSpec("The Crossing", ("A1 B1 A3", "B2 A2 B3", "C1 C2 C3"), ("R", 1), 13),
    LevelSpec("Borrowed Strip", ("A1 A2 A3 A4", "B4 C2 D3 C1", "C3 D4 B2 C4", "B3 B1 D2 D1"), ("R", 3), 91),
    LevelSpec(
        "Two Repair Fronts",
        ("A2 C2 B2 B3 E5", "B1 B4 A5 D3 B5", "C1 E4 E1 C4 C5", "D1 A1 D4 A4 D2", "E3 E2 C3 A3 D5"),
        ("R", 0),
        153,
    ),
    LevelSpec(
        "False Finish",
        ("D1 A4 A3 C3 A1", "E3 A2 B5 B4 C2", "B2 E5 C1 C4 A5", "B3 D2 D3 D4 D5", "E1 C5 E4 B1 E2"),
        ("R", 2),
        151,
    ),
    LevelSpec(
        "Closed Mosaic",
        ("E1 A3 E3 C5 D1", "B2 E2 D4 C2 A5", "A1 A2 C3 B4 C1", "D3 E5 B3 D5 B5", "A4 D2 E4 C4 B1"),
        ("R", 0),
        193,
    ),
)


def _parse_rows(rows: tuple[str, ...]) -> list[list[tuple[int, int]]]:
    size = len(rows)
    grid: list[list[tuple[int, int]]] = []
    for row in rows:
        tokens = row.split()
        if len(tokens) != size:
            raise ValueError(f"Expected {size} tile IDs in {row!r}")
        grid.append([(ord(token[0]) - ord("A"), int(token[1:]) - 1) for token in tokens])
    expected = {(row, col) for row in range(size) for col in range(size)}
    if {tile for line in grid for tile in line} != expected:
        raise ValueError("Initial mosaic must be a permutation of its targets")
    return grid


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        data={"spec": spec, "initial_grid": _parse_rows(spec.rows)},
        name=spec.title,
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class ShiftMosaicView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ShiftMosaic | None = None
        self._plaque_cache: dict[tuple[tuple[int, int], bool], np.ndarray] = {}

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_board(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: ShiftMosaic) -> None:
        size = game.size
        origin = game.board_origin
        span = size * CELL_SIZE
        bx, by = origin
        frame[by : by + span, bx : bx + span] = COLOR_BOARD
        if game.visual_mode == "complete":
            rail_color = COLOR_COMPLETE
        elif game.visual_mode == "failure":
            rail_color = COLOR_BUMP
        else:
            rail_color = COLOR_RAIL
        frame[by - 1, bx - 1 : bx + span + 1] = rail_color
        frame[by + span, bx - 1 : bx + span + 1] = rail_color
        frame[by - 1 : by + span + 1, bx - 1] = rail_color
        frame[by - 1 : by + span + 1, bx + span] = rail_color

        for row in range(size):
            for col in range(size):
                self._draw_target(frame, bx + col * CELL_SIZE, by + row * CELL_SIZE, (row, col))

        if game.visual_mode == "shift" and game.shift_visual is not None:
            pre_grid, handle, offset = game.shift_visual
            self._draw_shifted_tiles(frame, game, pre_grid, handle, offset)
            self._draw_shift_endpoints(frame, game, handle)
        else:
            for row in range(size):
                for col in range(size):
                    tile = game.grid[row][col]
                    self._draw_plaque(
                        frame,
                        bx + col * CELL_SIZE,
                        by + row * CELL_SIZE,
                        tile,
                        fitted=tile == (row, col),
                        clip=(bx, by, bx + span, by + span),
                    )

        self._draw_handles(frame, game)

    def _draw_target(self, frame: np.ndarray, x: int, y: int, target: tuple[int, int]) -> None:
        hue = ROW_HUES[target[0]]
        motif = MOTIFS[target[1]]
        for dy, motif_row in enumerate(motif):
            for dx, pixel in enumerate(motif_row):
                frame[y + dy, x + dx] = hue if pixel == "#" else COLOR_RAIL

    def _plaque_pixels(self, tile: tuple[int, int], fitted: bool) -> np.ndarray:
        cached = self._plaque_cache.get((tile, fitted))
        if cached is not None:
            return cached
        mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        mask[1:10, 1:10] = True
        mask[1:5, 1:5] = False
        pixels = np.full((CELL_SIZE, CELL_SIZE), -1, dtype=np.int8)
        pixels[mask] = COLOR_PLAQUE
        contour = COLOR_FITTED if fitted else COLOR_OUTLINE
        for y in range(CELL_SIZE):
            for x in range(CELL_SIZE):
                if not mask[y, x]:
                    continue
                if any(
                    ny < 0 or ny >= CELL_SIZE or nx < 0 or nx >= CELL_SIZE or not mask[ny, nx]
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                ):
                    pixels[y, x] = contour
        motif = MOTIFS[tile[1]]
        hue = ROW_HUES[tile[0]]
        for dy, motif_row in enumerate(motif):
            for dx, pixel in enumerate(motif_row):
                if pixel == "#":
                    pixels[5 + dy, 5 + dx] = hue
        self._plaque_cache[(tile, fitted)] = pixels
        return pixels

    def _draw_plaque(
        self, frame: np.ndarray, x: int, y: int, tile: tuple[int, int], *, fitted: bool, clip: tuple[int, int, int, int]
    ) -> None:
        pixels = self._plaque_pixels(tile, fitted)
        left, top, right, bottom = clip
        dst_left = max(left, x)
        dst_top = max(top, y)
        dst_right = min(right, x + CELL_SIZE)
        dst_bottom = min(bottom, y + CELL_SIZE)
        if dst_left >= dst_right or dst_top >= dst_bottom:
            return
        source = pixels[dst_top - y : dst_bottom - y, dst_left - x : dst_right - x]
        destination = frame[dst_top:dst_bottom, dst_left:dst_right]
        destination[:, :] = np.where(source >= 0, source, destination)

    def _draw_shifted_tiles(
        self,
        frame: np.ndarray,
        game: ShiftMosaic,
        pre_grid: list[list[tuple[int, int]]],
        handle: tuple[str, int],
        offset: int,
    ) -> None:
        kind, index = handle
        bx, by = game.board_origin
        span = game.size * CELL_SIZE
        clip = (bx, by, bx + span, by + span)
        horizontal = kind in {"L", "R"}
        sign = 1 if kind in {"R", "D"} else -1
        for row in range(game.size):
            for col in range(game.size):
                tile = pre_grid[row][col]
                moving = row == index if horizontal else col == index
                if not moving:
                    self._draw_plaque(
                        frame, bx + col * CELL_SIZE, by + row * CELL_SIZE, tile, fitted=tile == (row, col), clip=clip
                    )
                    continue
                dx = sign * offset if horizontal else 0
                dy = sign * offset if not horizontal else 0
                for periodic in (-span, 0, span):
                    px = bx + col * CELL_SIZE + dx + (periodic if horizontal else 0)
                    py = by + row * CELL_SIZE + dy + (periodic if not horizontal else 0)
                    self._draw_plaque(frame, px, py, tile, fitted=False, clip=clip)

    def _draw_shift_endpoints(self, frame: np.ndarray, game: ShiftMosaic, handle: tuple[str, int]) -> None:
        kind, index = handle
        bx, by = game.board_origin
        span = game.size * CELL_SIZE
        if kind in {"L", "R"}:
            center = by + index * CELL_SIZE + 5
            frame[center - 1 : center + 2, bx - 1] = COLOR_SHIFT
            frame[center - 1 : center + 2, bx + span] = COLOR_SHIFT
        else:
            center = bx + index * CELL_SIZE + 5
            frame[by - 1, center - 1 : center + 2] = COLOR_SHIFT
            frame[by + span, center - 1 : center + 2] = COLOR_SHIFT

    def _draw_handles(self, frame: np.ndarray, game: ShiftMosaic) -> None:
        for handle in game.handles:
            kind, _index = handle
            cx, cy = game.handle_center(handle)
            arrow = ARROW_RIGHT
            if kind == "L":
                arrow = np.fliplr(ARROW_RIGHT)
            elif kind == "U":
                arrow = np.rot90(ARROW_RIGHT)
            elif kind == "D":
                arrow = np.rot90(ARROW_RIGHT, -1)
            color = COLOR_SHIFT if handle == game.focus else COLOR_OUTLINE
            x0, y0 = cx - 2, cy - 2
            for y in range(5):
                for x in range(5):
                    if arrow[y, x] >= 0 and 0 <= x0 + x < VIEW_SIZE and 0 <= y0 + y < VIEW_SIZE:
                        frame[y0 + y, x0 + x] = color
        self._draw_focus(frame, game)

    def _draw_focus(self, frame: np.ndarray, game: ShiftMosaic) -> None:
        cx, cy = game.handle_center(game.focus)
        dx, dy = game.bump_offset if game.visual_mode == "bump" else (0, 0)
        cx += dx
        cy += dy
        color = COLOR_BUMP if game.visual_mode == "bump" else COLOR_FOCUS
        left, right = cx - 3, cx + 3
        top, bottom = cy - 3, cy + 3
        if 0 <= top < VIEW_SIZE:
            frame[top, max(0, left) : min(VIEW_SIZE, right + 1)] = color
        if 0 <= bottom < VIEW_SIZE:
            frame[bottom, max(0, left) : min(VIEW_SIZE, right + 1)] = color
        if 0 <= left < VIEW_SIZE:
            frame[max(0, top) : min(VIEW_SIZE, bottom + 1), left] = color
        if 0 <= right < VIEW_SIZE:
            frame[max(0, top) : min(VIEW_SIZE, bottom + 1), right] = color

    def _draw_budget(self, frame: np.ndarray, game: ShiftMosaic) -> None:
        bx, _by = game.board_origin
        width = game.size * CELL_SIZE
        frame[63, bx : bx + width] = COLOR_RAIL
        filled = int(width * game.remaining_actions / max(1, game.action_budget))
        if filled:
            frame[63, bx : bx + filled] = COLOR_FITTED


class ShiftMosaic(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = ShiftMosaicView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "shift_mosaic-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.size = spec.size
        self.board_origin = ((VIEW_SIZE - CELL_SIZE * self.size) // 2,) * 2
        self.grid = [[tuple(tile) for tile in row] for row in level.get_data("initial_grid")]
        self.focus = tuple(spec.initial_focus)
        self.done = False
        self.action_budget = spec.baseline_actions * 6
        self.remaining_actions = self.action_budget
        self.undo_history: list[tuple[list[list[tuple[int, int]]], tuple[str, int], bool]] = []
        self.visual_mode = "settled"
        self.shift_visual: tuple[list[list[tuple[int, int]]], tuple[str, int], int] | None = None
        self.bump_offset = (0, 0)
        self.animation_queue: list[tuple[str, object | None]] = []
        self.pending_win = False
        self.pending_loss = False
        self.handles = tuple(
            [("L", row) for row in range(self.size)]
            + [("U", col) for col in range(self.size)]
            + [("R", row) for row in range(self.size)]
            + [("D", col) for col in range(self.size)]
        )

    def handle_center(self, handle: tuple[str, int]) -> tuple[int, int]:
        kind, index = handle
        bx, by = self.board_origin
        span = self.size * CELL_SIZE
        if kind == "L":
            return bx - 4, by + CELL_SIZE * index + 5
        if kind == "R":
            return bx + span + 3, by + CELL_SIZE * index + 5
        if kind == "U":
            return bx + CELL_SIZE * index + 5, by - 4
        return bx + CELL_SIZE * index + 5, by + span + 3

    def _next_focus(self, direction: tuple[int, int]) -> tuple[str, int] | None:
        px, py = self.handle_center(self.focus)
        dx, dy = direction
        candidates: list[tuple[int, int, int, tuple[str, int]]] = []
        for rank, handle in enumerate(self.handles):
            if handle == self.focus:
                continue
            qx, qy = self.handle_center(handle)
            forward = (qx - px) * dx + (qy - py) * dy
            if forward <= 0:
                continue
            side = abs((qx - px) * -dy + (qy - py) * dx)
            candidates.append((side, forward, rank, handle))
        return min(candidates)[3] if candidates else None

    def _spend_action(self) -> None:
        self.remaining_actions = max(0, self.remaining_actions - 1)
        if self.remaining_actions == 0:
            self.pending_loss = True

    def _shift(self, handle: tuple[str, int]) -> None:
        kind, index = handle
        if kind == "R":
            self.grid[index] = self.grid[index][-1:] + self.grid[index][:-1]
        elif kind == "L":
            self.grid[index] = self.grid[index][1:] + self.grid[index][:1]
        elif kind == "D":
            values = [self.grid[row][index] for row in range(self.size)]
            values = values[-1:] + values[:-1]
            for row, tile in enumerate(values):
                self.grid[row][index] = tile
        else:
            values = [self.grid[row][index] for row in range(self.size)]
            values = values[1:] + values[:1]
            for row, tile in enumerate(values):
                self.grid[row][index] = tile

    def _is_solved(self) -> bool:
        return all(self.grid[row][col] == (row, col) for row in range(self.size) for col in range(self.size))

    def _start_shift(self) -> None:
        pre_grid = [[tile for tile in row] for row in self.grid]
        self.undo_history.append((pre_grid, self.focus, self.done))
        self._shift(self.focus)
        self.done = self._is_solved()
        self._spend_action()
        self.animation_queue = [
            ("shift", (pre_grid, self.focus, 3)),
            ("shift", (pre_grid, self.focus, 7)),
            ("settled", None),
        ]
        if self.done:
            self.animation_queue.extend([("complete", None), ("complete", None), ("settled", None)])
            self.pending_win = True
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        mode, payload = self.animation_queue.pop(0)
        self.visual_mode = mode
        self.shift_visual = payload if mode == "shift" else None
        if self.animation_queue:
            return
        if self.pending_win:
            self.next_level()
            self.complete_action()
        elif self.pending_loss:
            self.visual_mode = "failure"
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _undo(self) -> None:
        self._spend_action()
        if self.undo_history:
            grid, focus, done = self.undo_history.pop()
            self.grid = [[tile for tile in row] for row in grid]
            self.focus = focus
            self.done = done
        self.visual_mode = "settled"
        self.shift_visual = None
        if self.pending_loss:
            self.visual_mode = "failure"
            self.lose()
        self.complete_action()

    def step(self) -> None:
        if self.animation_queue:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self.visual_mode = "settled"
            self.complete_action()
            return
        if self.done:
            self.complete_action()
            return
        if action == ACTION_CLICK:
            self.complete_action()
            return
        if action == ACTION_UNDO:
            self._undo()
            return
        if action == ACTION_SPACE:
            self._start_shift()
            return
        directions = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
        if action in directions:
            direction = directions[action]
            next_focus = self._next_focus(direction)
            self._spend_action()
            if next_focus is not None:
                self.focus = next_focus
                self.visual_mode = "settled"
            else:
                self.visual_mode = "bump"
                self.bump_offset = direction
                self.animation_queue = [("settled", None)]
            if self.pending_loss and not self.animation_queue:
                self.visual_mode = "failure"
                self.lose()
            if not self.animation_queue:
                self.complete_action()
            return
        self.complete_action()
