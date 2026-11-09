from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "compass_weave-0001"
VIEW_SIZE = 64

BACKGROUND = 5
TOKEN = 0
HUD_DETAIL = 1
HUD_FRAME = 2
FLOOR = 3
HUD_USED = 4
CCW = 6
FAIL = 8
CW = 9
TIP = 10
SEAL = 11
DOCK_DARK = 13
GREEN = 14
HALF = 15

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_DIRECTIONS = {1: 0, 4: 1, 2: 2, 3: 3}
ROTATION_COLOR = {1: CW, -1: CCW, 2: HALF}
ARROW_EAST = np.array(
    [[-1, TOKEN, TIP, -1], [TOKEN, TOKEN, TIP, TIP], [TOKEN, TOKEN, TIP, TIP], [-1, TOKEN, TIP, -1]], dtype=np.int8
)

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


def _edge(a: Cell, b: Cell) -> Edge:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    segments: tuple[str, ...]
    orientation: int
    thresholds: tuple[tuple[Cell, Cell, int], ...]
    seals: tuple[tuple[Cell, Cell], ...]
    baseline: int

    @property
    def budget(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    LevelSpec(
        "First Bend",
        (
            "##########",
            "##########",
            "##########",
            "##..P..###",
            "##.#######",
            "#X.#######",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ("H 4 3 7", "V 3 4 6", "H 6 2 3"),
        1,
        (((4, 3), (4, 4), 1), ((5, 3), (6, 3), 1)),
        (((4, 6), (4, 7)),),
        9,
    ),
    LevelSpec(
        "Same Gate Twice",
        (
            "##########",
            "##########",
            "###P...###",
            "###..#.###",
            "###X...###",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ("H 3 4 7", "H 5 5 7", "V 4 3 5", "V 5 3 5", "V 7 3 5"),
        1,
        (((3, 4), (3, 5), 1), ((4, 5), (5, 5), 1), ((5, 6), (5, 7), 1), ((4, 7), (3, 7), 1)),
        (((3, 5), (4, 5)),),
        12,
    ),
    LevelSpec(
        "Turning Circuit",
        (
            "##########",
            "###...####",
            "###.#.####",
            "###.#.####",
            "##P.X.####",
            "###.######",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ("H 5 3 6", "H 2 4 6", "V 4 2 6", "V 6 2 5"),
        1,
        (((5, 3), (5, 4), 1), ((3, 4), (2, 4), 1), ((2, 5), (2, 6), 1), ((4, 6), (5, 6), 1)),
        (((5, 4), (6, 4)), ((5, 4), (4, 4))),
        16,
    ),
    LevelSpec(
        "Counter-Weave",
        (
            "##########",
            "##########",
            "####.#####",
            "####X.####",
            "#.P.....##",
            "#..#.##.##",
            "###.....##",
            "###..#####",
            "##########",
            "##########",
        ),
        (
            "H 5 2 8",
            "H 6 2 3",
            "H 4 5 6",
            "H 7 4 8",
            "H 8 4 5",
            "V 2 5 6",
            "V 3 5 6",
            "V 5 3 8",
            "V 6 4 5",
            "V 4 7 8",
            "V 5 7 8",
            "V 8 5 7",
        ),
        1,
        (
            ((5, 3), (5, 2), 1),
            ((5, 2), (6, 2), 1),
            ((6, 2), (6, 3), 1),
            ((6, 3), (5, 3), 1),
            ((5, 5), (5, 6), -1),
            ((5, 6), (4, 6), -1),
            ((4, 6), (4, 5), -1),
            ((7, 5), (8, 5), 1),
            ((8, 5), (8, 4), 1),
            ((8, 4), (7, 4), 1),
            ((7, 7), (7, 8), -1),
            ((6, 8), (5, 8), 1),
        ),
        (((5, 2), (6, 2)), ((4, 5), (3, 5)), ((6, 5), (7, 5)), ((7, 5), (7, 6))),
        27,
    ),
    LevelSpec(
        "Purple Parity",
        (
            "######...#",
            "######.#.#",
            "######.#..",
            "######.##.",
            "######.##X",
            "..P.....##",
            ".#.###..##",
            "...#######",
            "##########",
            "##########",
        ),
        (
            "H 6 1 8",
            "H 8 1 3",
            "H 7 7 8",
            "H 1 7 9",
            "H 3 9 10",
            "V 1 6 8",
            "V 3 6 8",
            "V 8 6 7",
            "V 7 1 7",
            "V 9 1 3",
            "V 10 3 5",
        ),
        1,
        (
            ((6, 3), (6, 2), 2),
            ((6, 2), (6, 1), 1),
            ((7, 1), (8, 1), 1),
            ((8, 2), (8, 3), 1),
            ((7, 3), (6, 3), 1),
            ((6, 6), (6, 7), 2),
            ((6, 7), (6, 8), 1),
            ((6, 8), (7, 8), 1),
            ((7, 8), (7, 7), 1),
            ((5, 7), (4, 7), 2),
            ((2, 7), (1, 7), -1),
            ((1, 8), (1, 9), -1),
            ((2, 9), (3, 9), 1),
            ((3, 9), (3, 10), 1),
        ),
        (((6, 6), (6, 7)), ((6, 3), (6, 2)), ((3, 7), (2, 7)), ((1, 7), (1, 8))),
        40,
    ),
    LevelSpec(
        "Three Crossings",
        (
            "#.########",
            "#.########",
            "#......###",
            "#.##..####",
            "#.##..####",
            "#...P.....",
            "#####X#.##",
            "#####...##",
            "#######.##",
            "#######.##",
        ),
        ("H 6 2 10", "H 8 6 8", "H 3 2 7", "V 6 4 8", "V 8 6 10", "V 5 3 6", "V 2 1 6"),
        1,
        (
            ((6, 5), (6, 6), 1),
            ((7, 6), (8, 6), 1),
            ((8, 7), (8, 8), 1),
            ((7, 8), (6, 8), 1),
            ((4, 5), (3, 5), -1),
            ((3, 5), (3, 6), 2),
            ((3, 3), (3, 2), 1),
            ((5, 2), (6, 2), -1),
        ),
        (((4, 6), (5, 6)), ((9, 8), (10, 8)), ((6, 9), (6, 10)), ((4, 5), (5, 5)), ((3, 6), (3, 7)), ((1, 2), (2, 2))),
        46,
    ),
    LevelSpec(
        "Final Compass Weave",
        (
            "######...#",
            "######.#.#",
            "######.#..",
            "######.##X",
            "######....",
            "..P.......",
            ".#.###....",
            "...#####..",
            "########.#",
            "########.#",
        ),
        (
            "H 6 1 8",
            "H 8 1 3",
            "H 7 7 8",
            "H 1 7 9",
            "H 3 9 10",
            "V 1 6 8",
            "V 3 6 8",
            "V 8 6 7",
            "V 7 1 7",
            "V 9 1 3",
            "V 10 3 8",
            "H 8 9 10",
            "H 5 8 10",
            "V 9 5 10",
        ),
        1,
        (
            ((6, 3), (6, 2), 2),
            ((6, 2), (6, 1), 1),
            ((7, 1), (8, 1), 1),
            ((8, 2), (8, 3), 1),
            ((7, 3), (6, 3), 1),
            ((6, 6), (6, 7), 2),
            ((6, 7), (6, 8), 1),
            ((6, 8), (7, 8), 1),
            ((7, 8), (7, 7), 1),
            ((5, 7), (4, 7), 2),
            ((2, 7), (1, 7), -1),
            ((1, 8), (1, 9), -1),
            ((2, 9), (3, 9), 1),
            ((3, 9), (3, 10), 1),
            ((7, 10), (8, 10), 1),
            ((8, 10), (8, 9), 1),
            ((6, 9), (5, 9), 1),
            ((5, 9), (5, 10), -1),
        ),
        (
            ((6, 6), (6, 7)),
            ((6, 3), (6, 2)),
            ((3, 7), (2, 7)),
            ((1, 7), (1, 8)),
            ((8, 10), (8, 9)),
            ((9, 9), (10, 9)),
            ((5, 8), (5, 9)),
        ),
        55,
    ),
)


def _parse_layout(spec: LevelSpec) -> tuple[set[Cell], set[Edge], Cell, Cell]:
    cells: set[Cell] = set()
    edges: set[Edge] = set()
    for segment in spec.segments:
        axis, *values = segment.split()
        a, b, c = (int(value) for value in values)
        if axis == "H":
            segment_cells = [(a, col) for col in range(b, c + 1)]
        else:
            segment_cells = [(row, a) for row in range(b, c + 1)]
        cells.update(segment_cells)
        edges.update(_edge(left, right) for left, right in pairwise(segment_cells))

    start = exit_cell = None
    ascii_cells: set[Cell] = set()
    for row, line in enumerate(spec.rows, 1):
        if len(line) != 10:
            raise ValueError(f"{spec.name}: every ASCII row must be ten cells")
        for col, symbol in enumerate(line, 1):
            if symbol != "#":
                ascii_cells.add((row, col))
            if symbol == "P":
                start = (row, col)
            elif symbol == "X":
                exit_cell = (row, col)
    if cells != ascii_cells or start is None or exit_cell is None:
        raise ValueError(f"{spec.name}: ASCII layout and segment graph disagree")
    return cells, edges, start, exit_cell


class CompassHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: CompassWeave | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[1:8, 1:8] = HUD_FRAME
        frame[2:7, 2:7] = BACKGROUND if not game._hud_pulse else HUD_DETAIL
        mask = _arrow_mask(game.orientation)
        for row in range(4):
            for col in range(4):
                if mask[row, col] >= 0:
                    frame[2 + row, 2 + col] = mask[row, col]

        frame[2:6, 11:62] = HUD_FRAME
        frame[3:5, 12:61] = HUD_USED
        remaining = max(0, game.step_limit - game.steps_used)
        width = int(49 * remaining / max(1, game.step_limit))
        if width:
            frame[3:5, 12 : 12 + width] = HUD_DETAIL
        if game._failed:
            frame[0, :] = FAIL
            frame[-1, :] = FAIL
            frame[:, 0] = FAIL
            frame[:, -1] = FAIL
        return frame


def _arrow_mask(orientation: int) -> np.ndarray:
    return np.rot90(ARROW_EAST, (1 - orientation) % 4)


def _xy(cell: Cell) -> tuple[int, int]:
    row, col = cell
    return 7 + 5 * (col - 1), 9 + 5 * (row - 1)


class CompassWeave(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = CompassHud()
        levels = [
            Level(
                sprites=[Sprite(np.full((64, 64), BACKGROUND, dtype=np.int8), name="board", collidable=False)],
                grid_size=(64, 64),
                data={"spec": spec},
                name=spec.name,
            )
            for spec in LEVEL_SPECS
        ]
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self._hud])
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.cells, self.edges, self.position, self.exit_cell = _parse_layout(self.spec)
        self.orientation = self.spec.orientation
        self.thresholds = {_edge(a, b): q for a, b, q in self.spec.thresholds}
        self.seal_edges = tuple(_edge(a, b) for a, b in self.spec.seals)
        self.seal_cut = [False] * len(self.seal_edges)
        self.steps_used = 0
        self.step_limit = self.spec.budget
        self.undo_stack: list[tuple[Cell, int, tuple[bool, ...]]] = []
        self._frames: list[tuple[np.ndarray, bool]] = []
        self._pending_terminal: str | None = None
        self._hud_pulse = False
        self._failed = False
        self.board = self.current_level.get_sprites_by_name("board")[0]
        self._set_board(self._render_board())
        # A deferred next_level() re-enters the engine loop with the solving action
        # still installed. Mark it complete so that input is not replayed in the
        # freshly initialized level.
        self.complete_action()

    def _set_board(self, pixels: np.ndarray) -> None:
        self.board.pixels = pixels.astype(np.int8, copy=False)

    def _draw_edge(self, frame: np.ndarray, edge: Edge, color: int) -> None:
        (r1, c1), (r2, c2) = edge
        x1, y1 = _xy((r1, c1))
        if r1 == r2:
            x = x1 + 4 if c2 > c1 else x1 - 1
            frame[y1 : y1 + 4, x] = color
        else:
            y = y1 + 4 if r2 > r1 else y1 - 1
            frame[y, x1 : x1 + 4] = color

    def _draw_threshold(self, frame: np.ndarray, edge: Edge, q: int) -> None:
        color = ROTATION_COLOR[q]
        self._draw_edge(frame, edge, color)
        (r1, c1), (r2, c2) = edge
        x1, y1 = _xy((r1, c1))
        x2, y2 = _xy((r2, c2))
        if r1 == r2:
            gap = min(x1, x2) + 4
            if q == 1:
                frame[min(y1, y2), gap - 1 : gap + 2] = color
                frame[min(y1, y2) + 1, gap + 1] = color
            elif q == -1:
                frame[min(y1, y2) + 3, gap - 1 : gap + 2] = color
                frame[min(y1, y2) + 2, gap - 1] = color
            else:
                frame[min(y1, y2), gap - 1 : gap + 2] = color
                frame[min(y1, y2) + 3, gap - 1 : gap + 2] = color
        else:
            gap = min(y1, y2) + 4
            if q == 1:
                frame[gap - 1 : gap + 2, min(x1, x2) + 3] = color
                frame[gap - 1, min(x1, x2) + 2] = color
            elif q == -1:
                frame[gap - 1 : gap + 2, min(x1, x2)] = color
                frame[gap + 1, min(x1, x2) + 1] = color
            else:
                frame[gap - 1 : gap + 2, min(x1, x2)] = color
                frame[gap - 1 : gap + 2, min(x1, x2) + 3] = color

    def _draw_seal(self, frame: np.ndarray, edge: Edge, cut: bool) -> None:
        (r1, c1), (r2, c2) = edge
        x1, y1 = _xy((r1, c1))
        x2, y2 = _xy((r2, c2))
        if r1 == r2:
            gap = min(x1, x2) + 4
            if cut:
                frame[min(y1, y2) : min(y1, y2) + 4, gap] = FLOOR
                frame[min(y1, y2) : min(y1, y2) + 4, gap - 1] = GREEN
                frame[min(y1, y2) : min(y1, y2) + 4, gap + 1] = GREEN
            else:
                frame[min(y1, y2) : min(y1, y2) + 4, gap - 1 : gap + 2] = SEAL
        else:
            gap = min(y1, y2) + 4
            if cut:
                frame[gap, min(x1, x2) : min(x1, x2) + 4] = FLOOR
                frame[gap - 1, min(x1, x2) : min(x1, x2) + 4] = GREEN
                frame[gap + 1, min(x1, x2) : min(x1, x2) + 4] = GREEN
            else:
                frame[gap - 1 : gap + 2, min(x1, x2) : min(x1, x2) + 4] = SEAL

    def _draw_dock(self, frame: np.ndarray, ready: bool, complete: bool = False) -> None:
        x, y = _xy(self.exit_cell)
        frame[y : y + 4, x : x + 4] = GREEN
        center = TIP if ready else DOCK_DARK
        frame[y + 1 : y + 3, x + 1 : x + 3] = TIP if complete else center

    def _draw_player(
        self, frame: np.ndarray, position: Cell, orientation: int, offset: tuple[int, int] = (0, 0), spin: bool = False
    ) -> None:
        x, y = _xy(position)
        x += offset[0]
        y += offset[1]
        mask = (
            np.array([[-1, TOKEN, -1, -1], [TOKEN, TIP, TOKEN, -1], [-1, TOKEN, -1, -1], [-1, -1, -1, -1]])
            if spin
            else _arrow_mask(orientation)
        )
        for row in range(4):
            for col in range(4):
                px, py = x + col, y + row
                if mask[row, col] >= 0 and 0 <= px < 64 and 0 <= py < 64:
                    frame[py, px] = mask[row, col]

    def _render_board(
        self,
        *,
        position: Cell | None = None,
        orientation: int | None = None,
        seals: tuple[bool, ...] | list[bool] | None = None,
        offset: tuple[int, int] = (0, 0),
        blocked: Edge | None = None,
        spin: bool = False,
        complete: bool = False,
    ) -> np.ndarray:
        frame = np.full((64, 64), BACKGROUND, dtype=np.int8)
        for cell in self.cells:
            x, y = _xy(cell)
            frame[y : y + 4, x : x + 4] = FLOOR
        for edge in self.edges:
            self._draw_edge(frame, edge, FLOOR)
        for edge, q in self.thresholds.items():
            self._draw_threshold(frame, edge, q)
        current_seals = self.seal_cut if seals is None else seals
        for index, edge in enumerate(self.seal_edges):
            self._draw_seal(frame, edge, bool(current_seals[index]))
        self._draw_dock(frame, all(current_seals), complete)
        if blocked is not None:
            self._draw_edge(frame, blocked, FAIL)
        self._draw_player(
            frame,
            self.position if position is None else position,
            self.orientation if orientation is None else orientation,
            offset,
            spin,
        )
        return frame

    def _queue(self, frame: np.ndarray, *, hud_pulse: bool = False) -> None:
        self._frames.append((frame, hud_pulse))

    def _finish_if_needed(self) -> None:
        if self._pending_terminal == "win":
            self.next_level()
        elif self._pending_terminal == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _spend(self) -> None:
        self.steps_used += 1

    def _expire_if_needed(self, won: bool = False) -> None:
        if not won and self.steps_used >= self.step_limit:
            self._failed = True
            self._pending_terminal = "lose"

    def _move(self, forward: bool) -> None:
        old_position = self.position
        old_orientation = self.orientation
        old_seals = tuple(self.seal_cut)
        movement_dir = self.orientation if forward else (self.orientation + 2) % 4
        dr, dc = DIRS[movement_dir]
        destination = (self.position[0] + dr, self.position[1] + dc)
        crossed = _edge(self.position, destination)
        if crossed not in self.edges:
            self._queue(self._render_board(offset=(dc, dr), blocked=crossed))
            self._queue(self._render_board())
            self._expire_if_needed()
            return

        self.undo_stack.append((self.position, self.orientation, tuple(self.seal_cut)))
        self.position = destination
        cut_now = False
        if forward and crossed in self.seal_edges:
            seal_index = self.seal_edges.index(crossed)
            if not self.seal_cut[seal_index]:
                self.seal_cut[seal_index] = True
                cut_now = True
        q = self.thresholds.get(crossed, 0)
        self.orientation = (self.orientation + q) % 4
        won = self.position == self.exit_cell and forward and all(self.seal_cut)

        if q:
            self._queue(self._render_board(position=old_position, orientation=old_orientation, seals=old_seals))
            self._queue(
                self._render_board(
                    position=old_position, orientation=old_orientation, seals=old_seals, offset=(dc * 2, dr * 2)
                )
            )
            self._queue(self._render_board(orientation=old_orientation, seals=old_seals))
            if cut_now:
                self._queue(self._render_board(orientation=old_orientation))
            if abs(q) == 2:
                self._queue(self._render_board(orientation=(old_orientation + 1) % 4))
            self._queue(self._render_board(spin=True))
        self._queue(self._render_board(complete=won))
        if won:
            self._queue(self._render_board(complete=True))
            self._pending_terminal = "win"
        self._expire_if_needed(won)

    def _undo(self) -> None:
        if not self.undo_stack:
            self._queue(self._render_board(), hud_pulse=True)
            self._expire_if_needed()
            return
        old_position = self.position
        position, orientation, seals = self.undo_stack.pop()
        self.position = position
        self.orientation = orientation
        self.seal_cut = list(seals)
        old_x, old_y = _xy(old_position)
        new_x, new_y = _xy(position)
        self._queue(self._render_board(position=old_position, offset=((new_x - old_x) // 2, (new_y - old_y) // 2)))
        self._queue(self._render_board())
        self._expire_if_needed()

    def step(self) -> None:
        if self._frames:
            pixels, self._hud_pulse = self._frames.pop(0)
            self._set_board(pixels)
            if not self._frames:
                self._finish_if_needed()
            return

        if self.action.id == GameAction.RESET:
            self._hud_pulse = False
            self._set_board(self._render_board())
            self.complete_action()
            return

        self._pending_terminal = None
        self._hud_pulse = False
        self._spend()
        action_id = int(self.action.id.value)
        if action_id in ACTION_DIRECTIONS:
            requested_direction = ACTION_DIRECTIONS[action_id]
            if requested_direction == self.orientation:
                self._move(True)
            elif requested_direction == (self.orientation + 2) % 4:
                self._move(False)
            else:
                self._queue(self._render_board(), hud_pulse=True)
                self._queue(self._render_board())
                self._expire_if_needed()
        elif action_id == 7:
            self._undo()
        else:
            self._queue(self._render_board())
            self._expire_if_needed()
