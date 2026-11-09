from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 6
CONNECTOR_GAP = 3
PITCH = CELL_SIZE + CONNECTOR_GAP

COLOR_WHITE = 0
COLOR_CONNECTOR_CORE = 1
COLOR_CONNECTOR = 2
COLOR_FRAME = 3
COLOR_SOCKET = 4
COLOR_BACKGROUND = 5
COLOR_A = 6
COLOR_CLICK = 7
COLOR_INVALID = 8
COLOR_B = 9
COLOR_ARROW = 10
COLOR_C = 11
COLOR_ORANGE = 12
COLOR_D = 14

STATE_COLORS = {"A": COLOR_A, "B": COLOR_B, "C": COLOR_C, "D": COLOR_D}
BASELINE_ACTIONS = (1, 2, 1, 4, 4, 4, 16)


@dataclass(frozen=True)
class Tile:
    state: str
    target: bool = False


Cell = Tile | None
Connector = tuple[str, int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    action_budget: int


@dataclass(frozen=True)
class VisualState:
    cells: tuple[tuple[Cell, ...], ...]
    connector: Connector | None = None
    removal: frozenset[tuple[int, int]] = frozenset()
    moving_tiles: tuple[tuple[int, int, Tile], ...] | None = None
    frame_color: int | None = None


LEVEL_SPECS = (
    LevelSpec("The Exchange", ("aBA", ".A."), BASELINE_ACTIONS[0] * 6),
    LevelSpec("Remove the Supports", (".a.", ".D.", "DB.", ".D.", "ACA"), BASELINE_ACTIONS[1] * 6),
    LevelSpec("No Pause Between Reactions", (".a.", ".D.", "DB.", "ADA", "BCD"), BASELINE_ACTIONS[2] * 6),
    LevelSpec("Replace the Support", ("BaBDB", "CCDBC", "DBCBC", "DADCA", "CBBDD"), BASELINE_ACTIONS[3] * 6),
    LevelSpec(
        "One Flash, Many Arms", ("ADBDAd", "CDCDDC", "CABBDD", "ACDDBA", "AdbDBA", "DbCBDb"), BASELINE_ACTIONS[4] * 6
    ),
    LevelSpec("Stage Before Ignition", ("CABBAB", "CBCBBa", "BDCCBC", "DAAcDD", "CCDBBC"), BASELINE_ACTIONS[5] * 6),
    LevelSpec(
        "Read the Reactor Twice", ("aDBDAd", "CDCDDc", "CABBDD", "ACDDBA", "AdbDBA", "DbCBDb"), BASELINE_ACTIONS[6] * 6
    ),
)


def _parse_rows(rows: tuple[str, ...]) -> list[list[Cell]]:
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Swap Reactor rows must have equal width")
    cells: list[list[Cell]] = []
    for row in rows:
        parsed: list[Cell] = []
        for symbol in row:
            if symbol == ".":
                parsed.append(None)
            elif symbol.upper() in STATE_COLORS:
                parsed.append(Tile(symbol.upper(), symbol.islower()))
            else:
                raise ValueError(f"Unknown Swap Reactor cell symbol: {symbol!r}")
        cells.append(parsed)
    return cells


def _freeze(cells: list[list[Cell]]) -> tuple[tuple[Cell, ...], ...]:
    return tuple(tuple(row) for row in cells)


def _board_origin(height: int, width: int) -> tuple[int, int]:
    board_width = PITCH * width - CONNECTOR_GAP
    board_height = PITCH * height - CONNECTOR_GAP
    return (VIEW_SIZE - board_width) // 2, (56 - board_height) // 2 + 1


class SwapReactorDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: SwapReactor | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        visual = game._visual or VisualState(_freeze(game.cells))
        frame[:, :] = COLOR_BACKGROUND
        self._draw_board(frame, game, visual)
        self._draw_arrow(frame)
        self._draw_budget(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: SwapReactor, visual: VisualState) -> None:
        height, width = game.height, game.width
        x0, y0 = _board_origin(height, width)
        board_width = PITCH * width - CONNECTOR_GAP
        board_height = PITCH * height - CONNECTOR_GAP
        frame_color = visual.frame_color if visual.frame_color is not None else COLOR_FRAME

        frame[y0 - 2 : y0, x0 - 2 : x0 + board_width + 2] = frame_color
        frame[y0 + board_height : y0 + board_height + 2, x0 - 2 : x0 + board_width + 2] = frame_color
        frame[y0 - 2 : y0 + board_height + 2, x0 - 2 : x0] = frame_color
        frame[y0 - 2 : y0 + board_height + 2, x0 + board_width : x0 + board_width + 2] = frame_color

        for row in range(height):
            y = y0 + row * PITCH
            for col in range(width - 1):
                x = x0 + col * PITCH + CELL_SIZE
                color = COLOR_CLICK if visual.connector == ("H", row, col) else COLOR_CONNECTOR
                frame[y : y + CELL_SIZE, x : x + CONNECTOR_GAP] = color
                if color == COLOR_CONNECTOR:
                    frame[y + 2 : y + 4, x : x + CONNECTOR_GAP] = COLOR_CONNECTOR_CORE
        for row in range(height - 1):
            y = y0 + row * PITCH + CELL_SIZE
            for col in range(width):
                x = x0 + col * PITCH
                color = COLOR_CLICK if visual.connector == ("V", row, col) else COLOR_CONNECTOR
                frame[y : y + CONNECTOR_GAP, x : x + CELL_SIZE] = color
                if color == COLOR_CONNECTOR:
                    frame[y : y + CONNECTOR_GAP, x + 2 : x + 4] = COLOR_CONNECTOR_CORE

        if visual.moving_tiles is None:
            for row, cells_row in enumerate(visual.cells):
                for col, tile in enumerate(cells_row):
                    self._draw_cell(frame, x0 + col * PITCH, y0 + row * PITCH, tile)
        else:
            for row in range(height):
                for col in range(width):
                    self._draw_cell(frame, x0 + col * PITCH, y0 + row * PITCH, None)
            for row, col, tile in visual.moving_tiles:
                self._draw_cell(frame, x0 + col * PITCH, y0 + row * PITCH, tile)

        for row, col in visual.removal:
            x = x0 + col * PITCH
            y = y0 + row * PITCH
            frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_WHITE

    def _draw_cell(self, frame: np.ndarray, x: int, y: int, tile: Cell) -> None:
        if tile is None:
            frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_SOCKET
            frame[y + 2 : y + 4, x + 2 : x + 4] = COLOR_BACKGROUND
            return
        frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = STATE_COLORS[tile.state]
        if tile.state == "A":
            for u, v in ((1, 1), (4, 1), (2, 2), (3, 2), (2, 3), (3, 3), (1, 4), (4, 4)):
                frame[y + v, x + u] = COLOR_WHITE
        elif tile.state == "B":
            frame[y + 1 : y + 5, x + 2 : x + 4] = COLOR_WHITE
        elif tile.state == "C":
            frame[y + 2 : y + 4, x + 1 : x + 5] = COLOR_SOCKET
        else:
            frame[y + 1 : y + 5, x + 2 : x + 4] = COLOR_WHITE
            frame[y + 2 : y + 4, x + 1 : x + 5] = COLOR_WHITE
        if tile.target:
            frame[y, x : x + CELL_SIZE] = COLOR_WHITE
            frame[y + 5, x : x + CELL_SIZE] = COLOR_WHITE
            frame[y : y + CELL_SIZE, x] = COLOR_WHITE
            frame[y : y + CELL_SIZE, x + 5] = COLOR_WHITE
            frame[y, x + 2 : x + 4] = COLOR_ORANGE
            frame[y + 5, x + 2 : x + 4] = COLOR_ORANGE
            frame[y + 2 : y + 4, x] = COLOR_ORANGE
            frame[y + 2 : y + 4, x + 5] = COLOR_ORANGE

    def _draw_arrow(self, frame: np.ndarray) -> None:
        frame[57:60, 31:33] = COLOR_ARROW
        frame[60, 29:35] = COLOR_ARROW
        frame[61, 30:34] = COLOR_ARROW
        frame[62, 31:33] = COLOR_ARROW

    def _draw_budget(self, frame: np.ndarray, game: SwapReactor) -> None:
        x0, y0 = _board_origin(game.height, game.width)
        board_height = PITCH * game.height - CONNECTOR_GAP
        x = max(0, x0 - 5)
        frame[y0 : y0 + board_height, x : x + 2] = COLOR_SOCKET
        filled = int(board_height * game.remaining_steps / max(1, game.action_budget))
        if filled:
            frame[y0 + board_height - filled : y0 + board_height, x : x + 2] = COLOR_ARROW


class SwapReactor(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = SwapReactorDisplay()
        self._display.game = self
        levels = [Level(grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, 64, 64, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display])
        self.cells: list[list[Cell]] = []
        self.height = 0
        self.width = 0
        self.action_budget = 1
        self.remaining_steps = 1
        self._history: list[list[list[Cell]]] = []
        self._visual: VisualState | None = None
        self._animation: list[VisualState] = []
        self._pending_win = False
        self._pending_loss = False
        self.last_clear_rounds: list[frozenset[tuple[int, int]]] = []
        super().__init__("swap_reactor-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.cells = _parse_rows(spec.rows)
        self.height = len(self.cells)
        self.width = len(self.cells[0])
        self.action_budget = spec.action_budget
        self.remaining_steps = spec.action_budget
        self._history = []
        self._visual = VisualState(_freeze(self.cells))
        self._animation = []
        self._pending_win = False
        self._pending_loss = False
        self.last_clear_rounds = []

    def step(self) -> None:
        if self._animation:
            self._show_next_animation_frame()
            return
        if self.action.id == GameAction.RESET:
            self._visual = VisualState(_freeze(self.cells))
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._spend_step()
            if self._history:
                self.cells = self._history.pop()
                self._visual = VisualState(_freeze(self.cells))
            else:
                self._visual = VisualState(_freeze(self.cells), frame_color=COLOR_INVALID)
            self._pending_loss = self.remaining_steps <= 0
            self._finish_or_fail()
            return

        connector = self._clicked_connector() if self.action.id == GameAction.ACTION6 else None
        self._spend_step()
        if connector is None:
            self._visual = VisualState(_freeze(self.cells), frame_color=COLOR_INVALID)
            self._pending_loss = self.remaining_steps <= 0
            self._finish_or_fail()
            return

        self._history.append(copy.deepcopy(self.cells))
        self._animation = self._resolve_connector(connector)
        self._pending_win = not any(tile is not None and tile.target for row in self.cells for tile in row)
        self._pending_loss = self.remaining_steps <= 0 and not self._pending_win
        self._show_next_animation_frame()

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _finish_or_fail(self) -> None:
        if self._pending_loss:
            if self._visual is not None:
                self._visual = replace(self._visual, frame_color=COLOR_INVALID)
            self.lose()
        self.complete_action()

    def _show_next_animation_frame(self) -> None:
        self._visual = self._animation.pop(0)
        if self._animation:
            return
        if self._pending_win:
            self.next_level()
            self.complete_action()
        elif self._pending_loss:
            self._visual = replace(self._visual, frame_color=COLOR_INVALID)
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _clicked_connector(self) -> Connector | None:
        data = self.action.data
        try:
            px, py = int(data["x"]), int(data["y"])
        except (KeyError, TypeError, ValueError):
            return None
        x0, y0 = _board_origin(self.height, self.width)
        for row in range(self.height):
            y = y0 + row * PITCH
            for col in range(self.width - 1):
                x = x0 + col * PITCH + CELL_SIZE
                if x <= px <= x + 2 and y <= py <= y + 5:
                    return ("H", row, col)
        for row in range(self.height - 1):
            y = y0 + row * PITCH + CELL_SIZE
            for col in range(self.width):
                x = x0 + col * PITCH
                if x <= px <= x + 5 and y <= py <= y + 2:
                    return ("V", row, col)
        return None

    def _resolve_connector(self, connector: Connector) -> list[VisualState]:
        before = copy.deepcopy(self.cells)
        kind, row, col = connector
        other_row, other_col = (row, col + 1) if kind == "H" else (row + 1, col)
        self.cells[row][col], self.cells[other_row][other_col] = (
            self.cells[other_row][other_col],
            self.cells[row][col],
        )
        frames = [VisualState(_freeze(before), connector=connector), VisualState(_freeze(self.cells))]
        self.last_clear_rounds = []

        while True:
            matched = self._find_matches(self.cells)
            if not matched:
                break
            removal = frozenset(matched)
            self.last_clear_rounds.append(removal)
            frames.append(VisualState(_freeze(self.cells), removal=removal))
            for matched_row, matched_col in matched:
                self.cells[matched_row][matched_col] = None
            frames.extend(self._settle_frames())

        stable = VisualState(_freeze(self.cells), connector=connector)
        frames.append(stable)
        if not any(tile is not None and tile.target for row_cells in self.cells for tile in row_cells):
            frames.extend(
                (
                    VisualState(_freeze(self.cells), frame_color=COLOR_D),
                    VisualState(_freeze(self.cells), frame_color=COLOR_WHITE),
                    VisualState(_freeze(self.cells), frame_color=COLOR_D),
                )
            )
        return frames

    def _find_matches(self, cells: list[list[Cell]]) -> set[tuple[int, int]]:
        matched: set[tuple[int, int]] = set()
        for row in range(self.height):
            start = 0
            while start < self.width:
                tile = cells[row][start]
                end = start + 1
                while (
                    end < self.width
                    and tile is not None
                    and cells[row][end] is not None
                    and cells[row][end].state == tile.state
                ):
                    end += 1
                if tile is not None and end - start >= 3:
                    matched.update((row, col) for col in range(start, end))
                start = end
        for col in range(self.width):
            start = 0
            while start < self.height:
                tile = cells[start][col]
                end = start + 1
                while (
                    end < self.height
                    and tile is not None
                    and cells[end][col] is not None
                    and cells[end][col].state == tile.state
                ):
                    end += 1
                if tile is not None and end - start >= 3:
                    matched.update((row, col) for row in range(start, end))
                start = end
        return matched

    def _settle_frames(self) -> list[VisualState]:
        moves: list[tuple[int, int, int, Tile]] = []
        settled: list[list[Cell]] = [[None for _ in range(self.width)] for _ in range(self.height)]
        for col in range(self.width):
            survivors = [(row, self.cells[row][col]) for row in range(self.height) if self.cells[row][col] is not None]
            first_destination = self.height - len(survivors)
            for offset, (start, tile) in enumerate(survivors):
                destination = first_destination + offset
                assert tile is not None
                settled[destination][col] = tile
                moves.append((start, destination, col, tile))
        max_distance = max((destination - start for start, destination, _, _ in moves), default=0)
        frames: list[VisualState] = []
        for distance in range(1, max_distance + 1):
            positions = tuple(
                (min(start + distance, destination), col, tile) for start, destination, col, tile in moves
            )
            frames.append(VisualState(_freeze(settled), moving_tiles=positions))
        self.cells = settled
        return frames
