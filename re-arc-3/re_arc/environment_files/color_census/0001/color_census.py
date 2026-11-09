from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

GAME_ID = "color_census-0001"
VIEW_SIZE = 64
BACKGROUND = 4
BLACK = 5
WHITE = 0
LIGHT_GRAY = 1
SEPARATOR = 2
INACTIVE = 3

ORDER = (8, 9, 14, 12, 15, 6, 10, 11)
NEXT = {8: 9, 9: 14, 14: 12, 12: 15, 15: 6, 6: 10, 10: 14, 11: 9}
SYMBOL_COLOR = {"R": 8, "B": 9, "G": 14, "O": 12, "P": 15, "M": 6, "C": 10, "Y": 11}
MASKS = {
    8: (".....", "..#..", "..#..", "..#..", "....."),
    9: (".....", ".##..", ".##..", ".....", "....."),
    14: (".....", ".#...", "..#..", "...#.", "....."),
    12: (".....", "...#.", "..#..", ".#...", "....."),
    15: (".....", "..#..", ".###.", "..#..", "....."),
    6: (".....", ".#.#.", ".....", ".#.#.", "....."),
    10: (".....", "..#..", ".#.#.", "..#..", "....."),
    11: (".....", ".....", ".###.", ".....", "....."),
}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    targets: tuple[int, ...]
    baseline: int


@dataclass
class PieceState:
    col: int
    row: int
    color: int
    covered: bool


LEVEL_SPECS = (
    LevelSpec(
        "One Click, Two Sites",
        (".....", "R....", ".....", "..B..", ".....", ".....", "....R", "....."),
        (0, 3, 0, 0, 0, 0, 0, 0),
        1,
    ),
    LevelSpec(
        "Origins Disappear",
        (".....", "R....", ".....", "..B..", ".....", ".....", "....R", "....."),
        (0, 0, 3, 0, 0, 0, 0, 0),
        2,
    ),
    LevelSpec(
        "Clear the Destination",
        (".....", "R....", ".....", "..B..", ".....", ".....", "....R", "....."),
        (0, 2, 0, 0, 1, 0, 0, 0),
        4,
    ),
    LevelSpec(
        "Borrowed Handle",
        (".....", ".O..g", ".....", ".cMc.", ".....", ".....", "B..Op", "....."),
        (0, 0, 0, 0, 0, 7, 1, 0),
        15,
    ),
    LevelSpec(
        "The Moving Vacancy",
        ("M.G.C", "...PM", "O.M..", "...C.", ".M..P", "P....", ".P.M.", "C.M.O"),
        (0, 0, 1, 2, 10, 0, 3, 0),
        20,
    ),
    LevelSpec(
        "Four Required Sums",
        ("OC...", "Rb..G", "..O..", "..p..", "C..bM", ".OG..", "....R", "b.YCO"),
        (0, 0, 0, 5, 4, 5, 3, 0),
        40,
    ),
    LevelSpec(
        "Collection Order",
        ("....R", "Yb...", "..pp.", "..Op.", "....M", "CgC..", ".C...", "M...Y"),
        (0, 0, 0, 2, 4, 3, 5, 0),
        51,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _build_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != 8 or any(len(row) != 5 for row in spec.rows):
        raise ValueError(f"Invalid 5x8 layout for {spec.name}")
    return Level(
        sprites=[], grid_size=(VIEW_SIZE, VIEW_SIZE), name=spec.name, data={"spec": spec, "budget": spec.baseline * 6}
    )


class CensusDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ColorCensus | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        frame[:, :] = BACKGROUND
        if game is None:
            return frame
        for piece in game.pieces:
            self._draw_piece(frame, piece, game)
        self._draw_hud(frame, game)
        self._draw_step_bar(frame, game)
        if game.feedback_kind == "invalid":
            self._draw_invalid(frame, game.feedback_click, game.feedback_phase)
        if game.failed:
            frame[0, :] = 8
            frame[-1, :] = 8
            frame[:, 0] = 8
            frame[:, -1] = 8
        return frame

    def _draw_piece(self, frame: np.ndarray, piece: PieceState, game: ColorCensus) -> None:
        sx, sy = 2 + 7 * piece.col, 4 + 7 * piece.row
        if piece.covered:
            frame[sy : sy + 7, sx : sx + 7] = LIGHT_GRAY
        body_x, body_y = sx + 1, sy + 1
        coord = (piece.col, piece.row)
        render_color = game.feedback_old_colors.get(coord, piece.color) if game.feedback_phase == 0 else piece.color
        frame[body_y : body_y + 5, body_x : body_x + 5] = render_color
        mask = MASKS[render_color]
        for y, row in enumerate(mask):
            for x, value in enumerate(row):
                if value == "#":
                    frame[body_y + y, body_x + x] = BLACK

        pulse_color = None
        if game.feedback_kind == "valid" and coord in game.feedback_affected and game.feedback_phase == 0:
            pulse_color = WHITE
        elif game.feedback_kind == "valid" and coord in game.feedback_preexisting and game.feedback_phase == 1:
            pulse_color = LIGHT_GRAY
        elif game.feedback_kind == "complete" and game.feedback_phase >= 0:
            pulse_color = WHITE if game.feedback_phase < 2 else 14
        if pulse_color is not None:
            frame[sy, sx : sx + 7] = pulse_color
            frame[sy + 6, sx : sx + 7] = pulse_color
            frame[sy : sy + 7, sx] = pulse_color
            frame[sy : sy + 7, sx + 6] = pulse_color
        elif game.counts()[piece.color] != game.targets[piece.color]:
            # Every member of an unsatisfied color class shares a bright rim,
            # making the global (not per-piece) effect of a click explicit.
            frame[sy, sx : sx + 7] = WHITE
            frame[sy + 6, sx : sx + 7] = WHITE
            frame[sy : sy + 7, sx] = WHITE
            frame[sy : sy + 7, sx + 6] = WHITE

    def _draw_hud(self, frame: np.ndarray, game: ColorCensus) -> None:
        counts = game.counts()
        for idx, color in enumerate(ORDER):
            if not game.revealed[color]:
                continue
            y = 4 + 7 * idx
            self._chip(frame, 39, y + 2, color)
            frame[y + 3, 42:44] = LIGHT_GRAY
            frame[y + 2, 43] = LIGHT_GRAY
            frame[y + 4, 43] = LIGHT_GRAY
            self._chip(frame, 44, y + 2, NEXT[color])
            self._pips(frame, 48, y + 1, counts[color], color)
            frame[y + 1 : y + 5, 53] = SEPARATOR
            self._pips(frame, 55, y + 1, game.targets[color], LIGHT_GRAY)
            seal = 14 if counts[color] == game.targets[color] else INACTIVE
            frame[y + 1 : y + 6, 61:63] = seal

    def _chip(self, frame: np.ndarray, x0: int, y0: int, color: int) -> None:
        frame[y0 : y0 + 3, x0 : x0 + 3] = color
        mask = MASKS[color]
        for yy, xx in ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)):
            source_y = min(4, yy * 2)
            source_x = min(4, xx * 2)
            if mask[source_y][source_x] == "#":
                frame[y0 + yy, x0 + xx] = BLACK

    def _pips(self, frame: np.ndarray, x0: int, y0: int, count: int, color: int) -> None:
        for idx in range(min(20, count)):
            frame[y0 + idx // 5, x0 + idx % 5] = color

    def _draw_step_bar(self, frame: np.ndarray, game: ColorCensus) -> None:
        frame[61:63, 2:63] = INACTIVE
        remaining = max(0, game.step_budget - game.step_count)
        filled = int(61 * remaining / max(1, game.step_budget))
        if filled:
            frame[61:63, 2 : 2 + filled] = 14

    def _draw_invalid(self, frame: np.ndarray, click: tuple[int, int], phase: int) -> None:
        x, y = click
        radius = 2 + phase
        for offset in range(-radius, radius + 1):
            for px, py in ((x + offset, y + offset), (x + offset, y - offset)):
                if 0 <= px < 64 and 0 <= py < 64:
                    frame[py, px] = LIGHT_GRAY


class ColorCensus(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = CensusDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self.display]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.pieces: list[PieceState] = []
        for row_idx, row in enumerate(spec.rows):
            for col_idx, symbol in enumerate(row):
                upper = symbol.upper()
                if upper in SYMBOL_COLOR:
                    self.pieces.append(PieceState(col_idx, row_idx, SYMBOL_COLOR[upper], symbol.islower()))
        self.targets = dict(zip(ORDER, spec.targets, strict=True))
        initial_counts = self.counts()
        self.revealed = {color: initial_counts[color] > 0 or self.targets[color] > 0 for color in ORDER}
        self.step_budget = int(level.get_data("budget"))
        self.step_count = 0
        self.done = False
        self.failed = False
        self.selected = None
        self.cursor = None
        self.undo_history: list[tuple[list[int], dict[int, bool]]] = []
        self.animation_queue: list[tuple[str | None, int]] = []
        self.feedback_kind: str | None = None
        self.feedback_phase = -1
        self.feedback_affected: set[tuple[int, int]] = set()
        self.feedback_preexisting: set[tuple[int, int]] = set()
        self.feedback_old_colors: dict[tuple[int, int], int] = {}
        self.feedback_click = (0, 0)

    def counts(self) -> dict[int, int]:
        result = {color: 0 for color in ORDER}
        for piece in self.pieces:
            result[piece.color] += 1
        return result

    def step(self) -> None:
        if self.animation_queue:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self._clear_feedback()
            self.complete_action()
            return

        self.step_count += 1
        if action == 7:
            self._undo()
            return
        if action != 6:
            self._finish_noop()
            return

        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        clicked = self._exposed_piece_at(x, y)
        if clicked is None:
            self.feedback_click = (x, y)
            self.animation_queue = [("invalid", 0), ("invalid", 1), (None, -1)]
            self._play_animation_frame()
            return

        old_colors = [piece.color for piece in self.pieces]
        old_revealed = dict(self.revealed)
        self.undo_history.append((old_colors, old_revealed))
        source = clicked.color
        destination = NEXT[source]
        self.feedback_affected = {(p.col, p.row) for p in self.pieces if p.color == source}
        self.feedback_preexisting = {(p.col, p.row) for p in self.pieces if p.color == destination}
        self.feedback_old_colors = {(p.col, p.row): p.color for p in self.pieces if p.color == source}
        for piece in self.pieces:
            if piece.color == source:
                piece.color = destination
        self.revealed[source] = True
        self.revealed[destination] = True
        self.done = self.counts() == self.targets
        if self.done:
            self.animation_queue = (
                [("valid", 0), ("valid", 1), ("valid", 2)] + [("complete", idx) for idx in range(4)] + [(None, -1)]
            )
        else:
            self.animation_queue = [("valid", 0), ("valid", 1), ("valid", 2), (None, -1)]
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        kind, phase = self.animation_queue.pop(0)
        self.feedback_kind = kind
        self.feedback_phase = phase
        if self.animation_queue:
            return
        self._clear_feedback()
        if self.done:
            self.next_level()
        elif self.step_count >= self.step_budget:
            self.failed = True
            self.lose()
        self.complete_action()

    def _finish_noop(self) -> None:
        self._clear_feedback()
        if self.step_count >= self.step_budget:
            self.failed = True
            self.lose()
        self.complete_action()

    def _undo(self) -> None:
        self._clear_feedback()
        if self.undo_history:
            colors, revealed = self.undo_history.pop()
            for piece, color in zip(self.pieces, colors, strict=True):
                piece.color = color
            self.revealed = dict(revealed)
            self.done = False
        if self.step_count >= self.step_budget:
            self.failed = True
            self.lose()
        self.complete_action()

    def _clear_feedback(self) -> None:
        self.feedback_kind = None
        self.feedback_phase = -1
        self.feedback_affected.clear()
        self.feedback_preexisting.clear()
        self.feedback_old_colors.clear()

    def _exposed_piece_at(self, x: int, y: int) -> PieceState | None:
        for piece in self.pieces:
            if piece.covered:
                continue
            sx, sy = 2 + 7 * piece.col, 4 + 7 * piece.row
            if sx + 1 <= x <= sx + 5 and sy + 1 <= y <= sy + 5:
                return piece
        return None

    def _get_hidden_state(self) -> np.ndarray:
        values = [self.step_count, self.step_budget]
        values.extend(piece.color for piece in self.pieces)
        values.extend(int(self.revealed[color]) for color in ORDER)
        return np.asarray(values, dtype=np.int16)
