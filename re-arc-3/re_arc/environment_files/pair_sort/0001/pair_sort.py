from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
COLOR_BACKGROUND = 4
COLOR_SLOT = 1
COLOR_FRAME = 2
COLOR_CONNECTOR = 3
COLOR_ARROW = 11
COLOR_MATCHED = 10
COLOR_CHECK = 14
COLOR_ERROR = 8
COLOR_BAR = 12

TOKEN_COLORS = (9, 11, 6, 14, 12, 15, 9, 11, 6, 14, 12, 15)
DIAMOND = ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0))
RING = ((0, 1, 1, 1, 0), (1, 1, 0, 1, 1), (1, 0, 0, 0, 1), (1, 1, 0, 1, 1), (0, 1, 1, 1, 0))

Cell = tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    ports: tuple[int, ...]
    edges: tuple[tuple[str, int, int], ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec(
        "Two at Once", ("......", "..Aa..", "..Bb..", "..Cc..", "......", "......"), (0, 2, 1), (("x", 1, 2),), 1
    ),
    LevelSpec(
        "Borrow the Glow",
        ("......", ".BAC..", ".bac..", "..dD..", "......", "......"),
        (0, 2, 3, 1),
        (("L", 0, 1), ("R", 0, 2), ("D", 0, 3)),
        4,
    ),
    LevelSpec(
        "Round the Supply",
        (".ABC..", ".abc..", "FfedD.", "..E...", "......", "......"),
        (3, 4, 5, 0, 1, 2),
        tuple((f"c{i}", i, (i + 1) % 6) for i in range(6)),
        9,
    ),
    LevelSpec(
        "The Quiet Bay",
        ("ABCD..", "abcd..", "Ii.eE.", ".hgf..", ".HGF..", "......"),
        (7, 1, 5, 4, 3, 2, 0, 8, 6),
        (*((f"e{i}", i, i + 1) for i in range(7)), ("b", 1, 8)),
        29,
    ),
    LevelSpec(
        "Use the Bay Twice",
        ("ABCDE.", "abcde.", "Jj..fF", "..ihg.", "..IHG.", "......"),
        (8, 1, 6, 5, 4, 3, 2, 0, 9, 7),
        (*((f"e{i}", i, i + 1) for i in range(8)), ("b", 1, 9)),
        37,
    ),
    LevelSpec(
        "Rotate Through the Bay",
        ("..BCD.", ".abcd.", ".AkKeE", ".J..fF", ".jihg.", "..IHG."),
        (9, 1, 7, 6, 5, 4, 3, 2, 0, 10, 8),
        (*((f"e{i}", i, i + 1) for i in range(9)), ("b", 1, 10)),
        46,
    ),
    LevelSpec(
        "Crossed Reservoirs",
        ("..BCD.", ".abcd.", ".AkKeE", ".JlLfF", ".jihg.", "..IHG."),
        (11, 6, 8, 5, 1, 4, 3, 7, 2, 10, 9, 0),
        (*((f"e{i}", i, i + 1) for i in range(9)), ("bL", 1, 10), ("bR", 8, 11)),
        51,
    ),
)


def _slot_top_left(cell: Cell) -> tuple[int, int]:
    row, col = cell
    return 4 + 10 * col, 4 + 10 * row


def _parse_layout(rows: tuple[str, ...]) -> tuple[tuple[Cell, Cell], ...]:
    found: dict[str, dict[str, Cell]] = {}
    for row, line in enumerate(rows):
        if len(line) != 6:
            raise ValueError("Pair Sort layouts must be 6 columns wide")
        for col, symbol in enumerate(line):
            if symbol == ".":
                continue
            key = symbol.upper()
            found.setdefault(key, {})["quiet" if symbol.isupper() else "port"] = (row, col)
    frames = []
    for key in sorted(found):
        pair = found[key]
        quiet, port = pair["quiet"], pair["port"]
        if abs(quiet[0] - port[0]) + abs(quiet[1] - port[1]) != 1:
            raise ValueError(f"Frame {key} is not orthogonally adjacent")
        frames.append((quiet, port))
    return tuple(frames)


def _hitbox(a: Cell, b: Cell) -> tuple[int, int, int, int]:
    (r0, c0), (r1, _c1) = sorted((a, b))
    if r0 == r1:
        return 9 + 10 * c0, 4 + 10 * r0, 13 + 10 * c0, 8 + 10 * r0
    return 4 + 10 * c0, 9 + 10 * r0, 8 + 10 * c0, 13 + 10 * r0


class PairSortDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PairSort | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_budget(frame, game)
        self._draw_frames(frame, game)
        self._draw_slots(frame, game)
        self._draw_connectors(frame, game)
        self._draw_tokens(frame, game)
        self._draw_feedback(frame, game)
        return frame

    def _draw_budget(self, frame: np.ndarray, game: PairSort) -> None:
        frame[0:2, 1:63] = 0
        filled = round(62 * max(0, game.step_budget - game.step_count) / game.step_budget)
        if filled:
            frame[0:2, 1 : 1 + filled] = COLOR_BAR

    def _draw_frames(self, frame: np.ndarray, game: PairSort) -> None:
        for index, (quiet, port) in enumerate(game.frames):
            xq, yq = _slot_top_left(quiet)
            xp, yp = _slot_top_left(port)
            x0, x1 = min(xq, xp) - 2, max(xq, xp) + 6
            y0, y1 = min(yq, yp) - 2, max(yq, yp) + 6
            color = COLOR_MATCHED if game.matched[index] else COLOR_FRAME
            if game.animation_phase == 3 and index in game.changed_frames:
                color = COLOR_ARROW if game.matched[index] else COLOR_ERROR
            if game.completion_phase is not None:
                color = COLOR_MATCHED if game.completion_phase % 2 == 0 else 0
            frame[y0 : y0 + 2, x0 : x1 + 1] = color
            frame[y1 - 1 : y1 + 1, x0 : x1 + 1] = color
            frame[y0 : y1 + 1, x0 : x0 + 2] = color
            frame[y0 : y1 + 1, x1 - 1 : x1 + 1] = color
            if game.matched[index] and game.completion_phase is None:
                if quiet[0] < port[0] or quiet[1] < port[1]:
                    self._draw_check(frame, x0, y0, 1, 1)
                else:
                    self._draw_check(frame, x1, y1, -1, -1)

    def _draw_check(self, frame: np.ndarray, x: int, y: int, sx: int, sy: int) -> None:
        # A three-pixel tick sits entirely on the two-pixel frame border, never
        # over a token silhouette.
        for dx, dy in ((0, 0), (1, 1), (2, 0)):
            px, py = x + sx * dx, y + sy * dy
            if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                frame[py, px] = COLOR_CHECK

    def _draw_slots(self, frame: np.ndarray, game: PairSort) -> None:
        for cell in game.token_at:
            x, y = _slot_top_left(cell)
            frame[y : y + 5, x : x + 5] = COLOR_SLOT
        if game.animation_phase == 0 and game.animation_edge is not None:
            for cell in game.animation_edge:
                x, y = _slot_top_left(cell)
                frame[y : y + 5, x : x + 5] = COLOR_ARROW

    def _draw_connectors(self, frame: np.ndarray, game: PairSort) -> None:
        for _name, a, b, box in game.edges:
            x0, y0, x1, y1 = box
            frame[y0 : y1 + 1, x0 : x1 + 1] = COLOR_CONNECTOR
            if a[0] == b[0]:
                frame[y0 + 2, x0 : x1 + 1] = COLOR_ARROW
                frame[y0 + 1, x0] = frame[y0 + 3, x0] = COLOR_ARROW
                frame[y0 + 1, x1] = frame[y0 + 3, x1] = COLOR_ARROW
            else:
                frame[y0 : y1 + 1, x0 + 2] = COLOR_ARROW
                frame[y0, x0 + 1] = frame[y0, x0 + 3] = COLOR_ARROW
                frame[y1, x0 + 1] = frame[y1, x0 + 3] = COLOR_ARROW

    def _draw_tokens(self, frame: np.ndarray, game: PairSort) -> None:
        moving = set(game.animation_edge or ()) if game.animation_phase is not None else set()
        for cell, token in game.token_at.items():
            if cell not in moving:
                self._draw_token(frame, token, *_slot_top_left(cell))
        if not moving or game.pre_swap is None or game.animation_edge is None:
            return
        a, b = game.animation_edge
        token_a, token_b = game.pre_swap[a], game.pre_swap[b]
        ax, ay = _slot_top_left(a)
        bx, by = _slot_top_left(b)
        if game.animation_phase == 0:
            positions = ((ax, ay), (bx, by))
        elif game.animation_phase in (1, 2):
            fraction = 0.35 if game.animation_phase == 1 else 0.65
            lane = -1 if a < b else 1
            dx, dy = bx - ax, by - ay
            if dx:
                positions = ((round(ax + dx * fraction), ay + lane), (round(bx - dx * fraction), by - lane))
            else:
                positions = ((ax + lane, round(ay + dy * fraction)), (bx - lane, round(by - dy * fraction)))
        else:
            positions = ((bx, by), (ax, ay))
        self._draw_token(frame, token_a, *positions[0])
        self._draw_token(frame, token_b, *positions[1])

    def _draw_token(self, frame: np.ndarray, token: int, x: int, y: int) -> None:
        mask = DIAMOND if token < 6 else RING
        color = TOKEN_COLORS[token]
        for row in range(5):
            for col in range(5):
                if mask[row][col] and 0 <= x + col < 64 and 0 <= y + row < 64:
                    frame[y + row, x + col] = color

    def _draw_feedback(self, frame: np.ndarray, game: PairSort) -> None:
        if game.feedback_kind == "invalid" and game.feedback_frames:
            x, y = game.feedback_point
            for offset in range(-2, 3):
                if 0 <= x + offset < 64 and 0 <= y + offset < 64:
                    frame[y + offset, x + offset] = COLOR_ERROR
                if 0 <= x + offset < 64 and 0 <= y - offset < 64:
                    frame[y - offset, x + offset] = COLOR_ERROR
        elif game.feedback_kind == "noop" and game.feedback_frames:
            frame[2, :] = COLOR_ERROR
            frame[-1, :] = COLOR_ERROR
            frame[:, 0] = COLOR_ERROR
            frame[:, -1] = COLOR_ERROR


class PairSort(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = PairSortDisplay()
        levels = [Level(sprites=[], grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, 64, 64, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display])
        super().__init__("pair_sort-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.spec = spec
        self.frames = _parse_layout(spec.rows)
        self.token_at: dict[Cell, int] = {}
        for index, (quiet, port) in enumerate(self.frames):
            self.token_at[quiet] = index
            self.token_at[port] = spec.ports[index]
        self.edges = []
        for name, left, right in spec.edges:
            a, b = self.frames[left][1], self.frames[right][1]
            self.edges.append((name, a, b, _hitbox(a, b)))
        self.matched = self._compute_matched()
        self.step_budget = spec.baseline * 6
        self.step_count = 0
        self.undo_history: list[tuple[dict[Cell, int], list[bool]]] = []
        self.pre_swap: dict[Cell, int] | None = None
        self.animation_edge: tuple[Cell, Cell] | None = None
        self.animation_phase: int | None = None
        self.changed_frames: set[int] = set()
        self.completion_phase: int | None = None
        self.feedback_kind: str | None = None
        self.feedback_frames = 0
        self.feedback_point = (32, 32)
        self.pending_loss = False

    def _compute_matched(self) -> list[bool]:
        return [self.token_at[a] == self.token_at[b] for a, b in self.frames]

    def _finish_or_lose(self) -> None:
        if self.pending_loss:
            self.lose()
        self.complete_action()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        if self.completion_phase is not None:
            if self.completion_phase < 5:
                self.completion_phase += 1
            else:
                self.completion_phase = None
                self.next_level()
                self.complete_action()
            return
        if self.animation_phase is not None:
            if self.animation_phase < 3:
                self.animation_phase += 1
                return
            self.animation_phase = None
            self.pre_swap = None
            self.animation_edge = None
            self.changed_frames.clear()
            if all(self.matched):
                self.completion_phase = 0
                return
            self._finish_or_lose()
            return
        if self.feedback_frames:
            self.feedback_frames -= 1
            if not self.feedback_frames:
                self.feedback_kind = None
                self._finish_or_lose()
            return

        self.step_count += 1
        self.pending_loss = self.step_count >= self.step_budget
        action_id = self.action.id.value
        if action_id == 7:
            if self.undo_history:
                self.token_at, self.matched = self.undo_history.pop()
            self._finish_or_lose()
            return
        if action_id != 6:
            self.feedback_kind = "noop"
            self.feedback_frames = 2
            return

        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        selected = next(
            (edge for edge in self.edges if edge[3][0] <= x <= edge[3][2] and edge[3][1] <= y <= edge[3][3]), None
        )
        if selected is None:
            self.feedback_kind = "invalid"
            self.feedback_point = (x, y)
            self.feedback_frames = 2
            return

        _name, a, b, _box = selected
        self.undo_history.append((dict(self.token_at), list(self.matched)))
        self.pre_swap = dict(self.token_at)
        old_matched = list(self.matched)
        self.token_at[a], self.token_at[b] = self.token_at[b], self.token_at[a]
        self.matched = self._compute_matched()
        self.changed_frames = {
            i for i, (old, new) in enumerate(zip(old_matched, self.matched, strict=True)) if old != new
        }
        self.animation_edge = (a, b)
        self.animation_phase = 0
        if all(self.matched):
            self.pending_loss = False
