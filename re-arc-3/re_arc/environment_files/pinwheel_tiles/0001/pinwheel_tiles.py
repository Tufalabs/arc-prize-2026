from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
PITCH = 9
PAD_SIZE = 7

BACKGROUND = 4
SOCKET = 3
SLAB = 1
WHITE = 0
UNUSED = 10
ARROW = 12
SPENT = 2
CRACK = 5
BLOCKED = 8
COMPLETE = 7
EXHAUSTED = 13

SYMBOLS = {
    "A": (9, (".#.", "###", ".#."), ("TL", "BR")),
    "B": (9, ("###", ".#.", ".#."), ("TR", "BL")),
    "C": (6, ("#..", "#..", "###"), ("TL", "BL")),
    "D": (6, ("#.#", "###", "#.#"), ("TR", "BR")),
    "E": (11, ("##.", ".##", "##."), ("TL", "TR")),
    "F": (11, ("###", "#.#", "###"), ("BL", "BR")),
    "G": (14, ("#..", "##.", "###"), ("TL", "TR", "BR")),
    "H": (14, ("#.#", "###", ".#."), ("TR", "BL", "BR")),
}

CORNER_MARKS = {
    "TL": ((0, 0), (1, 0), (0, 1)),
    "TR": ((6, 0), (5, 0), (6, 1)),
    "BL": ((0, 6), (1, 6), (0, 5)),
    "BR": ((6, 6), (5, 6), (6, 5)),
}
TAB_POS = {"TL": (1, 1), "TR": (5, 1), "BL": (1, 5), "BR": (5, 5)}


@dataclass(frozen=True)
class PivotSpec:
    id: str
    size: int
    row: int
    col: int
    style: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    sockets: tuple[str, ...]
    tiles: tuple[str, ...]
    pivots: tuple[PivotSpec, ...]
    solution: tuple[str, ...]
    baseline: int


def _p(id_: str, size: int, row: int, col: int, style: int) -> PivotSpec:
    return PivotSpec(id_, size, row - 1, col - 1, style)


LEVEL_SPECS = (
    LevelSpec("One Turn Home", ("ABC", "DEF"), ("ACF", "DBE"), (_p("P", 2, 1, 2, 0),), ("P",), 1),
    LevelSpec(
        "The Cracked Shortcut",
        ("AAB", "ABB"),
        ("BAB", "AAB"),
        (_p("L", 2, 1, 1, 0), _p("R", 2, 1, 2, 1)),
        ("L", "R"),
        2,
    ),
    LevelSpec(
        "Shared Column", ("ABC", "DEF"), ("BCF", "AED"), (_p("L", 2, 1, 1, 0), _p("R", 2, 1, 2, 1)), ("R", "L"), 2
    ),
    LevelSpec(
        "Still Point",
        ("GAAD", "BDEC", "HHFG", "BECF"),
        ("DAEF", "ADGC", "GBHF", "HEBC"),
        (
            _p("M", 3, 1, 1, 0),
            _p("T", 2, 1, 3, 2),
            _p("R", 2, 2, 3, 1),
            _p("Q", 2, 3, 2, 2),
            _p("B", 2, 3, 3, 3),
            _p("L", 2, 3, 1, 1),
        ),
        ("M", "T", "R", "Q", "B", "L"),
        6,
    ),
    LevelSpec(
        "Load the Anchor",
        (
            "D G C B H".replace(" ", ""),
            "A H A A F".replace(" ", ""),
            "E D B C F".replace(" ", ""),
            "D G E E C".replace(" ", ""),
            "F A H B G".replace(" ", ""),
        ),
        (
            "C H C H F".replace(" ", ""),
            "B G F A A".replace(" ", ""),
            "D A D C G".replace(" ", ""),
            "D G A E B".replace(" ", ""),
            "E F B E H".replace(" ", ""),
        ),
        (
            _p("C", 2, 1, 1, 1),
            _p("M", 3, 1, 1, 0),
            _p("T", 2, 1, 3, 1),
            _p("R", 2, 1, 4, 0),
            _p("L", 2, 3, 1, 2),
            _p("E", 2, 3, 2, 1),
            _p("Q", 3, 3, 3, 2),
            _p("Z", 2, 4, 1, 0),
        ),
        ("C", "M", "R", "E", "Z", "Q", "T", "L"),
        8,
    ),
    LevelSpec(
        "Crossed Centers",
        ("ECECD", "FHBBA", "DADGA", "ACGHB", "EHGFF"),
        ("EFDCA", "CAGBB", "EFDBF", "DHACH", "AEGHG"),
        (
            _p("A", 3, 1, 1, 1),
            _p("B", 3, 2, 2, 0),
            _p("C", 3, 3, 3, 2),
            _p("N", 2, 1, 4, 3),
            _p("W", 2, 3, 1, 2),
            _p("S", 2, 4, 1, 1),
            _p("E", 2, 2, 4, 1),
            _p("D", 2, 4, 4, 1),
            _p("X", 2, 1, 3, 2),
            _p("Y", 2, 3, 2, 3),
        ),
        ("X", "A", "N", "B", "C", "W", "E", "D", "Y", "S"),
        10,
    ),
    LevelSpec(
        "Whole Pinwheel",
        ("AEABBB", "FGDEHA", "CDCDFH", "ACGCEB", "HBHDGA", "FDFGCE"),
        ("ABBBAE", "EHBDEH", "AFDDBD", "HGCCFA", "GADEGC", "CHFCGF"),
        (
            _p("NW", 3, 1, 1, 2),
            _p("NE", 3, 1, 4, 2),
            _p("SW", 3, 4, 1, 2),
            _p("SE", 3, 4, 4, 2),
            _p("M", 3, 2, 2, 0),
            _p("T", 2, 1, 3, 1),
            _p("L", 2, 3, 1, 1),
            _p("R", 2, 3, 5, 0),
            _p("B", 2, 5, 3, 0),
            _p("C", 2, 3, 3, 1),
            _p("X", 2, 2, 4, 3),
            _p("Y", 2, 4, 2, 3),
        ),
        ("T", "NW", "C", "M", "SW", "X", "NE", "R", "SE", "B", "Y", "L"),
        12,
    ),
)


def _origin(spec: LevelSpec) -> tuple[int, int]:
    width = PITCH * len(spec.sockets[0]) - 2
    height = PITCH * len(spec.sockets) - 2
    return (VIEW_SIZE - width) // 2, (VIEW_SIZE - height) // 2


def _hitbox(spec: LevelSpec, pivot: PivotSpec) -> tuple[int, int, int, int]:
    ox, oy = _origin(spec)
    if pivot.size == 2:
        x = ox + PITCH * pivot.col + 6
        y = oy + PITCH * pivot.row + 6
        return x, y, x + 3, y + 3
    x = ox + PITCH * (pivot.col + 1) + 1
    y = oy + PITCH * (pivot.row + 1) + 1
    return x, y, x + 4, y + 4


def _validate_specs() -> None:
    if len(LEVEL_SPECS) != 7:
        raise ValueError("Pinwheel Tiles must contain seven levels")
    for spec in LEVEL_SPECS:
        h, w = len(spec.sockets), len(spec.sockets[0])
        if not (1 <= h <= 6 and 1 <= w <= 6):
            raise ValueError(f"{spec.name}: board out of bounds")
        if len(spec.tiles) != h or any(len(row) != w for row in (*spec.sockets, *spec.tiles)):
            raise ValueError(f"{spec.name}: malformed ASCII board")
        if sorted("".join(spec.tiles)) != sorted("".join(spec.sockets)):
            raise ValueError(f"{spec.name}: tile/socket multisets differ")
        if any(ch not in SYMBOLS for row in (*spec.sockets, *spec.tiles) for ch in row):
            raise ValueError(f"{spec.name}: unknown symbol")
        if spec.tiles == spec.sockets:
            raise ValueError(f"{spec.name}: starts solved")
        hitboxes = []
        for pivot in spec.pivots:
            if pivot.size not in (2, 3) or pivot.row < 0 or pivot.col < 0:
                raise ValueError(f"{spec.name}: bad pivot {pivot.id}")
            if pivot.row + pivot.size > h or pivot.col + pivot.size > w:
                raise ValueError(f"{spec.name}: pivot {pivot.id} out of bounds")
            box = _hitbox(spec, pivot)
            if not all(0 <= value < VIEW_SIZE for value in box):
                raise ValueError(f"{spec.name}: pivot {pivot.id} hitbox off canvas")
            for other in hitboxes:
                if not (box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]):
                    raise ValueError(f"{spec.name}: intersecting pivot hitboxes")
            hitboxes.append(box)
        for i, left in enumerate(spec.pivots):
            left_cells = {
                (r, c) for r in range(left.row, left.row + left.size) for c in range(left.col, left.col + left.size)
            }
            for right in spec.pivots[i + 1 :]:
                right_cells = {
                    (r, c)
                    for r in range(right.row, right.row + right.size)
                    for c in range(right.col, right.col + right.size)
                }
                if left_cells & right_cells and left.style == right.style:
                    raise ValueError(f"{spec.name}: overlapping pivots share style")


_validate_specs()


class PinwheelDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PinwheelTiles | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        self._draw_outlines(frame, game)
        self._draw_sockets(frame, game)
        self._draw_tiles(frame, game)
        self._draw_pivots(frame, game)
        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_outlines(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        requests: dict[tuple[int, int], list[int]] = {}
        ox, oy = game.origin
        for pivot in game.pivots:
            x0, y0 = ox + PITCH * pivot.col - 1, oy + PITCH * pivot.row - 1
            x1 = ox + PITCH * (pivot.col + pivot.size - 1) + PAD_SIZE
            y1 = oy + PITCH * (pivot.row + pivot.size - 1) + PAD_SIZE
            boundary = [(x, y0) for x in range(x0, x1 + 1)] + [(x, y1) for x in range(x0, x1 + 1)]
            boundary += [(x0, y) for y in range(y0 + 1, y1)] + [(x1, y) for y in range(y0 + 1, y1)]
            for idx, point in enumerate(boundary):
                if pivot.style == 1 and idx % 3 == 2:
                    continue
                requests.setdefault(point, []).append(pivot.style)
            caps = ((x0, y0), (x1, y1)) if pivot.style == 2 else ((x1, y0), (x0, y1))
            if pivot.style in (2, 3):
                for point in caps:
                    frame[point[1], point[0]] = WHITE
        for (x, y), styles in requests.items():
            styles.sort()
            style = styles[(x + y) % len(styles)]
            frame[y, x] = SLAB if style in (0, 1) else (SPENT if style == 2 else (SOCKET if (x + y) % 2 else SPENT))

    def _draw_sockets(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        ox, oy = game.origin
        for row, line in enumerate(game.spec.sockets):
            for col, symbol in enumerate(line):
                x, y = ox + PITCH * col, oy + PITCH * row
                frame[y : y + PAD_SIZE, x : x + PAD_SIZE] = SOCKET
                accent, _motif, corners = SYMBOLS[symbol]
                for corner in corners:
                    for dx, dy in CORNER_MARKS[corner]:
                        frame[y + dy, x + dx] = accent

    def _draw_tile(self, frame: np.ndarray, symbol: str, cx: int, cy: int, correct: bool, ghost: bool = False) -> None:
        x, y = cx - 2, cy - 2
        if x < 0 or y < 0 or x + 4 >= VIEW_SIZE or y + 4 >= VIEW_SIZE:
            return
        rim = UNUSED if ghost else (WHITE if correct else BACKGROUND)
        frame[y : y + 5, x : x + 5] = rim
        frame[y + 1 : y + 4, x + 1 : x + 4] = SLAB
        accent, motif, corners = SYMBOLS[symbol]
        for mr, line in enumerate(motif):
            for mc, value in enumerate(line):
                if value == "#":
                    frame[y + 1 + mr, x + 1 + mc] = accent
        for corner in corners:
            tx, ty = TAB_POS[corner]
            frame[y + ty - 1, x + tx - 1] = accent

    def _draw_tiles(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        ox, oy = game.origin
        anim = game.animation
        moving = anim is not None and anim["phase"] <= 2
        affected = anim["affected"] if moving else set()
        for row, line in enumerate(game.tiles):
            for col, symbol in enumerate(line):
                if (row, col) in affected:
                    continue
                self._draw_tile(
                    frame, symbol, ox + PITCH * col + 3, oy + PITCH * row + 3, symbol == game.spec.sockets[row][col]
                )
        if not moving:
            return
        phase = int(anim["phase"])
        pivot = anim["pivot"]
        center_col = pivot.col + (pivot.size - 1) / 2
        center_row = pivot.row + (pivot.size - 1) / 2
        for row, col, symbol in anim["sources"]:
            if phase == 0:
                dest_row, dest_col = row, col
                ghost = True
            else:
                rel_x, rel_y = col - center_col, row - center_row
                angle = math.radians(30 if phase == 1 else 60)
                rotated_x = rel_x * math.cos(angle) - rel_y * math.sin(angle)
                rotated_y = rel_x * math.sin(angle) + rel_y * math.cos(angle)
                dest_col, dest_row = center_col + rotated_x, center_row + rotated_y
                ghost = pivot.size == 3 and row == pivot.row + 1 and col == pivot.col + 1
            cx = round(ox + PITCH * dest_col + 3)
            cy = round(oy + PITCH * dest_row + 3)
            self._draw_tile(frame, symbol, cx, cy, False, ghost=ghost)

    def _style_ring(self, frame: np.ndarray, box: tuple[int, int, int, int], style: int, color: int) -> None:
        x0, y0, x1, y1 = box
        points = [(x, y0) for x in range(x0, x1 + 1)] + [(x1, y) for y in range(y0 + 1, y1 + 1)]
        points += [(x, y1) for x in range(x1 - 1, x0 - 1, -1)] + [(x0, y) for y in range(y1 - 1, y0, -1)]
        for idx, (x, y) in enumerate(points):
            if style != 1 or idx % 3 != 2:
                frame[y, x] = color if style != 3 or idx % 2 == 0 else SOCKET

    def _draw_pivots(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        active = game.animation["pivot"].id if game.animation is not None and game.animation["phase"] <= 2 else None
        for pivot in game.pivots:
            x0, y0, x1, y1 = _hitbox(game.spec, pivot)
            used = pivot.id in game.used
            body = SPENT if used else UNUSED
            if pivot.size == 3:
                mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
                frame[mid_y, x0 : x1 + 1] = body
                frame[y0 : y1 + 1, mid_x] = body
            else:
                frame[y0 : y1 + 1, x0 : x1 + 1] = body
            ring_color = WHITE if active == pivot.id else (SPENT if used else ARROW)
            self._style_ring(frame, (x0, y0, x1, y1), pivot.style, ring_color)
            if used:
                for offset in range(min(x1 - x0, y1 - y0) + 1):
                    frame[y0 + offset, x0 + offset] = CRACK
            else:
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                frame[cy, cx] = WHITE if active == pivot.id else ARROW
                if cx + 1 <= x1:
                    frame[cy, cx + 1] = WHITE if active == pivot.id else ARROW
                if cy + 1 <= y1:
                    frame[cy + 1, cx] = WHITE if active == pivot.id else ARROW
            if pivot.size == 3 and game.animation is None:
                symbol = game.tiles[pivot.row + 1][pivot.col + 1]
                required = game.spec.sockets[pivot.row + 1][pivot.col + 1]
                corner_color = WHITE if symbol == required else BACKGROUND
                for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                    frame[y, x] = corner_color
                accent, _motif, corners = SYMBOLS[symbol]
                for corner in corners:
                    tx, ty = {"TL": (x0, y0), "TR": (x1, y0), "BL": (x0, y1), "BR": (x1, y1)}[corner]
                    frame[ty, tx] = accent

    def _draw_feedback(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        if game.pulse_kind == "blocked" and game.pulse_frames:
            x, y = game.pulse_xy
            frame[max(0, y - 2) : min(64, y + 3), x] = BLOCKED
            frame[y, max(0, x - 2) : min(64, x + 3)] = BLOCKED
        border_color = None
        if game.pulse_kind == "completed" and game.pulse_frames:
            border_color = COMPLETE
        elif game.pulse_kind in ("exhausted", "budget") and game.pulse_frames:
            border_color = EXHAUSTED
        if border_color is not None:
            frame[0, :] = border_color
            frame[-1, :] = border_color
            frame[:, 0] = border_color
            frame[:, -1] = border_color

    def _draw_budget(self, frame: np.ndarray, game: PinwheelTiles) -> None:
        frame[61:63, 5:59] = SPENT
        filled = round(54 * game.remaining_steps / max(1, game.step_budget))
        if filled:
            frame[61:63, 5 : 5 + filled] = ARROW


class PinwheelTiles(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = PinwheelDisplay()
        levels = [Level(sprites=[], grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self.display])
        super().__init__("pinwheel_tiles-0001", levels, camera, False, len(levels), list(range(1, 8)), seed=seed)
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.tiles = [list(row) for row in self.spec.tiles]
        self.pivots = list(self.spec.pivots)
        self.pivot_by_id = {pivot.id: pivot for pivot in self.pivots}
        self.used: set[str] = set()
        self.origin = _origin(self.spec)
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[list[list[str]], set[str], bool, bool]] = []
        self.done = False
        self.exhausted = False
        self.animation: dict[str, object] | None = None
        self.pulse_kind = "none"
        self.pulse_frames = 0
        self.pulse_xy = (32, 32)

    def _spend_action(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0:
            self.animation = None
            self.pulse_kind = "budget"
            self.pulse_frames = 1
            self.lose()
            self.complete_action()
            return True
        return False

    def _pivot_at(self, x: int, y: int) -> PivotSpec | None:
        for pivot in self.pivots:
            x0, y0, x1, y1 = _hitbox(self.spec, pivot)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return pivot
        return None

    def _rotate(self, pivot: PivotSpec) -> None:
        old = [row[:] for row in self.tiles]
        for i in range(pivot.size):
            for j in range(pivot.size):
                self.tiles[pivot.row + j][pivot.col + pivot.size - 1 - i] = old[pivot.row + i][pivot.col + j]
        affected = {
            (r, c) for r in range(pivot.row, pivot.row + pivot.size) for c in range(pivot.col, pivot.col + pivot.size)
        }
        sources = [(r, c, old[r][c]) for r, c in sorted(affected)]
        self.animation = {"phase": 0, "pivot": pivot, "affected": affected, "sources": sources}

    def _advance_animation(self) -> None:
        assert self.animation is not None
        phase = int(self.animation["phase"])
        if phase < 2:
            self.animation["phase"] = phase + 1
            return
        self.animation = None
        if self.done:
            self.pulse_kind = "completed"
            self.pulse_frames = 3
            return
        if self.exhausted:
            self.pulse_kind = "exhausted"
            self.pulse_frames = 1
        self.complete_action()

    def _finish_completion_pulse(self) -> None:
        if self.pulse_frames > 1:
            self.pulse_frames -= 1
            return
        self.pulse_frames = 0
        self.next_level()
        self.complete_action()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        if self.animation is not None:
            self._advance_animation()
            return
        if self.pulse_kind == "completed" and self.pulse_frames:
            self._finish_completion_pulse()
            return
        if self.pulse_frames > 1:
            self.pulse_frames -= 1
            if self.pulse_frames == 1:
                self.complete_action()
            return

        self.pulse_kind = "none"
        self.pulse_frames = 0
        if self._spend_action():
            return

        if self.action.id == GameAction.ACTION7:
            if self.undo_history:
                tiles, used, done, exhausted = self.undo_history.pop()
                self.tiles = [row[:] for row in tiles]
                self.used = set(used)
                self.done = done
                self.exhausted = exhausted
            self.complete_action()
            return

        if self.done or self.exhausted or self.action.id != GameAction.ACTION6:
            self.complete_action()
            return

        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        pivot = self._pivot_at(x, y)
        if pivot is None or pivot.id in self.used:
            self.pulse_kind = "blocked"
            self.pulse_frames = 2
            self.pulse_xy = (x, y)
            return

        self.undo_history.append(([row[:] for row in self.tiles], set(self.used), self.done, self.exhausted))
        self._rotate(pivot)
        self.used.add(pivot.id)
        self.done = self.tiles == [list(row) for row in self.spec.sockets]
        self.exhausted = not self.done and len(self.used) == len(self.pivots)


__all__ = ["PinwheelTiles"]
