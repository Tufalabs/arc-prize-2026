from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "flood_paint-0001"
VIEW_SIZE = 64
CELL_SIZE = 3
BOARD_VIEW_SIZE = 45

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

COLOR_BLACK = 0
COLOR_WHITE = 1
COLOR_LIGHT_GRAY = 2
COLOR_PANEL = 3
COLOR_VOID = 4
COLOR_TEXTURE = 5
COLOR_BLUE = 9
COLOR_WET = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14
COLOR_PURPLE = 15

SYMBOL_COLORS = {"B": COLOR_BLUE, "Y": COLOR_YELLOW, "G": COLOR_GREEN, "O": COLOR_ORANGE, "P": COLOR_PURPLE}
COLOR_ORDER = (COLOR_BLUE, COLOR_YELLOW, COLOR_GREEN, COLOR_ORANGE, COLOR_PURPLE)
PETAL_RECTS = {
    COLOR_BLUE: (55, 2, 58, 5),
    COLOR_YELLOW: (60, 7, 63, 10),
    COLOR_GREEN: (58, 15, 61, 18),
    COLOR_ORANGE: (50, 15, 53, 18),
    COLOR_PURPLE: (48, 7, 51, 10),
}
TEXTURE_PIXELS = {
    COLOR_BLUE: ((1, 0), (1, 2)),
    COLOR_YELLOW: ((0, 1), (2, 1)),
    COLOR_GREEN: ((2, 0), (0, 2)),
    COLOR_ORANGE: ((0, 0), (2, 0)),
    COLOR_PURPLE: ((0, 2), (2, 2)),
}
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    current_symbol: str
    palette_symbols: str
    droplets: int | None
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec("First Bloom", (".BBBB", "SYYYB", ".BBBB"), "B", "BY", None, 2),
    LevelSpec("Behind the Bridge", ("..BBB..BBB", ".YBBBGGBBB", "SYY..GG..B"), "B", "BYG", None, 4),
    LevelSpec(
        "The Droplet Tray",
        (
            "OOOOOOOOO.",
            "OGGGGGGGO.",
            "OGBBBBBGO.",
            "OGBYYYBGOP",
            "OGBYSYBGOP",
            "OGBYYYBGOP",
            "OGBBBBBGO.",
            "OGGGGGGGO.",
            "OOOOOOOOO.",
        ),
        "P",
        "BYGOP",
        12,
        5,
    ),
    LevelSpec(
        "Small Fronts First",
        (
            "GBO..........",
            "Y............",
            "POB...BOPYB..",
            "..Y...G...G..",
            "..G...Y...OPB",
            "..P...Y.....Y",
            "..BOYYSBB...O",
            "........B....",
            "........BBB..",
            "..........O..",
            "....GPO...P..",
            "......Y...G..",
            "......GPOBY..",
        ),
        "P",
        "BYGOP",
        19,
        19,
    ),
    LevelSpec(
        "Around the Bend",
        (
            "GYO...OGYBP..",
            "B.P.......O..",
            "P.B.......G..",
            "..G.......Y..",
            "..YOPBG.OPB..",
            "......Y.G....",
            "PGB.GGSBY....",
            "Y.O.G.O......",
            "O.Y.G.P...GOP",
            "B...G.Y...B.Y",
            "GPYOB.BGOPY.B",
        ),
        "P",
        "BYGOP",
        25,
        25,
    ),
    LevelSpec(
        "Split Bloom",
        (
            "PYBGOPY.YGOPPBY",
            "O.......B.....G",
            "GBY.....P.BGO.O",
            "..P.....O.B.Y.P",
            "..OGBYP.G.P.P.B",
            "......O.Y.Y.B.Y",
            "....GYSBBGO.G.G",
            "....B.G.....O..",
            "....P.BOY...Y..",
            "....O...P...P..",
            "..BGY...G.OGB..",
            "..P.....B......",
            "..O...B.OYP...O",
            "..Y...G...G...B",
            "..GBPOY...BOYPG",
        ),
        "P",
        "BYGOP",
        33,
        33,
    ),
    LevelSpec(
        "All Fronts Wet",
        (
            "YOP...OPYBGOPYB",
            "G.B...G.......G",
            "B.BGY.B...GOY.O",
            "P...O.Y...B.P.P",
            "O...P.P.OYP.B.P",
            "Y...B.O.G...G.Y",
            "GBP.GYSBBYG.O.B",
            "..O...G...O.Y.G",
            "..Y...B.YBP.P.O",
            "......O.G...B..",
            "OOB...Y.O...G..",
            "Y.G...P.P...O..",
            "P.PYOBG.B.BPY.B",
            "G.......Y.....P",
            "BOYPG...GOPBYGO",
        ),
        "P",
        "BYGOP",
        39,
        39,
    ),
)


@dataclass(frozen=True)
class PaintSnapshot:
    absorbed: frozenset[tuple[int, int]]
    current_color: int
    status: str


def _parse_spec(spec: LevelSpec) -> dict[str, object]:
    width = len(spec.rows[0])
    if any(len(row) != width for row in spec.rows):
        raise ValueError(f"{spec.name}: inconsistent row widths")
    paintable: set[tuple[int, int]] = set()
    base_color: dict[tuple[int, int], int] = {}
    seed_cell: tuple[int, int] | None = None
    current_color = SYMBOL_COLORS[spec.current_symbol]
    palette = tuple(SYMBOL_COLORS[symbol] for symbol in spec.palette_symbols)
    for y, row in enumerate(spec.rows):
        for x, symbol in enumerate(row):
            if symbol == ".":
                continue
            cell = (x, y)
            paintable.add(cell)
            if symbol == "S":
                if seed_cell is not None:
                    raise ValueError(f"{spec.name}: expected one seed")
                seed_cell = cell
                base_color[cell] = current_color
            else:
                base_color[cell] = SYMBOL_COLORS[symbol]
    if seed_cell is None:
        raise ValueError(f"{spec.name}: missing seed")
    step_budget = 6 * spec.baseline_actions
    return {
        "spec": spec,
        "width": width,
        "height": len(spec.rows),
        "paintable": frozenset(paintable),
        "base_color": base_color,
        "seed_cell": seed_cell,
        "current_color": current_color,
        "palette": palette,
        "droplets": spec.droplets,
        "step_budget": step_budget,
    }


def _build_level(spec: LevelSpec) -> Level:
    return Level(sprites=[], grid_size=(VIEW_SIZE, VIEW_SIZE), data=_parse_spec(spec), name=spec.name)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class FloodPaintView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: FloodPaint | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_VOID
        self._draw_board(frame, game)
        self._draw_flower(frame, game)
        self._draw_droplets(frame, game)
        self._draw_step_bar(frame, game)
        self._draw_feedback(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: FloodPaint) -> None:
        ox, oy = game.board_origin
        shown_absorbed = game.display_absorbed
        for cell in game.paintable:
            x, y = cell
            px = ox + CELL_SIZE * x
            py = oy + CELL_SIZE * y
            if cell in shown_absorbed:
                frame[py : py + 3, px : px + 3] = game.display_region_color
            else:
                color = game.base_color[cell]
                frame[py : py + 3, px : px + 3] = color
                for tx, ty in TEXTURE_PIXELS[color]:
                    frame[py + ty, px + tx] = COLOR_TEXTURE

        boundary_color = game.boundary_flash_color if game.boundary_flash_color is not None else COLOR_WET
        for x, y in shown_absorbed:
            px = ox + CELL_SIZE * x
            py = oy + CELL_SIZE * y
            for dx, dy in NEIGHBORS:
                neighbor = (x + dx, y + dy)
                if neighbor in shown_absorbed:
                    continue
                if dx == -1:
                    frame[py : py + 3, px] = boundary_color
                elif dx == 1:
                    frame[py : py + 3, px + 2] = boundary_color
                elif dy == -1:
                    frame[py, px : px + 3] = boundary_color
                else:
                    frame[py + 2, px : px + 3] = boundary_color

        stamp_color = game.stamp_flash_color if game.stamp_flash_color is not None else COLOR_BLACK
        for x, y in shown_absorbed:
            px = ox + CELL_SIZE * x
            py = oy + CELL_SIZE * y
            frame[py + 1, px + 1] = stamp_color
            frame[py + 2, px + 1] = stamp_color
            frame[py + 2, px] = COLOR_WHITE if game.stamp_flash_color is None else stamp_color
            frame[py + 2, px + 2] = COLOR_WHITE if game.stamp_flash_color is None else stamp_color

        if game.failure_flash:
            for x, y in game.paintable - shown_absorbed:
                px = ox + CELL_SIZE * x
                py = oy + CELL_SIZE * y
                frame[py : py + 3, px : px + 3] = COLOR_MAROON

    def _draw_flower(self, frame: np.ndarray, game: FloodPaint) -> None:
        # Stems are deliberately narrow and inert; only the colored blocks are petals.
        for x, y in ((56, 6), (57, 7), (59, 9), (59, 10), (58, 13), (58, 14), (54, 13), (53, 14), (52, 9), (53, 9)):
            frame[y, x] = COLOR_PANEL
        frame[8:13, 54:59] = game.display_selector_color
        self._texture_block(frame, 54, 8, 5, 5, game.display_selector_color)

        for color in game.palette:
            x0, y0, x1, y1 = PETAL_RECTS[color]
            if color == game.display_selector_color:
                frame[y0 : y1 + 1, x0 : x1 + 1] = COLOR_PANEL
                frame[y0 + 1 : y1, x0 + 1 : x1] = color
            else:
                frame[y0 : y1 + 1, x0 : x1 + 1] = color
                self._texture_block(frame, x0, y0, 4, 4, color)

        if game.center_pulse:
            frame[7, 56:59] = COLOR_LIGHT_GRAY
            frame[13, 56:59] = COLOR_LIGHT_GRAY
            frame[9:12, 53] = COLOR_LIGHT_GRAY
            frame[9:12, 59] = COLOR_LIGHT_GRAY

    def _texture_block(self, frame: np.ndarray, x: int, y: int, width: int, height: int, color: int) -> None:
        txs = TEXTURE_PIXELS[color]
        for tx, ty in txs:
            px = x + round(tx * (width - 1) / 2)
            py = y + round(ty * (height - 1) / 2)
            frame[py, px] = COLOR_TEXTURE

    def _draw_droplets(self, frame: np.ndarray, game: FloodPaint) -> None:
        if game.droplets_remaining is None:
            return
        for idx in range(game.droplet_budget):
            x = 2 + idx + idx // 5
            filled = idx < game.droplets_remaining
            frame[52, x] = COLOR_WET if filled else COLOR_VOID
            frame[53:55, x] = COLOR_BLUE if filled else COLOR_LIGHT_GRAY

    def _draw_step_bar(self, frame: np.ndarray, game: FloodPaint) -> None:
        frame[59:63, 1:63] = COLOR_PANEL
        frame[60:62, 2:62] = COLOR_LIGHT_GRAY
        filled = round(60 * game.steps_remaining / max(1, game.step_budget))
        if filled:
            frame[60:62, 2 : 2 + filled] = COLOR_WET

    def _draw_feedback(self, frame: np.ndarray, game: FloodPaint) -> None:
        if game.ripple is None:
            return
        x, y, phase = game.ripple
        if phase == 1:
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                px, py = x + dx, y + dy
                if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                    frame[py, px] = COLOR_WHITE
        elif 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE:
            frame[y, x] = COLOR_LIGHT_GRAY


class FloodPaint(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = FloodPaintView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_VOID, COLOR_VOID, [self._view])
        super().__init__(
            GAME_ID,
            levels,
            camera,
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.width = int(level.get_data("width"))
        self.height = int(level.get_data("height"))
        self.paintable = set(level.get_data("paintable"))
        self.base_color = dict(level.get_data("base_color"))
        self.absorbed = {tuple(level.get_data("seed_cell"))}
        self.current_color = int(level.get_data("current_color"))
        self.palette = tuple(level.get_data("palette"))
        initial_droplets = level.get_data("droplets")
        self.droplet_budget = int(initial_droplets or 0)
        self.droplets_remaining = None if initial_droplets is None else int(initial_droplets)
        self.step_budget = int(level.get_data("step_budget"))
        self.steps_remaining = self.step_budget
        self.status = "ACTIVE"
        self.undo_history: list[PaintSnapshot] = []
        self.animation_frames: deque[dict[str, object]] = deque()
        self.display_absorbed = set(self.absorbed)
        self.display_region_color = self.current_color
        self.display_selector_color = self.current_color
        self.boundary_flash_color: int | None = None
        self.stamp_flash_color: int | None = None
        self.center_pulse = False
        self.failure_flash = False
        self.ripple: tuple[int, int, int] | None = None
        self.board_origin = (
            1 + (BOARD_VIEW_SIZE - CELL_SIZE * self.width) // 2,
            1 + (BOARD_VIEW_SIZE - CELL_SIZE * self.height) // 2,
        )
        self._assert_invariants()

    def _clear_feedback(self) -> None:
        self.boundary_flash_color = None
        self.stamp_flash_color = None
        self.center_pulse = False
        self.failure_flash = False
        self.ripple = None

    def _spend_step(self) -> None:
        self.steps_remaining = max(0, self.steps_remaining - 1)

    def _queue_frame(self, **values: object) -> None:
        self.animation_frames.append(values)

    def _play_animation_frame(self) -> None:
        values = self.animation_frames.popleft()
        for key, value in values.items():
            setattr(self, key, value)
        if self.animation_frames:
            return
        if self.status == "WON":
            self.next_level()
            self.complete_action()
        elif self.status == "FAILED":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _petal_at(self, x: int, y: int) -> int | None:
        for color in self.palette:
            x0, y0, x1, y1 = PETAL_RECTS[color]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return color
        return None

    def _distance_layers(self, target_color: int) -> list[set[tuple[int, int]]]:
        reached: set[tuple[int, int]] = set()
        frontier: set[tuple[int, int]] = set()
        for x, y in self.absorbed:
            for dx, dy in NEIGHBORS:
                cell = (x + dx, y + dy)
                if cell in self.paintable - self.absorbed and self.base_color[cell] == target_color:
                    frontier.add(cell)
        layers: list[set[tuple[int, int]]] = []
        while frontier:
            layers.append(set(frontier))
            reached.update(frontier)
            next_frontier: set[tuple[int, int]] = set()
            for x, y in frontier:
                for dx, dy in NEIGHBORS:
                    cell = (x + dx, y + dy)
                    if (
                        cell in self.paintable
                        and cell not in self.absorbed
                        and cell not in reached
                        and self.base_color[cell] == target_color
                    ):
                        next_frontier.add(cell)
            frontier = next_frontier
        return layers

    def _apply_color(self, color: int) -> None:
        old_absorbed = set(self.absorbed)
        old_color = self.current_color
        self.undo_history.append(PaintSnapshot(frozenset(old_absorbed), old_color, self.status))
        layers = self._distance_layers(color)
        reached = set().union(*layers) if layers else set()
        self.current_color = color
        self.absorbed.update(reached)
        if self.droplets_remaining is not None:
            self.droplets_remaining -= 1

        if self.absorbed == self.paintable:
            self.status = "WON"
        elif self.droplets_remaining == 0 or self.steps_remaining == 0:
            self.status = "FAILED"

        # Selection: selector and tray update while the old region still has its old color.
        self._queue_frame(
            display_selector_color=color, display_region_color=old_color, display_absorbed=set(old_absorbed)
        )
        # Region: the whole old mass recolors atomically.
        self._queue_frame(display_region_color=color)
        shown = set(old_absorbed)
        if not layers:
            self._queue_frame(boundary_flash_color=COLOR_LIGHT_GRAY)
            self._queue_frame(boundary_flash_color=None)
        else:
            for layer in layers:
                shown.update(layer)
                self._queue_frame(display_absorbed=set(shown), boundary_flash_color=None)

        if self.status == "WON":
            self._queue_frame(boundary_flash_color=COLOR_BLACK, stamp_flash_color=COLOR_BLACK)
            self._queue_frame(boundary_flash_color=COLOR_WET, stamp_flash_color=None)
        elif self.status == "FAILED":
            self._queue_frame(failure_flash=True)
            self._queue_frame(failure_flash=True)
            self._queue_frame(failure_flash=False)

    def _queue_invalid_click(self, x: int, y: int) -> None:
        self._queue_frame(ripple=(x, y, 1))
        self._queue_frame(ripple=(x, y, 2))
        self._queue_frame(ripple=None)

    def _queue_acknowledgement(self) -> None:
        self._queue_frame(center_pulse=True)
        self._queue_frame(center_pulse=False)

    def _undo(self) -> None:
        if not self.undo_history:
            self._queue_acknowledgement()
            return
        snapshot = self.undo_history.pop()
        self.absorbed = set(snapshot.absorbed)
        self.current_color = snapshot.current_color
        self.status = snapshot.status
        self.display_absorbed = set(self.absorbed)
        self.display_region_color = self.current_color
        self.display_selector_color = self.current_color
        self._queue_frame()

    def _finish_from_step_budget(self) -> bool:
        if self.steps_remaining > 0 or self.status != "ACTIVE":
            return False
        self.status = "FAILED"
        self._queue_frame(failure_flash=True)
        self._queue_frame(failure_flash=True)
        self._queue_frame(failure_flash=False)
        return True

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        self._clear_feedback()
        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self.complete_action()
            return
        if self.status != "ACTIVE":
            self.complete_action()
            return

        self._spend_step()
        if action == ACTION_UNDO:
            self._undo()
        elif action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE):
            self._queue_acknowledgement()
        elif action == ACTION_CLICK:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            color = self._petal_at(x, y)
            if color is None or color == self.current_color:
                self._queue_invalid_click(x, y)
            else:
                self._apply_color(color)
        else:
            self._queue_acknowledgement()

        self._finish_from_step_budget()
        self._assert_invariants()
        if not self.animation_frames:
            self.complete_action()

    def _assert_invariants(self) -> None:
        assert self.absorbed
        assert self.absorbed <= self.paintable
        assert self.current_color in self.palette
        assert all(color in self.palette for color in self.base_color.values())
        assert self.droplets_remaining is None or self.droplets_remaining >= 0
        assert self.steps_remaining >= 0
        if self.status == "FAILED":
            assert self.droplets_remaining == 0 or self.steps_remaining == 0
        if self.status == "WON":
            assert self.absorbed == self.paintable
        pending = set(self.absorbed)
        start = next(iter(pending))
        connected = {start}
        queue = [start]
        while queue:
            x, y = queue.pop()
            for dx, dy in NEIGHBORS:
                cell = (x + dx, y + dy)
                if cell in pending and cell not in connected:
                    connected.add(cell)
                    queue.append(cell)
        assert connected == pending

    def _get_hidden_state(self) -> np.ndarray:
        values = [self.current_color, self.steps_remaining, self.droplets_remaining or -1, len(self.absorbed)]
        return np.asarray(values, dtype=np.int16)
