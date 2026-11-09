from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

GAME_ID = "beam_paths-0001"
VIEW_SIZE = 64
CELL_SIZE = 4

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
MIRROR_TURNS = {"/": {"E": "N", "N": "E", "W": "S", "S": "W"}, "\\": {"E": "S", "S": "E", "W": "N", "N": "W"}}
ELBOW_SIDES = {"NE": ("N", "E"), "SE": ("S", "E"), "SW": ("S", "W"), "NW": ("N", "W")}
ELBOW_ORDER = ("NE", "SE", "SW", "NW")

OUTSIDE = 5
FLOOR = 1
WALL = 4
CASING = 3
CHANNEL = 2
PLAYER = 9
MIRROR = 7
WHITE = 0
EMITTER = 12
BEAM = 11
EXIT = 14
INVALID = 13
CHANNEL_COLORS = {"A": 6, "B": 15, "C": 8, "D": 10}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    emitter_direction: str
    orientations: tuple[tuple[tuple[int, int], str], ...]
    door_axes: tuple[tuple[tuple[int, int], str], ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec(
        "The Already-open Door",
        ("#########", "#E.a#...#", "#...#...#", "#P..A.X.#", "#...#...#", "#########"),
        "E",
        (),
        (((5, 4), "EW"),),
        5,
    ),
    LevelSpec(
        "The Beam Frontier",
        ("##########", "#E.I.a#..#", "#..P..#..#", "#.....A.X#", "#.....#..#", "##########"),
        "E",
        (((4, 2), "V"),),
        (((7, 4), "EW"),),
        7,
    ),
    LevelSpec(
        "Turning a Corner",
        ("###########", "#.....#...#", "#E..MP#...#", "#.....#...#", "#...a.A.X.#", "#.....#...#", "###########"),
        "E",
        (((5, 3), "/"),),
        (((7, 5), "EW"),),
        6,
    ),
    LevelSpec(
        "Prepare, Cross, Restore",
        (
            "#############",
            "#b...M#...#.#",
            "#E...M....#.#",
            "#.....#...#.#",
            "#.....#...#.#",
            "#....a#...#.#",
            "#.....#...BX#",
            "#.....A...#.#",
            "#P....#...#.#",
            "#.....#...#.#",
            "#############",
        ),
        "E",
        (((6, 2), "/"), ((6, 3), "/")),
        (((7, 8), "EW"), ((11, 7), "EW")),
        37,
    ),
    LevelSpec(
        "The Four-sided Elbow",
        (
            "###############",
            "#......#....#.#",
            "#......#....CX#",
            "#b.I..M....c#.#",
            "#......#....#.#",
            "#......###B####",
            "#......#....###",
            "#E....L.....###",
            "#......#....###",
            "#......A....###",
            "#P....a#....###",
            "#......#....###",
            "###############",
        ),
        "E",
        (((4, 4), "V"), ((7, 4), "\\"), ((7, 8), "NE")),
        (((8, 10), "EW"), ((11, 6), "NS"), ((13, 3), "EW")),
        43,
    ),
    LevelSpec(
        "Carry the Alignment Home",
        (
            "###############",
            "#......#.#....#",
            "#b....L..B.I.c#",
            "#......#.#....#",
            "#......#.#....#",
            "#......#.#....#",
            "#E....M..#....#",
            "#......#.#....#",
            "#.....a#.#....#",
            "#......A.#....#",
            "#P.....###....#",
            "#......CX######",
            "#......########",
            "###############",
        ),
        "E",
        (((7, 7), "/"), ((7, 3), "SW"), ((12, 3), "V")),
        (((8, 10), "EW"), ((10, 3), "EW"), ((8, 12), "EW")),
        60,
    ),
    LevelSpec(
        "Preserve the Longest Prefix",
        (
            "################",
            "#......#..#....#",
            "#......#..B....#",
            "#......#..#d...#",
            "#......#..#....#",
            "#......#..#....#",
            "#b....L...IM####",
            "#......#..#..#.#",
            "#......#..#..#.#",
            "#E....M...#c.#.#",
            "#......#..#..#.#",
            "#.....a#..C..#.#",
            "#P.....A..#..DX#",
            "#......#..#..#.#",
            "################",
        ),
        "E",
        (((7, 10), "/"), ((7, 7), "SW"), ((11, 7), "V"), ((12, 7), "/")),
        (((8, 13), "EW"), ((11, 3), "EW"), ((11, 12), "EW"), ((14, 13), "EW")),
        77,
    ),
)


@dataclass
class BeamTrace:
    segments: list[tuple[tuple[int, int], str | None, str | None]]
    impact: tuple[tuple[int, int], str] | None
    powered: set[str]


class BeamPathsHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BeamPaths | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        width = 60
        filled = max(0, min(width, round(width * game.remaining_steps / max(1, game.step_budget))))
        frame[62:64, 2:62] = WALL
        if filled:
            color = BEAM if game.remaining_steps > game.step_budget // 4 else INVALID
            frame[62:64, 2 : 2 + filled] = color
        return frame


def _level_from_spec(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), OUTSIDE, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level([board], grid_size=(64, 64), data={"spec": spec}, name=spec.name)


class BeamPaths(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = BeamPathsHud()
        self._frames: list[np.ndarray] = []
        self._finish_level_after_animation = False
        super().__init__(
            GAME_ID,
            [_level_from_spec(spec) for spec in LEVEL_SPECS],
            Camera(0, 0, 64, 64, OUTSIDE, OUTSIDE, [self._hud]),
            False,
            len(LEVEL_SPECS),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.rows = self.spec.rows
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        self.origin_x = (VIEW_SIZE - CELL_SIZE * self.width) // 2
        self.origin_y = (VIEW_SIZE - CELL_SIZE * self.height) // 2
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.player = self._find_symbol("P")
        self.emitter = self._find_symbol("E")
        self.exit_cell = self._find_symbol("X")
        self.components = {cell: orientation for cell, orientation in self.spec.orientations}
        self.component_kinds = {
            (x, y): {"M": "mirror", "I": "straight", "L": "elbow"}[self._symbol((x, y))] for x, y in self.components
        }
        self.door_axes = dict(self.spec.door_axes)
        self.receivers = {
            (x, y): char.upper()
            for y, row in enumerate(self.rows, 1)
            for x, char in enumerate(row, 1)
            if char in "abcd"
        }
        self.doors = {
            (x, y): char for y, row in enumerate(self.rows, 1) for x, char in enumerate(row, 1) if char in "ABCD"
        }
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[tuple[int, int], dict[tuple[int, int], str]]] = []
        self.trace = self._trace_beam()
        self.powered_channels = set(self.trace.powered)
        self._frames.clear()
        self._finish_level_after_animation = False
        self._sync_board()

    def _find_symbol(self, symbol: str) -> tuple[int, int]:
        for y, row in enumerate(self.rows, 1):
            x = row.find(symbol)
            if x >= 0:
                return x + 1, y
        raise ValueError(f"Missing {symbol!r} in {self.spec.name}")

    def _symbol(self, cell: tuple[int, int]) -> str:
        x, y = cell
        if not (1 <= x <= self.width and 1 <= y <= self.height):
            return "#"
        return self.rows[y - 1][x - 1]

    def _trace_beam(self) -> BeamTrace:
        segments: list[tuple[tuple[int, int], str | None, str | None]] = []
        powered: set[str] = set()
        direction = self.spec.emitter_direction
        cell = self.emitter
        seen: set[tuple[tuple[int, int], str]] = set()
        impact: tuple[tuple[int, int], str] | None = None
        while True:
            dx, dy = DIRS[direction]
            target = (cell[0] + dx, cell[1] + dy)
            state = (target, direction)
            if state in seen:
                impact = (target, OPPOSITE[direction])
                break
            seen.add(state)
            symbol = self._symbol(target)
            entry = OPPOSITE[direction]
            if symbol == "#":
                impact = (cell, direction)
                break
            if target in self.doors:
                axis = self.door_axes[target]
                if direction not in axis:
                    impact = (target, entry)
                    break
                segments.append((target, entry, direction))
                cell = target
                continue
            if target in self.components:
                kind = self.component_kinds[target]
                orientation = self.components[target]
                if kind == "mirror":
                    out = MIRROR_TURNS[orientation][direction]
                elif kind == "straight":
                    open_sides = ("E", "W") if orientation == "H" else ("N", "S")
                    if direction not in open_sides:
                        impact = (target, entry)
                        break
                    out = direction
                else:
                    sides = ELBOW_SIDES[orientation]
                    if entry not in sides:
                        impact = (target, entry)
                        break
                    out = sides[1] if sides[0] == entry else sides[0]
                segments.append((target, entry, out))
                cell, direction = target, out
                continue
            if target in self.receivers:
                segments.append((target, entry, None))
                powered.add(self.receivers[target])
                break
            if symbol == "E":
                impact = (target, entry)
                break
            segments.append((target, entry, direction))
            cell = target
        return BeamTrace(segments, impact, powered)

    def _rotate(self, cell: tuple[int, int]) -> None:
        kind = self.component_kinds[cell]
        old = self.components[cell]
        if kind == "mirror":
            self.components[cell] = "\\" if old == "/" else "/"
        elif kind == "straight":
            self.components[cell] = "H" if old == "V" else "V"
        else:
            self.components[cell] = ELBOW_ORDER[(ELBOW_ORDER.index(old) + 1) % 4]

    def _snapshot(self) -> tuple[tuple[int, int], dict[tuple[int, int], str]]:
        return self.player, dict(self.components)

    def _restore(self, snapshot: tuple[tuple[int, int], dict[tuple[int, int], str]]) -> None:
        self.player, components = snapshot
        self.components = dict(components)
        self.trace = self._trace_beam()
        self.powered_channels = set(self.trace.powered)

    def _walkable(self, cell: tuple[int, int]) -> bool:
        symbol = self._symbol(cell)
        if symbol == "#" or cell in self.components or symbol in "Eabcd":
            return False
        if cell in self.doors:
            return self.doors[cell] in self.powered_channels
        return symbol in ".PABCDX"

    def _spend_step(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.remaining_steps = 0
            self._sync_board()
            self.lose()
            self.complete_action()
            return True
        return False

    def step(self) -> None:
        if self._frames:
            self.board.pixels = self._frames.pop(0)
            if not self._frames:
                if self._finish_level_after_animation:
                    self.next_level()
                self.complete_action()
            return

        action = int(getattr(self.action.id, "value", self.action.id))
        if action == 0:
            self._sync_board()
            self.complete_action()
            return

        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            if self._spend_step():
                return
            self._sync_board()
            self.complete_action()
            return

        if action not in DELTAS:
            if self._spend_step():
                return
            self._sync_board()
            self.complete_action()
            return

        dx, dy = DELTAS[action]
        target = (self.player[0] + dx, self.player[1] + dy)
        old_trace = self.trace
        old_powered = set(self.powered_channels)
        if target in self.components:
            self.undo_history.append(self._snapshot())
            old_orientation = self.components[target]
            self._rotate(target)
            self.trace = self._trace_beam()
            self.powered_channels = set(self.trace.powered)
            if self._spend_step():
                return
            self._queue_rotation(target, old_orientation, old_trace, old_powered)
            return

        if self._walkable(target):
            self.undo_history.append(self._snapshot())
            self.player = target
            if self._spend_step():
                return
            if self.player == self.exit_cell:
                self._queue_completion()
            else:
                self._sync_board()
                self.complete_action()
            return

        if self._spend_step():
            return
        self._queue_invalid(target, (dx, dy))

    def _queue_rotation(
        self, cell: tuple[int, int], old_orientation: str, old_trace: BeamTrace, old_powered: set[str]
    ) -> None:
        frames = [
            self._render(
                trace=old_trace, powered=old_powered, highlight=cell, component_overrides={cell: old_orientation}
            )
        ]
        frames.append(self._render(trace=self.trace, powered=old_powered, trace_limit=0))
        total = len(self.trace.segments)
        for count in range(4, total + 4, 4):
            reveal_powered = self.powered_channels if count >= total else old_powered
            phase = 1 if count >= total and old_powered != self.powered_channels else 3
            frames.append(
                self._render(trace=self.trace, powered=reveal_powered, trace_limit=min(count, total), door_phase=phase)
            )
        if old_powered != self.powered_channels:
            frames.extend(
                self._render(trace=self.trace, powered=self.powered_channels, door_phase=phase) for phase in (2, 3)
            )
        frames.append(self._render())
        self.board.pixels = frames.pop(0)
        self._frames = frames

    def _queue_invalid(self, target: tuple[int, int], delta: tuple[int, int]) -> None:
        self._frames = [
            self._render(invalid=(target, delta), squash=delta),
            self._render(invalid=(target, delta)),
            self._render(),
        ]
        self.board.pixels = self._frames.pop(0)

    def _queue_completion(self) -> None:
        self._finish_level_after_animation = True
        self._frames = [self._render(exit_phase=phase) for phase in range(1, 5)]
        self.board.pixels = self._frames.pop(0)

    def _sync_board(self) -> None:
        self.board.pixels = self._render()

    def _cell_box(self, cell: tuple[int, int]) -> tuple[int, int]:
        return self.origin_x + CELL_SIZE * (cell[0] - 1), self.origin_y + CELL_SIZE * (cell[1] - 1)

    def _paint_arm(self, pixels: np.ndarray, cell: tuple[int, int], side: str, color: int) -> None:
        x, y = self._cell_box(cell)
        if side == "N":
            pixels[y : y + 2, x + 1 : x + 3] = color
        elif side == "S":
            pixels[y + 2 : y + 4, x + 1 : x + 3] = color
        elif side == "W":
            pixels[y + 1 : y + 3, x : x + 2] = color
        else:
            pixels[y + 1 : y + 3, x + 2 : x + 4] = color

    def _render(
        self,
        *,
        trace: BeamTrace | None = None,
        powered: set[str] | None = None,
        trace_limit: int | None = None,
        highlight: tuple[int, int] | None = None,
        invalid: tuple[tuple[int, int], tuple[int, int]] | None = None,
        squash: tuple[int, int] | None = None,
        door_phase: int = 3,
        exit_phase: int = 0,
        component_overrides: dict[tuple[int, int], str] | None = None,
    ) -> np.ndarray:
        trace = self.trace if trace is None else trace
        powered = self.powered_channels if powered is None else powered
        pixels = np.full((VIEW_SIZE, VIEW_SIZE), OUTSIDE, dtype=np.int8)
        for y, row in enumerate(self.rows, 1):
            for x, symbol in enumerate(row, 1):
                px, py = self._cell_box((x, y))
                pixels[py : py + 4, px : px + 4] = WALL if symbol == "#" else FLOOR

        for cell, channel in self.doors.items():
            self._draw_door(pixels, cell, channel, channel in powered, door_phase)
        for cell, channel in self.receivers.items():
            self._draw_receiver(pixels, cell, channel, channel in powered)
        self._draw_exit(pixels, exit_phase)
        self._draw_emitter(pixels)
        for cell, kind in self.component_kinds.items():
            orientation = (component_overrides or {}).get(cell, self.components[cell])
            self._draw_component(pixels, cell, kind, orientation)

        shown = trace.segments if trace_limit is None else trace.segments[:trace_limit]
        self._paint_arm(pixels, self.emitter, self.spec.emitter_direction, BEAM)
        for cell, entry, out in shown:
            if entry is not None:
                self._paint_arm(pixels, cell, entry, BEAM)
            if out is not None:
                self._paint_arm(pixels, cell, out, BEAM)
            px, py = self._cell_box(cell)
            pixels[py + 1 : py + 3, px + 1 : px + 3] = WHITE if cell in self.receivers else BEAM
        if trace_limit is None or trace_limit >= len(trace.segments):
            if trace.impact is not None:
                cell, side = trace.impact
                self._paint_arm(pixels, cell, side, BEAM)
                px, py = self._cell_box(cell)
                if side in "NS":
                    iy = py if side == "N" else py + 2
                    pixels[iy : iy + 2, px + 1 : px + 3] = WHITE
                else:
                    ix = px if side == "W" else px + 2
                    pixels[py + 1 : py + 3, ix : ix + 2] = WHITE

        if highlight is not None:
            px, py = self._cell_box(highlight)
            pixels[py, px + 2 : px + 4] = WHITE
            pixels[py : py + 2, px + 3] = WHITE
        if invalid is not None:
            target, delta = invalid
            px, py = self._cell_box(target)
            if delta[0] < 0:
                pixels[py + 1 : py + 3, px + 3] = INVALID
            elif delta[0] > 0:
                pixels[py + 1 : py + 3, px] = INVALID
            elif delta[1] < 0:
                pixels[py + 3, px + 1 : px + 3] = INVALID
            else:
                pixels[py, px + 1 : px + 3] = INVALID
        self._draw_player(pixels, squash)
        return pixels

    def _draw_emitter(self, pixels: np.ndarray) -> None:
        x, y = self._cell_box(self.emitter)
        pixels[y : y + 4, x : x + 4] = CASING
        pixels[y + 1 : y + 3, x + 1 : x + 3] = EMITTER
        self._paint_arm(pixels, self.emitter, self.spec.emitter_direction, BEAM)

    def _draw_component(self, pixels: np.ndarray, cell: tuple[int, int], kind: str, orientation: str) -> None:
        x, y = self._cell_box(cell)
        if kind == "mirror":
            pixels[y : y + 4, x : x + 4] = MIRROR
            for index in range(4):
                xx = 3 - index if orientation == "/" else index
                pixels[y + index, x + xx] = WHITE
            return
        pixels[y : y + 4, x : x + 4] = CASING
        sides = (("E", "W") if orientation == "H" else ("N", "S")) if kind == "straight" else ELBOW_SIDES[orientation]
        for side in sides:
            self._paint_arm(pixels, cell, side, CHANNEL)
        pixels[y + 1 : y + 3, x + 1 : x + 3] = CHANNEL

    def _draw_receiver(self, pixels: np.ndarray, cell: tuple[int, int], channel: str, powered: bool) -> None:
        x, y = self._cell_box(cell)
        color = CHANNEL_COLORS[channel]
        pixels[y : y + 4, x : x + 4] = color
        pixels[y + 1 : y + 3, x + 1 : x + 3] = WHITE if powered else WALL
        for side in DIRS:
            self._paint_arm(pixels, cell, side, FLOOR)
        pixels[y, x] = pixels[y, x + 3] = pixels[y + 3, x] = pixels[y + 3, x + 3] = color

    def _draw_door(self, pixels: np.ndarray, cell: tuple[int, int], channel: str, opened: bool, phase: int) -> None:
        x, y = self._cell_box(cell)
        color = CHANNEL_COLORS[channel]
        axis = self.door_axes[cell]
        pixels[y : y + 4, x : x + 4] = FLOOR
        progress = 0 if opened else 3
        if phase < 3:
            progress = (3 - phase) if opened else phase
        if axis == "EW":
            pixels[y, x : x + 4] = color
            pixels[y + 3, x : x + 4] = color
            pixels[y + 1, x : x + 4] = CHANNEL
            if progress >= 1:
                pixels[y + 2, (x, x + 3)] = CASING
            if progress >= 2:
                pixels[y + 2, x : x + 4] = CASING
            if progress >= 3:
                pixels[y + 2, x : x + 4] = WALL
        else:
            pixels[y : y + 4, x] = color
            pixels[y : y + 4, x + 3] = color
            pixels[y : y + 4, x + 1] = CHANNEL
            if progress >= 1:
                pixels[(y, y + 3), x + 2] = CASING
            if progress >= 2:
                pixels[y : y + 4, x + 2] = CASING
            if progress >= 3:
                pixels[y : y + 4, x + 2] = WALL

    def _draw_exit(self, pixels: np.ndarray, phase: int) -> None:
        x, y = self._cell_box(self.exit_cell)
        pixels[y : y + 4, x : x + 4] = EXIT
        pixels[y + 1 : y + 3, x + 1 : x + 3] = FLOOR
        if phase >= 2:
            pixels[max(0, y - 1) : min(64, y + 5), max(0, x - 1) : min(64, x + 5)] = EXIT
            pixels[y + 1 : y + 3, x + 1 : x + 3] = FLOOR
        if phase >= 4:
            pixels[y : y + 4, x : x + 4] = WHITE

    def _draw_player(self, pixels: np.ndarray, squash: tuple[int, int] | None) -> None:
        x, y = self._cell_box(self.player)
        if squash is not None:
            x += squash[0]
            y += squash[1]
        x = max(0, min(60, x))
        y = max(0, min(60, y))
        under = pixels[y : y + 4, x : x + 4].copy()
        pixels[y : y + 4, x : x + 4] = PLAYER
        pixels[y + 1 : y + 3, x + 1 : x + 3] = under[1:3, 1:3]
