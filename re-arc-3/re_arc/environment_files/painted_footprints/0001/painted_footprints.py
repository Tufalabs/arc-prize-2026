from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "painted_footprints-0001"
VIEW = 64
CELL = 5
LEFT = 2
TOP = 12

UP, DOWN, LEFT_ACTION, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT_ACTION: (-1, 0), RIGHT: (1, 0)}

BACKGROUND = 5
WALL = 4
BARE = 1
INK = 0
BODY = 2
BODY_ACCENT = 3
CORRECT = 14
FAMILY = {"B": (9, 10), "R": (8, 13), "Y": (11, 12)}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    pot: str
    initial_paint: dict[str, tuple[tuple[int, int], ...]]
    baseline: int

    @property
    def step_cap(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    LevelSpec(
        "The Wrong-Colored Step",
        (
            "############",
            "############",
            "#####b######",
            "#####.######",
            "####@B.#####",
            "####..######",
            "############",
            "############",
            "############",
            "############",
        ),
        "B",
        {"B": ((5, 5), (5, 6), (6, 6), (6, 5), (6, 4)), "R": ((7, 5),)},
        4,
    ),
    LevelSpec(
        "Paint, Then Circle",
        (
            "############",
            "############",
            "############",
            "###...######",
            "###@B.b#####",
            "###...######",
            "############",
            "############",
            "############",
            "############",
        ),
        "B",
        {"B": ((5, 5),)},
        10,
    ),
    LevelSpec(
        "Exchange the Pot",
        (
            "############",
            "############",
            "############",
            "###2..######",
            "###.#.######",
            "###@R.r#####",
            "###...######",
            "############",
            "############",
            "############",
        ),
        "B",
        {"R": ((5, 6),)},
        12,
    ),
    LevelSpec(
        "One Crossing, Two Times",
        (
            "############",
            "#####r######",
            "#####..1####",
            "#......@####",
            "#.B....b####",
            "#####...####",
            "#####R.2####",
            "#####..#####",
            "############",
            "############",
        ),
        "R",
        {"B": ((3, 5),), "R": ((6, 7),)},
        37,
    ),
    LevelSpec(
        "Passing Bay",
        (
            "############",
            "############",
            "###..@...###",
            "#.1.......##",
            "#rB.....Rb##",
            "#.........##",
            "##2#########",
            "############",
            "############",
            "############",
        ),
        "B",
        {"B": ((3, 5),), "R": ((9, 5),)},
        57,
    ),
    LevelSpec(
        "Brushback Spine",
        (
            "########b###",
            "........@###",
            ".#######.###",
            ".######..###",
            ".R.B.....###",
            "##.......###",
            "####2....###",
            "########.###",
            "########r###",
            "############",
        ),
        "B",
        {"B": ((4, 5), (9, 2)), "R": ((2, 5),)},
        58,
    ),
    LevelSpec(
        "Three-Color Relay",
        (
            "############",
            "#####y##2###",
            "........b###",
            ".##.#.##@.##",
            ".#........##",
            ".R.B.....r.#",
            "#####......#",
            "#####.##.#3#",
            "#####Y##.###",
            "#####....###",
        ),
        "B",
        {"B": ((4, 6), (9, 4)), "R": ((2, 6),), "Y": ((6, 9),)},
        92,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _cell_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return LEFT + CELL * (x - 1), TOP + CELL * (y - 1)


def _build_level(spec: LevelSpec) -> Level:
    canvas = Sprite(
        np.full((VIEW, VIEW), BACKGROUND, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        visible=True,
        collidable=False,
        tags=["canvas"],
    )
    return Level([canvas], (VIEW, VIEW), {"spec": spec}, spec.name)


class PaintedFootprints(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW, VIEW, BACKGROUND, BACKGROUND, []),
            False,
            len(levels),
            [UP, DOWN, LEFT_ACTION, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.floor: set[tuple[int, int]] = set()
        self.pads: dict[tuple[int, int], str] = {}
        self.stations: dict[tuple[int, int], str] = {}
        self.parcel_colors: dict[str, str] = {}
        self.initial_parcels: dict[str, tuple[int, int]] = {}
        self.initial_courier = (1, 1)
        counts = {"B": 0, "R": 0, "Y": 0}
        for y, row in enumerate(self.spec.rows, 1):
            if len(row) != 12:
                raise ValueError(f"{self.spec.name}: row {y} is not 12 cells")
            for x, glyph in enumerate(row, 1):
                cell = (x, y)
                if glyph in ".@BRY":
                    self.floor.add(cell)
                if glyph == "@":
                    self.initial_courier = cell
                elif glyph in "BRY":
                    counts[glyph] += 1
                    parcel_id = f"{glyph}{counts[glyph]}"
                    self.parcel_colors[parcel_id] = glyph
                    self.initial_parcels[parcel_id] = cell
                elif glyph in "bry":
                    self.pads[cell] = glyph.upper()
                elif glyph in "123":
                    self.stations[cell] = {"1": "B", "2": "R", "3": "Y"}[glyph]
        self.initial_paint = {cell: None for cell in self.floor}
        for color, cells in self.spec.initial_paint.items():
            for cell in cells:
                if cell not in self.floor:
                    raise ValueError(f"paint outside ordinary floor: {cell}")
                self.initial_paint[cell] = color
        self.step_cap = self.spec.step_cap
        self._undo_history: list[
            tuple[tuple[int, int], str, dict[str, tuple[int, int]], dict[tuple[int, int], str | None], str]
        ] = []
        self._animation: list[dict[str, object]] = []
        self._animation_terminal = False
        self._visual: dict[str, object] = {}
        self._restore_initial()

    def _restore_initial(self) -> None:
        self.courier_pos = self.initial_courier
        self.pot_color = self.spec.pot
        self.parcel_pos = dict(self.initial_parcels)
        self.paint = dict(self.initial_paint)
        self.step_count = 0
        self.status = "PLAYING"
        self._undo_history.clear()
        self._animation.clear()
        self._animation_terminal = False
        self._visual = {}
        self._render()

    def _snapshot(
        self,
    ) -> tuple[tuple[int, int], str, dict[str, tuple[int, int]], dict[tuple[int, int], str | None], str]:
        return self.courier_pos, self.pot_color, dict(self.parcel_pos), dict(self.paint), self.status

    def _restore_snapshot(
        self, snap: tuple[tuple[int, int], str, dict[str, tuple[int, int]], dict[tuple[int, int], str | None], str]
    ) -> None:
        self.courier_pos, self.pot_color, parcels, paint, self.status = snap
        self.parcel_pos = dict(parcels)
        self.paint = dict(paint)

    def _parcel_at(self, cell: tuple[int, int]) -> str | None:
        return next((parcel_id for parcel_id, pos in self.parcel_pos.items() if pos == cell), None)

    def _won(self) -> bool:
        return len(self.parcel_pos) == len(self.pads) and all(
            self.pads.get(pos) == self.parcel_colors[parcel_id] for parcel_id, pos in self.parcel_pos.items()
        )

    def _support(self, cell: tuple[int, int]) -> str | None:
        if cell in self.pads:
            return self.pads[cell]
        return self.paint.get(cell)

    def _start_animation(self, frames: list[dict[str, object]], terminal: bool = False) -> None:
        self._visual = frames[0]
        self._animation = frames[1:]
        self._animation_terminal = terminal
        self._render()

    def _advance_animation(self) -> None:
        if self._animation:
            self._visual = self._animation.pop(0)
            self._render()
            if self._animation:
                return
        if self._animation_terminal:
            self.next_level()
        self._animation_terminal = False
        self.complete_action()

    def _finish_direction(self, frames: list[dict[str, object]]) -> None:
        won = self._won()
        if won:
            self.status = "WON"
            frames.extend({"completion": phase} for phase in range(6))
        elif self.step_count >= self.step_cap:
            self.status = "FAILED"
            frames.extend(({"edge_flash": True}, {}))
        self._start_animation(frames, terminal=won)

    def _move(self, dx: int, dy: int) -> None:
        self.step_count += 1
        source = self.courier_pos
        adjacent = (source[0] + dx, source[1] + dy)
        parcel_id = self._parcel_at(adjacent)

        if parcel_id is not None:
            destination = (adjacent[0] + dx, adjacent[1] + dy)
            allowed_surface = destination in self.floor or destination in self.pads
            succeeds = (
                allowed_surface
                and destination not in self.stations
                and self._parcel_at(destination) is None
                and self._support(destination) == self.parcel_colors[parcel_id]
            )
            if not succeeds:
                frames = [
                    {"failed_push": (parcel_id, adjacent, dx, dy, 1)},
                    {"failed_push": (parcel_id, adjacent, dx, dy, 0), "outline": destination},
                    {},
                ]
                self._finish_direction(frames)
                return

            self._undo_history.append(self._snapshot())
            self.parcel_pos[parcel_id] = destination
            self.courier_pos = adjacent
            if adjacent in self.floor:
                self.paint[adjacent] = self.pot_color
            sx, sy = _cell_pixel(adjacent)
            dxp, dyp = dx * CELL, dy * CELL
            cx, cy = _cell_pixel(source)
            push_frames = [
                {"parcel_pixel": (parcel_id, sx + dxp // 3, sy + dyp // 3), "courier_pixel": (cx, cy)},
                {
                    "parcel_pixel": (parcel_id, sx + (2 * dxp) // 3, sy + (2 * dyp) // 3),
                    "courier_pixel": (cx + dxp // 3, cy + dyp // 3),
                },
                {},
            ]
            self._finish_direction(push_frames)
            return

        if adjacent in self.floor or adjacent in self.pads or adjacent in self.stations:
            self._undo_history.append(self._snapshot())
            self.courier_pos = adjacent
            station = self.stations.get(adjacent)
            if adjacent in self.floor:
                self.paint[adjacent] = self.pot_color
            elif station is not None:
                self.pot_color = station
            frames = [{"station_pulse": adjacent}, {}] if station is not None else [{}]
            self._finish_direction(frames)
            return

        ax, ay = _cell_pixel(source)
        frames = [{"courier_pixel": (ax + dx, ay + dy)}, {}]
        self._finish_direction(frames)

    def step(self) -> None:
        if self._animation:
            self._advance_animation()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._render()
            self.complete_action()
            return
        if action == SPACE:
            self._restore_initial()
            self.complete_action()
            return
        if action == CLICK:
            self.complete_action()
            return
        if action == UNDO:
            if self._undo_history:
                self._restore_snapshot(self._undo_history.pop())
            self._visual = {}
            self._render()
            self.complete_action()
            return
        if action in DELTAS and self.status == "PLAYING":
            self._move(*DELTAS[action])
            if not self._animation:
                self.complete_action()
            return
        self.complete_action()

    def _pattern(self, color: str, size: int = 5) -> np.ndarray:
        base, accent = FAMILY[color]
        arr = np.full((size, size), base, dtype=np.int8)
        center = size // 2
        if color == "B":
            arr[:, center] = accent
        elif color == "R":
            np.fill_diagonal(arr, accent)
        else:
            arr[center, :] = accent
        return arr

    def _draw_cell(self, frame: np.ndarray, cell: tuple[int, int], pixels: np.ndarray | int) -> None:
        x, y = _cell_pixel(cell)
        frame[y : y + CELL, x : x + CELL] = pixels

    def _draw_pad(self, frame: np.ndarray, cell: tuple[int, int], color: str, occupied: bool, pulse: bool) -> None:
        tile = self._pattern(color)
        border = CORRECT if occupied else FAMILY[color][0]
        tile[0, :] = tile[-1, :] = border
        tile[:, 0] = tile[:, -1] = border
        if pulse:
            tile[1, 1:-1] = tile[-2, 1:-1] = CORRECT
            tile[1:-1, 1] = tile[1:-1, -2] = CORRECT
        self._draw_cell(frame, cell, tile)

    def _draw_station(self, frame: np.ndarray, cell: tuple[int, int], color: str, pulse: bool) -> None:
        tile = np.full((CELL, CELL), FAMILY[color][0], dtype=np.int8)
        tile[0, :] = INK
        tile[:, 0] = INK
        tile[-1, :] = BODY_ACCENT
        tile[:, -1] = BODY_ACCENT
        interior = self._pattern(color, 3)
        if pulse:
            interior = np.where(interior == FAMILY[color][0], FAMILY[color][1], FAMILY[color][0])
        tile[1:4, 1:4] = interior
        self._draw_cell(frame, cell, tile)

    def _draw_parcel_pixel(self, frame: np.ndarray, pixel: tuple[int, int], color: str) -> None:
        x, y = pixel
        # Parcels use a dark rigid outline so they cannot be mistaken for a
        # painted footprint or for the courier standing on a goal pad.
        for ox, oy in ((0, 0), (4, 0), (0, 4), (4, 4)):
            frame[y + oy, x + ox] = INK
        frame[y + 1 : y + 4, x + 1 : x + 4] = self._pattern(color, 3)
        frame[y + 2, x + 2] = FAMILY[color][1]

    def _draw_push_hint(self, frame: np.ndarray) -> None:
        cx, cy = self.courier_pos
        for dx, dy in DELTAS.values():
            parcel_cell = (cx + dx, cy + dy)
            parcel_id = self._parcel_at(parcel_cell)
            if parcel_id is None:
                continue
            destination = (parcel_cell[0] + dx, parcel_cell[1] + dy)
            valid = (
                destination in self.floor | set(self.pads)
                and destination not in self.stations
                and self._parcel_at(destination) is None
                and self._support(destination) == self.parcel_colors[parcel_id]
            )
            px, py = _cell_pixel(parcel_cell)
            color = CORRECT if valid else INK
            if dx:
                edge = px + (CELL - 1 if dx > 0 else 0)
                frame[py + 1 : py + CELL - 1, edge] = color
            else:
                edge = py + (CELL - 1 if dy > 0 else 0)
                frame[edge, px + 1 : px + CELL - 1] = color

    def _draw_courier_pixel(self, frame: np.ndarray, pixel: tuple[int, int]) -> None:
        x, y = pixel
        body = np.array([[INK, BODY], [BODY, BODY], [BODY_ACCENT, INK]], dtype=np.int8)
        frame[y + 1 : y + 4, x + 1 : x + 3] = body
        pot = self._pattern(self.pot_color, 2)
        frame[y : y + 2, x + 2 : x + 4] = pot

    def _render(self) -> None:
        frame = np.full((VIEW, VIEW), BACKGROUND, dtype=np.int8)
        frame[TOP : TOP + 10 * CELL, LEFT : LEFT + 12 * CELL] = WALL
        for cell in self.floor:
            color = self.paint[cell]
            self._draw_cell(frame, cell, BARE if color is None else self._pattern(color))
        for cell, color in self.pads.items():
            parcel_id = self._parcel_at(cell)
            occupied = parcel_id is not None and self.parcel_colors[parcel_id] == color
            pulse_phase = self._visual.get("completion")
            pulse = occupied and pulse_phase is not None and int(pulse_phase) % 2 == 0
            self._draw_pad(frame, cell, color, occupied, pulse)
        station_pulse = self._visual.get("station_pulse")
        for cell, color in self.stations.items():
            self._draw_station(frame, cell, color, station_pulse == cell)

        parcel_override = self._visual.get("parcel_pixel")
        failed = self._visual.get("failed_push")
        for parcel_id, cell in self.parcel_pos.items():
            pixel = _cell_pixel(cell)
            if parcel_override and parcel_override[0] == parcel_id:
                pixel = tuple(parcel_override[1:])
            elif failed and failed[0] == parcel_id:
                base_x, base_y = _cell_pixel(failed[1])
                pixel = (base_x + failed[2] * failed[4], base_y + failed[3] * failed[4])
            self._draw_parcel_pixel(frame, pixel, self.parcel_colors[parcel_id])

        if not self._visual:
            self._draw_push_hint(frame)

        courier_pixel = self._visual.get("courier_pixel", _cell_pixel(self.courier_pos))
        self._draw_courier_pixel(frame, courier_pixel)

        outline = self._visual.get("outline")
        if outline is not None:
            ox, oy = _cell_pixel(outline)
            if 0 <= ox < VIEW - CELL + 1 and TOP <= oy < VIEW - CELL + 1:
                frame[oy, ox : ox + CELL] = INK
                frame[oy + CELL - 1, ox : ox + CELL] = INK
                frame[oy : oy + CELL, ox] = INK
                frame[oy : oy + CELL, ox + CELL - 1] = INK

        # HUD pot: a large, unmistakable version of the carried color motif.
        frame[1:8, 2:9] = self._pattern(self.pot_color, 7)
        frame[0:9, 0] = INK
        frame[0, 0:11] = INK
        frame[8, 0:11] = BODY_ACCENT
        frame[0:9, 10] = BODY_ACCENT
        spent = min(48, int(48 * self.step_count / max(1, self.step_cap)))
        frame[3:6, 14:62] = WALL
        remaining_color = 12 if self.step_cap - self.step_count < self.step_cap * 0.15 else BODY
        frame[3:6, 14 + spent : 62] = remaining_color
        if self._visual.get("edge_flash"):
            frame[0, :] = frame[-1, :] = 13
            frame[:, 0] = frame[:, -1] = 13
        self.canvas.pixels = frame
