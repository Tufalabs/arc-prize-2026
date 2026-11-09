from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

UP, DOWN, LEFT, RIGHT, SPACE, CLICK = 1, 2, 3, 4, 5, 6
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

BOARD_SIZE = 12
CELL = 4
BOARD_X = 8
BOARD_Y = 8

WHITE = 0
FLOOR = 1
SHADOW = 2
TAPE = 3
WALL_TOP = 4
OUTLINE = 5
MAGENTA = 6
RED = 8
RUG_DARK = 9
RUG_LIGHT = 10
FLASH = 11
ORANGE = 12
GREEN = 14
PURPLE = 15

RECT_3X2 = frozenset((x, y) for y in range(2) for x in range(3))
RECT_4X2 = frozenset((x, y) for y in range(2) for x in range(4))
RECT_3X3 = frozenset((x, y) for y in range(3) for x in range(3))
STEP = frozenset({(2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2)})
POCKET = frozenset({(3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 2), (1, 2), (0, 3), (1, 3)})
CROWN = frozenset(
    {(2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (4, 1), (0, 2), (1, 2), (4, 2), (5, 2), (0, 3), (1, 3), (4, 3), (5, 3)}
)


@dataclass(frozen=True)
class ParcelSpec:
    parcel_id: str
    color: int
    width: int
    height: int


@dataclass
class ParcelState:
    parcel_id: str
    color: int
    width: int
    height: int
    origin: tuple[int, int]

    def footprint(self, origin: tuple[int, int] | None = None) -> set[tuple[int, int]]:
        ox, oy = self.origin if origin is None else origin
        return {(ox + x, oy + y) for y in range(self.height) for x in range(self.width)}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    rug_shape: frozenset[tuple[int, int]]
    rug_origin: tuple[int, int]
    parcels: tuple[ParcelSpec, ...]
    baseline: int


# Layout markers are top-left anchors: uppercase letters are parcels, lowercase
# letters are their docks, # is every occupied wall cell, and . is floor.
# The rug origin/shape is separate because cargo deliberately overlaps the rug.
LEVEL_SPECS = (
    LevelSpec(
        "Ride the Rug",
        (
            "............",
            "............",
            "............",
            "............",
            "............",
            "...R.r......",
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
        ),
        RECT_3X2,
        (2, 5),
        (ParcelSpec("red", RED, 2, 2),),
        2,
    ),
    LevelSpec(
        "Slip Against the Wall",
        (
            "............",
            "............",
            "............",
            "............",
            "............",
            ".......#R.r.",
            ".......#....",
            "............",
            "............",
            "............",
            "............",
            "............",
        ),
        RECT_4X2,
        (8, 5),
        (ParcelSpec("red", RED, 2, 2),),
        4,
    ),
    LevelSpec(
        "Get All Four Corners Under",
        (
            "............",
            "............",
            "............",
            "........r...",
            "............",
            "....R.......",
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
        ),
        RECT_3X3,
        (2, 5),
        (ParcelSpec("red", RED, 2, 2),),
        7,
    ),
    LevelSpec(
        "Change Bays at the Corner",
        (
            "............",
            "............",
            "............",
            "...#R.......",
            "...#........",
            "...#........",
            "...#........",
            "...####.....",
            "............",
            "..........r.",
            "............",
            "............",
        ),
        STEP,
        (4, 2),
        (ParcelSpec("red", RED, 2, 2),),
        15,
    ),
    LevelSpec(
        "Split the Load",
        (
            "............",
            "..g.........",
            "............",
            "..##........",
            "......G.....",
            "....R.......",
            "............",
            "............",
            "............",
            "............",
            "........r...",
            "............",
        ),
        STEP,
        (4, 4),
        (ParcelSpec("red", RED, 2, 2), ParcelSpec("green", GREEN, 2, 2)),
        23,
    ),
    LevelSpec(
        "Long Crate, Small Pocket",
        (
            "..........p.",
            "............",
            "..........##",
            "............",
            "....P.#.....",
            ".O....#.....",
            "............",
            "............",
            "............",
            "o...........",
            "............",
            "............",
        ),
        POCKET,
        (1, 4),
        (ParcelSpec("orange", ORANGE, 2, 3), ParcelSpec("purple", PURPLE, 2, 2)),
        31,
    ),
    LevelSpec(
        "Exchange the Outer Bays",
        (
            "....G...g...",
            "............",
            "..R...M..#..",
            ".........#..",
            "............",
            "............",
            "..##........",
            "..##........",
            "............",
            "##..........",
            "m.........r.",
            "............",
        ),
        CROWN,
        (2, 0),
        (ParcelSpec("red", RED, 2, 2), ParcelSpec("green", GREEN, 2, 2), ParcelSpec("magenta", MAGENTA, 2, 2)),
        40,
    ),
)

PARCEL_MARKERS = {"R": "red", "G": "green", "O": "orange", "P": "purple", "M": "magenta"}
DOCK_MARKERS = {key.lower(): value for key, value in PARCEL_MARKERS.items()}


def _parse_layout(
    spec: LevelSpec,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], set[tuple[int, int]]]:
    if len(spec.rows) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in spec.rows):
        raise ValueError(f"{spec.name}: layout must be 12x12")
    origins: dict[str, tuple[int, int]] = {}
    docks: dict[str, tuple[int, int]] = {}
    walls: set[tuple[int, int]] = set()
    for y, row in enumerate(spec.rows):
        for x, marker in enumerate(row):
            if marker == "#":
                walls.add((x, y))
            elif marker in PARCEL_MARKERS:
                origins[PARCEL_MARKERS[marker]] = (x, y)
            elif marker in DOCK_MARKERS:
                docks[DOCK_MARKERS[marker]] = (x, y)
            elif marker != ".":
                raise ValueError(f"{spec.name}: unknown layout marker {marker!r}")
    expected = {parcel.parcel_id for parcel in spec.parcels}
    if set(origins) != expected or set(docks) != expected:
        raise ValueError(f"{spec.name}: parcel/dock markers do not match parcel specs")
    return origins, docks, walls


def _build_level(spec: LevelSpec) -> Level:
    origins, docks, walls = _parse_layout(spec)
    return Level(
        sprites=[],
        grid_size=(64, 64),
        data={"spec": spec, "origins": origins, "docks": docks, "walls": walls, "budget": spec.baseline * 6},
        name=spec.name,
    )


class RugShiftDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: RugShift | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = WHITE
        frame[BOARD_Y : BOARD_Y + BOARD_SIZE * CELL, BOARD_X : BOARD_X + BOARD_SIZE * CELL] = FLOOR
        self._draw_rug(frame, game)
        self._draw_parcels(frame, game)
        self._draw_docks(frame, game)
        self._draw_walls(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _visual_delta(self, game: RugShift, moving: bool) -> tuple[int, int]:
        visual = game._visual
        if visual is None:
            return (0, 0)
        if visual["kind"] == "move":
            pixels = int(visual["pixels"])
            dx, dy = visual["delta"]
            return (dx * (pixels - CELL), dy * (pixels - CELL)) if moving else (0, 0)
        if visual["kind"] == "bump" and moving:
            dx, dy = visual["delta"]
            return (dx * int(visual["pixels"]), dy * int(visual["pixels"]))
        return (0, 0)

    def _draw_rug(self, frame: np.ndarray, game: RugShift) -> None:
        off_x, off_y = self._visual_delta(game, True)
        rx, ry = game.rug_origin
        for lx, ly in game.rug_shape:
            px = BOARD_X + (rx + lx) * CELL + off_x
            py = BOARD_Y + (ry + ly) * CELL + off_y
            color = RUG_DARK if (lx + ly) % 2 == 0 else RUG_LIGHT
            self._fill_clipped(frame, px, py, CELL, CELL, color)

    def _draw_docks(self, frame: np.ndarray, game: RugShift) -> None:
        flash = game._visual is not None and game._visual["kind"] == "complete" and game._visual["flash"]
        for parcel in game.parcels:
            x, y = game.docks[parcel.parcel_id]
            x0, y0 = BOARD_X + x * CELL, BOARD_Y + y * CELL
            width, height = parcel.width * CELL, parcel.height * CELL
            color = FLASH if flash else parcel.color
            length = 3
            frame[y0 : y0 + 2, x0 : x0 + length] = color
            frame[y0 : y0 + length, x0 : x0 + 2] = color
            frame[y0 : y0 + 2, x0 + width - length : x0 + width] = color
            frame[y0 : y0 + length, x0 + width - 2 : x0 + width] = color
            frame[y0 + height - 2 : y0 + height, x0 : x0 + length] = color
            frame[y0 + height - length : y0 + height, x0 : x0 + 2] = color
            frame[y0 + height - 2 : y0 + height, x0 + width - length : x0 + width] = color
            frame[y0 + height - length : y0 + height, x0 + width - 2 : x0 + width] = color

    def _draw_parcels(self, frame: np.ndarray, game: RugShift) -> None:
        visual = game._visual
        moving_ids = set() if visual is None else set(visual.get("moving", ()))
        if visual is not None and visual.get("kind") == "move":
            support_key = "supported_after" if int(visual["pixels"]) == CELL else "supported_before"
            supported_ids = set(visual.get(support_key, ()))
        else:
            rug = game.rug_footprint()
            supported_ids = {parcel.parcel_id for parcel in game.parcels if parcel.footprint() <= rug}
        for parcel in game.parcels:
            moving = parcel.parcel_id in moving_ids
            supported = parcel.parcel_id in supported_ids
            off_x, off_y = self._visual_delta(game, moving)
            x0 = BOARD_X + parcel.origin[0] * CELL + off_x
            y0 = BOARD_Y + parcel.origin[1] * CELL + off_y
            width, height = parcel.width * CELL, parcel.height * CELL

            # Keep cargo visually independent from the patterned rug. A bright
            # outer rail means the parcel is fully supported and will ride with
            # the next valid rug move; a down-right shadow means it is grounded.
            if supported:
                self._outline(frame, x0 - 1, y0 - 1, width + 2, height + 2, FLASH)
            else:
                self._fill_clipped(frame, x0 + 1, y0 + height, width, 1, SHADOW)
                self._fill_clipped(frame, x0 + width, y0 + 1, 1, height, SHADOW)

            # Parcels are solid colored crates. The old rug-colored support
            # windows made the parcel and rug read as one articulated object.
            self._fill_clipped(frame, x0 + 1, y0 + 1, width - 2, height - 2, parcel.color)
            self._outline(frame, x0, y0, width, height, OUTLINE)
            flash_edges = () if visual is None else visual.get("flash_edges", {}).get(parcel.parcel_id, ())
            if visual is not None and visual.get("collision_frame"):
                for dx, dy in flash_edges:
                    self._draw_collision_edge(frame, x0, y0, width, height, dx, dy)

    def _draw_collision_edge(
        self, frame: np.ndarray, x0: int, y0: int, width: int, height: int, dx: int, dy: int
    ) -> None:
        if dx < 0:
            self._fill_clipped(frame, x0, y0, 1, height, FLASH)
        elif dx > 0:
            self._fill_clipped(frame, x0 + width - 1, y0, 1, height, FLASH)
        elif dy < 0:
            self._fill_clipped(frame, x0, y0, width, 1, FLASH)
        else:
            self._fill_clipped(frame, x0, y0 + height - 1, width, 1, FLASH)

    def _draw_walls(self, frame: np.ndarray, game: RugShift) -> None:
        for x, y in game.walls:
            x0, y0 = BOARD_X + x * CELL, BOARD_Y + y * CELL
            frame[y0 : y0 + CELL, x0 : x0 + CELL] = WALL_TOP
            if (x, y + 1) not in game.walls:
                frame[y0 + CELL - 1, x0 : x0 + CELL] = TAPE
            if (x + 1, y) not in game.walls:
                frame[y0 : y0 + CELL, x0 + CELL - 1] = TAPE

    def _draw_budget(self, frame: np.ndarray, game: RugShift) -> None:
        frame[3:5, BOARD_X : BOARD_X + BOARD_SIZE * CELL] = SHADOW
        fill = int(BOARD_SIZE * CELL * max(0, game.remaining_steps) / max(1, game.step_budget))
        if fill:
            frame[3:5, BOARD_X : BOARD_X + fill] = RUG_DARK
        # Seven tiny progress pips keep level position visible without text.
        for idx in range(len(LEVEL_SPECS)):
            color = GREEN if idx < game.level_index else (FLASH if idx == game.level_index else SHADOW)
            frame[59:61, 22 + idx * 3 : 24 + idx * 3] = color

    @staticmethod
    def _fill_clipped(frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
        x0, y0 = max(BOARD_X, x), max(BOARD_Y, y)
        x1, y1 = min(BOARD_X + BOARD_SIZE * CELL, x + w), min(BOARD_Y + BOARD_SIZE * CELL, y + h)
        if x0 < x1 and y0 < y1:
            frame[y0:y1, x0:x1] = color

    @classmethod
    def _outline(cls, frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
        cls._fill_clipped(frame, x, y, w, 1, color)
        cls._fill_clipped(frame, x, y + h - 1, w, 1, color)
        cls._fill_clipped(frame, x, y, 1, h, color)
        cls._fill_clipped(frame, x + w - 1, y, 1, h, color)


class RugShift(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = RugShiftDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "rug_shift-0001",
            levels,
            Camera(0, 0, 64, 64, WHITE, WHITE, [self._display]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK],
            seed,
        )
        self._display.game = self

    def on_set_level(self, _level: Level) -> None:
        spec: LevelSpec = self.current_level.get_data("spec")
        origins = self.current_level.get_data("origins")
        self.rug_shape = spec.rug_shape
        self.rug_origin = spec.rug_origin
        self.parcels = [
            ParcelState(p.parcel_id, p.color, p.width, p.height, tuple(origins[p.parcel_id])) for p in spec.parcels
        ]
        self.docks = {key: tuple(value) for key, value in self.current_level.get_data("docks").items()}
        self.walls = set(self.current_level.get_data("walls"))
        self.step_budget = int(self.current_level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.step_count = 0
        self.complete = False
        self.selected_object = None
        self.cursor_state = None
        self._visual: dict[str, object] | None = None
        self._frame_queue: list[dict[str, object]] = []

    def rug_footprint(self, origin: tuple[int, int] | None = None) -> set[tuple[int, int]]:
        ox, oy = self.rug_origin if origin is None else origin
        return {(ox + x, oy + y) for x, y in self.rug_shape}

    @staticmethod
    def _inside(cells: set[tuple[int, int]]) -> bool:
        return all(0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE for x, y in cells)

    def _is_solved(self) -> bool:
        return all(parcel.origin == self.docks[parcel.parcel_id] for parcel in self.parcels)

    def _resolve_move(self, delta: tuple[int, int]) -> None:
        dx, dy = delta
        proposed_origin = (self.rug_origin[0] + dx, self.rug_origin[1] + dy)
        if not self._inside(self.rug_footprint(proposed_origin)):
            self._frame_queue = [
                {"kind": "bump", "delta": delta, "pixels": 1},
                {"kind": "bump", "delta": delta, "pixels": 0},
            ]
            return

        rug_before = self.rug_footprint()
        eligible = {p.parcel_id for p in self.parcels if p.footprint() <= rug_before}
        by_id = {p.parcel_id: p for p in self.parcels}
        movable = set()
        wall_blocked: set[str] = set()
        destinations: dict[str, set[tuple[int, int]]] = {}
        for parcel_id in eligible:
            parcel = by_id[parcel_id]
            destination = parcel.footprint((parcel.origin[0] + dx, parcel.origin[1] + dy))
            destinations[parcel_id] = destination
            if destination & self.walls or not self._inside(destination):
                wall_blocked.add(parcel_id)
            else:
                movable.add(parcel_id)

        while True:
            old = set(movable)
            stationary = [parcel for parcel in self.parcels if parcel.parcel_id not in movable]
            for parcel_id in old:
                if any(destinations[parcel_id] & other.footprint() for other in stationary):
                    movable.remove(parcel_id)
            if movable == old:
                break

        # A stationary parcel and the cargo stopped against it both flash.
        flash_edges: dict[str, set[tuple[int, int]]] = {parcel_id: {delta} for parcel_id in wall_blocked}
        for parcel_id in eligible - movable:
            if parcel_id not in destinations:
                continue
            for other in self.parcels:
                if (
                    other.parcel_id != parcel_id
                    and other.parcel_id not in movable
                    and destinations[parcel_id] & other.footprint()
                ):
                    flash_edges.setdefault(parcel_id, set()).add(delta)
                    flash_edges.setdefault(other.parcel_id, set()).add((-dx, -dy))

        self.rug_origin = proposed_origin
        for parcel in self.parcels:
            if parcel.parcel_id in movable:
                parcel.origin = (parcel.origin[0] + dx, parcel.origin[1] + dy)

        rug_after = self.rug_footprint()
        supported_after = {p.parcel_id for p in self.parcels if p.footprint() <= rug_after}

        self.complete = self._is_solved()
        self._frame_queue = [
            {
                "kind": "move",
                "delta": delta,
                "pixels": pixels,
                "moving": tuple(movable),
                "supported_before": tuple(eligible),
                "supported_after": tuple(supported_after),
                "flash_edges": {key: tuple(value) for key, value in flash_edges.items()},
                "collision_frame": pixels == 1 and bool(flash_edges),
            }
            for pixels in range(1, CELL + 1)
        ]
        if self.complete:
            self._frame_queue.extend({"kind": "complete", "flash": bool(index % 2)} for index in range(4))

    def _finish_frame(self) -> None:
        if self._frame_queue:
            self._visual = self._frame_queue.pop(0)
        else:
            self._visual = None
        if self._frame_queue:
            return
        if self.complete:
            self.complete_action()
            self.next_level()
        elif self.remaining_steps <= 0:
            self.complete_action()
            self.lose()
        else:
            self.complete_action()

    def step(self) -> None:
        if self._frame_queue:
            self._finish_frame()
            return

        action = int(getattr(self.action.id, "value", self.action.id))
        if action == int(GameAction.RESET.value):
            self._visual = None
            self.complete_action()
            return
        if self.complete:
            self.complete_action()
            return

        self.step_count += 1
        self.remaining_steps -= 1
        self._visual = None
        if action in DELTAS:
            self._resolve_move(DELTAS[action])
        else:
            # SPACE and every valid click are deliberately inert, but still cost
            # one step under the mandatory derived budget contract.
            self.complete = self._is_solved()
            if self.complete:
                self._frame_queue = [{"kind": "complete", "flash": bool(index % 2)} for index in range(4)]

        if self._frame_queue:
            self._finish_frame()
        elif self.remaining_steps <= 0:
            self.complete_action()
            self.lose()
        else:
            self.complete_action()
