from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

ACTION_CLICK = 6
VIEW_SIZE = 64
CELL = 8

BG = 4
WHITE = 0
LIGHT = 1
TRACK = 2
BUTTON = 3
RED = 8
BLUE = 9
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 6
MAGENTA = 6

DIR_DELTA = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
DIR_FROM_DELTA = {(0, -1): "N", (0, 1): "S", (-1, 0): "W", (1, 0): "E"}
ROT = {"N": 0, "E": 90, "S": 180, "W": 270}


@dataclass(frozen=True)
class RiderSpec:
    rid: str
    color: int
    start: tuple[int, int]
    oriented: bool = False
    facing: str = "E"


@dataclass(frozen=True)
class TargetSpec:
    rid: str
    slot: tuple[int, int]
    facing: str | None = None


@dataclass(frozen=True)
class CycleSpec:
    cid: str
    color: int
    slots: tuple[tuple[int, int], ...]
    wrap_faces: dict[tuple[tuple[int, int], tuple[int, int]], str]


@dataclass(frozen=True)
class ButtonSpec:
    name: str
    cell: tuple[int, int]
    ops: tuple[tuple[str, int], ...]
    color: int
    forward: bool


@dataclass(frozen=True)
class LevelSpec:
    name: str
    riders: tuple[RiderSpec, ...]
    targets: tuple[TargetSpec, ...]
    cycles: tuple[CycleSpec, ...]
    buttons: tuple[ButtonSpec, ...]
    budget: int


def _cycle(
    cid: str,
    color: int,
    slots: list[tuple[int, int]],
    wraps: dict[tuple[tuple[int, int], tuple[int, int]], str] | None = None,
) -> CycleSpec:
    return CycleSpec(cid, color, tuple(slots), wraps or {})


def _button(name: str, x: int, ops: list[tuple[str, int]], color: int, forward: bool) -> ButtonSpec:
    return ButtonSpec(name, (x, 7), tuple(ops), color, forward)


LEVEL_SPECS = (
    LevelSpec(
        "Single Loop",
        (RiderSpec("M", MAGENTA, (2, 1)),),
        (TargetSpec("M", (4, 2)),),
        (_cycle("A", BLUE, [(2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (3, 3), (2, 3), (2, 2)]),),
        (_button("A-", 3, [("A", -1)], BLUE, False), _button("A+", 4, [("A", 1)], BLUE, True)),
        18,
    ),
    LevelSpec(
        "Overlapping Belts",
        (RiderSpec("M", MAGENTA, (2, 1)),),
        (TargetSpec("M", (5, 3)),),
        (
            _cycle("A", BLUE, [(1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2)]),
            _cycle("B", ORANGE, [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (4, 4), (3, 4), (3, 3)]),
        ),
        (
            _button("A-", 0, [("A", -1)], BLUE, False),
            _button("A+", 1, [("A", 1)], BLUE, True),
            _button("B-", 6, [("B", -1)], ORANGE, False),
            _button("B+", 7, [("B", 1)], ORANGE, True),
        ),
        30,
    ),
    LevelSpec(
        "Rotation By Approach",
        (RiderSpec("M", MAGENTA, (5, 1), True, "E"),),
        (TargetSpec("M", (4, 2), "E"),),
        (
            _cycle("A", BLUE, [(4, 0), (5, 0), (5, 1), (5, 2), (4, 2), (4, 1)]),
            _cycle("B", ORANGE, [(4, 2), (4, 3), (3, 3), (2, 3), (2, 2), (3, 2)]),
        ),
        (
            _button("B-", 0, [("B", -1)], ORANGE, False),
            _button("B+", 1, [("B", 1)], ORANGE, True),
            _button("A-", 3, [("A", -1)], BLUE, False),
            _button("A+", 4, [("A", 1)], BLUE, True),
        ),
        24,
    ),
    LevelSpec(
        "Two Targets",
        (RiderSpec("M", MAGENTA, (2, 1)), RiderSpec("G", GREEN, (3, 4))),
        (TargetSpec("M", (3, 2)), TargetSpec("G", (5, 3))),
        (
            _cycle("A", BLUE, [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)]),
            _cycle("B", ORANGE, [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (4, 4), (3, 4), (3, 3)]),
        ),
        (
            _button("A-", 0, [("A", -1)], BLUE, False),
            _button("A+", 1, [("A", 1)], BLUE, True),
            _button("B-", 6, [("B", -1)], ORANGE, False),
            _button("B+", 7, [("B", 1)], ORANGE, True),
        ),
        48,
    ),
    LevelSpec(
        "Edge Teleport",
        (RiderSpec("M", MAGENTA, (4, 4), True, "N"), RiderSpec("G", GREEN, (5, 2))),
        (TargetSpec("M", (0, 2), "E"), TargetSpec("G", (3, 2))),
        (
            _cycle(
                "C",
                BLUE,
                [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)],
                {((6, 2), (0, 2)): "E", ((0, 2), (6, 2)): "W"},
            ),
            _cycle("A", ORANGE, [(4, 4), (5, 4), (6, 4), (6, 3), (6, 2), (5, 2), (4, 2), (4, 3)]),
        ),
        (
            _button("C-", 0, [("C", -1)], BLUE, False),
            _button("C+", 1, [("C", 1)], BLUE, True),
            _button("A-", 6, [("A", -1)], ORANGE, False),
            _button("A+", 7, [("A", 1)], ORANGE, True),
        ),
        54,
    ),
    LevelSpec(
        "Subset Brace",
        (RiderSpec("M", MAGENTA, (1, 1)), RiderSpec("G", GREEN, (5, 2))),
        (TargetSpec("M", (2, 1)), TargetSpec("G", (1, 3))),
        (
            _cycle(
                "R",
                BLUE,
                [
                    (1, 1),
                    (2, 1),
                    (3, 1),
                    (4, 1),
                    (5, 1),
                    (5, 2),
                    (5, 3),
                    (5, 4),
                    (4, 4),
                    (3, 4),
                    (2, 4),
                    (1, 4),
                    (1, 3),
                    (1, 2),
                ],
            ),
            _cycle("S", PURPLE, [(1, 1), (5, 1), (4, 4), (1, 3)]),
        ),
        (
            _button("R-", 0, [("R", -1)], BLUE, False),
            _button("R+", 1, [("R", 1)], BLUE, True),
            _button("S-", 6, [("S", -1)], PURPLE, False),
            _button("S+", 7, [("S", 1)], PURPLE, True),
        ),
        54,
    ),
    LevelSpec(
        "Remote Echo",
        (RiderSpec("M", MAGENTA, (0, 1)), RiderSpec("H", GREEN, (5, 1))),
        (TargetSpec("M", (0, 2)), TargetSpec("H", (7, 3))),
        (
            _cycle(
                "A_main",
                ORANGE,
                [
                    (0, 1),
                    (1, 1),
                    (2, 1),
                    (3, 1),
                    (4, 1),
                    (4, 2),
                    (4, 3),
                    (4, 4),
                    (3, 4),
                    (2, 4),
                    (1, 4),
                    (0, 4),
                    (0, 3),
                    (0, 2),
                ],
            ),
            _cycle("E", ORANGE, [(5, 1), (6, 1), (7, 1), (7, 2), (7, 3), (6, 3), (5, 3), (5, 2)]),
            _cycle("B", PURPLE, [(5, 1), (6, 3), (5, 2), (7, 2)]),
        ),
        (
            _button("A+", 0, [("A_main", 1), ("E", 1)], ORANGE, True),
            _button("B-", 6, [("B", -1)], PURPLE, False),
            _button("B+", 7, [("B", 1)], PURPLE, True),
        ),
        90,
    ),
    LevelSpec(
        "Final Alignment",
        (RiderSpec("M", MAGENTA, (2, 0), True, "E"), RiderSpec("G", GREEN, (4, 3)), RiderSpec("Y", YELLOW, (0, 4))),
        (TargetSpec("M", (1, 3), "E"), TargetSpec("G", (3, 3)), TargetSpec("Y", (2, 6))),
        (
            _cycle(
                "W",
                BLUE,
                [
                    (7, 2),
                    (6, 2),
                    (5, 2),
                    (4, 2),
                    (3, 2),
                    (2, 2),
                    (1, 2),
                    (0, 2),
                    (0, 3),
                    (1, 3),
                    (2, 3),
                    (3, 3),
                    (4, 3),
                    (5, 3),
                    (6, 3),
                    (7, 3),
                ],
            ),
            _cycle(
                "A",
                ORANGE,
                [
                    (2, 0),
                    (3, 0),
                    (4, 0),
                    (5, 0),
                    (6, 0),
                    (7, 0),
                    (7, 1),
                    (7, 2),
                    (6, 2),
                    (5, 2),
                    (4, 2),
                    (3, 2),
                    (2, 2),
                    (2, 1),
                ],
            ),
            _cycle("E", BLUE, [(0, 4), (1, 4), (2, 4), (2, 5), (2, 6), (1, 6), (0, 6), (0, 5)]),
            _cycle("S", PURPLE, [(0, 4), (1, 4), (2, 4), (2, 5), (2, 6), (1, 6), (0, 6), (0, 5)]),
        ),
        (
            _button("W+", 0, [("W", 1), ("E", 1)], BLUE, True),
            _button("A-", 3, [("A", -1)], ORANGE, False),
            _button("A+", 4, [("A", 1)], ORANGE, True),
            _button("S-", 6, [("S", -1)], PURPLE, False),
            _button("S+", 7, [("S", 1)], PURPLE, True),
        ),
        150,
    ),
)


def _blank() -> np.ndarray:
    return np.full((VIEW_SIZE, VIEW_SIZE), BG, dtype=np.int8)


def _center(cell: tuple[int, int]) -> tuple[int, int]:
    return cell[0] * CELL + 4, cell[1] * CELL + 4


def _draw_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
    frame[max(0, y0) : min(VIEW_SIZE, y1), max(0, x0) : min(VIEW_SIZE, x1)] = color


def _draw_line(
    frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int, width: int = 1, offset: int = 0
) -> None:
    x0, y0 = _center(a)
    x1, y1 = _center(b)
    if x0 == x1:
        x0 += offset
        x1 += offset
    elif y0 == y1:
        y0 += offset
        y1 += offset
    if abs(x0 - x1) <= 8 and abs(y0 - y1) <= 8:
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        _draw_rect(frame, xmin - width // 2, ymin - width // 2, xmax + width // 2 + 1, ymax + width // 2 + 1, color)
        return
    for x, y in ((x0, y0), (x1, y1)):
        frame[max(0, y - 3) : min(VIEW_SIZE, y + 4), max(0, x - 1) : min(VIEW_SIZE, x + 2)] = color
        frame[max(0, y - 1) : min(VIEW_SIZE, y + 2), max(0, x - 3) : min(VIEW_SIZE, x + 4)] = color


def _cycle_edge_pairs(cycle: CycleSpec) -> set[frozenset[tuple[int, int]]]:
    return {frozenset((slot, cycle.slots[(idx + 1) % len(cycle.slots)])) for idx, slot in enumerate(cycle.slots)}


def _shared_edge_offsets(cycles: tuple[CycleSpec, ...]) -> dict[tuple[str, frozenset[tuple[int, int]]], int]:
    edge_cycles: dict[frozenset[tuple[int, int]], list[str]] = {}
    for cycle in cycles:
        for edge in _cycle_edge_pairs(cycle):
            edge_cycles.setdefault(edge, []).append(cycle.cid)

    offsets: dict[tuple[str, frozenset[tuple[int, int]]], int] = {}
    for edge, cycle_ids in edge_cycles.items():
        if len(cycle_ids) == 1:
            offsets[(cycle_ids[0], edge)] = 0
            continue
        start = -len(cycle_ids) + 1
        for index, cycle_id in enumerate(cycle_ids):
            offsets[(cycle_id, edge)] = start + index * 2
    return offsets


def _draw_false_adjacency_gaps(frame: np.ndarray, cycle: CycleSpec) -> None:
    slots = set(cycle.slots)
    edges = _cycle_edge_pairs(cycle)
    for x, y in slots:
        for dx, dy in ((1, 0), (0, 1)):
            other = (x + dx, y + dy)
            if other not in slots or frozenset(((x, y), other)) in edges:
                continue
            if dx:
                _draw_rect(frame, (x + 1) * CELL - 1, y * CELL + 2, (x + 1) * CELL + 1, y * CELL + 6, BG)
            else:
                _draw_rect(frame, x * CELL + 2, (y + 1) * CELL - 1, x * CELL + 6, (y + 1) * CELL + 1, BG)


def _draw_node(frame: np.ndarray, cell: tuple[int, int], color: int = LIGHT) -> None:
    x, y = _center(cell)
    _draw_rect(frame, x - 2, y - 2, x + 3, y + 3, color)


def _draw_outline(frame: np.ndarray, cell: tuple[int, int], color: int, facing: str | None = None) -> None:
    x0, y0 = cell[0] * CELL, cell[1] * CELL
    frame[y0, x0 : x0 + CELL] = color
    frame[y0 + 7, x0 : x0 + CELL] = color
    frame[y0 : y0 + CELL, x0] = color
    frame[y0 : y0 + CELL, x0 + 7] = color
    if facing:
        cx, cy = _center(cell)
        dx, dy = DIR_DELTA[facing]
        frame[cy + dy * 3, cx + dx * 3] = WHITE
        frame[cy + dy * 2, cx + dx * 2] = WHITE


def _draw_collar(frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
    cx, cy = _center(cell)
    frame[cy - 3, cx - 3 : cx + 4] = color
    frame[cy + 3, cx - 3 : cx + 4] = color
    frame[cy - 3 : cy + 4, cx - 3] = color
    frame[cy - 3 : cy + 4, cx + 3] = color


def _draw_button(frame: np.ndarray, button: ButtonSpec) -> None:
    x0, y0 = button.cell[0] * CELL, button.cell[1] * CELL
    _draw_rect(frame, x0, y0, x0 + CELL, y0 + CELL, BUTTON)
    frame[y0, x0 : x0 + CELL] = button.color
    frame[y0 + 7, x0 : x0 + CELL] = button.color
    frame[y0 : y0 + CELL, x0] = button.color
    frame[y0 : y0 + CELL, x0 + 7] = button.color
    if button.forward:
        pts = [(x0 + 2, y0 + 2), (x0 + 5, y0 + 4), (x0 + 2, y0 + 6), (x0 + 2, y0 + 2)]
    else:
        pts = [(x0 + 5, y0 + 2), (x0 + 2, y0 + 4), (x0 + 5, y0 + 6), (x0 + 5, y0 + 2)]
    for px, py in pts:
        frame[py, px] = WHITE
    frame[y0 + 4, min(pts[1][0], pts[0][0]) : max(pts[1][0], pts[0][0]) + 1] = WHITE


def _draw_rider(frame: np.ndarray, cell: tuple[int, int], color: int, oriented: bool, facing: str) -> None:
    cx, cy = _center(cell)
    if not oriented:
        mask = (
            (0, -2),
            (-1, -1),
            (0, -1),
            (1, -1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (0, 2),
        )
        for dx, dy in mask:
            frame[cy + dy, cx + dx] = color
        frame[cy - 1, cx - 1] = WHITE
        return
    dx, dy = DIR_DELTA[facing]
    frame[cy + dy * 3, cx + dx * 3] = color
    for side in range(-2, 3):
        if dx:
            frame[cy + side, cx - dx] = color
        else:
            frame[cy - dy, cx + side] = color
    _draw_rect(frame, cx - 1, cy - 1, cx + 2, cy + 2, color)
    frame[cy, cx] = WHITE


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


class LoopingChains(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(
                [Sprite(_blank(), name="board", layer=0, collidable=False, tags=["board"])],
                (VIEW_SIZE, VIEW_SIZE),
                {"spec": spec},
                spec.name,
            )
            for spec in LEVEL_SPECS
        ]
        super().__init__(
            "looping_chains-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BG, BG),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.cycles = {cycle.cid: cycle for cycle in self.spec.cycles}
        self.cycle_indices = {
            cycle.cid: {slot: index for index, slot in enumerate(cycle.slots)} for cycle in self.spec.cycles
        }
        self.buttons = {button.cell: button for button in self.spec.buttons}
        self.rider_cells = {r.rid: r.start for r in self.spec.riders}
        self.rider_facing = {r.rid: r.facing for r in self.spec.riders}
        self.remaining_steps = int(self.spec.budget)
        self.flash_cell: tuple[int, int] | None = None
        self._fresh_level_tick = True
        self._sync_visuals()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.flash_cell = None
            self._sync_visuals()
            self._fresh_level_tick = False
            self.complete_action()
            return
        if self._fresh_level_tick:
            self._fresh_level_tick = False
            self.complete_action()
            return
        self.flash_cell = None
        self._spend_step()
        if _action_id(self.action.id) == ACTION_CLICK:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            cell = (max(0, min(63, x)) // CELL, max(0, min(63, y)) // CELL)
            button = self.buttons.get(cell)
            if button is None:
                self.flash_cell = cell
            else:
                self._operate(button)
        self._sync_visuals()
        if self._is_solved():
            last_level = self.is_last_level()
            self.next_level()
            if last_level:
                self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _operate(self, button: ButtonSpec) -> None:
        moves: dict[str, tuple[tuple[int, int], str | None]] = {}
        for rider in self.spec.riders:
            old = self.rider_cells[rider.rid]
            for cycle_id, direction in button.ops:
                indices = self.cycle_indices[cycle_id]
                if old not in indices:
                    continue
                cycle = self.cycles[cycle_id]
                new_index = (indices[old] + direction) % len(cycle.slots)
                new = cycle.slots[new_index]
                face = cycle.wrap_faces.get((old, new))
                if face is None and rider.oriented:
                    delta = (new[0] - old[0], new[1] - old[1])
                    face = DIR_FROM_DELTA.get(delta)
                moves[rider.rid] = (new, face)
                break
        for rid, (cell, face) in moves.items():
            self.rider_cells[rid] = cell
            if face is not None:
                self.rider_facing[rid] = face

    def _is_solved(self) -> bool:
        for target in self.spec.targets:
            if self.rider_cells[target.rid] != target.slot:
                return False
            if target.facing is not None and self.rider_facing[target.rid] != target.facing:
                return False
        return True

    def _sync_visuals(self) -> None:
        frame = _blank()
        edge_offsets = _shared_edge_offsets(self.spec.cycles)
        for cycle in self.spec.cycles:
            for idx, slot in enumerate(cycle.slots):
                nxt = cycle.slots[(idx + 1) % len(cycle.slots)]
                edge = frozenset((slot, nxt))
                _draw_line(frame, slot, nxt, cycle.color, 1, edge_offsets.get((cycle.cid, edge), 0))
            _draw_false_adjacency_gaps(frame, cycle)
        for cycle in self.spec.cycles:
            for slot in cycle.slots:
                _draw_node(frame, slot)
        for target in self.spec.targets:
            rider = next(r for r in self.spec.riders if r.rid == target.rid)
            _draw_outline(frame, target.slot, rider.color, target.facing)
        for button in self.spec.buttons:
            _draw_button(frame, button)
        if self.flash_cell is not None:
            x0, y0 = self.flash_cell[0] * CELL, self.flash_cell[1] * CELL
            _draw_rect(frame, x0 + 2, y0 + 2, x0 + 6, y0 + 6, RED)
        for rider in self.spec.riders:
            _draw_rider(frame, self.rider_cells[rider.rid], rider.color, rider.oriented, self.rider_facing[rider.rid])
        filled = int(64 * self.remaining_steps / max(1, self.spec.budget))
        frame[0, :] = RED
        if filled:
            frame[0, :filled] = GREEN
        self.board.pixels = frame
