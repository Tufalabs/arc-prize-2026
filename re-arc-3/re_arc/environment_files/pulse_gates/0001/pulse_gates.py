from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pulse_gates-0001"
VIEW_SIZE = 64
BOARD_LEFT = 4
BOARD_TOP = 9
CELL_SIZE = 5

COLOR_HALO = 0
COLOR_DARK_ARM = 1
COLOR_DARK_CHANNEL = 2
COLOR_DARK_HUB = 3
COLOR_BACKGROUND = 4
COLOR_BLOCKED = 8
COLOR_SOURCE_CORE = 9
COLOR_SOURCE_BODY = 10
COLOR_PULSE = 11
COLOR_HINGE = 12
COLOR_RECEIVER = 14

DIRECTIONS = {"N": (0, -1, "S"), "E": (1, 0, "W"), "S": (0, 1, "N"), "W": (-1, 0, "E")}
ORIENTATION_ORDER = ("NE", "ES", "SW", "WN")
FIXED_SYMBOL_PORTS = {"-": "EW", "|": "NS", "T": "NSE", "t": "EWS", "p": "NE", "q": "NW"}
GATE_SYMBOLS = frozenset("ABCDEFGHI")

HUB = frozenset((u, v) for u in range(1, 4) for v in range(1, 4))
ARMS = {
    "N": frozenset((u, 0) for u in range(1, 4)),
    "E": frozenset((4, v) for v in range(1, 4)),
    "S": frozenset((u, 4) for u in range(1, 4)),
    "W": frozenset((0, v) for v in range(1, 4)),
}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    source_ports: str
    receiver_ports: str
    orientations: tuple[tuple[str, str], ...]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "One Bend",
        (
            "...........",
            "...........",
            "....R......",
            "....|......",
            ".S--A......",
            "...........",
            "...........",
            "...........",
            "...........",
        ),
        "E",
        "S",
        (("A", "SW"),),
        1,
    ),
    LevelSpec(
        "The Lit Frontier",
        (
            "...........",
            "...........",
            "...........",
            ".C-B.......",
            ".|.|.......",
            ".R.|.......",
            "S--A.......",
            "...........",
            "...........",
        ),
        "E",
        "N",
        (("A", "SW"), ("B", "ES"), ("C", "NE")),
        3,
    ),
    LevelSpec(
        "Keep the Feed",
        (
            "...........",
            "...........",
            "...........",
            "..R-A--t-S.",
            "....|..|...",
            "....B-C|...",
            "......pq...",
            "...........",
            "...........",
        ),
        "W",
        "E",
        (("A", "NE"), ("B", "WN"), ("C", "ES")),
        4,
    ),
    LevelSpec(
        "Three-Feed Weave",
        (
            "S..........",
            "T-A-B......",
            "|.|.|......",
            "T-C-D......",
            "|.|.|......",
            "p-E-FR.....",
            "...........",
            "...........",
            "...........",
        ),
        "S",
        "W",
        (("A", "ES"), ("B", "ES"), ("C", "SW"), ("D", "WN"), ("E", "SW"), ("F", "SW")),
        23,
    ),
    LevelSpec(
        "The Missing Corner",
        (
            "S..........",
            "T-A-B......",
            "|.|.|......",
            "T-C-D-E....",
            "|.|.|.|....",
            "p-F-G-HR...",
            "...........",
            "...........",
            "...........",
        ),
        "S",
        "W",
        (("A", "ES"), ("B", "ES"), ("C", "SW"), ("D", "WN"), ("E", "ES"), ("F", "SW"), ("G", "SW"), ("H", "SW")),
        37,
    ),
    LevelSpec(
        "Climb the Pulse",
        (
            "S.....R....",
            "T-A-B-C....",
            "|.|.|.|....",
            "T-D-E-F....",
            "|.|.|.|....",
            "p-G-H-I....",
            "...........",
            "...........",
            "...........",
        ),
        "S",
        "S",
        (
            ("A", "WN"),
            ("B", "ES"),
            ("C", "ES"),
            ("D", "ES"),
            ("E", "SW"),
            ("F", "ES"),
            ("G", "SW"),
            ("H", "SW"),
            ("I", "SW"),
        ),
        51,
    ),
    LevelSpec(
        "Full Handoff",
        (
            "S..........",
            "T-A-B-C....",
            "|.|.|.|....",
            "T-D-E-F....",
            "|.|.|.|....",
            "p-G-H-IR...",
            "...........",
            "...........",
            "...........",
        ),
        "S",
        "W",
        (
            ("A", "ES"),
            ("B", "SW"),
            ("C", "ES"),
            ("D", "SW"),
            ("E", "SW"),
            ("F", "WN"),
            ("G", "SW"),
            ("H", "SW"),
            ("I", "SW"),
        ),
        64,
    ),
)


def _build_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != 9 or any(len(row) != 11 for row in spec.rows):
        raise ValueError(f"{spec.name}: logical layout must be exactly 11x9")
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8),
        name="pulse_board",
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(
        sprites=[board],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={"spec": spec, "budget": spec.baseline_actions * 6},
    )


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class PulseGates(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board = level.get_sprites_by_tag("board")[0]
        self.occupied: dict[tuple[int, int], str] = {}
        self.gate_cells: dict[str, tuple[int, int]] = {}
        self.source_cell = (-1, -1)
        self.receiver_cell = (-1, -1)
        for y, row in enumerate(self.spec.rows):
            for x, symbol in enumerate(row):
                if symbol == ".":
                    continue
                self.occupied[(x, y)] = symbol
                if symbol == "S":
                    self.source_cell = (x, y)
                elif symbol == "R":
                    self.receiver_cell = (x, y)
                elif symbol in GATE_SYMBOLS:
                    self.gate_cells[symbol] = (x, y)
        self.orientations = dict(self.spec.orientations)
        self.reached_cells, self.distances = self._compute_reachability(self.orientations)
        self.reached_gates = self._gate_names_in(self.reached_cells)
        self.solved = False
        self.step_budget = int(level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[dict[str, str], bool]] = []
        self._frame_queue: list[dict[str, object]] = []
        self._pending_finish = False
        self._pending_loss = False
        self._queue_pulse_wave(self.orientations, self.reached_cells, self.distances)

    def _ports(self, cell: tuple[int, int], orientations: dict[str, str]) -> str:
        symbol = self.occupied[cell]
        if symbol == "S":
            return self.spec.source_ports
        if symbol == "R":
            return self.spec.receiver_ports
        if symbol in GATE_SYMBOLS:
            return orientations[symbol]
        return FIXED_SYMBOL_PORTS[symbol]

    def _compute_reachability(
        self, orientations: dict[str, str]
    ) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
        reached = {self.source_cell}
        distances = {self.source_cell: 0}
        queue = deque([self.source_cell])
        while queue:
            cell = queue.popleft()
            for direction in self._ports(cell, orientations):
                dx, dy, reciprocal = DIRECTIONS[direction]
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in self.occupied or reciprocal not in self._ports(neighbor, orientations):
                    continue
                if neighbor in reached:
                    continue
                reached.add(neighbor)
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
        return reached, distances

    def _gate_names_in(self, cells: set[tuple[int, int]]) -> set[str]:
        return {name for name, cell in self.gate_cells.items() if cell in cells}

    def _queue_pulse_wave(
        self, orientations: dict[str, str], reached: set[tuple[int, int]], distances: dict[tuple[int, int], int]
    ) -> None:
        maximum = max(distances.values(), default=0)
        for limit in range(0, maximum + 1, 2):
            visible = {cell for cell in reached if distances[cell] <= limit}
            self._frame_queue.append({"orientations": dict(orientations), "illuminated": visible})
        if not self._frame_queue or self._frame_queue[-1].get("illuminated") != reached:
            self._frame_queue.append({"orientations": dict(orientations), "illuminated": set(reached)})
        self._frame_queue.append({"orientations": dict(orientations), "illuminated": set(reached)})

    def _cell_origin(self, cell: tuple[int, int]) -> tuple[int, int]:
        return BOARD_LEFT + CELL_SIZE * cell[0], BOARD_TOP + CELL_SIZE * cell[1]

    def _draw_module_pixels(
        self, frame: np.ndarray, cell: tuple[int, int], pixels: set[tuple[int, int]], color: int
    ) -> None:
        x0, y0 = self._cell_origin(cell)
        for u, v in pixels:
            frame[y0 + v, x0 + u] = color

    def _draw_board(self, state: dict[str, object]) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)
        orientations = state.get("orientations", self.orientations)
        illuminated = state.get("illuminated", self.reached_cells)
        assert isinstance(orientations, dict)
        assert isinstance(illuminated, set)
        union_gate = state.get("union_gate")
        old_orientation = state.get("old_orientation")

        for cell, symbol in self.occupied.items():
            active = cell in illuminated
            ports = self._ports(cell, orientations)
            stencil = set(HUB)
            for port in ports:
                stencil.update(ARMS[port])

            if symbol == "S":
                self._draw_module_pixels(
                    frame, cell, set((u, v) for u in range(5) for v in range(5)), COLOR_SOURCE_BODY
                )
                self._draw_module_pixels(frame, cell, set(HUB), COLOR_SOURCE_CORE)
                continue

            if symbol == "R":
                x0, y0 = self._cell_origin(cell)
                frame[y0 : y0 + 5, x0 : x0 + 5] = COLOR_RECEIVER
                frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_RECEIVER if active else COLOR_BACKGROUND
                for port in ports:
                    self._draw_module_pixels(
                        frame, cell, set(ARMS[port]), COLOR_PULSE if active else COLOR_DARK_CHANNEL
                    )
                continue

            if symbol in GATE_SYMBOLS:
                arm_pixels = set().union(*(ARMS[port] for port in ports))
                self._draw_module_pixels(frame, cell, arm_pixels, COLOR_PULSE if active else COLOR_DARK_ARM)
                self._draw_module_pixels(frame, cell, set(HUB), COLOR_HALO if active else COLOR_DARK_HUB)
                if union_gate == symbol and isinstance(old_orientation, str):
                    sweep = set(HUB)
                    for port in set(old_orientation + orientations[symbol]):
                        sweep.update(ARMS[port])
                    self._draw_module_pixels(frame, cell, sweep, COLOR_HINGE)
                else:
                    cue = {(1, 1), (2, 1), (3, 2), (2, 3)}
                    if state.get("flash_gate") == symbol:
                        cue.update(HUB)
                    self._draw_module_pixels(frame, cell, cue, COLOR_HINGE)
                if active:
                    self._draw_module_pixels(frame, cell, {(0, 0), (4, 0), (0, 4), (4, 4)}, COLOR_HALO)
                continue

            self._draw_module_pixels(frame, cell, stencil, COLOR_PULSE if active else COLOR_DARK_CHANNEL)

        ripple = state.get("ripple")
        if isinstance(ripple, tuple):
            cx, cy, radius = ripple
            for px in range(max(0, cx - radius), min(VIEW_SIZE, cx + radius + 1)):
                for py in range(max(0, cy - radius), min(VIEW_SIZE, cy + radius + 1)):
                    if max(abs(px - cx), abs(py - cy)) == radius:
                        frame[py, px] = COLOR_BLOCKED

        completion_radius = state.get("completion_radius")
        if isinstance(completion_radius, int):
            cx0, cy0 = self._cell_origin(self.receiver_cell)
            cx, cy = cx0 + 2, cy0 + 2
            radius = completion_radius
            for px in range(max(0, cx - radius), min(VIEW_SIZE, cx + radius + 1)):
                for py in range(max(0, cy - radius), min(VIEW_SIZE, cy + radius + 1)):
                    if max(abs(px - cx), abs(py - cy)) == radius:
                        frame[py, px] = COLOR_HALO

        # Mandatory action bar: no text or grid decoration, just a compact decreasing meter.
        frame[58:61, BOARD_LEFT : BOARD_LEFT + 55] = COLOR_DARK_CHANNEL
        inner = int(53 * self.remaining_steps / max(1, self.step_budget))
        if inner > 0:
            frame[59, BOARD_LEFT + 1 : BOARD_LEFT + 1 + inner] = COLOR_PULSE
        self.board.pixels = frame

    def _play_frame(self) -> None:
        state = self._frame_queue.pop(0)
        self._draw_board(state)
        if self._frame_queue:
            return
        if self._pending_finish:
            self._pending_finish = False
            was_last = self.is_last_level()
            self.next_level()
            if was_last:
                self.complete_action()
            return
        if self._pending_loss:
            self._pending_loss = False
            self.lose()
        self.complete_action()

    def _spend_action(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _queue_invalid(self, x: int, y: int) -> None:
        self._frame_queue = [{"ripple": (x, y, 3)}, {"ripple": (x, y, 2)}, {}]

    def _resolve_click(self) -> None:
        data = self.action.data or {}
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        logical: tuple[int, int] | None = None
        if BOARD_LEFT <= x < BOARD_LEFT + 55 and BOARD_TOP <= y < BOARD_TOP + 45:
            logical = ((x - BOARD_LEFT) // CELL_SIZE, (y - BOARD_TOP) // CELL_SIZE)
        symbol = self.occupied.get(logical, "") if logical is not None else ""
        if symbol not in self.reached_gates:
            self._queue_invalid(x, y)
            return

        self.undo_history.append((dict(self.orientations), self.solved))
        old_orientations = dict(self.orientations)
        old_reached = set(self.reached_cells)
        old_orientation = self.orientations[symbol]
        next_index = (ORIENTATION_ORDER.index(old_orientation) + 1) % len(ORIENTATION_ORDER)
        self.orientations[symbol] = ORIENTATION_ORDER[next_index]
        self.reached_cells, self.distances = self._compute_reachability(self.orientations)
        self.reached_gates = self._gate_names_in(self.reached_cells)
        self.solved = self.receiver_cell in self.reached_cells

        self._frame_queue = [
            {"orientations": old_orientations, "illuminated": old_reached, "flash_gate": symbol},
            {
                "orientations": dict(self.orientations),
                "illuminated": set(),
                "union_gate": symbol,
                "old_orientation": old_orientation,
            },
            {"orientations": dict(self.orientations), "illuminated": set()},
        ]
        self._queue_pulse_wave(self.orientations, self.reached_cells, self.distances)
        if self.solved:
            for radius in (2, 4, 6):
                self._frame_queue.append(
                    {
                        "orientations": dict(self.orientations),
                        "illuminated": set(self.reached_cells),
                        "completion_radius": radius,
                    }
                )
            self._frame_queue.append({})
            self._pending_finish = True

    def _resolve_undo(self) -> None:
        if not self.undo_history:
            self._queue_invalid(2, 62)
            return
        orientations, solved = self.undo_history.pop()
        self.orientations = orientations
        self.solved = solved
        self.reached_cells, self.distances = self._compute_reachability(self.orientations)
        self.reached_gates = self._gate_names_in(self.reached_cells)
        self._frame_queue = [{}]

    def step(self) -> None:
        if self._frame_queue:
            self._play_frame()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._draw_board({})
            self.complete_action()
            return
        if self.solved:
            self._draw_board({})
            self.complete_action()
            return

        self._spend_action()
        if action == GameAction.ACTION6.value:
            self._resolve_click()
        elif action == GameAction.ACTION7.value:
            self._resolve_undo()
        else:
            self._frame_queue = [{}]

        if self.remaining_steps == 0 and not self.solved:
            self._pending_loss = True
        if self._frame_queue:
            self._play_frame()
        else:
            self.complete_action()
