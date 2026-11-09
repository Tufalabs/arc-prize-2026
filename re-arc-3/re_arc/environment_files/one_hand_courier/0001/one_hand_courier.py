from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "one_hand_courier-0001"
VIEW_SIZE = 64
CELL = 5
ORIGIN_X = 4
ORIGIN_Y = 14

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = 1, 2, 3, 4, 5, 6, 7
MOVE = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

BG = 5
WALL = 4
FLOOR = 1
NEUTRAL = 3
SOCKET_EMPTY = 2
WHITE = 0
A_DARK, A_LIGHT = 9, 10
B_DARK, B_LIGHT = 11, 12
PARCEL_DARK, PARCEL_LIGHT = 6, 7
INVALID = 8
SUCCESS = 14

MASKS = {
    "A": (".AAA.", "AA.AA", ".AaA.", "..A..", "..A.."),
    "B": ("..B..", ".BBB.", "BBbBB", "..B..", "..B.."),
    "P": (".PPP.", "PPpPP", "PpppP", "PPpPP", ".PPP."),
}


@dataclass(frozen=True)
class SocketSpec:
    socket_id: str
    cell: tuple[int, int]
    kind: str
    gate_id: str


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    kind: str
    a: tuple[int, int]
    b: tuple[int, int]
    socket_id: str | None


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    sockets: tuple[SocketSpec, ...]
    gates: tuple[GateSpec, ...]
    initial_hand: str | None
    baseline: int


def _s(socket_id: str, x: int, y: int, kind: str, gate_id: str) -> SocketSpec:
    return SocketSpec(socket_id, (x, y), kind, gate_id)


def _g(gate_id: str, kind: str, ax: int, ay: int, bx: int, by: int, socket_id: str | None) -> GateSpec:
    return GateSpec(gate_id, kind, (ax, ay), (bx, by), socket_id)


LEVEL_SPECS = (
    LevelSpec(
        "Empty the Hand",
        (
            "###########",
            "###########",
            "###########",
            "###########",
            "##@p..x####",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        (),
        (),
        "A",
        6,
    ),
    LevelSpec(
        "Permission in Hand",
        (
            "###########",
            "###########",
            "###########",
            "#######p.##",
            "#@a......##",
            "########x##",
            "###########",
            "###########",
            "###########",
        ),
        (),
        (_g("GA", "A", 5, 5, 6, 5, None),),
        None,
        13,
    ),
    LevelSpec(
        "Leave Permission Behind",
        (
            "###########",
            "###########",
            "###########",
            "####1######",
            "##@a..p####",
            "##x########",
            "###########",
            "###########",
            "###########",
        ),
        (_s("1", 5, 4, "A", "GA"),),
        (_g("GA", "A", 5, 5, 6, 5, "1"),),
        None,
        14,
    ),
    LevelSpec(
        "Move the Latch",
        (
            "###########",
            "###########",
            "###########",
            "#a#1..2####",
            "#@.....x###",
            "#b#..3.####",
            "####p######",
            "###########",
            "###########",
        ),
        (_s("1", 4, 4, "A", "GA1"), _s("2", 7, 4, "A", "GA2"), _s("3", 6, 6, "B", "GB")),
        (_g("GA1", "A", 3, 5, 4, 5, "1"), _g("GA2", "A", 7, 5, 8, 5, "2"), _g("GB", "B", 5, 6, 5, 7, "3")),
        None,
        39,
    ),
    LevelSpec(
        "Empty the Vault",
        (
            "###########",
            "###########",
            "####a######",
            "#b1#.2##3.#",
            "#.........#",
            "#p.#@.##.x#",
            "####.######",
            "###########",
            "###########",
        ),
        (_s("1", 3, 4, "A", "GA1"), _s("2", 6, 4, "A", "GA2"), _s("3", 9, 4, "B", "GB")),
        (_g("GA1", "A", 3, 5, 4, 5, "1"), _g("GA2", "A", 6, 5, 7, 5, "2"), _g("GB", "B", 8, 5, 9, 5, "3")),
        None,
        58,
    ),
    LevelSpec(
        "Close Behind the Cargo",
        (
            "###########",
            "###########",
            "#a#1.#2####",
            "#@........#",
            "#b##p#3.#x#",
            "#######.4##",
            "###########",
            "###########",
            "###########",
        ),
        (_s("1", 4, 3, "A", "GA1"), _s("2", 7, 3, "A", "GA2"), _s("3", 7, 5, "A", "GA3"), _s("4", 9, 6, "B", "GB")),
        (
            _g("GA1", "A", 3, 4, 4, 4, "1"),
            _g("GA2", "A", 6, 4, 7, 4, "2"),
            _g("GA3", "A", 8, 5, 8, 6, "3"),
            _g("GB", "B", 8, 4, 9, 4, "4"),
        ),
        None,
        62,
    ),
    LevelSpec(
        "Temporary Roads",
        (
            "###########",
            "###########",
            "#a#####..##",
            "#@#1..3.5##",
            "#......##x#",
            "#b##.#.4..#",
            "####.2#####",
            "####.p#####",
            "###########",
        ),
        (
            _s("1", 4, 4, "A", "GA_supply"),
            _s("2", 6, 7, "B", "GB_vault"),
            _s("3", 7, 4, "B", "GB_service"),
            _s("4", 8, 6, "B", "GB_final"),
            _s("5", 9, 4, "A", "GA_final"),
        ),
        (
            _g("GA_supply", "A", 3, 5, 4, 5, "1"),
            _g("GB_vault", "B", 5, 6, 5, 7, "2"),
            _g("GB_service", "B", 7, 4, 8, 4, "3"),
            _g("GB_final", "B", 7, 6, 8, 6, "4"),
            _g("GA_final", "A", 9, 6, 10, 6, "5"),
        ),
        None,
        68,
    ),
)


def _cell_xy(cell: tuple[int, int]) -> tuple[int, int]:
    return ORIGIN_X + (cell[0] - 1) * CELL, ORIGIN_Y + (cell[1] - 1) * CELL


def _object_kind(object_id: str) -> str:
    return "P" if object_id == "parcel" else object_id


def _mask_colors(kind: str) -> tuple[int, int]:
    if kind == "A":
        return A_DARK, A_LIGHT
    if kind == "B":
        return B_DARK, B_LIGHT
    return PARCEL_DARK, PARCEL_LIGHT


class OneHandCourier(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = []
        for spec in LEVEL_SPECS:
            canvas = Sprite(
                np.full((VIEW_SIZE, VIEW_SIZE), BG, dtype=np.int8),
                name="canvas",
                x=0,
                y=0,
                layer=0,
                collidable=False,
                tags=["canvas"],
            )
            levels.append(Level([canvas], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name))
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BG, BG)
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 7], seed=seed)

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.floor: set[tuple[int, int]] = set()
        self.courier = (1, 1)
        self.exit_cell = (1, 1)
        self.object_locations: dict[str, tuple[str, tuple[int, int] | str | None]] = {}
        for y, row in enumerate(self.spec.rows, start=1):
            for x, symbol in enumerate(row, start=1):
                if symbol == "#":
                    continue
                self.floor.add((x, y))
                if symbol == "@":
                    self.courier = (x, y)
                elif symbol == "x":
                    self.exit_cell = (x, y)
                elif symbol == "a":
                    self.object_locations["A"] = ("FLOOR", (x, y))
                elif symbol == "b":
                    self.object_locations["B"] = ("FLOOR", (x, y))
                elif symbol == "p":
                    self.object_locations["parcel"] = ("FLOOR", (x, y))
        self.hand = self.spec.initial_hand
        if self.hand is not None:
            self.object_locations[self.hand] = ("HAND", None)
        self.socket_occupants = {socket.socket_id: None for socket in self.spec.sockets}
        self.remaining_steps = self.spec.baseline * 6
        self.step_budget = self.remaining_steps
        self.undo_history: list[tuple[object, ...]] = []
        self.animation: list[dict[str, object]] = []
        self.pending_completion = False
        self.won = False
        self._render()

    def _snapshot(self) -> tuple[object, ...]:
        return (self.courier, self.hand, dict(self.object_locations), dict(self.socket_occupants), self.won)

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        courier, hand, objects, occupants, won = snapshot
        self.courier = courier  # type: ignore[assignment]
        self.hand = hand  # type: ignore[assignment]
        self.object_locations = dict(objects)  # type: ignore[arg-type]
        self.socket_occupants = dict(occupants)  # type: ignore[arg-type]
        self.won = bool(won)

    def _socket_at(self, cell: tuple[int, int]) -> SocketSpec | None:
        return next((socket for socket in self.spec.sockets if socket.cell == cell), None)

    def _gate_between(self, a: tuple[int, int], b: tuple[int, int]) -> GateSpec | None:
        edge = frozenset((a, b))
        return next((gate for gate in self.spec.gates if frozenset((gate.a, gate.b)) == edge), None)

    def _gate_open(self, gate: GateSpec) -> bool:
        return gate.socket_id is not None and self.socket_occupants[gate.socket_id] is not None

    def _loose_at(self, cell: tuple[int, int]) -> str | None:
        for object_id, location in self.object_locations.items():
            if location == ("FLOOR", cell):
                return object_id
        return None

    def _spend_step(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return False
        self.animation.clear()
        self._render({"failure": True})
        self.lose()
        self.complete_action()
        return True

    def step(self) -> None:
        if self.animation:
            frame = self.animation.pop(0)
            self._render(frame)
            if not self.animation:
                if self.pending_completion:
                    self.pending_completion = False
                    self.next_level()
                    self.complete_action()
                else:
                    self.complete_action()
            return

        action_id = self.action.id.value
        if action_id == GameAction.RESET.value:
            self._render()
            self.complete_action()
            return
        if action_id == UNDO:
            if self._spend_step():
                return
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._render()
            self.complete_action()
            return
        if action_id == CLICK:
            self._render()
            self.complete_action()
            return
        if action_id not in {UP, DOWN, LEFT, RIGHT, SPACE}:
            self.complete_action()
            return
        if self._spend_step():
            return

        if action_id in MOVE:
            changed, frames = self._move(action_id)
        else:
            changed, frames = self._interact()
        if changed:
            self.undo_history.append(changed)
        if self._check_success():
            self.won = True
            frames = [{"complete_phase": 1}, {"complete_phase": 2}]
            self.pending_completion = True
        if frames:
            self.animation = frames
            frame = self.animation.pop(0)
            self._render(frame)
            if not self.animation:
                if self.pending_completion:
                    self.pending_completion = False
                    self.next_level()
                    self.complete_action()
                else:
                    self.complete_action()
        else:
            self._render()
            self.complete_action()

    def _move(self, action_id: int) -> tuple[tuple[object, ...] | None, list[dict[str, object]]]:
        dx, dy = MOVE[action_id]
        source = self.courier
        destination = (source[0] + dx, source[1] + dy)
        if destination not in self.floor:
            return None, []
        gate = self._gate_between(source, destination)
        if gate is not None and not self._gate_open(gate) and self.hand != gate.kind:
            return None, [{"flash_gate": gate.gate_id, "gate_shift": -1}, {"flash_gate": gate.gate_id, "gate_shift": 1}]
        snapshot = self._snapshot()
        self.courier = destination
        frames: list[dict[str, object]] = []
        if gate is not None and not self._gate_open(gate):
            frames = [
                {"courier_override": source, "forced_open_gate": gate.gate_id},
                {"courier_edge": (source, destination), "forced_open_gate": gate.gate_id},
                {},
            ]
        elif destination == self.exit_cell and self.hand != "parcel":
            frames = [{"exit_pulse": True}, {}]
        return snapshot, frames

    def _interact(self) -> tuple[tuple[object, ...] | None, list[dict[str, object]]]:
        socket = self._socket_at(self.courier)
        if socket is not None:
            occupant = self.socket_occupants[socket.socket_id]
            if self.hand is None and occupant is not None:
                snapshot = self._snapshot()
                self.hand = occupant
                self.socket_occupants[socket.socket_id] = None
                self.object_locations[occupant] = ("HAND", None)
                return snapshot, [
                    {"socket_motion": socket.socket_id, "socket_direction": "retrieve"},
                    {"bright_cable": socket.socket_id},
                    {},
                ]
            if self.hand == socket.kind and occupant is None:
                snapshot = self._snapshot()
                object_id = self.hand
                self.hand = None
                self.socket_occupants[socket.socket_id] = object_id
                self.object_locations[object_id] = ("SOCKET", socket.socket_id)
                return snapshot, [
                    {"socket_motion": socket.socket_id, "socket_direction": "insert"},
                    {"bright_cable": socket.socket_id},
                    {},
                ]
            if self.hand is not None and occupant is None:
                return None, [{"flash_socket": socket.socket_id, "flash_hand": True}, {}]
            if self.hand is None and occupant is None:
                return None, [{"flash_socket": socket.socket_id}, {}]
            return None, [{"held_bob": True}, {}]

        loose = self._loose_at(self.courier)
        if self.hand is None and loose is not None:
            snapshot = self._snapshot()
            self.hand = loose
            self.object_locations[loose] = ("HAND", None)
            return snapshot, []
        if self.hand is not None and loose is None:
            if self.courier == self.exit_cell:
                return None, [{"held_bob": True}, {}]
            snapshot = self._snapshot()
            object_id = self.hand
            self.hand = None
            self.object_locations[object_id] = ("FLOOR", self.courier)
            return snapshot, []
        if self.hand is not None and loose is not None:
            return None, [{"flash_hand": True, "flash_object": loose}, {}]
        return None, []

    def _check_success(self) -> bool:
        return self.courier == self.exit_cell and self.hand == "parcel"

    def _draw_mask(
        self, frame: np.ndarray, x0: int, y0: int, kind: str, *, outline: bool = False, flash: bool = False
    ) -> None:
        dark, light = _mask_colors(kind)
        mask = MASKS[kind]
        occupied = {(x, y) for y, row in enumerate(mask) for x, char in enumerate(row) if char != "."}
        for y, row in enumerate(mask):
            for x, char in enumerate(row):
                if char == ".":
                    continue
                if outline and all((x + dx, y + dy) in occupied for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))):
                    continue
                frame[y0 + y, x0 + x] = INVALID if flash else (light if char.islower() else dark)

    def _draw_cable(self, frame: np.ndarray, socket: SocketSpec, gate: GateSpec, bright: bool) -> None:
        sx, sy = _cell_xy(socket.cell)
        start = (sx + 2, sy + 2)
        ax, ay = _cell_xy(gate.a)
        bx, by = _cell_xy(gate.b)
        end = ((ax + bx + 4) // 2, (ay + by + 4) // 2)
        color = WHITE if bright else (A_LIGHT if socket.kind == "A" else B_LIGHT)
        x1, x2 = sorted((start[0], end[0]))
        frame[start[1], x1 : x2 + 1] = color
        y1, y2 = sorted((start[1], end[1]))
        frame[y1 : y2 + 1, end[0]] = color

    def _draw_gate(
        self, frame: np.ndarray, gate: GateSpec, *, open_override: bool = False, flash: bool = False, shift: int = 0
    ) -> None:
        ax, ay = _cell_xy(gate.a)
        bx, by = _cell_xy(gate.b)
        is_open = self._gate_open(gate) or open_override
        color = INVALID if flash else (A_DARK if gate.kind == "A" else B_DARK)
        light = INVALID if flash else (A_LIGHT if gate.kind == "A" else B_LIGHT)
        if ay == by:
            x = max(ax, bx) + shift
            frame[ay, x - 1 : x + 1] = NEUTRAL
            frame[ay + 4, x - 1 : x + 1] = NEUTRAL
            if not is_open:
                frame[ay + 1 : ay + 4, x - 1 : x + 1] = color
                frame[ay + 2, x] = light
        else:
            y = max(ay, by) + shift
            frame[y - 1 : y + 1, ax] = NEUTRAL
            frame[y - 1 : y + 1, ax + 4] = NEUTRAL
            if not is_open:
                frame[y - 1 : y + 1, ax + 1 : ax + 4] = color
                frame[y, ax + 2] = light

    def _render(self, effect: dict[str, object] | None = None) -> None:
        effect = effect or {}
        frame = np.full((VIEW_SIZE, VIEW_SIZE), BG, dtype=np.int8)
        frame[ORIGIN_Y : ORIGIN_Y + 45, ORIGIN_X : ORIGIN_X + 55] = WALL
        for cell in self.floor:
            x, y = _cell_xy(cell)
            frame[y : y + CELL, x : x + CELL] = FLOOR

        for socket in self.spec.sockets:
            gate = next(gate for gate in self.spec.gates if gate.gate_id == socket.gate_id)
            self._draw_cable(frame, socket, gate, effect.get("bright_cable") == socket.socket_id)

        ex, ey = _cell_xy(self.exit_cell)
        exit_color = SUCCESS if self.won or effect.get("complete_phase") else PARCEL_DARK
        frame[ey : ey + CELL, ex : ex + CELL] = exit_color
        frame[ey + 1 : ey + 4, ex + 1 : ex + 4] = FLOOR
        self._draw_mask(
            frame,
            ex,
            ey,
            "P",
            outline=not (self.won or bool(effect.get("complete_phase")) or bool(effect.get("exit_pulse"))),
        )

        for socket in self.spec.sockets:
            x, y = _cell_xy(socket.cell)
            flash = effect.get("flash_socket") == socket.socket_id
            frame[y : y + CELL, x : x + CELL] = INVALID if flash else NEUTRAL
            occupant = self.socket_occupants[socket.socket_id]
            moving_from_socket = effect.get("socket_motion") == socket.socket_id
            self._draw_mask(frame, x, y, socket.kind, outline=occupant is None or moving_from_socket, flash=flash)

        for object_id, location in self.object_locations.items():
            if location[0] != "FLOOR":
                continue
            x, y = _cell_xy(location[1])  # type: ignore[arg-type]
            self._draw_mask(frame, x, y, _object_kind(object_id), flash=effect.get("flash_object") == object_id)

        for gate in self.spec.gates:
            self._draw_gate(
                frame,
                gate,
                open_override=effect.get("forced_open_gate") == gate.gate_id,
                flash=effect.get("flash_gate") == gate.gate_id,
                shift=int(effect.get("gate_shift", 0)),
            )

        self._draw_hand(frame, effect)
        moving_socket_id = effect.get("socket_motion")
        if moving_socket_id is not None:
            moving_socket = next(socket for socket in self.spec.sockets if socket.socket_id == moving_socket_id)
            sx, sy = _cell_xy(moving_socket.cell)
            self._draw_mask(frame, (5 + sx) // 2, (5 + sy) // 2, moving_socket.kind)
        if effect.get("courier_edge"):
            source, destination = effect["courier_edge"]  # type: ignore[misc]
            sx, sy = _cell_xy(source)
            dx, dy = _cell_xy(destination)
            self._draw_courier(frame, (sx + dx) // 2, (sy + dy) // 2, effect)
        else:
            courier = effect.get("courier_override", self.courier)
            cx, cy = _cell_xy(courier)  # type: ignore[arg-type]
            self._draw_courier(frame, cx, cy, effect)

        bar_x, bar_y, bar_width = 15, 7, 44
        frame[bar_y : bar_y + 2, bar_x : bar_x + bar_width] = NEUTRAL
        fill = max(0, int(bar_width * self.remaining_steps / max(1, self.step_budget)))
        if fill:
            frame[bar_y : bar_y + 2, bar_x : bar_x + fill] = SUCCESS
        if effect.get("failure"):
            frame[bar_y : bar_y + 2, bar_x : bar_x + bar_width] = INVALID
        self.canvas.pixels = frame

    def _draw_hand(self, frame: np.ndarray, effect: dict[str, object]) -> None:
        frame[3:12, 3:12] = NEUTRAL
        frame[4:11, 4:11] = WALL
        if self.hand is None or effect.get("socket_motion") is not None:
            frame[5:10, 6] = WHITE
            frame[5, 6:9] = WHITE
            frame[9, 6:9] = WHITE
            return
        flash = bool(effect.get("flash_hand"))
        y = 4 if effect.get("held_bob") else 5
        self._draw_mask(frame, 5, y, _object_kind(self.hand), flash=flash)

    def _draw_courier(self, frame: np.ndarray, x: int, y: int, effect: dict[str, object]) -> None:
        if self.won or effect.get("complete_phase"):
            frame[y + 1 : y + 4, x + 1 : x + 4] = SUCCESS
            frame[y + 2, x + 2] = PARCEL_LIGHT
            return
        frame[y + 1 : y + 5, x + 1 : x + 4] = WHITE
        frame[y + 2, x + 2] = NEUTRAL
        if self.hand is not None:
            dark, light = _mask_colors(_object_kind(self.hand))
            frame[y, x + 1 : x + 4] = dark
            frame[y + 1, x + 1] = dark
            frame[y + 1, x + 2] = light
            frame[y + 1, x + 3] = dark
