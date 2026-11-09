from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
MOVE = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

VIEW = 64
CELL = 4
ORIGIN = 2

OUTSIDE = 5
VOID = 4
FLOOR = 1
WALL = 3
BLACK = 0
AGENT = 9
ARTIFACT = 11
PEDESTAL = 2
EXIT = 14
EMPTY_GATE = 10
CARRIER_GATE = 15
ORANGE = 12
RED = 8
PULSE = 7


@dataclass(frozen=True)
class GateSpec:
    id: str
    gate_class: str
    initial_open: bool
    door: tuple[int, int]
    endpoint_a: tuple[int, int]
    endpoint_b: tuple[int, int]
    handle: tuple[int, int]
    swing: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    player: tuple[int, int]
    pedestal: tuple[int, int]
    exit: tuple[int, int]
    gates: tuple[GateSpec, ...]
    blocked_edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    baseline: int = 1


def gate(
    gate_id: str,
    gate_class: str,
    initial_open: bool,
    door: tuple[int, int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    handle: tuple[int, int],
    swing: tuple[int, int],
) -> GateSpec:
    return GateSpec(gate_id, gate_class, initial_open, door, endpoints[0], endpoints[1], handle, swing)


LEVELS = (
    LevelSpec(
        "Take It",
        (
            "###############",
            "###############",
            "###.###########",
            "###.###########",
            "###..##########",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
        ),
        (3, 4),
        (4, 4),
        (3, 2),
        (),
        baseline=3,
    ),
    LevelSpec(
        "The Handle",
        (
            "###############",
            "###############",
            "###..##########",
            "##....#########",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
        ),
        (2, 3),
        (5, 3),
        (4, 2),
        (gate("G", "CARRIER_ONLY", False, (3, 3), ((2, 3), (4, 3)), (2, 3), (3, 2)),),
        baseline=5,
    ),
    LevelSpec(
        "Different Way Back",
        (
            "###############",
            "####.##########",
            "####.##########",
            "####.##########",
            "##....#########",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
        ),
        (2, 4),
        (5, 4),
        (4, 1),
        (
            gate("E", "EMPTY_ONLY", False, (3, 4), ((2, 4), (4, 4)), (3, 5), (3, 3)),
            gate("C", "CARRIER_ONLY", False, (4, 3), ((4, 4), (4, 2)), (5, 3), (3, 3)),
        ),
        baseline=6,
    ),
    LevelSpec(
        "Doorway or Floor",
        (
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "##########.####",
            "##########.####",
            "###.........###",
            "###......######",
            "######...######",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
        ),
        (3, 8),
        (11, 7),
        (10, 5),
        (
            gate("G", "CARRIER_ONLY", False, (4, 8), ((3, 8), (5, 8)), (3, 8), (4, 7)),
            gate("H", "CARRIER_ONLY", False, (8, 7), ((7, 7), (9, 7)), (8, 8), (7, 8)),
        ),
        (((5, 8), (5, 7)), ((6, 8), (6, 7)), ((7, 8), (7, 7))),
        26,
    ),
    LevelSpec(
        "Before Your Hands Are Full",
        (
            "###############",
            "###############",
            "#########.#####",
            "########....###",
            "###......######",
            "###.####.######",
            "##.........####",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
        ),
        (2, 6),
        (10, 6),
        (11, 3),
        (
            gate("A", "EMPTY_ONLY", False, (4, 4), ((3, 4), (5, 4)), (4, 5), (4, 3)),
            gate("D", "EMPTY_ONLY", False, (5, 6), ((4, 6), (6, 6)), (5, 7), (5, 5)),
            gate("C", "CARRIER_ONLY", False, (8, 5), ((8, 6), (8, 4)), (9, 5), (7, 5)),
            gate("B", "EMPTY_ONLY", False, (9, 3), ((8, 3), (10, 3)), (8, 3), (9, 2)),
        ),
        baseline=32,
    ),
    LevelSpec(
        "Close What You Need",
        (
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "#########.#####",
            "####.#.....####",
            "##.....###...##",
            "#####...####.##",
            "#######.###...#",
            "#######.....###",
            "#######..######",
            "###############",
            "###############",
            "###############",
        ),
        (7, 11),
        (13, 9),
        (2, 7),
        (
            gate("T", "CARRIER_ONLY", False, (7, 10), ((7, 11), (7, 9)), (7, 11), (8, 10)),
            gate("F", "CARRIER_ONLY", True, (6, 7), ((6, 8), (6, 6)), (6, 8), (5, 7)),
            gate("R", "EMPTY_ONLY", False, (9, 6), ((8, 6), (10, 6)), (8, 6), (9, 5)),
            gate("B", "EMPTY_ONLY", False, (4, 7), ((5, 7), (3, 7)), (5, 7), (4, 6)),
            gate("D", "EMPTY_ONLY", False, (10, 10), ((9, 10), (11, 10)), (10, 11), (10, 9)),
            gate("K", "CARRIER_ONLY", False, (11, 7), ((10, 7), (12, 7)), (11, 6), (11, 8)),
        ),
        baseline=49,
    ),
    LevelSpec(
        "The Prepared House",
        (
            "###############",
            "###############",
            "###############",
            "###############",
            "####.##########",
            "####.#####.####",
            "###.........###",
            "####....###...#",
            "######..####..#",
            "#######..###..#",
            "########.###.##",
            "########.....##",
            "########..#####",
            "###############",
            "###############",
        ),
        (8, 12),
        (13, 9),
        (4, 4),
        (
            gate("T", "CARRIER_ONLY", False, (8, 11), ((8, 12), (8, 10)), (8, 12), (9, 11)),
            gate("F", "CARRIER_ONLY", True, (7, 7), ((7, 8), (7, 6)), (7, 8), (6, 7)),
            gate("Q", "EMPTY_ONLY", False, (4, 6), ((4, 7), (4, 5)), (5, 6), (3, 6)),
            gate("B", "EMPTY_ONLY", False, (5, 7), ((6, 7), (4, 7)), (6, 7), (5, 6)),
            gate("R", "EMPTY_ONLY", False, (10, 6), ((9, 6), (11, 6)), (9, 6), (10, 5)),
            gate("K", "CARRIER_ONLY", False, (12, 7), ((13, 7), (11, 7)), (12, 6), (12, 8)),
            gate("D", "EMPTY_ONLY", False, (11, 11), ((10, 11), (12, 11)), (11, 12), (11, 10)),
        ),
        (((6, 7), (6, 6)),),
        59,
    ),
)


def _origin(cell: tuple[int, int]) -> tuple[int, int]:
    return ORIGIN + CELL * cell[0], ORIGIN + CELL * cell[1]


def _diamond(frame: np.ndarray, cell: tuple[int, int], color: int = ARTIFACT) -> None:
    x, y = _origin(cell)
    frame[y + 1, x + 1 : x + 3] = color
    frame[y + 2, x : x + 4] = color


class TakeItBackDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: TakeItBack | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = OUTSIDE
        frame[1, :] = BLACK
        frame[2:62, 2:62] = VOID
        spec = game.spec
        for y, row in enumerate(spec.rows):
            for x, symbol in enumerate(row):
                if symbol == ".":
                    px, py = _origin((x, y))
                    frame[py : py + CELL, px : px + CELL] = FLOOR

        self._draw_recesses(frame, game)
        self._draw_blocked_edges(frame, spec)
        self._draw_exit(frame, game)
        self._draw_pedestal(frame, game)
        self._draw_gates(frame, game)
        self._draw_handles(frame, game)
        self._draw_player(frame, game)
        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_budget(self, frame: np.ndarray, game: TakeItBack) -> None:
        frame[0, :] = BLACK
        width = int(64 * max(0, game.steps_remaining) / game.step_budget)
        if width:
            frame[0, :width] = ORANGE

    def _draw_exit(self, frame: np.ndarray, game: TakeItBack) -> None:
        x, y = _origin(game.spec.exit)
        frame[y, x : x + 4] = EXIT
        frame[y + 1 : y + 4, x] = EXIT
        frame[y + 1 : y + 4, x + 3] = EXIT
        _diamond(frame, game.spec.exit)
        if game.transient and game.transient[0] in {"receiver", "success"}:
            color = ARTIFACT if game.transient[1] % 2 else EXIT
            frame[y : y + 4, x : x + 4] = color
            _diamond(frame, game.spec.exit, EXIT if color == ARTIFACT else ARTIFACT)

    def _draw_pedestal(self, frame: np.ndarray, game: TakeItBack) -> None:
        x, y = _origin(game.spec.pedestal)
        frame[y : y + 4, x : x + 4] = PEDESTAL
        frame[y + 3, x : x + 4] = WALL
        if game.pedestal_loaded:
            _diamond(frame, game.spec.pedestal)

    def _draw_recesses(self, frame: np.ndarray, game: TakeItBack) -> None:
        for gate_spec in game.spec.gates:
            if gate_spec.swing not in game.floor:
                continue
            x, y = _origin(gate_spec.swing)
            frame[y, x : x + 2] = WALL
            frame[y : y + 2, x] = WALL

    def _draw_blocked_edges(self, frame: np.ndarray, spec: LevelSpec) -> None:
        for a, b in spec.blocked_edges:
            ax, ay = _origin(a)
            if a[0] == b[0]:
                py = ay + CELL - 1 if b[1] > a[1] else ay
                frame[py, ax : ax + CELL] = WALL
            else:
                px = ax + CELL - 1 if b[0] > a[0] else ax
                frame[ay : ay + CELL, px] = WALL

    def _draw_gates(self, frame: np.ndarray, game: TakeItBack) -> None:
        animation_gate = None
        phase = None
        if game.transient and game.transient[0] == "gate":
            animation_gate, phase = game.transient[1], game.transient[2]
        for gate_spec in game.spec.gates:
            is_open = game.gate_open[gate_spec.id]
            if gate_spec.id == animation_gate and phase == 0:
                is_open = not is_open
            color = EMPTY_GATE if gate_spec.gate_class == "EMPTY_ONLY" else CARRIER_GATE
            if gate_spec.id == animation_gate and phase in {1, 2}:
                dx, dy = _origin(gate_spec.door)
                frame[dy, dx] = color
                frame[dy, dx + 3] = color
                frame[dy + 3, dx] = color
                frame[dy + 3, dx + 3] = color
                self._draw_swing_phase(frame, gate_spec, color, phase)
                continue
            if is_open:
                dx, dy = _origin(gate_spec.door)
                frame[dy, dx] = color
                frame[dy, dx + 3] = color
                frame[dy + 3, dx] = color
                frame[dy + 3, dx + 3] = color
                self._draw_leaf(frame, gate_spec.swing, gate_spec, color)
            else:
                self._draw_leaf(frame, gate_spec.door, gate_spec, color)

    def _draw_leaf(self, frame: np.ndarray, cell: tuple[int, int], gate_spec: GateSpec, color: int) -> None:
        x, y = _origin(cell)
        horizontal = gate_spec.endpoint_a[1] == gate_spec.endpoint_b[1]
        if cell == gate_spec.door:
            frame[y : y + 4, x : x + 4] = color
            if horizontal:
                frame[y + 1 : y + 3, x : x + 4] = BLACK
            else:
                frame[y : y + 4, x + 1 : x + 3] = BLACK
        else:
            frame[y : y + 4, x : x + 4] = color
        if gate_spec.gate_class == "CARRIER_ONLY":
            _diamond(frame, cell)
        hx = x if gate_spec.handle[0] <= gate_spec.door[0] else x + 3
        hy = y if gate_spec.handle[1] <= gate_spec.door[1] else y + 3
        frame[hy : hy + 2 if hy < y + 3 else hy + 1, hx : hx + 2 if hx < x + 3 else hx + 1] = ORANGE

    def _draw_swing_phase(self, frame: np.ndarray, gate_spec: GateSpec, color: int, phase: int) -> None:
        dx, dy = _origin(gate_spec.door)
        sx, sy = _origin(gate_spec.swing)
        x = (dx * (3 - phase) + sx * phase) // 3
        y = (dy * (3 - phase) + sy * phase) // 3
        frame[y : y + 3, x : x + 3] = color
        frame[y + 1 : y + 3, x + 1 : x + 3] = ORANGE

    def _draw_handles(self, frame: np.ndarray, game: TakeItBack) -> None:
        occupied = {g.swing for g in game.spec.gates if game.gate_open[g.id]}
        for gate_spec in game.spec.gates:
            if gate_spec.handle in occupied:
                continue
            x, y = _origin(gate_spec.handle)
            if gate_spec.handle in game.floor:
                frame[y + 1 : y + 3, x + 1 : x + 3] = ORANGE
            else:
                frame[y + 1 : y + 3, x : x + 2] = ORANGE

    def _draw_player(self, frame: np.ndarray, game: TakeItBack) -> None:
        cell = game.agent_position
        ox = oy = 0
        if game.transient and game.transient[0] == "bump":
            dx, dy = game.transient[1]
            ox, oy = dx, dy
        x, y = _origin(cell)
        x += ox
        y += oy
        frame[y : y + 3, x : x + 3] = AGENT
        frame[y + 1 : y + 4, x + 1 : x + 4] = AGENT
        pickup_in_flight = game.transient and game.transient[0] == "pickup" and game.transient[1] < 2
        if game.carrying and not pickup_in_flight:
            frame[y, x + 1 : x + 3] = AGENT
            frame[y + 1, x : x + 4] = ARTIFACT
            frame[y + 2, x + 1 : x + 3] = ARTIFACT

    def _draw_feedback(self, frame: np.ndarray, game: TakeItBack) -> None:
        if not game.transient:
            return
        kind = game.transient[0]
        if kind == "neutral":
            x, y = _origin(game.agent_position)
            frame[y, x : x + 4] = PULSE
            frame[y + 3, x : x + 4] = PULSE
            frame[y : y + 4, x] = PULSE
            frame[y : y + 4, x + 3] = PULSE
        elif kind == "hands":
            x, y = _origin(game.agent_position)
            rattle = 2 if game.transient[1] == 1 else 1
            frame[y + 1 : y + 3, x + rattle] = ORANGE
            for index in range(4):
                frame[y + index, x + index] = RED
                frame[y + index, x + 3 - index] = RED
        elif kind == "blocked":
            cell = game.transient[1]
            x, y = _origin(cell)
            frame[y : y + 4, x : x + 4] = RED
        elif kind == "reject_carrier":
            x, y = _origin(game.agent_position)
            for i in range(4):
                frame[y + i, x + 3 - i] = RED
        elif kind == "reject_empty":
            _diamond(frame, game.transient[1], PULSE)
        elif kind == "pickup":
            phase = game.transient[1]
            px, py = _origin(game.spec.pedestal)
            ax, ay = _origin(game.agent_position)
            x = (px * (2 - phase) + ax * phase) // 2
            y = (py * (2 - phase) + ay * phase) // 2
            frame[y + 1 : y + 3, x + 1 : x + 3] = ARTIFACT


def _build_level(spec: LevelSpec) -> Level:
    background = Sprite(np.full((64, 64), OUTSIDE, dtype=np.int8), name="background", layer=-20)
    return Level([background], (64, 64), {"spec": spec}, spec.name)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class TakeItBack(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = TakeItBackDisplay()
        levels = [_build_level(spec) for spec in LEVELS]
        super().__init__(
            "take_it_back-0001",
            levels,
            Camera(0, 0, VIEW, VIEW, OUTSIDE, OUTSIDE, [self._display]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.floor = {(x, y) for y, row in enumerate(self.spec.rows) for x, symbol in enumerate(row) if symbol == "."}
        self.agent_position = self.spec.player
        self.carrying = False
        self.pedestal_loaded = True
        self.gate_open = {gate_spec.id: gate_spec.initial_open for gate_spec in self.spec.gates}
        self.step_budget = self.spec.baseline * 6
        self.steps_remaining = self.step_budget
        self.won = False
        self.failed = False
        self.transient: tuple | None = None
        self.animation: list[tuple] = []
        self.finish_after_animation: str | None = None
        self.undo_history: list[tuple] = []

    def _snapshot(self) -> tuple:
        return (self.agent_position, self.carrying, self.pedestal_loaded, tuple(sorted(self.gate_open.items())))

    def _restore(self, snapshot: tuple) -> None:
        self.agent_position, self.carrying, self.pedestal_loaded, gates = snapshot
        self.gate_open = dict(gates)

    def _occupied_leaf(self, cell: tuple[int, int], except_gate: str | None = None) -> bool:
        return any(
            gate_spec.id != except_gate and self.gate_open[gate_spec.id] and gate_spec.swing == cell
            for gate_spec in self.spec.gates
        )

    def _gate_at_door(self, cell: tuple[int, int]) -> GateSpec | None:
        return next((gate_spec for gate_spec in self.spec.gates if gate_spec.door == cell), None)

    def _blocked_edge(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        edge = frozenset((a, b))
        return any(frozenset(pair) == edge for pair in self.spec.blocked_edges)

    def _can_move(self, target: tuple[int, int]) -> tuple[bool, str | None, tuple[int, int] | None]:
        source = self.agent_position
        if self._blocked_edge(source, target) or target not in self.floor or target == self.spec.pedestal:
            return False, "bump", None
        if self._occupied_leaf(target):
            return False, "bump", None

        source_gate = self._gate_at_door(source)
        if source_gate and target not in {source_gate.endpoint_a, source_gate.endpoint_b}:
            return False, "bump", None
        target_gate = self._gate_at_door(target)
        if target_gate:
            if source not in {target_gate.endpoint_a, target_gate.endpoint_b}:
                return False, "bump", None
            if not self.gate_open[target_gate.id]:
                authorized = (target_gate.gate_class == "EMPTY_ONLY") != self.carrying
                if not authorized:
                    kind = "reject_carrier" if self.carrying else "reject_empty"
                    return False, kind, target_gate.door
        return True, None, None

    def _toggle(self, gate_spec: GateSpec) -> bool:
        if self.carrying:
            self.animation = [("hands", 0), ("hands", 1), ("hands", 2)]
            return False
        opening = not self.gate_open[gate_spec.id]
        if opening:
            blocked = (
                gate_spec.swing not in self.floor
                or gate_spec.swing == self.agent_position
                or gate_spec.swing in {self.spec.pedestal, self.spec.exit}
                or self._gate_at_door(gate_spec.swing) is not None
                or self._occupied_leaf(gate_spec.swing, gate_spec.id)
            )
            if blocked:
                self.animation = [("blocked", gate_spec.swing), ("blocked", gate_spec.swing)]
                return False
        elif self.agent_position == gate_spec.door:
            self.animation = [("blocked", gate_spec.door), ("blocked", gate_spec.door)]
            return False

        self.undo_history.append(self._snapshot())
        self.gate_open[gate_spec.id] = opening
        self.animation = [
            ("gate", gate_spec.id, 0),
            ("gate", gate_spec.id, 1),
            ("gate", gate_spec.id, 2),
            ("gate", gate_spec.id, 3),
        ]
        return True

    def _resolve_action(self, action: int) -> None:
        self.transient = None
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            return
        if action == CLICK:
            return
        if action in MOVE:
            dx, dy = MOVE[action]
            target = (self.agent_position[0] + dx, self.agent_position[1] + dy)
            allowed, feedback, feedback_cell = self._can_move(target)
            if allowed:
                self.undo_history.append(self._snapshot())
                self.agent_position = target
            else:
                self.animation = [(feedback, (dx, dy)) if feedback == "bump" else (feedback, feedback_cell), None]
            return
        if action != SPACE:
            return
        if (
            self.pedestal_loaded
            and not self.carrying
            and sum(abs(a - b) for a, b in zip(self.agent_position, self.spec.pedestal, strict=True)) == 1
        ):
            self.undo_history.append(self._snapshot())
            self.pedestal_loaded = False
            self.carrying = True
            self.animation = [("pickup", 0), ("pickup", 1), ("pickup", 2)]
            if self.level_index == 0:
                self.animation.append(("receiver", 1))
            return
        gate_spec = next((g for g in self.spec.gates if g.handle == self.agent_position), None)
        if gate_spec:
            self._toggle(gate_spec)
        else:
            self.animation = [("neutral", 0), None]

    def _check_terminal(self) -> None:
        if self.carrying and self.agent_position == self.spec.exit:
            self.won = True
            self.finish_after_animation = "win"
            self.animation = [("success", index) for index in range(4)]
        elif self.steps_remaining <= 0:
            self.failed = True
            self.finish_after_animation = "lose"
            self.animation = [("blocked", self.agent_position), None]

    def _play_animation(self) -> None:
        self.transient = self.animation.pop(0)
        if self.animation:
            return
        outcome = self.finish_after_animation
        self.finish_after_animation = None
        if outcome == "win":
            self.next_level()
            self.complete_action()
        elif outcome == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def step(self) -> None:
        if self.animation:
            self._play_animation()
            return
        if self.action.id == GameAction.RESET:
            self.transient = None
            self.complete_action()
            return
        action = _action_id(self.action.id)
        self.steps_remaining = max(0, self.steps_remaining - 1)
        self._resolve_action(action)
        self._check_terminal()
        if self.animation:
            self._play_animation()
        else:
            self.complete_action()
