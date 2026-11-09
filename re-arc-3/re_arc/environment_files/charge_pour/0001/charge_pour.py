from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
ACTION_CLICK = 6
ACTION_UNDO = 7

BACKGROUND = 5
BODY = 4
PIPE = 3
PLAIN_RIM = 2
PIP = 10
GOAL_RIM = 11
ARROW = 12
REJECT = 8
COMPLETE = 14


@dataclass(frozen=True)
class ReservoirSpec:
    key: str
    symbol: str
    x: int
    y: int
    columns: int
    rows: int
    capacity: int
    initial: int
    goal: bool = False

    @property
    def width(self) -> int:
        return 5 * self.columns + 2

    @property
    def height(self) -> int:
        return 5 * self.rows + 2


@dataclass(frozen=True)
class EdgeSpec:
    key: str
    source: str
    destination: str
    center: tuple[int, int]
    direction: str
    pipe: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    reservoirs: tuple[ReservoirSpec, ...]
    edges: tuple[EdgeSpec, ...]
    solution: tuple[str, ...]

    @property
    def budget(self) -> int:
        return len(self.solution) * 6


@dataclass(frozen=True)
class Presentation:
    charges: dict[str, int]
    arrow_color: tuple[str, int] | None = None
    rim_color: tuple[tuple[str, ...], int] | None = None
    moving_pip: tuple[int, int] | None = None
    goal_pulse: int | None = None


def reservoir(
    key: str,
    symbol: str,
    position: tuple[int, int],
    layout: tuple[int, int],
    capacity: int,
    initial: int,
    *,
    goal: bool = False,
) -> ReservoirSpec:
    return ReservoirSpec(key, symbol, *position, *layout, capacity, initial, goal)


def edge(
    key: str, source: str, destination: str, center: tuple[int, int], direction: str, *pipe_points: tuple[int, int]
) -> EdgeSpec:
    return EdgeSpec(key, source, destination, center, direction, tuple(pipe_points))


def actions(program: str) -> tuple[str, ...]:
    return tuple(item for item in program.split() if item)


LEVEL_SPECS = (
    LevelSpec(
        "First Pour",
        (
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".S....B..G...",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("S", "S", (4, 27), (2, 1), 2, 2),
            reservoir("B", "B", (29, 27), (1, 1), 1, 0),
            reservoir("G", "G", (43, 27), (3, 1), 3, 1, goal=True),
        ),
        (
            edge("S_B", "S", "B", (22, 30), "right", (16, 30), (28, 30)),
            edge("B_G", "B", "G", (39, 30), "right", (36, 30), (42, 30)),
        ),
        actions("S_B B_G S_B B_G"),
    ),
    LevelSpec(
        "Full Cup",
        (
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".S....B..G...",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("S", "S", (4, 27), (2, 1), 2, 2),
            reservoir("B", "B", (29, 27), (1, 1), 1, 1),
            reservoir("G", "G", (43, 27), (3, 1), 3, 0, goal=True),
        ),
        (
            edge("S_B", "S", "B", (22, 30), "right", (16, 30), (28, 30)),
            edge("B_G", "B", "G", (39, 30), "right", (36, 30), (42, 30)),
        ),
        actions("B_G S_B B_G S_B B_G"),
    ),
    LevelSpec(
        "Backward Air",
        (
            ".....S.......",
            ".............",
            "......A......",
            ".............",
            "......B......",
            ".............",
            ".....G.......",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("S", "S", (24, 2), (3, 1), 3, 3),
            reservoir("A", "A", (29, 15), (1, 1), 1, 1),
            reservoir("B", "B", (29, 28), (1, 1), 1, 1),
            reservoir("G", "G", (24, 43), (3, 2), 5, 0, goal=True),
        ),
        (
            edge("S_A", "S", "A", (32, 11), "down", (32, 9), (32, 14)),
            edge("A_B", "A", "B", (32, 24), "down", (32, 22), (32, 27)),
            edge("B_G", "B", "G", (32, 39), "down", (32, 35), (32, 42)),
        ),
        actions("B_G A_B S_A B_G A_B S_A B_G A_B S_A B_G A_B B_G"),
    ),
    LevelSpec(
        "Split Headroom",
        (
            "C...R.H......",
            ".........N...",
            ".............",
            ".............",
            "E...J.P..Q...",
            ".........G...",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("C", "C", (1, 5), (3, 1), 3, 3),
            reservoir("R", "R", (23, 5), (1, 1), 1, 1),
            reservoir("H", "H", (35, 5), (1, 1), 1, 0),
            reservoir("GN", "N", (44, 19), (3, 2), 5, 0, goal=True),
            reservoir("E", "E", (1, 40), (2, 2), 4, 4),
            reservoir("J", "J", (19, 42), (1, 1), 1, 0),
            reservoir("P", "P", (32, 42), (1, 1), 1, 0),
            reservoir("Q", "Q", (45, 42), (1, 1), 1, 0),
            reservoir("GE", "G", (44, 55), (3, 1), 3, 0, goal=True),
        ),
        (
            edge("C_R", "C", "R", (20, 8), "right", (18, 8), (22, 8)),
            edge("R_H", "R", "H", (32, 8), "right", (30, 8), (34, 8)),
            edge("H_GN", "H", "GN", (46, 8), "right", (42, 8), (52, 8), (52, 18)),
            edge("E_J", "E", "J", (16, 45), "right", (13, 45), (18, 45)),
            edge("J_H", "J", "H", (38, 29), "up", (22, 41), (22, 33), (38, 33), (38, 12)),
            edge("J_P", "J", "P", (29, 45), "right", (26, 45), (31, 45)),
            edge("P_Q", "P", "Q", (42, 45), "right", (39, 45), (44, 45)),
            edge("Q_GE", "Q", "GE", (48, 52), "down", (48, 49), (48, 52), (52, 52), (52, 54)),
        ),
        actions(
            "R_H C_R H_GN R_H C_R H_GN R_H C_R H_GN R_H H_GN "
            "E_J J_H H_GN E_J J_P E_J P_Q J_P E_J Q_GE P_Q J_P Q_GE P_Q Q_GE"
        ),
    ),
    LevelSpec(
        "Turning Gap",
        (
            "....g........",
            ".............",
            "....A..B..p..",
            "....F..C.....",
            ".............",
            "P...E..D..G..",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("A", "A", (21, 20), (1, 1), 1, 1),
            reservoir("B", "B", (34, 20), (1, 1), 1, 1),
            reservoir("C", "C", (34, 33), (1, 1), 1, 1),
            reservoir("D", "D", (34, 46), (1, 1), 1, 0),
            reservoir("E", "E", (21, 46), (1, 1), 1, 1),
            reservoir("F", "F", (21, 33), (1, 1), 1, 1),
            reservoir("PE", "P", (2, 46), (2, 1), 2, 2),
            reservoir("PC", "p", (50, 20), (2, 2), 4, 4),
            reservoir("GD", "G", (47, 43), (3, 3), 9, 0, goal=True),
            reservoir("GA", "g", (18, 2), (2, 1), 2, 0, goal=True),
        ),
        (
            edge("A_B", "A", "B", (31, 23), "right", (28, 23), (33, 23)),
            edge("B_C", "B", "C", (37, 29), "down", (37, 27), (37, 32)),
            edge("C_D", "C", "D", (37, 42), "down", (37, 40), (37, 45)),
            edge("D_E", "D", "E", (31, 49), "left", (33, 49), (28, 49)),
            edge("E_F", "E", "F", (24, 42), "up", (24, 45), (24, 40)),
            edge("F_A", "F", "A", (24, 29), "up", (24, 32), (24, 27)),
            edge("PE_E", "PE", "E", (17, 49), "right", (14, 49), (20, 49)),
            edge("PC_C", "PC", "C", (46, 25), "left", (49, 25), (44, 25), (44, 36), (41, 36)),
            edge("D_GD", "D", "GD", (44, 49), "right", (41, 49), (46, 49)),
            edge("A_GA", "A", "GA", (24, 14), "up", (24, 19), (24, 9)),
        ),
        actions(
            "C_D B_C A_B F_A E_F PE_E D_GD C_D B_C A_B F_A E_F PE_E D_GD "
            "C_D B_C A_B F_A E_F D_GD C_D B_C D_GD C_D PC_C D_GD C_D PC_C D_GD "
            "C_D PC_C D_GD C_D PC_C D_GD C_D D_GD A_GA F_A A_GA"
        ),
    ),
    LevelSpec(
        "Golden Conduit",
        (
            "W....0.......",
            "E..J.........",
            "......A......",
            "...L..H..R.T.",
            ".............",
            "G........g...",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("W", "W", (2, 5), (2, 1), 2, 2),
            reservoir("E", "E", (2, 18), (2, 3), 5, 5),
            reservoir("J", "J", (19, 23), (1, 1), 1, 0),
            reservoir("G0", "0", (26, 4), (2, 2), 4, 4, goal=True),
            reservoir("A", "A", (29, 22), (1, 1), 1, 1),
            reservoir("H", "H", (29, 35), (1, 1), 1, 0),
            reservoir("L", "L", (16, 35), (1, 1), 1, 0),
            reservoir("R", "R", (42, 35), (1, 1), 1, 0),
            reservoir("T", "T", (55, 35), (1, 1), 1, 0),
            reservoir("G1", "G", (2, 50), (3, 1), 3, 0, goal=True),
            reservoir("G2", "g", (45, 49), (3, 2), 5, 0, goal=True),
        ),
        (
            edge("W_G0", "W", "G0", (20, 8), "right", (14, 8), (25, 8)),
            edge("E_J", "E", "J", (16, 26), "right", (14, 26), (18, 26)),
            edge("J_G0", "J", "G0", (22, 19), "up", (22, 22), (22, 18), (31, 18), (31, 16)),
            edge("G0_A", "G0", "A", (32, 19), "down", (32, 16), (32, 21)),
            edge("A_H", "A", "H", (32, 32), "down", (32, 29), (32, 34)),
            edge("H_L", "H", "L", (25, 38), "left", (28, 38), (23, 38)),
            edge("L_G1", "L", "G1", (19, 45), "down", (19, 42), (19, 46), (10, 46), (10, 49)),
            edge("H_R", "H", "R", (39, 38), "right", (36, 38), (41, 38)),
            edge("R_T", "R", "T", (52, 38), "right", (49, 38), (54, 38)),
            edge("T_G2", "T", "G2", (58, 45), "down", (58, 42), (58, 46), (53, 46), (53, 48)),
        ),
        actions(
            "E_J A_H G0_A W_G0 H_L A_H G0_A W_G0 L_G1 H_L A_H G0_A J_G0 E_J "
            "L_G1 H_L A_H G0_A J_G0 E_J L_G1 H_R A_H G0_A J_G0 E_J R_T H_R A_H "
            "G0_A J_G0 E_J T_G2 R_T H_R A_H G0_A J_G0 T_G2 R_T H_R A_H T_G2 R_T "
            "H_R T_G2 R_T T_G2"
        ),
    ),
    LevelSpec(
        "Whole Circuit",
        (
            "W...J..X.....",
            "0...A..B.....",
            "....F..C..G..",
            "g...E..D.....",
            "....Q........",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        (
            reservoir("A", "A", (23, 15), (1, 1), 1, 1),
            reservoir("B", "B", (36, 15), (1, 1), 1, 1),
            reservoir("C", "C", (36, 28), (1, 1), 1, 1),
            reservoir("D", "D", (36, 41), (1, 1), 1, 0),
            reservoir("E", "E", (23, 41), (1, 1), 1, 1),
            reservoir("F", "F", (23, 28), (1, 1), 1, 1),
            reservoir("G0", "0", (3, 15), (2, 1), 2, 2, goal=True),
            reservoir("W", "W", (3, 2), (2, 1), 2, 2),
            reservoir("X", "X", (35, 1), (2, 2), 4, 4),
            reservoir("J", "J", (22, 3), (1, 1), 1, 0),
            reservoir("Q", "Q", (23, 50), (1, 1), 1, 1),
            reservoir("G1", "G", (50, 23), (2, 3), 6, 0, goal=True),
            reservoir("G2", "g", (5, 39), (2, 3), 6, 0, goal=True),
        ),
        (
            edge("A_B", "A", "B", (33, 18), "right", (30, 18), (35, 18)),
            edge("B_C", "B", "C", (39, 24), "down", (39, 22), (39, 27)),
            edge("C_D", "C", "D", (39, 37), "down", (39, 35), (39, 40)),
            edge("D_E", "D", "E", (33, 44), "left", (35, 44), (30, 44)),
            edge("E_F", "E", "F", (26, 37), "up", (26, 40), (26, 35)),
            edge("F_A", "F", "A", (26, 24), "up", (26, 27), (26, 22)),
            edge("G0_A", "G0", "A", (18, 18), "right", (15, 18), (22, 18)),
            edge("W_G0", "W", "G0", (9, 11), "down", (9, 9), (9, 14)),
            edge("X_J", "X", "J", (31, 6), "left", (34, 6), (29, 6)),
            edge("J_G0", "J", "G0", (18, 12), "left", (25, 10), (25, 12), (12, 12), (12, 14)),
            edge("Q_G2", "Q", "G2", (20, 53), "left", (22, 53), (17, 53)),
            edge("C_G1", "C", "G1", (46, 31), "right", (43, 31), (49, 31)),
            edge("E_G2", "E", "G2", (20, 44), "left", (22, 44), (17, 44)),
        ),
        actions(
            "C_D B_C A_B F_A X_J Q_G2 C_G1 B_C A_B G0_A W_G0 C_G1 B_C A_B G0_A W_G0 "
            "C_G1 B_C A_B G0_A J_G0 X_J C_G1 B_C A_B G0_A J_G0 X_J C_G1 B_C A_B G0_A "
            "J_G0 X_J C_G1 B_C A_B G0_A J_G0 E_G2 D_E C_D B_C A_B E_G2 D_E C_D B_C "
            "E_G2 D_E C_D E_G2 D_E E_G2"
        ),
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _socket_center(spec: ReservoirSpec, index: int) -> tuple[float, float]:
    column = index % spec.columns
    row = index // spec.columns
    return spec.x + 2.5 + 5 * column, spec.y + 2.5 + 5 * row


def _orthogonalize(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = [points[0]]
    for point in points[1:]:
        previous = out[-1]
        if previous[0] != point[0] and previous[1] != point[1]:
            out.append((point[0], previous[1]))
        if out[-1] != point:
            out.append(point)
    return out


def _point_on_path(points: list[tuple[float, float]], fraction: float) -> tuple[int, int]:
    segments = list(pairwise(points))
    lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in segments]
    distance = sum(lengths) * fraction
    for (start, end), length in zip(segments, lengths, strict=True):
        if distance <= length:
            ratio = 0.0 if length == 0 else distance / length
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            return round(x - 0.5), round(y - 0.5)
        distance -= length
    end = points[-1]
    return round(end[0] - 0.5), round(end[1] - 0.5)


class ChargePourView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ChargePour | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        presentation = game.presentation
        charges = game.charge if presentation is None else presentation.charges
        self._draw_pipes(frame, game)
        self._draw_arrows(frame, game, presentation)
        self._draw_reservoirs(frame, game, charges, presentation)
        if presentation is not None and presentation.moving_pip is not None:
            x, y = presentation.moving_pip
            frame[max(0, y) : min(VIEW_SIZE, y + 2), max(0, x) : min(VIEW_SIZE, x + 2)] = PIP
        self._draw_budget(frame, game.remaining_steps, game.step_budget)
        return frame

    def _draw_pipes(self, frame: np.ndarray, game: ChargePour) -> None:
        for transfer in game.edges.values():
            for start, end in pairwise(transfer.pipe):
                x1, y1 = start
                x2, y2 = end
                if x1 == x2:
                    frame[min(y1, y2) : max(y1, y2) + 1, x1] = PIPE
                else:
                    frame[y1, min(x1, x2) : max(x1, x2) + 1] = PIPE

    def _draw_arrows(self, frame: np.ndarray, game: ChargePour, presentation: Presentation | None) -> None:
        for transfer in game.edges.values():
            color = ARROW
            if presentation is not None and presentation.arrow_color is not None:
                edge_key, override = presentation.arrow_color
                if edge_key == transfer.key:
                    color = override
            self._draw_arrow(frame, transfer.center, transfer.direction, color)

    def _draw_arrow(self, frame: np.ndarray, center: tuple[int, int], direction: str, color: int) -> None:
        pattern = np.array(
            [[0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0]], dtype=bool
        )
        turns = {"right": 0, "down": 1, "left": 2, "up": 3}[direction]
        pattern = np.rot90(pattern, -turns)
        x, y = center[0] - 2, center[1] - 2
        frame[y : y + 5, x : x + 5] = BODY
        plate = frame[y : y + 5, x : x + 5]
        plate[pattern] = color

    def _draw_reservoirs(
        self, frame: np.ndarray, game: ChargePour, charges: dict[str, int], presentation: Presentation | None
    ) -> None:
        flashing = set()
        flash_color = None
        if presentation is not None and presentation.rim_color is not None:
            keys, flash_color = presentation.rim_color
            flashing = set(keys)
        for spec in game.reservoirs.values():
            rim = GOAL_RIM if spec.goal else PLAIN_RIM
            if spec.key in flashing and flash_color is not None:
                rim = flash_color
            elif spec.goal and presentation is not None and presentation.goal_pulse is not None:
                rim = presentation.goal_pulse
            frame[spec.y : spec.y + spec.height, spec.x : spec.x + spec.width] = BODY
            frame[spec.y, spec.x : spec.x + spec.width] = rim
            frame[spec.y + spec.height - 1, spec.x : spec.x + spec.width] = rim
            frame[spec.y : spec.y + spec.height, spec.x] = rim
            frame[spec.y : spec.y + spec.height, spec.x + spec.width - 1] = rim
            for index in range(spec.capacity):
                column = index % spec.columns
                row = index // spec.columns
                x = spec.x + 1 + 5 * column
                y = spec.y + 1 + 5 * row
                socket_rim = GOAL_RIM if spec.goal else PIPE
                frame[y : y + 4, x : x + 4] = socket_rim
                frame[y + 1 : y + 3, x + 1 : x + 3] = PIP if index < charges[spec.key] else BACKGROUND

    def _draw_budget(self, frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[63, :] = BODY
        filled = round(64 * max(0, remaining) / max(1, budget))
        if filled:
            frame[63, :filled] = PIP


class ChargePour(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = ChargePourView()
        levels = [Level([], (64, 64), {"spec": spec}, spec.name) for spec in LEVEL_SPECS]
        super().__init__(
            "charge_pour-0001",
            levels,
            Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self.view]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self.view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        by_symbol = {item.symbol: item for item in self.spec.reservoirs}
        parsed: dict[str, ReservoirSpec] = {}
        for row in self.spec.rows:
            for symbol in row:
                if symbol in by_symbol:
                    item = by_symbol[symbol]
                    if item.key in parsed:
                        raise ValueError(f"Duplicate reservoir symbol {symbol!r} in {self.spec.name}")
                    parsed[item.key] = item
        if len(parsed) != len(self.spec.reservoirs):
            raise ValueError(f"ASCII layout for {self.spec.name} omits a reservoir")
        self.reservoirs = parsed
        self.edges = {item.key: item for item in self.spec.edges}
        self.charge = {key: item.initial for key, item in self.reservoirs.items()}
        self.initial_total_charge = sum(self.charge.values())
        goal_capacity = sum(item.capacity for item in self.reservoirs.values() if item.goal)
        if self.initial_total_charge != goal_capacity:
            raise ValueError(f"Charge conservation setup mismatch in {self.spec.name}")
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.completed = False
        self.feedback: tuple[str, bool, bool] | None = None
        self.presentation: Presentation | None = None
        self._animation: list[Presentation] = []
        self._finish_after_animation = False
        self._lose_after_animation = False
        self._undo_history: list[dict[str, int]] = []

    def step(self) -> None:
        if self._animation:
            self.presentation = self._animation.pop(0)
            if not self._animation:
                if self._finish_after_animation:
                    self.next_level()
                if self._lose_after_animation:
                    self.lose()
                self.complete_action()
            return

        action_id = _action_id(self.action.id)
        if action_id == GameAction.RESET.value:
            self.presentation = None
            self.complete_action()
            return

        self.feedback = None
        self.presentation = None
        self._spend_step()

        if action_id == ACTION_UNDO:
            if self._undo_history:
                self.charge = self._undo_history.pop()
                self._assert_invariants()
            self._finish_immediate_action()
            return

        if self.completed or action_id != ACTION_CLICK:
            self._finish_immediate_action()
            return

        transfer = self._clicked_edge()
        if transfer is None:
            self._finish_immediate_action()
            return

        source = self.reservoirs[transfer.source]
        destination = self.reservoirs[transfer.destination]
        source_empty = self.charge[source.key] == 0
        destination_full = self.charge[destination.key] == destination.capacity
        if source_empty or destination_full:
            self.feedback = (transfer.key, source_empty, destination_full)
            rims = tuple(
                key for key, blocked in ((source.key, source_empty), (destination.key, destination_full)) if blocked
            )
            settled = dict(self.charge)
            self._queue_animation(
                [
                    Presentation(settled, arrow_color=(transfer.key, REJECT)),
                    Presentation(settled, rim_color=(rims, REJECT)),
                    Presentation(settled),
                ]
            )
            return

        before = dict(self.charge)
        self._undo_history.append(before)
        removed_index = before[source.key] - 1
        filled_index = before[destination.key]
        self.charge[source.key] -= 1
        self.charge[destination.key] += 1
        self._assert_invariants()
        after = dict(self.charge)
        completed = self._goals_full()
        self.completed = completed

        travel_state = dict(before)
        travel_state[source.key] -= 1
        points = _orthogonalize(
            [_socket_center(source, removed_index), *transfer.pipe, _socket_center(destination, filled_index)]
        )
        frames = [Presentation(before, arrow_color=(transfer.key, PIP))]
        for fraction in (0.25, 0.5, 0.75):
            frames.append(Presentation(travel_state, moving_pip=_point_on_path(points, fraction)))
        frames.append(Presentation(after))
        if completed:
            frames.extend(Presentation(after, goal_pulse=color) for color in (GOAL_RIM, COMPLETE, GOAL_RIM, COMPLETE))
        self._queue_animation(frames, finish_level=completed)

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _finish_immediate_action(self) -> None:
        if self.remaining_steps <= 0 and not self.completed:
            self.lose()
        self.complete_action()

    def _queue_animation(self, frames: list[Presentation], *, finish_level: bool = False) -> None:
        self.presentation = frames[0]
        self._animation = frames[1:]
        self._finish_after_animation = finish_level
        self._lose_after_animation = self.remaining_steps <= 0 and not finish_level

    def _clicked_edge(self) -> EdgeSpec | None:
        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        matches = [
            item for item in self.edges.values() if abs(x - item.center[0]) <= 2 and abs(y - item.center[1]) <= 2
        ]
        if len(matches) > 1:
            raise AssertionError("Overlapping arrow hitboxes")
        return matches[0] if matches else None

    def _goals_full(self) -> bool:
        return all(self.charge[item.key] == item.capacity for item in self.reservoirs.values() if item.goal)

    def _assert_invariants(self) -> None:
        for key, amount in self.charge.items():
            if not 0 <= amount <= self.reservoirs[key].capacity:
                raise AssertionError(f"Charge out of bounds for {key}")
        if sum(self.charge.values()) != self.initial_total_charge:
            raise AssertionError("Charge was not conserved")
