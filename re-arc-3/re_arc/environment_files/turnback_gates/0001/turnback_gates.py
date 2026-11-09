from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

SIZE = 64
CELL = 5
ORIGIN = 4
BACKGROUND = 5
FLOOR = 1
EMPTY = 2
GATE = 3
WALL = 4
BLOCKED = 8
AGENT = 9
HIGHLIGHT = 10
ARROW = 11
RELIC = 12
CLOSED = 13
OPEN = 14
PEDESTAL = 15

Coord = tuple[int, int]


@dataclass(frozen=True)
class GateDef:
    id: str
    a: Coord
    b: Coord
    initial_entry_from: Coord


@dataclass(frozen=True)
class LevelDef:
    name: str
    rows: tuple[str, ...]
    gates: tuple[GateDef, ...]
    baseline: int


def _g(gate_id: str, a: Coord, b: Coord, entry: Coord) -> GateDef:
    return GateDef(gate_id, a, b, entry)


LEVEL_DEFS = (
    LevelDef(
        "Relic and Arch",
        (
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
            "####SRE####",
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        (),
        2,
    ),
    LevelDef(
        "The Arrow Turns",
        (
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
            "###ES.R####",
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        (_g("G1", (4, 5), (5, 5), (4, 5)),),
        5,
    ),
    LevelDef(
        "Leave by the Other Gate",
        (
            "###########",
            "###########",
            "###########",
            "###########",
            "#####R#####",
            "##S.....E##",
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        (_g("G1", (3, 5), (4, 5), (3, 5)), _g("G2", (6, 5), (7, 5), (6, 5))),
        8,
    ),
    LevelDef(
        "Rehearse from the Arch",
        (
            "###########",
            "#####..SE##",
            "#####.#####",
            "#####.#####",
            "#####.#####",
            "###...#####",
            "###.#.#####",
            "###....R###",
            "###########",
            "###########",
            "###########",
        ),
        (
            _g("G1", (7, 1), (6, 1), (7, 1)),
            _g("G2", (5, 1), (5, 2), (5, 1)),
            _g("G3", (5, 3), (5, 4), (5, 3)),
            _g("G4", (3, 5), (3, 6), (3, 5)),
            _g("G5", (3, 7), (4, 7), (3, 7)),
            _g("G6", (5, 5), (5, 6), (5, 6)),
        ),
        25,
    ),
    LevelDef(
        "The Lower Return",
        (
            "###########",
            "#S......E##",
            "#####.#####",
            "#.......###",
            "#.#####.###",
            "#.......###",
            "#.#########",
            "#......R###",
            "###########",
            "###########",
            "###########",
        ),
        (
            _g("G1", (5, 1), (5, 2), (5, 1)),
            _g("G2", (1, 7), (2, 7), (1, 7)),
            _g("G3", (5, 7), (6, 7), (5, 7)),
            _g("G4", (1, 5), (1, 4), (1, 5)),
            _g("G5", (1, 3), (2, 3), (1, 3)),
            _g("G6", (5, 1), (6, 1), (5, 1)),
        ),
        43,
    ),
    LevelDef(
        "The Last Open Throat",
        (
            "###########",
            "#S..#.....E",
            "###.#.###.#",
            "###...#...#",
            "#######.#.#",
            "#.......#.#",
            "#.#######.#",
            "#...#####.#",
            "###.#####.#",
            "###......R#",
            "###########",
        ),
        (
            _g("G1", (3, 1), (3, 2), (3, 1)),
            _g("G2", (5, 3), (5, 2), (5, 3)),
            _g("G3", (7, 1), (8, 1), (7, 1)),
            _g("G4", (9, 1), (9, 2), (9, 1)),
            _g("G5", (9, 3), (8, 3), (9, 3)),
            _g("G6", (7, 5), (6, 5), (7, 5)),
            _g("G7", (3, 5), (2, 5), (3, 5)),
            _g("G8", (1, 7), (2, 7), (1, 7)),
            _g("G9", (3, 9), (4, 9), (3, 9)),
            _g("G10", (7, 9), (8, 9), (7, 9)),
            _g("G11", (9, 9), (9, 8), (9, 9)),
            _g("G12", (9, 5), (9, 4), (9, 5)),
        ),
        45,
    ),
    LevelDef(
        "Three Routes, One Relay",
        (
            "###########",
            "#####....SE",
            "#####.#####",
            "###...#...#",
            "###.###.#.#",
            "#...###.#.#",
            "#.#####.#.#",
            "#...#...#.#",
            "###.#.###.#",
            "###......R#",
            "###########",
        ),
        (
            _g("G1", (9, 1), (8, 1), (9, 1)),
            _g("G2", (7, 1), (6, 1), (7, 1)),
            _g("G3", (5, 1), (5, 2), (5, 1)),
            _g("G4", (3, 3), (3, 4), (3, 3)),
            _g("G5", (1, 5), (1, 6), (1, 5)),
            _g("G6", (3, 7), (3, 8), (3, 7)),
            _g("G7", (3, 9), (4, 9), (3, 9)),
            _g("G8", (5, 9), (5, 8), (5, 9)),
            _g("G9", (7, 7), (7, 6), (7, 7)),
            _g("G10", (7, 5), (7, 4), (7, 5)),
            _g("G11", (9, 3), (9, 4), (9, 3)),
            _g("G12", (9, 7), (9, 8), (9, 7)),
            _g("G13", (7, 9), (6, 9), (7, 9)),
        ),
        61,
    ),
)


def _cell_xy(cell: Coord) -> tuple[int, int]:
    return ORIGIN + CELL * cell[0], ORIGIN + CELL * cell[1]


def _parse(spec: LevelDef) -> dict[str, Any]:
    floor: set[Coord] = set()
    start = relic = arch = None
    for y, row in enumerate(spec.rows):
        if len(row) != 11:
            raise ValueError(f"{spec.name}: row {y} is not 11 cells")
        for x, char in enumerate(row):
            if char in ".SR":
                floor.add((x, y))
            if char == "S":
                start = (x, y)
            elif char == "R":
                relic = (x, y)
            elif char == "E":
                arch = (x, y)
    if start is None or relic is None or arch is None:
        raise ValueError(f"{spec.name}: missing S, R, or E")
    return {"spec": spec, "floor": floor, "start": start, "relic": relic, "arch": arch}


def _level(spec: LevelDef) -> Level:
    return Level(sprites=[], grid_size=(SIZE, SIZE), data=_parse(spec), name=spec.name)


class TurnbackView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: TurnbackGates | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        for y in range(11):
            for x in range(11):
                x0, y0 = _cell_xy((x, y))
                frame[y0 : y0 + CELL, x0 : x0 + CELL] = FLOOR if (x, y) in game.floor else WALL

        self._draw_start(frame, game.start)
        self._draw_arch(frame, game)
        self._draw_pedestal(frame, game)
        self._draw_agent(frame, game)
        self._draw_gates(frame, game)
        self._draw_feedback(frame, game)
        self._draw_hud(frame, game)
        return frame

    def _draw_start(self, frame: np.ndarray, cell: Coord) -> None:
        x, y = _cell_xy(cell)
        frame[y + 1, x + 1 : x + 4] = HIGHLIGHT
        frame[y + 3, x + 1 : x + 4] = HIGHLIGHT
        frame[y + 1 : y + 4, x + 1] = HIGHLIGHT
        frame[y + 1 : y + 4, x + 3] = HIGHLIGHT

    def _draw_arch(self, frame: np.ndarray, game: TurnbackGates) -> None:
        x, y = _cell_xy(game.arch)
        arch_open = game._visual.get("arch_open", game.carrying_relic)
        color = OPEN if arch_open else CLOSED
        if game._visual.get("arch_pulse"):
            color = OPEN if arch_open else RELIC
        frame[y : y + 5, x : x + 5] = BACKGROUND
        frame[y, x : x + 5] = color
        frame[y + 4, x : x + 5] = color
        frame[y : y + 5, x] = color
        frame[y : y + 5, x + 4] = color
        adjacent = game.arch_neighbor
        dx, dy = adjacent[0] - game.arch[0], adjacent[1] - game.arch[1]
        if dx == -1:
            frame[y + 2, x] = BACKGROUND
        elif dx == 1:
            frame[y + 2, x + 4] = BACKGROUND
        elif dy == -1:
            frame[y, x + 2] = BACKGROUND
        else:
            frame[y + 4, x + 2] = BACKGROUND
        if not arch_open:
            frame[y + 2, x + 1 : x + 4] = CLOSED
            frame[y + 1 : y + 4, x + 2] = CLOSED
            self._diamond(frame, x + 2, y + 2, RELIC)

    def _draw_pedestal(self, frame: np.ndarray, game: TurnbackGates) -> None:
        x, y = _cell_xy(game.relic)
        frame[y + 2 : y + 5, x : x + 5] = PEDESTAL
        relic_present = game._visual.get("relic_present", game.relic_present)
        if relic_present:
            self._diamond(frame, x + 2, y + 1, RELIC)
        else:
            frame[y + 3, x + 2] = EMPTY
        if game._visual.get("relic_rise"):
            self._diamond(frame, x + 2, y, RELIC)

    def _draw_agent(self, frame: np.ndarray, game: TurnbackGates) -> None:
        pixel = game._visual.get("agent_pixel")
        if pixel is None:
            x, y = _cell_xy(game.agent_pos)
            x += 1
            y += 1
        else:
            x, y = pixel
        frame[y : y + 3, x : x + 3] = AGENT
        frame[y + 1, x + 1] = HIGHLIGHT
        visual_carry = game._visual.get("carrying", game.carrying_relic)
        if visual_carry:
            self._diamond(frame, x + 1, y + 1, RELIC)
        if game._visual.get("completion_outline"):
            for px, py in ((x - 1, y), (x + 3, y), (x, y - 1), (x, y + 3)):
                if 0 <= px < SIZE and 0 <= py < SIZE:
                    frame[py, px] = 0

    def _draw_gates(self, frame: np.ndarray, game: TurnbackGates) -> None:
        entries = game._visual.get("gate_entries", game.gate_entry_from)
        for gate in game.gates:
            flash = game._visual.get("blocked_gate") == gate.id
            self._draw_gate(frame, gate, entries[gate.id], flash)

    def _draw_gate(self, frame: np.ndarray, gate: GateDef, entry: Coord, flash: bool) -> None:
        left = min(gate.a, gate.b, key=lambda c: (c[0], c[1]))
        if gate.a[1] == gate.b[1]:
            left = gate.a if gate.a[0] < gate.b[0] else gate.b
            x0, y0 = _cell_xy(left)
            frame[y0 : y0 + 5, x0 + 4 : x0 + 6] = BLOCKED if flash else GATE
            xs = (4, 4, 4, 3, 2) if entry == left else (5, 5, 5, 6, 7)
            ys = (1, 2, 3, 2, 2)
            for px, py in zip(xs, ys, strict=True):
                frame[y0 + py, x0 + px] = ARROW
            if flash:
                frame[y0 + 2, x0 + 2 : x0 + 8] = BLOCKED
        else:
            top = gate.a if gate.a[1] < gate.b[1] else gate.b
            x0, y0 = _cell_xy(top)
            frame[y0 + 4 : y0 + 6, x0 : x0 + 5] = BLOCKED if flash else GATE
            ys = (4, 4, 4, 3, 2) if entry == top else (5, 5, 5, 6, 7)
            xs = (1, 2, 3, 2, 2)
            for px, py in zip(xs, ys, strict=True):
                frame[y0 + py, x0 + px] = ARROW
            if flash:
                frame[y0 + 2 : y0 + 8, x0 + 2] = BLOCKED

        # Persistent approach markers make the gate's current direction
        # readable without having to test both sides. Crossing swaps them.
        other = gate.b if entry == gate.a else gate.a
        ex, ey = _cell_xy(entry)
        ox, oy = _cell_xy(other)
        if entry[0] != other[0]:
            entry_edge = ex + (4 if other[0] > entry[0] else 0)
            blocked_edge = ox + (0 if other[0] > entry[0] else 4)
            frame[ey + 1 : ey + 4, entry_edge] = OPEN
            frame[oy + 1 : oy + 4, blocked_edge] = BLOCKED
        else:
            entry_edge = ey + (4 if other[1] > entry[1] else 0)
            blocked_edge = oy + (0 if other[1] > entry[1] else 4)
            frame[entry_edge, ex + 1 : ex + 4] = OPEN
            frame[blocked_edge, ox + 1 : ox + 4] = BLOCKED

    def _draw_feedback(self, frame: np.ndarray, game: TurnbackGates) -> None:
        if game._visual.get("bump"):
            x, y = _cell_xy(game.agent_pos)
            frame[y, x + 1 : x + 4] = EMPTY
        if game._visual.get("budget_fail"):
            frame[3:60, 3] = BLOCKED
            frame[3:60, 59] = BLOCKED
            frame[3, 3:60] = BLOCKED
            frame[59, 3:60] = BLOCKED

    def _draw_hud(self, frame: np.ndarray, game: TurnbackGates) -> None:
        frame[61:63, 4:59] = WALL
        width = int(55 * max(0, game.remaining_steps) / game.step_budget)
        if width:
            frame[61:63, 4 : 4 + width] = OPEN if width > 13 else BLOCKED
        for idx in range(7):
            frame[1:3, 4 + idx * 3 : 6 + idx * 3] = HIGHLIGHT if idx <= game.level_index else WALL

    @staticmethod
    def _diamond(frame: np.ndarray, cx: int, cy: int, color: int) -> None:
        for dx, dy in ((0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)):
            x, y = cx + dx, cy + dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                frame[y, x] = color


class TurnbackGates(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = TurnbackView()
        levels = [_level(spec) for spec in LEVEL_DEFS]
        super().__init__(
            "turnback_gates-0001",
            levels,
            Camera(0, 0, SIZE, SIZE, BACKGROUND, BACKGROUND, [self._view]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        data = self.current_level.get_data("spec")
        self.spec: LevelDef = data
        self.floor = set(self.current_level.get_data("floor"))
        self.start = tuple(self.current_level.get_data("start"))
        self.relic = tuple(self.current_level.get_data("relic"))
        self.arch = tuple(self.current_level.get_data("arch"))
        self.gates = self.spec.gates
        self.agent_pos = self.start
        self.relic_present = True
        self.carrying_relic = False
        self.won = False
        self.selection = None
        self.gate_entry_from = {gate.id: gate.initial_entry_from for gate in self.gates}
        self.gate_by_edge = {frozenset((gate.a, gate.b)): gate for gate in self.gates}
        neighbors = [cell for cell in self.floor if abs(cell[0] - self.arch[0]) + abs(cell[1] - self.arch[1]) == 1]
        if len(neighbors) != 1:
            raise ValueError(f"{self.spec.name}: arch needs exactly one floor neighbor")
        self.arch_neighbor = neighbors[0]
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[Coord, bool, bool, dict[str, Coord]]] = []
        self._visual: dict[str, Any] = {}
        self._animation: list[dict[str, Any]] = []
        self._finish_level_after_animation = False

    def _snapshot(self) -> tuple[Coord, bool, bool, dict[str, Coord]]:
        return (self.agent_pos, self.relic_present, self.carrying_relic, dict(self.gate_entry_from))

    def _restore(self, snapshot: tuple[Coord, bool, bool, dict[str, Coord]]) -> None:
        self.agent_pos, self.relic_present, self.carrying_relic, entries = snapshot
        self.gate_entry_from = dict(entries)
        self.won = False

    def _spend(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self._visual = {"budget_fail": True}
            self.lose()
            self.complete_action()
            return False
        return True

    def _queue(self, frames: list[dict[str, Any]], *, finish_level: bool = False) -> None:
        self._animation = frames
        self._finish_level_after_animation = finish_level
        self._play_frame()

    def _play_frame(self) -> None:
        self._visual = self._animation.pop(0)
        if self._animation:
            return
        if self._finish_level_after_animation:
            self.next_level()
            # Keep the solving input complete across ArcEngine's deferred
            # non-final level switch; otherwise it would be replayed once in
            # the freshly loaded level.
            self.complete_action()
        else:
            self.complete_action()

    @staticmethod
    def _agent_pixel(cell: Coord) -> tuple[int, int]:
        x, y = _cell_xy(cell)
        return x + 1, y + 1

    def _cross_frames(self, old: Coord, target: Coord, old_entries: dict[str, Coord]) -> list[dict[str, Any]]:
        old_px = self._agent_pixel(old)
        target_px = self._agent_pixel(target)
        touch = ((old_px[0] + target_px[0]) // 2, (old_px[1] + target_px[1]) // 2)
        return [
            {"agent_pixel": old_px, "gate_entries": old_entries},
            {"agent_pixel": touch, "gate_entries": old_entries},
            {"agent_pixel": target_px, "gate_entries": old_entries},
            {},
        ]

    def _move(self, dx: int, dy: int) -> None:
        old = self.agent_pos
        target = (old[0] + dx, old[1] + dy)
        if target == self.arch:
            if not self.carrying_relic:
                self._queue([{"arch_pulse": True}, {}])
                return
            self.undo_history.append(self._snapshot())
            self.agent_pos = target
            self.won = True
            self._queue(
                [{"agent_pixel": self._agent_pixel(old)}, {"arch_pulse": True, "completion_outline": True}, {}],
                finish_level=True,
            )
            return
        if target not in self.floor:
            self._queue([{"bump": True}, {}])
            return
        gate = self.gate_by_edge.get(frozenset((old, target)))
        if gate is not None and self.gate_entry_from[gate.id] != old:
            self._queue([{}, {"blocked_gate": gate.id}, {}])
            return

        self.undo_history.append(self._snapshot())
        old_entries = dict(self.gate_entry_from)
        self.agent_pos = target
        if gate is not None:
            self.gate_entry_from[gate.id] = target
        picked_up = self.relic_present and target == self.relic
        if picked_up:
            self.relic_present = False
            self.carrying_relic = True
            self._queue(
                [
                    {
                        "agent_pixel": self._agent_pixel(target),
                        "carrying": False,
                        "relic_present": True,
                        "arch_open": False,
                    },
                    {
                        "agent_pixel": self._agent_pixel(target),
                        "carrying": False,
                        "relic_present": False,
                        "relic_rise": True,
                        "arch_open": False,
                    },
                    {},
                ]
            )
        elif gate is not None:
            self._queue(self._cross_frames(old, target, old_entries))
        else:
            self._visual = {}
            self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._play_frame()
            return
        action = int(getattr(self.action.id, "value", self.action.id))
        if action == 0:
            self._visual = {}
            self.complete_action()
            return
        if not self._spend():
            return
        self._visual = {}
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.complete_action()
            return
        if action in DELTAS and not self.won:
            self._move(*DELTAS[action])
            return
        self.complete_action()
