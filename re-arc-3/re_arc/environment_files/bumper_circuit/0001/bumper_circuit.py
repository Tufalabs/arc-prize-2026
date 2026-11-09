from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

GAME_ID = "bumper_circuit-0001"
VIEW_SIZE = 64
CELL_SIZE = 3

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
SPACE, CLICK, UNDO = 5, 6, 7
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

BACKGROUND = 5
FLOOR = 4
WALL = 3
BEVEL = 2
WHITE = 0
METAL = 1
CABLE = 2
MAGENTA = 6
FAIL = 8
PUCK = 9
CYAN = 10
LIT = 11
BUMPER = 12
GATE = 13
COMPLETE = 14
F_LINK = 15


@dataclass(frozen=True)
class LinkSpec:
    switch: str
    obstacle: str
    color: int
    waypoints: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    links: tuple[LinkSpec, ...]
    solution: str

    @property
    def budget(self) -> int:
        return 6 * len(self.solution)


LEVEL_SPECS = (
    LevelSpec(
        "Catch the Wall",
        (
            "###############",
            "#.............#",
            "#.............#",
            "#..P..........#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#............A#",
            "#.............#",
            "#.............#",
            "#.............#",
            "###############",
        ),
        (),
        "RD",
    ),
    LevelSpec(
        "The Corner Shot",
        (
            "###############",
            "#.......P.....#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#..A....#.....#",
            "#..\\..........#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#......#......#",
            "###############",
        ),
        (),
        "RDLUL",
    ),
    LevelSpec(
        "Cross It, Then Build It",
        (
            "###############",
            "#.....1.......#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#\\.#...B......#",
            "#.............#",
            "#.............#",
            "#.............#",
            "#.A...........#",
            "#.............#",
            "#..........P..#",
            "###############",
        ),
        (LinkSpec("A", "1", MAGENTA, ((2, 9), (6, 9), (6, 1))),),
        "ULDDRULD",
    ),
    LevelSpec(
        "Two Branches, One Approach",
        (
            "###############",
            "#..#.......#..#",
            "#........\\..#P#",
            "#...#B....A...#",
            "#.............#",
            "##....#.......#",
            "#........#....#",
            "#......1..#...#",
            "#.....#.......#",
            "#..C..........#",
            "#2............#",
            "#.....#..#....#",
            "###############",
        ),
        (
            LinkSpec("A", "1", MAGENTA, ((10, 3), (7, 3), (7, 7))),
            LinkSpec("B", "2", COMPLETE, ((5, 3), (1, 3), (1, 10))),
        ),
        "DLULULDLURDRURDLULURDULDRLDLDR",
    ),
    LevelSpec(
        "Before and After",
        (
            "###################",
            "#...........#.....#",
            "#........C#.......#",
            "#.................#",
            "#.................#",
            "#.................#",
            "#..........#.....##",
            "#......2.....#....#",
            "#........\\#.......#",
            "#...........1.....#",
            "#...D.............#",
            "#...........#.....#",
            "#.......A..#......#",
            "#.................#",
            "#........B.......##",
            "#.....#..P........#",
            "###################",
        ),
        (
            LinkSpec("A", "R1", MAGENTA, ((8, 12), (9, 12), (9, 8))),
            LinkSpec("B", "1", COMPLETE, ((9, 14), (12, 14), (12, 9))),
            LinkSpec("D", "2", CYAN, ((4, 10), (7, 10), (7, 7))),
        ),
        "LURDRULDLDLRULDLURRULULDLDLULDRURDRULDLUR",
    ),
    LevelSpec(
        "The Gate You Already Crossed",
        (
            "###################",
            "#...........#.....#",
            "#........C#.......#",
            "#.................#",
            "#.................#",
            "#.................#",
            "#..........#.....##",
            "#......2.....#....#",
            "#........\\#.......#",
            "#...........1....E#",
            "#...D.............#",
            "#...........#.....#",
            "#.......A..#......#",
            "#.................#",
            "#........B.......##",
            "#.....3..P........#",
            "###################",
        ),
        (
            LinkSpec("A", "R1", MAGENTA, ((8, 12), (9, 12), (9, 8))),
            LinkSpec("B", "1", COMPLETE, ((9, 14), (12, 14), (12, 9))),
            LinkSpec("D", "2", CYAN, ((4, 10), (7, 10), (7, 7))),
            LinkSpec("E", "3", BUMPER, ((17, 9), (16, 9), (16, 15), (6, 15))),
        ),
        "LURDRULDLDRULURDRULDRDLURUURDRULDLDLULDRURDRULDLUR",
    ),
    LevelSpec(
        "Full Circuit",
        (
            "###################",
            "#...........#.....#",
            "#........C#.......#",
            "#.................#",
            "#......F..........#",
            "#.................#",
            "#..........#.....##",
            "#......2.....#....#",
            "#........\\4.......#",
            "#...........1....E#",
            "#...D.............#",
            "#...........#.....#",
            "#.......A.........#",
            "#.................#",
            "#........B.......##",
            "#.....3..P........#",
            "###################",
        ),
        (
            LinkSpec("A", "R1", MAGENTA, ((8, 12), (9, 12), (9, 8))),
            LinkSpec("B", "1", COMPLETE, ((9, 14), (12, 14), (12, 9))),
            LinkSpec("D", "2", CYAN, ((4, 10), (7, 10), (7, 7))),
            LinkSpec("E", "3", BUMPER, ((17, 9), (16, 9), (16, 15), (6, 15))),
            LinkSpec("F", "4", F_LINK, ((7, 4), (10, 4), (10, 8))),
        ),
        "LURDRULDLDRUDLURDRULDRDLURULDLDLURDRULDRDLURUURDRULDLDLULDRDLUR",
    ),
)


@dataclass(frozen=True)
class ParsedLevel:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    switches: dict[str, tuple[int, int]]
    bumpers: dict[str, tuple[tuple[int, int], str, bool]]
    gates: dict[str, tuple[int, int]]
    start: tuple[int, int]


@dataclass(frozen=True)
class RenderState:
    puck: tuple[int, int]
    lit: frozenset[str]
    remaining: int
    trail: tuple[tuple[int, int], ...] = ()
    bumper_flash: tuple[int, int] | None = None
    blocked: tuple[int, int] | None = None
    lit_pulse: str | None = None
    bright_link: str | None = None
    gate_stage: tuple[str, int] | None = None
    rotate_stage: int = 0
    outer_pulse: int | None = None


def _parse_level(spec: LevelSpec) -> ParsedLevel:
    height = len(spec.rows)
    width = len(spec.rows[0])
    assert all(len(row) == width for row in spec.rows)
    walls: set[tuple[int, int]] = set()
    switches: dict[str, tuple[int, int]] = {}
    gates: dict[str, tuple[int, int]] = {}
    start = None
    bumper_cell = None
    for y, row in enumerate(spec.rows):
        for x, symbol in enumerate(row):
            if symbol == "#":
                walls.add((x, y))
            elif symbol == "P":
                start = (x, y)
            elif symbol in "ABCDEF":
                switches[symbol] = (x, y)
            elif symbol in "1234":
                gates[symbol] = (x, y)
            elif symbol in "/\\":
                bumper_cell = (x, y, symbol)
    assert start is not None
    assert all((x, 0) in walls and (x, height - 1) in walls for x in range(width))
    assert all((0, y) in walls and (width - 1, y) in walls for y in range(height))
    occupied = set(walls) | set(switches.values()) | set(gates.values())
    assert start not in occupied
    bumpers = {}
    if bumper_cell is not None:
        x, y, orientation = bumper_cell
        assert (x, y) not in occupied
        bumpers["R1" if any(link.obstacle == "R1" for link in spec.links) else "M1"] = (
            (x, y),
            orientation,
            any(link.obstacle == "R1" for link in spec.links),
        )
    assert len(occupied) == len(walls) + len(switches) + len(gates)
    assert all(link.switch in switches for link in spec.links)
    assert all(link.obstacle in gates or link.obstacle in bumpers for link in spec.links)
    return ParsedLevel(width, height, frozenset(walls), switches, bumpers, gates, start)


PARSED_LEVELS = tuple(_parse_level(spec) for spec in LEVEL_SPECS)


def _build_level(spec: LevelSpec, parsed: ParsedLevel) -> Level:
    canvas = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=-20,
        collidable=False,
        tags=["canvas"],
    )
    return Level([canvas], (VIEW_SIZE, VIEW_SIZE), {"spec": spec, "parsed": parsed}, spec.name)


class BumperCircuitDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BumperCircuit | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is not None:
            self.game._render(frame)
        return frame


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _alternate(orientation: str) -> str:
    return "/" if orientation == "\\" else "\\"


def _reflect(direction: tuple[int, int], orientation: str) -> tuple[int, int]:
    dx, dy = direction
    if orientation == "/":
        return -dy, -dx
    return dy, dx


class BumperCircuit(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = BumperCircuitDisplay()
        self._validated_levels: set[int] = set()
        levels = [_build_level(spec, parsed) for spec, parsed in zip(LEVEL_SPECS, PARSED_LEVELS, strict=True)]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND, [self._display]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        for index in range(len(levels)):
            self.set_level(index)
        self.set_level(0)
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board: ParsedLevel = level.get_data("parsed")
        self.puck_position = self.board.start
        self.lit_switches: set[str] = set()
        self.remaining_actions = self.spec.budget
        self.terminal = "PLAYING"
        self.cursor = None
        self.selected_object = None
        self._history: list[tuple[tuple[int, int], frozenset[str]]] = []
        self._frames: list[RenderState] = []
        self._render_state = self._base_render_state()
        self._pending_outcome: str | None = None
        if self.level_index not in self._validated_levels:
            self._assert_reference_solution()
            self._validated_levels.add(self.level_index)

    def _base_render_state(self, *, lit: frozenset[str] | None = None) -> RenderState:
        return RenderState(
            self.puck_position, frozenset(self.lit_switches) if lit is None else lit, self.remaining_actions
        )

    def _source_for(self, obstacle: str) -> str | None:
        return next((link.switch for link in self.spec.links if link.obstacle == obstacle), None)

    def _gate_raised(self, gate: str, lit: frozenset[str] | set[str]) -> bool:
        source = self._source_for(gate)
        return source is not None and source in lit

    def _bumper_orientation(self, bumper: str, lit: frozenset[str] | set[str]) -> str:
        _, initial, rotating = self.board.bumpers[bumper]
        source = self._source_for(bumper)
        return _alternate(initial) if rotating and source in lit else initial

    def _solid_kind(self, cell: tuple[int, int], lit: frozenset[str] | set[str]) -> tuple[str, str | None] | None:
        if cell in self.board.walls:
            return "wall", None
        for switch, switch_cell in self.board.switches.items():
            if switch_cell == cell:
                return "switch", switch
        for gate, gate_cell in self.board.gates.items():
            if gate_cell == cell and self._gate_raised(gate, lit):
                return "gate", gate
        return None

    def _bumper_at(self, cell: tuple[int, int]) -> str | None:
        return next((name for name, (pos, _, _) in self.board.bumpers.items() if pos == cell), None)

    def _trajectory(
        self, start: tuple[int, int], direction: tuple[int, int], lit: frozenset[str] | set[str]
    ) -> tuple[tuple[int, int], list[list[tuple[int, int]]], tuple[str, str | None] | None, tuple[int, int] | None]:
        current = start
        segment: list[tuple[int, int]] = []
        segments: list[list[tuple[int, int]]] = []
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        while True:
            directed = (current, direction)
            assert directed not in seen, f"trajectory loop at {directed}"
            seen.add(directed)
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            solid = self._solid_kind(next_cell, lit)
            if solid is not None:
                if segment:
                    segments.append(segment)
                return current, segments, solid, next_cell
            current = next_cell
            segment.append(current)
            bumper = self._bumper_at(current)
            if bumper is not None:
                segments.append(segment)
                segment = []
                direction = _reflect(direction, self._bumper_orientation(bumper, lit))

    def _queue(self, state: RenderState) -> None:
        self._frames.append(state)

    def _finish_or_continue(self) -> None:
        if self._frames:
            self._render_state = self._frames.pop(0)
            if self._frames:
                return
        if self._pending_outcome == "win":
            self.terminal = "WON"
            self.next_level()
            self.complete_action()
        elif self._pending_outcome == "lose":
            self.terminal = "FAILED"
            self.lose()
            self.complete_action()
        else:
            self.complete_action()
        self._pending_outcome = None

    def _resolve_arrow(self, action: int) -> None:
        old_lit = frozenset(self.lit_switches)
        old_puck = self.puck_position
        settled, segments, collision, blocking_cell = self._trajectory(old_puck, DELTAS[action], old_lit)
        changed = settled != old_puck
        struck = collision[1] if collision and collision[0] == "switch" else None
        if struck is not None and struck not in old_lit:
            changed = True
        if changed:
            self._history.append((old_puck, old_lit))

        for segment in segments:
            if len(segment) > 1:
                trail = tuple(segment)
                self._queue(RenderState(segment[-1], old_lit, self.remaining_actions, trail=trail))
            endpoint_bumper = self._bumper_at(segment[-1]) if segment else None
            if endpoint_bumper is not None:
                self._queue(RenderState(segment[-1], old_lit, self.remaining_actions, bumper_flash=segment[-1]))

        self.puck_position = settled
        self._queue(self._base_render_state(lit=old_lit))

        if collision and collision[0] in {"wall", "gate"} and settled == old_puck:
            pulse = replace(self._base_render_state(lit=old_lit), blocked=blocking_cell)
            self._queue(pulse)
            self._queue(pulse)
        elif struck is not None and struck in old_lit:
            self._queue(replace(self._base_render_state(), lit_pulse=struck))
        elif struck is not None:
            self.lit_switches.add(struck)
            new_lit = frozenset(self.lit_switches)
            link = next((item for item in self.spec.links if item.switch == struck), None)
            self._queue(replace(self._base_render_state(), bright_link=struck))
            if link is not None and link.obstacle in self.board.gates:
                self._queue(replace(self._base_render_state(), gate_stage=(link.obstacle, 1)))
                self._queue(replace(self._base_render_state(), gate_stage=(link.obstacle, 2)))
            elif link is not None and link.obstacle in self.board.bumpers:
                self._queue(replace(self._base_render_state(), rotate_stage=1))
                self._queue(replace(self._base_render_state(), rotate_stage=2))
            self._queue(self._base_render_state())
            if new_lit == frozenset(self.board.switches):
                for _ in range(3):
                    self._queue(replace(self._base_render_state(), outer_pulse=COMPLETE))
                self._pending_outcome = "win"

    def _resolve_idle(self) -> None:
        self._queue(self._base_render_state())

    def _resolve_undo(self) -> None:
        if self._history:
            self.puck_position, restored_lit = self._history.pop()
            self.lit_switches = set(restored_lit)
        self._queue(self._base_render_state())

    def step(self) -> None:
        if self._frames:
            self._finish_or_continue()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self._render_state = self._base_render_state()
            self.complete_action()
            return

        self.remaining_actions = max(0, self.remaining_actions - 1)
        if action in DELTAS:
            self._resolve_arrow(action)
        elif action == UNDO:
            self._resolve_undo()
        else:
            self._resolve_idle()

        if self._pending_outcome is None and self.remaining_actions == 0:
            for _ in range(3):
                self._queue(replace(self._base_render_state(), outer_pulse=FAIL))
            self._pending_outcome = "lose"
        self._finish_or_continue()

    def _cell_origin(self, cell: tuple[int, int]) -> tuple[int, int]:
        ox = (VIEW_SIZE - CELL_SIZE * self.board.width) // 2
        oy = 62 - CELL_SIZE * self.board.height
        return ox + CELL_SIZE * cell[0], oy + CELL_SIZE * cell[1]

    def _center(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = self._cell_origin(cell)
        return x + 1, y + 1

    def _draw_cable(self, frame: np.ndarray, link: LinkSpec, color: int) -> None:
        points = [self._center(cell) for cell in link.waypoints]
        for (x1, y1), (x2, y2) in pairwise(points):
            if x1 == x2:
                frame[min(y1, y2) : max(y1, y2) + 1, x1] = color
            else:
                frame[y1, min(x1, x2) : max(x1, x2) + 1] = color
        for x, y in (points[0], points[-1]):
            frame[y, x] = color

    def _draw_bumper(
        self, frame: np.ndarray, cell: tuple[int, int], orientation: str, rotating: bool, flash: bool
    ) -> None:
        x, y = self._cell_origin(cell)
        frame[y : y + 3, x : x + 3] = FLOOR
        current = WHITE if flash else BUMPER
        if orientation == "\\":
            frame[y, x] = current
            frame[y + 1, x + 1] = current
            frame[y + 2, x + 2] = current
            if rotating:
                frame[y, x + 2] = BEVEL
                frame[y + 2, x] = BEVEL
        else:
            frame[y, x + 2] = current
            frame[y + 1, x + 1] = current
            frame[y + 2, x] = current
            if rotating:
                frame[y, x] = BEVEL
                frame[y + 2, x + 2] = BEVEL

    def _draw_gate(self, frame: np.ndarray, cell: tuple[int, int], raised: bool, stage: int) -> None:
        x, y = self._cell_origin(cell)
        if not raised and stage == 0:
            frame[y : y + 3, x : x + 3] = ((BEVEL, FLOOR, BEVEL), (FLOOR, METAL, FLOOR), (BEVEL, FLOOR, BEVEL))
            return
        if stage == 1:
            frame[y : y + 3, x : x + 3] = FLOOR
            frame[y + 1 : y + 3, x : x + 3] = GATE
            frame[y + 1, x + 1] = METAL
            return
        frame[y : y + 3, x : x + 3] = GATE
        frame[y + 1, x + 1] = METAL

    def _render(self, frame: np.ndarray) -> None:
        state = self._render_state
        frame[:, :] = BACKGROUND

        bar_x, bar_width = 3, 58
        frame[0, bar_x : bar_x + bar_width] = BEVEL
        filled = int(bar_width * state.remaining / max(1, self.spec.budget))
        frame[0, bar_x : bar_x + filled] = CYAN
        pip_total = len(self.board.switches)
        pip_start = (VIEW_SIZE - (pip_total * 4 - 1)) // 2
        for index, switch in enumerate(sorted(self.board.switches)):
            x = pip_start + 4 * index
            frame[3:6, x : x + 3] = (
                (WALL, BEVEL, WALL),
                (BEVEL, LIT if switch in state.lit else METAL, BEVEL),
                (WALL, BEVEL, WALL),
            )

        ox, oy = self._cell_origin((0, 0))
        frame[oy : oy + CELL_SIZE * self.board.height, ox : ox + CELL_SIZE * self.board.width] = FLOOR
        for link in self.spec.links:
            self._draw_cable(frame, link, WHITE if state.bright_link == link.switch else link.color)

        for wall in self.board.walls:
            x, y = self._cell_origin(wall)
            frame[y : y + 3, x : x + 3] = WALL
            if (wall[0], wall[1] - 1) not in self.board.walls:
                frame[y, x : x + 3] = BEVEL
            if (wall[0] - 1, wall[1]) not in self.board.walls:
                frame[y : y + 3, x] = BEVEL
        if state.blocked is not None:
            x, y = self._cell_origin(state.blocked)
            frame[y : y + 3, x : x + 3] = FAIL

        for gate, cell in self.board.gates.items():
            source = self._source_for(gate)
            raised = source in state.lit
            stage = state.gate_stage[1] if state.gate_stage and state.gate_stage[0] == gate else 0
            self._draw_gate(frame, cell, raised, stage)

        for bumper, (cell, _initial, rotating) in self.board.bumpers.items():
            orientation = self._bumper_orientation(bumper, state.lit)
            if state.rotate_stage == 1 and rotating:
                x, y = self._cell_origin(cell)
                frame[y : y + 3, x : x + 3] = FLOOR
                for slash in ("/", "\\"):
                    coords = ((0, 2), (1, 1), (2, 0)) if slash == "/" else ((0, 0), (1, 1), (2, 2))
                    color = METAL if slash == orientation else BUMPER
                    for dx, dy in coords:
                        frame[y + dy, x + dx] = color
            else:
                self._draw_bumper(frame, cell, orientation, rotating, state.bumper_flash == cell)

        for switch, cell in self.board.switches.items():
            x, y = self._cell_origin(cell)
            center = LIT if switch in state.lit else METAL
            frame[y : y + 3, x : x + 3] = ((WALL, BEVEL, WALL), (BEVEL, center, BEVEL), (WALL, BEVEL, WALL))
            if state.lit_pulse == switch:
                frame[y + 1, x : x + 3] = LIT
                frame[y : y + 3, x + 1] = LIT

        for cell in state.trail:
            x, y = self._center(cell)
            frame[y, x] = CYAN
        x, y = self._cell_origin(state.puck)
        frame[y : y + 3, x : x + 3] = ((FLOOR, CYAN, FLOOR), (CYAN, PUCK, CYAN), (FLOOR, CYAN, FLOOR))
        if state.outer_pulse is not None:
            frame[0, :] = state.outer_pulse
            frame[-1, :] = state.outer_pulse
            frame[:, 0] = state.outer_pulse
            frame[:, -1] = state.outer_pulse

    def _simulate_reference(self, actions: str) -> tuple[set[str], tuple[int, int], list[str]]:
        puck = self.board.start
        lit: set[str] = set()
        order: list[str] = []
        for index, letter in enumerate(actions):
            action = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}[letter]
            puck, _, collision, _ = self._trajectory(puck, DELTAS[action], lit)
            if collision and collision[0] == "switch" and collision[1] not in lit:
                lit.add(collision[1])
                order.append(collision[1])
            if index < len(actions) - 1:
                assert lit != set(self.board.switches), "strict reference prefix wins"
        return lit, puck, order

    def _assert_reference_solution(self) -> None:
        lit, _, _ = self._simulate_reference(self.spec.solution)
        assert lit == set(self.board.switches)

        start = (self.board.start, frozenset())
        queue = deque([start])
        reachable = {start}
        while queue:
            puck, active = queue.popleft()
            if active == frozenset(self.board.switches):
                continue
            for direction in DELTAS.values():
                settled, _, collision, _ = self._trajectory(puck, direction, active)
                next_active = set(active)
                if collision and collision[0] == "switch":
                    next_active.add(collision[1])
                frozen_active = frozenset(next_active)
                for gate, gate_cell in self.board.gates.items():
                    assert not (
                        self._gate_raised(gate, frozen_active)
                        and not self._gate_raised(gate, active)
                        and settled == gate_cell
                    ), f"activation raises gate {gate} under puck"
                next_state = (settled, frozen_active)
                if next_state not in reachable:
                    reachable.add(next_state)
                    queue.append(next_state)
