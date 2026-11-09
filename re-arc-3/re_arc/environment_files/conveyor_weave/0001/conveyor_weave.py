from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "conveyor_weave-0001"
VIEW_SIZE = 64
BOARD_OFFSET = 2
CELL_SIZE = 5

UP, RIGHT, DOWN, LEFT = "U", "R", "D", "L"
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}
ACTION_TO_DIRECTION = {1: UP, 2: DOWN, 3: LEFT, 4: RIGHT}

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_WALL = 4
COLOR_VOID = 5
COLOR_ROT_RIM = 7
COLOR_JAM = 8
COLOR_FIXED = 9
COLOR_FIXED_DETAIL = 10
COLOR_AGENT = 11
COLOR_NOSE = 12
COLOR_GATE = 13
COLOR_EXIT = 14
COLOR_ROTATABLE = 15

ARROW_UP = np.array(
    [[-1, -1, 0, -1, -1], [-1, 0, 0, 0, -1], [-1, -1, 0, -1, -1], [-1, -1, 0, -1, -1], [-1, -1, 0, -1, -1]],
    dtype=np.int8,
)

ROT_RIM = ((0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (3, 0), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4))


Coord = tuple[int, int]
Edge = tuple[Coord, Coord]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start_facing: str
    rotatable_initial: tuple[tuple[str, str], ...]
    gates: tuple[Edge, ...] = ()
    walls: tuple[Edge, ...] = ()
    baseline: int = 1


LEVEL_SPECS = (
    LevelSpec(
        "Ride the Arrow",
        (
            "############",
            "############",
            "############",
            "#P>>>E######",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
        RIGHT,
        (),
        baseline=1,
    ),
    LevelSpec(
        "Turn Before Boarding",
        (
            "############",
            "############",
            "############",
            "##..########",
            "#..T>E######",
            "#P..########",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
        UP,
        (("T", LEFT),),
        baseline=5,
    ),
    LevelSpec(
        "The Useful Jam",
        (
            "#####E######",
            "#####^######",
            "#####T######",
            "#####.######",
            "#..>>>######",
            "#P##########",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
        UP,
        (("T", LEFT),),
        baseline=6,
    ),
    LevelSpec(
        "Wrong-Side Gate",
        (
            "############",
            "#####E######",
            "#####^######",
            "#####^######",
            "#P>>>AB...##",
            "#####.###.##",
            "#####.###.##",
            "####..C>>.##",
            "############",
            "############",
            "############",
            "############",
        ),
        RIGHT,
        (("A", RIGHT), ("B", UP), ("C", LEFT)),
        gates=((((5, 7), (5, 6))), (((5, 6), (6, 6)))),
        baseline=22,
    ),
    LevelSpec(
        "Two Shores",
        (
            "#v<<<<<<<<<#",
            "#v####^###^#",
            "..U>^#^###^#",
            ".###..X#..Y#",
            ".###.###.###",
            ".P>>A>>>B>E#",
            "#.#.###.####",
            "#.......####",
            "############",
            "############",
            "############",
            "############",
        ),
        LEFT,
        (("U", DOWN), ("A", LEFT), ("X", RIGHT), ("B", DOWN), ("Y", LEFT)),
        gates=(((6, 5), (5, 5)), ((6, 9), (5, 9)), ((6, 4), (7, 4)), ((6, 8), (7, 8))),
        baseline=40,
    ),
    LevelSpec(
        "Borrowed Return Belt",
        (
            "############",
            "##v<<<<<<<<#",
            "#...L>>v##^#",
            "#.#.###v##^#",
            "#.#.^<<>>v^#",
            "#.PQ>....v^#",
            "##^>>^##B<.#",
            "##^###v<C.##",
            "##^###v##.##",
            "##^###E##R##",
            "##^<<<<<<<##",
            "############",
        ),
        RIGHT,
        (("Q", LEFT), ("L", DOWN), ("B", RIGHT), ("C", DOWN), ("R", LEFT)),
        gates=(((5, 4), (6, 4)), ((7, 9), (6, 9)), ((7, 10), (7, 11)), ((8, 9), (8, 10))),
        walls=(((3, 4), (4, 4)), ((5, 8), (6, 8)), ((5, 9), (6, 9)), ((6, 9), (6, 10)), ((7, 10), (8, 10))),
        baseline=42,
    ),
    LevelSpec(
        "Conveyor Weave",
        (
            "#v<<<<<<<<<<",
            "#v##.......I",
            "#v#>^v<<<<C#",
            "#<.T^E####^.",
            "###.######^.",
            "###.^<<###^.",
            "#.PQ>.....^.",
            "#.#>>^##..^.",
            "#.######..B.",
            "#..L>>>>>>^.",
            "##..........",
            "############",
        ),
        RIGHT,
        (("Q", LEFT), ("T", LEFT), ("I", RIGHT), ("C", DOWN), ("B", DOWN), ("L", DOWN)),
        gates=(
            ((6, 4), (7, 4)),
            ((4, 3), (4, 4)),
            ((3, 11), (2, 11)),
            ((9, 11), (9, 10)),
            ((4, 11), (4, 12)),
            ((10, 4), (11, 4)),
            ((10, 5), (11, 5)),
            ((10, 6), (11, 6)),
            ((10, 7), (11, 7)),
            ((10, 8), (11, 8)),
            ((10, 9), (11, 9)),
            ((10, 10), (11, 10)),
            ((10, 11), (11, 11)),
        ),
        walls=(
            ((1, 5), (2, 5)),
            ((1, 6), (2, 6)),
            ((1, 7), (2, 7)),
            ((1, 8), (2, 8)),
            ((1, 9), (2, 9)),
            ((1, 10), (2, 10)),
            ((1, 11), (2, 11)),
            ((2, 6), (3, 6)),
            ((2, 7), (3, 7)),
            ((2, 8), (3, 8)),
            ((2, 9), (3, 9)),
            ((2, 10), (3, 10)),
            ((5, 11), (5, 12)),
            ((6, 11), (6, 12)),
            ((7, 11), (7, 12)),
            ((8, 11), (8, 12)),
            ((9, 11), (9, 12)),
            ((10, 11), (10, 12)),
            ((7, 10), (7, 11)),
            ((8, 10), (8, 11)),
            ((9, 9), (10, 9)),
            ((9, 10), (10, 10)),
        ),
        baseline=50,
    ),
)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _add(coord: Coord, direction: str) -> Coord:
    dr, dc = DELTAS[direction]
    return coord[0] + dr, coord[1] + dc


def _unordered(edge: Edge) -> frozenset[Coord]:
    return frozenset(edge)


def _turns(direction: str) -> int:
    return {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}[direction]


def _make_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((60, 60), COLOR_VOID, dtype=np.int8),
        name="board",
        x=BOARD_OFFSET,
        y=BOARD_OFFSET,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level([board], (64, 64), {"spec": spec}, spec.name)


class ConveyorWeaveHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ConveyorWeave | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        width = int(60 * max(0, game.remaining_steps) / max(1, game.step_budget))
        frame[62:64, 2:62] = COLOR_WALL
        if width:
            frame[62:64, 2 : 2 + width] = COLOR_NOSE
        return frame


class ConveyorWeave(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = ConveyorWeaveHud()
        super().__init__(
            GAME_ID,
            [_make_level(spec) for spec in LEVEL_SPECS],
            Camera(0, 0, 64, 64, COLOR_VOID, COLOR_VOID, [self._hud]),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.floor: set[Coord] = set()
        self.fixed: dict[Coord, str] = {}
        self.rotatable_cells: dict[str, Coord] = {}
        initial = dict(self.spec.rotatable_initial)
        self.start: Coord | None = None
        self.exit: Coord | None = None
        for row, text in enumerate(self.spec.rows, 1):
            if len(text) != 12:
                raise ValueError(f"{self.spec.name}: row {row} is not 12 cells")
            for col, char in enumerate(text, 1):
                cell = (row, col)
                if char == "#":
                    continue
                if char in "^>v<":
                    self.fixed[cell] = {"^": UP, ">": RIGHT, "v": DOWN, "<": LEFT}[char]
                elif char in initial:
                    self.rotatable_cells[char] = cell
                else:
                    self.floor.add(cell)
                    if char == "P":
                        self.start = cell
                    elif char == "E":
                        self.exit = cell
        if self.start is None or self.exit is None:
            raise ValueError(f"{self.spec.name}: missing P or E")
        if set(initial) != set(self.rotatable_cells):
            raise ValueError(f"{self.spec.name}: rotatable labels do not match layout")
        self.rotatable_directions = {self.rotatable_cells[label]: direction for label, direction in initial.items()}
        self.gates = set(self.spec.gates)
        self.gate_seams = {_unordered(edge) for edge in self.spec.gates}
        self.walls = {_unordered(edge) for edge in self.spec.walls}
        self.agent_position = self.start
        self.agent_facing = self.spec.start_facing
        self.agent_jammed = False
        self.jammed_edge: Edge | None = None
        self.complete = False
        self.selected_object = None
        self.cursor_visible = False
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[Any, ...]] = []
        self.animation_frames: list[dict[str, Any]] = []
        self._render_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._pending_next = False
        self._pending_loss = False
        self._render()

    @property
    def conveyors(self) -> dict[Coord, str]:
        return self.fixed | self.rotatable_directions

    def _snapshot(self) -> tuple[Any, ...]:
        return (
            self.agent_position,
            self.agent_facing,
            self.agent_jammed,
            self.jammed_edge,
            dict(self.rotatable_directions),
        )

    def _restore(self, snapshot: tuple[Any, ...]) -> None:
        position, facing, jammed, edge, directions = snapshot
        self.agent_position = position
        self.agent_facing = facing
        self.agent_jammed = jammed
        self.jammed_edge = edge
        self.rotatable_directions = dict(directions)

    def _traversable(self, cell: Coord) -> bool:
        return cell in self.floor or cell in self.conveyors

    def _seam_allows(self, source: Coord, target: Coord) -> bool:
        seam = frozenset((source, target))
        if seam in self.walls:
            return False
        if seam in self.gate_seams:
            return (source, target) in self.gates
        return True

    def _resolve_conveyors(self, first: Coord) -> list[tuple[Coord, Coord, bool, str]]:
        trace: list[tuple[Coord, Coord, bool, str]] = []
        current = first
        visited = {current}
        self.agent_position = current
        self.agent_jammed = False
        self.jammed_edge = None
        while True:
            direction = self.conveyors[current]
            self.agent_facing = direction
            target = _add(current, direction)
            allowed = self._traversable(target) and self._seam_allows(current, target)
            if allowed and target in self.conveyors:
                target_direction = self.conveyors[target]
                allowed = target_direction != OPPOSITE[direction] and target not in visited
            trace.append((current, target, allowed, direction))
            if not allowed:
                self.agent_position = current
                self.agent_jammed = True
                self.jammed_edge = (current, target)
                break
            self.agent_position = target
            if target in self.floor:
                break
            visited.add(target)
            current = target
        return trace

    def _queue_conveyor_trace(self, trace: list[tuple[Coord, Coord, bool, str]]) -> None:
        frames: list[dict[str, Any]] = []
        for source, target, allowed, direction in trace:
            frames.append({"agent_pos": source, "agent_facing": direction, "stretch": direction})
            if allowed:
                frames.append({"agent_pos": target, "agent_facing": direction})
            else:
                frames.append({"agent_pos": source, "agent_facing": direction, "edge_flash": (source, target)})
        frames.append({})
        self.animation_frames = frames

    def _move(self, direction: str) -> None:
        self.agent_facing = direction
        target = _add(self.agent_position, direction)
        if self.agent_jammed:
            legal = target in self.floor and self._seam_allows(self.agent_position, target)
            if legal:
                self.agent_position = target
                self.agent_jammed = False
                self.jammed_edge = None
            else:
                self.animation_frames = [{"edge_flash": (self.agent_position, target)}, {}]
            return
        if not self._traversable(target) or not self._seam_allows(self.agent_position, target):
            self.animation_frames = [{"edge_flash": (self.agent_position, target)}, {}]
            return
        self.agent_position = target
        self.agent_jammed = False
        self.jammed_edge = None
        if target in self.conveyors:
            self._queue_conveyor_trace(self._resolve_conveyors(target))

    def _rotate(self) -> bool:
        if self.agent_jammed:
            return False
        target = _add(self.agent_position, self.agent_facing)
        if target not in self.rotatable_directions:
            return False
        if frozenset((self.agent_position, target)) in self.walls:
            return False
        old = self.rotatable_directions[target]
        new = DIRECTIONS[(DIRECTIONS.index(old) + 1) % 4]
        self.rotatable_directions[target] = new
        self.animation_frames = [{"dirs_override": {target: old}}, {"transition_cell": target}, {}]
        return True

    def _queue_invalid_space(self) -> None:
        edge = (self.agent_position, _add(self.agent_position, self.agent_facing))
        self.animation_frames = [{"lock_edge": edge}, {}]

    def _queue_completion(self) -> None:
        self.complete = True
        self.animation_frames.extend(
            [{"pulse": COLOR_WHITE}, {"pulse": COLOR_EXIT}, {"pulse": COLOR_WHITE}, {"pulse": COLOR_EXIT}, {}]
        )
        self._pending_next = True

    def _draw_cell(self, pixels: np.ndarray, cell: Coord, color: int) -> tuple[int, int]:
        y = (cell[0] - 1) * CELL_SIZE
        x = (cell[1] - 1) * CELL_SIZE
        pixels[y : y + CELL_SIZE, x : x + CELL_SIZE] = color
        return y, x

    def _draw_conveyor(self, pixels: np.ndarray, cell: Coord, direction: str, rotatable: bool) -> None:
        y, x = self._draw_cell(pixels, cell, COLOR_ROTATABLE if rotatable else COLOR_FIXED)
        if rotatable:
            for py, px in ROT_RIM:
                pixels[y + py, x + px] = COLOR_ROT_RIM
        else:
            rear = {UP: ((4, 0), (4, 4)), RIGHT: ((0, 0), (4, 0)), DOWN: ((0, 0), (0, 4)), LEFT: ((0, 4), (4, 4))}[
                direction
            ]
            for py, px in rear:
                pixels[y + py, x + px] = COLOR_FIXED_DETAIL
        arrow = np.rot90(ARROW_UP, -_turns(direction))
        mask = arrow >= 0
        tile = pixels[y : y + 5, x : x + 5]
        tile[mask] = arrow[mask]

    def _draw_edge(self, pixels: np.ndarray, edge: Edge, color: int, *, full: bool = False) -> None:
        (r1, c1), (r2, c2) = edge
        if r1 == r2:
            x = max(c1, c2) * 5 - 5
            y = (r1 - 1) * 5
            span = slice(y, y + 5) if full else slice(y + 1, y + 4)
            pixels[span, max(0, x - 1) : min(60, x + 1)] = color
        else:
            y = max(r1, r2) * 5 - 5
            x = (c1 - 1) * 5
            span = slice(x, x + 5) if full else slice(x + 1, x + 4)
            pixels[max(0, y - 1) : min(60, y + 1), span] = color

    def _draw_gate(self, pixels: np.ndarray, edge: Edge) -> None:
        self._draw_edge(pixels, edge, COLOR_GATE)
        source, target = edge
        dr, dc = target[0] - source[0], target[1] - source[1]
        sy, sx = (source[0] - 1) * 5, (source[1] - 1) * 5
        if dc == 1:
            points = ((2, 4), (1, 3), (3, 3))
        elif dc == -1:
            points = ((2, 0), (1, 1), (3, 1))
        elif dr == 1:
            points = ((4, 2), (3, 1), (3, 3))
        else:
            points = ((0, 2), (1, 1), (1, 3))
        for py, px in points:
            pixels[sy + py, sx + px] = COLOR_NOSE

    def _draw_agent(self, pixels: np.ndarray, position: Coord, facing: str, stretch: str | None) -> None:
        y, x = (position[0] - 1) * 5, (position[1] - 1) * 5
        for py, px in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)):
            pixels[y + py, x + px] = COLOR_AGENT
        nose = {UP: (0, 2), RIGHT: (2, 4), DOWN: (4, 2), LEFT: (2, 0)}[facing]
        pixels[y + nose[0], x + nose[1]] = COLOR_NOSE
        if stretch is not None:
            extras = {UP: ((0, 1), (0, 3)), RIGHT: ((1, 4), (3, 4)), DOWN: ((4, 1), (4, 3)), LEFT: ((1, 0), (3, 0))}[
                stretch
            ]
            for py, px in extras:
                pixels[y + py, x + px] = COLOR_NOSE

    def _render(self, visual: dict[str, Any] | None = None) -> None:
        visual = visual or {}
        directions = self.conveyors | visual.get("dirs_override", {})
        transition = visual.get("transition_cell")
        pulse = visual.get("pulse")
        cache_key = (tuple(sorted(directions.items())), transition, pulse)
        cached = self._render_cache.get(cache_key)
        if cached is None:
            pixels = np.full((60, 60), COLOR_VOID, dtype=np.int8)
            for cell in self.floor:
                self._draw_cell(pixels, cell, COLOR_FLOOR)
            if self.exit is not None:
                y, x = self._draw_cell(pixels, self.exit, COLOR_EXIT)
                pixels[y + 1, x + 1 : x + 4] = COLOR_WHITE
                pixels[y + 3, x + 1 : x + 4] = COLOR_WHITE
                pixels[y + 1 : y + 4, x + 1] = COLOR_WHITE
                pixels[y + 1 : y + 4, x + 3] = COLOR_WHITE
            for cell, direction in directions.items():
                self._draw_conveyor(pixels, cell, direction, cell in self.rotatable_directions)
            if transition is not None:
                y, x = self._draw_cell(pixels, transition, COLOR_ROTATABLE)
                for py, px in ROT_RIM:
                    pixels[y + py, x + px] = COLOR_ROT_RIM
                pixels[y + 2, x + 1 : x + 4] = COLOR_WHITE
                pixels[y + 1 : y + 4, x + 2] = COLOR_WHITE
            for wall in self.spec.walls:
                self._draw_edge(pixels, wall, COLOR_WALL, full=True)
            for gate in self.spec.gates:
                self._draw_gate(pixels, gate)
            if pulse is not None and self.exit is not None:
                self._draw_cell(pixels, self.exit, pulse)
            self._render_cache[cache_key] = pixels.copy()
        else:
            pixels = cached.copy()
        position = visual.get("agent_pos", self.agent_position)
        facing = visual.get("agent_facing", self.agent_facing)
        self._draw_agent(pixels, position, facing, visual.get("stretch"))
        if self.agent_jammed and self.jammed_edge is not None and position == self.agent_position:
            self._draw_edge(pixels, self.jammed_edge, COLOR_JAM)
        if visual.get("edge_flash") is not None:
            self._draw_edge(pixels, visual["edge_flash"], COLOR_JAM, full=True)
        if visual.get("lock_edge") is not None:
            source, target = visual["lock_edge"]
            self._draw_edge(pixels, (source, target), COLOR_JAM)
            sy, sx = (source[0] - 1) * 5, (source[1] - 1) * 5
            pixels[sy + 1 : sy + 4, sx + 1] = COLOR_JAM
            pixels[sy + 1, sx + 1 : sx + 4] = COLOR_JAM
            pixels[sy + 3, sx + 1 : sx + 4] = COLOR_JAM
        self.board.pixels = pixels

    def _finish_action(self) -> None:
        self._render()
        if self._pending_next:
            self._pending_next = False
            self.next_level()
        elif self._pending_loss:
            self._pending_loss = False
            self.lose()
        self.complete_action()

    def _play_animation_frame(self) -> None:
        visual = self.animation_frames.pop(0)
        self._render(visual)
        if not self.animation_frames:
            self._finish_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._render()
            self.complete_action()
            return

        self.remaining_steps -= 1
        if action == 7:
            if self.undo_history:
                self._restore(self.undo_history.pop())
        elif action in ACTION_TO_DIRECTION:
            before = self._snapshot()
            self._move(ACTION_TO_DIRECTION[action])
            if self._snapshot() != before:
                self.undo_history.append(before)
        elif action == 5:
            before = self._snapshot()
            if self._rotate():
                self.undo_history.append(before)
            else:
                self._queue_invalid_space()
        # ACTION6 is deliberately inert.

        if self.agent_position == self.exit:
            self._queue_completion()
        elif self.remaining_steps <= 0:
            self._pending_loss = True

        if self.animation_frames:
            self._play_animation_frame()
        else:
            self._finish_action()
