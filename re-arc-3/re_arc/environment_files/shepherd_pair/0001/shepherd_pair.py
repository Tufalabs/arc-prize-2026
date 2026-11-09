from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 4
LEFT_ORIGIN = (1, 16)
RIGHT_ORIGIN = (35, 16)

COLOR_FLOOR = 1
COLOR_FRAME = 3
COLOR_WALL = 4
COLOR_BACKGROUND = 5
COLOR_FLASH = 8
COLOR_SHEPHERD = 9
COLOR_TRACER = 10
COLOR_COMPANION = 11
COLOR_GATE = 12
COLOR_GOAL = 14

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

SHEPHERD_MASK = ((1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2))
COMPANION_MASK = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3))

Cell = tuple[int, int]
Gate = tuple[Cell, Cell]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    shepherd_rows: tuple[str, ...]
    companion_rows: tuple[str, ...]
    shepherd_start: Cell
    companion_start: Cell
    goal: Cell
    baseline: int
    shepherd_gates: tuple[Gate, ...] = ()
    companion_gates: tuple[Gate, ...] = ()

    @property
    def budget(self) -> int:
        return 6 * self.baseline


LEVEL_SPECS = (
    LevelSpec(
        "Echo Room",
        ("#######", "#.....#", "#.....#", "#.....#", "#######", "#######", "#######", "#######"),
        ("#######", "#.....#", "#.....#", "#.....#", "#######", "#######", "#######", "#######"),
        (3, 3),
        (3, 3),
        (3, 6),
        3,
    ),
    LevelSpec(
        "Useful Wall",
        ("#######", "#######", "#######", "#...###", "#.#.###", "###.###", "#######", "#######"),
        ("#######", "#...###", "###.###", "###.###", "#######", "#######", "#######", "#######"),
        (5, 2),
        (2, 2),
        (4, 4),
        5,
    ),
    LevelSpec(
        "Arrow Does Not Undo",
        ("#######", "#######", "#######", "#..####", "#..####", "#######", "#######", "#######"),
        ("#######", "#######", "#..####", "#...###", "#######", "#######", "#######", "#######"),
        (5, 2),
        (3, 2),
        (4, 4),
        5,
        companion_gates=(((3, 2), (3, 3)),),
    ),
    LevelSpec(
        "Borrowed Directions",
        ("#######", "#.....#", "#.###.#", "#.#...#", "#.#.#.#", "#...#.#", "#######", "#######"),
        ("#######", "#...###", "#.#...#", "#...#.#", "###...#", "#.....#", "#######", "#######"),
        (2, 3),
        (2, 3),
        (6, 2),
        27,
    ),
    LevelSpec(
        "The Command Mill",
        ("#######", "#.....#", "#.###.#", "#.#...#", "#.#.#.#", "#...#.#", "#######", "#######"),
        ("#######", "#.....#", "#.#.#.#", "#.....#", "#.#.#.#", "#.....#", "#######", "#######"),
        (6, 6),
        (5, 2),
        (6, 6),
        46,
        shepherd_gates=(((3, 6), (2, 6)),),
        companion_gates=(((4, 6), (4, 5)),),
    ),
    LevelSpec(
        "The Long Way Above",
        ("#######", "#...#.#", "#.#.#.#", "#.#...#", "#.#.#.#", "#...#.#", "#######", "#######"),
        ("#######", "#.....#", "#.###.#", "#.#...#", "#.#.#.#", "#...#.#", "#######", "#######"),
        (4, 2),
        (6, 2),
        (4, 2),
        52,
        shepherd_gates=(((5, 4), (4, 4)),),
        companion_gates=(((5, 2), (6, 2)),),
    ),
    LevelSpec(
        "Braided Escort",
        ("#######", "#.....#", "#.###.#", "#.#...#", "#.#.#.#", "#...#.#", "#######", "#######"),
        ("#######", "#...#.#", "#.#.#.#", "#.....#", "#.#.#.#", "#.#...#", "#######", "#######"),
        (4, 2),
        (5, 2),
        (6, 5),
        58,
        shepherd_gates=(((3, 2), (2, 2)), ((2, 4), (2, 5))),
        companion_gates=(((4, 5), (4, 4)),),
    ),
)


def _floor_cells(rows: tuple[str, ...]) -> frozenset[Cell]:
    assert len(rows) == 8
    assert all(len(row) == 7 for row in rows)
    return frozenset(
        (row + 1, col + 1) for row, line in enumerate(rows) for col, char in enumerate(line) if char == "."
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _cell_xy(origin: tuple[int, int], cell: Cell) -> tuple[int, int]:
    row, col = cell
    return origin[0] + CELL_SIZE * (col - 1), origin[1] + CELL_SIZE * (row - 1)


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "spec": spec,
            "shepherd_floor": _floor_cells(spec.shepherd_rows),
            "companion_floor": _floor_cells(spec.companion_rows),
        },
    )


class ShepherdPairDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ShepherdPair | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        frame[:, :] = COLOR_BACKGROUND
        if game is None:
            return frame

        self._draw_header(frame, game)
        self._draw_board(frame, LEFT_ORIGIN, game.spec.shepherd_rows, game.spec.shepherd_gates)
        self._draw_board(frame, RIGHT_ORIGIN, game.spec.companion_rows, game.spec.companion_gates)
        self._draw_pen(frame, game)
        self._draw_flash(frame, game)
        if game.visual_tracer is not None:
            self._draw_tracer(frame, game.visual_tracer)
        self._draw_character(
            frame, LEFT_ORIGIN, game.visual_shepherd, game.shepherd_offset, SHEPHERD_MASK, COLOR_SHEPHERD
        )
        self._draw_character(
            frame, RIGHT_ORIGIN, game.visual_companion, game.companion_offset, COMPANION_MASK, COLOR_COMPANION
        )
        self._draw_budget(frame, game)
        return frame

    def _draw_header(self, frame: np.ndarray, game: ShepherdPair) -> None:
        # Role silhouettes and a central arrow teach the relationship without text.
        for x, y in SHEPHERD_MASK:
            frame[4 + y, 4 + x] = COLOR_SHEPHERD
        frame[5, 12:27] = COLOR_TRACER
        frame[4:7, 25] = COLOR_TRACER
        for x, y in COMPANION_MASK:
            frame[3 + y, 31 + x] = COLOR_COMPANION
        for index in range(len(LEVEL_SPECS)):
            color = (
                COLOR_GOAL if index < game.level_index else (COLOR_FLOOR if index == game.level_index else COLOR_FRAME)
            )
            x0 = 43 + index * 3
            frame[4:7, x0 : x0 + 2] = color

    def _draw_board(
        self, frame: np.ndarray, origin: tuple[int, int], rows: tuple[str, ...], gates: tuple[Gate, ...]
    ) -> None:
        ox, oy = origin
        frame[oy - 1 : oy + 33, ox - 1] = COLOR_FRAME
        frame[oy - 1 : oy + 33, ox + 28] = COLOR_FRAME
        frame[oy - 1, ox - 1 : ox + 29] = COLOR_FRAME
        frame[oy + 32, ox - 1 : ox + 29] = COLOR_FRAME
        for row, line in enumerate(rows):
            for col, char in enumerate(line):
                color = COLOR_FLOOR if char == "." else COLOR_WALL
                x0 = ox + col * CELL_SIZE
                y0 = oy + row * CELL_SIZE
                frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = color
        for gate in gates:
            self._draw_gate(frame, origin, gate, COLOR_GATE)

    def _draw_gate(self, frame: np.ndarray, origin: tuple[int, int], gate: Gate, color: int) -> None:
        source, destination = gate
        dr = destination[0] - source[0]
        dc = destination[1] - source[1]
        right_points = ((2, 1), (3, 1), (4, 1), (5, 1), (4, 0), (4, 2))
        if (dr, dc) == (0, 1):
            anchor = _cell_xy(origin, source)
            points = right_points
        elif (dr, dc) == (0, -1):
            anchor = _cell_xy(origin, destination)
            points = tuple((7 - x, 3 - y) for x, y in right_points)
        elif (dr, dc) == (1, 0):
            anchor = _cell_xy(origin, source)
            points = tuple((3 - y, x) for x, y in right_points)
        elif (dr, dc) == (-1, 0):
            anchor = _cell_xy(origin, destination)
            points = tuple((y, 7 - x) for x, y in right_points)
        else:
            raise ValueError(f"Gate endpoints are not adjacent: {gate}")
        for x, y in points:
            frame[anchor[1] + y, anchor[0] + x] = color

    def _draw_pen(self, frame: np.ndarray, game: ShepherdPair) -> None:
        x0, y0 = _cell_xy(RIGHT_ORIGIN, game.goal_pos)
        frame[y0, x0 : x0 + 4] = COLOR_GOAL
        frame[y0 + 3, x0 : x0 + 4] = COLOR_GOAL
        frame[y0 : y0 + 4, x0] = COLOR_GOAL
        frame[y0 : y0 + 4, x0 + 3] = COLOR_GOAL
        fill_order = ((1, 1), (2, 1), (1, 2), (2, 2))
        for x, y in fill_order[: min(4, game.pen_fill)]:
            frame[y0 + y, x0 + x] = COLOR_GOAL

    def _draw_character(
        self,
        frame: np.ndarray,
        origin: tuple[int, int],
        cell: Cell,
        offset: tuple[int, int],
        mask: tuple[tuple[int, int], ...],
        color: int,
    ) -> None:
        x0, y0 = _cell_xy(origin, cell)
        x0 += offset[0]
        y0 += offset[1]
        for x, y in mask:
            px, py = x0 + x, y0 + y
            if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                frame[py, px] = color

    def _draw_tracer(self, frame: np.ndarray, direction: int) -> None:
        points = {
            ACTION_RIGHT: ((30, 31), (31, 31), (32, 31), (33, 31), (32, 30), (32, 32)),
            ACTION_LEFT: ((33, 31), (32, 31), (31, 31), (30, 31), (31, 30), (31, 32)),
            ACTION_UP: ((31, 33), (31, 32), (31, 31), (31, 30), (30, 31), (32, 31)),
            ACTION_DOWN: ((31, 30), (31, 31), (31, 32), (31, 33), (30, 32), (32, 32)),
        }[direction]
        for x, y in points:
            frame[y, x] = COLOR_TRACER

    def _draw_flash(self, frame: np.ndarray, game: ShepherdPair) -> None:
        if game.visual_flash is None:
            return
        side, kind, cell, gate = game.visual_flash
        origin = LEFT_ORIGIN if side == "shepherd" else RIGHT_ORIGIN
        if kind == "gate" and gate is not None:
            self._draw_gate(frame, origin, gate, COLOR_FLASH)
            return
        row, col = cell
        if 1 <= row <= 8 and 1 <= col <= 7:
            x0, y0 = _cell_xy(origin, cell)
            frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_FLASH
            return
        # Outside-board impacts light the corresponding enclosure edge.
        ox, oy = origin
        if row < 1:
            frame[oy - 1, ox - 1 : ox + 29] = COLOR_FLASH
        elif row > 8:
            frame[oy + 32, ox - 1 : ox + 29] = COLOR_FLASH
        elif col < 1:
            frame[oy - 1 : oy + 33, ox - 1] = COLOR_FLASH
        else:
            frame[oy - 1 : oy + 33, ox + 28] = COLOR_FLASH

    def _draw_budget(self, frame: np.ndarray, game: ShepherdPair) -> None:
        frame[55:58, 2:62] = COLOR_FRAME
        width = int(60 * game.remaining_steps / max(1, game.step_budget))
        if width > 0:
            frame[55:58, 2 : 2 + width] = COLOR_TRACER
        # The paired end caps distinguish this from a board or selectable region.
        frame[54:59, 0:2] = COLOR_SHEPHERD
        frame[54:59, 62:64] = COLOR_COMPANION


class ShepherdPair(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = ShepherdPairDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "shepherd_pair-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.shepherd_floor: frozenset[Cell] = level.get_data("shepherd_floor")
        self.companion_floor: frozenset[Cell] = level.get_data("companion_floor")
        self.shepherd_pos = self.spec.shepherd_start
        self.companion_pos = self.spec.companion_start
        self.goal_pos = self.spec.goal
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.won = False
        self.undo_history: list[tuple[Cell, Cell]] = []
        self._animation: list[dict[str, object]] = []
        self._pending_result = "complete"
        self._clear_visual_feedback()

    def _clear_visual_feedback(self) -> None:
        self.visual_shepherd = self.shepherd_pos
        self.visual_companion = self.companion_pos
        self.shepherd_offset = (0, 0)
        self.companion_offset = (0, 0)
        self.visual_tracer: int | None = None
        self.visual_flash: tuple[str, str, Cell, Gate | None] | None = None
        self.pen_fill = 0

    def _gate_on_edge(self, gates: tuple[Gate, ...], source: Cell, destination: Cell) -> Gate | None:
        for gate in gates:
            if {gate[0], gate[1]} == {source, destination}:
                return gate
        return None

    def _try_move(
        self, floor: frozenset[Cell], gates: tuple[Gate, ...], position: Cell, action: int
    ) -> tuple[Cell, bool, tuple[str, Cell, Gate | None] | None]:
        dr, dc = DELTAS[action]
        target = (position[0] + dr, position[1] + dc)
        if target not in floor:
            return position, False, ("wall", target, None)
        gate = self._gate_on_edge(gates, position, target)
        if gate is not None and gate != (position, target):
            return position, False, ("gate", target, gate)
        return target, True, None

    @staticmethod
    def _pixel_delta(action: int, distance: int) -> tuple[int, int]:
        dr, dc = DELTAS[action]
        return dc * distance, dr * distance

    def _queue_successful_move(
        self,
        action: int,
        old_shepherd: Cell,
        old_companion: Cell,
        companion_moved: bool,
        companion_obstruction: tuple[str, Cell, Gate | None] | None,
    ) -> None:
        half_s = self._pixel_delta(action, 2)
        companion_motion = self._pixel_delta(action, 2 if companion_moved else 1)
        self._animation = [
            {"visual_shepherd": old_shepherd, "visual_companion": old_companion, "shepherd_offset": half_s},
            {"visual_shepherd": self.shepherd_pos, "visual_companion": old_companion, "visual_tracer": action},
            {
                "visual_shepherd": self.shepherd_pos,
                "visual_companion": old_companion,
                "companion_offset": companion_motion,
            },
            {
                "visual_shepherd": self.shepherd_pos,
                "visual_companion": self.companion_pos,
                "visual_flash": (("companion", *companion_obstruction) if companion_obstruction is not None else None),
            },
        ]
        if self.won:
            for fill in range(1, 7):
                self._animation.append(
                    {"visual_shepherd": self.shepherd_pos, "visual_companion": self.companion_pos, "pen_fill": fill}
                )

    def _queue_blocked_move(self, action: int, obstruction: tuple[str, Cell, Gate | None]) -> None:
        kind, cell, gate = obstruction
        self._animation = [
            {
                "visual_shepherd": self.shepherd_pos,
                "visual_companion": self.companion_pos,
                "shepherd_offset": self._pixel_delta(action, 1),
            },
            {
                "visual_shepherd": self.shepherd_pos,
                "visual_companion": self.companion_pos,
                "shepherd_offset": self._pixel_delta(action, 1),
                "visual_flash": ("shepherd", kind, cell, gate),
            },
            {"visual_shepherd": self.shepherd_pos, "visual_companion": self.companion_pos},
        ]

    def _apply_animation_frame(self) -> None:
        state = self._animation.pop(0)
        self.visual_shepherd = state.get("visual_shepherd", self.shepherd_pos)
        self.visual_companion = state.get("visual_companion", self.companion_pos)
        self.shepherd_offset = state.get("shepherd_offset", (0, 0))
        self.companion_offset = state.get("companion_offset", (0, 0))
        self.visual_tracer = state.get("visual_tracer")
        self.visual_flash = state.get("visual_flash")
        self.pen_fill = int(state.get("pen_fill", 0))
        if self._animation:
            return
        if self._pending_result == "advance":
            self.next_level()
            self.complete_action()
        elif self._pending_result == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _resolve_direction(self, action: int) -> None:
        self._clear_visual_feedback()
        self.remaining_steps = max(0, self.remaining_steps - 1)
        old_shepherd = self.shepherd_pos
        old_companion = self.companion_pos
        new_shepherd, shepherd_moved, shepherd_obstruction = self._try_move(
            self.shepherd_floor, self.spec.shepherd_gates, old_shepherd, action
        )
        if not shepherd_moved:
            assert shepherd_obstruction is not None
            self._queue_blocked_move(action, shepherd_obstruction)
            self._pending_result = "lose" if self.remaining_steps == 0 else "complete"
            return

        self.undo_history.append((old_shepherd, old_companion))
        new_companion, companion_moved, companion_obstruction = self._try_move(
            self.companion_floor, self.spec.companion_gates, old_companion, action
        )
        self.shepherd_pos = new_shepherd
        self.companion_pos = new_companion
        self.won = self.companion_pos == self.goal_pos
        self._queue_successful_move(action, old_shepherd, old_companion, companion_moved, companion_obstruction)
        if self.won:
            self._pending_result = "advance"
        elif self.remaining_steps == 0:
            self._pending_result = "lose"
        else:
            self._pending_result = "complete"

    def _undo(self) -> None:
        if self.undo_history:
            self.shepherd_pos, self.companion_pos = self.undo_history.pop()
        self.won = False
        self._clear_visual_feedback()
        self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._apply_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._clear_visual_feedback()
            self.complete_action()
        elif action == ACTION_UNDO:
            self._undo()
        elif action in DELTAS:
            self._resolve_direction(action)
            self._apply_animation_frame()
        else:
            # SPACE and all in-bounds clicks intentionally have no game effect.
            self._clear_visual_feedback()
            self.complete_action()
