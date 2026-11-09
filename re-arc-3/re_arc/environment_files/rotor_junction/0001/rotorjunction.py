from __future__ import annotations

from typing import Literal

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

BOARD_SIZE = 10
CELL_SIZE = 5
BOARD_ORIGIN_X = 7
BOARD_ORIGIN_Y = 12
SCREEN_SIZE = 64
BACKGROUND_COLOR = 0
FRAME_COLOR = 1
SPENT_PIP_COLOR = 3
WALL_COLOR = 4
FAIL_COLOR = 8
LANE_COLOR = 9
GOAL_A_COLOR = 11
PUCK_A_COLOR = 12
GOAL_B_COLOR = 7
PUCK_B_COLOR = 6
REMAINING_PIP_COLOR = 14

Heading = Literal["N", "E", "S", "W"]
Axis = Literal["H", "V"]
CellKind = Literal["EMPTY", "WALL", "LANE_H", "LANE_V", "ROTOR", "START_A", "GOAL_A", "START_B", "GOAL_B"]
AckState = Literal["none", "win", "fail"]

DIR_TO_DELTA: dict[Heading, tuple[int, int]] = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
ROTATE_CW: dict[Heading, Heading] = {"N": "E", "E": "S", "S": "W", "W": "N"}


class PuckSpec:
    __slots__ = ("heading", "puck_id", "start")

    def __init__(self, puck_id: str, start: tuple[int, int], heading: Heading) -> None:
        self.puck_id = puck_id
        self.start = start
        self.heading = heading


class PuckState:
    __slots__ = ("docked", "heading", "puck_id", "x", "y")

    def __init__(self, puck_id: str, x: int, y: int, heading: Heading, docked: bool = False) -> None:
        self.puck_id = puck_id
        self.x = x
        self.y = y
        self.heading = heading
        self.docked = docked

    def as_tuple(self) -> tuple[str, int, int, Heading, bool]:
        return (self.puck_id, self.x, self.y, self.heading, self.docked)


class LevelSpec:
    __slots__ = ("budget", "goals", "lane_h", "lane_v", "rotors", "starts", "walls")

    def __init__(
        self,
        *,
        budget: int,
        starts: tuple[PuckSpec, ...],
        rotors: tuple[tuple[int, int, Heading], ...],
        lane_h: tuple[tuple[int, int], ...],
        lane_v: tuple[tuple[int, int], ...],
        walls: tuple[tuple[int, int], ...],
        goals: tuple[tuple[str, tuple[int, int]], ...],
    ) -> None:
        self.budget = budget
        self.starts = starts
        self.rotors = rotors
        self.lane_h = lane_h
        self.lane_v = lane_v
        self.walls = walls
        self.goals = goals


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=16,
        starts=(PuckSpec("A", (4, 8), "N"),),
        rotors=((4, 5, "N"), (7, 5, "E")),
        lane_h=((5, 5), (6, 5)),
        lane_v=((4, 6), (4, 7), (7, 6), (7, 7)),
        walls=(),
        goals=(("A", (7, 8)),),
    ),
    LevelSpec(
        budget=24,
        starts=(PuckSpec("A", (2, 7), "N"),),
        rotors=((2, 5, "N"), (7, 5, "E"), (7, 8, "S")),
        lane_h=((3, 5), (4, 5), (5, 5), (6, 5), (5, 8), (6, 8)),
        lane_v=((2, 6), (7, 6), (7, 7)),
        walls=(
            (1, 4),
            (2, 4),
            (3, 4),
            (8, 4),
            (8, 5),
            (8, 6),
            (9, 4),
            (9, 5),
            (9, 6),
            (6, 9),
            (7, 9),
            (8, 9),
            (4, 7),
            (5, 7),
        ),
        goals=(("A", (4, 8)),),
    ),
    LevelSpec(
        budget=30,
        starts=(PuckSpec("A", (1, 8), "N"), PuckSpec("B", (8, 1), "S")),
        rotors=((1, 6, "N"), (4, 6, "E"), (8, 3, "S"), (5, 3, "W")),
        lane_h=((2, 6), (3, 6), (6, 3), (7, 3)),
        lane_v=((1, 7), (4, 7), (8, 2), (5, 2)),
        walls=((0, 5), (1, 5), (2, 5), (5, 6), (5, 7), (6, 6), (7, 4), (8, 4), (9, 4), (3, 2), (4, 2), (4, 3)),
        goals=(("A", (4, 8)), ("B", (5, 1))),
    ),
)


def build_level_board(spec: LevelSpec) -> dict[tuple[int, int], CellKind]:
    board: dict[tuple[int, int], CellKind] = {}
    for x, y in spec.walls:
        board[(x, y)] = "WALL"
    for x, y in spec.lane_h:
        board[(x, y)] = "LANE_H"
    for x, y in spec.lane_v:
        board[(x, y)] = "LANE_V"
    for x, y, _ in spec.rotors:
        board[(x, y)] = "ROTOR"
    for start in spec.starts:
        board[start.start] = "START_A" if start.puck_id == "A" else "START_B"
    for puck_id, pos in spec.goals:
        board[pos] = "GOAL_A" if puck_id == "A" else "GOAL_B"
    return board


LEVEL_BOARDS: tuple[dict[tuple[int, int], CellKind], ...] = tuple(build_level_board(spec) for spec in LEVEL_SPECS)


def level_state_from_spec(level_index: int) -> tuple[list[PuckState], dict[tuple[int, int], Heading], int]:
    spec = LEVEL_SPECS[level_index]
    pucks = [
        PuckState(puck_id=start.puck_id, x=start.start[0], y=start.start[1], heading=start.heading)
        for start in spec.starts
    ]
    rotors = {(x, y): heading for x, y, heading in spec.rotors}
    return pucks, rotors, spec.budget


def _start_axis(spec: LevelSpec, pos: tuple[int, int]) -> Axis:
    for start in spec.starts:
        if start.start == pos:
            return "H" if start.heading in {"E", "W"} else "V"
    raise KeyError(pos)


def cell_axis(level_index: int, pos: tuple[int, int], kind: CellKind) -> Axis | None:
    if kind == "LANE_H":
        return "H"
    if kind == "LANE_V":
        return "V"
    if kind in {"START_A", "START_B"}:
        return _start_axis(LEVEL_SPECS[level_index], pos)
    return None


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def is_goal_for_puck(kind: CellKind, puck_id: str) -> bool:
    return (kind == "GOAL_A" and puck_id == "A") or (kind == "GOAL_B" and puck_id == "B")


def can_enter(level_index: int, pos: tuple[int, int], heading: Heading, puck_id: str) -> bool:
    if not in_bounds(pos[0], pos[1]):
        return False
    kind = LEVEL_BOARDS[level_index].get(pos, "EMPTY")
    if kind in {"EMPTY", "WALL"}:
        return False
    if kind == "ROTOR":
        return True
    if is_goal_for_puck(kind, puck_id):
        return True
    if kind in {"GOAL_A", "GOAL_B"}:
        return False
    axis = cell_axis(level_index, pos, kind)
    if axis == "H":
        return heading in {"E", "W"}
    if axis == "V":
        return heading in {"N", "S"}
    return False


def simulate_step(
    *,
    level_index: int,
    pucks: list[PuckState],
    rotors: dict[tuple[int, int], Heading],
    clicked_rotor: tuple[int, int] | None,
) -> tuple[list[PuckState], frozenset[tuple[int, int]], bool]:
    next_rotors = dict(rotors)
    if clicked_rotor in next_rotors:
        next_rotors[clicked_rotor] = ROTATE_CW[next_rotors[clicked_rotor]]

    flashes: set[tuple[int, int]] = set()
    moved: list[PuckState] = []
    board = LEVEL_BOARDS[level_index]
    invalid_route = False

    for puck in pucks:
        if puck.docked:
            moved.append(PuckState(puck_id=puck.puck_id, x=puck.x, y=puck.y, heading=puck.heading, docked=puck.docked))
            continue

        pos = (puck.x, puck.y)
        kind = board.get(pos, "EMPTY")
        if kind == "ROTOR":
            heading = next_rotors[pos]
        else:
            heading = puck.heading
        dx, dy = DIR_TO_DELTA[heading]
        target = (puck.x + dx, puck.y + dy)

        if kind == "ROTOR":
            if not can_enter(level_index, target, heading, puck.puck_id):
                flashes.add(pos)
                moved.append(PuckState(puck_id=puck.puck_id, x=puck.x, y=puck.y, heading=puck.heading, docked=False))
                continue
        else:
            if not can_enter(level_index, target, heading, puck.puck_id):
                invalid_route = True
                moved.append(PuckState(puck_id=puck.puck_id, x=puck.x, y=puck.y, heading=puck.heading, docked=False))
                continue

        target_kind = board.get(target, "EMPTY")
        docked = is_goal_for_puck(target_kind, puck.puck_id)
        moved.append(PuckState(puck_id=puck.puck_id, x=target[0], y=target[1], heading=heading, docked=docked))

    return moved, frozenset(flashes), invalid_route


class BudgetDisplay(RenderableUserDisplay):
    def __init__(self, game: RotorJunction) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        remaining = self._game.remaining_budget
        for index in range(30):
            row = index // 15
            col = index % 15
            x0 = 2 + 4 * col
            y0 = 1 + 4 * row
            color = SPENT_PIP_COLOR
            if index < remaining:
                color = PUCK_A_COLOR if remaining <= 3 else REMAINING_PIP_COLOR
            frame[y0 : y0 + 3, x0 : x0 + 3] = color
        return frame


class RotorJunction(ARCBaseGame):
    def __init__(self) -> None:
        self._route_score = 0
        self._ack_state: AckState = "none"
        self.remaining_budget = LEVEL_SPECS[0].budget
        self._rotor_flash_positions: frozenset[tuple[int, int]] = frozenset()
        self._board_sprite_tag = "rotor-junction-board"
        self._board_sprite: Sprite | None = None
        self._pucks: list[PuckState] = []
        self._rotors: dict[tuple[int, int], Heading] = {}
        self._budget_display = BudgetDisplay(self)
        levels = [
            Level(
                sprites=[
                    Sprite(
                        np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype=np.int16), x=0, y=0, tags=[self._board_sprite_tag]
                    )
                ],
                grid_size=(SCREEN_SIZE, SCREEN_SIZE),
                data={"level_index": idx},
                name=f"Rotor Junction {idx + 1}",
            )
            for idx in range(len(LEVEL_SPECS))
        ]
        camera = Camera(0, 0, SCREEN_SIZE, SCREEN_SIZE, BACKGROUND_COLOR, BACKGROUND_COLOR, [self._budget_display])
        super().__init__(
            game_id="rotor_junction",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5, 6],
            win_score=len(LEVEL_SPECS),
        )

    def on_set_level(self, level: Level) -> None:
        self._ack_state = "none"
        self._rotor_flash_positions = frozenset()
        self._route_score = int(level.get_data("level_index") or 0)
        self._pucks, self._rotors, self.remaining_budget = level_state_from_spec(self.level_index)
        board_sprite = level.get_sprites_by_tag(self._board_sprite_tag)[0]
        self._board_sprite = board_sprite
        self._render_board()

    def _cell_to_pixel(self, x: int, y: int) -> tuple[int, int]:
        return BOARD_ORIGIN_X + CELL_SIZE * x, BOARD_ORIGIN_Y + CELL_SIZE * y

    def _draw_lane_cell(self, canvas: np.ndarray, x: int, y: int, axis: Axis) -> None:
        px, py = self._cell_to_pixel(x, y)
        if axis == "H":
            canvas[py + 1 : py + 4, px : px + 5] = LANE_COLOR
        else:
            canvas[py : py + 5, px + 1 : px + 4] = LANE_COLOR

    def _draw_goal_cell(self, canvas: np.ndarray, x: int, y: int, puck_id: str) -> None:
        px, py = self._cell_to_pixel(x, y)
        ring_color = GOAL_A_COLOR if puck_id == "A" else GOAL_B_COLOR
        canvas[py : py + 5, px : px + 5] = ring_color
        canvas[py + 1 : py + 4, px + 1 : px + 4] = BACKGROUND_COLOR

    def _draw_rotor_cell(self, canvas: np.ndarray, pos: tuple[int, int], heading: Heading, flashing: bool) -> None:
        px, py = self._cell_to_pixel(*pos)
        hub_color = FAIL_COLOR if flashing else WALL_COLOR
        arm_color = FAIL_COLOR if flashing else GOAL_A_COLOR

        canvas[py + 1 : py + 4, px + 1 : px + 4] = hub_color
        for dx, dy in ((2, 0), (0, 2), (4, 2), (2, 4)):
            canvas[py + dy, px + dx] = SPENT_PIP_COLOR

        if heading == "N":
            canvas[py : py + 2, px + 2] = arm_color
        elif heading == "E":
            canvas[py + 2, px + 3 : px + 5] = arm_color
        elif heading == "S":
            canvas[py + 3 : py + 5, px + 2] = arm_color
        else:
            canvas[py + 2, px : px + 2] = arm_color

    def _draw_wall_cell(self, canvas: np.ndarray, x: int, y: int) -> None:
        px, py = self._cell_to_pixel(x, y)
        canvas[py : py + 5, px : px + 5] = WALL_COLOR
        canvas[py, px : px + 5] = SPENT_PIP_COLOR
        canvas[py : py + 5, px] = SPENT_PIP_COLOR

    def _draw_pucks(self, canvas: np.ndarray) -> None:
        by_pos: dict[tuple[int, int], list[PuckState]] = {}
        for puck in self._pucks:
            by_pos.setdefault((puck.x, puck.y), []).append(puck)

        for pos, occupants in by_pos.items():
            px, py = self._cell_to_pixel(*pos)
            if len(occupants) == 2:
                for puck in occupants:
                    color = PUCK_A_COLOR if puck.puck_id == "A" else PUCK_B_COLOR
                    if puck.puck_id == "A":
                        canvas[py + 1 : py + 3, px + 1 : px + 3] = color
                    else:
                        canvas[py + 2 : py + 4, px + 2 : px + 4] = color
                continue

            puck = occupants[0]
            color = PUCK_A_COLOR if puck.puck_id == "A" else PUCK_B_COLOR
            canvas[py + 1 : py + 4, px + 1 : px + 4] = color

    def _draw_border(self, canvas: np.ndarray, color: int) -> None:
        canvas[0, :] = color
        canvas[-1, :] = color
        canvas[:, 0] = color
        canvas[:, -1] = color

    def _render_board(self) -> None:
        if self._board_sprite is None:
            return

        canvas = np.zeros((SCREEN_SIZE, SCREEN_SIZE), dtype=np.int16)
        left = BOARD_ORIGIN_X - 1
        top = BOARD_ORIGIN_Y - 1
        right = BOARD_ORIGIN_X + CELL_SIZE * BOARD_SIZE
        bottom = BOARD_ORIGIN_Y + CELL_SIZE * BOARD_SIZE
        canvas[top, left : right + 1] = FRAME_COLOR
        canvas[bottom, left : right + 1] = FRAME_COLOR
        canvas[top : bottom + 1, left] = FRAME_COLOR
        canvas[top : bottom + 1, right] = FRAME_COLOR

        board = LEVEL_BOARDS[self.level_index]
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                kind = board.get((x, y), "EMPTY")
                if kind == "WALL":
                    self._draw_wall_cell(canvas, x, y)
                elif kind in {"LANE_H", "START_A", "START_B"} and cell_axis(self.level_index, (x, y), kind) == "H":
                    self._draw_lane_cell(canvas, x, y, "H")
                elif kind in {"LANE_V", "START_A", "START_B"} and cell_axis(self.level_index, (x, y), kind) == "V":
                    self._draw_lane_cell(canvas, x, y, "V")
                elif kind == "GOAL_A":
                    self._draw_goal_cell(canvas, x, y, "A")
                elif kind == "GOAL_B":
                    self._draw_goal_cell(canvas, x, y, "B")

        for pos, heading in self._rotors.items():
            self._draw_rotor_cell(canvas, pos, heading, pos in self._rotor_flash_positions)

        self._draw_pucks(canvas)

        state_name = str(getattr(getattr(self, "_state", None), "name", getattr(self, "_state", "")))
        if self._ack_state == "win" or state_name == "WIN":
            self._draw_border(canvas, REMAINING_PIP_COLOR)
        elif self._ack_state == "fail":
            self._draw_border(canvas, FAIL_COLOR)

        self._board_sprite.pixels = canvas

    def _clicked_rotor(self) -> tuple[int, int] | None:
        if self.action.id != GameAction.ACTION6:
            return None
        click_x = int(self.action.data.get("x", -1))
        click_y = int(self.action.data.get("y", -1))
        if not (0 <= click_x < SCREEN_SIZE and 0 <= click_y < SCREEN_SIZE):
            return None
        if click_x < BOARD_ORIGIN_X or click_y < BOARD_ORIGIN_Y:
            return None
        board_x = (click_x - BOARD_ORIGIN_X) // CELL_SIZE
        board_y = (click_y - BOARD_ORIGIN_Y) // CELL_SIZE
        if not in_bounds(board_x, board_y):
            return None
        pos = (board_x, board_y)
        if LEVEL_BOARDS[self.level_index].get(pos) != "ROTOR":
            return None
        return pos

    def _all_docked(self) -> bool:
        return all(puck.docked for puck in self._pucks)

    def _advance_from_overlay(self) -> None:
        if self._ack_state == "win":
            self._ack_state = "none"
            self.next_level()
            self.complete_action()
            return

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        if self._ack_state != "none":
            self._advance_from_overlay()
            return

        self.remaining_budget = max(0, self.remaining_budget - 1)
        clicked_rotor = self._clicked_rotor()
        self._pucks, self._rotor_flash_positions, invalid_route = simulate_step(
            level_index=self.level_index, pucks=self._pucks, rotors=self._rotors, clicked_rotor=clicked_rotor
        )
        if clicked_rotor in self._rotors:
            self._rotors[clicked_rotor] = ROTATE_CW[self._rotors[clicked_rotor]]

        if self._all_docked():
            if self.is_last_level():
                self._render_board()
                self.next_level()
                self.complete_action()
                return
            self._ack_state = "win"
            self._render_board()
            self.complete_action()
            return

        if invalid_route or self.remaining_budget == 0:
            self._ack_state = "fail"
            self._render_board()
            self.lose()
            self.complete_action()
            return

        self._render_board()
        self.complete_action()

    def _get_hidden_state(self) -> np.ndarray:
        state = np.zeros(64, dtype=np.int16)
        state[0] = self.remaining_budget
        state[1] = self.level_index
        state[2] = {"none": 0, "win": 1, "fail": 2}[self._ack_state]
        offset = 3
        for pos in sorted(self._rotors):
            state[offset] = pos[0]
            state[offset + 1] = pos[1]
            state[offset + 2] = {"N": 0, "E": 1, "S": 2, "W": 3}[self._rotors[pos]]
            offset += 3
        for puck in self._pucks:
            state[offset] = ord(puck.puck_id)
            state[offset + 1] = puck.x
            state[offset + 2] = puck.y
            state[offset + 3] = {"N": 0, "E": 1, "S": 2, "W": 3}[puck.heading]
            state[offset + 4] = int(puck.docked)
            offset += 5
        return state

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [ActionInput(id=GameAction.ACTION5)]
        for x, y, _ in LEVEL_SPECS[self.level_index].rotors:
            px, py = self._cell_to_pixel(x, y)
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": px + 2, "y": py + 2}))
        return actions
