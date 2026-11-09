from __future__ import annotations

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

DISPLAY_SIZE = 64
BOARD_SIZE = 8
CELL_SIZE = 6
BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 12

COLOR_BG = 0
COLOR_SOCKET_FILL = 1
COLOR_INACTIVE = 2
COLOR_SOCKET_FRAME = 3
COLOR_OUTLINE = 4
COLOR_FAIL = 8
COLOR_BEAM = 9
COLOR_EMITTER_HI = 10
COLOR_BUDGET = 11
COLOR_CRYSTAL = 14
COLOR_MIRROR = 15

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)

NOOP_ACTIONS = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}

LEVEL_SPECS = (
    {
        "name": "level_1",
        "budget": 6,
        "emitter": ("LEFT", 5),
        "mirrors": ((3, 5, "\\"),),
        "crystals": ((3, 1, True),),
        "blockers": (),
    },
    {
        "name": "level_2",
        "budget": 9,
        "emitter": ("BOTTOM", 1),
        "mirrors": ((1, 4, "\\"), (4, 4, "/")),
        "crystals": ((4, 6, True),),
        "blockers": (),
    },
    {
        "name": "level_3",
        "budget": 12,
        "emitter": ("LEFT", 6),
        "mirrors": ((2, 6, "\\"), (2, 3, "\\"), (6, 3, "/")),
        "crystals": ((6, 5, True),),
        "blockers": ((1, 1), (2, 1), (3, 1), (4, 4), (4, 5), (4, 6)),
    },
    {
        "name": "level_4",
        "budget": 12,
        "emitter": ("BOTTOM", 3),
        "mirrors": ((3, 5, "\\"), (1, 5, "\\"), (6, 5, "\\")),
        "crystals": ((1, 2, False), (6, 2, True)),
        "blockers": ((4, 3), (4, 4)),
    },
    {
        "name": "level_5",
        "budget": 16,
        "emitter": ("TOP", 1),
        "mirrors": ((1, 2, "/"), (5, 2, "\\"), (5, 6, "\\"), (2, 6, "/")),
        "crystals": ((2, 4, True),),
        "blockers": ((3, 3), (4, 3), (3, 4), (4, 4), (6, 5), (6, 6)),
    },
)


class BudgetDisplay(RenderableUserDisplay):
    def __init__(self, game: MirrorShotGame) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        budget = self._game._budget
        max_budget = self._game._max_budget
        for idx in range(max_budget):
            color = COLOR_BUDGET if idx < budget else COLOR_SOCKET_FILL
            x0 = 8 + idx * 3
            frame[2:6, x0 : x0 + 2] = color
        return frame


def cell_origin(cell_x: int, cell_y: int) -> tuple[int, int]:
    return BOARD_ORIGIN_X + cell_x * CELL_SIZE, BOARD_ORIGIN_Y + cell_y * CELL_SIZE


def cell_center(cell_x: int, cell_y: int) -> tuple[int, int]:
    x0, y0 = cell_origin(cell_x, cell_y)
    return x0 + 3, y0 + 3


def draw_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
    frame[y : y + h, x : x + w] = color


def draw_impact(frame: np.ndarray, px: int, py: int) -> None:
    x0 = max(0, min(DISPLAY_SIZE - 2, px - 1))
    y0 = max(0, min(DISPLAY_SIZE - 2, py - 1))
    frame[y0 : y0 + 2, x0 : x0 + 2] = COLOR_FAIL


def mirror_turn(orientation: str, direction: tuple[int, int]) -> tuple[int, int]:
    if orientation == "/":
        return {DIR_UP: DIR_RIGHT, DIR_RIGHT: DIR_UP, DIR_DOWN: DIR_LEFT, DIR_LEFT: DIR_DOWN}[direction]
    return {DIR_UP: DIR_LEFT, DIR_LEFT: DIR_UP, DIR_DOWN: DIR_RIGHT, DIR_RIGHT: DIR_DOWN}[direction]


class MirrorShotGame(ARCBaseGame):
    def __init__(self, game_id: str = "mirrorshot-0001") -> None:
        self._route_score = 0
        self._mode = "playing"
        self._budget = 0
        self._max_budget = 0
        self._mirror_order: tuple[tuple[int, int], ...] = ()
        self._mirror_state: dict[tuple[int, int], str] = {}
        self._crystals: dict[tuple[int, int], bool] = {}
        self._blockers: set[tuple[int, int]] = set()
        self._emitter = LEVEL_SPECS[0]["emitter"]
        self._beam_segments: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
        self._impact_points: list[tuple[int, int]] = []
        self._board_sprite = Sprite(
            pixels=np.full((DISPLAY_SIZE, DISPLAY_SIZE), COLOR_BG, dtype=np.int8),
            name="mirrorshot_board",
            x=0,
            y=0,
            layer=0,
            collidable=False,
        )
        interfaces = [BudgetDisplay(self)]
        camera = Camera(
            width=DISPLAY_SIZE, height=DISPLAY_SIZE, background=COLOR_BG, letter_box=COLOR_BG, interfaces=interfaces
        )
        levels = [Level(name=spec["name"], grid_size=(DISPLAY_SIZE, DISPLAY_SIZE)) for spec in LEVEL_SPECS]
        super().__init__(
            game_id=game_id, levels=levels, camera=camera, win_score=len(levels), available_actions=[1, 2, 3, 4, 5, 6]
        )

    @property
    def action(self) -> ActionInput:
        return self._action

    def on_set_level(self, _level: Level) -> None:
        current = self.current_level.get_sprite_at(0, 0, ignore_collidable=True)
        if current is not self._board_sprite:
            self._board_sprite = Sprite(
                pixels=np.full((DISPLAY_SIZE, DISPLAY_SIZE), COLOR_BG, dtype=np.int8),
                name="mirrorshot_board",
                x=0,
                y=0,
                layer=0,
                collidable=False,
            )
            self.current_level.add_sprite(self._board_sprite)
        self._load_level_state()

    def _get_valid_actions(self) -> list[ActionInput]:
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
            ActionInput(id=GameAction.ACTION5),
            ActionInput(id=GameAction.ACTION6, data={"x": 0, "y": 0}),
        ]

    def _load_level_state(self) -> None:
        spec = LEVEL_SPECS[self._current_level_index]
        self._mode = "playing"
        self._budget = spec["budget"]
        self._max_budget = spec["budget"]
        self._emitter = spec["emitter"]
        self._mirror_order = tuple((mirror[0], mirror[1]) for mirror in spec["mirrors"])
        self._mirror_state = {(mirror[0], mirror[1]): mirror[2] for mirror in spec["mirrors"]}
        self._crystals = {(crystal[0], crystal[1]): crystal[2] for crystal in spec["crystals"]}
        self._blockers = set(spec["blockers"])
        self._beam_segments = []
        self._impact_points = []
        self._render_board()

    def _render_board(self) -> None:
        frame = np.full((DISPLAY_SIZE, DISPLAY_SIZE), COLOR_BG, dtype=np.int8)

        draw_rect(
            frame,
            BOARD_ORIGIN_X - 1,
            BOARD_ORIGIN_Y - 1,
            BOARD_SIZE * CELL_SIZE + 2,
            BOARD_SIZE * CELL_SIZE + 2,
            COLOR_INACTIVE,
        )
        draw_rect(frame, BOARD_ORIGIN_X, BOARD_ORIGIN_Y, BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE, COLOR_BG)

        border_color = COLOR_INACTIVE
        if self._mode in {"level_complete", "game_complete"}:
            border_color = COLOR_CRYSTAL
        elif self._mode == "level_failed":
            border_color = COLOR_FAIL
        frame[BOARD_ORIGIN_Y - 1, BOARD_ORIGIN_X - 1 : BOARD_ORIGIN_X + BOARD_SIZE * CELL_SIZE + 1] = border_color
        frame[
            BOARD_ORIGIN_Y + BOARD_SIZE * CELL_SIZE, BOARD_ORIGIN_X - 1 : BOARD_ORIGIN_X + BOARD_SIZE * CELL_SIZE + 1
        ] = border_color
        frame[BOARD_ORIGIN_Y - 1 : BOARD_ORIGIN_Y + BOARD_SIZE * CELL_SIZE + 1, BOARD_ORIGIN_X - 1] = border_color
        frame[
            BOARD_ORIGIN_Y - 1 : BOARD_ORIGIN_Y + BOARD_SIZE * CELL_SIZE + 1, BOARD_ORIGIN_X + BOARD_SIZE * CELL_SIZE
        ] = border_color

        for blocker_x, blocker_y in self._blockers:
            self._draw_blocker(frame, blocker_x, blocker_y)
        self._draw_emitter(frame, self._emitter)
        for (crystal_x, crystal_y), active in self._crystals.items():
            self._draw_crystal(frame, crystal_x, crystal_y, active)
        for start, end, stop_at_center in self._beam_segments:
            self._draw_beam_segment(frame, start, end, stop_at_center)
        for impact_x, impact_y in self._impact_points:
            draw_impact(frame, impact_x, impact_y)
        for mirror_x, mirror_y in self._mirror_order:
            self._draw_socket(frame, mirror_x, mirror_y, self._mirror_state[(mirror_x, mirror_y)])

        self._board_sprite.pixels = frame

    def _draw_socket(self, frame: np.ndarray, cell_x: int, cell_y: int, orientation: str) -> None:
        x0, y0 = cell_origin(cell_x, cell_y)
        draw_rect(frame, x0, y0, CELL_SIZE, CELL_SIZE, COLOR_SOCKET_FRAME)
        draw_rect(frame, x0 + 1, y0 + 1, CELL_SIZE - 2, CELL_SIZE - 2, COLOR_SOCKET_FILL)
        if orientation == "/":
            for offset in range(4):
                frame[y0 + 4 - offset : y0 + 6 - offset, x0 + 1 + offset : x0 + 2 + offset] = COLOR_MIRROR
                frame[y0 + 4 - offset : y0 + 5 - offset, x0 + 2 + offset : x0 + 3 + offset] = COLOR_MIRROR
        else:
            for offset in range(4):
                frame[y0 + 1 + offset : y0 + 3 + offset, x0 + 1 + offset : x0 + 2 + offset] = COLOR_MIRROR
                frame[y0 + 1 + offset : y0 + 2 + offset, x0 + 2 + offset : x0 + 3 + offset] = COLOR_MIRROR

    def _draw_crystal(self, frame: np.ndarray, cell_x: int, cell_y: int, active: bool) -> None:
        x0, y0 = cell_origin(cell_x, cell_y)
        ring = COLOR_BUDGET if active else COLOR_INACTIVE
        points = [(2, 0, 2, ring), (1, 1, 4, ring), (0, 2, 6, ring), (1, 3, 4, ring), (2, 4, 2, ring)]
        for dx, dy, width, color in points:
            draw_rect(frame, x0 + dx, y0 + dy, width, 1, color)
        body = [(2, 1, 2), (1, 2, 4), (2, 3, 2)]
        for dx, dy, width in body:
            draw_rect(frame, x0 + dx, y0 + dy, width, 1, COLOR_CRYSTAL)

    def _draw_blocker(self, frame: np.ndarray, cell_x: int, cell_y: int) -> None:
        x0, y0 = cell_origin(cell_x, cell_y)
        draw_rect(frame, x0, y0, CELL_SIZE, CELL_SIZE, COLOR_OUTLINE)
        draw_rect(frame, x0 + 1, y0 + 1, CELL_SIZE - 2, CELL_SIZE - 2, COLOR_SOCKET_FRAME)

    def _draw_emitter(self, frame: np.ndarray, emitter: tuple[str, int]) -> None:
        side, idx = emitter
        if side == "LEFT":
            x0, y0 = BOARD_ORIGIN_X - 7, BOARD_ORIGIN_Y + idx * CELL_SIZE
            shape = (
                (2, 0, 2, 1, COLOR_OUTLINE),
                (1, 1, 4, 1, COLOR_OUTLINE),
                (0, 2, 6, 2, COLOR_OUTLINE),
                (1, 2, 4, 2, COLOR_BEAM),
                (2, 1, 2, 4, COLOR_BEAM),
                (2, 2, 2, 2, COLOR_EMITTER_HI),
            )
        elif side == "RIGHT":
            x0, y0 = BOARD_ORIGIN_X + BOARD_SIZE * CELL_SIZE + 1, BOARD_ORIGIN_Y + idx * CELL_SIZE
            shape = (
                (1, 0, 2, 1, COLOR_OUTLINE),
                (1, 1, 4, 1, COLOR_OUTLINE),
                (0, 2, 6, 2, COLOR_OUTLINE),
                (1, 2, 4, 2, COLOR_BEAM),
                (2, 1, 2, 4, COLOR_BEAM),
                (2, 2, 2, 2, COLOR_EMITTER_HI),
            )
        elif side == "TOP":
            x0, y0 = BOARD_ORIGIN_X + idx * CELL_SIZE, BOARD_ORIGIN_Y - 7
            shape = (
                (0, 2, 1, 2, COLOR_OUTLINE),
                (1, 1, 1, 4, COLOR_OUTLINE),
                (2, 0, 2, 6, COLOR_OUTLINE),
                (2, 1, 2, 4, COLOR_BEAM),
                (1, 2, 4, 2, COLOR_BEAM),
                (2, 2, 2, 2, COLOR_EMITTER_HI),
            )
        else:
            x0, y0 = BOARD_ORIGIN_X + idx * CELL_SIZE, BOARD_ORIGIN_Y + BOARD_SIZE * CELL_SIZE + 1
            shape = (
                (0, 1, 1, 2, COLOR_OUTLINE),
                (1, 1, 1, 4, COLOR_OUTLINE),
                (2, 0, 2, 6, COLOR_OUTLINE),
                (2, 1, 2, 4, COLOR_BEAM),
                (1, 2, 4, 2, COLOR_BEAM),
                (2, 2, 2, 2, COLOR_EMITTER_HI),
            )
        for dx, dy, width, height, color in shape:
            draw_rect(frame, x0 + dx, y0 + dy, width, height, color)

    def _draw_beam_segment(
        self, frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], stop_at_center: bool
    ) -> None:
        start_x, start_y = start
        end_x, end_y = end
        if start_x == end_x:
            x = start_x
            y0, y1 = sorted((start_y, end_y))
            frame[y0 : y1 + (0 if stop_at_center else 1), x - 1 : x + 1] = COLOR_BEAM
            return
        y = start_y
        x0, x1 = sorted((start_x, end_x))
        frame[y - 1 : y + 1, x0 : x1 + (0 if stop_at_center else 1)] = COLOR_BEAM

    def _socket_at_display(self, display_x: int, display_y: int) -> tuple[int, int] | None:
        for mirror_x, mirror_y in self._mirror_order:
            x0, y0 = cell_origin(mirror_x, mirror_y)
            if x0 <= display_x < x0 + CELL_SIZE and y0 <= display_y < y0 + CELL_SIZE:
                return mirror_x, mirror_y
        return None

    def _emitter_entry(self) -> tuple[tuple[int, int], tuple[int, int]]:
        side, idx = self._emitter
        if side == "LEFT":
            return (0, idx), DIR_RIGHT
        if side == "RIGHT":
            return (BOARD_SIZE - 1, idx), DIR_LEFT
        if side == "TOP":
            return (idx, 0), DIR_DOWN
        return (idx, BOARD_SIZE - 1), DIR_UP

    def _cell_edge_point(self, cell_x: int, cell_y: int, direction: tuple[int, int]) -> tuple[int, int]:
        x0, y0 = cell_origin(cell_x, cell_y)
        if direction == DIR_LEFT:
            return x0, y0 + 3
        if direction == DIR_RIGHT:
            return x0 + CELL_SIZE, y0 + 3
        if direction == DIR_UP:
            return x0 + 3, y0
        return x0 + 3, y0 + CELL_SIZE

    def _record_segment(
        self,
        start_cell: tuple[int, int],
        direction_in: tuple[int, int],
        end_cell: tuple[int, int],
        direction_out: tuple[int, int] | None,
        stop_at_center: bool,
    ) -> None:
        sx, sy = start_cell
        ex, ey = end_cell
        start_px = self._cell_edge_point(sx, sy, (-direction_in[0], -direction_in[1]))
        if stop_at_center:
            end_px = cell_center(ex, ey)
        else:
            out_dir = direction_out if direction_out is not None else direction_in
            end_px = self._cell_edge_point(ex, ey, out_dir)
        if start_px[0] != end_px[0] and start_px[1] != end_px[1]:
            center_px = cell_center(ex, ey)
            self._beam_segments.append((start_px, center_px, True))
            self._beam_segments.append((center_px, end_px, stop_at_center))
            return
        self._beam_segments.append((start_px, end_px, stop_at_center))

    def _simulate_shot(self) -> str:
        self._beam_segments = []
        self._impact_points = []

        (cell_x, cell_y), direction = self._emitter_entry()
        previous_cell = None
        previous_direction = direction
        visited: set[tuple[int, int, tuple[int, int]]] = set()

        while True:
            if not (0 <= cell_x < BOARD_SIZE and 0 <= cell_y < BOARD_SIZE):
                if previous_cell is not None:
                    self._impact_points.append(
                        self._cell_edge_point(previous_cell[0], previous_cell[1], previous_direction)
                    )
                return "wall"

            state_key = (cell_x, cell_y, direction)
            if state_key in visited:
                self._impact_points.append(cell_center(cell_x, cell_y))
                return "loop"
            visited.add(state_key)

            current_cell = (cell_x, cell_y)
            if current_cell in self._blockers:
                self._impact_points.append(cell_center(cell_x, cell_y))
                return "blocker"

            if current_cell in self._crystals:
                if previous_cell is not None:
                    self._record_segment(previous_cell, previous_direction, current_cell, None, True)
                active = self._crystals[current_cell]
                if active:
                    return "success"
                self._impact_points.append(cell_center(cell_x, cell_y))
                return "inactive_crystal"

            if current_cell in self._mirror_state:
                new_direction = mirror_turn(self._mirror_state[current_cell], direction)
                if previous_cell is None:
                    self._record_segment(current_cell, direction, current_cell, new_direction, False)
                else:
                    self._record_segment(previous_cell, previous_direction, current_cell, new_direction, False)
                previous_cell = current_cell
                previous_direction = new_direction
                cell_x += new_direction[0]
                cell_y += new_direction[1]
                direction = new_direction
                continue

            if previous_cell is not None:
                self._record_segment(previous_cell, previous_direction, current_cell, direction, False)
            previous_cell = current_cell
            previous_direction = direction
            cell_x += direction[0]
            cell_y += direction[1]

    def _toggle_socket(self, cell_x: int, cell_y: int) -> None:
        current = self._mirror_state[(cell_x, cell_y)]
        self._mirror_state[(cell_x, cell_y)] = "/" if current == "\\" else "\\"

    def _consume_budget(self) -> None:
        self._budget -= 1

    def _lose_level_if_needed(self) -> bool:
        if self._budget <= 0 and self._mode == "playing":
            self._mode = "level_failed"
            self._render_board()
            self.lose()
            return True
        return False

    def _advance_from_complete(self) -> None:
        if self._current_level_index == len(LEVEL_SPECS) - 1:
            self._mode = "game_complete"
        self.next_level()

    def step(self) -> None:
        if self._mode == "game_complete":
            self.complete_action()
            return

        if self._mode == "level_complete":
            self._advance_from_complete()
            self.complete_action()
            return

        if self._mode == "level_failed":
            self.lose()
            self.complete_action()
            return

        if self.action.id in NOOP_ACTIONS:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION6:
            display_x = int(self.action.data.get("x", 0))
            display_y = int(self.action.data.get("y", 0))
            socket = self._socket_at_display(display_x, display_y)
            if socket is not None:
                self._beam_segments = []
                self._impact_points = []
                self._toggle_socket(*socket)
                self._consume_budget()
                if not self._lose_level_if_needed():
                    self._render_board()
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION5:
            self._beam_segments = []
            self._impact_points = []
            self._consume_budget()
            outcome = self._simulate_shot()
            if outcome == "success":
                self._mode = "level_complete"
                self._render_board()
            elif not self._lose_level_if_needed():
                self._render_board()
            self.complete_action()
            return

        self.complete_action()


class Mirr(MirrorShotGame):
    pass
