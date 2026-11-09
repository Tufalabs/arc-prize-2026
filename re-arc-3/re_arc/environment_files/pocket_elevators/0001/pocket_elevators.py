from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
BOARD_ORIGIN = 2
TILE = 6
BOARD_CELLS = 10

COLOR_LIGHT = 1
COLOR_H2 = 2
COLOR_H1 = 3
COLOR_H0 = 4
COLOR_VOID = 5
COLOR_MAGENTA = 6
COLOR_RED = 8
COLOR_EXIT = 9
COLOR_CYAN = 10
COLOR_YELLOW = 11
COLOR_STONE = 12
COLOR_STONE_DETAIL = 13
COLOR_MINER = 14
COLOR_PURPLE = 15

HEIGHT_COLORS = {0: COLOR_H0, 1: COLOR_H1, 2: COLOR_H2}
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
RAIL_AXIS = {"H": {ACTION_LEFT, ACTION_RIGHT}, "V": {ACTION_UP, ACTION_DOWN}}
STONE_SHAPE = ((0, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 0))
SOCKET_SHAPE = ((0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0))


@dataclass(frozen=True)
class ElevatorSpec:
    name: str
    color: int
    a: tuple[int, int]
    ah: int
    b: tuple[int, int]
    bh: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    start: tuple[int, int]
    exit: tuple[int, int]
    exit_height: int
    fixed: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    elevators: tuple[ElevatorSpec, ...]
    sockets: tuple[tuple[int, int], ...]
    stones: tuple[tuple[int, int], ...]
    rails: tuple[tuple[int, int, str], ...]
    budget: int


class PocketElevatorsHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PocketElevators | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[62:64, 2:62] = COLOR_VOID
        width = int(60 * max(0, game.remaining_steps) / max(1, game.step_budget))
        if width:
            frame[62:64, 2 : 2 + width] = COLOR_YELLOW
        if game.remaining_steps <= 8:
            frame[62:64, 2 : 2 + max(0, width)] = COLOR_RED
        return frame


class PocketElevators(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = PocketElevatorsHud()
        self._board = Sprite(
            np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8),
            name="board",
            layer=-10,
            collidable=False,
            tags=["board"],
        )
        self._visual_frames: list[dict[str, object]] = []
        self._pending_win = False
        levels = [
            Level([self._board.clone()], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)
            for spec in LEVEL_SPECS
        ]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_VOID, COLOR_VOID, [self._hud])
        self._hud.game = self
        super().__init__(
            "pocket_elevators-0001",
            levels,
            camera,
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self._board = level.get_sprites_by_tag("board")[0]
        self.miner = self.spec.start
        self.stones = set(self.spec.stones)
        self.heights: dict[tuple[int, int], int] = {}
        for height, cells in self.spec.fixed:
            for cell in cells:
                self.heights[cell] = height
        self.elevator_cells: dict[tuple[int, int], str] = {}
        self.elevator_pairs: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self.elevator_colors: dict[str, int] = {}
        for elevator in self.spec.elevators:
            self.heights[elevator.a] = elevator.ah
            self.heights[elevator.b] = elevator.bh
            self.elevator_cells[elevator.a] = elevator.name
            self.elevator_cells[elevator.b] = elevator.name
            self.elevator_pairs[elevator.name] = (elevator.a, elevator.b)
            self.elevator_colors[elevator.name] = elevator.color
        self.heights[self.spec.exit] = self.spec.exit_height
        self.sockets = set(self.spec.sockets)
        self.rails = {(x, y): axis for x, y, axis in self.spec.rails}
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.exit_open = self._all_sockets_filled()
        self._visual_frames = []
        self._pending_win = False
        self._sync_visuals()

    def step(self) -> None:
        if self._visual_frames:
            self._play_visual_frame()
            return
        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        self.remaining_steps -= 1
        invalid_cell: tuple[int, int] | None = None
        action_id = int(self.action.id.value)
        if action_id in MOVE_DELTAS:
            invalid_cell = self._move(action_id)
        elif action_id == ACTION_SPACE:
            self._space()
        elif action_id == ACTION_CLICK:
            self._click()

        old_open = self.exit_open
        self.exit_open = self._all_sockets_filled()
        won = self.exit_open and self.miner == self.spec.exit
        lost = self.remaining_steps <= 0 and not won

        if invalid_cell is not None:
            self._sync_visuals(invalid_cell=invalid_cell)
            self.complete_action()
            return
        if not old_open and self.exit_open:
            self._queue_visuals([{"exit_phase": 1}, {"socket_flash": True}])
        else:
            self._sync_visuals()

        if won:
            self._pending_win = True
            if not self._visual_frames:
                self._pending_win = False
                self.next_level()
                self.complete_action()
        elif lost:
            self.lose()
            self.complete_action()
        elif not self._visual_frames:
            self.complete_action()

    def _play_visual_frame(self) -> None:
        frame = self._visual_frames.pop(0)
        self._sync_visuals(**frame)
        if self._visual_frames:
            return
        if self._pending_win:
            self._pending_win = False
            self.next_level()
            self.complete_action()
        else:
            self.complete_action()

    def _queue_visuals(self, frames: list[dict[str, object]]) -> None:
        if not frames:
            return
        first, rest = frames[0], frames[1:]
        self._visual_frames = rest
        self._sync_visuals(**first)

    def _move(self, action_id: int) -> tuple[int, int] | None:
        dx, dy = MOVE_DELTAS[action_id]
        to_cell = (self.miner[0] + dx, self.miner[1] + dy)
        if to_cell in self.stones:
            return self._push(action_id, to_cell)
        if not self._can_miner_enter(self.miner, to_cell):
            return to_cell
        self.miner = to_cell
        return None

    def _can_miner_enter(self, from_cell: tuple[int, int], to_cell: tuple[int, int]) -> bool:
        if to_cell not in self.heights:
            return False
        if to_cell == self.spec.exit and not self.exit_open:
            return False
        if to_cell in self.stones:
            return False
        return abs(self.heights[from_cell] - self.heights[to_cell]) <= 1

    def _push(self, action_id: int, stone: tuple[int, int]) -> tuple[int, int] | None:
        dx, dy = MOVE_DELTAS[action_id]
        dest = (stone[0] + dx, stone[1] + dy)
        if (
            dest not in self.heights
            or dest == self.spec.exit
            or dest in self.stones
            or dest == self.miner
            or self.heights[self.miner] != self.heights[stone]
            or self.heights[stone] != self.heights.get(dest)
            or self._is_filled_socket(stone)
            or not self._rail_allows(stone, action_id)
            or not self._rail_allows(dest, action_id)
        ):
            return dest
        self.stones.remove(stone)
        self.stones.add(dest)
        self.miner = stone
        return None

    def _rail_allows(self, cell: tuple[int, int], action_id: int) -> bool:
        axis = self.rails.get(cell)
        if axis is None:
            return True
        return action_id in RAIL_AXIS[axis]

    def _space(self) -> None:
        pair_name = self.elevator_cells.get(self.miner)
        if pair_name is None:
            self._sync_visuals()
            return
        a, b = self.elevator_pairs[pair_name]
        ah, bh = self.heights[a], self.heights[b]
        self.heights[a], self.heights[b] = bh, ah
        if ah == bh:
            self._sync_visuals()
            return
        frames: list[dict[str, object]] = []
        if abs(ah - bh) == 2:
            frames.append({"height_overrides": {a: 1, b: 1}})
        frames.append({})
        self._queue_visuals(frames)

    def _click(self) -> None:
        raw = self.camera.display_to_grid(int(self.action.data.get("x", 0)), int(self.action.data.get("y", 0)))
        if raw is None:
            self._sync_visuals()
            return
        gx = (raw[0] - BOARD_ORIGIN) // TILE
        gy = (raw[1] - BOARD_ORIGIN) // TILE
        cell = (gx, gy)
        if cell in self.rails:
            self.rails[cell] = "V" if self.rails[cell] == "H" else "H"
            self._sync_visuals(rail_flash=cell)
        else:
            self._sync_visuals()

    def _all_sockets_filled(self) -> bool:
        return all(self._is_filled_socket(cell) for cell in self.sockets)

    def _is_filled_socket(self, cell: tuple[int, int]) -> bool:
        return cell in self.sockets and cell in self.stones and self.heights.get(cell) == 0

    def _sync_visuals(
        self,
        height_overrides: dict[tuple[int, int], int] | None = None,
        socket_flash: bool = False,
        exit_phase: int = 0,
        invalid_cell: tuple[int, int] | None = None,
        rail_flash: tuple[int, int] | None = None,
    ) -> None:
        pixels = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8)
        heights = dict(self.heights)
        if height_overrides:
            heights.update(height_overrides)

        for y in range(BOARD_CELLS):
            for x in range(BOARD_CELLS):
                cell = (x, y)
                if cell not in heights:
                    continue
                self._draw_floor(pixels, cell, heights[cell])
        for cell, pair in self.elevator_cells.items():
            self._draw_elevator(pixels, cell, self.elevator_colors[pair], heights[cell])
        for cell, axis in self.rails.items():
            self._draw_rail(pixels, cell, axis, rail_flash == cell)
        for cell in self.sockets:
            self._draw_socket(pixels, cell, socket_flash and self._is_filled_socket(cell))
        self._draw_exit(pixels, self.spec.exit, self.exit_open, exit_phase)
        for stone in sorted(self.stones):
            self._draw_stone(pixels, stone)
        if socket_flash:
            for cell in self.sockets:
                if self._is_filled_socket(cell):
                    self._draw_socket(pixels, cell, flash=True)
        self._draw_miner(pixels, self.miner)
        for cell, axis in self.rails.items():
            self._draw_rail(pixels, cell, axis, rail_flash == cell)
        if invalid_cell is not None:
            self._draw_invalid(pixels, invalid_cell)
        self._board.pixels = pixels

    def _cell_rect(self, cell: tuple[int, int]) -> tuple[int, int]:
        return BOARD_ORIGIN + cell[0] * TILE, BOARD_ORIGIN + cell[1] * TILE

    def _draw_floor(self, pixels: np.ndarray, cell: tuple[int, int], height: int) -> None:
        x, y = self._cell_rect(cell)
        pixels[y : y + TILE, x : x + TILE] = HEIGHT_COLORS[height]

    def _draw_elevator(self, pixels: np.ndarray, cell: tuple[int, int], color: int, height: int) -> None:
        x, y = self._cell_rect(cell)
        pixels[y + 1 : y + 6, x + 1 : x + 6] = color
        pixels[y + 2 : y + 5, x + 2 : x + 5] = HEIGHT_COLORS[height]

    def _draw_socket(self, pixels: np.ndarray, cell: tuple[int, int], flash: bool) -> None:
        x, y = self._cell_rect(cell)
        color = COLOR_MINER if flash else COLOR_STONE
        for yy, row in enumerate(SOCKET_SHAPE):
            for xx, on in enumerate(row):
                if on:
                    pixels[y + 1 + yy, x + 1 + xx] = color

    def _draw_rail(self, pixels: np.ndarray, cell: tuple[int, int], axis: str, flash: bool) -> None:
        x, y = self._cell_rect(cell)
        color = COLOR_LIGHT if flash else COLOR_CYAN
        if axis == "H":
            pixels[y + 2 : y + 4, x + 1 : x + 6] = color
        else:
            pixels[y + 1 : y + 6, x + 2 : x + 4] = color

    def _draw_exit(self, pixels: np.ndarray, cell: tuple[int, int], open_exit: bool, phase: int) -> None:
        x, y = self._cell_rect(cell)
        pixels[y + 1 : y + 6, x + 1] = COLOR_EXIT
        pixels[y + 1 : y + 6, x + 5] = COLOR_EXIT
        pixels[y + 1, x + 2 : x + 5] = COLOR_EXIT
        pixels[y + 2, x + 1 : x + 6] = COLOR_EXIT
        if not open_exit or phase:
            rows = [y + 3, y + 4] if phase else [y + 2, y + 3, y + 4, y + 5]
            for row in rows:
                pixels[row, x + 2 : x + 5] = COLOR_RED

    def _draw_stone(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = self._cell_rect(cell)
        for yy, row in enumerate(STONE_SHAPE):
            for xx, on in enumerate(row):
                if on:
                    pixels[y + 1 + yy, x + 1 + xx] = COLOR_STONE
        pixels[y + 4, x + 2] = COLOR_STONE_DETAIL

    def _draw_miner(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = self._cell_rect(cell)
        pixels[y + 1 : y + 5, x + 2 : x + 5] = COLOR_MINER
        pixels[y + 1, x + 1] = COLOR_YELLOW

    def _draw_invalid(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        if not (0 <= cell[0] < BOARD_CELLS and 0 <= cell[1] < BOARD_CELLS):
            return
        x, y = self._cell_rect(cell)
        pixels[y : y + TILE, x : x + TILE] = COLOR_RED


def _cells(*cells: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return cells


LEVEL_SPECS = (
    LevelSpec(
        "One lift",
        (2, 6),
        (7, 4),
        2,
        ((0, _cells((2, 6), (3, 6))), (2, _cells((4, 5), (5, 5), (6, 5), (6, 4), (7, 4)))),
        (ElevatorSpec("A", COLOR_YELLOW, (4, 6), 0, (2, 4), 2),),
        (),
        (),
        (),
        48,
    ),
    LevelSpec(
        "Height blocker",
        (1, 6),
        (8, 6),
        0,
        ((0, _cells((1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (7, 6), (8, 6))), (1, _cells((4, 5), (5, 5)))),
        (ElevatorSpec("A", COLOR_YELLOW, (3, 5), 0, (6, 6), 2),),
        (),
        (),
        (),
        60,
    ),
    LevelSpec(
        "Lowering socket",
        (2, 6),
        (7, 6),
        1,
        (
            (
                1,
                _cells(
                    (2, 6),
                    (3, 6),
                    (4, 6),
                    (5, 6),
                    (6, 6),
                    (7, 6),
                    (2, 5),
                    (3, 5),
                    (4, 5),
                    (5, 5),
                    (6, 5),
                    (7, 5),
                    (3, 4),
                    (4, 4),
                ),
            ),
        ),
        (ElevatorSpec("A", COLOR_CYAN, (5, 4), 1, (3, 7), 0),),
        ((5, 4),),
        ((4, 4),),
        (),
        84,
    ),
    LevelSpec(
        "Recover route",
        (1, 6),
        (8, 5),
        2,
        (
            (2, _cells((2, 5), (3, 5), (4, 5), (5, 5), (4, 6), (5, 6), (6, 5), (8, 5))),
            (1, _cells((6, 6))),
            (0, _cells((1, 6), (6, 7), (6, 8), (6, 9), (7, 9), (7, 8), (7, 7), (8, 7))),
        ),
        (ElevatorSpec("A", COLOR_YELLOW, (2, 6), 0, (7, 5), 2), ElevatorSpec("B", COLOR_CYAN, (5, 4), 2, (7, 6), 0)),
        ((5, 4), (8, 7)),
        ((3, 5), (7, 8)),
        (),
        162,
    ),
    LevelSpec(
        "Rail pockets",
        (1, 6),
        (9, 6),
        2,
        (
            (2, _cells((2, 5), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (5, 5), (6, 5), (7, 5), (9, 6))),
            (1, _cells((7, 6))),
            (0, _cells((1, 6), (6, 6), (6, 7), (6, 8), (6, 9), (7, 9), (7, 8), (7, 7), (9, 7))),
        ),
        (ElevatorSpec("A", COLOR_YELLOW, (2, 6), 0, (8, 6), 2), ElevatorSpec("B", COLOR_CYAN, (6, 3), 2, (8, 7), 0)),
        ((6, 3), (9, 7)),
        ((3, 4), (7, 8)),
        ((4, 4, "V"), (6, 4, "H")),
        198,
    ),
    LevelSpec(
        "Two rail jobs",
        (1, 7),
        (9, 5),
        2,
        (
            (
                2,
                _cells(
                    (2, 6),
                    (2, 5),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (5, 4),
                    (6, 4),
                    (6, 3),
                    (7, 4),
                    (5, 5),
                    (6, 5),
                    (5, 6),
                    (6, 6),
                    (5, 7),
                    (6, 7),
                    (5, 8),
                    (6, 8),
                    (7, 5),
                    (9, 5),
                ),
            ),
            (1, _cells((7, 6), (8, 8), (9, 7))),
            (0, _cells((1, 7), (8, 6))),
        ),
        (
            ElevatorSpec("A", COLOR_YELLOW, (2, 7), 0, (9, 6), 2),
            ElevatorSpec("B", COLOR_CYAN, (6, 3), 2, (8, 7), 0),
            ElevatorSpec("C", COLOR_PURPLE, (7, 4), 2, (9, 8), 0),
        ),
        ((6, 3), (7, 4)),
        ((3, 4), (6, 7)),
        ((4, 4, "V"), (6, 4, "H")),
        246,
    ),
    LevelSpec(
        "Spending platform",
        (1, 7),
        (9, 5),
        2,
        (
            (
                2,
                _cells(
                    (2, 6),
                    (2, 5),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (5, 4),
                    (5, 3),
                    (6, 4),
                    (3, 5),
                    (4, 5),
                    (5, 5),
                    (5, 6),
                    (2, 8),
                    (3, 8),
                    (4, 8),
                    (5, 8),
                    (9, 5),
                ),
            ),
            (1, _cells((8, 7), (9, 7))),
            (0, _cells((1, 7), (6, 7), (7, 7), (6, 8), (7, 8))),
        ),
        (
            ElevatorSpec("A", COLOR_YELLOW, (2, 7), 0, (9, 6), 2),
            ElevatorSpec("K", COLOR_MAGENTA, (5, 7), 2, (1, 8), 0),
            ElevatorSpec("B", COLOR_CYAN, (5, 3), 2, (8, 8), 0),
            ElevatorSpec("C", COLOR_PURPLE, (6, 4), 2, (9, 8), 0),
        ),
        ((5, 3), (6, 4), (7, 7)),
        ((3, 4), (5, 6), (6, 7)),
        ((4, 4, "V"), (5, 4, "H")),
        360,
    ),
    LevelSpec(
        "Final bridge",
        (1, 7),
        (8, 4),
        2,
        (
            (
                2,
                _cells(
                    (2, 6),
                    (2, 5),
                    (2, 4),
                    (3, 4),
                    (4, 4),
                    (5, 4),
                    (5, 3),
                    (6, 4),
                    (3, 5),
                    (4, 5),
                    (5, 5),
                    (5, 6),
                    (2, 8),
                    (3, 8),
                    (4, 8),
                    (5, 8),
                    (8, 4),
                ),
            ),
            (1, _cells((8, 8), (9, 7), (9, 5))),
            (0, _cells((1, 7), (6, 7), (7, 7), (8, 7))),
        ),
        (
            ElevatorSpec("A", COLOR_YELLOW, (2, 7), 0, (9, 6), 2),
            ElevatorSpec("K", COLOR_MAGENTA, (5, 7), 2, (8, 5), 0),
            ElevatorSpec("B", COLOR_CYAN, (5, 3), 2, (7, 8), 0),
            ElevatorSpec("C", COLOR_PURPLE, (6, 4), 2, (9, 8), 0),
        ),
        ((5, 3), (6, 4), (8, 7)),
        ((3, 4), (5, 6), (6, 7)),
        ((4, 4, "V"), (5, 4, "H"), (7, 7, "V")),
        378,
    ),
)


__all__ = ["PocketElevators"]
