from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite
from arcengine.enums import GameAction

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

DIRECTIONS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

BOARD_LEFT = 7
BOARD_TOP = 3
CELL_SIZE = 5

WHITE = 0
FLOOR = 1
BLOCKED = 2
WALL = 4
BACKGROUND = 5
PLAYER = 6
BUCKET = 7
INVALID = 8
PIT_DEPTH = 9
PIT_RIM = 10
SAND_LIGHT = 11
SAND = 12
SAND_SHADOW = 13
EXIT = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    facing: int
    baseline: int

    @property
    def budget(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    LevelSpec(
        "First Crossing",
        (
            "##########",
            "...OE#####",
            "s#########",
            "P#########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ACTION_UP,
        8,
    ),
    LevelSpec(
        "Staging Bay",
        (
            "##########",
            "###.######",
            "Ps..sO.OE#",
            "##..######",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ACTION_RIGHT,
        20,
    ),
    LevelSpec(
        "Handoff",
        (
            "##########",
            "######.OE#",
            "######O###",
            "######.###",
            "##..sO..##",
            "##s.######",
            "##P#######",
            "##########",
            "##########",
        ),
        ACTION_UP,
        27,
    ),
    LevelSpec(
        "Two-Cell Crossing",
        (
            "##########",
            "######.OOE",
            "######.###",
            "######.###",
            "#...sO..##",
            "#s..######",
            "#P########",
            "##########",
            "##########",
        ),
        ACTION_UP,
        33,
    ),
    LevelSpec(
        "Rear Wall",
        (
            "##########",
            "##########",
            "####.OOO.E",
            "####.#####",
            "####ss.###",
            "#s####.###",
            "#P.O...###",
            "##########",
            "##########",
        ),
        ACTION_UP,
        58,
    ),
    LevelSpec(
        "Moving the Bridge",
        (
            "##########",
            "##########",
            "####.OOOOE",
            "####.#####",
            "#ss#....##",
            "#..###..s#",
            "#s.OOO.###",
            "#P########",
            "##########",
        ),
        ACTION_UP,
        72,
    ),
    LevelSpec(
        "Elbow Exchange",
        (
            "##########",
            "EOOOOO..##",
            "######..##",
            "###OOO.s##",
            "###O######",
            "###..#####",
            "#s#..s.#s#",
            "#.O.s..O.#",
            "####P#####",
        ),
        ACTION_UP,
        98,
    ),
)


def _cell_rect(cell: tuple[int, int]) -> tuple[slice, slice]:
    x, y = cell
    left = BOARD_LEFT + (x - 1) * CELL_SIZE
    top = BOARD_TOP + (y - 1) * CELL_SIZE
    return slice(top, top + CELL_SIZE), slice(left, left + CELL_SIZE)


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return BOARD_LEFT + (x - 1) * CELL_SIZE, BOARD_TOP + (y - 1) * CELL_SIZE


class SandFerryView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: SandFerry | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[:] = BACKGROUND
        frame[BOARD_TOP : BOARD_TOP + 45, BOARD_LEFT : BOARD_LEFT + 50] = WALL

        for cell in game.ordinary_floor:
            ys, xs = _cell_rect(cell)
            frame[ys, xs] = FLOOR
        for cell in game.pit_cells:
            self._draw_pit(frame, cell, cell in game.filled_pits)
        self._draw_exit(frame, game.exit_coordinate)

        hidden_destination = None
        source_echo = None
        if game.animation_kind in {"scoop", "deposit"} and game.animation_stage < 2:
            if game.animation_kind == "deposit":
                hidden_destination = game.animation_target
            else:
                source_echo = game.animation_target

        for cell in game.floor_sand:
            if cell != hidden_destination:
                self._draw_sand(frame, cell)
        if hidden_destination in game.filled_pits:
            self._draw_pit(frame, hidden_destination, False)

        if source_echo is not None:
            self._draw_contracting_sand(frame, source_echo, game.animation_stage)
        if game.animation_kind in {"scoop", "deposit"} and game.animation_stage == 1:
            self._draw_connector(frame, game.player_position, game.animation_target)

        if game.feedback_event == "invalid_space" and game.animation_stage == 0:
            ys, xs = _cell_rect(game._ahead())
            frame[ys, xs] = INVALID

        bucket_full = game.bucket_full
        if game.animation_kind == "scoop" and game.animation_stage < 2:
            bucket_full = False
        elif game.animation_kind == "deposit" and game.animation_stage < 2:
            bucket_full = True
        self._draw_player(frame, game.player_position, game.facing, bucket_full)

        if game.feedback_event == "blocked_move":
            self._draw_bump(frame, game.player_position, game.facing)
        if game.feedback_event == "invalid_space" and game.animation_stage == 1:
            self._flash_player_bucket(frame, game.player_position, game.facing)
        if game.feedback_event == "win":
            self._draw_win_pulse(frame, game.animation_stage)

        self._draw_hud(frame, bucket_full, game.remaining_steps, game.step_budget)
        return frame

    def _draw_pit(self, frame: np.ndarray, cell: tuple[int, int], filled: bool) -> None:
        x, y = _cell_origin(cell)
        frame[y : y + 5, x : x + 5] = PIT_RIM
        if filled:
            frame[y + 1 : y + 4, x : x + 5] = SAND
            frame[y + 4, x + 1 : x + 4] = SAND_SHADOW
        else:
            frame[y + 1 : y + 5, x : x + 5] = PIT_DEPTH
            frame[y + 1, x + 2] = PIT_RIM

    def _draw_sand(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _cell_origin(cell)
        frame[y + 1, x + 2] = SAND_LIGHT
        frame[y + 2, x + 1 : x + 4] = SAND
        frame[y + 3, x : x + 5] = SAND
        frame[y + 4, x + 1 : x + 4] = SAND_SHADOW

    def _draw_contracting_sand(self, frame: np.ndarray, cell: tuple[int, int], stage: int) -> None:
        x, y = _cell_origin(cell)
        if stage == 0:
            frame[y + 1 : y + 4, x + 1 : x + 4] = SAND
            frame[y + 3, x + 1 : x + 4] = SAND_SHADOW
        else:
            frame[y + 2, x + 2] = SAND_LIGHT
            frame[y + 3, x + 2] = SAND

    def _draw_exit(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _cell_origin(cell)
        frame[y, x + 1 : x + 4] = EXIT
        frame[y + 1 : y + 5, x] = EXIT
        frame[y + 1 : y + 5, x + 4] = EXIT
        frame[y + 1, x + 2] = WHITE

    def _draw_player(self, frame: np.ndarray, cell: tuple[int, int], facing: int, full: bool) -> None:
        x, y = _cell_origin(cell)
        frame[y + 1 : y + 4, x + 1 : x + 4] = PLAYER
        frame[y + 1, x + 2] = WHITE
        color = SAND if full else WHITE
        self._draw_bucket_attachment(frame, x, y, facing, BUCKET, color)

    def _draw_bucket_attachment(self, frame: np.ndarray, x: int, y: int, facing: int, rim: int, center: int) -> None:
        if facing == ACTION_UP:
            frame[y, x + 1 : x + 4] = rim
            frame[y, x + 2] = center
        elif facing == ACTION_DOWN:
            frame[y + 4, x + 1 : x + 4] = rim
            frame[y + 4, x + 2] = center
        elif facing == ACTION_LEFT:
            frame[y + 1 : y + 4, x] = rim
            frame[y + 2, x] = center
        else:
            frame[y + 1 : y + 4, x + 4] = rim
            frame[y + 2, x + 4] = center

    def _draw_connector(self, frame: np.ndarray, player: tuple[int, int], target: tuple[int, int]) -> None:
        px, py = _cell_origin(player)
        tx, ty = _cell_origin(target)
        pcx, pcy = px + 2, py + 2
        tcx, tcy = tx + 2, ty + 2
        if pcx == tcx:
            lo, hi = sorted((pcy, tcy))
            frame[lo + 2 : hi, pcx : pcx + 2] = SAND
        else:
            lo, hi = sorted((pcx, tcx))
            frame[pcy : pcy + 2, lo + 2 : hi] = SAND

    def _draw_bump(self, frame: np.ndarray, cell: tuple[int, int], facing: int) -> None:
        x, y = _cell_origin(cell)
        if facing == ACTION_UP:
            frame[y - 1, x + 1 : x + 4] = BLOCKED
        elif facing == ACTION_DOWN:
            frame[y + 5, x + 1 : x + 4] = BLOCKED
        elif facing == ACTION_LEFT:
            frame[y + 1 : y + 4, x - 1] = BLOCKED
        else:
            frame[y + 1 : y + 4, x + 5] = BLOCKED

    def _flash_player_bucket(self, frame: np.ndarray, cell: tuple[int, int], facing: int) -> None:
        x, y = _cell_origin(cell)
        self._draw_bucket_attachment(frame, x, y, facing, INVALID, INVALID)

    def _draw_win_pulse(self, frame: np.ndarray, stage: int) -> None:
        thickness = 1 if stage % 2 == 0 else 2
        frame[0:thickness, :] = EXIT
        frame[51 : 51 + thickness, :] = EXIT
        frame[:52, 0:thickness] = EXIT
        frame[:52, 64 - thickness : 64] = EXIT

    def _draw_hud(self, frame: np.ndarray, full: bool, remaining: int, budget: int) -> None:
        frame[52:64, :] = BACKGROUND
        frame[54:63, 3:12] = BUCKET
        frame[56:61, 5:10] = SAND if full else WHITE
        frame[61:63, 18:61] = WALL
        width = int(43 * max(0, remaining) / max(1, budget))
        if width:
            frame[61:63, 18 : 18 + width] = EXIT if remaining * 4 > budget else INVALID


class SandFerry(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = SandFerryView()
        levels = [
            Level(
                sprites=[Sprite([[-1]], name="logical_board", visible=False, collidable=False)],
                grid_size=(64, 64),
                data={"spec": spec},
                name=spec.name,
            )
            for spec in LEVEL_SPECS
        ]
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self._view])
        super().__init__(
            "sand_ferry-0001",
            levels,
            camera,
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed=seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.ordinary_floor: set[tuple[int, int]] = set()
        self.pit_cells: set[tuple[int, int]] = set()
        self.floor_sand: set[tuple[int, int]] = set()
        self.exit_coordinate = (0, 0)
        self.player_position = (0, 0)
        for y, row in enumerate(self.spec.rows, start=1):
            if len(row) != 10:
                raise ValueError("Sand Ferry rows must be exactly 10 cells wide")
            for x, symbol in enumerate(row, start=1):
                cell = (x, y)
                if symbol in ".sPE":
                    self.ordinary_floor.add(cell)
                if symbol == "O":
                    self.pit_cells.add(cell)
                elif symbol == "s":
                    self.floor_sand.add(cell)
                elif symbol == "P":
                    self.player_position = cell
                elif symbol == "E":
                    self.exit_coordinate = cell
        if len(self.spec.rows) != 9:
            raise ValueError("Sand Ferry levels must be exactly 9 cells tall")
        self.initial_sand_count = len(self.floor_sand)
        self.filled_pits: set[tuple[int, int]] = set()
        self.facing = self.spec.facing
        self.bucket_full = False
        self.feedback_event: str | None = None
        self.animation_kind: str | None = None
        self.animation_stage = 0
        self.animation_last_stage = 0
        self.animation_target = (0, 0)
        self._pending_loss = False
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self._assert_invariants()

    def _ahead(self) -> tuple[int, int]:
        dx, dy = DIRECTIONS[self.facing]
        return self.player_position[0] + dx, self.player_position[1] + dy

    def _traversable(self, cell: tuple[int, int]) -> bool:
        return (cell in self.ordinary_floor and cell not in self.floor_sand) or cell in self.filled_pits

    def _start_animation(self, kind: str, target: tuple[int, int], last_stage: int) -> None:
        self.animation_kind = kind
        self.animation_target = target
        self.animation_stage = 0
        self.animation_last_stage = last_stage

    def _finish_or_advance_animation(self) -> None:
        self.animation_stage += 1
        if self.animation_stage < self.animation_last_stage:
            return
        if self.feedback_event == "win":
            self.next_level()
        elif self._pending_loss:
            self.lose()
        self.complete_action()

    def _resolve_move(self, action_id: int) -> None:
        self.facing = action_id
        target = self._ahead()
        if self._traversable(target):
            self.player_position = target
        else:
            self.feedback_event = "blocked_move"

    def _resolve_space(self) -> None:
        target = self._ahead()
        if not self.bucket_full and target in self.floor_sand:
            self.floor_sand.remove(target)
            self.bucket_full = True
            self.feedback_event = "scoop"
            self._start_animation("scoop", target, 2)
        elif not self.bucket_full and target in self.filled_pits:
            self.filled_pits.remove(target)
            self.bucket_full = True
            self.feedback_event = "scoop"
            self._start_animation("scoop", target, 2)
        elif self.bucket_full and target in self.ordinary_floor and target not in self.floor_sand:
            self.floor_sand.add(target)
            self.bucket_full = False
            self.feedback_event = "deposit"
            self._start_animation("deposit", target, 2)
        elif self.bucket_full and target in self.pit_cells and target not in self.filled_pits:
            self.filled_pits.add(target)
            self.bucket_full = False
            self.feedback_event = "deposit"
            self._start_animation("deposit", target, 2)
        else:
            self.feedback_event = "invalid_space"
            self._start_animation("invalid", target, 1)

    def _assert_invariants(self) -> None:
        assert self._traversable(self.player_position)
        assert self.player_position not in self.floor_sand
        assert not (self.player_position in self.pit_cells and self.player_position not in self.filled_pits)
        assert self.floor_sand <= self.ordinary_floor
        assert self.filled_pits <= self.pit_cells
        assert self.floor_sand.isdisjoint(self.filled_pits)
        assert len(self.floor_sand) + len(self.filled_pits) + int(self.bucket_full) == self.initial_sand_count
        assert type(self.bucket_full) is bool

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        if self.animation_kind is not None and self.animation_stage < self.animation_last_stage:
            self._finish_or_advance_animation()
            return
        self.animation_kind = None

        self.feedback_event = None
        self.animation_stage = 0
        self.animation_target = (0, 0)
        action_id = int(self.action.id.value)
        self.remaining_steps -= 1

        if action_id in DIRECTIONS:
            self._resolve_move(action_id)
        elif action_id == ACTION_SPACE:
            self._resolve_space()
        # Click is deliberately a silent no-op.

        self._assert_invariants()
        if self.player_position == self.exit_coordinate:
            self.feedback_event = "win"
            self._start_animation("win", self.exit_coordinate, 3)
            return

        self._pending_loss = self.remaining_steps <= 0
        if self.animation_kind is None:
            if self._pending_loss:
                self.lose()
            self.complete_action()
