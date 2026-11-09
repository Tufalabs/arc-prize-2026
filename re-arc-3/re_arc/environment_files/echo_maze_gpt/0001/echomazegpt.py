from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 12
CELL_SIZE = 4
HUD_HEIGHT = 8
BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 12
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE

COLOR_HIDDEN = 5
COLOR_FLOOR_LOCAL = 4
COLOR_FLOOR_FADE = 3
COLOR_FLOOR_STRONG = 2
COLOR_WALL_DIM = 9
COLOR_WALL_STRONG = 10
COLOR_GOAL_BORDER = 11
COLOR_GOAL_CENTER = 12
COLOR_HAZARD_OUTLINE = 13
COLOR_HAZARD_SPIKE = 8
COLOR_PULSE = 15
COLOR_PLAYER_BORDER = 10
COLOR_PLAYER_CORE = 0
COLOR_WIN_FRAME = 14
COLOR_LOSE_FRAME = 8

CELL_WALL = 0
CELL_FLOOR = 1
CELL_HAZARD = 2
CELL_GOAL = 3


@dataclass(frozen=True)
class LevelSpec:
    start: tuple[int, int]
    goal: tuple[int, int]
    action_budget: int
    pulse_budget: int
    floors: frozenset[tuple[int, int]]
    hazards: frozenset[tuple[int, int]]


def _segment_horizontal(y: int, x0: int, x1: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(x0, x1 + 1)}


def _segment_vertical(x: int, y0: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(y0, y1 + 1)}


def _build_level_specs() -> list[LevelSpec]:
    level1_floors = (
        _segment_vertical(2, 3, 5)
        | _segment_horizontal(4, 2, 4)
        | _segment_vertical(4, 4, 7)
        | _segment_horizontal(7, 4, 7)
    )
    level2_floors = (
        _segment_horizontal(1, 1, 8)
        | _segment_vertical(4, 1, 5)
        | _segment_horizontal(5, 1, 8)
        | _segment_vertical(8, 1, 9)
        | _segment_horizontal(9, 8, 10)
        | _segment_vertical(1, 5, 8)
    )
    level2_hazards = frozenset({(8, 3), (8, 4)})
    level3_floors = (
        _segment_horizontal(1, 1, 9)
        | _segment_vertical(5, 1, 4)
        | _segment_horizontal(4, 1, 10)
        | _segment_vertical(10, 4, 8)
        | _segment_horizontal(8, 1, 10)
        | _segment_vertical(2, 8, 10)
        | _segment_horizontal(10, 2, 10)
        | _segment_vertical(7, 4, 7)
        | _segment_vertical(1, 6, 8)
    )
    level3_hazards = frozenset({(7, 1), (8, 1), (9, 1), (7, 6), (7, 7)})

    return [
        LevelSpec(
            start=(2, 4),
            goal=(7, 7),
            action_budget=16,
            pulse_budget=3,
            floors=frozenset(level1_floors),
            hazards=frozenset(),
        ),
        LevelSpec(
            start=(1, 1),
            goal=(10, 9),
            action_budget=28,
            pulse_budget=3,
            floors=frozenset(level2_floors),
            hazards=level2_hazards,
        ),
        LevelSpec(
            start=(1, 1),
            goal=(10, 10),
            action_budget=42,
            pulse_budget=4,
            floors=frozenset(level3_floors),
            hazards=level3_hazards,
        ),
    ]


LEVEL_SPECS = _build_level_specs()


def _full_screen(color: int) -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), int(color), dtype=np.int8)


class EchoMazeGpt(ARCBaseGame):
    def __init__(self) -> None:
        levels = [
            Level(
                name=f"Level {idx + 1}",
                grid_size=(GRID_SIZE, GRID_SIZE),
                sprites=[
                    Sprite(_full_screen(COLOR_HIDDEN), name="screen", x=0, y=0, layer=0, visible=True, collidable=False)
                ],
                data={"level_index": idx},
            )
            for idx in range(len(LEVEL_SPECS))
        ]
        super().__init__(
            game_id="echo_maze_gpt",
            levels=levels,
            camera=Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_HIDDEN),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
        )
        self._screen: Sprite | None = None
        self._board = np.full((BOARD_SIZE, BOARD_SIZE), CELL_WALL, dtype=np.int8)
        self._reveal_timers = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._player = (0, 0)
        self._goal = (0, 0)
        self._actions_remaining = 0
        self._action_budget = 0
        self._pulses_remaining = 0
        self._pulse_budget = 0
        self._active_pulse: tuple[int, int, int] | None = None
        self._transition_state: str | None = None
        self._route_score = 0

    def on_set_level(self, level: Level) -> None:
        level_idx = int(level.get_data("level_index") or 0)
        spec = LEVEL_SPECS[level_idx]

        self._screen = self.current_level.get_sprites_by_name("screen")[0]
        self._board = np.full((BOARD_SIZE, BOARD_SIZE), CELL_WALL, dtype=np.int8)
        for x, y in spec.floors:
            self._board[y, x] = CELL_FLOOR
        for x, y in spec.hazards:
            self._board[y, x] = CELL_HAZARD
        gx, gy = spec.goal
        self._board[gy, gx] = CELL_GOAL

        self._player = spec.start
        self._goal = spec.goal
        self._action_budget = int(spec.action_budget)
        self._actions_remaining = int(spec.action_budget)
        self._pulse_budget = int(spec.pulse_budget)
        self._pulses_remaining = int(spec.pulse_budget)
        self._reveal_timers = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._active_pulse = None
        self._transition_state = None
        self._render_frame()

    def _cell_origin(self, x: int, y: int) -> tuple[int, int]:
        return BOARD_ORIGIN_X + x * CELL_SIZE, BOARD_ORIGIN_Y + y * CELL_SIZE

    def _cell_center_distance(self, ox: int, oy: int, x: int, y: int) -> float:
        dx = float(x - ox)
        dy = float(y - oy)
        return math.sqrt((dx * dx) + (dy * dy))

    def _ring_cells(self, ox: int, oy: int, age: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                dist = self._cell_center_distance(ox, oy, x, y)
                if dist <= float(age) and dist > float(age - 1):
                    cells.append((x, y))
        return cells

    def _pulse_band_cells(self) -> set[tuple[int, int]]:
        if self._active_pulse is None:
            return set()
        ox, oy, age = self._active_pulse
        return set(self._ring_cells(ox, oy, age))

    def _in_local_visibility(self, x: int, y: int) -> bool:
        px, py = self._player
        return max(abs(x - px), abs(y - py)) <= 1

    def _draw_budget(self, frame: np.ndarray, row_start: int, budget: int, remaining: int, color: int) -> None:
        for row_idx in range(2):
            y0 = row_start + row_idx * 2
            for col_idx in range(21):
                slot_idx = row_idx * 21 + col_idx
                x0 = col_idx * 3
                if slot_idx >= budget:
                    pip_color = COLOR_HIDDEN
                elif slot_idx < remaining:
                    pip_color = color
                else:
                    pip_color = COLOR_FLOOR_FADE
                frame[y0 : y0 + 2, x0 : x0 + 2] = pip_color
                frame[y0 : y0 + 2, x0 + 2] = COLOR_HIDDEN

    def _paint_cell(self, frame: np.ndarray, x: int, y: int) -> None:
        x0, y0 = self._cell_origin(x, y)
        cell = frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
        timer = int(self._reveal_timers[y, x])
        local_visible = self._in_local_visibility(x, y)
        visible = local_visible or timer > 0

        if not visible:
            cell[:, :] = COLOR_HIDDEN
            return

        tile = int(self._board[y, x])
        strong = timer >= 4
        fading = timer > 0 and not strong

        if tile == CELL_WALL:
            border = COLOR_WALL_STRONG if strong else COLOR_WALL_DIM
            center = COLOR_WALL_DIM if strong else COLOR_FLOOR_FADE
            cell[:, :] = border
            cell[1:3, 1:3] = center
            return

        if tile == CELL_HAZARD:
            cell[:, :] = np.array(
                [
                    [COLOR_HAZARD_OUTLINE, COLOR_HAZARD_SPIKE, COLOR_HAZARD_SPIKE, COLOR_HAZARD_OUTLINE],
                    [COLOR_HAZARD_SPIKE, COLOR_HAZARD_OUTLINE, COLOR_HAZARD_OUTLINE, COLOR_HAZARD_SPIKE],
                    [COLOR_HAZARD_SPIKE, COLOR_HAZARD_OUTLINE, COLOR_HAZARD_OUTLINE, COLOR_HAZARD_SPIKE],
                    [COLOR_HAZARD_OUTLINE, COLOR_HAZARD_SPIKE, COLOR_HAZARD_SPIKE, COLOR_HAZARD_OUTLINE],
                ],
                dtype=np.int8,
            )
            return

        if tile == CELL_GOAL:
            cell[:, :] = COLOR_GOAL_BORDER
            cell[1:3, 1:3] = COLOR_GOAL_CENTER
            return

        floor_color = COLOR_FLOOR_LOCAL
        if strong:
            floor_color = COLOR_FLOOR_STRONG
        elif fading:
            floor_color = COLOR_FLOOR_FADE
        cell[:, :] = floor_color

    def _draw_player(self, frame: np.ndarray) -> None:
        x0, y0 = self._cell_origin(*self._player)
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = np.array(
            [
                [COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER],
                [COLOR_PLAYER_BORDER, COLOR_PLAYER_CORE, COLOR_PLAYER_CORE, COLOR_PLAYER_BORDER],
                [COLOR_PLAYER_BORDER, COLOR_PLAYER_CORE, COLOR_PLAYER_CORE, COLOR_PLAYER_BORDER],
                [COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER, COLOR_PLAYER_BORDER],
            ],
            dtype=np.int8,
        )

    def _draw_pulse_overlay(self, frame: np.ndarray) -> None:
        for x, y in self._pulse_band_cells():
            x0, y0 = self._cell_origin(x, y)
            cell = frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
            cell[0, :] = COLOR_PULSE
            cell[3, :] = COLOR_PULSE
            cell[:, 0] = COLOR_PULSE
            cell[:, 3] = COLOR_PULSE

    def _draw_transition_frame(self, frame: np.ndarray) -> None:
        if self._transition_state is None:
            return
        color = COLOR_WIN_FRAME if self._transition_state == "WIN_FLASH" else COLOR_LOSE_FRAME
        x0 = BOARD_ORIGIN_X
        y0 = BOARD_ORIGIN_Y
        x1 = x0 + BOARD_PIXEL_SIZE - 1
        y1 = y0 + BOARD_PIXEL_SIZE - 1
        frame[y0, x0 : x1 + 1] = color
        frame[y1, x0 : x1 + 1] = color
        frame[y0 : y1 + 1, x0] = color
        frame[y0 : y1 + 1, x1] = color

    def _render_frame(self) -> None:
        if self._screen is None:
            return
        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_HIDDEN, dtype=np.int8)
        self._draw_budget(frame, 0, self._action_budget, self._actions_remaining, COLOR_GOAL_BORDER)
        self._draw_budget(frame, 4, self._pulse_budget, self._pulses_remaining, COLOR_PULSE)

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self._paint_cell(frame, x, y)

        self._draw_pulse_overlay(frame)
        self._draw_player(frame)
        self._draw_transition_frame(frame)
        self._screen.pixels = frame

    def _try_move(self, dx: int, dy: int) -> None:
        px, py = self._player
        nx = px + dx
        ny = py + dy
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
            return
        if int(self._board[ny, nx]) == CELL_WALL:
            return
        self._player = (nx, ny)

    def _resolve_normal_step(self, action_id: int) -> None:
        self._actions_remaining -= 1
        self._reveal_timers = np.where(self._reveal_timers > 0, self._reveal_timers - 1, 0).astype(np.int8)

        if self._active_pulse is not None and self._active_pulse[2] >= 6:
            self._active_pulse = None

        if action_id == int(GameAction.ACTION1.value):
            self._try_move(0, -1)
        elif action_id == int(GameAction.ACTION2.value):
            self._try_move(0, 1)
        elif action_id == int(GameAction.ACTION3.value):
            self._try_move(-1, 0)
        elif action_id == int(GameAction.ACTION4.value):
            self._try_move(1, 0)
        elif action_id == int(GameAction.ACTION5.value):
            if self._active_pulse is None and self._pulses_remaining > 0:
                px, py = self._player
                self._pulses_remaining -= 1
                self._active_pulse = (px, py, 0)

        if self._active_pulse is not None:
            ox, oy, age = self._active_pulse
            age += 1
            self._active_pulse = (ox, oy, age)
            for x, y in self._ring_cells(ox, oy, age):
                self._reveal_timers[y, x] = 6

        px, py = self._player
        tile = int(self._board[py, px])
        if tile == CELL_HAZARD:
            self._transition_state = "LOSE_FLASH"
        elif tile == CELL_GOAL:
            self._route_score += 1
            self._transition_state = "WIN_FLASH"
        elif self._actions_remaining == 0:
            self._transition_state = "LOSE_FLASH"

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

        action_id = int(self.action.id.value if hasattr(self.action.id, "value") else self.action.id)

        if self._transition_state == "WIN_FLASH":
            self._transition_state = None
            self.next_level()
            self._render_frame()
            self.complete_action()
            return

        if self._transition_state == "LOSE_FLASH":
            self._transition_state = None
            self.lose()
            self._render_frame()
            self.complete_action()
            return

        self._resolve_normal_step(action_id)
        self._render_frame()
        self.complete_action()
