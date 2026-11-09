from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "wire_the_circuit-0001"

BOARD_ORIGIN_X = 5
BOARD_ORIGIN_Y = 8
CELL_SIZE = 6
BOARD_SIZE = 9
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE
BOARD_FRAME_MIN_X = BOARD_ORIGIN_X - 1
BOARD_FRAME_MIN_Y = BOARD_ORIGIN_Y - 1
BOARD_FRAME_MAX_X = BOARD_ORIGIN_X + BOARD_PIXEL_SIZE
BOARD_FRAME_MAX_Y = BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE

BUDGET_MIN_X = 5
BUDGET_MAX_X = 58
BUDGET_MIN_Y = 1
BUDGET_MAX_Y = 4

COLOR_BG = 0
COLOR_LIGHT_GRAY = 1
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_LOAD = 10
COLOR_POWERED = 11
COLOR_PLAYER = 12
COLOR_SOLVED = 14
COLOR_CURRENT = 15

MASK_U = 1
MASK_R = 2
MASK_D = 4
MASK_L = 8

STATE_EMPTY = 0
STATE_H = 1
STATE_V = 2
STATE_UR = 3
STATE_RD = 4
STATE_DL = 5
STATE_LU = 6

GAP_STATE_TO_MASK = {
    STATE_EMPTY: 0,
    STATE_H: MASK_L | MASK_R,
    STATE_V: MASK_U | MASK_D,
    STATE_UR: MASK_U | MASK_R,
    STATE_RD: MASK_R | MASK_D,
    STATE_DL: MASK_D | MASK_L,
    STATE_LU: MASK_L | MASK_U,
}

DELTA_TO_BITS = {
    (0, -1): (MASK_U, MASK_D),
    (1, 0): (MASK_R, MASK_L),
    (0, 1): (MASK_D, MASK_U),
    (-1, 0): (MASK_L, MASK_R),
}

TYPE_SOURCE = "source"
TYPE_LOAD = "load"
TYPE_FIXED = "fixed"
TYPE_GAP = "gap"

PLAYING = "playing"
SOLVED_HOLD = "solved_hold"
FAILED_HOLD = "failed_hold"
FINAL_HOLD = "final_hold"

ALL_ACTION_IDS = [1, 2, 3, 4, 6]


@dataclass(frozen=True)
class CellSpec:
    kind: str
    mask: int = 0


@dataclass(frozen=True)
class LevelSpec:
    budget: int
    cells: dict[tuple[int, int], CellSpec]
    gaps: tuple[tuple[int, int], ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=20,
        cells={
            (4, 0): CellSpec(TYPE_SOURCE, MASK_D),
            (4, 1): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 2): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 3): CellSpec(TYPE_GAP),
            (4, 4): CellSpec(TYPE_GAP),
            (4, 5): CellSpec(TYPE_GAP),
            (4, 6): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 7): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 8): CellSpec(TYPE_LOAD, MASK_U),
        },
        gaps=((4, 3), (4, 4), (4, 5)),
    ),
    LevelSpec(
        budget=48,
        cells={
            (1, 0): CellSpec(TYPE_SOURCE, MASK_D),
            (1, 1): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (1, 2): CellSpec(TYPE_GAP),
            (2, 2): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (3, 2): CellSpec(TYPE_FIXED, MASK_L | MASK_R | MASK_D),
            (3, 3): CellSpec(TYPE_FIXED, MASK_U),
            (4, 2): CellSpec(TYPE_GAP),
            (4, 3): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 4): CellSpec(TYPE_FIXED, MASK_U | MASK_D | MASK_R),
            (5, 4): CellSpec(TYPE_FIXED, MASK_L),
            (4, 5): CellSpec(TYPE_GAP),
            (5, 5): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (6, 5): CellSpec(TYPE_GAP),
            (6, 6): CellSpec(TYPE_LOAD, MASK_U),
        },
        gaps=((1, 2), (4, 2), (4, 5), (6, 5)),
    ),
    LevelSpec(
        budget=96,
        cells={
            (2, 0): CellSpec(TYPE_SOURCE, MASK_D),
            (2, 1): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (2, 2): CellSpec(TYPE_GAP),
            (3, 2): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (4, 2): CellSpec(TYPE_GAP),
            (4, 3): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (4, 4): CellSpec(TYPE_GAP),
            (3, 4): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (2, 4): CellSpec(TYPE_GAP),
            (2, 5): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (2, 6): CellSpec(TYPE_GAP),
            (3, 6): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (4, 6): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (5, 6): CellSpec(TYPE_GAP),
            (5, 7): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (5, 8): CellSpec(TYPE_GAP),
            (4, 8): CellSpec(TYPE_LOAD, MASK_R),
            (1, 1): CellSpec(TYPE_FIXED, MASK_R | MASK_D),
            (1, 2): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (1, 3): CellSpec(TYPE_FIXED, MASK_U | MASK_R),
            (0, 3): CellSpec(TYPE_FIXED, MASK_R),
            (6, 1): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (7, 1): CellSpec(TYPE_FIXED, MASK_D | MASK_L),
            (7, 2): CellSpec(TYPE_FIXED, MASK_U),
            (6, 3): CellSpec(TYPE_FIXED, MASK_R | MASK_D),
            (7, 3): CellSpec(TYPE_FIXED, MASK_L),
            (1, 5): CellSpec(TYPE_FIXED, MASK_R),
            (0, 5): CellSpec(TYPE_FIXED, MASK_R | MASK_D),
            (0, 6): CellSpec(TYPE_FIXED, MASK_L | MASK_U),
            (3, 0): CellSpec(TYPE_FIXED, MASK_D),
            (3, 1): CellSpec(TYPE_FIXED, MASK_U),
            (6, 6): CellSpec(TYPE_FIXED, MASK_L | MASK_R),
            (7, 6): CellSpec(TYPE_FIXED, MASK_L),
            (6, 7): CellSpec(TYPE_FIXED, MASK_U | MASK_D),
            (6, 8): CellSpec(TYPE_FIXED, MASK_U),
        },
        gaps=((2, 2), (4, 2), (4, 4), (2, 4), (2, 6), (5, 6), (5, 8)),
    ),
)


class BudgetDisplay(RenderableUserDisplay):
    def __init__(self, game: WireTheCircuit) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        return frame


def _module_origin(gx: int, gy: int) -> tuple[int, int]:
    return BOARD_ORIGIN_X + gx * CELL_SIZE, BOARD_ORIGIN_Y + gy * CELL_SIZE


def _click_center(gx: int, gy: int) -> tuple[int, int]:
    x0, y0 = _module_origin(gx, gy)
    return x0 + 3, y0 + 3


def _make_level(name: str) -> Level:
    screen = Sprite(
        pixels=np.full((64, 64), COLOR_BG, dtype=np.int16),
        name=name,
        x=0,
        y=0,
        visible=True,
        collidable=False,
        tags=["screen"],
    )
    return Level(sprites=[screen], grid_size=(64, 64), data={"screen_name": name}, name=name)


class WireTheCircuit(ARCBaseGame):
    def __init__(self) -> None:
        self._levels = [_make_level(f"wire-the-circuit-{idx}") for idx in range(len(LEVEL_SPECS))]
        self._budget_display = BudgetDisplay(self)
        self._route_score = 0
        self._level_state = PLAYING
        self._remaining_budget = 0
        self._gap_states: dict[tuple[int, int], int] = {}
        self._powered_cells: set[tuple[int, int]] = set()
        self._load_reached = False
        self._screen: Sprite | None = None
        super().__init__(
            game_id="wire_the_circuit",
            levels=self._levels,
            camera=Camera(0, 0, 64, 64, COLOR_BG, COLOR_BG, [self._budget_display]),
            win_score=len(self._levels),
            available_actions=ALL_ACTION_IDS,
        )

    def on_set_level(self, level: Level) -> None:
        del level
        self._screen = self.current_level.get_sprites_by_tag("screen")[0]
        self._reset_runtime_state()
        self._render_into_screen()

    def _reset_runtime_state(self) -> None:
        spec = LEVEL_SPECS[self.level_index]
        self._level_state = PLAYING
        self._remaining_budget = spec.budget
        self._gap_states = {coord: STATE_EMPTY for coord in spec.gaps}
        self._recompute_connectivity()

    def _cell_spec(self, gx: int, gy: int) -> CellSpec | None:
        return LEVEL_SPECS[self.level_index].cells.get((gx, gy))

    def _mask_at(self, gx: int, gy: int) -> int:
        cell = self._cell_spec(gx, gy)
        if cell is None:
            return 0
        if cell.kind == TYPE_GAP:
            return GAP_STATE_TO_MASK[self._gap_states[(gx, gy)]]
        return cell.mask

    def _recompute_connectivity(self) -> None:
        source = None
        for (gx, gy), cell in LEVEL_SPECS[self.level_index].cells.items():
            if cell.kind == TYPE_SOURCE:
                source = (gx, gy)
                break
        if source is None:
            self._powered_cells = set()
            self._load_reached = False
            return

        seen = {source}
        stack = [source]
        while stack:
            gx, gy = stack.pop()
            mask = self._mask_at(gx, gy)
            if mask == 0:
                continue
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                my_bit, other_bit = DELTA_TO_BITS[(dx, dy)]
                if (mask & my_bit) == 0:
                    continue
                nx = gx + dx
                ny = gy + dy
                if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                    continue
                other_mask = self._mask_at(nx, ny)
                if other_mask == 0 or (other_mask & other_bit) == 0:
                    continue
                nxt = (nx, ny)
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)

        self._powered_cells = seen
        self._load_reached = any(
            cell.kind == TYPE_LOAD and coord in seen for coord, cell in LEVEL_SPECS[self.level_index].cells.items()
        )

    def _step_playing(self) -> None:
        action_id = int(self.action.id.value)
        changed = False

        if action_id == 5:
            self._reset_runtime_state()
            self._render_into_screen()
            self.complete_action()
            return

        if action_id == 6:
            click = self.camera.display_to_grid(self.action.data.get("x", 0), self.action.data.get("y", 0))
            if click is not None:
                gx = (click[0] - BOARD_ORIGIN_X) // CELL_SIZE
                gy = (click[1] - BOARD_ORIGIN_Y) // CELL_SIZE
                if (
                    BOARD_ORIGIN_X <= click[0] < BOARD_ORIGIN_X + BOARD_PIXEL_SIZE
                    and BOARD_ORIGIN_Y <= click[1] < BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE
                    and 0 <= gx < BOARD_SIZE
                    and 0 <= gy < BOARD_SIZE
                ):
                    module_x0, module_y0 = _module_origin(gx, gy)
                    if module_x0 <= click[0] < module_x0 + CELL_SIZE and module_y0 <= click[1] < module_y0 + CELL_SIZE:
                        cell = self._cell_spec(gx, gy)
                        if cell is not None and cell.kind == TYPE_GAP:
                            self._gap_states[(gx, gy)] = (self._gap_states[(gx, gy)] + 1) % 7
                            self._remaining_budget -= 1
                            changed = True

        if changed:
            self._recompute_connectivity()
            if self._load_reached:
                self._level_state = FINAL_HOLD if self.is_last_level() else SOLVED_HOLD
            elif self._remaining_budget == 0:
                self._level_state = FAILED_HOLD

        self._render_into_screen()
        self.complete_action()

    def step(self) -> None:
        if self._level_state == SOLVED_HOLD:
            self.next_level()
            self._render_into_screen()
            self.complete_action()
            return

        if self._level_state == FINAL_HOLD:
            self._render_into_screen()
            self.next_level()
            self.complete_action()
            return

        if self._level_state == FAILED_HOLD:
            self._render_into_screen()
            self.lose()
            self.complete_action()
            return

        self._step_playing()

    def _cell_wire_color(self, coord: tuple[int, int], cell: CellSpec) -> int:
        if self._level_state in {SOLVED_HOLD, FINAL_HOLD} and coord in self._powered_cells:
            return COLOR_SOLVED
        if coord in self._powered_cells:
            return COLOR_POWERED
        if cell.kind == TYPE_GAP:
            return COLOR_PLAYER
        return COLOR_DARK_GRAY

    def _draw_wire(self, frame: np.ndarray, gx: int, gy: int, mask: int, color: int) -> None:
        x0, y0 = _module_origin(gx, gy)
        frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = color
        if mask & MASK_U:
            frame[y0 : y0 + 3, x0 + 2 : x0 + 4] = color
        if mask & MASK_D:
            frame[y0 + 3 : y0 + 6, x0 + 2 : x0 + 4] = color
        if mask & MASK_L:
            frame[y0 + 2 : y0 + 4, x0 : x0 + 3] = color
        if mask & MASK_R:
            frame[y0 + 2 : y0 + 4, x0 + 3 : x0 + 6] = color

    def _draw_source_or_load(self, frame: np.ndarray, gx: int, gy: int, mask: int, color: int) -> None:
        x0, y0 = _module_origin(gx, gy)
        frame[y0 + 1 : y0 + 5, x0 + 1 : x0 + 5] = color
        if mask & MASK_U:
            frame[y0 : y0 + 2, x0 + 2 : x0 + 4] = color
        if mask & MASK_D:
            frame[y0 + 4 : y0 + 6, x0 + 2 : x0 + 4] = color
        if mask & MASK_L:
            frame[y0 + 2 : y0 + 4, x0 : x0 + 2] = color
        if mask & MASK_R:
            frame[y0 + 2 : y0 + 4, x0 + 4 : x0 + 6] = color

    def _draw_gap_socket(self, frame: np.ndarray, gx: int, gy: int) -> None:
        x0, y0 = _module_origin(gx, gy)
        frame[y0 + 1 : y0 + 5, x0 + 1 : x0 + 5] = COLOR_VERY_DARK
        frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = COLOR_BLACK

    def _render_into_screen(self) -> None:
        frame = np.full((64, 64), COLOR_BG, dtype=np.int16)

        progress_colors = []
        for idx in range(len(LEVEL_SPECS)):
            if idx < self.level_index:
                progress_colors.append(COLOR_SOLVED)
            elif idx == self.level_index:
                progress_colors.append(COLOR_CURRENT)
            else:
                progress_colors.append(COLOR_DARK_GRAY)
        for idx, color in enumerate(progress_colors):
            x0 = 1 + idx * 3
            frame[1:3, x0 : x0 + 2] = color

        frame[BUDGET_MIN_Y : BUDGET_MAX_Y + 1, BUDGET_MIN_X : BUDGET_MAX_X + 1] = COLOR_LIGHT_GRAY
        total_budget = LEVEL_SPECS[self.level_index].budget
        for slot in range(BUDGET_MIN_X, BUDGET_MAX_X + 1):
            idx = slot - BUDGET_MIN_X
            if idx >= total_budget:
                frame[BUDGET_MIN_Y : BUDGET_MAX_Y + 1, slot] = COLOR_BG
            elif idx < self._remaining_budget:
                frame[BUDGET_MIN_Y : BUDGET_MAX_Y + 1, slot] = COLOR_PLAYER
            else:
                frame[BUDGET_MIN_Y : BUDGET_MAX_Y + 1, slot] = COLOR_DARK_GRAY

        frame_color = COLOR_GRAY
        if self._level_state == FAILED_HOLD:
            frame_color = COLOR_RED
        elif self._level_state in {SOLVED_HOLD, FINAL_HOLD}:
            frame_color = COLOR_SOLVED
        frame[BOARD_FRAME_MIN_Y : BOARD_FRAME_MAX_Y + 1, BOARD_FRAME_MIN_X] = frame_color
        frame[BOARD_FRAME_MIN_Y : BOARD_FRAME_MAX_Y + 1, BOARD_FRAME_MAX_X] = frame_color
        frame[BOARD_FRAME_MIN_Y, BOARD_FRAME_MIN_X : BOARD_FRAME_MAX_X + 1] = frame_color
        frame[BOARD_FRAME_MAX_Y, BOARD_FRAME_MIN_X : BOARD_FRAME_MAX_X + 1] = frame_color

        for gy in range(BOARD_SIZE):
            for gx in range(BOARD_SIZE):
                cell = self._cell_spec(gx, gy)
                if cell is None:
                    continue
                coord = (gx, gy)
                if cell.kind == TYPE_GAP and self._gap_states[coord] == STATE_EMPTY:
                    self._draw_gap_socket(frame, gx, gy)
                    continue
                if cell.kind == TYPE_SOURCE:
                    color = COLOR_SOLVED if self._level_state in {SOLVED_HOLD, FINAL_HOLD} else COLOR_POWERED
                    self._draw_source_or_load(frame, gx, gy, cell.mask, color)
                    continue
                if cell.kind == TYPE_LOAD:
                    core_color = COLOR_RED if self._level_state == FAILED_HOLD else COLOR_LOAD
                    if coord in self._powered_cells and self._level_state in {SOLVED_HOLD, FINAL_HOLD}:
                        core_color = COLOR_SOLVED
                    self._draw_source_or_load(frame, gx, gy, cell.mask, core_color)
                    continue
                mask = self._mask_at(gx, gy)
                if mask:
                    self._draw_wire(frame, gx, gy, mask, self._cell_wire_color(coord, cell))
                elif cell.kind == TYPE_GAP:
                    self._draw_gap_socket(frame, gx, gy)

        if self._screen is None:
            raise RuntimeError("Screen sprite was not initialized.")
        self._screen.pixels = frame

    def _get_hidden_state(self) -> np.ndarray:
        values = [self.level_index, self._remaining_budget]
        spec = LEVEL_SPECS[self.level_index]
        for coord in spec.gaps:
            values.append(self._gap_states[coord])
        return np.array([values], dtype=np.int16)

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [ActionInput(id=GameAction.ACTION1), ActionInput(id=GameAction.ACTION2)]
        actions.extend([ActionInput(id=GameAction.ACTION3), ActionInput(id=GameAction.ACTION4)])
        actions.append(ActionInput(id=GameAction.ACTION5))
        spec = LEVEL_SPECS[self.level_index]
        for gx, gy in spec.gaps:
            x, y = _click_center(gx, gy)
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
        return actions


AGENT_CLICK_TARGETS = tuple(tuple(_click_center(gx, gy) for gx, gy in spec.gaps) for spec in LEVEL_SPECS)
TARGET_GAP_STATES = (
    {(4, 3): STATE_V, (4, 4): STATE_V, (4, 5): STATE_V},
    {(1, 2): STATE_UR, (4, 2): STATE_DL, (4, 5): STATE_UR, (6, 5): STATE_DL},
    {
        (2, 2): STATE_UR,
        (4, 2): STATE_DL,
        (4, 4): STATE_LU,
        (2, 4): STATE_RD,
        (2, 6): STATE_UR,
        (5, 6): STATE_DL,
        (5, 8): STATE_LU,
    },
)
