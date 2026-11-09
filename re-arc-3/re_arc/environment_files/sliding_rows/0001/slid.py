from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BODY_X = 8
BODY_Y = 18
BODY_W = 48
TILE = 4
LOGICAL_COLS = 12
ROW_STRIDE = 5

LEFT_PAD_X = 2
RIGHT_PAD_X = 58
PAD_W = 4
RAIL_LEFT_X = 7
RAIL_RIGHT_X = 56

BUDGET_X = 4
BUDGET_Y = 4
BUDGET_W = 56
BUDGET_H = 4

FRAME_X0 = 1
FRAME_X1 = 62
FRAME_Y0 = 16

COLOR_WHITE = 0
COLOR_BLUE = 9
COLOR_YELLOW = 11
COLOR_BLACK = 5
COLOR_DARK_GRAY = 3
COLOR_GRAY = 2
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_RED = 8
COLOR_VERY_DARK = 4
COLOR_ARROW = 1

PHASE_PLAY = "play"
PHASE_SUCCESS_WAIT = "success_wait"
PHASE_FAIL_WAIT = "fail_wait"


def _solid(size: int, color: int) -> np.ndarray:
    return np.full((size, size), np.int8(color), dtype=np.int8)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    strips: int
    budget: int
    selected_strip: int
    base_pattern: tuple[int, ...]
    offsets: tuple[int, ...]

    @property
    def frame_bottom(self) -> int:
        return BODY_Y + (ROW_STRIDE * (self.strips - 1)) + 5


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Teach The Bars",
        strips=3,
        budget=16,
        selected_strip=1,
        base_pattern=(
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_YELLOW,
        ),
        offsets=(0, 1, 11),
    ),
    LevelSpec(
        name="Five-Row Weave",
        strips=5,
        budget=32,
        selected_strip=2,
        base_pattern=(
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_GREEN,
            COLOR_GREEN,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_GREEN,
            COLOR_GREEN,
            COLOR_BLUE,
        ),
        offsets=(0, 2, 10, 3, 11),
    ),
    LevelSpec(
        name="Full Rack",
        strips=8,
        budget=72,
        selected_strip=3,
        base_pattern=(
            COLOR_BLUE,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_YELLOW,
            COLOR_GREEN,
            COLOR_RED,
            COLOR_RED,
            COLOR_GREEN,
            COLOR_BLUE,
            COLOR_YELLOW,
            COLOR_GREEN,
            COLOR_RED,
        ),
        offsets=(0, 3, 7, 10, 2, 5, 9, 11),
    ),
)


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(LEVEL_SPECS):
        sprite = Sprite(
            pixels=np.full((GRID_SIZE, GRID_SIZE), np.int8(COLOR_BLACK), dtype=np.int8),
            name=f"screen_{idx}",
            x=0,
            y=0,
            layer=0,
            collidable=False,
        )
        levels.append(
            Level(
                name=spec.name,
                grid_size=(GRID_SIZE, GRID_SIZE),
                sprites=[sprite],
                data={
                    "level_index": idx,
                    "strips": spec.strips,
                    "budget": spec.budget,
                    "selected_strip": spec.selected_strip,
                    "base_pattern": list(spec.base_pattern),
                    "offsets": list(spec.offsets),
                },
            )
        )
    return levels


class SlidingRows(ARCBaseGame):
    def __init__(self) -> None:
        self._route_score = 0
        self._phase = PHASE_PLAY
        self._level_index = 0
        self._spec = LEVEL_SPECS[0]
        self._selected_strip = 0
        self._budget_remaining = 0
        self._offsets: list[int] = []
        self._screen: Sprite | None = None

        super().__init__(
            game_id="sliding_rows-0001",
            levels=_build_levels(),
            camera=Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BLACK, COLOR_BLACK),
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 6],
        )

    def on_set_level(self, level: Level) -> None:
        self._level_index = int(level.get_data("level_index") or 0)
        self._spec = LEVEL_SPECS[self._level_index]
        self._phase = PHASE_PLAY
        self._selected_strip = int(level.get_data("selected_strip") or self._spec.selected_strip)
        self._budget_remaining = int(level.get_data("budget") or self._spec.budget)
        self._offsets = [int(v) % LOGICAL_COLS for v in (level.get_data("offsets") or list(self._spec.offsets))]
        screens = self.current_level.get_sprites_by_name(f"screen_{self._level_index}")
        self._screen = screens[0] if screens else None
        self._render()

    def _strip_y(self, strip_idx: int) -> int:
        return BODY_Y + (ROW_STRIDE * strip_idx)

    def _displayed_color(self, strip_idx: int, logical_col: int) -> int:
        offset = self._offsets[strip_idx]
        return int(self._spec.base_pattern[(logical_col - offset) % LOGICAL_COLS])

    def _aligned_columns(self) -> set[int]:
        aligned: set[int] = set()
        for col in range(LOGICAL_COLS):
            color = self._displayed_color(0, col)
            if all(self._displayed_color(strip_idx, col) == color for strip_idx in range(1, self._spec.strips)):
                aligned.add(col)
        return aligned

    def _is_solved(self) -> bool:
        return len(self._aligned_columns()) == LOGICAL_COLS

    def _frame_color(self) -> int:
        if self._phase == PHASE_SUCCESS_WAIT:
            return COLOR_GREEN
        if self._phase == PHASE_FAIL_WAIT:
            return COLOR_RED
        return COLOR_DARK_GRAY

    def _budget_fill_width(self) -> int:
        budget_cap = max(1, self._spec.budget)
        if budget_cap <= BUDGET_W:
            return max(0, min(BUDGET_W, self._budget_remaining))
        ratio = self._budget_remaining / float(budget_cap)
        return max(0, min(BUDGET_W, round(ratio * BUDGET_W)))

    def _budget_color(self) -> int:
        if self._budget_remaining <= 3:
            return COLOR_RED
        if self._budget_remaining <= 8:
            return COLOR_ORANGE
        return COLOR_GREEN

    def _draw_rect(self, frame: np.ndarray, x0: int, y0: int, width: int, height: int, color: int) -> None:
        frame[y0 : y0 + height, x0 : x0 + width] = np.int8(color)

    def _draw_chevron(self, frame: np.ndarray, x0: int, y0: int, direction: int, bg: int, fg: int) -> None:
        self._draw_rect(frame, x0, y0, PAD_W, TILE, bg)
        if direction < 0:
            pattern = ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2))
        else:
            pattern = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1))
        for dy, dx in pattern:
            frame[y0 + dy, x0 + dx] = np.int8(fg)

    def _draw_budget_bar(self, frame: np.ndarray) -> None:
        self._draw_rect(frame, BUDGET_X, BUDGET_Y, BUDGET_W, BUDGET_H, COLOR_VERY_DARK)
        filled = self._budget_fill_width()
        if filled > 0:
            self._draw_rect(frame, BUDGET_X, BUDGET_Y, filled, BUDGET_H, self._budget_color())

    def _draw_frame(self, frame: np.ndarray) -> None:
        color = self._frame_color()
        bottom = self._spec.frame_bottom
        frame[FRAME_Y0, FRAME_X0 : FRAME_X1 + 1] = np.int8(color)
        frame[bottom, FRAME_X0 : FRAME_X1 + 1] = np.int8(color)
        frame[FRAME_Y0 : bottom + 1, FRAME_X0] = np.int8(color)
        frame[FRAME_Y0 : bottom + 1, FRAME_X1] = np.int8(color)

    def _draw_separators(self, frame: np.ndarray) -> None:
        for strip_idx in range(self._spec.strips - 1):
            sep_y = self._strip_y(strip_idx) + TILE
            frame[sep_y, BODY_X : BODY_X + BODY_W] = np.int8(COLOR_GRAY)

    def _draw_rows(self, frame: np.ndarray) -> None:
        for strip_idx in range(self._spec.strips):
            y0 = self._strip_y(strip_idx)
            is_selected = strip_idx == self._selected_strip
            pad_bg = COLOR_ORANGE if is_selected else COLOR_VERY_DARK
            pad_fg = COLOR_YELLOW if is_selected else COLOR_ARROW
            self._draw_chevron(frame, LEFT_PAD_X, y0, -1, pad_bg, pad_fg)
            self._draw_chevron(frame, RIGHT_PAD_X, y0, 1, pad_bg, pad_fg)

            for col in range(LOGICAL_COLS):
                x0 = BODY_X + (TILE * col)
                color = self._displayed_color(strip_idx, col)
                self._draw_rect(frame, x0, y0, TILE, TILE, color)

    def _draw_selection(self, frame: np.ndarray) -> None:
        y0 = self._strip_y(self._selected_strip)
        y1 = y0 + TILE
        frame[y0:y1, RAIL_LEFT_X] = np.int8(COLOR_ORANGE)
        frame[y0:y1, RAIL_RIGHT_X] = np.int8(COLOR_ORANGE)
        frame[y0 - 1, RAIL_LEFT_X : RAIL_RIGHT_X + 1] = np.int8(COLOR_ORANGE)
        frame[y0 + TILE, RAIL_LEFT_X : RAIL_RIGHT_X + 1] = np.int8(COLOR_ORANGE)

    def _draw_white_columns(self, frame: np.ndarray) -> None:
        for col in self._aligned_columns():
            x0 = BODY_X + (TILE * col)
            top = self._strip_y(0)
            bottom = self._strip_y(self._spec.strips - 1) + TILE
            frame[top:bottom, x0 : x0 + TILE] = np.int8(COLOR_WHITE)

    def _render(self) -> None:
        frame = np.full((GRID_SIZE, GRID_SIZE), np.int8(COLOR_BLACK), dtype=np.int8)
        self._draw_budget_bar(frame)
        self._draw_frame(frame)
        self._draw_separators(frame)
        self._draw_rows(frame)
        self._draw_selection(frame)
        self._draw_white_columns(frame)
        if self._screen is not None:
            self._screen.pixels = frame

    def _click_strip_body(self, x: int, y: int) -> int | None:
        if not (BODY_X <= x < BODY_X + BODY_W):
            return None
        for strip_idx in range(self._spec.strips):
            y0 = self._strip_y(strip_idx)
            if y0 <= y < y0 + TILE:
                return strip_idx
        return None

    def _click_pad(self, x: int, y: int) -> tuple[int, int] | None:
        for strip_idx in range(self._spec.strips):
            y0 = self._strip_y(strip_idx)
            if not (y0 <= y < y0 + TILE):
                continue
            if LEFT_PAD_X <= x < LEFT_PAD_X + PAD_W:
                return strip_idx, -1
            if RIGHT_PAD_X <= x < RIGHT_PAD_X + PAD_W:
                return strip_idx, 1
        return None

    def _advance_success_wait(self) -> None:
        self.next_level()

    def _advance_fail_wait(self) -> None:
        self.lose()

    def _handle_click(self, x: int, y: int) -> bool:
        pad_hit = self._click_pad(x, y)
        if pad_hit is not None:
            strip_idx, direction = pad_hit
            self._selected_strip = strip_idx
            self._offsets[strip_idx] = (self._offsets[strip_idx] + direction) % LOGICAL_COLS
            return True

        body_hit = self._click_strip_body(x, y)
        if body_hit is not None and body_hit != self._selected_strip:
            self._selected_strip = body_hit
            return True

        return False

    def _handle_play_action(self, action_id: int, payload: dict[str, int]) -> bool:
        if action_id == int(GameAction.ACTION1.value):
            if self._selected_strip > 0:
                self._selected_strip -= 1
                return True
            return False
        if action_id == int(GameAction.ACTION2.value):
            if self._selected_strip < self._spec.strips - 1:
                self._selected_strip += 1
                return True
            return False
        if action_id == int(GameAction.ACTION3.value):
            self._offsets[self._selected_strip] = (self._offsets[self._selected_strip] - 1) % LOGICAL_COLS
            return True
        if action_id == int(GameAction.ACTION4.value):
            self._offsets[self._selected_strip] = (self._offsets[self._selected_strip] + 1) % LOGICAL_COLS
            return True
        if action_id == int(GameAction.ACTION6.value):
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                return self._handle_click(x, y)
        return False

    def step(self) -> None:
        action_id = int(self.action.id.value)
        payload = self.action.data if isinstance(self.action.data, dict) else {}

        if self._phase == PHASE_SUCCESS_WAIT:
            self._advance_success_wait()
            self.complete_action()
            return

        if self._phase == PHASE_FAIL_WAIT:
            self._advance_fail_wait()
            self.complete_action()
            return

        changed = self._handle_play_action(action_id, payload)
        if changed:
            self._budget_remaining -= 1
            if self._is_solved():
                self._phase = PHASE_SUCCESS_WAIT
            elif self._budget_remaining <= 0:
                self._phase = PHASE_FAIL_WAIT

        self._render()
        self.complete_action()


GAME_ID = "sliding_rows-0001"

Slid = SlidingRows
