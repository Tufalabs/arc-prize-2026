from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 6
CELL_SIZE = 8
BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 16

COLOR_BG = 0
COLOR_GRID = 1
COLOR_SPENT = 2
COLOR_STAMP_SEP = 3
COLOR_DARK = 4
COLOR_MAGENTA = 6
COLOR_MAGENTA_HINT = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_BLUE_HINT = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_PRESS = 5
ACTION_CLICK = 6

RESET_RECT = (2, 7, 9, 14)
MAGENTA_PAD_RECT = (13, 7, 20, 14)
LEFT_KNOB_RECT = (24, 7, 31, 14)
PREVIEW_RECT = (32, 7, 39, 14)
RIGHT_KNOB_RECT = (40, 7, 47, 14)
BLUE_PAD_RECT = (51, 7, 58, 14)

EMPTY = 0
MAGENTA = 1
BLUE = 2


class Orientation(StrEnum):
    H = "H"
    V = "V"


class Phase(StrEnum):
    PLAYING = "PLAYING"
    WON = "WON"
    LOST = "LOST"
    COMPLETE = "COMPLETE"


class LevelSpec(TypedDict):
    name: str
    budget: int
    start_center: tuple[int, int]
    start_orientation: Orientation
    start_ink: int
    targets_any: frozenset[tuple[int, int]]
    targets_exact: dict[tuple[int, int], int]
    forbidden: frozenset[tuple[int, int]]
    inks_enabled: tuple[int, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    {
        "name": "Level 1",
        "budget": 14,
        "start_center": (1, 4),
        "start_orientation": Orientation.H,
        "start_ink": MAGENTA,
        "targets_any": frozenset({(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)}),
        "targets_exact": {},
        "forbidden": frozenset(),
        "inks_enabled": (MAGENTA,),
    },
    {
        "name": "Level 2",
        "budget": 18,
        "start_center": (1, 4),
        "start_orientation": Orientation.H,
        "start_ink": MAGENTA,
        "targets_any": frozenset({(1, 1), (2, 1), (3, 1), (2, 2), (2, 3)}),
        "targets_exact": {},
        "forbidden": frozenset({(0, 1), (4, 1), (2, 0), (1, 2), (3, 2), (1, 3), (3, 3), (2, 4)}),
        "inks_enabled": (MAGENTA,),
    },
    {
        "name": "Level 3",
        "budget": 20,
        "start_center": (4, 1),
        "start_orientation": Orientation.H,
        "start_ink": MAGENTA,
        "targets_any": frozenset(),
        "targets_exact": {(1, 3): MAGENTA, (3, 3): MAGENTA, (2, 2): BLUE, (2, 3): BLUE, (2, 4): BLUE},
        "forbidden": frozenset({(0, 3), (4, 3), (2, 1), (2, 5)}),
        "inks_enabled": (MAGENTA, BLUE),
    },
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _inside_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


class StampPress(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._canvas_sprite: Sprite | None = None
        self._phase = Phase.PLAYING
        self._remaining_actions = 0
        self._stamp_center = (0, 0)
        self._stamp_orientation = Orientation.H
        self._current_ink = MAGENTA
        self._printed = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._route_score = 0

        levels = [self._build_level(index, spec) for index, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG)
        super().__init__(
            game_id="stamp_press-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PRESS, ACTION_CLICK],
            seed=seed,
        )

    def _build_level(self, index: int, spec: LevelSpec) -> Level:
        canvas = Sprite(
            _solid(GRID_SIZE, GRID_SIZE, COLOR_BG),
            name="canvas",
            x=0,
            y=0,
            layer=0,
            visible=True,
            collidable=False,
            tags=["canvas"],
        )
        return Level(
            name=str(spec["name"]), grid_size=(GRID_SIZE, GRID_SIZE), sprites=[canvas], data={"level_index": index}
        )

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        spec = LEVEL_SPECS[level_index]
        self._phase = Phase.PLAYING
        self._remaining_actions = int(spec["budget"])
        self._stamp_center = spec["start_center"]
        self._stamp_orientation = spec["start_orientation"]
        self._current_ink = int(spec["start_ink"])
        self._printed = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self._canvas_sprite = self.current_level.get_sprites_by_name("canvas")[0]
        self._render()

    def level_reset(self) -> None:
        super().level_reset()
        self._route_score = 0

    def full_reset(self) -> None:
        super().full_reset()
        self._route_score = 0

    def _spec(self) -> LevelSpec:
        return LEVEL_SPECS[self.level_index]

    def _footprint(
        self, center: tuple[int, int] | None = None, orientation: Orientation | None = None
    ) -> list[tuple[int, int]]:
        cx, cy = center or self._stamp_center
        facing = orientation or self._stamp_orientation
        if facing == Orientation.H:
            return [(cx - 1, cy), (cx, cy), (cx + 1, cy)]
        return [(cx, cy - 1), (cx, cy), (cx, cy + 1)]

    def _footprint_inside(self, center: tuple[int, int], orientation: Orientation) -> bool:
        return all(0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE for x, y in self._footprint(center, orientation))

    def _toggle_orientation(self) -> Orientation:
        return Orientation.V if self._stamp_orientation == Orientation.H else Orientation.H

    def _board_rect(self, gx: int, gy: int) -> tuple[int, int, int, int]:
        x0 = BOARD_ORIGIN_X + gx * CELL_SIZE
        y0 = BOARD_ORIGIN_Y + gy * CELL_SIZE
        return x0, y0, x0 + CELL_SIZE - 1, y0 + CELL_SIZE - 1

    def _cell_center(self, gx: int, gy: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self._board_rect(gx, gy)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def _win_matches(self) -> bool:
        spec = self._spec()
        for x, y in spec["forbidden"]:
            if int(self._printed[y, x]) != EMPTY:
                return False

        targets_exact = spec["targets_exact"]
        if targets_exact:
            for (x, y), ink in targets_exact.items():
                if int(self._printed[y, x]) != ink:
                    return False
            return True

        for x, y in spec["targets_any"]:
            if int(self._printed[y, x]) == EMPTY:
                return False
        return True

    def _consume_budget(self) -> None:
        self._remaining_actions -= 1
        if self._remaining_actions <= 0 and not self._win_matches():
            self._phase = Phase.LOST
            self.lose()

    def _apply_move(self, dx: int, dy: int) -> None:
        cx, cy = self._stamp_center
        target = (cx + dx, cy + dy)
        if self._footprint_inside(target, self._stamp_orientation):
            self._stamp_center = target

    def _apply_rotation(self) -> None:
        rotated = self._toggle_orientation()
        if self._footprint_inside(self._stamp_center, rotated):
            self._stamp_orientation = rotated

    def _apply_press(self) -> None:
        for x, y in self._footprint():
            self._printed[y, x] = self._current_ink

    def _advance_after_phase_frame(self) -> None:
        if self._phase == Phase.WON:
            self.next_level()
            self._route_score += 1
            return
        if self._phase == Phase.LOST:
            self.lose()
            return
        if self._phase == Phase.COMPLETE:
            self.full_reset()

    def _apply_click(self, x: int, y: int) -> bool:
        spec = self._spec()
        if _inside_rect(x, y, RESET_RECT):
            self.level_reset()
            return False
        if _inside_rect(x, y, LEFT_KNOB_RECT) or _inside_rect(x, y, RIGHT_KNOB_RECT):
            self._apply_rotation()
            return True
        if MAGENTA in spec["inks_enabled"] and _inside_rect(x, y, MAGENTA_PAD_RECT):
            self._current_ink = MAGENTA
            return True
        if BLUE in spec["inks_enabled"] and _inside_rect(x, y, BLUE_PAD_RECT):
            self._current_ink = BLUE
            return True
        return True

    def step(self) -> None:
        if self._phase != Phase.PLAYING:
            self._advance_after_phase_frame()
            self._render()
            self.complete_action()
            return

        action_id_raw = getattr(self.action.id, "value", self.action.id)
        action_id = action_id_raw if isinstance(action_id_raw, int) else GameAction.RESET.value
        should_consume = True
        if action_id == GameAction.RESET.value:
            should_consume = False
        elif action_id == ACTION_UP:
            self._apply_move(0, -1)
        elif action_id == ACTION_DOWN:
            self._apply_move(0, 1)
        elif action_id == ACTION_LEFT:
            self._apply_move(-1, 0)
        elif action_id == ACTION_RIGHT:
            self._apply_move(1, 0)
        elif action_id == ACTION_PRESS:
            self._apply_press()
        elif action_id == ACTION_CLICK:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            should_consume = self._apply_click(x, y)

        if should_consume:
            self._consume_budget()

        if self._phase == Phase.PLAYING and self._win_matches():
            if self.is_last_level():
                self._phase = Phase.COMPLETE
                self.next_level()
            else:
                self._phase = Phase.WON

        self._render()
        self.complete_action()

    def _draw_rect(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        frame[y0 : y1 + 1, x0 : x1 + 1] = color

    def _draw_frame(
        self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int, thickness: int = 1
    ) -> None:
        frame[y0 : y0 + thickness, x0 : x1 + 1] = color
        frame[y1 - thickness + 1 : y1 + 1, x0 : x1 + 1] = color
        frame[y0 : y1 + 1, x0 : x0 + thickness] = color
        frame[y0 : y1 + 1, x1 - thickness + 1 : x1 + 1] = color

    def _draw_target_hint(self, frame: np.ndarray, gx: int, gy: int, color: int) -> None:
        x0, y0, x1, y1 = self._board_rect(gx, gy)
        self._draw_frame(frame, x0 + 1, y0 + 1, x1 - 1, y1 - 1, color, thickness=1)
        frame[y0 + 2 : y1 - 1, x0 + 2 : x1 - 1] = COLOR_BG

    def _draw_forbidden(self, frame: np.ndarray, gx: int, gy: int) -> None:
        x0, y0, x1, y1 = self._board_rect(gx, gy)
        self._draw_rect(frame, x0 + 1, y0 + 1, x1 - 1, y1 - 1, COLOR_DARK)
        self._draw_frame(frame, x0 + 1, y0 + 1, x1 - 1, y1 - 1, COLOR_MAROON, thickness=1)
        for i in range(6):
            frame[y0 + 1 + i, x0 + 1 + i] = COLOR_RED
            frame[y0 + 6 - i, x0 + 1 + i] = COLOR_RED

    def _draw_ink(self, frame: np.ndarray, gx: int, gy: int, ink: int) -> None:
        if ink == EMPTY:
            return
        color = COLOR_MAGENTA if ink == MAGENTA else COLOR_BLUE
        x0, y0, _x1, _y1 = self._board_rect(gx, gy)
        self._draw_rect(frame, x0 + 1, y0 + 1, x0 + 6, y0 + 6, color)

    def _draw_board(self, frame: np.ndarray) -> None:
        self._draw_rect(
            frame, BOARD_ORIGIN_X - 1, BOARD_ORIGIN_Y - 1, BOARD_ORIGIN_X + 48, BOARD_ORIGIN_Y + 48, COLOR_DARK
        )
        self._draw_rect(frame, BOARD_ORIGIN_X, BOARD_ORIGIN_Y, BOARD_ORIGIN_X + 47, BOARD_ORIGIN_Y + 47, COLOR_BG)

        spec = self._spec()
        for gy in range(BOARD_SIZE):
            for gx in range(BOARD_SIZE):
                x0, y0, x1, y1 = self._board_rect(gx, gy)
                self._draw_frame(frame, x0, y0, x1, y1, COLOR_GRID, thickness=1)
                if (gx, gy) in spec["targets_any"]:
                    self._draw_target_hint(frame, gx, gy, COLOR_YELLOW)
                if (gx, gy) in spec["targets_exact"]:
                    hint = COLOR_MAGENTA_HINT if spec["targets_exact"][(gx, gy)] == MAGENTA else COLOR_BLUE_HINT
                    self._draw_target_hint(frame, gx, gy, hint)
                self._draw_ink(frame, gx, gy, int(self._printed[gy, gx]))
                if (gx, gy) in spec["forbidden"]:
                    self._draw_forbidden(frame, gx, gy)

    def _draw_budget_bar(self, frame: np.ndarray) -> None:
        budget = int(self._spec()["budget"])
        x_start = 2
        y0 = 1
        segment_width = 4 if budget <= 14 else 3
        gap = 0 if budget > 18 else 1
        for idx in range(budget):
            sx = x_start + idx * (segment_width + gap)
            color = COLOR_SPENT
            if idx < self._remaining_actions:
                color = COLOR_ORANGE if self._remaining_actions <= 3 else COLOR_GREEN
            self._draw_rect(frame, sx, y0, sx + segment_width - 1, y0 + 3, color)

    def _draw_knob(self, frame: np.ndarray, rect: tuple[int, int, int, int], clockwise: bool) -> None:
        x0, y0, x1, y1 = rect
        self._draw_rect(frame, x0, y0, x1, y1, COLOR_YELLOW)
        self._draw_frame(frame, x0, y0, x1, y1, COLOR_DARK, thickness=1)
        cy = (y0 + y1) // 2
        if clockwise:
            frame[cy, x0 + 2 : x1 - 1] = COLOR_DARK
            frame[cy - 1 : cy + 2, x1 - 2] = COLOR_DARK
            frame[y0 + 2, x0 + 2] = COLOR_DARK
            frame[y0 + 3, x0 + 1] = COLOR_DARK
        else:
            frame[cy, x0 + 1 : x1 - 2] = COLOR_DARK
            frame[cy - 1 : cy + 2, x0 + 2] = COLOR_DARK
            frame[y0 + 2, x1 - 2] = COLOR_DARK
            frame[y0 + 3, x1 - 1] = COLOR_DARK

    def _draw_preview(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = PREVIEW_RECT
        self._draw_rect(frame, x0, y0, x1, y1, COLOR_BG)
        self._draw_frame(frame, x0, y0, x1, y1, COLOR_DARK, thickness=1)
        ink_color = COLOR_MAGENTA if self._current_ink == MAGENTA else COLOR_BLUE
        if self._stamp_orientation == Orientation.H:
            self._draw_rect(frame, x0 + 1, y0 + 2, x1 - 1, y0 + 4, COLOR_DARK)
            frame[y0 + 3, x0 + 2 : x1 - 1] = ink_color
        else:
            self._draw_rect(frame, x0 + 2, y0 + 1, x0 + 4, y1 - 1, COLOR_DARK)
            frame[y0 + 2 : y1 - 1, x0 + 3] = ink_color

    def _draw_pad(self, frame: np.ndarray, rect: tuple[int, int, int, int], color: int, selected: bool) -> None:
        x0, y0, x1, y1 = rect
        self._draw_rect(frame, x0, y0, x1, y1, color)
        outline = COLOR_DARK if selected else COLOR_GRID
        self._draw_frame(frame, x0, y0, x1, y1, outline, thickness=1)

    def _draw_reset(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = RESET_RECT
        self._draw_rect(frame, x0, y0, x1, y1, COLOR_ORANGE)
        self._draw_frame(frame, x0, y0, x1, y1, COLOR_DARK, thickness=1)
        self._draw_rect(frame, x0 + 2, y0 + 2, x1 - 2, y1 - 2, COLOR_DARK)

    def _draw_stamp_overlay(self, frame: np.ndarray) -> None:
        cells = self._footprint()
        xs = [gx for gx, _gy in cells]
        ys = [gy for _gx, gy in cells]
        x0 = BOARD_ORIGIN_X + min(xs) * CELL_SIZE
        y0 = BOARD_ORIGIN_Y + min(ys) * CELL_SIZE
        x1 = BOARD_ORIGIN_X + (max(xs) + 1) * CELL_SIZE - 1
        y1 = BOARD_ORIGIN_Y + (max(ys) + 1) * CELL_SIZE - 1
        self._draw_frame(frame, x0, y0, x1, y1, COLOR_DARK, thickness=1)
        ink_color = COLOR_MAGENTA if self._current_ink == MAGENTA else COLOR_BLUE
        if self._stamp_orientation == Orientation.H:
            for gx, _gy in cells[:-1]:
                x_sep = BOARD_ORIGIN_X + (gx + 1) * CELL_SIZE - 1
                frame[y0 + 1 : y1, x_sep] = COLOR_STAMP_SEP
            for gx, gy in cells:
                cx, cy = self._cell_center(gx, gy)
                frame[cy, cx - 2 : cx + 3] = ink_color
        else:
            for _gx, gy in cells[:-1]:
                y_sep = BOARD_ORIGIN_Y + (gy + 1) * CELL_SIZE - 1
                frame[y_sep, x0 + 1 : x1] = COLOR_STAMP_SEP
            for gx, gy in cells:
                cx, cy = self._cell_center(gx, gy)
                frame[cy - 2 : cy + 3, cx] = ink_color

    def _draw_status_overlay(self, frame: np.ndarray) -> None:
        if self._phase == Phase.PLAYING:
            return
        accent = COLOR_RED if self._phase == Phase.LOST else COLOR_GREEN
        self._draw_frame(frame, 0, 0, GRID_SIZE - 1, GRID_SIZE - 1, accent, thickness=2)
        self._draw_frame(
            frame, BOARD_ORIGIN_X - 2, BOARD_ORIGIN_Y - 2, BOARD_ORIGIN_X + 49, BOARD_ORIGIN_Y + 49, accent, thickness=2
        )

    def _render(self) -> None:
        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8)
        frame[0:16, :] = COLOR_GRID

        self._draw_budget_bar(frame)
        self._draw_reset(frame)
        self._draw_knob(frame, LEFT_KNOB_RECT, clockwise=False)
        self._draw_preview(frame)
        self._draw_knob(frame, RIGHT_KNOB_RECT, clockwise=True)

        if MAGENTA in self._spec()["inks_enabled"]:
            self._draw_pad(frame, MAGENTA_PAD_RECT, COLOR_MAGENTA, self._current_ink == MAGENTA)
        if BLUE in self._spec()["inks_enabled"]:
            self._draw_pad(frame, BLUE_PAD_RECT, COLOR_BLUE, self._current_ink == BLUE)

        self._draw_board(frame)
        if self._phase in {Phase.PLAYING, Phase.WON}:
            self._draw_stamp_overlay(frame)
        self._draw_status_overlay(frame)

        if self._canvas_sprite is not None:
            self._canvas_sprite.pixels = frame
