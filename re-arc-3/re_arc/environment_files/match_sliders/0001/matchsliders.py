from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "match_sliders-0001"

GRID = 64
SCALE = 4  # pixels per logical cell
STRIPE_H = 4  # height of each slider stripe in pixels
GAP = 1  # 1 px gap between stripes

COLOR_BG = 5  # black
COLOR_REF = 9  # blue  — fixed reference rows
COLOR_SLIDER = 8  # red   — movable row
COLOR_MID = 3  # dark gray — divider cell between the two blocks
COLOR_MATCH = 11  # yellow — all rows aligned

# The gray divider is centered on the grid (x=32) in the solved state.
GX_REF = GRID // 2 - SCALE // 2  # = 30  (divider center ≈ col 32)
LX_REF = GX_REF - 4 * SCALE  # = 14

# Clamp bounds: left block stays >= 0, right block stays < GRID
OFFSET_MIN = -LX_REF  # = -14
OFFSET_MAX = GRID - LX_REF - 9 * SCALE  # = 14

# Per-level config: (n_rows, mover_row, init_offset_cells)
#   level 1: 5 rows, middle row (2)          offset left  by 3
#   level 2: 7 rows, second row (1)          offset right by 3
#   level 3: 12 rows, 2nd from bottom (10)   offset left  by 3
_LEVELS = [(5, 2, -3 * SCALE), (7, 1, +3 * SCALE), (12, 10, -3 * SCALE)]


def _build_level() -> Level:
    floor = Sprite(
        pixels=np.full((GRID, GRID), COLOR_BG, dtype=np.int8),
        name="floor",
        x=0,
        y=0,
        layer=0,
        tags=["floor"],
        collidable=False,
    )
    return Level(grid_size=(GRID, GRID), sprites=[floor], data={})


def _draw(floor: Sprite, offset: int, n_rows: int, mover: int) -> None:
    grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)

    total_h = n_rows * STRIPE_H + (n_rows - 1) * GAP
    y0 = (GRID - total_h) // 2

    aligned = offset == 0

    for row_i in range(n_rows):
        y1 = y0 + row_i * (STRIPE_H + GAP)
        y2 = y1 + STRIPE_H

        px_off = offset if row_i == mover else 0
        fg = COLOR_MATCH if aligned else (COLOR_SLIDER if row_i == mover else COLOR_REF)

        lx = LX_REF + px_off
        gx = lx + 4 * SCALE
        rx = gx + SCALE

        grid[y1:y2, max(0, lx) : min(GRID, lx + 4 * SCALE)] = fg
        grid[y1:y2, max(0, gx) : min(GRID, gx + SCALE)] = COLOR_MID
        grid[y1:y2, max(0, rx) : min(GRID, rx + 4 * SCALE)] = fg

    floor.pixels = grid


class MatchSliders(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._offset = _LEVELS[0][2]
        self._n_rows = _LEVELS[0][0]
        self._mover = _LEVELS[0][1]
        self._floor: Sprite | None = None

        camera = Camera(width=GRID, height=GRID, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LEVELS],
            camera=camera,
            win_score=len(_LEVELS),
            available_actions=[3, 4],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        n_rows, mover, init_offset = _LEVELS[self.level_index]
        self._n_rows = n_rows
        self._mover = mover
        self._offset = init_offset
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        if self._floor:
            _draw(self._floor, self._offset, self._n_rows, self._mover)

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id == int(GameAction.ACTION4.value):
            new = self._offset + SCALE
        elif action_id == int(GameAction.ACTION3.value):
            new = self._offset - SCALE
        else:
            new = self._offset
        if OFFSET_MIN <= new <= OFFSET_MAX:
            self._offset = new

        if self._floor:
            _draw(self._floor, self._offset, self._n_rows, self._mover)

        if self._offset == 0:
            self.next_level()
            self.complete_action()
            return

        self.complete_action()
