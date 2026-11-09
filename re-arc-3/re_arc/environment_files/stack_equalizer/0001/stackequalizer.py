from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "stack_equalizer-0001"

GRID = 64
PLAT_W = 12  # platform/block width in pixels
PLAT_H = 4  # platform base height in pixels
BTN_W = 4  # button width in pixels
BTN_H = 4  # button height in pixels
BTN_GAP = 1  # gap between platform base bottom and button top

# x positions (symmetric around x=32)
LEFT_X = 10
RIGHT_X = 42
LEFT_BTN_X = LEFT_X + (PLAT_W - BTN_W) // 2  # = 14
RIGHT_BTN_X = RIGHT_X + (PLAT_W - BTN_W) // 2  # = 46

# y positions (fixed across all levels)
PLAT_BASE_TOP = 52  # platform base: y=52..55
BTN_TOP = PLAT_BASE_TOP + PLAT_H + BTN_GAP  # button: y=57..60

# Colors
COLOR_BG = 0
COLOR_BLOCK = 2
COLOR_PLATFORM = 5
COLOR_BUTTON = 9
COLOR_WIN = 11

# Per-level config: (init_left, init_right, block_h)
#   Level 1:  7 vs  3, 4px blocks → goal  5 each
#   Level 2: 15 vs  1, 3px blocks → goal  8 each
#   Level 3:  3 vs 19, 2px blocks → goal 11 each
_LEVELS = [(7, 3, 4), (15, 1, 3), (3, 19, 2)]


def _build_level() -> Level:
    floor = Sprite(
        pixels=np.full((GRID, GRID), COLOR_BG, dtype=np.int8),
        name="floor",
        x=0,
        y=0,
        layer=0,
        tags=["floor", "sys_click", "sys_every_pixel"],
        collidable=False,
    )
    return Level(grid_size=(GRID, GRID), sprites=[floor], data={})


def _draw(floor: Sprite, left_count: int, right_count: int, block_h: int, won: bool) -> None:
    grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)

    block_color = COLOR_WIN if won else COLOR_BLOCK
    plat_color = COLOR_WIN if won else COLOR_PLATFORM
    btn_color = COLOR_WIN if won else COLOR_BUTTON

    # Platform bases
    grid[PLAT_BASE_TOP : PLAT_BASE_TOP + PLAT_H, LEFT_X : LEFT_X + PLAT_W] = plat_color
    grid[PLAT_BASE_TOP : PLAT_BASE_TOP + PLAT_H, RIGHT_X : RIGHT_X + PLAT_W] = plat_color

    # Buttons
    grid[BTN_TOP : BTN_TOP + BTN_H, LEFT_BTN_X : LEFT_BTN_X + BTN_W] = btn_color
    grid[BTN_TOP : BTN_TOP + BTN_H, RIGHT_BTN_X : RIGHT_BTN_X + BTN_W] = btn_color

    # Left stack (bottom-aligned on platform base, growing upward)
    for i in range(left_count):
        top = PLAT_BASE_TOP - block_h * (i + 1)
        grid[top : top + block_h, LEFT_X : LEFT_X + PLAT_W] = block_color

    # Right stack
    for i in range(right_count):
        top = PLAT_BASE_TOP - block_h * (i + 1)
        grid[top : top + block_h, RIGHT_X : RIGHT_X + PLAT_W] = block_color

    floor.pixels = grid


def _in_rect(x: int, y: int, rx: int, ry: int, rw: int, rh: int) -> bool:
    return rx <= x < rx + rw and ry <= y < ry + rh


class StackEqualizer(ARCBaseGame):
    """
    Two block towers start unequal. Click the button below a platform to move one
    block FROM the other platform TO it. Win when both towers are equal.

    Level 1:  7 vs  3 blocks (4px each) → balance at 5
    Level 2: 15 vs  1 blocks (3px each) → balance at 8
    Level 3:  3 vs 19 blocks (2px each) → balance at 11

    ACTION6 click on left button  → move top block from right tower to left
    ACTION6 click on right button → move top block from left tower to right
    """

    def __init__(self, seed: int = 0):
        self._left = 0
        self._right = 0
        self._block_h = 4
        self._floor: Sprite | None = None

        camera = Camera(width=GRID, height=GRID, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LEVELS],
            camera=camera,
            win_score=len(_LEVELS),
            available_actions=[6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        init_left, init_right, block_h = _LEVELS[self.level_index]
        self._left = init_left
        self._right = init_right
        self._block_h = block_h
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        if self._floor:
            _draw(self._floor, self._left, self._right, self._block_h, won=False)

    def _decode_click(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else None
        if not data:
            return None
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return None
        cell = self.camera.display_to_grid(raw_x, raw_y)
        if cell is None:
            return None
        return int(cell[0]), int(cell[1])

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id == int(GameAction.ACTION6.value):
            click = self._decode_click()
            if click is not None:
                cx, cy = click
                if _in_rect(cx, cy, LEFT_BTN_X, BTN_TOP, BTN_W, BTN_H):
                    if self._right > 0:
                        self._right -= 1
                        self._left += 1
                elif _in_rect(cx, cy, RIGHT_BTN_X, BTN_TOP, BTN_W, BTN_H):
                    if self._left > 0:
                        self._left -= 1
                        self._right += 1

        won = self._left == self._right
        if self._floor:
            _draw(self._floor, self._left, self._right, self._block_h, won=won)

        if won:
            self.next_level()
            self.complete_action()
            return

        self.complete_action()
