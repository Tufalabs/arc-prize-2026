from __future__ import annotations

import random

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "rotate_pattern-0002"
COLOR_BG = 0  # white        (255, 255, 255) — grid background
COLOR_FILL = 9  # blue         ( 30, 147, 255) — filled (#) cells in pattern
COLOR_INNER = 6  # red          (249,  60,  49) — 0-cells inside pattern bounding box
COLOR_BTN_OFF = 8  # red          (249,  60,  49) — indicator button (not matched)
COLOR_BTN_ON = 14  # green        ( 79, 204,  48) — indicator button (matched)

# Level 2 uses different color schemes for left vs right pattern
L2_LEFT_FILL = 12  # orange       (255, 133,  27)
L2_LEFT_INNER = 11  # yellow       (255, 220,   0)
L2_RIGHT_FILL = 15  # purple       (163,  86, 214)
L2_RIGHT_INNER = 10  # cyan        (136, 216, 241)

GRID = 64  # all levels use the full 64x64 grid
SCALE = 3  # each pattern cell → SCALExSCALE pixels

# Patterns are centered in each half of the 64x64 grid.
# Left-half center col = 16, right-half center col = 48, row center = 32.
# Scaled pixel size = pattern_size * SCALE; offset = scaled_size // 2.

# ── Level 1: 3x3 pattern → 9x9 pixels ───────────────────────────────────────
#   vertical:   row 32 - 9//2 = 28  → rows 28-36
#   left  col:  col 16 - 9//2 = 12  → cols 12-20
#   right col:  col 48 - 9//2 = 44  → cols 44-52
PAD_ROW_L1 = 28
PAD_COL_L_L1, PAD_COL_R_L1 = 12, 44

#  # # #
#  # . .
#  # . #
PATTERN_L1 = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1]], dtype=np.int8)

START_ROT_L1 = 1  # 90° CW — one ACTION3 press from matching

# ── Level 2: 5x5 pattern → 15x15 pixels ─────────────────────────────────────
#   vertical:   row 32 - 15//2 = 25  → rows 25-39
#   left  col:  col 16 - 15//2 = 9   → cols  9-23
#   right col:  col 48 - 15//2 = 41  → cols 41-55
PAD_ROW_L2 = 25
PAD_COL_L_L2, PAD_COL_R_L2 = 9, 41

#  # # . . #       all four corners filled, 5x5
#  # # # # .
#  # . . # .
#  # . . # #
#  # # # # #
PATTERN_L2 = np.array(
    [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 0, 1, 1], [1, 1, 1, 1, 1]], dtype=np.int8
)

# ── Level 3: 7x7 star (SW + NW branches missing), 45° rotations ────────────
# Star arms: 8 directions from center (3,3), each with 3 cells (distance 1,2,3).
# Order: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
_STAR_ARMS = {
    0: [(2, 3), (1, 3), (0, 3)],  # N
    1: [(2, 4), (1, 5), (0, 6)],  # NE
    2: [(3, 4), (3, 5), (3, 6)],  # E
    3: [(4, 4), (5, 5), (6, 6)],  # SE
    4: [(4, 3), (5, 3), (6, 3)],  # S
    5: [(4, 2), (5, 1), (6, 0)],  # SW
    6: [(3, 2), (3, 1), (3, 0)],  # W
    7: [(2, 2), (1, 1), (0, 0)],  # NW
}
_STAR_MISSING_ARMS = (5, 7)  # SW and NW missing (separated by 2 = 90°)


def _star_rotation(k: int) -> np.ndarray:
    """Return the 7x7 star with the missing arms rotated by k*45° CW."""
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[3, 3] = 1  # center always present
    missing = {(arm + k) % 8 for arm in _STAR_MISSING_ARMS}
    for arm_idx, cells in _STAR_ARMS.items():
        if arm_idx in missing:
            continue
        for r, c in cells:
            grid[r, c] = 1
    return grid


PATTERN_L3 = _star_rotation(0)
L3_NUM_ROTATIONS = 8  # 45° per step
L3_START_ROT = 3  # 135° CW — minimum 3 turns to solve

# ── Level 4: 3x3 pattern, left target enlarged 2x to 6x6 ───────────────────
#  # # #
#  # . #
#  # . #
PATTERN_L4 = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=np.int8)
PATTERN_L4_BIG = np.repeat(np.repeat(PATTERN_L4, 2, axis=0), 2, axis=1)  # 6x6

# ── Layout helpers ──────────────────────────────────────────────────────────
# Left-half center col=16, right-half center col=48, row center=32.
PAD_ROW_L3 = 32 - (7 * SCALE) // 2  # 7x7 pattern, always same size
PAD_COL_L_L3 = 16 - (7 * SCALE) // 2
PAD_COL_R_L3 = 48 - (7 * SCALE) // 2

# L4: left is 6x6 (18x18 px), right is 3x3 (9x9 px) — different centering
PAD_ROW_L4_L = 32 - (6 * SCALE) // 2
PAD_COL_L4_L = 16 - (6 * SCALE) // 2
PAD_ROW_L4_R = 32 - (3 * SCALE) // 2
PAD_COL_L4_R = 48 - (3 * SCALE) // 2


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


def _place(grid: np.ndarray, pattern: np.ndarray, row0: int, col0: int, fg: int, inner: int = COLOR_INNER) -> None:
    """Fill the bounding box at SCALExSCALE pixels per cell."""
    h, w = pattern.shape
    for r in range(h):
        for c in range(w):
            color = fg if pattern[r, c] else inner
            for dr in range(SCALE):
                for dc in range(SCALE):
                    grid[row0 + r * SCALE + dr, col0 + c * SCALE + dc] = color


BTN_SIZE = 7  # button is a square
BTN_ROW = 5  # top padding before button
BTN_COL = GRID // 2 - BTN_SIZE // 2  # centered horizontally


def _place_button(grid: np.ndarray, matched: bool) -> None:
    """Draw a small indicator button at middle-top of the grid."""
    color = COLOR_BTN_ON if matched else COLOR_BTN_OFF
    for dr in range(BTN_SIZE):
        for dc in range(BTN_SIZE):
            grid[BTN_ROW + dr, BTN_COL + dc] = color


class RotatePattern(ARCBaseGame):
    def __init__(self, seed: int = 0):
        # Pre-draw the random 90° offsets for levels 1, 2, 4 so they are stable
        # across repeated on_set_level calls — e.g. when the player issues a
        # mid-play RESET, which re-enters on_set_level and would otherwise
        # draw a fresh rotation from the rng and desynchronise from the
        # rotation clients have been manipulating.
        rng = random.Random(seed)
        self._level_rotations: dict[int, int] = {level_index: rng.choice([1, 2, 3]) for level_index in (1, 2, 4)}
        self._rotation = START_ROT_L1
        self._floor: Sprite | None = None

        camera = Camera(width=GRID, height=GRID, background=0)
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in range(5)],
            camera=camera,
            win_score=5,
            available_actions=[3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None

        if self.level_index == 0:
            self._rotation = START_ROT_L1
        elif self.level_index == 3:
            self._rotation = L3_START_ROT  # 135° — at least 3 turns needed
        else:
            self._rotation = self._level_rotations[self.level_index]

        self._update_grid()

    def _pattern(self) -> np.ndarray:
        # index: 0=L1, 1=L2(same as L1), 2=5x5, 3=star, 4=enlarged
        return (PATTERN_L1, PATTERN_L1, PATTERN_L2, PATTERN_L3, PATTERN_L4)[self.level_index]

    def _num_rotations(self) -> int:
        """Number of discrete rotation steps (4 for 90°, 8 for 45°)."""
        return L3_NUM_ROTATIONS if self.level_index == 3 else 4

    def _rotated_pattern(self) -> np.ndarray:
        """Return the player pattern rotated by current rotation state."""
        if self.level_index == 3:
            return _star_rotation(self._rotation)
        return np.rot90(self._pattern(), k=-self._rotation)

    def _update_grid(self) -> None:
        if self._floor is None:
            return
        rotated = self._rotated_pattern()
        matched = self._rotation == 0

        grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)

        if self.level_index == 1:
            # L2: same pattern as L1, but different colors per side
            pad_r, col_l, col_r = PAD_ROW_L1, PAD_COL_L_L1, PAD_COL_R_L1
            _place(grid, self._pattern(), pad_r, col_l, L2_LEFT_FILL, L2_LEFT_INNER)
            _place(grid, rotated, pad_r, col_r, L2_RIGHT_FILL, L2_RIGHT_INNER)
        elif self.level_index == 4:
            # L5: left target is enlarged 6x6, right player is 3x3
            _place(grid, PATTERN_L4_BIG, PAD_ROW_L4_L, PAD_COL_L4_L, COLOR_FILL)
            _place(grid, rotated, PAD_ROW_L4_R, PAD_COL_L4_R, COLOR_FILL)
        else:
            if self.level_index == 0:
                pad_r, col_l, col_r = PAD_ROW_L1, PAD_COL_L_L1, PAD_COL_R_L1
            elif self.level_index == 2:
                pad_r, col_l, col_r = PAD_ROW_L2, PAD_COL_L_L2, PAD_COL_R_L2
            else:
                pad_r, col_l, col_r = PAD_ROW_L3, PAD_COL_L_L3, PAD_COL_R_L3
            _place(grid, self._pattern(), pad_r, col_l, COLOR_FILL)
            _place(grid, rotated, pad_r, col_r, COLOR_FILL)

        _place_button(grid, matched)
        self._floor.pixels = grid

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        n = self._num_rotations()

        if action_id == int(GameAction.ACTION3.value):
            self._rotation = (self._rotation - 1) % n  # rotate left (CCW)
        elif action_id == int(GameAction.ACTION4.value):
            self._rotation = (self._rotation + 1) % n  # rotate right (CW)
        elif action_id == int(GameAction.ACTION5.value):
            if self._rotation == 0:  # match → advance
                self.next_level()
                self.complete_action()
                return

        self._update_grid()
        self.complete_action()
