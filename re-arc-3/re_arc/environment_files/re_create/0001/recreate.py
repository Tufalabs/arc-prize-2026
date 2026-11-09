from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

SIDES = ("top", "bottom", "left", "right")


class EnergyBar(RenderableUserDisplay):
    def __init__(
        self,
        *,
        side: str = "top",
        rows: int = 1,
        pip_width: int = 2,
        actions_per_tick: int = 1,
        pips_per_tick: int = 1,
        pip_color: int = 11,
        spent_color: int = 3,
        gap: int = 1,
        margin: int = 0,
        tier_colors: list[int] | None = None,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        self.side = side
        self.rows = max(1, min(int(rows), 3))
        self.pip_width = max(1, min(int(pip_width), 3))
        self.actions_per_tick = max(1, int(actions_per_tick))
        self.pips_per_tick = max(1, int(pips_per_tick))
        self.pip_color = int(pip_color)
        self.spent_color = int(spent_color)
        self.gap = max(0, int(gap))
        self.margin = max(0, int(margin))
        self.tier_colors: list[int] = list(tier_colors) if tier_colors else [self.pip_color]

        self.capacity_actions = 0
        self.remaining_actions = 0

    def set_capacity(self, capacity_actions: int) -> None:
        self.capacity_actions = max(0, int(capacity_actions))
        self.remaining_actions = self.capacity_actions

    def set_remaining_actions(self, remaining_actions: int) -> None:
        self.remaining_actions = max(0, min(int(remaining_actions), self.capacity_actions))

    def tick(self) -> int:
        if self.remaining_actions > 0:
            self.remaining_actions -= 1
        return self.remaining_actions

    def _actions_to_pips(self, actions: int) -> int:
        if actions <= 0:
            return 0
        return (actions * self.pips_per_tick + self.actions_per_tick - 1) // self.actions_per_tick

    @property
    def total_pips(self) -> int:
        return self._actions_to_pips(self.capacity_actions)

    @property
    def remaining_pips(self) -> int:
        return self._actions_to_pips(self.remaining_actions)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.capacity_actions <= 0:
            return frame
        h, w = int(frame.shape[0]), int(frame.shape[1])
        total = self.total_pips
        remaining = self.remaining_pips
        if total <= 0:
            return frame

        pw = self.pip_width
        ph = self.pip_width
        stride = pw + self.gap
        horizontal = self.side in ("top", "bottom")
        long_dim = w if horizontal else h
        pips_per_row = max(1, (long_dim - self.margin) // stride)
        slot_count = pips_per_row * self.rows

        if total <= slot_count:
            visible = total
            colored = remaining
            color = self.pip_color
        else:
            visible = slot_count
            consumed = total - remaining
            tier_index = consumed // slot_count
            consumed_in_tier = consumed - tier_index * slot_count
            if tier_index >= len(self.tier_colors):
                tier_index = len(self.tier_colors) - 1
                consumed_in_tier = slot_count
            colored = slot_count - consumed_in_tier
            color = self.tier_colors[tier_index]

        for i in range(visible):
            row = i // pips_per_row
            col = i % pips_per_row
            if row >= self.rows:
                break
            cell_color = color if i < colored else self.spent_color
            if horizontal:
                x = self.margin + col * stride
                if self.side == "top":
                    y = self.margin + row * stride
                else:
                    y = h - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
            else:
                y = self.margin + col * stride
                if self.side == "left":
                    x = self.margin + row * stride
                else:
                    x = w - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
        return frame

    @staticmethod
    def _fill(frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
        h_frame, w_frame = int(frame.shape[0]), int(frame.shape[1])
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_frame, x + w)
        y1 = min(h_frame, y + h)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = color


ENERGY_CONFIG = {
    "side": "top",
    "rows": 2,
    "pip_width": 2,
    "actions_per_tick": 1,
    "pips_per_tick": 2,
    "pip_color": 13,
    "spent_color": 0,
    "gap": 0,
    "margin": 0,
    "tier_colors": [13, 9, 11, 1, 14],
}
ENERGY_CAPACITIES = [12, 27, 78]

GAME_ID = "re_create-0001"
VARIANT = "0001"

SPEC = {
    "spec_id": "spec-custom-re-create-0001",
    "objective": "obj.recreate_pattern",
    "mechanics": ["mech.cursor_navigation", "mech.color_cycling", "mech.pattern_match"],
    "layout_template": "dual_pane",
    "action_set": ["up", "down", "left", "right", "space"],
    "target_optimal_steps": 50,
    "grid_size": [32, 18],
    "level_count": 3,
}

# ── ARC colour indices ────────────────────────────────────────────────
COLOR_LEFT_BG = 1  # dark  -target-pane background
COLOR_RIGHT_BG = 0  # white -canvas-pane background
COLOR_GAP = 1  # dark  -divider between panes
COLOR_HIGHLIGHT = 11  # yellow -cursor border
COLOR_EMPTY = 10  # cyan  -unpainted canvas cell
CELL_SIZE = 3  # each logical cell is 3x3 pixels
CELL_GAP = 1  # 1-pixel gap between cells
GRID_W = 32
GRID_H = 18
PANE_W = 15
GAP_W = 2
LEFT_X = 0
RIGHT_X = PANE_W + GAP_W  # 17

MOVE = {
    int(GameAction.ACTION1.value): (0, -1),  # up
    int(GameAction.ACTION2.value): (0, 1),  # down
    int(GameAction.ACTION3.value): (-1, 0),  # left
    int(GameAction.ACTION4.value): (1, 0),  # right
}

# ── Levels -placeholder patterns (to be replaced later) ─────────────
RAW_LEVELS: list[dict[str, object]] = [
    {
        "pattern": [[1, 2]],
        "colors": [9, 6],  # blue, red
    },
    {
        "pattern": [[1, 2], [2, 1]],
        "colors": [9, 6],  # blue, red
    },
    {
        "pattern": [[1, 2, 3], [3, 1, 2], [2, 3, 1]],
        "colors": [9, 6, 12],  # blue, red, orange
    },
]


def _build_level(index: int, raw: dict[str, object]) -> Level:
    board = Sprite(
        np.full((GRID_H, GRID_W), COLOR_GAP, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        tags=["board"],
        collidable=False,
    )
    pattern = [list(row) for row in raw["pattern"]]
    colors = list(raw["colors"])
    return Level(
        name=f"ReCreate L{index + 1}",
        grid_size=(GRID_W, GRID_H),
        sprites=[board],
        data={"pattern": pattern, "colors": colors, "spec": SPEC},
    )


class ReCreate(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(i, raw) for i, raw in enumerate(RAW_LEVELS)]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_GAP, interfaces=[self._energy_bar])

        # Defaults must be set before super().__init__() which calls on_set_level()
        self._board: Sprite | None = None
        self._pattern: list[list[int]] = []
        self._colors: list[int] = []
        self._canvas: list[list[int]] = []
        self._cursor_x = 0
        self._cursor_y = 0
        self._pat_rows = 0
        self._pat_cols = 0
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    # ── level lifecycle ───────────────────────────────────────────────

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        boards = level.get_sprites_by_name("board")
        self._board = boards[0] if boards else None

        self._pattern = [list(row) for row in (level.get_data("pattern") or [[]])]
        self._colors = list(level.get_data("colors") or [])
        self._pat_rows = len(self._pattern)
        self._pat_cols = len(self._pattern[0]) if self._pattern else 0

        self._canvas = [[0] * self._pat_cols for _ in range(self._pat_rows)]
        self._cursor_x = 0
        self._cursor_y = 0
        self._render()

    # ── helpers ────────────────────────────────────────────────────────

    def _pattern_pixel_size(self) -> tuple[int, int]:
        pw = self._pat_cols * CELL_SIZE + max(0, self._pat_cols - 1) * CELL_GAP
        ph = self._pat_rows * CELL_SIZE + max(0, self._pat_rows - 1) * CELL_GAP
        return pw, ph

    def _cell_origin(self, pane_x: int, cx: int, cy: int) -> tuple[int, int]:
        """Top-left pixel coordinate for cell (cx, cy) inside a pane."""
        pw, ph = self._pattern_pixel_size()
        ox = pane_x + (PANE_W - pw) // 2
        oy = (GRID_H - ph) // 2
        return ox + cx * (CELL_SIZE + CELL_GAP), oy + cy * (CELL_SIZE + CELL_GAP)

    def _color_for(self, val: int) -> int:
        if val == 0:
            return COLOR_EMPTY
        if 1 <= val <= len(self._colors):
            return self._colors[val - 1]
        return COLOR_EMPTY

    # ── rendering ─────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._board is None:
            return
        grid = np.full((GRID_H, GRID_W), COLOR_GAP, dtype=np.int8)

        # pane backgrounds
        grid[:, LEFT_X : LEFT_X + PANE_W] = COLOR_LEFT_BG
        grid[:, RIGHT_X : RIGHT_X + PANE_W] = COLOR_RIGHT_BG

        # target cells (left pane)
        for cy in range(self._pat_rows):
            for cx in range(self._pat_cols):
                color = self._color_for(self._pattern[cy][cx])
                px, py = self._cell_origin(LEFT_X, cx, cy)
                grid[py : py + CELL_SIZE, px : px + CELL_SIZE] = color

        # canvas cells (right pane)
        for cy in range(self._pat_rows):
            for cx in range(self._pat_cols):
                color = self._color_for(self._canvas[cy][cx])
                px, py = self._cell_origin(RIGHT_X, cx, cy)
                grid[py : py + CELL_SIZE, px : px + CELL_SIZE] = color

        # highlight border around selected cell
        px, py = self._cell_origin(RIGHT_X, self._cursor_x, self._cursor_y)
        for x in range(px - 1, px + CELL_SIZE + 1):
            if 0 <= x < GRID_W:
                if 0 <= py - 1:
                    grid[py - 1, x] = COLOR_HIGHLIGHT
                if py + CELL_SIZE < GRID_H:
                    grid[py + CELL_SIZE, x] = COLOR_HIGHLIGHT
        for y in range(py - 1, py + CELL_SIZE + 1):
            if 0 <= y < GRID_H:
                if 0 <= px - 1:
                    grid[y, px - 1] = COLOR_HIGHLIGHT
                if px + CELL_SIZE < GRID_W:
                    grid[y, px + CELL_SIZE] = COLOR_HIGHLIGHT

        self._board.pixels = grid

    # ── game logic ────────────────────────────────────────────────────

    def _check_win(self) -> bool:
        return self._canvas == self._pattern

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id in MOVE:
            dx, dy = MOVE[action_id]
            nx, ny = self._cursor_x + dx, self._cursor_y + dy
            if 0 <= nx < self._pat_cols and 0 <= ny < self._pat_rows:
                self._cursor_x = nx
                self._cursor_y = ny
        elif action_id == int(GameAction.ACTION5.value):
            cur = self._canvas[self._cursor_y][self._cursor_x]
            self._canvas[self._cursor_y][self._cursor_x] = (cur + 1) % (len(self._colors) + 1)

        if self._check_win():
            self.next_level()

        self._render()
        self.complete_action()


def _novel_signature_re_create(seed: int) -> int:
    acc = int(seed)
    for idx in range(1, 25):
        acc = (acc * 37 + 19 * idx + 7) % 104729
        if idx % 3 == 0:
            acc = (acc ^ (idx * 97)) % 104729
    return acc


__all__ = ["GAME_ID", "SPEC", "ReCreate"]
