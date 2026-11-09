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
    "side": "bottom",
    "rows": 2,
    "pip_width": 1,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 6,
    "spent_color": 15,
    "gap": 0,
    "margin": 0,
    "tier_colors": [6, 1, 9, 12, 11, 2, 4],
}
ENERGY_CAPACITIES = [90, 255, 783]

GAME_ID = "tunnel_search-0001"
VP = 64
CELL = 4
STEP = 2

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)

C_FLOOR = 0
C_WALL = 5
C_PLAYER = 9
C_DIAMOND = 11
C_DIM = 1

DIRS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

# 32x32 grid. Room in center, tunnels branching up and left.
LAYOUT_1 = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 0
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 1
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 2
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 3
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 4
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 5
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 6
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 7
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 8
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 9
    "WWWWWWWWWWWWWWW..WWWWWWWWWWWWWWW",  # 10 tunnel up (diamond)
    "WWWWWWWWWWWWWWW..WWWWWWWWWWWWWWW",  # 11
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 12 room top
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 13
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 14
    "WWWWWWWW............WWWWWWWWWWWW",  # 15 left stub+room
    "WWWWWWWW............WWWWWWWWWWWW",  # 16
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 17
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 18
    "WWWWWWWWWWWW........WWWWWWWWWWWW",  # 19 room bottom
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 20
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 21
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 22
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 23
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 24
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 25
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 26
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 27
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 28
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 29
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 30
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",  # 31
]

# Level 2: long horizontal corridor, tunnel down at col 10-11, tunnel up at col 24-25.
LAYOUT_2 = (
    ["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"] * 2
    + ["WWWWWWWWWWWWWWWWWWWWWWWW..WWWWWW"] * 13
    + ["WW............................WW"] * 2
    + ["WWWWWWWWWW..WWWWWWWWWWWWWWWWWWWW"] * 5
    + ["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"] * 10
)


# Level 3: 48x48 map. Main corridor with branches. Left-up is dead end (1 diamond).
def _make_layout_3() -> list[str]:
    g = [["W"] * 48 for _ in range(48)]
    # Main horizontal corridor: rows 23-24, cols 2-45
    for r in (23, 24):
        for c in range(2, 46):
            g[r][c] = "."
    # Dead-end UP branch (left): cols 8-9, rows 4-22
    for r in range(4, 23):
        for c in (8, 9):
            g[r][c] = "."
    # DOWN branch (middle): cols 24-25, rows 25-40
    for r in range(25, 41):
        for c in (24, 25):
            g[r][c] = "."
    # UP branch (right): cols 40-41, rows 8-22
    for r in range(8, 23):
        for c in (40, 41):
            g[r][c] = "."
    return ["".join(row) for row in g]


LAYOUT_3 = _make_layout_3()

# All positions in pixels. Diamonds centered in 2-cell-wide corridors.
LEVEL_SPECS: list[dict] = [
    {"name": "Level 1", "layout": LAYOUT_1, "start": [60, 60], "diamonds": [[62, 42], [34, 62]]},
    {"name": "Level 2", "layout": LAYOUT_2, "start": [10, 62], "diamonds": [[42, 78], [72, 62], [98, 10]]},
    {"name": "Level 3", "layout": LAYOUT_3, "start": [10, 94], "diamonds": [[34, 18], [98, 158], [162, 34], [178, 94]]},
]


def _parse_layout(lines: list[str]) -> np.ndarray:
    mh = len(lines)
    mw = len(lines[0]) if lines else 0
    grid = np.zeros((mh, mw), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _build_level(spec: dict) -> Level:
    initial = np.full((VP, VP), C_FLOOR, dtype=np.int8)
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(VP, VP),
        sprites=[board],
        data={"layout": spec["layout"], "start": spec["start"], "diamonds": spec["diamonds"]},
    )


class TunnelSearch(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(s) for s in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, VP, VP, background=C_WALL, letter_box=C_WALL, interfaces=[self._energy_bar])
        super().__init__(
            GAME_ID,
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[UP, DOWN, LEFT, RIGHT],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._grid = _parse_layout(level.get_data("layout"))
        self._mh, self._mw = self._grid.shape
        self._map_w = self._mw * CELL
        self._map_h = self._mh * CELL
        start = level.get_data("start")
        self._px, self._py = int(start[0]), int(start[1])
        self._diamonds: set[tuple[int, int]] = {(int(d[0]), int(d[1])) for d in level.get_data("diamonds")}
        self._total = len(self._diamonds)
        self._board = level.get_sprites_by_name("board")[0]
        # Pre-render static walls
        self._static = np.full((self._map_h, self._map_w), C_FLOOR, dtype=np.int8)
        for gy in range(self._mh):
            for gx in range(self._mw):
                if self._grid[gy, gx] == 1:
                    y0, x0 = gy * CELL, gx * CELL
                    self._static[y0 : y0 + CELL, x0 : x0 + CELL] = C_WALL
        self._redraw()

    def _blocked(self, px: int, py: int) -> bool:
        for cx, cy in [(px, py), (px + 3, py), (px, py + 3), (px + 3, py + 3)]:
            gx, gy = cx // CELL, cy // CELL
            if gx < 0 or gx >= self._mw or gy < 0 or gy >= self._mh:
                return True
            if self._grid[gy, gx] == 1:
                return True
        return False

    def _cam_offset(self) -> tuple[int, int]:
        cx = self._px + CELL // 2 - VP // 2
        cy = self._py + CELL // 2 - VP // 2
        cx = max(0, min(self._map_w - VP, cx))
        cy = max(0, min(self._map_h - VP, cy))
        return cx, cy

    def _redraw(self) -> None:
        full = self._static.copy()
        # Diamonds (diamond shape in 4x4 area)
        for dx, dy in self._diamonds:
            full[dy, dx + 1 : dx + 3] = C_DIAMOND
            full[dy + 1, dx : dx + 4] = C_DIAMOND
            full[dy + 2, dx : dx + 4] = C_DIAMOND
            full[dy + 3, dx + 1 : dx + 3] = C_DIAMOND
        # Player (blue ring, white center)
        full[self._py : self._py + CELL, self._px : self._px + CELL] = C_PLAYER
        full[self._py + 1 : self._py + CELL - 1, self._px + 1 : self._px + CELL - 1] = C_FLOOR
        # Crop viewport centered on player
        cx, cy = self._cam_offset()
        canvas = full[cy : cy + VP, cx : cx + VP].copy()
        # Counter overlay (full-width bar)
        remaining = len(self._diamonds)
        canvas[0:6, 0:VP] = C_DIM
        for i in range(self._total):
            color = C_DIAMOND if i < remaining else C_WALL
            x0 = 2 + i * 6
            canvas[1, x0 + 1 : x0 + 3] = color
            canvas[2, x0 : x0 + 4] = color
            canvas[3, x0 : x0 + 4] = color
            canvas[4, x0 + 1 : x0 + 3] = color
        self._board.pixels = canvas

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in DIRS:
            ddx, ddy = DIRS[action_id]
            nx = self._px + ddx * STEP
            ny = self._py + ddy * STEP
            if not self._blocked(nx, ny):
                self._px, self._py = nx, ny
                collected = {
                    (dx, dy) for dx, dy in self._diamonds if abs(self._px - dx) < CELL and abs(self._py - dy) < CELL
                }
                self._diamonds -= collected
                self._redraw()
                if not self._diamonds:
                    self.next_level()
        self.complete_action()
