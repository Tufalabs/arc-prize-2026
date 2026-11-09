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
    "side": "right",
    "rows": 1,
    "pip_width": 2,
    "actions_per_tick": 1,
    "pips_per_tick": 3,
    "pip_color": 13,
    "spent_color": 0,
    "gap": 0,
    "margin": 0,
    "tier_colors": [13, 9],
}
ENERGY_CAPACITIES = [6, 12, 6, 12]

GAME_ID = "ss99-0001"
W = 64
H = 64

A_SPACE = int(GameAction.ACTION5.value)
A_CLICK = int(GameAction.ACTION6.value)

# Colors (see cli.py COLOR_MAP)
C_WHITE = 0  # (255, 255, 255) background
C_BORDER = 5  # (0,   0,   0)   black border
C_CANNON = 8  # (249, 60,  49)  red cannon tube
C_AIM = 1  # (204, 204, 204) light gray aim line
C_TARGET = 14  # (79,  204, 48)  green targets
C_BULLET = 11  # (255, 220, 0)   yellow bullet trail
C_WALL = 12  # (255, 133, 27)  orange/brown blockage

CANNON_TIP_X = 5
CANNON_Y = 31


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def _extend_to_border(sx: int, sy: int, ax: int, ay: int) -> tuple[int, int]:
    dx = ax - sx
    dy = ay - sy
    if dx == 0 and dy == 0:
        return ax, ay
    candidates: list[float] = []
    if dx > 0:
        candidates.append((W - 2 - sx) / dx)
    elif dx < 0:
        candidates.append((1 - sx) / dx)
    if dy > 0:
        candidates.append((H - 2 - sy) / dy)
    elif dy < 0:
        candidates.append((1 - sy) / dy)
    if not candidates:
        return ax, ay
    t = min(c for c in candidates if c > 0)
    ex = max(1, min(W - 2, int(sx + dx * t)))
    ey = max(1, min(H - 2, int(sy + dy * t)))
    return ex, ey


def _build_base_grid() -> np.ndarray:
    grid = np.full((H, W), C_WHITE, dtype=np.int8)
    grid[0, :] = C_BORDER
    grid[H - 1, :] = C_BORDER
    grid[:, 0] = C_BORDER
    grid[:, W - 1] = C_BORDER
    return grid


def _draw_cannon(grid: np.ndarray) -> None:
    # 5x3 black tube extending from left border
    for dy in range(-1, 2):
        for dx in range(5):
            y = CANNON_Y + dy
            x = 1 + dx
            if 0 < y < H - 1 and 0 < x < W - 1:
                grid[y, x] = C_CANNON


LEVEL_SPECS: list[dict] = [
    {"name": "Level 1", "targets": [(50, 31)], "walls": [], "bounces": 0},
    {"name": "Level 2", "targets": [(45, 20), (50, 45)], "walls": [], "bounces": 0},
    {
        "name": "Level 3",
        "targets": [(50, 15)],
        "walls": [(28, 20, 30, 45)],  # (x1, y1, x2, y2) rectangle
        "bounces": 1,
    },
    {"name": "Level 4", "targets": [(50, 12), (50, 52)], "walls": [(40, 1, 42, 22), (40, 40, 42, 58)], "bounces": 1},
]


def _wall_cells(walls: list) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for x1, y1, x2, y2 in walls:
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if 0 < x < W - 1 and 0 < y < H - 1:
                    cells.add((x, y))
    return cells


def _trace_ray(
    sx: int, sy: int, ax: int, ay: int, walls: set[tuple[int, int]], max_bounces: int = 0
) -> list[tuple[int, int]]:
    dx = ax - sx
    dy = ay - sy
    if dx == 0 and dy == 0:
        return []

    path: list[tuple[int, int]] = []

    # First segment: cannon toward aim, extended to border
    ex, ey = _extend_to_border(sx, sy, ax, ay)
    segment = _bresenham(sx, sy, ex, ey)
    for px, py in segment:
        if (px, py) in walls:
            return path
        path.append((px, py))

    if max_bounces <= 0:
        return path

    # Reflect off border
    rdx, rdy = dx, dy
    if ex <= 1 or ex >= W - 2:
        rdx = -rdx
    if ey <= 1 or ey >= H - 2:
        rdy = -rdy

    # Second segment from bounce point
    ex2, ey2 = _extend_to_border(ex, ey, ex + rdx, ey + rdy)
    segment2 = _bresenham(ex, ey, ex2, ey2)
    for i, (px, py) in enumerate(segment2):
        if i == 0:
            continue
        if (px, py) in walls:
            return path
        path.append((px, py))

    return path


def _draw_walls(grid: np.ndarray, walls: list) -> None:
    for x, y in _wall_cells(walls):
        grid[y, x] = C_WALL


def _build_level(spec: dict) -> Level:
    grid = _build_base_grid()
    _draw_cannon(grid)
    _draw_walls(grid, spec.get("walls", []))
    for tx, ty in spec["targets"]:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = ty + dy, tx + dx
                if 0 < y < H - 1 and 0 < x < W - 1:
                    grid[y, x] = C_TARGET
    board = Sprite(
        pixels=grid, name="board", x=0, y=0, collidable=False, layer=0, tags=["board", "sys_click", "sys_every_pixel"]
    )
    return Level(
        name=spec["name"],
        grid_size=(W, H),
        sprites=[board],
        data={"targets": spec["targets"], "walls": spec.get("walls", []), "bounces": spec.get("bounces", 0)},
    )


class Ss99(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, W, H, background=C_WHITE, letter_box=C_WHITE, interfaces=[self._energy_bar])
        super().__init__(
            GAME_ID,
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[A_SPACE, A_CLICK],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        raw = level.get_data("targets") or []
        self._targets: list[tuple[int, int]] = [(int(t[0]), int(t[1])) for t in raw]
        self._walls: list = level.get_data("walls") or []
        self._wall_cells: set[tuple[int, int]] = _wall_cells(self._walls)
        self._max_bounces: int = int(level.get_data("bounces") or 0)
        self._aim: tuple[int, int] | None = None
        self._bullet_trail: list[tuple[int, int]] = []
        self._board = level.get_sprites_by_name("board")[0]
        self._render()

    def _render(self) -> None:
        canvas = _build_base_grid()
        _draw_cannon(canvas)
        _draw_walls(canvas, self._walls)

        for tx, ty in self._targets:
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y, x = ty + dy, tx + dx
                    if 0 < y < H - 1 and 0 < x < W - 1:
                        canvas[y, x] = C_TARGET

        if self._aim is not None:
            ax, ay = self._aim
            aim_path = _trace_ray(CANNON_TIP_X + 1, CANNON_Y, ax, ay, self._wall_cells, self._max_bounces)
            for px, py in aim_path:
                if 0 < px < W - 1 and 0 < py < H - 1 and canvas[py, px] == C_WHITE:
                    canvas[py, px] = C_AIM

        for bx, by in self._bullet_trail:
            if 0 < bx < W - 1 and 0 < by < H - 1 and canvas[by, bx] not in (C_BORDER, C_CANNON, C_WALL):
                canvas[by, bx] = C_BULLET

        self._board.pixels = canvas

    def _handle_click(self) -> None:
        data = self.action.data or {}
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return
        grid_pos = self.camera.display_to_grid(raw_x, raw_y)
        if grid_pos is None:
            return
        gx, gy = int(grid_pos[0]), int(grid_pos[1])
        if gx > CANNON_TIP_X and 0 < gy < H - 1:
            self._aim = (gx, gy)
            self._bullet_trail = []
            self._render()

    def _handle_shoot(self) -> None:
        if self._aim is None:
            return
        ax, ay = self._aim
        path = _trace_ray(CANNON_TIP_X + 1, CANNON_Y, ax, ay, self._wall_cells, self._max_bounces)

        hits: set[int] = set()
        for px, py in path:
            for i, (tx, ty) in enumerate(self._targets):
                if abs(px - tx) <= 1 and abs(py - ty) <= 1:
                    hits.add(i)

        self._bullet_trail = path
        for i in sorted(hits, reverse=True):
            self._targets.pop(i)
        self._render()

        if not self._targets:
            self.next_level()

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == A_CLICK:
            self._handle_click()
        elif action_id == A_SPACE:
            self._handle_shoot()
        self.complete_action()
