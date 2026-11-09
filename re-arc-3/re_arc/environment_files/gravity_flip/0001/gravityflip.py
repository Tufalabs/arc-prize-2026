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
    "side": "left",
    "rows": 2,
    "pip_width": 1,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 14,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [14],
}
ENERGY_CAPACITIES = [12, 27, 18]

GAME_ID = "gravity_flip-0001"
W = H = 64
CELL = 4
GW = GH = 16

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)

C_FLOOR = 0
C_WALL = 5
C_PLAYER = 9
C_START_WALL = 10
C_GOAL = 14
C_KILL = 8
C_ARROW_DIM = 1
C_ARROW_LIT = 11

# Gravity directions: arrow key sets which way the player "falls"
GRAVITY = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

LAYOUT_1 = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WWWW..WWWWWWWWWW",
    "WWWW..WWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WW....WWWWWWWWWW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LAYOUT_2 = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "WW.....WWWWWWWWW",
    "WW.....WWWWWWWWW",
    "WW.WW..WWWWWWWWW",
    "WW.....WWWWWWWWW",
    "WW............WW",
    "WWW...........WW",
    "WWW.WWWW......WW",
    "WW...........WWW",
    "WW.....WW.....WW",
    "WW.....WW.....WW",
    "WW.....WW.....WW",
    "WWW.WWWWWWWWWWWW",
    "WWW.WWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LAYOUT_S = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW......WWWWWWWW",
    "WW......WWWWWWWW",
    "WWW...........WW",
    "WWW...........WW",
    "WWWWWWWW......WW",
    "WWWWWWWW......WW",
    "WW...........WWW",
    "WW...........WWW",
    "WW...........WWW",
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "layout": LAYOUT_1,
        "start": [12, 12],
        "goal_line": [[x, 2] for x in range(2, 6)],
        "start_wall": [[13, y] for y in range(10, 14)],
        "kill_zone": [],
        "gravity": RIGHT,
    },
    {
        "name": "Level 2",
        "layout": LAYOUT_S,
        "start": [12, 3],
        "goal_line": [[2, y] for y in range(11, 14)],
        "start_wall": [[13, y] for y in range(2, 5)],
        "kill_zone": [],
        "gravity": RIGHT,
    },
    {
        "name": "Level 3",
        "layout": LAYOUT_2,
        "start": [5, 3],
        "goal_line": [[2, y] for y in range(9, 13)],
        "start_wall": [[x, 2] for x in range(2, 7)],
        "kill_zone": [],
        "gravity": UP,
    },
]


def _parse_layout(lines: list[str]) -> np.ndarray:
    grid = np.zeros((GH, GW), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _apply_gravity(base: np.ndarray, px: int, py: int, dx: int, dy: int) -> tuple[int, int]:
    while True:
        nx, ny = px + dx, py + dy
        if nx < 0 or nx >= GW or ny < 0 or ny >= GH:
            break
        if base[ny, nx] == 1:
            break
        px, py = nx, ny
    return px, py


def _build_path(base: np.ndarray, px: int, py: int, dx: int, dy: int) -> list[tuple[int, int]]:
    path = [(px, py)]
    while True:
        nx, ny = px + dx, py + dy
        if nx < 0 or nx >= GW or ny < 0 or ny >= GH:
            break
        if base[ny, nx] == 1:
            break
        px, py = nx, ny
        path.append((px, py))
    return path


# Gravity indicator: 9x9 pixel cross in top-right corner, flush with edges
_SZ = 9
_MID = _SZ // 2
_ARM = 2  # arm width

_ARROW_CELLS: dict[int, list[tuple[int, int]]] = {
    UP: [(x, y) for y in range(_MID) for x in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)],
    DOWN: [(x, y) for y in range(_MID + 1, _SZ) for x in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)],
    LEFT: [(x, y) for x in range(_MID) for y in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)],
    RIGHT: [(x, y) for x in range(_MID + 1, _SZ) for y in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)],
}
_ARROW_ALL: list[tuple[int, int]] = []
for _cells in _ARROW_CELLS.values():
    _ARROW_ALL.extend(_cells)
_ARROW_ALL.extend(
    [
        (x, y)
        for x in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)
        for y in range(_MID - _ARM // 2, _MID + _ARM // 2 + 1)
    ]
)
_ARROW_ALL = list(set(_ARROW_ALL))


def _draw_gravity_indicator(canvas: np.ndarray, gravity: int) -> None:
    ox, oy = W - _SZ, 0
    for dx, dy in _ARROW_ALL:
        canvas[oy + dy, ox + dx] = C_ARROW_DIM
    for dx, dy in _ARROW_CELLS[gravity]:
        canvas[oy + dy, ox + dx] = C_ARROW_LIT


def _render(
    base: np.ndarray, player: list[int], goal_line: list, start_wall: list, kill_zone: list, gravity: int = DOWN
) -> np.ndarray:
    canvas = np.full((H, W), C_FLOOR, dtype=np.int8)

    for gy in range(GH):
        for gx in range(GW):
            if base[gy, gx] == 1:
                y0, x0 = gy * CELL, gx * CELL
                canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_WALL

    for sx, sy in start_wall:
        y0, x0 = sy * CELL, sx * CELL
        canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_START_WALL

    for kx, ky in kill_zone:
        y0, x0 = ky * CELL, kx * CELL
        canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_KILL

    for gx, gy in goal_line:
        y0, x0 = gy * CELL, gx * CELL
        canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_GOAL

    # Player
    px, py = player
    py0, px0 = py * CELL, px * CELL
    canvas[py0 : py0 + CELL, px0 : px0 + CELL] = C_PLAYER
    canvas[py0 + 1 : py0 + 3, px0 + 1 : px0 + 3] = C_FLOOR

    _draw_gravity_indicator(canvas, gravity)

    return canvas


def _build_level(spec: dict) -> Level:
    base = _parse_layout(spec["layout"])
    start_wall = spec.get("start_wall", [])
    for sx, sy in start_wall:
        base[sy, sx] = 1
    goal_line = spec.get("goal_line", [])
    for gx, gy in goal_line:
        base[gy, gx] = 1
    kill_zone = spec.get("kill_zone", [])
    for kx, ky in kill_zone:
        base[ky, kx] = 1
    grav = spec["gravity"]
    dx, dy = GRAVITY[grav]
    start = list(spec["start"])
    sx, sy = _apply_gravity(base, start[0], start[1], dx, dy)
    settled_start = [sx, sy]
    initial = _render(base, settled_start, goal_line, start_wall, kill_zone, grav)
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(W, H),
        sprites=[board],
        data={
            "layout": spec["layout"],
            "start": settled_start,
            "goal_line": goal_line,
            "start_wall": start_wall,
            "kill_zone": kill_zone,
            "gravity": grav,
        },
    )


class GravityFlip(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, W, H, background=C_FLOOR, letter_box=C_FLOOR, interfaces=[self._energy_bar])
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
        self._base = _parse_layout(level.get_data("layout"))
        self._start_wall = level.get_data("start_wall") or []
        for sx, sy in self._start_wall:
            self._base[sy, sx] = 1
        self._goal_line = level.get_data("goal_line") or []
        self._goal_set = {(int(g[0]), int(g[1])) for g in self._goal_line}
        for gx, gy in self._goal_line:
            self._base[gy, gx] = 1
        self._kill_zone = level.get_data("kill_zone") or []
        self._kill_set = {(int(k[0]), int(k[1])) for k in self._kill_zone}
        for kx, ky in self._kill_zone:
            self._base[ky, kx] = 1
        start = level.get_data("start")
        self._pos = [int(start[0]), int(start[1])]
        self._gravity = int(level.get_data("gravity"))
        self._board = level.get_sprites_by_name("board")[0]
        self._anim_path: list[tuple[int, int]] = []
        self._anim_idx = 0
        self._redraw()

    def _touching_goal(self) -> bool:
        px, py = self._pos
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (px + dx, py + dy) in self._goal_set:
                return True
        return False

    def _touching_kill(self) -> bool:
        px, py = self._pos
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (px + dx, py + dy) in self._kill_set:
                return True
        return False

    def _apply_terminal_failure(self) -> bool:
        if self._touching_kill():
            self.lose()
            return True
        return False

    def _redraw(self) -> None:
        self._board.pixels = _render(
            self._base, self._pos, self._goal_line, self._start_wall, self._kill_zone, self._gravity
        )

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        # Continue animation if in progress
        if self._anim_path:
            self._anim_idx += 1
            if self._anim_idx < len(self._anim_path):
                px, py = self._anim_path[self._anim_idx]
                self._pos = [px, py]
                self._redraw()
                return
            # Animation finished
            self._anim_path = []
            self._anim_idx = 0
            if not self._apply_terminal_failure() and self._touching_goal():
                self.next_level()
            self.complete_action()
            return

        # Start new action
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in GRAVITY:
            self._gravity = action_id
            dx, dy = GRAVITY[action_id]
            path = _build_path(self._base, self._pos[0], self._pos[1], dx, dy)
            if len(path) > 1:
                self._anim_path = path
                self._anim_idx = 1
                px, py = path[1]
                self._pos = [px, py]
                self._redraw()
                return
        self._apply_terminal_failure()
        self.complete_action()
