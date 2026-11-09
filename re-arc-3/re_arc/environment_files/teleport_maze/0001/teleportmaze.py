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
    "pips_per_tick": 3,
    "pip_color": 4,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [4, 13],
}
ENERGY_CAPACITIES = [18, 63, 42]

GAME_ID = "teleport_maze-0001"
GRID = 16
CELL = 4
PX = GRID * CELL  # 64

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)
SPACE = int(GameAction.ACTION5.value)

MOVE_DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

C_FLOOR = 0
C_WALL = 5
C_GOAL = 14
TP_COLORS = [8, 9, 15]  # red, blue, purple

LAYOUTS = [
    # Level 1: two rooms divided by vertical wall at x=7
    [
        "WWWWWWWWWWWWWWWW",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "WWWWWWWWWWWWWWWW",
    ],
    # Level 2: three rooms divided at x=5 and x=10
    [
        "WWWWWWWWWWWWWWWW",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "W....W....W....W",
        "WWWWWWWWWWWWWWWW",
    ],
    # Level 3: 2x2 rooms
    [
        "WWWWWWWWWWWWWWWW",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "WWWWWWWWWWWWWWWW",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "W......W.......W",
        "WWWWWWWWWWWWWWWW",
    ],
]

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "layout_idx": 0,
        "start": [2, 7],
        "start_color": 0,
        "goal": [13, 7],
        "teleporters": [[[5, 7], [10, 7], 0]],
    },
    {
        "name": "Level 2",
        "layout_idx": 1,
        "start": [2, 7],
        "start_color": 0,
        "goal": [13, 7],
        "teleporters": [[[3, 3], [7, 3], 1], [[8, 11], [12, 11], 0]],
    },
    {
        "name": "Level 3",
        "layout_idx": 2,
        "start": [2, 2],
        "start_color": 1,
        "goal": [13, 12],
        "teleporters": [[[4, 4], [13, 2], 2], [[4, 6], [4, 12], 0], [[4, 14], [10, 12], 1]],
    },
]


def _parse_layout(lines: list[str]) -> np.ndarray:
    grid = np.zeros((GRID, GRID), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _render(
    base: np.ndarray, player_pos: list[int], player_color: int, goal: list[int], teleporters: list
) -> np.ndarray:
    canvas = np.full((PX, PX), C_FLOOR, dtype=np.int8)

    for gy in range(GRID):
        for gx in range(GRID):
            if base[gy, gx] == 1:
                y0, x0 = gy * CELL, gx * CELL
                canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_WALL

    for pos1, pos2, ci in teleporters:
        tc = TP_COLORS[ci]
        for tx, ty in [pos1, pos2]:
            canvas[ty * CELL + 1 : ty * CELL + 3, tx * CELL + 1 : tx * CELL + 3] = tc

    gx, gy = goal
    canvas[gy * CELL + 1 : gy * CELL + 3, gx * CELL + 1 : gx * CELL + 3] = C_GOAL

    px, py = player_pos
    py0, px0 = py * CELL, px * CELL
    canvas[py0 : py0 + CELL, px0 : px0 + CELL] = TP_COLORS[player_color]
    canvas[py0 + 1 : py0 + 3, px0 + 1 : px0 + 3] = C_FLOOR

    # Color legend in top-left corner
    used_colors = sorted({ci for _, _, ci in teleporters})
    for i, ci in enumerate(used_colors):
        sx = 1 + i * CELL
        canvas[1:CELL, sx : sx + CELL - 1] = TP_COLORS[ci]

    return canvas


def _build_level(spec: dict) -> Level:
    base = _parse_layout(LAYOUTS[spec["layout_idx"]])
    initial = _render(base, spec["start"], spec["start_color"], spec["goal"], spec["teleporters"])
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(PX, PX),
        sprites=[board],
        data={
            "layout_idx": spec["layout_idx"],
            "start": spec["start"],
            "start_color": spec["start_color"],
            "goal": spec["goal"],
            "teleporters": spec["teleporters"],
        },
    )


class TeleportMaze(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, PX, PX, background=C_FLOOR, letter_box=C_FLOOR, interfaces=[self._energy_bar])
        super().__init__(
            GAME_ID,
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[UP, DOWN, LEFT, RIGHT, SPACE],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._base = _parse_layout(LAYOUTS[int(level.get_data("layout_idx"))])
        start = level.get_data("start")
        self._pos = [int(start[0]), int(start[1])]
        self._color = int(level.get_data("start_color"))
        goal = level.get_data("goal")
        self._goal = [int(goal[0]), int(goal[1])]
        self._teleporters = level.get_data("teleporters") or []
        self._available_colors = sorted({ci for _, _, ci in self._teleporters})
        self._board = level.get_sprites_by_name("board")[0]
        self._redraw()

    def _redraw(self) -> None:
        self._board.pixels = _render(self._base, self._pos, self._color, self._goal, self._teleporters)

    def _try_move(self, dx: int, dy: int) -> None:
        nx, ny = self._pos[0] + dx, self._pos[1] + dy
        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            return
        if self._base[ny, nx] == 1:
            return
        self._pos = [nx, ny]

        for pos1, pos2, ci in self._teleporters:
            if ci != self._color:
                continue
            if nx == pos1[0] and ny == pos1[1]:
                self._pos = [int(pos2[0]), int(pos2[1])]
                break
            if nx == pos2[0] and ny == pos2[1]:
                self._pos = [int(pos1[0]), int(pos1[1])]
                break

        self._redraw()

        if self._pos[0] == self._goal[0] and self._pos[1] == self._goal[1]:
            self.next_level()

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            self._try_move(dx, dy)
        elif action_id == SPACE and len(self._available_colors) > 1:
            idx = self._available_colors.index(self._color)
            self._color = self._available_colors[(idx + 1) % len(self._available_colors)]
            self._redraw()
        self.complete_action()
