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
    "rows": 2,
    "pip_width": 2,
    "actions_per_tick": 1,
    "pips_per_tick": 3,
    "pip_color": 6,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [6, 11, 13, 4],
}
ENERGY_CAPACITIES = [63, 51, 78]

GAME_ID = "light_out-0001"
VP = 64
CELL = 4
GW = GH = 16

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)
SPACE = int(GameAction.ACTION5.value)

C_DARK = 5
C_FLOOR = 0
C_WALL = 1
C_PLAYER = 9
C_GOAL = 14

DIRS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

FLASH_FRAMES = 2

LAYOUT_1 = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
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
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WW............WW",
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LAYOUT_3 = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "WW..WW......WWWW",
    "WW..WW......WWWW",
    "WW..WW..WW....WW",
    "WW......WW....WW",
    "WW......WW....WW",
    "WWWWWW..WW....WW",
    "WW......WW....WW",
    "WW..WWWWWW....WW",
    "WW..........WWWW",
    "WW..........WWWW",
    "WWWWWW..WW....WW",
    "WW..........WWWW",
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LEVEL_SPECS: list[dict] = [
    {"name": "Level 1", "layout": LAYOUT_1, "start": [3, 12], "goal": [12, 3], "show_player": True, "show_goal": False},
    {"name": "Level 2", "layout": LAYOUT_2, "start": [3, 3], "goal": [7, 13], "show_player": False, "show_goal": False},
    {
        "name": "Level 3",
        "layout": LAYOUT_3,
        "start": [2, 2],
        "goal": [11, 12],
        "show_player": False,
        "show_goal": False,
    },
]


def _parse_layout(lines: list[str]) -> np.ndarray:
    grid = np.zeros((GH, GW), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _render(
    grid: np.ndarray, player: tuple[int, int], goal: tuple[int, int], lit: bool, show_player: bool, show_goal: bool
) -> np.ndarray:
    canvas = np.full((VP, VP), C_DARK, dtype=np.int8)
    if lit:
        for gy in range(GH):
            for gx in range(GW):
                y0, x0 = gy * CELL, gx * CELL
                if grid[gy, gx] == 1:
                    canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_WALL
                else:
                    canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_FLOOR
    # Draw goal as diamond shape
    if show_goal or lit:
        gx, gy = goal
        y0, x0 = gy * CELL, gx * CELL
        canvas[y0, x0 + 1 : x0 + 3] = C_GOAL
        canvas[y0 + 1, x0 : x0 + 4] = C_GOAL
        canvas[y0 + 2, x0 : x0 + 4] = C_GOAL
        canvas[y0 + 3, x0 + 1 : x0 + 3] = C_GOAL
    # Draw player as diamond shape
    if show_player or lit:
        px, py = player
        y0, x0 = py * CELL, px * CELL
        bg = C_DARK if not lit else C_FLOOR
        canvas[y0, x0 + 1 : x0 + 3] = C_PLAYER
        canvas[y0 + 1, x0 : x0 + 4] = C_PLAYER
        canvas[y0 + 2, x0 : x0 + 4] = C_PLAYER
        canvas[y0 + 3, x0 + 1 : x0 + 3] = C_PLAYER
        canvas[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = bg
    return canvas


def _build_level(spec: dict) -> Level:
    initial = np.full((VP, VP), C_DARK, dtype=np.int8)
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(VP, VP),
        sprites=[board],
        data={
            "layout": spec["layout"],
            "start": spec["start"],
            "goal": spec["goal"],
            "show_player": spec["show_player"],
            "show_goal": spec["show_goal"],
        },
    )


class LightOut(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(s) for s in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, VP, VP, background=C_DARK, letter_box=C_DARK, interfaces=[self._energy_bar])
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
        self._grid = _parse_layout(level.get_data("layout"))
        start = level.get_data("start")
        self._px, self._py = int(start[0]), int(start[1])
        goal = level.get_data("goal")
        self._goal = (int(goal[0]), int(goal[1]))
        self._show_player = bool(level.get_data("show_player"))
        self._show_goal = bool(level.get_data("show_goal"))
        self._board = level.get_sprites_by_name("board")[0]
        self._lit = False
        self._flash_remaining = 0
        self._redraw()

    def _redraw(self) -> None:
        self._board.pixels = _render(
            self._grid, (self._px, self._py), self._goal, self._lit, self._show_player, self._show_goal
        )

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        # Flash animation in progress
        if self._flash_remaining > 0:
            self._flash_remaining -= 1
            if self._flash_remaining == 0:
                self._lit = False
                self._redraw()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id == SPACE:
            self._lit = True
            self._flash_remaining = FLASH_FRAMES
            self._redraw()
            self.complete_action()
            return

        if action_id in DIRS:
            dx, dy = DIRS[action_id]
            nx, ny = self._px + dx, self._py + dy
            if 0 <= nx < GW and 0 <= ny < GH and self._grid[ny, nx] == 0:
                self._px, self._py = nx, ny
                self._redraw()
                if (self._px, self._py) == self._goal:
                    self.next_level()

        self.complete_action()
