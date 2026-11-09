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
    "pip_width": 2,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 14,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [14],
}
ENERGY_CAPACITIES = [27, 33, 18]

GAME_ID = "dodge_ball-0001"

GRID = 64
TILE = 4
TILES = GRID // TILE

COLOR_BG = 0
COLOR_PLAYER = 1
COLOR_ENEMY = 2
COLOR_DIAMOND = 4
COLOR_WALL = 5
COLOR_OBSTACLE = 8

DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

_LEVELS = [
    {"player": (2, 13), "enemy": (13, 2), "diamonds": [(7, 9)], "obstacles": []},
    {
        "player": (13, 2),
        "enemy": (13, 7),
        "diamonds": [(12, 12)],
        "obstacles": [(7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8), (9, 6), (9, 7), (9, 8)],
    },
    {
        "player": (2, 7),
        "enemy": (6, 2),
        "enemy2": (11, 3),
        "diamonds": [(3, 2)],
        "obstacles": [
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 1),
            (5, 2),
            (5, 3),
            (9, 9),
            (9, 10),
            (9, 11),
            (10, 9),
            (10, 10),
            (10, 11),
        ],
    },
]


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


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


class DodgeBall(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._px = 0
        self._py = 0
        self._enemies: list[list[int]] = []
        self._diamonds: list[tuple[int, int]] = []
        self._obstacles: set[tuple[int, int]] = set()
        self._floor: Sprite | None = None

        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(width=GRID, height=GRID, background=COLOR_BG, interfaces=[self._energy_bar])
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LEVELS],
            camera=camera,
            win_score=len(_LEVELS),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _apply_level_config(self) -> None:
        cfg = _LEVELS[self.level_index]
        self._px, self._py = cfg["player"]
        self._enemies = self._load_enemies(cfg)
        self._diamonds = list(cfg["diamonds"])
        self._obstacles = set(cfg["obstacles"])

    def _load_enemies(self, cfg: dict) -> list[list[int]]:
        enemies = [list(cfg["enemy"])]
        i = 2
        while f"enemy{i}" in cfg:
            enemies.append(list(cfg[f"enemy{i}"]))
            i += 1
        return enemies

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._apply_level_config()
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        self._draw()

    def _can_move_to(self, x: int, y: int) -> bool:
        return 1 <= x <= TILES - 2 and 1 <= y <= TILES - 2 and (x, y) not in self._obstacles

    def _enemy_step(self) -> None:
        for e in self._enemies:
            best_d = _manhattan(e[0], e[1], self._px, self._py)
            best = (e[0], e[1])
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = e[0] + dx, e[1] + dy
                if self._can_move_to(nx, ny):
                    d = _manhattan(nx, ny, self._px, self._py)
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
            e[0], e[1] = best

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        delta = DELTAS.get(action_id)
        if delta:
            nx, ny = self._px + delta[0], self._py + delta[1]
            if self._can_move_to(nx, ny):
                self._px, self._py = nx, ny

        self._enemy_step()

        if any(e[0] == self._px and e[1] == self._py for e in self._enemies):
            self.lose()
            self.complete_action()
            return

        self._diamonds = [(dx, dy) for dx, dy in self._diamonds if (dx, dy) != (self._px, self._py)]

        if not self._diamonds:
            self._draw()
            self.next_level()
            self.complete_action()
            return

        self._draw()
        self.complete_action()

    def _draw(self) -> None:
        if not self._floor:
            return
        grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)

        for x in range(TILES):
            for y_wall in (0, TILES - 1):
                wy, wx = y_wall * TILE, x * TILE
                grid[wy : wy + TILE, wx : wx + TILE] = COLOR_WALL
        for y in range(TILES):
            for x_wall in (0, TILES - 1):
                wy, wx = y * TILE, x_wall * TILE
                grid[wy : wy + TILE, wx : wx + TILE] = COLOR_WALL

        for ox, oy in self._obstacles:
            y0, x0 = oy * TILE, ox * TILE
            grid[y0 : y0 + TILE, x0 : x0 + TILE] = COLOR_OBSTACLE

        for dx, dy in self._diamonds:
            y0, x0 = dy * TILE, dx * TILE
            grid[y0 + 1 : y0 + TILE - 1, x0 + 1 : x0 + TILE - 1] = COLOR_DIAMOND

        for e in self._enemies:
            ey0, ex0 = e[1] * TILE, e[0] * TILE
            grid[ey0 : ey0 + TILE, ex0 : ex0 + TILE] = COLOR_ENEMY

        py0, px0 = self._py * TILE, self._px * TILE
        grid[py0 : py0 + TILE, px0 : px0 + TILE] = COLOR_PLAYER

        self._floor.pixels = grid
