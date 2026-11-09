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
    "pip_width": 1,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 2,
    "spent_color": 15,
    "gap": 0,
    "margin": 0,
    "tier_colors": [2],
}
ENERGY_CAPACITIES = [36, 39, 39]

GAME_ID = "catch_eggs-0001"

GRID = 64
TILE = 4
TILES = GRID // TILE

COLOR_BG = 5
COLOR_PLAYER = 12
COLOR_COOP = 3
COLOR_EGG = 0
COLOR_CRASH = 8
COLOR_WALL = 7

PLAYER_Y = TILES - 2

DELTAS = {int(GameAction.ACTION2.value): 0, int(GameAction.ACTION3.value): -1, int(GameAction.ACTION4.value): 1}

_LEVELS = [
    {"player_x": 8, "eggs": [(5, 9), (10, 3), (8, 1)]},
    {"player_x": 3, "coops": [10], "eggs": [(3, 9), (12, 8), (5, 6), (10, 3), (8, 1)]},
    {
        "player_x": 8,
        "coops": [3, 12],
        "eggs": [(2, 10), (4, 6), (3, 2), (7, 8), (9, 4), (4, 1), (10, 3), (13, 9), (11, 5), (12, 1)],
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


class CatchEggs(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._px = 0
        self._coops: list[int] = []
        self._eggs: list[list[int]] = []
        self._total_eggs = 0
        self._caught = 0
        self._crashed_egg: list[int] | None = None
        self._floor: Sprite | None = None

        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(width=GRID, height=GRID, background=COLOR_BG, interfaces=[self._energy_bar])
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LEVELS],
            camera=camera,
            win_score=len(_LEVELS),
            available_actions=[2, 3, 4],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        cfg = _LEVELS[self.level_index]
        self._px = cfg["player_x"]
        self._coops = list(cfg.get("coops", []))
        self._eggs = [[x, y] for x, y in cfg["eggs"]]
        self._total_eggs = len(cfg["eggs"])
        self._caught = 0
        self._crashed_egg = None
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        self._draw()

    def _coop_step(self) -> None:
        if not self._coops or not self._eggs:
            return
        for i, cx in enumerate(self._coops):
            best_egg = min(self._eggs, key=lambda e, c=cx: abs(e[0] - c) + (PLAYER_Y - e[1]))  # type: ignore[arg-type]
            if best_egg[0] < cx:
                self._coops[i] -= 1
            elif best_egg[0] > cx:
                self._coops[i] += 1

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
        delta = DELTAS.get(action_id, 0)
        nx = self._px + delta
        if 1 <= nx <= TILES - 2:
            self._px = nx

        self._coop_step()

        for egg in self._eggs:
            egg[1] += 1

        remaining = []
        for egg in self._eggs:
            if egg[1] >= PLAYER_Y:
                if egg[0] == self._px or egg[0] in self._coops:
                    self._caught += 1
                else:
                    self._crashed_egg = egg
                    self._draw()
                    self.lose()
                    self.complete_action()
                    return
            else:
                remaining.append(egg)
        self._eggs = remaining

        if self._caught >= self._total_eggs:
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

        for egg in self._eggs:
            ex, ey = egg
            y0, x0 = ey * TILE, ex * TILE
            grid[y0 : y0 + TILE, x0 : x0 + TILE] = COLOR_EGG

        if self._crashed_egg is not None:
            cx, cy = self._crashed_egg
            y0, x0 = cy * TILE, cx * TILE
            grid[y0 : y0 + TILE, x0 : x0 + TILE] = COLOR_CRASH

        for cx in self._coops:
            cy0, cx0 = PLAYER_Y * TILE, cx * TILE
            grid[cy0 : cy0 + TILE, cx0 : cx0 + TILE] = COLOR_COOP

        py0, px0 = PLAYER_Y * TILE, self._px * TILE
        grid[py0 : py0 + TILE, px0 : px0 + TILE] = COLOR_PLAYER

        self._floor.pixels = grid
