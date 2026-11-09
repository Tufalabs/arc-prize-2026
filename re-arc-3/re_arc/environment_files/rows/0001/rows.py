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
    "pips_per_tick": 2,
    "pip_color": 4,
    "spent_color": 0,
    "gap": 0,
    "margin": 0,
    "tier_colors": [4],
}
ENERGY_CAPACITIES = [15, 27, 24, 33, 42, 39, 48, 57]

BACKGROUND_COLOR = 0
OFF_COLOR = 5
ON_COLOR = 11
TARGET_OFF_COLOR = 4
TARGET_BORDER_COLOR = 1
TARGET_ON_COLOR = 11
MODE_COLORS = {
    0: 10,  # row mode
    1: 15,  # column mode
}


def _mode_pixels(mode: int) -> list[list[int]]:
    b = MODE_COLORS[0]  # 10 blue background
    p = MODE_COLORS[1]  # 15 purple active indicator
    if mode == 0:  # row: horizontal bar in purple
        return [[b, b, b], [p, p, p], [b, b, b]]
    # column: vertical bar in purple
    return [[b, p, b], [b, p, b], [b, p, b]]


LEVEL_SPECS: list[tuple[str, int, int]] = [
    ("Level 1", 3, 4),
    ("Level 2", 3, 6),
    ("Level 3", 4, 6),
    ("Level 4", 4, 8),
    ("Level 5", 4, 10),
    ("Level 6", 5, 10),
    ("Level 7", 5, 12),
    ("Level 8", 5, 14),
]


def _bit_index(x: int, y: int, n: int) -> int:
    return y * n + x


def _toggle_row(mask: int, n: int, row: int) -> int:
    out = int(mask)
    for x in range(n):
        out ^= 1 << _bit_index(x, row, n)
    return out


def _toggle_col(mask: int, n: int, col: int) -> int:
    out = int(mask)
    for y in range(n):
        out ^= 1 << _bit_index(col, y, n)
    return out


def _build_recipe(n: int, steps: int, seed: int):
    recipe = []
    mode = 0
    for i in range(steps):
        if ((seed + i) % 3) == 0:
            mode ^= 1
        index = (seed * 5 + i * 7) % n
        recipe.append({"mode": int(mode), "index": int(index)})
    return recipe


def _apply_recipe(start_mask: int, n: int, recipe: list[dict]) -> int:
    mask = int(start_mask)
    for step in recipe:
        mode = int(step["mode"])
        idx = int(step["index"])
        mask = _toggle_row(mask, n, idx) if mode == 0 else _toggle_col(mask, n, idx)
    return int(mask)


def _build_level(spec: tuple[str, int, int], seed: int) -> Level:
    name, n, steps = spec
    logical_w = n + 2
    logical_h = (2 * n) + 3
    cell = min(64 // logical_w, 64 // logical_h)
    total_w = logical_w * cell
    total_h = logical_h * cell
    ox = (64 - total_w) // 2
    oy = (64 - total_h) // 2

    start_mask = 0
    start_mode = 0
    recipe = _build_recipe(n, steps, seed)
    target_mask = _apply_recipe(start_mask, n, recipe)

    if target_mask == start_mask:
        recipe.append({"mode": 0, "index": n // 2})
        target_mask = _apply_recipe(start_mask, n, recipe)

    floor_pixels = [[BACKGROUND_COLOR] * 64 for _ in range(64)]

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(
            pixels=_mode_pixels(start_mode),
            name="mode_indicator",
            x=0,
            y=0,
            collidable=False,
            layer=8,
            tags=["mode_indicator"],
            scale=max(2, cell * 2 // 3),
        ),
    ]

    # Current board
    for y in range(n):
        for x in range(n):
            sprites.append(
                Sprite(
                    pixels=[[OFF_COLOR]],
                    name=f"cell_{x}_{y}",
                    x=ox + (x + 1) * cell,
                    y=oy + (y + 1) * cell,
                    collidable=False,
                    layer=6,
                    tags=["cell"],
                    scale=cell,
                )
            )

    # Target board border (2px gray around target area)
    border_w = n * cell + 4
    border_h = n * cell + 4
    border_pixels = [[TARGET_BORDER_COLOR] * border_w for _ in range(border_h)]
    sprites.append(
        Sprite(
            pixels=border_pixels,
            name="target_border",
            x=ox + cell - 2,
            y=oy + (n + 2) * cell - 2,
            collidable=False,
            layer=2,
        )
    )

    # Target board
    for y in range(n):
        for x in range(n):
            bit = (target_mask >> _bit_index(x, y, n)) & 1
            sprites.append(
                Sprite(
                    pixels=[[TARGET_ON_COLOR if bit else TARGET_OFF_COLOR]],
                    name=f"target_{x}_{y}",
                    x=ox + (x + 1) * cell,
                    y=oy + (n + 2 + y) * cell,
                    collidable=False,
                    layer=4,
                    tags=["target"],
                    scale=cell,
                )
            )

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(64, 64),
        data={
            "size": int(n),
            "cell": int(cell),
            "ox": int(ox),
            "oy": int(oy),
            "start_mode": int(start_mode),
            "start_mask": int(start_mask),
            "target_mask": int(target_mask),
            "recipe": [{"mode": int(step["mode"]), "index": int(step["index"])} for step in recipe],
        },
    )


class Rows(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 191) for idx, spec in enumerate(LEVEL_SPECS)]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, 64, 64, BACKGROUND_COLOR, BACKGROUND_COLOR, [self._energy_bar])
        super().__init__("rows", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[5, 6])

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._n = int(level.get_data("size"))
        self._cell = int(level.get_data("cell"))
        self._ox = int(level.get_data("ox"))
        self._oy = int(level.get_data("oy"))
        self._mode = int(level.get_data("start_mode") or 0) % 2
        self._mask = int(level.get_data("start_mask") or 0)
        self._target_mask = int(level.get_data("target_mask") or 0)

        self._indicator = self.current_level.get_sprites_by_name("mode_indicator")[0]
        self._cells = {
            ((int(sprite.x) - self._ox) // self._cell - 1, (int(sprite.y) - self._oy) // self._cell - 1): sprite
            for sprite in self.current_level.get_sprites_by_tag("cell")
        }

        self._sync_visuals()

    def _sync_visuals(self) -> None:
        self._indicator.pixels = _mode_pixels(self._mode)
        for y in range(self._n):
            for x in range(self._n):
                bit = (self._mask >> _bit_index(x, y, self._n)) & 1
                self._cells[(x, y)].pixels[0][0] = ON_COLOR if bit else OFF_COLOR

    def _toggle_mode(self) -> None:
        self._mode ^= 1
        self._sync_visuals()

    def _click_board(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return

        gx = (int(grid_pos[0]) - self._ox) // self._cell - 1
        gy = (int(grid_pos[1]) - self._oy) // self._cell - 1
        if gx < 0 or gy < 0 or gx >= self._n or gy >= self._n:
            return

        if self._mode == 0:
            self._mask = _toggle_row(self._mask, self._n, gy)
        else:
            self._mask = _toggle_col(self._mask, self._n, gx)

        self._sync_visuals()
        if self._mask == self._target_mask:
            self.next_level()

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action = self.action.id
        if action == GameAction.ACTION5:
            self._toggle_mode()
        elif action == GameAction.ACTION6:
            self._click_board()
        self.complete_action()
