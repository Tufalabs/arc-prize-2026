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
    "pips_per_tick": 3,
    "pip_color": 12,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [12, 13],
}
ENERGY_CAPACITIES = [18, 36, 66]

GAME_ID = "make_all_same-0001"

GRID = 64
SCALE = 3  # pixels per size unit

SIZE_MIN = 1
SIZE_MAX = 6

COLOR_BG = 0
COLOR_BLOCK = 8  # unselected resizable blocks (cyan)
COLOR_SELECTED = 6  # currently selected block (magenta)
COLOR_TARGET = 4  # immovable target reference (yellow)

# 3x3 cell centers; index 0-7 are the 8 surrounding slots, center is target
_CELL_CENTERS = [(12, 12), (32, 12), (52, 12), (12, 32), (52, 32), (12, 52), (32, 52), (52, 52)]
_TARGET_CENTER = (32, 32)

_LEVELS = [
    (3, [0, 4], [1, 5]),
    (4, [0, 2, 5, 7], [2, 6, 1, 5]),
    (3, [0, 1, 2, 3, 4, 5, 6, 7], [1, 5, 2, 6, 4, 1, 5, 2]),
]


def _build_level() -> Level:
    floor = Sprite(
        pixels=np.full((GRID, GRID), COLOR_BG, dtype=np.int8),
        name="floor",
        x=0,
        y=0,
        layer=0,
        tags=["floor", "sys_click", "sys_every_pixel"],
        collidable=False,
    )
    return Level(grid_size=(GRID, GRID), sprites=[floor], data={})


def _draw_block(grid: np.ndarray, cx: int, cy: int, size: int, color: int) -> None:
    half = (size * SCALE) // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size * SCALE
    y1 = y0 + size * SCALE
    grid[max(0, y0) : min(GRID, y1), max(0, x0) : min(GRID, x1)] = color


class MakeAllSame(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._blocks: list[dict] = []
        self._target_size = 3
        self._selected: int | None = None
        self._floor: Sprite | None = None

        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(width=GRID, height=GRID, background=COLOR_BG, interfaces=[self._energy_bar])
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LEVELS],
            camera=camera,
            win_score=len(_LEVELS),
            available_actions=[1, 2, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        target_size, pos_indices, init_sizes = _LEVELS[self.level_index]
        self._target_size = target_size
        self._selected = None
        self._blocks = []
        for i, idx in enumerate(pos_indices):
            cx, cy = _CELL_CENTERS[idx]
            self._blocks.append({"cx": cx, "cy": cy, "size": init_sizes[i]})
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        self._draw()

    def _decode_click(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else None
        if not data:
            return None
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return None
        cell = self.camera.display_to_grid(raw_x, raw_y)
        if cell is None:
            return None
        return int(cell[0]), int(cell[1])

    def _hit_test(self, cx: int, cy: int) -> int | None:
        for i, b in enumerate(self._blocks):
            half = (b["size"] * SCALE) // 2
            x0 = b["cx"] - half
            y0 = b["cy"] - half
            x1 = x0 + b["size"] * SCALE
            y1 = y0 + b["size"] * SCALE
            if x0 <= cx < x1 and y0 <= cy < y1:
                return i
        return None

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id == int(GameAction.ACTION6.value):
            click = self._decode_click()
            if click is not None:
                hit = self._hit_test(*click)
                self._selected = hit

        elif action_id == int(GameAction.ACTION1.value):  # up = grow
            if self._selected is not None:
                b = self._blocks[self._selected]
                if b["size"] < SIZE_MAX:
                    b["size"] += 1

        elif action_id == int(GameAction.ACTION2.value):  # down = shrink
            if self._selected is not None:
                b = self._blocks[self._selected]
                if b["size"] > SIZE_MIN:
                    b["size"] -= 1

        self._draw()

        if all(b["size"] == self._target_size for b in self._blocks):
            self.next_level()

        self.complete_action()

    def _draw(self) -> None:
        if not self._floor:
            return
        grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)
        _draw_block(grid, *_TARGET_CENTER, self._target_size, COLOR_TARGET)
        for i, b in enumerate(self._blocks):
            color = COLOR_SELECTED if i == self._selected else COLOR_BLOCK
            _draw_block(grid, b["cx"], b["cy"], b["size"], color)
        self._floor.pixels = grid
