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
    "pips_per_tick": 3,
    "pip_color": 12,
    "spent_color": 0,
    "gap": 0,
    "margin": 0,
    "tier_colors": [12, 11],
}
ENERGY_CAPACITIES = [15, 24, 60]

GAME_ID = "mirror_push-0001"
VP = 64
CELL = 4
GW = GH = 16

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)

C_FLOOR = 0
C_RED = 2
C_GREEN = 3
C_YELLOW = 4
C_WALL = 5
C_PINK = 6
C_ORANGE = 7
C_DARK = 8
C_PLAYER = 9

DIRS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

RIGHT_COL0 = 9
LEFT_COL0 = 1
ROW0 = 2
INTERIOR_W = 6
INTERIOR_H = 12

LAYOUT = [
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "W......WW......W",
    "WWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWW",
]

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "layout": LAYOUT,
        "start": [3, 12],
        "left_blocks": [{"x": 3, "y": 10, "cells": [[0, 0], [0, 1], [1, 1]], "color": C_ORANGE, "z": 0}],
        "right_blocks": [{"x": 10, "y": 8, "cells": [[1, 0], [0, 1], [1, 1]], "color": C_ORANGE, "z": 0}],
        "precedence": [C_ORANGE],
    },
    {
        "name": "Level 2",
        "layout": LAYOUT,
        "start": [4, 10],
        "left_blocks": [
            {"x": 3, "y": 5, "cells": [[0, 0], [1, 0]], "color": C_GREEN, "z": 0},
            {"x": 4, "y": 7, "cells": [[0, 0], [0, 1]], "color": C_PINK, "z": 1},
        ],
        "right_blocks": [
            {"x": 10, "y": 5, "cells": [[0, 0], [1, 0]], "color": C_GREEN, "z": 0},
            {"x": 11, "y": 5, "cells": [[0, 0], [0, 1]], "color": C_PINK, "z": 1},
        ],
        "precedence": [C_GREEN, C_PINK],
    },
    {
        "name": "Level 3",
        "layout": LAYOUT,
        "start": [2, 2],
        "left_blocks": [
            {"x": 2, "y": 3, "cells": [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]], "color": C_YELLOW, "z": 0},
            {"x": 4, "y": 8, "cells": [[0, 0], [1, 0], [1, 1], [1, 2]], "color": C_RED, "z": 1},
            {"x": 2, "y": 12, "cells": [[0, 0], [1, 0], [2, 0]], "color": C_GREEN, "z": 2},
        ],
        "right_blocks": [
            {"x": 11, "y": 6, "cells": [[2, 0], [2, 1], [0, 2], [1, 2], [2, 2]], "color": C_YELLOW, "z": 0},
            {"x": 11, "y": 8, "cells": [[0, 0], [1, 0], [0, 1], [0, 2]], "color": C_RED, "z": 1},
            {"x": 11, "y": 10, "cells": [[0, 0], [1, 0], [2, 0]], "color": C_GREEN, "z": 2},
        ],
        "precedence": [C_YELLOW, C_RED, C_GREEN],
    },
]


def _parse_layout(lines: list[str]) -> np.ndarray:
    grid = np.zeros((GH, GW), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _block_covers(block: dict, x: int, y: int) -> bool:
    bx, by = block["x"], block["y"]
    return any(bx + dx == x and by + dy == y for dx, dy in block["cells"])


def _render_pattern(blocks: list[dict], col0: int) -> np.ndarray:
    pat = np.zeros((INTERIOR_H, INTERIOR_W), dtype=np.int8)
    for b in sorted(blocks, key=lambda b: b["z"]):
        for dx, dy in b["cells"]:
            lx = b["x"] - col0 + dx
            ly = b["y"] - ROW0 + dy
            if 0 <= lx < INTERIOR_W and 0 <= ly < INTERIOR_H:
                pat[ly, lx] = b["color"]
    return pat


def _render(
    grid: np.ndarray, player: tuple[int, int], left_blocks: list[dict], right_blocks: list[dict], precedence: list[int]
) -> np.ndarray:
    canvas = np.full((VP, VP), C_WALL, dtype=np.int8)
    for gy in range(GH):
        for gx in range(GW):
            if grid[gy, gx] == 0:
                y0, x0 = gy * CELL, gx * CELL
                color = C_DARK if gx >= RIGHT_COL0 else C_FLOOR
                canvas[y0 : y0 + CELL, x0 : x0 + CELL] = color
    for block_list in (right_blocks, left_blocks):
        for b in sorted(block_list, key=lambda b: b["z"]):
            for dx, dy in b["cells"]:
                y0 = (b["y"] + dy) * CELL
                x0 = (b["x"] + dx) * CELL
                canvas[y0 : y0 + CELL, x0 : x0 + CELL] = b["color"]
    px, py = player
    y0, x0 = py * CELL, px * CELL
    canvas[y0, x0 + 1 : x0 + 3] = C_PLAYER
    canvas[y0 + 1, x0 : x0 + 4] = C_PLAYER
    canvas[y0 + 2, x0 : x0 + 4] = C_PLAYER
    canvas[y0 + 3, x0 + 1 : x0 + 3] = C_PLAYER
    canvas[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = C_FLOOR
    for i, color in enumerate(precedence):
        px0 = 2 + i
        py0 = 1 + i
        canvas[py0 : py0 + 3, px0 : px0 + 3] = color
    return canvas


def _build_level(spec: dict) -> Level:
    initial = np.full((VP, VP), C_WALL, dtype=np.int8)
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(VP, VP),
        sprites=[board],
        data={
            "layout": spec["layout"],
            "start": spec["start"],
            "left_blocks": spec["left_blocks"],
            "right_blocks": spec["right_blocks"],
            "precedence": spec["precedence"],
        },
    )


class MirrorPush(ARCBaseGame):
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
        start = level.get_data("start")
        self._px, self._py = int(start[0]), int(start[1])
        self._blocks: list[dict] = [
            {**b, "cells": [tuple(c) for c in b["cells"]]} for b in level.get_data("left_blocks")
        ]
        self._right_blocks: list[dict] = [
            {**b, "cells": [tuple(c) for c in b["cells"]]} for b in level.get_data("right_blocks")
        ]
        self._precedence: list[int] = list(level.get_data("precedence"))
        self._board = level.get_sprites_by_name("board")[0]
        self._redraw()

    def _redraw(self) -> None:
        self._board.pixels = _render(
            self._grid, (self._px, self._py), self._blocks, self._right_blocks, self._precedence
        )

    def _can_push(self, block: dict, dx: int, dy: int) -> bool:
        for cdx, cdy in block["cells"]:
            nx = block["x"] + cdx + dx
            ny = block["y"] + cdy + dy
            if nx < 0 or nx >= GW or ny < 0 or ny >= GH or self._grid[ny, nx] == 1:
                return False
        return True

    def _check_win(self) -> bool:
        left_pat = _render_pattern(self._blocks, LEFT_COL0)
        right_pat = _render_pattern(self._right_blocks, RIGHT_COL0)
        return bool(np.array_equal(left_pat, np.fliplr(right_pat)))

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in DIRS:
            dx, dy = DIRS[action_id]
            nx, ny = self._px + dx, self._py + dy
            if 0 <= nx < GW and 0 <= ny < GH and self._grid[ny, nx] == 0:
                blocks_here = [b for b in self._blocks if _block_covers(b, nx, ny)]
                if blocks_here:
                    if len(blocks_here) == 1 and self._can_push(blocks_here[0], dx, dy):
                        blocks_here[0]["x"] += dx
                        blocks_here[0]["y"] += dy
                        self._px, self._py = nx, ny
                        self._redraw()
                        if self._check_win():
                            self.next_level()
                else:
                    self._px, self._py = nx, ny
                    self._redraw()
        self.complete_action()
