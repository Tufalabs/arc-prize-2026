from __future__ import annotations

from dataclasses import dataclass

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
    "pips_per_tick": 1,
    "pip_color": 1,
    "spent_color": 15,
    "gap": 0,
    "margin": 0,
    "tier_colors": [1],
}
ENERGY_CAPACITIES = [6, 12, 24]

COLOR_FLOOR = 5  # black
COLOR_NEUTRAL = 2  # medium gray — unselected blocks

GRID_W = 64
GRID_H = 64

# Target / real-block colors: red, blue, green, yellow
COLORS = [8, 9, 14, 11]


@dataclass(frozen=True)
class BlockSpec:
    cells: tuple[tuple[int, int], ...]
    origin: tuple[int, int]
    real_color: int  # revealed on select / after placement
    layer: int


@dataclass(frozen=True)
class TargetSpec:
    cells: tuple[tuple[int, int], ...]
    origin: tuple[int, int]
    color: int


def _outer_contour(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """Outer ring (8-connected), 1 cell wide."""
    cell_set = set(cells)
    contour: set[tuple[int, int]] = set()
    for cx, cy in cells:
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            nb = (cx + dx, cy + dy)
            if nb not in cell_set:
                contour.add(nb)
    return contour


_SQ = ((0, 0), (1, 0), (0, 1), (1, 1))


def _spread_ys(n: int) -> list[int]:
    """Return n y-positions centered vertically on the 64-high grid."""
    gap = 14
    total = (n - 1) * gap
    y0 = (GRID_H - total) // 2
    return [y0 + i * gap for i in range(n)]


LEVEL_SPECS: list[tuple[str, list[BlockSpec], list[TargetSpec]]] = [
    # Level 1 — one block, one target
    ("Level 1", [BlockSpec(_SQ, (15, 31), COLORS[0], layer=10)], [TargetSpec(_SQ, (47, 31), COLORS[0])]),
    # Level 2 — two blocks, two targets (swapped order)
    (
        "Level 2",
        [
            BlockSpec(_SQ, (15, _spread_ys(2)[0]), COLORS[0], layer=10),
            BlockSpec(_SQ, (15, _spread_ys(2)[1]), COLORS[1], layer=10),
        ],
        [TargetSpec(_SQ, (47, _spread_ys(2)[0]), COLORS[1]), TargetSpec(_SQ, (47, _spread_ys(2)[1]), COLORS[0])],
    ),
    # Level 3 — four blocks, four targets (scrambled)
    (
        "Level 3",
        [
            BlockSpec(_SQ, (15, _spread_ys(4)[0]), COLORS[0], layer=10),
            BlockSpec(_SQ, (15, _spread_ys(4)[1]), COLORS[1], layer=10),
            BlockSpec(_SQ, (15, _spread_ys(4)[2]), COLORS[2], layer=10),
            BlockSpec(_SQ, (15, _spread_ys(4)[3]), COLORS[3], layer=10),
        ],
        [
            TargetSpec(_SQ, (47, _spread_ys(4)[0]), COLORS[2]),
            TargetSpec(_SQ, (47, _spread_ys(4)[1]), COLORS[3]),
            TargetSpec(_SQ, (47, _spread_ys(4)[2]), COLORS[0]),
            TargetSpec(_SQ, (47, _spread_ys(4)[3]), COLORS[1]),
        ],
    ),
]


def _build_level(spec: tuple[str, list[BlockSpec], list[TargetSpec]]) -> Level:
    name, blocks, targets = spec

    floor_pixels = [[COLOR_FLOOR] * GRID_W for _ in range(GRID_H)]
    sprites: list[Sprite] = [Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10)]

    # Target outlines
    for tidx, target in enumerate(targets):
        contour = _outer_contour(target.cells)
        for cx, cy in contour:
            gx, gy = target.origin[0] + cx, target.origin[1] + cy
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                sprites.append(
                    Sprite(
                        pixels=[[target.color]],
                        name=f"target_{tidx}_ol_{cx}_{cy}",
                        x=gx,
                        y=gy,
                        collidable=False,
                        layer=1,
                        tags=["target_outline", f"target_{tidx}"],
                    )
                )

    # Blocks — start neutral
    for bidx, block in enumerate(blocks):
        for cx, cy in block.cells:
            gx, gy = block.origin[0] + cx, block.origin[1] + cy
            sprites.append(
                Sprite(
                    pixels=[[COLOR_NEUTRAL]],
                    name=f"block_{bidx}_cell_{cx}_{cy}",
                    x=gx,
                    y=gy,
                    collidable=False,
                    layer=block.layer,
                    tags=["block_cell", f"block_{bidx}"],
                )
            )

    blocks_data = [
        {
            "cells": list(b.cells),
            "origin": list(b.origin),
            "home": list(b.origin),
            "real_color": b.real_color,
            "layer": b.layer,
        }
        for b in blocks
    ]
    targets_data = [{"cells": list(t.cells), "origin": list(t.origin), "color": t.color} for t in targets]

    return Level(
        name=name, sprites=sprites, grid_size=(GRID_W, GRID_H), data={"blocks": blocks_data, "targets": targets_data}
    )


class PickAndPlace(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, GRID_W, GRID_H, 5, 5, [self._energy_bar])
        super().__init__(
            "pick_and_place", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[6]
        )

    _held: int | None = None
    _locked: set[int]  # block indices correctly placed
    _block_at_target: dict[int, int]  # block_idx → target_idx
    _target_occupied: dict[int, int]  # target_idx → block_idx

    def on_set_level(self, _level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._held = None
        self._locked = set()
        self._block_at_target = {}
        self._target_occupied = {}

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        if self.action.id == GameAction.ACTION6:
            self._handle_click()
        self.complete_action()

    def _handle_click(self) -> None:
        data = self.action.data or {}
        pos = self.camera.display_to_grid(int(data.get("x", -1)), int(data.get("y", -1)))
        if pos is None:
            return
        gx, gy = pos

        if self._held is None:
            self._try_pick(gx, gy)
        else:
            self._try_place(gx, gy)

    # ---- pick ----

    def _try_pick(self, gx: int, gy: int) -> None:
        blocks = self.current_level.get_data("blocks") or []
        best: int | None = None
        best_layer = -1
        for bidx, b in enumerate(blocks):
            if bidx in self._locked:
                continue
            ox, oy = b["origin"]
            for cx, cy in b["cells"]:
                if ox + cx == gx and oy + cy == gy and b["layer"] > best_layer:
                    best = bidx
                    best_layer = b["layer"]
                    break
        if best is None:
            return

        # If this block was sitting at a wrong target, free that target
        if best in self._block_at_target:
            tid = self._block_at_target.pop(best)
            self._target_occupied.pop(tid, None)
            # Move block back home
            b = blocks[best]
            self._move_block_sprites(best, b["home"][0], b["home"][1])
            b["origin"][0] = b["home"][0]
            b["origin"][1] = b["home"][1]

        self._held = best
        # Reveal real color
        self._set_block_color(best, blocks[best]["real_color"])

    # ---- place ----

    def _try_place(self, gx: int, gy: int) -> None:
        held = self._held
        if held is None:
            return

        targets = self.current_level.get_data("targets") or []
        blocks = self.current_level.get_data("blocks") or []

        # Find which target was clicked
        clicked_tid: int | None = None
        for tidx, t in enumerate(targets):
            tox, toy = t["origin"]
            for cx, cy in t["cells"]:
                if tox + cx == gx and toy + cy == gy:
                    clicked_tid = tidx
                    break
            if clicked_tid is not None:
                break

        if clicked_tid is None:
            # Clicked empty space — deselect, revert to neutral
            self._deselect(held)
            return

        # If target already has a wrong block, send it home
        if clicked_tid in self._target_occupied:
            evicted = self._target_occupied.pop(clicked_tid)
            self._block_at_target.pop(evicted, None)
            eb = blocks[evicted]
            self._move_block_sprites(evicted, eb["home"][0], eb["home"][1])
            eb["origin"][0] = eb["home"][0]
            eb["origin"][1] = eb["home"][1]
            self._set_block_color(evicted, COLOR_NEUTRAL)

        # Place held block at target
        t = targets[clicked_tid]
        b = blocks[held]
        tox, toy = t["origin"]
        self._move_block_sprites(held, tox, toy)
        b["origin"][0] = tox
        b["origin"][1] = toy
        # Reveal real color (stays after placement)
        self._set_block_color(held, b["real_color"])
        self._held = None

        # Check correctness
        if b["real_color"] == t["color"]:
            self._locked.add(held)
            self._block_at_target[held] = clicked_tid
            self._target_occupied[clicked_tid] = held
            if len(self._locked) == len(targets):
                self.next_level()
        else:
            self._block_at_target[held] = clicked_tid
            self._target_occupied[clicked_tid] = held

    # ---- deselect ----

    def _deselect(self, bidx: int) -> None:
        self._set_block_color(bidx, COLOR_NEUTRAL)
        self._held = None

    # ---- sprite helpers ----

    def _set_block_color(self, bidx: int, color: int) -> None:
        for sprite in self.current_level.get_sprites_by_tag(f"block_{bidx}"):
            sprite.pixels[0][0] = color

    def _move_block_sprites(self, bidx: int, ox: int, oy: int) -> None:
        blocks = self.current_level.get_data("blocks") or []
        b = blocks[bidx]
        sprite_map: dict[tuple[int, int], Sprite] = {}
        for s in self.current_level.get_sprites_by_tag(f"block_{bidx}"):
            parts = s.name.split("_")
            sprite_map[(int(parts[3]), int(parts[4]))] = s
        for cx, cy in b["cells"]:
            sp = sprite_map.get((cx, cy))
            if sp is not None:
                sp.set_position(ox + cx, oy + cy)
