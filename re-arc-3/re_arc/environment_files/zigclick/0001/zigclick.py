from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
TIME_LIMIT = 150
NODE_SIZE = 3

COLOR_BG = 0
COLOR_TRACK = 1
COLOR_TIMER_EMPTY = 2
COLOR_TIMER_FILL = 3
COLOR_NODE_IDLE = 4
COLOR_NODE_DONE = 5
COLOR_NODE_EXPECTED = 6
COLOR_SEGMENT_OFF = 7
COLOR_SEGMENT_ON = 8

NODE_TOP_LEFTS = [(6, 7), (53, 14), (9, 22), (50, 30), (12, 38), (47, 46)]


def _make_track_pixels() -> np.ndarray:
    pixels = np.full((GRID_HEIGHT - 1, GRID_WIDTH), COLOR_BG, dtype=np.int8)

    centers = [(x + 1, y + 1) for x, y in NODE_TOP_LEFTS]
    for idx in range(len(centers) - 1):
        x1, y1 = centers[idx]
        x2, y2 = centers[idx + 1]

        x_start = min(x1, x2)
        x_end = max(x1, x2)
        pixels[y1 - 1, x_start : x_end + 1] = COLOR_TRACK

        y_start = min(y1, y2)
        y_end = max(y1, y2)
        pixels[y_start - 1 : y_end, x2] = COLOR_TRACK

    return pixels


def _segment_sprites() -> list[Sprite]:
    sprites: list[Sprite] = []
    centers = [(x + 1, y + 1) for x, y in NODE_TOP_LEFTS]
    for idx in range(len(centers) - 1):
        x1, y1 = centers[idx]
        x2, y2 = centers[idx + 1]
        transition = idx + 2

        x_start = min(x1, x2)
        width = abs(x2 - x1) + 1
        sprites.append(
            Sprite(
                pixels=np.full((1, width), COLOR_SEGMENT_OFF, dtype=np.int8),
                name=f"segment_{idx}_h",
                x=x_start,
                y=y1,
                layer=4,
                tags=["segment", f"to_{transition}"],
                collidable=False,
            )
        )

        y_start = min(y1, y2)
        height = abs(y2 - y1) + 1
        sprites.append(
            Sprite(
                pixels=np.full((height, 1), COLOR_SEGMENT_OFF, dtype=np.int8),
                name=f"segment_{idx}_v",
                x=x2,
                y=y_start,
                layer=4,
                tags=["segment", f"to_{transition}"],
                collidable=False,
            )
        )

    return sprites


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        Sprite(
            pixels=np.full((1, GRID_WIDTH), COLOR_TIMER_FILL, dtype=np.int8),
            name="timer_bar",
            x=0,
            y=0,
            layer=6,
            tags=["hud", "timer"],
            collidable=False,
        )
    )

    sprites.append(
        Sprite(pixels=_make_track_pixels(), name="zig_track", x=0, y=1, layer=1, tags=["track"], collidable=False)
    )

    for idx, (x, y) in enumerate(NODE_TOP_LEFTS, start=1):
        sprites.append(
            Sprite(
                pixels=np.full((NODE_SIZE, NODE_SIZE), COLOR_NODE_IDLE, dtype=np.int8),
                name=f"node_{idx}",
                x=x,
                y=y,
                layer=5,
                tags=["node", f"node_{idx}", "sys_click", "sys_every_pixel"],
                collidable=False,
            )
        )

    sprites.extend(_segment_sprites())

    return Level(
        name="Zigclick 0001",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": TIME_LIMIT, "sequence_length": len(NODE_TOP_LEFTS)},
    )


class Zigclick(ARCBaseGame):
    def __init__(self, seed: int = 0):
        level = _build_level()
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(
            game_id="zigclick-0001", levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed
        )

        self._time_limit = TIME_LIMIT
        self._time_remaining = TIME_LIMIT
        self._progress = 0
        self._nodes: list[Sprite] = []
        self._segments: list[tuple[Sprite, int]] = []

    def on_set_level(self, level: Level) -> None:
        self._time_limit = int(level.get_data("time_limit") or TIME_LIMIT)
        self._time_remaining = self._time_limit
        self._progress = 0

        self._nodes = sorted(
            level.get_sprites_by_tag("node"),
            key=lambda sprite: int(next(tag.split("_", 1)[1] for tag in sprite.tags if tag.startswith("node_"))),
        )

        self._segments = []
        for sprite in level.get_sprites_by_tag("segment"):
            to_step = next((tag for tag in sprite.tags if tag.startswith("to_")), None)
            if to_step is None:
                continue
            self._segments.append((sprite, int(to_step.split("_", 1)[1])))

        self._sync_nodes_and_segments()
        self._sync_timer()

    def _sync_timer(self) -> None:
        timer = self.current_level.get_sprites_by_name("timer_bar")
        if not timer:
            return
        filled = round((self._time_remaining / max(1, self._time_limit)) * GRID_WIDTH)
        filled = max(0, min(GRID_WIDTH, filled))
        pixels = np.full((1, GRID_WIDTH), COLOR_TIMER_EMPTY, dtype=np.int8)
        if filled > 0:
            pixels[:, :filled] = COLOR_TIMER_FILL
        timer[0].pixels = pixels

    def _sync_nodes_and_segments(self) -> None:
        expected = self._progress + 1
        for idx, sprite in enumerate(self._nodes, start=1):
            if idx <= self._progress:
                color = COLOR_NODE_DONE
            elif idx == expected:
                color = COLOR_NODE_EXPECTED
            else:
                color = COLOR_NODE_IDLE
            sprite.pixels = np.full((sprite.height, sprite.width), color, dtype=np.int8)

        for sprite, to_step in self._segments:
            color = COLOR_SEGMENT_ON if self._progress >= to_step else COLOR_SEGMENT_OFF
            sprite.pixels = np.full((sprite.height, sprite.width), color, dtype=np.int8)

    def _click_cell(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else None
        if not data:
            return None
        try:
            x = int(data.get("x", -1))
            y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return None
        return x, y

    @staticmethod
    def _in_bounds(x: int, y: int, sprite: Sprite) -> bool:
        return sprite.x <= x < sprite.x + sprite.width and sprite.y <= y < sprite.y + sprite.height

    def _clicked_node_index(self, x: int, y: int) -> int | None:
        for idx, sprite in enumerate(self._nodes, start=1):
            if self._in_bounds(x, y, sprite):
                return idx
        return None

    def step(self) -> None:
        click = self._click_cell()
        if click is not None:
            node_idx = self._clicked_node_index(click[0], click[1])
            if node_idx is not None:
                expected = self._progress + 1
                if node_idx == expected:
                    self._progress += 1
                else:
                    self._progress = 0

        self._time_remaining -= 1

        if self._progress >= len(self._nodes):
            self.next_level()
        elif self._time_remaining <= 0:
            self.lose()

        self._sync_nodes_and_segments()
        self._sync_timer()
        self.complete_action()
