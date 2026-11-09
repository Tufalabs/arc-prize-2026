from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "path_connect_sequence-0001"
VARIANT = "0001"
SPEC = {
    "spec_id": "spec-hard-0004",
    "objective": "obj.sequence",
    "mechanics": ["mech.path_connect", "mech.pattern_match", "mech.region_rotate"],
    "layout_template": "hooked_spine",
    "action_set": ["click"],
    "time_limit": 309,
    "target_optimal_steps": 84,
    "grid_size": [64, 64],
}

GRID_WIDTH = int(SPEC["grid_size"][0])
GRID_HEIGHT = int(SPEC["grid_size"][1])
TIME_LIMIT = int(SPEC["time_limit"])

COLOR_BG = 0
COLOR_TRACK = 1
COLOR_TIMER_EMPTY = 2
COLOR_TIMER_FILL = 3
COLOR_NODE_IDLE = 4
COLOR_NODE_DONE = 5
COLOR_NODE_EXPECTED = 6
COLOR_NODE_START = 8
COLOR_SEGMENT_ON = 10
COLOR_NODE_GOAL = 11

NODE_SIZE = 3
NODE_CENTERS = [(7, 8), (20, 8), (20, 20), (42, 20), (42, 34), (16, 34), (16, 48), (52, 48)]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _line_axis(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_track() -> np.ndarray:
    pixels = np.full((GRID_HEIGHT - 1, GRID_WIDTH), COLOR_BG, dtype=np.int8)
    for x, y in NODE_CENTERS:
        if 0 <= x < GRID_WIDTH and 1 <= y < GRID_HEIGHT:
            pixels[y - 1, x] = COLOR_TRACK

    for idx in range(len(NODE_CENTERS) - 1):
        x1, y1 = NODE_CENTERS[idx]
        x2, y2 = NODE_CENTERS[idx + 1]

        start_x, end_x = _line_axis(x1, x2)
        pixels[y1 - 1, start_x : end_x + 1] = COLOR_TRACK

        start_y, end_y = _line_axis(y1, y2)
        pixels[start_y - 1 : end_y, x2] = COLOR_TRACK

    return pixels


def _segment_sprites() -> list[tuple[Sprite, int]]:
    segments: list[tuple[Sprite, int]] = []
    for idx in range(len(NODE_CENTERS) - 1):
        x1, y1 = NODE_CENTERS[idx]
        x2, y2 = NODE_CENTERS[idx + 1]
        to_step = idx + 2

        start_x, end_x = _line_axis(x1, x2)
        width = end_x - start_x + 1
        h_seg = Sprite(
            pixels=np.full((1, width), COLOR_TRACK, dtype=np.int8),
            name=f"segment_{idx}_h",
            x=start_x,
            y=y1,
            layer=4,
            tags=["segment", f"to_{to_step}"],
            collidable=False,
        )
        segments.append((h_seg, to_step))

        start_y, end_y = _line_axis(y1, y2)
        height = end_y - start_y + 1
        v_seg = Sprite(
            pixels=np.full((height, 1), COLOR_TRACK, dtype=np.int8),
            name=f"segment_{idx}_v",
            x=x2,
            y=start_y,
            layer=4,
            tags=["segment", f"to_{to_step}"],
            collidable=False,
        )
        segments.append((v_seg, to_step))
    return segments


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        Sprite(
            _solid(GRID_WIDTH, 1, COLOR_TIMER_FILL),
            name="timer_bar",
            x=0,
            y=0,
            layer=6,
            tags=["hud", "timer"],
            collidable=False,
        )
    )

    sprites.append(Sprite(_build_track(), name="track", x=0, y=1, layer=1, tags=["track"], collidable=False))

    for idx, (cx, cy) in enumerate(NODE_CENTERS, start=1):
        top_left_x = cx - (NODE_SIZE // 2)
        top_left_y = cy - (NODE_SIZE // 2)
        sprites.append(
            Sprite(
                pixels=np.full((NODE_SIZE, NODE_SIZE), COLOR_NODE_IDLE, dtype=np.int8),
                name=f"node_{idx}",
                x=top_left_x,
                y=top_left_y,
                layer=5,
                tags=["node", f"node_{idx}", "sys_click", "sys_every_pixel"],
                collidable=False,
            )
        )

    for segment, _ in _segment_sprites():
        sprites.append(segment)

    return Level(
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": TIME_LIMIT, "spec": SPEC, "click_sequence": [{"x": x, "y": y} for x, y in NODE_CENTERS]},
    )


class PathConnectSequence(ARCBaseGame):
    def __init__(self, seed: int = 0):
        level = _build_level()
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(game_id=GAME_ID, levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed)
        self._time_limit = TIME_LIMIT
        self._time_left = TIME_LIMIT
        self._progress = 0
        self._nodes: list[Sprite] = []
        self._segments: list[tuple[Sprite, int]] = []

    def on_set_level(self, level: Level) -> None:
        self._time_limit = int(level.get_data("time_limit") or TIME_LIMIT)
        self._time_left = self._time_limit
        self._progress = 0
        self._nodes = sorted(
            level.get_sprites_by_tag("node"),
            key=lambda sprite: int(next(tag.split("_", 1)[1] for tag in sprite.tags if tag.startswith("node_"))),
        )

        self._segments = []
        for sprite in level.get_sprites_by_tag("segment"):
            to_step_tag = next((tag for tag in sprite.tags if tag.startswith("to_")), None)
            if to_step_tag is None:
                continue
            self._segments.append((sprite, int(to_step_tag.split("_", 1)[1])))

        self._sync_path_visuals()
        self._sync_timer()

    def _sync_timer(self) -> None:
        fill = round((self._time_left / max(1, self._time_limit)) * GRID_WIDTH)
        fill = max(0, min(GRID_WIDTH, fill))
        pixels = np.full((1, GRID_WIDTH), COLOR_TIMER_EMPTY, dtype=np.int8)
        if fill > 0:
            pixels[:, :fill] = COLOR_TIMER_FILL
        timers = self.current_level.get_sprites_by_name("timer_bar")
        if timers:
            timers[0].pixels = pixels

    def _sync_path_visuals(self) -> None:
        expected = self._progress + 1
        last_node = len(self._nodes)
        for idx, sprite in enumerate(self._nodes, start=1):
            if idx <= self._progress:
                base_color = COLOR_NODE_DONE
            elif idx == expected:
                base_color = COLOR_NODE_EXPECTED
            else:
                base_color = COLOR_NODE_IDLE

            if idx == 1 and idx > self._progress:
                base_color = COLOR_NODE_START
            if idx == last_node and idx > self._progress:
                base_color = COLOR_NODE_GOAL if idx != expected else COLOR_NODE_EXPECTED

            sprite.pixels = np.full((sprite.height, sprite.width), base_color, dtype=np.int8)

        for sprite, to_step in self._segments:
            color = COLOR_SEGMENT_ON if self._progress >= to_step else COLOR_TRACK
            sprite.pixels = np.full((sprite.height, sprite.width), color, dtype=np.int8)

    def _click_cell(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if not payload:
            return None
        try:
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
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

    def _apply_click(self, x: int, y: int) -> None:
        node_idx = self._clicked_node_index(x, y)
        if node_idx is None:
            return
        expected = self._progress + 1
        if node_idx == expected:
            self._progress += 1
            return
        if node_idx == 1:
            self._progress = 1
            return
        self._progress = 0

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == int(GameAction.ACTION6.value):
            click = self._click_cell()
            if click is not None:
                self._apply_click(click[0], click[1])

        self._time_left -= 1
        if self._progress >= len(self._nodes):
            self.next_level()
        elif self._time_left <= 0:
            self.lose()

        self._sync_path_visuals()
        self._sync_timer()
        self.complete_action()
