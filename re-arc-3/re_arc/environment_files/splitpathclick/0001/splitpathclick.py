from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
TIMER_HEIGHT = 1
PLAY_Y_MIN = TIMER_HEIGHT
PLAY_Y_MAX = GRID_HEIGHT - 1

TIME_LIMIT = 160

COLORS = {
    "bg": 0,
    "floor": 1,
    "wall": 5,
    "timer_bg": 2,
    "timer_fill": 11,
    "node_1": 12,
    "node_2": 8,
    "node_3": 9,
    "node_4": 6,
    "node_on": 10,
    "bridge_off": 3,
    "bridge_on": 14,
    "goal_off": 4,
    "goal_on": 13,
    "accent": 7,
}

NODE_SPECS = [
    {"name": "node-1", "x": 8, "y": 9, "w": 5, "h": 5, "color": COLORS["node_1"]},
    {"name": "node-2", "x": 29, "y": 10, "w": 5, "h": 5, "color": COLORS["node_2"]},
    {"name": "node-3", "x": 49, "y": 28, "w": 5, "h": 5, "color": COLORS["node_3"]},
    {"name": "node-4", "x": 30, "y": 49, "w": 5, "h": 5, "color": COLORS["node_4"]},
]

SEQUENCE = ["node-1", "node-2", "node-3", "node-4"]

BRIDGE_SPECS = [
    {"name": "bridge-1", "x": 13, "y": 11, "w": 16, "h": 2},
    {"name": "bridge-2", "x": 33, "y": 13, "w": 18, "h": 2},
    {"name": "bridge-3", "x": 50, "y": 33, "w": 2, "h": 16},
    {"name": "bridge-4", "x": 34, "y": 50, "w": 18, "h": 2},
]

GOAL_SPEC = {"name": "goal-core", "x": 54, "y": 50, "w": 6, "h": 6}

BORDER_WALLS = [
    (0, 1, GRID_WIDTH, 1),
    (0, GRID_HEIGHT - 1, GRID_WIDTH, 1),
    (0, 1, 1, GRID_HEIGHT - 1),
    (GRID_WIDTH - 1, 1, 1, GRID_HEIGHT - 1),
]

# 3x3 split rooms with mismatched doorway offsets.
INNER_WALLS = [
    (21, 1, 1, 20),
    (21, 25, 1, 16),
    (21, 45, 1, 19),
    (42, 1, 1, 14),
    (42, 19, 1, 24),
    (42, 47, 1, 17),
    (1, 22, 20, 1),
    (25, 22, 18, 1),
    (47, 22, 16, 1),
    (1, 43, 14, 1),
    (19, 43, 24, 1),
    (47, 43, 16, 1),
]


def _rect_pixels(color: int, w: int, h: int) -> np.ndarray:
    return np.full((h, w), color, dtype=np.int8)


def _rect_sprite(
    *, name: str, x: int, y: int, w: int, h: int, color: int, layer: int, tags: list[str], collidable: bool = False
) -> Sprite:
    return Sprite(pixels=_rect_pixels(color, w, h), name=name, x=x, y=y, layer=layer, tags=tags, collidable=collidable)


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        _rect_sprite(
            name="floor",
            x=0,
            y=0,
            w=GRID_WIDTH,
            h=GRID_HEIGHT,
            color=COLORS["floor"],
            layer=0,
            tags=["floor", "sys_static"],
        )
    )

    sprites.append(
        _rect_sprite(
            name="timer-bg",
            x=0,
            y=0,
            w=GRID_WIDTH,
            h=TIMER_HEIGHT,
            color=COLORS["timer_bg"],
            layer=1,
            tags=["hud", "timer_bg", "sys_static"],
        )
    )

    sprites.append(
        _rect_sprite(
            name="timer-fill",
            x=0,
            y=0,
            w=GRID_WIDTH,
            h=TIMER_HEIGHT,
            color=COLORS["timer_fill"],
            layer=2,
            tags=["hud", "timer_fill"],
        )
    )

    for wall_idx, (x, y, w, h) in enumerate(BORDER_WALLS + INNER_WALLS, start=1):
        sprites.append(
            _rect_sprite(
                name=f"wall-{wall_idx}",
                x=x,
                y=y,
                w=w,
                h=h,
                color=COLORS["wall"],
                layer=3,
                tags=["wall", "blocker", "sys_static"],
                collidable=True,
            )
        )

    # Decorative path hints to emphasize split-room routing.
    sprites.append(
        _rect_sprite(
            name="hint-left", x=4, y=27, w=12, h=2, color=COLORS["accent"], layer=4, tags=["hint", "sys_static"]
        )
    )
    sprites.append(
        _rect_sprite(
            name="hint-mid", x=26, y=29, w=12, h=2, color=COLORS["accent"], layer=4, tags=["hint", "sys_static"]
        )
    )

    for idx, spec in enumerate(BRIDGE_SPECS):
        sprites.append(
            _rect_sprite(
                name=spec["name"],
                x=spec["x"],
                y=spec["y"],
                w=spec["w"],
                h=spec["h"],
                color=COLORS["bridge_off"],
                layer=5,
                tags=["bridge", f"bridge_idx_{idx}"],
            )
        )

    for idx, spec in enumerate(NODE_SPECS):
        sprites.append(
            _rect_sprite(
                name=spec["name"],
                x=spec["x"],
                y=spec["y"],
                w=spec["w"],
                h=spec["h"],
                color=spec["color"],
                layer=6,
                tags=["node", "sys_click", "sys_every_pixel", f"seq_{idx + 1}"],
            )
        )

    sprites.append(
        _rect_sprite(
            name=GOAL_SPEC["name"],
            x=GOAL_SPEC["x"],
            y=GOAL_SPEC["y"],
            w=GOAL_SPEC["w"],
            h=GOAL_SPEC["h"],
            color=COLORS["goal_off"],
            layer=6,
            tags=["goal", "sys_static"],
        )
    )

    return Level(
        name="SplitPathClick 0001",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": TIME_LIMIT, "sequence": list(SEQUENCE)},
    )


class Splitpathclick(ARCBaseGame):
    def __init__(self, seed: int = 0):
        level = _build_level()
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLORS["bg"])
        super().__init__(
            game_id="splitpathclick-0001", levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed
        )
        self._time_limit = TIME_LIMIT
        self._time_left = TIME_LIMIT
        self._progress = 0
        self._sequence: list[str] = list(SEQUENCE)
        self._timer: Sprite | None = None
        self._goal: Sprite | None = None
        self._nodes_by_name: dict[str, Sprite] = {}
        self._bridges_by_name: dict[str, Sprite] = {}
        self._default_node_colors: dict[str, int] = {}

    def on_set_level(self, level: Level) -> None:
        self._time_limit = int(level.get_data("time_limit") or TIME_LIMIT)
        self._time_left = self._time_limit
        self._progress = 0

        seq = level.get_data("sequence")
        if isinstance(seq, list) and seq:
            self._sequence = [str(name) for name in seq]
        else:
            self._sequence = list(SEQUENCE)

        timer_sprites = level.get_sprites_by_name("timer-fill")
        goal_sprites = level.get_sprites_by_name("goal-core")
        if not timer_sprites or not goal_sprites:
            raise RuntimeError("splitpathclick level missing timer or goal sprite")

        self._timer = timer_sprites[0]
        self._goal = goal_sprites[0]

        self._nodes_by_name = {}
        self._default_node_colors = {}
        for sprite in level.get_sprites_by_tag("node"):
            rendered = sprite.render()
            base_color = int(rendered[0][0]) if rendered.size > 0 else COLORS["node_1"]
            self._nodes_by_name[sprite.name] = sprite
            self._default_node_colors[sprite.name] = base_color

        self._bridges_by_name = {sprite.name: sprite for sprite in level.get_sprites_by_tag("bridge")}

        for bridge in self._bridges_by_name.values():
            self._set_sprite_color(bridge, COLORS["bridge_off"])
        for node_name, node in self._nodes_by_name.items():
            self._set_sprite_color(node, self._default_node_colors.get(node_name, COLORS["node_1"]))
        self._set_sprite_color(self._goal, COLORS["goal_off"])

        self._refresh_timer()

    def _set_sprite_color(self, sprite: Sprite | None, color: int) -> None:
        if sprite is None:
            return
        sprite.pixels = _rect_pixels(color, sprite.width, sprite.height)

    def _refresh_timer(self) -> None:
        if self._timer is None:
            return
        ratio = max(0.0, min(1.0, self._time_left / max(1, self._time_limit)))
        fill_width = round(ratio * GRID_WIDTH)
        row = [COLORS["timer_fill"] if x < fill_width else -1 for x in range(GRID_WIDTH)]
        self._timer.pixels = np.array([row], dtype=np.int8)

    def _sprite_at_click(self, x: int, y: int) -> Sprite | None:
        if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
            return None

        for sprite in sorted(self.current_level.get_sprites_by_tag("sys_click"), key=lambda s: s.layer, reverse=True):
            if not sprite.is_visible:
                continue
            if x < sprite.x or y < sprite.y or x >= sprite.x + sprite.width or y >= sprite.y + sprite.height:
                continue

            rendered = sprite.render()
            local_x = x - sprite.x
            local_y = y - sprite.y
            if local_y < 0 or local_x < 0 or local_y >= rendered.shape[0] or local_x >= rendered.shape[1]:
                continue
            if int(rendered[local_y][local_x]) < 0:
                continue
            return sprite
        return None

    def _reset_sequence(self) -> None:
        self._progress = 0
        for bridge in self._bridges_by_name.values():
            self._set_sprite_color(bridge, COLORS["bridge_off"])
        for node_name, node in self._nodes_by_name.items():
            self._set_sprite_color(node, self._default_node_colors.get(node_name, COLORS["node_1"]))
        self._set_sprite_color(self._goal, COLORS["goal_off"])

    def _apply_click(self, x: int, y: int) -> None:
        clicked = self._sprite_at_click(x, y)
        if clicked is None or "node" not in clicked.tags:
            return

        if self._progress >= len(self._sequence):
            return

        expected_name = self._sequence[self._progress]
        if clicked.name != expected_name:
            self._reset_sequence()
            return

        self._set_sprite_color(clicked, COLORS["node_on"])
        bridge_name = f"bridge-{self._progress + 1}"
        bridge = self._bridges_by_name.get(bridge_name)
        self._set_sprite_color(bridge, COLORS["bridge_on"])

        self._progress += 1
        if self._progress >= len(self._sequence):
            self._set_sprite_color(self._goal, COLORS["goal_on"])
            self.next_level()

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        state_name = self.current_state_name()

        if state_name == "NOT_FINISHED" and action_id == int(GameAction.ACTION6.value):
            raw_x = self.action.data.get("x") if isinstance(self.action.data, dict) else None
            raw_y = self.action.data.get("y") if isinstance(self.action.data, dict) else None
            try:
                click_x = int(raw_x)
                click_y = int(raw_y)
            except (TypeError, ValueError):
                click_x = -1
                click_y = -1
            self._apply_click(click_x, click_y)

        if state_name == "NOT_FINISHED" and self._time_left > 0:
            self._time_left -= 1
        if state_name == "NOT_FINISHED" and self._time_left <= 0:
            self.lose()

        self._refresh_timer()
        self.complete_action()

    def current_state_name(self) -> str:
        state = getattr(self, "_state", None)
        return getattr(state, "name", str(state))
