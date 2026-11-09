from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
PLAYFIELD_TOP = 1

COLOR_BG = 0
COLOR_WALL = 5
COLOR_FLOOR = 1
COLOR_TIMER_FILL = 11
COLOR_TIMER_EMPTY = 2
COLOR_NODE_OFF = 8
COLOR_NODE_ON = 3
COLOR_NODE_ACCENT = 12
COLOR_NODE_ERROR = 14
COLOR_CORE_OFF = 9
COLOR_CORE_ON = 10
COLOR_GATE_OFF = 6
COLOR_GATE_ON = 4

TEMPLATE_RECTS = {
    "rooms": {
        "rooms": [(2, 4, 18, 18), (22, 4, 18, 18), (2, 24, 18, 18), (22, 24, 18, 18), (44, 20, 18, 18)],
        "corridors": [(20, 13, 2, 1), (20, 33, 2, 1), (11, 22, 1, 2), (31, 22, 1, 2), (40, 29, 4, 1)],
        "nodes": {1: (9, 9), 2: (29, 9), 3: (9, 29), 4: (29, 29), 9: (51, 29)},
    },
    "lanes": {
        "rooms": [
            (2, 6, 14, 10),
            (2, 22, 14, 10),
            (2, 38, 14, 10),
            (22, 14, 14, 10),
            (22, 30, 14, 10),
            (42, 22, 20, 14),
        ],
        "corridors": [(16, 10, 6, 1), (16, 26, 6, 1), (16, 42, 6, 1), (36, 19, 6, 1), (36, 35, 6, 1)],
        "nodes": {1: (7, 9), 2: (7, 25), 3: (7, 41), 4: (27, 17), 9: (52, 27)},
    },
    "zigzag": {
        "rooms": [(2, 4, 16, 14), (24, 12, 16, 14), (2, 24, 16, 14), (24, 36, 16, 14), (46, 24, 16, 14)],
        "corridors": [(18, 11, 6, 1), (18, 31, 6, 1), (40, 43, 6, 1), (32, 26, 1, 10)],
        "nodes": {1: (8, 8), 2: (30, 16), 3: (8, 28), 4: (30, 40), 9: (52, 28)},
    },
    "hooked_spine": {
        "rooms": [(4, 8, 14, 14), (22, 8, 14, 14), (22, 26, 14, 14), (22, 44, 14, 14), (42, 26, 18, 18)],
        "corridors": [(18, 15, 4, 1), (29, 22, 1, 4), (29, 40, 1, 4), (36, 34, 6, 1)],
        "nodes": {1: (10, 13), 2: (28, 13), 3: (28, 31), 4: (28, 49), 9: (50, 33)},
    },
}


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _node_pixels(base: int, accent: int) -> np.ndarray:
    return np.array([[base, accent, base], [accent, base, accent], [base, accent, base]], dtype=np.int8)


class BatchHClickGame(ARCBaseGame):
    def __init__(self, seed: int, config: dict):
        self._cfg = config
        self._time_remaining = int(config["time_limit"])
        self._objective = str(config["objective"])
        self._required_nodes = list(config["required_nodes"])
        self._sequence = list(config.get("sequence", self._required_nodes))
        self._template = str(config["layout_template"])
        self._progress = 0
        self._error_flash = 0
        self._armed = False
        self._activated: set[int] = set()
        self._node_sprites: dict[int, Sprite] = {}
        self._meter_sprite: Sprite | None = None
        self._gate_sprite: Sprite | None = None
        self._timer_sprite: Sprite | None = None

        level = self._build_level(config)
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(
            game_id=str(config["game_id"]), levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed
        )

    def _build_level(self, config: dict) -> Level:
        tpl = TEMPLATE_RECTS.get(self._template, TEMPLATE_RECTS["rooms"])
        node_positions = dict(tpl["nodes"])
        sprites: list[Sprite] = []

        sprites.append(
            Sprite(
                _solid(GRID_WIDTH, 1, COLOR_TIMER_FILL),
                name="timer_bar",
                x=0,
                y=0,
                layer=0,
                tags=["hud", "timer"],
                collidable=False,
            )
        )
        sprites.append(
            Sprite(
                _solid(GRID_WIDTH, GRID_HEIGHT - PLAYFIELD_TOP, COLOR_WALL),
                name="walls",
                x=0,
                y=PLAYFIELD_TOP,
                layer=0,
                tags=["wall", "blocker"],
                collidable=False,
            )
        )

        for idx, (x, y, w, h) in enumerate(tpl["rooms"]):
            sprites.append(
                Sprite(
                    _solid(w, h, COLOR_FLOOR),
                    name=f"room_{idx}",
                    x=x,
                    y=y,
                    layer=1,
                    tags=["floor", "room"],
                    collidable=False,
                )
            )

        for idx, (x, y, w, h) in enumerate(tpl["corridors"]):
            sprites.append(
                Sprite(
                    _solid(w, h, COLOR_FLOOR),
                    name=f"corridor_{idx}",
                    x=x,
                    y=y,
                    layer=1,
                    tags=["floor", "corridor"],
                    collidable=False,
                )
            )

        for node_id in (1, 2, 3, 4):
            x, y = node_positions[node_id]
            sprites.append(
                Sprite(
                    _node_pixels(COLOR_NODE_OFF, COLOR_NODE_ACCENT),
                    name=f"node_{node_id}",
                    x=x,
                    y=y,
                    layer=3,
                    tags=["node", "sys_click", "sys_every_pixel", f"node_{node_id}"],
                    collidable=False,
                )
            )

        core_x, core_y = node_positions[9]
        sprites.append(
            Sprite(
                _node_pixels(COLOR_CORE_OFF, COLOR_NODE_ACCENT),
                name="goal_core",
                x=core_x,
                y=core_y,
                layer=3,
                tags=["node", "core", "sys_click", "sys_every_pixel", "node_9"],
                collidable=False,
            )
        )

        sprites.append(
            Sprite(
                _solid(12, 3, COLOR_GATE_OFF),
                name="gate_status",
                x=49,
                y=18,
                layer=2,
                tags=["gate", "status"],
                collidable=False,
            )
        )

        sprites.append(
            Sprite(
                _solid(12, 1, COLOR_GATE_OFF),
                name="progress_meter",
                x=49,
                y=38,
                layer=4,
                tags=["hud", "status"],
                collidable=False,
            )
        )

        return Level(
            name=str(config["title"]),
            grid_size=(GRID_WIDTH, GRID_HEIGHT),
            sprites=sprites,
            data={
                "time_limit": int(config["time_limit"]),
                "required_nodes": list(config["required_nodes"]),
                "sequence": list(config.get("sequence", config["required_nodes"])),
                "objective": str(config["objective"]),
                "layout_template": str(config["layout_template"]),
            },
        )

    def on_set_level(self, level: Level) -> None:
        self._time_remaining = int(level.get_data("time_limit") or self._cfg["time_limit"])
        self._required_nodes = list(level.get_data("required_nodes") or self._cfg["required_nodes"])
        self._sequence = list(level.get_data("sequence") or self._required_nodes)
        self._objective = str(level.get_data("objective") or self._cfg["objective"])
        self._progress = 0
        self._error_flash = 0
        self._armed = False
        self._activated.clear()

        self._node_sprites = {}
        for sprite in self.current_level.get_sprites_by_tag("node"):
            node_id = self._node_id(sprite)
            if node_id is not None:
                self._node_sprites[node_id] = sprite

        meters = self.current_level.get_sprites_by_name("progress_meter")
        self._meter_sprite = meters[0] if meters else None

        gates = self.current_level.get_sprites_by_name("gate_status")
        self._gate_sprite = gates[0] if gates else None

        timers = self.current_level.get_sprites_by_name("timer_bar")
        self._timer_sprite = timers[0] if timers else None

        self._sync_visuals()

    def _node_id(self, sprite: Sprite) -> int | None:
        for tag in sprite.tags:
            if not tag.startswith("node_"):
                continue
            try:
                return int(tag.split("_", 1)[1])
            except ValueError:
                return None
        return None

    def _hit_node(self, x: int, y: int) -> int | None:
        for node_id, sprite in self._node_sprites.items():
            sx = int(sprite.x)
            sy = int(sprite.y)
            if sx <= x < sx + int(sprite.width) and sy <= y < sy + int(sprite.height):
                return node_id
        return None

    def _mark_progress(self, node_id: int) -> None:
        if self._objective == "obj.sequence":
            expected = self._sequence[self._progress] if self._progress < len(self._sequence) else None
            if node_id == expected:
                self._progress += 1
                if self._progress >= len(self._sequence):
                    self.win()
                return
            self._progress = 0
            self._activated.clear()
            self._error_flash = 2
            return

        if node_id in self._required_nodes:
            self._activated.add(node_id)
        self._progress = len(self._activated)
        self._armed = all(r in self._activated for r in self._required_nodes)

        if self._objective == "obj.collect_all" and self._armed:
            self.win()

    def _apply_click(self, x: int, y: int) -> None:
        node_id = self._hit_node(x, y)
        if node_id is None:
            return

        if self._objective == "obj.activate_all":
            if node_id == 9:
                if self._armed:
                    self.win()
                else:
                    self._error_flash = 2
                return
            self._mark_progress(node_id)
            return

        if node_id == 9:
            self._error_flash = 2
            return

        self._mark_progress(node_id)

    def _sync_visuals(self) -> None:
        solved_nodes = (
            set(self._sequence[: self._progress]) if self._objective == "obj.sequence" else set(self._activated)
        )

        for node_id in (1, 2, 3, 4):
            sprite = self._node_sprites.get(node_id)
            if sprite is None:
                continue
            if self._error_flash > 0:
                sprite.pixels = _node_pixels(COLOR_NODE_ERROR, COLOR_NODE_ACCENT)
            elif node_id in solved_nodes:
                sprite.pixels = _node_pixels(COLOR_NODE_ON, COLOR_NODE_ACCENT)
            else:
                sprite.pixels = _node_pixels(COLOR_NODE_OFF, COLOR_NODE_ACCENT)

        core = self._node_sprites.get(9)
        if core is not None:
            core.pixels = _node_pixels(COLOR_CORE_ON if self._armed else COLOR_CORE_OFF, COLOR_NODE_ACCENT)

        if self._gate_sprite is not None:
            self._gate_sprite.pixels = _solid(12, 3, COLOR_GATE_ON if self._armed else COLOR_GATE_OFF)

        if self._meter_sprite is not None:
            width = 12
            total = len(self._sequence) if self._objective == "obj.sequence" else len(self._required_nodes)
            filled = round((self._progress / max(1, total)) * width)
            filled = max(0, min(width, filled))
            meter = np.full((1, width), COLOR_GATE_OFF, dtype=np.int8)
            meter[:, :filled] = np.int8(COLOR_GATE_ON)
            self._meter_sprite.pixels = meter

        if self._timer_sprite is not None:
            limit = int(self._cfg["time_limit"])
            ratio = self._time_remaining / float(max(1, limit))
            filled = round(ratio * GRID_WIDTH)
            filled = max(0, min(GRID_WIDTH, filled))
            timer_row = np.full((1, GRID_WIDTH), COLOR_TIMER_EMPTY, dtype=np.int8)
            timer_row[:, :filled] = np.int8(COLOR_TIMER_FILL)
            self._timer_sprite.pixels = timer_row

    def step(self) -> None:
        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data if isinstance(self.action.data, dict) else {}
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
            self._apply_click(x, y)

        self._time_remaining -= 1
        if self._error_flash > 0:
            self._error_flash -= 1

        state_name = getattr(getattr(self, "_state", None), "name", "")
        if self._time_remaining <= 0 and state_name != "WIN":
            self.lose()

        self._sync_visuals()
        self.complete_action()
