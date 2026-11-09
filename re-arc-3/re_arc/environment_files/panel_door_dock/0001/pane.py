from __future__ import annotations

from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
METER_HEIGHT = 8
PLAYFIELD_TOP = 8
PLAYFIELD_HEIGHT = 40
CONTROLS_TOP = 48
CELL_SIZE = 8
LOGICAL_WIDTH = 8
LOGICAL_HEIGHT = 5

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_FLOOR_DIM = 2
COLOR_PLATE = 3
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_C_ACCENT = 7
COLOR_DANGER = 8
COLOR_B = 9
COLOR_B_LIT = 10
COLOR_YELLOW = 11
COLOR_A = 12
COLOR_MAROON = 13
COLOR_DOCK = 14
COLOR_C = 15

MOVE_DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ARROW_RECTS = {"LEFT": (0, 48, 15, 63), "UP": (16, 48, 31, 63), "DOWN": (32, 48, 47, 63), "RIGHT": (48, 48, 63, 63)}
DOOR_COLORS = {"A": (COLOR_A, COLOR_YELLOW), "B": (COLOR_B, COLOR_B_LIT), "C": (COLOR_C, COLOR_C_ACCENT)}
VALID_PANEL_IDS = frozenset({"A", "B", "C"})


class PanelRegion:
    def __init__(
        self, panel_id: str, rect: tuple[int, int, int, int], cable_from: tuple[int, int], cable_to: tuple[int, int]
    ) -> None:
        self.panel_id = str(panel_id)
        self.rect = tuple(rect)
        self.cable_from = tuple(cable_from)
        self.cable_to = tuple(cable_to)


class PanelDoorDockLevel:
    def __init__(
        self,
        name: str,
        rows: tuple[str, ...],
        start: tuple[int, int],
        dock: tuple[int, int],
        panels: tuple[PanelRegion, ...],
        budget: int,
    ) -> None:
        self.name = str(name)
        self.rows = tuple(rows)
        self.start = tuple(start)
        self.dock = tuple(dock)
        self.panels = tuple(panels)
        self.budget = int(budget)

    @property
    def width(self) -> int:
        return LOGICAL_WIDTH

    @property
    def height(self) -> int:
        return LOGICAL_HEIGHT


LEVELS = (
    PanelDoorDockLevel(
        name="Teach One Door",
        rows=("########", "#S.A.D##", "#..#..##", "########", "########"),
        start=(1, 1),
        dock=(5, 1),
        panels=(PanelRegion("A", (24, 8, 31, 15), (27, 15), (27, 16)),),
        budget=15,
    ),
    PanelDoorDockLevel(
        name="Open In Order",
        rows=("########", "#SABD###", "#.#.#.##", "########", "########"),
        start=(1, 1),
        dock=(4, 1),
        panels=(
            PanelRegion("A", (16, 8, 23, 15), (19, 15), (19, 16)),
            PanelRegion("B", (32, 24, 39, 31), (35, 24), (35, 23)),
        ),
        budget=18,
    ),
    PanelDoorDockLevel(
        name="Reopen The Old Door",
        rows=("########", "##SA..##", "###B..##", "##DA..##", "########"),
        start=(2, 1),
        dock=(2, 3),
        panels=(
            PanelRegion("A", (24, 8, 31, 15), (27, 15), (27, 16)),
            PanelRegion("B", (32, 16, 39, 23), (32, 19), (31, 19)),
        ),
        budget=20,
    ),
)


def logical_to_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0] * CELL_SIZE), int(PLAYFIELD_TOP + (cell[1] * CELL_SIZE))


def rect_pixels(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def floor_cell_pixels() -> np.ndarray:
    pixels = rect_pixels(CELL_SIZE, CELL_SIZE, COLOR_FLOOR)
    pixels[0, :] = COLOR_FLOOR_DIM
    pixels[:, 0] = COLOR_FLOOR_DIM
    pixels[7, 7] = COLOR_WHITE
    return pixels


def wall_background_pixels(height: int) -> np.ndarray:
    pixels = rect_pixels(GRID_WIDTH, height, COLOR_WALL)
    pixels[:, 0] = COLOR_BLACK
    pixels[:, -1] = COLOR_BLACK
    return pixels


def meter_pixels(remaining: int, capacity: int, fail_flash: bool) -> np.ndarray:
    pixels = rect_pixels(GRID_WIDTH, METER_HEIGHT, COLOR_MAROON if fail_flash else COLOR_WALL)
    pixels[0, :] = COLOR_BLACK
    pixels[-1, :] = COLOR_BLACK
    pixels[:, 0] = COLOR_BLACK
    pixels[:, -1] = COLOR_BLACK

    usable = max(0, capacity)
    if usable <= 0:
        return pixels

    pip_count = min(usable, 20)
    left_margin = 2
    pip_gap = 1
    pip_width = 2
    for pip_idx in range(pip_count):
        start_x = left_margin + pip_idx * (pip_width + pip_gap)
        end_x = start_x + pip_width
        if pip_idx < remaining:
            color = COLOR_YELLOW if remaining > 3 or pip_idx < max(0, remaining - 3) else COLOR_A
        else:
            color = COLOR_DANGER
        pixels[2:6, start_x:end_x] = color
    return pixels


def button_pixels(direction: str) -> np.ndarray:
    pixels = rect_pixels(16, 16, COLOR_PLATE)
    pixels[0, :] = COLOR_BLACK
    pixels[-1, :] = COLOR_BLACK
    pixels[:, 0] = COLOR_BLACK
    pixels[:, -1] = COLOR_BLACK
    pixels[2:14, 2:14] = COLOR_WALL

    arrow = COLOR_YELLOW
    if direction == "LEFT":
        pixels[7, 3:12] = arrow
        pixels[6, 4:10] = arrow
        pixels[8, 4:10] = arrow
        pixels[5, 5:8] = arrow
        pixels[9, 5:8] = arrow
        pixels[4:11, 3] = arrow
    elif direction == "RIGHT":
        pixels[7, 4:13] = arrow
        pixels[6, 6:12] = arrow
        pixels[8, 6:12] = arrow
        pixels[5, 8:11] = arrow
        pixels[9, 8:11] = arrow
        pixels[4:11, 12] = arrow
    elif direction == "UP":
        pixels[3, 7] = arrow
        pixels[4, 6:9] = arrow
        pixels[5, 5:10] = arrow
        pixels[6, 4:11] = arrow
        pixels[7:12, 7] = arrow
        pixels[7:12, 8] = arrow
    elif direction == "DOWN":
        pixels[12, 7] = arrow
        pixels[11, 6:9] = arrow
        pixels[10, 5:10] = arrow
        pixels[9, 4:11] = arrow
        pixels[4:9, 7] = arrow
        pixels[4:9, 8] = arrow
    return pixels


def cargo_pixels(on_dock: bool) -> np.ndarray:
    body = COLOR_YELLOW if on_dock else COLOR_A
    pixels = np.zeros((6, 6), dtype=np.int8)
    pixels[:, :] = body
    pixels[0, :] = COLOR_WALL
    pixels[-1, :] = COLOR_WALL
    pixels[:, 0] = COLOR_WALL
    pixels[:, -1] = COLOR_WALL
    pixels[2:4, 2:4] = COLOR_WALL
    pixels[1, 1] = COLOR_YELLOW
    pixels[1, 4] = COLOR_YELLOW
    return pixels


def dock_pixels(active: bool) -> np.ndarray:
    pixels = floor_cell_pixels()
    ring = COLOR_B_LIT if active else COLOR_DOCK
    pixels[1:7, 3] = ring
    pixels[1:7, 4] = ring
    pixels[3, 1:7] = ring
    pixels[4, 1:7] = ring
    pixels[2:6, 2:6] = COLOR_WHITE if active else COLOR_FLOOR
    pixels[2, 2] = COLOR_WALL
    pixels[2, 5] = COLOR_WALL
    pixels[5, 2] = COLOR_WALL
    pixels[5, 5] = COLOR_WALL
    return pixels


def door_pixels(panel_id: str, open_state: bool) -> np.ndarray:
    color, lit = DOOR_COLORS[panel_id]
    pixels = rect_pixels(CELL_SIZE, CELL_SIZE, color if not open_state else COLOR_FLOOR)
    pixels[0, :] = COLOR_WALL
    pixels[-1, :] = COLOR_WALL
    pixels[:, 0] = COLOR_WALL
    pixels[:, -1] = COLOR_WALL
    if open_state:
        pixels[:, 1] = color
        pixels[:, -2] = color
        pixels[1:7, 2:6] = COLOR_FLOOR
        pixels[1:7, 3] = lit
        pixels[1:7, 4] = lit
        pixels[2:6, 3:5] = COLOR_FLOOR
    else:
        pixels[2, 1:7] = lit
        pixels[3, 1:7] = color
        pixels[4, 1:7] = lit
        pixels[5, 1:7] = color
    return pixels


def panel_pixels(panel_id: str, active: bool) -> np.ndarray:
    color, lit = DOOR_COLORS[panel_id]
    pixels = rect_pixels(8, 8, color)
    pixels[0, :] = COLOR_BLACK
    pixels[-1, :] = COLOR_BLACK
    pixels[:, 0] = COLOR_BLACK
    pixels[:, -1] = COLOR_BLACK
    pixels[2:6, 2:6] = COLOR_FLOOR_DIM
    pixels[3:5, 3:5] = lit if active else COLOR_BLACK
    return pixels


def panel_center(panel: PanelRegion) -> tuple[int, int]:
    x0, y0, x1, y1 = panel.rect
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def action_payload_for_click(point: tuple[int, int]) -> dict[str, int]:
    return {"x": int(point[0]), "y": int(point[1])}


def level_action_programs() -> list[list[tuple[int, dict[str, int]]]]:
    level0 = [
        (6, action_payload_for_click(panel_center(LEVELS[0].panels[0]))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
    ]
    level1 = [
        (6, action_payload_for_click(panel_center(LEVELS[1].panels[0]))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click(panel_center(LEVELS[1].panels[1]))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click((56, 56))),
    ]
    level2 = [
        (6, action_payload_for_click(panel_center(LEVELS[2].panels[0]))),
        (6, action_payload_for_click((56, 56))),
        (6, action_payload_for_click(panel_center(LEVELS[2].panels[1]))),
        (6, action_payload_for_click((40, 56))),
        (6, action_payload_for_click(panel_center(LEVELS[2].panels[0]))),
        (6, action_payload_for_click((40, 56))),
        (6, action_payload_for_click((8, 56))),
        (6, action_payload_for_click((8, 56))),
    ]
    return [level0, level1, level2]


def _panel_hit(level: PanelDoorDockLevel, x: int, y: int) -> str | None:
    for panel in level.panels:
        x0, y0, x1, y1 = panel.rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            return panel.panel_id
    return None


def _arrow_hit(x: int, y: int) -> str | None:
    for direction, (x0, y0, x1, y1) in ARROW_RECTS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return direction
    return None


def _tile_at(level: PanelDoorDockLevel, cell: tuple[int, int]) -> str:
    cx, cy = cell
    if not (0 <= cx < LOGICAL_WIDTH and 0 <= cy < LOGICAL_HEIGHT):
        return "#"
    return level.rows[cy][cx]


def _passable(tile: str, active_door: str | None) -> bool:
    if tile in {".", "S", "D"}:
        return True
    if tile in VALID_PANEL_IDS:
        return active_door == tile
    return False


def shortest_solution_for_level(level: PanelDoorDockLevel) -> list[str]:
    start_state = (level.start[0], level.start[1], "", 0, "")
    queue = deque([start_state])
    visited = {(level.start[0], level.start[1], "", 0)}

    while queue:
        cx, cy, active_door, panels_used, panel_history = queue.popleft()
        if (cx, cy) == level.dock:
            path = panel_history.split(",") if panel_history else []
            return [entry for entry in path if entry]

        for panel in level.panels:
            next_active = panel.panel_id if active_door != panel.panel_id else ""
            next_state = (cx, cy, next_active, panels_used + 1, f"{panel_history},P{panel.panel_id}")
            key = next_state[:4]
            if key not in visited:
                visited.add(key)
                queue.append(next_state)

        for direction, (dx, dy) in MOVE_DELTAS.items():
            nx = cx + dx
            ny = cy + dy
            tile = _tile_at(level, (nx, ny))
            if not _passable(tile, active_door or None):
                continue
            next_state = (nx, ny, active_door, panels_used + 1, f"{panel_history},M{direction}")
            key = next_state[:4]
            if key not in visited:
                visited.add(key)
                queue.append(next_state)

    raise RuntimeError(f"Level {level.name!r} is not solvable.")


def validate_levels() -> None:
    level0 = shortest_solution_for_level(LEVELS[0])
    level1 = shortest_solution_for_level(LEVELS[1])
    level2 = shortest_solution_for_level(LEVELS[2])

    assert len(level0) == 5
    assert any(step == "PA" for step in level0)
    assert level0.count("MRIGHT") >= 4

    assert len(level1) == 5
    assert [step for step in level1 if step.startswith("P")] == ["PA", "PB"]

    assert len(level2) == 7
    panel_steps = [step for step in level2 if step.startswith("P")]
    assert panel_steps == ["PA", "PB", "PA"]


validate_levels()


class Pane(ARCBaseGame):
    levels_spec: tuple[PanelDoorDockLevel, ...]

    def __init__(self, seed: int | None = None) -> None:
        self.levels_spec = LEVELS
        self._remaining_actions = 0
        self._active_door: str | None = None
        self._cargo_pos = (0, 0)
        self._mode = "play"
        self._door_sprites: list[tuple[Sprite, str]] = []
        self._panel_sprites: dict[str, Sprite] = {}
        self._meter_sprite: Sprite | None = None
        self._dock_sprite: Sprite | None = None
        self._cargo_sprite: Sprite | None = None
        self._fail_overlay: Sprite | None = None

        levels = [self._build_level(level_idx, level) for level_idx, level in enumerate(self.levels_spec)]
        camera = Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, COLOR_WHITE)
        super().__init__("panel_door_dock-0001", levels, camera, False, len(levels), [6], seed=seed)

    def level_action_programs(self) -> list[list[tuple[int, dict[str, int]]]]:
        return level_action_programs()

    def _build_level(self, level_idx: int, spec: PanelDoorDockLevel) -> Level:
        sprites: list[Sprite] = [
            Sprite(rect_pixels(GRID_WIDTH, GRID_HEIGHT, COLOR_WHITE), name="bg", x=0, y=0, layer=0, collidable=False),
            Sprite(
                meter_pixels(spec.budget, spec.budget, False),
                name="meter",
                x=0,
                y=0,
                layer=1,
                tags=["hud"],
                collidable=False,
            ),
            Sprite(
                wall_background_pixels(PLAYFIELD_HEIGHT),
                name="playfield_wall",
                x=0,
                y=PLAYFIELD_TOP,
                layer=1,
                tags=["wall"],
                collidable=False,
            ),
            Sprite(
                rect_pixels(GRID_WIDTH, GRID_HEIGHT - CONTROLS_TOP, COLOR_FLOOR_DIM),
                name="controls_back",
                x=0,
                y=CONTROLS_TOP,
                layer=1,
                collidable=False,
            ),
        ]

        for direction, (x0, y0, _, _) in ARROW_RECTS.items():
            sprites.append(
                Sprite(
                    button_pixels(direction), name=f"button_{direction.lower()}", x=x0, y=y0, layer=2, collidable=False
                )
            )

        for cy, row in enumerate(spec.rows):
            for cx, tile in enumerate(row):
                if tile in {".", "S", "D"}:
                    px, py = logical_to_pixel((cx, cy))
                    sprites.append(
                        Sprite(
                            floor_cell_pixels(),
                            name=f"floor_{cx}_{cy}",
                            x=px,
                            y=py,
                            layer=2,
                            tags=["floor"],
                            collidable=False,
                        )
                    )

        dock_x, dock_y = logical_to_pixel(spec.dock)
        sprites.append(Sprite(dock_pixels(False), name="dock", x=dock_x, y=dock_y, layer=4, collidable=False))

        for cy, row in enumerate(spec.rows):
            for cx, tile in enumerate(row):
                if tile in VALID_PANEL_IDS:
                    px, py = logical_to_pixel((cx, cy))
                    sprites.append(
                        Sprite(
                            door_pixels(tile, False),
                            name=f"door_{tile}_{cx}_{cy}",
                            x=px,
                            y=py,
                            layer=5,
                            tags=["door", f"door_{tile}"],
                            collidable=False,
                        )
                    )

        for panel in spec.panels:
            x0, y0, x1, y1 = panel.rect
            width = x1 - x0 + 1
            height = y1 - y0 + 1
            pixels = panel_pixels(panel.panel_id, False)
            if pixels.shape != (height, width):
                pixels = np.resize(pixels, (height, width)).astype(np.int8)
            sprites.append(
                Sprite(
                    pixels,
                    name=f"panel_{panel.panel_id}",
                    x=x0,
                    y=y0,
                    layer=6,
                    tags=["panel", f"panel_{panel.panel_id}"],
                    collidable=False,
                )
            )
            cable_x0 = min(panel.cable_from[0], panel.cable_to[0])
            cable_y0 = min(panel.cable_from[1], panel.cable_to[1])
            cable_w = abs(panel.cable_from[0] - panel.cable_to[0]) + 1
            cable_h = abs(panel.cable_from[1] - panel.cable_to[1]) + 1
            cable_pixels = rect_pixels(max(1, cable_w), max(1, cable_h), DOOR_COLORS[panel.panel_id][0])
            sprites.append(
                Sprite(cable_pixels, name=f"cable_{panel.panel_id}", x=cable_x0, y=cable_y0, layer=5, collidable=False)
            )

        start_x, start_y = logical_to_pixel(spec.start)
        sprites.append(
            Sprite(
                cargo_pixels(False),
                name="cargo",
                x=start_x + 1,
                y=start_y + 1,
                layer=7,
                tags=["cargo"],
                collidable=False,
            )
        )
        sprites.append(
            Sprite(
                rect_pixels(GRID_WIDTH, PLAYFIELD_HEIGHT, COLOR_MAROON),
                name="fail_overlay",
                x=0,
                y=PLAYFIELD_TOP,
                layer=8,
                collidable=False,
            )
        )
        return Level(sprites, (GRID_WIDTH, GRID_HEIGHT), {"level_index": level_idx}, spec.name)

    @property
    def current_spec(self) -> PanelDoorDockLevel:
        return self.levels_spec[int(self.level_index)]

    def on_set_level(self, _level: Level) -> None:
        spec = self.current_spec
        self._remaining_actions = int(spec.budget)
        self._active_door = None
        self._cargo_pos = tuple(spec.start)
        self._mode = "play"
        self._door_sprites = []
        self._panel_sprites = {}
        self._meter_sprite = None
        self._dock_sprite = None
        self._cargo_sprite = None
        self._fail_overlay = None

        for sprite in self.current_level.get_sprites():
            if sprite.name == "meter":
                self._meter_sprite = sprite
            elif sprite.name == "dock":
                self._dock_sprite = sprite
            elif sprite.name == "cargo":
                self._cargo_sprite = sprite
            elif sprite.name == "fail_overlay":
                self._fail_overlay = sprite
            elif sprite.name.startswith("door_"):
                panel_id = sprite.name.split("_", 1)[1][0]
                self._door_sprites.append((sprite, panel_id))
            elif sprite.name.startswith("panel_"):
                panel_id = sprite.name.split("_", 1)[1]
                self._panel_sprites[panel_id] = sprite

        self._sync_visuals()

    def _sync_visuals(self) -> None:
        spec = self.current_spec
        if self._meter_sprite is not None:
            self._meter_sprite.pixels = meter_pixels(self._remaining_actions, spec.budget, self._mode == "fail_pause")

        if self._dock_sprite is not None:
            self._dock_sprite.pixels = dock_pixels(self._cargo_pos == spec.dock)

        if self._cargo_sprite is not None:
            px, py = logical_to_pixel(self._cargo_pos)
            self._cargo_sprite.set_position(px + 1, py + 1)
            self._cargo_sprite.pixels = cargo_pixels(self._cargo_pos == spec.dock)

        for sprite, panel_id in self._door_sprites:
            sprite.pixels = door_pixels(panel_id, self._active_door == panel_id)

        for panel_id, sprite in self._panel_sprites.items():
            sprite.pixels = panel_pixels(panel_id, self._active_door == panel_id)

        if self._fail_overlay is not None:
            self._fail_overlay.set_visible(self._mode == "fail_pause")

    def _consume_action(self) -> None:
        self._remaining_actions = max(0, self._remaining_actions - 1)

    def _toggle_panel(self, panel_id: str) -> None:
        self._active_door = None if self._active_door == panel_id else panel_id

    def _attempt_move(self, direction: str) -> None:
        dx, dy = MOVE_DELTAS[direction]
        target = (self._cargo_pos[0] + dx, self._cargo_pos[1] + dy)
        tile = _tile_at(self.current_spec, target)
        if _passable(tile, self._active_door):
            self._cargo_pos = target

    def _resolve_click(self, x: int, y: int) -> None:
        if self._mode == "success_pause":
            self.next_level()
            self.complete_action()
            return

        self._consume_action()
        direction = _arrow_hit(x, y)
        if direction is not None:
            self._attempt_move(direction)
        else:
            panel_id = _panel_hit(self.current_spec, x, y)
            if panel_id is not None:
                self._toggle_panel(panel_id)

        if self._cargo_pos == self.current_spec.dock:
            self._mode = "success_pause"
        elif self._remaining_actions <= 0:
            self.lose()
            self.complete_action()
            return

        self._sync_visuals()
        self.complete_action()

    def step(self) -> None:
        raw_action_id = getattr(self.action, "id", 0)
        action_id = int(getattr(raw_action_id, "value", raw_action_id))
        payload = dict(getattr(self.action, "data", {}) or {})

        if action_id != 6:
            self.complete_action()
            return

        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        self._resolve_click(x, y)
