from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

CELL_SIZE = 8
HUD_HEIGHT = 8
BOARD_WIDTH = 8
BOARD_HEIGHT = 7
FRAME_WIDTH = 64
FRAME_HEIGHT = 64
PLAYFIELD_Y = HUD_HEIGHT

ACTION_TO_DELTA: dict[int, tuple[int, int]] = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
    int(GameAction.ACTION5.value): (0, 0),
}

PAIR_COLORS: dict[str, tuple[int, int]] = {"blue": (9, 10), "purple": (15, 7), "magenta": (6, 7), "orange": (12, 11)}
PortalSpec = dict[str, object]


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    return cell[0] * CELL_SIZE, PLAYFIELD_Y + cell[1] * CELL_SIZE


def _empty_canvas(width: int, height: int, color: int = 0) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _make_floor_pixels() -> np.ndarray:
    pixels = _empty_canvas(FRAME_WIDTH, FRAME_HEIGHT, 0)
    for gy in range(BOARD_HEIGHT):
        for gx in range(BOARD_WIDTH):
            px, py = _cell_origin((gx, gy))
            inset_color = 1 if (gx + gy) % 2 == 0 else 0
            pixels[py : py + CELL_SIZE, px : px + CELL_SIZE] = 0
            pixels[py + 1 : py + CELL_SIZE - 1, px + 1 : px + CELL_SIZE - 1] = inset_color

    border = 5
    pixels[PLAYFIELD_Y, :] = border
    pixels[FRAME_HEIGHT - 1, :] = border
    pixels[PLAYFIELD_Y:, 0] = border
    pixels[PLAYFIELD_Y:, FRAME_WIDTH - 1] = border
    return pixels


def _make_hazard_pixels(cell: tuple[int, int]) -> np.ndarray:
    px, py = _cell_origin(cell)
    pixels = _empty_canvas(CELL_SIZE, CELL_SIZE)
    stripe = (8, 12, 13)
    for y in range(CELL_SIZE):
        for x in range(CELL_SIZE):
            pixels[y, x] = stripe[(px + py + x - y) % len(stripe)]
    return pixels


def _make_portal_pixels(
    width: int, height: int, primary: int, secondary: int, *, entry: bool, orientation: str
) -> np.ndarray:
    pixels = _empty_canvas(width, height, secondary)
    pixels[1:-1, 1:-1] = primary
    pixels[2:-2, 2:-2] = secondary
    light = 1

    if entry:
        if orientation == "horizontal":
            chevrons = ((4, 5), (6, 4), (6, 6), (8, 5), (10, 4), (10, 6), (12, 5))
        else:
            chevrons = ((3, 4), (4, 6), (2, 6), (3, 8), (4, 10), (2, 10), (3, 12))
        for x, y in chevrons:
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = light
                if x + 1 < width:
                    pixels[y, x + 1] = light
    else:
        cx = width // 2
        cy = height // 2
        for dx, dy in ((-3, 0), (-2, -2), (0, -3), (2, -2), (3, 0), (2, 2), (0, 3), (-2, 2)):
            x = cx + dx
            y = cy + dy
            if 0 <= x < width and 0 <= y < height:
                pixels[y, x] = light
        pixels[cy - 1 : cy + 2, cx - 1 : cx + 2] = primary
        pixels[cy, cx] = light
    return pixels


def _make_dock_pixels(width: int, height: int) -> np.ndarray:
    pixels = _empty_canvas(width, height, 14)
    pixels[0, :] = 5
    pixels[-1, :] = 5
    pixels[:, 0] = 5
    pixels[:, -1] = 5
    pixels[1:-1, 1:-1] = 14
    pixels[2:-2, 2:-2] = 10
    pixels[3:-3, 3:-3] = 14
    if width >= 8 and height >= 8:
        pixels[height // 2 - 1 : height // 2 + 1, 2:-2] = 14
        pixels[-2, width // 2 - 1 : width // 2 + 1] = 0
        pixels[1, 2:-2] = 10
    return pixels


def _make_player_pixels(body: int, highlight: int) -> np.ndarray:
    pixels = _empty_canvas(7, 7, 0)
    pixels[1:6, 1:6] = body
    pixels[0, 2:5] = 5
    pixels[6, 2:5] = 5
    pixels[2:5, 0] = 5
    pixels[2:5, 6] = 5
    pixels[1:3, 2:5] = highlight
    pixels[2:5, 3] = highlight
    return pixels


def _make_edge_pixels(width: int, height: int, color: int) -> np.ndarray:
    return _empty_canvas(width, height, color)


class MoveMeterDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.max_moves = 1
        self.remaining_moves = 1
        self.freeze_mode = "play"
        self.fail_reason = ""

    def configure(self, *, remaining_moves: int, max_moves: int, freeze_mode: str, fail_reason: str) -> None:
        self.remaining_moves = max(0, int(remaining_moves))
        self.max_moves = max(1, int(max_moves))
        self.freeze_mode = freeze_mode
        self.fail_reason = fail_reason

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[0:HUD_HEIGHT, :] = 0
        frame[1:7, 2:62] = 2
        frame[0, 1:63] = 5
        frame[7, 1:63] = 5
        frame[1:7, 1] = 5
        frame[1:7, 62] = 5

        if self.freeze_mode == "failure":
            fill_color = 8
            filled = 0 if self.fail_reason == "budget" else max(1, round(60 * self.remaining_moves / self.max_moves))
            frame[1:7, 2:62] = 13
        elif self.freeze_mode == "success":
            fill_color = 11
            filled = 60
        else:
            ratio = self.remaining_moves / self.max_moves
            fill_color = 12 if ratio < 0.3 else 11
            filled = round(60 * ratio)

        if filled > 0:
            frame[1:7, 2 : 2 + filled] = fill_color
        return frame


LEVEL_SPECS = (
    {
        "start": (3, 0),
        "budget": 12,
        "hazards": frozenset((x, y) for y in (2, 3, 4) for x in range(8)),
        "dock": frozenset({(5, 5), (6, 5), (5, 6), (6, 6)}),
        "portals": (
            {"pair_id": "blue", "orientation": "horizontal", "entry": ((3, 1), (4, 1)), "exit": ((2, 5), (3, 5))},
        ),
    },
    {
        "start": (0, 5),
        "budget": 24,
        "hazards": frozenset(
            {(x, y) for x in (2, 3) for y in range(7)}
            | {(x, y) for x in (5, 6) for y in range(7)}
            | {(7, 0), (7, 1), (7, 2)}
        ),
        "dock": frozenset({(7, 5), (7, 6)}),
        "portals": (
            {"pair_id": "blue", "orientation": "vertical", "entry": ((1, 1), (1, 2)), "exit": ((4, 1), (4, 2))},
            {"pair_id": "purple", "orientation": "vertical", "entry": ((4, 4), (4, 5)), "exit": ((7, 3), (7, 4))},
        ),
    },
    {
        "start": (0, 3),
        "budget": 12,
        "hazards": frozenset(
            {
                (x, y)
                for y in range(7)
                for x in range(8)
                if (x, y)
                not in {
                    (0, 2),
                    (1, 2),
                    (0, 3),
                    (1, 3),
                    (0, 4),
                    (1, 4),
                    (3, 0),
                    (4, 0),
                    (3, 1),
                    (4, 1),
                    (3, 3),
                    (4, 3),
                    (3, 4),
                    (4, 4),
                    (3, 5),
                    (4, 5),
                    (3, 6),
                    (4, 6),
                    (6, 2),
                    (7, 2),
                    (6, 3),
                    (7, 3),
                    (6, 4),
                    (7, 4),
                }
            }
        ),
        "dock": frozenset({(6, 2), (7, 2), (6, 3), (7, 3)}),
        "portals": (
            {"pair_id": "blue", "orientation": "horizontal", "entry": ((0, 2), (1, 2)), "exit": ((3, 0), (4, 0))},
            {"pair_id": "purple", "orientation": "horizontal", "entry": ((0, 4), (1, 4)), "exit": ((3, 5), (4, 5))},
            {"pair_id": "magenta", "orientation": "horizontal", "entry": ((3, 1), (4, 1)), "exit": ((3, 3), (4, 3))},
            {"pair_id": "orange", "orientation": "horizontal", "entry": ((3, 4), (4, 4)), "exit": ((6, 4), (7, 4))},
        ),
    },
)


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for level_index, spec in enumerate(LEVEL_SPECS):
        sprites: list[Sprite] = [
            Sprite(_make_floor_pixels(), name="floor", x=0, y=0, layer=0),
            Sprite(_make_edge_pixels(FRAME_WIDTH, 1, 5), name="edge_top", x=0, y=PLAYFIELD_Y, layer=4),
            Sprite(_make_edge_pixels(FRAME_WIDTH, 1, 5), name="edge_bottom", x=0, y=FRAME_HEIGHT - 1, layer=4),
            Sprite(_make_edge_pixels(1, FRAME_HEIGHT - PLAYFIELD_Y, 5), name="edge_left", x=0, y=PLAYFIELD_Y, layer=4),
            Sprite(
                _make_edge_pixels(1, FRAME_HEIGHT - PLAYFIELD_Y, 5),
                name="edge_right",
                x=FRAME_WIDTH - 1,
                y=PLAYFIELD_Y,
                layer=4,
            ),
        ]

        for hazard in sorted(spec["hazards"]):
            px, py = _cell_origin(hazard)
            sprites.append(
                Sprite(_make_hazard_pixels(hazard), name=f"hazard_{hazard[0]}_{hazard[1]}", x=px, y=py, layer=1)
            )

        for portal in spec["portals"]:
            primary, secondary = PAIR_COLORS[portal["pair_id"]]
            ex0, ey0 = portal["entry"][0]
            ox0, oy0 = portal["exit"][0]
            entry_w = CELL_SIZE * (2 if portal["orientation"] == "horizontal" else 1)
            entry_h = CELL_SIZE * (1 if portal["orientation"] == "horizontal" else 2)
            exit_w = entry_w
            exit_h = entry_h
            entry_px, entry_py = _cell_origin((ex0, ey0))
            exit_px, exit_py = _cell_origin((ox0, oy0))
            sprites.append(
                Sprite(
                    _make_portal_pixels(
                        entry_w, entry_h, primary, secondary, entry=True, orientation=portal["orientation"]
                    ),
                    name=f"{portal['pair_id']}_entry",
                    x=entry_px,
                    y=entry_py,
                    layer=2,
                )
            )
            sprites.append(
                Sprite(
                    _make_portal_pixels(
                        exit_w, exit_h, primary, secondary, entry=False, orientation=portal["orientation"]
                    ),
                    name=f"{portal['pair_id']}_exit",
                    x=exit_px,
                    y=exit_py,
                    layer=2,
                )
            )

        min_dock_x = min(x for x, _ in spec["dock"])
        max_dock_x = max(x for x, _ in spec["dock"])
        min_dock_y = min(y for _, y in spec["dock"])
        max_dock_y = max(y for _, y in spec["dock"])
        dock_px, dock_py = _cell_origin((min_dock_x, min_dock_y))
        dock_width = (max_dock_x - min_dock_x + 1) * CELL_SIZE
        dock_height = (max_dock_y - min_dock_y + 1) * CELL_SIZE
        sprites.append(Sprite(_make_dock_pixels(dock_width, dock_height), name="dock", x=dock_px, y=dock_py, layer=1))

        start_px, start_py = _cell_origin(spec["start"])
        sprites.append(Sprite(_make_player_pixels(9, 10), name="player", x=start_px + 1, y=start_py + 1, layer=5))
        levels.append(Level(sprites=sprites, grid_size=(FRAME_WIDTH, FRAME_HEIGHT), data={"spec_index": level_index}))
    return levels


levels = _build_levels()


class HazardSkipPortals(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._meter = MoveMeterDisplay()
        self._remaining_moves = 1
        self._max_moves = 1
        self._player_cell = (0, 0)
        self._freeze_mode = "play"
        self._fail_reason = ""
        self._portal_lookup: dict[tuple[int, int], tuple[PortalSpec, int]] = {}
        self._route_score = 0
        super().__init__(
            game_id="hazard_skip_portals",
            levels=levels,
            camera=Camera(0, 0, FRAME_WIDTH, FRAME_HEIGHT, 0, 0, [self._meter]),
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec_index = int(level.get_data("spec_index"))
        self._spec = LEVEL_SPECS[spec_index]
        self._player_cell = self._spec["start"]
        self._remaining_moves = self._spec["budget"]
        self._max_moves = self._spec["budget"]
        self._freeze_mode = "play"
        self._fail_reason = ""
        self._route_score = 0
        self._portal_lookup = {}
        for portal in self._spec["portals"]:
            for idx, cell in enumerate(portal["entry"]):
                self._portal_lookup[cell] = (portal, idx)
        self._sync_visuals()

    def _player_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_name("player")[0]

    def _dock_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_name("dock")[0]

    def _edge_sprites(self) -> list[Sprite]:
        return [
            self.current_level.get_sprites_by_name("edge_top")[0],
            self.current_level.get_sprites_by_name("edge_bottom")[0],
            self.current_level.get_sprites_by_name("edge_left")[0],
            self.current_level.get_sprites_by_name("edge_right")[0],
        ]

    def _set_player_visual(self) -> None:
        player = self._player_sprite()
        px, py = _cell_origin(self._player_cell)
        player.set_position(px, py + 1)
        if self._freeze_mode == "failure":
            player.pixels = _make_player_pixels(8, 13)
        elif self._freeze_mode == "success":
            player.pixels = _make_player_pixels(11, 14)
        else:
            player.pixels = _make_player_pixels(9, 10)

    def _set_dock_visual(self) -> None:
        dock = self._dock_sprite()
        if self._freeze_mode == "success":
            dock.pixels = np.where(dock.pixels == 10, 11, dock.pixels)
        else:
            width = dock.pixels.shape[1]
            height = dock.pixels.shape[0]
            dock.pixels = _make_dock_pixels(width, height)

    def _set_edge_visual(self) -> None:
        edge_color = 5
        if self._freeze_mode == "success":
            edge_color = 11
        elif self._freeze_mode == "failure":
            edge_color = 13
        top, bottom, left, right = self._edge_sprites()
        top.pixels = _make_edge_pixels(FRAME_WIDTH, 1, edge_color)
        bottom.pixels = _make_edge_pixels(FRAME_WIDTH, 1, edge_color)
        left.pixels = _make_edge_pixels(1, FRAME_HEIGHT - PLAYFIELD_Y, edge_color)
        right.pixels = _make_edge_pixels(1, FRAME_HEIGHT - PLAYFIELD_Y, edge_color)

    def _sync_visuals(self) -> None:
        self._set_player_visual()
        self._set_dock_visual()
        self._set_edge_visual()
        self._meter.configure(
            remaining_moves=self._remaining_moves,
            max_moves=self._max_moves,
            freeze_mode=self._freeze_mode,
            fail_reason=self._fail_reason,
        )

    def _trigger_failure(self, reason: str) -> None:
        self._freeze_mode = "failure"
        self._fail_reason = reason
        self._sync_visuals()
        self.lose()

    def _trigger_success(self) -> None:
        if self.is_last_level():
            self._freeze_mode = "success"
            self._fail_reason = ""
            self._sync_visuals()
            self.next_level()
            return
        self._freeze_mode = "success"
        self._fail_reason = ""
        self._sync_visuals()

    def _teleport_if_entry(self, old_cell: tuple[int, int], new_cell: tuple[int, int]) -> tuple[int, int]:
        if new_cell == old_cell:
            return new_cell
        portal_data = self._portal_lookup.get(new_cell)
        if portal_data is None:
            return new_cell
        portal, offset = portal_data
        return portal["exit"][offset]

    def _acknowledge_freeze(self) -> None:
        if self._freeze_mode == "success":
            self.next_level()

    def step(self) -> None:
        if self._freeze_mode == "success":
            self._acknowledge_freeze()
            self.complete_action()
            return

        raw_action_id = getattr(self.action.id, "value", self.action.id)
        action_id = int(raw_action_id)
        if action_id == 0:
            self._sync_visuals()
            self.complete_action()
            return
        dx, dy = ACTION_TO_DELTA.get(action_id, (0, 0))
        self._remaining_moves -= 1

        old_x, old_y = self._player_cell
        dest_x = old_x + dx
        dest_y = old_y + dy
        if 0 <= dest_x < BOARD_WIDTH and 0 <= dest_y < BOARD_HEIGHT:
            moved_cell = (dest_x, dest_y)
        else:
            moved_cell = self._player_cell

        self._player_cell = self._teleport_if_entry(self._player_cell, moved_cell)

        if self._player_cell in self._spec["hazards"]:
            self._trigger_failure("hazard")
        elif self._player_cell in self._spec["dock"]:
            self._route_score += 1
            self._trigger_success()
        elif self._remaining_moves <= 0:
            self._trigger_failure("budget")
        else:
            self._sync_visuals()

        self.complete_action()
