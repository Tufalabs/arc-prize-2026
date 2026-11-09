from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_SIZE = 64
BACKGROUND = 0
PADDING = 0
PLAYFIELD_FILL = 1
PLAYFIELD_BORDER = 3
BUTTON_BORDER = 4
FAIL_RED = 8
ARROW_BLUE = 9
DOCK_BLUE = 10
CRATE_HIGHLIGHT = 11
CRATE_FILL = 12
CRATE_SEAM = 13
SUCCESS_GREEN = 14
DECOY_FILL = 6
DECOY_ACCENT = 15

PLAYFIELD_X0 = 12
PLAYFIELD_Y0 = 12
CELL_SIZE = 4
LOGICAL_SIZE = 10

UP_ACTION = 1
DOWN_ACTION = 2
LEFT_ACTION = 3
RIGHT_ACTION = 4
ACTION_TO_DELTA = {UP_ACTION: (0, -1), DOWN_ACTION: (0, 1), LEFT_ACTION: (-1, 0), RIGHT_ACTION: (1, 0)}


LEVEL_SPECS = [
    {
        "crate_shape": (2, 2),
        "crate_start": (3, 5),
        "dock_pos": (5, 4),
        "decoy_shape": None,
        "decoy_pos": None,
        "move_limit": None,
    },
    {
        "crate_shape": (2, 2),
        "crate_start": (4, 5),
        "dock_pos": (1, 2),
        "decoy_shape": (3, 2),
        "decoy_pos": (6, 1),
        "move_limit": None,
    },
    {
        "crate_shape": (3, 2),
        "crate_start": (1, 6),
        "dock_pos": (5, 3),
        "decoy_shape": (2, 2),
        "decoy_pos": (6, 0),
        "move_limit": 10,
    },
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _panel_pixels(direction: str) -> np.ndarray:
    pixels = _solid(10, 10, PLAYFIELD_FILL)
    pixels[0, :] = BUTTON_BORDER
    pixels[-1, :] = BUTTON_BORDER
    pixels[:, 0] = BUTTON_BORDER
    pixels[:, -1] = BUTTON_BORDER

    if direction == "up":
        coords = {(2, 4), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 4), (6, 4), (7, 4)}
    elif direction == "down":
        coords = {(2, 4), (3, 4), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 3), (5, 4), (5, 5), (6, 4), (7, 4)}
    elif direction == "left":
        coords = {(4, 2), (3, 3), (4, 3), (5, 3), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (3, 5), (4, 5), (5, 5)}
    else:
        coords = {(4, 2), (3, 3), (4, 3), (5, 3), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (3, 5), (4, 5), (5, 5)}
        coords = {(y, 8 - x) for (y, x) in coords}

    for y, x in coords:
        pixels[y, x] = ARROW_BLUE
    return pixels


def _dock_pixels(shape: tuple[int, int], outline_color: int, accent_color: int) -> np.ndarray:
    width = shape[0] * CELL_SIZE
    height = shape[1] * CELL_SIZE
    pixels = _solid(width, height, PLAYFIELD_FILL)
    pixels[0, :] = outline_color
    pixels[-1, :] = outline_color
    pixels[:, 0] = outline_color
    pixels[:, -1] = outline_color
    for y, x in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        pixels[y, x] = accent_color
    return pixels


def _crate_pixels(shape: tuple[int, int], fill_color: int) -> np.ndarray:
    width = shape[0] * CELL_SIZE
    height = shape[1] * CELL_SIZE
    pixels = _solid(width, height, fill_color)
    pixels[0, :] = BUTTON_BORDER
    pixels[-1, :] = BUTTON_BORDER
    pixels[:, 0] = BUTTON_BORDER
    pixels[:, -1] = BUTTON_BORDER
    if height > 2 and width > 2:
        pixels[1, 1:-1] = CRATE_HIGHLIGHT
    for seam_x in range(CELL_SIZE, width, CELL_SIZE):
        pixels[:, seam_x] = CRATE_SEAM
    for seam_y in range(CELL_SIZE, height, CELL_SIZE):
        pixels[seam_y, :] = CRATE_SEAM
    pixels[0, :] = BUTTON_BORDER
    pixels[-1, :] = BUTTON_BORDER
    pixels[:, 0] = BUTTON_BORDER
    pixels[:, -1] = BUTTON_BORDER
    return pixels


def _logical_to_pixels(x: int, y: int) -> tuple[int, int]:
    return PLAYFIELD_X0 + (x * CELL_SIZE), PLAYFIELD_Y0 + (y * CELL_SIZE)


def _named_sprite(name: str, pixels: np.ndarray, x: int, y: int, layer: int) -> Sprite:
    return Sprite(pixels, name=name, x=x, y=y, layer=layer, collidable=False)


def _build_level(spec: dict[str, tuple[int, int] | int | None], index: int) -> Level:
    sprites: list[Sprite] = [
        _named_sprite("playfield_fill", _solid(40, 40, PLAYFIELD_FILL), PLAYFIELD_X0, PLAYFIELD_Y0, 0),
        _named_sprite("border_top", _solid(42, 1, PLAYFIELD_BORDER), 11, 11, 1),
        _named_sprite("border_bottom", _solid(42, 1, PLAYFIELD_BORDER), 11, 52, 1),
        _named_sprite("border_left", _solid(1, 42, PLAYFIELD_BORDER), 11, 11, 1),
        _named_sprite("border_right", _solid(1, 42, PLAYFIELD_BORDER), 52, 11, 1),
        _named_sprite("panel_up", _panel_pixels("up"), 27, 1, 2),
        _named_sprite("panel_right", _panel_pixels("right"), 53, 27, 2),
        _named_sprite("panel_down", _panel_pixels("down"), 27, 53, 2),
        _named_sprite("panel_left", _panel_pixels("left"), 1, 27, 2),
    ]

    crate_shape = tuple(spec["crate_shape"])
    crate_start = tuple(spec["crate_start"])
    dock_pos = tuple(spec["dock_pos"])
    decoy_shape = spec["decoy_shape"]
    decoy_pos = spec["decoy_pos"]
    move_limit = spec["move_limit"]

    dock_x, dock_y = _logical_to_pixels(*dock_pos)
    sprites.append(_named_sprite("dock", _dock_pixels(crate_shape, DOCK_BLUE, ARROW_BLUE), dock_x, dock_y, 2))

    if decoy_shape is not None and decoy_pos is not None:
        decoy_x, decoy_y = _logical_to_pixels(*tuple(decoy_pos))
        sprites.append(
            _named_sprite("decoy", _dock_pixels(tuple(decoy_shape), DECOY_FILL, DECOY_ACCENT), decoy_x, decoy_y, 2)
        )

    crate_x, crate_y = _logical_to_pixels(*crate_start)
    sprites.append(_named_sprite("crate", _crate_pixels(crate_shape, CRATE_FILL), crate_x, crate_y, 3))

    for edge_name in ("up", "right", "down", "left"):
        if edge_name in {"up", "down"}:
            pixels = _solid(42, 1, FAIL_RED)
            x, y = (11, 11) if edge_name == "up" else (11, 52)
        else:
            pixels = _solid(1, 42, FAIL_RED)
            x, y = (52, 11) if edge_name == "right" else (11, 11)
        sprite = _named_sprite(f"fail_edge_{edge_name}", pixels, x, y, 4)
        sprite.set_visible(False)
        sprites.append(sprite)

    if move_limit is not None:
        lamp_rects = [(12, 4), (15, 4), (18, 4), (21, 4), (24, 4), (38, 4), (41, 4), (44, 4), (47, 4), (50, 4)]
        for lamp_idx, (x, y) in enumerate(lamp_rects):
            sprites.append(_named_sprite(f"lamp_{lamp_idx}", _solid(2, 5, SUCCESS_GREEN), x, y, 2))

    return Level(
        sprites=sprites,
        grid_size=(GRID_SIZE, GRID_SIZE),
        data={
            "index": index,
            "crate_shape": crate_shape,
            "crate_start": crate_start,
            "dock_pos": dock_pos,
            "move_limit": move_limit,
        },
        name=f"Level {index + 1}",
    )


levels = [_build_level(spec, idx) for idx, spec in enumerate(LEVEL_SPECS)]


class ArrowCrateDock(ARCBaseGame):
    def __init__(self) -> None:
        self._crate_x = 0
        self._crate_y = 0
        self._crate_shape = (2, 2)
        self._remaining_moves: int | None = None
        self._route_score = 0
        self._crate_sprite: Sprite | None = None
        self._border_sprites: dict[str, Sprite] = {}
        self._lamp_sprites: list[Sprite] = []
        super().__init__(
            "arrow_crate_dock",
            levels,
            Camera(0, 0, GRID_SIZE, GRID_SIZE, BACKGROUND, PADDING),
            False,
            len(LEVEL_SPECS),
            [UP_ACTION, DOWN_ACTION, LEFT_ACTION, RIGHT_ACTION],
        )

    def on_set_level(self, level: Level) -> None:
        self._route_score = int(level.get_data("index") or 0)
        self._crate_shape = tuple(level.get_data("crate_shape"))
        self._crate_x, self._crate_y = tuple(level.get_data("crate_start"))
        self._remaining_moves = level.get_data("move_limit")
        self._crate_sprite = level.get_sprites_by_name("crate")[0]
        self._border_sprites = {
            "up": level.get_sprites_by_name("border_top")[0],
            "right": level.get_sprites_by_name("border_right")[0],
            "down": level.get_sprites_by_name("border_bottom")[0],
            "left": level.get_sprites_by_name("border_left")[0],
        }
        self._lamp_sprites = []
        if self._remaining_moves is not None:
            self._lamp_sprites = [level.get_sprites_by_name(f"lamp_{idx}")[0] for idx in range(10)]
            self._sync_lamps(timeout=False)
        self._sync_crate(CRATE_FILL)
        self._reset_border_colors()

    def _sync_crate(self, fill_color: int) -> None:
        if self._crate_sprite is None:
            return
        self._crate_sprite.pixels = _crate_pixels(self._crate_shape, fill_color)
        px, py = _logical_to_pixels(self._crate_x, self._crate_y)
        self._crate_sprite.set_position(px, py)

    def _sync_lamps(self, *, timeout: bool) -> None:
        if not self._lamp_sprites:
            return
        if timeout:
            for sprite in self._lamp_sprites:
                sprite.pixels = _solid(2, 5, FAIL_RED)
            return
        remaining = int(self._remaining_moves or 0)
        for idx, sprite in enumerate(self._lamp_sprites):
            sprite.pixels = _solid(2, 5, SUCCESS_GREEN if idx < remaining else PLAYFIELD_BORDER)

    def _reset_border_colors(self) -> None:
        for sprite in self._border_sprites.values():
            sprite.pixels[:, :] = PLAYFIELD_BORDER
        for name in ("up", "right", "down", "left"):
            fail_sprite = self.current_level.get_sprites_by_name(f"fail_edge_{name}")[0]
            fail_sprite.set_visible(False)

    def _highlight_fail_edge(self, edge: str) -> None:
        fail_sprite = self.current_level.get_sprites_by_name(f"fail_edge_{edge}")[0]
        fail_sprite.set_visible(True)

    def _dock_matches_crate(self) -> bool:
        dock_x, dock_y = tuple(self.current_level.get_data("dock_pos"))
        return self._crate_x == int(dock_x) and self._crate_y == int(dock_y)

    def _in_bounds(self, x: int, y: int) -> bool:
        shape_w, shape_h = self._crate_shape
        return 0 <= x and 0 <= y and x + shape_w <= LOGICAL_SIZE and y + shape_h <= LOGICAL_SIZE

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        dx, dy = ACTION_TO_DELTA.get(action_id, (0, 0))
        next_x = self._crate_x + dx
        next_y = self._crate_y + dy

        if not self._in_bounds(next_x, next_y):
            self._sync_crate(FAIL_RED)
            if dx < 0:
                self._highlight_fail_edge("left")
            elif dx > 0:
                self._highlight_fail_edge("right")
            elif dy < 0:
                self._highlight_fail_edge("up")
            else:
                self._highlight_fail_edge("down")
            self.lose()
            self.complete_action()
            return

        self._crate_x = next_x
        self._crate_y = next_y
        self._sync_crate(CRATE_FILL)

        if self._remaining_moves is not None:
            self._remaining_moves -= 1
            self._sync_lamps(timeout=False)

        if self._dock_matches_crate():
            self._sync_crate(SUCCESS_GREEN)
            self.next_level()
            self.complete_action()
            return

        if self._remaining_moves == 0:
            self._sync_crate(FAIL_RED)
            self._sync_lamps(timeout=True)
            self.lose()

        self.complete_action()


class ArrowCrateDock0001(ArrowCrateDock):
    pass


class Arrow_crate_dock(ArrowCrateDock):
    pass
