from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

WHITE = 0
HUD_BG = 1
BORDER = 2
SPENT = 3
ANCHOR_NOTCH = 4
OUTLINE = 5
MAGENTA = 6
MAGENTA_LIGHT = 7
FAIL = 8
BLUE = 9
BLUE_LIGHT = 10
ACTIVE = 11
REMAINING = 14

CELL_SIZE = 4
HUD_HEIGHT = 8
PLAY_ORIGIN_X = 4
PLAY_ORIGIN_Y = 8
PLAY_SIZE = 56
PAWN_SIZE_CELLS = 2
PAWN_SIZE_PX = PAWN_SIZE_CELLS * CELL_SIZE
LEASH_DISTANCE = 4
BOARD_LIMIT = 12


LEVEL_SPECS = [
    {
        "name": "one_switch",
        "blue_pos": (5, 6),
        "blue_mark": (1, 6),
        "magenta_pos": (9, 6),
        "magenta_mark": (9, 6),
        "active": "blue",
        "budget": 16,
        "marker_style": "standard",
    },
    {
        "name": "leapfrog",
        "blue_pos": (0, 9),
        "blue_mark": (0, 9),
        "magenta_pos": (12, 7),
        "magenta_mark": (12, 7),
        "active": "blue",
        "budget": 52,
        "marker_style": "standard",
    },
    {
        "name": "dock_exactly",
        "blue_pos": (1, 10),
        "blue_mark": (1, 10),
        "magenta_pos": (11, 5),
        "magenta_mark": (11, 5),
        "active": "blue",
        "budget": 56,
        "marker_style": "large",
    },
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _transparent(width: int, height: int) -> np.ndarray:
    return np.full((height, width), -1, dtype=np.int8)


def _logical_to_pixel(position: tuple[int, int]) -> tuple[int, int]:
    x, y = position
    return PLAY_ORIGIN_X + x * CELL_SIZE, PLAY_ORIGIN_Y + y * CELL_SIZE


def _pawn_pixels(fill: int, trim: int, active: bool) -> np.ndarray:
    pixels = np.full((PAWN_SIZE_PX, PAWN_SIZE_PX), fill, dtype=np.int8)
    pixels[0, :] = trim
    pixels[-1, :] = trim
    pixels[:, 0] = trim
    pixels[:, -1] = trim
    pixels[1, 1:-1] = fill
    pixels[-2, 1:-1] = fill
    pixels[1:-1, 1] = fill
    pixels[1:-1, -2] = fill
    pixels[2:6, 2:6] = fill
    if active:
        pixels[0:2, 0:2] = ACTIVE
        pixels[0:2, -2:] = ACTIVE
        pixels[-2:, 0:2] = ACTIVE
        pixels[-2:, -2:] = ACTIVE
    return pixels


def _standard_marker_pixels(color: int) -> np.ndarray:
    pixels = _transparent(PAWN_SIZE_PX, PAWN_SIZE_PX)
    pixels[0, 0:3] = color
    pixels[0:3, 0] = color
    pixels[0, -3:] = color
    pixels[0:3, -1] = color
    pixels[-1, 0:3] = color
    pixels[-3:, 0] = color
    pixels[-1, -3:] = color
    pixels[-3:, -1] = color
    return pixels


def _large_marker_pixels(color: int) -> np.ndarray:
    size = 3 * CELL_SIZE
    pixels = _transparent(size, size)
    pixels[0, :] = color
    pixels[-1, :] = color
    pixels[:, 0] = color
    pixels[:, -1] = color
    pixels[4:8, 4:8] = color
    pixels[1:3, 1:3] = ANCHOR_NOTCH
    pixels[1:3, 3:7:2] = ANCHOR_NOTCH
    pixels[3:7:2, 1:3] = ANCHOR_NOTCH
    return pixels


def _tether_pixels(mark: tuple[int, int], pawn: tuple[int, int], color: int) -> tuple[np.ndarray, int, int]:
    mark_px = _logical_to_pixel(mark)
    pawn_px = _logical_to_pixel(pawn)
    start_x = mark_px[0] + PAWN_SIZE_PX // 2
    start_y = mark_px[1] + PAWN_SIZE_PX // 2
    end_x = pawn_px[0] + PAWN_SIZE_PX // 2
    end_y = pawn_px[1] + PAWN_SIZE_PX // 2

    min_x = min(start_x, end_x)
    max_x = max(start_x, end_x)
    min_y = min(start_y, end_y)
    max_y = max(start_y, end_y)
    width = max(1, max_x - min_x + 1)
    height = max(1, max_y - min_y + 1)
    pixels = _transparent(width, height)

    for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
        if (x - min(start_x, end_x)) % 2 == 0:
            pixels[start_y - min_y, x - min_x] = color
    for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
        if (y - min(start_y, end_y)) % 2 == 0:
            pixels[y - min_y, end_x - min_x] = color

    return pixels, min_x, min_y


def _background_pixels() -> np.ndarray:
    pixels = _solid(64, 64, WHITE)
    pixels[:HUD_HEIGHT, :] = HUD_BG

    left = PLAY_ORIGIN_X
    right = PLAY_ORIGIN_X + PLAY_SIZE - 1
    top = PLAY_ORIGIN_Y
    bottom = PLAY_ORIGIN_Y + PLAY_SIZE - 1
    pixels[top : bottom + 1, left] = BORDER
    pixels[top : bottom + 1, right] = BORDER
    pixels[top, left : right + 1] = BORDER
    pixels[bottom, left : right + 1] = BORDER
    return pixels


class SelectAndSettle(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._actions_used = 0
        self._budget = 0
        self._marker_style = "standard"
        self._active = "blue"
        self._blue_pos = (0, 0)
        self._blue_mark = (0, 0)
        self._magenta_pos = (0, 0)
        self._magenta_mark = (0, 0)
        levels = [Level(name=str(spec["name"]), grid_size=(64, 64), data={"spec": spec}) for spec in LEVEL_SPECS]
        super().__init__(
            "select_and_settle",
            levels,
            Camera(0, 0, 64, 64, WHITE, WHITE),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6],
            seed=self._seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        if not isinstance(spec, dict):
            raise ValueError("Level spec missing.")
        self._blue_pos = tuple(spec["blue_pos"])
        self._blue_mark = tuple(spec["blue_mark"])
        self._magenta_pos = tuple(spec["magenta_pos"])
        self._magenta_mark = tuple(spec["magenta_mark"])
        self._active = str(spec["active"])
        self._budget = int(spec["budget"])
        self._actions_used = 0
        self._marker_style = str(spec["marker_style"])
        self._render_state()

    def _pawn_rect(self, position: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = position
        return x, y, x + PAWN_SIZE_CELLS - 1, y + PAWN_SIZE_CELLS - 1

    def _within_bounds(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x <= BOARD_LIMIT and 0 <= y <= BOARD_LIMIT

    def _leash_ok(self, position: tuple[int, int], mark: tuple[int, int]) -> bool:
        return abs(position[0] - mark[0]) + abs(position[1] - mark[1]) <= LEASH_DISTANCE

    def _click_target(self) -> tuple[int, int] | None:
        payload = getattr(self.action, "data", {}) or {}
        x = payload.get("x")
        y = payload.get("y")
        if x is None or y is None:
            return None
        tile = self.camera.display_to_grid(int(x), int(y))
        if tile is None:
            return None
        return int(tile[0]), int(tile[1])

    def _resolve_click(self) -> None:
        target = self._click_target()
        if target is None:
            return

        sprite = self.current_level.get_sprite_at(target[0], target[1], tag="pawn")
        if sprite is None:
            return

        clicked = str(sprite.name)
        if clicked == self._active:
            return

        if clicked == "blue":
            self._magenta_mark = self._magenta_pos
            self._active = "blue"
        elif clicked == "magenta":
            self._blue_mark = self._blue_pos
            self._active = "magenta"

    def _try_move_active(self, dx: int, dy: int) -> None:
        if self._active == "blue":
            current = self._blue_pos
            mark = self._blue_mark
        else:
            current = self._magenta_pos
            mark = self._magenta_mark

        candidate = (current[0] + dx, current[1] + dy)
        if not self._within_bounds(candidate):
            return
        if not self._leash_ok(candidate, mark):
            return

        if self._active == "blue":
            self._blue_pos = candidate
        else:
            self._magenta_pos = candidate

    def _check_win(self) -> bool:
        return self._blue_pos == self._magenta_mark or self._magenta_pos == self._blue_mark

    def _build_budget_sprite(self) -> Sprite:
        pixels = _solid(64, HUD_HEIGHT, HUD_BG)
        pip_count = min(self._budget, 28)
        if self._budget <= 28:
            used_slots = min(self._actions_used, self._budget)
        else:
            used_slots = min(28, (self._actions_used * 28 + self._budget - 1) // self._budget)
        for slot in range(pip_count):
            row = slot // 14
            col = slot % 14
            x0 = PLAY_ORIGIN_X + col * CELL_SIZE
            y0 = row * CELL_SIZE
            color = SPENT if slot < used_slots else REMAINING
            pixels[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = color
        return Sprite(pixels, name="budget", x=0, y=0, layer=6, collidable=False)

    def _build_marker_sprite(self, name: str, position: tuple[int, int], color: int) -> Sprite:
        x, y = _logical_to_pixel(position)
        pixels = _standard_marker_pixels(color) if self._marker_style == "standard" else _large_marker_pixels(color)
        return Sprite(pixels, name=name, x=x, y=y, layer=2, collidable=False)

    def _build_pawn_sprite(self, name: str, position: tuple[int, int], fill: int, trim: int, active: bool) -> Sprite:
        x, y = _logical_to_pixel(position)
        return Sprite(
            _pawn_pixels(fill, trim, active),
            name=name,
            x=x,
            y=y,
            layer=4,
            collidable=True,
            tags=["pawn", "sys_click", "sys_every_pixel"],
        )

    def _build_tether_sprite(self, name: str, mark: tuple[int, int], pawn: tuple[int, int], color: int) -> Sprite:
        pixels, x, y = _tether_pixels(mark, pawn, color)
        return Sprite(pixels, name=name, x=x, y=y, layer=1, collidable=False)

    def _render_state(self) -> None:
        level = self.current_level
        level.remove_all_sprites()
        level.add_sprite(Sprite(_background_pixels(), name="background", x=0, y=0, layer=0, collidable=False))
        level.add_sprite(self._build_tether_sprite("blue_tether", self._blue_mark, self._blue_pos, BLUE_LIGHT))
        level.add_sprite(
            self._build_tether_sprite("magenta_tether", self._magenta_mark, self._magenta_pos, MAGENTA_LIGHT)
        )
        level.add_sprite(self._build_marker_sprite("blue_marker", self._blue_mark, BLUE_LIGHT))
        level.add_sprite(self._build_marker_sprite("magenta_marker", self._magenta_mark, MAGENTA_LIGHT))
        level.add_sprite(self._build_pawn_sprite("blue", self._blue_pos, BLUE, BLUE_LIGHT, self._active == "blue"))
        level.add_sprite(
            self._build_pawn_sprite("magenta", self._magenta_pos, MAGENTA, MAGENTA_LIGHT, self._active == "magenta")
        )
        level.add_sprite(self._build_budget_sprite())

    def _action_id(self) -> int:
        return int(getattr(self.action, "id", GameAction.RESET).value)

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

        if self._state.name in {"WIN", "GAME_OVER"}:
            self.complete_action()
            return

        self._actions_used += 1
        action_id = self._action_id()

        if action_id == GameAction.ACTION1.value:
            self._try_move_active(0, -1)
        elif action_id == GameAction.ACTION2.value:
            self._try_move_active(0, 1)
        elif action_id == GameAction.ACTION3.value:
            self._try_move_active(-1, 0)
        elif action_id == GameAction.ACTION4.value:
            self._try_move_active(1, 0)
        elif action_id == GameAction.ACTION6.value:
            self._resolve_click()

        self._render_state()

        if self._check_win():
            self.next_level()
        elif self._actions_used >= self._budget:
            self.lose()

        self.complete_action()
