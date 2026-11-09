from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 12
BOARD_SIZE = 12
TILE_SIZE = 4
BOARD_PIXEL_WIDTH = BOARD_SIZE * TILE_SIZE

COLOR_WHITE = 0
COLOR_BACKGROUND = 1
COLOR_NEUTRAL = 2
COLOR_SPENT_PIP = 3
COLOR_WALL = 4
COLOR_OUTLINE = 5
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_BLUE_HIGHLIGHT = 10
COLOR_RED_HIGHLIGHT = 13
COLOR_GREEN = 14

PLAYER_TAG = "player"
BORDER_TAG = "border"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

COLOR_BY_NAME = {"neutral": COLOR_NEUTRAL, "red": COLOR_RED, "blue": COLOR_BLUE}
DELTA_BY_ACTION = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    title: str
    budget: int
    rows: tuple[str, ...]


LEVEL_SPECS = (
    LevelSpec(
        name="lesson_one",
        title="Washer, Gate, Dock",
        budget=45,
        rows=(
            "############",
            "############",
            "############",
            "############",
            "##B.....d###",
            "##.#########",
            "##.#########",
            "##.#########",
            "##..b#######",
            "##S#########",
            "############",
            "############",
        ),
    ),
    LevelSpec(
        name="lesson_two",
        title="Overwrite Order",
        budget=45,
        rows=(
            "############",
            "############",
            "############",
            "############",
            "############",
            "##R..bB.d###",
            "##.###.#####",
            "##.###.#####",
            "##..r..#####",
            "##S#########",
            "############",
            "############",
        ),
    ),
    LevelSpec(
        name="lesson_three",
        title="Neutral Reset",
        budget=51,
        rows=(
            "############",
            "############",
            "############",
            "############",
            "############",
            "##R.nN.Bd###",
            "##.###..####",
            "##..r##b####",
            "##S#########",
            "############",
            "############",
            "############",
        ),
    ),
)


def _solid_sprite(
    width: int, height: int, color: int, *, name: str, x: int, y: int, layer: int = 0, tags: list[str] | None = None
) -> Sprite:
    return Sprite(
        pixels=np.full((height, width), color, dtype=np.int16),
        name=name,
        x=x,
        y=y,
        layer=layer,
        collidable=False,
        tags=list(tags or []),
    )


def _tile_pixels(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int16)


def _player_pixels(color: int) -> np.ndarray:
    return _tile_pixels(
        [[-1, color, color, -1], [color, color, color, color], [color, color, color, color], [-1, color, color, -1]]
    )


def _washer_pixels(color: int, highlight: int) -> np.ndarray:
    return _tile_pixels(
        [
            [color, color, color, color],
            [color, COLOR_WHITE, COLOR_WHITE, color],
            [color, COLOR_WHITE, COLOR_WHITE, color],
            [color, highlight, highlight, color],
        ]
    )


def _gate_pixels(color: int) -> np.ndarray:
    return _tile_pixels(
        [
            [COLOR_OUTLINE, color, color, COLOR_OUTLINE],
            [COLOR_OUTLINE, color, color, COLOR_OUTLINE],
            [COLOR_OUTLINE, color, color, COLOR_OUTLINE],
            [COLOR_OUTLINE, color, color, COLOR_OUTLINE],
        ]
    )


def _dock_pixels() -> np.ndarray:
    return _tile_pixels(
        [
            [COLOR_OUTLINE, COLOR_OUTLINE, COLOR_OUTLINE, COLOR_OUTLINE],
            [COLOR_OUTLINE, -1, -1, COLOR_OUTLINE],
            [COLOR_OUTLINE, COLOR_BLUE, COLOR_BLUE, COLOR_OUTLINE],
            [COLOR_OUTLINE, COLOR_OUTLINE, COLOR_OUTLINE, COLOR_OUTLINE],
        ]
    )


def _neutralizer_pixels() -> np.ndarray:
    return _tile_pixels(
        [
            [COLOR_NEUTRAL, COLOR_WHITE, COLOR_WHITE, COLOR_NEUTRAL],
            [COLOR_WHITE, COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_WHITE],
            [COLOR_WHITE, COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_WHITE],
            [COLOR_NEUTRAL, COLOR_WHITE, COLOR_WHITE, COLOR_NEUTRAL],
        ]
    )


def _logical_to_pixel(x: int, y: int) -> tuple[int, int]:
    return (BOARD_ORIGIN_X + (x * TILE_SIZE), BOARD_ORIGIN_Y + (y * TILE_SIZE))


def _build_tile_sprite(char: str, x: int, y: int) -> Sprite | None:
    px, py = _logical_to_pixel(x, y)
    if char == "#":
        return _solid_sprite(TILE_SIZE, TILE_SIZE, COLOR_WALL, name=f"wall_{x}_{y}", x=px, y=py, layer=-1)
    if char == "b":
        return Sprite(_washer_pixels(COLOR_BLUE, COLOR_BLUE_HIGHLIGHT), name=f"washer_blue_{x}_{y}", x=px, y=py)
    if char == "r":
        return Sprite(_washer_pixels(COLOR_RED, COLOR_RED_HIGHLIGHT), name=f"washer_red_{x}_{y}", x=px, y=py)
    if char == "n":
        return Sprite(_neutralizer_pixels(), name=f"neutralizer_{x}_{y}", x=px, y=py)
    if char == "B":
        return Sprite(_gate_pixels(COLOR_BLUE), name=f"gate_blue_{x}_{y}", x=px, y=py)
    if char == "R":
        return Sprite(_gate_pixels(COLOR_RED), name=f"gate_red_{x}_{y}", x=px, y=py)
    if char == "N":
        return Sprite(_gate_pixels(COLOR_NEUTRAL), name=f"gate_neutral_{x}_{y}", x=px, y=py)
    if char == "d":
        return Sprite(_dock_pixels(), name=f"dock_blue_{x}_{y}", x=px, y=py)
    return None


def _parse_start(rows: tuple[str, ...]) -> tuple[int, int]:
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == "S":
                return (x, y)
    raise ValueError("Level map is missing a start tile.")


def _level_data(spec: LevelSpec) -> dict[str, object]:
    return {
        "rows": list(spec.rows),
        "budget": int(spec.budget),
        "start": list(_parse_start(spec.rows)),
        "title": spec.title,
    }


def _build_level(spec: LevelSpec) -> Level:
    sprites: list[Sprite] = [
        _solid_sprite(64, 64, COLOR_BACKGROUND, name="background", x=0, y=0, layer=-5),
        _solid_sprite(
            BOARD_PIXEL_WIDTH,
            BOARD_PIXEL_WIDTH,
            COLOR_WHITE,
            name="board_floor",
            x=BOARD_ORIGIN_X,
            y=BOARD_ORIGIN_Y,
            layer=-4,
        ),
        _solid_sprite(
            BOARD_PIXEL_WIDTH,
            1,
            COLOR_WALL,
            name="border_top",
            x=BOARD_ORIGIN_X,
            y=BOARD_ORIGIN_Y,
            layer=5,
            tags=[BORDER_TAG],
        ),
        _solid_sprite(
            BOARD_PIXEL_WIDTH,
            1,
            COLOR_WALL,
            name="border_bottom",
            x=BOARD_ORIGIN_X,
            y=BOARD_ORIGIN_Y + BOARD_PIXEL_WIDTH - 1,
            layer=5,
            tags=[BORDER_TAG],
        ),
        _solid_sprite(
            1,
            BOARD_PIXEL_WIDTH,
            COLOR_WALL,
            name="border_left",
            x=BOARD_ORIGIN_X,
            y=BOARD_ORIGIN_Y,
            layer=5,
            tags=[BORDER_TAG],
        ),
        _solid_sprite(
            1,
            BOARD_PIXEL_WIDTH,
            COLOR_WALL,
            name="border_right",
            x=BOARD_ORIGIN_X + BOARD_PIXEL_WIDTH - 1,
            y=BOARD_ORIGIN_Y,
            layer=5,
            tags=[BORDER_TAG],
        ),
    ]

    start_x, start_y = _parse_start(spec.rows)
    for y, row in enumerate(spec.rows):
        for x, char in enumerate(row):
            tile_sprite = _build_tile_sprite(char, x, y)
            if tile_sprite is not None:
                sprites.append(tile_sprite)

    player_px, player_py = _logical_to_pixel(start_x, start_y)
    sprites.append(
        Sprite(
            _player_pixels(COLOR_NEUTRAL),
            name="player",
            x=player_px,
            y=player_py,
            layer=10,
            collidable=False,
            tags=[PLAYER_TAG],
        )
    )

    return Level(sprites=sprites, grid_size=(64, 64), data=_level_data(spec), name=spec.name)


LEVELS = [_build_level(spec) for spec in LEVEL_SPECS]


class BudgetPipDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.capacity = 0
        self.remaining = 0

    def configure(self, capacity: int, remaining: int | None = None) -> None:
        self.capacity = max(0, int(capacity))
        if remaining is None:
            self.remaining = self.capacity
            return
        self.remaining = max(0, min(int(remaining), self.capacity))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.capacity <= 0:
            return frame

        start_x = 13
        for pip_idx in range(self.capacity):
            color = COLOR_GREEN if pip_idx < self.remaining else COLOR_SPENT_PIP
            x = start_x + pip_idx
            if 0 <= x < frame.shape[1]:
                frame[3:7, x] = color
        return frame


class ColorWasherRoute(ARCBaseGame):
    def __init__(self) -> None:
        self._budget_display = BudgetPipDisplay()
        self._status = "playing"
        self._route_score = 0
        self.player_x = 0
        self.player_y = 0
        self.player_color = "neutral"
        self.remaining_budget = 0
        super().__init__(
            "color_washer_route",
            LEVELS,
            Camera(0, 0, 64, 64, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._budget_display]),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE],
        )

    def on_set_level(self, _level: Level) -> None:
        start = self.current_level.get_data("start")
        self.player_x = int(start[0])
        self.player_y = int(start[1])
        self.player_color = "neutral"
        self.remaining_budget = int(self.current_level.get_data("budget"))
        self._status = "playing"
        self._budget_display.configure(self.remaining_budget)
        self._set_border_color(COLOR_WALL)
        self._sync_player_sprite()

    def _sync_player_sprite(self) -> None:
        player = self.current_level.get_sprites_by_tag(PLAYER_TAG)[0]
        px, py = _logical_to_pixel(self.player_x, self.player_y)
        player.set_position(px, py)
        player.pixels = _player_pixels(COLOR_BY_NAME[self.player_color])

    def _set_border_color(self, color: int) -> None:
        for border in self.current_level.get_sprites_by_tag(BORDER_TAG):
            border.pixels[:] = color

    def _rows(self) -> list[str]:
        rows = self.current_level.get_data("rows")
        return [str(row) for row in rows]

    def _tile_at(self, x: int, y: int) -> str:
        rows = self._rows()
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return "#"
        return rows[y][x]

    def _enter_won_state(self) -> None:
        self._status = "level_won"
        self._set_border_color(COLOR_GREEN)

    def _advance_from_won_state(self) -> None:
        if self.level_index >= len(LEVELS) - 1:
            self._set_border_color(COLOR_GREEN)
            self.next_level()
            return
        self.next_level()

    def _spend_budget(self) -> bool:
        if self.remaining_budget > 0:
            self.remaining_budget -= 1
        self._budget_display.configure(self.current_level.get_data("budget"), self.remaining_budget)
        return self.remaining_budget > 0

    def _resolve_move(self, action_id: int) -> None:
        has_budget_left = self._spend_budget()
        dx, dy = DELTA_BY_ACTION[action_id]
        target_x = self.player_x + dx
        target_y = self.player_y + dy
        tile = self._tile_at(target_x, target_y)
        next_color = self.player_color
        moved = False
        won = False

        if tile in {"#", ""}:
            moved = False
        elif tile in {".", "S"}:
            moved = True
        elif tile == "b":
            moved = True
            next_color = "blue"
        elif tile == "r":
            moved = True
            next_color = "red"
        elif tile == "n":
            moved = True
            next_color = "neutral"
        elif tile == "B":
            moved = self.player_color == "blue"
        elif tile == "R":
            moved = self.player_color == "red"
        elif tile == "N":
            moved = self.player_color == "neutral"
        elif tile == "d":
            moved = self.player_color == "blue"
            won = moved

        if moved:
            self.player_x = target_x
            self.player_y = target_y
            self.player_color = next_color
            self._sync_player_sprite()

        if won:
            if self.level_index >= len(LEVELS) - 1:
                self._set_border_color(COLOR_GREEN)
                self.next_level()
                return
            self._enter_won_state()
            return

        if not has_budget_left:
            self._status = "level_failed"
            self._set_border_color(COLOR_RED)
            self.lose()

    def step(self) -> None:
        action_id = int(self.action.id.value)

        if action_id == ACTION_CLICK:
            self.complete_action()
            return

        if self._status == "level_failed":
            self.complete_action()
            return

        if self._status == "level_won":
            if action_id in {ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE}:
                self._advance_from_won_state()
            self.complete_action()
            return

        if action_id == ACTION_SPACE:
            self.set_level(self.level_index)
            self.complete_action()
            return

        if action_id in DELTA_BY_ACTION:
            self._resolve_move(action_id)

        self.complete_action()

    def _get_hidden_state(self) -> np.ndarray:
        color_code = {"neutral": 0, "red": 1, "blue": 2}[self.player_color]
        status_code = {"playing": 0, "level_won": 1, "level_failed": 2}.get(self._status, 0)
        return np.array(
            [self.level_index, self.player_x, self.player_y, color_code, self.remaining_budget, status_code],
            dtype=np.int16,
        )


class ColorWasherRoute0001(ColorWasherRoute):
    pass
