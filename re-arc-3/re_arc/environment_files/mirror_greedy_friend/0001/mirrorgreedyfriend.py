from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

BOARD_ORIGIN_X = 4
BOARD_ORIGIN_Y = 8
CELL_SIZE = 4
BOARD_SIZE = 14
SCREEN_SIZE = 64
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE
BEACON_CELLS = frozenset({(6, 3), (7, 3), (6, 4), (7, 4)})

COLOR_FLOOR = 0
COLOR_FLOOR_ALT = 1
COLOR_AXIS_A = 2
COLOR_AXIS_B = 3
COLOR_FRAME = 4
COLOR_FRAME_INSET = 5
COLOR_BEACON_RING = 11
COLOR_BEACON_CORE = 6
COLOR_PLAYER = 9
COLOR_PLAYER_HI = 10
COLOR_HELPER = 15
COLOR_HELPER_HI = 7
COLOR_PIP_REMAINING = 12
COLOR_PIP_SPENT = 13

LEVEL_SPECS = (
    {"player": (2, 11), "helper": (11, 11), "walls": frozenset(), "budget": 34},
    {"player": (2, 10), "helper": (10, 10), "walls": frozenset({(2, 9), (9, 10)}), "budget": 30},
    {
        "player": (1, 11),
        "helper": (10, 11),
        "walls": frozenset({(1, 10), (3, 11), (9, 11), (2, 8), (8, 8), (5, 8)}),
        "budget": 36,
    },
)

ACTION_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (0, 0)}


def logical_to_pixels(x: int, y: int) -> tuple[int, int]:
    return BOARD_ORIGIN_X + (CELL_SIZE * x), BOARD_ORIGIN_Y + (CELL_SIZE * y)


def make_board_pixels() -> np.ndarray:
    pixels = np.full((SCREEN_SIZE, SCREEN_SIZE), COLOR_FLOOR, dtype=np.int16)
    pixels[:BOARD_ORIGIN_Y, :] = COLOR_FRAME

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            color = COLOR_FLOOR if (x + y) % 2 == 0 else COLOR_FLOOR_ALT
            px, py = logical_to_pixels(x, y)
            pixels[py : py + CELL_SIZE, px : px + CELL_SIZE] = color

    border_left = BOARD_ORIGIN_X - 1
    border_top = BOARD_ORIGIN_Y - 1
    border_right = BOARD_ORIGIN_X + BOARD_PIXEL_SIZE
    border_bottom = SCREEN_SIZE - 1
    pixels[border_top : border_bottom + 1, border_left] = COLOR_FRAME
    pixels[border_top : border_bottom + 1, border_right] = COLOR_FRAME
    pixels[border_top, border_left : border_right + 1] = COLOR_FRAME
    pixels[border_bottom, border_left : border_right + 1] = COLOR_FRAME
    pixels[BOARD_ORIGIN_Y : BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE, BOARD_ORIGIN_X] = COLOR_FRAME_INSET
    pixels[BOARD_ORIGIN_Y : BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE, BOARD_ORIGIN_X + BOARD_PIXEL_SIZE - 1] = (
        COLOR_FRAME_INSET
    )
    pixels[BOARD_ORIGIN_Y, BOARD_ORIGIN_X : BOARD_ORIGIN_X + BOARD_PIXEL_SIZE] = COLOR_FRAME_INSET
    pixels[BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE - 1, BOARD_ORIGIN_X : BOARD_ORIGIN_X + BOARD_PIXEL_SIZE] = (
        COLOR_FRAME_INSET
    )

    axis_x = BOARD_ORIGIN_X + (CELL_SIZE * 7) - 1
    for y in range(BOARD_ORIGIN_Y, BOARD_ORIGIN_Y + BOARD_PIXEL_SIZE):
        pixels[y, axis_x] = COLOR_AXIS_A if y % 2 == 0 else COLOR_AXIS_B
        pixels[y, axis_x + 1] = COLOR_AXIS_B if y % 2 == 0 else COLOR_AXIS_A
    return pixels


def make_wall_pixels() -> np.ndarray:
    return np.array(
        [
            [COLOR_FRAME, COLOR_FRAME, COLOR_FRAME, COLOR_FRAME],
            [COLOR_FRAME, COLOR_FRAME_INSET, COLOR_FRAME_INSET, COLOR_FRAME],
            [COLOR_FRAME, COLOR_FRAME_INSET, COLOR_FRAME_INSET, COLOR_FRAME],
            [COLOR_FRAME, COLOR_FRAME, COLOR_FRAME, COLOR_FRAME],
        ],
        dtype=np.int16,
    )


def make_beacon_pixels() -> np.ndarray:
    pixels = np.full((8, 8), COLOR_BEACON_RING, dtype=np.int16)
    pixels[1:7, 1:7] = COLOR_BEACON_RING
    pixels[2:6, 2:6] = COLOR_BEACON_CORE
    pixels[1:7, 1:7][1:5, 1:5] = COLOR_BEACON_RING
    pixels[2:6, 2:6] = COLOR_BEACON_CORE
    return pixels


def make_avatar_pixels(main: int, highlight: int) -> np.ndarray:
    return np.array(
        [[-1, highlight, -1, -1], [main, main, main, -1], [main, -1, main, -1], [-1, -1, -1, -1]], dtype=np.int16
    )


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.capacity = 0
        self.remaining = 0

    def set_budget(self, capacity: int, remaining: int) -> None:
        self.capacity = max(0, int(capacity))
        self.remaining = max(0, min(int(remaining), self.capacity))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[:BOARD_ORIGIN_Y, :] = COLOR_FRAME
        if self.capacity <= 0:
            return frame

        total_width = self.capacity
        start_x = max(0, (SCREEN_SIZE - total_width) // 2)
        for idx in range(self.capacity):
            color = COLOR_PIP_REMAINING if idx < self.remaining else COLOR_PIP_SPENT
            x = start_x + idx
            frame[1:7, x] = color
        return frame


class MirrorGreedyFriend(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._board_pixels = make_board_pixels()
        self._wall_pixels = make_wall_pixels()
        self._beacon_pixels = make_beacon_pixels()
        self._player_pixels = make_avatar_pixels(COLOR_PLAYER, COLOR_PLAYER_HI)
        self._helper_pixels = make_avatar_pixels(COLOR_HELPER, COLOR_HELPER_HI)
        self._move_display = MoveBudgetDisplay()
        self._level_specs = LEVEL_SPECS
        self._remaining_moves = 0
        self._move_capacity = 0
        self._walls: frozenset[tuple[int, int]] = frozenset()
        self._player_pos = (0, 0)
        self._helper_pos = (0, 0)
        self._transition_frame_pending = False
        self._player_sprite: Sprite | None = None
        self._helper_sprite: Sprite | None = None

        super().__init__(
            "mirror_greedy_friend",
            [self._build_level(spec, idx) for idx, spec in enumerate(self._level_specs)],
            Camera(0, 0, SCREEN_SIZE, SCREEN_SIZE, COLOR_FLOOR, COLOR_FLOOR, [self._move_display]),
            False,
            len(self._level_specs),
            [1, 2, 3, 4, 5],
            seed,
        )

    def _build_level(self, spec: dict[str, object], index: int) -> Level:
        sprites: list[Sprite] = [
            Sprite(np.array(self._board_pixels, copy=True), name="board", x=0, y=0, layer=0, collidable=False),
            Sprite(np.array(self._beacon_pixels, copy=True), name="beacon", x=28, y=20, layer=2, collidable=False),
            Sprite(np.array(self._player_pixels, copy=True), name="player", layer=3, collidable=False),
            Sprite(np.array(self._helper_pixels, copy=True), name="helper", layer=3, collidable=False),
        ]

        for wall_x, wall_y in sorted(spec["walls"]):
            px, py = logical_to_pixels(wall_x, wall_y)
            sprites.append(
                Sprite(
                    np.array(self._wall_pixels, copy=True),
                    name=f"wall_{wall_x}_{wall_y}",
                    x=px,
                    y=py,
                    layer=1,
                    collidable=False,
                )
            )

        return Level(
            sprites=sprites,
            grid_size=(SCREEN_SIZE, SCREEN_SIZE),
            data={
                "index": index,
                "player_start": tuple(spec["player"]),
                "helper_start": tuple(spec["helper"]),
                "walls": tuple(sorted(spec["walls"])),
                "budget": int(spec["budget"]),
            },
            name=f"Level {index + 1}",
        )

    def on_set_level(self, level: Level) -> None:
        self._player_sprite = level.get_sprites_by_name("player")[0]
        self._helper_sprite = level.get_sprites_by_name("helper")[0]
        self._player_pos = tuple(level.get_data("player_start"))
        self._helper_pos = tuple(level.get_data("helper_start"))
        self._walls = frozenset(tuple(cell) for cell in level.get_data("walls"))
        self._move_capacity = int(level.get_data("budget"))
        self._remaining_moves = self._move_capacity
        self._move_display.set_budget(self._move_capacity, self._remaining_moves)
        self._sync_avatar_sprites()

    def _sync_avatar_sprites(self) -> None:
        if self._player_sprite is not None:
            px, py = logical_to_pixels(*self._player_pos)
            self._player_sprite.set_position(px, py)
        if self._helper_sprite is not None:
            hx, hy = logical_to_pixels(*self._helper_pos)
            self._helper_sprite.set_position(hx, hy)

    def _blocked(self, pos: tuple[int, int], occupied: tuple[int, int]) -> bool:
        x, y = pos
        return not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE) or pos in self._walls or pos == occupied

    def _won(self) -> bool:
        return (
            self._player_pos in BEACON_CELLS
            and self._helper_pos in BEACON_CELLS
            and self._player_pos != self._helper_pos
        )

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

        if self._transition_frame_pending:
            self._transition_frame_pending = False
            self.complete_action()
            return

        raw_action_id = getattr(self.action, "id", 5)
        action_id = int(getattr(raw_action_id, "value", raw_action_id))
        dx, dy = ACTION_TO_DELTA.get(action_id, (0, 0))

        self._remaining_moves = max(0, self._remaining_moves - 1)

        moved = False
        if dx != 0 or dy != 0:
            candidate = (self._player_pos[0] + dx, self._player_pos[1] + dy)
            if not self._blocked(candidate, self._helper_pos):
                self._player_pos = candidate
                moved = True

        if moved:
            helper_candidate = (self._helper_pos[0] - dx, self._helper_pos[1] + dy)
            if not self._blocked(helper_candidate, self._player_pos):
                self._helper_pos = helper_candidate

        self._sync_avatar_sprites()
        self._move_display.set_budget(self._move_capacity, self._remaining_moves)

        if self._won():
            if self.is_last_level():
                self.next_level()
                self.complete_action()
                return
            self.next_level()
            self._transition_frame_pending = True
            return

        if self._remaining_moves <= 0:
            self.lose()

        self.complete_action()
