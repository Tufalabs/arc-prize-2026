from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

BOARD_W = 16
BOARD_H = 15
CELL_SIZE = 4
HUD_HEIGHT = 4
SCREEN_W = 64
SCREEN_H = 64
ACTIONS_PER_PIP = 3

COLOR_WHITE = 0
COLOR_DARK_GRAY = 3
COLOR_MAGENTA = 6
COLOR_PINK = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

TRANSPARENT = -1

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

PLAYER_PIXELS = np.array([[0, 12, 12, 0], [12, 11, 11, 12], [12, 12, 12, 12], [0, 12, 12, 0]], dtype=np.int8)
HELPER_PIXELS = np.array([[0, 14, 14, 0], [14, 10, 10, 14], [14, 14, 14, 14], [0, 14, 14, 0]], dtype=np.int8)
BEACON_PIXELS = np.array([[0, 11, 11, 0], [11, 15, 15, 11], [11, 15, 15, 11], [0, 11, 11, 0]], dtype=np.int8)
FAIL_BEACON_PIXELS = np.array([[0, 8, 8, 0], [8, 15, 15, 8], [8, 15, 15, 8], [0, 8, 8, 0]], dtype=np.int8)
WIN_BEACON_PIXELS = np.array([[11, 12, 14, 11], [12, 15, 15, 14], [12, 15, 15, 14], [11, 12, 14, 11]], dtype=np.int8)
PORTAL_A_PIXELS = np.array(
    [
        [0, 9, 9, 0, 0, 9, 9, 0],
        [9, 10, 10, 9, 9, 10, 10, 9],
        [9, 10, 9, 10, 10, 9, 10, 9],
        [0, 9, 10, 9, 9, 10, 9, 0],
        [0, 9, 10, 9, 9, 10, 9, 0],
        [9, 10, 9, 10, 10, 9, 10, 9],
        [9, 10, 10, 9, 9, 10, 10, 9],
        [0, 9, 9, 0, 0, 9, 9, 0],
    ],
    dtype=np.int8,
)
PORTAL_B_PIXELS = np.array(
    [
        [0, 6, 6, 0, 0, 6, 6, 0],
        [6, 7, 7, 6, 6, 7, 7, 6],
        [6, 7, 6, 7, 7, 6, 7, 6],
        [0, 6, 7, 6, 6, 7, 6, 0],
        [0, 6, 7, 6, 6, 7, 6, 0],
        [6, 7, 6, 7, 7, 6, 7, 6],
        [6, 7, 7, 6, 6, 7, 7, 6],
        [0, 6, 6, 0, 0, 6, 6, 0],
    ],
    dtype=np.int8,
)
OVERLAP_PIXELS = np.array([[12, 12, 14, 14], [12, 11, 10, 14], [12, 12, 14, 14], [0, 12, 14, 0]], dtype=np.int8)
TRANSPARENT_4 = np.full((4, 4), TRANSPARENT, dtype=np.int8)


def cell_to_pixels(cell: tuple[int, int]) -> tuple[int, int]:
    return (cell[0] * CELL_SIZE, HUD_HEIGHT + cell[1] * CELL_SIZE)


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.capacity_pips = 0
        self.remaining_actions = 0

    def configure(self, capacity_pips: int) -> None:
        self.capacity_pips = int(capacity_pips)
        self.remaining_actions = int(capacity_pips) * ACTIONS_PER_PIP

    def set_remaining_actions(self, remaining_actions: int) -> None:
        self.remaining_actions = max(0, int(remaining_actions))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[0:HUD_HEIGHT, :] = COLOR_WHITE
        x = 2
        filled = min(self.capacity_pips, (self.remaining_actions + ACTIONS_PER_PIP - 1) // ACTIONS_PER_PIP)
        for idx in range(self.capacity_pips):
            color = COLOR_YELLOW if idx < filled else COLOR_DARK_GRAY
            frame[:, x : x + 3][0:HUD_HEIGHT, :] = color
            x += 4
        return frame


def make_sprite(pixels: np.ndarray, name: str, cell: tuple[int, int], *, layer: int) -> Sprite:
    x, y = cell_to_pixels(cell)
    return Sprite(pixels=pixels, name=name, x=x, y=y, layer=layer)


def build_level(
    *,
    name: str,
    player_start: tuple[int, int],
    helper_start: tuple[int, int],
    beacon: tuple[int, int],
    portal_specs: list[dict[str, object]],
    budget_pips: int,
    solution: list[int],
) -> Level:
    sprites = [
        make_sprite(BEACON_PIXELS, "beacon", beacon, layer=1),
        make_sprite(PLAYER_PIXELS, "player", player_start, layer=3),
        make_sprite(HELPER_PIXELS, "helper", helper_start, layer=3),
    ]
    for spec in portal_specs:
        top_left = tuple(spec["top_left"])  # type: ignore[arg-type]
        pixels = PORTAL_A_PIXELS if str(spec["pair"]) == "A" else PORTAL_B_PIXELS
        sprites.append(make_sprite(pixels, str(spec["name"]), top_left, layer=0))
    return Level(
        sprites=sprites,
        grid_size=(SCREEN_W, SCREEN_H),
        data={
            "player_start": player_start,
            "helper_start": helper_start,
            "beacon": beacon,
            "portal_specs": portal_specs,
            "budget_pips": int(budget_pips),
            "solution": list(solution),
        },
        name=name,
    )


levels = [
    build_level(
        name="Level 1",
        player_start=(1, 7),
        helper_start=(13, 1),
        beacon=(11, 7),
        portal_specs=[
            {"pair": "A", "name": "portal_a1", "top_left": (3, 6)},
            {"pair": "A", "name": "portal_a2", "top_left": (7, 6)},
        ],
        budget_pips=10,
        solution=[4, 4, 4, 4, 4, 4, 1, 2],
    ),
    build_level(
        name="Level 2",
        player_start=(1, 7),
        helper_start=(13, 1),
        beacon=(11, 7),
        portal_specs=[
            {"pair": "A", "name": "portal_a1", "top_left": (3, 6)},
            {"pair": "A", "name": "portal_a2", "top_left": (7, 6)},
            {"pair": "B", "name": "portal_b1", "top_left": (3, 9)},
            {"pair": "B", "name": "portal_b2", "top_left": (12, 3)},
        ],
        budget_pips=10,
        solution=[4, 4, 4, 4, 4, 4, 1, 2],
    ),
    build_level(
        name="Level 3",
        player_start=(0, 7),
        helper_start=(15, 0),
        beacon=(11, 7),
        portal_specs=[
            {"pair": "A", "name": "portal_a1", "top_left": (5, 6)},
            {"pair": "A", "name": "portal_a2", "top_left": (8, 6)},
        ],
        budget_pips=12,
        solution=[3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2],
    ),
]


class RelayPortals(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._budget_display = MoveBudgetDisplay()
        self._relay_score = 0
        self._remaining_actions = 0
        self._beacon_mode = "normal"
        self._portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._player: Sprite | None = None
        self._helper: Sprite | None = None
        self._beacon_sprite: Sprite | None = None
        self._beacon_cell = (0, 0)
        super().__init__(
            game_id="relay_portals-0001",
            levels=levels,
            camera=Camera(0, 0, SCREEN_W, SCREEN_H, COLOR_WHITE, COLOR_WHITE, [self._budget_display]),
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._player = level.get_sprites_by_name("player")[0]
        self._helper = level.get_sprites_by_name("helper")[0]
        self._beacon_sprite = level.get_sprites_by_name("beacon")[0]
        self._beacon_cell = tuple(level.get_data("beacon"))
        self._remaining_actions = int(level.get_data("budget_pips")) * ACTIONS_PER_PIP
        self._budget_display.configure(int(level.get_data("budget_pips")))
        self._budget_display.set_remaining_actions(self._remaining_actions)
        self._beacon_mode = "normal"
        self._portal_pairs = self._build_portal_pairs(level)
        self._sync_entity_sprites()

    def _build_portal_pairs(self, level: Level) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        grouped: dict[str, list[tuple[int, int]]] = {}
        for spec in level.get_data("portal_specs"):
            grouped.setdefault(str(spec["pair"]), []).append(tuple(spec["top_left"]))  # type: ignore[arg-type]
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for _pair_id, pads in grouped.items():
            first, second = sorted(pads)
            pairs.append((first, second))
        return pairs

    def _action_id(self) -> int:
        return int(getattr(self._action.id, "value", self._action.id))

    def _get_cell(self, sprite: Sprite) -> tuple[int, int]:
        return (int(sprite.x) // CELL_SIZE, (int(sprite.y) - HUD_HEIGHT) // CELL_SIZE)

    def _set_cell(self, sprite: Sprite, cell: tuple[int, int]) -> None:
        x, y = cell_to_pixels(cell)
        sprite.set_position(x, y)

    def _inside_pad(self, cell: tuple[int, int], top_left: tuple[int, int]) -> bool:
        return top_left[0] <= cell[0] <= top_left[0] + 1 and top_left[1] <= cell[1] <= top_left[1] + 1

    def _portal_destination(self, prev_cell: tuple[int, int], current_cell: tuple[int, int]) -> tuple[int, int]:
        for first, second in self._portal_pairs:
            if self._inside_pad(current_cell, first) and not self._inside_pad(prev_cell, first):
                ox = current_cell[0] - first[0]
                oy = current_cell[1] - first[1]
                return (second[0] + ox, second[1] + oy)
            if self._inside_pad(current_cell, second) and not self._inside_pad(prev_cell, second):
                ox = current_cell[0] - second[0]
                oy = current_cell[1] - second[1]
                return (first[0] + ox, first[1] + oy)
        return current_cell

    def _move_player(self, start: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
        target = (start[0] + delta[0], start[1] + delta[1])
        if 0 <= target[0] < BOARD_W and 0 <= target[1] < BOARD_H:
            return target
        return start

    def _move_helper(self, start: tuple[int, int]) -> tuple[int, int]:
        if start[0] != self._beacon_cell[0]:
            step_x = 1 if start[0] < self._beacon_cell[0] else -1
            return (start[0] + step_x, start[1])
        if start[1] != self._beacon_cell[1]:
            step_y = 1 if start[1] < self._beacon_cell[1] else -1
            return (start[0], start[1] + step_y)
        return start

    def _sync_entity_sprites(self) -> None:
        assert self._player is not None
        assert self._helper is not None
        assert self._beacon_sprite is not None
        self._budget_display.set_remaining_actions(self._remaining_actions)

        if self._beacon_mode == "fail":
            self._beacon_sprite.pixels = FAIL_BEACON_PIXELS.copy()
        elif self._beacon_mode == "win":
            self._beacon_sprite.pixels = WIN_BEACON_PIXELS.copy()
        else:
            self._beacon_sprite.pixels = BEACON_PIXELS.copy()

        player_cell = self._get_cell(self._player)
        helper_cell = self._get_cell(self._helper)
        if player_cell == helper_cell:
            self._player.pixels = (
                WIN_BEACON_PIXELS.copy() if player_cell == self._beacon_cell else OVERLAP_PIXELS.copy()
            )
            self._helper.pixels = TRANSPARENT_4.copy()
            self._set_cell(self._player, player_cell)
            self._set_cell(self._helper, helper_cell)
            return

        self._player.pixels = PLAYER_PIXELS.copy()
        self._helper.pixels = HELPER_PIXELS.copy()
        self._set_cell(self._player, player_cell)
        self._set_cell(self._helper, helper_cell)

    def step(self) -> None:
        assert self._player is not None
        assert self._helper is not None

        if self._action_id() == 0:
            self._sync_entity_sprites()
            self.complete_action()
            return

        player_prev = self._get_cell(self._player)
        helper_prev = self._get_cell(self._helper)

        delta = MOVE_DELTAS.get(self._action_id(), (0, 0))
        player_now = self._move_player(player_prev, delta)
        player_now = self._portal_destination(player_prev, player_now)
        helper_now = self._move_helper(helper_prev)

        self._set_cell(self._player, player_now)
        self._set_cell(self._helper, helper_now)

        self._remaining_actions = max(0, self._remaining_actions - 1)
        player_entered_beacon = player_prev != self._beacon_cell and player_now == self._beacon_cell
        helper_entered_beacon = helper_prev != self._beacon_cell and helper_now == self._beacon_cell

        win_now = (
            player_now == self._beacon_cell
            and helper_now == self._beacon_cell
            and player_entered_beacon
            and helper_entered_beacon
        )
        if win_now:
            self._relay_score += 1
            self._beacon_mode = "win"
            self._sync_entity_sprites()
            self.next_level()
            self.complete_action()
            return

        if helper_entered_beacon or self._remaining_actions == 0:
            self._beacon_mode = "fail"
            self._sync_entity_sprites()
            self.lose()
            self.complete_action()
            return

        self._beacon_mode = "normal"
        self._sync_entity_sprites()
        self.complete_action()
