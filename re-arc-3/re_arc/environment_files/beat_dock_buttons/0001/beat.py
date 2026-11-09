from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
LOGICAL_ORIGIN = 11
CELL_SIZE = 6

COLOR_BG = 0
COLOR_TILE_INNER = 1
COLOR_DARK = 3
COLOR_BLACK = 5
COLOR_DRUM = 6
COLOR_DRUM_RIM = 7
COLOR_DANGER = 8
COLOR_DOCK_BORDER = 9
COLOR_TILE = 10
COLOR_BEAT = 11
COLOR_COUNTDOWN_WARN = 12
COLOR_REJECT = 13
COLOR_ACCEPT = 14

UP_HITBOX = (28, 35, 1, 8)
RIGHT_HITBOX = (55, 62, 28, 35)
DOWN_HITBOX = (28, 35, 55, 62)
LEFT_HITBOX = (1, 8, 28, 35)

TOKEN_MASK = np.array([[0, 11, 11, 0], [11, 12, 12, 11], [11, 12, 12, 11], [0, 11, 11, 0]], dtype=np.int8)

DOCK_MASK = np.array(
    [[0, 9, 9, 9, 9], [9, 0, 0, 0, 9], [9, 0, 0, 0, 9], [9, 0, 0, 0, 9], [0, 9, 9, 9, 9]], dtype=np.int8
)

DRUM_UP = np.array(
    [
        [0, 0, 7, 7, 7, 7, 0, 0],
        [0, 7, 6, 6, 6, 6, 7, 0],
        [7, 6, 0, 0, 0, 0, 6, 7],
        [7, 6, 6, 0, 0, 6, 6, 7],
        [7, 6, 6, 6, 6, 6, 6, 7],
        [7, 6, 6, 6, 6, 6, 6, 7],
        [0, 7, 6, 6, 6, 6, 7, 0],
        [0, 0, 7, 7, 7, 7, 0, 0],
    ],
    dtype=np.int8,
)

RING_PIPS = {1: ((2, 0),), 2: ((5, 2),), 3: ((2, 5),), 4: ((0, 2),), 0: ((2, 0), (5, 2), (2, 5), (0, 2))}

DIRECTION_DELTAS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}


LEVEL_SPECS = (
    {
        "name": "Straight lesson",
        "start_pos": (1, 3),
        "dock_pos": (4, 3),
        "beat_phase": 3,
        "step_budget": 9,
        "walkable_cells": frozenset({(1, 3), (2, 3), (3, 3)}),
    },
    {
        "name": "Arrival planning",
        "start_pos": (1, 3),
        "dock_pos": (4, 3),
        "beat_phase": 1,
        "step_budget": 15,
        "walkable_cells": frozenset({(1, 3), (2, 3), (3, 3), (1, 2), (2, 2), (3, 2)}),
    },
    {
        "name": "Extra loop sync",
        "start_pos": (1, 3),
        "dock_pos": (4, 3),
        "beat_phase": 4,
        "step_budget": 21,
        "walkable_cells": frozenset({(1, 3), (2, 3), (3, 3), (2, 2), (3, 2)}),
    },
)


def _logical_to_px(cell: tuple[int, int]) -> tuple[int, int]:
    return LOGICAL_ORIGIN + CELL_SIZE * int(cell[0]), LOGICAL_ORIGIN + CELL_SIZE * int(cell[1])


def _rot90(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns).astype(np.int8, copy=False)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


class BeatDockButtons(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._route_score = 0
        self._token_pos = (0, 0)
        self._beat_phase = 0
        self._steps_remaining = 0
        self._step_budget = 0
        self._last_reject = False
        self._board_sprite: Sprite | None = None
        self._current_spec = LEVEL_SPECS[0]

        levels = [self._build_level(index, spec) for index, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG)
        super().__init__(
            game_id="beat_dock_buttons",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[6],
            seed=seed,
        )

    def _build_level(self, index: int, spec: dict[str, object]) -> Level:
        board = Sprite(
            _solid(GRID_SIZE, GRID_SIZE, COLOR_BG), name="board", x=0, y=0, layer=0, tags=["board"], collidable=False
        )
        return Level(
            name=f"Level {index + 1}: {spec['name']}",
            grid_size=(GRID_SIZE, GRID_SIZE),
            sprites=[board],
            data={
                "spec_index": index,
                "start_pos": spec["start_pos"],
                "dock_pos": spec["dock_pos"],
                "beat_phase": spec["beat_phase"],
                "step_budget": spec["step_budget"],
                "walkable_cells": list(spec["walkable_cells"]),
            },
        )

    def on_set_level(self, level: Level) -> None:
        spec_index = int(level.get_data("spec_index") or 0)
        self._current_spec = LEVEL_SPECS[spec_index]
        self._token_pos = tuple(level.get_data("start_pos"))
        self._beat_phase = int(level.get_data("beat_phase"))
        self._steps_remaining = int(level.get_data("step_budget"))
        self._step_budget = int(level.get_data("step_budget"))
        self._last_reject = False

        boards = level.get_sprites_by_name("board")
        self._board_sprite = boards[0] if boards else None
        self._sync_board()

    @property
    def _walkable_cells(self) -> frozenset[tuple[int, int]]:
        return self._current_spec["walkable_cells"]

    def _sync_board(self) -> None:
        if self._board_sprite is None:
            return

        frame = _solid(GRID_SIZE, GRID_SIZE, COLOR_BG)
        self._draw_countdown(frame)
        self._draw_drums(frame)
        self._draw_walkable_tiles(frame)
        self._draw_dock(frame)
        self._draw_beat_ring(frame)
        self._draw_token(frame)
        self._board_sprite.pixels = frame

    def _draw_countdown(self, frame: np.ndarray) -> None:
        x0, y0 = 2, 2
        width, height = 16, 3
        frame[y0 : y0 + height, x0 : x0 + width] = np.int8(COLOR_DARK)
        inner_width = width - 2
        fill = 0
        if self._step_budget > 0:
            fill = round(inner_width * (self._steps_remaining / float(self._step_budget)))
        fill = max(0, min(inner_width, fill))
        if self._steps_remaining <= 3:
            fill_color = COLOR_DANGER
        elif self._steps_remaining * 2 <= self._step_budget:
            fill_color = COLOR_COUNTDOWN_WARN
        else:
            fill_color = COLOR_ACCEPT
        if fill > 0:
            frame[y0 + 1, x0 + 1 : x0 + 1 + fill] = np.int8(fill_color)

    def _draw_drums(self, frame: np.ndarray) -> None:
        drums = {
            UP_HITBOX: DRUM_UP,
            RIGHT_HITBOX: _rot90(DRUM_UP, 1),
            DOWN_HITBOX: _rot90(DRUM_UP, 2),
            LEFT_HITBOX: _rot90(DRUM_UP, 3),
        }
        for (x1, x2, y1, y2), mask in drums.items():
            frame[y1 : y2 + 1, x1 : x2 + 1] = mask

    def _draw_walkable_tiles(self, frame: np.ndarray) -> None:
        for cell in self._walkable_cells:
            px, py = _logical_to_px(cell)
            frame[py + 1 : py + 5, px + 1 : px + 5] = np.int8(COLOR_TILE)
            frame[py + 2 : py + 4, px + 2 : px + 4] = np.int8(COLOR_TILE_INNER)

    def _draw_dock(self, frame: np.ndarray) -> None:
        px, py = _logical_to_px(self._current_spec["dock_pos"])
        dock = DOCK_MASK.copy()
        if self._last_reject:
            core_color = COLOR_REJECT
        elif self._beat_phase == 0:
            core_color = COLOR_ACCEPT
        else:
            core_color = COLOR_TILE
        dock[dock == 0] = np.int8(core_color)
        dock[0, 0] = np.int8(COLOR_BG)
        dock[4, 0] = np.int8(COLOR_BG)
        frame[py : py + 5, px : px + 5] = dock

    def _draw_beat_ring(self, frame: np.ndarray) -> None:
        dock_px, dock_py = _logical_to_px(self._current_spec["dock_pos"])
        ring_x = dock_px + 7
        ring_y = dock_py
        for pip_x, pip_y in ((2, 0), (5, 2), (2, 5), (0, 2)):
            frame[ring_y + pip_y : ring_y + pip_y + 2, ring_x + pip_x : ring_x + pip_x + 2] = np.int8(COLOR_DARK)
        for pip_x, pip_y in RING_PIPS[self._beat_phase]:
            frame[ring_y + pip_y : ring_y + pip_y + 2, ring_x + pip_x : ring_x + pip_x + 2] = np.int8(COLOR_BEAT)

    def _draw_token(self, frame: np.ndarray) -> None:
        px, py = _logical_to_px(self._token_pos)
        frame[py + 1 : py + 5, px + 1 : px + 5] = TOKEN_MASK

    def _drum_direction(self, click_x: int, click_y: int) -> str | None:
        hitboxes = (("up", UP_HITBOX), ("right", RIGHT_HITBOX), ("down", DOWN_HITBOX), ("left", LEFT_HITBOX))
        for name, (x1, x2, y1, y2) in hitboxes:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                return name
        return None

    def _apply_drum_click(self, direction: str) -> None:
        dx, dy = DIRECTION_DELTAS[direction]
        target = (self._token_pos[0] + dx, self._token_pos[1] + dy)
        self._last_reject = False

        if target in self._walkable_cells:
            self._token_pos = target
        elif target == self._current_spec["dock_pos"]:
            if self._beat_phase == 0:
                self._token_pos = target
                self._route_score = self.level_index + 1
                self.next_level()
            else:
                self._last_reject = True

        self._steps_remaining -= 1
        self._beat_phase = (self._beat_phase + 1) % 5

        if self._token_pos != self._current_spec["dock_pos"] and self._steps_remaining <= 0:
            self.lose()

    def step(self) -> None:
        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data if isinstance(self.action.data, dict) else {}
            click_x = int(payload.get("x", -1))
            click_y = int(payload.get("y", -1))
            direction = self._drum_direction(click_x, click_y)
            if direction is not None:
                self._apply_drum_click(direction)

        self._sync_board()
        self.complete_action()


class Beat(BeatDockButtons):
    pass
