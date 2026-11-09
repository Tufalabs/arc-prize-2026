from __future__ import annotations

from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, GameState, Level, RenderableUserDisplay, Sprite

GRID_SIZE = 64
PLAYFIELD_LEFT = 4
PLAYFIELD_RIGHT = 59
PLAYFIELD_TOP = 10
PLAYFIELD_BOTTOM = 59

COLOR_BACKGROUND = 0
COLOR_INTERIOR = 1
COLOR_BORDER = 3
COLOR_SPENT = 4
COLOR_FAIL = 8
COLOR_GHOST = 10
COLOR_WARNING = 11
COLOR_AVATAR = 14

MASKS: dict[str, list[tuple[int, int]]] = {
    "A": [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
    "A_mirror": [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)],
    "B": [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)],
    "B_mirror": [
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    ],
}
MASK_DIMS: dict[str, tuple[int, int]] = {"A": (3, 3), "A_mirror": (3, 3), "B": (4, 4), "B_mirror": (4, 4)}

LEVEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Level 1",
        "avatar_mask": "A",
        "avatar_start": (16, 28),
        "targets": [{"mask": "A", "anchor": (22, 30)}],
        "move_budget": 24,
        "optimal_actions": 8,
    },
    {
        "name": "Level 2",
        "avatar_mask": "A",
        "avatar_start": (14, 20),
        "targets": [{"mask": "A", "anchor": (26, 24)}, {"mask": "A_mirror", "anchor": (20, 30)}],
        "move_budget": 48,
        "optimal_actions": 16,
    },
    {
        "name": "Level 3",
        "avatar_mask": "B",
        "avatar_start": (16, 18),
        "targets": [{"mask": "B", "anchor": (28, 24)}, {"mask": "B_mirror", "anchor": (22, 30)}],
        "move_budget": 54,
        "optimal_actions": 18,
    },
]


def _rectangle(width: int, height: int, color: int) -> list[list[int]]:
    return [[int(color) for _ in range(width)] for _ in range(height)]


def _mask_pixels(mask_name: str, color: int) -> list[list[int]]:
    width, height = MASK_DIMS[mask_name]
    pixels = [[-1 for _ in range(width)] for _ in range(height)]
    for x, y in MASKS[mask_name]:
        pixels[y][x] = int(color)
    return pixels


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for spec in LEVEL_SPECS:
        sprites = [
            Sprite(
                _rectangle(PLAYFIELD_RIGHT - PLAYFIELD_LEFT + 1, PLAYFIELD_BOTTOM - PLAYFIELD_TOP + 1, COLOR_INTERIOR),
                name="playfield",
                x=PLAYFIELD_LEFT,
                y=PLAYFIELD_TOP,
                layer=0,
            ),
            Sprite(_rectangle(58, 1, COLOR_BORDER), name="border_top", x=3, y=9, layer=1),
            Sprite(_rectangle(58, 1, COLOR_BORDER), name="border_bottom", x=3, y=60, layer=1),
            Sprite(_rectangle(1, 52, COLOR_BORDER), name="border_left", x=3, y=9, layer=1),
            Sprite(_rectangle(1, 52, COLOR_BORDER), name="border_right", x=60, y=9, layer=1),
        ]
        for index, target in enumerate(spec["targets"]):
            tx, ty = target["anchor"]
            sprites.append(
                Sprite(
                    _mask_pixels(target["mask"], COLOR_GHOST),
                    name=f"target_{index}",
                    x=tx,
                    y=ty,
                    layer=2,
                    collidable=False,
                    visible=True,
                )
            )
        avatar_x, avatar_y = spec["avatar_start"]
        sprites.append(
            Sprite(
                _mask_pixels(spec["avatar_mask"], COLOR_AVATAR),
                name="avatar",
                x=avatar_x,
                y=avatar_y,
                layer=3,
                collidable=False,
                visible=True,
            )
        )
        levels.append(Level(sprites=sprites, grid_size=(GRID_SIZE, GRID_SIZE), data={"spec": spec}, name=spec["name"]))
    return levels


class FootprintHud(RenderableUserDisplay):
    def __init__(self, game: FootprintWalker) -> None:
        self._game = game

    def _pip_color(self, remaining_moves: int) -> int:
        if remaining_moves <= 2:
            return COLOR_FAIL
        if remaining_moves <= 5:
            return COLOR_WARNING
        return COLOR_AVATAR

    def _draw_budget(self, frame: np.ndarray) -> None:
        total = int(self._game._move_capacity)
        remaining = 0 if self._game._state == GameState.GAME_OVER else int(self._game._remaining_moves)
        for pip_index in range(total):
            column = pip_index % 18
            row = pip_index // 18
            x0 = 5 + (column * 3)
            y0 = 1 + (row * 2)
            color = self._pip_color(remaining) if pip_index < remaining else COLOR_SPENT
            frame[y0 : y0 + 2, x0 : x0 + 2] = color

    def _draw_terminal_frame(self, frame: np.ndarray, color: int) -> None:
        frame[:2, :] = color
        frame[-2:, :] = color
        frame[:, :2] = color
        frame[:, -2:] = color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        self._draw_budget(frame)
        if self._game._state == GameState.WIN:
            self._draw_terminal_frame(frame, COLOR_AVATAR)
        elif self._game._state == GameState.GAME_OVER:
            self._draw_terminal_frame(frame, COLOR_FAIL)
        return frame


class FootprintWalker(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._move_capacity = 0
        self._remaining_moves = 0
        self._route_score = 0
        self._avatar_mask_name = "A"
        self._targets: list[dict[str, Any]] = []
        self._avatar: Sprite | None = None
        self._hud = FootprintHud(self)
        super().__init__(
            "footprint_walker",
            _build_levels(),
            Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._hud]),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec") or {}
        self._move_capacity = int(spec.get("move_budget", 0))
        self._remaining_moves = int(spec.get("move_budget", 0))
        self._avatar_mask_name = str(spec.get("avatar_mask", "A"))
        self._targets = list(spec.get("targets", []))
        avatars = level.get_sprites_by_name("avatar")
        self._avatar = avatars[0] if avatars else None

    def _mask_cells(self, mask_name: str) -> list[tuple[int, int]]:
        return MASKS[mask_name]

    def _anchor_fits_playfield(self, anchor_x: int, anchor_y: int) -> bool:
        for local_x, local_y in self._mask_cells(self._avatar_mask_name):
            world_x = anchor_x + local_x
            world_y = anchor_y + local_y
            if not (PLAYFIELD_LEFT <= world_x <= PLAYFIELD_RIGHT and PLAYFIELD_TOP <= world_y <= PLAYFIELD_BOTTOM):
                return False
        return True

    def _is_winning_anchor(self) -> bool:
        if self._avatar is None:
            return False
        for target in self._targets:
            if str(target.get("mask")) != self._avatar_mask_name:
                continue
            if tuple(target.get("anchor", ())) == (int(self._avatar.x), int(self._avatar.y)):
                return True
        return False

    def _movement_delta(self, action_id: Any) -> tuple[int, int]:
        if action_id == GameAction.ACTION1:
            return (0, -1)
        if action_id == GameAction.ACTION2:
            return (0, 1)
        if action_id == GameAction.ACTION3:
            return (-1, 0)
        if action_id == GameAction.ACTION4:
            return (1, 0)
        return (0, 0)

    def step(self) -> None:
        action_id = getattr(self.action, "id", GameAction.RESET)
        if action_id == GameAction.RESET or self._avatar is None:
            self.complete_action()
            return
        if self._state in {GameState.WIN, GameState.GAME_OVER}:
            self.complete_action()
            return

        dx, dy = self._movement_delta(action_id)
        next_x = int(self._avatar.x) + dx
        next_y = int(self._avatar.y) + dy
        if self._anchor_fits_playfield(next_x, next_y):
            self._avatar.set_position(next_x, next_y)

        self._remaining_moves = max(0, int(self._remaining_moves) - 1)

        if self._is_winning_anchor():
            self._route_score += 1
            self.next_level()
        elif self._remaining_moves == 0:
            self.lose()

        self.complete_action()
