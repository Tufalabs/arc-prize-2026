from __future__ import annotations

import random

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
LEVEL_COUNT = 10
BACKGROUND_COLOR = 0
SUCCESS_COLOR = 14
TARGET_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _seeded_target_positions(width: int, height: int, count: int, seed: int) -> list[tuple[int, int]]:
    cells = [(x, y) for y in range(height) for x in range(width)]
    if count > len(cells):
        raise ValueError("Requested targets exceed available grid cells.")
    rng = random.Random(seed)
    rng.shuffle(cells)
    return cells[:count]


def _build_level(level_idx: int, target_position: tuple[int, int], target_color: int) -> Level:
    x, y = int(target_position[0]), int(target_position[1])
    floor_pixels = [[BACKGROUND_COLOR] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    return Level(
        name=f"Level {level_idx + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[
            Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
            Sprite(pixels=[[int(target_color)]], name="target", x=x, y=y, collidable=False, layer=2, tags=["target"]),
        ],
        data={"target": (x, y)},
    )


class EasyTaps(ARCBaseGame):
    def __init__(self, seed: int = 0):
        positions = _seeded_target_positions(GRID_WIDTH, GRID_HEIGHT, LEVEL_COUNT, int(seed))
        levels = [
            _build_level(level_idx, positions[level_idx], TARGET_COLORS[level_idx]) for level_idx in range(LEVEL_COUNT)
        ]
        camera = Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, BACKGROUND_COLOR, 5, [])
        super().__init__(
            game_id="easy_taps-0001",
            levels=levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[6],
            seed=seed,
        )

        self._target = (0, 0)
        self._target_sprite: Sprite | None = None
        self._pending_level_advance = False

    def on_set_level(self, level: Level) -> None:
        target = level.get_data("target") or (0, 0)
        self._target = (int(target[0]), int(target[1]))
        targets = self.current_level.get_sprites_by_name("target")
        self._target_sprite = targets[0] if targets else None
        self._pending_level_advance = False

    def _try_click_target(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return
        if (int(grid_pos[0]), int(grid_pos[1])) == self._target:
            if self._target_sprite is not None:
                self._target_sprite.pixels[0][0] = SUCCESS_COLOR
            self._pending_level_advance = True

    def step(self) -> None:
        if self._pending_level_advance:
            self._pending_level_advance = False
            self.next_level()
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION6:
            self._try_click_target()
            if self._pending_level_advance:
                return
        self.complete_action()
