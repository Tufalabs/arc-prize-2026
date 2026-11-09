from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_CLICK = 6

GRID_SIZE = 32
VIEW_SIZE = 64
TILE_SIZE = 3
TILE_SPACING = 4

COLOR_BACKGROUND = 4
COLOR_TIMER_FULL = 12
COLOR_TIMER_EMPTY = 11
COLOR_STENCIL = 6

NEIGHBOR_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
CENTER_CLICK_PATTERN = ((0, 0, 0), (0, 1, 0), (0, 0, 0))


TargetSpec = tuple[int, int, list[list[int]]]
StencilSpec = tuple[int, int, list[list[int]]]


@dataclass(frozen=True)
class ExampleSpriteSpec:
    name: str
    x: int
    y: int
    pixels: tuple[tuple[int, ...], ...]
    layer: int = -1


@dataclass(frozen=True)
class LevelSpec:
    name: str
    palette: tuple[int, ...]
    step_budget: int
    normal_tiles: tuple[tuple[int, int], ...]
    stencil_tiles: tuple[StencilSpec, ...]
    targets: tuple[TargetSpec, ...]
    examples: tuple[ExampleSpriteSpec, ...] = ()


FIRST_LEVEL_EXAMPLES = (
    ExampleSpriteSpec(
        name="top_example_panel",
        x=0,
        y=0,
        pixels=tuple(tuple(5 for _ in range(GRID_SIZE)) for _ in range(16)),
        layer=-8,
    ),
    ExampleSpriteSpec(
        name="bottom_example_panel", x=0, y=14, pixels=tuple(tuple(5 for _ in range(16)) for _ in range(18)), layer=-8
    ),
    ExampleSpriteSpec(
        name="same_color_example",
        x=19,
        y=1,
        pixels=(
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
        ),
    ),
    ExampleSpriteSpec(
        name="different_color_example",
        x=2,
        y=1,
        pixels=(
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9),
            (8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9),
            (8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8),
            (9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8),
            (9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8),
        ),
    ),
    ExampleSpriteSpec(
        name="bottom_left_example",
        x=2,
        y=18,
        pixels=(
            (8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9),
            (8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9),
            (8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8),
            (9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8),
            (9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8),
        ),
    ),
    ExampleSpriteSpec(name="top_left_rule", x=6, y=5, pixels=((2, 0, 2), (0, 8, 2), (2, 0, 0))),
    ExampleSpriteSpec(name="top_right_rule", x=23, y=5, pixels=((2, 0, 2), (0, 8, 0), (2, 0, 2))),
    ExampleSpriteSpec(name="bottom_left_rule", x=6, y=22, pixels=((0, 2, 2), (0, 8, 0), (2, 2, 0))),
    ExampleSpriteSpec(
        name="play_area_hint",
        x=16,
        y=16,
        pixels=(
            (2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2),
            (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2),
            (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
            (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2),
            (2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2),
            (2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2),
        ),
        layer=5,
    ),
)


LEVEL_SPECS = (
    LevelSpec(
        name="THR",
        palette=(9, 8),
        step_budget=32,
        normal_tiles=((18, 18), (22, 18), (26, 18), (18, 22), (26, 22), (18, 26), (22, 26), (26, 26)),
        stencil_tiles=(),
        targets=((22, 22, [[0, 2, 2], [0, 8, 0], [0, 2, 2]]),),
        examples=FIRST_LEVEL_EXAMPLES,
    ),
    LevelSpec(
        name="hxv",
        palette=(9, 12),
        step_budget=32,
        normal_tiles=(
            (10, 7),
            (14, 7),
            (18, 7),
            (10, 11),
            (18, 11),
            (10, 15),
            (14, 15),
            (18, 15),
            (18, 19),
            (14, 23),
            (10, 23),
            (10, 19),
            (18, 23),
        ),
        stencil_tiles=(),
        targets=((14, 19, [[0, 2, 0], [2, 12, 2], [0, 0, 2]]), (14, 11, [[0, 2, 2], [0, 12, 0], [0, 2, 0]])),
    ),
    LevelSpec(
        name="Fmh",
        palette=(8, 12),
        step_budget=96,
        normal_tiles=(
            (10, 2),
            (10, 6),
            (10, 10),
            (14, 10),
            (14, 14),
            (18, 10),
            (18, 6),
            (22, 10),
            (18, 2),
            (14, 2),
            (6, 14),
            (6, 10),
            (10, 18),
            (6, 18),
            (14, 18),
            (18, 18),
            (22, 18),
            (22, 14),
            (10, 22),
            (18, 22),
            (14, 26),
            (10, 26),
            (18, 26),
        ),
        stencil_tiles=(),
        targets=(
            (18, 14, [[2, 0, 0], [0, 8, 2], [2, 0, 2]]),
            (14, 22, [[2, 0, 2], [0, 12, 2], [0, 0, 0]]),
            (14, 6, [[0, 0, 0], [0, 12, 2], [2, 0, 2]]),
            (10, 14, [[2, 0, 2], [2, 8, 0], [0, 0, 2]]),
        ),
    ),
    LevelSpec(
        name="oea",
        palette=(9, 8, 12),
        step_budget=96,
        normal_tiles=(
            (10, 7),
            (14, 11),
            (14, 15),
            (22, 11),
            (22, 15),
            (18, 7),
            (14, 7),
            (6, 11),
            (10, 15),
            (18, 15),
            (6, 15),
            (6, 7),
            (22, 7),
            (10, 19),
            (18, 19),
            (10, 23),
            (14, 23),
            (18, 23),
        ),
        stencil_tiles=(),
        targets=(
            (18, 11, [[2, 0, 2], [2, 9, 2], [2, 2, 0]]),
            (10, 11, [[2, 0, 2], [2, 12, 2], [2, 0, 2]]),
            (14, 19, [[0, 2, 2], [2, 12, 2], [0, 0, 0]]),
        ),
    ),
    LevelSpec(
        name="INW",
        palette=(14, 15),
        step_budget=128,
        normal_tiles=(
            (7, 6),
            (15, 18),
            (15, 6),
            (11, 10),
            (11, 2),
            (23, 14),
            (15, 10),
            (19, 18),
            (23, 22),
            (15, 22),
            (19, 26),
            (15, 26),
            (23, 18),
            (19, 10),
            (7, 10),
            (15, 2),
            (7, 18),
            (7, 22),
            (7, 26),
            (11, 26),
            (7, 14),
            (3, 10),
            (3, 18),
            (15, 14),
            (23, 10),
            (27, 18),
            (27, 10),
        ),
        stencil_tiles=(
            (11, 6, [[7, 6, 7], [6, 10, 6], [7, 6, 7]]),
            (19, 22, [[7, 6, 7], [6, 10, 6], [7, 6, 7]]),
            (11, 14, [[7, 6, 7], [6, 10, 6], [7, 6, 7]]),
        ),
        targets=(
            (19, 6, [[0, 3, 3], [2, 15, 3], [0, 2, 0]]),
            (11, 18, [[0, 2, 0], [2, 14, 2], [0, 3, 0]]),
            (3, 14, [[3, 2, 0], [3, 15, 2], [3, 2, 0]]),
            (7, 2, [[3, 3, 3], [3, 14, 0], [3, 0, 2]]),
            (23, 26, [[2, 0, 3], [0, 14, 3], [3, 3, 3]]),
            (11, 22, [[2, 3, 2], [0, 14, 0], [2, 0, 2]]),
            (27, 14, [[2, 0, 3], [0, 14, 3], [2, 0, 3]]),
            (19, 14, [[2, 0, 2], [0, 14, 0], [2, 0, 2]]),
        ),
    ),
    LevelSpec(
        name="DFx",
        palette=(11, 14),
        step_budget=128,
        normal_tiles=(),
        stencil_tiles=(
            (6, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (10, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (14, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (18, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (22, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (6, 11, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (22, 11, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (10, 11, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (14, 11, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (18, 15, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (6, 15, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (22, 15, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (14, 15, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (18, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (6, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (22, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (10, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (14, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (26, 19, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (2, 7, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (2, 3, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
            (26, 23, [[7, 6, 7], [7, 10, 7], [7, 7, 7]]),
        ),
        targets=(
            (22, 23, [[2, 0, 0], [3, 14, 2], [3, 3, 3]]),
            (10, 15, [[2, 0, 0], [0, 14, 0], [2, 0, 2]]),
            (18, 11, [[2, 0, 2], [0, 14, 0], [0, 0, 2]]),
            (6, 3, [[3, 3, 3], [2, 14, 3], [0, 0, 2]]),
        ),
    ),
)


class TimerBar(RenderableUserDisplay):
    def __init__(self) -> None:
        self.step_budget = 1
        self.steps_left = 1

    def reset(self, step_budget: int) -> None:
        self.step_budget = max(1, int(step_budget))
        self.steps_left = self.step_budget

    def tick(self) -> bool:
        self.steps_left = max(0, self.steps_left - 1)
        return self.steps_left > 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        filled = round(VIEW_SIZE * self.steps_left / self.step_budget)
        for x in range(VIEW_SIZE):
            frame[VIEW_SIZE - 1, x] = COLOR_TIMER_FULL if x < filled else COLOR_TIMER_EMPTY
        return frame


def _tile_pixels(color: int) -> np.ndarray:
    return np.full((TILE_SIZE, TILE_SIZE), color, dtype=np.int8)


def _stencil_tile_pixels(mask: list[list[int]], color: int) -> np.ndarray:
    pixels = np.full((TILE_SIZE, TILE_SIZE), color, dtype=np.int8)
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value == COLOR_STENCIL:
                pixels[y, x] = COLOR_STENCIL
    return pixels


def _target_pixels(mask: list[list[int]]) -> np.ndarray:
    return np.asarray(mask, dtype=np.int8)


def _background_sprite() -> Sprite:
    return Sprite(
        np.full((GRID_SIZE, GRID_SIZE), COLOR_BACKGROUND, dtype=np.int8), name="background", layer=-10, collidable=False
    )


def _normal_tile_sprite(x: int, y: int, color: int) -> Sprite:
    return Sprite(_tile_pixels(color), name=f"normal_tile_{x}_{y}", x=x, y=y, tags=["Hkx", "gOi"])


def _stencil_tile_sprite(x: int, y: int, mask: list[list[int]], color: int) -> Sprite:
    return Sprite(_stencil_tile_pixels(mask, color), name=f"stencil_tile_{x}_{y}", x=x, y=y, tags=["NTi", "gOi"])


def _target_sprite(x: int, y: int, mask: list[list[int]]) -> Sprite:
    return Sprite(_target_pixels(mask), name=f"target_{x}_{y}", x=x, y=y, collidable=False, tags=["bsT"])


def _example_sprite(spec: ExampleSpriteSpec) -> Sprite:
    return Sprite(
        np.asarray(spec.pixels, dtype=np.int8), name=spec.name, x=spec.x, y=spec.y, layer=spec.layer, collidable=False
    )


def _build_level(spec: LevelSpec) -> Level:
    sprites: list[Sprite] = [_background_sprite()]
    initial_color = spec.palette[0]
    for example in spec.examples:
        sprites.append(_example_sprite(example))
    for x, y in spec.normal_tiles:
        sprites.append(_normal_tile_sprite(x, y, initial_color))
    for x, y, mask in spec.stencil_tiles:
        sprites.append(_stencil_tile_sprite(x, y, mask, initial_color))
    for x, y, mask in spec.targets:
        sprites.append(_target_sprite(x, y, mask))

    return Level(
        grid_size=(GRID_SIZE, GRID_SIZE),
        sprites=sprites,
        name=spec.name,
        data={"palette": spec.palette, "step_budget": spec.step_budget, "click_pattern": CENTER_CLICK_PATTERN},
    )


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class Ft09Close(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.timer = TimerBar()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "ft09_close",
            levels,
            Camera(0, 0, 16, 16, COLOR_BACKGROUND, COLOR_BACKGROUND, [self.timer]),
            False,
            len(levels),
            [ACTION_CLICK],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.tiles = level.get_sprites_by_tag("Hkx") + level.get_sprites_by_tag("NTi")
        self.targets = level.get_sprites_by_tag("bsT")
        self.palette = list(level.get_data("palette") or (9, 8))
        self.default_click_pattern = level.get_data("click_pattern") or CENTER_CLICK_PATTERN
        self.timer.reset(int(level.get_data("step_budget") or 1))

        for tile in self.tiles:
            self._set_tile_color(tile, self.palette[0])

    @property
    def gqb(self) -> list[int]:
        return self.palette

    def step(self) -> None:
        if _action_id(self.action.id) != ACTION_CLICK:
            self.complete_action()
            return

        clicked = self._clicked_tile()
        if clicked is None:
            self.complete_action()
            return

        for tile in self._affected_tiles(clicked):
            self._advance_tile_color(tile)

        if self._is_level_solved():
            self.next_level()
            self.complete_action()
            return

        if not self.timer.tick():
            self.lose()
        self.complete_action()

    def _clicked_tile(self) -> Sprite | None:
        display_x = int(self.action.data.get("x", 0))
        display_y = int(self.action.data.get("y", 0))
        grid_position = self.camera.display_to_grid(display_x, display_y)
        if grid_position is None:
            return None
        grid_x, grid_y = grid_position
        return self._tile_at(grid_x, grid_y)

    def _tile_at(self, x: int, y: int) -> Sprite | None:
        tile = self.current_level.get_sprite_at(x, y, "Hkx")
        if tile is None:
            tile = self.current_level.get_sprite_at(x, y, "NTi")
        return tile

    def _affected_tiles(self, clicked: Sprite) -> list[Sprite]:
        pattern = self._click_pattern_for(clicked)
        out: list[Sprite] = []
        for row, col in np.ndindex((TILE_SIZE, TILE_SIZE)):
            if pattern[row][col] != 1:
                continue
            dx, dy = NEIGHBOR_OFFSETS[row * TILE_SIZE + col]
            tile = self._tile_at(int(clicked.x) + dx * TILE_SPACING, int(clicked.y) + dy * TILE_SPACING)
            if tile is not None:
                out.append(tile)
        return out

    def _click_pattern_for(self, tile: Sprite) -> tuple[tuple[int, int, int], ...]:
        if "NTi" not in tile.tags:
            return self.default_click_pattern
        pattern = [[1 if int(value) == COLOR_STENCIL else 0 for value in row] for row in tile.pixels]
        pattern[1][1] = 1
        return tuple(tuple(row) for row in pattern)

    def _advance_tile_color(self, tile: Sprite) -> None:
        current_color = int(tile.pixels[1][1])
        color_index = self.palette.index(current_color)
        next_color = self.palette[(color_index + 1) % len(self.palette)]
        self._set_tile_color(tile, next_color)

    def _set_tile_color(self, tile: Sprite, color: int) -> None:
        pixels = tile.pixels
        if "NTi" in tile.tags:
            pixels[pixels != COLOR_STENCIL] = color
        else:
            pixels[:, :] = color

    def _is_level_solved(self) -> bool:
        for target in self.targets:
            target_color = int(target.pixels[1][1])
            for row, col in np.ndindex((TILE_SIZE, TILE_SIZE)):
                dx, dy = NEIGHBOR_OFFSETS[row * TILE_SIZE + col]
                neighbor = self._tile_at(int(target.x) + dx * TILE_SPACING, int(target.y) + dy * TILE_SPACING)
                if neighbor is None:
                    continue
                same_color = int(neighbor.pixels[1][1]) == target_color
                requires_same_color = int(target.pixels[row][col]) == 0
                if same_color != requires_same_color:
                    return False
        return True

    def _get_hidden_state(self) -> np.ndarray:
        return np.asarray([self.timer.steps_left], dtype=np.int16)

    def _get_valid_actions(self) -> list[ActionInput]:
        scale, x_offset, y_offset = self.camera._calculate_scale_and_offset()
        actions: list[ActionInput] = []
        for tile in sorted(self.tiles, key=lambda sprite: (int(sprite.y), int(sprite.x), sprite.name)):
            if not self._is_sprite_clickable_now(tile):
                continue
            actions.append(
                ActionInput(
                    id=GameAction.ACTION6,
                    data={"x": int(tile.x) * scale + x_offset, "y": int(tile.y) * scale + y_offset},
                )
            )
        return actions
