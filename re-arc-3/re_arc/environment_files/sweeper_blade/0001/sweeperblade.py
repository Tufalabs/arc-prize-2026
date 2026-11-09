from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

LOGICAL_GRID_WIDTH = 16
LOGICAL_GRID_HEIGHT = 16
PLAYFIELD_Y_OFFSET = 2
PLAYFIELD_WIDTH = 16
PLAYFIELD_HEIGHT = 14
CELL_SIZE = 4
HUD_ROWS = 2
DISPLAY_SIZE = 64
MAX_BUDGET_PIPS = 32

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6

DIR_TO_DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

DIR_TO_NAME = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}

YELLOW = "yellow"
MAGENTA = "magenta"

COLOR_FLOOR = 0
COLOR_HUD_BG = 1
COLOR_SPENT = 2
COLOR_WALL = 4
COLOR_BIN_CAVITY = 5
COLOR_MAGENTA_PELLET = 6
COLOR_BLADE = 9
COLOR_BLADE_EDGE = 10
COLOR_YELLOW_PELLET = 11
COLOR_YELLOW_BIN = 12
COLOR_MAGENTA_BIN = 15

FLOOR_SPRITE = "floor"
HUD_SPRITE = "hud"
WALL_SPRITE = "wall"
BIN_SPRITE = "bin"
PELLET_SPRITE = "pellet"
BLADE_SPRITE = "blade"


@dataclass(frozen=True)
class LevelSpec:
    budget: int
    blade_start: tuple[int, int]
    yellow_pellets: frozenset[tuple[int, int]]
    magenta_pellets: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]
    bins: tuple[tuple[str, frozenset[tuple[int, int]]], ...]


@dataclass(frozen=True)
class SweepState:
    blade: tuple[int, int]
    yellow: frozenset[tuple[int, int]]
    magenta: frozenset[tuple[int, int]]
    budget: int


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=20,
        blade_start=(1, 7),
        yellow_pellets=frozenset({(4, 7), (5, 7), (4, 8), (5, 8)}),
        magenta_pellets=frozenset(),
        walls=frozenset(),
        bins=((YELLOW, frozenset({(9, 7), (10, 7), (9, 8), (10, 8)})),),
    ),
    LevelSpec(
        budget=26,
        blade_start=(1, 6),
        yellow_pellets=frozenset({(4, 6), (5, 6), (4, 7), (5, 7), (10, 6)}),
        magenta_pellets=frozenset(),
        walls=frozenset({(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12)}),
        bins=((YELLOW, frozenset({(11, 6), (12, 6), (11, 7), (12, 7)})),),
    ),
    LevelSpec(
        budget=32,
        blade_start=(9, 6),
        yellow_pellets=frozenset({(9, 5), (10, 5), (10, 4)}),
        magenta_pellets=frozenset({(9, 9), (10, 9), (10, 10)}),
        walls=frozenset({(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12)}),
        bins=(
            (YELLOW, frozenset({(9, 1), (10, 1), (9, 2), (10, 2)})),
            (MAGENTA, frozenset({(9, 12), (10, 12), (9, 13), (10, 13)})),
        ),
    ),
)


def logical_to_pixel(cell: tuple[int, int], *, playfield: bool) -> tuple[int, int]:
    x, y = cell
    return x * CELL_SIZE, (y + PLAYFIELD_Y_OFFSET) * CELL_SIZE if playfield else y * CELL_SIZE


def blade_cells(top_left: tuple[int, int]) -> frozenset[tuple[int, int]]:
    bx, by = top_left
    return frozenset({(bx, by), (bx + 1, by), (bx + 2, by), (bx, by + 1), (bx + 1, by + 1), (bx + 2, by + 1)})


def leading_cells(top_left: tuple[int, int], action_id: int) -> tuple[tuple[int, int], ...]:
    bx, by = top_left
    if action_id == UP:
        return ((bx, by - 1), (bx + 1, by - 1), (bx + 2, by - 1))
    if action_id == DOWN:
        return ((bx, by + 2), (bx + 1, by + 2), (bx + 2, by + 2))
    if action_id == LEFT:
        return ((bx - 1, by), (bx - 1, by + 1))
    if action_id == RIGHT:
        return ((bx + 3, by), (bx + 3, by + 1))
    raise ValueError(f"Unsupported action {action_id}.")


def in_bounds(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < PLAYFIELD_WIDTH and 0 <= y < PLAYFIELD_HEIGHT


def initial_state(level_index: int) -> SweepState:
    spec = LEVEL_SPECS[level_index]
    return SweepState(
        blade=spec.blade_start, yellow=spec.yellow_pellets, magenta=spec.magenta_pellets, budget=spec.budget
    )


def board_key(state: SweepState) -> tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    return state.blade, state.yellow, state.magenta


def bin_lookup(spec: LevelSpec) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    for color, cells in spec.bins:
        for cell in cells:
            lookup[cell] = color
    return lookup


def _component(start: tuple[int, int], pellets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            neighbor = (nx, ny)
            if neighbor in pellets and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def apply_directional_move(spec: LevelSpec, state: SweepState, action_id: int) -> SweepState:
    if action_id not in DIR_TO_DELTA:
        raise ValueError(f"Unsupported move action {action_id}.")

    next_budget = max(0, state.budget - 1)
    dx, dy = DIR_TO_DELTA[action_id]
    new_blade = (state.blade[0] + dx, state.blade[1] + dy)
    blade_destination = blade_cells(new_blade)

    pellets_by_color = {YELLOW: set(state.yellow), MAGENTA: set(state.magenta)}
    all_pellets = pellets_by_color[YELLOW] | pellets_by_color[MAGENTA]
    moving: set[tuple[int, int]] = set()

    for cell in leading_cells(state.blade, action_id):
        if cell in all_pellets:
            moving.update(_component(cell, all_pellets))

    bins = bin_lookup(spec)
    non_moving = all_pellets - moving

    for cell in moving:
        target = (cell[0] + dx, cell[1] + dy)
        if not in_bounds(target):
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)
        if target in spec.walls:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)
        if target in non_moving:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)
        if target in blade_destination:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)
        bin_color = bins.get(target)
        if bin_color is None:
            continue
        if cell in pellets_by_color[YELLOW] and bin_color != YELLOW:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)
        if cell in pellets_by_color[MAGENTA] and bin_color != MAGENTA:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)

    moved_yellow: set[tuple[int, int]] = set()
    for cell in state.yellow:
        if cell not in moving:
            moved_yellow.add(cell)
            continue
        target = (cell[0] + dx, cell[1] + dy)
        if bins.get(target) == YELLOW:
            continue
        moved_yellow.add(target)

    moved_magenta: set[tuple[int, int]] = set()
    for cell in state.magenta:
        if cell not in moving:
            moved_magenta.add(cell)
            continue
        target = (cell[0] + dx, cell[1] + dy)
        if bins.get(target) == MAGENTA:
            continue
        moved_magenta.add(target)

    occupied_after = moved_yellow | moved_magenta
    for cell in blade_destination:
        if not in_bounds(cell) or cell in spec.walls or cell in bins or cell in occupied_after:
            return SweepState(blade=state.blade, yellow=state.yellow, magenta=state.magenta, budget=next_budget)

    return SweepState(
        blade=new_blade, yellow=frozenset(moved_yellow), magenta=frozenset(moved_magenta), budget=next_budget
    )


def pellet_count(state: SweepState) -> int:
    return len(state.yellow) + len(state.magenta)


def blade_sprite_pixels(last_dir: int) -> list[list[int]]:
    pixels = np.full((2 * CELL_SIZE, 3 * CELL_SIZE), COLOR_BLADE, dtype=np.int16)
    if last_dir == UP:
        pixels[:CELL_SIZE, :] = COLOR_BLADE_EDGE
    elif last_dir == DOWN:
        pixels[CELL_SIZE:, :] = COLOR_BLADE_EDGE
    elif last_dir == LEFT:
        pixels[:, :CELL_SIZE] = COLOR_BLADE_EDGE
    else:
        pixels[:, 2 * CELL_SIZE :] = COLOR_BLADE_EDGE
    return pixels.tolist()


def pellet_pixels(color_id: int) -> list[list[int]]:
    return [
        [COLOR_FLOOR, color_id, color_id, COLOR_FLOOR],
        [color_id, color_id, color_id, color_id],
        [color_id, color_id, color_id, color_id],
        [COLOR_FLOOR, color_id, color_id, COLOR_FLOOR],
    ]


def bin_pixels(rim_color: int) -> list[list[int]]:
    cell = np.array(
        [
            [rim_color, rim_color, rim_color, rim_color],
            [rim_color, COLOR_BIN_CAVITY, COLOR_BIN_CAVITY, rim_color],
            [rim_color, COLOR_BIN_CAVITY, COLOR_BIN_CAVITY, rim_color],
            [rim_color, rim_color, rim_color, rim_color],
        ],
        dtype=np.int16,
    )
    return np.tile(cell, (2, 2)).tolist()


def build_floor_pixels() -> list[list[int]]:
    pixels = np.full((DISPLAY_SIZE, DISPLAY_SIZE), COLOR_FLOOR, dtype=np.int16)
    pixels[: HUD_ROWS * CELL_SIZE, :] = COLOR_HUD_BG
    return pixels.tolist()


def build_hud_pixels(remaining_budget: int, max_budget: int) -> list[list[int]]:
    pixels = np.full((HUD_ROWS * CELL_SIZE, DISPLAY_SIZE), COLOR_HUD_BG, dtype=np.int16)
    for idx in range(MAX_BUDGET_PIPS):
        row = idx // LOGICAL_GRID_WIDTH
        col = idx % LOGICAL_GRID_WIDTH
        y0 = row * CELL_SIZE
        x0 = col * CELL_SIZE
        if idx < remaining_budget:
            fill = COLOR_YELLOW_PELLET
        elif idx < max_budget:
            fill = COLOR_SPENT
        else:
            fill = COLOR_HUD_BG
        pixels[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = fill
    return pixels.tolist()


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.remaining = capacity

    def set_remaining(self, remaining: int) -> None:
        self.remaining = max(0, min(int(remaining), self.capacity))

    def reset(self) -> None:
        self.remaining = self.capacity

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[: HUD_ROWS * CELL_SIZE, :] = np.array(build_hud_pixels(self.remaining, self.capacity), dtype=np.int16)
        return frame


class SweeperBlade(ARCBaseGame):
    def __init__(self, starting_budget: int | None = None):
        levels = [
            Level(grid_size=(DISPLAY_SIZE, DISPLAY_SIZE), data={"level_index": idx}, name=f"sweeper_blade_{idx + 1}")
            for idx in range(len(LEVEL_SPECS))
        ]
        self._move_budget = MoveBudgetDisplay(starting_budget or LEVEL_SPECS[0].budget)
        camera = Camera(width=DISPLAY_SIZE, height=DISPLAY_SIZE, background=COLOR_FLOOR, interfaces=[self._move_budget])
        self._route_score = 0
        self._state_data = initial_state(0)
        self._last_dir = RIGHT
        self._complete = False
        super().__init__(
            game_id="sweeper_blade-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[UP, DOWN, LEFT, RIGHT],
        )

    def _level_spec(self) -> LevelSpec:
        return LEVEL_SPECS[self.level_index]

    def _render_level(self) -> None:
        level = self.current_level
        level.remove_all_sprites()
        level.add_sprite(Sprite(pixels=build_floor_pixels(), name=FLOOR_SPRITE, layer=-10, collidable=False))
        level.add_sprite(
            Sprite(
                pixels=build_hud_pixels(self._state_data.budget, self._level_spec().budget),
                name=HUD_SPRITE,
                layer=-9,
                collidable=False,
            )
        )

        for wall in sorted(self._level_spec().walls):
            x, y = logical_to_pixel(wall, playfield=True)
            level.add_sprite(
                Sprite(
                    pixels=np.full((CELL_SIZE, CELL_SIZE), COLOR_WALL, dtype=np.int16).tolist(),
                    name=f"{WALL_SPRITE}_{wall[0]}_{wall[1]}",
                    x=x,
                    y=y,
                    layer=-4,
                )
            )

        for color, cells in self._level_spec().bins:
            rim = COLOR_YELLOW_BIN if color == YELLOW else COLOR_MAGENTA_BIN
            anchor = min(cells, key=lambda cell: (cell[1], cell[0]))
            x, y = logical_to_pixel(anchor, playfield=True)
            level.add_sprite(
                Sprite(
                    pixels=bin_pixels(rim),
                    name=f"{BIN_SPRITE}_{color}_{anchor[0]}_{anchor[1]}",
                    x=x,
                    y=y,
                    layer=-3,
                    tags=[BIN_SPRITE, color],
                )
            )

        for cell in sorted(self._state_data.yellow):
            x, y = logical_to_pixel(cell, playfield=True)
            level.add_sprite(
                Sprite(
                    pixels=pellet_pixels(COLOR_YELLOW_PELLET),
                    name=f"{PELLET_SPRITE}_yellow_{cell[0]}_{cell[1]}",
                    x=x,
                    y=y,
                    layer=-2,
                    collidable=False,
                    tags=[PELLET_SPRITE, YELLOW],
                )
            )

        for cell in sorted(self._state_data.magenta):
            x, y = logical_to_pixel(cell, playfield=True)
            level.add_sprite(
                Sprite(
                    pixels=pellet_pixels(COLOR_MAGENTA_PELLET),
                    name=f"{PELLET_SPRITE}_magenta_{cell[0]}_{cell[1]}",
                    x=x,
                    y=y,
                    layer=-2,
                    collidable=False,
                    tags=[PELLET_SPRITE, MAGENTA],
                )
            )

        bx, by = logical_to_pixel(self._state_data.blade, playfield=True)
        level.add_sprite(
            Sprite(
                pixels=blade_sprite_pixels(self._last_dir),
                name=BLADE_SPRITE,
                x=bx,
                y=by,
                layer=1,
                collidable=False,
                tags=[BLADE_SPRITE, DIR_TO_NAME[self._last_dir]],
            )
        )
        self._move_budget.capacity = self._level_spec().budget
        self._move_budget.set_remaining(self._state_data.budget)

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index"))
        self._state_data = initial_state(level_index)
        self._last_dir = RIGHT
        self._move_budget.capacity = self._level_spec().budget
        self._move_budget.reset()
        self._render_level()

    def _get_hidden_state(self) -> np.ndarray:
        hidden = np.zeros((PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH), dtype=np.int16)
        for x, y in self._state_data.yellow:
            hidden[y, x] = 1
        for x, y in self._state_data.magenta:
            hidden[y, x] = 2
        for x, y in blade_cells(self._state_data.blade):
            if in_bounds((x, y)):
                hidden[y, x] = 3
        return hidden

    def step(self) -> None:
        if self._complete:
            self.complete_action()
            return

        raw_action_id = self.action.id
        action_id = int(getattr(raw_action_id, "value", raw_action_id))

        if action_id in {SPACE, CLICK}:
            self.complete_action()
            return

        if action_id in DIR_TO_DELTA:
            self._last_dir = action_id
            self._state_data = apply_directional_move(self._level_spec(), self._state_data, action_id)
            self._render_level()

        if pellet_count(self._state_data) == 0:
            if self.is_last_level():
                self._complete = True
                self.next_level()
                self.complete_action()
                return
            self.next_level()
            self.complete_action()
            return

        if self._state_data.budget == 0:
            self.lose()
            self.complete_action()
            return

        self.complete_action()
