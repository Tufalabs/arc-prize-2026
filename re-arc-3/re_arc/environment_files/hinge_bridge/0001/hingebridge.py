from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_SIZE: Final[int] = 12
CELL_SIZE: Final[int] = 4
PLAYFIELD_ORIGIN_X: Final[int] = 8
PLAYFIELD_ORIGIN_Y: Final[int] = 16
SCREEN_SIZE: Final[int] = 64
MAX_BUDGET: Final[int] = 30

COLOR_BG: Final[int] = 5
COLOR_FLOOR: Final[int] = 3
COLOR_FLOOR_HI: Final[int] = 2
COLOR_BAR: Final[int] = 12
COLOR_BAR_DARK: Final[int] = 13
COLOR_ANCHOR_RING: Final[int] = 15
COLOR_ANCHOR_HUB: Final[int] = 11
COLOR_WALKER: Final[int] = 9
COLOR_WALKER_HI: Final[int] = 10
COLOR_GOAL: Final[int] = 14
COLOR_FAIL: Final[int] = 8

ACTION_UP: Final[int] = 1
ACTION_DOWN: Final[int] = 2
ACTION_LEFT: Final[int] = 3
ACTION_RIGHT: Final[int] = 4
ACTION_SPACE: Final[int] = 5
ACTION_CLICK: Final[int] = 6

MOVE_DELTAS: Final[dict[int, tuple[int, int]]] = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


class AnchorSpec(NamedTuple):
    x: int
    y: int
    length: int
    initial_horizontal: bool


class LevelSpec(NamedTuple):
    budget: int
    start: tuple[int, int]
    floor: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    anchors: tuple[AnchorSpec, ...]


class LevelState(NamedTuple):
    walker: tuple[int, int]
    horizontal: tuple[bool, ...]
    remaining_budget: int
    failed: bool = False


def _rect(x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}


def build_level_specs() -> tuple[LevelSpec, ...]:
    level1_floor = _rect(1, 5, 3, 7) | _rect(9, 5, 11, 7) | {(6, 6)}
    level1_goals = frozenset(_rect(10, 5, 11, 6))

    level2_floor = {
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (4, 8),
        (5, 8),
        (5, 7),
        (4, 6),
        (5, 6),
        (6, 6),
        (7, 6),
        (8, 6),
        (7, 5),
        (8, 5),
        (7, 4),
        (8, 4),
    }
    level2_goals = frozenset({(7, 4), (8, 4), (7, 5), (8, 5)})

    level3_floor = {(3, 7), (4, 6), (4, 4), (8, 4), (9, 5), (10, 5), (6, 1), (7, 1), (6, 2), (7, 2)}
    level3_goals = frozenset({(6, 1), (7, 1), (6, 2), (7, 2)})

    return (
        LevelSpec(
            budget=24,
            start=(2, 6),
            floor=frozenset(level1_floor),
            goals=level1_goals,
            anchors=(AnchorSpec(x=6, y=6, length=5, initial_horizontal=False),),
        ),
        LevelSpec(
            budget=30,
            start=(3, 8),
            floor=frozenset(level2_floor),
            goals=level2_goals,
            anchors=(
                AnchorSpec(x=3, y=8, length=5, initial_horizontal=False),
                AnchorSpec(x=6, y=6, length=5, initial_horizontal=True),
            ),
        ),
        LevelSpec(
            budget=30,
            start=(3, 7),
            floor=frozenset(level3_floor),
            goals=level3_goals,
            anchors=(
                AnchorSpec(x=3, y=7, length=7, initial_horizontal=True),
                AnchorSpec(x=8, y=4, length=7, initial_horizontal=True),
            ),
        ),
    )


LEVEL_SPECS: Final[tuple[LevelSpec, ...]] = build_level_specs()


def make_initial_state(level_spec: LevelSpec) -> LevelState:
    return LevelState(
        walker=level_spec.start,
        horizontal=tuple(anchor.initial_horizontal for anchor in level_spec.anchors),
        remaining_budget=level_spec.budget,
    )


def _inside_grid(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def bar_cells(anchor: AnchorSpec, horizontal: bool) -> tuple[tuple[int, int], ...]:
    radius = (anchor.length - 1) // 2
    if horizontal:
        return tuple((anchor.x + dx, anchor.y) for dx in range(-radius, radius + 1) if 0 <= anchor.x + dx < GRID_SIZE)
    return tuple((anchor.x, anchor.y + dy) for dy in range(-radius, radius + 1) if 0 <= anchor.y + dy < GRID_SIZE)


def anchor_click_point(anchor: AnchorSpec) -> tuple[int, int]:
    return (PLAYFIELD_ORIGIN_X + anchor.x * CELL_SIZE + 1, PLAYFIELD_ORIGIN_Y + anchor.y * CELL_SIZE + 1)


def pixel_to_logical(px: int, py: int) -> tuple[int, int] | None:
    if px < PLAYFIELD_ORIGIN_X or py < PLAYFIELD_ORIGIN_Y:
        return None
    rel_x = px - PLAYFIELD_ORIGIN_X
    rel_y = py - PLAYFIELD_ORIGIN_Y
    if rel_x >= GRID_SIZE * CELL_SIZE or rel_y >= GRID_SIZE * CELL_SIZE:
        return None
    return (rel_x // CELL_SIZE, rel_y // CELL_SIZE)


def _cell_has_bar(level_spec: LevelSpec, horizontal: tuple[bool, ...], cell: tuple[int, int]) -> tuple[bool, bool]:
    for idx, anchor in enumerate(level_spec.anchors):
        cells = bar_cells(anchor, horizontal[idx])
        if cell not in cells:
            continue
        if cell == (anchor.x, anchor.y):
            return True, False
        return True, True
    return False, False


def is_walkable(level_spec: LevelSpec, horizontal: tuple[bool, ...], cell: tuple[int, int]) -> bool:
    if not _inside_grid(cell):
        return False

    has_bar, is_non_anchor_bar = _cell_has_bar(level_spec, horizontal, cell)
    if has_bar and not is_non_anchor_bar:
        return True

    base_is_floor = cell in level_spec.floor or cell in level_spec.goals
    if has_bar and is_non_anchor_bar:
        return not base_is_floor
    return base_is_floor


def try_move(level_spec: LevelSpec, state: LevelState, delta: tuple[int, int]) -> LevelState:
    next_cell = (state.walker[0] + delta[0], state.walker[1] + delta[1])
    if is_walkable(level_spec, state.horizontal, next_cell):
        return LevelState(
            walker=next_cell, horizontal=state.horizontal, remaining_budget=state.remaining_budget, failed=state.failed
        )
    return state


def toggle_anchor(level_spec: LevelSpec, state: LevelState, anchor_index: int) -> LevelState:
    anchor = level_spec.anchors[anchor_index]
    current_horizontal = state.horizontal[anchor_index]
    current_cells = set(bar_cells(anchor, current_horizontal))
    next_cells = set(bar_cells(anchor, not current_horizontal))
    if state.walker in current_cells | next_cells:
        return state

    updated = list(state.horizontal)
    updated[anchor_index] = not current_horizontal
    return LevelState(
        walker=state.walker, horizontal=tuple(updated), remaining_budget=state.remaining_budget, failed=state.failed
    )


def apply_abstract_action(
    level_spec: LevelSpec, state: LevelState, action_id: int, anchor_index: int | None = None
) -> LevelState:
    if state.failed:
        return state

    if action_id == ACTION_SPACE:
        return make_initial_state(level_spec)

    next_state = state
    if action_id in MOVE_DELTAS:
        next_state = try_move(level_spec, state, MOVE_DELTAS[action_id])
    elif action_id == ACTION_CLICK and anchor_index is not None:
        next_state = toggle_anchor(level_spec, state, anchor_index)

    if next_state.walker in level_spec.goals:
        return next_state

    remaining = next_state.remaining_budget - 1
    failed = remaining <= 0
    return LevelState(
        walker=next_state.walker, horizontal=next_state.horizontal, remaining_budget=max(0, remaining), failed=failed
    )


class HingeBridge(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_specs = LEVEL_SPECS
        self._level_state = make_initial_state(self._level_specs[0])
        levels = [
            Level(
                name=f"Level {index + 1}",
                grid_size=(SCREEN_SIZE, SCREEN_SIZE),
                sprites=[
                    Sprite(name="framebuffer", pixels=np.full((SCREEN_SIZE, SCREEN_SIZE), COLOR_BG, dtype=np.int8))
                ],
            )
            for index in range(len(LEVEL_SPECS))
        ]
        super().__init__(
            game_id="hinge_bridge-0001",
            levels=levels,
            camera=Camera(width=SCREEN_SIZE, height=SCREEN_SIZE, background=COLOR_BG, letter_box=COLOR_BG),
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        del level
        self._level_state = make_initial_state(self._level_specs[self.level_index])
        self._refresh_framebuffer()

    def describe_level_for_solver(self, level_index: int | None = None) -> LevelSpec:
        if level_index is None:
            level_index = self.level_index
        return self._level_specs[int(level_index)]

    def _refresh_framebuffer(self) -> None:
        framebuffer = self.current_level.get_sprites_by_name("framebuffer")[0]
        framebuffer.pixels = self._render_frame()

    def _fill_cell(self, frame: np.ndarray, cell: tuple[int, int], pixels: list[list[int]]) -> None:
        x0 = PLAYFIELD_ORIGIN_X + cell[0] * CELL_SIZE
        y0 = PLAYFIELD_ORIGIN_Y + cell[1] * CELL_SIZE
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = np.array(pixels, dtype=np.int8)

    def _render_budget(self, frame: np.ndarray, remaining_budget: int) -> None:
        pip_w = 2
        pip_h = 3
        gap = 1
        total_width = (15 * pip_w) + (14 * gap)
        origin_x = (SCREEN_SIZE - total_width) // 2
        origin_y = 1
        for idx in range(MAX_BUDGET):
            row = idx // 15
            col = idx % 15
            x0 = origin_x + col * (pip_w + gap)
            y0 = origin_y + row * 4
            color = COLOR_GOAL if idx < remaining_budget else COLOR_FLOOR
            if remaining_budget <= 3 and idx < remaining_budget:
                color = COLOR_FAIL
            frame[y0 : y0 + pip_h, x0 : x0 + pip_w] = color

    def _render_frame(self) -> np.ndarray:
        level_spec = self._level_specs[self.level_index]
        frame = np.full((SCREEN_SIZE, SCREEN_SIZE), COLOR_BG, dtype=np.int8)

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell = (x, y)
                tile = [
                    [COLOR_BG, COLOR_BG, COLOR_BG, COLOR_BG],
                    [COLOR_BG, COLOR_BG, COLOR_BG, COLOR_BG],
                    [COLOR_BG, COLOR_BG, COLOR_BG, COLOR_BG],
                    [COLOR_BG, COLOR_BG, COLOR_BG, COLOR_BG],
                ]
                if cell in level_spec.floor or cell in level_spec.goals:
                    tile = [
                        [COLOR_FLOOR, COLOR_FLOOR, COLOR_FLOOR, COLOR_FLOOR],
                        [COLOR_FLOOR, COLOR_FLOOR_HI, COLOR_FLOOR_HI, COLOR_FLOOR],
                        [COLOR_FLOOR, COLOR_FLOOR_HI, COLOR_FLOOR_HI, COLOR_FLOOR],
                        [COLOR_FLOOR, COLOR_FLOOR, COLOR_FLOOR, COLOR_FLOOR],
                    ]

                for idx, anchor in enumerate(level_spec.anchors):
                    bar = bar_cells(anchor, self._level_state.horizontal[idx])
                    if cell not in bar or cell == (anchor.x, anchor.y):
                        continue
                    if self._level_state.horizontal[idx]:
                        tile[1] = [COLOR_BAR, COLOR_BAR, COLOR_BAR, COLOR_BAR]
                        tile[2] = [COLOR_BAR, COLOR_BAR_DARK, COLOR_BAR_DARK, COLOR_BAR]
                    else:
                        tile[0][1] = COLOR_BAR
                        tile[0][2] = COLOR_BAR
                        tile[1][1] = COLOR_BAR
                        tile[1][2] = COLOR_BAR_DARK
                        tile[2][1] = COLOR_BAR_DARK
                        tile[2][2] = COLOR_BAR
                        tile[3][1] = COLOR_BAR
                        tile[3][2] = COLOR_BAR

                for anchor in level_spec.anchors:
                    if cell == (anchor.x, anchor.y):
                        tile = [
                            [COLOR_ANCHOR_RING, COLOR_ANCHOR_RING, COLOR_ANCHOR_RING, COLOR_ANCHOR_RING],
                            [COLOR_ANCHOR_RING, COLOR_ANCHOR_HUB, COLOR_ANCHOR_HUB, COLOR_ANCHOR_RING],
                            [COLOR_ANCHOR_RING, COLOR_ANCHOR_HUB, COLOR_ANCHOR_HUB, COLOR_ANCHOR_RING],
                            [COLOR_ANCHOR_RING, COLOR_ANCHOR_RING, COLOR_ANCHOR_RING, COLOR_ANCHOR_RING],
                        ]

                if cell in level_spec.goals:
                    tile = [
                        [COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL],
                        [COLOR_GOAL, COLOR_ANCHOR_HUB, COLOR_ANCHOR_HUB, COLOR_GOAL],
                        [COLOR_GOAL, COLOR_ANCHOR_HUB, COLOR_ANCHOR_HUB, COLOR_GOAL],
                        [COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL],
                    ]

                if cell == self._level_state.walker:
                    tile = [
                        [0, COLOR_WALKER, COLOR_WALKER, 0],
                        [COLOR_WALKER, COLOR_WALKER_HI, COLOR_WALKER_HI, COLOR_WALKER],
                        [COLOR_WALKER, COLOR_WALKER, COLOR_WALKER, COLOR_WALKER],
                        [0, COLOR_WALKER, COLOR_WALKER, 0],
                    ]

                self._fill_cell(frame, cell, tile)

        self._render_budget(frame, self._level_state.remaining_budget)

        if self._level_state.failed:
            x0 = PLAYFIELD_ORIGIN_X
            y0 = PLAYFIELD_ORIGIN_Y
            x1 = PLAYFIELD_ORIGIN_X + GRID_SIZE * CELL_SIZE - 1
            y1 = PLAYFIELD_ORIGIN_Y + GRID_SIZE * CELL_SIZE - 1
            frame[y0, x0 : x1 + 1] = COLOR_FAIL
            frame[y1, x0 : x1 + 1] = COLOR_FAIL
            frame[y0 : y1 + 1, x0] = COLOR_FAIL
            frame[y0 : y1 + 1, x1] = COLOR_FAIL

        return frame

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
        level_spec = self._level_specs[self.level_index]

        if self._level_state.failed:
            self.lose()
            self.complete_action()
            return

        if action_id == ACTION_SPACE:
            self._level_state = make_initial_state(level_spec)
            self._refresh_framebuffer()
            self.complete_action()
            return

        if action_id in MOVE_DELTAS:
            next_state = apply_abstract_action(level_spec, self._level_state, action_id)
        elif action_id == ACTION_CLICK:
            click_x = int(self.action.data.get("x", -1))
            click_y = int(self.action.data.get("y", -1))
            logical = pixel_to_logical(click_x, click_y)
            anchor_index = None
            if logical is not None:
                for idx, anchor in enumerate(level_spec.anchors):
                    if logical == (anchor.x, anchor.y):
                        anchor_index = idx
                        break
            next_state = apply_abstract_action(level_spec, self._level_state, action_id, anchor_index=anchor_index)
        else:
            next_state = apply_abstract_action(level_spec, self._level_state, action_id)

        self._level_state = next_state
        self._refresh_framebuffer()

        if self._level_state.failed:
            self.lose()
            self.complete_action()
            return

        if self._level_state.walker in level_spec.goals and not self._level_state.failed:
            self.next_level()

        self.complete_action()
