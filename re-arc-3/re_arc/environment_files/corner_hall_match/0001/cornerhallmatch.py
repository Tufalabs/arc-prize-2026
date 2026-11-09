from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

BACKGROUND_COLOR = 1
FLOOR_COLOR = 0
WALL_COLOR = 4
AVATAR_BODY_COLOR = 9
AVATAR_HIGHLIGHT_COLOR = 10
GOAL_COLOR = 11
ACTIVE_PIP_COLOR = 12
SPENT_PIP_COLOR = 3
SUCCESS_COLOR = 14
FAILURE_COLOR = 8

MAZE_ORIGIN_X = 8
MAZE_ORIGIN_Y = 12
CELL_SIZE = 4
LOGICAL_SIZE = 12
MAX_DISPLAYED_BUDGET = 24

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.budget = 0
        self.remaining = 0
        self.failed = False

    def set_state(self, budget: int, remaining: int, *, failed: bool = False) -> None:
        self.budget = max(0, min(int(budget), MAX_DISPLAYED_BUDGET))
        self.remaining = max(0, min(int(remaining), self.budget))
        self.failed = bool(failed)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for index in range(MAX_DISPLAYED_BUDGET):
            row = index // 12
            col = index % 12
            x0 = 8 + 4 * col
            y0 = 2 + 4 * row
            if index >= self.budget:
                continue
            if self.failed:
                color = FAILURE_COLOR
            elif index < self.remaining:
                color = ACTIVE_PIP_COLOR
            else:
                color = SPENT_PIP_COLOR
            frame[y0 : y0 + 3, x0 : x0 + 3] = color
        return frame


def _cell_origin(cell_x: int, cell_y: int) -> tuple[int, int]:
    return MAZE_ORIGIN_X + CELL_SIZE * cell_x, MAZE_ORIGIN_Y + CELL_SIZE * cell_y


def _make_avatar_sprite(color: int | None = None) -> Sprite:
    base = np.array(
        [
            [-1, AVATAR_BODY_COLOR, AVATAR_BODY_COLOR, -1],
            [AVATAR_BODY_COLOR, AVATAR_HIGHLIGHT_COLOR, AVATAR_HIGHLIGHT_COLOR, AVATAR_BODY_COLOR],
            [AVATAR_BODY_COLOR, AVATAR_HIGHLIGHT_COLOR, AVATAR_HIGHLIGHT_COLOR, AVATAR_BODY_COLOR],
            [-1, AVATAR_BODY_COLOR, AVATAR_BODY_COLOR, -1],
        ],
        dtype=np.int8,
    )
    if color is not None:
        base = np.where(base >= 0, np.int8(color), np.int8(-1))
    return Sprite(base, name="avatar", layer=2)


def _make_goal_sprite(color: int) -> Sprite:
    pixels = np.array(
        [[color, color, color, color], [color, -1, -1, color], [color, -1, -1, color], [color, color, color, color]],
        dtype=np.int8,
    )
    return Sprite(pixels, name="goal", layer=1)


def _parse_layout(rows: tuple[str, ...]) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    walkable: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    if len(rows) != LOGICAL_SIZE:
        raise ValueError("Each level must have exactly 12 rows.")
    for y, row in enumerate(rows):
        if len(row) != LOGICAL_SIZE:
            raise ValueError("Each level row must have exactly 12 columns.")
        for x, char in enumerate(row):
            if char in ".SG":
                walkable.add((x, y))
            if char == "S":
                start = (x, y)
            elif char == "G":
                goal = (x, y)
    if start is None or goal is None:
        raise ValueError("Each level requires exactly one start and one goal.")
    return walkable, start, goal


def _build_maze_sprite(walkable: set[tuple[int, int]]) -> Sprite:
    pixels = np.full((64, 64), -1, dtype=np.int8)
    for y in range(LOGICAL_SIZE):
        for x in range(LOGICAL_SIZE):
            px, py = _cell_origin(x, y)
            color = FLOOR_COLOR if (x, y) in walkable else WALL_COLOR
            pixels[py : py + CELL_SIZE, px : px + CELL_SIZE] = color
    return Sprite(pixels, name="maze", layer=0, collidable=False)


def _build_level(name: str, rows: tuple[str, ...], budget: int, optimal_moves: int) -> Level:
    walkable, start, goal = _parse_layout(rows)
    goal_sprite = _make_goal_sprite(GOAL_COLOR)
    goal_sprite.set_position(*_cell_origin(*goal))
    avatar_sprite = _make_avatar_sprite()
    avatar_sprite.set_position(*_cell_origin(*start))
    sprites = [_build_maze_sprite(walkable), goal_sprite, avatar_sprite]
    data = {
        "layout": rows,
        "walkable": tuple(sorted(walkable)),
        "start": start,
        "goal": goal,
        "budget": int(budget),
        "optimal_moves": int(optimal_moves),
    }
    return Level(sprites=sprites, grid_size=(64, 64), data=data, name=name)


levels = [
    _build_level(
        "One bend teaches the rule",
        (
            "############",
            "############",
            "############",
            "############",
            "############",
            "#####G######",
            "#####.######",
            "#####.######",
            "#S....######",
            "############",
            "############",
            "############",
        ),
        budget=21,
        optimal_moves=7,
    ),
    _build_level(
        "A dead end is not the route",
        (
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
            "####.##G####",
            "####.##.####",
            "#S......####",
            "############",
            "############",
            "############",
        ),
        budget=24,
        optimal_moves=8,
    ),
    _build_level(
        "Loop: shorter path starts by moving away",
        (
            "############",
            "############",
            "####...#####",
            "####.#..####",
            "##G..##S####",
            "####....####",
            "############",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
        budget=21,
        optimal_moves=7,
    ),
]


class CornerHallMatch(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._budget_display = MoveBudgetDisplay()
        self._player_cell = (0, 0)
        self._goal_cell = (0, 0)
        self._walkable_cells: set[tuple[int, int]] = set()
        self._remaining_moves = 0
        self._level_budget = 0
        self._avatar_sprite: Sprite | None = None
        self._goal_sprite: Sprite | None = None
        super().__init__(
            "corner_hall_match",
            levels,
            Camera(0, 0, 64, 64, BACKGROUND_COLOR, BACKGROUND_COLOR, [self._budget_display]),
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._player_cell = tuple(level.get_data("start"))
        self._goal_cell = tuple(level.get_data("goal"))
        self._walkable_cells = {tuple(cell) for cell in level.get_data("walkable")}
        self._level_budget = int(level.get_data("budget"))
        self._remaining_moves = self._level_budget
        self._avatar_sprite = level.get_sprites_by_name("avatar")[0]
        self._goal_sprite = level.get_sprites_by_name("goal")[0]
        self._avatar_sprite.pixels = _make_avatar_sprite().pixels
        self._goal_sprite.pixels = _make_goal_sprite(GOAL_COLOR).pixels
        self._avatar_sprite.set_position(*_cell_origin(*self._player_cell))
        self._goal_sprite.set_position(*_cell_origin(*self._goal_cell))
        self._budget_display.set_state(self._level_budget, self._remaining_moves, failed=False)

    def _move_avatar_to_cell(self, cell: tuple[int, int]) -> None:
        self._player_cell = cell
        if self._avatar_sprite is not None:
            self._avatar_sprite.set_position(*_cell_origin(*cell))

    def _recolor_success(self) -> None:
        if self._avatar_sprite is not None:
            self._avatar_sprite.pixels = _make_avatar_sprite(SUCCESS_COLOR).pixels
        if self._goal_sprite is not None:
            self._goal_sprite.pixels = _make_goal_sprite(SUCCESS_COLOR).pixels

    def _recolor_failure(self) -> None:
        if self._avatar_sprite is not None:
            self._avatar_sprite.pixels = _make_avatar_sprite(FAILURE_COLOR).pixels
        self._budget_display.set_state(self._level_budget, self._remaining_moves, failed=True)

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
        self._remaining_moves = max(0, self._remaining_moves - 1)

        move_delta = MOVE_DELTAS.get(action_id)
        if move_delta is not None:
            next_cell = (self._player_cell[0] + move_delta[0], self._player_cell[1] + move_delta[1])
            if next_cell in self._walkable_cells:
                self._move_avatar_to_cell(next_cell)

        if self._player_cell == self._goal_cell:
            self._recolor_success()
            self._budget_display.set_state(self._level_budget, self._remaining_moves, failed=False)
            self.next_level()
            self.complete_action()
            return

        if self._remaining_moves == 0:
            self._recolor_failure()
            self.lose()
            self.complete_action()
            return

        self._budget_display.set_state(self._level_budget, self._remaining_moves, failed=False)
        self.complete_action()
