from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

BOARD_ORIGIN_X = 4
BOARD_ORIGIN_Y = 8
CELL_SIZE = 7
TILE_INSET = 1
INNER_SIZE = 5
TOTAL_LEVELS = 3

COLOR_BG = 0
COLOR_SPENT_PIP = 3
COLOR_HAZARD_ACCENT = 8
COLOR_SAFE_CENTER = 9
COLOR_SAFE_BORDER = 10
COLOR_GOAL = 11
COLOR_HAZARD_BODY = 12
COLOR_HAZARD_CORE = 13
COLOR_ACTIVE = 14
COLOR_TOKEN_CORE = 15

SAFE_TILE = np.array(
    [[10, 10, 10, 10, 10], [10, 9, 9, 9, 10], [10, 9, 10, 9, 10], [10, 9, 9, 9, 10], [10, 10, 10, 10, 10]],
    dtype=np.int8,
)
HAZARD_TILE = np.array(
    [[0, 8, 0, 8, 0], [8, 12, 8, 12, 8], [0, 8, 13, 8, 0], [8, 12, 8, 12, 8], [0, 8, 0, 8, 0]], dtype=np.int8
)
GOAL_OUTLINE = np.array(
    [[11, 11, 11, 11, 11], [11, 0, 0, 0, 11], [11, 0, 11, 0, 11], [11, 0, 0, 0, 11], [11, 11, 11, 11, 11]],
    dtype=np.int8,
)
GOAL_SUCCESS = np.array(
    [[11, 11, 11, 11, 11], [11, 14, 14, 14, 11], [11, 14, 11, 14, 11], [11, 14, 14, 14, 11], [11, 11, 11, 11, 11]],
    dtype=np.int8,
)
TOKEN = np.array([[0, 14, 0], [14, 15, 14], [0, 14, 0]], dtype=np.int8)
FAILURE_X = np.array(
    [[8, 0, 0, 0, 8], [0, 8, 0, 8, 0], [0, 0, 13, 0, 0], [0, 8, 0, 8, 0], [8, 0, 0, 0, 8]], dtype=np.int8
)

ACTION_TO_DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


LEVEL_SPECS = (
    {
        "name": "level_1",
        "rows": ("........", "........", "........", ".S.XX.G.", "...XX...", "........", "........", "........"),
        "start": (3, 1),
        "goal": (3, 6),
        "budget": 13,
    },
    {
        "name": "level_2",
        "rows": ("........", "..XX..G.", "..XX....", "....XX..", "....XX..", "..XX....", ".SXX....", "........"),
        "start": (6, 1),
        "goal": (1, 6),
        "budget": 16,
    },
    {
        "name": "level_3",
        "rows": ("XXXXXXXX", "XXXXX.XX", "X...X..X", "X.X.X..X", "X.X....X", "X.X.XXGX", "XSX.XXXX", "XXXXXXXX"),
        "start": (6, 1),
        "goal": (5, 6),
        "budget": 20,
    },
)


def _level_from_spec(spec: dict[str, object]) -> Level:
    safe_cells = []
    hazard_cells = []
    rows = tuple(str(row) for row in spec["rows"])
    start = tuple(int(value) for value in spec["start"])
    goal = tuple(int(value) for value in spec["goal"])
    budget = int(spec["budget"])
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            if cell == "X":
                hazard_cells.append((row_idx, col_idx))
            else:
                safe_cells.append((row_idx, col_idx))
    return Level(
        grid_size=(64, 64),
        data={
            "rows": list(rows),
            "start": list(start),
            "goal": list(goal),
            "budget": budget,
            "safe_cells": [list(cell) for cell in safe_cells],
            "hazard_cells": [list(cell) for cell in hazard_cells],
            "optimal_path_len": _shortest_path_len(spec),
        },
        name=str(spec["name"]),
    )


def _shortest_path_len(spec: dict[str, object]) -> int:
    rows = tuple(str(row) for row in spec["rows"])
    start = tuple(int(value) for value in spec["start"])
    goal = tuple(int(value) for value in spec["goal"])
    queue = [(start, 0)]
    visited = {start}
    while queue:
        (row, col), dist = queue.pop(0)
        if (row, col) == goal:
            return dist
        for d_row, d_col in ACTION_TO_DELTA.values():
            next_row = row + d_row
            next_col = col + d_col
            if not (0 <= next_row < 8 and 0 <= next_col < 8):
                continue
            if rows[next_row][next_col] == "X":
                continue
            next_pos = (next_row, next_col)
            if next_pos in visited:
                continue
            visited.add(next_pos)
            queue.append((next_pos, dist + 1))
    raise ValueError(f"Level {spec['name']} is unsolvable.")


def _transparent_canvas(size: int) -> np.ndarray:
    return np.full((size, size), -1, dtype=np.int8)


def _tile_sprite(pattern: np.ndarray, *, name: str, x: int, y: int, layer: int) -> Sprite:
    pixels = _transparent_canvas(CELL_SIZE)
    pixels[TILE_INSET : TILE_INSET + INNER_SIZE, TILE_INSET : TILE_INSET + INNER_SIZE] = pattern
    return Sprite(pixels=pixels, name=name, x=x, y=y, layer=layer, tags=["sys_static"])


def _overlay_sprite(pattern: np.ndarray, *, size: int, offset: int, name: str, x: int, y: int, layer: int) -> Sprite:
    pixels = _transparent_canvas(CELL_SIZE)
    pixels[offset : offset + size, offset : offset + size] = pattern
    return Sprite(pixels=pixels, name=name, x=x, y=y, layer=layer)


class SafeTileTaps(ARCBaseGame):
    def __init__(self) -> None:
        self._level_spec: dict[str, object] | None = None
        self._token_pos = (0, 0)
        self._remaining_budget = 0
        self._phase = "play"
        self._failed_target: tuple[int, int] | None = None
        levels = [_level_from_spec(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "safe_tile_taps-0001", levels, Camera(0, 0, 64, 64, COLOR_BG, COLOR_BG), False, TOTAL_LEVELS, [1, 2, 3, 4]
        )

    def on_set_level(self, _level: Level) -> None:
        spec = LEVEL_SPECS[self.level_index]
        self._level_spec = spec
        self._token_pos = tuple(int(value) for value in spec["start"])
        self._remaining_budget = int(spec["budget"])
        self._phase = "play"
        self._failed_target = None
        self._rebuild_level_sprites()

    def _rebuild_level_sprites(self) -> None:
        level = self.current_level
        level.remove_all_sprites()

        if self._level_spec is None:
            return

        rows = tuple(str(row) for row in self._level_spec["rows"])
        goal = tuple(int(value) for value in self._level_spec["goal"])
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                x = BOARD_ORIGIN_X + (col_idx * CELL_SIZE)
                y = BOARD_ORIGIN_Y + (row_idx * CELL_SIZE)
                base_pattern = HAZARD_TILE if cell == "X" else SAFE_TILE
                level.add_sprite(_tile_sprite(base_pattern, name=f"cell_{row_idx}_{col_idx}", x=x, y=y, layer=1))

                if (row_idx, col_idx) == goal:
                    goal_pattern = GOAL_SUCCESS if self._phase in {"win", "finished"} else GOAL_OUTLINE
                    level.add_sprite(_tile_sprite(goal_pattern, name=f"goal_{row_idx}_{col_idx}", x=x, y=y, layer=3))

        self._add_budget_sprites()

        if self._phase in {"play", "win", "finished"}:
            self._add_token_sprite(self._token_pos)
        elif self._phase == "fail":
            failure_cell = self._failed_target if self._failed_target is not None else self._token_pos
            self._add_failure_sprite(failure_cell)

    def _add_budget_sprites(self) -> None:
        if self._level_spec is None:
            return
        budget = int(self._level_spec["budget"])
        total_width = (budget * 2) + (budget - 1)
        start_x = (64 - total_width) // 2
        for index in range(budget):
            remaining_slot = budget - index
            if index < self._remaining_budget:
                color = (
                    COLOR_HAZARD_BODY
                    if self._remaining_budget <= 3 and remaining_slot <= self._remaining_budget
                    else COLOR_ACTIVE
                )
            else:
                color = COLOR_SPENT_PIP
            pixels = np.full((5, 2), color, dtype=np.int8)
            self.current_level.add_sprite(
                Sprite(pixels=pixels, name=f"pip_{index}", x=start_x + (index * 3), y=1, layer=0, tags=["sys_static"])
            )

    def _add_token_sprite(self, cell: tuple[int, int]) -> None:
        x, y = self._cell_origin(cell)
        self.current_level.add_sprite(_overlay_sprite(TOKEN, size=3, offset=2, name="token", x=x, y=y, layer=4))

    def _add_failure_sprite(self, cell: tuple[int, int]) -> None:
        x, y = self._cell_origin(cell)
        pixels = np.full((CELL_SIZE, CELL_SIZE), COLOR_BG, dtype=np.int8)
        pixels[TILE_INSET : TILE_INSET + INNER_SIZE, TILE_INSET : TILE_INSET + INNER_SIZE] = FAILURE_X
        self.current_level.add_sprite(Sprite(pixels=pixels, name="failure", x=x, y=y, layer=5))

    def _cell_origin(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return BOARD_ORIGIN_X + (col * CELL_SIZE), BOARD_ORIGIN_Y + (row * CELL_SIZE)

    def _cell_value(self, row: int, col: int) -> str:
        if self._level_spec is None:
            return "X"
        rows = tuple(str(value) for value in self._level_spec["rows"])
        return rows[row][col]

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

        if self._phase == "fail":
            self.lose()
            self.complete_action()
            return

        if self._phase == "win":
            self._phase = "finished" if self.is_last_level() else "play"
            self.next_level()
            self.complete_action()
            return

        if self._phase == "finished":
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        d_row, d_col = ACTION_TO_DELTA.get(action_id, (0, 0))
        row, col = self._token_pos
        next_row = row + d_row
        next_col = col + d_col

        if not (0 <= next_row < 8 and 0 <= next_col < 8):
            self._rebuild_level_sprites()
            self.complete_action()
            return

        target = (next_row, next_col)
        target_value = self._cell_value(next_row, next_col)

        self._remaining_budget -= 1
        if target_value == "X":
            self._failed_target = target
            self._phase = "fail"
            self._rebuild_level_sprites()
            self.complete_action()
            return

        self._token_pos = target
        goal = tuple(int(value) for value in self._level_spec["goal"])
        if target == goal:
            self._phase = "win"
            self._rebuild_level_sprites()
            self.complete_action()
            return

        if self._remaining_budget <= 0:
            self._failed_target = self._token_pos
            self._phase = "fail"
            self._rebuild_level_sprites()
            self.complete_action()
            return

        self._rebuild_level_sprites()
        self.complete_action()


Game = SafeTileTaps
