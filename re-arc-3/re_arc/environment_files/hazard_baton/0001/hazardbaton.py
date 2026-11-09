from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 10
CELL_SIZE = 5
ORIGIN_X = 7
ORIGIN_Y = 12

COLOR_WHITE = 0
COLOR_LIGHT_GRAY = 1
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK_GRAY = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_PINK = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14

STATUS_PLAYING = "playing"
STATUS_WON = "won"
STATUS_FAILED_HAZARD = "failed_hazard"
STATUS_FAILED_BUDGET = "failed_budget"

SAFE_TILE = np.array(
    [[3, 3, 3, 3, 3], [3, 2, 2, 2, 3], [3, 2, 2, 2, 3], [3, 2, 2, 2, 3], [3, 3, 3, 3, 3]], dtype=np.int8
)
HAZARD_TILE = np.array(
    [[12, 1, 12, 1, 12], [1, 8, 13, 8, 1], [12, 13, 8, 13, 12], [1, 8, 13, 8, 1], [12, 1, 12, 1, 12]], dtype=np.int8
)
PINK_HOME_TILE = np.array(
    [[11, 2, 2, 2, 11], [2, 4, 2, 4, 2], [2, 2, 2, 2, 2], [2, 4, 2, 4, 2], [11, 2, 2, 2, 11]], dtype=np.int8
)
BLUE_HOME_TILE = np.array(
    [[10, 2, 2, 2, 10], [2, 4, 2, 4, 2], [2, 2, 2, 2, 2], [2, 4, 2, 4, 2], [10, 2, 2, 2, 10]], dtype=np.int8
)

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}


class LevelSpec:
    def __init__(
        self,
        *,
        title: str,
        arena_min_x: int,
        arena_max_x: int,
        arena_min_y: int,
        arena_max_y: int,
        pink_home: tuple[int, int],
        blue_home: tuple[int, int],
        safe_cells: tuple[tuple[int, int], ...],
        move_budget: int,
    ) -> None:
        self.title = title
        self.arena_min_x = arena_min_x
        self.arena_max_x = arena_max_x
        self.arena_min_y = arena_min_y
        self.arena_max_y = arena_max_y
        self.pink_home = pink_home
        self.blue_home = blue_home
        self.safe_cells = safe_cells
        self.move_budget = move_budget


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        title="Level 1",
        arena_min_x=1,
        arena_max_x=8,
        arena_min_y=2,
        arena_max_y=5,
        pink_home=(1, 4),
        blue_home=(8, 4),
        safe_cells=(
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (1, 3),
            (8, 3),
            (1, 5),
            (2, 5),
            (7, 5),
            (8, 5),
            (1, 4),
            (8, 4),
        ),
        move_budget=48,
    ),
    LevelSpec(
        title="Level 2",
        arena_min_x=1,
        arena_max_x=8,
        arena_min_y=2,
        arena_max_y=6,
        pink_home=(1, 4),
        blue_home=(8, 4),
        safe_cells=(
            (1, 4),
            (8, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (4, 3),
            (4, 2),
            (5, 2),
            (6, 2),
            (6, 3),
            (6, 4),
            (4, 5),
            (4, 6),
            (5, 6),
            (6, 6),
            (6, 5),
            (7, 4),
            (1, 5),
            (2, 5),
        ),
        move_budget=54,
    ),
    LevelSpec(
        title="Level 3",
        arena_min_x=1,
        arena_max_x=8,
        arena_min_y=2,
        arena_max_y=6,
        pink_home=(1, 4),
        blue_home=(8, 4),
        safe_cells=(
            (1, 4),
            (8, 4),
            (2, 4),
            (3, 4),
            (1, 5),
            (2, 5),
            (2, 6),
            (3, 3),
            (3, 2),
            (4, 2),
            (5, 2),
            (5, 3),
            (5, 4),
            (3, 5),
            (3, 6),
            (4, 6),
            (5, 6),
            (5, 5),
            (6, 3),
            (7, 3),
            (7, 4),
            (6, 5),
            (7, 5),
        ),
        move_budget=60,
    ),
)


def _solid(color: int) -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), int(color), dtype=np.int8)


def _cell_bounds(cell: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = cell
    px = ORIGIN_X + (x * CELL_SIZE)
    py = ORIGIN_Y + (y * CELL_SIZE)
    return px, py, px + CELL_SIZE, py + CELL_SIZE


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    px0, py0, _, _ = _cell_bounds(cell)
    return px0 + 2, py0 + 2


class HazardBaton(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._level_bases: list[np.ndarray] = []
        self._board_sprite: Sprite | None = None
        self._pink_pos = (0, 0)
        self._blue_pos = (0, 0)
        self._selected = "blue"
        self._moves_used = 0
        self._move_budget = 0
        self._status = STATUS_PLAYING

        levels = [self._build_level(idx, spec) for idx, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_WHITE)
        super().__init__(
            game_id="hazard_baton-0001",
            levels=levels,
            camera=camera,
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 6],
            seed=seed,
        )

    def _build_level(self, idx: int, spec: LevelSpec) -> Level:
        base = _solid(COLOR_WHITE)
        safe_set = set(spec.safe_cells)
        for y in range(spec.arena_min_y, spec.arena_max_y + 1):
            for x in range(spec.arena_min_x, spec.arena_max_x + 1):
                tile = SAFE_TILE if (x, y) in safe_set else HAZARD_TILE
                px0, py0, px1, py1 = _cell_bounds((x, y))
                base[py0:py1, px0:px1] = tile

        for cell, tile in ((spec.blue_home, BLUE_HOME_TILE), (spec.pink_home, PINK_HOME_TILE)):
            px0, py0, px1, py1 = _cell_bounds(cell)
            base[py0:py1, px0:px1] = tile

        self._render_budget(base, moves_used=0, move_budget=spec.move_budget)
        self._level_bases.append(base)
        board_sprite = Sprite(base.copy(), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"])
        return Level(
            name=spec.title,
            grid_size=(GRID_SIZE, GRID_SIZE),
            sprites=[board_sprite],
            data={
                "level_index": idx,
                "pink_home": spec.pink_home,
                "blue_home": spec.blue_home,
                "safe_cells": list(spec.safe_cells),
                "move_budget": spec.move_budget,
            },
        )

    def on_set_level(self, level: Level) -> None:
        level_idx = int(level.get_data("level_index") or self.level_index)
        spec = LEVEL_SPECS[level_idx]
        boards = level.get_sprites_by_name("board")
        self._board_sprite = boards[0] if boards else None
        self._pink_pos = spec.pink_home
        self._blue_pos = spec.blue_home
        self._selected = "blue"
        self._moves_used = 0
        self._move_budget = spec.move_budget
        self._status = STATUS_PLAYING
        self._sync_board()

    def _sync_board(self) -> None:
        if self._board_sprite is None:
            return
        level_idx = int(self.current_level.get_data("level_index") or self.level_index)
        frame = self._level_bases[level_idx].copy()
        self._render_budget(frame, moves_used=self._moves_used, move_budget=self._move_budget)
        self._draw_pawn(frame, self._pink_pos, kind="pink", selected=self._selected == "pink")
        self._draw_pawn(frame, self._blue_pos, kind="blue", selected=self._selected == "blue")
        if self._status == STATUS_FAILED_HAZARD or self._status == STATUS_FAILED_BUDGET:
            self._draw_border(frame, COLOR_MAROON)
        elif self._status == STATUS_WON:
            self._draw_border(frame, COLOR_GREEN)
        self._board_sprite.pixels = frame

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[8:10, ORIGIN_X - 2 : ORIGIN_X + (BOARD_SIZE * CELL_SIZE) + 2] = color
        frame[62:64, ORIGIN_X - 2 : ORIGIN_X + (BOARD_SIZE * CELL_SIZE) + 2] = color
        frame[8:64, ORIGIN_X - 2 : ORIGIN_X] = color
        frame[8:64, ORIGIN_X + (BOARD_SIZE * CELL_SIZE) : ORIGIN_X + (BOARD_SIZE * CELL_SIZE) + 2] = color

    def _render_budget(self, frame: np.ndarray, *, moves_used: int, move_budget: int) -> None:
        frame[0:8, :] = COLOR_WHITE
        max_per_row = 21
        for idx in range(move_budget):
            row = idx // max_per_row
            col = idx % max_per_row
            x0 = 1 + (col * 3)
            y0 = row * 3
            color = COLOR_DARK_GRAY if idx < moves_used else COLOR_YELLOW
            frame[y0 : y0 + 2, x0 : x0 + 2] = color

    def _draw_pawn(self, frame: np.ndarray, cell: tuple[int, int], *, kind: str, selected: bool) -> None:
        px0, py0, _, _ = _cell_bounds(cell)
        if selected:
            for dx, dy in ((0, 0), (2, 0), (4, 0), (0, 2), (4, 2), (0, 4), (2, 4), (4, 4)):
                frame[py0 + dy, px0 + dx] = COLOR_WHITE
        body_outer = COLOR_PINK if kind == "pink" else COLOR_LIGHT_BLUE
        body_center = COLOR_MAGENTA if kind == "pink" else COLOR_BLUE
        frame[py0 + 1 : py0 + 4, px0 + 1 : px0 + 4] = body_outer
        frame[py0 + 2, px0 + 2] = body_center
        if kind == "blue":
            frame[py0 + 1, px0 + 2] = COLOR_YELLOW

    def _selected_position(self) -> tuple[int, int]:
        return self._pink_pos if self._selected == "pink" else self._blue_pos

    def _other_position(self) -> tuple[int, int]:
        return self._blue_pos if self._selected == "pink" else self._pink_pos

    def _safe_cells(self) -> set[tuple[int, int]]:
        raw = self.current_level.get_data("safe_cells") or []
        return {tuple(cell) for cell in raw}

    def _pink_home(self) -> tuple[int, int]:
        return tuple(self.current_level.get_data("pink_home"))

    def _cell_type(self, cell: tuple[int, int]) -> str:
        x, y = cell
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return "void"
        spec = LEVEL_SPECS[int(self.current_level.get_data("level_index") or self.level_index)]
        if not (spec.arena_min_x <= x <= spec.arena_max_x and spec.arena_min_y <= y <= spec.arena_max_y):
            return "void"
        if cell in self._safe_cells():
            return "safe"
        return "hazard"

    def _set_selected_position(self, cell: tuple[int, int]) -> None:
        if self._selected == "pink":
            self._pink_pos = cell
        else:
            self._blue_pos = cell

    def _apply_click(self, x: int, y: int) -> None:
        for name, cell in (("pink", self._pink_pos), ("blue", self._blue_pos)):
            px0, py0, px1, py1 = _cell_bounds(cell)
            if px0 <= x < px1 and py0 <= y < py1:
                self._selected = name
                return

    def _attempt_move(self, action_id: int) -> None:
        delta = MOVE_DELTAS.get(action_id)
        if delta is None:
            return
        current = self._selected_position()
        target = (current[0] + delta[0], current[1] + delta[1])
        if target == self._other_position():
            return
        cell_type = self._cell_type(target)
        if cell_type == "void":
            return

        self._set_selected_position(target)
        self._moves_used += 1

        if cell_type == "hazard":
            self._status = STATUS_FAILED_HAZARD
            return
        if self._blue_pos == self._pink_home():
            self._status = STATUS_WON
            return
        if self._moves_used >= self._move_budget:
            self._status = STATUS_FAILED_BUDGET

    def step(self) -> None:
        if self._status == STATUS_WON:
            self.next_level()
            self.complete_action()
            return

        if self._status in {STATUS_FAILED_HAZARD, STATUS_FAILED_BUDGET}:
            self.lose()
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data if isinstance(self.action.data, dict) else {}
            self._apply_click(int(payload.get("x", -1)), int(payload.get("y", -1)))
        else:
            self._attempt_move(action_id)

        self._sync_board()
        if self._status in {STATUS_FAILED_HAZARD, STATUS_FAILED_BUDGET}:
            self.lose()
        self.complete_action()


def cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    return _cell_center(cell)
