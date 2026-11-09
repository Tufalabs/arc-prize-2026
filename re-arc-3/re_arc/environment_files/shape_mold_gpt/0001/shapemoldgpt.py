from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
BOARD_W = 14
BOARD_H = 12
CELL = 4
BOARD_X = 4
BOARD_Y = 4
HUD_Y = 56
MAX_HUD_PIPS = 32

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_SPENT = 3
COLOR_WALL = 4
COLOR_OUTLINE = 5
COLOR_CUTTER = 8
COLOR_BUDGET = 11
COLOR_CLAY = 12
COLOR_ACCENT = 13
COLOR_GOAL = 14

PHASE_PLAYING = "playing"
PHASE_CLEARED = "cleared"
PHASE_FAILED = "failed"

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def _rect(x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}


def _to_i8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.int8:
        return frame
    return frame.astype(np.int8, copy=False)


def _screen_sprite() -> Sprite:
    return Sprite(
        pixels=np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_BG, dtype=np.int8),
        name="screen",
        x=0,
        y=0,
        layer=0,
        visible=True,
        collidable=False,
    )


def _level_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "budget_total": 24,
            "open_cells": frozenset(_rect(1, 4, 12, 5)),
            "start_cells": frozenset({(2, 4), (3, 4), (2, 5), (3, 5)}),
            "goal_cells": frozenset({(10, 4), (10, 5)}),
            "cutters": (("V", 6, 4, 5),),
        },
        {
            "budget_total": 30,
            "open_cells": frozenset(_rect(1, 6, 7, 8) | _rect(7, 2, 7, 8)),
            "start_cells": frozenset({(2, 6), (3, 6), (2, 7), (3, 7), (2, 8), (3, 8)}),
            "goal_cells": frozenset({(7, 3), (7, 4)}),
            "cutters": (("V", 5, 6, 8), ("H", 6, 7, 7)),
        },
        {
            "budget_total": 36,
            "open_cells": frozenset(_rect(1, 7, 8, 9) | _rect(7, 4, 8, 9) | _rect(7, 4, 12, 8)),
            "start_cells": frozenset({(3, 7), (2, 8), (3, 8), (4, 8), (2, 9), (3, 9)}),
            "goal_cells": frozenset({(10, 4), (11, 4), (10, 5), (11, 5)}),
            "cutters": (("V", 5, 8, 8), ("H", 7, 8, 8)),
        },
        {
            "budget_total": 42,
            "open_cells": frozenset(_rect(1, 7, 7, 10) | _rect(6, 4, 7, 10) | _rect(6, 4, 11, 6)),
            "start_cells": frozenset({(3, 7), (2, 8), (3, 8), (4, 8), (2, 9), (3, 9), (2, 10)}),
            "goal_cells": frozenset({(8, 4), (9, 4), (8, 5), (9, 5)}),
            "cutters": (("V", 5, 8, 8), ("H", 7, 7, 7), ("V", 9, 6, 6)),
        },
    )


def _make_levels(specs: tuple[dict[str, object], ...]) -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(specs):
        levels.append(
            Level(
                name=f"Level {idx + 1}",
                grid_size=(GRID_WIDTH, GRID_HEIGHT),
                sprites=[_screen_sprite()],
                data={
                    "budget_total": spec["budget_total"],
                    "open_cells": tuple(sorted(spec["open_cells"])),
                    "start_cells": tuple(sorted(spec["start_cells"])),
                    "goal_cells": tuple(sorted(spec["goal_cells"])),
                    "cutters": tuple(spec["cutters"]),
                },
            )
        )
    return levels


class ShapeMoldGpt(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._specs = _level_specs()
        self._clay_cells: frozenset[tuple[int, int]] = frozenset()
        self._open_cells: frozenset[tuple[int, int]] = frozenset()
        self._goal_cells: frozenset[tuple[int, int]] = frozenset()
        self._cutters: tuple[tuple[str, int, int, int], ...] = ()
        self._budget_total = 0
        self._remaining_budget = 0
        self._phase = PHASE_PLAYING
        self._used_cutters: set[tuple[str, int, int, int]] = set()
        self._screen: Sprite | None = None
        super().__init__(
            game_id="shape_mold_gpt-0001",
            levels=_make_levels(self._specs),
            camera=Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG),
            win_score=4,
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._budget_total = int(level.get_data("budget_total"))
        self._remaining_budget = self._budget_total
        self._open_cells = frozenset(tuple(cell) for cell in level.get_data("open_cells"))
        self._clay_cells = frozenset(tuple(cell) for cell in level.get_data("start_cells"))
        self._goal_cells = frozenset(tuple(cell) for cell in level.get_data("goal_cells"))
        self._cutters = tuple(tuple(item) for item in level.get_data("cutters"))
        self._phase = PHASE_PLAYING
        self._used_cutters = set()
        screens = level.get_sprites_by_name("screen")
        self._screen = screens[0] if screens else None
        self._render()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.ACTION6:
            self._render()
            self.complete_action()
            return

        if self._phase == PHASE_CLEARED:
            if self.is_last_level():
                self.next_level()
            else:
                self.next_level()
            self._render()
            self.complete_action()
            return

        if self._phase == PHASE_FAILED:
            self.lose()
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            self.level_reset()
            self.complete_action()
            return

        delta = MOVE_DELTAS.get(action)
        if delta is None:
            self._render()
            self.complete_action()
            return

        self._remaining_budget -= 1
        self._attempt_move(*delta)

        if self._clay_cells == self._goal_cells:
            self._phase = PHASE_CLEARED
        elif self._remaining_budget <= 0 or not self._clay_cells:
            self._phase = PHASE_FAILED
            self._render()
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _attempt_move(self, dx: int, dy: int) -> None:
        cut_cells, used_now = self._cut_cells(dx, dy)
        moved_cells = frozenset((x + dx, y + dy) for (x, y) in self._clay_cells if (x, y) not in cut_cells)
        if any(cell not in self._open_cells for cell in moved_cells):
            return
        self._clay_cells = moved_cells
        self._used_cutters.update(used_now)

    def _cut_cells(self, dx: int, dy: int) -> tuple[set[tuple[int, int]], set[tuple[str, int, int, int]]]:
        cut: set[tuple[int, int]] = set()
        used_now: set[tuple[str, int, int, int]] = set()
        for x, y in self._clay_cells:
            for cutter in self._cutters:
                kind, b, a0, a1 = cutter
                if cutter in self._used_cutters:
                    continue
                if kind == "V":
                    if dx == 1 and x == b - 1 and a0 <= y <= a1:
                        cut.add((x, y))
                        used_now.add(cutter)
                    elif dx == -1 and x == b and a0 <= y <= a1:
                        cut.add((x, y))
                        used_now.add(cutter)
                elif kind == "H":
                    if dy == 1 and y == b - 1 and a0 <= x <= a1:
                        cut.add((x, y))
                        used_now.add(cutter)
                    elif dy == -1 and y == b and a0 <= x <= a1:
                        cut.add((x, y))
                        used_now.add(cutter)
        return cut, used_now

    def _render(self) -> None:
        if self._screen is None:
            return

        frame = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_BG, dtype=np.int8)

        self._draw_board(frame)
        self._draw_goal(frame)
        self._draw_clay(frame)
        self._draw_cutters(frame)
        self._draw_budget(frame)

        if self._phase == PHASE_CLEARED:
            self._draw_screen_border(frame, COLOR_GOAL)
        elif self._phase == PHASE_FAILED:
            self._draw_screen_border(frame, COLOR_ACCENT)

        self._screen.pixels = _to_i8(frame)

    def _draw_board(self, frame: np.ndarray) -> None:
        bx0 = BOARD_X - 1
        by0 = BOARD_Y - 1
        bx1 = BOARD_X + BOARD_W * CELL
        by1 = BOARD_Y + BOARD_H * CELL
        frame[by0 : by1 + 1, bx0] = COLOR_OUTLINE
        frame[by0 : by1 + 1, bx1] = COLOR_OUTLINE
        frame[by0, bx0 : bx1 + 1] = COLOR_OUTLINE
        frame[by1, bx0 : bx1 + 1] = COLOR_OUTLINE

        for y in range(BOARD_H):
            for x in range(BOARD_W):
                color = COLOR_FLOOR if (x, y) in self._open_cells else COLOR_WALL
                px = BOARD_X + x * CELL
                py = BOARD_Y + y * CELL
                frame[py : py + CELL, px : px + CELL] = color

    def _draw_goal(self, frame: np.ndarray) -> None:
        goal_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        for x, y in self._goal_cells:
            px = BOARD_X + x * CELL
            py = BOARD_Y + y * CELL
            goal_mask[py : py + CELL, px : px + CELL] = True

        for py, px in np.argwhere(goal_mask):
            if (
                py == 0
                or px == 0
                or py == GRID_HEIGHT - 1
                or px == GRID_WIDTH - 1
                or not goal_mask[py - 1, px]
                or not goal_mask[py + 1, px]
                or not goal_mask[py, px - 1]
                or not goal_mask[py, px + 1]
            ):
                frame[py, px] = COLOR_GOAL

    def _draw_clay(self, frame: np.ndarray) -> None:
        if not self._clay_cells:
            return

        clay_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        for x, y in self._clay_cells:
            px = BOARD_X + x * CELL
            py = BOARD_Y + y * CELL
            clay_mask[py : py + CELL, px : px + CELL] = True
            frame[py : py + CELL, px : px + CELL] = COLOR_CLAY

        for py, px in np.argwhere(clay_mask):
            if (
                py == 0
                or px == 0
                or py == GRID_HEIGHT - 1
                or px == GRID_WIDTH - 1
                or not clay_mask[py - 1, px]
                or not clay_mask[py + 1, px]
                or not clay_mask[py, px - 1]
                or not clay_mask[py, px + 1]
            ):
                frame[py, px] = COLOR_ACCENT

        for x, y in self._clay_cells:
            if (x - 1, y) not in self._clay_cells and (x, y - 1) not in self._clay_cells:
                frame[BOARD_Y + y * CELL, BOARD_X + x * CELL] = COLOR_BUDGET

    def _draw_cutters(self, frame: np.ndarray) -> None:
        for cutter in self._cutters:
            kind, b, a0, a1 = cutter
            if kind == "V":
                px = BOARD_X + b * CELL
                py0 = BOARD_Y + a0 * CELL
                py1 = BOARD_Y + (a1 + 1) * CELL
                for py in range(py0, py1):
                    frame[py, px - 1] = COLOR_CUTTER if py % 2 == 0 else COLOR_ACCENT
                    frame[py, px] = COLOR_ACCENT if py % 2 == 0 else COLOR_CUTTER
            else:
                py = BOARD_Y + b * CELL
                px0 = BOARD_X + a0 * CELL
                px1 = BOARD_X + (a1 + 1) * CELL
                for px in range(px0, px1):
                    frame[py - 1, px] = COLOR_CUTTER if px % 2 == 0 else COLOR_ACCENT
                    frame[py, px] = COLOR_ACCENT if px % 2 == 0 else COLOR_CUTTER

    def _draw_budget(self, frame: np.ndarray) -> None:
        for i in range(MAX_HUD_PIPS):
            x = (i % 16) * CELL
            y = HUD_Y + (i // 16) * CELL
            color = COLOR_BG
            if i < self._budget_total:
                color = COLOR_BUDGET if i < self._remaining_budget else COLOR_SPENT
            frame[y : y + CELL, x : x + CELL] = color

    def _draw_screen_border(self, frame: np.ndarray, color: int) -> None:
        frame[0, :] = color
        frame[-1, :] = color
        frame[:, 0] = color
        frame[:, -1] = color
