from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_UNDO = 7

CELL_SIZE = 8
VIEW_SIZE = 64

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_BLANK_FRAME = 2
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_PURPLE = 6
COLOR_YELLOW = 11
COLOR_MAROON = 13
COLOR_GREEN = 14
COLOR_EMPTY_BAR = 15

MOVE_DELTAS = {ACTION_UP: (0, -1, "V"), ACTION_DOWN: (0, 1, "V"), ACTION_LEFT: (-1, 0, "H"), ACTION_RIGHT: (1, 0, "H")}
MIXED_COLORS = {
    frozenset((COLOR_RED, COLOR_BLUE)): COLOR_PURPLE,
    frozenset((COLOR_BLUE, COLOR_YELLOW)): COLOR_GREEN,
    frozenset((COLOR_RED, COLOR_YELLOW)): COLOR_MAROON,
}


Paint = tuple[str, int | None, int | None, str | None]
PaintSnakesSnapshot = tuple[list[dict[str, object]], int, dict[tuple[int, int], Paint], set[tuple[int, int]]]


@dataclass(frozen=True)
class SnakeSpec:
    snake_id: str
    color: int
    start: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    open_cells: tuple[tuple[int, int], ...]
    snakes: tuple[SnakeSpec, ...]
    targets: tuple[tuple[int, int, int | None], ...]
    pressure_gates: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    color_changers: tuple[tuple[tuple[int, int], int], ...] = ()
    allow_occupied_targets: bool = False
    color_mixing: bool = False
    step_budget: int = 1


LEVEL_SPECS = (
    LevelSpec(
        "First Stroke",
        (
            (3, 1),
            (4, 1),
            (3, 2),
            (4, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (3, 4),
            (6, 4),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
        ),
        (SnakeSpec("R", COLOR_RED, (1, 3)),),
        (
            (1, 3, COLOR_RED),
            (2, 3, COLOR_RED),
            (3, 3, COLOR_RED),
            (3, 2, COLOR_RED),
            (3, 1, COLOR_RED),
            (4, 3, COLOR_RED),
            (5, 3, COLOR_RED),
            (6, 3, COLOR_RED),
            (6, 4, COLOR_RED),
            (6, 5, COLOR_RED),
            (5, 5, COLOR_RED),
            (4, 5, COLOR_RED),
            (3, 5, COLOR_RED),
            (3, 4, COLOR_RED),
        ),
        allow_occupied_targets=True,
        step_budget=42,
    ),
    LevelSpec(
        "Two Crossings",
        (
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (0, 1),
            (1, 1),
            (3, 1),
            (4, 1),
            (4, 2),
            (0, 3),
            (1, 3),
            (3, 3),
            (4, 3),
        ),
        (SnakeSpec("R", COLOR_RED, (2, 0)), SnakeSpec("B", COLOR_BLUE, (0, 1))),
        (
            (2, 0, COLOR_RED),
            (2, 1, COLOR_RED),
            (2, 2, COLOR_RED),
            (2, 4, COLOR_RED),
            (0, 1, COLOR_BLUE),
            (1, 1, COLOR_BLUE),
            (3, 1, COLOR_BLUE),
            (4, 1, COLOR_BLUE),
            (4, 2, COLOR_BLUE),
            (4, 3, COLOR_BLUE),
            (3, 3, COLOR_BLUE),
            (2, 3, COLOR_BLUE),
            (1, 3, COLOR_BLUE),
        ),
        step_budget=108,
    ),
    LevelSpec(
        "Gate Stand",
        (
            (3, 1),
            (4, 1),
            (6, 1),
            (3, 2),
            (4, 2),
            (6, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
        ),
        (SnakeSpec("R", COLOR_RED, (3, 5)), SnakeSpec("B", COLOR_BLUE, (1, 5)), SnakeSpec("Y", COLOR_YELLOW, (4, 1))),
        (
            (3, 3, COLOR_RED),
            (5, 3, COLOR_RED),
            (6, 3, COLOR_RED),
            (6, 2, COLOR_RED),
            (2, 5, COLOR_BLUE),
            (4, 2, COLOR_YELLOW),
            (4, 4, COLOR_YELLOW),
            (4, 5, COLOR_YELLOW),
        ),
        pressure_gates=(((2, 3), (4, 3)), ((3, 4), (2, 4))),
        step_budget=66,
    ),
    LevelSpec(
        "Circuit Relay",
        (
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (0, 1),
            (1, 1),
            (3, 1),
            (4, 1),
            (4, 2),
            (0, 3),
            (1, 3),
            (4, 3),
            (3, 3),
            (0, 7),
            (1, 7),
            (2, 7),
            (3, 7),
            (4, 7),
            (5, 7),
        ),
        (SnakeSpec("R", COLOR_RED, (2, 0)), SnakeSpec("B", COLOR_BLUE, (0, 1)), SnakeSpec("G", COLOR_GREEN, (0, 7))),
        (
            (2, 0, COLOR_RED),
            (2, 1, COLOR_RED),
            (2, 2, COLOR_RED),
            (2, 4, COLOR_RED),
            (3, 3, COLOR_BLUE),
            (4, 3, COLOR_BLUE),
            (0, 1, COLOR_BLUE),
            (1, 1, COLOR_BLUE),
            (3, 1, COLOR_BLUE),
            (4, 1, COLOR_BLUE),
            (4, 2, COLOR_BLUE),
            (2, 3, None),
            (0, 7, COLOR_GREEN),
            (1, 7, COLOR_GREEN),
            (3, 7, COLOR_GREEN),
            (4, 7, COLOR_GREEN),
        ),
        pressure_gates=(((2, 7), (3, 1)), ((5, 7), (2, 3))),
        step_budget=138,
    ),
    LevelSpec(
        "Three Dependencies",
        (
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (0, 1),
            (1, 1),
            (2, 1),
            (4, 1),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (4, 4),
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            (3, 6),
            (4, 6),
            (5, 6),
        ),
        (SnakeSpec("B", COLOR_BLUE, (0, 1)), SnakeSpec("R", COLOR_RED, (3, 0)), SnakeSpec("G", COLOR_GREEN, (7, 5))),
        (
            (3, 0, COLOR_RED),
            (3, 2, COLOR_RED),
            (3, 4, COLOR_RED),
            (0, 1, COLOR_BLUE),
            (1, 1, COLOR_BLUE),
            (2, 1, COLOR_BLUE),
            (4, 1, COLOR_BLUE),
            (5, 1, COLOR_BLUE),
            (5, 2, COLOR_BLUE),
            (5, 3, COLOR_BLUE),
            (5, 4, COLOR_BLUE),
            (0, 5, COLOR_YELLOW),
            (1, 5, COLOR_YELLOW),
            (5, 5, COLOR_GREEN),
            (6, 5, COLOR_GREEN),
            (3, 3, None),
            (3, 6, None),
        ),
        pressure_gates=(((3, 6), (3, 1)), ((3, 6), (2, 5)), ((3, 3), (3, 5))),
        color_changers=(((4, 5), COLOR_YELLOW),),
        allow_occupied_targets=True,
        step_budget=168,
    ),
    LevelSpec(
        "Mixing Rails",
        (
            (3, 0),
            (5, 0),
            (3, 1),
            (4, 1),
            (5, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (1, 3),
            (2, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (5, 5),
            (6, 5),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 6),
            (5, 7),
        ),
        (
            SnakeSpec("B1", COLOR_BLUE, (3, 0)),
            SnakeSpec("Y1", COLOR_YELLOW, (5, 0)),
            SnakeSpec("R", COLOR_RED, (0, 2)),
            SnakeSpec("G", COLOR_GREEN, (6, 2)),
        ),
        (
            (5, 1, COLOR_YELLOW),
            (1, 2, COLOR_RED),
            (2, 2, COLOR_RED),
            (3, 2, COLOR_BLUE),
            (4, 2, COLOR_BLUE),
            (5, 2, COLOR_YELLOW),
            (1, 3, COLOR_RED),
            (0, 4, COLOR_BLUE),
            (1, 4, COLOR_RED),
            (6, 3, COLOR_GREEN),
            (2, 4, COLOR_BLUE),
            (3, 4, COLOR_BLUE),
            (4, 4, COLOR_BLUE),
            (5, 4, COLOR_YELLOW),
            (6, 4, COLOR_GREEN),
            (1, 5, COLOR_RED),
            (3, 5, COLOR_BLUE),
            (5, 5, COLOR_YELLOW),
            (1, 6, COLOR_GREEN),
            (2, 6, COLOR_GREEN),
            (3, 6, COLOR_GREEN),
            (4, 6, COLOR_GREEN),
            (5, 6, COLOR_GREEN),
            (6, 6, COLOR_GREEN),
            (5, 7, COLOR_YELLOW),
            (4, 3, None),
        ),
        pressure_gates=(((2, 3), (3, 1)), ((4, 3), (5, 3)), ((2, 5), (6, 5))),
        allow_occupied_targets=True,
        step_budget=168,
    ),
    LevelSpec(
        "Wrong Loop",
        (
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (0, 4),
            (1, 4),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 6),
            (7, 5),
            (6, 5),
            (5, 5),
            (4, 5),
            (3, 5),
            (2, 5),
            (1, 5),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 6),
            (4, 7),
        ),
        (SnakeSpec("R", COLOR_RED, (0, 2)), SnakeSpec("B", COLOR_BLUE, (7, 6)), SnakeSpec("Y", COLOR_YELLOW, (4, 0))),
        (
            (0, 2, COLOR_RED),
            (1, 2, COLOR_RED),
            (3, 2, COLOR_RED),
            (5, 2, COLOR_RED),
            (7, 6, COLOR_BLUE),
            (7, 5, COLOR_BLUE),
            (6, 5, COLOR_BLUE),
            (5, 5, COLOR_BLUE),
            (3, 5, COLOR_BLUE),
            (2, 5, COLOR_BLUE),
            (1, 5, COLOR_BLUE),
            (1, 4, COLOR_BLUE),
            (4, 0, COLOR_YELLOW),
            (4, 1, COLOR_YELLOW),
            (4, 2, COLOR_YELLOW),
            (4, 3, COLOR_YELLOW),
            (4, 4, COLOR_YELLOW),
            (4, 5, COLOR_YELLOW),
            (4, 6, COLOR_YELLOW),
        ),
        step_budget=156,
    ),
)

LEVEL_SPECS = (
    LEVEL_SPECS[0],
    LEVEL_SPECS[1],
    LEVEL_SPECS[6],
    LEVEL_SPECS[3],
    LEVEL_SPECS[2],
    LEVEL_SPECS[4],
    LEVEL_SPECS[5],
)


class PaintSnakeHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PaintSnakes | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        filled = int(VIEW_SIZE * game.steps_left / max(1, game.step_budget))
        frame[63, :] = COLOR_EMPTY_BAR
        if filled > 0:
            frame[63, :filled] = COLOR_GREEN
        active = game.snakes[game.active_snake]["color"] if game.snakes else COLOR_WHITE
        frame[0:2, 62:64] = active
        return frame


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


def _blank_paint() -> Paint:
    return ("blank", None, None, None)


def _dry_paint(color: int) -> Paint:
    return ("dry", color, None, None)


def _wet_paint(color: int, owner: int, axis: str) -> Paint:
    return ("wet", color, owner, axis)


def _base_sprite() -> Sprite:
    return Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BLACK, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[_base_sprite()],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "open_cells": spec.open_cells,
            "snakes": spec.snakes,
            "targets": spec.targets,
            "pressure_gates": spec.pressure_gates,
            "color_changers": spec.color_changers,
            "allow_occupied_targets": spec.allow_occupied_targets,
            "color_mixing": spec.color_mixing,
            "step_budget": spec.step_budget,
        },
    )


class PaintSnakes(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = PaintSnakeHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "paint_snakes",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BLACK, COLOR_BLACK, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_UNDO],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.open_cells = {tuple(cell) for cell in level.get_data("open_cells")}
        self.targets = {tuple(target[:2]): target[2] for target in level.get_data("targets")}
        self.pressure_gates = tuple((tuple(pad), tuple(gate)) for pad, gate in (level.get_data("pressure_gates") or ()))
        self.pad_cells = {pad for pad, _gate in self.pressure_gates}
        self.gate_cells = {gate for _pad, gate in self.pressure_gates}
        self._pip_gate_cells = {pad: index + 1 for index, (pad, _gate) in enumerate(self.pressure_gates)}
        self._pip_gate_cells.update({gate: index + 1 for index, (_pad, gate) in enumerate(self.pressure_gates)})
        if (3, 6) in self.pad_cells:
            self._pip_gate_cells[(3, 6)] = 1
        if (3, 1) in self.gate_cells:
            self._pip_gate_cells[(3, 1)] = 1
        if (2, 5) in self.gate_cells:
            self._pip_gate_cells[(2, 5)] = 1
        self._use_pip_gates = bool(self.pressure_gates)
        self.color_changers = {tuple(cell): int(color) for cell, color in (level.get_data("color_changers") or ())}
        self.used_color_changers: set[tuple[int, int]] = set()
        self.allow_occupied_targets = bool(level.get_data("allow_occupied_targets") or False)
        self.color_mixing = bool(level.get_data("color_mixing") or False)
        snake_specs = level.get_data("snakes")
        self.snakes = [{"id": spec.snake_id, "color": spec.color, "pos": tuple(spec.start)} for spec in snake_specs]
        self.active_snake = 0
        self.paint: dict[tuple[int, int], Paint] = {cell: _blank_paint() for cell in self.open_cells}
        self.step_budget = int(level.get_data("step_budget") or 1)
        self.steps_left = self.step_budget
        self.flash_cells: set[tuple[int, int]] = set()
        self.undo_history: list[PaintSnakesSnapshot] = []
        self._sync_board()

    def step(self) -> None:
        if _action_id(self.action.id) == 0:
            self.flash_cells.clear()
            self._sync_board()
            self.complete_action()
            return

        self.flash_cells.clear()
        action_id = _action_id(self.action.id)
        changed = False
        solved = False

        if action_id in MOVE_DELTAS:
            self._push_undo_snapshot()
            changed = self._move_active(action_id)
        elif action_id == ACTION_SPACE:
            self._push_undo_snapshot()
            self.active_snake = (self.active_snake + 1) % len(self.snakes)
            changed = True
        elif action_id == ACTION_UNDO:
            changed = self._undo()

        self.steps_left = max(0, self.steps_left - 1)
        if changed:
            solved = self._is_solved()
        invalid_flash = bool(self.flash_cells)
        self._sync_board()
        if invalid_flash:
            self.flash_cells.clear()
            self._sync_board()

        if solved:
            self.next_level()
            self.complete_action()
        elif self.steps_left <= 0:
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _push_undo_snapshot(self) -> None:
        snakes = [
            {"id": str(snake["id"]), "color": int(snake["color"]), "pos": tuple(snake["pos"])} for snake in self.snakes
        ]
        self.undo_history.append((snakes, int(self.active_snake), dict(self.paint), set(self.used_color_changers)))

    def _undo(self) -> bool:
        if not self.undo_history:
            return False
        snakes, active_snake, paint, used_color_changers = self.undo_history.pop()
        self.snakes = [
            {"id": str(snake["id"]), "color": int(snake["color"]), "pos": tuple(snake["pos"])} for snake in snakes
        ]
        self.active_snake = active_snake
        self.paint = dict(paint)
        self.used_color_changers = set(used_color_changers)
        self.flash_cells.clear()
        return True

    def _move_active(self, action_id: int) -> bool:
        dx, dy, axis = MOVE_DELTAS[action_id]
        snake = self.snakes[self.active_snake]
        from_cell = snake["pos"]
        to_cell = (from_cell[0] + dx, from_cell[1] + dy)
        if not self._can_enter(to_cell, axis):
            self.flash_cells.update({from_cell, to_cell})
            return False

        to_paint = self.paint.get(to_cell, _blank_paint())
        if to_paint[0] == "wet":
            self.paint[to_cell] = _dry_paint(int(to_paint[1]))

        self.paint[from_cell] = _wet_paint(self._paint_color(from_cell, int(snake["color"])), self.active_snake, axis)
        snake["pos"] = to_cell
        if to_cell in self.color_changers:
            snake["color"] = self.color_changers[to_cell]
            self.used_color_changers.add(to_cell)
        return True

    def _paint_color(self, cell: tuple[int, int], color: int) -> int:
        if not self.color_mixing:
            return color
        _kind, existing_color, _owner, _axis = self.paint.get(cell, _blank_paint())
        if existing_color is None or existing_color == color:
            return color
        return MIXED_COLORS.get(frozenset((int(existing_color), color)), color)

    def _can_enter(self, cell: tuple[int, int], axis: str) -> bool:
        if cell not in self.open_cells:
            return False
        if cell in self.gate_cells and not self._gate_is_open(cell):
            return False
        if any(snake["pos"] == cell for snake in self.snakes):
            return False
        kind, color, owner, wet_axis = self.paint.get(cell, _blank_paint())
        if kind != "wet":
            return True
        if owner == self.active_snake:
            return False
        if self.color_mixing and color != self.snakes[self.active_snake]["color"]:
            return True
        return axis != wet_axis

    def _gate_is_open(self, gate_cell: tuple[int, int]) -> bool:
        return any(
            gate == gate_cell and any(snake["pos"] == pad for snake in self.snakes) for pad, gate in self.pressure_gates
        )

    def _is_solved(self) -> bool:
        for cell, desired in self.targets.items():
            kind, color, _owner, _axis = self.paint.get(cell, _blank_paint())
            if desired is None:
                continue
            if any(snake["pos"] == cell and snake["color"] == desired for snake in self.snakes):
                continue
            if kind == "blank" or color != desired:
                return False
        return True

    def _sync_board(self) -> None:
        pixels = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_WALL, dtype=np.int8)
        for y in range(8):
            for x in range(8):
                cell = (x, y)
                x0 = x * CELL_SIZE
                y0 = y * CELL_SIZE
                if cell in self.open_cells:
                    pixels[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_FLOOR
                    self._draw_target(pixels, cell)
                    self._draw_pressure_gate(pixels, cell)
                    self._draw_color_changer(pixels, cell)
                    self._draw_paint(pixels, cell)
        for cell in self.flash_cells:
            if 0 <= cell[0] < 8 and 0 <= cell[1] < 8:
                x0 = cell[0] * CELL_SIZE
                y0 = cell[1] * CELL_SIZE
                pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = COLOR_MAROON
        for index, snake in enumerate(self.snakes):
            self._draw_snake(pixels, tuple(snake["pos"]), int(snake["color"]), index == self.active_snake)
        self.board.pixels[:, :] = pixels

    def _draw_target(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        if cell not in self.targets:
            return
        color = self.targets[cell]
        frame_color = COLOR_BLANK_FRAME if color is None else int(color)
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        pixels[y0, x0 : x0 + CELL_SIZE] = frame_color
        pixels[y0 + CELL_SIZE - 1, x0 : x0 + CELL_SIZE] = frame_color
        pixels[y0 : y0 + CELL_SIZE, x0] = frame_color
        pixels[y0 : y0 + CELL_SIZE, x0 + CELL_SIZE - 1] = frame_color

    def _draw_pressure_gate(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        if self._use_pip_gates and (cell in self.pad_cells or cell in self.gate_cells):
            self._draw_pip_pressure_gate(pixels, cell)
            return
        if cell in self.pad_cells:
            pixels[y0 + 2 : y0 + 6, x0 + 2 : x0 + 6] = COLOR_YELLOW
        if cell in self.gate_cells:
            if self._gate_is_open(cell):
                pixels[y0 + 1 : y0 + 7, x0 + 3 : x0 + 5] = COLOR_GREEN
            else:
                pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = COLOR_MAROON

    def _draw_pip_pressure_gate(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        pip_count = self._pip_gate_cells.get(cell, 1)
        if cell in self.pad_cells:
            self._draw_pip_tile(pixels, cell, COLOR_WHITE, COLOR_WHITE, pip_count)
            return

        if self._gate_is_open(cell):
            self._draw_open_pip_gate(pixels, cell)
        else:
            self._draw_pip_tile(pixels, cell, COLOR_MAROON, COLOR_MAROON, pip_count)

    def _draw_pip_tile(
        self, pixels: np.ndarray, cell: tuple[int, int], background_color: int, pip_color: int, pip_count: int
    ) -> None:
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = background_color
        pixels[y0 + 2 : y0 + 6, x0 + 2 : x0 + 6] = COLOR_FLOOR
        self._draw_gate_pips(pixels, cell, pip_count, pip_color)

    def _draw_open_pip_gate(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = COLOR_WHITE

    def _draw_gate_pips(self, pixels: np.ndarray, cell: tuple[int, int], pip_count: int, pip_color: int) -> None:
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        pip_positions = {
            1: ((3, 3), (4, 3), (3, 4), (4, 4)),
            2: ((1, 2), (2, 2), (1, 3), (2, 3), (5, 2), (6, 2), (5, 3), (6, 3)),
            3: ((1, 2), (2, 2), (1, 3), (2, 3), (5, 2), (6, 2), (5, 3), (6, 3), (3, 5), (4, 5)),
        }.get(pip_count, ((3, 2), (4, 2), (1, 3), (2, 3), (5, 3), (6, 3), (3, 5), (4, 5)))
        for dx, dy in pip_positions:
            pixels[y0 + dy, x0 + dx] = pip_color

    def _draw_color_changer(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        if cell not in self.color_changers:
            return
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        if cell not in self.used_color_changers:
            pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = self.color_changers[cell]
            return
        pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = COLOR_WHITE
        pixels[y0 + 2 : y0 + 6, x0 + 2 : x0 + 6] = self.color_changers[cell]

    def _draw_paint(self, pixels: np.ndarray, cell: tuple[int, int]) -> None:
        kind, color, _owner, axis = self.paint.get(cell, _blank_paint())
        if kind == "blank":
            return
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = int(color)
        if kind == "wet":
            if axis == "H":
                pixels[y0 + 3 : y0 + 5, x0 + 1 : x0 + 7] = COLOR_BLACK
            else:
                pixels[y0 + 1 : y0 + 7, x0 + 3 : x0 + 5] = COLOR_BLACK

    def _draw_snake(self, pixels: np.ndarray, cell: tuple[int, int], color: int, active: bool) -> None:
        x0 = cell[0] * CELL_SIZE
        y0 = cell[1] * CELL_SIZE
        outline = COLOR_WHITE if active else COLOR_BLACK
        pixels[y0 + 1 : y0 + 7, x0 + 1 : x0 + 7] = outline
        pixels[y0 + 2 : y0 + 6, x0 + 2 : x0 + 6] = color

    def _get_hidden_state(self) -> np.ndarray:
        paint_values = []
        for y in range(8):
            for x in range(8):
                kind, color, owner, axis = self.paint.get((x, y), _blank_paint())
                kind_value = {"blank": 0, "dry": 1, "wet": 2}[kind]
                axis_value = 1 if axis == "H" else 2 if axis == "V" else 0
                paint_values.extend([kind_value, int(color or 0), int(owner or 0), axis_value])
        snake_values = [value for snake in self.snakes for value in (*snake["pos"], snake["color"])]
        return np.asarray(
            [self.level_index, self.active_snake, self.steps_left, *snake_values, *paint_values], dtype=np.int16
        )

    def _get_valid_actions(self) -> list[ActionInput]:
        return [ActionInput(id=GameAction.from_id(action_id)) for action_id in (1, 2, 3, 4, 5)]
