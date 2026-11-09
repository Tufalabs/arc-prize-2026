from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
BOARD_CELLS = 21
CELL_SIZE = 3
VIEW_SIZE = 64

COLOR_WHITE = 0
COLOR_AXIS_FIXED = 3
COLOR_BACKGROUND = 4
COLOR_INVALID = 8
COLOR_BLUE = 9
COLOR_AXIS = 10
COLOR_TARGET = 11
COLOR_SOLVED = 14
COLOR_GHOST = 15

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

SHAPES = {
    "L3": ((0, 0), (0, 1), (1, 1)),
    "Q4": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "I3": ((0, 0), (1, 0), (2, 0)),
    "I2": ((0, 0), (1, 0)),
    "T4": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "Z4": ((0, 0), (1, 0), (1, 1), (2, 1)),
}


@dataclass(frozen=True)
class ShapeSpec:
    id: str
    kind: str
    anchor: tuple[int, int]


@dataclass(frozen=True)
class AxisSpec:
    id: str
    kind: str
    position: int
    movable: bool


@dataclass(frozen=True)
class LevelSpec:
    name: str
    shapes: tuple[ShapeSpec, ...]
    axes: tuple[AxisSpec, ...]
    cycle: tuple[str, ...]
    selected: str
    targets: frozenset[tuple[int, int]]
    step_budget: int


LEVEL_SPECS = (
    LevelSpec(
        "Slide the silhouette",
        (ShapeSpec("A", "L3", (7, 8)),),
        (),
        ("A",),
        "A",
        frozenset({(10, 9), (10, 10), (11, 10)}),
        24,
    ),
    LevelSpec(
        "Fixed vertical echo",
        (ShapeSpec("A", "L3", (5, 8)),),
        (AxisSpec("V", "V", 10, False),),
        ("A",),
        "A",
        frozenset({(7, 10), (7, 11), (8, 11), (14, 10), (14, 11), (13, 11)}),
        24,
    ),
    LevelSpec(
        "Move the mirror",
        (ShapeSpec("A", "Q4", (5, 8)),),
        (AxisSpec("V", "V", 9, True),),
        ("A", "V"),
        "A",
        frozenset({(7, 9), (8, 9), (7, 10), (8, 10), (15, 9), (16, 9), (15, 10), (16, 10)}),
        36,
    ),
    LevelSpec(
        "Two shapes share one mirror",
        (ShapeSpec("A", "L3", (3, 4)), ShapeSpec("B", "I3", (9, 15))),
        (AxisSpec("V", "V", 10, False),),
        ("A", "B"),
        "A",
        frozenset(
            {(5, 6), (5, 7), (6, 7), (16, 6), (16, 7), (15, 7), (6, 13), (7, 13), (8, 13), (15, 13), (14, 13), (13, 13)}
        ),
        60,
    ),
    LevelSpec(
        "Horizontal echo",
        (ShapeSpec("A", "L3", (3, 4)), ShapeSpec("B", "I3", (12, 7))),
        (AxisSpec("H", "H", 8, True),),
        ("A", "B", "H"),
        "A",
        frozenset(
            {(5, 6), (5, 7), (6, 7), (5, 15), (5, 14), (6, 14), (12, 5), (13, 5), (14, 5), (12, 16), (13, 16), (14, 16)}
        ),
        60,
    ),
    LevelSpec(
        "The diagonal ghost",
        (ShapeSpec("A", "L3", (2, 10)),),
        (AxisSpec("V", "V", 10, False), AxisSpec("H", "H", 10, False)),
        ("A",),
        "A",
        frozenset(
            {(6, 6), (6, 7), (7, 7), (15, 6), (15, 7), (14, 7), (6, 15), (6, 14), (7, 14), (15, 15), (15, 14), (14, 14)}
        ),
        48,
    ),
    LevelSpec(
        "Shared movable axes",
        (ShapeSpec("A", "L3", (4, 7)), ShapeSpec("B", "I2", (15, 9))),
        (AxisSpec("V", "V", 9, True), AxisSpec("H", "H", 12, True)),
        ("A", "B", "V", "H"),
        "A",
        frozenset(
            {
                (6, 4),
                (6, 5),
                (7, 5),
                (17, 4),
                (17, 5),
                (16, 5),
                (6, 15),
                (6, 14),
                (7, 14),
                (17, 15),
                (17, 14),
                (16, 14),
                (13, 6),
                (14, 6),
                (10, 6),
                (9, 6),
                (13, 13),
                (14, 13),
                (10, 13),
                (9, 13),
            }
        ),
        108,
    ),
    LevelSpec(
        "Off-center dependency network",
        (ShapeSpec("A", "L3", (7, 8)), ShapeSpec("B", "T4", (16, 1)), ShapeSpec("C", "Z4", (5, 17))),
        (AxisSpec("V", "V", 13, True), AxisSpec("H", "H", 8, True)),
        ("A", "B", "C", "V", "H"),
        "A",
        frozenset(
            {
                (4, 4),
                (4, 5),
                (5, 5),
                (15, 4),
                (15, 5),
                (14, 5),
                (4, 19),
                (4, 18),
                (5, 18),
                (15, 19),
                (15, 18),
                (14, 18),
                (12, 7),
                (13, 7),
                (14, 7),
                (13, 8),
                (7, 7),
                (6, 7),
                (5, 7),
                (6, 8),
                (12, 16),
                (13, 16),
                (14, 16),
                (13, 15),
                (7, 16),
                (6, 16),
                (5, 16),
                (6, 15),
                (2, 13),
                (3, 13),
                (3, 14),
                (4, 14),
                (17, 13),
                (16, 13),
                (16, 14),
                (15, 14),
                (2, 10),
                (3, 10),
                (3, 9),
                (4, 9),
                (17, 10),
                (16, 10),
                (16, 9),
                (15, 9),
            }
        ),
        210,
    ),
)


class AxisReflectHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: AxisReflect | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[63, :] = COLOR_AXIS_FIXED
        filled = max(0, min(VIEW_SIZE, int(VIEW_SIZE * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[63, :filled] = COLOR_AXIS
        return frame


def _empty_frame() -> np.ndarray:
    return np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)


def _cell_pixels(cell: tuple[int, int]) -> tuple[slice, slice]:
    x, y = cell
    return slice(y * CELL_SIZE, y * CELL_SIZE + CELL_SIZE), slice(x * CELL_SIZE, x * CELL_SIZE + CELL_SIZE)


def _shape_cells(shape: dict[str, object]) -> set[tuple[int, int]]:
    ax, ay = shape["anchor"]
    return {(ax + dx, ay + dy) for dx, dy in SHAPES[str(shape["kind"])]}


def _axis_reflect(axis: dict[str, object], cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    pos = int(axis["position"])
    if axis["kind"] == "V":
        return 2 * pos + 1 - x, y
    return x, 2 * pos + 1 - y


def _in_board(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < BOARD_CELLS and 0 <= y < BOARD_CELLS


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        _empty_frame(), name="board", layer=0, collidable=False, tags=["board", "sys_click", "sys_every_pixel"]
    )
    return Level(grid_size=(VIEW_SIZE, VIEW_SIZE), sprites=[board], name=spec.name, data={"spec": spec})


class AxisReflect(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = AxisReflectHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "axis_reflect",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, _level: Level) -> None:
        spec = self.current_level.get_data("spec")
        self.board = self.current_level.get_sprites_by_name("board")[0]
        self.shapes = {
            item.id: {"id": item.id, "kind": item.kind, "anchor": tuple(item.anchor)} for item in spec.shapes
        }
        self.axes = {
            item.id: {"id": item.id, "kind": item.kind, "position": item.position, "movable": item.movable}
            for item in spec.axes
        }
        self.cycle = list(spec.cycle)
        self.selected_id = spec.selected
        self.targets = set(spec.targets)
        self.step_budget = int(spec.step_budget)
        self.remaining_steps = self.step_budget
        self.flash_invalid = False
        self.flash_solved = False
        self._sync_board()

    def _real_cells_by_shape(self) -> dict[str, set[tuple[int, int]]]:
        return {shape_id: _shape_cells(shape) for shape_id, shape in self.shapes.items()}

    def _real_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for shape in self.shapes.values():
            cells.update(_shape_cells(shape))
        return cells

    def _coverage(self) -> set[tuple[int, int]]:
        real = self._real_cells()
        vertical = [axis for axis in self.axes.values() if axis["kind"] == "V"]
        horizontal = [axis for axis in self.axes.values() if axis["kind"] == "H"]
        coverage = set(real)
        if vertical:
            coverage.update(cell for source in real if _in_board(cell := _axis_reflect(vertical[0], source)))
        if horizontal:
            coverage.update(cell for source in real if _in_board(cell := _axis_reflect(horizontal[0], source)))
        if vertical and horizontal:
            coverage.update(
                cell
                for source in real
                if _in_board(cell := _axis_reflect(vertical[0], _axis_reflect(horizontal[0], source)))
            )
        return coverage

    def _solved(self) -> bool:
        return self.targets.issubset(self._coverage())

    def _selected_cycle_index(self) -> int:
        return self.cycle.index(self.selected_id)

    def _select_next(self) -> None:
        self.selected_id = self.cycle[(self._selected_cycle_index() + 1) % len(self.cycle)]

    def _select_at(self, cell: tuple[int, int]) -> None:
        real_by_shape = self._real_cells_by_shape()
        for item_id in self.cycle:
            if item_id in real_by_shape and cell in real_by_shape[item_id]:
                self.selected_id = item_id
                return

        x, y = cell
        for item_id in self.cycle:
            axis = self.axes.get(item_id)
            if axis is None or not axis["movable"]:
                continue
            pos = int(axis["position"])
            if axis["kind"] == "V" and x in {pos, pos + 1}:
                self.selected_id = item_id
                return
            if axis["kind"] == "H" and y in {pos, pos + 1}:
                self.selected_id = item_id
                return

    def _try_move_shape(self, shape_id: str, dx: int, dy: int) -> bool:
        shape = self.shapes[shape_id]
        ax, ay = shape["anchor"]
        old_cells = _shape_cells(shape)
        candidate = dict(shape)
        candidate["anchor"] = (ax + dx, ay + dy)
        new_cells = _shape_cells(candidate)
        if any(not _in_board(cell) for cell in new_cells):
            return False
        occupied = self._real_cells() - old_cells
        if occupied & new_cells:
            return False
        shape["anchor"] = candidate["anchor"]
        return True

    def _try_move_axis(self, axis_id: str, action: int) -> bool:
        axis = self.axes[axis_id]
        pos = int(axis["position"])
        if axis["kind"] == "V":
            if action == ACTION_LEFT:
                pos -= 1
            elif action == ACTION_RIGHT:
                pos += 1
            else:
                return False
        else:
            if action == ACTION_UP:
                pos -= 1
            elif action == ACTION_DOWN:
                pos += 1
            else:
                return False
        if not 0 <= pos <= BOARD_CELLS - 2:
            return False
        axis["position"] = pos
        return True

    def _spend_step(self) -> None:
        self.remaining_steps -= 1

    def _draw_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        ys, xs = _cell_pixels(cell)
        frame[ys, xs] = color

    def _draw_selected_shape(self, frame: np.ndarray, cells: set[tuple[int, int]]) -> None:
        for x, y in cells:
            px = x * CELL_SIZE
            py = y * CELL_SIZE
            if (x, y - 1) not in cells:
                frame[py, px : px + CELL_SIZE] = COLOR_WHITE
            if (x, y + 1) not in cells:
                frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = COLOR_WHITE
            if (x - 1, y) not in cells:
                frame[py : py + CELL_SIZE, px] = COLOR_WHITE
            if (x + 1, y) not in cells:
                frame[py : py + CELL_SIZE, px + CELL_SIZE - 1] = COLOR_WHITE

    def _draw_axes(self, frame: np.ndarray) -> None:
        for axis in self.axes.values():
            color = COLOR_AXIS if axis["movable"] else COLOR_AXIS_FIXED
            handle = COLOR_WHITE if axis["id"] == self.selected_id and axis["movable"] else color
            pos = int(axis["position"])
            if axis["kind"] == "V":
                x = pos * CELL_SIZE + CELL_SIZE - 1
                frame[: BOARD_CELLS * CELL_SIZE, x] = COLOR_AXIS
                frame[0:2, max(0, x - 1) : min(VIEW_SIZE, x + 2)] = handle
                frame[61:63, max(0, x - 1) : min(VIEW_SIZE, x + 2)] = handle
            else:
                y = pos * CELL_SIZE + CELL_SIZE - 1
                frame[y, : BOARD_CELLS * CELL_SIZE] = COLOR_AXIS
                frame[max(0, y - 1) : min(VIEW_SIZE, y + 2), 0:2] = handle
                frame[max(0, y - 1) : min(VIEW_SIZE, y + 2), 61:63] = handle

    def _sync_board(self) -> None:
        frame = _empty_frame()
        real_by_shape = self._real_cells_by_shape()
        real = set().union(*real_by_shape.values()) if real_by_shape else set()
        coverage = self._coverage()

        for cell in self.targets:
            self._draw_cell(frame, cell, COLOR_TARGET)
        for cell in coverage - real:
            self._draw_cell(frame, cell, COLOR_GHOST)
        for cell in real:
            self._draw_cell(frame, cell, COLOR_BLUE)
        for cell in self.targets:
            x, y = cell
            px = x * CELL_SIZE
            py = y * CELL_SIZE
            frame[py, px] = COLOR_TARGET
            frame[py + 2, px + 2] = COLOR_TARGET

        self._draw_axes(frame)
        if self.flash_solved:
            frame[0, :] = COLOR_SOLVED
            frame[:, 0] = COLOR_SOLVED
            frame[62, :] = COLOR_SOLVED
            frame[:, 62] = COLOR_SOLVED
        elif self.flash_invalid:
            selected_cells = real_by_shape.get(self.selected_id)
            if selected_cells:
                for cell in selected_cells:
                    self._draw_cell(frame, cell, COLOR_INVALID)
            else:
                axis = self.axes.get(self.selected_id)
                if axis is not None:
                    if axis["kind"] == "V":
                        x = int(axis["position"]) * CELL_SIZE + CELL_SIZE - 1
                        frame[:63, x] = COLOR_INVALID
                    else:
                        y = int(axis["position"]) * CELL_SIZE + CELL_SIZE - 1
                        frame[y, :63] = COLOR_INVALID
        else:
            selected_cells = real_by_shape.get(self.selected_id)
            if selected_cells:
                self._draw_selected_shape(frame, selected_cells)

        self.board.pixels = frame

    def step(self) -> None:
        self.flash_invalid = False
        self.flash_solved = False
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._sync_board()
            self.complete_action()
            return

        changed_or_valid_selection = False
        if action in MOVE_DELTAS:
            if self.selected_id in self.shapes:
                dx, dy = MOVE_DELTAS[action]
                changed_or_valid_selection = self._try_move_shape(self.selected_id, dx, dy)
            elif self.selected_id in self.axes:
                changed_or_valid_selection = self._try_move_axis(self.selected_id, action)
        elif action == ACTION_SPACE:
            self._select_next()
            changed_or_valid_selection = True
        elif action == ACTION_CLICK:
            point = self.camera.display_to_grid(int(self.action.data.get("x", 0)), int(self.action.data.get("y", 0)))
            if point is not None:
                x, y = point
                if 0 <= x < BOARD_CELLS * CELL_SIZE and 0 <= y < BOARD_CELLS * CELL_SIZE:
                    previous = self.selected_id
                    self._select_at((x // CELL_SIZE, y // CELL_SIZE))
                    changed_or_valid_selection = self.selected_id != previous

        if not changed_or_valid_selection:
            self.flash_invalid = True
        self._spend_step()
        if self._solved():
            self.flash_solved = True
            self._sync_board()
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self._sync_board()
            self.lose()
            self.complete_action()
            return

        self._sync_board()
        self.complete_action()
