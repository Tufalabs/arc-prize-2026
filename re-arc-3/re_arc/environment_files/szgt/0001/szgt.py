from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BOARD_WIDTH = 16
BOARD_HEIGHT = 15
CELL_SIZE = 4
HUD_HEIGHT = 4
FRAME_WIDTH = 64
FRAME_HEIGHT = 64

COLOR_FLOOR = 0
COLOR_HUD_EMPTY = 3
COLOR_WALL = 4
COLOR_EDGE = 5
COLOR_PLATE_B_BORDER = 6
COLOR_PLATE_B_FILL = 7
COLOR_AVATAR_BASE = 9
COLOR_AVATAR_HIGHLIGHT = 10
COLOR_PLATE_A_BORDER = 11
COLOR_PLATE_A_FILL = 12
COLOR_GATE_FRAME = 13
COLOR_HUD_FILL = 14
COLOR_EXIT_FRAME = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
TOGGLE_ACTION = int(GameAction.ACTION5.value)


def _rect_cells(x0: int, y0: int, x1: int, y1: int) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))


def _frame_pixels(fill: int) -> np.ndarray:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH), int(fill), dtype=np.int8)


class PlateSpec:
    def __init__(self, cells: frozenset[tuple[int, int]], border_color: int, fill_color: int) -> None:
        self.cells = cells
        self.border_color = int(border_color)
        self.fill_color = int(fill_color)


class GateSpec:
    def __init__(self, cells: frozenset[tuple[int, int]], plate_index: int, closed_fill_color: int) -> None:
        self.cells = cells
        self.plate_index = int(plate_index)
        self.closed_fill_color = int(closed_fill_color)


class LevelSpec:
    def __init__(
        self,
        *,
        name: str,
        budget: int,
        start_center: tuple[int, int],
        start_large: bool,
        floor_cells: frozenset[tuple[int, int]],
        plates: tuple[PlateSpec, ...],
        gates: tuple[GateSpec, ...],
        exit_cells: frozenset[tuple[int, int]],
    ) -> None:
        self.name = name
        self.budget = int(budget)
        self.start_center = start_center
        self.start_large = bool(start_large)
        self.floor_cells = floor_cells
        self.plates = plates
        self.gates = gates
        self.exit_cells = exit_cells


def _plate(x0: int, y0: int, border_color: int, fill_color: int) -> PlateSpec:
    return PlateSpec(cells=_rect_cells(x0, y0, x0 + 2, y0 + 2), border_color=border_color, fill_color=fill_color)


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1",
        budget=27,
        start_center=(4, 7),
        start_large=True,
        floor_cells=_rect_cells(1, 3, 6, 11) | _rect_cells(7, 7, 9, 7) | _rect_cells(10, 4, 14, 10),
        plates=(),
        gates=(),
        exit_cells=_rect_cells(12, 6, 13, 7),
    ),
    LevelSpec(
        name="Level 2",
        budget=48,
        start_center=(4, 10),
        start_large=True,
        floor_cells=_rect_cells(1, 2, 7, 12) | _rect_cells(9, 7, 11, 7) | _rect_cells(12, 5, 14, 9),
        plates=(_plate(3, 4, COLOR_PLATE_A_BORDER, COLOR_PLATE_A_FILL),),
        gates=(GateSpec(cells=frozenset({(8, 7)}), plate_index=0, closed_fill_color=COLOR_PLATE_A_FILL),),
        exit_cells=_rect_cells(12, 6, 13, 7),
    ),
    LevelSpec(
        name="Level 3",
        budget=81,
        start_center=(3, 9),
        start_large=True,
        floor_cells=(
            _rect_cells(1, 3, 5, 11)
            | _rect_cells(7, 7, 9, 7)
            | _rect_cells(10, 3, 14, 8)
            | _rect_cells(12, 10, 12, 10)
            | _rect_cells(10, 11, 14, 13)
        ),
        plates=(
            _plate(2, 4, COLOR_PLATE_A_BORDER, COLOR_PLATE_A_FILL),
            _plate(11, 4, COLOR_PLATE_B_BORDER, COLOR_PLATE_B_FILL),
        ),
        gates=(
            GateSpec(cells=frozenset({(6, 7)}), plate_index=0, closed_fill_color=COLOR_PLATE_A_FILL),
            GateSpec(cells=frozenset({(12, 9)}), plate_index=1, closed_fill_color=COLOR_PLATE_B_FILL),
        ),
        exit_cells=_rect_cells(12, 12, 13, 13),
    ),
    LevelSpec(
        name="Level 4",
        budget=99,
        start_center=(4, 7),
        start_large=True,
        floor_cells=_rect_cells(1, 2, 6, 8)
        | _rect_cells(8, 5, 9, 5)
        | _rect_cells(10, 2, 14, 8)
        | _rect_cells(1, 10, 6, 13),
        plates=(
            _plate(2, 3, COLOR_PLATE_A_BORDER, COLOR_PLATE_A_FILL),
            _plate(11, 3, COLOR_PLATE_B_BORDER, COLOR_PLATE_B_FILL),
        ),
        gates=(
            GateSpec(cells=frozenset({(7, 5)}), plate_index=0, closed_fill_color=COLOR_PLATE_A_FILL),
            GateSpec(cells=frozenset({(4, 9)}), plate_index=1, closed_fill_color=COLOR_PLATE_B_FILL),
        ),
        exit_cells=_rect_cells(3, 11, 4, 12),
    ),
)


class Szgt(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._level_index = 0
        self._remaining_budget = 0
        self._avatar_cx = 0
        self._avatar_cy = 0
        self._is_large = False
        self._activated_mask = 0
        self._display: Sprite | None = None
        levels = [
            Level(
                name=spec.name,
                grid_size=(FRAME_WIDTH, FRAME_HEIGHT),
                sprites=[Sprite(_frame_pixels(COLOR_WALL), name="display", collidable=False)],
                data={"level_index": idx},
            )
            for idx, spec in enumerate(LEVEL_SPECS)
        ]
        camera = Camera(width=FRAME_WIDTH, height=FRAME_HEIGHT, background=COLOR_WALL)
        super().__init__(
            game_id="szgt",
            levels=levels,
            camera=camera,
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._level_index = int(level.get_data("level_index") or 0)
        spec = LEVEL_SPECS[self._level_index]
        self._remaining_budget = int(spec.budget)
        self._avatar_cx, self._avatar_cy = spec.start_center
        self._is_large = bool(spec.start_large)
        self._activated_mask = 0
        displays = self.current_level.get_sprites_by_name("display")
        self._display = displays[0] if displays else None
        self._render()

    def _current_spec(self) -> LevelSpec:
        return LEVEL_SPECS[self._level_index]

    def _occupied_cells(self, cx: int, cy: int, is_large: bool) -> tuple[tuple[int, int], ...]:
        if not is_large:
            return ((cx, cy),)
        cells: list[tuple[int, int]] = []
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                cells.append((x, y))
        return tuple(cells)

    def _gate_is_open(self, gate_index: int, mask: int | None = None) -> bool:
        resolved_mask = self._activated_mask if mask is None else int(mask)
        return bool(resolved_mask & (1 << LEVEL_SPECS[self._level_index].gates[gate_index].plate_index))

    def _is_passable(self, x: int, y: int, mask: int | None = None) -> bool:
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return False
        spec = self._current_spec()
        if (x, y) in spec.floor_cells or (x, y) in spec.exit_cells:
            return True
        for plate in spec.plates:
            if (x, y) in plate.cells:
                return True
        for gate_index, gate in enumerate(spec.gates):
            if (x, y) not in gate.cells:
                continue
            return self._gate_is_open(gate_index, mask)
        return False

    def _can_occupy(self, cx: int, cy: int, is_large: bool, mask: int | None = None) -> bool:
        for x, y in self._occupied_cells(cx, cy, is_large):
            if not self._is_passable(x, y, mask):
                return False
        return True

    def _update_plates(self) -> None:
        if not self._is_large:
            return
        occupied = frozenset(self._occupied_cells(self._avatar_cx, self._avatar_cy, True))
        for plate_index, plate in enumerate(self._current_spec().plates):
            if occupied == plate.cells:
                self._activated_mask |= 1 << plate_index

    def _draw_cell(self, frame: np.ndarray, x: int, y: int, color: int) -> None:
        px = x * CELL_SIZE
        py = HUD_HEIGHT + y * CELL_SIZE
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = int(color)

    def _draw_plate(self, frame: np.ndarray, plate: PlateSpec, active: bool) -> None:
        xs = [cell[0] for cell in plate.cells]
        ys = [cell[1] for cell in plate.cells]
        center = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
        for cell_x, cell_y in plate.cells:
            px = cell_x * CELL_SIZE
            py = HUD_HEIGHT + cell_y * CELL_SIZE
            frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = int(plate.fill_color)
            frame[py, px : px + CELL_SIZE] = int(plate.border_color)
            frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = int(plate.border_color)
            frame[py : py + CELL_SIZE, px] = int(plate.border_color)
            frame[py : py + CELL_SIZE, px + CELL_SIZE - 1] = int(plate.border_color)
            if active and (cell_x, cell_y) == center:
                frame[py + 1 : py + 3, px + 1 : px + 3] = int(COLOR_HUD_FILL)

    def _draw_gate(
        self, frame: np.ndarray, cells: frozenset[tuple[int, int]], open_fill: int, closed_fill: int, is_open: bool
    ) -> None:
        fill_color = open_fill if is_open else closed_fill
        for cell_x, cell_y in cells:
            px = cell_x * CELL_SIZE
            py = HUD_HEIGHT + cell_y * CELL_SIZE
            frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = int(fill_color)
            frame[py, px : px + CELL_SIZE] = int(COLOR_GATE_FRAME)
            frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = int(COLOR_GATE_FRAME)
            frame[py : py + CELL_SIZE, px] = int(COLOR_GATE_FRAME)
            frame[py : py + CELL_SIZE, px + CELL_SIZE - 1] = int(COLOR_GATE_FRAME)

    def _draw_exit(self, frame: np.ndarray) -> None:
        for cell_x, cell_y in self._current_spec().exit_cells:
            px = cell_x * CELL_SIZE
            py = HUD_HEIGHT + cell_y * CELL_SIZE
            frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = int(COLOR_PLATE_B_FILL)
            frame[py, px : px + CELL_SIZE] = int(COLOR_EXIT_FRAME)
            frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = int(COLOR_EXIT_FRAME)
            frame[py : py + CELL_SIZE, px] = int(COLOR_EXIT_FRAME)
            frame[py : py + CELL_SIZE, px + CELL_SIZE - 1] = int(COLOR_EXIT_FRAME)
            frame[py + 1 : py + 3, px + 1 : px + 3] = int(COLOR_PLATE_B_FILL)

    def _draw_avatar(self, frame: np.ndarray) -> None:
        if self._is_large:
            min_x = (self._avatar_cx - 1) * CELL_SIZE
            min_y = HUD_HEIGHT + (self._avatar_cy - 1) * CELL_SIZE
            frame[min_y : min_y + 12, min_x : min_x + 12] = int(COLOR_AVATAR_HIGHLIGHT)
            frame[min_y, min_x : min_x + 12] = int(COLOR_AVATAR_BASE)
            frame[min_y + 11, min_x : min_x + 12] = int(COLOR_AVATAR_BASE)
            frame[min_y : min_y + 12, min_x] = int(COLOR_AVATAR_BASE)
            frame[min_y : min_y + 12, min_x + 11] = int(COLOR_AVATAR_BASE)
            return

        px = self._avatar_cx * CELL_SIZE
        py = HUD_HEIGHT + self._avatar_cy * CELL_SIZE
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = int(COLOR_AVATAR_BASE)
        frame[py + 1 : py + 3, px + 1 : px + 3] = int(COLOR_AVATAR_HIGHLIGHT)

    def _render_frame(self) -> np.ndarray:
        spec = self._current_spec()
        frame = _frame_pixels(COLOR_WALL)
        frame[0:HUD_HEIGHT, :] = int(COLOR_HUD_EMPTY)
        frame[0, :] = int(COLOR_EDGE)
        frame[HUD_HEIGHT - 1, :] = int(COLOR_EDGE)

        if spec.budget > 0:
            fill_width = (FRAME_WIDTH * max(self._remaining_budget, 0)) // spec.budget
            if fill_width > 0:
                frame[1 : HUD_HEIGHT - 1, 0:fill_width] = int(COLOR_HUD_FILL)

        for cell_x, cell_y in spec.floor_cells:
            self._draw_cell(frame, cell_x, cell_y, COLOR_FLOOR)
        for plate_index, plate in enumerate(spec.plates):
            self._draw_plate(frame, plate, bool(self._activated_mask & (1 << plate_index)))
        self._draw_exit(frame)
        for gate_index, gate in enumerate(spec.gates):
            self._draw_gate(
                frame,
                gate.cells,
                open_fill=COLOR_FLOOR,
                closed_fill=gate.closed_fill_color,
                is_open=self._gate_is_open(gate_index),
            )
        self._draw_avatar(frame)
        return frame

    def _render(self) -> None:
        if self._display is not None:
            self._display.pixels = self._render_frame()

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

        self._remaining_budget = max(0, self._remaining_budget - 1)
        action_id = int(self.action.id.value)

        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            next_cx = self._avatar_cx + dx
            next_cy = self._avatar_cy + dy
            if self._can_occupy(next_cx, next_cy, self._is_large):
                self._avatar_cx = next_cx
                self._avatar_cy = next_cy
        elif action_id == TOGGLE_ACTION:
            if self._is_large:
                self._is_large = False
            elif self._can_occupy(self._avatar_cx, self._avatar_cy, True):
                self._is_large = True

        self._update_plates()

        if (self._avatar_cx, self._avatar_cy) in self._current_spec().exit_cells:
            self._render()
            self.next_level()
            self.complete_action()
            return

        if self._remaining_budget == 0:
            self._render()
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()
