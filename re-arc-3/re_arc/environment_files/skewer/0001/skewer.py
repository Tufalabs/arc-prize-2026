from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
GRID_ORIGIN = 2
CELL_SIZE = 6
GRID_W = 10
PLAY_ROWS = 8

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_SEGMENT = 2
COLOR_HANDLE = 3
COLOR_OBSTACLE = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_RAIL = 10
COLOR_SELECTED = 11
COLOR_SCORING = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

BASELINE_ACTIONS = [3, 11, 10, 7, 11, 9, 9, 45]
STEP_BUDGETS = [count * 6 for count in BASELINE_ACTIONS]


@dataclass(frozen=True)
class SkewerSpec:
    sid: str
    direction: tuple[int, int]
    handle: tuple[int, int]
    length: int
    movable: bool = True
    rail: tuple[tuple[int, int], ...] = ()
    target: tuple[int, ...] = ()
    slots: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class LevelSpec:
    name: str
    skewers: tuple[SkewerSpec, ...]
    selected: str
    order: tuple[str, ...]
    loose: tuple[tuple[int, int, int], ...] = ()
    obstacles: tuple[tuple[int, int], ...] = ()
    step_budget: int = 30


@dataclass
class SkewerState:
    sid: str
    direction: tuple[int, int]
    handle: tuple[int, int]
    length: int
    movable: bool
    rail: set[tuple[int, int]]
    target: tuple[int, ...]
    slots: list[int | None]

    def slot_cell(self, index: int) -> tuple[int, int]:
        dx, dy = self.direction
        hx, hy = self.handle
        return hx + (index + 1) * dx, hy + (index + 1) * dy

    def slot_cells(self) -> list[tuple[int, int]]:
        return [self.slot_cell(index) for index in range(self.length)]

    @property
    def orientation(self) -> str:
        return "h" if self.direction[1] == 0 else "v"


LEVEL_SPECS = (
    LevelSpec(
        name="First Pierce",
        selected="A",
        order=("A",),
        skewers=(
            SkewerSpec("A", (1, 0), (1, 4), 3),
            SkewerSpec("S7", (0, -1), (7, 7), 4, movable=False, target=(COLOR_RED,)),
        ),
        loose=((5, 4, COLOR_RED),),
        step_budget=STEP_BUDGETS[0],
    ),
    LevelSpec(
        name="Pushed Apart",
        selected="A",
        order=("A",),
        skewers=(
            SkewerSpec("A", (1, 0), (1, 4), 3),
            SkewerSpec("S8", (0, -1), (8, 7), 4, movable=False, target=(COLOR_BLUE, COLOR_RED)),
        ),
        loose=((5, 4, COLOR_RED), (6, 4, COLOR_BLUE)),
        step_budget=STEP_BUDGETS[1],
    ),
    LevelSpec(
        name="Sticks Give Back",
        selected="A",
        order=("A",),
        skewers=(
            SkewerSpec("A", (1, 0), (2, 5), 3),
            SkewerSpec("S5", (0, -1), (5, 7), 4, movable=False, target=(COLOR_RED,)),
            SkewerSpec(
                "S8", (0, -1), (8, 7), 4, movable=False, target=(COLOR_BLUE,), slots=(None, COLOR_RED, None, None)
            ),
        ),
        loose=((6, 4, COLOR_BLUE),),
        step_budget=STEP_BUDGETS[2],
    ),
    LevelSpec(
        name="Loaded Comb",
        selected="A",
        order=("A",),
        skewers=(
            SkewerSpec("A", (1, 0), (1, 4), 4, slots=(COLOR_GREEN, None, COLOR_RED, COLOR_BLUE)),
            SkewerSpec("S3", (0, -1), (3, 7), 4, movable=False, target=(COLOR_GREEN,)),
            SkewerSpec("S6", (0, -1), (6, 7), 4, movable=False, target=(COLOR_BLUE,)),
            SkewerSpec("S8", (0, -1), (8, 7), 4, movable=False, target=(COLOR_RED,)),
        ),
        step_budget=STEP_BUDGETS[3],
    ),
    LevelSpec(
        name="Notch and Bumper",
        selected="A",
        order=("A",),
        skewers=(
            SkewerSpec("A", (1, 0), (1, 4), 3),
            SkewerSpec("S8", (0, -1), (8, 7), 4, movable=False, target=(COLOR_BLUE, COLOR_RED)),
        ),
        loose=((5, 4, COLOR_RED), (6, 4, COLOR_BLUE)),
        obstacles=((2, 2), (2, 3), (2, 5), (2, 6)),
        step_budget=STEP_BUDGETS[4],
    ),
    LevelSpec(
        name="Two Handles, One Stack",
        selected="Upper",
        order=("Upper", "Lower"),
        skewers=(
            SkewerSpec("Upper", (1, 0), (1, 5), 3, rail=((1, 5), (2, 5), (3, 5), (4, 5), (5, 5))),
            SkewerSpec("Lower", (1, 0), (1, 6), 3, rail=((1, 6), (2, 6), (3, 6), (4, 6), (5, 6))),
            SkewerSpec("S8", (0, -1), (8, 7), 2, movable=False, target=(COLOR_RED, COLOR_BLUE)),
        ),
        loose=((5, 5, COLOR_BLUE), (5, 6, COLOR_RED)),
        step_budget=STEP_BUDGETS[5],
    ),
    LevelSpec(
        name="Shared Crossing",
        selected="H",
        order=("H", "V"),
        skewers=(
            SkewerSpec("H", (1, 0), (1, 5), 3, rail=((1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 6), (6, 6))),
            SkewerSpec("V", (0, 1), (4, 1), 3, rail=((4, 1), (4, 2)), slots=(None, None, COLOR_RED)),
            SkewerSpec("S8", (0, -1), (8, 7), 2, movable=False, target=(COLOR_BLUE, COLOR_RED)),
        ),
        loose=((7, 6, COLOR_BLUE),),
        step_budget=STEP_BUDGETS[6],
    ),
    LevelSpec(
        name="Short-Rail Weave",
        selected="H",
        order=("H", "V"),
        skewers=(
            SkewerSpec(
                "H",
                (1, 0),
                (2, 6),
                3,
                rail=(
                    (0, 5),
                    (1, 5),
                    (2, 5),
                    (3, 5),
                    (4, 5),
                    (5, 5),
                    (6, 5),
                    (0, 6),
                    (1, 6),
                    (2, 6),
                    (3, 6),
                    (4, 6),
                    (5, 6),
                ),
            ),
            SkewerSpec(
                "V",
                (0, 1),
                (4, 2),
                3,
                rail=((4, 1), (4, 2), (4, 3), (4, 4)),
                slots=(COLOR_PURPLE, COLOR_BLUE, COLOR_GREEN),
            ),
            SkewerSpec("S1", (0, -1), (1, 7), 1, movable=False, target=(COLOR_BLUE,)),
            SkewerSpec("S6", (0, -1), (6, 7), 1, movable=False, target=(COLOR_GREEN,)),
            SkewerSpec("S8", (0, -1), (8, 7), 2, movable=False, target=(COLOR_RED, COLOR_PURPLE)),
        ),
        loose=((7, 6, COLOR_RED),),
        step_budget=STEP_BUDGETS[7],
    ),
)


class Skewer(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(
                sprites=[
                    Sprite(np.full((VIEW_SIZE, VIEW_SIZE), COLOR_WHITE, dtype=np.int8), name="board", collidable=False)
                ],
                grid_size=(VIEW_SIZE, VIEW_SIZE),
                data={"spec": spec},
                name=spec.name,
            )
            for spec in LEVEL_SPECS
        ]
        super().__init__(
            "skewer-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_WHITE, COLOR_WHITE),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.spec: LevelSpec = spec
        self.skewers: dict[str, SkewerState] = {}
        for skewer in spec.skewers:
            slots = list(skewer.slots) if skewer.slots else [None] * skewer.length
            slots.extend([None] * (skewer.length - len(slots)))
            self.skewers[skewer.sid] = SkewerState(
                skewer.sid,
                skewer.direction,
                skewer.handle,
                skewer.length,
                skewer.movable,
                set(skewer.rail),
                skewer.target,
                slots[: skewer.length],
            )
        self.selection_order = list(spec.order)
        self.selected_id = spec.selected
        self.loose = {(x, y): color for x, y, color in spec.loose}
        self.obstacles = set(spec.obstacles)
        self.step_budget = int(spec.step_budget)
        self.remaining_steps = int(spec.step_budget)
        self.flash_cells: set[tuple[int, int]] = set()
        self.invalid_flash = False
        self.board_sprite = self.current_level.get_sprites_by_name("board")[0]
        self._sync_visuals()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        self.flash_cells = set()
        self.invalid_flash = False
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id in MOVE_DELTAS:
            accepted = self._try_move_selected(MOVE_DELTAS[action_id])
            if not accepted:
                self.invalid_flash = True
        elif action_id == ACTION_SPACE:
            self._cycle_selection()
        elif action_id == ACTION_CLICK:
            self._select_clicked_handle()

        self.remaining_steps -= 1
        self._sync_visuals()

        if self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
            return
        self.complete_action()

    def _cycle_selection(self) -> None:
        if not self.selection_order:
            return
        idx = self.selection_order.index(self.selected_id)
        self.selected_id = self.selection_order[(idx + 1) % len(self.selection_order)]

    def _select_clicked_handle(self) -> None:
        data = self.action.data or {}
        cell = _pixel_to_cell(int(data.get("x", 0)), int(data.get("y", 0)))
        if cell is None:
            return
        for sid in self.selection_order:
            skewer = self.skewers[sid]
            if skewer.handle == cell:
                self.selected_id = sid
                return

    def _try_move_selected(self, delta: tuple[int, int]) -> bool:
        before = copy.deepcopy((self.skewers, self.loose))
        active = self.skewers[self.selected_id]
        old_handle = active.handle
        new_handle = (old_handle[0] + delta[0], old_handle[1] + delta[1])
        candidate_slots = [
            (new_handle[0] + (idx + 1) * active.direction[0], new_handle[1] + (idx + 1) * active.direction[1])
            for idx in range(active.length)
        ]

        if not self._candidate_skewer_cells_are_valid(active, new_handle, candidate_slots):
            return False

        pierce: dict[int, tuple[int, int]] = {}
        push_starts: list[tuple[int, int]] = []
        candidate_by_cell = {new_handle: None}
        for idx, cell in enumerate(candidate_slots):
            candidate_by_cell[cell] = idx
        for cell, idx in candidate_by_cell.items():
            if cell not in self.loose:
                continue
            if idx is not None and active.slots[idx] is None:
                pierce[idx] = cell
            else:
                push_starts.append(cell)

        push_chains: list[list[tuple[int, int]]] = []
        pushed_cells: set[tuple[int, int]] = set()
        for start in push_starts:
            chain = self._push_chain(start, delta, new_handle, candidate_slots, pushed_cells)
            if chain is None:
                return False
            push_chains.append(chain)
            pushed_cells.update(chain)

        active.handle = new_handle
        for chain in push_chains:
            for cell in reversed(chain):
                color = self.loose.pop(cell)
                self.loose[(cell[0] + delta[0], cell[1] + delta[1])] = color
        for idx, cell in pierce.items():
            active.slots[idx] = self.loose.pop(cell)
            self.flash_cells.add(cell)

        if not self._resolve_transfers(active):
            self.skewers, self.loose = before
            self.flash_cells = set()
            return False
        return True

    def _candidate_skewer_cells_are_valid(
        self, active: SkewerState, new_handle: tuple[int, int], candidate_slots: list[tuple[int, int]]
    ) -> bool:
        candidate_cells = [new_handle, *candidate_slots]
        for x, y in candidate_cells:
            if not (0 <= x < GRID_W and 0 <= y < PLAY_ROWS):
                return False
            if (x, y) in self.obstacles:
                return False

        for other in self.skewers.values():
            if other.sid == active.sid:
                continue
            other_cells = {other.handle: "handle"}
            for idx, cell in enumerate(other.slot_cells()):
                other_cells[cell] = idx
            if new_handle in other_cells:
                occupant = other_cells[new_handle]
                if occupant == "handle":
                    return False
                if isinstance(occupant, int) and other.movable and any(color is not None for color in other.slots):
                    return False
                if isinstance(occupant, int) and (not other.movable) and other.slots[occupant] is not None:
                    return False
            for cell in candidate_slots:
                occupant = other_cells.get(cell)
                if occupant == "handle":
                    return False
                if isinstance(occupant, int) and other.orientation == active.orientation:
                    return False
        return True

    def _push_chain(
        self,
        start: tuple[int, int],
        delta: tuple[int, int],
        new_handle: tuple[int, int],
        candidate_slots: list[tuple[int, int]],
        already_pushed: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        chain = []
        cell = start
        active_cells = {new_handle, *candidate_slots}
        blocked_skewer_cells = self._all_skewer_cells_after_active_move(new_handle, candidate_slots)
        while cell in self.loose:
            if cell in already_pushed:
                return None
            chain.append(cell)
            cell = (cell[0] + delta[0], cell[1] + delta[1])

        if not (0 <= cell[0] < GRID_W and 0 <= cell[1] < PLAY_ROWS):
            return None
        if cell in self.obstacles or cell in blocked_skewer_cells or cell in active_cells:
            return None
        return chain

    def _all_skewer_cells_after_active_move(
        self, new_handle: tuple[int, int], candidate_slots: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        cells = {new_handle, *candidate_slots}
        for skewer in self.skewers.values():
            if skewer.sid == self.selected_id:
                continue
            cells.add(skewer.handle)
            cells.update(skewer.slot_cells())
        return cells

    def _resolve_transfers(self, active: SkewerState) -> bool:
        for active_idx, active_cell in enumerate(active.slot_cells()):
            for other in self.skewers.values():
                if other.sid == active.sid or other.orientation == active.orientation:
                    continue
                for other_idx, other_cell in enumerate(other.slot_cells()):
                    if active_cell != other_cell:
                        continue
                    active_color = active.slots[active_idx]
                    other_color = other.slots[other_idx]
                    if active_color is not None and other_color is not None:
                        return False
                    if active_color is not None:
                        other.slots[other_idx] = active_color
                        active.slots[active_idx] = None
                        self.flash_cells.add(active_cell)
                    elif other_color is not None:
                        active.slots[active_idx] = other_color
                        other.slots[other_idx] = None
                        self.flash_cells.add(active_cell)
        return True

    def _is_solved(self) -> bool:
        for skewer in self.skewers.values():
            if not skewer.target:
                continue
            colors = tuple(color for color in skewer.slots if color is not None)
            if colors != skewer.target:
                return False
        return True

    def _sync_visuals(self) -> None:
        self.board_sprite.pixels = self._render_board()

    def _render_board(self) -> np.ndarray:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_WHITE, dtype=np.int8)
        frame[0:2, GRID_ORIGIN : GRID_ORIGIN + GRID_W * CELL_SIZE] = COLOR_HANDLE
        filled = int((GRID_W * CELL_SIZE) * self.remaining_steps / max(1, self.step_budget))
        if filled > 0:
            frame[0:2, GRID_ORIGIN : GRID_ORIGIN + filled] = COLOR_SELECTED

        for y in range(PLAY_ROWS):
            for x in range(GRID_W):
                _fill_cell(frame, x, y, COLOR_FLOOR)
        for y in range(PLAY_ROWS, GRID_W):
            for x in range(GRID_W):
                _fill_cell(frame, x, y, COLOR_WHITE)

        for cell in self.obstacles:
            _fill_cell(frame, *cell, COLOR_OBSTACLE)
            _outline_cell(frame, *cell, COLOR_BLACK)
        for skewer in self.skewers.values():
            for cell in skewer.slot_cells():
                _draw_segment(frame, cell, skewer.orientation)
        for skewer in self.skewers.values():
            _draw_handle(
                frame,
                skewer.handle,
                skewer.movable,
                skewer.sid == self.selected_id,
                self.invalid_flash and skewer.sid == self.selected_id,
            )
        for skewer in self.skewers.values():
            for idx, color in enumerate(skewer.slots):
                if color is not None:
                    _draw_block(frame, skewer.slot_cell(idx), color, skewer.orientation)
        for cell, color in self.loose.items():
            _draw_loose_block(frame, cell, color)
        for skewer in self.skewers.values():
            if skewer.target:
                _draw_targets(frame, skewer.handle[0], skewer.target)
        for cell in self.flash_cells:
            _outline_cell(frame, *cell, COLOR_SELECTED)
        return frame


def _pixel_to_cell(px: int, py: int) -> tuple[int, int] | None:
    gx = (px - GRID_ORIGIN) // CELL_SIZE
    gy = (py - GRID_ORIGIN) // CELL_SIZE
    if 0 <= gx < GRID_W and 0 <= gy < GRID_W:
        return gx, gy
    return None


def _cell_origin(x: int, y: int) -> tuple[int, int]:
    return GRID_ORIGIN + x * CELL_SIZE, GRID_ORIGIN + y * CELL_SIZE


def _fill_cell(frame: np.ndarray, x: int, y: int, color: int) -> None:
    px, py = _cell_origin(x, y)
    frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = color


def _outline_cell(frame: np.ndarray, x: int, y: int, color: int) -> None:
    px, py = _cell_origin(x, y)
    frame[py, px : px + CELL_SIZE] = color
    frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = color
    frame[py : py + CELL_SIZE, px] = color
    frame[py : py + CELL_SIZE, px + CELL_SIZE - 1] = color


def _draw_rail(frame: np.ndarray, cell: tuple[int, int], orientation: str) -> None:
    px, py = _cell_origin(*cell)
    if orientation == "h":
        frame[py + 2 : py + 4, px + 1 : px + 5] = COLOR_RAIL
    else:
        frame[py + 1 : py + 5, px + 2 : px + 4] = COLOR_RAIL


def _draw_segment(frame: np.ndarray, cell: tuple[int, int], orientation: str) -> None:
    px, py = _cell_origin(*cell)
    if orientation == "h":
        frame[py + 2 : py + 4, px : px + CELL_SIZE] = COLOR_SEGMENT
    else:
        frame[py : py + CELL_SIZE, px + 2 : px + 4] = COLOR_SEGMENT


def _draw_handle(frame: np.ndarray, cell: tuple[int, int], movable: bool, selected: bool, invalid: bool) -> None:
    color = COLOR_HANDLE if movable else COLOR_SCORING
    _fill_cell(frame, *cell, color)
    outline = COLOR_RED if invalid else COLOR_SELECTED if selected else COLOR_SEGMENT if movable else COLOR_OBSTACLE
    _outline_cell(frame, *cell, outline)


def _draw_loose_block(frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
    px, py = _cell_origin(*cell)
    frame[py + 1 : py + 5, px + 1 : px + 5] = color


def _draw_block(frame: np.ndarray, cell: tuple[int, int], color: int, orientation: str) -> None:
    _draw_loose_block(frame, cell, color)
    px, py = _cell_origin(*cell)
    if orientation == "h":
        frame[py + 2 : py + 4, px + 1 : px + 5] = COLOR_SEGMENT
    else:
        frame[py + 1 : py + 5, px + 2 : px + 4] = COLOR_SEGMENT


def _draw_targets(frame: np.ndarray, x: int, target: tuple[int, ...]) -> None:
    if len(target) >= 1:
        _draw_loose_block(frame, (x, 9), target[0])
    if len(target) >= 2:
        _draw_loose_block(frame, (x, 8), target[1])
