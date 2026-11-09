from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_SIZE = 64
MODULE_SIZE = 8
HUD_ROW = 0
PLAYFIELD_ROWS = range(1, 8)

FLOOR = 1
FLOOR_ACCENT = 2
BELT_OFF = 3
OUTLINE = 4
CAVITY = 5
FAIL = 8
BELT_ON = 9
CHEVRON = 10
HIGHLIGHT = 11
CRATE = 12
CRATE_ACCENT = 13
SUCCESS = 14
INPUT = 15

PHASE_EDITING = "editing"
PHASE_SUCCESS = "success"
PHASE_FAILURE = "failure"

STATE_NAMES = ("up", "right", "down")


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


@dataclass(frozen=True)
class SplitterSpec:
    id: str
    module: tuple[int, int]
    states: tuple[str, str]
    initial_state: str
    branches: dict[str, tuple[tuple[int, int], ...]]
    targets: dict[str, str]


@dataclass(frozen=True)
class TerminalSpec:
    id: str
    kind: str
    module: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    budget: int
    input_module: tuple[int, int]
    trunk: tuple[tuple[int, int], ...]
    splitters: tuple[SplitterSpec, ...]
    terminals: tuple[TerminalSpec, ...]
    goal_id: str

    def to_level_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "budget": self.budget,
            "input_module": list(self.input_module),
            "trunk": [list(point) for point in self.trunk],
            "goal_id": self.goal_id,
            "splitters": [
                {
                    "id": spec.id,
                    "module": list(spec.module),
                    "states": list(spec.states),
                    "initial_state": spec.initial_state,
                    "branches": {state: [list(point) for point in points] for state, points in spec.branches.items()},
                    "targets": dict(spec.targets),
                }
                for spec in self.splitters
            ],
            "terminals": [
                {"id": terminal.id, "kind": terminal.kind, "module": list(terminal.module)}
                for terminal in self.terminals
            ],
        }


LEVEL_SPECS = (
    LevelSpec(
        name="Level 1",
        budget=5,
        input_module=(0, 4),
        trunk=((0, 4), (1, 4), (2, 4), (3, 4)),
        splitters=(
            SplitterSpec(
                id="S1",
                module=(3, 4),
                states=("up", "down"),
                initial_state="down",
                branches={
                    "up": ((3, 4), (3, 3), (4, 3), (5, 3), (6, 3)),
                    "down": ((3, 4), (3, 5), (4, 5), (5, 5), (6, 5)),
                },
                targets={"up": "G", "down": "B1"},
            ),
        ),
        terminals=(
            TerminalSpec(id="G", kind="goal", module=(6, 3)),
            TerminalSpec(id="B1", kind="bumper", module=(6, 5)),
        ),
        goal_id="G",
    ),
    LevelSpec(
        name="Level 2",
        budget=7,
        input_module=(0, 5),
        trunk=((0, 5), (1, 5), (2, 5)),
        splitters=(
            SplitterSpec(
                id="S1",
                module=(2, 5),
                states=("up", "right"),
                initial_state="right",
                branches={
                    "up": ((2, 5), (2, 4), (2, 3), (3, 3), (4, 3)),
                    "right": ((2, 5), (3, 5), (4, 5), (5, 5), (6, 5)),
                },
                targets={"up": "S2", "right": "B1"},
            ),
            SplitterSpec(
                id="S2",
                module=(4, 3),
                states=("right", "up"),
                initial_state="up",
                branches={"right": ((4, 3), (5, 3), (6, 3)), "up": ((4, 3), (4, 2), (4, 1))},
                targets={"right": "G", "up": "B2"},
            ),
        ),
        terminals=(
            TerminalSpec(id="G", kind="goal", module=(6, 3)),
            TerminalSpec(id="B1", kind="bumper", module=(6, 5)),
            TerminalSpec(id="B2", kind="bumper", module=(4, 1)),
        ),
        goal_id="G",
    ),
    LevelSpec(
        name="Level 3",
        budget=9,
        input_module=(0, 6),
        trunk=((0, 6), (1, 6), (2, 6)),
        splitters=(
            SplitterSpec(
                id="S1",
                module=(2, 6),
                states=("up", "right"),
                initial_state="right",
                branches={
                    "up": ((2, 6), (2, 5), (2, 4), (3, 4), (4, 4)),
                    "right": ((2, 6), (3, 6), (4, 6), (5, 6), (6, 6)),
                },
                targets={"up": "S2", "right": "B1"},
            ),
            SplitterSpec(
                id="S2",
                module=(4, 4),
                states=("right", "up"),
                initial_state="right",
                branches={"right": ((4, 4), (5, 4)), "up": ((4, 4), (4, 3), (4, 2))},
                targets={"right": "S3", "up": "B2"},
            ),
            SplitterSpec(
                id="S3",
                module=(5, 4),
                states=("up", "right"),
                initial_state="right",
                branches={"up": ((5, 4), (5, 3), (5, 2)), "right": ((5, 4), (6, 4), (7, 4))},
                targets={"up": "G", "right": "B3"},
            ),
        ),
        terminals=(
            TerminalSpec(id="G", kind="goal", module=(5, 2)),
            TerminalSpec(id="B1", kind="bumper", module=(6, 6)),
            TerminalSpec(id="B2", kind="bumper", module=(4, 2)),
            TerminalSpec(id="B3", kind="bumper", module=(7, 4)),
        ),
        goal_id="G",
    ),
)

GAME_ID = "conveyor_sort_gpt-0001"


def _module_rect(module: tuple[int, int]) -> tuple[int, int, int, int]:
    left = module[0] * MODULE_SIZE
    top = module[1] * MODULE_SIZE
    return left, top, MODULE_SIZE, MODULE_SIZE


def _module_center(module: tuple[int, int]) -> tuple[int, int]:
    return module[0] * MODULE_SIZE + 4, module[1] * MODULE_SIZE + 4


class ConveyorSortGpt(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [self._build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=GRID_SIZE, height=GRID_SIZE, background=FLOOR),
            win_score=len(levels),
            available_actions=[5, 6],
            seed=seed,
        )
        self._phase = PHASE_EDITING
        self._level_budget = 0
        self._route_score = 0
        self._terminal_id: str | None = None
        self._traveled_path: tuple[tuple[int, int], ...] = ()
        self._board_sprite: Sprite | None = None
        self._splitter_click_sprites: dict[str, Sprite] = {}
        self._level_spec: LevelSpec = LEVEL_SPECS[0]
        self._splitter_specs: dict[str, SplitterSpec] = {}
        self._splitter_states: dict[str, str] = {}
        self._terminal_specs: dict[str, TerminalSpec] = {}

    def _build_level(self, spec: LevelSpec) -> Level:
        sprites = [Sprite(_solid(GRID_SIZE, GRID_SIZE, FLOOR), name="board", x=0, y=0, layer=0, collidable=False)]
        for splitter in spec.splitters:
            left, top, _, _ = _module_rect(splitter.module)
            sprites.append(
                Sprite(
                    _solid(MODULE_SIZE, MODULE_SIZE, FLOOR),
                    name=f"click_{splitter.id}",
                    x=left,
                    y=top,
                    layer=-10,
                    visible=False,
                    collidable=False,
                    tags=["sys_click", "sys_every_pixel", "splitter_click", splitter.id],
                )
            )
        return Level(sprites=sprites, grid_size=(GRID_SIZE, GRID_SIZE), data=spec.to_level_data(), name=spec.name)

    def on_set_level(self, level: Level) -> None:
        level_idx = int(self._current_level_index)
        self._level_spec = LEVEL_SPECS[level_idx]
        self._phase = PHASE_EDITING
        self._level_budget = self._level_spec.budget
        self._route_score = 0
        self._terminal_id = None
        self._traveled_path = ()
        self._splitter_specs = {spec.id: spec for spec in self._level_spec.splitters}
        self._terminal_specs = {terminal.id: terminal for terminal in self._level_spec.terminals}
        self._splitter_states = {spec.id: spec.initial_state for spec in self._level_spec.splitters}
        boards = level.get_sprites_by_name("board")
        self._board_sprite = boards[0] if boards else None
        self._splitter_click_sprites = {}
        for sprite in level.get_sprites_by_tag("splitter_click"):
            for tag in sprite.tags:
                if tag in self._splitter_specs:
                    self._splitter_click_sprites[tag] = sprite
                    break
        self._render_board()

    def _is_sprite_clickable_now(self, sprite: Sprite) -> bool:
        if "splitter_click" not in sprite.tags:
            return False
        return self._phase == PHASE_EDITING

    def step(self) -> None:
        action_id = int(self.action.id.value if hasattr(self.action.id, "value") else self.action.id)
        if action_id in (1, 2, 3, 4):
            self.complete_action()
            return

        if action_id == 6:
            self._handle_click(int(self.action.data.get("x", -999)), int(self.action.data.get("y", -999)))
            self.complete_action()
            return

        if action_id == 5:
            self._handle_space()
            self.complete_action()
            return

        self.complete_action()

    def _handle_click(self, x: int, y: int) -> None:
        if self._phase != PHASE_EDITING:
            return
        splitter_id = self._find_splitter_at(x, y)
        if splitter_id is None:
            return
        states = self._splitter_specs[splitter_id].states
        current = self._splitter_states[splitter_id]
        self._splitter_states[splitter_id] = states[1] if current == states[0] else states[0]
        self._spend_budget()
        if self._phase == PHASE_EDITING:
            self._render_board()

    def _handle_space(self) -> None:
        if self._phase == PHASE_SUCCESS:
            self.next_level()
            return
        if self._phase == PHASE_FAILURE:
            return
        if self._phase != PHASE_EDITING:
            return

        self._spend_budget()
        if self._phase == PHASE_FAILURE:
            return
        path, terminal_id = self._trace_route()
        self._traveled_path = path
        self._terminal_id = terminal_id
        if terminal_id == self._level_spec.goal_id:
            self._phase = PHASE_SUCCESS
            self._route_score = 1
        elif self._phase == PHASE_EDITING:
            self._enter_failure()
            return
        self._render_board()

    def _spend_budget(self) -> None:
        if self._phase != PHASE_EDITING:
            return
        self._level_budget -= 1
        if self._level_budget <= 0:
            self._level_budget = 0
            self._enter_failure()

    def _enter_failure(self) -> None:
        if self._phase == PHASE_FAILURE:
            return
        self._phase = PHASE_FAILURE
        self._render_board()
        self.lose()

    def _find_splitter_at(self, x: int, y: int) -> str | None:
        for splitter_id, spec in self._splitter_specs.items():
            left, top, width, height = _module_rect(spec.module)
            if left <= x < left + width and top <= y < top + height:
                return splitter_id
        return None

    def _trace_route(self) -> tuple[tuple[tuple[int, int], ...], str]:
        points = list(self._level_spec.trunk)
        current_target = self._level_spec.splitters[0].id
        while True:
            splitter = self._splitter_specs[current_target]
            state = self._splitter_states[current_target]
            branch = splitter.branches[state]
            points.extend(branch[1:])
            current_target = splitter.targets[state]
            if current_target in self._splitter_specs:
                continue
            return tuple(points), current_target

    def _active_modules(self) -> set[tuple[int, int]]:
        route, _ = self._trace_route()
        return set(route)

    def _render_board(self) -> None:
        if self._board_sprite is None:
            return
        board = np.full((GRID_SIZE, GRID_SIZE), FLOOR, dtype=np.int8)
        self._draw_floor_accents(board)
        self._draw_budget(board)
        self._draw_all_belts(board)
        self._draw_terminals(board)
        self._draw_input(board)
        self._draw_splitters(board)
        self._draw_crate(board)
        self._board_sprite.pixels = board

    def _draw_floor_accents(self, board: np.ndarray) -> None:
        for row in PLAYFIELD_ROWS:
            y = row * MODULE_SIZE + 1
            for col in range(1, 8, 2):
                x = col * MODULE_SIZE + 1
                board[y, x] = FLOOR_ACCENT
                if y + 3 < GRID_SIZE and x + 3 < GRID_SIZE:
                    board[y + 3, x + 3] = FLOOR_ACCENT

    def _draw_budget(self, board: np.ndarray) -> None:
        max_slots = 9
        for idx in range(max_slots):
            left = 2 + idx * 7
            top = 2
            fill = HIGHLIGHT if idx < self._level_budget else BELT_OFF
            board[top : top + 4, left : left + 4] = fill
            board[top, left : left + 4] = OUTLINE
            board[top + 3, left : left + 4] = OUTLINE
            board[top : top + 4, left] = OUTLINE
            board[top : top + 4, left + 3] = OUTLINE

    def _draw_all_belts(self, board: np.ndarray) -> None:
        active_route = self._traveled_path if self._phase in {PHASE_SUCCESS, PHASE_FAILURE} else self._trace_route()[0]
        active_modules = set(active_route)
        all_polylines = [self._level_spec.trunk]
        for splitter in self._level_spec.splitters:
            all_polylines.extend(splitter.branches.values())
        for polyline in all_polylines:
            self._draw_polyline(board, polyline, surface=BELT_OFF, chevrons=False)
        for polyline in self._active_polylines(active_modules):
            self._draw_polyline(board, polyline, surface=BELT_ON, chevrons=True)

    def _active_polylines(self, active_modules: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
        polylines = [self._level_spec.trunk]
        next_splitter = self._level_spec.splitters[0].id
        while True:
            splitter = self._splitter_specs[next_splitter]
            state = self._splitter_states[next_splitter]
            branch = splitter.branches[state]
            polylines.append(branch)
            target = splitter.targets[state]
            if target not in self._splitter_specs:
                break
            next_splitter = target
        return [polyline for polyline in polylines if any(point in active_modules for point in polyline)]

    def _draw_polyline(
        self, board: np.ndarray, polyline: tuple[tuple[int, int], ...], *, surface: int, chevrons: bool
    ) -> None:
        for start, end in pairwise(polyline):
            self._draw_segment(board, start, end, surface=surface, chevrons=chevrons)

    def _draw_segment(
        self, board: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, surface: int, chevrons: bool
    ) -> None:
        x1, y1 = _module_center(start)
        x2, y2 = _module_center(end)
        if x1 == x2:
            left = max(0, x1 - 3)
            right = min(GRID_SIZE, x1 + 4)
            top = max(0, min(y1, y2) - 3)
            bottom = min(GRID_SIZE, max(y1, y2) + 4)
        else:
            left = max(0, min(x1, x2) - 3)
            right = min(GRID_SIZE, max(x1, x2) + 4)
            top = max(0, y1 - 3)
            bottom = min(GRID_SIZE, y1 + 4)

        board[top:bottom, left:right] = OUTLINE
        inner_left = min(GRID_SIZE, left + 1)
        inner_right = max(0, right - 1)
        inner_top = min(GRID_SIZE, top + 1)
        inner_bottom = max(0, bottom - 1)
        if inner_left < inner_right and inner_top < inner_bottom:
            board[inner_top:inner_bottom, inner_left:inner_right] = surface

        if not chevrons:
            return
        chevron_color = CHEVRON
        if x1 == x2:
            step = 6 if y2 > y1 else -6
            direction = 1 if y2 > y1 else -1
            for cy in range(y1 + step, y2, step):
                self._draw_vertical_chevron(board, x1, cy, chevron_color, direction)
        else:
            step = 6 if x2 > x1 else -6
            direction = 1 if x2 > x1 else -1
            for cx in range(x1 + step, x2, step):
                self._draw_horizontal_chevron(board, cx, y1, chevron_color, direction)

    def _draw_vertical_chevron(self, board: np.ndarray, cx: int, cy: int, color: int, direction: int) -> None:
        points = [(cx, cy), (cx - 1, cy - direction), (cx + 1, cy - direction)]
        for px, py in points:
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                board[py, px] = color

    def _draw_horizontal_chevron(self, board: np.ndarray, cx: int, cy: int, color: int, direction: int) -> None:
        points = [(cx, cy), (cx - direction, cy - 1), (cx - direction, cy + 1)]
        for px, py in points:
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                board[py, px] = color

    def _draw_terminals(self, board: np.ndarray) -> None:
        for terminal in self._level_spec.terminals:
            if terminal.kind == "goal":
                self._draw_goal(board, terminal)
            else:
                self._draw_bumper(board, terminal)

    def _draw_goal(self, board: np.ndarray, terminal: TerminalSpec) -> None:
        left, top, _, _ = _module_rect(terminal.module)
        glow = self._phase == PHASE_SUCCESS and self._terminal_id == terminal.id
        rim = SUCCESS if glow else HIGHLIGHT
        board[top + 1 : top + 7, left + 1 : left + 7] = rim
        board[top + 2 : top + 6, left + 2 : left + 6] = CAVITY
        board[top + 4 : top + 6, left + 2 : left + 6] = CAVITY
        if glow:
            board[max(0, top) : min(GRID_SIZE, top + 8), max(0, left) : min(GRID_SIZE, left + 8)] = np.maximum(
                board[max(0, top) : min(GRID_SIZE, top + 8), max(0, left) : min(GRID_SIZE, left + 8)], np.int8(SUCCESS)
            )
            board[top + 2 : top + 6, left + 2 : left + 6] = CAVITY

    def _draw_bumper(self, board: np.ndarray, terminal: TerminalSpec) -> None:
        left, top, _, _ = _module_rect(terminal.module)
        body = FAIL
        if self._phase == PHASE_FAILURE and self._terminal_id == terminal.id:
            body = FAIL
        board[top + 1 : top + 7, left + 1 : left + 7] = body
        for offset in range(1, 7):
            stripe_x = left + offset
            stripe_y = top + offset
            if stripe_x < left + 7 and stripe_y < top + 7:
                board[top + 1 : top + 7, stripe_x] = CRATE_ACCENT if offset % 2 else body
        board[top + 1, left + 1 : left + 7] = OUTLINE
        board[top + 6, left + 1 : left + 7] = OUTLINE
        board[top + 1 : top + 7, left + 1] = OUTLINE
        board[top + 1 : top + 7, left + 6] = OUTLINE

    def _draw_input(self, board: np.ndarray) -> None:
        left, top, _, _ = _module_rect(self._level_spec.input_module)
        board[top + 1 : top + 7, left + 1 : left + 7] = INPUT
        board[top + 2 : top + 6, left + 1 : left + 3] = CAVITY
        board[top + 1, left + 1 : left + 7] = OUTLINE
        board[top + 6, left + 1 : left + 7] = OUTLINE
        board[top + 1 : top + 7, left + 6] = OUTLINE

    def _draw_splitters(self, board: np.ndarray) -> None:
        for splitter in self._level_spec.splitters:
            self._draw_splitter(board, splitter)

    def _draw_splitter(self, board: np.ndarray, splitter: SplitterSpec) -> None:
        left, top, _, _ = _module_rect(splitter.module)
        board[top + 1 : top + 7, left + 1 : left + 7] = FLOOR
        board[top + 1, left + 1 : left + 7] = OUTLINE
        board[top + 6, left + 1 : left + 7] = OUTLINE
        board[top + 1 : top + 7, left + 1] = OUTLINE
        board[top + 1 : top + 7, left + 6] = OUTLINE

        state = self._splitter_states[splitter.id]
        for branch_state in splitter.states:
            lane_color = BELT_ON if branch_state == state else BELT_OFF
            self._draw_splitter_lane(board, splitter, branch_state, lane_color)

        arrow = HIGHLIGHT
        self._draw_splitter_arrow(board, splitter, state, arrow)

    def _draw_splitter_lane(self, board: np.ndarray, splitter: SplitterSpec, state: str, color: int) -> None:
        left, top, _, _ = _module_rect(splitter.module)
        cx, cy = left + 4, top + 4
        board[cy - 1 : cy + 1, left + 1 : cx + 1] = color
        if state == "up":
            board[top + 1 : cy + 1, cx - 1 : cx + 1] = color
        elif state == "down":
            board[cy : top + 7, cx - 1 : cx + 1] = color
        elif state == "right":
            board[cy - 1 : cy + 1, cx : left + 7] = color

    def _draw_splitter_arrow(self, board: np.ndarray, splitter: SplitterSpec, state: str, color: int) -> None:
        left, top, _, _ = _module_rect(splitter.module)
        cx, cy = left + 4, top + 4
        if state == "up":
            points = [(cx, cy - 2), (cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1), (cx, cy), (cx, cy + 1)]
        elif state == "down":
            points = [(cx, cy + 2), (cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1), (cx, cy), (cx, cy - 1)]
        else:
            points = [(cx + 2, cy), (cx + 1, cy - 1), (cx + 1, cy), (cx + 1, cy + 1), (cx, cy), (cx - 1, cy)]
        for px, py in points:
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                board[py, px] = color

    def _draw_crate(self, board: np.ndarray) -> None:
        if self._phase == PHASE_EDITING:
            module = self._level_spec.input_module
        else:
            terminal = self._terminal_specs.get(self._terminal_id or "")
            module = terminal.module if terminal is not None else self._level_spec.input_module
        cx, cy = _module_center(module)
        for py in range(cy - 1, cy + 2):
            for px in range(cx - 1, cx + 2):
                if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                    board[py, px] = CRATE
        if 0 <= cy < GRID_SIZE:
            for px in range(cx - 1, cx + 2):
                if 0 <= px < GRID_SIZE:
                    board[cy, px] = CRATE_ACCENT
        if 0 <= cx < GRID_SIZE:
            board[cy - 1 : cy + 2, cx] = CRATE_ACCENT


__all__ = ["GAME_ID", "ConveyorSortGpt"]
