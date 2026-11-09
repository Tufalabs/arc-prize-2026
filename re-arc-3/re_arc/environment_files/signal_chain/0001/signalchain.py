from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
UI_BOTTOM = 8
DIVIDER_Y = 9
PLAYFIELD_Y0 = 10

COLOR_BG = 0
COLOR_UI = 1
COLOR_DIVIDER = 2
COLOR_SPENT = 3
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

SYMBOL_BITMAPS = {
    "DOT": ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0)),
    "BAR": ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    "TRIANGLE": ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    "CROSS": ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0), (0, 0, 1, 0, 0), (0, 1, 0, 1, 0), (1, 0, 0, 0, 1)),
    "DIAMOND": ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0)),
    "CHEVRON": ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    "SQUARE": ((1, 1, 1, 1, 1), (1, 0, 0, 0, 1), (1, 0, 0, 0, 1), (1, 0, 0, 0, 1), (1, 1, 1, 1, 1)),
}


class TargetSpec(tuple):
    __slots__ = ()

    def __new__(cls, symbol: str, center: tuple[int, int]):
        return super().__new__(cls, (symbol, center))

    @property
    def symbol(self) -> str:
        return self[0]

    @property
    def center(self) -> tuple[int, int]:
        return self[1]


class WallSpec(tuple):
    __slots__ = ()

    def __new__(cls, x1: int, x2: int, y1: int, y2: int):
        return super().__new__(cls, (x1, x2, y1, y2))

    @property
    def x1(self) -> int:
        return self[0]

    @property
    def x2(self) -> int:
        return self[1]

    @property
    def y1(self) -> int:
        return self[2]

    @property
    def y2(self) -> int:
        return self[3]


class LevelSpec(tuple):
    __slots__ = ()

    def __new__(
        cls,
        name: str,
        budget: int,
        sequence: tuple[str, ...],
        targets: tuple[TargetSpec, ...],
        walls: tuple[WallSpec, ...],
    ):
        return super().__new__(cls, (name, budget, sequence, targets, walls))

    @property
    def name(self) -> str:
        return self[0]

    @property
    def budget(self) -> int:
        return self[1]

    @property
    def sequence(self) -> tuple[str, ...]:
        return self[2]

    @property
    def targets(self) -> tuple[TargetSpec, ...]:
        return self[3]

    @property
    def walls(self) -> tuple[WallSpec, ...]:
        return self[4]


LEVEL_SPECS = (
    LevelSpec(
        name="level_1",
        budget=9,
        sequence=("DOT", "TRIANGLE", "CROSS"),
        targets=(TargetSpec("DOT", (14, 22)), TargetSpec("TRIANGLE", (32, 22)), TargetSpec("CROSS", (32, 42))),
        walls=(),
    ),
    LevelSpec(
        name="level_2",
        budget=12,
        sequence=("BAR", "DIAMOND", "DOT", "CROSS"),
        targets=(
            TargetSpec("BAR", (12, 20)),
            TargetSpec("DIAMOND", (28, 20)),
            TargetSpec("DOT", (28, 44)),
            TargetSpec("CROSS", (48, 44)),
            TargetSpec("TRIANGLE", (48, 20)),
        ),
        walls=(WallSpec(36, 42, 14, 30), WallSpec(16, 22, 30, 36)),
    ),
    LevelSpec(
        name="level_3",
        budget=16,
        sequence=("DOT", "CHEVRON", "BAR", "DIAMOND", "CROSS"),
        targets=(
            TargetSpec("DOT", (12, 18)),
            TargetSpec("CHEVRON", (48, 18)),
            TargetSpec("BAR", (48, 48)),
            TargetSpec("DIAMOND", (16, 48)),
            TargetSpec("CROSS", (16, 30)),
            TargetSpec("TRIANGLE", (56, 32)),
            TargetSpec("SQUARE", (32, 58)),
        ),
        walls=(WallSpec(26, 38, 24, 42), WallSpec(40, 44, 28, 38), WallSpec(22, 32, 52, 56)),
    ),
)


def _blank_board() -> np.ndarray:
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)


def _make_board_sprite() -> Sprite:
    return Sprite(pixels=_blank_board(), name="board", x=0, y=0, layer=-10, visible=True, collidable=False)


LEVELS = tuple(
    Level(sprites=[_make_board_sprite()], grid_size=(GRID_SIZE, GRID_SIZE), name=spec.name) for spec in LEVEL_SPECS
)


class SignalChain(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            "signal_chain",
            list(LEVELS),
            Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BG, COLOR_BG, []),
            False,
            len(LEVEL_SPECS),
            [5, 6],
            seed=seed,
        )
        self._route_score = 0
        self._spec: LevelSpec | None = None
        self._board: Sprite | None = None
        self._progress_idx = 0
        self._actions_remaining = 0
        self._success_wait = False
        self._failed = False
        self._final_victory = False
        self._clicked_target_ids: list[int] = []
        self._placed_segments: list[tuple[int, int]] = []
        self._last_feedback: tuple[str, object] | None = None
        self._wall_cells: set[tuple[int, int]] = set()

    def on_set_level(self, _level: Level) -> None:
        self._spec = LEVEL_SPECS[self.level_index]
        self._board = self.current_level.get_sprites_by_name("board")[0]
        self._route_score = 0
        self._progress_idx = 0
        self._actions_remaining = self._spec.budget
        self._success_wait = False
        self._failed = False
        self._final_victory = False
        self._clicked_target_ids = []
        self._placed_segments = []
        self._last_feedback = None
        self._wall_cells = set()
        for wall in self._spec.walls:
            for y in range(wall.y1, wall.y2 + 1):
                for x in range(wall.x1, wall.x2 + 1):
                    self._wall_cells.add((x, y))
        self._render()

    def step(self) -> None:
        action_id = self._action_id()

        if self._final_victory:
            if action_id == int(GameAction.ACTION5.value):
                self.full_reset()
            self.complete_action()
            return

        if self._success_wait:
            self._last_feedback = None
            if self.is_last_level():
                self._final_victory = True
                self._render()
                self.next_level()
            else:
                self.next_level()
            self.complete_action()
            return

        if self._failed:
            self._render()
            self.complete_action()
            return

        self._last_feedback = None

        if action_id in (
            int(GameAction.ACTION1.value),
            int(GameAction.ACTION2.value),
            int(GameAction.ACTION3.value),
            int(GameAction.ACTION4.value),
        ):
            self._render()
            self.complete_action()
            return

        if action_id == int(GameAction.ACTION5.value):
            self.level_reset()
            self.complete_action()
            return

        if action_id == int(GameAction.ACTION6.value):
            self._handle_click()

        self._render()
        self.complete_action()

    def _handle_click(self) -> None:
        if self._spec is None:
            return

        x = int(self.action.data.get("x", -1))
        y = int(self.action.data.get("y", -1))
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return

        self._actions_remaining -= 1
        target_id = self._target_at(x, y)
        if target_id is None:
            self._last_feedback = ("empty_click", (x, y))
            self._post_click_resolution()
            return

        expected = self._spec.sequence[self._progress_idx]
        target = self._spec.targets[target_id]
        if target.symbol != expected:
            self._last_feedback = ("wrong_target", target_id)
            self._post_click_resolution()
            return

        if self._progress_idx == 0:
            self._clicked_target_ids.append(target_id)
            self._progress_idx += 1
            self._post_click_resolution()
            return

        prev_id = self._clicked_target_ids[-1]
        cells = self._bresenham(self._spec.targets[prev_id].center, target.center)
        if self._segment_blocked(cells, prev_id, target_id):
            self._last_feedback = ("blocked_segment", (cells, target_id))
            self._post_click_resolution()
            return

        self._placed_segments.append((prev_id, target_id))
        self._clicked_target_ids.append(target_id)
        self._progress_idx += 1
        self._post_click_resolution()

    def _post_click_resolution(self) -> None:
        if self._spec is None:
            return
        if self._progress_idx == len(self._spec.sequence):
            self._route_score += 1
            self._success_wait = True
            return
        if self._actions_remaining <= 0:
            self._failed = True
            self.lose()

    def _render(self) -> None:
        if self._board is None or self._spec is None:
            return

        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int16)
        frame[: UI_BOTTOM + 1, :] = COLOR_UI
        frame[DIVIDER_Y, :] = COLOR_DIVIDER

        self._draw_preview(frame)
        self._draw_budget_bar(frame)
        self._draw_walls(frame)
        self._draw_segments(frame)
        self._draw_targets(frame)
        self._draw_feedback(frame)

        if self._failed:
            self._draw_border(frame, COLOR_RED, COLOR_RED)
        if self._final_victory:
            self._draw_border(frame, COLOR_GREEN, COLOR_PURPLE)

        self._board.pixels = frame

    def _draw_preview(self, frame: np.ndarray) -> None:
        assert self._spec is not None
        for idx, symbol in enumerate(self._spec.sequence):
            x0 = 2 + idx * 7
            y0 = 2
            if self._success_wait or self._final_victory:
                frame_color = COLOR_GREEN
            elif idx < self._progress_idx:
                frame_color = COLOR_GREEN
            elif idx == self._progress_idx:
                frame_color = COLOR_MAGENTA
            else:
                frame_color = COLOR_YELLOW
            frame[y0 : y0 + 5, x0 : x0 + 5] = frame_color
            frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_BG
            self._draw_symbol(frame, symbol, x0, y0, 5)

    def _draw_budget_bar(self, frame: np.ndarray) -> None:
        assert self._spec is not None
        start_x = GRID_SIZE - self._spec.budget - 2
        for idx in range(self._spec.budget):
            x = start_x + idx
            if idx < self._actions_remaining:
                color = COLOR_ORANGE if self._actions_remaining <= 3 else COLOR_GREEN
            else:
                color = COLOR_SPENT
            frame[2:4, x] = color

    def _draw_walls(self, frame: np.ndarray) -> None:
        assert self._spec is not None
        for wall in self._spec.walls:
            frame[wall.y1 : wall.y2 + 1, wall.x1 : wall.x2 + 1] = COLOR_WALL
            frame[wall.y1, wall.x1 : wall.x2 + 1] = COLOR_BLACK
            frame[wall.y2, wall.x1 : wall.x2 + 1] = COLOR_BLACK
            frame[wall.y1 : wall.y2 + 1, wall.x1] = COLOR_BLACK
            frame[wall.y1 : wall.y2 + 1, wall.x2] = COLOR_BLACK

    def _draw_segments(self, frame: np.ndarray) -> None:
        color = COLOR_GREEN if (self._success_wait or self._final_victory) else COLOR_YELLOW
        assert self._spec is not None
        for start_id, end_id in self._placed_segments:
            cells = self._bresenham(self._spec.targets[start_id].center, self._spec.targets[end_id].center)
            for x, y in cells:
                frame[y, x] = color

    def _draw_targets(self, frame: np.ndarray) -> None:
        assert self._spec is not None
        feedback_kind = self._last_feedback[0] if self._last_feedback else None
        wrong_target_id = self._last_feedback[1] if feedback_kind == "wrong_target" else None
        blocked_target_id = None
        if feedback_kind == "blocked_segment":
            blocked_target_id = self._last_feedback[1][1]

        for target_id, target in enumerate(self._spec.targets):
            ring_color = COLOR_BLUE
            if self._success_wait or self._final_victory:
                ring_color = COLOR_GREEN
            elif target_id == wrong_target_id or target_id == blocked_target_id:
                ring_color = COLOR_RED
            elif target_id in self._clicked_target_ids:
                ring_color = COLOR_YELLOW
                if self._clicked_target_ids and target_id == self._clicked_target_ids[-1]:
                    ring_color = COLOR_MAGENTA

            self._draw_node(frame, target.center[0], target.center[1], target.symbol, ring_color)

    def _draw_feedback(self, frame: np.ndarray) -> None:
        if self._last_feedback is None:
            return
        kind, payload = self._last_feedback
        if kind == "empty_click":
            x, y = payload
            for dx, dy in ((-1, -1), (0, 0), (1, 1), (-1, 1), (1, -1)):
                px = x + dx
                py = y + dy
                if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                    frame[py, px] = COLOR_RED
            return
        if kind == "blocked_segment":
            cells, _target_id = payload
            for x, y in cells:
                frame[y, x] = COLOR_RED

    def _draw_node(self, frame: np.ndarray, cx: int, cy: int, symbol: str, ring_color: int) -> None:
        x0 = cx - 3
        y0 = cy - 3
        ring_cells = (
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (0, 1),
            (6, 1),
            (0, 2),
            (6, 2),
            (0, 3),
            (6, 3),
            (0, 4),
            (6, 4),
            (0, 5),
            (6, 5),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
        )
        for dx in range(7):
            for dy in range(7):
                px = x0 + dx
                py = y0 + dy
                if not (0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE):
                    continue
                if (dx, dy) in ring_cells:
                    frame[py, px] = ring_color
                elif 1 <= dx <= 5 and 1 <= dy <= 5:
                    frame[py, px] = COLOR_BG
        self._draw_symbol(frame, symbol, cx - 2, cy - 2, 5)
        self._draw_node_highlight(frame, cx, cy)

    def _draw_node_highlight(self, frame: np.ndarray, cx: int, cy: int) -> None:
        for px, py in ((cx - 2, cy - 2), (cx + 2, cy - 2), (cx - 2, cy + 2), (cx + 2, cy + 2)):
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE and frame[py, px] == COLOR_BG:
                frame[py, px] = COLOR_LIGHT_BLUE

    def _draw_symbol(self, frame: np.ndarray, symbol: str, x0: int, y0: int, size: int) -> None:
        del size
        bitmap = SYMBOL_BITMAPS[symbol]
        for dy, row in enumerate(bitmap):
            for dx, value in enumerate(row):
                if value:
                    frame[y0 + dy, x0 + dx] = COLOR_BLACK

    def _draw_border(self, frame: np.ndarray, border_color: int, corner_color: int) -> None:
        frame[0, :] = border_color
        frame[-1, :] = border_color
        frame[:, 0] = border_color
        frame[:, -1] = border_color
        frame[0, 0] = corner_color
        frame[0, -1] = corner_color
        frame[-1, 0] = corner_color
        frame[-1, -1] = corner_color

    def _segment_blocked(self, cells: list[tuple[int, int]], start_id: int, end_id: int) -> bool:
        assert self._spec is not None
        start_box = self._node_box(self._spec.targets[start_id].center)
        end_box = self._node_box(self._spec.targets[end_id].center)
        for x, y in cells:
            if self._in_box(x, y, start_box) or self._in_box(x, y, end_box):
                continue
            if (x, y) in self._wall_cells:
                return True
        return False

    def _target_at(self, x: int, y: int) -> int | None:
        assert self._spec is not None
        for idx, target in enumerate(self._spec.targets):
            left, right, top, bottom = self._node_box(target.center)
            if left <= x <= right and top <= y <= bottom:
                return idx
        return None

    @staticmethod
    def _node_box(center: tuple[int, int]) -> tuple[int, int, int, int]:
        cx, cy = center
        return (cx - 3, cx + 3, cy - 3, cy + 3)

    @staticmethod
    def _in_box(x: int, y: int, box: tuple[int, int, int, int]) -> bool:
        left, right, top, bottom = box
        return left <= x <= right and top <= y <= bottom

    @staticmethod
    def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cells: list[tuple[int, int]] = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                return cells
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _action_id(self) -> int:
        raw = getattr(self.action, "id", -1)
        value = getattr(raw, "value", raw)
        return int(value[0] if isinstance(value, tuple) else value)
