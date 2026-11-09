from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
HUD_HEIGHT = 8
CELL_SIZE = 4
LOGICAL_WIDTH = 16
LOGICAL_HEIGHT = 14
PLAYFIELD_TOP = HUD_HEIGHT

COLOR_BG = 5
COLOR_FLOOR = 1
COLOR_FLOOR_EDGE = 2
COLOR_SPENT = 3
COLOR_CABLE = 15
COLOR_BAR = 11
COLOR_BAR_ACCENT = 12
COLOR_WINCH = 9
COLOR_WINCH_HILITE = 10
COLOR_GOAL = 14
COLOR_FAIL = 8
COLOR_AVATAR = 7
COLOR_AVATAR_ACCENT = 15

BAR_TILE = np.array([[12, 11, 11, 12], [11, 12, 12, 11], [11, 12, 12, 11], [12, 11, 11, 12]], dtype=np.int8)
AVATAR_TILE = np.array([[5, 15, 15, 5], [15, 7, 7, 15], [15, 7, 7, 15], [5, 15, 15, 5]], dtype=np.int8)
WINCH_TILE = np.array(
    [
        [9, 9, 9, 9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 10, 5, 10, 10, 5, 10, 9],
        [9, 10, 10, 15, 15, 10, 10, 9],
        [9, 10, 10, 15, 15, 10, 10, 9],
        [9, 10, 5, 10, 10, 5, 10, 9],
        [9, 10, 10, 10, 10, 10, 10, 9],
        [9, 9, 9, 9, 9, 9, 9, 9],
    ],
    dtype=np.int8,
)
GOAL_TILE = np.array(
    [
        [14, 14, 14, 14, 14, 14, 14, 14],
        [14, 5, 5, 5, 5, 5, 5, 14],
        [14, 5, 11, 11, 11, 11, 5, 14],
        [14, 5, 11, 14, 14, 11, 5, 14],
        [14, 5, 11, 14, 14, 11, 5, 14],
        [14, 5, 11, 11, 11, 11, 5, 14],
        [14, 5, 5, 5, 5, 5, 5, 14],
        [14, 14, 14, 14, 14, 14, 14, 14],
    ],
    dtype=np.int8,
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _render_xy(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return x * CELL_SIZE, PLAYFIELD_TOP + y * CELL_SIZE


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def cells(self) -> set[tuple[int, int]]:
        return {(x, y) for x in range(self.x0, self.x1 + 1) for y in range(self.y0, self.y1 + 1)}

    def contains(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class BarSpec:
    key: str
    state0: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    state1: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    initial_state: int

    def cells_for_state(self, state: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return self.state1 if state else self.state0


@dataclass(frozen=True)
class WinchSpec:
    key: str
    footprint: Rect
    bar_key: str


@dataclass(frozen=True)
class LevelSpec:
    budget: int
    floor_rects: tuple[Rect, ...]
    start: tuple[int, int]
    goal: Rect
    winches: tuple[WinchSpec, ...]
    bars: tuple[BarSpec, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=16,
        floor_rects=(Rect(6, 8, 8, 10), Rect(8, 6, 8, 7), Rect(8, 5, 11, 5), Rect(12, 8, 14, 10)),
        start=(8, 9),
        goal=Rect(12, 8, 13, 9),
        winches=(WinchSpec("W1", Rect(1, 2, 2, 3), "B1"),),
        bars=(BarSpec("B1", ((9, 4), (10, 4), (11, 4)), ((9, 9), (10, 9), (11, 9)), 0),),
    ),
    LevelSpec(
        budget=18,
        floor_rects=(Rect(6, 8, 8, 10), Rect(9, 5, 11, 6), Rect(12, 8, 14, 10)),
        start=(8, 9),
        goal=Rect(12, 8, 13, 9),
        winches=(WinchSpec("W1", Rect(1, 2, 2, 3), "B1"), WinchSpec("W2", Rect(1, 4, 2, 5), "B2")),
        bars=(
            BarSpec("B1", ((9, 5), (10, 5), (11, 5)), ((9, 9), (10, 9), (11, 9)), 0),
            BarSpec("B2", ((11, 9), (12, 9), (13, 9)), ((13, 9), (14, 9), (15, 9)), 0),
        ),
    ),
    LevelSpec(
        budget=30,
        floor_rects=(Rect(1, 8, 4, 12), Rect(4, 6, 5, 7), Rect(5, 5, 7, 5), Rect(8, 8, 8, 12), Rect(14, 8, 15, 11)),
        start=(4, 10),
        goal=Rect(14, 9, 15, 10),
        winches=(WinchSpec("W1", Rect(1, 2, 2, 3), "B1"), WinchSpec("W2", Rect(6, 11, 7, 12), "B2")),
        bars=(
            BarSpec("B1", ((5, 5), (6, 5), (7, 5)), ((5, 10), (6, 10), (7, 10)), 0),
            BarSpec("B2", ((9, 10), (10, 10), (11, 10)), ((11, 10), (12, 10), (13, 10)), 0),
        ),
    ),
)


class WinchLine(ARCBaseGame):
    def __init__(self) -> None:
        levels = [
            Level(
                name=f"Winch Line {index + 1}",
                grid_size=(GRID_SIZE, GRID_SIZE),
                sprites=[Sprite(_solid(GRID_SIZE, GRID_SIZE, COLOR_BG), name="screen", layer=0, collidable=False)],
                data={"level_index": index},
            )
            for index in range(len(LEVEL_SPECS))
        ]
        super().__init__(
            game_id="winch_line",
            levels=levels,
            camera=Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG, letter_box=COLOR_BG),
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 6],
        )
        self._route_score = 0
        self._screen: Sprite | None = None
        self._level_spec = LEVEL_SPECS[0]
        self._floor_cells: set[tuple[int, int]] = set()
        self._winch_cells: dict[str, set[tuple[int, int]]] = {}
        self._avatar = (0, 0)
        self._bar_states: dict[str, int] = {}
        self._budget_remaining = 0
        self._terminal_mode: str | None = None

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        self._level_spec = LEVEL_SPECS[level_index]
        self._screen = self.current_level.get_sprites_by_name("screen")[0]
        self._floor_cells = set()
        for rect in self._level_spec.floor_rects:
            self._floor_cells.update(rect.cells())
        self._winch_cells = {winch.key: winch.footprint.cells() for winch in self._level_spec.winches}
        self._avatar = self._level_spec.start
        self._bar_states = {bar.key: bar.initial_state for bar in self._level_spec.bars}
        self._budget_remaining = self._level_spec.budget
        self._terminal_mode = None
        self._render()

    def _active_bar_cells(self, *, exclude: str | None = None) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        for bar in self._level_spec.bars:
            if bar.key == exclude:
                continue
            occupied.update(bar.cells_for_state(self._bar_states[bar.key]))
        return occupied

    def _bar_by_key(self, key: str) -> BarSpec:
        for bar in self._level_spec.bars:
            if bar.key == key:
                return bar
        raise KeyError(key)

    def _walkable(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if not (0 <= x < LOGICAL_WIDTH and 0 <= y < LOGICAL_HEIGHT):
            return False
        active_bars = self._active_bar_cells()
        if cell in self._floor_cells:
            return cell not in active_bars and all(cell not in cells for cells in self._winch_cells.values())
        return cell in active_bars

    def _attempt_move(self, delta: tuple[int, int]) -> None:
        target = (self._avatar[0] + delta[0], self._avatar[1] + delta[1])
        if self._walkable(target):
            self._avatar = target

    def _clicked_winch(self, click_x: int, click_y: int) -> WinchSpec | None:
        if not (0 <= click_x < GRID_SIZE and 0 <= click_y < GRID_SIZE):
            return None
        if click_y < PLAYFIELD_TOP:
            return None
        cell = (click_x // CELL_SIZE, (click_y - PLAYFIELD_TOP) // CELL_SIZE)
        for winch in self._level_spec.winches:
            if winch.footprint.contains(cell):
                return winch
        return None

    def _attempt_toggle(self, winch: WinchSpec) -> None:
        bar = self._bar_by_key(winch.bar_key)
        current_state = self._bar_states[bar.key]
        current_cells = bar.cells_for_state(current_state)
        target_cells = bar.cells_for_state(1 - current_state)
        dx = target_cells[0][0] - current_cells[0][0]
        dy = target_cells[0][1] - current_cells[0][1]

        other_bars = self._active_bar_cells(exclude=bar.key)
        winch_footprints = set().union(*self._winch_cells.values())

        for cell in target_cells:
            x, y = cell
            if not (0 <= x < LOGICAL_WIDTH and 0 <= y < LOGICAL_HEIGHT):
                return
            if cell in winch_footprints:
                return
            if cell in other_bars:
                return

        avatar_on_bar = self._avatar in set(current_cells)
        if not avatar_on_bar and self._avatar in target_cells:
            return

        if avatar_on_bar:
            carried_destination = (self._avatar[0] + dx, self._avatar[1] + dy)
            if not (0 <= carried_destination[0] < LOGICAL_WIDTH and 0 <= carried_destination[1] < LOGICAL_HEIGHT):
                return
            if carried_destination not in target_cells:
                return
            if carried_destination in self._floor_cells:
                return

        self._bar_states[bar.key] = 1 - current_state
        if avatar_on_bar:
            self._avatar = (self._avatar[0] + dx, self._avatar[1] + dy)

    def _goal_reached(self) -> bool:
        return self._level_spec.goal.contains(self._avatar)

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

        if self._terminal_mode == "won":
            self._terminal_mode = None
            self.next_level()
            self.complete_action()
            return

        self._budget_remaining = max(0, self._budget_remaining - 1)

        action_id = int(self.action.id.value)
        if action_id == 1:
            self._attempt_move((0, -1))
        elif action_id == 2:
            self._attempt_move((0, 1))
        elif action_id == 3:
            self._attempt_move((-1, 0))
        elif action_id == 4:
            self._attempt_move((1, 0))
        elif action_id == 6:
            winch = self._clicked_winch(int(self.action.data.get("x", 0)), int(self.action.data.get("y", 0)))
            if winch is not None:
                self._attempt_toggle(winch)

        if self._goal_reached():
            self._route_score += 1
            if self.is_last_level():
                self._render(border="win")
                self.next_level()
                self.complete_action()
                return
            self._terminal_mode = "won"
            self._render(border="win")
            self.complete_action()
            return

        if self._budget_remaining == 0:
            self._render(border="fail")
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _render(self, border: str | None = None) -> None:
        frame = _solid(GRID_SIZE, GRID_SIZE, COLOR_BG)
        self._draw_budget(frame)
        self._draw_floor(frame)
        self._draw_inactive_slots(frame)
        self._draw_goal(frame)
        self._draw_cables(frame)
        self._draw_active_bars(frame)
        self._draw_winches(frame)
        self._draw_avatar(frame)
        if border == "win":
            self._draw_border(frame, COLOR_GOAL)
        elif border == "fail":
            self._draw_border(frame, COLOR_FAIL)
        if self._screen is not None:
            self._screen.pixels[:, :] = frame

    def _draw_budget(self, frame: np.ndarray) -> None:
        warning = self._budget_remaining <= 3
        for slot in range(20):
            if slot >= self._level_spec.budget:
                continue
            row = slot // 10
            col = slot % 10
            x0 = 12 + (col * 4)
            y0 = row * 4
            color = COLOR_GOAL if slot < self._budget_remaining else COLOR_SPENT
            if warning and slot < self._budget_remaining:
                color = COLOR_BAR_ACCENT
            frame[y0 : y0 + 4, x0 : x0 + 4] = color

    def _draw_floor(self, frame: np.ndarray) -> None:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for cell in self._floor_cells:
            px, py = _render_xy(cell)
            frame[py : py + 4, px : px + 4] = COLOR_FLOOR
            x, y = cell
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in self._floor_cells:
                    continue
                if dx == -1:
                    frame[py : py + 4, px] = COLOR_FLOOR_EDGE
                elif dx == 1:
                    frame[py : py + 4, px + 3] = COLOR_FLOOR_EDGE
                elif dy == -1:
                    frame[py, px : px + 4] = COLOR_FLOOR_EDGE
                else:
                    frame[py + 3, px : px + 4] = COLOR_FLOOR_EDGE

    def _draw_inactive_slots(self, frame: np.ndarray) -> None:
        for bar in self._level_spec.bars:
            inactive_state = 1 - self._bar_states[bar.key]
            for cell in bar.cells_for_state(inactive_state):
                px, py = _render_xy(cell)
                frame[py, px] = COLOR_SPENT
                frame[py + 1, px + 1] = COLOR_SPENT
                frame[py + 2, px + 2] = COLOR_SPENT
                frame[py + 3, px + 3] = COLOR_SPENT
                frame[py, px + 3] = COLOR_SPENT
                frame[py + 3, px] = COLOR_SPENT

    def _draw_goal(self, frame: np.ndarray) -> None:
        px, py = _render_xy((self._level_spec.goal.x0, self._level_spec.goal.y0))
        frame[py : py + 8, px : px + 8] = GOAL_TILE

    def _center_of_winch(self, winch: WinchSpec) -> tuple[int, int]:
        return (winch.footprint.x0 * 4 + 4, PLAYFIELD_TOP + winch.footprint.y0 * 4 + 4)

    def _center_of_bar(self, bar: BarSpec) -> tuple[int, int]:
        cells = bar.cells_for_state(self._bar_states[bar.key])
        return (cells[0][0] * 4 + 6, PLAYFIELD_TOP + cells[0][1] * 4 + 2)

    def _draw_cables(self, frame: np.ndarray) -> None:
        for winch in self._level_spec.winches:
            bar = self._bar_by_key(winch.bar_key)
            self._draw_line(frame, self._center_of_winch(winch), self._center_of_bar(bar), COLOR_CABLE)

    def _draw_line(self, frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= x0 < GRID_SIZE and 0 <= y0 < GRID_SIZE:
                frame[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _draw_active_bars(self, frame: np.ndarray) -> None:
        for bar in self._level_spec.bars:
            for cell in bar.cells_for_state(self._bar_states[bar.key]):
                px, py = _render_xy(cell)
                frame[py : py + 4, px : px + 4] = BAR_TILE

    def _draw_winches(self, frame: np.ndarray) -> None:
        for winch in self._level_spec.winches:
            px, py = _render_xy((winch.footprint.x0, winch.footprint.y0))
            frame[py : py + 8, px : px + 8] = WINCH_TILE

    def _draw_avatar(self, frame: np.ndarray) -> None:
        px, py = _render_xy(self._avatar)
        frame[py : py + 4, px : px + 4] = AVATAR_TILE

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[PLAYFIELD_TOP, :] = color
        frame[GRID_SIZE - 1, :] = color
        frame[PLAYFIELD_TOP:, 0] = color
        frame[PLAYFIELD_TOP:, GRID_SIZE - 1] = color

    def _get_hidden_state(self) -> np.ndarray:
        hidden = np.zeros(5, dtype=np.int16)
        hidden[0] = self._avatar[0]
        hidden[1] = self._avatar[1]
        hidden[2] = self._bar_states.get("B1", 0)
        hidden[3] = self._bar_states.get("B2", 0)
        hidden[4] = self._budget_remaining
        return hidden

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
            ActionInput(id=GameAction.ACTION5),
        ]
        for winch in self._level_spec.winches:
            click_x, click_y = self._center_of_winch(winch)
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": click_x, "y": click_y}))
        return actions
