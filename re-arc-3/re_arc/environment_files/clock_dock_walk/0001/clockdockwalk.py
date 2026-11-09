from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BACKGROUND_COLOR = 9
PADDING_COLOR = 9
CELL_SIZE = 7
PLAYFIELD_ORIGIN = (4, 15)

COLOR_WHITE = 0
COLOR_LIGHT_GRAY = 1
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_OUTLINE = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_LIGHT_MAGENTA = 7
COLOR_RED = 8
COLOR_WATER = 9
COLOR_LIGHT_WATER = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14

ACTION_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

PHASE_TO_BEACON = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def _action_id_value(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    if isinstance(value, tuple) and value:
        value = value[0]
    return int(value)


class LevelSpec:
    def __init__(
        self,
        *,
        start: tuple[int, int],
        walkable: frozenset[tuple[int, int]],
        dock: tuple[int, int],
        start_phase: int,
        accept_phase: int,
        max_actions: int,
        optimal_actions: int,
        name: str,
    ) -> None:
        self.start = start
        self.walkable = walkable
        self.dock = dock
        self.start_phase = int(start_phase)
        self.accept_phase = int(accept_phase)
        self.max_actions = int(max_actions)
        self.optimal_actions = int(optimal_actions)
        self.name = str(name)

    def to_data(self) -> dict[str, object]:
        return {
            "start": list(self.start),
            "walkable": [list(cell) for cell in sorted(self.walkable)],
            "dock": list(self.dock),
            "start_phase": int(self.start_phase),
            "accept_phase": int(self.accept_phase),
            "max_actions": int(self.max_actions),
            "optimal_actions": int(self.optimal_actions),
            "name": self.name,
        }

    @classmethod
    def from_data(cls, data: dict[str, object]) -> LevelSpec:
        walkable = frozenset(tuple(cell) for cell in data["walkable"])
        return cls(
            start=tuple(data["start"]),
            walkable=walkable,
            dock=tuple(data["dock"]),
            start_phase=int(data["start_phase"]),
            accept_phase=int(data["accept_phase"]),
            max_actions=int(data["max_actions"]),
            optimal_actions=int(data["optimal_actions"]),
            name=str(data["name"]),
        )


LEVEL_SPECS = [
    LevelSpec(
        name="See the beat",
        start=(1, 3),
        walkable=frozenset({(1, 3), (2, 3), (3, 3), (4, 3)}),
        dock=(5, 3),
        start_phase=0,
        accept_phase=0,
        max_actions=15,
        optimal_actions=5,
    ),
    LevelSpec(
        name="Longer route, same rules",
        start=(1, 5),
        walkable=frozenset({(1, 5), (1, 4), (1, 3), (2, 3), (3, 3), (4, 3), (4, 2), (5, 2)}),
        dock=(6, 2),
        start_phase=0,
        accept_phase=1,
        max_actions=30,
        optimal_actions=10,
    ),
    LevelSpec(
        name="Timing correction by route choice",
        start=(1, 4),
        walkable=frozenset({(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (6, 3), (4, 5), (5, 5), (6, 5)}),
        dock=(7, 3),
        start_phase=0,
        accept_phase=0,
        max_actions=27,
        optimal_actions=9,
    ),
]


def _make_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(LEVEL_SPECS):
        levels.append(Level([], (GRID_SIZE, GRID_SIZE), data={"spec": spec.to_data()}, name=f"Level {idx + 1}"))
    return levels


class ClockDockWalk(ARCBaseGame):
    def __init__(self) -> None:
        self._level_spec = LEVEL_SPECS[0]
        self._player = self._level_spec.start
        self._current_phase = self._level_spec.start_phase
        self._actions_used = 0
        self._dock_reject = False
        self._avatar_on_dock = False
        self._failed = False
        self._route_score = 0
        super().__init__(
            "clock_dock_walk-0001",
            _make_levels(),
            Camera(0, 0, GRID_SIZE, GRID_SIZE, BACKGROUND_COLOR, PADDING_COLOR),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5],
        )

    def on_set_level(self, level: Level) -> None:
        self._level_spec = LevelSpec.from_data(level.get_data("spec"))
        self._player = self._level_spec.start
        self._current_phase = self._level_spec.start_phase
        self._actions_used = 0
        self._dock_reject = False
        self._avatar_on_dock = False
        self._failed = False
        self._refresh_level_sprite()

    def _cell_origin(self, col: int, row: int) -> tuple[int, int]:
        return (PLAYFIELD_ORIGIN[0] + CELL_SIZE * col, PLAYFIELD_ORIGIN[1] + CELL_SIZE * row)

    def _fill_rect(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        x0 = max(0, min(GRID_SIZE - 1, int(x0)))
        y0 = max(0, min(GRID_SIZE - 1, int(y0)))
        x1 = max(0, min(GRID_SIZE - 1, int(x1)))
        y1 = max(0, min(GRID_SIZE - 1, int(y1)))
        if x0 > x1 or y0 > y1:
            return
        frame[y0 : y1 + 1, x0 : x1 + 1] = color

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[0, :] = color
        frame[-1, :] = color
        frame[:, 0] = color
        frame[:, -1] = color

    def _draw_walkway(self, frame: np.ndarray) -> None:
        for col, row in self._level_spec.walkable:
            x, y = self._cell_origin(col, row)
            self._fill_rect(frame, x + 1, y + 1, x + 5, y + 5, COLOR_ORANGE)
            if (col, row) == self._level_spec.start:
                self._fill_rect(frame, x + 2, y + 2, x + 4, y + 4, COLOR_GREEN)
            if (col + 1, row) in self._level_spec.walkable:
                self._fill_rect(frame, x + 4, y + 2, x + 6, y + 4, COLOR_ORANGE)
            if (col, row + 1) in self._level_spec.walkable:
                self._fill_rect(frame, x + 2, y + 4, x + 4, y + 6, COLOR_ORANGE)
            self._fill_rect(frame, x + 1, y + 1, x + 5, y + 1, COLOR_OUTLINE)
            self._fill_rect(frame, x + 1, y + 5, x + 5, y + 5, COLOR_OUTLINE)
            self._fill_rect(frame, x + 1, y + 1, x + 1, y + 5, COLOR_OUTLINE)
            self._fill_rect(frame, x + 5, y + 1, x + 5, y + 5, COLOR_OUTLINE)

    def _draw_dock(self, frame: np.ndarray) -> None:
        col, row = self._level_spec.dock
        x, y = self._cell_origin(col, row)
        platform_color = COLOR_RED if self._dock_reject else COLOR_LIGHT_GRAY
        beacon_color = COLOR_GREEN
        if self._avatar_on_dock:
            platform_color = COLOR_GREEN
            beacon_color = COLOR_GREEN
        elif self._current_phase == self._level_spec.accept_phase:
            beacon_color = COLOR_YELLOW

        self._fill_rect(frame, x + 1, y + 1, x + 5, y + 5, COLOR_OUTLINE)
        self._fill_rect(frame, x + 2, y + 2, x + 4, y + 4, platform_color)
        self._fill_rect(frame, x + 3, y + 3, x + 3, y + 3, COLOR_DARK_GRAY)

        dx, dy = PHASE_TO_BEACON[self._level_spec.accept_phase]
        if dx > 0:
            self._fill_rect(frame, x + 5, y + 2, x + 6, y + 4, beacon_color)
        elif dx < 0:
            self._fill_rect(frame, x + 0, y + 2, x + 1, y + 4, beacon_color)
        elif dy > 0:
            self._fill_rect(frame, x + 2, y + 5, x + 4, y + 6, beacon_color)
        else:
            self._fill_rect(frame, x + 2, y + 0, x + 4, y + 1, beacon_color)

    def _draw_avatar(self, frame: np.ndarray) -> None:
        col, row = self._level_spec.dock if self._avatar_on_dock else self._player
        x, y = self._cell_origin(col, row)
        avatar = np.array(
            [
                [-1, COLOR_LIGHT_MAGENTA, -1, COLOR_LIGHT_MAGENTA, -1],
                [-1, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA, -1],
                [-1, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA, -1],
                [COLOR_MAGENTA, -1, COLOR_MAGENTA, -1, COLOR_MAGENTA],
                [-1, COLOR_MAGENTA, -1, COLOR_MAGENTA, -1],
            ],
            dtype=np.int16,
        )
        for ay in range(5):
            for ax in range(5):
                color = int(avatar[ay, ax])
                if color >= 0:
                    frame[y + 1 + ay, x + 1 + ax] = color

    def _draw_clock(self, frame: np.ndarray) -> None:
        lamp_color = [COLOR_GRAY, COLOR_GRAY, COLOR_GRAY, COLOR_GRAY]
        lamp_color[self._current_phase] = COLOR_YELLOW

        self._fill_rect(frame, 31, 3, 33, 5, COLOR_OUTLINE)
        self._fill_rect(frame, 31, 1, 33, 2, lamp_color[0])
        self._fill_rect(frame, 34, 3, 35, 5, lamp_color[1])
        self._fill_rect(frame, 31, 6, 33, 7, lamp_color[2])
        self._fill_rect(frame, 29, 3, 30, 5, lamp_color[3])

    def _draw_countdown(self, frame: np.ndarray) -> None:
        self._fill_rect(frame, 4, 10, 59, 13, COLOR_OUTLINE)
        self._fill_rect(frame, 5, 11, 58, 12, COLOR_BLACK)

        remaining = max(0, self._level_spec.max_actions - self._actions_used)
        if remaining <= 0:
            return

        ratio = remaining / float(self._level_spec.max_actions)
        if ratio > 0.5:
            fill_color = COLOR_GREEN
        elif ratio > 0.25:
            fill_color = COLOR_YELLOW
        else:
            fill_color = COLOR_ORANGE
        width = round(54 * ratio)
        if width > 0:
            self._fill_rect(frame, 5, 11, 4 + width, 12, fill_color)

    def _draw_water_accents(self, frame: np.ndarray) -> None:
        accent_points = ((9, 7), (18, 5), (48, 6), (55, 8), (11, 28), (58, 24), (14, 55), (52, 51))
        for x, y in accent_points:
            self._fill_rect(frame, x, y, x + 1, y, COLOR_LIGHT_WATER)

    def _build_frame(self) -> np.ndarray:
        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_WATER, dtype=np.uint8)
        self._draw_water_accents(frame)
        self._draw_walkway(frame)
        self._draw_dock(frame)
        self._draw_clock(frame)
        self._draw_countdown(frame)
        self._draw_avatar(frame)
        if self._avatar_on_dock:
            self._draw_border(frame, COLOR_GREEN)
        elif self._failed:
            self._draw_border(frame, COLOR_MAROON)
            frame[1, :] = COLOR_RED
            frame[-2, :] = COLOR_RED
            frame[:, 1] = COLOR_RED
            frame[:, -2] = COLOR_RED
        return frame

    def _refresh_level_sprite(self) -> None:
        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(Sprite(self._build_frame(), name="frame", x=0, y=0, collidable=False))

    def _attempt_directional_move(self, action_id: int) -> bool:
        delta = ACTION_TO_DELTA.get(action_id)
        if delta is None:
            return False

        target = (self._player[0] + delta[0], self._player[1] + delta[1])
        if target == self._level_spec.dock:
            if self._current_phase == self._level_spec.accept_phase:
                self._avatar_on_dock = True
                return True
            self._dock_reject = True
            return False

        if target in self._level_spec.walkable:
            self._player = target
        return False

    def step(self) -> None:
        if self._state.name in {"GAME_OVER", "WIN"}:
            self.complete_action()
            return

        self._dock_reject = False
        action_id = _action_id_value(getattr(self.action, "id", 0))
        if action_id == _action_id_value(GameAction.RESET):
            self.complete_action()
            return

        success = False
        if action_id in ACTION_TO_DELTA:
            success = self._attempt_directional_move(action_id)
        elif action_id == 5:
            success = False

        self._actions_used += 1
        self._route_score = self._actions_used

        if not success:
            self._current_phase = (self._current_phase + 1) % 4

        if success:
            self._refresh_level_sprite()
            self.next_level()
            self.complete_action()
            return

        if self._actions_used >= self._level_spec.max_actions:
            self._failed = True
            self._refresh_level_sprite()
            self.lose()
            self.complete_action()
            return

        self._refresh_level_sprite()
        self.complete_action()
