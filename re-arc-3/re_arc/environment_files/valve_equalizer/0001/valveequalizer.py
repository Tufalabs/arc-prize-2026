from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
UI_BOTTOM = 13
TANK_Y0 = 24
TANK_WIDTH = 12
TANK_HEIGHT = 26
INTERIOR_WIDTH = 10
INTERIOR_HEIGHT = 24
INTERIOR_BOTTOM_Y = TANK_Y0 + 24
WAIT_BUTTON_BBOX = (3, 3, 10, 10)
BUDGET_PIPS_X = 16
BUDGET_PIPS_Y = 4

COLOR_BG = 0
COLOR_TANK_INTERIOR = 1
COLOR_SPENT_PIP = 3
COLOR_FRAME = 4
COLOR_ACCENT = 5
COLOR_PURPLE_FLOAT_BOTTOM = 6
COLOR_CLOSED = 8
COLOR_WATER = 9
COLOR_SURFACE = 10
COLOR_TARGET_FLOAT_TOP = 11
COLOR_TARGET_FLOAT_BOTTOM = 12
COLOR_OPEN = 14
COLOR_GUARD_LEDGE = 15
COLOR_GUARD_FLOAT_TOP = 15

ACTION_IDS = [1, 2, 3, 4, 5, 6]
WAIT_ACTION_IDS = {1, 2, 3, 4, 5}

LEVEL_SPECS = (
    {
        "name": "Level 1",
        "tanks": (
            {"x0": 17, "float_kind": None, "target_level": None, "guard_min_level": None},
            {"x0": 35, "float_kind": "target", "target_level": 10, "guard_min_level": None},
        ),
        "valves": ({"left": 0, "right": 1, "bbox": (29, 41, 34, 46), "center": (31, 43)},),
        "initial_water": (12, 8),
        "initial_valves_open": (False,),
        "budget": 6,
    },
    {
        "name": "Level 2",
        "tanks": (
            {"x0": 4, "float_kind": None, "target_level": None, "guard_min_level": None},
            {"x0": 22, "float_kind": "target", "target_level": 11, "guard_min_level": None},
            {"x0": 40, "float_kind": None, "target_level": None, "guard_min_level": None},
        ),
        "valves": (
            {"left": 0, "right": 1, "bbox": (16, 41, 21, 46), "center": (18, 43)},
            {"left": 1, "right": 2, "bbox": (34, 41, 39, 46), "center": (36, 43)},
        ),
        "initial_water": (16, 8, 2),
        "initial_valves_open": (False, False),
        "budget": 10,
    },
    {
        "name": "Level 3",
        "tanks": (
            {"x0": 4, "float_kind": "guard", "target_level": None, "guard_min_level": 14},
            {"x0": 22, "float_kind": None, "target_level": None, "guard_min_level": None},
            {"x0": 40, "float_kind": "target", "target_level": 8, "guard_min_level": None},
        ),
        "valves": (
            {"left": 0, "right": 1, "bbox": (16, 41, 21, 46), "center": (18, 43)},
            {"left": 1, "right": 2, "bbox": (34, 41, 39, 46), "center": (36, 43)},
        ),
        "initial_water": (18, 10, 2),
        "initial_valves_open": (False, False),
        "budget": 36,
    },
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), np.int8(color), dtype=np.int8)


class ValveEqualizer(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_states = LEVEL_SPECS
        self._board_sprite: Sprite | None = None
        self._water_levels: list[int] = []
        self._valves_open: list[bool] = []
        self._remaining_budget = 0
        self._attempt_failed = False
        self._attempt_succeeded = False

        levels = [self._make_level(idx, spec) for idx, spec in enumerate(self._level_states)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG)
        super().__init__(
            game_id="valve_equalizer-0001",
            levels=levels,
            camera=camera,
            available_actions=ACTION_IDS,
            win_score=len(levels),
            seed=seed,
        )

    def _make_level(self, idx: int, spec: dict[str, object]) -> Level:
        board = Sprite(
            pixels=_solid(GRID_SIZE, GRID_SIZE, COLOR_BG),
            name="board",
            x=0,
            y=0,
            layer=0,
            tags=["board"],
            collidable=False,
        )
        return Level(
            name=f"{spec['name']}-{idx}", grid_size=(GRID_SIZE, GRID_SIZE), sprites=[board], data={"level_index": idx}
        )

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        spec = self._level_states[level_index]
        self._board_sprite = level.get_sprites_by_name("board")[0]
        self._water_levels = list(spec["initial_water"])
        self._valves_open = list(spec["initial_valves_open"])
        self._remaining_budget = int(spec["budget"])
        self._attempt_failed = False
        self._attempt_succeeded = False
        self._redraw()

    def _current_spec(self) -> dict[str, object]:
        return self._level_states[int(self.level_index)]

    def _point_in_bbox(self, x: int, y: int, bbox: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = bbox
        return x0 <= x <= x1 and y0 <= y <= y1

    def _toggle_valve_at(self, x: int, y: int) -> None:
        for idx, valve in enumerate(self._current_spec()["valves"]):
            if self._point_in_bbox(x, y, valve["bbox"]):
                self._valves_open[idx] = not self._valves_open[idx]
                return

    def _apply_tick(self) -> None:
        spec = self._current_spec()
        deltas = [0] * len(self._water_levels)
        before = tuple(self._water_levels)
        for is_open, valve in zip(self._valves_open, spec["valves"], strict=True):
            if not is_open:
                continue
            diff = before[int(valve["left"])] - before[int(valve["right"])]
            if diff >= 2:
                deltas[int(valve["left"])] -= 1
                deltas[int(valve["right"])] += 1
            elif diff <= -2:
                deltas[int(valve["left"])] += 1
                deltas[int(valve["right"])] -= 1

        next_levels: list[int] = []
        for water, delta in zip(before, deltas, strict=True):
            next_levels.append(max(0, min(INTERIOR_HEIGHT, water + delta)))
        self._water_levels = next_levels

    def _win_condition_met(self) -> bool:
        for idx, tank in enumerate(self._current_spec()["tanks"]):
            water = self._water_levels[idx]
            if tank["target_level"] is not None and water < int(tank["target_level"]):
                return False
            if tank["guard_min_level"] is not None and water < int(tank["guard_min_level"]):
                return False
        return True

    def _handle_active_step(self) -> None:
        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data or {}
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self._toggle_valve_at(x, y)

        self._remaining_budget -= 1
        self._apply_tick()

        if self._win_condition_met():
            self._attempt_succeeded = True
            self.next_level()
            return

        if self._remaining_budget <= 0:
            self._attempt_failed = True
            self.lose()

    def _redraw(self) -> None:
        if self._board_sprite is None:
            return
        board = _solid(GRID_SIZE, GRID_SIZE, COLOR_BG)
        self._draw_wait_affordance(board)
        self._draw_budget(board)

        spec = self._current_spec()
        for tank_index, tank in enumerate(spec["tanks"]):
            self._draw_tank(board, tank, self._water_levels[tank_index])
        for valve_index, valve in enumerate(spec["valves"]):
            self._draw_valve(board, valve, self._valves_open[valve_index])

        if self._attempt_failed:
            self._draw_border(board, COLOR_CLOSED)
        elif self._attempt_succeeded or getattr(getattr(self, "_state", None), "name", "") == "WIN":
            self._draw_border(board, COLOR_OPEN)

        self._board_sprite.pixels = board

    def _draw_wait_affordance(self, board: np.ndarray) -> None:
        x0, y0, x1, y1 = WAIT_BUTTON_BBOX
        board[y0 : y1 + 1, x0 : x1 + 1] = np.int8(COLOR_TARGET_FLOAT_BOTTOM)
        board[y0 + 1 : y1, x0 + 1 : x1] = np.int8(COLOR_TARGET_FLOAT_TOP)
        arrow = ((5, 4), (6, 4), (7, 4), (6, 5), (7, 5), (8, 5), (5, 6), (6, 6), (7, 6), (6, 7), (7, 7), (8, 7))
        for x, y in arrow:
            board[y, x] = np.int8(COLOR_ACCENT)

    def _draw_budget(self, board: np.ndarray) -> None:
        total = int(self._current_spec()["budget"])
        for idx in range(total):
            x0 = BUDGET_PIPS_X + idx * 3
            color = COLOR_TARGET_FLOAT_TOP if idx < self._remaining_budget else COLOR_SPENT_PIP
            board[BUDGET_PIPS_Y : BUDGET_PIPS_Y + 2, x0 : x0 + 2] = np.int8(color)

    def _draw_tank(self, board: np.ndarray, tank: dict[str, object], water_level: int) -> None:
        x0 = int(tank["x0"])
        board[TANK_Y0 + 1 : TANK_Y0 + TANK_HEIGHT, x0 : x0 + TANK_WIDTH] = np.int8(COLOR_FRAME)
        board[TANK_Y0 + 1 : TANK_Y0 + TANK_HEIGHT - 1, x0 + 1 : x0 + TANK_WIDTH - 1] = np.int8(COLOR_TANK_INTERIOR)

        if water_level > 0:
            water_top = INTERIOR_BOTTOM_Y - water_level + 1
            board[water_top : INTERIOR_BOTTOM_Y + 1, x0 + 1 : x0 + 11] = np.int8(COLOR_WATER)
            board[water_top, x0 + 1 : x0 + 11] = np.int8(COLOR_SURFACE)

        self._draw_float(board, tank, water_level)
        if tank["target_level"] is not None:
            self._draw_goal_ledge(board, tank, int(tank["target_level"]), right_side=True, color=COLOR_OPEN)
        if tank["guard_min_level"] is not None:
            self._draw_goal_ledge(board, tank, int(tank["guard_min_level"]), right_side=False, color=COLOR_GUARD_LEDGE)

    def _float_rows(self, water_level: int) -> tuple[int, int]:
        bottom = INTERIOR_BOTTOM_Y - water_level
        top = bottom - 1
        return top, bottom

    def _draw_float(self, board: np.ndarray, tank: dict[str, object], water_level: int) -> None:
        if tank["float_kind"] is None:
            return
        top, bottom = self._float_rows(water_level)
        x0 = int(tank["x0"]) + 3
        if tank["float_kind"] == "target":
            top_color = COLOR_TARGET_FLOAT_TOP
            bottom_color = COLOR_TARGET_FLOAT_BOTTOM
        else:
            top_color = COLOR_GUARD_FLOAT_TOP
            bottom_color = COLOR_PURPLE_FLOAT_BOTTOM
        board[top, x0 : x0 + 6] = np.int8(top_color)
        board[bottom, x0 : x0 + 6] = np.int8(bottom_color)

    def _draw_goal_ledge(
        self, board: np.ndarray, tank: dict[str, object], level: int, *, right_side: bool, color: int
    ) -> None:
        top, _bottom = self._float_rows(level)
        if right_side:
            x0 = int(tank["x0"]) + 8
        else:
            x0 = int(tank["x0"]) + 1
        board[top : top + 2, x0 : x0 + 3] = np.int8(color)

    def _draw_valve(self, board: np.ndarray, valve: dict[str, object], is_open: bool) -> None:
        x0, y0, x1, y1 = valve["bbox"]
        board[y0 : y1 + 1, x0 : x1 + 1] = np.int8(COLOR_FRAME)
        if is_open:
            board[y0 + 1 : y1, x0 + 1 : x1] = np.int8(COLOR_TANK_INTERIOR)
            board[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = np.int8(COLOR_OPEN)
            board[y1 - 1 : y1, x0 + 1 : x0 + 3] = np.int8(COLOR_OPEN)
            board[y0 + 1 : y0 + 3, x1 - 2 : x1] = np.int8(COLOR_OPEN)
            board[y1 - 1 : y1, x1 - 2 : x1] = np.int8(COLOR_OPEN)
        else:
            board[y0 + 1 : y1, x0 + 1 : x1] = np.int8(COLOR_CLOSED)

    def _draw_border(self, board: np.ndarray, color: int) -> None:
        board[0, :] = np.int8(color)
        board[-1, :] = np.int8(color)
        board[:, 0] = np.int8(color)
        board[:, -1] = np.int8(color)

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

        state_name = getattr(getattr(self, "_state", None), "name", "")
        if state_name in {"GAME_OVER", "WIN"}:
            self._redraw()
            self.complete_action()
            return

        self._handle_active_step()
        self._redraw()
        self.complete_action()
