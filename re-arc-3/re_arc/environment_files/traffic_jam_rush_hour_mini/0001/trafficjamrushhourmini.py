from __future__ import annotations

from math import ceil

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "traffic_jam_rush_hour_mini-0001"
WIDTH = 16
HEIGHT = 16
PLAYFIELD_TOP = 1
EXIT_CELLS: tuple[tuple[int, int], ...] = ((15, 8), (15, 9))
WIN_CELEBRATION_STEPS = 6

COLOR_EMPTY = 0
COLOR_WALL = 1
COLOR_EXIT = 2
COLOR_GOAL = 3
COLOR_TIMEBAR_FILL = 12
COLOR_INDICATOR = 13
COLOR_SELECTED = 14
COLOR_MOVING = 15


class CarSpec:
    __slots__ = ("axis", "color", "is_goal", "length", "symbol")

    def __init__(self, symbol: str, color: int, axis: str, length: int, is_goal: bool):
        self.symbol = str(symbol)
        self.color = int(color)
        self.axis = str(axis)
        self.length = int(length)
        self.is_goal = bool(is_goal)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _time_segments_to_ticks(segments: int) -> int:
    return max(0, int(segments) * 10)


def _level_specs() -> list[dict]:
    return [
        {
            "name": "Traffic Jam 1",
            "layout": [
                "#==============#",
                "################",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#.......+......#",
                "#..>>...+......]",
                "#.......+......]",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "+": 4},
            "timebar_segments": 14,
        },
        {
            "name": "Traffic Jam 2",
            "layout": [
                "#==============#",
                "################",
                "#..............#",
                "#....%.........#",
                "#....%.........#",
                "#....%.........#",
                "#.......***....#",
                "#.........+....#",
                "#..>>.....+....]",
                "#.........+....]",
                "#........--....#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "%": 4, "*": 5, "+": 6, "-": 7},
            "timebar_segments": 14,
        },
        {
            "name": "Traffic Jam 3",
            "layout": [
                "#==============#",
                "################",
                "#..............#",
                "#..@...........#",
                "#..@...........#",
                "#..@...........#",
                "#...:::........#",
                "#......+..%..!.#",
                "#..>>..+..%..!.]",
                "#......+..%&&..]",
                "#..............#",
                "#..............#",
                "#.....***~~~...#",
                "#..............#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "@": 4, ":": 5, "+": 6, "%": 7, "!": 8, "&": 9, "*": 10, "~": 11},
            "timebar_segments": 14,
        },
        {
            "name": "Traffic Jam 4",
            "layout": [
                "#==========....#",
                "################",
                "#..............#",
                "#..............#",
                "#.......#......#",
                "#......~~~.....#",
                "#.......%..:::.#",
                "#.....+.%...?..#",
                "#..>>.+.%.!.?..]",
                "#.....+.&&!....]",
                "#.........!..*.#",
                "#............*.#",
                "#.....#...#..*.#",
                "#..............#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "~": 4, "%": 5, ":": 6, "+": 7, "?": 8, "&": 9, "!": 10, "*": 11},
            "timebar_segments": 10,
        },
        {
            "name": "Traffic Jam 5",
            "layout": [
                "#========......#",
                "################",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#.....**..~~~.@#",
                "#......+...%..@#",
                "#..>>..+...%..@]",
                "#......+.!.%.&&]",
                "#........!:::..#",
                "#........!.....#",
                "#......#...#...#",
                "#..............#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "*": 4, "~": 5, "@": 6, "+": 7, "%": 8, "!": 9, ":": 10, "&": 11},
            "timebar_segments": 8,
        },
        {
            "name": "Traffic Jam 6",
            "layout": [
                "#========......#",
                "################",
                "#..............#",
                "#..............#",
                "#..............#",
                "#..............#",
                "#........%.~~~@#",
                "#.....+..%....@#",
                "#..>>.+..%..!.@]",
                "#.....+*....!.|]",
                "#......*:::.!.|#",
                "#......*......|#",
                "#.....#..#.....#",
                "#...........#..#",
                "#..............#",
                "################",
            ],
            "symbol_colors": {">": 3, "~": 4, "%": 5, "@": 6, "+": 7, "*": 8, ":": 9, "!": 10, "|": 11},
            "timebar_segments": 8,
        },
    ]


def _assert_layout(layout: list[str]) -> None:
    if len(layout) != HEIGHT:
        raise ValueError(f"Layout must have {HEIGHT} rows.")
    for row in layout:
        if len(row) != WIDTH:
            raise ValueError(f"Layout row must have {WIDTH} columns: {row!r}")


def _normalize_layout(layout: list[str]) -> list[str]:
    out = list(layout)
    _assert_layout(out)
    return out


def _extract_car_specs(layout: list[str], symbol_colors: dict[str, int]) -> tuple[list[CarSpec], list[tuple[int, int]]]:
    by_symbol: dict[str, list[tuple[int, int]]] = {}
    for y in range(PLAYFIELD_TOP, HEIGHT):
        row = layout[y]
        for x, cell in enumerate(row):
            if cell in {"#", ".", "]"}:
                continue
            by_symbol.setdefault(cell, []).append((x, y))

    car_specs: list[CarSpec] = []
    starts: list[tuple[int, int]] = []

    for symbol, cells in by_symbol.items():
        xs = sorted({x for x, _ in cells})
        ys = sorted({y for _, y in cells})
        if len(xs) > 1 and len(ys) > 1:
            raise ValueError(f"Car {symbol!r} is not axis-aligned.")
        axis = "h" if len(ys) == 1 else "v"

        if axis == "h":
            y0 = ys[0]
            expected = [(x, y0) for x in range(xs[0], xs[-1] + 1)]
        else:
            x0 = xs[0]
            expected = [(x0, y) for y in range(ys[0], ys[-1] + 1)]
        if sorted(expected) != sorted(cells):
            raise ValueError(f"Car {symbol!r} must be contiguous.")

        length = len(expected)
        if length not in {2, 3}:
            raise ValueError(f"Car {symbol!r} has invalid length {length}; expected 2 or 3.")

        color = int(symbol_colors.get(symbol, -1))
        if color < 3 or color > 11:
            raise ValueError(f"Car {symbol!r} has invalid color mapping: {color}")

        is_goal = symbol == ">"
        if is_goal and color != COLOR_GOAL:
            raise ValueError("Goal car must use color 3.")

        start = (xs[0], y0) if axis == "h" else (x0, ys[0])

        car_specs.append(CarSpec(symbol=symbol, color=color, axis=axis, length=length, is_goal=is_goal))
        starts.append(start)

    if sum(1 for spec in car_specs if spec.is_goal) != 1:
        raise ValueError("Exactly one goal car ('>') is required.")

    order = sorted(range(len(car_specs)), key=lambda idx: (car_specs[idx].color, car_specs[idx].symbol))
    return [car_specs[idx] for idx in order], [starts[idx] for idx in order]


def _cells_for(spec: CarSpec, pos: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = int(pos[0]), int(pos[1])
    if spec.axis == "h":
        return [(x0 + i, y0) for i in range(spec.length)]
    return [(x0, y0 + i) for i in range(spec.length)]


def _wall_and_exit_cells(layout: list[str]) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    walls: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    for y in range(PLAYFIELD_TOP, HEIGHT):
        for x, cell in enumerate(layout[y]):
            if cell == "#":
                walls.add((x, y))
            elif cell == "]":
                exits.add((x, y))

    expected_exits = set(EXIT_CELLS)
    if exits != expected_exits:
        raise ValueError(f"Exit doorway must be exactly {sorted(expected_exits)}; got {sorted(exits)}")
    return frozenset(walls), frozenset(exits)


def _serialize_car_specs(specs: list[CarSpec]) -> list[dict]:
    return [
        {
            "symbol": spec.symbol,
            "color": int(spec.color),
            "axis": spec.axis,
            "length": int(spec.length),
            "is_goal": bool(spec.is_goal),
        }
        for spec in specs
    ]


def _deserialize_car_specs(rows: list[dict]) -> list[CarSpec]:
    return [
        CarSpec(
            symbol=str(row["symbol"]),
            color=int(row["color"]),
            axis=str(row["axis"]),
            length=int(row["length"]),
            is_goal=bool(row["is_goal"]),
        )
        for row in rows
    ]


def _build_level(spec: dict) -> Level:
    layout = _normalize_layout(spec["layout"])
    car_specs, starts = _extract_car_specs(layout, spec["symbol_colors"])
    walls, exits = _wall_and_exit_cells(layout)

    board = Sprite(pixels=_solid(WIDTH, HEIGHT, COLOR_EMPTY), name="board", x=0, y=0, layer=0, collidable=False)

    return Level(
        name=str(spec["name"]),
        grid_size=(WIDTH, HEIGHT),
        sprites=[board],
        data={
            "timer_ticks": _time_segments_to_ticks(int(spec["timebar_segments"])),
            "walls": sorted((int(x), int(y)) for x, y in walls),
            "exits": sorted((int(x), int(y)) for x, y in exits),
            "cars": _serialize_car_specs(car_specs),
            "starts": [(int(x), int(y)) for x, y in starts],
        },
    )


def _build_levels() -> list[Level]:
    return [_build_level(spec) for spec in _level_specs()]


def _flatten_positions(positions: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    out: list[int] = []
    for x, y in positions:
        out.extend((int(x), int(y)))
    return tuple(out)


def _unflatten_positions(state: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(state) % 2 != 0:
        raise ValueError("Invalid state length.")
    return tuple((int(state[i]), int(state[i + 1])) for i in range(0, len(state), 2))


def _deserialize_level_model(level: Level) -> dict:
    return {
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "walls": frozenset(tuple(int(v) for v in row) for row in (level.get_data("walls") or [])),
        "exits": frozenset(tuple(int(v) for v in row) for row in (level.get_data("exits") or [])),
        "car_specs": tuple(_deserialize_car_specs(level.get_data("cars") or [])),
        "starts": tuple((int(x), int(y)) for x, y in (level.get_data("starts") or [])),
        "timer_ticks": int(level.get_data("timer_ticks") or 0),
    }


def initial_search_state_from_model(model: dict) -> tuple[int, ...]:
    starts = tuple((int(x), int(y)) for x, y in model["starts"])
    return _flatten_positions(starts)


def _occupancy(car_specs: tuple[CarSpec, ...], positions: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    occ: dict[tuple[int, int], int] = {}
    for idx, spec in enumerate(car_specs):
        for cell in _cells_for(spec, positions[idx]):
            occ[cell] = idx
    return occ


def _is_valid_position(
    *, model: dict, car_idx: int, candidate_pos: tuple[int, int], positions: tuple[tuple[int, int], ...]
) -> bool:
    car_specs: tuple[CarSpec, ...] = model["car_specs"]
    walls: frozenset[tuple[int, int]] = model["walls"]
    exits: frozenset[tuple[int, int]] = model["exits"]
    width = int(model["width"])
    height = int(model["height"])

    occ = _occupancy(car_specs, positions)
    spec = car_specs[car_idx]
    for cell in _cells_for(spec, candidate_pos):
        x, y = cell
        if x < 0 or y < PLAYFIELD_TOP or x >= width or y >= height:
            return False
        if cell in walls:
            return False
        if cell in exits and not spec.is_goal:
            return False
        blocker = occ.get(cell)
        if blocker is not None and blocker != car_idx:
            return False
    return True


def _goal_reached(model: dict, positions: tuple[tuple[int, int], ...]) -> bool:
    car_specs: tuple[CarSpec, ...] = model["car_specs"]
    exits: frozenset[tuple[int, int]] = model["exits"]
    for idx, spec in enumerate(car_specs):
        if not spec.is_goal:
            continue
        return any(cell in exits for cell in _cells_for(spec, positions[idx]))
    return False


def apply_slide_transition(
    model: dict, state: tuple[int, ...], car_idx: int, dx: int, dy: int
) -> tuple[tuple[int, ...] | None, int]:
    positions = list(_unflatten_positions(state))
    car_specs: tuple[CarSpec, ...] = model["car_specs"]
    if car_idx < 0 or car_idx >= len(car_specs):
        return None, 0

    spec = car_specs[car_idx]
    if spec.axis == "h" and dy != 0:
        return None, 0
    if spec.axis == "v" and dx != 0:
        return None, 0

    moved = 0
    while True:
        x, y = positions[car_idx]
        candidate = (int(x + dx), int(y + dy))
        if not _is_valid_position(model=model, car_idx=car_idx, candidate_pos=candidate, positions=tuple(positions)):
            break
        positions[car_idx] = candidate
        moved += 1

    if moved <= 0:
        return None, 0

    return _flatten_positions(tuple(positions)), moved


def iter_search_moves(model: dict, state: tuple[int, ...]):
    car_specs: tuple[CarSpec, ...] = model["car_specs"]
    for idx, spec in enumerate(car_specs):
        dirs = ((-1, 0), (1, 0)) if spec.axis == "h" else ((0, -1), (0, 1))
        for dx, dy in dirs:
            next_state, distance = apply_slide_transition(model, state, idx, dx, dy)
            if next_state is None or distance <= 0:
                continue
            yield (idx, dx, dy, distance), next_state, float(2 + distance)


def is_goal_state(model: dict, state: tuple[int, ...]) -> bool:
    return _goal_reached(model, _unflatten_positions(state))


class TrafficJamRushHourMini(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = _build_levels()
        camera = Camera(width=WIDTH, height=HEIGHT, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID, levels=levels, camera=camera, win_score=len(levels), available_actions=[5, 6], seed=seed
        )

        self._board_sprite: Sprite | None = None
        self._walls: frozenset[tuple[int, int]] = frozenset()
        self._exits: frozenset[tuple[int, int]] = frozenset(EXIT_CELLS)
        self._car_specs: list[CarSpec] = []
        self._car_positions: list[tuple[int, int]] = []
        self._timer_ticks = 0
        self._selected_idx: int | None = None
        self._sliding_idx: int | None = None
        self._sliding_dir: tuple[int, int] = (0, 0)
        self._pending_click: tuple[int, int] | None = None
        self._blink_on = False
        self._mode = "running"
        self._celebration_ticks = 0

    def on_set_level(self, level: Level) -> None:
        model = _deserialize_level_model(level)
        self._walls = model["walls"]
        self._exits = model["exits"]
        self._car_specs = list(model["car_specs"])
        self._car_positions = list(model["starts"])
        self._timer_ticks = int(model["timer_ticks"])
        self._selected_idx = None
        self._sliding_idx = None
        self._sliding_dir = (0, 0)
        self._pending_click = None
        self._blink_on = False
        self._mode = "running"
        self._celebration_ticks = 0

        boards = self.current_level.get_sprites_by_name("board")
        self._board_sprite = boards[0] if boards else None
        self._sync_board()

    def _positions_tuple(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(x), int(y)) for x, y in self._car_positions)

    def _model_snapshot(self) -> dict:
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "walls": self._walls,
            "exits": self._exits,
            "car_specs": tuple(self._car_specs),
            "starts": self._positions_tuple(),
        }

    def _occupied_cells(self) -> dict[tuple[int, int], int]:
        return _occupancy(tuple(self._car_specs), self._positions_tuple())

    def _can_shift(self, car_idx: int, dx: int, dy: int) -> bool:
        current = self._car_positions[car_idx]
        candidate = (int(current[0] + dx), int(current[1] + dy))
        return _is_valid_position(
            model=self._model_snapshot(), car_idx=car_idx, candidate_pos=candidate, positions=self._positions_tuple()
        )

    def _indicator_cells(self, car_idx: int) -> dict[tuple[int, int], tuple[int, int]]:
        if car_idx < 0 or car_idx >= len(self._car_specs):
            return {}
        spec = self._car_specs[car_idx]
        x, y = self._car_positions[car_idx]
        out: dict[tuple[int, int], tuple[int, int]] = {}

        if spec.axis == "h":
            if self._can_shift(car_idx, -1, 0):
                out[(int(x - 1), int(y))] = (-1, 0)
            if self._can_shift(car_idx, 1, 0):
                out[(int(x + spec.length), int(y))] = (1, 0)
        else:
            if self._can_shift(car_idx, 0, -1):
                out[(int(x), int(y - 1))] = (0, -1)
            if self._can_shift(car_idx, 0, 1):
                out[(int(x), int(y + spec.length))] = (0, 1)

        return out

    def _car_at(self, x: int, y: int) -> int | None:
        target = (int(x), int(y))
        for idx, spec in enumerate(self._car_specs):
            if target in _cells_for(spec, self._car_positions[idx]):
                return idx
        return None

    def _set_mode_failed(self) -> None:
        self._mode = "failed"
        self._selected_idx = None
        self._sliding_idx = None
        self._sliding_dir = (0, 0)
        self._pending_click = None

    def _lose_level(self) -> None:
        self._set_mode_failed()
        self._sync_board()
        self.lose()

    def _set_mode_win(self) -> None:
        self._mode = "win"
        self._celebration_ticks = WIN_CELEBRATION_STEPS
        self._selected_idx = None
        self._sliding_idx = None
        self._sliding_dir = (0, 0)
        self._pending_click = None

    def _goal_touches_exit(self) -> bool:
        model = self._model_snapshot()
        return _goal_reached(model, self._positions_tuple())

    def _handle_pending_click(self, click: tuple[int, int]) -> None:
        if self._mode != "running":
            return
        if self._sliding_idx is not None:
            return

        x, y = int(click[0]), int(click[1])
        car_idx = self._car_at(x, y)
        if car_idx is not None:
            self._selected_idx = car_idx
            return

        if self._selected_idx is not None:
            indicators = self._indicator_cells(self._selected_idx)
            direction = indicators.get((x, y))
            if direction is not None:
                self._sliding_idx = int(self._selected_idx)
                self._sliding_dir = (int(direction[0]), int(direction[1]))
                self._selected_idx = None
                return

        self._selected_idx = None

    def _step_sliding(self) -> None:
        if self._sliding_idx is None:
            return

        idx = int(self._sliding_idx)
        dx, dy = int(self._sliding_dir[0]), int(self._sliding_dir[1])
        if not self._can_shift(idx, dx, dy):
            self._sliding_idx = None
            self._sliding_dir = (0, 0)
            return

        x, y = self._car_positions[idx]
        self._car_positions[idx] = (int(x + dx), int(y + dy))

        if not self._can_shift(idx, dx, dy):
            self._sliding_idx = None
            self._sliding_dir = (0, 0)

    def _timebar_segments(self) -> int:
        if self._timer_ticks <= 0:
            return 0
        return int(max(0, min(14, ceil(self._timer_ticks / 10.0))))

    def _sync_board(self) -> None:
        if self._board_sprite is None:
            return

        grid = np.full((HEIGHT, WIDTH), COLOR_EMPTY, dtype=np.int8)

        grid[0, 0] = COLOR_WALL
        grid[0, WIDTH - 1] = COLOR_WALL
        segments = self._timebar_segments()
        for i in range(14):
            x = 1 + i
            grid[0, x] = COLOR_TIMEBAR_FILL if i < segments else COLOR_EMPTY

        for x, y in self._walls:
            grid[int(y), int(x)] = COLOR_WALL

        exit_color = COLOR_EXIT
        if self._mode == "win" and self._blink_on:
            exit_color = COLOR_MOVING
        for x, y in self._exits:
            grid[int(y), int(x)] = int(exit_color)

        if self._mode == "running" and self._selected_idx is not None and self._sliding_idx is None:
            for ix, iy in self._indicator_cells(self._selected_idx):
                if 0 <= ix < WIDTH and PLAYFIELD_TOP <= iy < HEIGHT and self._car_at(ix, iy) is None:
                    grid[int(iy), int(ix)] = COLOR_INDICATOR

        for idx, spec in enumerate(self._car_specs):
            color = int(spec.color)
            if self._mode == "failed":
                if self._blink_on:
                    color = COLOR_MOVING
            elif self._mode == "win":
                if spec.is_goal and self._blink_on:
                    color = COLOR_MOVING
            else:
                if self._sliding_idx == idx and self._blink_on:
                    color = COLOR_MOVING
                elif self._selected_idx == idx and self._blink_on:
                    color = COLOR_SELECTED

            for x, y in _cells_for(spec, self._car_positions[idx]):
                if 0 <= x < WIDTH and PLAYFIELD_TOP <= y < HEIGHT:
                    grid[int(y), int(x)] = np.int8(color)

        self._board_sprite.pixels = grid

    def _capture_click_for_next_step(self) -> tuple[int, int] | None:
        if self._mode != "running":
            return None
        if self._sliding_idx is not None:
            return None
        if self.action.id != GameAction.ACTION6:
            return None

        payload = self.action.data if isinstance(self.action.data, dict) else {}
        dx = int(payload.get("x", -1))
        dy = int(payload.get("y", -1))
        grid_pos = self.camera.display_to_grid(dx, dy)
        if grid_pos is None:
            return None
        x, y = int(grid_pos[0]), int(grid_pos[1])
        if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
            return None
        return x, y

    def step(self) -> None:
        action_id = int(self.action.id.value)

        if action_id == int(GameAction.ACTION5.value):
            self.level_reset()
            self.complete_action()
            return

        if self._mode == "win":
            self._blink_on = not self._blink_on
            self._celebration_ticks -= 1
            if self._celebration_ticks <= 0:
                self.next_level()
                self.complete_action()
                return
            self._sync_board()
            self.complete_action()
            return

        queued_click = self._capture_click_for_next_step()

        if self._pending_click is not None:
            self._handle_pending_click(self._pending_click)

        self._step_sliding()

        if self._goal_touches_exit():
            self._set_mode_win()
        else:
            self._timer_ticks -= 1
            if self._timer_ticks <= 0:
                self._timer_ticks = 0
                self._lose_level()
                self.complete_action()
                return

        self._blink_on = not self._blink_on

        if self._mode == "running" and self._sliding_idx is None:
            self._pending_click = queued_click
        else:
            self._pending_click = None

        self._sync_board()
        self.complete_action()
