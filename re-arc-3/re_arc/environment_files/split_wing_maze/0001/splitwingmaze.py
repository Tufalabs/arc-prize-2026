from __future__ import annotations

import heapq

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 64
GRID_HEIGHT = 64
BOARD_ORIGIN_X = 2
BOARD_ORIGIN_Y = 4
BOARD_SIZE = 10
CELL_SIZE = 6

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_SPENT = 3
COLOR_WALL = 4
COLOR_SEAM = 5
COLOR_MAGENTA = 6
COLOR_MAGENTA_LIGHT = 7
COLOR_FAILURE = 8
COLOR_BLUE = 9
COLOR_BLUE_LIGHT = 10
COLOR_SELECTION = 11
COLOR_BUDGET = 14

MOVE_ACTIONS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

SELECT_BLUE = "blue"
SELECT_MAGENTA = "magenta"
MODE_NORMAL = "normal"
MODE_FAILURE_FLASH = "failure_flash"
MODE_WIN_FLASH = "win_flash"
MODE_COMPLETED = "completed"


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _pixel_rect(cell: tuple[int, int]) -> tuple[int, int, int, int]:
    cell_x, cell_y = cell
    px = BOARD_ORIGIN_X + (cell_x * CELL_SIZE)
    py = BOARD_ORIGIN_Y + (cell_y * CELL_SIZE)
    return px, py, CELL_SIZE, CELL_SIZE


def _apply_pattern(frame: np.ndarray, cell: tuple[int, int], pattern: tuple[tuple[int | None, ...], ...]) -> None:
    px, py, _, _ = _pixel_rect(cell)
    for dy, row in enumerate(pattern):
        for dx, color in enumerate(row):
            if color is None:
                continue
            frame[py + dy, px + dx] = int(color)


def _parse_map(rows: tuple[str, ...]) -> tuple[frozenset[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    blue_home: tuple[int, int] | None = None
    magenta_home: tuple[int, int] | None = None
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))
            elif char == "A":
                blue_home = (x, y)
            elif char == "B":
                magenta_home = (x, y)
    if blue_home is None or magenta_home is None:
        raise ValueError("Each level must define both A and B start cells.")
    return frozenset(walls), blue_home, magenta_home


def _is_open(level: dict[str, object], cell: tuple[int, int]) -> bool:
    x, y = cell
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    return cell not in level["walls"]


def _minimum_directional_cost(level: dict[str, object]) -> int:
    blue_home = level["blue_home"]
    magenta_home = level["magenta_home"]
    start_state = (blue_home, magenta_home, SELECT_BLUE)
    frontier: list[tuple[int, tuple[tuple[int, int], tuple[int, int], str]]] = [(0, start_state)]
    best = {start_state: 0}

    while frontier:
        move_cost, state = heapq.heappop(frontier)
        blue_pos, magenta_pos, selected = state
        if move_cost != best.get(state):
            continue
        if blue_pos == magenta_home:
            return int(move_cost)

        other_selected = SELECT_MAGENTA if selected == SELECT_BLUE else SELECT_BLUE
        switched = (blue_pos, magenta_pos, other_selected)
        prior_switch_cost = best.get(switched)
        if prior_switch_cost is None or move_cost < prior_switch_cost:
            best[switched] = move_cost
            heapq.heappush(frontier, (move_cost, switched))

        for dx, dy in MOVE_ACTIONS.values():
            if selected == SELECT_BLUE:
                current = blue_pos
                blocker = magenta_pos
            else:
                current = magenta_pos
                blocker = blue_pos
            destination = (current[0] + dx, current[1] + dy)
            if not _is_open(level, destination) or destination == blocker:
                destination = current
            next_state = (
                destination if selected == SELECT_BLUE else blue_pos,
                destination if selected == SELECT_MAGENTA else magenta_pos,
                selected,
            )
            next_cost = move_cost + 1
            prior = best.get(next_state)
            if prior is None or next_cost < prior:
                best[next_state] = next_cost
                heapq.heappush(frontier, (next_cost, next_state))

    raise RuntimeError(f"Level {level['name']!r} is unsolvable.")


LEVEL_SPECS = (
    {
        "name": "Move the blocker, then run",
        "original_budget": 12,
        "rows": (
            "##########",
            "##########",
            "##A.######",
            "##.#######",
            "##.....###",
            "######.###",
            "######B.##",
            "##########",
            "##########",
            "##########",
        ),
    },
    {
        "name": "Dead end temptation",
        "original_budget": 14,
        "rows": (
            "##########",
            "##########",
            "##A#######",
            "##.#######",
            "##......##",
            "##.####.##",
            "##.####B.#",
            "##########",
            "##########",
            "##########",
        ),
    },
    {
        "name": "Passing bay",
        "original_budget": 18,
        "rows": (
            "##########",
            "##########",
            "#######B##",
            "##A.....##",
            "####..####",
            "####..####",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
    },
)


def _build_level_payloads() -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for spec in LEVEL_SPECS:
        rows = tuple(spec["rows"])
        walls, blue_home, magenta_home = _parse_map(rows)
        payload = {
            "name": str(spec["name"]),
            "rows": rows,
            "walls": walls,
            "blue_home": blue_home,
            "magenta_home": magenta_home,
            "blue_start": blue_home,
            "magenta_start": magenta_home,
            "original_budget": int(spec["original_budget"]),
        }
        optimal_directional_cost = _minimum_directional_cost(payload)
        payload["optimal_directional_cost"] = optimal_directional_cost
        payload["starting_budget"] = max(int(spec["original_budget"]), (optimal_directional_cost * 3) + 3)
        payloads.append(payload)
    return payloads


LEVEL_PAYLOADS = _build_level_payloads()


BLUE_HOME_PATTERN = (
    (COLOR_BLUE, None, None, None, None, COLOR_BLUE),
    (None, COLOR_BLUE, None, None, COLOR_BLUE, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, COLOR_BLUE, None, None, COLOR_BLUE, None),
    (COLOR_BLUE, None, None, None, None, COLOR_BLUE),
)

MAGENTA_HOME_PATTERN = (
    (COLOR_MAGENTA, None, None, None, None, COLOR_MAGENTA),
    (None, COLOR_MAGENTA, None, None, COLOR_MAGENTA, None),
    (None, None, COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT, None, None),
    (None, None, COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT, None, None),
    (None, COLOR_MAGENTA, None, None, COLOR_MAGENTA, None),
    (COLOR_MAGENTA, None, None, None, None, COLOR_MAGENTA),
)

BLUE_PAWN_PATTERN = (
    (None, None, None, None, None, None),
    (None, None, COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT, None, None),
    (None, COLOR_BLUE_LIGHT, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE_LIGHT, None),
    (None, COLOR_BLUE_LIGHT, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE_LIGHT, None),
    (None, None, COLOR_BLUE_LIGHT, COLOR_BLUE_LIGHT, None, None),
    (None, None, None, None, None, None),
)

MAGENTA_PAWN_PATTERN = (
    (None, None, None, None, None, None),
    (None, None, COLOR_MAGENTA_LIGHT, COLOR_MAGENTA_LIGHT, None, None),
    (None, COLOR_MAGENTA_LIGHT, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA_LIGHT, None),
    (None, COLOR_MAGENTA_LIGHT, COLOR_MAGENTA, COLOR_MAGENTA, COLOR_MAGENTA_LIGHT, None),
    (None, None, COLOR_MAGENTA_LIGHT, COLOR_MAGENTA_LIGHT, None, None),
    (None, None, None, None, None, None),
)

SELECTION_PATTERN = (
    (COLOR_SELECTION, None, None, None, None, COLOR_SELECTION),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (None, None, None, None, None, None),
    (COLOR_SELECTION, None, None, None, None, COLOR_SELECTION),
)


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for index, payload in enumerate(LEVEL_PAYLOADS):
        board = Sprite(
            _solid(GRID_WIDTH, GRID_HEIGHT, COLOR_WHITE),
            name="board",
            x=0,
            y=0,
            layer=0,
            tags=["board"],
            collidable=False,
        )
        levels.append(
            Level(
                name=f"split_wing_maze_level_{index + 1}",
                grid_size=(GRID_WIDTH, GRID_HEIGHT),
                sprites=[board],
                data=payload,
            )
        )
    return levels


class SplitWingMaze(ARCBaseGame):
    def __init__(self) -> None:
        self._selected_pawn = SELECT_BLUE
        self._blue_pos = (0, 0)
        self._magenta_pos = (0, 0)
        self._budget_remaining = 0
        self._budget_capacity = 0
        self._mode = MODE_NORMAL
        self._completed = False
        self._board_sprite: Sprite | None = None
        self._level_data: dict[str, object] = {}

        super().__init__(
            game_id="split_wing_maze-0001",
            levels=_build_levels(),
            camera=Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_WHITE),
            win_score=len(LEVEL_PAYLOADS),
            available_actions=[1, 2, 3, 4, 6],
        )

    def on_set_level(self, level: Level) -> None:
        self._level_data = {
            "name": str(level.get_data("name")),
            "rows": tuple(level.get_data("rows")),
            "walls": frozenset(tuple(cell) for cell in level.get_data("walls")),
            "blue_home": tuple(level.get_data("blue_home")),
            "magenta_home": tuple(level.get_data("magenta_home")),
            "blue_start": tuple(level.get_data("blue_start")),
            "magenta_start": tuple(level.get_data("magenta_start")),
            "optimal_directional_cost": int(level.get_data("optimal_directional_cost")),
            "starting_budget": int(level.get_data("starting_budget")),
        }
        boards = self.current_level.get_sprites_by_name("board")
        self._board_sprite = boards[0] if boards else None
        self._completed = False
        self._reset_level_state()

    def _reset_level_state(self) -> None:
        self._blue_pos = tuple(self._level_data["blue_start"])
        self._magenta_pos = tuple(self._level_data["magenta_start"])
        self._selected_pawn = SELECT_BLUE
        self._budget_capacity = int(self._level_data["starting_budget"])
        self._budget_remaining = int(self._level_data["starting_budget"])
        self._mode = MODE_NORMAL
        self._render_board()

    def _set_board_pixels(self, frame: np.ndarray) -> None:
        if self._board_sprite is None:
            raise RuntimeError("Board sprite was not initialized.")
        self._board_sprite.pixels = frame

    def _action_id(self) -> int:
        action = getattr(self, "action", None)
        action_id = getattr(action, "id", 0)
        value = getattr(action_id, "value", action_id)
        return int(value)

    def _click_point(self) -> tuple[int, int] | None:
        data = getattr(self.action, "data", {}) or {}
        try:
            x = int(data["x"])
            y = int(data["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return None
        return x, y

    def _pixel_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        if x < BOARD_ORIGIN_X or y < BOARD_ORIGIN_Y:
            return None
        cell_x = (x - BOARD_ORIGIN_X) // CELL_SIZE
        cell_y = (y - BOARD_ORIGIN_Y) // CELL_SIZE
        if not (0 <= cell_x < BOARD_SIZE and 0 <= cell_y < BOARD_SIZE):
            return None
        return cell_x, cell_y

    def _hit_selected_pawn(self, target: tuple[int, int]) -> str | None:
        if target == self._blue_pos:
            return SELECT_BLUE
        if target == self._magenta_pos:
            return SELECT_MAGENTA
        return None

    def _attempt_move(self, dx: int, dy: int) -> None:
        current = self._blue_pos if self._selected_pawn == SELECT_BLUE else self._magenta_pos
        blocker = self._magenta_pos if self._selected_pawn == SELECT_BLUE else self._blue_pos
        destination = (current[0] + dx, current[1] + dy)
        if not _is_open(self._level_data, destination) or destination == blocker:
            return
        if self._selected_pawn == SELECT_BLUE:
            self._blue_pos = destination
        else:
            self._magenta_pos = destination

    def _draw_floor_cell(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        px, py, _, _ = _pixel_rect(cell)
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_FLOOR
        frame[py, px] = COLOR_WHITE
        frame[py + CELL_SIZE - 1, px + CELL_SIZE - 1] = COLOR_WHITE

    def _draw_wall_cell(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        px, py, _, _ = _pixel_rect(cell)
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_WALL
        frame[py, px : px + CELL_SIZE] = COLOR_SEAM
        frame[py : py + CELL_SIZE, px] = COLOR_SEAM

    def _draw_budget(self, frame: np.ndarray) -> None:
        capacity = max(0, int(self._budget_capacity))
        if capacity <= 0:
            return
        total_width = min(capacity, GRID_WIDTH)
        start_x = max(0, (GRID_WIDTH - total_width) // 2)
        remaining = capacity if self._completed else max(0, int(self._budget_remaining))
        for index in range(total_width):
            color = COLOR_BUDGET if index < remaining else COLOR_SPENT
            frame[0:3, start_x + index] = color

    def _apply_flash_tint(self, frame: np.ndarray, tint: int) -> None:
        board = frame[
            BOARD_ORIGIN_Y : BOARD_ORIGIN_Y + (BOARD_SIZE * CELL_SIZE),
            BOARD_ORIGIN_X : BOARD_ORIGIN_X + (BOARD_SIZE * CELL_SIZE),
        ]
        board[board == COLOR_FLOOR] = tint
        frame[3, 1:63] = tint
        frame[63, 1:63] = tint
        frame[4:64, 1] = tint
        frame[4:64, 62] = tint

    def _apply_completed_border(self, frame: np.ndarray) -> None:
        frame[3, 1:63] = COLOR_BUDGET
        frame[63, 1:63] = COLOR_BUDGET
        frame[4:64, 1] = COLOR_BUDGET
        frame[4:64, 62] = COLOR_BUDGET

    def _render_board(self) -> None:
        frame = _solid(GRID_WIDTH, GRID_HEIGHT, COLOR_WHITE)
        rows = tuple(self._level_data["rows"])
        for y, row in enumerate(rows):
            for x, tile in enumerate(row):
                cell = (x, y)
                if tile == "#":
                    self._draw_wall_cell(frame, cell)
                else:
                    self._draw_floor_cell(frame, cell)

        _apply_pattern(frame, tuple(self._level_data["blue_home"]), BLUE_HOME_PATTERN)
        _apply_pattern(frame, tuple(self._level_data["magenta_home"]), MAGENTA_HOME_PATTERN)
        _apply_pattern(frame, self._blue_pos, BLUE_PAWN_PATTERN)
        _apply_pattern(frame, self._magenta_pos, MAGENTA_PAWN_PATTERN)

        if self._selected_pawn == SELECT_BLUE:
            _apply_pattern(frame, self._blue_pos, SELECTION_PATTERN)
        else:
            _apply_pattern(frame, self._magenta_pos, SELECTION_PATTERN)

        self._draw_budget(frame)

        if self._mode == MODE_FAILURE_FLASH:
            self._apply_flash_tint(frame, COLOR_FAILURE)
        elif self._mode == MODE_WIN_FLASH:
            self._apply_flash_tint(frame, COLOR_BUDGET)
        elif self._completed:
            self._apply_completed_border(frame)

        self._set_board_pixels(frame)

    def step(self) -> None:
        if self._mode == MODE_FAILURE_FLASH:
            self.lose()
            self.complete_action()
            return

        if self._mode == MODE_WIN_FLASH:
            if self.is_last_level():
                self._completed = True
                self._mode = MODE_COMPLETED
                self._render_board()
                self.next_level()
            else:
                self.next_level()
            self.complete_action()
            return

        action_id = self._action_id()
        if action_id in MOVE_ACTIONS:
            self._budget_remaining = max(0, self._budget_remaining - 1)
            dx, dy = MOVE_ACTIONS[action_id]
            self._attempt_move(dx, dy)
            if self._blue_pos == tuple(self._level_data["magenta_home"]):
                self._mode = MODE_WIN_FLASH
            elif self._budget_remaining == 0:
                self._mode = MODE_FAILURE_FLASH
        elif action_id == int(GameAction.ACTION6.value):
            point = self._click_point()
            if point is not None:
                clicked_cell = self._pixel_to_cell(point[0], point[1])
                if clicked_cell is not None:
                    clicked_pawn = self._hit_selected_pawn(clicked_cell)
                    if clicked_pawn is not None and clicked_pawn != self._selected_pawn:
                        self._selected_pawn = clicked_pawn

        self._render_board()
        self.complete_action()
