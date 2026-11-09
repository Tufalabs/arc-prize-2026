from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

CELL_SIZE = 5
MAZE_ORIGIN_X = 2
MAZE_ORIGIN_Y = 4
GRID_WIDTH = 12
GRID_HEIGHT = 12

COLOR_WHITE = 0
COLOR_LIGHT_GRAY = 1
COLOR_DARK_GRAY = 3
COLOR_VERY_DARK_GRAY = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_PINK = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_PURPLE = 15

PLAYER_PATTERN = np.array(
    [[0, 9, 9, 9, 0], [9, 10, 10, 10, 9], [9, 10, 5, 10, 9], [9, 10, 10, 10, 9], [0, 9, 9, 9, 0]], dtype=np.int8
)
GOAL_PATTERN = np.array(
    [[0, 11, 11, 11, 0], [11, 12, 12, 12, 11], [11, 12, 5, 12, 11], [11, 12, 12, 12, 11], [0, 11, 11, 11, 0]],
    dtype=np.int8,
)
PORTAL_A_PATTERN = np.array(
    [[6, 6, 6, 6, 6], [6, 7, 0, 7, 6], [6, 0, 7, 0, 6], [6, 7, 0, 7, 6], [6, 6, 6, 6, 6]], dtype=np.int8
)
PORTAL_B_PATTERN = np.array(
    [[15, 15, 15, 15, 15], [15, 10, 0, 10, 15], [15, 0, 10, 0, 15], [15, 10, 0, 10, 15], [15, 15, 15, 15, 15]],
    dtype=np.int8,
)
FLASH_PLAYER_PATTERN = np.full((CELL_SIZE, CELL_SIZE), COLOR_BLACK, dtype=np.int8)
FLASH_GOAL_PATTERN = np.array(
    [[0, 12, 12, 12, 0], [12, 12, 12, 12, 12], [12, 12, 5, 12, 12], [12, 12, 12, 12, 12], [0, 12, 12, 12, 0]],
    dtype=np.int8,
)
FLOOR_PATTERN = np.full((CELL_SIZE, CELL_SIZE), COLOR_WHITE, dtype=np.int8)
WALL_PATTERN = np.full((CELL_SIZE, CELL_SIZE), COLOR_VERY_DARK_GRAY, dtype=np.int8)
FLASH_FLOOR_PATTERN = np.full((CELL_SIZE, CELL_SIZE), COLOR_RED, dtype=np.int8)
FLASH_WALL_PATTERN = np.full((CELL_SIZE, CELL_SIZE), COLOR_MAROON, dtype=np.int8)

MOVE_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


LEVEL_SPECS = (
    {
        "name": "Shortcut",
        "budget": 48,
        "rows": (
            "############",
            "#S...###...#",
            "###.####A###",
            "#...####.###",
            "#.#####..###",
            "#...A##..G##",
            "#######...##",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
    },
    {
        "name": "Two Outcomes",
        "budget": 42,
        "rows": (
            "############",
            "#S...###B..#",
            "#.######.###",
            "#...####...#",
            "###.########",
            "#...A###A..#",
            "###.####.###",
            "#...B###...#",
            "##########.#",
            "########..G#",
            "########..##",
            "############",
        ),
    },
    {
        "name": "False Near, True Far",
        "budget": 42,
        "rows": (
            "############",
            "############",
            "#######A..##",
            "#########.##",
            "#...A####.G#",
            "#.####....##",
            "#...##.##.##",
            "#.#.##....##",
            "#.#B##.#####",
            "#S####...B##",
            "############",
            "############",
        ),
    },
)


def _cell_to_pixels(cell_x: int, cell_y: int) -> tuple[int, int]:
    return MAZE_ORIGIN_X + (CELL_SIZE * cell_x), MAZE_ORIGIN_Y + (CELL_SIZE * cell_y)


def _make_cell_sprite(
    pattern: np.ndarray, *, name: str, cell_x: int, cell_y: int, layer: int, tags: list[str]
) -> Sprite:
    pixel_x, pixel_y = _cell_to_pixels(cell_x, cell_y)
    return Sprite(pattern.copy(), name=name, x=pixel_x, y=pixel_y, layer=layer, tags=tags, collidable=False)


def _parse_level(spec: dict[str, object]) -> Level:
    name = str(spec["name"])
    budget = int(spec["budget"])
    rows = tuple(str(row) for row in spec["rows"])
    sprites: list[Sprite] = []
    traversable: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    portal_cells: dict[str, list[tuple[int, int]]] = {}

    for cell_y, row in enumerate(rows):
        for cell_x, tile in enumerate(row):
            if tile == "#":
                sprites.append(
                    _make_cell_sprite(
                        WALL_PATTERN,
                        name=f"wall_{cell_x}_{cell_y}",
                        cell_x=cell_x,
                        cell_y=cell_y,
                        layer=0,
                        tags=["wall", "cell_base"],
                    )
                )
                continue

            traversable.add((cell_x, cell_y))
            sprites.append(
                _make_cell_sprite(
                    FLOOR_PATTERN,
                    name=f"floor_{cell_x}_{cell_y}",
                    cell_x=cell_x,
                    cell_y=cell_y,
                    layer=0,
                    tags=["floor", "cell_base"],
                )
            )

            if tile == "S":
                start = (cell_x, cell_y)
            elif tile == "G":
                goal = (cell_x, cell_y)
                sprites.append(
                    _make_cell_sprite(GOAL_PATTERN, name="goal", cell_x=cell_x, cell_y=cell_y, layer=1, tags=["goal"])
                )
            elif tile in {"A", "B"}:
                portal_cells.setdefault(tile, []).append((cell_x, cell_y))
                sprites.append(
                    _make_cell_sprite(
                        PORTAL_A_PATTERN if tile == "A" else PORTAL_B_PATTERN,
                        name=f"portal_{tile}_{len(portal_cells[tile]) - 1}",
                        cell_x=cell_x,
                        cell_y=cell_y,
                        layer=1,
                        tags=["portal", f"portal_{tile}"],
                    )
                )

    if start is None or goal is None:
        raise ValueError(f"Level {name} is missing S or G.")

    portal_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for portal_name, cells in portal_cells.items():
        if len(cells) != 2:
            raise ValueError(f"Portal {portal_name} in {name} must appear exactly twice.")
        first, second = cells
        portal_pairs[first] = second
        portal_pairs[second] = first

    sprites.append(
        _make_cell_sprite(PLAYER_PATTERN, name="player", cell_x=start[0], cell_y=start[1], layer=2, tags=["player"])
    )

    return Level(
        sprites=sprites,
        grid_size=(64, 64),
        data={
            "name": name,
            "rows": list(rows),
            "budget": budget,
            "start": start,
            "goal": goal,
            "traversable": sorted(traversable),
            "portal_pairs": {f"{x},{y}": [tx, ty] for (x, y), (tx, ty) in portal_pairs.items()},
        },
        name=name,
    )


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self, game: WarpHallways) -> None:
        self._game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame[0:4, 2:62] = COLOR_DARK_GRAY

        if self._game.fail_flash_active:
            frame[0:4, 2:62] = COLOR_RED
            return frame

        capacity = max(1, self._game.move_capacity)
        remaining = max(0, min(self._game.remaining_moves, capacity))
        fill = round((remaining / capacity) * 60)
        if remaining > capacity / 2:
            fill_color = COLOR_YELLOW
        elif remaining >= capacity / 4:
            fill_color = COLOR_ORANGE
        else:
            fill_color = COLOR_RED
        if fill > 0:
            frame[0:4, 2 : 2 + fill] = fill_color
        return frame


class WarpHallways(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._budget_display = MoveBudgetDisplay(self)
        self._player: Sprite | None = None
        self._portal_lookup: dict[tuple[int, int], tuple[int, int]] = {}
        self._traversable: set[tuple[int, int]] = set()
        self._start_cell = (0, 0)
        self._goal_cell = (0, 0)
        self._player_cell = (0, 0)
        self._move_capacity = 1
        self._remaining_moves = 1
        self._fail_flash = False
        super().__init__(
            "warp_hallways",
            [_parse_level(spec) for spec in LEVEL_SPECS],
            Camera(0, 0, 64, 64, COLOR_LIGHT_GRAY, COLOR_LIGHT_GRAY, [self._budget_display]),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6],
            seed,
        )

    @property
    def move_capacity(self) -> int:
        return int(self._move_capacity)

    @property
    def remaining_moves(self) -> int:
        return int(self._remaining_moves)

    @property
    def fail_flash_active(self) -> bool:
        return bool(self._fail_flash)

    def on_set_level(self, level: Level) -> None:
        self._move_capacity = int(level.get_data("budget"))
        self._remaining_moves = self._move_capacity
        self._fail_flash = False
        start_x, start_y = level.get_data("start")
        goal_x, goal_y = level.get_data("goal")
        self._start_cell = (int(start_x), int(start_y))
        self._goal_cell = (int(goal_x), int(goal_y))
        self._player_cell = self._start_cell
        self._traversable = {tuple(cell) for cell in level.get_data("traversable")}
        self._portal_lookup = {}
        for key, value in level.get_data("portal_pairs").items():
            cell_x, cell_y = (int(part) for part in str(key).split(","))
            target_x, target_y = int(value[0]), int(value[1])
            self._portal_lookup[(cell_x, cell_y)] = (target_x, target_y)
        self._player = level.get_sprites_by_name("player")[0]
        self._place_player_sprite()

    def _place_player_sprite(self) -> None:
        if self._player is None:
            return
        pixel_x, pixel_y = _cell_to_pixels(self._player_cell[0], self._player_cell[1])
        self._player.set_position(pixel_x, pixel_y)

    def _apply_failure_palette(self) -> None:
        for sprite in self.current_level.get_sprites_by_tag("wall"):
            sprite.pixels = FLASH_WALL_PATTERN.copy()
        for sprite in self.current_level.get_sprites_by_tag("floor"):
            sprite.pixels = FLASH_FLOOR_PATTERN.copy()
        for sprite in self.current_level.get_sprites_by_tag("goal"):
            sprite.pixels = FLASH_GOAL_PATTERN.copy()
        if self._player is not None:
            self._player.pixels = FLASH_PLAYER_PATTERN.copy()

    def _normalize_action_id(self) -> int:
        return int(getattr(self.action.id, "value", self.action.id))

    def _attempt_move(self, action_id: int) -> None:
        delta = MOVE_TO_DELTA.get(action_id)
        if delta is None:
            return

        target = (self._player_cell[0] + delta[0], self._player_cell[1] + delta[1])
        if target not in self._traversable:
            return

        self._player_cell = target
        if target in self._portal_lookup:
            self._player_cell = self._portal_lookup[target]

    def step(self) -> None:
        action_id = self._normalize_action_id()
        if action_id == 0:
            self.complete_action()
            return

        if self._fail_flash:
            self.lose()
            self.complete_action()
            return

        self._remaining_moves = max(0, self._remaining_moves - 1)
        self._attempt_move(action_id)
        self._place_player_sprite()

        if self._player_cell == self._goal_cell:
            self.next_level()
            self.complete_action()
            return

        if self._remaining_moves == 0:
            self._fail_flash = True
            self._apply_failure_palette()

        self.complete_action()
