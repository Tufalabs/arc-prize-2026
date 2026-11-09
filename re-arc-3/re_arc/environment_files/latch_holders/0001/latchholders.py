from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

SCREEN_SIZE = 64
BOARD_SIZE = 12
CELL_SIZE = 4
BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 8
HUD_X0 = BOARD_ORIGIN_X
HUD_X1 = BOARD_ORIGIN_X + (BOARD_SIZE * CELL_SIZE) - 1

COLOR_FLOOR = 0
COLOR_FLOOR_ACCENT = 1
COLOR_WALL_FILL = 4
COLOR_OUTLINE = 5
COLOR_LATCH_3 = 6
COLOR_FAIL = 8
COLOR_BLUE = 9
COLOR_HINT = 10
COLOR_LATCH_1 = 11
COLOR_ORANGE = 12
COLOR_SUCCESS = 14
COLOR_LATCH_2 = 14
COLOR_SELECTION = 15

PAIR_COLORS = {1: COLOR_LATCH_1, 2: COLOR_LATCH_2, 3: COLOR_LATCH_3}
PAWN_COLORS = {"blue": COLOR_BLUE, "orange": COLOR_ORANGE}
PAWN_ORDER = ("blue", "orange")
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
MODE_PLAYING = "playing"
MODE_LEVEL_WON = "level_won"
MODE_LEVEL_FAILED = "level_failed"


def _solid(color: int) -> np.ndarray:
    return np.full((SCREEN_SIZE, SCREEN_SIZE), int(color), dtype=np.int8)


def _cell_to_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_ORIGIN_X + (cell[0] * CELL_SIZE), BOARD_ORIGIN_Y + (cell[1] * CELL_SIZE)


def _wall_set(*extra_walls: tuple[int, int]) -> frozenset[tuple[int, int]]:
    walls = {
        (x, y)
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if x == 0 or x == BOARD_SIZE - 1 or y == 0 or y == BOARD_SIZE - 1
    }
    walls.update(extra_walls)
    return frozenset(walls)


LEVEL_SPECS = (
    {
        "budget": 24,
        "initial_selected": "orange",
        "blue_start": (2, 8),
        "orange_start": (4, 5),
        "walls": _wall_set(*((5, y) for y in range(1, 11) if y != 5)),
        "latches": {1: (2, 5)},
        "doors": {1: (5, 5)},
        "orange_goal": (9, 5),
        "blue_goal": None,
        "final_blue_goal": None,
        "final_orange_goal": None,
    },
    {
        "budget": 45,
        "initial_selected": "blue",
        "blue_start": (3, 9),
        "orange_start": (2, 3),
        "walls": _wall_set(
            *((x, 6) for x in range(1, 11)),
            *((4, y) for y in range(1, 6) if y != 3),
            *((8, y) for y in range(7, 11) if y != 8),
        ),
        "latches": {1: (2, 8), 2: (7, 3)},
        "doors": {1: (4, 3), 2: (8, 8)},
        "orange_goal": None,
        "blue_goal": (9, 9),
        "final_blue_goal": None,
        "final_orange_goal": None,
    },
    {
        "budget": 93,
        "initial_selected": "blue",
        "blue_start": (3, 9),
        "orange_start": (2, 2),
        "walls": _wall_set(
            *((x, 6) for x in range(1, 9)),
            *((4, y) for y in range(1, 6) if y != 2),
            *((8, y) for y in range(1, 6) if y != 4),
            *((8, y) for y in range(7, 11) if y != 8),
        ),
        "latches": {1: (2, 8), 2: (6, 2), 3: (9, 7)},
        "doors": {1: (4, 2), 2: (8, 8), 3: (8, 4)},
        "orange_goal": None,
        "blue_goal": None,
        "final_blue_goal": (10, 2),
        "final_orange_goal": (10, 9),
    },
)


class LatchHolders(ARCBaseGame):
    _canvas: Sprite
    _spec: dict[str, object]
    _positions: dict[str, tuple[int, int]]
    _selected: str | None
    _mode: str
    _remaining_moves: int

    def __init__(self, seed: int = 0) -> None:
        levels = []
        for index in range(len(LEVEL_SPECS)):
            levels.append(
                Level(
                    name=f"level_{index + 1}",
                    grid_size=(SCREEN_SIZE, SCREEN_SIZE),
                    sprites=[
                        Sprite(_solid(COLOR_FLOOR), name="canvas", x=0, y=0, layer=0, tags=["canvas"], collidable=False)
                    ],
                    data={"level_index": index},
                )
            )
        super().__init__(
            "latch_holders",
            levels=levels,
            camera=Camera(0, 0, SCREEN_SIZE, SCREEN_SIZE, COLOR_FLOOR, COLOR_FLOOR),
            win_score=len(levels),
            available_actions=[6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._canvas = level.get_sprites_by_name("canvas")[0]
        level_index = int(level.get_data("level_index"))
        self._spec = LEVEL_SPECS[level_index]
        self._positions = {"blue": self._spec["blue_start"], "orange": self._spec["orange_start"]}
        self._selected = self._spec["initial_selected"]
        self._mode = MODE_PLAYING
        self._remaining_moves = int(self._spec["budget"])
        self._render()

    def _click_to_cell(self, x: int | None, y: int | None) -> tuple[int, int] | None:
        if x is None or y is None:
            return None
        if not (BOARD_ORIGIN_X <= x < BOARD_ORIGIN_X + (BOARD_SIZE * CELL_SIZE)):
            return None
        if not (BOARD_ORIGIN_Y <= y < BOARD_ORIGIN_Y + (BOARD_SIZE * CELL_SIZE)):
            return None
        return ((int(x) - BOARD_ORIGIN_X) // CELL_SIZE, (int(y) - BOARD_ORIGIN_Y) // CELL_SIZE)

    def _door_is_open(self, pair_id: int) -> bool:
        latch_cell = self._spec["latches"][pair_id]
        door_cell = self._spec["doors"][pair_id]
        return any(position == latch_cell or position == door_cell for position in self._positions.values())

    def _pawn_at(self, cell: tuple[int, int]) -> str | None:
        for pawn_name in PAWN_ORDER:
            if self._positions[pawn_name] == cell:
                return pawn_name
        return None

    def _cell_is_blocked(self, cell: tuple[int, int], moving_pawn: str) -> bool:
        if cell in self._spec["walls"]:
            return True
        other_pawn = "orange" if moving_pawn == "blue" else "blue"
        if self._positions[other_pawn] == cell:
            return True
        for pair_id, door_cell in self._spec["doors"].items():
            if cell == door_cell and not self._door_is_open(pair_id):
                return True
        return False

    def _legal_destinations(self, pawn_name: str) -> list[tuple[int, int]]:
        x, y = self._positions[pawn_name]
        out: list[tuple[int, int]] = []
        for dx, dy in DIRS:
            cell = (x + dx, y + dy)
            if not self._cell_is_blocked(cell, pawn_name):
                out.append(cell)
        return out

    def _move_selected_pawn(self, destination: tuple[int, int]) -> bool:
        if self._selected is None:
            return False
        if destination not in self._legal_destinations(self._selected):
            return False
        self._positions[self._selected] = destination
        self._remaining_moves = max(0, self._remaining_moves - 1)
        return True

    def _check_win(self) -> bool:
        if self._spec["orange_goal"] is not None:
            return self._positions["orange"] == self._spec["orange_goal"]
        if self._spec["blue_goal"] is not None:
            return self._positions["blue"] == self._spec["blue_goal"]
        return (
            self._positions["blue"] == self._spec["final_blue_goal"]
            and self._positions["orange"] == self._spec["final_orange_goal"]
        )

    def step(self) -> None:
        if self.action.id != GameAction.ACTION6:
            self._render()
            self.complete_action()
            return

        if self._mode == MODE_LEVEL_WON:
            self.next_level()
            self.complete_action()
            return

        cell = self._click_to_cell(self.action.data.get("x"), self.action.data.get("y"))
        moved = False
        if cell is not None:
            clicked_pawn = self._pawn_at(cell)
            if clicked_pawn is not None:
                self._selected = clicked_pawn
            elif self._selected is not None:
                sx, sy = self._positions[self._selected]
                if abs(cell[0] - sx) + abs(cell[1] - sy) == 1:
                    moved = self._move_selected_pawn(cell)

        if moved and self._check_win():
            self._mode = MODE_LEVEL_WON
        elif moved and self._remaining_moves <= 0:
            self._mode = MODE_LEVEL_FAILED
            self._render()
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()

    def _render(self) -> None:
        frame = _solid(COLOR_FLOOR)
        self._draw_hud(frame)
        self._draw_floor(frame)
        self._draw_goal_overlays(frame)
        self._draw_latches(frame)
        self._draw_walls(frame)
        self._draw_doors(frame)
        if self._mode == MODE_PLAYING and self._selected is not None:
            self._draw_hints(frame, self._selected)
        self._draw_pawns(frame)
        self._draw_border(frame)
        self._canvas.pixels = frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        frame[0:8, :] = COLOR_FLOOR
        total = max(1, int(self._spec["budget"]))
        remaining = max(0, int(self._remaining_moves))
        bar_color = COLOR_FAIL if remaining == 0 and self._mode == MODE_PLAYING else COLOR_SUCCESS
        width = HUD_X1 - HUD_X0 + 1
        filled = round(width * (remaining / total))
        for x in range(HUD_X0, HUD_X1 + 1):
            color = bar_color if x < HUD_X0 + filled else COLOR_WALL_FILL
            frame[2:6, x] = color
            if (x - HUD_X0) % CELL_SIZE == 0:
                frame[1:7, x] = COLOR_OUTLINE
        frame[1:7, HUD_X1] = COLOR_OUTLINE

    def _draw_floor(self, frame: np.ndarray) -> None:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                px, py = _cell_to_pixel((x, y))
                frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_FLOOR
                if (x + y) % 2 == 0:
                    frame[py, px] = COLOR_FLOOR_ACCENT
                    frame[py + CELL_SIZE - 1, px + CELL_SIZE - 1] = COLOR_FLOOR_ACCENT

    def _draw_goal_overlays(self, frame: np.ndarray) -> None:
        if self._spec["orange_goal"] is not None:
            self._draw_goal_ring(frame, self._spec["orange_goal"], COLOR_ORANGE)
        if self._spec["blue_goal"] is not None:
            self._draw_goal_ring(frame, self._spec["blue_goal"], COLOR_BLUE)
        if self._spec["final_blue_goal"] is not None:
            self._draw_goal_ring(frame, self._spec["final_blue_goal"], COLOR_ORANGE)
        if self._spec["final_orange_goal"] is not None:
            self._draw_goal_ring(frame, self._spec["final_orange_goal"], COLOR_BLUE)

    def _draw_goal_ring(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        px, py = _cell_to_pixel(cell)
        frame[py, px + 1 : px + 3] = color
        frame[py + 3, px + 1 : px + 3] = color
        frame[py + 1 : py + 3, px] = color
        frame[py + 1 : py + 3, px + 3] = color

    def _draw_latches(self, frame: np.ndarray) -> None:
        for pair_id, cell in self._spec["latches"].items():
            px, py = _cell_to_pixel(cell)
            color = PAIR_COLORS[pair_id]
            frame[py : py + 4, px : px + 4] = color
            frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_WALL_FILL

    def _draw_walls(self, frame: np.ndarray) -> None:
        for cell in self._spec["walls"]:
            px, py = _cell_to_pixel(cell)
            frame[py : py + 4, px : px + 4] = COLOR_WALL_FILL
            x, y = cell
            if (x, y - 1) not in self._spec["walls"]:
                frame[py, px : px + 4] = COLOR_OUTLINE
            if (x, y + 1) not in self._spec["walls"]:
                frame[py + 3, px : px + 4] = COLOR_OUTLINE
            if (x - 1, y) not in self._spec["walls"]:
                frame[py : py + 4, px] = COLOR_OUTLINE
            if (x + 1, y) not in self._spec["walls"]:
                frame[py : py + 4, px + 3] = COLOR_OUTLINE

    def _draw_doors(self, frame: np.ndarray) -> None:
        for pair_id, cell in self._spec["doors"].items():
            px, py = _cell_to_pixel(cell)
            if self._door_is_open(pair_id):
                frame[py : py + 4, px] = COLOR_WALL_FILL
                frame[py : py + 4, px + 3] = COLOR_WALL_FILL
            else:
                color = PAIR_COLORS[pair_id]
                frame[py : py + 4, px] = color
                frame[py : py + 4, px + 2] = color
                frame[py : py + 4, px + 1] = COLOR_WALL_FILL
                frame[py : py + 4, px + 3] = COLOR_WALL_FILL

    def _draw_hints(self, frame: np.ndarray, pawn_name: str) -> None:
        for cell in self._legal_destinations(pawn_name):
            px, py = _cell_to_pixel(cell)
            frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_HINT

    def _draw_pawns(self, frame: np.ndarray) -> None:
        for pawn_name in PAWN_ORDER:
            px, py = _cell_to_pixel(self._positions[pawn_name])
            color = PAWN_COLORS[pawn_name]
            pattern = np.array(
                [[0, color, color, 0], [color, color, color, color], [0, color, color, 0], [color, 0, 0, color]],
                dtype=np.int8,
            )
            mask = pattern != 0
            frame[py : py + 4, px : px + 4][mask] = pattern[mask]
            if self._selected == pawn_name and self._mode == MODE_PLAYING:
                frame[py, px] = COLOR_SELECTION
                frame[py, px + 3] = COLOR_SELECTION
                frame[py + 2, px] = COLOR_SELECTION
                frame[py + 2, px + 3] = COLOR_SELECTION

    def _draw_border(self, frame: np.ndarray) -> None:
        if self._mode == MODE_LEVEL_WON:
            color = COLOR_SUCCESS
        elif self._mode == MODE_LEVEL_FAILED:
            color = COLOR_FAIL
        else:
            return
        frame[0, :] = color
        frame[-1, :] = color
        frame[:, 0] = color
        frame[:, -1] = color
