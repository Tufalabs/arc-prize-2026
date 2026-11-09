from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

CELL_SIZE = 8
HUD_HEIGHT = 8
LOGICAL_WIDTH = 8
LOGICAL_HEIGHT = 7
FRAME_SIZE = 64

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_SPENT = 3
COLOR_BLACK = 5
COLOR_WARP_A = 6
COLOR_WARP_A_HI = 7
COLOR_FAIL = 8
COLOR_WARP_B = 9
COLOR_WARP_B_HI = 10
COLOR_GOAL = 11
COLOR_WARNING = 12
COLOR_WALL = 3
COLOR_AVATAR = 14
COLOR_WARP_C = 15

ACTION_TO_DELTA = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


LEVEL_SPECS = (
    {"rows": ("########", "#S.A...#", "###.####", "#...#..#", "#.###aG#", "#......#", "########"), "max_moves": 9},
    {"rows": ("########", "#S.A#G##", "#..##..#", "#.###..#", "#b##Ba.#", "########", "########"), "max_moves": 15},
    {"rows": ("########", "#####G##", "#####.c#", "#b##aB##", "#.##..##", "#SA##C##", "########"), "max_moves": 18},
)


def _make_levels() -> list[Level]:
    return [
        Level(grid_size=(FRAME_SIZE, FRAME_SIZE), data={"level_index": index}, name=f"Level {index + 1}")
        for index in range(len(LEVEL_SPECS))
    ]


def _cell_rect(cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
    px = cell_x * CELL_SIZE
    py = HUD_HEIGHT + cell_y * CELL_SIZE
    return px, py, px + CELL_SIZE, py + CELL_SIZE


class one_way_warp_maze(ARCBaseGame):
    def __init__(self) -> None:
        self._phase = "normal"
        self._avatar = (0, 0)
        self._start = (0, 0)
        self._goal = (0, 0)
        self._remaining_moves = 0
        self._move_budget = 0
        self._walls: frozenset[tuple[int, int]] = frozenset()
        self._warp_entries: dict[tuple[int, int], tuple[int, int]] = {}
        self._warp_exits: set[tuple[int, int]] = set()
        self._entry_symbols: dict[tuple[int, int], str] = {}
        self._exit_symbols: dict[tuple[int, int], str] = {}
        self._teleport_hint: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._board_sprite: Sprite | None = None
        super().__init__(
            "one_way_warp_maze",
            _make_levels(),
            Camera(0, 0, FRAME_SIZE, FRAME_SIZE, COLOR_WHITE, COLOR_WHITE, []),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4],
        )

    def on_set_level(self, level: Level) -> None:
        spec = LEVEL_SPECS[int(level.get_data("level_index"))]
        self._load_spec(spec)
        board = Sprite(np.full((FRAME_SIZE, FRAME_SIZE), COLOR_WHITE, dtype=np.int8), name="board", collidable=False)
        level.add_sprite(board)
        self._board_sprite = board
        self._redraw()

    def _load_spec(self, spec: dict[str, object]) -> None:
        walls: set[tuple[int, int]] = set()
        warp_markers: dict[str, tuple[int, int]] = {}
        self._entry_symbols = {}
        self._exit_symbols = {}
        self._warp_entries = {}
        self._warp_exits = set()
        self._teleport_hint = None
        self._phase = "normal"
        self._move_budget = int(spec["max_moves"])
        self._remaining_moves = int(spec["max_moves"])

        for y, row in enumerate(spec["rows"]):
            for x, tile in enumerate(row):
                if tile == "#":
                    walls.add((x, y))
                elif tile == "S":
                    self._start = (x, y)
                    self._avatar = (x, y)
                elif tile == "G":
                    self._goal = (x, y)
                elif tile in "ABCabc":
                    warp_markers[tile] = (x, y)

        for entry_symbol, exit_symbol in (("A", "a"), ("B", "b"), ("C", "c")):
            if entry_symbol in warp_markers and exit_symbol in warp_markers:
                entry = warp_markers[entry_symbol]
                exit_cell = warp_markers[exit_symbol]
                self._warp_entries[entry] = exit_cell
                self._warp_exits.add(exit_cell)
                self._entry_symbols[entry] = entry_symbol
                self._exit_symbols[exit_cell] = exit_symbol

        self._walls = frozenset(walls)

    def step(self) -> None:
        if self._phase == "fail_flash":
            self.lose()
            self.complete_action()
            return

        if self._phase == "win_flash":
            if self.is_last_level():
                self._phase = "game_complete"
                self._teleport_hint = None
                self._redraw()
            self.next_level()
            self.complete_action()
            return

        delta = ACTION_TO_DELTA.get(self._action.id)
        if delta is None:
            self.complete_action()
            return

        self._teleport_hint = None
        self._remaining_moves = max(0, self._remaining_moves - 1)

        next_pos = (self._avatar[0] + delta[0], self._avatar[1] + delta[1])
        if self._is_walkable(next_pos):
            self._avatar = next_pos

        warp_exit = self._warp_entries.get(self._avatar)
        if warp_exit is not None:
            warp_entry = self._avatar
            self._avatar = warp_exit
            self._teleport_hint = (warp_entry, warp_exit)

        if self._avatar == self._goal:
            self._phase = "win_flash"
            self._teleport_hint = None
        elif self._remaining_moves == 0:
            self._phase = "fail_flash"

        self._redraw()

        # Fold the fail-flash transition into the same action that depleted
        # the budget: skip complete_action so arcengine's perform_action
        # loop re-enters step(), which hits the _phase == "fail_flash"
        # branch at the top and calls self.lose() before completing. The
        # first iter's render is captured as an animation frame (fail-flash
        # overlay), and the final observation carries state=GAME_OVER so
        # clients see the loss registered on the move that caused it.
        if self._phase == "fail_flash":
            return
        self.complete_action()

    def _is_walkable(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < LOGICAL_WIDTH and 0 <= y < LOGICAL_HEIGHT and pos not in self._walls

    def _redraw(self) -> None:
        frame = np.full((FRAME_SIZE, FRAME_SIZE), COLOR_WHITE, dtype=np.int8)
        self._draw_hud(frame)
        self._draw_board(frame)
        if self._board_sprite is not None:
            self._board_sprite.pixels = frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        pip_width = 2
        gap = 1
        pip_height = 6
        left = (FRAME_SIZE - (self._move_budget * pip_width + (self._move_budget - 1) * gap)) // 2
        top = 1

        for index in range(self._move_budget):
            x0 = left + index * (pip_width + gap)
            x1 = x0 + pip_width
            color = COLOR_SPENT
            remaining_after_spend = self._move_budget - index
            if self._phase == "fail_flash":
                color = COLOR_FAIL
            elif self._phase == "game_complete":
                color = COLOR_AVATAR
            elif index < self._remaining_moves:
                color = COLOR_WARNING if self._remaining_moves <= 3 else COLOR_AVATAR
            elif remaining_after_spend <= 3 and index >= self._remaining_moves and self._remaining_moves > 0:
                color = COLOR_SPENT
            frame[top : top + pip_height, x0:x1] = color

    def _draw_board(self, frame: np.ndarray) -> None:
        for y in range(LOGICAL_HEIGHT):
            for x in range(LOGICAL_WIDTH):
                self._draw_floor(frame, x, y)

        for x, y in self._walls:
            self._draw_wall(frame, x, y)

        for pos, symbol in self._exit_symbols.items():
            self._draw_warp_exit(frame, pos, symbol)
        for pos, symbol in self._entry_symbols.items():
            self._draw_warp_entry(frame, pos, symbol)

        self._draw_goal(frame)

        if self._teleport_hint is not None and self._phase == "normal":
            self._draw_highlight(frame, self._teleport_hint[0])
            self._draw_highlight(frame, self._teleport_hint[1])

        if self._phase == "fail_flash":
            self._draw_fail_border(frame)
            self._draw_fail_avatar(frame)
        else:
            self._draw_avatar(frame)

    def _draw_floor(self, frame: np.ndarray, cell_x: int, cell_y: int) -> None:
        x0, y0, x1, y1 = _cell_rect(cell_x, cell_y)
        frame[y0:y1, x0:x1] = COLOR_WHITE
        frame[y0 + 1 : y1 - 1, x0 + 1] = COLOR_FLOOR
        frame[y0 + 1 : y1 - 1, x1 - 2] = COLOR_FLOOR
        frame[y0 + 1, x0 + 1 : x1 - 1] = COLOR_FLOOR
        frame[y1 - 2, x0 + 1 : x1 - 1] = COLOR_FLOOR

    def _draw_wall(self, frame: np.ndarray, cell_x: int, cell_y: int) -> None:
        x0, y0, x1, y1 = _cell_rect(cell_x, cell_y)
        frame[y0:y1, x0:x1] = COLOR_WALL
        neighbors = {
            "up": (cell_x, cell_y - 1),
            "down": (cell_x, cell_y + 1),
            "left": (cell_x - 1, cell_y),
            "right": (cell_x + 1, cell_y),
        }
        if neighbors["up"] not in self._walls:
            frame[y0, x0:x1] = COLOR_BLACK
        if neighbors["down"] not in self._walls:
            frame[y1 - 1, x0:x1] = COLOR_BLACK
        if neighbors["left"] not in self._walls:
            frame[y0:y1, x0] = COLOR_BLACK
        if neighbors["right"] not in self._walls:
            frame[y0:y1, x1 - 1] = COLOR_BLACK
        frame[y0 + 3 : y0 + 5, x0 + 2 : x0 + 3] = COLOR_BLACK
        frame[y0 + 2 : y0 + 3, x0 + 4 : x0 + 6] = COLOR_BLACK

    def _pair_colors(self, symbol: str) -> tuple[int, int]:
        pair = symbol.upper()
        if pair == "A":
            return COLOR_WARP_A, COLOR_WARP_A_HI
        if pair == "B":
            return COLOR_WARP_B, COLOR_WARP_B_HI
        return COLOR_WARP_C, COLOR_WHITE

    def _draw_warp_entry(self, frame: np.ndarray, pos: tuple[int, int], symbol: str) -> None:
        fill, accent = self._pair_colors(symbol)
        x0, y0, _, _ = _cell_rect(*pos)
        coords = (
            (3, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (3, 5),
        )
        for dx, dy in coords:
            frame[y0 + dy, x0 + dx] = fill
        frame[y0 + 2, x0 + 4] = accent
        frame[y0 + 3, x0 + 4] = accent
        frame[y0 + 4, x0 + 4] = accent
        frame[y0 + 3, x0 + 2] = COLOR_BLACK

    def _draw_warp_exit(self, frame: np.ndarray, pos: tuple[int, int], symbol: str) -> None:
        fill, accent = self._pair_colors(symbol)
        x0, y0, _, _ = _cell_rect(*pos)
        frame[y0 + 1 : y0 + 5, x0 + 1] = fill
        frame[y0 + 1 : y0 + 5, x0 + 6] = fill
        frame[y0 + 5, x0 + 2 : x0 + 6] = fill
        frame[y0 + 2 : y0 + 5, x0 + 2 : x0 + 6] = COLOR_WHITE
        frame[y0 + 2 : y0 + 4, x0 + 2] = accent
        frame[y0 + 2 : y0 + 4, x0 + 5] = accent

    def _draw_goal(self, frame: np.ndarray) -> None:
        x0, y0, _, _ = _cell_rect(*self._goal)
        rim = COLOR_AVATAR if self._phase in {"win_flash", "game_complete"} else COLOR_GOAL
        accent = COLOR_AVATAR if self._phase in {"win_flash", "game_complete"} else COLOR_WARNING
        frame[y0 + 1, x0 + 2 : x0 + 6] = rim
        frame[y0 + 5, x0 + 2 : x0 + 6] = rim
        frame[y0 + 2 : y0 + 5, x0 + 1] = rim
        frame[y0 + 2 : y0 + 5, x0 + 6] = rim
        frame[y0 + 2 : y0 + 5, x0 + 3 : x0 + 5] = accent
        frame[y0 + 3, x0 + 3 : x0 + 5] = COLOR_BLACK

    def _draw_avatar(self, frame: np.ndarray) -> None:
        x0, y0, _, _ = _cell_rect(*self._avatar)
        fill = COLOR_AVATAR
        coords = (
            (3, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (3, 5),
        )
        outline = ((3, 0), (0, 3), (6, 3), (3, 6))
        for dx, dy in outline:
            frame[y0 + dy, x0 + dx] = COLOR_BLACK
        for dx, dy in coords:
            frame[y0 + dy, x0 + dx] = fill
        frame[y0 + 3, x0 + 3] = COLOR_WHITE

    def _draw_fail_avatar(self, frame: np.ndarray) -> None:
        x0, y0, _, _ = _cell_rect(*self._avatar)
        for offset in range(1, 6):
            frame[y0 + offset, x0 + offset] = COLOR_FAIL
            frame[y0 + offset, x0 + 6 - offset] = COLOR_FAIL

    def _draw_highlight(self, frame: np.ndarray, pos: tuple[int, int]) -> None:
        x0, y0, x1, y1 = _cell_rect(*pos)
        frame[y0 + 1, x0 + 1 : x1 - 1] = COLOR_WHITE
        frame[y1 - 2, x0 + 1 : x1 - 1] = COLOR_WHITE
        frame[y0 + 1 : y1 - 1, x0 + 1] = COLOR_WHITE
        frame[y0 + 1 : y1 - 1, x1 - 2] = COLOR_WHITE

    def _draw_fail_border(self, frame: np.ndarray) -> None:
        frame[HUD_HEIGHT, :] = COLOR_FAIL
        frame[FRAME_SIZE - 1, :] = COLOR_FAIL
        frame[HUD_HEIGHT:, 0] = COLOR_FAIL
        frame[HUD_HEIGHT:, FRAME_SIZE - 1] = COLOR_FAIL


AGENT_GAME_CLASS = one_way_warp_maze
