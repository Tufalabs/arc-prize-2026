from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BOARD_SIZE = 8
DISPLAY_SIZE = 64
HUD_HEIGHT = 8
TILE_SIZE = 7
BOARD_LEFT = 4
BOARD_TOP = 8
BOARD_RIGHT = BOARD_LEFT + (BOARD_SIZE * TILE_SIZE)
BOARD_BOTTOM = BOARD_TOP + (BOARD_SIZE * TILE_SIZE)

COLOR_BG = 0
COLOR_HUD = 1
COLOR_SPENT = 3
COLOR_WALL = 4
COLOR_OUTLINE = 5
COLOR_PUCK = 8
COLOR_ICE_DARK = 9
COLOR_ICE = 10
COLOR_STAR = 11
COLOR_BUDGET = 12
COLOR_LOCK = 13
COLOR_GOAL = 14
COLOR_POST = 15

ACTION_TO_DELTA = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


class IceStopLevel:
    def __init__(self, name: str, rows: tuple[str, ...], budget: int, requires_star: bool, optimal_moves: int) -> None:
        self.name = name
        self.rows = rows
        self.budget = budget
        self.requires_star = requires_star
        self.optimal_moves = optimal_moves


LEVEL_SPECS: tuple[IceStopLevel, ...] = (
    IceStopLevel(
        name="level_1",
        rows=("########", "#S...#.#", "#....#.#", "#....#.#", "#....#.#", "#....#.#", "#...D..#", "########"),
        budget=6,
        requires_star=False,
        optimal_moves=2,
    ),
    IceStopLevel(
        name="level_2",
        rows=("########", "#..D...#", "#......#", "#S..P..#", "#......#", "#......#", "#......#", "########"),
        budget=7,
        requires_star=False,
        optimal_moves=2,
    ),
    IceStopLevel(
        name="level_3",
        rows=("########", "#S..P..#", "#......#", "#......#", "#D.....#", "#..P...#", "#......#", "########"),
        budget=10,
        requires_star=False,
        optimal_moves=3,
    ),
    IceStopLevel(
        name="level_4",
        rows=("########", "#......#", "#...#.D#", "#P..#.P#", "#...#..#", "#S..#..#", "#......#", "########"),
        budget=12,
        requires_star=False,
        optimal_moves=4,
    ),
    IceStopLevel(
        name="level_5",
        rows=("########", "#*.....#", "#...#.D#", "#P..#.P#", "#...#..#", "#S..#..#", "#......#", "########"),
        budget=15,
        requires_star=True,
        optimal_moves=5,
    ),
)


def _cell_origin(gx: int, gy: int) -> tuple[int, int]:
    return BOARD_LEFT + (gx * TILE_SIZE), BOARD_TOP + (gy * TILE_SIZE)


class IceStop(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [Level(grid_size=(DISPLAY_SIZE, DISPLAY_SIZE), name=spec.name) for spec in LEVEL_SPECS]
        self._board = np.full((BOARD_SIZE, BOARD_SIZE), ".", dtype="<U1")
        self._dock = (0, 0)
        self._star = None
        self._player = (0, 0)
        self._start = (0, 0)
        self._remaining_budget = 0
        self._mode = "play"
        self._star_collected = False
        self._current_spec = LEVEL_SPECS[0]
        super().__init__(
            game_id="ice_stop-0001",
            levels=levels,
            camera=Camera(0, 0, DISPLAY_SIZE, DISPLAY_SIZE, COLOR_BG, COLOR_BG),
            debug=False,
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self._load_level(self.level_index)

    def full_reset(self) -> None:
        super().full_reset()
        self._mode = "play"

    def level_reset(self) -> None:
        super().level_reset()
        self._mode = "play"

    def _load_level(self, level_index: int) -> None:
        spec = LEVEL_SPECS[level_index]
        self._current_spec = spec
        self._board = np.full((BOARD_SIZE, BOARD_SIZE), ".", dtype="<U1")
        self._dock = (0, 0)
        self._star = None
        self._player = (0, 0)
        self._start = (0, 0)
        self._remaining_budget = int(spec.budget)
        self._mode = "play"
        self._star_collected = False

        for gy, row in enumerate(spec.rows):
            for gx, cell in enumerate(row):
                if cell == "#":
                    self._board[gy, gx] = "#"
                elif cell == "P":
                    self._board[gy, gx] = "P"
                elif cell == "D":
                    self._dock = (gx, gy)
                elif cell == "*":
                    self._star = (gx, gy)
                elif cell == "S":
                    self._player = (gx, gy)
                    self._start = (gx, gy)

        self._redraw()

    def _is_solid(self, x: int, y: int) -> bool:
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return True
        return self._board[y, x] in {"#", "P"}

    def _resolve_slide(self, dx: int, dy: int) -> None:
        x, y = self._player
        nx = x + dx
        ny = y + dy
        if self._is_solid(nx, ny):
            return

        while not self._is_solid(nx, ny):
            x, y = nx, ny
            if self._star is not None and not self._star_collected and (x, y) == self._star:
                self._star_collected = True
            nx += dx
            ny += dy
        self._player = (x, y)

    def _handle_directional_action(self, action: GameAction) -> None:
        self._remaining_budget = max(0, self._remaining_budget - 1)
        dx, dy = ACTION_TO_DELTA[action]
        self._resolve_slide(dx, dy)

        on_dock = self._player == self._dock
        if on_dock and (not self._current_spec.requires_star or self._star_collected):
            self._mode = "won"
        elif self._remaining_budget == 0:
            self._mode = "lost"

    def _advance_after_win(self) -> None:
        if self.is_last_level():
            self._mode = "game_complete"
            self.next_level()
            return
        self.next_level()
        self._mode = "play"

    def _redraw(self) -> None:
        frame = np.full((DISPLAY_SIZE, DISPLAY_SIZE), COLOR_BG, dtype=np.int8)
        self._draw_hud(frame)
        self._draw_board(frame)
        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(Sprite(pixels=frame, x=0, y=0, layer=0, collidable=False, visible=True))

    def _draw_hud(self, frame: np.ndarray) -> None:
        frame[:HUD_HEIGHT, :] = COLOR_HUD
        max_pips = 16
        remaining = max(0, min(self._remaining_budget, max_pips))
        for idx in range(max_pips):
            row = 0 if idx < 8 else 1
            col = idx % 8
            x0 = 4 + (col * 6)
            y0 = 1 + (row * 3)
            color = COLOR_BUDGET if idx < remaining else COLOR_SPENT
            frame[y0 : y0 + 3, x0 : x0 + 5] = color
            frame[y0, x0 : x0 + 5] = COLOR_BG

        if self._current_spec.requires_star:
            self._draw_hud_star(frame, 55, 1, COLOR_GOAL if self._star_collected else COLOR_STAR)

    def _draw_hud_star(self, frame: np.ndarray, x0: int, y0: int, color: int) -> None:
        star = ("..x..", ".xxx.", "xxxxx", ".xxx.", "..x..")
        for dy, row in enumerate(star):
            for dx, cell in enumerate(row):
                if cell == "x":
                    frame[y0 + dy, x0 + dx] = color

    def _draw_board(self, frame: np.ndarray) -> None:
        for gy in range(BOARD_SIZE):
            for gx in range(BOARD_SIZE):
                px0, py0 = _cell_origin(gx, gy)
                self._draw_ice_tile(frame, px0, py0)

        for gy in range(BOARD_SIZE):
            for gx in range(BOARD_SIZE):
                px0, py0 = _cell_origin(gx, gy)
                cell = self._board[gy, gx]
                if cell == "#":
                    self._draw_wall_tile(frame, gx, gy, px0, py0)
                elif cell == "P":
                    self._draw_post_tile(frame, px0, py0)

        if self._star is not None and not self._star_collected:
            self._draw_star_tile(frame, *_cell_origin(*self._star))

        locked_dock = self._current_spec.requires_star and not self._star_collected
        self._draw_dock_tile(frame, *_cell_origin(*self._dock), locked=locked_dock)
        self._draw_puck_tile(frame, *_cell_origin(*self._player))

        if self._mode == "won":
            self._draw_border(frame, COLOR_GOAL)
        elif self._mode == "lost":
            self._draw_border(frame, 8)
        elif self._mode == "game_complete":
            self._draw_border(frame, COLOR_GOAL)
            self._draw_hud_star(frame, 48, 1, COLOR_STAR)
            self._draw_hud_star(frame, 55, 1, COLOR_STAR)

    def _draw_ice_tile(self, frame: np.ndarray, px0: int, py0: int) -> None:
        frame[py0 : py0 + TILE_SIZE, px0 : px0 + TILE_SIZE] = COLOR_ICE
        for offset in range(1, 5):
            frame[py0 + offset, px0 + offset] = COLOR_BG
        frame[py0 + 1, px0 + 4 : px0 + 6] = COLOR_ICE_DARK
        frame[py0 + 5, px0 + 1 : px0 + 3] = COLOR_ICE_DARK

    def _draw_wall_tile(self, frame: np.ndarray, gx: int, gy: int, px0: int, py0: int) -> None:
        frame[py0 : py0 + TILE_SIZE, px0 : px0 + TILE_SIZE] = COLOR_WALL
        if gy == 0 or self._board[gy - 1, gx] != "#":
            frame[py0, px0 : px0 + TILE_SIZE] = COLOR_OUTLINE
        if gx == 0 or self._board[gy, gx - 1] != "#":
            frame[py0 : py0 + TILE_SIZE, px0] = COLOR_OUTLINE
        if gy == BOARD_SIZE - 1 or self._board[gy + 1, gx] != "#":
            frame[py0 + TILE_SIZE - 1, px0 : px0 + TILE_SIZE] = COLOR_BG
        if gx == BOARD_SIZE - 1 or self._board[gy, gx + 1] != "#":
            frame[py0 : py0 + TILE_SIZE, px0 + TILE_SIZE - 1] = COLOR_BG

    def _draw_post_tile(self, frame: np.ndarray, px0: int, py0: int) -> None:
        pattern = (".......", "...p...", "..ppp..", "...p...", "..ppp..", "...s...", ".......")
        for dy, row in enumerate(pattern):
            for dx, cell in enumerate(row):
                if cell == "p":
                    frame[py0 + dy, px0 + dx] = COLOR_POST
                elif cell == "s":
                    frame[py0 + dy, px0 + dx] = COLOR_LOCK

    def _draw_dock_tile(self, frame: np.ndarray, px0: int, py0: int, *, locked: bool) -> None:
        pattern = (".......", ".g...g.", ".g...g.", ".g...g.", ".ggggg.", ".......", ".......")
        for dy, row in enumerate(pattern):
            for dx, cell in enumerate(row):
                if cell == "g":
                    frame[py0 + dy, px0 + dx] = COLOR_GOAL
        if locked:
            for dx in range(2, 5):
                frame[py0 + 1, px0 + dx] = COLOR_LOCK
            frame[py0 + 2, px0 + 3] = COLOR_LOCK
            frame[py0 + 3, px0 + 3] = COLOR_LOCK

    def _draw_star_tile(self, frame: np.ndarray, px0: int, py0: int) -> None:
        pattern = (".......", "...y...", "..yyy..", ".yyyyy.", "..yyy..", "...y...", ".......")
        for dy, row in enumerate(pattern):
            for dx, cell in enumerate(row):
                if cell == "y":
                    frame[py0 + dy, px0 + dx] = COLOR_STAR

    def _draw_puck_tile(self, frame: np.ndarray, px0: int, py0: int) -> None:
        pattern = (".......", "..rrr..", ".rrwrr.", ".rrrrr.", ".rrrrr.", "..rrr..", ".......")
        for dy, row in enumerate(pattern):
            for dx, cell in enumerate(row):
                if cell == "r":
                    frame[py0 + dy, px0 + dx] = COLOR_PUCK
                elif cell == "w":
                    frame[py0 + dy, px0 + dx] = COLOR_BUDGET

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[BOARD_TOP:BOARD_BOTTOM, BOARD_LEFT] = color
        frame[BOARD_TOP:BOARD_BOTTOM, BOARD_RIGHT - 1] = color
        frame[BOARD_TOP, BOARD_LEFT:BOARD_RIGHT] = color
        frame[BOARD_BOTTOM - 1, BOARD_LEFT:BOARD_RIGHT] = color

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.RESET:
            self._redraw()
            self.complete_action()
            return

        if self._mode == "game_complete":
            self._redraw()
            self.complete_action()
            return

        if self._mode == "won":
            self._advance_after_win()
            self._redraw()
            self.complete_action()
            return

        if self._mode == "lost":
            self._redraw()
            self.lose()
            self.complete_action()
            return

        if action == GameAction.ACTION5:
            self.level_reset()
            self._redraw()
            self.complete_action()
            return

        if action == GameAction.ACTION6:
            self._redraw()
            self.complete_action()
            return

        if action in ACTION_TO_DELTA:
            self._handle_directional_action(action)
            if self._mode == "lost":
                self._redraw()
                self.lose()
                self.complete_action()
                return
            self._redraw()
            self.complete_action()
            return

        self._redraw()
        self.complete_action()
