from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

STONE = "stone"
PIT = "pit"
FLAT = "flat"
MOUND = "mound"
MOLD_EMPTY = "mold_empty"
MOLD_FILLED = "mold_filled"
KILN = "kiln"

DIRS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
DIR_ROTATION = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}

COLOR_BG = 0
COLOR_FLAT = 1
COLOR_RIM = 2
COLOR_PIT = 3
COLOR_STONE = 4
COLOR_DEEP = 5
COLOR_KILN_HOT = 8
COLOR_FACE = 9
COLOR_BODY = 10
COLOR_GLINT = 11
COLOR_MOUND = 12
COLOR_SHADOW = 13
COLOR_PLUG = 14
COLOR_MOLD = 15


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start_facing: int
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "First Footprint",
        (
            "##########",
            "##########",
            "##.S_.K###",
            "##.^.M####",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ACTION_LEFT,
        8,
    ),
    LevelSpec(
        "The Wall You Made",
        (
            "##########",
            "##########",
            "####K#####",
            "####M#####",
            "#...S..###",
            "#.^....###",
            "##...#####",
            "##########",
            "##########",
            "##########",
        ),
        ACTION_RIGHT,
        19,
    ),
    LevelSpec(
        "Stopper Becomes Clay",
        (
            "##########",
            "##########",
            "####K#####",
            "####M#####",
            "#......###",
            "#.^...S###",
            "##....####",
            "#####M####",
            "##########",
            "##########",
        ),
        ACTION_RIGHT,
        20,
    ),
    LevelSpec(
        "Fired Floor",
        (
            "##########",
            "##########",
            "####...K##",
            "####M...##",
            "#S.....^##",
            "#.^.....##",
            "##......##",
            "#######M##",
            "##########",
            "##########",
        ),
        ACTION_RIGHT,
        27,
    ),
    LevelSpec(
        "Clay Falls Flat",
        (
            "##########",
            "##########",
            "####....K#",
            "####M...##",
            "#S.....^##",
            "#.^.....##",
            "##_.^...##",
            "#####.^_##",
            "#####M#M##",
            "##########",
        ),
        ACTION_RIGHT,
        40,
    ),
    LevelSpec(
        "Braced Bridge",
        (
            "##########",
            "##########",
            "#######._K",
            "####M...##",
            "#S.....^##",
            "#.^.....##",
            "##_.^...##",
            "#####.^_##",
            "#####M#M##",
            "##########",
        ),
        ACTION_RIGHT,
        43,
    ),
    LevelSpec(
        "Worse First",
        (
            "##########",
            "##########",
            "#######._K",
            "####M...##",
            "#.S....^##",
            "#.^.....##",
            "#._.^...##",
            "####..^_##",
            "####MM#M##",
            "##########",
        ),
        ACTION_RIGHT,
        50,
    ),
    LevelSpec(
        "Double Footprint Kiln",
        (
            "############",
            "############",
            "#######._._K",
            "####M.....##",
            "#.S....^.^##",
            "#.^.......##",
            "#._.^.....##",
            "####..^_^_##",
            "####MM#M#M##",
            "############",
        ),
        ACTION_RIGHT,
        58,
    ),
)


class ClayFootprintsHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ClayFootprints | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        x0 = 8
        y0 = 61
        width = 48
        frame[y0 : y0 + 2, x0 : x0 + width] = COLOR_STONE
        filled = max(0, min(width, int(width * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[y0 : y0 + 2, x0 : x0 + filled] = COLOR_MOUND
        return frame


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


def _tile_metrics(rows: tuple[str, ...]) -> tuple[int, int, int]:
    h = len(rows)
    w = len(rows[0])
    tile = min(6, 60 // max(w, h))
    return tile, (VIEW_SIZE - w * tile) // 2, (VIEW_SIZE - h * tile) // 2


def _player_pixels(facing: tuple[int, int], size: int = 6) -> np.ndarray:
    p = np.full((size, size), -1, dtype=np.int8)
    p[1 : size - 1, 1 : size - 1] = COLOR_BODY
    mid = max(1, size // 2 - 1)
    p[0, mid : mid + 2] = COLOR_BODY
    dx, dy = facing
    if dx == 1:
        p[mid : mid + 2, size - 1] = COLOR_FACE
    elif dx == -1:
        p[mid : mid + 2, 0] = COLOR_FACE
    elif dy == -1:
        p[0, mid : mid + 2] = COLOR_FACE
    else:
        p[size - 1, mid : mid + 2] = COLOR_FACE
    return p


def _build_level(spec: LevelSpec) -> Level:
    start: tuple[int, int] | None = None
    molds: list[tuple[int, int]] = []
    grid: list[list[str]] = []
    for y, row in enumerate(spec.rows):
        out_row: list[str] = []
        for x, cell in enumerate(row):
            if cell == "#":
                out_row.append(STONE)
            elif cell == "_":
                out_row.append(PIT)
            elif cell == "^":
                out_row.append(MOUND)
            elif cell == "M":
                out_row.append(MOLD_EMPTY)
                molds.append((x, y))
            elif cell == "K":
                out_row.append(KILN)
            elif cell == "S":
                start = (x, y)
                out_row.append(FLAT)
            else:
                out_row.append(FLAT)
        grid.append(out_row)
    if start is None:
        raise ValueError(f"{spec.name} is missing S.")

    board = Sprite(_solid(VIEW_SIZE, VIEW_SIZE, COLOR_BG), name="board", layer=0, collidable=False, tags=["board"])
    tile, _, _ = _tile_metrics(spec.rows)
    player = Sprite(
        _player_pixels(DIRS[spec.start_facing], tile), name="player", layer=20, collidable=False, tags=["player"]
    )
    return Level(
        sprites=[board, player],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "grid": tuple(tuple(row) for row in grid),
            "molds": tuple(molds),
            "start": start,
            "start_facing": spec.start_facing,
            "tile": _tile_metrics(spec.rows),
            "step_budget": spec.baseline_actions * 6,
        },
    )


class ClayFootprints(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = ClayFootprintsHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "clay_footprints",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BG, COLOR_BG, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self.grid = [list(row) for row in level.get_data("grid")]
        self.mold_cells = set(tuple(cell) for cell in level.get_data("molds"))
        self.pos = tuple(level.get_data("start"))
        self.facing = DIRS[int(level.get_data("start_facing"))]
        self.tile, self.origin_x, self.origin_y = level.get_data("tile")
        self.step_budget = int(level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self._animation_frames: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
        self._terminal_after_animation: str | None = None
        self._sync_visuals()

    def step(self) -> None:
        if self._animation_frames:
            board_pixels, pos, facing = self._animation_frames.pop(0)
            self.board.pixels = board_pixels
            self._sync_player(pos, facing)
            if not self._animation_frames:
                self._finish_or_continue()
            return

        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        action = _action_id(self.action.id)
        before_grid = [row[:] for row in self.grid]
        before_pos = self.pos
        before_facing = self.facing
        moved_slide: list[tuple[int, int]] = []
        slide_result: str | None = None
        stamp_info: tuple[tuple[int, int], tuple[int, int], str] | None = None

        if action in DIRS:
            moved_slide, slide_result = self._resolve_direction(action)
        elif action == ACTION_SPACE:
            stamp_info = self._resolve_stamp()
        elif action == ACTION_CLICK:
            pass

        self.remaining_steps -= 1
        won = self._is_win()
        if won:
            self._terminal_after_animation = "next"
        elif self.remaining_steps <= 0:
            self._terminal_after_animation = "lose"
        else:
            self._terminal_after_animation = None

        self._queue_animation(before_grid, before_pos, before_facing, moved_slide, slide_result, stamp_info)
        if not self._animation_frames:
            self._sync_visuals()
            self._finish_or_continue()

    def _finish_or_continue(self) -> None:
        if self._terminal_after_animation == "next":
            self._terminal_after_animation = None
            self._sync_visuals(glint=True)
            self.next_level()
            self.complete_action()
            return
        if self._terminal_after_animation == "lose":
            self._terminal_after_animation = None
            self._sync_visuals()
            self.lose()
            self.complete_action()
            return
        self._sync_visuals()
        self.complete_action()

    def _resolve_direction(self, action: int) -> tuple[list[tuple[int, int]], str | None]:
        self.facing = DIRS[action]
        dx, dy = self.facing
        fx, fy = self.pos[0] + dx, self.pos[1] + dy
        front = self._cell(fx, fy)
        if self._walkable(front):
            self.pos = (fx, fy)
            return [], None
        if front == MOUND and self._cell(*self.pos) != PIT:
            return self._push_mound((fx, fy), self.facing)
        return [], None

    def _push_mound(
        self, mound: tuple[int, int], direction: tuple[int, int]
    ) -> tuple[list[tuple[int, int]], str | None]:
        dx, dy = direction
        cx, cy = mound[0] + dx, mound[1] + dy
        flats: list[tuple[int, int]] = []
        while self._cell(cx, cy) == FLAT:
            flats.append((cx, cy))
            cx += dx
            cy += dy
        stop = self._cell(cx, cy)
        if stop in (MOLD_EMPTY, PIT):
            self.grid[mound[1]][mound[0]] = FLAT
            self.grid[cy][cx] = MOLD_FILLED if stop == MOLD_EMPTY else FLAT
            self.pos = mound
            return [*flats, (cx, cy)], "mold" if stop == MOLD_EMPTY else "pit"
        if flats:
            dest = flats[-1]
            self.grid[mound[1]][mound[0]] = FLAT
            self.grid[dest[1]][dest[0]] = MOUND
            self.pos = mound
            return flats, None
        return [], None

    def _resolve_stamp(self) -> tuple[tuple[int, int], tuple[int, int], str] | None:
        cx, cy = self.pos
        dx, dy = self.facing
        bx, by = cx - dx, cy - dy
        current = self._cell(cx, cy)
        behind = self._cell(bx, by)
        if current != FLAT or behind not in (PIT, FLAT):
            return None
        self.grid[cy][cx] = PIT
        self.grid[by][bx] = FLAT if behind == PIT else MOUND
        return (cx, cy), (bx, by), behind

    def _queue_animation(
        self,
        before_grid: list[list[str]],
        before_pos: tuple[int, int],
        before_facing: tuple[int, int],
        moved_slide: list[tuple[int, int]],
        slide_result: str | None,
        stamp_info: tuple[tuple[int, int], tuple[int, int], str] | None,
    ) -> None:
        if stamp_info is not None:
            current, behind, behind_kind = stamp_info
            self._animation_frames.append(
                (self._render_board(before_grid, stamp_current=current), before_pos, before_facing)
            )
            self._animation_frames.append(
                (self._render_board(before_grid, stamp_current=current), before_pos, before_facing)
            )
            rise_kind = FLAT if behind_kind == PIT else MOUND
            self._animation_frames.append(
                (
                    self._render_board(before_grid, stamp_current=current, stamp_behind=(behind, rise_kind)),
                    self.pos,
                    self.facing,
                )
            )
            self._animation_frames.append(
                (
                    self._render_board(before_grid, stamp_current=current, stamp_behind=(behind, rise_kind)),
                    self.pos,
                    self.facing,
                )
            )
            self._animation_frames.append((self._render_board(self.grid), self.pos, self.facing))
            return

        if len(moved_slide) > 1:
            origin = self.pos
            visual_grid = [row[:] for row in before_grid]
            visual_grid[origin[1]][origin[0]] = FLAT
            for cell in moved_slide:
                self._animation_frames.append(
                    (self._render_board(visual_grid, moving_mound=cell), self.pos, self.facing)
                )
                self._animation_frames.append(
                    (self._render_board(visual_grid, moving_mound=cell), self.pos, self.facing)
                )
            if slide_result in {"mold", "pit"}:
                self._animation_frames.append(
                    (self._render_board(self.grid, pulse=moved_slide[-1]), self.pos, self.facing)
                )
            self._animation_frames.append((self._render_board(self.grid), self.pos, self.facing))

    def _is_win(self) -> bool:
        return self._cell(*self.pos) == KILN and all(self.grid[y][x] == MOLD_FILLED for x, y in self.mold_cells)

    def _walkable(self, cell: str) -> bool:
        return cell in (FLAT, MOLD_FILLED, KILN)

    def _cell(self, x: int, y: int) -> str:
        if y < 0 or y >= len(self.grid) or x < 0 or x >= len(self.grid[0]):
            return STONE
        return self.grid[y][x]

    def _stamp_preview_cell(self) -> tuple[int, int] | None:
        cx, cy = self.pos
        dx, dy = self.facing
        bx, by = cx - dx, cy - dy
        if self._cell(cx, cy) == FLAT and self._cell(bx, by) in (PIT, FLAT):
            return bx, by
        return None

    def _sync_visuals(self, *, glint: bool = False) -> None:
        self.board.pixels = self._render_board(self.grid, stamp_preview=self._stamp_preview_cell(), glint=glint)
        self._sync_player(self.pos, self.facing)

    def _sync_player(self, pos: tuple[int, int], facing: tuple[int, int]) -> None:
        self.player.pixels = _player_pixels(facing, self.tile)
        x0, y0 = self._cell_top_left(*pos)
        self.player.set_position(x0, y0)

    def _cell_top_left(self, x: int, y: int) -> tuple[int, int]:
        return self.origin_x + x * self.tile, self.origin_y + y * self.tile

    def _render_board(
        self,
        grid: list[list[str]],
        *,
        moving_mound: tuple[int, int] | None = None,
        stamp_current: tuple[int, int] | None = None,
        stamp_behind: tuple[tuple[int, int], str] | None = None,
        stamp_preview: tuple[int, int] | None = None,
        pulse: tuple[int, int] | None = None,
        glint: bool = False,
    ) -> np.ndarray:
        frame = _solid(VIEW_SIZE, VIEW_SIZE, COLOR_BG)
        for y, row in enumerate(grid):
            for x, kind in enumerate(row):
                if moving_mound == (x, y):
                    kind = MOUND
                if stamp_behind is not None and stamp_behind[0] == (x, y):
                    kind = stamp_behind[1]
                self._draw_tile(frame, x, y, kind)
                if stamp_current == (x, y):
                    self._draw_stamp_depress(frame, x, y)
                if stamp_preview == (x, y):
                    self._draw_stamp_preview(frame, x, y)
                if pulse == (x, y):
                    self._draw_pulse(frame, x, y)
        if glint:
            for x, y in self.mold_cells:
                if self._cell(x, y) == MOLD_FILLED:
                    x0, y0 = self._cell_top_left(x, y)
                    frame[y0 : y0 + 2, x0 : x0 + 2] = COLOR_GLINT
        return frame

    def _draw_tile(self, frame: np.ndarray, x: int, y: int, kind: str) -> None:
        x0, y0 = self._cell_top_left(x, y)
        s = self.tile
        if kind == STONE:
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_STONE
            return
        if kind in (FLAT, MOUND, PIT, MOLD_EMPTY, MOLD_FILLED):
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_FLAT
        if kind == PIT:
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_RIM
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_PIT
            frame[y0 + s - 2 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_DEEP
        elif kind == MOUND:
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_MOUND
            frame[y0 + s - 2 : y0 + s - 1, x0 + 2 : x0 + s - 1] = COLOR_SHADOW
            frame[y0 + 1, x0 + 1] = COLOR_FLAT
            frame[y0 + 1, x0 + s - 2] = COLOR_FLAT
        elif kind == MOLD_EMPTY:
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_MOLD
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_STONE
        elif kind == MOLD_FILLED:
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_MOLD
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_PLUG
            frame[y0 + 1, x0 + 1 : x0 + s - 2] = COLOR_MOUND
        elif kind == KILN:
            frame[y0 : y0 + s, x0 : x0 + s] = COLOR_SHADOW
            frame[y0 + 1 : y0 + s, x0 + 1 : x0 + s - 1] = COLOR_KILN_HOT
            frame[y0 + s - 2 : y0 + s, x0 + 2 : x0 + s - 2] = COLOR_BG

    def _draw_stamp_depress(self, frame: np.ndarray, x: int, y: int) -> None:
        x0, y0 = self._cell_top_left(x, y)
        s = self.tile
        frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_PIT
        frame[y0 + s - 2 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_DEEP

    def _draw_stamp_preview(self, frame: np.ndarray, x: int, y: int) -> None:
        x0, y0 = self._cell_top_left(x, y)
        s = self.tile
        phase = self.remaining_steps % 2
        crack_color = COLOR_GLINT if phase == 0 else COLOR_DEEP
        inset = max(2, s // 3)
        left = x0 + inset
        right = x0 + s - inset - 1
        top = y0 + inset
        bottom = y0 + s - inset - 1
        if left > right or top > bottom:
            cx = x0 + s // 2
            cy = y0 + s // 2
            frame[cy, cx] = crack_color
            return
        frame[top, left : right + 1] = crack_color
        frame[bottom, left : right + 1] = crack_color
        frame[top : bottom + 1, left] = crack_color
        frame[top : bottom + 1, right] = crack_color
        frame[top, left] = COLOR_FLAT
        frame[bottom, right] = COLOR_FLAT

    def _draw_pulse(self, frame: np.ndarray, x: int, y: int) -> None:
        x0, y0 = self._cell_top_left(x, y)
        s = self.tile
        frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_GLINT
