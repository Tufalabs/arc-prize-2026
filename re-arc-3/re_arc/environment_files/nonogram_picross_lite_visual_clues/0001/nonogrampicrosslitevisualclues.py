from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "nonogram_picross_lite_visual_clues-0001"

COLOR_BG = 0
COLOR_BOARD_BG = 1
COLOR_FRAME = 2
COLOR_UNKNOWN = 3
COLOR_MARKED_EMPTY = 4
COLOR_FILL_A = 5
COLOR_FILL_B = 6
COLOR_CLUE_A = 7
COLOR_CLUE_B = 8
COLOR_TIME_REMAIN = 9
COLOR_TIME_SPENT = 10
COLOR_PROGRESS_FILL = 11
COLOR_PROGRESS_EMPTY = 12
COLOR_STRIKE_EMPTY = 13
COLOR_STRIKE_FILLED = 14
COLOR_WARNING = 15

PLAYER_UNKNOWN = 0
PLAYER_MARKED_EMPTY = 1
PLAYER_FILL_A = 2
PLAYER_FILL_B = 3

SOLUTION_EMPTY = 0
SOLUTION_A = 1
SOLUTION_B = 2

MAX_WIDTH = 47
MAX_HEIGHT = 43


@dataclass(frozen=True)
class LevelSpec:
    name: str
    n: int
    w: int
    h: int
    time_limit: int
    two_color: bool
    solution_rows: tuple[str, ...]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _parse_solution(rows: tuple[str, ...], n: int) -> tuple[tuple[int, ...], ...]:
    parsed: list[tuple[int, ...]] = []
    if len(rows) != n:
        raise ValueError(f"Expected {n} solution rows, got {len(rows)}")
    for row in rows:
        if len(row) != n:
            raise ValueError("Solution row length must equal N")
        out: list[int] = []
        for ch in row:
            if ch == ".":
                out.append(SOLUTION_EMPTY)
            elif ch == "#":
                out.append(SOLUTION_A)
            elif ch == "@":
                out.append(SOLUTION_B)
            else:
                raise ValueError(f"Unsupported solution character: {ch!r}")
        parsed.append(tuple(out))
    return tuple(parsed)


def _extract_runs(line: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    color = 0
    length = 0
    for value in line:
        if value == 0:
            if length > 0:
                runs.append((color, length))
                color = 0
                length = 0
            continue
        if color == value:
            length += 1
            continue
        if length > 0:
            runs.append((color, length))
        color = value
        length = 1
    if length > 0:
        runs.append((color, length))
    return runs


def _build_clues(
    solution: tuple[tuple[int, ...], ...],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    n = len(solution)
    row_clues: list[list[tuple[int, int]]] = []
    col_clues: list[list[tuple[int, int]]] = []
    for y in range(n):
        row_clues.append(_extract_runs([int(v) for v in solution[y]]))
    for x in range(n):
        col = [int(solution[y][x]) for y in range(n)]
        col_clues.append(_extract_runs(col))
    return row_clues, col_clues


def _grid_size_for(n: int, w: int, h: int) -> tuple[int, int]:
    return (w + n + 3, h + n + 7)


def _build_level(spec: LevelSpec) -> Level:
    solution = _parse_solution(spec.solution_rows, spec.n)
    row_clues, col_clues = _build_clues(solution)
    width, height = _grid_size_for(spec.n, spec.w, spec.h)

    return Level(
        name=spec.name,
        grid_size=(width, height),
        sprites=[
            Sprite(
                pixels=_solid(width, height, COLOR_BG),
                name="board",
                x=0,
                y=0,
                layer=0,
                tags=["board"],
                collidable=False,
            )
        ],
        data={
            "n": int(spec.n),
            "w": int(spec.w),
            "h": int(spec.h),
            "time_limit": int(spec.time_limit),
            "two_color": bool(spec.two_color),
            "solution": [list(row) for row in solution],
            "row_clues": [[(int(c), int(k)) for c, k in line] for line in row_clues],
            "col_clues": [[(int(c), int(k)) for c, k in line] for line in col_clues],
        },
    )


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Nonogram Visual Clues L1",
        n=5,
        w=5,
        h=5,
        time_limit=40,
        two_color=False,
        solution_rows=("..#..", "..#..", "#####", "..#..", "..#.."),
    ),
    LevelSpec(
        name="Nonogram Visual Clues L2",
        n=8,
        w=5,
        h=5,
        time_limit=70,
        two_color=False,
        solution_rows=("##....##", "##....##", "........", "..####..", "..####..", "........", "##....##", "##....##"),
    ),
    LevelSpec(
        name="Nonogram Visual Clues L3",
        n=10,
        w=10,
        h=10,
        time_limit=110,
        two_color=False,
        solution_rows=(
            "..##..##..",
            ".########.",
            "###.##.###",
            "##########",
            "##.####.##",
            "##......##",
            "..######..",
            ".##....##.",
            "##......##",
            "..##..##..",
        ),
    ),
    LevelSpec(
        name="Nonogram Visual Clues L4",
        n=12,
        w=12,
        h=12,
        time_limit=150,
        two_color=False,
        solution_rows=(
            "############",
            "#..######..#",
            "#..#....#..#",
            "####.##.####",
            "...#....#...",
            "..###..###..",
            "..###..###..",
            "...#....#...",
            "####.##.####",
            "#..#....#..#",
            "#..######..#",
            "############",
        ),
    ),
    LevelSpec(
        name="Nonogram Visual Clues L5",
        n=12,
        w=14,
        h=14,
        time_limit=180,
        two_color=True,
        solution_rows=(
            ".....##.....",
            "....#@@#....",
            "...#@@@@#...",
            "..#@@@@@@#..",
            ".#@@@@@@@@#.",
            "#@@@@@@@@@@#",
            "#@@@@@@@@@@#",
            ".#@@@@@@@@#.",
            "..#@@@@@@#..",
            "...#@@@@#...",
            "....#@@#....",
            ".....##.....",
        ),
    ),
    LevelSpec(
        name="Nonogram Visual Clues L6",
        n=16,
        w=24,
        h=24,
        time_limit=260,
        two_color=True,
        solution_rows=(
            "....##....##....",
            "...####..####...",
            "..##@@####@@##..",
            ".##@@@@##@@@@##.",
            "##@@@@@##@@@@@##",
            "##@@##@##@##@@##",
            "###@@@####@@@###",
            "####@######@####",
            "####@######@####",
            "###@@@####@@@###",
            "##@@##@##@##@@##",
            "##@@@@@##@@@@@##",
            ".##@@@@##@@@@##.",
            "..##@@####@@##..",
            "...####..####...",
            "....##....##....",
        ),
    ),
)


LEVELS = [_build_level(spec) for spec in LEVEL_SPECS]


def _deserialize_model(level: Level) -> dict:
    return {
        "n": int(level.get_data("n")),
        "w": int(level.get_data("w")),
        "h": int(level.get_data("h")),
        "time_limit": int(level.get_data("time_limit")),
        "two_color": bool(level.get_data("two_color")),
        "solution": tuple(tuple(int(v) for v in row) for row in (level.get_data("solution") or [])),
        "row_clues": [[(int(c), int(k)) for c, k in line] for line in (level.get_data("row_clues") or [])],
        "col_clues": [[(int(c), int(k)) for c, k in line] for line in (level.get_data("col_clues") or [])],
    }


class NonogramPicrossLiteVisualClues(ARCBaseGame):
    def __init__(self, seed: int = 0):
        camera = Camera(width=MAX_WIDTH, height=MAX_HEIGHT, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID, levels=LEVELS, camera=camera, win_score=len(LEVELS), available_actions=[5, 6], seed=seed
        )

        self._board_sprite: Sprite | None = None
        self._model: dict | None = None
        self._player_grid: list[list[int]] = []
        self._time_left = 0
        self._strikes = 0
        self._paint = SOLUTION_A
        self._prev_mismatches = 0
        self._initial_mismatches = 1
        self._fail_animation_steps = 0
        self._fail_frame_on = False
        self._row_flash: list[int] = []
        self._col_flash: list[int] = []
        self._puzzle_x0 = 0
        self._puzzle_y0 = 0

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        n = int(self._model["n"])
        self._board_sprite = next(iter(level.get_sprites_by_name("board")), None)
        self._puzzle_x0 = 2 + int(self._model["w"])
        self._puzzle_y0 = 2 + int(self._model["h"])
        self._initialize_level_runtime_state(n)
        self._render()

    def _initialize_level_runtime_state(self, n: int) -> None:
        self._player_grid = [[PLAYER_UNKNOWN for _ in range(n)] for _ in range(n)]
        self._time_left = int(self._model["time_limit"])
        self._strikes = 0
        self._paint = SOLUTION_A
        self._fail_animation_steps = 0
        self._fail_frame_on = False
        self._row_flash = [0 for _ in range(n)]
        self._col_flash = [0 for _ in range(n)]

        self._initial_mismatches = max(1, self._mismatch_count())
        self._prev_mismatches = self._mismatch_count()

    def _color_for_solution_fill(self, value: int) -> int:
        if value == SOLUTION_B:
            return COLOR_FILL_B
        return COLOR_FILL_A

    def _color_for_clue_fill(self, value: int) -> int:
        if value == SOLUTION_B:
            return COLOR_CLUE_B
        return COLOR_CLUE_A

    def _player_to_solution_value(self, value: int) -> int:
        if value == PLAYER_FILL_A:
            return SOLUTION_A
        if value == PLAYER_FILL_B:
            return SOLUTION_B
        return SOLUTION_EMPTY

    def _mismatch_count(self) -> int:
        if self._model is None:
            return 0
        solution = self._model["solution"]
        n = int(self._model["n"])
        mismatches = 0
        for y in range(n):
            for x in range(n):
                expected = int(solution[y][x])
                actual = self._player_to_solution_value(int(self._player_grid[y][x]))
                if expected != actual:
                    mismatches += 1
        return mismatches

    def _in_puzzle(self, gx: int, gy: int) -> bool:
        if self._model is None:
            return False
        n = int(self._model["n"])
        return self._puzzle_x0 <= gx < self._puzzle_x0 + n and self._puzzle_y0 <= gy < self._puzzle_y0 + n

    def _cycle_cell(self, px: int, py: int) -> None:
        if self._model is None:
            return
        value = int(self._player_grid[py][px])
        two_color = bool(self._model["two_color"])
        if value == PLAYER_UNKNOWN:
            if two_color and self._paint == SOLUTION_B:
                self._player_grid[py][px] = PLAYER_FILL_B
            else:
                self._player_grid[py][px] = PLAYER_FILL_A
        elif value in (PLAYER_FILL_A, PLAYER_FILL_B):
            self._player_grid[py][px] = PLAYER_MARKED_EMPTY
        else:
            self._player_grid[py][px] = PLAYER_UNKNOWN

    def _extract_player_runs(self, values: list[int]) -> list[tuple[int, int]]:
        normalized = [self._player_to_solution_value(v) for v in values]
        return _extract_runs(normalized)

    def _line_feedback(self) -> tuple[list[int], list[int]]:
        if self._model is None:
            return [], []

        n = int(self._model["n"])
        row_clues = self._model["row_clues"]
        col_clues = self._model["col_clues"]

        next_rows = [0 for _ in range(n)]
        next_cols = [0 for _ in range(n)]

        for y in range(n):
            player_runs = self._extract_player_runs([int(v) for v in self._player_grid[y]])
            clue_runs = list(row_clues[y])
            if self._line_impossible(player_runs, clue_runs):
                next_rows[y] = COLOR_WARNING
            elif player_runs == clue_runs:
                next_rows[y] = COLOR_STRIKE_FILLED

        for x in range(n):
            col = [int(self._player_grid[y][x]) for y in range(n)]
            player_runs = self._extract_player_runs(col)
            clue_runs = list(col_clues[x])
            if self._line_impossible(player_runs, clue_runs):
                next_cols[x] = COLOR_WARNING
            elif player_runs == clue_runs:
                next_cols[x] = COLOR_STRIKE_FILLED

        return next_rows, next_cols

    def _line_impossible(self, player_runs: list[tuple[int, int]], clue_runs: list[tuple[int, int]]) -> bool:
        if len(player_runs) > len(clue_runs):
            return True
        for idx, (color, length) in enumerate(player_runs):
            if idx >= len(clue_runs):
                return True
            clue_color, clue_len = clue_runs[idx]
            if int(color) != int(clue_color):
                return True
            if int(length) > int(clue_len):
                return True
        return False

    def _apply_click(self, action_data: dict) -> None:
        if self._model is None:
            return
        x = int(action_data.get("x", -1))
        y = int(action_data.get("y", -1))
        point = self.camera.display_to_grid(x, y)
        if point is None:
            return
        gx, gy = int(point[0]), int(point[1])
        if not self._in_puzzle(gx, gy):
            return
        px = gx - self._puzzle_x0
        py = gy - self._puzzle_y0
        self._cycle_cell(px, py)

    def _trigger_failure(self) -> None:
        self.lose()

    def _render(self) -> None:
        if self._model is None or self._board_sprite is None:
            return

        n = int(self._model["n"])
        w = int(self._model["w"])
        h = int(self._model["h"])
        width, height = _grid_size_for(n, w, h)

        grid = _solid(width, height, COLOR_BG)

        # Board interior and fixed separators.
        grid[1 : height - 1, 1 : width - 1] = np.int8(COLOR_BOARD_BG)

        frame_color = COLOR_WARNING if (self._fail_animation_steps > 0 and self._fail_frame_on) else COLOR_FRAME
        grid[0, :] = np.int8(frame_color)
        grid[height - 1, :] = np.int8(frame_color)
        grid[:, 0] = np.int8(frame_color)
        grid[:, width - 1] = np.int8(frame_color)

        x_sep = 1 + w
        y_sep = 1 + h
        y_status_sep = y_sep + 1 + n

        grid[y_sep, 1 : width - 1] = np.int8(COLOR_FRAME)
        grid[1:y_sep, x_sep] = np.int8(COLOR_FRAME)
        grid[y_sep, x_sep] = np.int8(COLOR_FRAME)
        grid[y_status_sep, 1 : width - 1] = np.int8(COLOR_FRAME)

        # Progress row.
        y_progress = y_status_sep + 1
        if 1 <= y_progress < height - 1:
            total = w + n
            mismatch = self._mismatch_count()
            filled = round((1.0 - (mismatch / float(max(1, self._initial_mismatches)))) * total)
            filled = max(0, min(total, filled))
            for i in range(total):
                color = COLOR_PROGRESS_FILL if i < filled else COLOR_PROGRESS_EMPTY
                if i < w:
                    grid[y_progress, 1 + i] = np.int8(color)
                else:
                    grid[y_progress, x_sep + 1 + (i - w)] = np.int8(color)
            grid[y_progress, x_sep] = np.int8(COLOR_FRAME)

        # Time row.
        y_time = y_status_sep + 2
        if 1 <= y_time < height - 1:
            total = w + n
            filled = round((self._time_left / float(max(1, int(self._model["time_limit"])))) * total)
            filled = max(0, min(total, filled))
            for i in range(total):
                color = COLOR_TIME_REMAIN if i < filled else COLOR_TIME_SPENT
                if i < w:
                    grid[y_time, 1 + i] = np.int8(color)
                else:
                    grid[y_time, x_sep + 1 + (i - w)] = np.int8(color)
            grid[y_time, x_sep] = np.int8(COLOR_FRAME)

        # Strikes + paint indicator row.
        y_strikes = y_status_sep + 3
        if 1 <= y_strikes < height - 1:
            for sx in range(3):
                grid[y_strikes, 1 + sx] = np.int8(COLOR_STRIKE_FILLED if sx < self._strikes else COLOR_STRIKE_EMPTY)
            grid[y_strikes, x_sep] = np.int8(COLOR_FRAME)

            paint_color = COLOR_FILL_B if self._paint == SOLUTION_B else COLOR_FILL_A
            right_start = x_sep + 1
            if right_start + n - 1 < width - 1 and n >= 2:
                grid[y_strikes, right_start + n - 2] = np.int8(paint_color)
                grid[y_strikes, right_start + n - 1] = np.int8(paint_color)

        # Puzzle region.
        for py in range(n):
            for px in range(n):
                value = int(self._player_grid[py][px])
                gx = self._puzzle_x0 + px
                gy = self._puzzle_y0 + py
                if value == PLAYER_UNKNOWN:
                    color = COLOR_UNKNOWN
                elif value == PLAYER_MARKED_EMPTY:
                    color = COLOR_MARKED_EMPTY
                elif value == PLAYER_FILL_B:
                    color = COLOR_FILL_B
                else:
                    color = COLOR_FILL_A
                grid[gy, gx] = np.int8(color)

        # Clue regions.
        row_clues = self._model["row_clues"]
        col_clues = self._model["col_clues"]

        for y in range(n):
            clues = list(row_clues[y])
            required = sum(k for _, k in clues) + max(0, len(clues) - 1)
            cursor = max(0, w - required)
            flash = int(self._row_flash[y])
            for color_value, length in clues:
                for _ in range(int(length)):
                    gx = 1 + cursor
                    gy = self._puzzle_y0 + y
                    base_color = self._color_for_clue_fill(int(color_value))
                    grid[gy, gx] = np.int8(flash if flash else base_color)
                    cursor += 1
                cursor += 1

        for x in range(n):
            clues = list(col_clues[x])
            required = sum(k for _, k in clues) + max(0, len(clues) - 1)
            cursor = max(0, h - required)
            flash = int(self._col_flash[x])
            for color_value, length in clues:
                for _ in range(int(length)):
                    gx = self._puzzle_x0 + x
                    gy = 1 + cursor
                    base_color = self._color_for_clue_fill(int(color_value))
                    grid[gy, gx] = np.int8(flash if flash else base_color)
                    cursor += 1
                cursor += 1

        self._board_sprite.pixels = grid

    def step(self) -> None:
        action_id = int(self.action.id.value)

        if action_id == int(GameAction.ACTION6.value):
            self._apply_click(self.action.data or {})
        elif action_id == int(GameAction.ACTION5.value) and self._model is not None and bool(self._model["two_color"]):
            self._paint = SOLUTION_B if self._paint == SOLUTION_A else SOLUTION_A

        self._time_left -= 1

        mismatches = self._mismatch_count()
        if mismatches > self._prev_mismatches:
            self._strikes += 1
        self._prev_mismatches = mismatches

        if mismatches == 0:
            self._render()
            self.next_level()
            self.complete_action()
            return

        if self._time_left <= 0 or self._strikes >= 3:
            self._render()
            self._trigger_failure()
            self.complete_action()
            return

        self._render()
        next_row_flash, next_col_flash = self._line_feedback()
        self._row_flash = next_row_flash
        self._col_flash = next_col_flash

        self.complete_action()
