from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from math import gcd

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "frogger-0001"

GRID_W = 32
GRID_H = 15
INTERIOR_X_MIN = 1
INTERIOR_X_MAX = 30

COLOR_WALL = 0
COLOR_GRASS = 1
COLOR_ROAD = 2
COLOR_WATER = 3
COLOR_FROG = 4
COLOR_CAR = 5
COLOR_TRUCK = 6
COLOR_LOG = 7
COLOR_TURTLE_SAFE = 8
COLOR_HOME_EMPTY = 9
COLOR_HOME_FILLED = 10
COLOR_CROC = 11
COLOR_TIME_FILLED = 12
COLOR_TIME_EMPTY = 13
COLOR_WAKE = 14
COLOR_FLASH = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION_ID = int(GameAction.ACTION5.value)
ACTION_IDS = [
    int(GameAction.ACTION1.value),
    int(GameAction.ACTION2.value),
    int(GameAction.ACTION3.value),
    int(GameAction.ACTION4.value),
    WAIT_ACTION_ID,
]


def _action_id(value) -> int:
    return int(getattr(value, "value", value))


VEHICLE_CHARS = {"<", ">", "{", "}", "[", "]"}
FLOAT_CHARS = {"=", "(", ")", "*", "o", "&", "%"}
CROC_CHARS = {"&", "%"}

TURTLE_CYCLE = ("safe", "safe", "safe", "warning", "submerged", "submerged")


class LaneDef:
    def __init__(self, *, row: int, kind: str, direction: int, move_period: int, uses_turtles: bool = False):
        self.row = int(row)
        self.kind = str(kind)
        self.direction = int(direction)
        self.move_period = int(move_period)
        self.uses_turtles = bool(uses_turtles)


class LevelDef:
    def __init__(
        self,
        *,
        name: str,
        rows: tuple[str, ...],
        time_fill: int,
        time_drain_period: int,
        homes_required: int,
        lanes: tuple[LaneDef, ...],
    ):
        self.name = str(name)
        self.rows = tuple(str(row) for row in rows)
        self.time_fill = int(time_fill)
        self.time_drain_period = int(time_drain_period)
        self.homes_required = int(homes_required)
        self.lanes = tuple(lanes)


LEVEL_DEFS: tuple[LevelDef, ...] = (
    LevelDef(
        name="Level 1",
        rows=(
            "#==============================#",
            "#..............^...............#",
            "#~~~~o~~~~o~~~~o~~~~o~~~~o~~~~~#",
            "#~~~~o~~~~o~~~~o~~~~o~~~~o~~~~~#",
            "#..............................#",
            "#~~~~o~~~~o~~~~o~~~~o~~~~o~~~~~#",
            "#~~~~o~~~~o~~~~o~~~~o~~~~o~~~~~#",
            "#..............................#",
            "#______________________________#",
            "#______________________________#",
            "#______________________________#",
            "#______________________________#",
            "#__{<<______{<<_______{<<______#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=30,
        time_drain_period=2,
        homes_required=1,
        lanes=(
            LaneDef(row=2, kind="water", direction=0, move_period=0),
            LaneDef(row=3, kind="water", direction=0, move_period=0),
            LaneDef(row=5, kind="water", direction=0, move_period=0),
            LaneDef(row=6, kind="water", direction=0, move_period=0),
            LaneDef(row=12, kind="road", direction=-1, move_period=3),
        ),
    ),
    LevelDef(
        name="Level 2",
        rows=(
            "#==============================#",
            "#..~~~~..~~~~..^..~~~~..~~~~...#",
            "#====)~~~~~====)~~~~~====)~~~~~#",
            "#~~~~~(====~~~~~(====~~~~~(====#",
            "#~~(====~~~~~~(====~~~~~~(====~#",
            "#===)~~~===)~~~===)~~~===)~~~~~#",
            "#~~(===~~~(===~~~(===~~~(===~~~#",
            "#..............................#",
            "#}>>____}>>____}>>____}>>______#",
            "#___{<<___{<<___{<<___{<<______#",
            "#__}>>____}>>____}>>____}>>____#",
            "#__{<<____{<<____{<<____{<<____#",
            "#______}>>___}>>___}>>___}>>___#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=30,
        time_drain_period=2,
        homes_required=1,
        lanes=(
            LaneDef(row=2, kind="water", direction=1, move_period=2),
            LaneDef(row=3, kind="water", direction=-1, move_period=3),
            LaneDef(row=4, kind="water", direction=-1, move_period=2),
            LaneDef(row=5, kind="water", direction=1, move_period=2),
            LaneDef(row=6, kind="water", direction=-1, move_period=2),
            LaneDef(row=8, kind="road", direction=1, move_period=2),
            LaneDef(row=9, kind="road", direction=-1, move_period=1),
            LaneDef(row=10, kind="road", direction=1, move_period=3),
            LaneDef(row=11, kind="road", direction=-1, move_period=2),
            LaneDef(row=12, kind="road", direction=1, move_period=2),
        ),
    ),
    LevelDef(
        name="Level 3",
        rows=(
            "#==============================#",
            "#..^..~~~~..^..~~~~..^..~~~~...#",
            "#====)~~~~~====)~~~~~====)~~~~~#",
            "#***~~~~***~~~~***~~~~***~~~~~~#",
            "#~~~~~(====~~~~~(====~~~~~(====#",
            "#===)~~~===)~~~===)~~~===)~~~~~#",
            "#~~(===~~~(===~~~(===~~~(===~~~#",
            "#..............................#",
            "#}>>_}>>_}>>_}>>_}>>_}>>_}>>___#",
            "#{<<__{<<__{<<__{<<__{<<__{<<__#",
            "#__}>>____}>>____}>>____}>>____#",
            "#{<<____{<<____{<<____{<<______#",
            "#______}>>______}>>______}>>___#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=30,
        time_drain_period=1,
        homes_required=3,
        lanes=(
            LaneDef(row=2, kind="water", direction=1, move_period=2),
            LaneDef(row=3, kind="water", direction=-1, move_period=2, uses_turtles=True),
            LaneDef(row=4, kind="water", direction=-1, move_period=1),
            LaneDef(row=5, kind="water", direction=1, move_period=2),
            LaneDef(row=6, kind="water", direction=-1, move_period=2),
            LaneDef(row=8, kind="road", direction=1, move_period=1),
            LaneDef(row=9, kind="road", direction=-1, move_period=1),
            LaneDef(row=10, kind="road", direction=1, move_period=2),
            LaneDef(row=11, kind="road", direction=-1, move_period=2),
            LaneDef(row=12, kind="road", direction=1, move_period=3),
        ),
    ),
    LevelDef(
        name="Level 4",
        rows=(
            "#==============================#",
            "#.^..~~.^..~~.^..~~.^..~~......#",
            "#=====&)~~~~~~~~=====&)~~~~~~~~#",
            "#***~~~***~~~***~~~***~~~***~~~#",
            "#~~~~~(&===~~~~~(&===~~~~~(&===#",
            "#=&=)~~=&=)~~=&=)~~=&=)~~=&=)~~#",
            "#~~(&==~~(&==~~(&==~~(&==~~(&==#",
            "#..............................#",
            "#}]]]]____}]]]]____}]]]]_______#",
            "#{<<_{<<_{<<_{<<_{<<_{<<_{<<___#",
            "#{[[[[_____{[[[[_____{[[[[_____#",
            "#}>>_}>>_}>>_}>>_}>>_}>>_}>>___#",
            "#}]]]]__}>>_____}]]]]__}>>_____#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=30,
        time_drain_period=1,
        homes_required=4,
        lanes=(
            LaneDef(row=2, kind="water", direction=1, move_period=2),
            LaneDef(row=3, kind="water", direction=-1, move_period=2, uses_turtles=True),
            LaneDef(row=4, kind="water", direction=-1, move_period=1),
            LaneDef(row=5, kind="water", direction=1, move_period=1),
            LaneDef(row=6, kind="water", direction=-1, move_period=1),
            LaneDef(row=8, kind="road", direction=1, move_period=3),
            LaneDef(row=9, kind="road", direction=-1, move_period=1),
            LaneDef(row=10, kind="road", direction=-1, move_period=2),
            LaneDef(row=11, kind="road", direction=1, move_period=1),
            LaneDef(row=12, kind="road", direction=1, move_period=2),
        ),
    ),
    LevelDef(
        name="Level 5",
        rows=(
            "#========================------#",
            "#........^...^...^...^.........#",
            "#==)~~~~==)~~~~==)~~~~==)~~~~==#",
            "#***~~~~***~~~~***~~~~***~~~~~~#",
            "#(===~~~~(===~~~~(===~~~~(===~~#",
            "#=&)~~~~=&)~~~~=&)~~~~=&)~~~~=&#",
            "#***~~~~***~~~~***~~~~***~~~~~~#",
            "#..............................#",
            "#{<<_____{<<_____{<<_____{<<___#",
            "#}]]]]___}]]]]___}]]]]___}]]]]_#",
            "#{[[[___{[[[____{[[[____{[[[___#",
            "#}>>____{>>_____{>>_____{>>____#",
            "#{[[[[__{<<_____{[[[[___{<<____#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=24,
        time_drain_period=1,
        homes_required=4,
        lanes=(
            LaneDef(row=2, kind="water", direction=1, move_period=1),
            LaneDef(row=3, kind="water", direction=1, move_period=1, uses_turtles=True),
            LaneDef(row=4, kind="water", direction=-1, move_period=1),
            LaneDef(row=5, kind="water", direction=1, move_period=1),
            LaneDef(row=6, kind="water", direction=-1, move_period=2, uses_turtles=True),
            LaneDef(row=8, kind="road", direction=-1, move_period=1),
            LaneDef(row=9, kind="road", direction=1, move_period=1),
            LaneDef(row=10, kind="road", direction=-1, move_period=1),
            LaneDef(row=11, kind="road", direction=1, move_period=1),
            LaneDef(row=12, kind="road", direction=-1, move_period=1),
        ),
    ),
    LevelDef(
        name="Level 6",
        rows=(
            "#====================----------#",
            "#..^...^...^...^...^...........#",
            "#=====&================&=======#",
            "#******************************#",
            "#(&===(&===(&===(&===(&===(&===#",
            "#******************************#",
            "#(=============================#",
            "#..............................#",
            "#}>>___________________________#",
            "#___________________________{<<#",
            "#____}>>_______________________#",
            "#_________________{<<__________#",
            "#______}]]]]___________________#",
            "#..............@...............#",
            "################################",
        ),
        time_fill=20,
        time_drain_period=1,
        homes_required=5,
        lanes=(
            LaneDef(row=2, kind="water", direction=1, move_period=1),
            LaneDef(row=3, kind="water", direction=-1, move_period=1, uses_turtles=True),
            LaneDef(row=4, kind="water", direction=-1, move_period=1),
            LaneDef(row=5, kind="water", direction=1, move_period=1, uses_turtles=True),
            LaneDef(row=6, kind="water", direction=-1, move_period=1),
            LaneDef(row=8, kind="road", direction=1, move_period=1),
            LaneDef(row=9, kind="road", direction=-1, move_period=1),
            LaneDef(row=10, kind="road", direction=1, move_period=1),
            LaneDef(row=11, kind="road", direction=-1, move_period=1),
            LaneDef(row=12, kind="road", direction=1, move_period=2),
        ),
    ),
)


def _lcm(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        if value <= 0:
            continue
        out = (out * value) // gcd(out, value)
    return out


class FroggerModel:
    """Pure simulation model shared by runtime and DSL planner."""

    def __init__(self, level_def: LevelDef):
        self.level_def = level_def
        self.static_rows = [list(row) for row in level_def.rows]
        self.lanes_by_row = {lane.row: lane for lane in level_def.lanes}
        self.row_patterns = {
            lane.row: list(level_def.rows[lane.row][INTERIOR_X_MIN : INTERIOR_X_MAX + 1]) for lane in level_def.lanes
        }
        self.home_positions: list[tuple[int, int]] = []
        self.home_index_by_pos: dict[tuple[int, int], int] = {}

        spawn: tuple[int, int] | None = None
        for y, row in enumerate(self.static_rows):
            if len(row) != GRID_W:
                raise ValueError(f"Invalid row width on y={y}: expected {GRID_W}, got {len(row)}")
            for x, char in enumerate(row):
                if char == "@":
                    spawn = (x, y)
                    self.static_rows[y][x] = "."
                elif char == "^":
                    idx = len(self.home_positions)
                    self.home_positions.append((x, y))
                    self.home_index_by_pos[(x, y)] = idx
        if spawn is None:
            raise ValueError("Each level must define a spawn '@' cell.")

        self.spawn = spawn
        self.homes_required = int(level_def.homes_required)
        self.time_fill = int(level_def.time_fill)
        self.time_drain_period = int(level_def.time_drain_period)

        periods = [lane.move_period * 30 for lane in level_def.lanes if lane.move_period > 0]
        if any(lane.uses_turtles for lane in level_def.lanes):
            periods.append(len(TURTLE_CYCLE))
        self.phase_cycle = max(1, _lcm(periods) or 1)

        self.player_x = int(spawn[0])
        self.player_y = int(spawn[1])
        self.time_left = int(self.time_fill)
        self.drain_tick = 0
        self.phase = 0
        self.filled_mask = 0
        self.pending_advance = False

    def clone(self) -> FroggerModel:
        other = FroggerModel(self.level_def)
        other.player_x = self.player_x
        other.player_y = self.player_y
        other.time_left = self.time_left
        other.drain_tick = self.drain_tick
        other.phase = self.phase
        other.filled_mask = self.filled_mask
        other.pending_advance = self.pending_advance
        other.row_patterns = {y: chars[:] for y, chars in self.row_patterns.items()}
        return other

    def state_key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.player_x),
            int(self.player_y),
            int(self.filled_mask),
            int(self.phase),
            int(self.drain_tick % max(1, self.time_drain_period)),
        )

    def is_solved(self) -> bool:
        return self._filled_count() >= self.homes_required

    def _filled_count(self) -> int:
        return int(self.filled_mask).bit_count()

    def _is_home_filled(self, x: int, y: int) -> bool:
        idx = self.home_index_by_pos.get((x, y))
        if idx is None:
            return False
        return bool(self.filled_mask & (1 << idx))

    def _is_blocked_for_move(self, x: int, y: int) -> bool:
        if x <= 0 or x >= GRID_W - 1:
            return True
        if y <= 0 or y >= GRID_H:
            return True
        if y == GRID_H - 1:
            return True
        if self._is_home_filled(x, y):
            return True
        cell = self.static_rows[y][x]
        return cell == "#"

    def _lane_char(self, row: int, x: int) -> str | None:
        pattern = self.row_patterns.get(row)
        if pattern is None or x < INTERIOR_X_MIN or x > INTERIOR_X_MAX:
            return None
        return pattern[x - INTERIOR_X_MIN]

    def _is_turtle_submerged(self) -> bool:
        return TURTLE_CYCLE[self.phase % len(TURTLE_CYCLE)] == "submerged"

    def _is_float_char(self, char: str, lane: LaneDef | None) -> bool:
        if char not in FLOAT_CHARS:
            return False
        return not (char == "*" and lane is not None and lane.uses_turtles and self._is_turtle_submerged())

    def _cell_is_road_hazard(self, x: int, y: int) -> bool:
        lane = self.lanes_by_row.get(y)
        if lane is None or lane.kind != "road":
            return False
        char = self._lane_char(y, x)
        return bool(char in VEHICLE_CHARS)

    def _cell_has_water_background(self, _x: int, y: int) -> bool:
        lane = self.lanes_by_row.get(y)
        return bool(lane is not None and lane.kind == "water")

    def _cell_is_float(self, x: int, y: int) -> bool:
        lane = self.lanes_by_row.get(y)
        if lane is None:
            return False
        char = self._lane_char(y, x)
        if char is None:
            return False
        return self._is_float_char(char, lane)

    def _cell_is_croc(self, x: int, y: int) -> bool:
        lane = self.lanes_by_row.get(y)
        if lane is None or lane.kind != "water":
            return False
        char = self._lane_char(y, x)
        return bool(char in CROC_CHARS)

    def _do_lane_moves(self) -> dict[int, int]:
        moved: dict[int, int] = {}
        for lane in self.level_def.lanes:
            if lane.move_period <= 0:
                continue
            if (self.phase + 1) % lane.move_period != 0:
                continue
            pattern = self.row_patterns[lane.row]
            if lane.direction > 0:
                self.row_patterns[lane.row] = [pattern[-1], *pattern[:-1]]
                moved[lane.row] = 1
            elif lane.direction < 0:
                self.row_patterns[lane.row] = [*pattern[1:], pattern[0]]
                moved[lane.row] = -1
        return moved

    def _respawn_with_time_reset(self) -> None:
        self.player_x = int(self.spawn[0])
        self.player_y = int(self.spawn[1])
        self.time_left = int(self.time_fill)
        self.drain_tick = 0

    def apply_action(self, action_id: int) -> str:
        if self.pending_advance:
            return "level_complete"

        action_num = _action_id(action_id)
        dx, dy = MOVE_DELTAS.get(action_num, (0, 0))
        if action_num == WAIT_ACTION_ID:
            dx, dy = 0, 0

        nx = self.player_x + int(dx)
        ny = self.player_y + int(dy)
        if not self._is_blocked_for_move(nx, ny):
            self.player_x = nx
            self.player_y = ny

        if self._cell_is_road_hazard(self.player_x, self.player_y):
            self.phase = (self.phase + 1) % self.phase_cycle
            return "death"

        carry_lane = self.lanes_by_row.get(self.player_y)
        was_on_float = bool(
            carry_lane is not None and carry_lane.kind == "water" and self._cell_is_float(self.player_x, self.player_y)
        )

        moved_rows = self._do_lane_moves()

        if was_on_float and self.player_y in moved_rows:
            self.player_x += int(moved_rows[self.player_y])
            if self.player_x <= 0 or self.player_x >= GRID_W - 1:
                self.phase = (self.phase + 1) % self.phase_cycle
                return "death"

        if self._cell_has_water_background(self.player_x, self.player_y) and not self._cell_is_float(
            self.player_x, self.player_y
        ):
            self.phase = (self.phase + 1) % self.phase_cycle
            return "death"

        if self._cell_is_croc(self.player_x, self.player_y):
            self.phase = (self.phase + 1) % self.phase_cycle
            return "death"

        home_idx = self.home_index_by_pos.get((self.player_x, self.player_y))
        if home_idx is not None and not (self.filled_mask & (1 << home_idx)):
            self.filled_mask |= 1 << home_idx
            self._respawn_with_time_reset()
            if self._filled_count() >= self.homes_required:
                self.pending_advance = True

        self.drain_tick += 1
        if self.drain_tick % max(1, self.time_drain_period) == 0:
            self.time_left -= 1
            if self.time_left <= 0:
                self.phase = (self.phase + 1) % self.phase_cycle
                return "death"

        self.phase = (self.phase + 1) % self.phase_cycle
        if self.pending_advance:
            return "solved"
        return "alive"


def solve_frogger_level(level_def: LevelDef, *, max_expansions: int = 400_000) -> list[int] | None:
    model = FroggerModel(level_def)
    start = model.clone()

    queue = deque([start])
    previous: dict[tuple, tuple | None] = {(*start.state_key(), start.time_left): None}
    previous_action: dict[tuple, int] = {}
    canonical_to_best: dict[tuple[int, int, int, int, int], int] = {start.state_key(): start.time_left}

    solved_key: tuple | None = None
    expansions = 0

    while queue:
        current = queue.popleft()
        full_key = (*current.state_key(), current.time_left)

        if current.is_solved() or current.pending_advance:
            solved_key = full_key
            break

        expansions += 1
        if expansions > max_expansions:
            break

        for action_id in ACTION_IDS:
            nxt = current.clone()
            outcome = nxt.apply_action(action_id)
            if outcome == "death":
                continue
            if outcome == "level_complete":
                solved_key = full_key
                break

            canon = nxt.state_key()
            best_time = canonical_to_best.get(canon, -1)
            if nxt.time_left <= best_time:
                continue
            canonical_to_best[canon] = nxt.time_left

            next_full = (*canon, nxt.time_left)
            if next_full in previous:
                continue
            previous[next_full] = full_key
            previous_action[next_full] = int(action_id)
            queue.append(nxt)
            if nxt.pending_advance or nxt.is_solved():
                solved_key = next_full
                queue.clear()
                break

        if solved_key is not None:
            break

    if solved_key is None:
        return None

    plan: list[int] = []
    cursor = solved_key
    while previous[cursor] is not None:
        plan.append(previous_action[cursor])
        cursor = previous[cursor]
    plan.reverse()
    return plan


def _build_level(level_def: LevelDef) -> Level:
    board = np.full((GRID_H, GRID_W), COLOR_GRASS, dtype=np.int8)
    board[0, :] = COLOR_TIME_EMPTY
    board[:, 0] = COLOR_WALL
    board[:, GRID_W - 1] = COLOR_WALL
    board[GRID_H - 1, :] = COLOR_WALL
    sprite = Sprite(board, name="board", x=0, y=0, layer=0, collidable=False)
    return Level(name=level_def.name, grid_size=(GRID_W, GRID_H), sprites=[sprite], data={"level_index": 0})


class Frogger(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(level_def) for level_def in LEVEL_DEFS]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_WALL)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(LEVEL_DEFS),
            available_actions=ACTION_IDS,
            seed=seed,
        )
        self._model: FroggerModel | None = None
        self._death_flash = False
        self._death_pos = (0, 0)
        self._hop_flash = False
        self._anim_tick = 0

    def on_set_level(self, _level: Level) -> None:
        level_idx = int(self._score)
        level_idx = max(0, min(level_idx, len(LEVEL_DEFS) - 1))
        self._model = FroggerModel(LEVEL_DEFS[level_idx])
        self._death_flash = False
        self._death_pos = self._model.spawn
        self._hop_flash = False
        self._anim_tick = 0
        self._render_board()

    def _set_board(self, grid: np.ndarray) -> None:
        board = self.current_level.get_sprites_by_name("board")
        if board:
            board[0].pixels = grid.astype(np.int8)

    def _front_color_for_cell(self, pattern: list[str], idx: int) -> int:
        near = []
        if idx > 0:
            near.append(pattern[idx - 1])
        if idx + 1 < len(pattern):
            near.append(pattern[idx + 1])
        body_color = COLOR_TRUCK if any(ch in {"[", "]"} for ch in near) else COLOR_CAR
        if self._anim_tick % 2 == 0:
            return COLOR_FLASH
        return body_color

    def _char_to_color(self, row: int, col: int, char: str) -> int:
        if (col, row) in self._model.home_index_by_pos:
            if self._model._is_home_filled(col, row):
                return COLOR_HOME_FILLED
            return COLOR_HOME_EMPTY

        if char == "#":
            return COLOR_WALL
        if char in {".", "@"}:
            return COLOR_GRASS
        if char == "_":
            return COLOR_ROAD
        if char == "~":
            return COLOR_WATER
        if char in {"<", ">"}:
            return COLOR_CAR
        if char in {"[", "]"}:
            return COLOR_TRUCK
        if char in {"{", "}"}:
            pattern = self._model.row_patterns.get(row, ["_"] * 30)
            idx = max(0, min(29, col - INTERIOR_X_MIN))
            return self._front_color_for_cell(pattern, idx)
        if char in {"=", "(", ")"}:
            return COLOR_LOG if char == "=" else COLOR_WAKE
        if char == "o":
            return COLOR_TURTLE_SAFE
        if char == "*":
            phase = TURTLE_CYCLE[self._model.phase % len(TURTLE_CYCLE)]
            if phase == "safe":
                return COLOR_TURTLE_SAFE
            if phase == "warning":
                return COLOR_FLASH
            return COLOR_WATER
        if char in CROC_CHARS:
            return COLOR_CROC if self._anim_tick % 2 == 0 else COLOR_FLASH
        if char == "^":
            return COLOR_HOME_EMPTY
        return COLOR_GRASS

    def _render_board(self) -> None:
        if self._model is None:
            return

        grid = np.full((GRID_H, GRID_W), COLOR_GRASS, dtype=np.int8)

        for y in range(GRID_H):
            for x in range(GRID_W):
                if x == 0 or x == GRID_W - 1 or y == GRID_H - 1:
                    grid[y, x] = COLOR_WALL
                    continue
                if y == 0:
                    grid[y, x] = COLOR_TIME_FILLED if x <= self._model.time_left else COLOR_TIME_EMPTY
                    continue

                base_char = self._model.static_rows[y][x]
                lane_char = self._model._lane_char(y, x)
                draw_char = lane_char if lane_char is not None else base_char
                grid[y, x] = self._char_to_color(y, x, draw_char)

        if self._death_flash:
            grid[self._death_pos[1], self._death_pos[0]] = COLOR_FLASH
        else:
            frog_color = COLOR_FLASH if self._hop_flash else COLOR_FROG
            grid[self._model.player_y, self._model.player_x] = frog_color

        self._set_board(grid)

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

        if self._model is None:
            self.complete_action()
            return

        if self._model.pending_advance:
            self.next_level()
            self.complete_action()
            return

        self._hop_flash = False

        action_id = _action_id(self.action.id)
        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            nx = self._model.player_x + dx
            ny = self._model.player_y + dy
            if not self._model._is_blocked_for_move(nx, ny):
                self._hop_flash = True

        outcome = self._model.apply_action(action_id)
        if outcome == "death":
            self._death_flash = True
            self._death_pos = (self._model.player_x, self._model.player_y)
            self.lose()

        self._anim_tick += 1
        self._render_board()
        self.complete_action()


def solve_all_levels() -> list[list[int] | None]:
    return [solve_frogger_level(level_def) for level_def in LEVEL_DEFS]
