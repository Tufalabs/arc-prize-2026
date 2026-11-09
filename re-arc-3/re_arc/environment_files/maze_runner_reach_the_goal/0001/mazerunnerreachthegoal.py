from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "maze_runner_reach_the_goal-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5

PLAY = "PLAY"
WIN_ANIM = "WIN_ANIM"

COLOR_FLOOR = 0
COLOR_DANGER = 1
COLOR_TIMEBAR_FILLED = 2
COLOR_TIMEBAR_EMPTY = 3
COLOR_START_PAD = 4
COLOR_PLAYER_A = 5
COLOR_PLAYER_B = 6
COLOR_WALL = 7
COLOR_PLAYER_BUMP = 8
COLOR_GOAL_DIM = 9
COLOR_GOAL_BRIGHT = 10
COLOR_WIN_BURST = 11

TICKS_PER_BAR_CELL = 3
WIN_ANIM_STEPS = 6

MOVE_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_SPACE: (0, 0),
}


LEVEL_BLUEPRINTS = [
    [
        "========================",
        "########################",
        "#@@.........#..........#",
        "#@@.........#..........#",
        "#...........#..........#",
        "#...........#..........#",
        "#...........#..........#",
        "#...........#..........#",
        "#......................#",
        "#..................**..#",
        "#...........#......**..#",
        "#...........#..........#",
        "########################",
    ],
    [
        "==========================",
        "##########################",
        "#@@......................#",
        "#@@......#...............#",
        "#........#......##.#.....#",
        "#........#.......#.#.....#",
        "#..####..#..##...#.#.....#",
        "#..#.....#..##..##.#.....#",
        "#...............#........#",
        "#...............#........#",
        "#........#####..#........#",
        "#........#......#....**..#",
        "#........#......#....**..#",
        "#........#......#........#",
        "##########################",
    ],
    [
        "============================",
        "############################",
        "#@@.....#.....#.....#......#",
        "#@@.....#.....#.....#......#",
        "#...#...#.....#.....#......#",
        "#...#...#...........#......#",
        "#.......#........#..#......#",
        "#.......###..##..#..#......#",
        "#...#.........#..#..#......#",
        "#...#.........#..#..#......#",
        "#...#...#.....#.....#......#",
        "#...#...#.....#............#",
        "#...#...#.....#..#.........#",
        "#...#...#.....#..#..#......#",
        "#...#...#.....#..#..#....**#",
        "#...#...#.....#..#..#....**#",
        "############################",
    ],
    [
        "==============================",
        "##############################",
        "#@@....#.....#.....#.....#...#",
        "#@@....#.....#.....#.....#...#",
        "#......#...........#.....#.**#",
        "#......#...........#.....#.**#",
        "#......#.....#.....#.....#...#",
        "#.##...#.....#.....#.....#...#",
        "#.##...#..##.#.....#.....#...#",
        "#......#.....#...........#...#",
        "#......#.....#........##.#...#",
        "#......#.....#.....#.....#...#",
        "#......#.....#.....#.........#",
        "#............#.....#.........#",
        "#............#.....#.....#...#",
        "#......#.....#.....#.....#...#",
        "##############################",
    ],
    [
        "================================",
        "################################",
        "#@@...#.....#.....#.....#..#...#",
        "#@@...#.....#.....#.....#..#...#",
        "#.....#...........#.....#..#.**#",
        "#.....#...........#.....#..#.**#",
        "#.....#.....#.....#.....#......#",
        "#.....#.....#.....#.....#......#",
        "#.##..#.....#..........##..#...#",
        "#.##..#.....#...........#..#...#",
        "#.....#.....#..##.......#..#...#",
        "#.....#.....#.....#.....#..#...#",
        "#.....#.....#.....#........#...#",
        "#.....#.....#.....#........#...#",
        "#.....#.....#.....#.....#..#...#",
        "#...........#.....#.....#..#...#",
        "#...........#.....#.....#..#...#",
        "#.....#.....#.....#.....#..#...#",
        "################################",
    ],
    [
        "==================================",
        "##################################",
        "#@@...#....#....#....#....#..#...#",
        "#@@...#....#....#....#....#..#...#",
        "#.....#.........#....#....#..#...#",
        "#.....#.........#....#....#..#...#",
        "#.....#....#....#....#....#......#",
        "#.....#....#....#.........#......#",
        "#.....#....#....#.......###..#...#",
        "#.....#....#....#....#....#..#...#",
        "#.....#....#....#....#....#..#...#",
        "#.....#....#....#....#.......#...#",
        "#.....#....#....#....#.......#...#",
        "#.....#.#..#....#....#....#..#...#",
        "#.....#....#.........#....#..#...#",
        "#.....#....#.........#....#..#...#",
        "#.....#....#....#....#....#..#...#",
        "#..........#....#....#....#..#...#",
        "#..........#....#....#....#..#.**#",
        "#.....#....#....#....#....#..#.**#",
        "##################################",
    ],
]


@dataclass(frozen=True)
class LevelModel:
    width: int
    height: int
    walls: tuple[tuple[int, int], ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    time_limit: int


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw - 1 < bx or bx + bw - 1 < ax or ay + ah - 1 < by or by + bh - 1 < ay)


def _action_to_delta(action_id: int) -> tuple[int, int]:
    return MOVE_DELTAS.get(int(action_id), (0, 0))


def _can_place_player(model: dict, px: int, py: int) -> bool:
    width = int(model["width"])
    height = int(model["height"])
    if px < 0 or py < 1 or px + 1 >= width or py + 1 >= height:
        return False
    walls = model["walls_set"]
    for yy in (py, py + 1):
        for xx in (px, px + 1):
            if (xx, yy) in walls:
                return False
    return True


def initial_search_state_from_model(model: dict) -> tuple[int, int, int, int, int, int]:
    sx, sy = model["start"]
    return (int(sx), int(sy), int(model["time_limit"]), 0, 0, 0)


def apply_action_transition(
    model: dict, state: tuple[int, int, int, int, int, int], action_id: int
) -> tuple[tuple[int, int, int, int, int, int] | None, bool]:
    px, py, time_remaining, tick, walk_frame, _bump_timer = state
    if int(time_remaining) <= 0:
        return None, False

    dx, dy = _action_to_delta(int(action_id))
    next_px, next_py = int(px), int(py)
    next_walk = int(walk_frame)
    next_bump = 0

    if (dx, dy) != (0, 0):
        cand_x = next_px + dx
        cand_y = next_py + dy
        if _can_place_player(model, cand_x, cand_y):
            next_px, next_py = cand_x, cand_y
            next_walk = 1 - next_walk
        else:
            next_bump = 1

    next_time = int(time_remaining) - 1
    next_tick = int(tick) + 1

    goal_x, goal_y = model["goal"]
    won = _rects_overlap((next_px, next_py, 2, 2), (int(goal_x), int(goal_y), 2, 2))
    if won:
        return (next_px, next_py, next_time, next_tick, next_walk, next_bump), True

    if next_time <= 0:
        return None, False

    return (next_px, next_py, next_time, next_tick, next_walk, next_bump), False


def _extract_top_left(points: list[tuple[int, int]], token: str) -> tuple[int, int]:
    if len(points) != 4:
        raise ValueError(f"Expected exactly 4 cells for `{token}`, found {len(points)}.")
    xs = sorted({x for x, _ in points})
    ys = sorted({y for _, y in points})
    if len(xs) != 2 or len(ys) != 2:
        raise ValueError(f"`{token}` must be a 2x2 block.")
    expected = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])}
    if set(points) != expected:
        raise ValueError(f"`{token}` footprint must be contiguous 2x2.")
    return xs[0], ys[0]


def _parse_blueprint(rows: list[str]) -> LevelModel:
    if not rows:
        raise ValueError("Blueprint cannot be empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Blueprint rows must have consistent width.")

    at_cells: list[tuple[int, int]] = []
    goal_cells: list[tuple[int, int]] = []
    walls: list[tuple[int, int]] = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.append((x, y))
            elif ch == "@":
                at_cells.append((x, y))
            elif ch == "*":
                goal_cells.append((x, y))
            elif ch in {".", "="}:
                continue
            else:
                raise ValueError(f"Unsupported tile `{ch}` in blueprint.")

    start = _extract_top_left(at_cells, "@")
    goal = _extract_top_left(goal_cells, "*")
    time_limit = int(width * TICKS_PER_BAR_CELL)

    model = {
        "width": int(width),
        "height": len(rows),
        "walls": tuple((int(x), int(y)) for x, y in walls),
        "start": (int(start[0]), int(start[1])),
        "goal": (int(goal[0]), int(goal[1])),
        "time_limit": int(time_limit),
    }
    return LevelModel(**model)


def _serialize_model(model: LevelModel) -> dict:
    return {
        "width": int(model.width),
        "height": int(model.height),
        "walls": [[int(x), int(y)] for x, y in model.walls],
        "start": [int(model.start[0]), int(model.start[1])],
        "goal": [int(model.goal[0]), int(model.goal[1])],
        "time_limit": int(model.time_limit),
    }


def _deserialize_model(level: Level) -> dict:
    raw = dict(level.get_data("model") or {})
    walls = tuple((int(cell[0]), int(cell[1])) for cell in raw.get("walls", []))
    start_raw = raw.get("start") or [0, 0]
    goal_raw = raw.get("goal") or [0, 0]

    return {
        "width": int(raw.get("width", 0)),
        "height": int(raw.get("height", 0)),
        "walls": walls,
        "walls_set": set(walls),
        "start": (int(start_raw[0]), int(start_raw[1])),
        "goal": (int(goal_raw[0]), int(goal_raw[1])),
        "time_limit": int(raw.get("time_limit", 0)),
        "ticks_per_bar_cell": TICKS_PER_BAR_CELL,
    }


def _build_level(idx: int, blueprint_rows: list[str]) -> Level:
    model = _parse_blueprint(blueprint_rows)
    width, height = int(model.width), int(model.height)
    canvas = np.full((height, width), COLOR_FLOOR, dtype=np.int8)

    return Level(
        name=f"Maze Runner Reach Goal L{idx + 1}",
        grid_size=(width, height),
        sprites=[Sprite(pixels=canvas, name="canvas", x=0, y=0, layer=0, tags=["board"], collidable=False)],
        data={"model": _serialize_model(model), "level_index": int(idx)},
    )


class MazeRunnerReachTheGoal(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(idx, blueprint) for idx, blueprint in enumerate(LEVEL_BLUEPRINTS)]
        max_w = max(int(level.grid_size[0]) for level in levels)
        max_h = max(int(level.grid_size[1]) for level in levels)
        camera = Camera(width=max_w, height=max_h, background=COLOR_FLOOR)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._model: dict | None = None
        self._canvas: Sprite | None = None

        self._mode = PLAY
        self._tick = 0
        self._time_remaining_ticks = 0

        self._px = 0
        self._py = 0
        self._start_x = 0
        self._start_y = 0
        self._goal_x = 0
        self._goal_y = 0

        self._player_walk_frame = 0
        self._player_bump_timer = 0
        self._win_anim_counter = 0
        self._route_score = 0

    def _action_id(self) -> int:
        raw = getattr(self.action, "id", 0)
        return int(getattr(raw, "value", raw))

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        sprites = level.get_sprites_by_name("canvas")
        self._canvas = sprites[0] if sprites else None
        self._reset_runtime_for_current_level()

    def _reset_runtime_for_current_level(self) -> None:
        assert self._model is not None
        self._mode = PLAY
        self._tick = 0
        self._time_remaining_ticks = int(self._model["time_limit"])

        self._start_x, self._start_y = self._model["start"]
        self._goal_x, self._goal_y = self._model["goal"]
        self._px = int(self._start_x)
        self._py = int(self._start_y)

        self._player_walk_frame = 0
        self._player_bump_timer = 0
        self._win_anim_counter = 0
        self._route_score = 0
        self._redraw()

    def _player_overlaps_goal(self) -> bool:
        return _rects_overlap((int(self._px), int(self._py), 2, 2), (int(self._goal_x), int(self._goal_y), 2, 2))

    def _draw_timebar(self, frame: np.ndarray) -> None:
        assert self._model is not None
        width = int(self._model["width"])
        capacity = int(width * TICKS_PER_BAR_CELL)
        remaining = max(0, int(self._time_remaining_ticks))

        frame[0, :] = COLOR_TIMEBAR_EMPTY

        for x in range(width):
            threshold = int((width - x) * TICKS_PER_BAR_CELL)
            if remaining >= threshold:
                frame[0, x] = COLOR_TIMEBAR_FILLED

        full_cells = remaining // TICKS_PER_BAR_CELL
        partial = remaining % TICKS_PER_BAR_CELL
        if partial != 0:
            partial_x = width - full_cells - 1
            if 0 <= partial_x < width:
                frame[0, partial_x] = COLOR_TIMEBAR_FILLED if (self._tick % 2 == 0) else COLOR_TIMEBAR_EMPTY

        if remaining <= capacity // 4:
            danger_color = COLOR_TIMEBAR_FILLED if (self._tick % 2 == 0) else COLOR_DANGER
            for x in range(width):
                if frame[0, x] == COLOR_TIMEBAR_FILLED:
                    frame[0, x] = danger_color

    def _draw_goal(self, frame: np.ndarray) -> None:
        gx, gy = int(self._goal_x), int(self._goal_y)
        if self._mode == WIN_ANIM:
            frame[gy : gy + 2, gx : gx + 2] = COLOR_WIN_BURST
            return

        if self._tick % 2 == 0:
            frame[gy : gy + 2, gx : gx + 2] = COLOR_GOAL_BRIGHT
            return

        frame[gy, gx] = COLOR_GOAL_DIM
        frame[gy, gx + 1] = COLOR_GOAL_BRIGHT
        frame[gy + 1, gx] = COLOR_GOAL_BRIGHT
        frame[gy + 1, gx + 1] = COLOR_GOAL_DIM

    def _draw_player(self, frame: np.ndarray) -> None:
        px, py = int(self._px), int(self._py)
        if self._mode == WIN_ANIM:
            frame[py : py + 2, px : px + 2] = COLOR_WIN_BURST
            return

        if self._player_bump_timer > 0:
            frame[py : py + 2, px : px + 2] = COLOR_PLAYER_BUMP
            return

        if self._player_walk_frame == 0:
            frame[py : py + 2, px : px + 2] = COLOR_PLAYER_A
            return

        frame[py, px] = COLOR_PLAYER_A
        frame[py, px + 1] = COLOR_PLAYER_B
        frame[py + 1, px] = COLOR_PLAYER_B
        frame[py + 1, px + 1] = COLOR_PLAYER_A

    def _redraw(self) -> None:
        if self._model is None or self._canvas is None:
            return

        width = int(self._model["width"])
        height = int(self._model["height"])
        frame = np.full((height, width), COLOR_FLOOR, dtype=np.int8)

        self._draw_timebar(frame)

        sx, sy = int(self._start_x), int(self._start_y)
        frame[sy : sy + 2, sx : sx + 2] = COLOR_START_PAD

        for wx, wy in self._model["walls"]:
            frame[int(wy), int(wx)] = COLOR_WALL

        self._draw_goal(frame)
        self._draw_player(frame)
        self._canvas.pixels = frame

    def _handle_win_anim(self) -> None:
        self._tick += 1
        self._win_anim_counter += 1
        if self._win_anim_counter >= WIN_ANIM_STEPS:
            self.next_level()
            return
        self._redraw()

    def _handle_play(self, action_id: int) -> None:
        assert self._model is not None
        dx, dy = _action_to_delta(action_id)
        if (dx, dy) != (0, 0):
            cand_x = int(self._px) + dx
            cand_y = int(self._py) + dy
            if _can_place_player(self._model, cand_x, cand_y):
                self._px = cand_x
                self._py = cand_y
                self._player_walk_frame = 1 - int(self._player_walk_frame)
            else:
                self._player_bump_timer = 1

        self._time_remaining_ticks -= 1
        self._tick += 1

        if self._player_overlaps_goal():
            self._mode = WIN_ANIM
            self._win_anim_counter = 0
        elif self._time_remaining_ticks <= 0:
            self.lose()
            return

        self._redraw()

        if self._player_bump_timer > 0:
            self._player_bump_timer -= 1

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

        action_id = self._action_id()

        if self._mode == WIN_ANIM:
            self._handle_win_anim()
        else:
            self._handle_play(action_id)

        self.complete_action()
