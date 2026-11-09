from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "dash_puzzle-0001"
GRID_WIDTH = 24
GRID_HEIGHT = 16
PLAYFIELD_Y0 = 2
TOTAL_LEVELS = 6

COLOR_VOID = 0
COLOR_HUD_BG = 1
COLOR_TIME_FILL = 2
COLOR_FLOOR = 3
COLOR_WALL = 4
COLOR_TRAIL = 5
COLOR_PLAYER = 6
COLOR_FACING = 7
COLOR_CRUMBLE_INTACT = 8
COLOR_CRUMBLE_CRACKED = 9
COLOR_GOAL_A = 10
COLOR_GOAL_B = 11
COLOR_SPIKE_ACTIVE = 12
COLOR_PLATFORM_A = 14
COLOR_PLATFORM_B = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

FACING_FROM_ACTION = {
    int(GameAction.ACTION1.value): 0,  # up
    int(GameAction.ACTION2.value): 1,  # down
    int(GameAction.ACTION3.value): 2,  # left
    int(GameAction.ACTION4.value): 3,  # right
}

FACING_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

LAYOUTS: list[tuple[str, int, tuple[str, ...]]] = [
    (
        "Level 1",
        120,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#..........~~..........#",
            "#..........~~..........#",
            "#..........~~..........#",
            "#..@]......~~..........#",
            "#..........~~..........#",
            "#..........~~..........#",
            "#..........~~....&&....#",
            "#..........~~....&&....#",
            "#..........~~..........#",
            "#..........~~..........#",
            "#..........~~..........#",
            "#..........~~..........#",
            "########################",
        ),
    ),
    (
        "Level 2",
        140,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "#..@].....~~~~.........#",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "#.........~~~~...&&....#",
            "#.........~~~~...&&....#",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "#.........~~~~.........#",
            "########################",
        ),
    ),
    (
        "Level 3",
        160,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#......................#",
            "#......................#",
            "#..........@]..........#",
            "#......................#",
            "#......................#",
            "############^###########",
            "#......................#",
            "#......................#",
            "#......................#",
            "#..............&&......#",
            "#..............&&......#",
            "#......................#",
            "########################",
        ),
    ),
    (
        "Level 4",
        170,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#.........~~~..........#",
            "#.........~~~..........#",
            "#.........~~~..........#",
            "#.........~~~.#........#",
            "#...@]....~~~.#........#",
            "#.........~~~.#........#",
            "#.........~~~.#........#",
            "#.........~~~.#........#",
            "#.........~~~..........#",
            "#.........~~~..........#",
            "#.........~~~......&&..#",
            "#.........~~~......&&..#",
            "########################",
        ),
    ),
    (
        "Level 5",
        190,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#..@]....~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#........%%%%%%....^...#",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^...#",
            "#........~~~~~~....^&&.#",
            "#........~~~~~~....^&&.#",
            "########################",
        ),
    ),
    (
        "Level 6",
        220,
        (
            "========================",
            ":::..                   ",
            "########################",
            "#...................#..#",
            "#...................#..#",
            "#......~~~~~~~~~~...#..#",
            "#......~~~~~~~~~~...#..#",
            "#..@]..~~~()~~~~~...#..#",
            "#......~~~()~~~~~...#..#",
            "#......~~~~~~~~~~...#..#",
            "#......~~~~~~~~~~...#..#",
            "#.................%%#..#",
            "#.................%%^&&#",
            "#.................%%^&&#",
            "#...................#..#",
            "########################",
        ),
    ),
]


@dataclass(frozen=True)
class LevelModel:
    index: int
    name: str
    time_limit: int
    start: tuple[int, int]
    start_facing: int
    walls: frozenset[tuple[int, int]]
    voids: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]
    spikes: frozenset[tuple[int, int]]
    crumbles: frozenset[tuple[int, int]]
    platform_start_x: int
    platform_y: int
    platform_dir: int


@dataclass(frozen=True)
class DashState:
    px: int
    py: int
    facing: int
    time_left: int
    tick: int
    collapsed: frozenset[tuple[int, int]]
    cracked_fresh: frozenset[tuple[int, int]]
    cracked_old: frozenset[tuple[int, int]]
    platform_x: int
    platform_dir: int


@dataclass(frozen=True)
class StepResult:
    state: DashState
    won: bool
    failed: bool
    trail: tuple[tuple[int, int], ...]


def _spike_active(tick: int) -> bool:
    return int(tick) % 4 in (0, 1)


def _dash_length(model: LevelModel, time_left: int) -> int:
    spent = int(model.time_limit) - int(time_left)
    return 3 + (spent % 3)


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= int(x) < GRID_WIDTH and PLAYFIELD_Y0 <= int(y) < GRID_HEIGHT


def _platform_cells(model: LevelModel, platform_x: int) -> frozenset[tuple[int, int]]:
    if model.platform_start_x < 0 or model.platform_y < 0:
        return frozenset()
    x0 = int(platform_x)
    y0 = int(model.platform_y)
    return frozenset({(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)})


def _is_wall(model: LevelModel, x: int, y: int) -> bool:
    if not _in_bounds(x, y):
        return True
    return (int(x), int(y)) in model.walls


def _is_active_spike(model: LevelModel, state: DashState, x: int, y: int) -> bool:
    return (int(x), int(y)) in model.spikes and _spike_active(state.tick)


def _is_void(model: LevelModel, state: DashState, x: int, y: int) -> bool:
    pos = (int(x), int(y))
    if pos in _platform_cells(model, state.platform_x):
        return False
    return pos in model.voids or pos in state.collapsed


def _is_intact_crumble(model: LevelModel, state: DashState, x: int, y: int) -> bool:
    pos = (int(x), int(y))
    if pos not in model.crumbles:
        return False
    return pos not in state.collapsed and pos not in state.cracked_fresh and pos not in state.cracked_old


def _can_platform_shift(model: LevelModel, new_x: int) -> bool:
    cells = _platform_cells(model, new_x)
    if not cells:
        return True
    for x, y in cells:
        if not _in_bounds(x, y):
            return False
        if (x, y) in model.walls:
            return False
    return True


def _initial_state(model: LevelModel) -> DashState:
    return DashState(
        px=int(model.start[0]),
        py=int(model.start[1]),
        facing=int(model.start_facing),
        time_left=int(model.time_limit),
        tick=0,
        collapsed=frozenset(),
        cracked_fresh=frozenset(),
        cracked_old=frozenset(),
        platform_x=int(model.platform_start_x),
        platform_dir=int(model.platform_dir),
    )


def _simulate_step(model: LevelModel, state: DashState, action_id: int) -> StepResult:
    px = int(state.px)
    py = int(state.py)
    facing = int(state.facing)
    time_left = int(state.time_left)
    tick = int(state.tick)
    collapsed = set(state.collapsed)
    cracked_fresh = set(state.cracked_fresh)
    cracked_old = set(state.cracked_old)
    platform_x = int(state.platform_x)
    platform_dir = int(state.platform_dir)

    failed = False
    trail: list[tuple[int, int]] = []

    if action_id in MOVE_DELTAS:
        facing = int(FACING_FROM_ACTION[action_id])
        dx, dy = MOVE_DELTAS[action_id]
        nx, ny = px + dx, py + dy
        if not _is_wall(model, nx, ny):
            if _is_active_spike(model, state, nx, ny) or _is_void(model, state, nx, ny):
                failed = True
                px, py = int(nx), int(ny)
            else:
                px, py = int(nx), int(ny)
                if _is_intact_crumble(model, state, px, py):
                    cracked_fresh.add((px, py))

    elif int(action_id) == int(GameAction.ACTION5.value):
        dx, dy = FACING_DELTAS.get(facing, (1, 0))
        dash_len = _dash_length(model, time_left)
        cx, cy = px, py
        for _ in range(dash_len):
            nx, ny = cx + dx, cy + dy
            if _is_wall(model, nx, ny):
                break
            cx, cy = int(nx), int(ny)
            trail.append((cx, cy))
            if _is_active_spike(model, state, cx, cy):
                failed = True
                break

        if trail:
            px, py = int(cx), int(cy)
            if _is_intact_crumble(model, state, px, py):
                cracked_fresh.add((px, py))
            if _is_void(model, state, px, py):
                failed = True

    # Step-order world update:
    # 1) spikes toggle
    tick = (tick + 1) % 4

    # 2) crumble advances
    for cell in list(cracked_old):
        if (px, py) == cell:
            failed = True
        cracked_old.remove(cell)
        collapsed.add(cell)

    cracked_old.update(cracked_fresh)
    cracked_fresh.clear()

    # 3) platform advances and carries player
    old_platform_cells = _platform_cells(model, platform_x)
    if old_platform_cells and platform_dir != 0:
        target_x = platform_x + platform_dir
        if not _can_platform_shift(model, target_x):
            platform_dir *= -1
            target_x = platform_x + platform_dir
        if _can_platform_shift(model, target_x):
            if (px, py) in old_platform_cells:
                px += target_x - platform_x
            platform_x = int(target_x)

    # 4) decrease time
    time_left = max(0, time_left - 1)

    next_state = DashState(
        px=int(px),
        py=int(py),
        facing=int(facing),
        time_left=int(time_left),
        tick=int(tick),
        collapsed=frozenset(collapsed),
        cracked_fresh=frozenset(cracked_fresh),
        cracked_old=frozenset(cracked_old),
        platform_x=int(platform_x),
        platform_dir=int(platform_dir),
    )

    # 5/6) terminal checks after updates
    if _is_active_spike(model, next_state, next_state.px, next_state.py):
        failed = True
    if _is_void(model, next_state, next_state.px, next_state.py):
        failed = True
    if next_state.time_left <= 0:
        failed = True

    won = (next_state.px, next_state.py) in model.goals and not failed
    return StepResult(state=next_state, won=bool(won), failed=bool(failed), trail=tuple(trail))


def _dominance_key(state: DashState):
    return (
        state.px,
        state.py,
        state.facing,
        state.tick,
        state.collapsed,
        state.cracked_fresh,
        state.cracked_old,
        state.platform_x,
        state.platform_dir,
    )


def _find_plan(model: LevelModel) -> list[int]:
    start = _initial_state(model)
    queue = deque([start])
    previous: dict[DashState, DashState | None] = {start: None}
    previous_action: dict[DashState, int] = {}
    best_time: dict[tuple, int] = {_dominance_key(start): int(start.time_left)}

    while queue:
        state = queue.popleft()
        if (state.px, state.py) in model.goals:
            break

        for action_id in (1, 2, 3, 4, 5):
            result = _simulate_step(model, state, action_id)
            if result.failed:
                continue

            next_state = result.state
            key = _dominance_key(next_state)
            prior_best = best_time.get(key)
            if prior_best is not None and prior_best >= next_state.time_left:
                continue
            best_time[key] = int(next_state.time_left)

            if next_state in previous:
                continue

            previous[next_state] = state
            previous_action[next_state] = int(action_id)

            if result.won:
                actions: list[int] = [int(action_id)]
                cursor = state
                while previous[cursor] is not None:
                    actions.append(int(previous_action[cursor]))
                    cursor = previous[cursor]  # type: ignore[assignment]
                actions.reverse()
                return actions

            queue.append(next_state)

    raise RuntimeError(f"No valid plan found for {model.name}.")


def _parse_layout(index: int, name: str, time_limit: int, rows: tuple[str, ...]) -> LevelModel:
    if len(rows) != GRID_HEIGHT:
        raise ValueError(f"{name}: expected {GRID_HEIGHT} rows, got {len(rows)}")

    walls: set[tuple[int, int]] = set()
    voids: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    crumbles: set[tuple[int, int]] = set()
    platform_cells: set[tuple[int, int]] = set()

    start: tuple[int, int] | None = None
    start_facing = 3

    for y, row in enumerate(rows):
        if len(row) != GRID_WIDTH:
            raise ValueError(f"{name}: row {y} width {len(row)} != {GRID_WIDTH}")
        if y < PLAYFIELD_Y0:
            continue

        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            elif cell == "~":
                voids.add((x, y))
            elif cell == "&":
                goals.add((x, y))
            elif cell == "^":
                spikes.add((x, y))
            elif cell in {"%", "$"}:
                crumbles.add((x, y))
            elif cell in {"(", ")"}:
                platform_cells.add((x, y))
                voids.add((x, y))
            elif cell == "@":
                start = (x, y)
            elif cell == "[":
                start_facing = 2
            elif cell == "]":
                start_facing = 3
            elif cell == "{":
                start_facing = 0
            elif cell == "}":
                start_facing = 1

    if start is None:
        raise ValueError(f"{name}: missing player start '@'")
    if not goals:
        raise ValueError(f"{name}: missing goal '&'")

    platform_start_x = -1
    platform_y = -1
    platform_dir = 0
    if platform_cells:
        min_x = min(x for x, _ in platform_cells)
        min_y = min(y for _, y in platform_cells)
        platform_start_x = int(min_x)
        platform_y = int(min_y)
        platform_dir = 1

    return LevelModel(
        index=int(index),
        name=str(name),
        time_limit=int(time_limit),
        start=start,
        start_facing=int(start_facing),
        walls=frozenset(walls),
        voids=frozenset(voids),
        goals=frozenset(goals),
        spikes=frozenset(spikes),
        crumbles=frozenset(crumbles),
        platform_start_x=int(platform_start_x),
        platform_y=int(platform_y),
        platform_dir=int(platform_dir),
    )


LEVEL_MODELS: tuple[LevelModel, ...] = tuple(
    _parse_layout(index=i, name=name, time_limit=time_limit, rows=rows)
    for i, (name, time_limit, rows) in enumerate(LAYOUTS)
)


@cache
def compute_level_plan(level_index: int) -> tuple[int, ...]:
    model = LEVEL_MODELS[int(level_index)]
    return tuple(_find_plan(model))


def _blank_board() -> np.ndarray:
    return np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_HUD_BG, dtype=np.int8)


def _build_level(model: LevelModel) -> Level:
    board = Sprite(
        pixels=_blank_board(), name="board", x=0, y=0, layer=0, collidable=False, tags=["board", "sys_static"]
    )
    player = Sprite(
        pixels=np.array([[COLOR_PLAYER]], dtype=np.int8),
        name="player",
        x=model.start[0],
        y=model.start[1],
        layer=3,
        collidable=False,
        tags=["player"],
    )
    facing = Sprite(
        pixels=np.array([[COLOR_FACING]], dtype=np.int8),
        name="facing",
        x=model.start[0] + 1,
        y=model.start[1],
        layer=4,
        collidable=False,
        tags=["facing"],
    )

    return Level(
        name=model.name,
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[board, player, facing],
        data={"level_index": int(model.index), "time_limit": int(model.time_limit)},
    )


class DashPuzzle(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(model) for model in LEVEL_MODELS]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_HUD_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._model: LevelModel | None = None
        self._sim_state: DashState | None = None
        self._board: Sprite | None = None
        self._player: Sprite | None = None
        self._facing_sprite: Sprite | None = None
        self._trail_cells: tuple[tuple[int, int], ...] = ()
        self._trail_visible = False

    def on_set_level(self, level: Level) -> None:
        level_idx = int(level.get_data("level_index") or 0)
        self._model = LEVEL_MODELS[level_idx]
        self._board = level.get_sprites_by_name("board")[0]
        self._player = level.get_sprites_by_name("player")[0]
        self._facing_sprite = level.get_sprites_by_name("facing")[0]
        self._reset_level_state()
        self._render()

    def _reset_level_state(self) -> None:
        if self._model is None:
            return
        self._sim_state = _initial_state(self._model)
        self._trail_cells = ()
        self._trail_visible = False

    def _render(self) -> None:
        if self._model is None or self._sim_state is None:
            return
        if self._board is None or self._player is None or self._facing_sprite is None:
            return

        model = self._model
        state = self._sim_state
        board = _blank_board()

        # Row 0: timebar.
        time_ratio = max(0.0, min(1.0, float(state.time_left) / float(max(1, model.time_limit))))
        time_fill = round(time_ratio * GRID_WIDTH)
        board[0, :] = np.int8(COLOR_VOID)
        if time_fill > 0:
            board[0, :time_fill] = np.int8(COLOR_TIME_FILL)

        # Row 1: dash meter (3-5 lit pips).
        board[1, :] = np.int8(COLOR_HUD_BG)
        dash_len = _dash_length(model, state.time_left)
        for x in range(5):
            board[1, x] = np.int8(COLOR_TIME_FILL if x < dash_len else COLOR_FLOOR)

        platform_cells = _platform_cells(model, state.platform_x)
        spike_on = _spike_active(state.tick)
        goal_color = COLOR_GOAL_A if state.tick % 2 == 0 else COLOR_GOAL_B
        platform_color = COLOR_PLATFORM_A if state.tick % 2 == 0 else COLOR_PLATFORM_B

        for y in range(PLAYFIELD_Y0, GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                pos = (x, y)
                color = COLOR_FLOOR
                if pos in model.walls:
                    color = COLOR_WALL
                elif pos in model.goals:
                    color = goal_color
                elif pos in platform_cells:
                    color = platform_color
                elif pos in model.spikes:
                    color = COLOR_SPIKE_ACTIVE if spike_on else COLOR_FLOOR
                elif pos in state.collapsed or pos in model.voids:
                    color = COLOR_VOID
                elif pos in state.cracked_fresh or pos in state.cracked_old:
                    color = COLOR_CRUMBLE_CRACKED
                elif pos in model.crumbles:
                    color = COLOR_CRUMBLE_INTACT
                board[y, x] = np.int8(color)

        if self._trail_visible:
            for tx, ty in self._trail_cells:
                if not _in_bounds(tx, ty):
                    continue
                if (tx, ty) in model.walls:
                    continue
                if (tx, ty) in model.goals:
                    continue
                if (tx, ty) in model.spikes and spike_on:
                    continue
                if (tx, ty) == (state.px, state.py):
                    continue
                board[ty, tx] = np.int8(COLOR_TRAIL)

        self._board.pixels = board

        self._player.set_position(int(state.px), int(state.py))
        fx, fy = FACING_DELTAS.get(int(state.facing), (1, 0))
        mx, my = int(state.px + fx), int(state.py + fy)
        if 0 <= mx < GRID_WIDTH and 0 <= my < GRID_HEIGHT:
            self._facing_sprite.set_visible(True)
            self._facing_sprite.set_position(mx, my)
        else:
            self._facing_sprite.set_visible(False)

    def step(self) -> None:
        if self._model is None or self._sim_state is None:
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id not in {1, 2, 3, 4, 5}:
            self._render()
            self.complete_action()
            return

        # Dash trail persists exactly one full step.
        if self._trail_visible:
            self._trail_visible = False
            self._trail_cells = ()

        result = _simulate_step(self._model, self._sim_state, action_id)

        self._sim_state = result.state
        if action_id == int(GameAction.ACTION5.value) and result.trail:
            self._trail_visible = True
            self._trail_cells = result.trail

        if result.won:
            self.next_level()
            self.complete_action()
            return

        if result.failed:
            self._render()
            self.lose()
            self.complete_action()
            return

        self._render()
        self.complete_action()
