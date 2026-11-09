from __future__ import annotations

import heapq
from collections import deque
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 31
GRID_HEIGHT = 19
TIME_TICKS = 124
EXPLOSION_RANGE = 3
OUTCOME_FLASH_TICKS = 4

COLOR_VOID = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_BLOCK = 3
COLOR_EXIT_LOCKED = 4
COLOR_EXIT_OPEN = 5
COLOR_PLAYER_A = 6
COLOR_PLAYER_B = 7
COLOR_ENEMY_A = 8
COLOR_ENEMY_B = 9
COLOR_BOMB_3 = 10
COLOR_BOMB_2 = 11
COLOR_BOMB_1 = 12
COLOR_EXPLOSION_A = 13
COLOR_EXPLOSION_B = 14
COLOR_TIME_FILLED = 15

DIR_LEFT = 0
DIR_RIGHT = 1

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

# Solvable 31x19 layouts that exercise bombs, blast blocking, time pressure,
# restart, and locked-exit progression across six levels.
LEVEL_LAYOUTS: list[tuple[str, list[str]]] = [
    (
        "Level 1",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#.............................#",
            "#...........+...+.............#",
            "#.............................#",
            "#.............................#",
            "#..............+..............#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
    (
        "Level 2",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#.............................#",
            "#.............................#",
            "#..........+...+..............#",
            "#...................#.........#",
            "#..................#!#........#",
            "#............+................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
    (
        "Level 3",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#.............................#",
            "#.............................#",
            "#.........+.+.................#",
            "#..........#..................#",
            "#.........#!#.................#",
            "#.............................#",
            "#.............####............#",
            "#.............................#",
            "#.................+.+.........#",
            "#..................#..........#",
            "#.................#!#.........#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
    (
        "Level 4",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#.............................#",
            "#.......+.+...................#",
            "#.............................#",
            "#.............................#",
            "#...........+.+...............#",
            "#.............................#",
            "#.................+.+.........#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
    (
        "Level 5",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#......+.+....................#",
            "#.............................#",
            "#.............................#",
            "#..........+.+................#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#..............+.+............#",
            "#.............................#",
            "#.....................+.+.....#",
            "#.............................#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
    (
        "Level 6",
        [
            "===============================",
            "###############################",
            "#..........................XX.#",
            "#..........................XX.#",
            "#.....+.+.....................#",
            "#.............................#",
            "#.............................#",
            "#.........+.+.................#",
            "#.............................#",
            "#.............+.+.+.+.........#",
            "#.............................#",
            "#.............................#",
            "#.....................+.+.....#",
            "#........+...+................#",
            "#.............................#",
            "#@............................#",
            "#.............................#",
            "#.............................#",
            "###############################",
        ],
    ),
]


class StaticLevelModel(NamedTuple):
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    block_positions: tuple[tuple[int, int], ...]
    enemy_positions: tuple[tuple[int, int], ...]
    player_start: tuple[int, int]
    exit_tiles: frozenset[tuple[int, int]]


class BombState(NamedTuple):
    x: int
    y: int
    fuse: int


class ExplosionState(NamedTuple):
    phase: int
    cells: tuple[tuple[int, int], ...]


class DynamicState(NamedTuple):
    player_x: int
    player_y: int
    player_alive: bool
    block_mask: int
    enemies: tuple[tuple[int, int, int, bool], ...]
    bomb: BombState | None
    explosion: ExplosionState | None
    time_ticks: int


class TransitionResult(NamedTuple):
    state: DynamicState
    won: bool
    failed: bool
    placed_bomb: bool
    explosion_created: bool


def _validate_layout(rows: list[str]) -> None:
    if len(rows) != GRID_HEIGHT:
        raise ValueError(f"Expected {GRID_HEIGHT} rows, got {len(rows)}")
    for row in rows:
        if len(row) != GRID_WIDTH:
            raise ValueError(f"Expected row width {GRID_WIDTH}, got {len(row)}")


def _build_static_model(rows: list[str]) -> StaticLevelModel:
    _validate_layout(rows)

    walls: set[tuple[int, int]] = set()
    blocks: list[tuple[int, int]] = []
    enemies: list[tuple[int, int]] = []
    exit_tiles: set[tuple[int, int]] = set()
    player_start: tuple[int, int] | None = None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch == "+":
                blocks.append((x, y))
            elif ch == "!":
                enemies.append((x, y))
            elif ch == "@":
                player_start = (x, y)
            elif ch == "X":
                exit_tiles.add((x, y))

    if player_start is None:
        raise ValueError("Layout missing player start '@'")
    if len(exit_tiles) != 4:
        raise ValueError("Layout must contain a 2x2 exit with 4 X tiles")

    return StaticLevelModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        walls=frozenset(walls),
        block_positions=tuple(blocks),
        enemy_positions=tuple(enemies),
        player_start=player_start,
        exit_tiles=frozenset(exit_tiles),
    )


def _initial_dynamic_state(model: StaticLevelModel) -> DynamicState:
    enemies = tuple((x, y, DIR_LEFT, True) for x, y in model.enemy_positions)
    all_blocks_mask = (1 << len(model.block_positions)) - 1
    return DynamicState(
        player_x=model.player_start[0],
        player_y=model.player_start[1],
        player_alive=True,
        block_mask=all_blocks_mask,
        enemies=enemies,
        bomb=None,
        explosion=None,
        time_ticks=TIME_TICKS,
    )


def _alive_enemy_count(enemies: tuple[tuple[int, int, int, bool], ...]) -> int:
    return sum(1 for _x, _y, _d, alive in enemies if alive)


def _exit_is_open(_model: StaticLevelModel, state: DynamicState) -> bool:
    return state.block_mask == 0 and _alive_enemy_count(state.enemies) == 0


def _block_positions_from_mask(model: StaticLevelModel, block_mask: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, pos in enumerate(model.block_positions):
        if (block_mask >> idx) & 1:
            out.add(pos)
    return out


def _enemy_occupied_positions(enemies: tuple[tuple[int, int, int, bool], ...]) -> set[tuple[int, int]]:
    return {(x, y) for x, y, _d, alive in enemies if alive}


def _is_player_blocked(model: StaticLevelModel, state: DynamicState, x: int, y: int) -> bool:
    if x < 0 or y < 1 or x >= model.width or y >= model.height:
        return True

    if (x, y) in model.walls:
        return True
    if (x, y) in _block_positions_from_mask(model, state.block_mask):
        return True

    if (x, y) in model.exit_tiles and not _exit_is_open(model, state):
        return True

    bomb = state.bomb
    return bool(
        bomb is not None and (x, y) == (bomb.x, bomb.y) and (state.player_x, state.player_y) != (bomb.x, bomb.y)
    )


def _is_enemy_blocked(
    model: StaticLevelModel, state: DynamicState, candidate: tuple[int, int], occupied_by_enemies: set[tuple[int, int]]
) -> bool:
    x, y = candidate
    if x < 0 or y < 1 or x >= model.width or y >= model.height:
        return True
    if candidate in model.walls:
        return True
    if candidate in _block_positions_from_mask(model, state.block_mask):
        return True
    if candidate in model.exit_tiles:
        return True
    if state.bomb is not None and candidate == (state.bomb.x, state.bomb.y):
        return True
    return candidate in occupied_by_enemies


def _blast_cells(model: StaticLevelModel, state: DynamicState, bx: int, by: int) -> tuple[tuple[int, int], ...]:
    alive_blocks = _block_positions_from_mask(model, state.block_mask)
    cells = [(bx, by)]
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        for step in range(1, EXPLOSION_RANGE + 1):
            x = bx + dx * step
            y = by + dy * step
            if x < 0 or y < 1 or x >= model.width or y >= model.height:
                break
            if (x, y) in model.walls:
                break
            if (x, y) in model.exit_tiles:
                break
            cells.append((x, y))
            if (x, y) in alive_blocks:
                break
    return tuple(cells)


def transition_step(model: StaticLevelModel, state: DynamicState, action_id: int) -> TransitionResult:
    player_x = state.player_x
    player_y = state.player_y
    player_alive = state.player_alive
    block_mask = state.block_mask
    enemies = [list(enemy) for enemy in state.enemies]
    bomb = state.bomb
    explosion = state.explosion
    time_ticks = state.time_ticks

    placed_bomb = False
    explosion_created = False

    # 1) Apply action (restart click intentionally omitted here for solver model).
    if action_id in (1, 2, 3, 4):
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action_id - 1]
        nx = player_x + dx
        ny = player_y + dy
        temp_state = DynamicState(
            player_x=player_x,
            player_y=player_y,
            player_alive=player_alive,
            block_mask=block_mask,
            enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
            bomb=bomb,
            explosion=explosion,
            time_ticks=time_ticks,
        )
        if not _is_player_blocked(model, temp_state, nx, ny):
            player_x, player_y = nx, ny
    elif action_id == 5:
        if bomb is None:
            bomb = BombState(player_x, player_y, 3)
            placed_bomb = True

    # 3) decrease time.
    time_ticks -= 1
    if time_ticks <= 0:
        dynamic = DynamicState(
            player_x=player_x,
            player_y=player_y,
            player_alive=False,
            block_mask=block_mask,
            enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
            bomb=bomb,
            explosion=explosion,
            time_ticks=0,
        )
        return TransitionResult(state=dynamic, won=False, failed=True, placed_bomb=placed_bomb, explosion_created=False)

    # 4) advance bomb (fuse tick / explode creation).
    created_explosion_cells: tuple[tuple[int, int], ...] | None = None
    if bomb is not None:
        if bomb.fuse > 1:
            bomb = BombState(bomb.x, bomb.y, bomb.fuse - 1)
        else:
            created_explosion_cells = _blast_cells(
                model,
                DynamicState(
                    player_x=player_x,
                    player_y=player_y,
                    player_alive=player_alive,
                    block_mask=block_mask,
                    enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
                    bomb=bomb,
                    explosion=explosion,
                    time_ticks=time_ticks,
                ),
                bomb.x,
                bomb.y,
            )
            bomb = None
            explosion = ExplosionState(phase=1, cells=created_explosion_cells)
            explosion_created = True

    # 5) apply explosion effects from newly created frame A.
    if created_explosion_cells is not None:
        blast = set(created_explosion_cells)
        for idx, (bx, by) in enumerate(model.block_positions):
            if ((block_mask >> idx) & 1) and (bx, by) in blast:
                block_mask &= ~(1 << idx)

        for enemy in enemies:
            ex, ey, _dir, alive = enemy
            if alive and (ex, ey) in blast:
                enemy[3] = False

        if (player_x, player_y) in blast:
            player_alive = False

    if not player_alive:
        dynamic = DynamicState(
            player_x=player_x,
            player_y=player_y,
            player_alive=False,
            block_mask=block_mask,
            enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
            bomb=bomb,
            explosion=explosion,
            time_ticks=time_ticks,
        )
        return TransitionResult(
            state=dynamic, won=False, failed=True, placed_bomb=placed_bomb, explosion_created=explosion_created
        )

    # 6) move enemies.
    occupied = _enemy_occupied_positions(tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies))
    for enemy in enemies:
        ex, ey, direction, alive = (int(enemy[0]), int(enemy[1]), int(enemy[2]), bool(enemy[3]))
        if not alive:
            continue
        occupied.discard((ex, ey))

        forward_dx = -1 if direction == DIR_LEFT else 1
        cand = (ex + forward_dx, ey)
        temp_state = DynamicState(
            player_x=player_x,
            player_y=player_y,
            player_alive=player_alive,
            block_mask=block_mask,
            enemies=tuple((int(v[0]), int(v[1]), int(v[2]), bool(v[3])) for v in enemies),
            bomb=bomb,
            explosion=explosion,
            time_ticks=time_ticks,
        )
        if not _is_enemy_blocked(model, temp_state, cand, occupied):
            enemy[0], enemy[1] = cand
            occupied.add(cand)
            continue

        reverse_direction = DIR_RIGHT if direction == DIR_LEFT else DIR_LEFT
        reverse_dx = -1 if reverse_direction == DIR_LEFT else 1
        cand2 = (ex + reverse_dx, ey)
        if not _is_enemy_blocked(model, temp_state, cand2, occupied):
            enemy[0], enemy[1], enemy[2] = cand2[0], cand2[1], reverse_direction
            occupied.add(cand2)
            continue

        enemy[2] = reverse_direction
        occupied.add((ex, ey))

    # 7) check entities walking into active explosions (A/B).
    if explosion is not None:
        blast = set(explosion.cells)
        if (player_x, player_y) in blast:
            player_alive = False
        for enemy in enemies:
            ex, ey, _direction, alive = enemy
            if alive and (ex, ey) in blast:
                enemy[3] = False

    if not player_alive:
        dynamic = DynamicState(
            player_x=player_x,
            player_y=player_y,
            player_alive=False,
            block_mask=block_mask,
            enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
            bomb=bomb,
            explosion=explosion,
            time_ticks=time_ticks,
        )
        return TransitionResult(
            state=dynamic, won=False, failed=True, placed_bomb=placed_bomb, explosion_created=explosion_created
        )

    # 8) advance explosions A->B->clear.
    if explosion is not None:
        if explosion.phase == 1:
            explosion = ExplosionState(phase=2, cells=explosion.cells)
        else:
            explosion = None

    # 9) exit lock is derived from live targets.
    exit_open = (
        block_mask == 0
        and _alive_enemy_count(tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies)) == 0
    )

    # 10) win check.
    won = player_alive and exit_open and (player_x, player_y) in model.exit_tiles

    dynamic = DynamicState(
        player_x=player_x,
        player_y=player_y,
        player_alive=player_alive,
        block_mask=block_mask,
        enemies=tuple((int(e[0]), int(e[1]), int(e[2]), bool(e[3])) for e in enemies),
        bomb=bomb,
        explosion=explosion,
        time_ticks=time_ticks,
    )
    return TransitionResult(
        state=dynamic, won=won, failed=not player_alive, placed_bomb=placed_bomb, explosion_created=explosion_created
    )


def _state_priority(model: StaticLevelModel, state: DynamicState, steps: int) -> int:
    remaining_blocks = state.block_mask.bit_count()
    remaining_enemies = _alive_enemy_count(state.enemies)
    if remaining_blocks + remaining_enemies == 0:
        # Favor moving toward exit when all targets are gone.
        dist_to_exit = min(abs(state.player_x - ex) + abs(state.player_y - ey) for ex, ey in model.exit_tiles)
        return steps + dist_to_exit

    target_positions: list[tuple[int, int]] = []
    for idx, pos in enumerate(model.block_positions):
        if (state.block_mask >> idx) & 1:
            target_positions.append(pos)
    for ex, ey, _d, alive in state.enemies:
        if alive:
            target_positions.append((ex, ey))

    nearest = min(abs(state.player_x - tx) + abs(state.player_y - ty) for tx, ty in target_positions)
    bombs_needed_lb = (remaining_blocks + remaining_enemies + 2) // 3
    return steps + nearest + bombs_needed_lb


def compute_level_solution(model: StaticLevelModel, max_expansions: int = 320000) -> list[int]:
    start = _initial_dynamic_state(model)

    frontier: list[tuple[int, int, int, DynamicState]] = []
    push_counter = 0
    heapq.heappush(frontier, (_state_priority(model, start, 0), 0, push_counter, start))

    previous: dict[DynamicState, DynamicState | None] = {start: None}
    previous_action: dict[DynamicState, int] = {}

    # Dominance by non-time state key; keep best remaining time.
    best_time_for_key: dict[tuple, int] = {}

    expansions = 0
    while frontier:
        _f, steps, _counter, state = heapq.heappop(frontier)
        expansions += 1
        if expansions > max_expansions:
            break

        for action_id in (1, 2, 3, 4, 5):
            result = transition_step(model, state, action_id)
            if result.failed:
                continue

            next_state = result.state
            if next_state in previous:
                continue

            key = (
                next_state.player_x,
                next_state.player_y,
                next_state.block_mask,
                next_state.enemies,
                next_state.bomb,
                next_state.explosion,
            )
            best_time = best_time_for_key.get(key)
            if best_time is not None and best_time >= next_state.time_ticks:
                continue
            best_time_for_key[key] = next_state.time_ticks

            previous[next_state] = state
            previous_action[next_state] = action_id

            if result.won:
                out: list[int] = []
                cursor = next_state
                while previous[cursor] is not None:
                    out.append(previous_action[cursor])
                    cursor = previous[cursor]  # type: ignore[index]
                out.reverse()
                return out

            next_steps = steps + 1
            push_counter += 1
            heapq.heappush(
                frontier, (_state_priority(model, next_state, next_steps), next_steps, push_counter, next_state)
            )

    # Fallback: receding-horizon progress search.
    current = start
    stitched_plan: list[int] = []

    def remaining_targets(s: DynamicState) -> int:
        return s.block_mask.bit_count() + _alive_enemy_count(s.enemies)

    max_macro_iters = 96
    horizon = 22

    for _ in range(max_macro_iters):
        if _exit_is_open(model, current) and (current.player_x, current.player_y) in model.exit_tiles:
            return stitched_plan

        baseline_targets = remaining_targets(current)
        queue = deque([(current, [])])
        seen: set[DynamicState] = {current}

        best_progress_state: DynamicState | None = None
        best_progress_actions: list[int] | None = None
        best_progress_score = -(10**9)

        local_expansions = 0
        local_cap = 160000

        while queue:
            state, actions = queue.popleft()
            local_expansions += 1
            if local_expansions > local_cap:
                break
            depth = len(actions)
            if depth >= horizon:
                continue

            for action_id in (1, 2, 3, 4, 5):
                result = transition_step(model, state, action_id)
                if result.failed:
                    continue
                nxt = result.state
                if nxt in seen:
                    continue
                seen.add(nxt)

                nxt_actions = [*actions, action_id]
                if result.won:
                    return stitched_plan + nxt_actions

                gain = baseline_targets - remaining_targets(nxt)
                score = gain * 1000 - len(nxt_actions)
                if _exit_is_open(model, nxt):
                    exit_dist = min(abs(nxt.player_x - ex) + abs(nxt.player_y - ey) for ex, ey in model.exit_tiles)
                    score += 200 - exit_dist
                else:
                    score += max(0, 40 - _state_priority(model, nxt, len(stitched_plan) + len(nxt_actions)))

                if score > best_progress_score:
                    best_progress_score = score
                    best_progress_state = nxt
                    best_progress_actions = nxt_actions

                queue.append((nxt, nxt_actions))

        if best_progress_state is None or best_progress_actions is None:
            break

        # Require strict target progress when possible to avoid cycling forever.
        if remaining_targets(best_progress_state) >= baseline_targets and len(stitched_plan) > 0:
            break

        stitched_plan.extend(best_progress_actions)
        current = best_progress_state

    raise RuntimeError("BombDropTactics solver failed to find a solution")


def _level_to_model(name: str, rows: list[str]) -> tuple[Level, StaticLevelModel]:
    model = _build_static_model(rows)

    floor_pixels = [[COLOR_FLOOR] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    wall_pixels = [[-1] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    timebar_pixels = [[COLOR_TIME_FILLED] * GRID_WIDTH]

    for x, y in model.walls:
        wall_pixels[y][x] = COLOR_WALL

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-20),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=-5, tags=["wall"]),
        Sprite(pixels=timebar_pixels, name="timebar", x=0, y=0, collidable=False, layer=30, tags=["timebar"]),
        Sprite(
            pixels=[[-1 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)],
            name="explosion_overlay",
            x=0,
            y=0,
            collidable=False,
            layer=18,
            tags=["explosion_overlay"],
        ),
    ]

    for idx, (bx, by) in enumerate(model.block_positions):
        sprites.append(
            Sprite(pixels=[[COLOR_BLOCK]], name=f"block_{idx}", x=bx, y=by, collidable=True, layer=8, tags=["block"])
        )

    for idx, (ex, ey) in enumerate(model.enemy_positions):
        sprites.append(
            Sprite(
                pixels=[[COLOR_ENEMY_A]], name=f"enemy_{idx}", x=ex, y=ey, collidable=False, layer=10, tags=["enemy"]
            )
        )

    exit_min_x = min(x for x, _y in model.exit_tiles)
    exit_min_y = min(y for _x, y in model.exit_tiles)
    sprites.append(
        Sprite(
            pixels=[[COLOR_EXIT_LOCKED, COLOR_EXIT_LOCKED], [COLOR_EXIT_LOCKED, COLOR_EXIT_LOCKED]],
            name="exit",
            x=exit_min_x,
            y=exit_min_y,
            collidable=True,
            layer=7,
            tags=["exit"],
        )
    )

    sprites.append(
        Sprite(
            pixels=[[COLOR_PLAYER_A]],
            name="player",
            x=model.player_start[0],
            y=model.player_start[1],
            collidable=False,
            layer=20,
            tags=["player"],
        )
    )

    sprites.append(
        Sprite(pixels=[[COLOR_BOMB_3]], name="bomb", x=0, y=0, collidable=False, layer=14, visible=False, tags=["bomb"])
    )

    level = Level(name=name, sprites=sprites, grid_size=(GRID_WIDTH, GRID_HEIGHT), data={"layout": tuple(rows)})
    return level, model


class Bombdroptactics(ARCBaseGame):
    def __init__(self) -> None:
        built = [_level_to_model(name, rows) for name, rows in LEVEL_LAYOUTS]
        self._models = [model for _level, model in built]
        levels = [level for level, _model in built]
        camera = Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, 0, 0, [])
        super().__init__(
            "bombdroptactics",
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def on_set_level(self, level: Level) -> None:
        self._model = self._models[self._current_level_index]
        self._world = _initial_dynamic_state(self._model)
        self._pending_restart = False
        self._anim_tick = 0
        self._outcome: str | None = None
        self._outcome_ticks = 0

        self._player = level.get_sprites_by_name("player")[0]
        self._exit = level.get_sprites_by_name("exit")[0]
        self._bomb_sprite = level.get_sprites_by_name("bomb")[0]
        self._timebar = level.get_sprites_by_name("timebar")[0]
        self._explosion_overlay = level.get_sprites_by_name("explosion_overlay")[0]

        self._block_sprites = {
            idx: level.get_sprites_by_name(f"block_{idx}")[0] for idx in range(len(self._model.block_positions))
        }
        self._enemy_sprites = {
            idx: level.get_sprites_by_name(f"enemy_{idx}")[0] for idx in range(len(self._model.enemy_positions))
        }

        self._sync_visuals()

    def plan_current_level(self) -> list[int]:
        return compute_level_solution(self._model, max_expansions=400_000)

    def _reset_level_now(self) -> None:
        self.set_level(self._current_level_index)

    def _open_exit(self) -> bool:
        return _exit_is_open(self._model, self._world)

    def _set_outcome(self, outcome: str) -> None:
        if self._outcome is None:
            self._outcome = outcome
            self._outcome_ticks = OUTCOME_FLASH_TICKS

    def _apply_outcome_flash(self) -> None:
        if self._outcome is None:
            return

        flash_on = (self._outcome_ticks % 2) == 0
        if self._outcome == "win":
            c = COLOR_TIME_FILLED if flash_on else COLOR_EXIT_OPEN
            self._exit.pixels = np.array([[c, c], [c, c]], dtype=np.int8)
        elif self._outcome == "fail":
            c = COLOR_EXPLOSION_A if flash_on else COLOR_EXPLOSION_B
            self._player.pixels[0][0] = c

        self._outcome_ticks -= 1
        if self._outcome_ticks > 0:
            return

        if self._outcome == "win":
            self.next_level()
        else:
            self.lose()
        self._outcome = None

    def _sync_timebar(self) -> None:
        segments_filled = min(GRID_WIDTH, max(0, (self._world.time_ticks + 3) // 4))
        low_time = self._world.time_ticks <= (TIME_TICKS // 5)
        fill_color = COLOR_BOMB_1 if (low_time and (self._anim_tick % 2 == 1)) else COLOR_TIME_FILLED
        row = [COLOR_VOID] * GRID_WIDTH
        for x in range(segments_filled):
            row[x] = fill_color
        self._timebar.pixels = np.array([row], dtype=np.int8)

    def _sync_explosions(self) -> None:
        pixels = [[-1 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        if self._world.explosion is not None:
            color = COLOR_EXPLOSION_A if self._world.explosion.phase == 1 else COLOR_EXPLOSION_B
            for x, y in self._world.explosion.cells:
                pixels[y][x] = color
        self._explosion_overlay.pixels = np.array(pixels, dtype=np.int8)

    def _sync_bomb(self) -> None:
        bomb = self._world.bomb
        if bomb is None:
            self._bomb_sprite.set_visible(False)
            return

        self._bomb_sprite.set_visible(True)
        self._bomb_sprite.set_position(bomb.x, bomb.y)
        if bomb.fuse == 3:
            self._bomb_sprite.pixels[0][0] = COLOR_BOMB_3
        elif bomb.fuse == 2:
            self._bomb_sprite.pixels[0][0] = COLOR_BOMB_2
        else:
            self._bomb_sprite.pixels[0][0] = COLOR_BOMB_1

    def _sync_entities(self) -> None:
        self._player.set_position(self._world.player_x, self._world.player_y)
        self._player.pixels[0][0] = COLOR_PLAYER_A if (self._anim_tick % 2 == 0) else COLOR_PLAYER_B

        for idx, (ex, ey, _d, alive) in enumerate(self._world.enemies):
            sprite = self._enemy_sprites[idx]
            sprite.set_visible(bool(alive))
            if not alive:
                continue
            sprite.set_position(ex, ey)
            sprite.pixels[0][0] = COLOR_ENEMY_A if (self._anim_tick % 2 == 0) else COLOR_ENEMY_B

        for idx, pos in enumerate(self._model.block_positions):
            sprite = self._block_sprites[idx]
            alive = ((self._world.block_mask >> idx) & 1) == 1
            sprite.set_visible(alive)
            sprite.set_collidable(alive)
            if alive:
                sprite.set_position(pos[0], pos[1])

        open_exit = self._open_exit()
        self._exit.set_collidable(not open_exit)
        exit_color = COLOR_EXIT_OPEN if open_exit else COLOR_EXIT_LOCKED
        self._exit.pixels = np.array([[exit_color, exit_color], [exit_color, exit_color]], dtype=np.int8)

    def _sync_visuals(self) -> None:
        self._sync_entities()
        self._sync_bomb()
        self._sync_explosions()
        self._sync_timebar()

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

        if self._outcome is not None:
            self._anim_tick += 1
            self._apply_outcome_flash()
            self.complete_action()
            return

        action_id = int(self.action.id.value)

        # 1) apply action
        if action_id == int(GameAction.ACTION6.value):
            self._pending_restart = True

        if self._pending_restart:
            self._pending_restart = False
            self._reset_level_now()
            self.complete_action()
            return

        model_action = action_id
        if model_action not in (1, 2, 3, 4, 5):
            model_action = 0

        result = transition_step(self._model, self._world, model_action)
        self._world = result.state

        self._anim_tick += 1

        if result.won:
            self._set_outcome("win")
        elif result.failed or not self._world.player_alive:
            self._set_outcome("fail")

        self._sync_visuals()
        self.complete_action()


def level_models() -> list[StaticLevelModel]:
    return [_build_static_model(rows) for _name, rows in LEVEL_LAYOUTS]
