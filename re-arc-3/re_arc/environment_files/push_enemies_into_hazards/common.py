from __future__ import annotations

import heapq
from dataclasses import dataclass

ACTION_DELTAS: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
WAIT_ACTION = 5

PATROLLER_DIRS: dict[str, tuple[int, int]] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "time_max_steps": 60,
        "patroller_dirs": [],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#....................^^....#",
            "#............o.......^^....#",
            "#..........................#",
            "#..........................#",
            "#...@......................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 2",
        "time_max_steps": 80,
        "patroller_dirs": [],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#...##.############.####...#",
            "#...#.......o.....@..^^#...#",
            "#...####################...#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 3",
        "time_max_steps": 110,
        "patroller_dirs": [],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#..........................#",
            "#...^^.....................#",
            "#...^^.....................#",
            "#............o.............#",
            "#..........................#",
            "#............##............#",
            "#.........o..##............#",
            "#.......##.................#",
            "#.......##........o........#",
            "#..........................#",
            "#....................^^....#",
            "#....................^^....#",
            "#...@......................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 4",
        "time_max_steps": 120,
        "patroller_dirs": ["right"],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#..........................#",
            "#...................^^.....#",
            "#..............o....^^.....#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#...######.######.######...#",
            "#...#.8..............^^#...#",
            "#...####################...#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#...@......................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 5",
        "time_max_steps": 130,
        "patroller_dirs": [],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#.............#............#",
            "#.............#............#",
            "#.............#............#",
            "#.........o...#.....^^.....#",
            "#...................^^.....#",
            "#.............#............#",
            "#............6#............#",
            "#.............#............#",
            "#...##############.#####...#",
            "#.............#............#",
            "#.....^^...................#",
            "#.....^^......#....o.......#",
            "#...@.........#............#",
            "#.............#............#",
            "############################",
        ],
    },
    {
        "name": "Level 6",
        "time_max_steps": 150,
        "patroller_dirs": ["right", "left"],
        "layout": [
            "############################",
            "#==========================#",
            "############################",
            "#..........................#",
            "#..........................#",
            "#....o.....................#",
            "#.....#...........8..#.....#",
            "#...........####...........#",
            "#...........^^^^...........#",
            "#........32.^^^^...........#",
            "#........22.^^^^...........#",
            "#...........^^^^...........#",
            "#...........####......6....#",
            "#.....#...........8..#.....#",
            "#.....o....................#",
            "#.............@............#",
            "#..........................#",
            "############################",
        ],
    },
]


@dataclass(frozen=True)
class LevelModel:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    spikes: frozenset[tuple[int, int]]
    player_start: tuple[int, int]
    enemies_start: tuple[tuple[int, int, int, int, int, int], ...]
    brute_start: tuple[int, int]
    has_brute: bool
    time_max_steps: int


def _enemy_type_code(ch: str) -> int:
    if ch == "o":
        return 0
    if ch == "8":
        return 1
    if ch == "6":
        return 2
    raise ValueError(f"Unsupported enemy type: {ch!r}")


def _parse_level(spec: dict) -> LevelModel:
    rows = [str(row) for row in spec.get("layout") or []]
    if not rows:
        raise ValueError("Level layout must not be empty.")

    width = len(rows[0])
    height = len(rows)
    if width <= 0 or width > 64 or height <= 0 or height > 64:
        raise ValueError("Level grid must be within 1..64 in each dimension.")

    for row in rows:
        if len(row) != width:
            raise ValueError("All level rows must have the same width.")

    walls: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None
    enemies: list[tuple[int, int, int, int, int, int]] = []
    brute_cells: set[tuple[int, int]] = set()

    patroller_dirs = [str(name).lower() for name in (spec.get("patroller_dirs") or [])]
    patroller_idx = 0

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
                continue
            if y == 1:
                walls.add((x, y))
            if ch in {"^", "v"}:
                spikes.add((x, y))
            elif ch == "@":
                player = (x, y)
            elif ch in {"o", "8", "6"}:
                etype = _enemy_type_code(ch)
                if etype == 1:
                    if patroller_idx >= len(patroller_dirs):
                        raise ValueError("Missing initial patroller direction in level spec.")
                    dname = patroller_dirs[patroller_idx]
                    patroller_idx += 1
                    dx, dy = PATROLLER_DIRS.get(dname, (None, None))
                    if dx is None:
                        raise ValueError(f"Invalid patroller direction: {dname!r}")
                else:
                    dx, dy = (0, 0)
                enemies.append((etype, x, y, 0, dx, dy))
            elif ch in {"2", "3"}:
                brute_cells.add((x, y))

    if player is None:
        raise ValueError("Each level must include one player '@'.")
    if patroller_idx != len(patroller_dirs):
        raise ValueError("Extra patroller directions provided for level.")

    has_brute = bool(brute_cells)
    brute_top_left = (-1, -1)
    if has_brute:
        min_x = min(x for x, _ in brute_cells)
        min_y = min(y for _, y in brute_cells)
        expected = {(min_x, min_y), (min_x + 1, min_y), (min_x, min_y + 1), (min_x + 1, min_y + 1)}
        if brute_cells != expected:
            raise ValueError("Brute must be exactly one contiguous 2x2 block.")
        brute_top_left = (min_x, min_y)

    time_max_steps = int(spec.get("time_max_steps") or 0)
    if time_max_steps <= 0:
        raise ValueError("time_max_steps must be positive.")

    return LevelModel(
        width=width,
        height=height,
        walls=frozenset(walls),
        spikes=frozenset(spikes),
        player_start=player,
        enemies_start=tuple(enemies),
        brute_start=brute_top_left,
        has_brute=has_brute,
        time_max_steps=time_max_steps,
    )


def build_level_models() -> list[LevelModel]:
    return [_parse_level(spec) for spec in LEVEL_SPECS]


def initial_state(model: LevelModel):
    bx, by = model.brute_start if model.has_brute else (-1, -1)
    return (
        int(model.player_start[0]),
        int(model.player_start[1]),
        int(model.time_max_steps),
        int(bx),
        int(by),
        tuple(model.enemies_start),
    )


def _brute_cells(brute_x: int, brute_y: int) -> set[tuple[int, int]]:
    if brute_x < 0 or brute_y < 0:
        return set()
    return {(brute_x, brute_y), (brute_x + 1, brute_y), (brute_x, brute_y + 1), (brute_x + 1, brute_y + 1)}


def _is_wall(model: LevelModel, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= model.width or y >= model.height:
        return True
    return (x, y) in model.walls


def _occupancy(enemies: list[list[int]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for idx, enemy in enumerate(enemies):
        if enemy[1] < 0:
            continue
        out[(enemy[1], enemy[2])] = idx
    return out


def _blocked_for_enemy(
    model: LevelModel, x: int, y: int, enemies: list[list[int]], enemy_idx: int, brute_cells: set[tuple[int, int]]
) -> bool:
    if _is_wall(model, x, y):
        return True
    if (x, y) in brute_cells:
        return True
    for idx, enemy in enumerate(enemies):
        if idx == enemy_idx or enemy[1] < 0:
            continue
        if enemy[1] == x and enemy[2] == y:
            return True
    return False


def _remaining_targets(enemies: list[list[int]], brute_x: int, brute_y: int) -> int:
    alive = sum(1 for enemy in enemies if enemy[1] >= 0)
    if brute_x >= 0 and brute_y >= 0:
        alive += 1
    return alive


def step_state(model: LevelModel, state, action_id: int):
    px, py, time_left, brute_x, brute_y, enemies_tuple = state
    if action_id not in (1, 2, 3, 4, 5):
        action_id = WAIT_ACTION

    enemies = [list(enemy) for enemy in enemies_tuple]
    player_x = int(px)
    player_y = int(py)

    moved_to_spike = False

    if action_id in ACTION_DELTAS:
        dx, dy = ACTION_DELTAS[action_id]
        tx = player_x + dx
        ty = player_y + dy
        brute_cells = _brute_cells(brute_x, brute_y)
        occ = _occupancy(enemies)

        if _is_wall(model, tx, ty):
            pass
        elif (tx, ty) in model.spikes:
            player_x, player_y = tx, ty
            moved_to_spike = True
        elif (tx, ty) in occ:
            idx = occ[(tx, ty)]
            nx = tx + dx
            ny = ty + dy
            if _is_wall(model, nx, ny) or (nx, ny) in _occupancy(enemies) or (nx, ny) in brute_cells:
                pass
            elif (nx, ny) in model.spikes:
                enemies[idx][1] = -1
                enemies[idx][2] = -1
                enemies[idx][3] = 0
                player_x, player_y = tx, ty
            else:
                enemies[idx][1] = nx
                enemies[idx][2] = ny
                enemies[idx][3] = 2
                player_x, player_y = tx, ty
        elif (tx, ty) in brute_cells:
            old_cells = set(brute_cells)
            new_cells = {(bx + dx, by + dy) for bx, by in old_cells}
            blocked = False
            for cx, cy in new_cells:
                if _is_wall(model, cx, cy):
                    blocked = True
                    break
                if (cx, cy) == (player_x, player_y) and (cx, cy) not in old_cells:
                    blocked = True
                    break
                for enemy in enemies:
                    if enemy[1] >= 0 and (enemy[1], enemy[2]) == (cx, cy):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                player_x, player_y = tx, ty
                if any((cx, cy) in model.spikes for cx, cy in new_cells):
                    brute_x, brute_y = (-1, -1)
                else:
                    brute_x += dx
                    brute_y += dy
        else:
            player_x, player_y = tx, ty

    if moved_to_spike:
        return None

    brute_cells = _brute_cells(brute_x, brute_y)

    # Enemy AI phase.
    for idx, enemy in enumerate(enemies):
        etype, ex, ey, stun, edir_x, edir_y = enemy
        if ex < 0:
            continue
        if stun > 0:
            continue
        if etype == 0:
            continue

        if etype == 1:
            nx = ex + edir_x
            ny = ey + edir_y
            if _blocked_for_enemy(model, nx, ny, enemies, idx, brute_cells):
                enemy[4] = -edir_x
                enemy[5] = -edir_y
            else:
                enemy[1] = nx
                enemy[2] = ny
            continue

        if etype == 2:
            moved = False
            dx = player_x - ex
            dy = player_y - ey
            candidates: list[tuple[int, int]] = []
            if dx != 0:
                candidates.append((1 if dx > 0 else -1, 0))
            if dy != 0:
                candidates.append((0, 1 if dy > 0 else -1))
            for mx, my in candidates:
                nx = ex + mx
                ny = ey + my
                if _blocked_for_enemy(model, nx, ny, enemies, idx, brute_cells):
                    continue
                enemy[1] = nx
                enemy[2] = ny
                moved = True
                break
            if moved:
                continue

    # Collisions and hazard cleanup.
    for enemy in enemies:
        if enemy[1] < 0:
            continue
        if (enemy[1], enemy[2]) == (player_x, player_y):
            return None

    for enemy in enemies:
        if enemy[1] < 0:
            continue
        if (enemy[1], enemy[2]) in model.spikes:
            enemy[1] = -1
            enemy[2] = -1
            enemy[3] = 0

    if _remaining_targets(enemies, brute_x, brute_y) == 0:
        return (
            int(player_x),
            int(player_y),
            int(time_left),
            int(brute_x),
            int(brute_y),
            tuple((int(e[0]), int(e[1]), int(e[2]), int(e[3]), int(e[4]), int(e[5])) for e in enemies),
        )

    for enemy in enemies:
        if enemy[1] >= 0 and enemy[3] > 0:
            enemy[3] -= 1

    next_time = int(time_left) - 1
    if next_time <= 0:
        return None

    return (
        int(player_x),
        int(player_y),
        int(next_time),
        int(brute_x),
        int(brute_y),
        tuple((int(e[0]), int(e[1]), int(e[2]), int(e[3]), int(e[4]), int(e[5])) for e in enemies),
    )


def _dominance_key(state):
    px, py, _, bx, by, enemies = state
    return (px, py, bx, by, enemies)


def find_plan(model: LevelModel) -> list[int] | None:
    start = initial_state(model)
    parents: dict[tuple, tuple | None] = {start: None}
    parent_action: dict[tuple, int] = {}
    g_score: dict[tuple, int] = {start: 0}
    best_time: dict[tuple, int] = {_dominance_key(start): int(model.time_max_steps)}
    goal_state: tuple | None = None

    spike_cells = list(model.spikes)

    def heuristic(state) -> int:
        px, py, _, bx, by, enemies = state
        alive_enemy_cells: list[tuple[int, int]] = []
        h = 0
        for enemy in enemies:
            _, ex, ey, _, _, _ = enemy
            if ex < 0:
                continue
            alive_enemy_cells.append((ex, ey))
            if spike_cells:
                h += min(abs(ex - sx) + abs(ey - sy) for sx, sy in spike_cells)
        if bx >= 0 and by >= 0:
            brute = [(bx, by), (bx + 1, by), (bx, by + 1), (bx + 1, by + 1)]
            alive_enemy_cells.extend(brute)
            if spike_cells:
                h += min(abs(cx - sx) + abs(cy - sy) for cx, cy in brute for sx, sy in spike_cells)
        if alive_enemy_cells:
            d = min(abs(px - tx) + abs(py - ty) for tx, ty in alive_enemy_cells)
            h += max(0, d - 1)
        h += 3 * len(alive_enemy_cells)
        return int(h)

    open_heap: list[tuple[int, int, int, tuple]] = []
    heapq.heappush(open_heap, (heuristic(start), -int(model.time_max_steps), 0, start))

    expanded = 0
    max_expansions = 600_000
    while open_heap and expanded < max_expansions:
        _, _neg_time_left, g, state = heapq.heappop(open_heap)
        if g != g_score.get(state, -1):
            continue
        expanded += 1

        _, _, _, bx, by, enemies = state
        if _remaining_targets([list(enemy) for enemy in enemies], bx, by) == 0:
            goal_state = state
            break

        for action_id in (1, 2, 3, 4, 5):
            nxt = step_state(model, state, action_id)
            if nxt is None:
                continue

            t_left = int(nxt[2])
            dom_key = _dominance_key(nxt)
            prior_time = best_time.get(dom_key)
            if prior_time is not None and prior_time >= t_left:
                continue
            best_time[dom_key] = t_left

            cand_g = g + 1
            prior_g = g_score.get(nxt)
            if prior_g is not None and prior_g <= cand_g:
                continue
            g_score[nxt] = cand_g
            parents[nxt] = state
            parent_action[nxt] = action_id
            f = cand_g + heuristic(nxt)
            heapq.heappush(open_heap, (f, -t_left, cand_g, nxt))

    if goal_state is None:
        return None

    actions: list[int] = []
    cursor = goal_state
    while parents[cursor] is not None:
        actions.append(parent_action[cursor])
        cursor = parents[cursor]  # type: ignore[index]
    actions.reverse()
    return actions
