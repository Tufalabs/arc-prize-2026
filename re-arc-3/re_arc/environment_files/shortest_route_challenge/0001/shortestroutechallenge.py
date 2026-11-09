from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "shortest_route_challenge-0001"

WIDTH = 20
HEIGHT = 18

COLOR_FLOOR = 0
COLOR_WALL = 1
COLOR_PLAYER = 2
COLOR_PLAYER_KEY = 3
COLOR_EXIT = 4
COLOR_ACCENT = 5
COLOR_EMPTY = 6
COLOR_TIMEBAR = 7
COLOR_CLOSED = 8
COLOR_HAZARD = 9
COLOR_CRATE = 10
COLOR_PLATE_INACTIVE = 11
COLOR_PLATE_ACTIVE = 12
COLOR_ENEMY = 13
COLOR_ICE = 14
COLOR_TELEPORT = 15

DIR_NONE = 0
DIR_UP = 1
DIR_DOWN = 2
DIR_LEFT = 3
DIR_RIGHT = 4

MOVE_BY_ACTION = {
    int(GameAction.ACTION1.value): DIR_UP,
    int(GameAction.ACTION2.value): DIR_DOWN,
    int(GameAction.ACTION3.value): DIR_LEFT,
    int(GameAction.ACTION4.value): DIR_RIGHT,
}

DELTA_BY_DIR = {DIR_UP: (0, -1), DIR_DOWN: (0, 1), DIR_LEFT: (-1, 0), DIR_RIGHT: (1, 0)}

OPPOSITE_DIR = {DIR_UP: DIR_DOWN, DIR_DOWN: DIR_UP, DIR_LEFT: DIR_RIGHT, DIR_RIGHT: DIR_LEFT}

GATE_CLOSED = 0
GATE_OPENING = 1
GATE_OPEN = 2

DOOR_CLOSED = 0
DOOR_OPENING = 1
DOOR_OPEN = 2
DOOR_CLOSING = 3

NO_PENDING_TELEPORT = -1
NO_CRATE = -1

# State tuple layout:
# 0 px
# 1 py
# 2 crate_x
# 3 crate_y
# 4 has_key
# 5 key_present
# 6 gate_state
# 7 door_state
# 8 sliding_dir
# 9 teleport_pending (destination pad index or -1)
# 10 moves_left
# 11 tick_mod4
# 12 anim_phase (0/1)
# 13... enemies, packed as (x, y, dir_sign) * enemy_count
BASE_STATE_FIELDS = 13


LEVEL_SPECS = [
    {
        "name": "Level 1",
        "max_moves": 18,
        "layout": [
            "==================--",
            "####################",
            "####################",
            "#P.............EE..#",
            "#..............EE..#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..##############..#",
            "#..................#",
            "#..##############..#",
            "#..................#",
            "#..................#",
            "#..##############..#",
            "#..................#",
            "#..................#",
            "#..................#",
            "####################",
        ],
    },
    {
        "name": "Level 2",
        "max_moves": 16,
        "layout": [
            "================----",
            "####################",
            "####################",
            "#P..K..............#",
            "##########G######..#",
            "##########G######..#",
            "#..........EE......#",
            "#..........EE......#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "#..................#",
            "####################",
        ],
    },
    {
        "name": "Level 3",
        "max_moves": 16,
        "layout": [
            "================----",
            "####################",
            "####################",
            "#......EE..........#",
            "#......EE..........#",
            "#######.############",
            "#######.############",
            "#######.############",
            "#######.############",
            "#######.############",
            "#######^############",
            "#######^############",
            "#######^############",
            "#####...############",
            "#######.############",
            "#######P############",
            "####################",
            "####################",
        ],
    },
    {
        "name": "Level 4",
        "max_moves": 20,
        "layout": [
            "====================",
            "####################",
            "####################",
            "#P.................#",
            "#..................#",
            "#..COO.............#",
            "#...OO.............#",
            "############DD######",
            "#..................#",
            "#..................#",
            "#............EE....#",
            "#............EE....#",
            "#..................#",
            "#........M.........#",
            "#..................#",
            "#..................#",
            "#..................#",
            "####################",
        ],
    },
    {
        "name": "Level 5",
        "max_moves": 18,
        "layout": [
            "==================--",
            "####################",
            "####################",
            "#P..IIII......##...#",
            "#...I..I.....EETT..#",
            "#...I..I.....EETT..#",
            "#...IIII.....^##...#",
            "#.............##...#",
            "#..TT.........##...#",
            "#..TT.........##...#",
            "#.............##...#",
            "#.............##...#",
            "#.............##...#",
            "#.............##...#",
            "#.............##...#",
            "#.............##...#",
            "#.............##...#",
            "####################",
        ],
    },
    {
        "name": "Level 6",
        "max_moves": 20,
        "layout": [
            "====================",
            "####################",
            "####################",
            "#P..TT.............#",
            "#...TT....K........#",
            "#..................#",
            "######G#############",
            "######G...........##",
            "#.....^^...........#",
            "#.OO..^^....M......#",
            "#.OOC.^^...........#",
            "#..................#",
            "##########DD########",
            "#............EE....#",
            "#............EE....#",
            "#..TT..............#",
            "#..TT..............#",
            "####################",
        ],
    },
]


@dataclass(frozen=True)
class EnemySpec:
    x: int
    y: int
    axis: int  # 0 = horizontal, 1 = vertical
    dir_sign: int


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _overlay(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    out = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    for x, y in cells:
        out[y, x] = int(color)
    return out


def _overlay_checker(cells: set[tuple[int, int]], a: int, b: int, phase: int) -> np.ndarray:
    out = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    for x, y in cells:
        out[y, x] = int(a if ((x + y + phase) & 1) == 0 else b)
    return out


def _is_2x2_block(cells: set[tuple[int, int]]) -> bool:
    if len(cells) != 4:
        return False
    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells})
    if len(xs) != 2 or len(ys) != 2:
        return False
    expected = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])}
    return cells == expected


def _top_left(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return min(cells, key=lambda item: (item[1], item[0]))


def _cluster_4_connected(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    clusters: list[set[tuple[int, int]]] = []
    pending = set(cells)
    while pending:
        start = next(iter(pending))
        queue: deque[tuple[int, int]] = deque([start])
        cluster = {start}
        pending.remove(start)
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt not in pending:
                    continue
                pending.remove(nxt)
                cluster.add(nxt)
                queue.append(nxt)
        clusters.append(cluster)
    return clusters


def _static_walkable_for_axis(walls: set[tuple[int, int]], x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
        return False
    return (x, y) not in walls


def _infer_enemy_axis(walls: set[tuple[int, int]], x: int, y: int) -> int:
    horizontal_open = int(_static_walkable_for_axis(walls, x - 1, y)) + int(_static_walkable_for_axis(walls, x + 1, y))
    vertical_open = int(_static_walkable_for_axis(walls, x, y - 1)) + int(_static_walkable_for_axis(walls, x, y + 1))
    if horizontal_open >= vertical_open:
        return 0
    return 1


def _parse_level(layout: list[str], max_moves: int) -> dict:
    if len(layout) != HEIGHT:
        raise ValueError(f"layout height must be {HEIGHT}")
    for row in layout:
        if len(row) != WIDTH:
            raise ValueError(f"layout width must be {WIDTH}")

    walls: set[tuple[int, int]] = set()
    exit_cells: set[tuple[int, int]] = set()
    plate_cells: set[tuple[int, int]] = set()
    gate_cells: set[tuple[int, int]] = set()
    door_cells: set[tuple[int, int]] = set()
    spike_cells: set[tuple[int, int]] = set()
    ice_cells: set[tuple[int, int]] = set()
    teleport_cells: set[tuple[int, int]] = set()

    player_start: tuple[int, int] | None = None
    key_pos: tuple[int, int] | None = None
    crate_pos: tuple[int, int] | None = None
    enemy_positions: list[tuple[int, int]] = []

    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            if y == 0:
                # HUD row is always non-walkable regardless of glyph.
                walls.add((x, y))
                continue

            if ch == "#":
                walls.add((x, y))
            elif ch == "P":
                if player_start is not None:
                    raise ValueError("multiple player starts")
                player_start = (x, y)
            elif ch == "E":
                exit_cells.add((x, y))
            elif ch == "K":
                if key_pos is not None:
                    raise ValueError("multiple keys")
                key_pos = (x, y)
            elif ch == "G":
                gate_cells.add((x, y))
            elif ch == "D":
                door_cells.add((x, y))
            elif ch == "C":
                if crate_pos is not None:
                    raise ValueError("multiple crates")
                crate_pos = (x, y)
            elif ch == "O":
                plate_cells.add((x, y))
            elif ch == "^":
                spike_cells.add((x, y))
            elif ch == "M":
                enemy_positions.append((x, y))
            elif ch == "I":
                ice_cells.add((x, y))
            elif ch == "T":
                teleport_cells.add((x, y))
            elif ch in {".", "=", "-"}:
                pass
            else:
                raise ValueError(f"unsupported glyph {ch!r}")

    if player_start is None:
        raise ValueError("missing player start")
    if not _is_2x2_block(exit_cells):
        raise ValueError("exit must be a single 2x2 block")
    if gate_cells and len(gate_cells) != 2:
        raise ValueError("locked gate must be 1x2")
    if door_cells and len(door_cells) != 2:
        raise ValueError("door must be 2x1")
    if plate_cells and not _is_2x2_block(plate_cells):
        raise ValueError("plate must be 2x2")

    teleport_pads: list[set[tuple[int, int]]] = []
    if teleport_cells:
        clusters = _cluster_4_connected(teleport_cells)
        if len(clusters) != 2:
            raise ValueError("teleporter requires exactly two pads")
        for cluster in clusters:
            if not _is_2x2_block(cluster):
                raise ValueError("each teleporter pad must be 2x2")
            teleport_pads.append(cluster)
        teleport_pads.sort(key=lambda cells: (_top_left(cells)[1], _top_left(cells)[0]))

    tele_cell_to_pad: dict[tuple[int, int], int] = {}
    tele_landing: list[tuple[int, int]] = []
    for idx, pad in enumerate(teleport_pads):
        tele_landing.append(_top_left(pad))
        for cell in pad:
            tele_cell_to_pad[cell] = idx

    enemy_specs: list[EnemySpec] = []
    for x, y in enemy_positions:
        axis = _infer_enemy_axis(walls, x, y)
        enemy_specs.append(EnemySpec(x=x, y=y, axis=axis, dir_sign=1))

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "max_moves": int(max_moves),
        "player_start": [player_start[0], player_start[1]],
        "crate_start": [crate_pos[0], crate_pos[1]] if crate_pos is not None else [NO_CRATE, NO_CRATE],
        "key_pos": [key_pos[0], key_pos[1]] if key_pos is not None else [NO_CRATE, NO_CRATE],
        "walls": [[x, y] for x, y in sorted(walls)],
        "exit_cells": [[x, y] for x, y in sorted(exit_cells)],
        "gate_cells": [[x, y] for x, y in sorted(gate_cells)],
        "door_cells": [[x, y] for x, y in sorted(door_cells)],
        "plate_cells": [[x, y] for x, y in sorted(plate_cells)],
        "spike_cells": [[x, y] for x, y in sorted(spike_cells)],
        "ice_cells": [[x, y] for x, y in sorted(ice_cells)],
        "tele_cells": [[x, y] for x, y in sorted(teleport_cells)],
        "tele_landing": [[x, y] for x, y in tele_landing],
        "tele_cell_to_pad": [[x, y, idx] for (x, y), idx in sorted(tele_cell_to_pad.items())],
        "enemy_specs": [
            {"x": int(spec.x), "y": int(spec.y), "axis": int(spec.axis), "dir": int(spec.dir_sign)}
            for spec in enemy_specs
        ],
    }


def _deserialize_model(level_or_model) -> dict:
    model = level_or_model
    if isinstance(level_or_model, Level):
        model = level_or_model.get_data("model") or {}

    walls = frozenset((int(x), int(y)) for x, y in (model.get("walls") or []))
    exit_cells = frozenset((int(x), int(y)) for x, y in (model.get("exit_cells") or []))
    gate_cells = frozenset((int(x), int(y)) for x, y in (model.get("gate_cells") or []))
    door_cells = frozenset((int(x), int(y)) for x, y in (model.get("door_cells") or []))
    plate_cells = frozenset((int(x), int(y)) for x, y in (model.get("plate_cells") or []))
    spike_cells = frozenset((int(x), int(y)) for x, y in (model.get("spike_cells") or []))
    ice_cells = frozenset((int(x), int(y)) for x, y in (model.get("ice_cells") or []))
    tele_cells = frozenset((int(x), int(y)) for x, y in (model.get("tele_cells") or []))

    tele_landing_raw = model.get("tele_landing") or []
    tele_landing = tuple((int(item[0]), int(item[1])) for item in tele_landing_raw)

    tele_cell_to_pad: dict[tuple[int, int], int] = {}
    for item in model.get("tele_cell_to_pad") or []:
        tele_cell_to_pad[(int(item[0]), int(item[1]))] = int(item[2])

    enemy_specs: tuple[EnemySpec, ...] = tuple(
        EnemySpec(
            x=int(item.get("x", 0)),
            y=int(item.get("y", 0)),
            axis=int(item.get("axis", 0)),
            dir_sign=int(item.get("dir", 1)) or 1,
        )
        for item in (model.get("enemy_specs") or [])
    )

    player_start_raw = model.get("player_start") or [1, 2]
    crate_start_raw = model.get("crate_start") or [NO_CRATE, NO_CRATE]
    key_pos_raw = model.get("key_pos") or [NO_CRATE, NO_CRATE]

    return {
        "width": int(model.get("width", WIDTH)),
        "height": int(model.get("height", HEIGHT)),
        "max_moves": int(model.get("max_moves", 1)),
        "player_start": (int(player_start_raw[0]), int(player_start_raw[1])),
        "crate_start": (int(crate_start_raw[0]), int(crate_start_raw[1])),
        "key_pos": (int(key_pos_raw[0]), int(key_pos_raw[1])),
        "walls": walls,
        "exit_cells": exit_cells,
        "gate_cells": gate_cells,
        "door_cells": door_cells,
        "plate_cells": plate_cells,
        "spike_cells": spike_cells,
        "ice_cells": ice_cells,
        "tele_cells": tele_cells,
        "tele_landing": tele_landing,
        "tele_cell_to_pad": tele_cell_to_pad,
        "enemy_specs": enemy_specs,
    }


def initial_search_state_from_model(model: dict) -> tuple[int, ...]:
    player_start = model["player_start"]
    crate_start = model["crate_start"]
    key_pos = model["key_pos"]
    key_present = 1 if key_pos[0] >= 0 else 0

    gate_state = GATE_CLOSED if model["gate_cells"] else GATE_OPEN
    door_state = DOOR_CLOSED if model["door_cells"] else DOOR_OPEN

    state = [
        int(player_start[0]),
        int(player_start[1]),
        int(crate_start[0]),
        int(crate_start[1]),
        0,
        key_present,
        gate_state,
        door_state,
        DIR_NONE,
        NO_PENDING_TELEPORT,
        int(model["max_moves"]),
        0,
        0,
    ]

    for enemy in model["enemy_specs"]:
        state.extend([int(enemy.x), int(enemy.y), 1 if int(enemy.dir_sign) >= 0 else -1])
    return tuple(state)


def _enemy_count(model: dict) -> int:
    return len(model["enemy_specs"])


def _enemy_state(state: tuple[int, ...], enemy_idx: int) -> tuple[int, int, int]:
    offset = BASE_STATE_FIELDS + (enemy_idx * 3)
    return int(state[offset]), int(state[offset + 1]), int(state[offset + 2])


def _set_enemy_state(buf: list[int], enemy_idx: int, x: int, y: int, dir_sign: int) -> None:
    offset = BASE_STATE_FIELDS + (enemy_idx * 3)
    buf[offset] = int(x)
    buf[offset + 1] = int(y)
    buf[offset + 2] = 1 if int(dir_sign) >= 0 else -1


def _blocked_by_static(model: dict, x: int, y: int, gate_state: int, door_state: int) -> bool:
    if x < 0 or y < 0 or x >= model["width"] or y >= model["height"]:
        return True
    if (x, y) in model["walls"]:
        return True
    if gate_state in (GATE_CLOSED, GATE_OPENING) and (x, y) in model["gate_cells"]:
        return True
    return bool(door_state in (DOOR_CLOSED, DOOR_OPENING) and (x, y) in model["door_cells"])


def _advance_door_state(door_state: int, plate_active: bool) -> int:
    if plate_active:
        if door_state == DOOR_CLOSED:
            return DOOR_OPENING
        if door_state == DOOR_OPENING:
            return DOOR_OPEN
        if door_state == DOOR_CLOSING:
            return DOOR_OPENING
        return DOOR_OPEN

    if door_state == DOOR_OPEN:
        return DOOR_CLOSING
    if door_state == DOOR_CLOSING:
        return DOOR_CLOSED
    if door_state == DOOR_OPENING:
        return DOOR_CLOSED
    return DOOR_CLOSED


def _attempt_player_move(*, model: dict, state_buf: list[int], move_dir: int, from_auto_slide: bool) -> None:
    if move_dir not in DELTA_BY_DIR:
        if from_auto_slide:
            state_buf[8] = DIR_NONE
        return

    px, py = int(state_buf[0]), int(state_buf[1])
    cx, cy = int(state_buf[2]), int(state_buf[3])
    has_key = int(state_buf[4])
    key_present = int(state_buf[5])
    gate_state = int(state_buf[6])
    door_state = int(state_buf[7])

    dx, dy = DELTA_BY_DIR[move_dir]
    tx, ty = px + dx, py + dy
    moved = False

    blocked = _blocked_by_static(model, tx, ty, gate_state, door_state)
    if blocked:
        if has_key and gate_state == GATE_CLOSED and (tx, ty) in model["gate_cells"]:
            state_buf[6] = GATE_OPENING
            state_buf[4] = 0
        if from_auto_slide:
            state_buf[8] = DIR_NONE
        return

    if (tx, ty) == (cx, cy):
        bx, by = tx + dx, ty + dy
        if (
            _blocked_by_static(model, bx, by, int(state_buf[6]), door_state)
            or (bx, by) == (cx, cy)
            or any((bx, by) == _enemy_state(tuple(state_buf), idx)[:2] for idx in range(_enemy_count(model)))
        ):
            if from_auto_slide:
                state_buf[8] = DIR_NONE
            return
        state_buf[2] = int(bx)
        state_buf[3] = int(by)
        state_buf[0] = int(tx)
        state_buf[1] = int(ty)
        moved = True
    else:
        state_buf[0] = int(tx)
        state_buf[1] = int(ty)
        moved = True

    px, py = int(state_buf[0]), int(state_buf[1])

    key_pos = model["key_pos"]
    if key_present and (px, py) == key_pos:
        state_buf[5] = 0
        state_buf[4] = 1

    tele_idx = model["tele_cell_to_pad"].get((px, py))
    if tele_idx is not None and len(model["tele_landing"]) == 2:
        state_buf[9] = int(1 - tele_idx)
        state_buf[8] = DIR_NONE
        return

    on_ice = (px, py) in model["ice_cells"]
    if moved and on_ice:
        state_buf[8] = int(move_dir)
    elif from_auto_slide:
        state_buf[8] = DIR_NONE
    else:
        if not on_ice:
            state_buf[8] = DIR_NONE


def apply_action_transition(
    model: dict, state: tuple[int, ...], action_id: int
) -> tuple[tuple[int, ...] | None, bool, bool]:
    if state is None:
        return None, False, True

    s = list(state)

    # Advance one-frame opening from the previous step before processing movement.
    if int(s[6]) == GATE_OPENING:
        s[6] = GATE_OPEN

    # Sliding and teleport continuation happen before reading the new intent.
    tele_pending = int(s[9])
    if tele_pending != NO_PENDING_TELEPORT and len(model["tele_landing"]) == 2:
        lx, ly = model["tele_landing"][tele_pending]
        s[0] = int(lx)
        s[1] = int(ly)
        s[8] = DIR_NONE
        s[9] = NO_PENDING_TELEPORT
    elif int(s[8]) != DIR_NONE:
        _attempt_player_move(model=model, state_buf=s, move_dir=int(s[8]), from_auto_slide=True)
    else:
        move_dir = MOVE_BY_ACTION.get(int(action_id), DIR_NONE)
        if move_dir == DIR_NONE:
            s[8] = DIR_NONE
        else:
            _attempt_player_move(model=model, state_buf=s, move_dir=move_dir, from_auto_slide=False)

    s[10] = int(s[10]) - 1
    s[11] = (int(s[11]) + 1) % 4
    s[12] = 1 - int(s[12])

    crate_pos = (int(s[2]), int(s[3]))
    plate_active = crate_pos[0] >= 0 and crate_pos in model["plate_cells"]
    s[7] = _advance_door_state(int(s[7]), plate_active)

    enemy_total = _enemy_count(model)
    old_enemy_positions = [_enemy_state(tuple(s), idx) for idx in range(enemy_total)]
    new_enemy_positions: list[tuple[int, int, int]] = []

    for idx in range(enemy_total):
        ex, ey, edir = old_enemy_positions[idx]
        spec = model["enemy_specs"][idx]
        if int(spec.axis) == 0:
            step_dx, step_dy = int(edir), 0
        else:
            step_dx, step_dy = 0, int(edir)

        occupied_by_others = {pos[:2] for j, pos in enumerate(old_enemy_positions) if j != idx}
        occupied_by_others.update((x, y) for x, y, _ in new_enemy_positions)

        tx, ty = ex + step_dx, ey + step_dy
        blocked_forward = (
            _blocked_by_static(model, tx, ty, int(s[6]), int(s[7]))
            or (tx, ty) == crate_pos
            or (tx, ty) in occupied_by_others
        )

        if blocked_forward:
            edir = -int(edir)
            if int(spec.axis) == 0:
                step_dx, step_dy = int(edir), 0
            else:
                step_dx, step_dy = 0, int(edir)

            tx, ty = ex + step_dx, ey + step_dy
            blocked_reverse = (
                _blocked_by_static(model, tx, ty, int(s[6]), int(s[7]))
                or (tx, ty) == crate_pos
                or (tx, ty) in occupied_by_others
            )
            if blocked_reverse:
                tx, ty = ex, ey

        new_enemy_positions.append((int(tx), int(ty), int(edir)))

    for idx, (ex, ey, edir) in enumerate(new_enemy_positions):
        _set_enemy_state(s, idx, ex, ey, edir)

    px, py = int(s[0]), int(s[1])
    spikes_up = int(s[11]) == 3
    hit_spike = spikes_up and ((px, py) in model["spike_cells"])
    hit_enemy = any((px, py) == (ex, ey) for ex, ey, _ in new_enemy_positions)
    dead = hit_spike or hit_enemy

    won = (px, py) in model["exit_cells"]
    failed = (not won) and (dead or int(s[10]) <= 0)

    if failed:
        return tuple(s), False, True
    return tuple(s), won, False


def _build_level(spec: dict) -> Level:
    name = str(spec["name"])
    max_moves = int(spec["max_moves"])
    layout = [str(row) for row in (spec["layout"] or [])]
    model = _parse_level(layout, max_moves=max_moves)

    walls = {(int(x), int(y)) for x, y in model["walls"]}
    {(int(x), int(y)) for x, y in model["spike_cells"]}
    ice = {(int(x), int(y)) for x, y in model["ice_cells"]}

    sprites: list[Sprite] = [
        Sprite(
            pixels=_solid(WIDTH, HEIGHT, COLOR_FLOOR),
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        ),
        Sprite(pixels=_overlay(walls, COLOR_WALL), name="walls", x=0, y=0, layer=1, tags=["wall"], collidable=True),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="teleporter",
            x=0,
            y=0,
            layer=2,
            tags=["teleporter"],
            collidable=False,
        ),
        Sprite(pixels=_overlay(ice, COLOR_ICE), name="ice", x=0, y=0, layer=2, tags=["ice"], collidable=False),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="spikes",
            x=0,
            y=0,
            layer=3,
            tags=["spikes"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="plate",
            x=0,
            y=0,
            layer=3,
            tags=["plate"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="exit",
            x=0,
            y=0,
            layer=4,
            tags=["exit"],
            collidable=False,
        ),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="gate",
            x=0,
            y=0,
            layer=4,
            tags=["gate"],
            collidable=True,
        ),
        Sprite(
            pixels=np.full((HEIGHT, WIDTH), -1, dtype=np.int8),
            name="door",
            x=0,
            y=0,
            layer=4,
            tags=["door"],
            collidable=True,
        ),
        Sprite(
            pixels=_solid(WIDTH, 1, COLOR_TIMEBAR),
            name="timebar",
            x=0,
            y=0,
            layer=6,
            tags=["hud", "timer"],
            collidable=False,
        ),
    ]

    key_pos = tuple(model["key_pos"])
    if key_pos[0] >= 0:
        sprites.append(
            Sprite(
                pixels=[[COLOR_EXIT]],
                name="key",
                x=int(key_pos[0]),
                y=int(key_pos[1]),
                layer=5,
                tags=["key"],
                collidable=False,
            )
        )

    crate_pos = tuple(model["crate_start"])
    if crate_pos[0] >= 0:
        sprites.append(
            Sprite(
                pixels=[[COLOR_CRATE]],
                name="crate",
                x=int(crate_pos[0]),
                y=int(crate_pos[1]),
                layer=5,
                tags=["crate"],
                collidable=True,
            )
        )

    for idx, enemy in enumerate(model["enemy_specs"]):
        sprites.append(
            Sprite(
                pixels=[[COLOR_ENEMY]],
                name=f"enemy_{idx}",
                x=int(enemy["x"]),
                y=int(enemy["y"]),
                layer=5,
                tags=["enemy"],
                collidable=True,
            )
        )

    player_start = tuple(model["player_start"])
    sprites.append(
        Sprite(
            pixels=[[COLOR_PLAYER]],
            name="player",
            x=int(player_start[0]),
            y=int(player_start[1]),
            layer=7,
            tags=["player"],
            collidable=True,
        )
    )

    return Level(name=name, grid_size=(WIDTH, HEIGHT), sprites=sprites, data={"model": model, "max_moves": max_moves})


class ShortestRouteChallenge(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        camera = Camera(width=WIDTH, height=HEIGHT, background=COLOR_FLOOR)
        super().__init__(
            game_id="shortest_route_challenge",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

        self._model: dict | None = None
        self._state_tuple: tuple[int, ...] | None = None
        self._mode = "playing"
        self._route_score = 0

        self._player: Sprite | None = None
        self._timebar: Sprite | None = None
        self._exit: Sprite | None = None
        self._spikes: Sprite | None = None
        self._plate: Sprite | None = None
        self._door: Sprite | None = None
        self._gate: Sprite | None = None
        self._teleporter: Sprite | None = None
        self._key: Sprite | None = None
        self._crate: Sprite | None = None
        self._enemies: list[Sprite] = []

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._state_tuple = initial_search_state_from_model(self._model)
        self._mode = "playing"

        self._player = next(iter(level.get_sprites_by_name("player")), None)
        self._timebar = next(iter(level.get_sprites_by_name("timebar")), None)
        self._exit = next(iter(level.get_sprites_by_name("exit")), None)
        self._spikes = next(iter(level.get_sprites_by_name("spikes")), None)
        self._plate = next(iter(level.get_sprites_by_name("plate")), None)
        self._door = next(iter(level.get_sprites_by_name("door")), None)
        self._gate = next(iter(level.get_sprites_by_name("gate")), None)
        self._teleporter = next(iter(level.get_sprites_by_name("teleporter")), None)
        self._key = next(iter(level.get_sprites_by_name("key")), None)
        self._crate = next(iter(level.get_sprites_by_name("crate")), None)
        self._enemies = sorted(level.get_sprites_by_tag("enemy"), key=lambda s: s.name)

        self._render_from_state(dead=False)

    def _render_from_state(self, *, dead: bool) -> None:
        if self._model is None or self._state_tuple is None:
            return

        state = self._state_tuple
        px, py = int(state[0]), int(state[1])
        cx, cy = int(state[2]), int(state[3])
        has_key = int(state[4]) == 1
        key_present = int(state[5]) == 1
        gate_state = int(state[6])
        door_state = int(state[7])
        sliding_dir = int(state[8])
        tele_pending = int(state[9])
        moves_left = int(state[10])
        tick_mod4 = int(state[11])
        anim_phase = int(state[12])

        enemy_positions = [_enemy_state(state, idx) for idx in range(_enemy_count(self._model))]

        if self._timebar is not None:
            if self._mode == "fail":
                row = [COLOR_EMPTY for _ in range(WIDTH)]
            else:
                fill = max(0, min(WIDTH, moves_left))
                flash = moves_left <= 3 and anim_phase == 1
                fill_color = COLOR_ACCENT if flash else COLOR_TIMEBAR
                row = [fill_color if x < fill else COLOR_EMPTY for x in range(WIDTH)]
            self._timebar.pixels = np.array([row], dtype=np.int8)

        if self._exit is not None:
            self._exit.pixels = _overlay_checker(set(self._model["exit_cells"]), COLOR_EXIT, COLOR_ACCENT, anim_phase)

        if self._spikes is not None:
            spike_color = COLOR_HAZARD if tick_mod4 == 3 else COLOR_EMPTY
            self._spikes.pixels = _overlay(set(self._model["spike_cells"]), spike_color)

        if self._teleporter is not None:
            self._teleporter.pixels = _overlay_checker(
                set(self._model["tele_cells"]), COLOR_TELEPORT, COLOR_EMPTY, anim_phase
            )

        if self._plate is not None:
            plate_active = (cx, cy) in self._model["plate_cells"]
            plate_color = COLOR_PLATE_ACTIVE if plate_active else COLOR_PLATE_INACTIVE
            self._plate.pixels = _overlay(set(self._model["plate_cells"]), plate_color)

        if self._gate is not None:
            gate_pixels = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
            if gate_state in (GATE_CLOSED, GATE_OPENING):
                gate_color = COLOR_CLOSED if gate_state == GATE_CLOSED else COLOR_ACCENT
                for gx, gy in self._model["gate_cells"]:
                    gate_pixels[gy, gx] = gate_color
            self._gate.pixels = gate_pixels

        if self._door is not None:
            door_pixels = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
            if door_state in (DOOR_CLOSED, DOOR_OPENING, DOOR_CLOSING):
                door_color = COLOR_CLOSED if door_state == DOOR_CLOSED else COLOR_ACCENT
                for dx, dy in self._model["door_cells"]:
                    door_pixels[dy, dx] = door_color
            self._door.pixels = door_pixels

        if self._key is not None:
            if key_present:
                kx, ky = self._model["key_pos"]
                self._key.set_position(int(kx), int(ky))
                self._key.pixels = np.array([[COLOR_ACCENT if anim_phase else COLOR_EXIT]], dtype=np.int8)
            else:
                self._key.set_position(-10, -10)

        if self._crate is not None:
            if cx >= 0:
                self._crate.set_position(int(cx), int(cy))
            else:
                self._crate.set_position(-10, -10)

        for idx, enemy_sprite in enumerate(self._enemies):
            if idx < len(enemy_positions):
                ex, ey, _ = enemy_positions[idx]
                enemy_sprite.set_position(int(ex), int(ey))
            else:
                enemy_sprite.set_position(-10, -10)

        if self._player is not None:
            self._player.set_position(int(px), int(py))
            if dead:
                color = COLOR_HAZARD
            elif tele_pending != NO_PENDING_TELEPORT or sliding_dir != DIR_NONE:
                color = COLOR_TELEPORT
            elif has_key:
                color = COLOR_PLAYER_KEY
            else:
                color = COLOR_PLAYER
            self._player.pixels = np.array([[color]], dtype=np.int8)

    @staticmethod
    def _action_id(action_obj) -> int:
        return int(getattr(action_obj, "value", action_obj))

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

        action_id = self._action_id(self.action.id)

        if self._mode != "playing" or self._model is None or self._state_tuple is None:
            self.complete_action()
            return

        next_state, won, failed = apply_action_transition(self._model, self._state_tuple, action_id)
        if next_state is None:
            self._mode = "fail"
            self._render_from_state(dead=True)
            self.lose()
            self.complete_action()
            return

        self._state_tuple = next_state

        if won:
            self._route_score += 1
            self.next_level()
            self.complete_action()
            return

        if failed:
            self._mode = "fail"
            self._render_from_state(dead=True)
            self.lose()
            self.complete_action()
            return

        self._render_from_state(dead=False)
        self.complete_action()
