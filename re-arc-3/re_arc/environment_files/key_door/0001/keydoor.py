from __future__ import annotations

from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "key_door-0001"
VARIANT = "0001"

GRID_WIDTH = 31
GRID_HEIGHT = 19
PLAYER_SIZE = (2, 2)
KEY_SIZE = (2, 2)
DOOR_SIZE = (2, 3)
GUARD_SIZE = (2, 2)
FREEZE_STEPS = 6

# Colors
COLOR_VOID = 0
COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_TIMEBAR_FILL = 3
COLOR_TIMEBAR_EMPTY = 4
COLOR_PLAYER_FILL = 5
COLOR_PLAYER_OUTLINE_NO_KEY = 6
COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY = 7
COLOR_DOOR_LOCKED = 8
COLOR_DOOR_OPEN = 9
COLOR_DOOR_OPENING = 10
COLOR_EXIT_LOCKED = 11
COLOR_EXIT_OPEN = 12
COLOR_HAZARD = 13
COLOR_GUARD = 14
COLOR_HIGHLIGHT = 15

DOOR_PHASE_LOCKED = 0
DOOR_PHASE_OPENING = 1
DOOR_PHASE_OPEN = 2

LASER_OFF = 0
LASER_WARN = 1
LASER_ON = 2

FREEZE_NONE = 0
FREEZE_FAIL = 1
FREEZE_WIN = 2

MOVE_DELTAS = {
    1: (0, -1),  # up
    2: (0, 1),  # down
    3: (-1, 0),  # left
    4: (1, 0),  # right
    5: (0, 0),  # wait/space
}

LEVEL_LAYOUTS = [
    {
        "name": "Level 1",
        "time_limit": 140,
        "rows": [
            "===============================",
            "###############################",
            "#...................##........#",
            "#...................##........#",
            "#...................##........#",
            "#.....kk............##........#",
            "#.....kk............##........#",
            "#...................##........#",
            "#...................XX........#",
            "#...................XX........#",
            "#...................XX........#",
            "#...................##........#",
            "#..PP...............##........#",
            "#..PP...............##........#",
            "#...................##........#",
            "#...................##........#",
            "#...................##........#",
            "#...................##........#",
            "###############################",
        ],
    },
    {
        "name": "Level 2",
        "time_limit": 180,
        "rows": [
            "===============================",
            "###############################",
            "#..............##.............#",
            "#.XX...........##.......kk....#",
            "#.XX...........##.......kk....#",
            "#.XX...........##.............#",
            "#..............##.............#",
            "#..............DD.............#",
            "#..............DD.............#",
            "#..............DD.............#",
            "#....PP........##.............#",
            "#....PP........##.............#",
            "#..............##.............#",
            "#..............##.............#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#..............##.............#",
            "###############################",
        ],
    },
    {
        "name": "Level 3",
        "time_limit": 200,
        "rows": [
            "===============================",
            "###############################",
            "##..#..........##.............#",
            "##XX#..........##.............#",
            "##XX#..........##.......kk....#",
            "##XX#..........##.......kk....#",
            "##DD#..........##.............#",
            "##DD#..........DD.............#",
            "##DD#..........DD.............#",
            "##..#..........DD.............#",
            "##..#..........##.............#",
            "#..............##.............#",
            "#.....PP.......##.............#",
            "#.....PP.......##.............#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#..............##.............#",
            "###############################",
        ],
    },
    {
        "name": "Level 4",
        "time_limit": 220,
        "rows": [
            "===============================",
            "###############################",
            "##..#..........##.............#",
            "##XX#..........##.............#",
            "##XX#..........##.......kk....#",
            "##XX#..........##.......kk....#",
            "##DD#..........##.............#",
            "##DD#..........DD.............#",
            "##DD#..........DD.............#",
            "##..#..........DD.............#",
            "##..#..........##.............#",
            "#..............##.............#",
            "#.....PP.......##.............#",
            "#.....PP.......##.............#",
            "#.............................#",
            "#.............................#",
            "#.............................#",
            "#..............##.............#",
            "###############################",
        ],
    },
    {
        "name": "Level 5",
        "time_limit": 260,
        "rows": [
            "===============================",
            "###############################",
            "##..#..........##......####...#",
            "##XX#..........##......#..#...#",
            "##XX#..........##......#kk#...#",
            "##XX#..........##......#kk#...#",
            "##DD#..........##......#..#...#",
            "##DD#..........DD......#~.#...#",
            "##DD#..........DD......#~.#...#",
            "##..#..........DD......#~.#...#",
            "##..#..........##......#..#...#",
            "#..............##......#..#...#",
            "#.....PP.......##......#..#...#",
            "#.....PP.......##.............#",
            "#...............~.............#",
            "#...............~.............#",
            "#...............~.............#",
            "#..............##.............#",
            "###############################",
        ],
    },
    {
        "name": "Level 6",
        "time_limit": 320,
        "guard_waypoints": [(9, 10), (12, 10), (12, 12), (9, 12)],
        "rows": [
            "===============================",
            "###############################",
            "##..#..........##......####...#",
            "##XX#..........##......#..#...#",
            "##XX#..........##......#kk#...#",
            "##XX#..........##......#kk#...#",
            "##DD#..........##......#..#...#",
            "##DD#..........DD......#~.#...#",
            "##DD#..........DD......#~.#...#",
            "##~.#..........DD......#~.#...#",
            "##~.#....GG....##......#..#...#",
            "#........GG....##......#..#...#",
            "#.....PP.......##......#..#...#",
            "#.....PP.......##.............#",
            "#...............~.............#",
            "#...............~.............#",
            "#...............~.............#",
            "#..............##.............#",
            "###############################",
        ],
    },
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _cells_for_rect(anchor: tuple[int, int], size: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ax, ay = anchor
    w, h = size
    return tuple((ax + dx, ay + dy) for dy in range(h) for dx in range(w))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _componentize(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    remaining = set(cells)
    groups: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        root = next(iter(remaining))
        queue = deque([root])
        remaining.remove(root)
        group = [root]
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt not in remaining:
                    continue
                remaining.remove(nxt)
                queue.append(nxt)
                group.append(nxt)
        groups.append(tuple(sorted(group)))
    groups.sort(key=lambda g: (min(y for _, y in g), min(x for x, _ in g)))
    return groups


def _parse_rect_anchors(rows: list[str], token: str, width: int, height: int) -> list[tuple[int, int]]:
    anchors: list[tuple[int, int]] = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell != token:
                continue
            if x > 0 and row[x - 1] == token:
                continue
            if y > 0 and rows[y - 1][x] == token:
                continue
            ok = True
            for dy in range(height):
                yy = y + dy
                if yy >= len(rows):
                    ok = False
                    break
                for dx in range(width):
                    xx = x + dx
                    if xx >= len(row) or rows[yy][xx] != token:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                anchors.append((x, y))
    return anchors


def _laser_initial_phases(level_index: int, groups: list[tuple[tuple[int, int], ...]]) -> list[int]:
    if not groups:
        return []
    if level_index < 4:
        return [LASER_OFF for _ in groups]

    centroids: list[tuple[int, float, float]] = []
    for idx, group in enumerate(groups):
        avg_x = sum(x for x, _ in group) / float(len(group))
        avg_y = sum(y for _, y in group) / float(len(group))
        centroids.append((idx, avg_x, avg_y))

    phases = [LASER_OFF for _ in groups]

    if level_index == 4:
        # Level 5: bottom passage starts WARN, key-corridor starts OFF.
        bottom_idx = max(centroids, key=lambda item: item[2])[0]
        phases[bottom_idx] = LASER_WARN
        return phases

    # Level 6: bottom WARN, key-corridor OFF, exit-corridor WARN.
    bottom_idx = max(centroids, key=lambda item: item[2])[0]
    phases[bottom_idx] = LASER_WARN

    remaining = [item for item in centroids if item[0] != bottom_idx]
    if remaining:
        key_idx = max(remaining, key=lambda item: item[1])[0]
        phases[key_idx] = LASER_OFF
        for idx, _, _ in remaining:
            if idx != key_idx:
                phases[idx] = LASER_WARN
    return phases


def _parse_level_model(level_index: int, spec: dict) -> dict:
    rows = list(spec["rows"])
    if len(rows) != GRID_HEIGHT:
        raise ValueError(f"{spec['name']}: expected {GRID_HEIGHT} rows, got {len(rows)}")
    if any(len(row) != GRID_WIDTH for row in rows):
        raise ValueError(f"{spec['name']}: expected width={GRID_WIDTH}")

    wall_cells: set[tuple[int, int]] = set()
    spike_cells: set[tuple[int, int]] = set()
    laser_cells: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "#":
                wall_cells.add((x, y))
            elif cell == "^":
                spike_cells.add((x, y))
            elif cell == "~":
                laser_cells.add((x, y))

    player_anchors = _parse_rect_anchors(rows, "P", PLAYER_SIZE[0], PLAYER_SIZE[1])
    key_anchors = _parse_rect_anchors(rows, "k", KEY_SIZE[0], KEY_SIZE[1])
    door_anchors = _parse_rect_anchors(rows, "D", DOOR_SIZE[0], DOOR_SIZE[1])
    exit_anchors = _parse_rect_anchors(rows, "X", DOOR_SIZE[0], DOOR_SIZE[1])
    guard_anchors = _parse_rect_anchors(rows, "G", GUARD_SIZE[0], GUARD_SIZE[1])

    if len(player_anchors) != 1:
        raise ValueError(f"{spec['name']}: expected exactly one player anchor")
    if len(key_anchors) != 1:
        raise ValueError(f"{spec['name']}: expected exactly one key anchor")
    if len(exit_anchors) != 1:
        raise ValueError(f"{spec['name']}: expected exactly one exit anchor")

    if guard_anchors and "guard_waypoints" not in spec:
        raise ValueError(f"{spec['name']}: guard exists but no waypoint list was provided")

    laser_groups = _componentize(laser_cells)
    laser_initial = _laser_initial_phases(level_index, laser_groups)

    player_anchor = player_anchors[0]
    key_anchor = key_anchors[0]
    exit_anchor = exit_anchors[0]

    guard_anchor = guard_anchors[0] if guard_anchors else None
    guard_waypoints = [tuple(pair) for pair in (spec.get("guard_waypoints") or [])]

    if guard_anchor is not None and guard_waypoints:
        target_idx = 1 if tuple(guard_anchor) == tuple(guard_waypoints[0]) and len(guard_waypoints) > 1 else 0
    else:
        target_idx = 0

    return {
        "name": spec["name"],
        "time_limit": int(spec["time_limit"]),
        "wall_cells": sorted((int(x), int(y)) for x, y in wall_cells),
        "spike_cells": sorted((int(x), int(y)) for x, y in spike_cells),
        "player_start": [int(player_anchor[0]), int(player_anchor[1])],
        "key_anchor": [int(key_anchor[0]), int(key_anchor[1])],
        "door_anchors": [[int(x), int(y)] for x, y in sorted(door_anchors)],
        "exit_anchor": [int(exit_anchor[0]), int(exit_anchor[1])],
        "laser_groups": [[[int(x), int(y)] for x, y in group] for group in laser_groups],
        "laser_initial": [int(value) for value in laser_initial],
        "guard_start": [int(guard_anchor[0]), int(guard_anchor[1])] if guard_anchor is not None else None,
        "guard_waypoints": [[int(x), int(y)] for x, y in guard_waypoints],
        "guard_target_idx": int(target_idx),
    }


def _deserialize_model(level: Level) -> dict:
    model = level.get_data("model")
    if not isinstance(model, dict):
        raise RuntimeError("key_door level data is missing model payload")
    return model


def _build_level(level_index: int, spec: dict) -> Level:
    model = _parse_level_model(level_index, spec)
    scene = Sprite(
        pixels=_solid(GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR), name="scene", x=0, y=0, layer=0, collidable=False
    )
    return Level(name=str(spec["name"]), grid_size=(GRID_WIDTH, GRID_HEIGHT), sprites=[scene], data={"model": model})


def _all_player_cells(px: int, py: int) -> tuple[tuple[int, int], ...]:
    return _cells_for_rect((px, py), PLAYER_SIZE)


def _door_cells(model: dict) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for anchor in model["door_anchors"]:
        cells.extend(_cells_for_rect((int(anchor[0]), int(anchor[1])), DOOR_SIZE))
    cells.extend(_cells_for_rect((int(model["exit_anchor"][0]), int(model["exit_anchor"][1])), DOOR_SIZE))
    return tuple(cells)


def _runtime_cache(model: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    wall_set = model.get("_wall_set")
    if wall_set is None:
        wall_set = {(int(x), int(y)) for x, y in model["wall_cells"]}
        model["_wall_set"] = wall_set

    door_set = model.get("_door_cells_set")
    if door_set is None:
        door_set = {(int(x), int(y)) for x, y in _door_cells(model)}
        model["_door_cells_set"] = door_set

    return wall_set, door_set


def _laser_cells_active(model: dict, laser_phases: tuple[int, ...]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, group in enumerate(model["laser_groups"]):
        if idx >= len(laser_phases) or int(laser_phases[idx]) != LASER_ON:
            continue
        for x, y in group:
            out.add((int(x), int(y)))
    return out


def _player_overlaps_hazard(model: dict, px: int, py: int, laser_phases: tuple[int, ...]) -> bool:
    fill_cell = (int(px) + 1, int(py) + 1)

    for x, y in model["spike_cells"]:
        if (int(x), int(y)) == fill_cell:
            return True

    for cell in _laser_cells_active(model, laser_phases):
        if cell == fill_cell:
            return True

    guard_start = model.get("guard_start")
    if guard_start is not None:
        gx = int(model["guard_current"][0])
        gy = int(model["guard_current"][1])
        if _rects_overlap((px, py, PLAYER_SIZE[0], PLAYER_SIZE[1]), (gx, gy, GUARD_SIZE[0], GUARD_SIZE[1])):
            return True

    return False


def _advance_guard(model: dict) -> None:
    guard_start = model.get("guard_start")
    if guard_start is None:
        return

    waypoints = [tuple(map(int, pair)) for pair in (model.get("guard_waypoints") or [])]
    if not waypoints:
        return

    gx = int(model["guard_current"][0])
    gy = int(model["guard_current"][1])
    target_idx = int(model.get("guard_target_idx") or 0) % len(waypoints)
    tx, ty = waypoints[target_idx]

    dx = 0
    dy = 0
    if gx != tx:
        dx = 1 if tx > gx else -1
    elif gy != ty:
        dy = 1 if ty > gy else -1

    gx += dx
    gy += dy

    if gx == tx and gy == ty:
        target_idx = (target_idx + 1) % len(waypoints)

    model["guard_current"] = [int(gx), int(gy)]
    model["guard_target_idx"] = int(target_idx)


def _passable(model: dict, x: int, y: int, door_phase: int) -> bool:
    wall_set, door_set = _runtime_cache(model)

    if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
        return False

    if (x, y) in wall_set:
        return False

    return not (int(door_phase) != DOOR_PHASE_OPEN and (x, y) in door_set)


def _apply_move(model: dict, state: tuple, action_id: int) -> tuple[int, int]:
    px, py = int(state[0]), int(state[1])
    dx, dy = MOVE_DELTAS.get(int(action_id), (0, 0))
    tx = px + dx
    ty = py + dy

    door_phase = int(state[3])

    for cx, cy in _all_player_cells(tx, ty):
        if not _passable(model, cx, cy, door_phase):
            return px, py

    return tx, ty


def _state_with_pre_step_doors(state: tuple) -> tuple:
    (px, py, has_key, door_phase, key_collected, laser_phases, guard_x, guard_y, guard_target_idx, time_left) = state
    if int(door_phase) == DOOR_PHASE_OPENING:
        door_phase = DOOR_PHASE_OPEN
    return (
        int(px),
        int(py),
        int(has_key),
        int(door_phase),
        int(key_collected),
        tuple(int(v) for v in laser_phases),
        int(guard_x),
        int(guard_y),
        int(guard_target_idx),
        int(time_left),
    )


def initial_search_state_from_model(model: dict) -> tuple:
    guard_start = model.get("guard_start")
    if guard_start is None:
        guard_x, guard_y = -100, -100
    else:
        guard_x, guard_y = int(guard_start[0]), int(guard_start[1])

    laser_initial = tuple(int(v) for v in model.get("laser_initial") or [])
    return (
        int(model["player_start"][0]),
        int(model["player_start"][1]),
        0,
        DOOR_PHASE_LOCKED,
        0,
        laser_initial,
        int(guard_x),
        int(guard_y),
        int(model.get("guard_target_idx") or 0),
        int(model["time_limit"]),
    )


def is_win_state(model: dict, state: tuple) -> bool:
    px, py = int(state[0]), int(state[1])
    door_phase = int(state[3])
    if door_phase != DOOR_PHASE_OPEN:
        return False
    exit_x = int(model["exit_anchor"][0])
    exit_y = int(model["exit_anchor"][1])
    return _rects_overlap((px, py, PLAYER_SIZE[0], PLAYER_SIZE[1]), (exit_x, exit_y, DOOR_SIZE[0], DOOR_SIZE[1]))


def apply_action_transition(model: dict, state: tuple, action_id: int) -> tuple[tuple | None, bool]:
    state = _state_with_pre_step_doors(state)
    (px, py, has_key, door_phase, key_collected, laser_phases, guard_x, guard_y, guard_target_idx, time_left) = state

    # 1) Consume action.
    action_id = int(action_id)
    if action_id in MOVE_DELTAS:
        work_state = (
            int(px),
            int(py),
            int(has_key),
            int(door_phase),
            int(key_collected),
            tuple(int(v) for v in laser_phases),
            int(guard_x),
            int(guard_y),
            int(guard_target_idx),
            int(time_left),
        )
        px, py = _apply_move(model, work_state, action_id)

    # 2) Pickups.
    just_collected = False
    if not int(key_collected):
        key_x = int(model["key_anchor"][0])
        key_y = int(model["key_anchor"][1])
        if _rects_overlap((px, py, PLAYER_SIZE[0], PLAYER_SIZE[1]), (key_x, key_y, KEY_SIZE[0], KEY_SIZE[1])):
            has_key = 1
            key_collected = 1
            just_collected = True

    # 3) Doors update.
    if just_collected and int(door_phase) == DOOR_PHASE_LOCKED:
        door_phase = DOOR_PHASE_OPENING

    # 4) Hazards/NPC update.
    next_laser = []
    for value in laser_phases:
        next_laser.append((int(value) + 1) % 3)
    laser_phases = tuple(next_laser)

    if model.get("guard_start") is not None:
        temp_model = dict(model)
        temp_model["guard_current"] = [int(guard_x), int(guard_y)]
        temp_model["guard_target_idx"] = int(guard_target_idx)
        _advance_guard(temp_model)
        guard_x = int(temp_model["guard_current"][0])
        guard_y = int(temp_model["guard_current"][1])
        guard_target_idx = int(temp_model["guard_target_idx"])

    # 5) Resolve outcomes.
    hazard_model = dict(model)
    hazard_model["guard_current"] = [int(guard_x), int(guard_y)]
    if _player_overlaps_hazard(hazard_model, px, py, laser_phases):
        return None, False

    won = False
    if int(door_phase) == DOOR_PHASE_OPEN:
        exit_x = int(model["exit_anchor"][0])
        exit_y = int(model["exit_anchor"][1])
        if _rects_overlap((px, py, PLAYER_SIZE[0], PLAYER_SIZE[1]), (exit_x, exit_y, DOOR_SIZE[0], DOOR_SIZE[1])):
            won = True

    # 6) Timer.
    time_left = int(time_left) - 1
    if time_left <= 0 and not won:
        return None, False

    next_state = (
        int(px),
        int(py),
        int(has_key),
        int(door_phase),
        int(key_collected),
        tuple(int(v) for v in laser_phases),
        int(guard_x),
        int(guard_y),
        int(guard_target_idx),
        int(time_left),
    )
    return next_state, bool(won)


class KeyDoor(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(idx, spec) for idx, spec in enumerate(LEVEL_LAYOUTS)]
        camera = Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, 5, 5, [])
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._scene: Sprite | None = None
        self._model: dict | None = None

        self._player_x = 0
        self._player_y = 0
        self._has_key = False
        self._key_collected = False
        self._door_phase = DOOR_PHASE_LOCKED
        self._laser_phases: list[int] = []
        self._guard_x = -100
        self._guard_y = -100
        self._guard_target_idx = 0
        self._time_limit = 1
        self._time_left = 1

        self._freeze_mode = FREEZE_NONE
        self._freeze_steps_left = 0
        self._anim_tick = 0

        self._wall_cells: set[tuple[int, int]] = set()
        self._spike_cells: set[tuple[int, int]] = set()
        self._door_cells_cache: set[tuple[int, int]] = set()

    def on_set_level(self, level: Level) -> None:
        self._scene = level.get_sprites_by_name("scene")[0]
        self._model = _deserialize_model(level)

        self._player_x = int(self._model["player_start"][0])
        self._player_y = int(self._model["player_start"][1])
        self._has_key = False
        self._key_collected = False
        self._door_phase = DOOR_PHASE_LOCKED
        self._laser_phases = [int(v) for v in (self._model.get("laser_initial") or [])]
        self._time_limit = int(self._model["time_limit"])
        self._time_left = int(self._time_limit)

        guard_start = self._model.get("guard_start")
        if guard_start is None:
            self._guard_x = -100
            self._guard_y = -100
            self._guard_target_idx = 0
        else:
            self._guard_x = int(guard_start[0])
            self._guard_y = int(guard_start[1])
            self._guard_target_idx = int(self._model.get("guard_target_idx") or 0)

        self._freeze_mode = FREEZE_NONE
        self._freeze_steps_left = 0
        self._anim_tick = 0

        self._wall_cells = {(int(x), int(y)) for x, y in self._model["wall_cells"]}
        self._spike_cells = {(int(x), int(y)) for x, y in self._model["spike_cells"]}
        self._door_cells_cache = {(int(x), int(y)) for x, y in _door_cells(self._model)}

        self._render_scene()

    def _player_cells(self) -> tuple[tuple[int, int], ...]:
        return _all_player_cells(self._player_x, self._player_y)

    def _laser_on_cells(self) -> set[tuple[int, int]]:
        if self._model is None:
            return set()
        out: set[tuple[int, int]] = set()
        for idx, group in enumerate(self._model["laser_groups"]):
            phase = int(self._laser_phases[idx]) if idx < len(self._laser_phases) else LASER_OFF
            if phase != LASER_ON:
                continue
            for x, y in group:
                out.add((int(x), int(y)))
        return out

    def _guard_overlap_player(self) -> bool:
        if self._model is None or self._model.get("guard_start") is None:
            return False
        return _rects_overlap(
            (self._player_x, self._player_y, PLAYER_SIZE[0], PLAYER_SIZE[1]),
            (self._guard_x, self._guard_y, GUARD_SIZE[0], GUARD_SIZE[1]),
        )

    def _trigger_fail(self) -> None:
        self._freeze_mode = FREEZE_FAIL
        self._freeze_steps_left = FREEZE_STEPS

    def _trigger_win(self) -> None:
        self._freeze_mode = FREEZE_WIN
        self._freeze_steps_left = FREEZE_STEPS

    def _passable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
            return False
        if (x, y) in self._wall_cells:
            return False
        return not (self._door_phase != DOOR_PHASE_OPEN and (x, y) in self._door_cells_cache)

    def _try_move(self, dx: int, dy: int) -> None:
        tx = self._player_x + int(dx)
        ty = self._player_y + int(dy)
        for cx, cy in _all_player_cells(tx, ty):
            if not self._passable(cx, cy):
                return
        self._player_x = tx
        self._player_y = ty

    def _collect_key_if_needed(self) -> bool:
        if self._model is None or self._key_collected:
            return False
        key_x = int(self._model["key_anchor"][0])
        key_y = int(self._model["key_anchor"][1])
        if _rects_overlap(
            (self._player_x, self._player_y, PLAYER_SIZE[0], PLAYER_SIZE[1]), (key_x, key_y, KEY_SIZE[0], KEY_SIZE[1])
        ):
            self._key_collected = True
            self._has_key = True
            return True
        return False

    def _advance_hazards(self) -> None:
        self._laser_phases = [((int(value) + 1) % 3) for value in self._laser_phases]

        if self._model is None or self._model.get("guard_start") is None:
            return

        waypoints = [tuple(map(int, pair)) for pair in (self._model.get("guard_waypoints") or [])]
        if not waypoints:
            return

        target_idx = int(self._guard_target_idx) % len(waypoints)
        tx, ty = waypoints[target_idx]

        if self._guard_x != tx:
            self._guard_x += 1 if tx > self._guard_x else -1
        elif self._guard_y != ty:
            self._guard_y += 1 if ty > self._guard_y else -1

        if self._guard_x == tx and self._guard_y == ty:
            self._guard_target_idx = (target_idx + 1) % len(waypoints)

    def _check_fail_hazards(self) -> bool:
        fill_cell = (self._player_x + 1, self._player_y + 1)
        if fill_cell in self._spike_cells:
            return True
        if fill_cell in self._laser_on_cells():
            return True

        return self._guard_overlap_player()

    def _check_open_exit_overlap(self) -> bool:
        if self._model is None or self._door_phase != DOOR_PHASE_OPEN:
            return False
        exit_x = int(self._model["exit_anchor"][0])
        exit_y = int(self._model["exit_anchor"][1])
        return _rects_overlap(
            (self._player_x, self._player_y, PLAYER_SIZE[0], PLAYER_SIZE[1]),
            (exit_x, exit_y, DOOR_SIZE[0], DOOR_SIZE[1]),
        )

    def _render_scene(self) -> None:
        if self._scene is None or self._model is None:
            return

        grid = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_FLOOR, dtype=np.int8)

        # Static terrain.
        for x, y in self._wall_cells:
            grid[y, x] = COLOR_WALL
        for x, y in self._spike_cells:
            grid[y, x] = COLOR_HAZARD

        # Doors and exit.
        for anchor in self._model["door_anchors"]:
            ax, ay = int(anchor[0]), int(anchor[1])
            if self._door_phase == DOOR_PHASE_LOCKED:
                door_pixels = (
                    (COLOR_DOOR_LOCKED, COLOR_DOOR_LOCKED),
                    (COLOR_DOOR_LOCKED, COLOR_DOOR_LOCKED),
                    (COLOR_DOOR_LOCKED, COLOR_DOOR_LOCKED),
                )
            elif self._door_phase == DOOR_PHASE_OPENING:
                door_pixels = (
                    (COLOR_DOOR_LOCKED, COLOR_DOOR_OPENING),
                    (COLOR_DOOR_OPENING, COLOR_DOOR_LOCKED),
                    (COLOR_DOOR_LOCKED, COLOR_DOOR_OPENING),
                )
            else:
                door_pixels = (
                    (COLOR_DOOR_OPEN, COLOR_DOOR_OPEN),
                    (COLOR_FLOOR, COLOR_FLOOR),
                    (COLOR_DOOR_OPEN, COLOR_DOOR_OPEN),
                )
            for dy in range(DOOR_SIZE[1]):
                for dx in range(DOOR_SIZE[0]):
                    grid[ay + dy, ax + dx] = int(door_pixels[dy][dx])

        exit_x = int(self._model["exit_anchor"][0])
        exit_y = int(self._model["exit_anchor"][1])
        if self._freeze_mode == FREEZE_WIN:
            flash_color = COLOR_EXIT_OPEN if (self._anim_tick % 2 == 0) else COLOR_HIGHLIGHT
            exit_pixels = ((flash_color, flash_color), (COLOR_FLOOR, COLOR_FLOOR), (flash_color, flash_color))
        elif self._door_phase == DOOR_PHASE_LOCKED:
            exit_pixels = (
                (COLOR_EXIT_LOCKED, COLOR_EXIT_LOCKED),
                (COLOR_EXIT_LOCKED, COLOR_EXIT_LOCKED),
                (COLOR_EXIT_LOCKED, COLOR_EXIT_LOCKED),
            )
        elif self._door_phase == DOOR_PHASE_OPENING:
            exit_pixels = (
                (COLOR_EXIT_LOCKED, COLOR_HIGHLIGHT),
                (COLOR_HIGHLIGHT, COLOR_EXIT_LOCKED),
                (COLOR_EXIT_LOCKED, COLOR_HIGHLIGHT),
            )
        else:
            pulse = COLOR_EXIT_OPEN if (self._anim_tick % 2 == 0) else COLOR_HIGHLIGHT
            exit_pixels = ((pulse, pulse), (COLOR_FLOOR, COLOR_FLOOR), (COLOR_EXIT_OPEN, COLOR_EXIT_OPEN))
        for dy in range(DOOR_SIZE[1]):
            for dx in range(DOOR_SIZE[0]):
                grid[exit_y + dy, exit_x + dx] = int(exit_pixels[dy][dx])

        # Key sparkle.
        if not self._key_collected:
            key_x = int(self._model["key_anchor"][0])
            key_y = int(self._model["key_anchor"][1])
            if self._anim_tick % 2 == 0:
                key_pixels = (
                    (COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY, COLOR_HIGHLIGHT),
                    (COLOR_HIGHLIGHT, COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY),
                )
            else:
                key_pixels = (
                    (COLOR_HIGHLIGHT, COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY),
                    (COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY, COLOR_HIGHLIGHT),
                )
            for dy in range(KEY_SIZE[1]):
                for dx in range(KEY_SIZE[0]):
                    grid[key_y + dy, key_x + dx] = int(key_pixels[dy][dx])

        # Laser visuals.
        for idx, group in enumerate(self._model["laser_groups"]):
            phase = int(self._laser_phases[idx]) if idx < len(self._laser_phases) else LASER_OFF
            if phase == LASER_OFF:
                color = COLOR_FLOOR
            elif phase == LASER_WARN:
                color = COLOR_HIGHLIGHT
            else:
                color = COLOR_HAZARD
            for x, y in group:
                grid[int(y), int(x)] = int(color)

        # Guard.
        if self._model.get("guard_start") is not None:
            guard_blink = (self._anim_tick % 2) == 1
            for dy in range(GUARD_SIZE[1]):
                for dx in range(GUARD_SIZE[0]):
                    color = COLOR_GUARD
                    if guard_blink and dx == 0 and dy == 0:
                        color = COLOR_HIGHLIGHT
                    gy = self._guard_y + dy
                    gx = self._guard_x + dx
                    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
                        grid[gy, gx] = int(color)

        # Player.
        if self._freeze_mode == FREEZE_FAIL:
            flash = COLOR_HAZARD if (self._anim_tick % 2 == 0) else COLOR_HIGHLIGHT
            for y, x in ((0, 0), (0, 1), (1, 0), (1, 1)):
                grid[self._player_y + y, self._player_x + x] = int(flash)
        else:
            outline = COLOR_KEY_AND_PLAYER_OUTLINE_WITH_KEY if self._has_key else COLOR_PLAYER_OUTLINE_NO_KEY
            player_pixels = ((outline, outline), (outline, COLOR_PLAYER_FILL))
            for dy in range(PLAYER_SIZE[1]):
                for dx in range(PLAYER_SIZE[0]):
                    grid[self._player_y + dy, self._player_x + dx] = int(player_pixels[dy][dx])

        # Timebar (row 0, rendered last).
        fill = round((float(self._time_left) / float(max(1, self._time_limit))) * GRID_WIDTH)
        fill = max(0, min(GRID_WIDTH, fill))
        low_time = self._time_left * 5 <= self._time_limit
        fill_color = COLOR_HIGHLIGHT if (low_time and (self._anim_tick % 2 == 1)) else COLOR_TIMEBAR_FILL
        for x in range(GRID_WIDTH):
            grid[0, x] = int(fill_color if x < fill else COLOR_TIMEBAR_EMPTY)

        self._scene.pixels = grid

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

        action_id = int(getattr(self.action.id, "value", self.action.id))

        # Frozen fail/win transition windows.
        if self._freeze_mode != FREEZE_NONE:
            self._freeze_steps_left -= 1
            self._anim_tick += 1
            self._render_scene()

            if self._freeze_steps_left <= 0:
                if self._freeze_mode == FREEZE_FAIL:
                    self.lose()
                else:
                    self.next_level()
                self.complete_action()
                return

            self.complete_action()
            return

        # Door opening transitions complete at the start of the next step.
        if self._door_phase == DOOR_PHASE_OPENING:
            self._door_phase = DOOR_PHASE_OPEN

        # 1) Consume action.
        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            self._try_move(dx, dy)

        # 2) Pickups.
        just_collected = self._collect_key_if_needed()

        # 3) Doors update.
        if just_collected and self._door_phase == DOOR_PHASE_LOCKED:
            self._door_phase = DOOR_PHASE_OPENING

        # 4) Hazards/NPC update.
        self._advance_hazards()

        # 5) Resolve outcomes.
        if self._check_fail_hazards():
            self._trigger_fail()
        elif self._check_open_exit_overlap():
            self._trigger_win()

        # 6) Timer.
        if self._freeze_mode == FREEZE_NONE:
            self._time_left -= 1
            if self._time_left <= 0:
                self._trigger_fail()

        self._anim_tick += 1
        self._render_scene()
        self.complete_action()


__all__ = [
    "GAME_ID",
    "KeyDoor",
    "_deserialize_model",
    "_rects_overlap",
    "apply_action_transition",
    "initial_search_state_from_model",
    "is_win_state",
]
