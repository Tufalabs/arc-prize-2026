from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "light_shadow-0001"

COLOR_VOID = 0
COLOR_SHADOW = 1
COLOR_LAMP_LIT = 2
COLOR_WALL = 3
COLOR_PLAYER = 4
COLOR_BEAM_LIT = 5
COLOR_GUARD = 6
COLOR_ALERT = 7
COLOR_EXIT = 8
COLOR_EXIT_HIGHLIGHT = 9
COLOR_LAMP = 10
COLOR_SWITCH_OFF = 11
COLOR_SWITCH_ON = 12
COLOR_DOOR_CLOSED = 13
COLOR_CRATE = 14
COLOR_TIMEBAR = 15

LAMP_RADIUS = 4
GUARD_RANGE = 8
TURRET_RANGE = 10

MAX_GRID_W = 32
MAX_GRID_H = 18

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
DIR_ORDER = ("right", "down", "left", "up")
MOVE_BY_ACTION = {
    int(GameAction.ACTION1.value): "up",
    int(GameAction.ACTION2.value): "down",
    int(GameAction.ACTION3.value): "left",
    int(GameAction.ACTION4.value): "right",
}
ACTION_WAIT_TOGGLE = int(GameAction.ACTION5.value)


@dataclass(frozen=True)
class PatrolSpec:
    axis: str
    min_pos: int
    max_pos: int


@dataclass(frozen=True)
class GuardStatic:
    x: int
    y: int
    facing: str
    patrol: PatrolSpec | None = None


@dataclass(frozen=True)
class TurretStatic:
    x: int
    y: int
    start_dir: int = 0
    beam_range: int = TURRET_RANGE


@dataclass(frozen=True)
class LevelStatic:
    name: str
    width: int
    height: int
    time_limit: int
    walls: frozenset[tuple[int, int]]
    walkable: frozenset[tuple[int, int]]
    exits: frozenset[tuple[int, int]]
    switches: frozenset[tuple[int, int]]
    door_cells: frozenset[tuple[int, int]]
    lamp_cells: frozenset[tuple[int, int]]
    turret_cells: frozenset[tuple[int, int]]
    start_player: tuple[int, int]
    start_crates: tuple[tuple[int, int], ...]
    guards: tuple[GuardStatic, ...]
    turrets: tuple[TurretStatic, ...]
    switch_controls_door: bool


@dataclass(frozen=True)
class GuardDyn:
    x: int
    y: int
    facing: str
    move_dir: int
    pause: int


@dataclass(frozen=True)
class PlayState:
    player: tuple[int, int]
    crates: tuple[tuple[int, int], ...]
    lamp_on: bool
    door_open: bool
    guards: tuple[GuardDyn, ...]
    turret_dirs: tuple[int, ...]
    time_left: int
    play_steps: int


def _normalize_layout(lines: list[str], width: int, height: int) -> list[str]:
    if len(lines) != height:
        raise ValueError(f"Expected {height} rows, got {len(lines)}")
    out: list[str] = []
    for row in lines:
        if len(row) < width:
            out.append(row + ("#" * (width - len(row))))
        else:
            out.append(row[:width])
    return out


def _guard_facing_from_arrows(arrows: dict[tuple[int, int], str], gx: int, gy: int) -> str:
    candidates = [((gx - 1, gy), "left"), ((gx + 2, gy), "right"), ((gx, gy - 1), "up"), ((gx, gy + 2), "down")]
    for pos, direction in candidates:
        arrow = arrows.get(pos)
        if arrow is None:
            continue
        if arrow == "<" and direction == "left":
            return "left"
        if arrow == ">" and direction == "right":
            return "right"
        if arrow == "^" and direction == "up":
            return "up"
        if arrow == "v" and direction == "down":
            return "down"
    return "left"


def _find_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    rem = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while rem:
        start = rem.pop()
        comp = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in rem:
                    rem.remove(nxt)
                    comp.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _build_level_static(
    *,
    name: str,
    layout: list[str],
    width: int,
    height: int,
    time_limit: int,
    patrol_overrides: dict[int, PatrolSpec] | None = None,
    switch_controls_door: bool = False,
) -> LevelStatic:
    lines = _normalize_layout(layout, width, height)
    walls: set[tuple[int, int]] = set()
    walkable: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    switches: set[tuple[int, int]] = set()
    door_cells: set[tuple[int, int]] = set()
    lamp_cells: set[tuple[int, int]] = set()
    turret_cells_raw: set[tuple[int, int]] = set()
    crates: list[tuple[int, int]] = []
    guards_raw: set[tuple[int, int]] = set()
    arrows: dict[tuple[int, int], str] = {}
    player: tuple[int, int] | None = None

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
                continue
            walkable.add((x, y))
            if ch == "@":
                player = (x, y)
            elif ch == "X":
                exits.add((x, y))
            elif ch in {"o", "O"}:
                switches.add((x, y))
            elif ch == "|":
                door_cells.add((x, y))
            elif ch in {"*", "+"}:
                lamp_cells.add((x, y))
            elif ch == "B":
                crates.append((x, y))
            elif ch == "G":
                guards_raw.add((x, y))
            elif ch == "S":
                turret_cells_raw.add((x, y))
            elif ch in {"<", ">", "^", "v"}:
                arrows[(x, y)] = ch

    if player is None:
        raise ValueError(f"{name}: missing player")

    guard_components = _find_components(guards_raw)
    guards: list[GuardStatic] = []
    for comp in guard_components:
        min_x = min(p[0] for p in comp)
        min_y = min(p[1] for p in comp)
        facing = _guard_facing_from_arrows(arrows, min_x, min_y)
        guards.append(GuardStatic(x=min_x, y=min_y, facing=facing, patrol=None))
    guards = sorted(guards, key=lambda g: (g.y, g.x))

    if patrol_overrides:
        patched: list[GuardStatic] = []
        for idx, g in enumerate(guards):
            patched.append(GuardStatic(x=g.x, y=g.y, facing=g.facing, patrol=patrol_overrides.get(idx)))
        guards = patched

    turret_components = _find_components(turret_cells_raw)
    turrets: list[TurretStatic] = []
    turret_cells: set[tuple[int, int]] = set()
    for comp in turret_components:
        min_x = min(p[0] for p in comp)
        min_y = min(p[1] for p in comp)
        turrets.append(TurretStatic(x=min_x, y=min_y, start_dir=0, beam_range=TURRET_RANGE))
        turret_cells.update(comp)

    return LevelStatic(
        name=name,
        width=width,
        height=height,
        time_limit=time_limit,
        walls=frozenset(walls),
        walkable=frozenset(walkable),
        exits=frozenset(exits),
        switches=frozenset(switches),
        door_cells=frozenset(door_cells),
        lamp_cells=frozenset(lamp_cells),
        turret_cells=frozenset(turret_cells),
        start_player=player,
        start_crates=tuple(sorted(crates)),
        guards=tuple(guards),
        turrets=tuple(sorted(turrets, key=lambda t: (t.y, t.x))),
        switch_controls_door=bool(switch_controls_door),
    )


def _base_level_rows() -> list[dict]:
    return [
        {
            "name": "Level 1 - Light Is Dangerous",
            "width": 24,
            "height": 16,
            "time_limit": 180,
            "layout": [
                "========================",
                "########################",
                "#@.....................#",
                "#......############....#",
                "#......#..**::::::.....#",
                "#......#..++:::::<GG...#",
                "#......#..:::::...GG...#",
                "#....###############...#",
                "#......................#",
                "#...###############....#",
                "#...#.............#....#",
                "#...#.............#....#",
                "#...#.............#..XX#",
                "#...#.............#..XX#",
                "#......................#",
                "########################",
            ],
        },
        {
            "name": "Level 2 - Switch Changes Light",
            "width": 24,
            "height": 16,
            "time_limit": 170,
            "layout": [
                "========================",
                "########################",
                "#@o...##################",
                "#.....##################",
                "#..**.##################",
                "#..++.##################",
                "#......::::<GG::::XX...#",
                "#......:::::GG::::XX...#",
                "#......................#",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
            ],
        },
        {
            "name": "Level 3 - Patrol Timing",
            "width": 24,
            "height": 16,
            "time_limit": 140,
            "layout": [
                "========================",
                "########################",
                "########################",
                "########################",
                "########################",
                "#@....::::**:::<GG...XX#",
                "#.....::::++::::GG...XX#",
                "#......................#",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
                "########################",
            ],
            "patrol_overrides": {0: PatrolSpec(axis="x", min_pos=16, max_pos=19)},
        },
        {
            "name": "Level 4 - Moving Light",
            "width": 24,
            "height": 16,
            "time_limit": 140,
            "layout": [
                "========================",
                "########################",
                "#@.....................#",
                "#.............##.......#",
                "#.............##.......#",
                "#....##.............##.#",
                "#....##.............##.#",
                "#......................#",
                "#...........SS;;;;;;...#",
                "#...........SS.........#",
                "#......................#",
                "#..##.............##...#",
                "#..................XX..#",
                "#..................XX..#",
                "#......................#",
                "########################",
            ],
        },
        {
            "name": "Level 5 - Crate Blocks Beam",
            "width": 24,
            "height": 16,
            "time_limit": 120,
            "layout": [
                "========================",
                "########################",
                "#@............#........#",
                "#.............#........#",
                "#.............#........#",
                "#.............#........#",
                "#.............#........#",
                "#........B....#........#",
                "#....SS;;;;;;.;;;;;;;;.#",
                "#....SS.......#........#",
                "#.............#........#",
                "#.............#...XX...#",
                "#.............#...XX...#",
                "#.............#........#",
                "#.............#........#",
                "########################",
            ],
        },
        {
            "name": "Level 6 - Combined Systems",
            "width": 32,
            "height": 18,
            "time_limit": 170,
            "layout": [
                "================================",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "#.....:::**:::GG:::.|....SS.XX.#",
                "#@....:::++::<GG:o:.|..B.SS;XX.#",
                "#...................|..........#",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
                "################################",
            ],
            "patrol_overrides": {0: PatrolSpec(axis="x", min_pos=12, max_pos=15)},
            "switch_controls_door": True,
        },
    ]


def _build_static_levels() -> list[LevelStatic]:
    out: list[LevelStatic] = []
    for spec in _base_level_rows():
        out.append(
            _build_level_static(
                name=str(spec["name"]),
                layout=list(spec["layout"]),
                width=int(spec["width"]),
                height=int(spec["height"]),
                time_limit=int(spec["time_limit"]),
                patrol_overrides=dict(spec.get("patrol_overrides") or {}),
                switch_controls_door=bool(spec.get("switch_controls_door", False)),
            )
        )
    return out


LEVELS_STATIC = _build_static_levels()


def _initial_state(level: LevelStatic) -> PlayState:
    guards: list[GuardDyn] = []
    for g in level.guards:
        move_dir = 0
        if g.patrol is not None:
            move_dir = 1
        guards.append(GuardDyn(x=g.x, y=g.y, facing=g.facing, move_dir=move_dir, pause=0))
    return PlayState(
        player=level.start_player,
        crates=tuple(sorted(level.start_crates)),
        lamp_on=bool(level.lamp_cells),
        door_open=False,
        guards=tuple(guards),
        turret_dirs=tuple(t.start_dir for t in level.turrets),
        time_left=level.time_limit,
        play_steps=0,
    )


def _in_bounds(level: LevelStatic, x: int, y: int) -> bool:
    return 0 <= x < level.width and 0 <= y < level.height


def _guard_body_cells(g: GuardDyn) -> set[tuple[int, int]]:
    return {(g.x, g.y), (g.x + 1, g.y), (g.x, g.y + 1), (g.x + 1, g.y + 1)}


def _all_guard_body_cells(guards: tuple[GuardDyn, ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for g in guards:
        cells.update(_guard_body_cells(g))
    return cells


def _guard_eye_cell(g: GuardDyn) -> tuple[int, int]:
    if g.facing == "left":
        return (g.x - 1, g.y)
    if g.facing == "right":
        return (g.x + 2, g.y)
    if g.facing == "up":
        return (g.x, g.y - 1)
    return (g.x, g.y + 2)


def _turret_beam_cells(level: LevelStatic, state: PlayState) -> set[tuple[int, int]]:
    blockers = set(level.walls)
    if not state.door_open:
        blockers.update(level.door_cells)
    blockers.update(state.crates)

    out: set[tuple[int, int]] = set()
    for idx, turret in enumerate(level.turrets):
        dname = DIR_ORDER[state.turret_dirs[idx] % len(DIR_ORDER)]
        if dname == "right":
            sx, sy = turret.x + 2, turret.y
        elif dname == "down":
            sx, sy = turret.x + 1, turret.y + 2
        elif dname == "left":
            sx, sy = turret.x - 1, turret.y + 1
        else:
            sx, sy = turret.x, turret.y - 1

        dx, dy = DIRS[dname]
        x, y = sx, sy
        for _ in range(turret.beam_range):
            if not _in_bounds(level, x, y):
                break
            if (x, y) in blockers:
                break
            out.add((x, y))
            x += dx
            y += dy
    return out


def _lamp_distances(level: LevelStatic, state: PlayState) -> dict[tuple[int, int], int]:
    if not state.lamp_on or not level.lamp_cells:
        return {}

    blockers = set(level.walls)
    if not state.door_open:
        blockers.update(level.door_cells)
    blockers.update(state.crates)

    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int, int]] = deque()

    for src in level.lamp_cells:
        dist[src] = 0
        q.append((src[0], src[1], 0))

    while q:
        x, y, d = q.popleft()
        if d >= LAMP_RADIUS:
            continue
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if not _in_bounds(level, nx, ny):
                continue
            if ny == 0:
                continue
            if (nx, ny) in blockers:
                continue
            nd = d + 1
            prev = dist.get((nx, ny))
            if prev is not None and prev <= nd:
                continue
            dist[(nx, ny)] = nd
            q.append((nx, ny, nd))

    return dist


def _lighting_maps(
    level: LevelStatic, state: PlayState
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    dist = _lamp_distances(level, state)
    flicker = state.play_steps > 0 and state.play_steps % 4 == 0
    lamp_visual = {cell for cell, d in dist.items() if not (flicker and d == LAMP_RADIUS)}
    lamp_full = set(dist.keys())
    beam = _turret_beam_cells(level, state)
    return lamp_visual, lamp_full, beam


def _blocked_for_movement(
    level: LevelStatic, state: PlayState, target: tuple[int, int], guard_cells: set[tuple[int, int]]
) -> bool:
    x, y = target
    if not _in_bounds(level, x, y):
        return True
    if y == 0:
        return True
    if (x, y) not in level.walkable:
        return True
    if (x, y) in level.walls:
        return True
    if (x, y) in level.lamp_cells:
        return True
    if (x, y) in level.turret_cells:
        return True
    if (x, y) in guard_cells:
        return True
    if (x, y) in state.crates:
        return True
    return bool((x, y) in level.door_cells and not state.door_open)


def _advance_guards(level: LevelStatic, state: PlayState) -> tuple[GuardDyn, ...]:
    updated: list[GuardDyn] = []
    for idx, guard in enumerate(state.guards):
        static = level.guards[idx]
        patrol = static.patrol
        if patrol is None:
            updated.append(guard)
            continue

        if guard.pause > 0:
            updated.append(
                GuardDyn(x=guard.x, y=guard.y, facing=guard.facing, move_dir=guard.move_dir, pause=guard.pause - 1)
            )
            continue

        if patrol.axis == "x":
            nx = guard.x + guard.move_dir
            ny = guard.y
            facing = "right" if guard.move_dir > 0 else "left"
            at_end = nx <= patrol.min_pos or nx >= patrol.max_pos
        else:
            nx = guard.x
            ny = guard.y + guard.move_dir
            facing = "down" if guard.move_dir > 0 else "up"
            at_end = ny <= patrol.min_pos or ny >= patrol.max_pos

        move_dir = guard.move_dir
        pause = 0
        if at_end:
            move_dir = -move_dir
            pause = 1
            facing = "right" if move_dir > 0 else "left"
            if patrol.axis == "y":
                facing = "down" if move_dir > 0 else "up"

        updated.append(GuardDyn(x=nx, y=ny, facing=facing, move_dir=move_dir, pause=pause))

    return tuple(updated)


def _advance_turrets(level: LevelStatic, state: PlayState) -> tuple[int, ...]:
    if not level.turrets:
        return state.turret_dirs
    return tuple((direction + 1) % 4 for direction in state.turret_dirs)


def _spotted_by_guard(level: LevelStatic, state: PlayState, guard: GuardDyn, lit_cells: set[tuple[int, int]]) -> bool:
    player = state.player
    if player not in lit_cells:
        return False

    eye_x, eye_y = _guard_eye_cell(guard)
    dx, dy = DIRS[guard.facing]

    blockers = set(level.walls)
    if not state.door_open:
        blockers.update(level.door_cells)
    blockers.update(state.crates)

    x, y = eye_x, eye_y
    for _ in range(GUARD_RANGE):
        x += dx
        y += dy
        if not _in_bounds(level, x, y):
            break
        if (x, y) in blockers:
            break
        if (x, y) not in lit_cells:
            break
        if (x, y) == player:
            return True
    return False


def _apply_play_step(level: LevelStatic, state: PlayState, action_id: int) -> tuple[PlayState, str]:
    player = state.player
    crates = set(state.crates)
    lamp_on = state.lamp_on
    door_open = state.door_open
    guards = state.guards
    turret_dirs = state.turret_dirs

    guard_cells = _all_guard_body_cells(guards)

    move_name = MOVE_BY_ACTION.get(int(action_id))
    if move_name is not None:
        dx, dy = DIRS[move_name]
        nx, ny = player[0] + dx, player[1] + dy
        if (nx, ny) in crates:
            bx, by = nx + dx, ny + dy
            if not _blocked_for_movement(level, state, (bx, by), guard_cells) and (bx, by) not in crates:
                crates.remove((nx, ny))
                crates.add((bx, by))
                player = (nx, ny)
        elif not _blocked_for_movement(level, state, (nx, ny), guard_cells):
            player = (nx, ny)
    elif int(action_id) == ACTION_WAIT_TOGGLE:
        if player in level.switches:
            lamp_on = not lamp_on
            if level.switch_controls_door and level.door_cells:
                door_open = not door_open

    time_left = state.time_left - 1

    guard_next = _advance_guards(
        level,
        PlayState(
            player=player,
            crates=tuple(sorted(crates)),
            lamp_on=lamp_on,
            door_open=door_open,
            guards=guards,
            turret_dirs=turret_dirs,
            time_left=time_left,
            play_steps=state.play_steps + 1,
        ),
    )
    turret_next = _advance_turrets(level, state)

    next_state = PlayState(
        player=player,
        crates=tuple(sorted(crates)),
        lamp_on=lamp_on,
        door_open=door_open,
        guards=guard_next,
        turret_dirs=turret_next,
        time_left=time_left,
        play_steps=state.play_steps + 1,
    )

    lamp_visual, lamp_full, beam = _lighting_maps(level, next_state)
    del lamp_visual
    lit_for_detection = set(lamp_full)
    lit_for_detection.update(beam)

    if next_state.player in beam:
        return next_state, "alarm"

    for g in next_state.guards:
        if _spotted_by_guard(level, next_state, g, lit_for_detection):
            return next_state, "alarm"

    if next_state.player in level.exits:
        return next_state, "win"

    if next_state.time_left <= 0:
        return next_state, "timeout"

    return next_state, "play"


def _dominance_key(state: PlayState) -> tuple:
    return (
        state.player,
        state.crates,
        state.lamp_on,
        state.door_open,
        tuple((g.x, g.y, g.facing, g.move_dir, g.pause) for g in state.guards),
        state.turret_dirs,
    )


def plan_level_actions(level: LevelStatic) -> list[int]:
    start = _initial_state(level)
    if start.player in level.exits:
        return []

    queue = deque([start])
    prev: dict[PlayState, PlayState | None] = {start: None}
    prev_action: dict[PlayState, int] = {}
    best_time: dict[tuple, int] = {_dominance_key(start): start.time_left}
    goal_state: PlayState | None = None

    while queue:
        cur = queue.popleft()
        for action_id in (1, 2, 3, 4, 5):
            nxt, status = _apply_play_step(level, cur, action_id)
            if status in {"alarm", "timeout"}:
                continue
            key = _dominance_key(nxt)
            prior_time = best_time.get(key)
            if prior_time is not None and prior_time >= nxt.time_left:
                continue
            best_time[key] = nxt.time_left

            if nxt not in prev:
                prev[nxt] = cur
                prev_action[nxt] = action_id
                if status == "win":
                    goal_state = nxt
                    queue.clear()
                    break
                queue.append(nxt)
        if goal_state is not None:
            break

    if goal_state is None:
        raise RuntimeError(f"No safe plan found for {level.name}")

    actions: list[int] = []
    cur = goal_state
    while prev[cur] is not None:
        actions.append(prev_action[cur])
        cur = prev[cur]  # type: ignore[index]
    actions.reverse()
    return actions


def _make_level(index: int) -> Level:
    return Level(
        name=LEVELS_STATIC[index].name,
        grid_size=(MAX_GRID_W, MAX_GRID_H),
        sprites=[
            Sprite(
                pixels=np.full((MAX_GRID_H, MAX_GRID_W), COLOR_VOID, dtype=np.int8),
                name="world",
                x=0,
                y=0,
                layer=0,
                collidable=False,
                tags=["world"],
            )
        ],
        data={"level_index": index},
    )


class LightShadow(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_make_level(i) for i in range(len(LEVELS_STATIC))]
        camera = Camera(width=MAX_GRID_W, height=MAX_GRID_H, background=COLOR_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._level_index = 0
        self._level: LevelStatic = LEVELS_STATIC[0]
        self._play_state: PlayState = _initial_state(self._level)
        self._mode = "play"
        self._mode_steps_left = 0
        self._world_sprite: Sprite | None = None

    def on_set_level(self, level: Level) -> None:
        self._level_index = int(level.get_data("level_index") or 0)
        self._level = LEVELS_STATIC[self._level_index]
        self._play_state = _initial_state(self._level)
        self._mode = "play"
        self._mode_steps_left = 0
        world = level.get_sprites_by_name("world")
        self._world_sprite = world[0] if world else None
        self._render()

    def _timebar_fill(self) -> int:
        if self._mode == "timeout":
            return 0
        ratio = 0.0
        if self._level.time_limit > 0:
            ratio = max(0.0, min(1.0, self._play_state.time_left / self._level.time_limit))
        return round(self._level.width * ratio)

    def _render(self) -> None:
        if self._world_sprite is None:
            return

        frame = np.full((MAX_GRID_H, MAX_GRID_W), COLOR_VOID, dtype=np.int8)

        fill = self._timebar_fill()
        if self._mode == "alarm" and self._mode_steps_left % 2 == 1:
            frame[0, : self._level.width] = np.int8(COLOR_ALERT)
        else:
            for x in range(self._level.width):
                frame[0, x] = np.int8(COLOR_TIMEBAR if x < fill else COLOR_VOID)

        lamp_visual, _lamp_full, beam = _lighting_maps(self._level, self._play_state)

        crates = set(self._play_state.crates)
        _all_guard_body_cells(self._play_state.guards)

        exit_hot = False
        for ex in self._level.exits:
            if ex == self._play_state.player:
                exit_hot = True
                break
            if abs(ex[0] - self._play_state.player[0]) + abs(ex[1] - self._play_state.player[1]) == 1:
                exit_hot = True
                break

        for y in range(1, self._level.height):
            for x in range(self._level.width):
                cell = (x, y)
                color = COLOR_SHADOW
                if cell in self._level.walls:
                    color = COLOR_WALL
                elif cell in self._level.door_cells and not self._play_state.door_open:
                    color = COLOR_DOOR_CLOSED
                elif cell in self._level.lamp_cells or cell in self._level.turret_cells:
                    color = COLOR_LAMP
                elif cell in crates:
                    color = COLOR_CRATE
                else:
                    if cell in beam:
                        color = COLOR_BEAM_LIT
                    elif cell in lamp_visual:
                        color = COLOR_LAMP_LIT
                    else:
                        color = COLOR_SHADOW

                if cell in self._level.exits:
                    color = COLOR_EXIT_HIGHLIGHT if exit_hot else COLOR_EXIT

                if cell in self._level.switches:
                    color = COLOR_SWITCH_ON if self._play_state.lamp_on else COLOR_SWITCH_OFF

                frame[y, x] = np.int8(color)

        guard_flash = self._mode == "alarm" and self._mode_steps_left % 2 == 1
        for guard in self._play_state.guards:
            gcolor = COLOR_ALERT if guard_flash else COLOR_GUARD
            for cx, cy in _guard_body_cells(guard):
                if _in_bounds(self._level, cx, cy):
                    frame[cy, cx] = np.int8(gcolor)
            ex, ey = _guard_eye_cell(guard)
            if _in_bounds(self._level, ex, ey):
                frame[ey, ex] = np.int8(gcolor)

        px, py = self._play_state.player
        pcolor = COLOR_PLAYER
        if self._mode == "timeout" and self._mode_steps_left % 2 == 1:
            pcolor = COLOR_ALERT
        if _in_bounds(self._level, px, py):
            frame[py, px] = np.int8(pcolor)

        self._world_sprite.pixels = frame

    def compute_solver_plan_for_current_level(self) -> list[int]:
        return plan_level_actions(self._level)

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

        action_id = int(self.action.id.value)

        if self._mode in {"alarm", "timeout"}:
            self.lose()
            self.complete_action()
            return

        self._play_state, status = _apply_play_step(self._level, self._play_state, action_id)

        if status == "alarm":
            self.lose()
            self.complete_action()
            return

        if status == "timeout":
            self.lose()
            self.complete_action()
            return

        if status == "win":
            self.next_level()
            self.complete_action()
            return

        self._render()
        self.complete_action()
