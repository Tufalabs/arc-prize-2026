from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "gravity_platformer-0001"
GRID_WIDTH = 40
GRID_HEIGHT = 18
TIMEBAR_TILES = 80
JUMP_STEPS = 6

# Controls used by this game.
ACTION_UP = 1
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_INTERACT = 5
ACTION_IDLE = 7
AVAILABLE_ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_INTERACT]

# Colors (0-15) from spec.
C_AIR = 0
C_SOLID = 1
C_PLAYER = 2
C_EXIT = 3
C_HAZARD = 4
C_PLATFORM = 5
C_SWITCH = 6
C_SWITCH_DOOR = 7
C_KEY = 8
C_CRUMBLE = 9
C_ONEWAY = 10
C_TIME_FULL = 11
C_TIME_EMPTY = 12
C_ENEMY = 13
C_TIMED = 14
C_KEY_DOOR = 15

TIMED_PATTERN = (",", ",", ",", "!", "^", "^", "^", "!")

LEVEL_TEXTS: tuple[tuple[str, ...], ...] = (
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#......................................#",
        "#......................................#",
        "#....................####..............#",
        "#.............................###}{....#",
        "#................................}{....#",
        "#................................}{....#",
        "#........................########......#",
        "#..................####................#",
        "#............####......................#",
        "#.......###............................#",
        "#......................................#",
        "#...###................................#",
        "#......................................#",
        "#..@...................................#",
        "#......................................#",
        "###############^^^^#####################",
    ),
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#......................................#",
        "#......................................#",
        "#................................}{....#",
        "#................................}{....#",
        "#................................}{....#",
        "#........................########......#",
        "#..........................###.........#",
        "#.............................###......#",
        "#......................................#",
        "#.............==>..............####....#",
        "#......................................#",
        "#.............###......................#",
        "#........###...........................#",
        "#.....###..............................#",
        "#..@...................................#",
        "###########^^^^^^^^^^^^^^^^^^###########",
    ),
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#.......................##.............#",
        "#.......................##.............#",
        "#.........####..........##.............#",
        "#.......................##........}{...#",
        "#.......................##........}{...#",
        "#.......................##........}{...#",
        "#.......................##......######.#",
        "#.......................##.............#",
        "#.......................##....#####....#",
        "#.......................##..########...#",
        "#.......................##.....###.....#",
        "#.......................##.......###...#",
        "#.......................%%.............#",
        "#.....................^.%%.............#",
        "#..@..o.................%%.............#",
        "################^^##############^^######",
    ),
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#......................................#",
        "#......................................#",
        "#.................................}{...#",
        "#.................................}{...#",
        "#....................::::::::.....}{...#",
        "#......................................#",
        "#.................------...............#",
        "#......................................#",
        "#............------....................#",
        "#......................................#",
        "#.......------.........................#",
        "#......................................#",
        "#......................................#",
        "#..@...................................#",
        "#......................................#",
        "################^^^^^^^#################",
    ),
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#......................................#",
        "#......................................#",
        "#.................................}{...#",
        "#.................................}{...#",
        "#.................................}{...#",
        "#........................########......#",
        "#.....................###..............#",
        "#..........................###.........#",
        "#.............................###......#",
        "#......................................#",
        "#.......................###............#",
        "#..................$...................#",
        "#..............,..###.........XX.......#",
        "#..............,..............XX.......#",
        "#..@....&......,..............XX.......#",
        "#############^^^^^^#####################",
    ),
    (
        "++++++++++++++++++++++++++++++++++++++++",
        "++++++++++++++++++++++++++++++++++++++++",
        "#......................................#",
        "#..........................XX}{........#",
        "#..........................XX}{........#",
        "#..........................XX}{........#",
        "#.....------..........#####............#",
        "#.........------....#####..............#",
        "#.............------....#####..........#",
        "#.................------....#####......#",
        "#...........................######.....#",
        "#..................::::::.....$........#",
        "#....................############......#",
        "#.............==>......................#",
        "#........###%%............,............#",
        "#.....###...%%............,............#",
        "#..@.o......%%............,..&.........#",
        "#############^^^^^^^^^^#################",
    ),
)


@dataclass(frozen=True)
class PlatformMeta:
    y: int
    min_left: int
    max_left: int
    start_left: int
    start_dir: int
    switch_controlled: bool
    start_active: bool


@dataclass(frozen=True)
class DoorModel:
    door_type: str  # '%' or 'X'
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LevelModel:
    index: int
    spawn: tuple[int, int]
    exits: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]
    one_way: frozenset[tuple[int, int]]
    static_spikes: frozenset[tuple[int, int]]
    switch_pos: tuple[int, int] | None
    key_pos: tuple[int, int] | None
    doors: tuple[DoorModel, ...]
    timed_cells: tuple[tuple[int, int], ...]
    crumble_cells: tuple[tuple[int, int], ...]
    crumble_index: dict[tuple[int, int], int]
    platform: PlatformMeta | None
    enemy_start: tuple[int, int] | None


@dataclass(frozen=True)
class SimState:
    x: int
    y: int
    jump_remaining: int
    has_key: bool
    switch_on: bool
    time_left: int
    key_present: bool
    door_open_mask: int
    door_anim_1: int
    door_anim_2: int
    timed_phase: int
    crumble_state: tuple[int, ...]  # 0 intact, 1 armed, 2 cracked, 3 gone
    platform_left: int
    platform_dir: int
    platform_active: bool
    enemy_x: int
    enemy_y: int
    enemy_dir: int
    enemy_alive: bool
    enemy_anim: int


def _assert_grid(lines: tuple[str, ...]) -> None:
    if len(lines) != GRID_HEIGHT:
        raise ValueError("Each level must be 18 rows")
    for row in lines:
        if len(row) != GRID_WIDTH:
            raise ValueError("Each level row must be 40 columns")


def _collect_cells(lines: tuple[str, ...], chars: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    wanted = set(chars)
    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch in wanted:
                out.append((x, y))
    return out


def _parse_platform(lines: tuple[str, ...], level_idx: int) -> PlatformMeta | None:
    found: tuple[int, int] | None = None
    for y, row in enumerate(lines):
        for token in ("==>", "<==", "==="):
            x = row.find(token)
            if x >= 0:
                found = (x, y)
                break
        if found is not None:
            break
    if found is None:
        return None

    x, y = found
    if level_idx == 1:
        return PlatformMeta(
            y=y, min_left=10, max_left=26, start_left=x, start_dir=1, switch_controlled=False, start_active=True
        )
    if level_idx == 5:
        return PlatformMeta(
            y=y, min_left=14, max_left=24, start_left=x, start_dir=1, switch_controlled=True, start_active=False
        )
    return None


def _parse_level_model(level_idx: int, lines: tuple[str, ...]) -> LevelModel:
    _assert_grid(lines)

    spawn_cells = _collect_cells(lines, "@")
    if len(spawn_cells) != 1:
        raise ValueError("Level must include exactly one player spawn")
    spawn = spawn_cells[0]

    switch_cells = _collect_cells(lines, "oO")
    switch_pos = switch_cells[0] if switch_cells else None

    key_cells = _collect_cells(lines, "$")
    key_pos = key_cells[0] if key_cells else None

    enemy_cells = _collect_cells(lines, "&8")
    enemy_start = enemy_cells[0] if enemy_cells else None

    exits = frozenset(_collect_cells(lines, "{}"))
    walls = frozenset(_collect_cells(lines, "#"))
    one_way = frozenset(_collect_cells(lines, "-"))
    static_spikes = frozenset(_collect_cells(lines, "^"))
    timed_cells = tuple(_collect_cells(lines, ",!"))
    crumble_cells = tuple(_collect_cells(lines, ":;"))
    crumble_index = {pos: idx for idx, pos in enumerate(crumble_cells)}

    pct_cells = tuple(_collect_cells(lines, "%"))
    x_cells = tuple(_collect_cells(lines, "X"))
    doors: list[DoorModel] = []
    if pct_cells:
        doors.append(DoorModel("%", pct_cells))
    if x_cells:
        doors.append(DoorModel("X", x_cells))

    return LevelModel(
        index=level_idx,
        spawn=spawn,
        exits=exits,
        walls=walls,
        one_way=one_way,
        static_spikes=static_spikes,
        switch_pos=switch_pos,
        key_pos=key_pos,
        doors=tuple(doors),
        timed_cells=timed_cells,
        crumble_cells=crumble_cells,
        crumble_index=crumble_index,
        platform=_parse_platform(lines, level_idx),
        enemy_start=enemy_start,
    )


LEVEL_MODELS: tuple[LevelModel, ...] = tuple(_parse_level_model(idx, lines) for idx, lines in enumerate(LEVEL_TEXTS))


def initial_state(model: LevelModel) -> SimState:
    platform = model.platform
    return SimState(
        x=model.spawn[0],
        y=model.spawn[1],
        jump_remaining=0,
        has_key=False,
        switch_on=False,
        time_left=TIMEBAR_TILES,
        key_present=(model.key_pos is not None),
        door_open_mask=0,
        door_anim_1=0,
        door_anim_2=0,
        timed_phase=0,
        crumble_state=tuple(0 for _ in model.crumble_cells),
        platform_left=(platform.start_left if platform else 0),
        platform_dir=(platform.start_dir if platform else 1),
        platform_active=(platform.start_active if platform else False),
        enemy_x=(model.enemy_start[0] if model.enemy_start else 0),
        enemy_y=(model.enemy_start[1] if model.enemy_start else 0),
        enemy_dir=1,
        enemy_alive=(model.enemy_start is not None),
        enemy_anim=0,
    )


def _door_anim_for_index(state: SimState, idx: int) -> int:
    if idx == 0:
        return state.door_anim_1
    if idx == 1:
        return state.door_anim_2
    return 0


def _set_door_anim(state: SimState, idx: int, value: int) -> SimState:
    if idx == 0:
        return SimState(**{**state.__dict__, "door_anim_1": int(value)})
    if idx == 1:
        return SimState(**{**state.__dict__, "door_anim_2": int(value)})
    return state


def _is_door_open(state: SimState, idx: int) -> bool:
    return bool((state.door_open_mask >> idx) & 1)


def _set_door_open(state: SimState, idx: int) -> SimState:
    return SimState(**{**state.__dict__, "door_open_mask": state.door_open_mask | (1 << idx)})


def _timed_char(state: SimState) -> str:
    return TIMED_PATTERN[state.timed_phase % len(TIMED_PATTERN)]


def _platform_cells(model: LevelModel, state: SimState) -> set[tuple[int, int]]:
    if model.platform is None:
        return set()
    return {(state.platform_left + i, model.platform.y) for i in range(3)}


def _door_solid_cells(model: LevelModel, state: SimState) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, door in enumerate(model.doors):
        if _is_door_open(state, idx):
            continue
        out.update(door.cells)
    return out


def _door_anim_cells(model: LevelModel, state: SimState) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, door in enumerate(model.doors):
        if _is_door_open(state, idx):
            continue
        if _door_anim_for_index(state, idx) > 0:
            out.update(door.cells)
    return out


def _timed_extended_cells(model: LevelModel, state: SimState) -> set[tuple[int, int]]:
    if _timed_char(state) != "^":
        return set()
    return set(model.timed_cells)


def _crumble_is_solid(crumble_state_value: int) -> bool:
    return crumble_state_value in (0, 1, 2)


def _is_solid_for_move(model: LevelModel, state: SimState, x: int, y: int, *, down_entry: bool) -> bool:
    if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
        return True
    if y < 2:
        return True
    if (x, y) in model.walls:
        return True
    if (x, y) in model.static_spikes:
        return True
    if (x, y) in _door_solid_cells(model, state):
        return True
    if (x, y) in _timed_extended_cells(model, state):
        return True
    if (x, y) in _platform_cells(model, state):
        return True
    if (x, y) in model.one_way and down_entry:
        return True
    cidx = model.crumble_index.get((x, y))
    return bool(cidx is not None and _crumble_is_solid(state.crumble_state[cidx]))


def _is_supported_below(model: LevelModel, state: SimState, x: int, y: int) -> bool:
    return _is_solid_for_move(model, state, x, y + 1, down_entry=True)


def _is_grounded(model: LevelModel, state: SimState) -> bool:
    return _is_supported_below(model, state, state.x, state.y)


def _is_hazard_cell(model: LevelModel, state: SimState, x: int, y: int) -> bool:
    if (x, y) in model.static_spikes:
        return True
    return (x, y) in _timed_extended_cells(model, state)


def _maybe_open_adjacent_door(model: LevelModel, state: SimState) -> SimState:
    px, py = state.x, state.y
    for idx, door in enumerate(model.doors):
        if _is_door_open(state, idx):
            continue
        if _door_anim_for_index(state, idx) > 0:
            continue
        adjacent = False
        for cx, cy in door.cells:
            if abs(cx - px) + abs(cy - py) == 1:
                adjacent = True
                break
        if not adjacent:
            continue
        can_open = False
        if door.door_type == "%":
            can_open = bool(state.switch_on)
        elif door.door_type == "X":
            can_open = bool(state.has_key)
        if not can_open:
            continue
        state = _set_door_anim(state, idx, 2)
    return state


def _advance_door_animation(model: LevelModel, state: SimState) -> SimState:
    for idx, _door in enumerate(model.doors):
        if _is_door_open(state, idx):
            continue
        anim = _door_anim_for_index(state, idx)
        if anim <= 0:
            continue
        anim -= 1
        state = _set_door_anim(state, idx, anim)
        if anim <= 0:
            state = _set_door_open(state, idx)
    return state


def _advance_crumble_decay(model: LevelModel, state: SimState) -> SimState:
    if not model.crumble_cells:
        return state
    vals = list(state.crumble_state)
    changed = False
    for i, value in enumerate(vals):
        if value == 1:
            vals[i] = 2
            changed = True
        elif value == 2:
            vals[i] = 3
            changed = True
    if changed:
        return SimState(**{**state.__dict__, "crumble_state": tuple(vals)})
    return state


def _trigger_crumble_under_player(model: LevelModel, state: SimState) -> SimState:
    idx = model.crumble_index.get((state.x, state.y))
    if idx is None:
        return state
    if state.crumble_state[idx] != 0:
        return state
    vals = list(state.crumble_state)
    vals[idx] = 1
    return SimState(**{**state.__dict__, "crumble_state": tuple(vals)})


def _collect_key_if_present(model: LevelModel, state: SimState) -> SimState:
    if not state.key_present or model.key_pos is None:
        return state
    if (state.x, state.y) != model.key_pos:
        return state
    return SimState(**{**state.__dict__, "key_present": False, "has_key": True})


def _move_horizontal(model: LevelModel, state: SimState, dx: int) -> SimState:
    nx = state.x + dx
    ny = state.y
    if _is_solid_for_move(model, state, nx, ny, down_entry=False):
        return state
    return SimState(**{**state.__dict__, "x": nx, "y": ny})


def _do_vertical(model: LevelModel, state: SimState, do_jump: bool) -> SimState:
    jump_remaining = state.jump_remaining
    if do_jump and _is_grounded(model, state):
        jump_remaining = JUMP_STEPS

    x, y = state.x, state.y
    if jump_remaining > 0:
        if not _is_solid_for_move(model, state, x, y - 1, down_entry=False):
            y -= 1
            jump_remaining -= 1
        else:
            jump_remaining = 0
    else:
        if not _is_solid_for_move(model, state, x, y + 1, down_entry=True):
            y += 1

    return SimState(**{**state.__dict__, "x": x, "y": y, "jump_remaining": jump_remaining})


def _update_platform(model: LevelModel, state: SimState) -> SimState:
    platform = model.platform
    if platform is None:
        return state

    active = state.platform_active
    if platform.switch_controlled:
        active = bool(state.switch_on)

    delta = 0
    left = state.platform_left
    direction = state.platform_dir

    if active:
        candidate = left + direction
        if candidate < platform.min_left or candidate > platform.max_left:
            direction *= -1
            candidate = left + direction
        left = candidate
        delta = direction

    out = SimState(**{**state.__dict__, "platform_left": left, "platform_dir": direction, "platform_active": active})

    old_cells = {(state.platform_left + i, platform.y) for i in range(3)}
    was_on = (out.x, out.y + 1) in old_cells
    if was_on and delta != 0:
        tx = out.x + delta
        ty = out.y
        if not _is_solid_for_move(model, out, tx, ty, down_entry=False):
            out = SimState(**{**out.__dict__, "x": tx, "y": ty})

    return out


def _enemy_can_step_to(model: LevelModel, state: SimState, x: int, y: int) -> bool:
    if _is_solid_for_move(model, state, x, y, down_entry=False):
        return False
    return _is_supported_below(model, state, x, y)


def _update_enemy(model: LevelModel, state: SimState) -> SimState:
    if not state.enemy_alive:
        return state

    enemy_dir = state.enemy_dir
    ex, ey = state.enemy_x, state.enemy_y

    nx = ex + enemy_dir
    if not _enemy_can_step_to(model, state, nx, ey):
        enemy_dir *= -1
        nx = ex + enemy_dir

    if _enemy_can_step_to(model, state, nx, ey):
        ex = nx

    return SimState(
        **{**state.__dict__, "enemy_x": ex, "enemy_y": ey, "enemy_dir": enemy_dir, "enemy_anim": 1 - state.enemy_anim}
    )


def _update_timed_gate(model: LevelModel, state: SimState) -> SimState:
    if not model.timed_cells:
        return state
    return SimState(**{**state.__dict__, "timed_phase": (state.timed_phase + 1) % len(TIMED_PATTERN)})


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def advance_state(model: LevelModel, state: SimState, action_id: int) -> tuple[SimState, bool, bool]:
    s = state

    # Start-of-step animated door/crumble progression.
    s = _advance_door_animation(model, s)
    s = _advance_crumble_decay(model, s)

    # Interact phase.
    if action_id == ACTION_INTERACT:
        if model.switch_pos is not None and _adjacent((s.x, s.y), model.switch_pos):
            switch_on = not s.switch_on
            s = SimState(**{**s.__dict__, "switch_on": switch_on})
        s = _maybe_open_adjacent_door(model, s)

    # Horizontal phase.
    if action_id == ACTION_LEFT:
        s = _move_horizontal(model, s, -1)
    elif action_id == ACTION_RIGHT:
        s = _move_horizontal(model, s, 1)
    s = _collect_key_if_present(model, s)

    # Vertical phase.
    s = _do_vertical(model, s, do_jump=(action_id == ACTION_UP))
    s = _collect_key_if_present(model, s)

    # Platform phase.
    s = _update_platform(model, s)
    s = _collect_key_if_present(model, s)

    # Timed gate, crumble trigger, enemy.
    s = _update_timed_gate(model, s)
    s = _trigger_crumble_under_player(model, s)
    s = _update_enemy(model, s)

    # Collision/death.
    dead = _is_hazard_cell(model, s, s.x, s.y)
    if s.enemy_alive and (s.x, s.y) == (s.enemy_x, s.enemy_y):
        dead = True

    # Win check before time decrement.
    won = (s.x, s.y) in model.exits

    # Timebar decrement unless already won this frame.
    if not won:
        next_time = max(0, s.time_left - 1)
        s = SimState(**{**s.__dict__, "time_left": next_time})
        if next_time <= 0:
            dead = True

    return s, won, dead


def _make_level(level_idx: int, _model: LevelModel) -> Level:
    frame = np.full((GRID_HEIGHT, GRID_WIDTH), C_AIR, dtype=np.int8)
    return Level(
        name=f"Level {level_idx + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[Sprite(pixels=frame, name="frame", x=0, y=0, layer=0, tags=["frame"], collidable=False)],
        data={"level_index": level_idx},
    )


class GravityPlatformer(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_make_level(i, model) for i, model in enumerate(LEVEL_MODELS)]
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=C_AIR),
            win_score=len(levels),
            available_actions=AVAILABLE_ACTIONS,
            seed=seed,
        )
        self._route_score = 0
        self._model: LevelModel | None = None
        self._sim_state: SimState | None = None
        self._frame_sprite: Sprite | None = None
        self._burst_pos: tuple[int, int] | None = None

    def on_set_level(self, level: Level) -> None:
        idx = int(level.get_data("level_index") or 0)
        self._model = LEVEL_MODELS[idx]
        self._sim_state = initial_state(self._model)
        self._frame_sprite = level.get_sprites_by_name("frame")[0]
        self._burst_pos = None
        self._render()

    def _timebar_indices(self, time_left: int) -> set[tuple[int, int]]:
        filled: set[tuple[int, int]] = set()
        remaining = max(0, min(TIMEBAR_TILES, int(time_left)))
        for idx in range(remaining):
            if idx < GRID_WIDTH:
                x = GRID_WIDTH - 1 - idx
                y = 1
            else:
                x = GRID_WIDTH - 1 - (idx - GRID_WIDTH)
                y = 0
            filled.add((x, y))
        return filled

    def _render(self) -> None:
        if self._model is None or self._sim_state is None or self._frame_sprite is None:
            return
        model = self._model
        state = self._sim_state

        frame = np.full((GRID_HEIGHT, GRID_WIDTH), C_AIR, dtype=np.int8)

        for x, y in self._timebar_indices(state.time_left):
            frame[y, x] = C_TIME_FULL
        for y in (0, 1):
            for x in range(GRID_WIDTH):
                if frame[y, x] == C_AIR:
                    frame[y, x] = C_TIME_EMPTY

        for x, y in model.walls:
            frame[y, x] = C_SOLID
        for x, y in model.one_way:
            frame[y, x] = C_ONEWAY
        for x, y in model.static_spikes:
            frame[y, x] = C_HAZARD
        for x, y in model.exits:
            frame[y, x] = C_EXIT

        if model.switch_pos is not None:
            sx, sy = model.switch_pos
            frame[sy, sx] = C_SWITCH

        if model.key_pos is not None and state.key_present:
            kx, ky = model.key_pos
            frame[ky, kx] = C_KEY

        for idx, door in enumerate(model.doors):
            if _is_door_open(state, idx):
                continue
            color = C_SWITCH_DOOR if door.door_type == "%" else C_KEY_DOOR
            for x, y in door.cells:
                frame[y, x] = color

        gate_char = _timed_char(state)
        for x, y in model.timed_cells:
            frame[y, x] = C_HAZARD if gate_char == "^" else C_TIMED

        for i, (x, y) in enumerate(model.crumble_cells):
            if state.crumble_state[i] == 3:
                continue
            frame[y, x] = C_CRUMBLE

        for x, y in _platform_cells(model, state):
            frame[y, x] = C_PLATFORM

        if state.enemy_alive:
            frame[state.enemy_y, state.enemy_x] = C_ENEMY

        if self._burst_pos is not None:
            bx, by = self._burst_pos
            if 0 <= bx < GRID_WIDTH and 0 <= by < GRID_HEIGHT:
                frame[by, bx] = C_HAZARD
        else:
            color = C_KEY if state.has_key else C_PLAYER
            frame[state.y, state.x] = color

        self._frame_sprite.pixels = frame

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

        if self._model is None or self._sim_state is None:
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        next_state, won, dead = advance_state(self._model, self._sim_state, action_id)
        self._sim_state = next_state

        if won:
            self._route_score += 1
            self._burst_pos = None
            self.next_level()
            self.complete_action()
            return

        if dead:
            self._burst_pos = (next_state.x, next_state.y)
            self._render()
            self.lose()
            self.complete_action()
            return

        self._burst_pos = None
        self._render()
        self.complete_action()
