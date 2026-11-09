from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "herding"

COLOR_EMPTY = 0
COLOR_WALL = 1
COLOR_GRASS = 2
COLOR_PEN_FLOOR = 3
COLOR_PEN_WALL = 4
COLOR_SHEEP_A = 5
COLOR_SHEEP_B = 6
COLOR_PLAYER = 7
COLOR_HIGHLIGHT = 8
COLOR_DOOR_CLOSED = 9
COLOR_TIME_NORMAL = 10
COLOR_TIME_LOW = 11
COLOR_WIRE = 12
COLOR_WAVE = 13
COLOR_ONE_WAY = 14
COLOR_SHEEP_PEN = 15

MOVE_BY_ACTION = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

DIRS4 = ((0, -1), (0, 1), (-1, 0), (1, 0))
GATE_DIR = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}

LEVEL_SPECS: list[tuple[str, int, list[str]]] = [
    (
        "First pen",
        90,
        [
            "============================",
            "############################",
            "#P.........................#",
            "#..........................#",
            "#............s.............#",
            "#..........................#",
            "#................+++++++...#",
            "#................+ppppp+...#",
            "#................+ppppp+...#",
            "#.................ppppp+...#",
            "#................+ppppp+...#",
            "#................+ppppp+...#",
            "#................+++++++...#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    ),
    (
        "Two sheep choke",
        120,
        [
            "============================",
            "############################",
            "#P..........#..............#",
            "#...........#..............#",
            "#....s......#....s.........#",
            "#...........#..............#",
            "#....###....#.....+++++++..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#..................ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+++++++..#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    ),
    (
        "One-way gates",
        140,
        [
            "============================",
            "############################",
            "#P..........#..............#",
            "#..s........#..............#",
            "#...........#....s.........#",
            "#....s......#..............#",
            "#....###....#.....+++++++..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#...........>.....>ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+ppppp+..#",
            "#...........#.....+++++++..#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    ),
    (
        "Switch and door",
        140,
        [
            "============================",
            "############################",
            "#P....o::::::::::::........#",
            "#.................:........#",
            "#.....s...........:....s...#",
            "#...........s.....:........#",
            "#.................:+++++++.#",
            "#.................:+ppppp+.#",
            "#.................:Dppppp+.#",
            "#.................:Dppppp+.#",
            "#.................:Dppppp+.#",
            "#.................:+ppppp+.#",
            "#.................:+++++++.#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    ),
    (
        "Whistle call",
        160,
        [
            "============================",
            "############################",
            "#P....o::::::::::::........#",
            "#.................:+++++++.#",
            "#.....s...........:+ppppp+.#",
            "#.................:Dppppp+.#",
            "#.................:Dppppp+.#",
            "#....s............:Dppppp+.#",
            "#.................:+ppppp+.#",
            "#.................:+++++++.#",
            "#..........................#",
            "#..###.....................#",
            "#..<s.s....................#",
            "#..###.....................#",
            "#..........................#",
            "############################",
        ],
    ),
    (
        "Final roundup",
        150,
        [
            "============================",
            "############################",
            "#P..o::::::::::::::........#",
            "#....s............:........#",
            "#..........s#.....:........#",
            "#....###....#.....:+++++++.#",
            "#...........#.....:Dppppp+.#",
            "#....s......#.....:Dppppp+.#",
            "#...........#.o::::Dppppp+.#",
            "#...........>.....:+ppppp+.#",
            "#....###....#.....:+++++++.#",
            "#.#########.#..............#",
            "#.<..s..s.#.#..............#",
            "#.#.......#.#....###.......#",
            "#...........#..............#",
            "############################",
        ],
    ),
]


class SheepState(NamedTuple):
    sid: int
    x: int
    y: int
    bump_cooldown: int


class HerdingStatic(NamedTuple):
    width: int
    height: int
    time_limit: int
    walls: frozenset[tuple[int, int]]
    pen_walls: frozenset[tuple[int, int]]
    pen_floor: frozenset[tuple[int, int]]
    gates: dict[tuple[int, int], tuple[int, int]]
    switches: frozenset[tuple[int, int]]
    wires: frozenset[tuple[int, int]]
    doors: tuple[tuple[int, int], ...]
    switch_component_cells: dict[tuple[int, int], frozenset[tuple[int, int]]]
    switch_component_doors: dict[tuple[int, int], tuple[int, ...]]
    sheep_start: tuple[tuple[int, int], ...]
    player_start: tuple[int, int]


class HerdingState(NamedTuple):
    player: tuple[int, int]
    sheep: tuple[SheepState, ...]
    door_phase: tuple[int, ...]
    door_target: tuple[int, ...]
    whistle_timer: int
    time_left: int
    step_index: int
    sheep_frame_a: bool


class StepResult(NamedTuple):
    state: HerdingState
    wave_cells: frozenset[tuple[int, int]]
    flash_cells: frozenset[tuple[int, int]]
    won: bool
    lost: bool


class ModelAction(NamedTuple):
    move: tuple[int, int] | None = None
    click: tuple[int, int] | None = None
    whistle: bool = False


def _component_maps(
    switches: set[tuple[int, int]], wires: set[tuple[int, int]], doors: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], frozenset[tuple[int, int]]], dict[tuple[int, int], tuple[int, ...]]]:
    graph_cells = set(wires) | set(switches)
    by_switch_cells: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    by_switch_doors: dict[tuple[int, int], tuple[int, ...]] = {}

    for switch in switches:
        queue: deque[tuple[int, int]] = deque([switch])
        seen = {switch}
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS4:
                nxt = (x + dx, y + dy)
                if nxt in seen or nxt not in graph_cells:
                    continue
                seen.add(nxt)
                queue.append(nxt)

        door_idx: set[int] = set()
        for idx, door_pos in enumerate(doors):
            dx, dy = door_pos
            for ox, oy in DIRS4:
                if (dx + ox, dy + oy) in seen:
                    door_idx.add(idx)
                    break

        by_switch_cells[switch] = frozenset(seen)
        by_switch_doors[switch] = tuple(sorted(door_idx))

    return by_switch_cells, by_switch_doors


def _parse_level(time_limit: int, lines: list[str]) -> tuple[Level, HerdingStatic]:
    if len(lines) != 16:
        raise ValueError("Herding levels must have exactly 16 rows.")
    width = len(lines[0])
    if width != 28:
        raise ValueError("Herding levels must be exactly 28 columns wide.")
    if any(len(row) != width for row in lines):
        raise ValueError("All herding level rows must have the same width.")

    walls: set[tuple[int, int]] = set()
    pen_walls: set[tuple[int, int]] = set()
    pen_floor: set[tuple[int, int]] = set()
    gates: dict[tuple[int, int], tuple[int, int]] = {}
    switches: set[tuple[int, int]] = set()
    wires: set[tuple[int, int]] = set()
    doors: list[tuple[int, int]] = []
    sheep_start: list[tuple[int, int]] = []
    player_start: tuple[int, int] | None = None

    terrain_pixels = np.full((16, width), COLOR_EMPTY, dtype=np.int8)

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if y == 0:
                terrain_pixels[y, x] = COLOR_EMPTY
                continue

            if ch == "#":
                walls.add((x, y))
                terrain_pixels[y, x] = COLOR_WALL
            elif ch == "+":
                pen_walls.add((x, y))
                terrain_pixels[y, x] = COLOR_PEN_WALL
            elif ch == "p":
                pen_floor.add((x, y))
                terrain_pixels[y, x] = COLOR_PEN_FLOOR
            else:
                terrain_pixels[y, x] = COLOR_GRASS

            if ch in GATE_DIR:
                gates[(x, y)] = GATE_DIR[ch]
            elif ch == "o":
                switches.add((x, y))
            elif ch == ":":
                wires.add((x, y))
            elif ch == "D":
                doors.append((x, y))
            elif ch == "P":
                player_start = (x, y)
            elif ch == "s":
                sheep_start.append((x, y))

    if player_start is None:
        raise ValueError("Each herding level needs exactly one player start.")
    if not sheep_start:
        raise ValueError("Each herding level needs at least one sheep.")

    switch_component_cells, switch_component_doors = _component_maps(switches, wires, doors)

    sprites = [
        Sprite(pixels=terrain_pixels, name="terrain", x=0, y=0, layer=0, collidable=False, tags=["terrain"]),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="wires",
            x=0,
            y=0,
            layer=3,
            collidable=False,
            tags=["wires", "sys_click", "sys_every_pixel"],
        ),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="doors",
            x=0,
            y=0,
            layer=4,
            collidable=False,
            tags=["doors"],
        ),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="gates",
            x=0,
            y=0,
            layer=4,
            collidable=False,
            tags=["gates"],
        ),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="player",
            x=0,
            y=0,
            layer=5,
            collidable=False,
            tags=["player"],
        ),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="sheep",
            x=0,
            y=0,
            layer=6,
            collidable=False,
            tags=["sheep"],
        ),
        Sprite(
            pixels=np.full((16, width), -1, dtype=np.int8),
            name="wave",
            x=0,
            y=0,
            layer=7,
            collidable=False,
            tags=["wave"],
        ),
        Sprite(
            pixels=np.full((1, width), COLOR_TIME_NORMAL, dtype=np.int8),
            name="timer",
            x=0,
            y=0,
            layer=8,
            collidable=False,
            tags=["hud", "timer"],
        ),
    ]

    static = HerdingStatic(
        width=width,
        height=16,
        time_limit=int(time_limit),
        walls=frozenset(walls),
        pen_walls=frozenset(pen_walls),
        pen_floor=frozenset(pen_floor),
        gates=gates,
        switches=frozenset(switches),
        wires=frozenset(wires),
        doors=tuple(doors),
        switch_component_cells=switch_component_cells,
        switch_component_doors=switch_component_doors,
        sheep_start=tuple(sheep_start),
        player_start=player_start,
    )

    level = Level(name="Herding", grid_size=(width, 16), sprites=sprites, data={"time_limit": int(time_limit)})
    return level, static


def _initial_state(static: HerdingStatic) -> HerdingState:
    sheep = tuple(
        SheepState(sid=idx, x=pos[0], y=pos[1], bump_cooldown=0) for idx, pos in enumerate(static.sheep_start)
    )
    door_count = len(static.doors)
    return HerdingState(
        player=static.player_start,
        sheep=sheep,
        door_phase=tuple(0 for _ in range(door_count)),
        door_target=tuple(0 for _ in range(door_count)),
        whistle_timer=0,
        time_left=int(static.time_limit),
        step_index=0,
        sheep_frame_a=True,
    )


def _door_is_open(state: HerdingState, door_idx: int) -> bool:
    return int(state.door_phase[door_idx]) == 2


def _is_in_pen(static: HerdingStatic, state: HerdingState, x: int, y: int) -> bool:
    if (x, y) in static.pen_floor:
        return True
    for idx, pos in enumerate(static.doors):
        if pos == (x, y) and _door_is_open(state, idx):
            return True
    return False


def _all_sheep_in_pen(static: HerdingStatic, state: HerdingState) -> bool:
    return all(_is_in_pen(static, state, sheep.x, sheep.y) for sheep in state.sheep)


def _passable(static: HerdingStatic, state: HerdingState, x: int, y: int, dx: int, dy: int) -> bool:
    if x < 0 or y < 1 or x >= static.width or y >= static.height:
        return False
    if (x, y) in static.walls or (x, y) in static.pen_walls:
        return False

    for idx, pos in enumerate(static.doors):
        if pos == (x, y):
            return _door_is_open(state, idx)

    required = static.gates.get((x, y))
    return not (required is not None and required != (dx, dy))


def _det_choice(count: int, level_seed: int, state: HerdingState, sheep: SheepState, salt: int) -> int:
    if count <= 1:
        return 0
    mix = (
        (level_seed * 1103515245)
        ^ (state.step_index * 2246822519)
        ^ (sheep.sid * 3266489917)
        ^ (sheep.x * 668265263)
        ^ (sheep.y * 374761393)
        ^ (salt * 1274126177)
    ) & 0xFFFFFFFF
    return int(mix % count)


def _wave_cells(static: HerdingStatic, player: tuple[int, int], radius: int) -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    px, py = player
    for dx in range(-radius, radius + 1):
        rem = radius - abs(dx)
        for dy in (-rem, rem):
            x = px + dx
            y = py + dy
            if y < 1 or x < 0 or x >= static.width or y >= static.height:
                continue
            out.add((x, y))
    return frozenset(out)


def step_model(static: HerdingStatic, state: HerdingState, action: ModelAction, level_seed: int) -> StepResult:
    door_target = list(state.door_target)
    flash_cells: frozenset[tuple[int, int]] = frozenset()

    # 1) Click toggle.
    if action.click is not None and action.click in static.switches:
        switch_cell = action.click
        flash_cells = static.switch_component_cells.get(switch_cell, frozenset())
        for door_idx in static.switch_component_doors.get(switch_cell, ()):  # type: ignore[arg-type]
            door_target[door_idx] = 0 if int(door_target[door_idx]) == 2 else 2

    player_x, player_y = state.player
    sheep = [SheepState(sid=s.sid, x=s.x, y=s.y, bump_cooldown=int(s.bump_cooldown)) for s in state.sheep]

    # 2) Move + push.
    if action.move is not None:
        dx, dy = action.move
        tx, ty = player_x + dx, player_y + dy
        sheep_at = next((idx for idx, s in enumerate(sheep) if s.x == tx and s.y == ty), None)
        if sheep_at is None:
            if _passable(static, state, tx, ty, dx, dy):
                player_x, player_y = tx, ty
        else:
            bx, by = tx + dx, ty + dy
            occupied = any((idx != sheep_at and s.x == bx and s.y == by) for idx, s in enumerate(sheep))
            if _passable(static, state, bx, by, dx, dy) and not occupied and (bx, by) != (player_x, player_y):
                pushed = sheep[sheep_at]
                sheep[sheep_at] = SheepState(sid=pushed.sid, x=bx, y=by, bump_cooldown=1)
                player_x, player_y = tx, ty

    # 3) Door animation.
    door_phase = list(state.door_phase)
    for idx in range(len(door_phase)):
        phase = int(door_phase[idx])
        target = int(door_target[idx])
        if phase == target:
            continue
        if target == 2:
            door_phase[idx] = 1 if phase == 0 else 2
        else:
            door_phase[idx] = 1 if phase == 2 else 0

    phase_state = HerdingState(
        player=(player_x, player_y),
        sheep=tuple(sheep),
        door_phase=tuple(int(v) for v in door_phase),
        door_target=tuple(int(v) for v in door_target),
        whistle_timer=int(state.whistle_timer),
        time_left=int(state.time_left),
        step_index=int(state.step_index),
        sheep_frame_a=bool(state.sheep_frame_a),
    )

    # 4) Whistle resolve and wave.
    whistle_timer = int(phase_state.whistle_timer)
    if action.whistle:
        whistle_timer = 2
    whistle_active = whistle_timer > 0
    wave = _wave_cells(static, (player_x, player_y), 1 if whistle_timer == 2 else 2) if whistle_active else frozenset()
    if whistle_timer > 0:
        whistle_timer -= 1

    # 5) Sheep autonomous movement.
    occupied = {(s.x, s.y) for s in sheep}
    moved_sheep: list[SheepState] = []
    for s in sorted(sheep, key=lambda item: item.sid):
        if s.bump_cooldown > 0:
            moved_sheep.append(SheepState(sid=s.sid, x=s.x, y=s.y, bump_cooldown=s.bump_cooldown - 1))
            continue

        move = (0, 0)
        if whistle_active and abs(s.x - player_x) + abs(s.y - player_y) <= 6:
            current_dist = abs(s.x - player_x) + abs(s.y - player_y)
            reducing = []
            for dx, dy in DIRS4:
                nx, ny = s.x + dx, s.y + dy
                if abs(nx - player_x) + abs(ny - player_y) >= current_dist:
                    continue
                if not _passable(static, phase_state, nx, ny, dx, dy):
                    continue
                if (nx, ny) in occupied or (nx, ny) == (player_x, player_y):
                    continue
                reducing.append((dx, dy))
            if reducing:
                move = reducing[_det_choice(len(reducing), level_seed, phase_state, s, salt=11)]
        else:
            options = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
            move = options[_det_choice(len(options), level_seed, phase_state, s, salt=29)]

        nx, ny = s.x + move[0], s.y + move[1]
        if move != (0, 0):
            occupied.remove((s.x, s.y))
            if (
                _passable(static, phase_state, nx, ny, move[0], move[1])
                and (nx, ny) not in occupied
                and (nx, ny) != (player_x, player_y)
            ):
                occupied.add((nx, ny))
                moved_sheep.append(SheepState(sid=s.sid, x=nx, y=ny, bump_cooldown=0))
            else:
                occupied.add((s.x, s.y))
                moved_sheep.append(SheepState(sid=s.sid, x=s.x, y=s.y, bump_cooldown=0))
        else:
            moved_sheep.append(SheepState(sid=s.sid, x=s.x, y=s.y, bump_cooldown=0))

    next_state = HerdingState(
        player=(player_x, player_y),
        sheep=tuple(sorted(moved_sheep, key=lambda item: item.sid)),
        door_phase=tuple(int(v) for v in door_phase),
        door_target=tuple(int(v) for v in door_target),
        whistle_timer=whistle_timer,
        time_left=max(0, int(state.time_left) - 1),
        step_index=int(state.step_index) + 1,
        sheep_frame_a=not bool(state.sheep_frame_a),
    )

    won = _all_sheep_in_pen(static, next_state)
    lost = (next_state.time_left <= 0) and not won
    return StepResult(state=next_state, wave_cells=wave, flash_cells=flash_cells, won=won, lost=lost)


def decode_action(action_id: int, click_cell: tuple[int, int] | None = None) -> ModelAction:
    if isinstance(action_id, GameAction):
        action_id = int(action_id.value)
    if action_id in MOVE_BY_ACTION:
        return ModelAction(move=MOVE_BY_ACTION[action_id])
    if int(action_id) == int(GameAction.ACTION5.value):
        return ModelAction(whistle=True)
    if int(action_id) == int(GameAction.ACTION6.value):
        return ModelAction(click=click_cell)
    return ModelAction()


class Herding(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels: list[Level] = []
        statics: list[HerdingStatic] = []
        for _name, time_limit, lines in LEVEL_SPECS:
            level, static = _parse_level(time_limit, lines)
            levels.append(level)
            statics.append(static)

        self._seed = int(seed)
        self._static_by_level = tuple(statics)
        self._level_index = 0
        self._herding_state = _initial_state(self._static_by_level[0])
        self._last_wave_cells: frozenset[tuple[int, int]] = frozenset()
        self._last_flash_cells: frozenset[tuple[int, int]] = frozenset()

        camera = Camera(width=28, height=16, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        for idx, candidate in enumerate(self._levels):
            if candidate is level:
                self._level_index = idx
                break
        static = self._static_by_level[self._level_index]
        self._herding_state = _initial_state(static)
        self._last_wave_cells = frozenset()
        self._last_flash_cells = frozenset()
        self._sync_render()

    @property
    def _static(self) -> HerdingStatic:
        return self._static_by_level[self._level_index]

    def _level_seed(self) -> int:
        return (self._seed * 10007) + (self._level_index * 7919) + 17

    def _click_cell(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else None
        if not data:
            return None
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return None

        by_display = self.camera.display_to_grid(raw_x, raw_y)
        if by_display is not None:
            x, y = int(by_display[0]), int(by_display[1])
            if 0 <= x < self._static.width and 0 <= y < self._static.height:
                return x, y

        if 0 <= raw_x < self._static.width and 0 <= raw_y < self._static.height:
            return raw_x, raw_y
        return None

    def _sync_render(self) -> None:
        static = self._static
        state = self._herding_state

        wires = np.full((static.height, static.width), -1, dtype=np.int8)
        wire_color = COLOR_HIGHLIGHT if self._last_flash_cells else COLOR_WIRE
        for pos in static.wires:
            wires[pos[1], pos[0]] = wire_color
        for pos in static.switches:
            wires[pos[1], pos[0]] = wire_color

        doors = np.full((static.height, static.width), -1, dtype=np.int8)
        for idx, (x, y) in enumerate(static.doors):
            phase = int(state.door_phase[idx])
            if phase == 0:
                doors[y, x] = COLOR_DOOR_CLOSED
            elif phase == 1:
                doors[y, x] = COLOR_HIGHLIGHT
            else:
                doors[y, x] = COLOR_PEN_FLOOR

        gates = np.full((static.height, static.width), -1, dtype=np.int8)
        for x, y in static.gates:
            gates[y, x] = COLOR_ONE_WAY

        player = np.full((static.height, static.width), -1, dtype=np.int8)
        player[state.player[1], state.player[0]] = COLOR_PLAYER

        sheep_pixels = np.full((static.height, static.width), -1, dtype=np.int8)
        for sheep in state.sheep:
            in_pen = _is_in_pen(static, state, sheep.x, sheep.y)
            if in_pen:
                color = COLOR_SHEEP_PEN if state.sheep_frame_a else COLOR_SHEEP_B
            else:
                color = COLOR_SHEEP_A if state.sheep_frame_a else COLOR_SHEEP_B
            sheep_pixels[sheep.y, sheep.x] = int(color)

        wave = np.full((static.height, static.width), -1, dtype=np.int8)
        for x, y in self._last_wave_cells:
            wave[y, x] = COLOR_WAVE

        timer = np.full((1, static.width), COLOR_EMPTY, dtype=np.int8)
        fill = round((state.time_left / max(1, static.time_limit)) * static.width)
        fill = max(0, min(static.width, fill))
        time_color = COLOR_TIME_LOW if state.time_left <= max(1, static.time_limit // 4) else COLOR_TIME_NORMAL
        if fill > 0:
            timer[0, :fill] = int(time_color)

        self.current_level.get_sprites_by_name("wires")[0].pixels = wires
        self.current_level.get_sprites_by_name("doors")[0].pixels = doors
        self.current_level.get_sprites_by_name("gates")[0].pixels = gates
        self.current_level.get_sprites_by_name("player")[0].pixels = player
        self.current_level.get_sprites_by_name("sheep")[0].pixels = sheep_pixels
        self.current_level.get_sprites_by_name("wave")[0].pixels = wave
        self.current_level.get_sprites_by_name("timer")[0].pixels = timer

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

        action_raw = self.action.id
        action_id = int(action_raw.value) if isinstance(action_raw, GameAction) else int(action_raw)
        click_cell = self._click_cell() if action_id == int(GameAction.ACTION6.value) else None
        model_action = decode_action(action_id, click_cell=click_cell)

        result = step_model(self._static, self._herding_state, model_action, self._level_seed())
        self._herding_state = result.state
        self._last_wave_cells = result.wave_cells
        self._last_flash_cells = result.flash_cells

        if result.won:
            self.next_level()
        elif result.lost:
            self.lose()
        else:
            self._sync_render()

        self.complete_action()
