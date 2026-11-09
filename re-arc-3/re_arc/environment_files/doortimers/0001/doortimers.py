from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "doortimers-0001"
GRID_WIDTH = 28
GRID_HEIGHT = 20
TIMEBAR_INNER_WIDTH = 26

COLOR_BG = 0
COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_PLAYER = 3
COLOR_EXIT_A = 4
COLOR_EXIT_B = 5
COLOR_SPIKE = 6
COLOR_DOOR_CLOSED = 7
COLOR_DOOR_OPEN = 8
COLOR_DOOR_SOON = 9
COLOR_TIME_FILL = 10
COLOR_TIME_EMPTY = 11
COLOR_DEATH = 14

PHASE_CLOSED = 0
PHASE_OPEN = 1
PHASE_SOON = 2

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION = int(GameAction.ACTION5.value)

GLYPH_TO_PHASE = {"|": PHASE_CLOSED, ":": PHASE_OPEN, ";": PHASE_SOON}
PHASE_TO_COLOR = {PHASE_CLOSED: COLOR_DOOR_CLOSED, PHASE_OPEN: COLOR_DOOR_OPEN, PHASE_SOON: COLOR_DOOR_SOON}

LEVELS: list[tuple[int, tuple[str, ...]]] = [
    (
        24,
        (
            "[====================------]",
            "[--------------------------]",
            "############################",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............|..........$$.#",
            "#...@........|..........$$.#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "#............#.............#",
            "############################",
        ),
    ),
    (
        32,
        (
            "[========================--]",
            "[--------------------------]",
            "############################",
            "#.........#.......#........#",
            "#.........#.......#........#",
            "#.........#.......#........#",
            "#.........#.......;....$$..#",
            "#.........#.......;....$$..#",
            "#.........#.......#........#",
            "#.........#.......#........#",
            "#.........#..^^^..#........#",
            "#.........#..^^^..#........#",
            "#.........#..^^^..#...^^...#",
            "#.........#.......#........#",
            "#.........|.......#........#",
            "#.........|.......#........#",
            "#..@......#.......#........#",
            "#.........#.......#........#",
            "#.........#.......#........#",
            "############################",
        ),
    ),
    (
        34,
        (
            "[==========================]",
            "[====----------------------]",
            "############################",
            "#.......#.....#.....#......#",
            "#.......#.....#.....#......#",
            "#.......#.....#.....#......#",
            "#.......#.....#.....#...$$.#",
            "#.......#.....#.....;...$$.#",
            "#.......#.....#.....;......#",
            "#.......#.....#..^..#...^^.#",
            "#.......#.....#..^..#......#",
            "#.......#.....:.....#......#",
            "#.......#.....:..^..#......#",
            "#.......#.....#.....#......#",
            "#.......#.....#.....#......#",
            "#.......#.....#.....#......#",
            "#.......|.....#.....#......#",
            "#..@....|.....#.....#......#",
            "#.......#.....#.....#......#",
            "############################",
        ),
    ),
    (
        50,
        (
            "[==========================]",
            "[==========----------------]",
            "############################",
            "#......#....#....#....#....#",
            "#......#....#....#....#....#",
            "#......#....#....#....#....#",
            "#......#....;....#....#....#",
            "#......#....;....#....#....#",
            "#......#....#....#....#....#",
            "#......#....#....#....;.$$.#",
            "#......#....#....#....;.$$.#",
            "#......#....#^..^#....#....#",
            "#......#....#^..^#....#....#",
            "#......#....#....#....#....#",
            "#......#....#....|....#....#",
            "#......#....#....|....#....#",
            "#......|....#....#....#....#",
            "#..@...|....#....#....#....#",
            "#......#....#....#....#....#",
            "############################",
        ),
    ),
    (
        44,
        (
            "[==========================]",
            "[==================--------]",
            "############################",
            "############################",
            "############################",
            "###...........|;......######",
            "###...........|;......######",
            "###.#######.#########.######",
            "###.#######.#########......#",
            "###.#######.#########...$$.#",
            "###@#######.#########...$$.#",
            "###.#######.#########......#",
            "###.#######.#########.######",
            "###......:.......|....######",
            "###......:.......|....######",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
        ),
    ),
    (
        50,
        (
            "[==========================]",
            "[========================--]",
            "############################",
            "############################",
            "#######################.$$.#",
            "##################..;.|..$$#",
            "##################..;.|....#",
            "#######################....#",
            "#######################.####",
            "#######################.####",
            "##############..:...;...####",
            "##############..:...;...####",
            "##################.#########",
            "##################.#########",
            "##################.#########",
            "##################.#########",
            "#......|.....;.....#########",
            "#..@...|.....;.....#########",
            "############################",
            "############################",
        ),
    ),
]


class LevelModel:
    def __init__(
        self,
        *,
        time_limit: int,
        terrain: tuple[str, ...],
        start: tuple[int, int],
        exits: tuple[tuple[int, int], ...],
        spikes: frozenset[tuple[int, int]],
        door_groups: tuple[tuple[tuple[int, int], ...], ...],
        door_initial_phases: tuple[int, ...],
        door_cell_to_group: dict[tuple[int, int], int],
    ) -> None:
        self.time_limit = int(time_limit)
        self.terrain = terrain
        self.start = start
        self.exits = exits
        self.spikes = spikes
        self.door_groups = door_groups
        self.door_initial_phases = door_initial_phases
        self.door_cell_to_group = door_cell_to_group


def _neighbors4(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _parse_level(time_limit: int, rows: tuple[str, ...]) -> LevelModel:
    if len(rows) != GRID_HEIGHT:
        raise ValueError(f"Expected {GRID_HEIGHT} rows, got {len(rows)}.")
    for row in rows:
        if len(row) != GRID_WIDTH:
            raise ValueError(f"Expected row width {GRID_WIDTH}, got {len(row)} in {row!r}.")

    terrain_grid: list[list[str]] = [["#"] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    start: tuple[int, int] | None = None
    exits: list[tuple[int, int]] = []
    spikes: set[tuple[int, int]] = set()
    door_chars: dict[tuple[int, int], str] = {}

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in "#.^":
                terrain_grid[y][x] = ch
            elif ch in GLYPH_TO_PHASE:
                terrain_grid[y][x] = "."
                door_chars[(x, y)] = ch
            elif ch == "@":
                terrain_grid[y][x] = "."
                if start is not None:
                    raise ValueError("Level contains multiple player start positions.")
                start = (x, y)
            elif ch == "$":
                terrain_grid[y][x] = "."
                exits.append((x, y))
            elif ch in "[]=-":
                terrain_grid[y][x] = ch
            else:
                raise ValueError(f"Unsupported glyph {ch!r} at ({x}, {y}).")

            if ch == "^":
                spikes.add((x, y))

    if start is None:
        raise ValueError("Level is missing player start '@'.")
    if len(exits) != 4:
        raise ValueError("Level must contain exactly a 2x2 exit block (4 '$' tiles).")

    visited: set[tuple[int, int]] = set()
    door_groups: list[tuple[tuple[int, int], ...]] = []
    door_phases: list[int] = []
    door_cell_to_group: dict[tuple[int, int], int] = {}

    for cell, glyph in sorted(door_chars.items(), key=lambda item: (item[0][1], item[0][0])):
        if cell in visited:
            continue
        stack = [cell]
        component: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            if door_chars.get(cur) != glyph:
                continue
            visited.add(cur)
            component.append(cur)
            for nxt in _neighbors4(cur[0], cur[1]):
                if nxt in visited:
                    continue
                if door_chars.get(nxt) == glyph:
                    stack.append(nxt)
        if not component:
            continue
        component.sort(key=lambda pos: (pos[1], pos[0]))
        group_idx = len(door_groups)
        door_groups.append(tuple(component))
        door_phases.append(GLYPH_TO_PHASE[glyph])
        for pos in component:
            door_cell_to_group[pos] = group_idx

    return LevelModel(
        time_limit=int(time_limit),
        terrain=tuple("".join(row) for row in terrain_grid),
        start=start,
        exits=tuple(sorted(exits, key=lambda pos: (pos[1], pos[0]))),
        spikes=frozenset(spikes),
        door_groups=tuple(door_groups),
        door_initial_phases=tuple(door_phases),
        door_cell_to_group=door_cell_to_group,
    )


def _terrain_colors(model: LevelModel) -> np.ndarray:
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_BG, dtype=np.int8)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            ch = model.terrain[y][x]
            if ch == "#":
                grid[y, x] = np.int8(COLOR_WALL)
            elif ch == ".":
                grid[y, x] = np.int8(COLOR_FLOOR)
            elif ch == "^":
                grid[y, x] = np.int8(COLOR_SPIKE)
            elif ch in "[]-":
                grid[y, x] = np.int8(COLOR_TIME_EMPTY)
    return grid


def _build_level(model: LevelModel, level_idx: int) -> Level:
    sprites: list[Sprite] = [
        Sprite(
            _terrain_colors(model), name="terrain", x=0, y=0, layer=0, tags=["terrain", "sys_static"], collidable=False
        ),
        Sprite(
            np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8),
            name="timer",
            x=0,
            y=0,
            layer=1,
            tags=["hud", "timer"],
            collidable=False,
        ),
        Sprite(
            np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8),
            name="doors",
            x=0,
            y=0,
            layer=2,
            tags=["doors"],
            collidable=False,
        ),
        Sprite(
            np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8),
            name="exit",
            x=0,
            y=0,
            layer=3,
            tags=["exit"],
            collidable=False,
        ),
        Sprite(
            np.array([[COLOR_DEATH]], dtype=np.int8), name="death", x=0, y=0, layer=5, tags=["death"], collidable=False
        ),
        Sprite(
            np.array([[COLOR_PLAYER]], dtype=np.int8),
            name="player",
            x=model.start[0],
            y=model.start[1],
            layer=6,
            tags=["player"],
            collidable=True,
        ),
    ]

    return Level(
        name=f"Door Timers {level_idx + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": int(model.time_limit), "level_index": int(level_idx)},
    )


class Doortimers(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_models = [_parse_level(time_limit, rows) for time_limit, rows in LEVELS]
        levels = [_build_level(model, idx) for idx, model in enumerate(self._level_models)]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._level_idx = 0
        self._time_left = 0
        self._pulse_tick = 0
        self._door_phases: list[int] = []
        self._player: Sprite | None = None
        self._timer_sprite: Sprite | None = None
        self._door_sprite: Sprite | None = None
        self._exit_sprite: Sprite | None = None
        self._death_sprite: Sprite | None = None

    @property
    def _model(self) -> LevelModel:
        return self._level_models[self._level_idx]

    def _phase_at(self, group_idx: int) -> int:
        return int(self._door_phases[group_idx] % 3)

    def _door_phase_at_cell(self, x: int, y: int) -> int | None:
        idx = self._model.door_cell_to_group.get((x, y))
        if idx is None:
            return None
        return self._phase_at(idx)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

    def _is_walkable(self, x: int, y: int) -> bool:
        if not self._in_bounds(x, y):
            return False
        if (x, y) in self._model.exits:
            return True
        door_phase = self._door_phase_at_cell(x, y)
        if door_phase is not None:
            return door_phase in {PHASE_OPEN, PHASE_SOON}
        return self._model.terrain[y][x] in {".", "^"}

    def _sync_timer(self) -> None:
        if self._timer_sprite is None:
            return
        timer = np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8)
        remaining = max(0, min(52, int(self._time_left)))
        for row in (0, 1):
            timer[row, 0] = np.int8(COLOR_TIME_EMPTY)
            timer[row, GRID_WIDTH - 1] = np.int8(COLOR_TIME_EMPTY)
            for x in range(1, GRID_WIDTH - 1):
                timer[row, x] = np.int8(COLOR_TIME_EMPTY)
        for i in range(remaining):
            row = 0 if i < TIMEBAR_INNER_WIDTH else 1
            col = 1 + (i % TIMEBAR_INNER_WIDTH)
            timer[row, col] = np.int8(COLOR_TIME_FILL)
        self._timer_sprite.pixels = timer

    def _sync_doors(self) -> None:
        if self._door_sprite is None:
            return
        pixels = np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8)
        for idx, cells in enumerate(self._model.door_groups):
            color = np.int8(PHASE_TO_COLOR[self._phase_at(idx)])
            for x, y in cells:
                pixels[y, x] = color
        self._door_sprite.pixels = pixels

    def _sync_exit(self) -> None:
        if self._exit_sprite is None:
            return
        color = COLOR_EXIT_A if (self._pulse_tick % 2 == 0) else COLOR_EXIT_B
        pixels = np.full((GRID_HEIGHT, GRID_WIDTH), np.int8(-1), dtype=np.int8)
        for x, y in self._model.exits:
            pixels[y, x] = np.int8(color)
        self._exit_sprite.pixels = pixels

    def _sync_visuals(self) -> None:
        self._sync_timer()
        self._sync_doors()
        self._sync_exit()

    def on_set_level(self, level: Level) -> None:
        self._level_idx = int(level.get_data("level_index") or 0)
        self._time_left = int(level.get_data("time_limit") or self._model.time_limit)
        self._pulse_tick = 0
        self._door_phases = list(self._model.door_initial_phases)

        players = level.get_sprites_by_name("player")
        self._player = players[0] if players else None
        if self._player is not None:
            self._player.set_position(self._model.start[0], self._model.start[1])
            self._player.set_visible(True)

        timers = level.get_sprites_by_name("timer")
        self._timer_sprite = timers[0] if timers else None
        doors = level.get_sprites_by_name("doors")
        self._door_sprite = doors[0] if doors else None
        exits = level.get_sprites_by_name("exit")
        self._exit_sprite = exits[0] if exits else None
        deaths = level.get_sprites_by_name("death")
        self._death_sprite = deaths[0] if deaths else None
        if self._death_sprite is not None:
            self._death_sprite.set_visible(False)

        self._sync_visuals()

    def _player_pos(self) -> tuple[int, int]:
        if self._player is None:
            return (0, 0)
        return int(self._player.x), int(self._player.y)

    def _step_move(self, action_id: int) -> None:
        if self._player is None:
            return
        if action_id == WAIT_ACTION:
            return
        delta = MOVE_DELTAS.get(action_id)
        if delta is None:
            return
        px, py = self._player_pos()
        nx = px + int(delta[0])
        ny = py + int(delta[1])
        if self._is_walkable(nx, ny):
            self._player.set_position(nx, ny)

    def _advance_doors(self) -> None:
        for i in range(len(self._door_phases)):
            self._door_phases[i] = (self._door_phases[i] + 1) % 3

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
        self._step_move(action_id)

        px, py = self._player_pos()

        if (px, py) in self._model.spikes:
            self.lose()
            self.complete_action()
            return

        if (px, py) in self._model.exits:
            self.next_level()
            self.complete_action()
            return

        self._advance_doors()
        self._pulse_tick += 1

        phase = self._door_phase_at_cell(px, py)
        if phase == PHASE_CLOSED:
            self.lose()
            self.complete_action()
            return

        self._time_left -= 1
        if self._time_left <= 0:
            self.lose()
            self.complete_action()
            return

        self._sync_visuals()
        self.complete_action()
