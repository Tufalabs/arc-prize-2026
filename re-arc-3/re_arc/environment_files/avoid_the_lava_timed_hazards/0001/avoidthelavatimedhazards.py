from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "avoid_the_lava_timed_hazards-0001"

COLOR_VOID = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER_A = 3
COLOR_PLAYER_B = 4
COLOR_HOT_A = 5
COLOR_HOT_B = 6
COLOR_COOLED = 7
COLOR_WARNING = 8
COLOR_EXIT_A = 9
COLOR_EXIT_B = 10
COLOR_TIMEBAR_FILL = 11
COLOR_TIMEBAR_EMPTY = 12
COLOR_VENT = 13
COLOR_DEATH_FLASH = 14
COLOR_WIN_FLASH = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION = int(GameAction.ACTION5.value)


@dataclass(frozen=True)
class VentModel:
    body: tuple[int, int]
    nozzle: tuple[int, int]
    direction: tuple[int, int]
    offset: int
    jet_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MoverModel:
    track: tuple[tuple[int, int], ...]
    length: int
    initial_head_index: int
    initial_direction: int


@dataclass(frozen=True)
class LevelModel:
    width: int
    height: int
    time_limit: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    exits: frozenset[tuple[int, int]]
    static_hot: frozenset[tuple[int, int]]
    pulser_offsets: tuple[tuple[int, int, int], ...]
    vents: tuple[VentModel, ...]
    movers: tuple[MoverModel, ...]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), np.int8(color), dtype=np.int8)


def _overlay(width: int, height: int) -> np.ndarray:
    return np.full((height, width), np.int8(-1), dtype=np.int8)


def _parse_level(rows: list[str], *, time_limit: int, level_index: int) -> LevelModel:
    height = len(rows)
    width = len(rows[0])
    if height < 2:
        raise ValueError("Each level must include a timebar row and at least one play row.")
    for row in rows:
        if len(row) != width:
            raise ValueError("Level rows must have consistent width.")

    walls: set[tuple[int, int]] = set()
    static_hot: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    pulser_cells: list[tuple[int, int]] = []
    plus_cells: list[tuple[int, int]] = []
    start: tuple[int, int] | None = None
    vent_bodies: list[tuple[int, int]] = []
    nozzle_dirs: dict[tuple[int, int], tuple[int, int]] = {}

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch == "@":
                start = (x, y)
            elif ch == "~":
                static_hot.add((x, y))
            elif ch == "*":
                pulser_cells.append((x, y))
            elif ch == "+":
                plus_cells.append((x, y))
            elif ch in "[]{}":
                exits.add((x, y))
            elif ch == "!":
                vent_bodies.append((x, y))
            elif ch == ">":
                nozzle_dirs[(x, y)] = (1, 0)
            elif ch == "<":
                nozzle_dirs[(x, y)] = (-1, 0)
            elif ch == "^":
                nozzle_dirs[(x, y)] = (0, -1)
            elif ch == "v":
                nozzle_dirs[(x, y)] = (0, 1)

    if start is None:
        raise ValueError(f"Level {level_index + 1}: missing player start")
    if len(exits) != 4:
        raise ValueError(f"Level {level_index + 1}: exit must be a 2x2 tile group")

    pulser_offsets: list[tuple[int, int, int]] = []
    if level_index == 2 and pulser_cells:
        leftmost = min(x for x, _ in pulser_cells)
        for x, y in sorted(pulser_cells):
            pulser_offsets.append((x, y, (x - leftmost) % 4))
    else:
        for x, y in sorted(pulser_cells):
            pulser_offsets.append((x, y, 0))

    vents: list[VentModel] = []
    for bx, by in sorted(vent_bodies, key=lambda p: (p[1], p[0])):
        nozzle: tuple[int, int] | None = None
        direction: tuple[int, int] | None = None
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            cand = (bx + dx, by + dy)
            cand_dir = nozzle_dirs.get(cand)
            if cand_dir is None:
                continue
            nozzle = cand
            direction = cand_dir
            break
        if nozzle is None or direction is None:
            raise ValueError(f"Level {level_index + 1}: vent body at {(bx, by)} missing nozzle")

        offset = 0
        if level_index == 4:
            # Spec order for level 5: down, right, left -> offsets 0,2,4.
            order = {(12, 5): 0, (19, 9): 2, (5, 11): 4}
            offset = order.get((bx, by), 0)
        elif level_index == 5:
            # Spec order for level 6 vents.
            if direction == (1, 0):
                offset = 1
            elif direction == (-1, 0):
                offset = 3

        jet_cells: list[tuple[int, int]] = []
        cx, cy = nozzle[0] + direction[0], nozzle[1] + direction[1]
        while 0 <= cx < width and 1 <= cy < height and (cx, cy) not in walls:
            jet_cells.append((cx, cy))
            cx += direction[0]
            cy += direction[1]

        vents.append(
            VentModel(body=(bx, by), nozzle=nozzle, direction=direction, offset=offset, jet_cells=tuple(jet_cells))
        )

    movers: list[MoverModel] = []
    if plus_cells:
        if level_index == 3:
            track = [(x, 6) for x in range(9, 21)]
        elif level_index == 5:
            track = [(x, 6) for x in range(6, 27)]
        else:
            track = []

        if not track:
            raise ValueError(f"Level {level_index + 1}: mover track not configured")

        track_set = set(track)
        if not set(plus_cells).issubset(track_set):
            raise ValueError(f"Level {level_index + 1}: mover starts outside configured track")

        plus_sorted = sorted(plus_cells)
        rightmost_plus = max(plus_sorted, key=lambda p: p[0])
        head_index = track.index(rightmost_plus)
        movers.append(MoverModel(track=tuple(track), length=4, initial_head_index=head_index, initial_direction=1))

    return LevelModel(
        width=width,
        height=height,
        time_limit=int(time_limit),
        walls=frozenset(walls),
        start=start,
        exits=frozenset(exits),
        static_hot=frozenset(static_hot),
        pulser_offsets=tuple(pulser_offsets),
        vents=tuple(vents),
        movers=tuple(movers),
    )


def _next_mover_head(head: int, direction: int, track_len: int) -> tuple[int, int]:
    candidate = head + direction
    if 0 <= candidate < track_len:
        return candidate, direction
    bounced = -direction
    return head + bounced, bounced


def _hot_color_for_tick(tick: int) -> int:
    return COLOR_HOT_A if tick % 2 == 0 else COLOR_HOT_B


def _mover_hot_and_warning(
    mover: MoverModel, head_index: int, direction: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    hot: set[tuple[int, int]] = set()
    for i in range(mover.length):
        idx = head_index - direction * i
        if 0 <= idx < len(mover.track):
            hot.add(mover.track[idx])

    warnings: set[tuple[int, int]] = set()
    nxt_head, _ = _next_mover_head(head_index, direction, len(mover.track))
    next_cell = mover.track[nxt_head]
    if next_cell not in hot:
        warnings.add(next_cell)
    return hot, warnings


def _hazard_state_for_tick(
    model: LevelModel, tick: int, mover_heads: tuple[int, ...], mover_dirs: tuple[int, ...]
) -> tuple[
    set[tuple[int, int]], set[tuple[int, int]], dict[tuple[int, int], int], set[tuple[int, int]], set[tuple[int, int]]
]:
    hot: set[tuple[int, int]] = set(model.static_hot)
    warnings: set[tuple[int, int]] = set()
    pulser_colors: dict[tuple[int, int], int] = {}
    vent_nozzle_warning: set[tuple[int, int]] = set()
    vent_base_cells: set[tuple[int, int]] = set()

    hot_color = _hot_color_for_tick(tick)

    for x, y, offset in model.pulser_offsets:
        phase = (tick + offset) % 4
        if phase in (0, 1):
            pulser_colors[(x, y)] = COLOR_COOLED
        elif phase == 2:
            pulser_colors[(x, y)] = COLOR_WARNING
            warnings.add((x, y))
        else:
            pulser_colors[(x, y)] = hot_color
            hot.add((x, y))

    for mover, head, direction in zip(model.movers, mover_heads, mover_dirs, strict=False):
        mover_hot, mover_warn = _mover_hot_and_warning(mover, int(head), int(direction))
        hot.update(mover_hot)
        warnings.update(mover_warn)

    for vent in model.vents:
        vent_base_cells.add(vent.body)
        vent_base_cells.add(vent.nozzle)
        phase = (tick + vent.offset) % 6
        if phase == 3:
            vent_nozzle_warning.add(vent.nozzle)
        elif phase == 4:
            warnings.update(vent.jet_cells)
        elif phase == 5:
            hot.update(vent.jet_cells)

    return hot, warnings, pulser_colors, vent_nozzle_warning, vent_base_cells


def initial_search_state_from_model(model: LevelModel) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...]]:
    heads = tuple(m.initial_head_index for m in model.movers)
    dirs = tuple(m.initial_direction for m in model.movers)
    return int(model.start[0]), int(model.start[1]), 0, heads, dirs


def apply_action_transition(
    model: LevelModel, state: tuple[int, int, int, tuple[int, ...], tuple[int, ...]], action_id: int
) -> tuple[tuple[int, int, int, tuple[int, ...], tuple[int, ...]] | None, bool]:
    px, py, tick, heads, dirs = state

    nx, ny = int(px), int(py)
    if action_id in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[action_id]
        tx = nx + dx
        ty = ny + dy
        if 0 <= tx < model.width and 1 <= ty < model.height and (tx, ty) not in model.walls:
            nx, ny = tx, ty

    next_tick = int(tick) + 1

    next_heads: list[int] = []
    next_dirs: list[int] = []
    for mover, head, direction in zip(model.movers, heads, dirs, strict=False):
        n_head, n_dir = _next_mover_head(int(head), int(direction), len(mover.track))
        next_heads.append(int(n_head))
        next_dirs.append(int(n_dir))

    hot, _warnings, _pulser_colors, _vent_nozzle_warn, _vent_base = _hazard_state_for_tick(
        model, next_tick, tuple(next_heads), tuple(next_dirs)
    )

    if (nx, ny) in hot:
        return None, False

    if (nx, ny) in model.exits:
        return (int(nx), int(ny), int(next_tick), tuple(next_heads), tuple(next_dirs)), True

    if next_tick >= model.time_limit:
        return None, False

    return (int(nx), int(ny), int(next_tick), tuple(next_heads), tuple(next_dirs)), False


LEVEL_LAYOUTS = [
    {
        "time_limit": 90,
        "rows": [
            "========================",
            "########################",
            "#@...........#.........#",
            "#.######....#..~~~~....#",
            "#......#....#..~~~~..[]#",
            "#......#.........~~..{}#",
            "#..#######.............#",
            "#......................#",
            "#....~~~~..............#",
            "#....~~~~..#######.....#",
            "#..........#.....#.....#",
            "#..........#.....#.....#",
            "########################",
        ],
    },
    {
        "time_limit": 110,
        "rows": [
            "========================",
            "########################",
            "#@....#..........#.....#",
            "#.....#..........#.....#",
            "#.....#~~~~~~~~~~#.....#",
            "#........**.**.........#",
            "#.....#~~~~~~~~~~#.....#",
            "#.....#..........#..[].#",
            "#.....#..........#..{}.#",
            "#.....#..........#.....#",
            "#.....#..........#.....#",
            "#.....#..........#.....#",
            "########################",
        ],
    },
    {
        "time_limit": 130,
        "rows": [
            "============================",
            "############################",
            "#@....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~..[].#",
            "#.....~~~~~~~~~~~~~~~~..{}.#",
            "#.....******.******.**.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "#.....~~~~~~~~~~~~~~~~.....#",
            "############################",
        ],
    },
    {
        "time_limit": 120,
        "rows": [
            "============================",
            "############################",
            "############################",
            "############################",
            "############################",
            "#####.####.####.####.#[]...#",
            "#.@.......++++........{}...#",
            "#.####.####.####.####.######",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
        ],
    },
    {
        "time_limit": 160,
        "rows": [
            "============================",
            "############################",
            "#..................***[]...#",
            "#..................***{}...#",
            "#....#####......#####......#",
            "#....#####..!...#####......#",
            "#....#####..v...#####......#",
            "#....#####......#####......#",
            "#....#####......#####......#",
            "#....#####......###!>......#",
            "#....#####......#####......#",
            "#....<!###......#####......#",
            "#....#####......#####......#",
            "#....#####......#####......#",
            "#.@..#####......#####......#",
            "############################",
        ],
    },
    {
        "time_limit": 180,
        "rows": [
            "============================",
            "############################",
            "#.....................[]...#",
            "#.....................{}...#",
            "#..........................#",
            "#########.##.##.##.#########",
            "######......++++...........#",
            "######.#####################",
            "#..........####............#",
            "#......<!....####..........#",
            "#....!>......####..........#",
            "#..........####............#",
            "######.##############.######",
            "#..........................#",
            "#....~~~~....~~~~..........#",
            "#~~~~~~~~****~~~~~~~~~~~~~~#",
            "#@.........................#",
            "############################",
        ],
    },
]

LEVEL_MODELS: tuple[LevelModel, ...] = tuple(
    _parse_level(spec["rows"], time_limit=int(spec["time_limit"]), level_index=idx)
    for idx, spec in enumerate(LEVEL_LAYOUTS)
)


def _build_level(index: int, model: LevelModel) -> Level:
    wall_pixels = _overlay(model.width, model.height)
    for x, y in model.walls:
        wall_pixels[y, x] = np.int8(COLOR_WALL)

    exit_pixels = _overlay(model.width, model.height)
    for x, y in model.exits:
        exit_pixels[y, x] = np.int8(COLOR_EXIT_A)

    sprites: list[Sprite] = [
        Sprite(
            pixels=_solid(model.width, model.height, COLOR_FLOOR),
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        ),
        Sprite(pixels=wall_pixels, name="walls", x=0, y=0, layer=2, tags=["wall", "blocker"], collidable=True),
        Sprite(pixels=exit_pixels, name="exit", x=0, y=0, layer=3, tags=["exit"], collidable=False),
        Sprite(
            pixels=_overlay(model.width, model.height),
            name="vent_overlay",
            x=0,
            y=0,
            layer=4,
            tags=["vent"],
            collidable=False,
        ),
        Sprite(
            pixels=_overlay(model.width, model.height),
            name="hazard_overlay",
            x=0,
            y=0,
            layer=5,
            tags=["hazard"],
            collidable=False,
        ),
        Sprite(
            pixels=np.array([[np.int8(COLOR_PLAYER_A)]], dtype=np.int8),
            name="player",
            x=int(model.start[0]),
            y=int(model.start[1]),
            layer=6,
            tags=["player"],
            collidable=False,
        ),
        Sprite(
            pixels=np.array([[np.int8(COLOR_TIMEBAR_FILL) for _ in range(model.width)]], dtype=np.int8),
            name="timebar",
            x=0,
            y=0,
            layer=7,
            tags=["hud", "timer"],
            collidable=False,
        ),
    ]

    return Level(
        name=f"AvoidTheLava L{index + 1}",
        grid_size=(model.width, model.height),
        sprites=sprites,
        data={"level_index": int(index), "time_limit": int(model.time_limit)},
    )


class AvoidTheLavaTimedHazards(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(i, model) for i, model in enumerate(LEVEL_MODELS)]
        max_width = max(model.width for model in LEVEL_MODELS)
        max_height = max(model.height for model in LEVEL_MODELS)
        camera = Camera(width=max_width, height=max_height, background=COLOR_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._model: LevelModel | None = None
        self._player: Sprite | None = None
        self._exit_sprite: Sprite | None = None
        self._hazard_overlay: Sprite | None = None
        self._vent_overlay: Sprite | None = None
        self._timebar_sprite: Sprite | None = None

        self._tick = 0
        self._heads: tuple[int, ...] = tuple()
        self._dirs: tuple[int, ...] = tuple()
        self._phase = "play"

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        self._model = LEVEL_MODELS[level_index]

        self._player = level.get_sprites_by_name("player")[0]
        self._exit_sprite = level.get_sprites_by_name("exit")[0]
        self._hazard_overlay = level.get_sprites_by_name("hazard_overlay")[0]
        self._vent_overlay = level.get_sprites_by_name("vent_overlay")[0]
        self._timebar_sprite = level.get_sprites_by_name("timebar")[0]

        self._tick = 0
        self._heads = tuple(m.initial_head_index for m in self._model.movers)
        self._dirs = tuple(m.initial_direction for m in self._model.movers)
        self._phase = "play"

        self.camera.width = int(self._model.width)
        self.camera.height = int(self._model.height)
        self._sync_visuals()

    def _player_pos(self) -> tuple[int, int]:
        if self._player is None:
            return 0, 0
        return int(self._player.x), int(self._player.y)

    def _set_player_pos(self, x: int, y: int) -> None:
        if self._player is not None:
            self._player.set_position(int(x), int(y))

    def _apply_player_action(self, action_id: int) -> None:
        if self._model is None:
            return
        if action_id not in MOVE_DELTAS:
            return

        px, py = self._player_pos()
        dx, dy = MOVE_DELTAS[action_id]
        nx = px + dx
        ny = py + dy

        if not (0 <= nx < self._model.width and 1 <= ny < self._model.height):
            return
        if (nx, ny) in self._model.walls:
            return

        self._set_player_pos(nx, ny)

    def _sync_timebar(self) -> None:
        if self._model is None or self._timebar_sprite is None:
            return

        time_remaining = max(0, int(self._model.time_limit) - int(self._tick))
        filled = (int(self._model.width) * int(time_remaining)) // max(1, int(self._model.time_limit))
        filled = max(0, min(int(self._model.width), int(filled)))
        row = np.array(
            [[np.int8(COLOR_TIMEBAR_FILL if x < filled else COLOR_TIMEBAR_EMPTY) for x in range(self._model.width)]],
            dtype=np.int8,
        )
        self._timebar_sprite.pixels = row

    def _sync_visuals(self) -> None:
        if (
            self._model is None
            or self._player is None
            or self._exit_sprite is None
            or self._hazard_overlay is None
            or self._vent_overlay is None
        ):
            return

        hot, warnings, pulser_colors, nozzle_warning, vent_base = _hazard_state_for_tick(
            self._model, int(self._tick), self._heads, self._dirs
        )

        hot_color = _hot_color_for_tick(self._tick)

        hazard_canvas = _overlay(self._model.width, self._model.height)
        for x, y in self._model.static_hot:
            hazard_canvas[y, x] = np.int8(hot_color)
        for (x, y), c in pulser_colors.items():
            hazard_canvas[y, x] = np.int8(c)
        for x, y in warnings:
            if (x, y) not in hot:
                hazard_canvas[y, x] = np.int8(COLOR_WARNING)
        for x, y in hot:
            hazard_canvas[y, x] = np.int8(hot_color)
        self._hazard_overlay.pixels = hazard_canvas

        vent_canvas = _overlay(self._model.width, self._model.height)
        for x, y in vent_base:
            vent_canvas[y, x] = np.int8(COLOR_VENT)
        for x, y in nozzle_warning:
            vent_canvas[y, x] = np.int8(COLOR_WARNING)
        self._vent_overlay.pixels = vent_canvas

        exit_color = COLOR_EXIT_A if self._tick % 2 == 0 else COLOR_EXIT_B
        exit_pixels = _overlay(self._model.width, self._model.height)
        for x, y in self._model.exits:
            exit_pixels[y, x] = np.int8(exit_color)
        self._exit_sprite.pixels = exit_pixels

        if self._phase == "dead":
            player_color = COLOR_DEATH_FLASH
        elif self._phase == "won":
            player_color = COLOR_WIN_FLASH
        else:
            player_color = COLOR_PLAYER_A if self._tick % 2 == 0 else COLOR_PLAYER_B
        self._player.pixels = np.array([[np.int8(player_color)]], dtype=np.int8)

        self._sync_timebar()

    def _resolve_play_step(self) -> None:
        if self._model is None:
            return

        self._tick += 1

        next_heads: list[int] = []
        next_dirs: list[int] = []
        for mover, head, direction in zip(self._model.movers, self._heads, self._dirs, strict=False):
            n_head, n_dir = _next_mover_head(int(head), int(direction), len(mover.track))
            next_heads.append(int(n_head))
            next_dirs.append(int(n_dir))
        self._heads = tuple(next_heads)
        self._dirs = tuple(next_dirs)

        hot, _warnings, _pulser, _nozzle, _base = _hazard_state_for_tick(
            self._model, int(self._tick), self._heads, self._dirs
        )

        px, py = self._player_pos()
        if (px, py) in hot:
            self._phase = "dead"
            return
        if (px, py) in self._model.exits:
            self._phase = "won"
            return
        if self._tick >= self._model.time_limit:
            self._phase = "dead"

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

        if self._phase == "dead":
            self.lose()
            self.complete_action()
            return

        if self._phase == "won":
            self.next_level()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in MOVE_DELTAS:
            self._apply_player_action(action_id)
        elif action_id != WAIT_ACTION:
            # Unsupported actions are treated as wait.
            pass

        self._resolve_play_step()
        self._sync_visuals()
        self.complete_action()
