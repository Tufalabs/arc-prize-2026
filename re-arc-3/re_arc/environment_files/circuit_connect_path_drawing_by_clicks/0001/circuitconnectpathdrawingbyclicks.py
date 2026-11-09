from __future__ import annotations

from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "circuit_connect_path_drawing_by_clicks-0001"
WIDTH = 24
HEIGHT = 16

COLOR_EMPTY = 0
COLOR_FLASH = 1
COLOR_GROUND = 2
COLOR_WIRE = 3
COLOR_WIRE_TRAIL = 4
COLOR_WAVE = 5
COLOR_WALL = 6
COLOR_TIME_FILL = 7
COLOR_TIME_EMPTY = 8
COLOR_SOURCE_A = 9
COLOR_SOURCE_B = 10
COLOR_TARGET_A = 11
COLOR_TARGET_B = 12
COLOR_RELAY_OFF = 13
COLOR_RELAY_ON = 14
COLOR_DOOR_CLOSED = 15

TRAIL_DECAY_STEPS = 3
ILLEGAL_FLASH_STEPS = 4
DIODE_BLOCK_FLASH_STEPS = 2
DOOR_OPEN_FLASH_STEPS = 6
FAIL_SPARK_STEPS = 6

LEVEL_LAYOUTS = [
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#......................#",
        "#.SS...................#",
        "#.SS...................#",
        "#......................#",
        "#......................#",
        "#......................#",
        "#......................#",
        "#......................#",
        "#...............EE.....#",
        "#...............EE.....#",
        "#......................#",
        "#......................#",
        "#......................#",
        "########################",
    ],
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#......................#",
        "#.SS........######.....#",
        "#.SS........######.....#",
        "#...........######.....#",
        "#.....####..######..####",
        "#.....####..........####",
        "#...........######.....#",
        "#...#####..............#",
        "#...............EE.....#",
        "#...............EE.....#",
        "#.............#####....#",
        "#......................#",
        "#......................#",
        "########################",
    ],
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#....X.................#",
        "#.SS........######.....#",
        "#.SS........######.....#",
        "#...........######..X..#",
        "#.....####..######..####",
        "#.....####..........####",
        "#...........######.....#",
        "#...#####....X.........#",
        "#...............EE.....#",
        "#...............EE.....#",
        "#.............#####..X.#",
        "#......................#",
        "#......................#",
        "########################",
    ],
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#..........#...........#",
        "#.SS.......#...........#",
        "#.SS.......#...........#",
        "#..........#...........#",
        "#..........<...........#",
        "#..........#...........#",
        "#..........>...........#",
        "#..........#...........#",
        "#..........#....EE.....#",
        "#..........#....EE.....#",
        "#..........#...........#",
        "#..........#...........#",
        "#..........#...........#",
        "########################",
    ],
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#..........#...........#",
        "#.SS.......#...........#",
        "#.SS.......#...........#",
        "#..........#.....EE....#",
        "#..........#.....EE....#",
        "#..........#...........#",
        "#..........|...........#",
        "#..........|...........#",
        "#..........|...........#",
        "#..........#...........#",
        "#...RR.....#...........#",
        "#...RR.....#...........#",
        "#..........#...........#",
        "########################",
    ],
    [
        "~~~~~~~~~~~~~~~~~~~~~~~~",
        "########################",
        "#.......#......#.......#",
        "#.SS....#......#.......#",
        "#.SS....#.QQ...#.......#",
        "#.......#.QQ...#.......#",
        "#.......#......#.......#",
        "#.......|......#.......#",
        "#.......|..RR..#.......#",
        "#.......|..RR..|.......#",
        "#..RR...#......|>..EE..#",
        "#..RR...#...X..|...EE..#",
        "#.......#......#.......#",
        "#.......#......#.......#",
        "#.......#......#.......#",
        "########################",
    ],
]

LEVEL_TIMES = [140, 160, 170, 190, 220, 260]
LEVEL_DOOR_CONTROLS = [[], [], [], [], ["R"], ["R", "Q"]]

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIODE_DIR = {">": (1, 0), "<": (-1, 0), "^": (0, -1), "v": (0, 1)}
DIODE_COLOR = {">": COLOR_RELAY_ON, "<": COLOR_RELAY_OFF, "^": COLOR_SOURCE_B, "v": COLOR_SOURCE_A}


class LevelModel:
    def __init__(
        self,
        *,
        rows: tuple[str, ...],
        source_tiles: frozenset[tuple[int, int]],
        target_tiles: frozenset[tuple[int, int]],
        ground_tiles: frozenset[tuple[int, int]],
        relay_tiles_by_id: dict[str, frozenset[tuple[int, int]]],
        diodes: dict[tuple[int, int], str],
        door_groups: list[frozenset[tuple[int, int]]],
        door_control_by_group: list[str],
        door_group_for_pos: dict[tuple[int, int], int],
    ) -> None:
        self.rows = rows
        self.source_tiles = source_tiles
        self.target_tiles = target_tiles
        self.ground_tiles = ground_tiles
        self.relay_tiles_by_id = relay_tiles_by_id
        self.diodes = diodes
        self.door_groups = door_groups
        self.door_control_by_group = door_control_by_group
        self.door_group_for_pos = door_group_for_pos


def _solid(w: int, h: int, color: int) -> np.ndarray:
    return np.full((h, w), int(color), dtype=np.int8)


def _validate_layout(rows: list[str]) -> tuple[str, ...]:
    if len(rows) != HEIGHT:
        raise ValueError(f"Expected {HEIGHT} rows, got {len(rows)}")
    for row in rows:
        if len(row) != WIDTH:
            raise ValueError(f"Expected width {WIDTH}, got {len(row)}")
    return tuple(rows)


def _collect_doors(rows: tuple[str, ...]) -> list[frozenset[tuple[int, int]]]:
    doors = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if rows[y][x] == "|"}
    groups: list[frozenset[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()

    for root in sorted(doors, key=lambda p: (p[0], p[1])):
        if root in seen:
            continue
        queue: deque[tuple[int, int]] = deque([root])
        component: set[tuple[int, int]] = set()
        seen.add(root)
        while queue:
            x, y = queue.popleft()
            component.add((x, y))
            for dx, dy in DIRS:
                nxt = (x + dx, y + dy)
                if nxt in doors and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        groups.append(frozenset(component))

    groups.sort(key=lambda cells: (min(pos[0] for pos in cells), min(pos[1] for pos in cells)))
    return groups


def build_level_model(layout_rows: list[str], door_controls: list[str]) -> LevelModel:
    rows = _validate_layout(layout_rows)

    source_tiles: set[tuple[int, int]] = set()
    target_tiles: set[tuple[int, int]] = set()
    ground_tiles: set[tuple[int, int]] = set()
    relay_tiles_by_id: dict[str, set[tuple[int, int]]] = {}
    diodes: dict[tuple[int, int], str] = {}

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "S":
                source_tiles.add((x, y))
            elif ch == "E":
                target_tiles.add((x, y))
            elif ch == "X":
                ground_tiles.add((x, y))
            elif ch in DIODE_DIR:
                diodes[(x, y)] = ch
            elif "A" <= ch <= "Z" and ch not in {"S", "E", "X"}:
                relay_tiles_by_id.setdefault(ch, set()).add((x, y))

    door_groups = _collect_doors(rows)
    if door_controls and len(door_controls) != len(door_groups):
        raise ValueError(f"Door control count mismatch: controls={door_controls} groups={len(door_groups)}")
    if not door_controls:
        door_controls = [""] * len(door_groups)

    door_group_for_pos: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(door_groups):
        for pos in cells:
            door_group_for_pos[pos] = idx

    return LevelModel(
        rows=rows,
        source_tiles=frozenset(source_tiles),
        target_tiles=frozenset(target_tiles),
        ground_tiles=frozenset(ground_tiles),
        relay_tiles_by_id={k: frozenset(v) for k, v in relay_tiles_by_id.items()},
        diodes=dict(diodes),
        door_groups=list(door_groups),
        door_control_by_group=list(door_controls),
        door_group_for_pos=door_group_for_pos,
    )


def _build_level(level_idx: int) -> Level:
    model = build_level_model(LEVEL_LAYOUTS[level_idx], LEVEL_DOOR_CONTROLS[level_idx])
    board = Sprite(
        pixels=_solid(WIDTH, HEIGHT, COLOR_EMPTY),
        name="board",
        x=0,
        y=0,
        layer=0,
        tags=["board", "sys_click", "sys_every_pixel"],
        collidable=False,
    )
    return Level(
        name=f"CircuitConnect L{level_idx + 1}",
        grid_size=(WIDTH, HEIGHT),
        sprites=[board],
        data={
            "level_index": int(level_idx),
            "layout": "\n".join(model.rows),
            "time_limit": int(LEVEL_TIMES[level_idx]),
            "door_controls": list(LEVEL_DOOR_CONTROLS[level_idx]),
        },
    )


class CircuitConnectPathDrawingByClicks(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(idx) for idx in range(len(LEVEL_LAYOUTS))]
        camera = Camera(width=WIDTH, height=HEIGHT, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID, levels=levels, camera=camera, win_score=len(levels), available_actions=[6], seed=seed
        )
        self._level_idx = 0
        self._model: LevelModel | None = None
        self._board: Sprite | None = None
        self._time_limit = LEVEL_TIMES[0]
        self._time_remaining = LEVEL_TIMES[0]
        self._mode = "playing"
        self._anim_tick = 0
        self._pending_click: tuple[int, int] | None = None
        self._wire_cells: set[tuple[int, int]] = set()
        self._wavefront: set[tuple[int, int]] = set()
        self._trail_decay: dict[tuple[int, int], int] = {}
        self._pulse_cooldown = 0
        self._relays_on: set[str] = set()
        self._door_open_groups: set[int] = set()
        self._door_opening: dict[int, int] = {}
        self._illegal_flash: dict[tuple[int, int], int] = {}
        self._diode_flash: dict[tuple[int, int], int] = {}
        self._fail_spark: dict[tuple[int, int], int] = {}

    def on_set_level(self, level: Level) -> None:
        self._level_idx = int(level.get_data("level_index") or 0)
        layout = str(level.get_data("layout") or "")
        rows = [line for line in layout.splitlines() if line]
        controls = [str(v) for v in (level.get_data("door_controls") or [])]
        self._model = build_level_model(rows, controls)
        self._time_limit = int(level.get_data("time_limit") or LEVEL_TIMES[self._level_idx])

        boards = level.get_sprites_by_name("board")
        if not boards:
            raise RuntimeError("circuit_connect level is missing board sprite")
        self._board = boards[0]

        self._reset_runtime_state()
        self._render_board()

    def _reset_runtime_state(self) -> None:
        self._time_remaining = self._time_limit
        self._mode = "playing"
        self._anim_tick = 0
        self._pending_click = None
        self._wire_cells = set()
        self._wavefront = set()
        self._trail_decay = {}
        self._pulse_cooldown = 0
        self._relays_on = set()
        self._door_open_groups = set()
        self._door_opening = {}
        self._illegal_flash = {}
        self._diode_flash = {}
        self._fail_spark = {}

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < WIDTH and 0 <= y < HEIGHT

    def _parse_click(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if not payload:
            return None
        try:
            x = int(payload.get("x", -1))
            y = int(payload.get("y", -1))
        except (TypeError, ValueError):
            return None

        grid = self.camera.display_to_grid(x, y)
        if grid is not None:
            gx, gy = int(grid[0]), int(grid[1])
            if self._in_bounds((gx, gy)):
                return (gx, gy)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            return (x, y)
        return None

    def _base_char(self, pos: tuple[int, int]) -> str:
        if self._model is None:
            return "#"
        x, y = pos
        ch = self._model.rows[y][x]
        if ch == "|":
            group_idx = self._model.door_group_for_pos.get(pos)
            if group_idx is not None and group_idx in self._door_open_groups:
                return "."
        return ch

    def _is_placeable(self, pos: tuple[int, int]) -> bool:
        _x, y = pos
        if y == 0:
            return False
        return self._base_char(pos) == "."

    def _is_conductor(self, pos: tuple[int, int]) -> bool:
        if pos in self._wire_cells:
            return True
        base = self._base_char(pos)
        return base in {"S", "E", "X", "R", "Q", "<", ">", "^", "v"}

    def _diode_allows_exit(self, pos: tuple[int, int], dx: int, dy: int) -> bool:
        if self._model is None:
            return False
        diode = self._model.diodes.get(pos)
        if diode is None:
            return True
        return DIODE_DIR[diode] == (dx, dy)

    def _diode_allows_entry(self, pos: tuple[int, int], dx: int, dy: int) -> bool:
        if self._model is None:
            return False
        diode = self._model.diodes.get(pos)
        if diode is None:
            return True
        ddx, ddy = DIODE_DIR[diode]
        return (ddx, ddy) == (dx, dy)

    def _flow_allowed(self, src: tuple[int, int], dst: tuple[int, int], dx: int, dy: int) -> bool:
        if not self._diode_allows_exit(src, dx, dy):
            return False
        return self._diode_allows_entry(dst, dx, dy)

    def _tick_map(self, values: dict[tuple[int, int], int]) -> None:
        expired: list[tuple[int, int]] = []
        for key in list(values):
            values[key] -= 1
            if values[key] <= 0:
                expired.append(key)
        for key in expired:
            values.pop(key, None)

    def _consume_click_request(self, click: tuple[int, int]) -> bool:
        if self._model is None:
            return False
        if not self._in_bounds(click):
            return False
        x, y = click
        if y == 0:
            self._illegal_flash[(x, y)] = ILLEGAL_FLASH_STEPS
            return False

        if click in self._wire_cells:
            self._wire_cells.remove(click)
            return True

        if self._is_placeable(click):
            self._wire_cells.add(click)
            return True

        self._illegal_flash[(x, y)] = ILLEGAL_FLASH_STEPS
        return False

    def _activate_relay_if_hit(self, wavefront: set[tuple[int, int]]) -> None:
        if self._model is None:
            return
        for relay_id, relay_cells in self._model.relay_tiles_by_id.items():
            if relay_id in self._relays_on:
                continue
            if any(cell in wavefront for cell in relay_cells):
                self._relays_on.add(relay_id)
                for idx, controller in enumerate(self._model.door_control_by_group):
                    if controller != relay_id:
                        continue
                    if idx in self._door_open_groups or idx in self._door_opening:
                        continue
                    self._door_opening[idx] = DOOR_OPEN_FLASH_STEPS

    def _propagate_wave(self) -> set[tuple[int, int]]:
        if self._model is None:
            return set()

        emitted = set(self._model.source_tiles) if self._pulse_cooldown <= 0 else set()
        start_frontier = set(self._wavefront) | emitted

        for pos in self._wavefront:
            self._trail_decay[pos] = TRAIL_DECAY_STEPS

        next_wave: set[tuple[int, int]] = set()
        for x, y in start_frontier:
            for dx, dy in DIRS:
                nxt = (x + dx, y + dy)
                if not self._in_bounds(nxt):
                    continue
                if not self._is_conductor(nxt):
                    continue
                if not self._flow_allowed((x, y), nxt, dx, dy):
                    if self._model.diodes.get(nxt) and not self._diode_allows_entry(nxt, dx, dy):
                        self._diode_flash[nxt] = DIODE_BLOCK_FLASH_STEPS
                    continue
                next_wave.add(nxt)

        self._wavefront = next_wave
        return next_wave

    def _set_failed(self, cells: set[tuple[int, int]]) -> None:
        self._mode = "failed"
        if not cells and self._model is not None:
            cells = set(self._model.source_tiles)
        self._fail_spark = {cell: FAIL_SPARK_STEPS for cell in cells}
        self.lose()

    def _update_playing_state(self) -> None:
        topology_changed = False
        if self._pending_click is not None:
            topology_changed = self._consume_click_request(self._pending_click)
        self._pending_click = None

        if topology_changed:
            self._wavefront.clear()
            self._pulse_cooldown = 0

        wave = self._propagate_wave()
        self._activate_relay_if_hit(wave)

        if self._model is not None:
            if any(cell in self._model.ground_tiles for cell in wave):
                failed_cells = {cell for cell in wave if cell in self._model.ground_tiles}
                self._set_failed(failed_cells)
            elif any(cell in self._model.target_tiles for cell in wave):
                self._mode = "won"

        self._tick_map(self._trail_decay)
        self._tick_map(self._illegal_flash)
        self._tick_map(self._diode_flash)

        for door_idx in list(self._door_opening):
            self._door_opening[door_idx] -= 1
            if self._door_opening[door_idx] <= 0:
                self._door_opening.pop(door_idx, None)
                self._door_open_groups.add(door_idx)

        if self._mode == "playing":
            self._time_remaining -= 1
            if self._time_remaining <= 0:
                self._set_failed(set())

    def _step_post_state(self) -> None:
        if self._mode == "failed":
            self._tick_map(self._fail_spark)

    def _advance_after_win_click(self) -> None:
        self.next_level()

    def _render_board(self) -> None:
        if self._model is None or self._board is None:
            return

        pixels = np.full((HEIGHT, WIDTH), COLOR_EMPTY, dtype=np.int8)

        ratio = self._time_remaining / max(1, self._time_limit)
        filled = round(max(0.0, min(1.0, ratio)) * WIDTH)
        for x in range(WIDTH):
            pixels[0, x] = COLOR_TIME_FILL if x < filled else COLOR_TIME_EMPTY

        source_color = COLOR_SOURCE_A if (self._anim_tick % 2 == 0) else COLOR_SOURCE_B
        target_color = (
            COLOR_TARGET_B
            if self._mode == "won"
            else (COLOR_TARGET_A if (self._anim_tick % 2 == 0) else COLOR_TARGET_B)
        )

        for y in range(1, HEIGHT):
            for x in range(WIDTH):
                pos = (x, y)
                base = self._base_char(pos)
                color = COLOR_EMPTY
                if base == "#":
                    color = COLOR_WALL
                elif base == "X":
                    color = COLOR_GROUND
                elif base == "S":
                    color = source_color
                elif base == "E":
                    color = target_color
                elif base in {"R", "Q"}:
                    color = COLOR_RELAY_ON if base in self._relays_on else COLOR_RELAY_OFF
                elif base in DIODE_COLOR:
                    color = DIODE_COLOR[base]
                elif base == "|":
                    group_idx = self._model.door_group_for_pos.get(pos)
                    if group_idx is not None and group_idx in self._door_opening:
                        rem = self._door_opening[group_idx]
                        color = COLOR_DOOR_CLOSED if (rem % 2 == 0) else COLOR_EMPTY
                    else:
                        color = COLOR_DOOR_CLOSED
                pixels[y, x] = color

        for x, y in self._wire_cells:
            pixels[y, x] = COLOR_WIRE
        for (x, y), _rem in self._trail_decay.items():
            if (x, y) in self._wire_cells:
                pixels[y, x] = COLOR_WIRE_TRAIL
        for x, y in self._wavefront:
            if (x, y) in self._wire_cells:
                pixels[y, x] = COLOR_WAVE

        for (x, y), rem in self._illegal_flash.items():
            if rem % 2 == 0:
                pixels[y, x] = COLOR_FLASH
        for (x, y), rem in self._diode_flash.items():
            if rem % 2 == 0:
                pixels[y, x] = COLOR_FLASH

        if self._mode == "failed":
            for (x, y), rem in self._fail_spark.items():
                pixels[y, x] = COLOR_FLASH if rem % 2 == 0 else COLOR_GROUND

        self._board.pixels = pixels

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

        if self._mode in {"won", "failed"}:
            if action_id == int(GameAction.ACTION6.value):
                if self._mode == "won":
                    self._advance_after_win_click()
                else:
                    self.lose()
            else:
                self._step_post_state()
                self._anim_tick += 1
                self._render_board()
            self.complete_action()
            return

        self._update_playing_state()

        if action_id == int(GameAction.ACTION6.value):
            self._pending_click = self._parse_click()

        self._anim_tick += 1
        self._render_board()
        self.complete_action()
