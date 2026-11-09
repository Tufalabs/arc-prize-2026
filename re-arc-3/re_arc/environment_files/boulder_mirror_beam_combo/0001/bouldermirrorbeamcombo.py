from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "boulder_mirror_beam_combo-0001"

COLOR_EMPTY = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER = 3
COLOR_BOULDER = 4
COLOR_EMITTER = 5
COLOR_MIRROR = 6
COLOR_MIRROR_PENDING = 7
COLOR_BEAM_WARM = 8
COLOR_BEAM_A = 9
COLOR_BEAM_B = 10
COLOR_RECEIVER_OFF = 11
COLOR_RECEIVER_A = 12
COLOR_RECEIVER_B = 13
COLOR_TIMEBAR_FILL = 14
COLOR_ALERT = 15

DIR_BY_ACTION = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

DIR_TO_IDX = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

# right, down, left, up
IDX_TO_DIR = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# orientation 0 => '/', orientation 1 => '\\'
REFLECT = {(0, 0): 3, (0, 1): 1, (1, 0): 2, (1, 1): 0, (2, 0): 1, (2, 1): 3, (3, 0): 0, (3, 1): 2}

LEVEL_LAYOUTS = [
    (
        "Level 1 — Block, cross, rotate, unblock",
        "generous",
        [
            "================================",
            "################################",
            "#..............##..............#",
            "#..............##........[]....#",
            "#..............##........[]....#",
            "#..............##..............#",
            "#..............##..............#",
            "#..............##..............#",
            "#..............##..............#",
            "#.>------------..---------\\....#",
            "#..............##.........|....#",
            "#......OO......##.........|....#",
            "#......OO......##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "#..@...........##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "################################",
        ],
    ),
    (
        "Level 2 — Two mirrors: route up, then right",
        "generous",
        [
            "================================",
            "################################",
            "#..............##..............#",
            "#..............##..........[]..#",
            "#..............##..........[]..#",
            "#..............##..............#",
            "#..............##.......\\..[]..#",
            "#..............##..............#",
            "#..............##..............#",
            "#.>------------..-------\\......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "#......OO......##.......|......#",
            "#......OO......##.......|......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "#..@...........##.......|......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "################################",
        ],
    ),
    (
        "Level 3 — Reach the blocker from the safe passage",
        "generous",
        [
            "====================================",
            "####################################",
            "#................##................#",
            "#................##.............[].#",
            "#................##.............[].#",
            "#................##...........\\.[].#",
            "#................##................#",
            "#................##................#",
            "#................##................#",
            "#.>--------------..-----------\\....#",
            "#................##...........|....#",
            "#................##....OO.....|....#",
            "#................##....OO.....|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "#..................................#",
            "#................##...........|....#",
            "#.......@........##...........|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "####################################",
        ],
    ),
    (
        "Level 4 — Three mirrors, two boulders",
        "moderate",
        [
            "================================",
            "################################",
            "#..............##..............#",
            "#..............##........[]....#",
            "#..............##........[]....#",
            "#..............##..............#",
            "#..............##..............#",
            "#..............##..............#",
            "#..............##..............#",
            "#.>------------..---------\\....#",
            "#..............##.........|....#",
            "#......OO......##.........|....#",
            "#......OO......##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "#..@...........##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "#..............##.........|....#",
            "################################",
        ],
    ),
    (
        "Level 5 — Four mirrors: long route, manage two boulders",
        "moderate",
        [
            "================================",
            "################################",
            "#..............##..............#",
            "#..............##..........[]..#",
            "#..............##..........[]..#",
            "#..............##..............#",
            "#..............##.......\\..[]..#",
            "#..............##..............#",
            "#..............##..............#",
            "#.>------------..-------\\......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "#......OO......##.......|......#",
            "#......OO......##.......|......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "#..@...........##.......|......#",
            "#..............##.......|......#",
            "#..............##.......|......#",
            "################################",
        ],
    ),
    (
        "Level 6 — Final: isolate, block twice, then commit",
        "tight",
        [
            "====================================",
            "####################################",
            "#................##................#",
            "#................##.............[].#",
            "#................##.............[].#",
            "#................##...........\\.[].#",
            "#................##................#",
            "#................##................#",
            "#................##................#",
            "#.>--------------..-----------\\....#",
            "#................##...........|....#",
            "#................##....OO.....|....#",
            "#................##....OO.....|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "#..................................#",
            "#................##...........|....#",
            "#.......@........##...........|....#",
            "#................##...........|....#",
            "#................##...........|....#",
            "####################################",
        ],
    ),
]

TIME_LIMIT_BY_BUCKET = {"generous": 220, "moderate": 180, "tight": 160}


@dataclass(frozen=True)
class ParsedLevel:
    name: str
    width: int
    height: int
    walls: tuple[tuple[int, int], ...]
    emitter: tuple[int, int]
    mirrors: tuple[tuple[int, int, int], ...]
    receiver_cells: tuple[tuple[int, int], ...]
    boulders: tuple[tuple[int, int], ...]
    player_start: tuple[int, int]
    time_limit: int


def _normalize_rows(rows: list[str]) -> list[str]:
    width = max(len(row) for row in rows)
    normalized: list[str] = []
    for y, row in enumerate(rows):
        if len(row) == width:
            normalized.append(row)
            continue
        pad_char = "=" if y == 0 else "#"
        normalized.append(row + (pad_char * (width - len(row))))
    return normalized


def _parse_layout(name: str, bucket: str, rows: list[str]) -> ParsedLevel:
    grid = _normalize_rows(rows)
    width = len(grid[0])
    height = len(grid)

    walls: set[tuple[int, int]] = set()
    mirrors: list[tuple[int, int, int]] = []
    receiver_cells: list[tuple[int, int]] = []
    boulder_cells: set[tuple[int, int]] = set()

    emitter: tuple[int, int] | None = None
    player_start: tuple[int, int] | None = None

    for y, row in enumerate(grid):
        for x, glyph in enumerate(row):
            if y == 0:
                continue
            if glyph == "#":
                walls.add((x, y))
            elif glyph == ">":
                if emitter is not None:
                    raise ValueError(f"{name}: multiple emitters")
                emitter = (x, y)
            elif glyph == "@":
                if player_start is not None:
                    raise ValueError(f"{name}: multiple player starts")
                player_start = (x, y)
            elif glyph == "/":
                mirrors.append((x, y, 0))
            elif glyph == "\\":
                mirrors.append((x, y, 1))
            elif glyph in {"[", "]"}:
                receiver_cells.append((x, y))
            elif glyph == "O":
                boulder_cells.add((x, y))
            elif glyph in {".", "-", "|", "+", "="}:
                pass
            else:
                raise ValueError(f"{name}: unsupported glyph {glyph!r} at {(x, y)}")

    if emitter is None:
        raise ValueError(f"{name}: missing emitter")
    if player_start is None:
        raise ValueError(f"{name}: missing player")
    if not receiver_cells:
        raise ValueError(f"{name}: missing receiver")

    boulders: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    for cell in sorted(boulder_cells):
        if cell in visited:
            continue
        cx, cy = cell
        block = {(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)}
        if not block.issubset(boulder_cells):
            raise ValueError(f"{name}: boulder must be exactly 2x2 near {(cx, cy)}")
        boulders.append((cx, cy))
        visited.update(block)

    for x, y in receiver_cells:
        if (x, y) in walls:
            raise ValueError(f"{name}: receiver overlaps wall at {(x, y)}")

    time_limit = int(TIME_LIMIT_BY_BUCKET[bucket])

    return ParsedLevel(
        name=name,
        width=width,
        height=height,
        walls=tuple(sorted(walls)),
        emitter=emitter,
        mirrors=tuple(sorted(mirrors, key=lambda item: (item[1], item[0]))),
        receiver_cells=tuple(sorted(set(receiver_cells))),
        boulders=tuple(sorted(boulders)),
        player_start=player_start,
        time_limit=time_limit,
    )


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _build_level(parsed: ParsedLevel, idx: int) -> Level:
    board = Sprite(
        pixels=_solid(parsed.width, parsed.height, COLOR_FLOOR),
        name="board",
        x=0,
        y=0,
        layer=0,
        tags=["board"],
        collidable=False,
    )
    return Level(
        name=f"{parsed.name} ({idx + 1})",
        grid_size=(parsed.width, parsed.height),
        sprites=[board],
        data={
            "name": parsed.name,
            "width": parsed.width,
            "height": parsed.height,
            "walls": [list(cell) for cell in parsed.walls],
            "emitter": list(parsed.emitter),
            "mirrors": [{"x": x, "y": y, "orientation": o} for x, y, o in parsed.mirrors],
            "receiver_cells": [list(cell) for cell in parsed.receiver_cells],
            "boulders": [list(cell) for cell in parsed.boulders],
            "player_start": list(parsed.player_start),
            "time_limit": parsed.time_limit,
            "spec": {
                "objective": "light_receiver",
                "mechanics": [
                    "move",
                    "push_2x2_boulder",
                    "adjacent_click_rotate_mirror",
                    "beam_warm_then_lethal",
                    "timebar_top_row",
                ],
                "available_actions": [1, 2, 3, 4, 5, 6],
            },
        },
    )


def _cell_bit(width: int, x: int, y: int) -> int:
    return 1 << (y * width + x)


def _iter_boulder_cells(top_left: tuple[int, int]):
    bx, by = top_left
    yield (bx, by)
    yield (bx + 1, by)
    yield (bx, by + 1)
    yield (bx + 1, by + 1)


class BoulderMirrorBeamCombo(ARCBaseGame):
    def __init__(self, seed: int = 0):
        parsed_levels = [_parse_layout(name, bucket, rows) for name, bucket, rows in LEVEL_LAYOUTS]
        levels = [_build_level(parsed, idx) for idx, parsed in enumerate(parsed_levels)]
        max_width = max(level.grid_size[0] for level in levels)
        max_height = max(level.grid_size[1] for level in levels)

        camera = Camera(width=max_width, height=max_height, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 6],
            seed=seed,
        )

        self._board: Sprite | None = None
        self._width = 0
        self._height = 0

        self._walls: set[tuple[int, int]] = set()
        self._emitter = (0, 0)
        self._receiver_cells: set[tuple[int, int]] = set()

        self._mirror_cells: list[tuple[int, int]] = []
        self._mirror_orient_start_mask = 0

        self._boulders_start: tuple[tuple[int, int], ...] = tuple()
        self._player_start = (0, 0)
        self._time_limit = 1

        self._solid_static: set[tuple[int, int]] = set()

        self._player_pos = (0, 0)
        self._boulders: tuple[tuple[int, int], ...] = tuple()
        self._mirror_mask = 0
        self._pending_rotation_idx = -1
        self._click_feedback_idx = -1

        self._time_left = 1
        self._tick = 0
        self._route_score = 0

        self._prev_beam_bits = 0
        self._beam_bits = 0
        self._receiver_lit = False

    def on_set_level(self, level: Level) -> None:
        boards = level.get_sprites_by_tag("board")
        if not boards:
            raise RuntimeError("boulder_mirror_beam_combo level missing board sprite")
        self._board = boards[0]

        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))

        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._emitter = tuple(int(v) for v in level.get_data("emitter"))
        self._receiver_cells = {tuple(int(v) for v in item) for item in (level.get_data("receiver_cells") or [])}

        mirror_entries = list(level.get_data("mirrors") or [])
        self._mirror_cells = []
        self._mirror_orient_start_mask = 0
        for idx, entry in enumerate(mirror_entries):
            x = int(entry["x"])
            y = int(entry["y"])
            orientation = int(entry["orientation"]) & 1
            self._mirror_cells.append((x, y))
            if orientation:
                self._mirror_orient_start_mask |= 1 << idx

        self._boulders_start = tuple(sorted(tuple(int(v) for v in item) for item in (level.get_data("boulders") or [])))
        self._player_start = tuple(int(v) for v in level.get_data("player_start"))
        self._time_limit = max(1, int(level.get_data("time_limit") or 1))

        self._solid_static = set(self._walls)
        self._solid_static.add(self._emitter)
        self._solid_static.update(self._receiver_cells)
        self._solid_static.update(self._mirror_cells)

        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._player_pos = self._player_start
        self._boulders = tuple(self._boulders_start)
        self._mirror_mask = int(self._mirror_orient_start_mask)
        self._pending_rotation_idx = -1
        self._click_feedback_idx = -1

        self._time_left = int(self._time_limit)
        self._prev_beam_bits = 0
        self._beam_bits = 0
        self._receiver_lit = False

        self._render_board(warm_bits=0, lethal_bits=0, death_flash=False)

    def _mirror_orientation(self, idx: int, mask: int | None = None) -> int:
        state = self._mirror_mask if mask is None else int(mask)
        return 1 if (state & (1 << idx)) else 0

    def _boulder_cells(self, boulders: tuple[tuple[int, int], ...] | None = None) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for top_left in self._boulders if boulders is None else boulders:
            for cell in _iter_boulder_cells(top_left):
                cells.add(cell)
        return cells

    def _find_boulder_at_in(self, x: int, y: int, boulders: tuple[tuple[int, int], ...]) -> int:
        for idx, top_left in enumerate(boulders):
            if (x, y) in set(_iter_boulder_cells(top_left)):
                return idx
        return -1

    def _can_place_boulder_in(
        self, top_left: tuple[int, int], *, ignore_idx: int, boulders: tuple[tuple[int, int], ...]
    ) -> bool:
        ox, oy = top_left
        candidate = set(_iter_boulder_cells((ox, oy)))
        for x, y in candidate:
            if not self._in_bounds(x, y):
                return False
            if (x, y) in self._solid_static:
                return False
        for idx, other in enumerate(boulders):
            if idx == ignore_idx:
                continue
            if candidate.intersection(set(_iter_boulder_cells(other))):
                return False
        return True

    def _trace_beam_bits(self, *, mirror_mask: int, boulders: tuple[tuple[int, int], ...]) -> tuple[int, bool]:
        boulder_cells = self._boulder_cells(boulders)
        mirror_index = {cell: idx for idx, cell in enumerate(self._mirror_cells)}

        x, y = self._emitter
        direction = 0  # right
        seen: set[tuple[int, int, int]] = set()

        beam_bits = 0
        receiver_lit = False

        while True:
            state = (x, y, direction)
            if state in seen:
                break
            seen.add(state)

            dx, dy = IDX_TO_DIR[direction]
            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= self._width or ny >= self._height:
                break
            cell = (nx, ny)
            if cell in self._walls:
                break
            if cell in boulder_cells:
                break
            if cell in self._receiver_cells:
                receiver_lit = True
                break

            beam_bits |= _cell_bit(self._width, nx, ny)

            mirror_idx = mirror_index.get(cell)
            if mirror_idx is not None:
                orient = self._mirror_orientation(mirror_idx, mirror_mask)
                direction = REFLECT[(direction, orient)]

            x, y = nx, ny

        return beam_bits, receiver_lit

    def _player_on_lethal(self, lethal_bits: int) -> bool:
        px, py = self._player_pos
        return bool(lethal_bits & _cell_bit(self._width, px, py))

    def _find_boulder_at(self, x: int, y: int) -> int:
        return self._find_boulder_at_in(x, y, self._boulders)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 1 <= y < self._height

    def _can_place_boulder(
        self, top_left: tuple[int, int], *, ignore_idx: int, boulders: tuple[tuple[int, int], ...]
    ) -> bool:
        return self._can_place_boulder_in(top_left, ignore_idx=ignore_idx, boulders=boulders)

    def _solver_initial_state(self) -> tuple:
        return (
            self._player_start[0],
            self._player_start[1],
            tuple(self._boulders_start),
            int(self._mirror_orient_start_mask),
            -1,
            0,
            int(self._time_limit),
        )

    def _solver_adjacent_mirrors(self, px: int, py: int) -> list[int]:
        out: list[int] = []
        for idx, (mx, my) in enumerate(self._mirror_cells):
            if abs(px - mx) + abs(py - my) == 1:
                out.append(idx)
        return out

    def _solver_step_state(
        self, state: tuple, action_id: int, click_idx: int | None = None
    ) -> tuple[tuple | None, bool]:
        px, py, boulders, mirror_mask, pending_idx, prev_beam_bits, time_left = state
        npx, npy = int(px), int(py)
        nboulders = tuple(boulders)

        if int(action_id) in DIR_BY_ACTION:
            dx, dy = DIR_BY_ACTION[int(action_id)]
            tx, ty = npx + dx, npy + dy
            if self._in_bounds(tx, ty):
                boulder_idx = self._find_boulder_at_in(tx, ty, nboulders)
                if boulder_idx >= 0:
                    old_top = nboulders[boulder_idx]
                    new_top = (old_top[0] + dx, old_top[1] + dy)
                    if self._can_place_boulder_in(new_top, ignore_idx=boulder_idx, boulders=nboulders):
                        mutable = list(nboulders)
                        mutable[boulder_idx] = new_top
                        nboulders = tuple(mutable)
                        npx, npy = tx, ty
                elif (tx, ty) not in self._solid_static:
                    npx, npy = tx, ty

        nmirror_mask = int(mirror_mask)
        if int(pending_idx) >= 0:
            nmirror_mask ^= 1 << int(pending_idx)

        beam_bits, receiver_lit = self._trace_beam_bits(mirror_mask=nmirror_mask, boulders=nboulders)
        lethal_bits = int(beam_bits) & int(prev_beam_bits)

        if lethal_bits & _cell_bit(self._width, npx, npy):
            return None, False

        if receiver_lit:
            return None, True

        ntime = int(time_left) - 1
        if ntime <= 0:
            return None, False

        npending = -1
        if int(action_id) == int(GameAction.ACTION6.value) and click_idx is not None:
            idx = int(click_idx)
            if 0 <= idx < len(self._mirror_cells):
                mx, my = self._mirror_cells[idx]
                if abs(npx - mx) + abs(npy - my) == 1:
                    npending = idx

        next_state = (npx, npy, nboulders, nmirror_mask, npending, int(beam_bits), ntime)
        return next_state, False

    def _try_move_player(self, dx: int, dy: int) -> None:
        px, py = self._player_pos
        nx, ny = px + dx, py + dy

        if not self._in_bounds(nx, ny):
            return

        boulder_idx = self._find_boulder_at(nx, ny)
        if boulder_idx >= 0:
            old_top_left = self._boulders[boulder_idx]
            new_top_left = (old_top_left[0] + dx, old_top_left[1] + dy)
            if not self._can_place_boulder(new_top_left, ignore_idx=boulder_idx, boulders=self._boulders):
                return
            mutable = list(self._boulders)
            mutable[boulder_idx] = new_top_left
            self._boulders = tuple(mutable)
            self._player_pos = (nx, ny)
            return

        if (nx, ny) in self._solid_static:
            return

        self._player_pos = (nx, ny)

    def _decode_click_target(self) -> int:
        data = self.action.data or {}
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return -1
        grid_pos = self.camera.display_to_grid(raw_x, raw_y)
        if grid_pos is None:
            return -1
        gx = int(grid_pos[0])
        gy = int(grid_pos[1])

        px, py = self._player_pos
        if abs(px - gx) + abs(py - gy) != 1:
            return -1

        for idx, (mx, my) in enumerate(self._mirror_cells):
            if gx == mx and gy == my:
                return idx
        return -1

    def _timebar_color(self) -> int:
        if self._time_left <= max(1, self._time_limit // 4):
            return COLOR_ALERT
        return COLOR_TIMEBAR_FILL

    def _render_board(self, *, warm_bits: int, lethal_bits: int, death_flash: bool) -> None:
        if self._board is None:
            return

        canvas = np.full((self._height, self._width), COLOR_FLOOR, dtype=np.int8)

        for x, y in self._walls:
            canvas[y, x] = COLOR_WALL

        if death_flash:
            flash_color = COLOR_ALERT if (self._tick % 2 == 0) else COLOR_BEAM_A
            canvas[1:, :] = np.where(canvas[1:, :] == COLOR_WALL, COLOR_WALL, flash_color)
        else:
            for y in range(1, self._height):
                row_offset = y * self._width
                for x in range(self._width):
                    bit = 1 << (row_offset + x)
                    if warm_bits & bit:
                        canvas[y, x] = COLOR_BEAM_WARM
                    elif lethal_bits & bit:
                        canvas[y, x] = COLOR_BEAM_A if (self._tick % 2 == 0) else COLOR_BEAM_B

        ex, ey = self._emitter
        canvas[ey, ex] = COLOR_EMITTER

        receiver_color = COLOR_RECEIVER_OFF
        if self._receiver_lit and not death_flash:
            receiver_color = COLOR_RECEIVER_A if (self._tick % 2 == 0) else COLOR_RECEIVER_B
        for x, y in self._receiver_cells:
            canvas[y, x] = receiver_color

        for idx, (mx, my) in enumerate(self._mirror_cells):
            if idx == self._click_feedback_idx and not death_flash:
                canvas[my, mx] = COLOR_MIRROR_PENDING
            else:
                canvas[my, mx] = COLOR_MIRROR

        for top_left in self._boulders:
            for bx, by in _iter_boulder_cells(top_left):
                canvas[by, bx] = COLOR_BOULDER

        px, py = self._player_pos
        canvas[py, px] = COLOR_ALERT if death_flash else COLOR_PLAYER

        fill = round(self._width * (float(self._time_left) / float(self._time_limit)))
        fill = max(0, min(fill, self._width))
        canvas[0, :] = COLOR_EMPTY
        if fill > 0:
            canvas[0, :fill] = self._timebar_color() if not death_flash else COLOR_ALERT

        self._board.pixels = canvas

    def _lose_level(self, *, click_feedback_idx: int) -> None:
        self._route_score += 1
        self._click_feedback_idx = click_feedback_idx
        self._tick += 1
        self._render_board(warm_bits=0, lethal_bits=0, death_flash=True)
        self.lose()

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
        clicked_idx = -1

        if action_id in DIR_BY_ACTION:
            dx, dy = DIR_BY_ACTION[action_id]
            self._try_move_player(dx, dy)
        elif action_id == int(GameAction.ACTION6.value):
            clicked_idx = self._decode_click_target()

        previous_beam = self._prev_beam_bits

        if self._pending_rotation_idx >= 0:
            self._mirror_mask ^= 1 << self._pending_rotation_idx
        self._pending_rotation_idx = -1

        beam_bits, receiver_lit = self._trace_beam_bits(mirror_mask=self._mirror_mask, boulders=self._boulders)

        warm_bits = beam_bits & (~previous_beam)
        lethal_bits = beam_bits & previous_beam

        self._receiver_lit = bool(receiver_lit)
        self._beam_bits = int(beam_bits)

        if self._player_on_lethal(lethal_bits):
            self._lose_level(click_feedback_idx=clicked_idx)
            self.complete_action()
            return

        if self._receiver_lit:
            self.next_level()
            self.complete_action()
            return

        self._time_left -= 1
        if self._time_left <= 0:
            self._lose_level(click_feedback_idx=clicked_idx)
            self.complete_action()
            return

        self._pending_rotation_idx = clicked_idx
        self._click_feedback_idx = clicked_idx
        self._prev_beam_bits = beam_bits

        self._tick += 1
        self._render_board(warm_bits=warm_bits, lethal_bits=lethal_bits, death_flash=False)
        self.complete_action()
