from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pipe_connect_rotate_tiles-0001"
GRID_SIZE = 64

COLOR_BG = 0
COLOR_WALL = 1
COLOR_TIME_FULL = 2
COLOR_TIME_PARTIAL = 3
COLOR_PIPE = 4
COLOR_PIPE_FILLED = 5
COLOR_PIPE_FRONTIER = 6
COLOR_SOURCE = 7
COLOR_SINK = 8
COLOR_SINK_ON = 9
COLOR_LOCKED = 10
COLOR_LOCKED_FILLED = 11
COLOR_LEAK_A = 12
COLOR_LEAK_B = 13
COLOR_FAIL = 14

U = 1
R = 2
D = 4
L = 8
DIRS = [U, R, D, L]
OPPOSITE = {U: D, R: L, D: U, L: R}
DELTA = {U: (0, -1), R: (1, 0), D: (0, 1), L: (-1, 0)}

ROTATABLE_MASKS = {
    "─": L | R,
    "│": U | D,
    "└": U | R,
    "┌": R | D,
    "┐": D | L,
    "┘": L | U,
    "├": U | D | R,
    "┬": L | R | D,
    "┤": U | D | L,
    "┴": L | R | U,
    "┼": U | D | L | R,
}

LOCKED_MASKS = {
    "═": L | R,
    "║": U | D,
    "╚": U | R,
    "╔": R | D,
    "╗": D | L,
    "╝": L | U,
    "╠": U | D | R,
    "╦": L | R | D,
    "╣": U | D | L,
    "╩": L | R | U,
    "╬": U | D | L | R,
}

ALL_PIPE_MASKS = dict(ROTATABLE_MASKS)
ALL_PIPE_MASKS.update(LOCKED_MASKS)

SOURCE_PARTS = {"▛", "▜", "▙", "▟"}
SINK_PARTS = {"◰", "◳", "◲", "◱"}

NOOP_CLICK = (-1, -1)


@dataclass(frozen=True)
class LevelSpec:
    rows: tuple[str, ...]
    time_left_ticks: int
    leak_penalty_ticks: int
    click_cost_ticks: int


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓",
            "############",
            "#..........#",
            "#▛▜.....┌◰◳#",
            "#▙▟─────┐◲◱#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "############",
        ),
        time_left_ticks=12 * 8,
        leak_penalty_ticks=6,
        click_cost_ticks=0,
    ),
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓▓▓",
            "##############",
            "#............#",
            "#............#",
            "#▛▜........◰◳#",
            "#▙▟──┴──┴──◲◱#",
            "#....└──┘....#",
            "#............#",
            "#............#",
            "#............#",
            "##############",
        ),
        time_left_ticks=14 * 10,
        leak_penalty_ticks=8,
        click_cost_ticks=0,
    ),
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓",
            "################",
            "#..............#",
            "#..............#",
            "#.....┌─┐......#",
            "#▛▜...│.│....◰◳#",
            "#▙▟───┤─┼─┬──◲◱#",
            "#.......│.│....#",
            "#.......└─┘....#",
            "#..............#",
            "#..............#",
            "#..............#",
            "################",
        ),
        time_left_ticks=16 * 12,
        leak_penalty_ticks=10,
        click_cost_ticks=1,
    ),
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓",
            "##################",
            "#................#",
            "#................#",
            "#.....╔═════╗....#",
            "#.....║.....│....#",
            "#.....║.....│....#",
            "#▛▜...║.....│..◰◳#",
            "#▙▟───╣.....┤──◲◱#",
            "#.....║.....│....#",
            "#.....┌─────┐....#",
            "#................#",
            "#................#",
            "#................#",
            "##################",
        ),
        time_left_ticks=18 * 12,
        leak_penalty_ticks=12,
        click_cost_ticks=1,
    ),
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓",
            "####################",
            "#..................#",
            "#..................#",
            "#.........┐──────◰◳#",
            "#.........│......◲◱#",
            "#.........│........#",
            "#.........│........#",
            "#▛▜.......─........#",
            "#▙▟──│────┬........#",
            "#.........│........#",
            "#.........│........#",
            "#.........│......◰◳#",
            "#.........┘──────◲◱#",
            "#..................#",
            "#..................#",
            "####################",
        ),
        time_left_ticks=20 * 14,
        leak_penalty_ticks=12,
        click_cost_ticks=1,
    ),
    LevelSpec(
        rows=(
            "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓",
            "########################",
            "#......................#",
            "#......................#",
            "#......................#",
            "#....................◰◳#",
            "#...........╔═════┤──◲◱#",
            "#...........║.....│....#",
            "#...........║.....│....#",
            "#▛▜.........║.....│....#",
            "#▙▟═════════╬═════╣....#",
            "#...........║.....│....#",
            "#...........║.....│....#",
            "#...........║.....│..◰◳#",
            "#...........╚═════├──◲◱#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
        time_left_ticks=24 * 14,
        leak_penalty_ticks=15,
        click_cost_ticks=1,
    ),
)


def rotate_mask_cw(mask: int) -> int:
    out = 0
    if mask & U:
        out |= R
    if mask & R:
        out |= D
    if mask & D:
        out |= L
    if mask & L:
        out |= U
    return out


def render_layout_geometry(rows: tuple[str, ...]) -> dict:
    height = len(rows)
    width = len(rows[0])
    tile = max(1, min(GRID_SIZE // max(1, width), GRID_SIZE // max(1, height)))
    x0 = (GRID_SIZE - width * tile) // 2
    y0 = (GRID_SIZE - height * tile) // 2

    base_chars: dict[tuple[int, int], str] = {}
    pipe_masks: dict[tuple[int, int], int] = {}
    rotatable_cells: list[tuple[int, int]] = []
    source_output_cells: set[tuple[int, int]] = set()
    sink_inlet_cells: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            pos = (x, y)
            base_chars[pos] = ch
            if ch in ROTATABLE_MASKS:
                pipe_masks[pos] = ROTATABLE_MASKS[ch]
                rotatable_cells.append(pos)
            elif ch in LOCKED_MASKS:
                pipe_masks[pos] = LOCKED_MASKS[ch]

            if ch in {"▜", "▟"}:
                source_output_cells.add(pos)
            if ch in {"◰", "◲"}:
                sink_inlet_cells.add(pos)

    sink_cells = {(x, y) for (x, y), ch in base_chars.items() if ch in SINK_PARTS}

    sink_objects: list[tuple[tuple[int, int], ...]] = []
    sink_inlet_to_obj: dict[tuple[int, int], int] = {}
    for (x, y), ch in base_chars.items():
        if ch != "◰":
            continue
        bottom = (x, y + 1)
        if base_chars.get(bottom) != "◲":
            continue
        obj_idx = len(sink_objects)
        inlets = ((x, y), bottom)
        sink_objects.append(inlets)
        sink_inlet_to_obj[(x, y)] = obj_idx
        sink_inlet_to_obj[bottom] = obj_idx

    return {
        "rows": rows,
        "width": width,
        "height": height,
        "tile": tile,
        "x0": x0,
        "y0": y0,
        "base_chars": base_chars,
        "pipe_masks": pipe_masks,
        "rotatable_cells": tuple(sorted(rotatable_cells)),
        "source_output_cells": frozenset(source_output_cells),
        "sink_inlet_cells": frozenset(sink_inlet_cells),
        "sink_objects": tuple(sink_objects),
        "sink_inlet_to_obj": sink_inlet_to_obj,
        "sink_cells": frozenset(sink_cells),
    }


def _inside(geo: dict, x: int, y: int) -> bool:
    return 0 <= x < int(geo["width"]) and 0 <= y < int(geo["height"])


def _pipe_mask(geo: dict, pos: tuple[int, int], orient: tuple[int, ...]) -> int:
    idx = int(geo["rotatable_index"].get(pos, -1))
    if idx >= 0:
        base = int(geo["rotatable_base_masks"][idx])
        turns = int(orient[idx] & 3)
        out = base
        for _ in range(turns):
            out = rotate_mask_cw(out)
        return out
    return int(geo["pipe_masks"].get(pos, 0))


def _special_mask(geo: dict, pos: tuple[int, int]) -> int:
    if pos in geo["source_output_cells"]:
        return R
    if pos in geo["sink_inlet_cells"]:
        return L
    return 0


def _reciprocates(geo: dict, orient: tuple[int, ...], pos: tuple[int, int], direction: int) -> bool:
    dx, dy = DELTA[direction]
    nbr = (pos[0] + dx, pos[1] + dy)
    if not _inside(geo, nbr[0], nbr[1]):
        return False

    nmask = _pipe_mask(geo, nbr, orient)
    if nmask:
        return bool(nmask & OPPOSITE[direction])

    smask = _special_mask(geo, nbr)
    if smask:
        return bool(smask & OPPOSITE[direction])

    return False


def compute_distances(geo: dict, orient: tuple[int, ...]) -> dict[tuple[int, int], int]:
    start_cells: list[tuple[int, int]] = []
    for sx, sy in geo["source_output_cells"]:
        nbr = (sx + 1, sy)
        if not _inside(geo, nbr[0], nbr[1]):
            continue
        nmask = _pipe_mask(geo, nbr, orient)
        if nmask & L:
            start_cells.append(nbr)

    dist: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int]] = deque()

    for cell in start_cells:
        if cell in dist:
            continue
        dist[cell] = 0
        queue.append(cell)

    while queue:
        cx, cy = queue.popleft()
        cmask = _pipe_mask(geo, (cx, cy), orient)
        if not cmask:
            continue
        cdist = dist[(cx, cy)]
        for direction in DIRS:
            if not (cmask & direction):
                continue
            dx, dy = DELTA[direction]
            nbr = (cx + dx, cy + dy)
            if not _inside(geo, nbr[0], nbr[1]):
                continue
            nmask = _pipe_mask(geo, nbr, orient)
            if not nmask or not (nmask & OPPOSITE[direction]):
                continue
            if nbr in dist:
                continue
            dist[nbr] = cdist + 1
            queue.append(nbr)

    return dist


def leak_cells_from_filled(geo: dict, orient: tuple[int, ...], filled: set[tuple[int, int]]) -> set[tuple[int, int]]:
    leaks: set[tuple[int, int]] = set()
    for cell in filled:
        cmask = _pipe_mask(geo, cell, orient)
        if not cmask:
            continue
        for direction in DIRS:
            if not (cmask & direction):
                continue
            if not _reciprocates(geo, orient, cell, direction):
                leaks.add(cell)
                break
    return leaks


def supplied_sinks(geo: dict, orient: tuple[int, ...], filled: set[tuple[int, int]]) -> set[int]:
    supplied: set[int] = set()
    for sink_cell in geo["sink_inlet_cells"]:
        left = (sink_cell[0] - 1, sink_cell[1])
        if left not in filled:
            continue
        pmask = _pipe_mask(geo, left, orient)
        if pmask & R:
            obj_idx = geo["sink_inlet_to_obj"].get(sink_cell)
            if obj_idx is not None:
                supplied.add(int(obj_idx))
    return supplied


def is_static_solution(geo: dict, orient: tuple[int, ...]) -> bool:
    dist = compute_distances(geo, orient)
    if not dist:
        return False
    filled = set(dist.keys())
    if leak_cells_from_filled(geo, orient, filled):
        return False
    got = supplied_sinks(geo, orient, filled)
    need = set(range(len(geo["sink_objects"])))
    return need.issubset(got)


def can_supply_without_pre_sink_leak(geo: dict, orient: tuple[int, ...]) -> bool:
    """
    True when water can reach all sink inlets before the first leak event.

    The game checks leaks before win each step, so a leak at or before the
    sink-reaching frontier prevents completion for that wave.
    """
    dist = compute_distances(geo, orient)
    if not dist:
        return False

    need = set(range(len(geo["sink_objects"])))
    max_dist = max(dist.values())
    for radius in range(max_dist + 1):
        filled = {cell for cell, d in dist.items() if d <= radius}
        if leak_cells_from_filled(geo, orient, filled):
            return False
        got = supplied_sinks(geo, orient, filled)
        if need.issubset(got):
            return True
    return False


def logical_cell_to_display_click(geo: dict, gx: int, gy: int) -> dict[str, int]:
    tile = int(geo["tile"])
    x = int(geo["x0"]) + gx * tile + tile // 2
    y = int(geo["y0"]) + gy * tile + tile // 2
    return {"x": int(x), "y": int(y)}


def _draw_pipe(canvas: np.ndarray, px: int, py: int, tile: int, mask: int, color: int, bg: int) -> None:
    canvas[py : py + tile, px : px + tile] = np.int8(bg)
    cx = px + tile // 2
    cy = py + tile // 2
    thickness = max(1, tile // 3)
    half = thickness // 2

    x1 = max(px, cx - half)
    x2 = min(px + tile, cx - half + thickness)
    y1 = max(py, cy - half)
    y2 = min(py + tile, cy - half + thickness)
    canvas[y1:y2, x1:x2] = np.int8(color)

    if mask & U:
        canvas[py : cy + 1, x1:x2] = np.int8(color)
    if mask & D:
        canvas[cy : py + tile, x1:x2] = np.int8(color)
    if mask & L:
        canvas[y1:y2, px : cx + 1] = np.int8(color)
    if mask & R:
        canvas[y1:y2, cx : px + tile] = np.int8(color)


def _build_level(level_idx: int, spec: LevelSpec) -> Level:
    geo = render_layout_geometry(spec.rows)
    data = {
        "level_idx": int(level_idx),
        "rows": list(spec.rows),
        "time_left_ticks": int(spec.time_left_ticks),
        "leak_penalty_ticks": int(spec.leak_penalty_ticks),
        "click_cost_ticks": int(spec.click_cost_ticks),
        "tile": int(geo["tile"]),
        "x0": int(geo["x0"]),
        "y0": int(geo["y0"]),
        "width": int(geo["width"]),
        "height": int(geo["height"]),
        "rotatable_cells": [list(pos) for pos in geo["rotatable_cells"]],
    }

    board = Sprite(
        pixels=np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=1,
        tags=["board", "sys_click", "sys_every_pixel"],
        collidable=False,
    )

    return Level(grid_size=(GRID_SIZE, GRID_SIZE), sprites=[board], data=data)


class PipeConnectRotateTiles(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID, levels=levels, camera=camera, win_score=len(levels), available_actions=[5, 6], seed=seed
        )

        self._route_score = 0
        self._geo: dict = {}
        self._orientation: list[int] = []
        self._queued_actions: list[tuple[str, tuple[int, int] | None]] = []

        self._time_limit_ticks = 0
        self._time_left_ticks = 0
        self._leak_penalty_ticks = 0
        self._click_cost_ticks = 0
        self._flow_radius = 0
        self._flow_phase = 0
        self._leak_anim_timer = 0
        self._leak_cells: set[tuple[int, int]] = set()
        self._distances: dict[tuple[int, int], int] = {}
        self._filled_cells: set[tuple[int, int]] = set()
        self._frontier_cells: set[tuple[int, int]] = set()
        self._sink_supplied_ids: set[int] = set()
        self._cell_flash: dict[tuple[int, int], int] = {}
        self._fail_flash_timer = 0

    @staticmethod
    def _parse_geo(level: Level) -> dict:
        rows = tuple(str(row) for row in (level.get_data("rows") or ()))
        geo = render_layout_geometry(rows)
        rot_cells = list(geo["rotatable_cells"])
        geo["rotatable_index"] = {cell: idx for idx, cell in enumerate(rot_cells)}
        geo["rotatable_base_masks"] = tuple(int(geo["pipe_masks"][cell]) for cell in rot_cells)
        return geo

    def on_set_level(self, level: Level) -> None:
        self._geo = self._parse_geo(level)
        self._orientation = [0 for _ in self._geo["rotatable_cells"]]
        self._queued_actions = []

        self._time_limit_ticks = int(level.get_data("time_left_ticks") or 1)
        self._time_left_ticks = self._time_limit_ticks
        self._leak_penalty_ticks = int(level.get_data("leak_penalty_ticks") or 0)
        self._click_cost_ticks = int(level.get_data("click_cost_ticks") or 0)

        self._flow_radius = 0
        self._flow_phase = 0
        self._leak_anim_timer = 0
        self._leak_cells = set()
        self._distances = {}
        self._filled_cells = set()
        self._frontier_cells = set()
        self._sink_supplied_ids = set()
        self._cell_flash = {}
        self._fail_flash_timer = 0

        self._recompute_flow()
        self._sync_board()

    def _orient_tuple(self) -> tuple[int, ...]:
        return tuple(int(v & 3) for v in self._orientation)

    def _pipe_mask_cell(self, pos: tuple[int, int]) -> int:
        return _pipe_mask(self._geo, pos, self._orient_tuple())

    def _click_to_logical(self, display_x: int, display_y: int) -> tuple[int, int] | None:
        grid_pos = self.camera.display_to_grid(int(display_x), int(display_y))
        if not isinstance(grid_pos, (tuple, list)) or len(grid_pos) != 2:
            return None
        gx, gy = int(grid_pos[0]), int(grid_pos[1])
        tile = int(self._geo["tile"])
        x0 = int(self._geo["x0"])
        y0 = int(self._geo["y0"])
        if gx < x0 or gy < y0:
            return None
        lx = (gx - x0) // tile
        ly = (gy - y0) // tile
        if not _inside(self._geo, lx, ly):
            return None
        return (int(lx), int(ly))

    def _queue_current_action(self) -> None:
        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION5.value):
            self._queued_actions.append(("restart", None))
            return

        if action_id != int(GameAction.ACTION6.value):
            return

        payload = self.action.data if isinstance(self.action.data, dict) else {}
        try:
            dx = int(payload.get("x", -1))
            dy = int(payload.get("y", -1))
        except (TypeError, ValueError):
            dx, dy = NOOP_CLICK

        logical = self._click_to_logical(dx, dy)
        if logical is None:
            return

        self._queued_actions.append(("click", logical))

    def _restart_level_state(self) -> None:
        self._orientation = [0 for _ in self._geo["rotatable_cells"]]
        self._time_left_ticks = self._time_limit_ticks
        self._flow_radius = 0
        self._flow_phase = 0
        self._leak_anim_timer = 0
        self._leak_cells = set()
        self._distances = {}
        self._filled_cells = set()
        self._frontier_cells = set()
        self._sink_supplied_ids = set()
        self._cell_flash = {}
        self._fail_flash_timer = 0
        self._queued_actions = []
        self._recompute_flow()

    def _apply_queued_actions(self, actions: list[tuple[str, tuple[int, int] | None]]) -> int:
        click_penalty = 0

        for kind, data in actions:
            if kind == "restart":
                self._restart_level_state()
                return 0

            if kind != "click" or data is None:
                continue

            x, y = data
            pos = (int(x), int(y))
            ch = self._geo["base_chars"].get(pos, ".")

            if ch in ROTATABLE_MASKS:
                idx = self._geo["rotatable_index"].get(pos)
                if idx is not None:
                    self._orientation[int(idx)] = (int(self._orientation[int(idx)]) + 1) & 3
                click_penalty += self._click_cost_ticks
            else:
                self._cell_flash[pos] = 2

        return click_penalty

    def _recompute_flow(self) -> None:
        orient = self._orient_tuple()
        self._distances = compute_distances(self._geo, orient)

        if self._distances:
            max_dist = max(self._distances.values())
            self._flow_radius = min(max_dist, max(0, int(self._flow_radius)))
            self._filled_cells = {cell for cell, d in self._distances.items() if d <= self._flow_radius}
            self._frontier_cells = {cell for cell, d in self._distances.items() if d == self._flow_radius}
        else:
            self._flow_radius = 0
            self._filled_cells = set()
            self._frontier_cells = set()

        self._sink_supplied_ids = supplied_sinks(self._geo, orient, self._filled_cells)

    def _tick_normal(self, actions_to_apply: list[tuple[str, tuple[int, int] | None]]) -> None:
        click_penalty = self._apply_queued_actions(actions_to_apply)

        self._time_left_ticks -= 1 + int(click_penalty)

        if self._leak_anim_timer > 0:
            self._leak_anim_timer -= 1
            if self._leak_anim_timer <= 0:
                self._leak_cells = set()

        self._flow_phase ^= 1

        self._recompute_flow()

        if self._distances:
            max_dist = max(self._distances.values())
            if self._flow_radius < max_dist:
                self._flow_radius += 1

        self._recompute_flow()

        leaks = leak_cells_from_filled(self._geo, self._orient_tuple(), self._filled_cells)
        if leaks:
            self._leak_cells = set(leaks)
            self._leak_anim_timer = 6
            self._flow_radius = 0
            self._time_left_ticks -= self._leak_penalty_ticks
            self._recompute_flow()

        all_sink_ids = set(range(len(self._geo["sink_objects"])))
        if all_sink_ids.issubset(self._sink_supplied_ids):
            self.next_level()
            return

        if self._time_left_ticks <= 0:
            self._time_left_ticks = 0
            self._fail_flash_timer = 10

    def _tick_fail_flash(self, actions_to_apply: list[tuple[str, tuple[int, int] | None]]) -> None:
        if any(kind == "restart" for kind, _ in actions_to_apply):
            self.lose()
            return

        self._fail_flash_timer -= 1
        if self._fail_flash_timer <= 0:
            self.lose()

    def _sync_board(self) -> None:
        board = self.current_level.get_sprites_by_name("board")
        if not board:
            return
        canvas = np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8)

        w = int(self._geo["width"])
        h = int(self._geo["height"])
        tile = int(self._geo["tile"])
        x0 = int(self._geo["x0"])
        y0 = int(self._geo["y0"])

        low_time = self._time_left_ticks <= max(6, int(self._time_limit_ticks * 0.15))
        for x in range(w):
            filled_ratio = float(self._time_left_ticks) / float(max(1, self._time_limit_ticks))
            full_cells = int(filled_ratio * w)
            full_cells = max(0, min(w, full_cells))

            color = COLOR_BG
            if x < full_cells:
                color = COLOR_TIME_FULL
            elif x == full_cells and self._time_left_ticks > 0:
                blink = (self._time_left_ticks + self._flow_phase) % 2 == 0
                color = COLOR_TIME_PARTIAL if blink else COLOR_BG
            if low_time and ((self._time_left_ticks + x) % 3 == 0) and x >= max(0, full_cells - 1):
                color = COLOR_TIME_PARTIAL

            px = x0 + x * tile
            py = y0
            canvas[py : py + tile, px : px + tile] = np.int8(color)

        for y in range(1, h):
            for x in range(w):
                pos = (x, y)
                ch = self._geo["base_chars"].get(pos, ".")
                px = x0 + x * tile
                py = y0 + y * tile

                if ch == "#":
                    canvas[py : py + tile, px : px + tile] = np.int8(COLOR_WALL)
                    continue
                if ch == ".":
                    canvas[py : py + tile, px : px + tile] = np.int8(COLOR_BG)
                    continue
                if ch in SOURCE_PARTS:
                    canvas[py : py + tile, px : px + tile] = np.int8(COLOR_SOURCE)
                    continue
                if ch in SINK_PARTS:
                    sink_obj = self._geo["sink_inlet_to_obj"].get((x, y))
                    if sink_obj is None and self._geo["base_chars"].get((x, y - 1)) == "◰":
                        sink_obj = self._geo["sink_inlet_to_obj"].get((x, y - 1))
                    sink_color = COLOR_SINK_ON if sink_obj in self._sink_supplied_ids else COLOR_SINK
                    canvas[py : py + tile, px : px + tile] = np.int8(sink_color)
                    continue

                mask = self._pipe_mask_cell(pos)
                is_locked = ch in LOCKED_MASKS
                is_filled = pos in self._filled_cells
                is_frontier = pos in self._frontier_cells and is_filled

                color = COLOR_LOCKED if is_locked else COLOR_PIPE
                if is_filled:
                    color = COLOR_LOCKED_FILLED if is_locked else COLOR_PIPE_FILLED
                if is_frontier and not is_locked:
                    color = COLOR_PIPE_FRONTIER if self._flow_phase == 0 else COLOR_PIPE_FILLED

                if pos in self._cell_flash:
                    color = COLOR_FAIL
                if self._fail_flash_timer > 0:
                    color = COLOR_FAIL if ((self._fail_flash_timer + self._flow_phase) % 2 == 0) else color
                if pos in self._leak_cells and self._leak_anim_timer > 0:
                    color = COLOR_LEAK_A if (self._leak_anim_timer % 2 == 0) else COLOR_LEAK_B

                _draw_pipe(canvas, px, py, tile, mask, color, COLOR_BG)

        board[0].pixels = canvas

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

        previous_queue = list(self._queued_actions)
        self._queued_actions = []

        if self._fail_flash_timer > 0:
            self._tick_fail_flash(previous_queue)
        else:
            self._tick_normal(previous_queue)

        new_flash: dict[tuple[int, int], int] = {}
        for pos, timer in self._cell_flash.items():
            if timer > 1:
                new_flash[pos] = timer - 1
        self._cell_flash = new_flash

        self._queue_current_action()
        self._sync_board()
        self.complete_action()


def _serialize_level_model(level: Level) -> dict:
    rows = tuple(str(row) for row in (level.get_data("rows") or ()))
    geo = render_layout_geometry(rows)
    rot_cells = tuple(tuple(int(v) for v in pos) for pos in geo["rotatable_cells"])
    return {
        "rows": rows,
        "time_left_ticks": int(level.get_data("time_left_ticks") or 1),
        "leak_penalty_ticks": int(level.get_data("leak_penalty_ticks") or 0),
        "click_cost_ticks": int(level.get_data("click_cost_ticks") or 0),
        "rotatable_cells": rot_cells,
    }


def _deserialize_level_model(level: Level) -> dict:
    model = _serialize_level_model(level)
    geo = render_layout_geometry(model["rows"])
    rot_cells = [tuple(pos) for pos in model["rotatable_cells"]]
    geo["rotatable_cells"] = tuple(rot_cells)
    geo["rotatable_index"] = {cell: idx for idx, cell in enumerate(rot_cells)}
    geo["rotatable_base_masks"] = tuple(int(geo["pipe_masks"][cell]) for cell in rot_cells)
    model["geo"] = geo
    return model


def _action_click_for_cell(level: Level, gx: int, gy: int) -> dict[str, int]:
    rows = tuple(str(row) for row in (level.get_data("rows") or ()))
    geo = render_layout_geometry(rows)
    return logical_cell_to_display_click(geo, int(gx), int(gy))
