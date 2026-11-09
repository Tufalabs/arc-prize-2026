from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

SIDES = ("top", "bottom", "left", "right")


class EnergyBar(RenderableUserDisplay):
    def __init__(
        self,
        *,
        side: str = "top",
        rows: int = 1,
        pip_width: int = 2,
        actions_per_tick: int = 1,
        pips_per_tick: int = 1,
        pip_color: int = 11,
        spent_color: int = 3,
        gap: int = 1,
        margin: int = 0,
        tier_colors: list[int] | None = None,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        self.side = side
        self.rows = max(1, min(int(rows), 3))
        self.pip_width = max(1, min(int(pip_width), 3))
        self.actions_per_tick = max(1, int(actions_per_tick))
        self.pips_per_tick = max(1, int(pips_per_tick))
        self.pip_color = int(pip_color)
        self.spent_color = int(spent_color)
        self.gap = max(0, int(gap))
        self.margin = max(0, int(margin))
        self.tier_colors: list[int] = list(tier_colors) if tier_colors else [self.pip_color]

        self.capacity_actions = 0
        self.remaining_actions = 0

    def set_capacity(self, capacity_actions: int) -> None:
        self.capacity_actions = max(0, int(capacity_actions))
        self.remaining_actions = self.capacity_actions

    def set_remaining_actions(self, remaining_actions: int) -> None:
        self.remaining_actions = max(0, min(int(remaining_actions), self.capacity_actions))

    def tick(self) -> int:
        if self.remaining_actions > 0:
            self.remaining_actions -= 1
        return self.remaining_actions

    def _actions_to_pips(self, actions: int) -> int:
        if actions <= 0:
            return 0
        return (actions * self.pips_per_tick + self.actions_per_tick - 1) // self.actions_per_tick

    @property
    def total_pips(self) -> int:
        return self._actions_to_pips(self.capacity_actions)

    @property
    def remaining_pips(self) -> int:
        return self._actions_to_pips(self.remaining_actions)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.capacity_actions <= 0:
            return frame
        h, w = int(frame.shape[0]), int(frame.shape[1])
        total = self.total_pips
        remaining = self.remaining_pips
        if total <= 0:
            return frame

        pw = self.pip_width
        ph = self.pip_width
        stride = pw + self.gap
        horizontal = self.side in ("top", "bottom")
        long_dim = w if horizontal else h
        pips_per_row = max(1, (long_dim - self.margin) // stride)
        slot_count = pips_per_row * self.rows

        if total <= slot_count:
            visible = total
            colored = remaining
            color = self.pip_color
        else:
            visible = slot_count
            consumed = total - remaining
            tier_index = consumed // slot_count
            consumed_in_tier = consumed - tier_index * slot_count
            if tier_index >= len(self.tier_colors):
                tier_index = len(self.tier_colors) - 1
                consumed_in_tier = slot_count
            colored = slot_count - consumed_in_tier
            color = self.tier_colors[tier_index]

        for i in range(visible):
            row = i // pips_per_row
            col = i % pips_per_row
            if row >= self.rows:
                break
            cell_color = color if i < colored else self.spent_color
            if horizontal:
                x = self.margin + col * stride
                if self.side == "top":
                    y = self.margin + row * stride
                else:
                    y = h - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
            else:
                y = self.margin + col * stride
                if self.side == "left":
                    x = self.margin + row * stride
                else:
                    x = w - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
        return frame

    @staticmethod
    def _fill(frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
        h_frame, w_frame = int(frame.shape[0]), int(frame.shape[1])
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_frame, x + w)
        y1 = min(h_frame, y + h)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = color


ENERGY_CONFIG = {
    "side": "left",
    "rows": 1,
    "pip_width": 2,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 1,
    "spent_color": 15,
    "gap": 0,
    "margin": 0,
    "tier_colors": [1],
}
ENERGY_CAPACITIES = [6, 12, 21]

GAME_ID = "scale_gate_flood-0001"

GRID = 64
TILE = 4
TILES = GRID // TILE

COLOR_BG = 0
COLOR_WATER = 1
COLOR_FRONT = 8
COLOR_SOURCE = 9
COLOR_TARGET = 6
COLOR_WALL = 5
COLOR_GATE_CLOSED = 7
COLOR_WIN = 3
COLOR_HAZARD = 2
COLOR_HAZARD_FLASH = 4

T_FLOOR = 0
T_WALL = 1
T_SOURCE = 2
T_TARGET = 3
T_GATE = 4
T_HAZARD = 5

_CHAR_MAP = {"#": T_WALL, ".": T_FLOOR, "S": T_SOURCE, "T": T_TARGET, "G": T_GATE, "!": T_HAZARD}

_LAYOUTS = [
    [
        "################",
        "####S.....######",
        "###........#####",
        "##..........####",
        "##..........####",
        "###........#####",
        "####......######",
        "######GG########",
        "####......######",
        "###........#####",
        "##..T.......####",
        "##..........####",
        "###........#####",
        "####....T.######",
        "####......######",
        "################",
    ],
    [
        "################",
        "###.......######",
        "##........######",
        "###......#######",
        "#####GG#########",
        "###......#######",
        "##.....T..######",
        "###......#######",
        "#####GG#########",
        "###......#######",
        "###.......G..###",
        "###.......G..###",
        "#####GG#########",
        "#T.G.....#######",
        "#..G.S....######",
        "################",
    ],
    [
        "################",
        "#S...#....#....#",
        "#....G....G....#",
        "#....G....G....#",
        "#....#....#....#",
        "##GG########GG##",
        "#....#....#....#",
        "#.!..#....G....#",
        "#....#....G....#",
        "#....#....#....#",
        "##GG########GG##",
        "#....#....#....#",
        "#....G....G....#",
        "#.T..G....G....#",
        "#....#....#....#",
        "################",
    ],
]


def _parse_layout(
    layout: list[str],
) -> tuple[
    list[list[int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    set[tuple[int, int]],
    list[frozenset[tuple[int, int]]],
    dict[tuple[int, int], int],
]:
    tiles: list[list[int]] = []
    sources: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    hazards: set[tuple[int, int]] = set()
    gate_cells_set: set[tuple[int, int]] = set()
    for r, row in enumerate(layout):
        tile_row: list[int] = []
        for c, ch in enumerate(row):
            t = _CHAR_MAP.get(ch, T_FLOOR)
            tile_row.append(t)
            if t == T_SOURCE:
                sources.append((r, c))
            elif t == T_TARGET:
                targets.append((r, c))
            elif t == T_HAZARD:
                hazards.add((r, c))
            elif t == T_GATE:
                gate_cells_set.add((r, c))
        tiles.append(tile_row)

    remaining = set(gate_cells_set)
    gate_groups: list[frozenset[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    group.add((nr, nc))
                    stack.append((nr, nc))
        gate_groups.append(frozenset(group))

    cell_to_group: dict[tuple[int, int], int] = {}
    for i, group in enumerate(gate_groups):
        for cell in group:
            cell_to_group[cell] = i

    return tiles, sources, targets, hazards, gate_groups, cell_to_group


def _build_level() -> Level:
    floor = Sprite(
        pixels=np.full((GRID, GRID), COLOR_BG, dtype=np.int8),
        name="floor",
        x=0,
        y=0,
        layer=0,
        tags=["floor", "sys_click", "sys_every_pixel"],
        collidable=False,
    )
    return Level(grid_size=(GRID, GRID), sprites=[floor], data={})


class ScaleGateFlood(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._tiles: list[list[int]] = []
        self._sources: list[tuple[int, int]] = []
        self._targets: list[tuple[int, int]] = []
        self._gate_groups: list[frozenset[tuple[int, int]]] = []
        self._hazards: set[tuple[int, int]] = set()
        self._cell_to_group: dict[tuple[int, int], int] = {}
        self._gate_open: set[int] = set()
        self._phase = "setup"
        self._water: set[tuple[int, int]] = set()
        self._frontier: set[tuple[int, int]] = set()
        self._won = False
        self._explode_tick = 0
        self._floor: Sprite | None = None

        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(width=GRID, height=GRID, background=COLOR_BG, interfaces=[self._energy_bar])
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level() for _ in _LAYOUTS],
            camera=camera,
            win_score=len(_LAYOUTS),
            available_actions=[5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        layout = _LAYOUTS[self.level_index]
        parsed = _parse_layout(layout)
        self._tiles, self._sources, self._targets = parsed[0], parsed[1], parsed[2]
        self._hazards = parsed[3]
        self._gate_groups, self._cell_to_group = parsed[4], parsed[5]
        self._gate_open = set()
        self._phase = "setup"
        self._water = set()
        self._frontier = set()
        self._won = False
        self._explode_tick = 0
        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        self._draw()

    def _reset_flow(self) -> None:
        self._phase = "setup"
        self._water = set()
        self._frontier = set()
        self._won = False

    def _decode_click(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else None
        if not data:
            return None
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return None
        cell = self.camera.display_to_grid(raw_x, raw_y)
        if cell is None:
            return None
        gx, gy = int(cell[0]), int(cell[1])
        return gy // TILE, gx // TILE

    def _is_passable(self, r: int, c: int) -> bool:
        if r < 0 or r >= TILES or c < 0 or c >= TILES:
            return False
        t = self._tiles[r][c]
        if t == T_WALL:
            return False
        if t == T_GATE:
            gi = self._cell_to_group.get((r, c))
            return gi is not None and gi in self._gate_open
        return True

    def _flood_step(self) -> None:
        new_frontier: set[tuple[int, int]] = set()
        occupied = self._water | self._frontier
        for r, c in self._frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in occupied and self._is_passable(nr, nc):
                    new_frontier.add((nr, nc))
        self._water |= self._frontier
        self._frontier = new_frontier

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if self._phase == "setup":
            if action_id == int(GameAction.ACTION6.value):
                click = self._decode_click()
                if click is not None:
                    r, c = click
                    gi = self._cell_to_group.get((r, c))
                    if gi is not None:
                        self._gate_open.symmetric_difference_update({gi})
            elif action_id == int(GameAction.ACTION5.value):
                self._phase = "flow"
                self._frontier = set(self._sources)

        if self._phase == "explode":
            self._explode_tick += 1
            if self._explode_tick >= 6:
                self.lose()
                self._phase = "lost"
            self._draw()
            if self._phase == "explode":
                return
            self.complete_action()
            return

        if self._phase == "flow":
            self._flood_step()
            flooded = self._water | self._frontier
            if self._hazards & flooded:
                self._phase = "explode"
                self._explode_tick = 0
                self._draw()
                return
            if not self._won:
                target_set = set(self._targets)
                if target_set <= flooded:
                    self._won = True
            if not self._frontier:
                self._draw()
                if self._won:
                    self.next_level()
                else:
                    self._reset_flow()
                self.complete_action()
                return
            self._draw()
            return

        self._draw()
        self.complete_action()

    def _draw(self) -> None:
        if not self._floor:
            return
        grid = np.full((GRID, GRID), COLOR_BG, dtype=np.int8)

        for r in range(TILES):
            for c in range(TILES):
                t = self._tiles[r][c]
                y0, x0 = r * TILE, c * TILE
                y1, x1 = y0 + TILE, x0 + TILE
                if t == T_WALL:
                    grid[y0:y1, x0:x1] = COLOR_WALL
                elif t == T_SOURCE:
                    grid[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_SOURCE
                elif t == T_TARGET:
                    grid[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_TARGET
                elif t == T_HAZARD:
                    if self._phase == "explode":
                        flash = self._explode_tick % 2 == 0
                        color = COLOR_HAZARD_FLASH if flash else COLOR_HAZARD
                        grid[y0:y1, x0:x1] = color
                    else:
                        grid[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_HAZARD
                elif t == T_GATE:
                    gi = self._cell_to_group.get((r, c))
                    is_open = gi is not None and gi in self._gate_open
                    grid[y0:y1, x0:x1] = COLOR_BG if is_open else COLOR_GATE_CLOSED

        for r, c in self._water:
            y0, x0 = r * TILE, c * TILE
            color = COLOR_WIN if self._tiles[r][c] == T_TARGET else COLOR_WATER
            grid[y0 : y0 + TILE, x0 : x0 + TILE] = color

        for r, c in self._frontier:
            y0, x0 = r * TILE, c * TILE
            color = COLOR_WIN if self._tiles[r][c] == T_TARGET else COLOR_FRONT
            grid[y0 : y0 + TILE, x0 : x0 + TILE] = color

        self._floor.pixels = grid
