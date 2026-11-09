from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite
from arcengine.enums import BlockingMode

GAME_ID = "escape_velocity-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

GRID = 16
TILE = 4
CANVAS = 64

WHITE = 0
WHITE_WALL = 1
FLOOR = 2
PIP_DARK = 3
DARK_WALL = 4
BLACK = 5
PAD = 6
BUTTON = 8
BLUE = 9
WATER = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
GOAL = 15
DOOR = BLUE

NORTH = (0, -1)
SOUTH = (0, 1)
WEST = (-1, 0)
EAST = (1, 0)
MOVE_DIRECTIONS = {ACTION_UP: NORTH, ACTION_DOWN: SOUTH, ACTION_LEFT: WEST, ACTION_RIGHT: EAST}
DIR_BY_NAME = {"N": NORTH, "S": SOUTH, "W": WEST, "E": EAST}

BLUE_KEY = "blue"
ORANGE_KEY = "orange"
PORTAL_COLORS = {BLUE_KEY: BLUE, ORANGE_KEY: ORANGE}

MAX_FLIGHT_TELEPORTS = 8
WIN_FLICKER = 4
WIN_WIPE = 8
WIN_HOME = 28
HINT_FRAMES = 16
BOLT_CELLS_PER_FRAME = 2
SPLASH_STAGES = 3

BAR_X = 4
BAR_WIDTH = 56
BAR_Y = 61

TILE_DARK = "#"
TILE_WHITE = "%"
TILE_FLOOR = "."
TILE_WATER = "~"


@dataclass(frozen=True)
class DoorSpec:
    cells: tuple[tuple[int, int], ...]
    button: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    budget: int
    spawn: tuple[int, int]
    facing: tuple[int, int]
    white: frozenset[tuple[int, int]]
    dark: frozenset[tuple[int, int]]
    floor: frozenset[tuple[int, int]]
    pads: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    doors: tuple[DoorSpec, ...]
    exit_cells: frozenset[tuple[int, int]]
    start_portals: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]
    gun: bool
    boxes: tuple[tuple[int, int], ...]
    hint: str | None = None


@dataclass
class BoltAnim:
    color: str
    path: list[tuple[int, int]] = field(default_factory=list)
    idx: int = 0
    hit: tuple[int, int] | None = None
    normal: tuple[int, int] | None = None
    place: bool = False


@dataclass
class SplashAnim:
    cell: tuple[int, int]
    stage: int = 0
    box_idx: int | None = None


def _cell(value: object, context: str) -> tuple[int, int]:
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value)):
        raise ValueError(f"{context}: expected [col, row], got {value!r}.")
    col, row = value
    if not (0 <= col < GRID and 0 <= row < GRID):
        raise ValueError(f"{context}: cell {value!r} is out of bounds.")
    return (col, row)


def _parse_level(data: dict, source: str) -> LevelSpec:
    tiles = data.get("tiles")
    if not (
        isinstance(tiles, list) and len(tiles) == GRID and all(isinstance(r, str) and len(r) == GRID for r in tiles)
    ):
        raise ValueError(f"{source}: `tiles` must be {GRID} strings of {GRID} characters.")
    white: set[tuple[int, int]] = set()
    dark: set[tuple[int, int]] = set()
    floor: set[tuple[int, int]] = set()
    for row, line in enumerate(tiles):
        for col, char in enumerate(line):
            if char == TILE_WHITE:
                white.add((col, row))
            elif char == TILE_DARK:
                dark.add((col, row))
            elif char == TILE_FLOOR:
                floor.add((col, row))
            elif char != TILE_WATER:
                raise ValueError(f"{source}: invalid tile character {char!r} at ({col},{row}).")

    exit_cells = frozenset(_cell(c, f"{source}: exit") for c in data.get("exit", []))
    if not exit_cells:
        raise ValueError(f"{source}: every level needs `exit` goal cells.")
    for cell in exit_cells:
        if cell not in dark:
            raise ValueError(f"{source}: exit cell {cell} must replace a dark-wall tile.")
        dark.discard(cell)

    doors = []
    for entry in data.get("doors", []):
        cells = tuple(_cell(c, f"{source}: door") for c in entry.get("cells", []))
        if not cells:
            raise ValueError(f"{source}: a door needs `cells`.")
        for cell in cells:
            if cell not in dark:
                raise ValueError(f"{source}: door cell {cell} must replace a dark-wall tile.")
            dark.discard(cell)
        door_cols = sorted(c for c, _r in cells)
        door_rows = sorted(r for _c, r in cells)
        is_column = len(set(door_cols)) == 1 and door_rows == list(range(door_rows[0], door_rows[0] + len(cells)))
        is_row = len(set(door_rows)) == 1 and door_cols == list(range(door_cols[0], door_cols[0] + len(cells)))
        if not (is_column or is_row):
            raise ValueError(f"{source}: door cells must form one contiguous horizontal or vertical run.")
        if entry.get("button") is None:
            raise ValueError(f"{source}: every door is a switch-gate and needs a `button`.")
        doors.append(DoorSpec(cells=cells, button=_cell(entry["button"], f"{source}: button")))

    start_portals = []
    for key in (BLUE_KEY, ORANGE_KEY):
        entry = (data.get("portals") or {}).get(key)
        if entry is not None:
            cell = _cell(entry.get("cell"), f"{source}: portal {key}")
            normal = DIR_BY_NAME.get(entry.get("normal"))
            if normal is None:
                raise ValueError(f"{source}: portal {key} normal must be N/S/E/W.")
            if cell not in white:
                raise ValueError(f"{source}: portal {key} cell {cell} must be a white wall.")
            start_portals.append((key, cell, normal))
    gun = bool(data.get("gun", True))

    pads = []
    for pad in data.get("pads", []):
        direction = DIR_BY_NAME.get(pad.get("dir"))
        if direction is None:
            raise ValueError(f"{source}: pad direction must be one of N/S/E/W.")
        pads.append((_cell(pad.get("cell"), f"{source}: pad"), direction))
    boxes = [_cell(c, f"{source}: box") for c in data.get("boxes", [])]
    if data.get("box") is not None:
        boxes.append(_cell(data["box"], f"{source}: box"))
    spawn = _cell(data.get("spawn"), f"{source}: spawn")
    facing = DIR_BY_NAME.get(data.get("spawn_facing"))
    if facing is None:
        raise ValueError(f"{source}: `spawn_facing` must be one of N/S/E/W.")

    pad_cells = {cell for cell, _d in pads}
    buttons = [d.button for d in doors]
    for cell, label in [
        (spawn, "spawn"),
        *[(b, "box") for b in boxes],
        *[(c, "pad") for c in pad_cells],
        *[(b, "button") for b in buttons],
    ]:
        if cell is not None and cell not in floor:
            raise ValueError(f"{source}: {label} cell {cell} must sit on floor.")
    if spawn in pad_cells or spawn in buttons:
        raise ValueError(f"{source}: spawn may not sit on a pad or button.")
    for d in doors:
        if min(abs(d.button[0] - c) + abs(d.button[1] - r) for c, r in d.cells) < 2:
            raise ValueError(f"{source}: a button must be >=2 cells from every cell of its door.")

    budget = data.get("time_budget")
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError(f"{source}: `time_budget` must be a positive integer.")
    return LevelSpec(
        name=str(data.get("name", source)),
        budget=budget,
        spawn=spawn,
        facing=facing,
        white=frozenset(white),
        dark=frozenset(dark),
        floor=frozenset(floor),
        pads=tuple(pads),
        doors=tuple(doors),
        exit_cells=exit_cells,
        start_portals=tuple(start_portals),
        gun=gun,
        boxes=tuple(boxes),
        hint=(str(data["hint"]) if data.get("hint") is not None else None),
    )


def _load_level_specs() -> tuple[LevelSpec, ...]:
    levels_dir = Path(__file__).resolve().parent / "levels"
    paths = sorted(levels_dir.glob("level_*.json"))
    if not paths:
        raise ValueError(f"No level_*.json files found in {levels_dir}.")
    return tuple(_parse_level(json.loads(path.read_text(encoding="utf-8")), path.name) for path in paths)


class EscapeVelocity(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        specs = _load_level_specs()
        self._total_levels = len(specs)
        levels = [
            Level(
                sprites=[_screen_sprite()],
                grid_size=(CANVAS, CANVAS),
                data={"spec": spec, "index": idx},
                name=f"{idx + 1}. {spec.name}",
            )
            for idx, spec in enumerate(specs)
        ]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, CANVAS, CANVAS, background=BLACK, letter_box=BLACK),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self._level_index: int = int(level.get_data("index"))
        self.remaining_steps: int = self.spec.budget
        # The action space MUST stay constant across the whole game — agents (and
        # the conformance test) require it. CLICK no-ops on gun-less levels (see
        # step()), so exposing it everywhere changes nothing about play.
        self._available_actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK]
        self.pads = dict(self.spec.pads)
        self.player = tuple(self.spec.spawn)
        self.facing = self.spec.facing
        self.armed = BLUE_KEY
        self.portals: dict[str, tuple[tuple[int, int], tuple[int, int]] | None] = {BLUE_KEY: None, ORANGE_KEY: None}
        for key, cell, normal in self.spec.start_portals:
            self.portals[key] = (cell, normal)
        self.carrying = False
        self.carried_idx: int | None = None
        self.box_cells: list[tuple[int, int] | None] = list(self.spec.boxes)
        self.door_open: list[bool] = [self._button_weighted(d) for d in self.spec.doors]
        self._flight: tuple[int, int] | None = None
        self._flight_teleports = 0
        self._box_flight: tuple[int, tuple[int, int]] | None = None
        self._box_teleports = 0
        self._bolt: BoltAnim | None = None
        self._splash: SplashAnim | None = None
        self._flash: set[tuple[int, int]] = set()
        self._flash_frames = 0
        self._flash_color = WHITE
        self._win_frames = 0
        self._win_overlay = 0
        self._hint_frames = 0
        self._hint_shown = False
        self._level_advanced = False
        self._screen = self.current_level.get_sprites_by_tag("screen")[0]
        self._sync()

    # ------------------------------------------------------------------ step

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._flight = None
            self._box_flight = None
            self._bolt = None
            self._splash = None
            self._flash = set()
            self._flash_frames = 0
            self._flash_color = WHITE
            self._win_frames = 0
            self._win_overlay = 0
            self._hint_frames = 0
            self._sync()
            self.complete_action()
            return

        if self._win_frames > 0:
            exit_cells = self.spec.exit_cells
            self._win_frames -= 1
            transition = WIN_HOME if self._is_final_level() else WIN_WIPE
            if self._win_frames >= transition:
                self._flash = set(exit_cells) if self._win_frames % 2 == 1 else set()
                self._win_overlay = 0
            else:
                self._flash = set()
                self._win_overlay = transition - self._win_frames
            self._sync()
            if self._win_frames <= 0:
                self._flash = set()
                self._win_overlay = 0
                self.next_level()
                self._finish_action()
            return

        if self._bolt is not None:
            self._flash = set()
            self._advance_bolt()
            self._after_frame()
            return

        if self._splash is not None:
            self._flash = set()
            self._advance_splash()
            self._after_frame()
            return

        if self._flight is not None:
            self._flash = set()
            self._advance_flight()
            self._after_frame()
            return

        if self._box_flight is not None:
            self._flash = set()
            self._advance_box_flight()
            self._after_frame()
            return

        if self._flash_frames > 0:
            self._flash_frames -= 1
            if self._flash_frames <= 0:
                self._flash = set()
                self._flash_color = WHITE
            self._sync()
            if self._idle():
                self._finish_action()
            return

        if self._hint_frames > 0:
            self._hint_frames -= 1
            self._sync()
            if self._idle():
                self._finish_action()
            return

        self._flash = set()
        self._flash_color = WHITE
        self._level_advanced = False
        self.remaining_steps = max(0, self.remaining_steps - 1)
        self._handle_input()
        self._after_frame()

    def _after_frame(self) -> None:
        self._sync()
        if self._idle():
            self._finish_action()

    def _idle(self) -> bool:
        return (
            self._bolt is None
            and self._splash is None
            and self._flight is None
            and self._box_flight is None
            and self._flash_frames <= 0
            and self._win_frames <= 0
            and self._hint_frames <= 0
        )

    def _finish_action(self) -> None:
        if not self._level_advanced and not self._terminal() and self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _is_final_level(self) -> bool:
        return self._level_index >= self._total_levels - 1

    def _terminal(self) -> bool:
        return self._state.name in {"WIN", "GAME_OVER"}

    def _handle_input(self) -> None:
        aid = self.action.id.value if isinstance(self.action.id, GameAction) else int(self.action.id)
        if aid == ACTION_SPACE:
            if self.carrying:
                idx = self.carried_idx
                assert idx is not None
                front = (self.player[0] + self.facing[0], self.player[1] + self.facing[1])
                neighbors = [front] + [
                    (self.player[0] + dx, self.player[1] + dy)
                    for dx, dy in (NORTH, EAST, SOUTH, WEST)
                    if (self.player[0] + dx, self.player[1] + dy) != front
                ]
                fed = False
                for cell in neighbors:
                    pad_direction = self.pads.get(cell)
                    if pad_direction is not None and self._box_at(cell) is None:
                        self.carrying = False
                        self.carried_idx = None
                        self.box_cells[idx] = cell
                        self._box_flight = (idx, pad_direction)
                        self._box_teleports = 0
                        fed = True
                        break
                if not fed:
                    self.carrying = False
                    self.carried_idx = None
                    self.box_cells[idx] = self.player
                self._update_doors()
            elif self._box_at(self.player) is not None:
                idx = self._box_at(self.player)
                assert idx is not None
                for key, placement in self.portals.items():
                    if placement is not None and placement[0] == self.player:
                        self.portals[key] = None
                self.carrying = True
                self.carried_idx = idx
                self.box_cells[idx] = None
                self._update_doors()
                if self.spec.hint == "booster" and not self._hint_shown:
                    self._hint_shown = True
                    self._hint_frames = HINT_FRAMES
            elif self.spec.gun:
                self.armed = ORANGE_KEY if self.armed == BLUE_KEY else BLUE_KEY
        elif aid in MOVE_DIRECTIONS:
            self.facing = MOVE_DIRECTIONS[aid]
            self._try_move(MOVE_DIRECTIONS[aid])
        elif aid == ACTION_CLICK and self.spec.gun:
            self._fire(self.action.data or {})

    # ------------------------------------------------------------- movement

    def _try_move(self, direction: tuple[int, int]) -> None:
        target = (self.player[0] + direction[0], self.player[1] + direction[1])
        if not self._in_bounds(target):
            self._reject({self.player})
            return
        portal = self._enterable_portal_at(target, direction)
        if portal is not None:
            self._walk_teleport(portal)
            return
        if self._solid(target):
            self._reject({self.player})
            return
        self.player = target
        self._arrive()

    def _linked(self) -> bool:
        return self.portals[BLUE_KEY] is not None and self.portals[ORANGE_KEY] is not None

    def _portal_with_normal(self, cell: tuple[int, int], normal: tuple[int, int]) -> str | None:
        for key, placement in self.portals.items():
            if placement is not None and placement[0] == cell and placement[1] == normal:
                return key
        return None

    def _enterable_portal_at(self, cell: tuple[int, int], motion: tuple[int, int]) -> str | None:
        """A portal is entered only through its marked face: moving against its
        normal. One cell may host both portals on different faces; the motion
        direction selects which one is entered."""
        if not self._linked():
            return None
        return self._portal_with_normal(cell, (-motion[0], -motion[1]))

    def _walk_teleport(self, entered_key: str) -> None:
        partner_key = ORANGE_KEY if entered_key == BLUE_KEY else BLUE_KEY
        placement = self.portals[partner_key]
        entered = self.portals[entered_key]
        assert placement is not None
        assert entered is not None
        exit_cell, exit_normal = placement
        front = (exit_cell[0] + exit_normal[0], exit_cell[1] + exit_normal[1])
        if not self._in_bounds(front) or self._solid(front):
            self._reject({entered[0], exit_cell})
            return
        self.player = front
        self._flash = {entered[0], exit_cell}
        self._flash_frames = max(self._flash_frames, 2)
        self._arrive()

    def _reject(self, cells: set[tuple[int, int]]) -> None:
        self._flash = set(cells)
        self._flash_frames = 2

    # ----------------------------------------------------------------- fire

    def _fire(self, data: dict) -> None:
        x = int(data.get("x", -1))
        y = int(data.get("y", -1))
        if not (0 <= x < CANVAS and 0 <= y < CANVAS):
            return
        cell = (x // TILE, y // TILE)
        ox, oy = x - cell[0] * TILE, y - cell[1] * TILE
        indicated: list[tuple[int, int]] = []
        if oy == 0:
            indicated.append(NORTH)
        if oy == TILE - 1:
            indicated.append(SOUTH)
        if ox == 0:
            indicated.append(WEST)
        if ox == TILE - 1:
            indicated.append(EAST)
        dx = cell[0] - self.player[0]
        dy = cell[1] - self.player[1]
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            self.facing = EAST if dx > 0 else WEST
        else:
            self.facing = SOUTH if dy > 0 else NORTH
        bolt = BoltAnim(color=self.armed)
        prev = self.player
        entry: tuple[int, int] | None = None
        surface_hit: tuple[tuple[int, int], tuple[int, int]] | None = None
        same_surface = self._white_component(cell) if cell in self.spec.white else frozenset()
        for cursor in self._ray_cells((dx, dy)):
            if cursor == cell:
                bolt.hit = cursor
                entry = (prev[0] - cursor[0], prev[1] - cursor[1])
                break
            if self._blocks_bolt(cursor):
                if cursor in same_surface:
                    # The line clips another cell of the wall being aimed at:
                    # skim along the surface toward the clicked cell.
                    if surface_hit is None:
                        surface_hit = (cursor, (prev[0] - cursor[0], prev[1] - cursor[1]))
                    bolt.path.append(cursor)
                    continue
                bolt.hit = cursor
                entry = (prev[0] - cursor[0], prev[1] - cursor[1])
                break
            prev = cursor
            bolt.path.append(cursor)
        if bolt.hit is not None and (bolt.hit in self.spec.white or self._box_at(bolt.hit) is not None):
            normal = None
            if bolt.hit == cell:
                for face in indicated:
                    if face[0] * dx + face[1] * dy >= 0:
                        continue  # the face must look toward the shooter
                    front = (bolt.hit[0] + face[0], bolt.hit[1] + face[1])
                    if not self._in_bounds(front) or self._solid(front):
                        continue
                    occupant = self._portal_with_normal(bolt.hit, face)
                    if occupant is not None and occupant != self.armed:
                        continue
                    normal = face
                    break
            if normal is None:
                normal = self._entry_or_fallback(bolt.hit, entry, (dx, dy))
            if normal is None and surface_hit is not None:
                bolt.hit, fallback_entry = surface_hit
                normal = self._entry_or_fallback(bolt.hit, fallback_entry, (dx, dy))
            if normal is not None:
                occupant = self._portal_with_normal(bolt.hit, normal)
                if occupant is None or occupant == self.armed:
                    bolt.normal = normal
                    bolt.place = True
        if not bolt.path:
            self._resolve_bolt(bolt)
            return
        self._bolt = bolt

    def _ray_cells(self, delta: tuple[int, int]) -> list[tuple[int, int]]:
        """Exact-angle grid traversal (DDA) from the player's cell center
        through the clicked cell center, extended to the grid edge. Exact
        corner crossings step diagonally."""
        dx, dy = delta
        sx = (dx > 0) - (dx < 0)
        sy = (dy > 0) - (dy < 0)
        ax, ay = abs(dx), abs(dy)
        x, y = self.player
        # tMaxX = nx / (2*ax), tMaxY = ny / (2*ay); compare via cross products.
        nx, ny = 1, 1
        cells: list[tuple[int, int]] = []
        for _ in range(4 * GRID):
            if ax == 0:
                y += sy
                ny += 2
            elif ay == 0:
                x += sx
                nx += 2
            else:
                lhs = nx * ay
                rhs = ny * ax
                if lhs < rhs:
                    x += sx
                    nx += 2
                elif rhs < lhs:
                    y += sy
                    ny += 2
                else:
                    x += sx
                    y += sy
                    nx += 2
                    ny += 2
            if not self._in_bounds((x, y)):
                break
            cells.append((x, y))
        return cells

    def _white_component(self, cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
        if cell not in self.spec.white:
            return frozenset()
        seen = {cell}
        queue = [cell]
        while queue:
            cx, cy = queue.pop()
            for dx2, dy2 in (NORTH, SOUTH, EAST, WEST):
                nxt = (cx + dx2, cy + dy2)
                if nxt in self.spec.white and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return frozenset(seen)

    def _entry_or_fallback(
        self, cell: tuple[int, int], entry: tuple[int, int] | None, ray: tuple[int, int]
    ) -> tuple[int, int] | None:
        if (
            entry is not None
            and entry != (0, 0)
            and entry[0] * entry[1] == 0
            and abs(entry[0]) <= 1
            and abs(entry[1]) <= 1
        ):
            front = (cell[0] + entry[0], cell[1] + entry[1])
            if self._in_bounds(front) and not self._solid(front):
                return entry
        return self._surface_normal(cell, ray)

    def _surface_normal(self, cell: tuple[int, int], ray: tuple[int, int]) -> tuple[int, int] | None:
        candidates = []
        if ray[1] != 0:
            candidates.append((0, -((ray[1] > 0) - (ray[1] < 0))))
        if ray[0] != 0:
            candidates.append((-((ray[0] > 0) - (ray[0] < 0)), 0))
        for normal in candidates:
            adjacent = (cell[0] + normal[0], cell[1] + normal[1])
            if self._in_bounds(adjacent) and not self._solid(adjacent):
                return normal
        return None

    def _advance_bolt(self) -> None:
        bolt = self._bolt
        assert bolt is not None
        bolt.idx += BOLT_CELLS_PER_FRAME
        if bolt.idx >= len(bolt.path):
            self._bolt = None
            self._resolve_bolt(bolt)

    def _resolve_bolt(self, bolt: BoltAnim) -> None:
        if bolt.hit is None:
            return
        if bolt.place and bolt.normal is not None:
            self.portals[bolt.color] = (bolt.hit, bolt.normal)
            return
        self._flash_color = PORTAL_COLORS[bolt.color]
        self._reject({bolt.hit})

    # --------------------------------------------------------------- flight

    def _advance_flight(self) -> None:
        direction = self._flight
        assert direction is not None
        nxt = (self.player[0] + direction[0], self.player[1] + direction[1])
        if self._in_bounds(nxt):
            portal = self._enterable_portal_at(nxt, direction)
            if portal is not None and self._flight_teleports < MAX_FLIGHT_TELEPORTS:
                self._flight_teleports += 1
                partner_key = ORANGE_KEY if portal == BLUE_KEY else BLUE_KEY
                placement = self.portals[partner_key]
                entered = self.portals[portal]
                assert placement is not None
                assert entered is not None
                exit_cell, exit_normal = placement
                front = (exit_cell[0] + exit_normal[0], exit_cell[1] + exit_normal[1])
                if self._in_bounds(front) and not self._solid(front):
                    self.player = front
                    self._flight = exit_normal
                    self._flash = {entered[0], exit_cell}
                    return
                self._land()
                return
            if not self._solid(nxt):
                self.player = nxt
                return
        self._land()

    def _land(self) -> None:
        self._flight = None
        self._flight_teleports = 0
        self._arrive()

    def _advance_box_flight(self) -> None:
        assert self._box_flight is not None
        idx, direction = self._box_flight
        box = self.box_cells[idx]
        assert box is not None
        nxt = (box[0] + direction[0], box[1] + direction[1])
        if self._in_bounds(nxt):
            portal = self._enterable_portal_at(nxt, direction)
            if portal is not None and self._box_teleports < MAX_FLIGHT_TELEPORTS:
                self._box_teleports += 1
                partner_key = ORANGE_KEY if portal == BLUE_KEY else BLUE_KEY
                placement = self.portals[partner_key]
                entered = self.portals[portal]
                assert placement is not None
                assert entered is not None
                exit_cell, exit_normal = placement
                front = (exit_cell[0] + exit_normal[0], exit_cell[1] + exit_normal[1])
                if self._in_bounds(front) and not self._solid(front) and self._box_at(front) is None:
                    self.box_cells[idx] = front
                    self._box_flight = (idx, exit_normal)
                    self._flash = {entered[0], exit_cell}
                    return
                self._box_land()
                return
            if not self._solid(nxt) and self._box_at(nxt) is None:
                self.box_cells[idx] = nxt
                return
        self._box_land()

    def _box_land(self) -> None:
        assert self._box_flight is not None
        idx, _direction = self._box_flight
        self._box_flight = None
        self._box_teleports = 0
        cell = self.box_cells[idx]
        assert cell is not None
        if not self._walkable(cell):
            self._splash = SplashAnim(cell=cell, box_idx=idx)
            self.box_cells[idx] = None
            return
        pad_direction = self.pads.get(cell)
        if pad_direction is not None:
            self._box_flight = (idx, pad_direction)
            self._box_teleports = 0
            return
        self._update_doors()

    # -------------------------------------------------------------- arrival

    def _arrive(self) -> None:
        cell = self.player
        walkable = self._walkable(cell)
        self._update_doors()
        if not walkable:
            self._splash = SplashAnim(cell=cell)
            return
        pad_direction = self.pads.get(cell)
        if pad_direction is not None:
            self._flight = pad_direction
            self._flight_teleports = 0
            return
        if cell in self.spec.exit_cells:
            self._level_advanced = True
            self._flash = set()
            self._flash_frames = 0
            self._win_frames = WIN_FLICKER + (WIN_HOME if self._is_final_level() else WIN_WIPE)
            self._flight = None

    def _button_weighted(self, door: DoorSpec) -> bool:
        return self.player == door.button or self._box_at(door.button) is not None

    def _door_should_be_open(self, door: DoorSpec) -> bool:
        return self._button_weighted(door)

    def _update_doors(self) -> None:
        for idx, door in enumerate(self.spec.doors):
            new_state = self._door_should_be_open(door)
            if new_state != self.door_open[idx]:
                self.door_open[idx] = new_state
                self._flash |= set(door.cells)
                self._flash_frames = max(self._flash_frames, 2)

    def _advance_splash(self) -> None:
        splash = self._splash
        assert splash is not None
        splash.stage += 1
        if splash.stage < SPLASH_STAGES:
            return
        self._splash = None
        if splash.box_idx is not None:
            home = self.spec.boxes[splash.box_idx]
            self.box_cells[splash.box_idx] = home
            self._update_doors()
            self._flash |= {home}
            self._flash_frames = max(self._flash_frames, 2)
            return
        if self.carrying:
            assert self.carried_idx is not None
            self.box_cells[self.carried_idx] = self.spec.boxes[self.carried_idx]
            self.carrying = False
            self.carried_idx = None
        self.player = tuple(self.spec.spawn)
        self._update_doors()
        self._flash |= {self.player}
        self._flash_frames = max(self._flash_frames, 2)

    # -------------------------------------------------------------- queries

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < GRID and 0 <= cell[1] < GRID

    def _box_at(self, cell: tuple[int, int]) -> int | None:
        for idx, box in enumerate(self.box_cells):
            if box == cell:
                return idx
        return None

    def _blocks_bolt(self, cell: tuple[int, int]) -> bool:
        if self._box_at(cell) is not None:
            return True
        return self._solid(cell)

    def _solid(self, cell: tuple[int, int]) -> bool:
        if cell in self.spec.white or cell in self.spec.dark:
            return True
        return any(not self.door_open[idx] and cell in door.cells for idx, door in enumerate(self.spec.doors))

    def _walkable(self, cell: tuple[int, int]) -> bool:
        if cell in self.spec.floor or cell in self.spec.exit_cells:
            return True
        return any(self.door_open[idx] and cell in door.cells for idx, door in enumerate(self.spec.doors))

    # ------------------------------------------------------------ rendering

    def _sync(self) -> None:
        self._screen.pixels = self._render().astype(np.int8)

    def _render(self) -> np.ndarray:
        frame = np.full((CANVAS, CANVAS), WATER, dtype=np.int8)
        for cell in self.spec.floor:
            self._fill(frame, cell, FLOOR)
        for cell in self.spec.dark:
            self._fill(frame, cell, DARK_WALL)
        for cell in self.spec.white:
            self._fill(frame, cell, WHITE_WALL)
        for cell in self.spec.exit_cells:
            self._draw_goal(frame, cell)
        self._draw_doors(frame)
        for door in self.spec.doors:
            self._draw_button(frame, door)
        for cell, direction in self.pads.items():
            self._draw_pad(frame, cell, direction)
        for box in self.box_cells:
            if box is not None:
                self._draw_box(frame, box)
        linked = self._linked()
        active = [(key, placement) for key, placement in self.portals.items() if placement is not None]
        for _key, (cell, _normal) in active:
            if cell in self.spec.white:
                self._fill(frame, cell, WHITE_WALL)
        for key, (cell, normal) in active:
            self._draw_portal_strip(frame, cell, normal, PORTAL_COLORS[key])
        self._split_shared_corner(frame, active)
        if linked:
            for _key, (cell, normal) in active:
                self._draw_portal_pip(frame, cell, normal)
        if self._splash is not None:
            self._draw_splash(frame, self._splash)
        if self._splash is None or self._splash.box_idx is not None:
            self._draw_player(frame)
        if self._bolt is not None:
            self._draw_bolt(frame, self._bolt)
        self._draw_time_bar(frame)
        if self._hint_frames > 0:
            self._draw_hint(frame)
        if self._win_overlay > 0:
            if self._is_final_level():
                self._draw_homecoming(frame, self._win_overlay)
            else:
                frame[:, : min(CANVAS, self._win_overlay * 8)] = DOOR
        for cell in self._flash:
            strips = [p for p in self.portals.values() if p is not None and p[0] == cell]
            if strips:
                for _cell, normal in strips:
                    self._draw_face_edge(frame, cell, normal, self._flash_color)
            else:
                self._draw_outline(frame, cell, self._flash_color)
        return frame

    def _draw_face_edge(self, frame: np.ndarray, cell: tuple[int, int], normal: tuple[int, int], color: int) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        if normal == WEST:
            frame[y : y + TILE, x] = color
        elif normal == EAST:
            frame[y : y + TILE, x + TILE - 1] = color
        elif normal == NORTH:
            frame[y, x : x + TILE] = color
        else:
            frame[y + TILE - 1, x : x + TILE] = color

    def _fill(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y : y + TILE, x : x + TILE] = color

    def _draw_outline(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y, x : x + TILE] = color
        frame[y + TILE - 1, x : x + TILE] = color
        frame[y : y + TILE, x] = color
        frame[y : y + TILE, x + TILE - 1] = color

    def _draw_open_door(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        self._fill(frame, cell, DOOR)
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y + 1 : y + 3, x + 1 : x + 3] = BLACK

    def _draw_goal(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        self._fill(frame, cell, GOAL)
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y + 1 : y + 3, x + 1 : x + 3] = WHITE

    def _draw_doors(self, frame: np.ndarray) -> None:
        for door_idx, door in enumerate(self.spec.doors):
            for cell in door.cells:
                if self.door_open[door_idx]:
                    self._draw_open_door(frame, cell)
                else:
                    self._fill(frame, cell, DOOR)

    def _draw_button(self, frame: np.ndarray, door: DoorSpec) -> None:
        cell = door.button
        self._fill(frame, cell, BUTTON)
        x, y = cell[0] * TILE, cell[1] * TILE
        for px, py in ((0, 0), (TILE - 1, 0), (0, TILE - 1), (TILE - 1, TILE - 1)):
            frame[y + py, x + px] = MAROON
        if self._button_weighted(door):
            frame[y + 1 : y + 3, x + 1 : x + 3] = WHITE

    def _draw_pad(self, frame: np.ndarray, cell: tuple[int, int], direction: tuple[int, int]) -> None:
        self._fill(frame, cell, PAD)
        x, y = cell[0] * TILE, cell[1] * TILE
        arrows = {
            NORTH: ((1, 1), (2, 1), (0, 2), (3, 2)),
            SOUTH: ((0, 1), (3, 1), (1, 2), (2, 2)),
            EAST: ((1, 0), (2, 1), (2, 2), (1, 3)),
            WEST: ((2, 0), (1, 1), (1, 2), (2, 3)),
        }
        for px, py in arrows[direction]:
            frame[y + py, x + px] = WHITE

    def _draw_box(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y : y + TILE, x : x + TILE] = WHITE_WALL
        frame[y + 1 : y + 3, x + 1 : x + 3] = YELLOW

    def _draw_portal_strip(self, frame: np.ndarray, cell: tuple[int, int], normal: tuple[int, int], color: int) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        if normal == WEST:
            frame[y : y + TILE, x : x + 2] = color
        elif normal == EAST:
            frame[y : y + TILE, x + 2 : x + 4] = color
        elif normal == NORTH:
            frame[y : y + 2, x : x + TILE] = color
        else:
            frame[y + 2 : y + 4, x : x + TILE] = color

    def _draw_portal_pip(self, frame: np.ndarray, cell: tuple[int, int], normal: tuple[int, int]) -> None:
        x, y = cell[0] * TILE, cell[1] * TILE
        if normal == WEST:
            frame[y + 1 : y + 3, x] = WHITE
        elif normal == EAST:
            frame[y + 1 : y + 3, x + 3] = WHITE
        elif normal == NORTH:
            frame[y, x + 1 : x + 3] = WHITE
        else:
            frame[y + 3, x + 1 : x + 3] = WHITE

    def _split_shared_corner(self, frame: np.ndarray, active: list) -> None:
        """When both portals sit on one cell with perpendicular faces, their
        strips overlap in a 2x2 corner square: split it half-and-half — the
        E/W strip keeps the square's outer column, the N/S strip the inner."""
        if len(active) != 2:
            return
        (key_a, (cell_a, normal_a)), (key_b, (cell_b, normal_b)) = active
        if cell_a != cell_b or normal_a[0] * normal_b[0] + normal_a[1] * normal_b[1] != 0:
            return
        if normal_a[0] != 0:
            ew_key, ew_normal, ns_key, ns_normal = key_a, normal_a, key_b, normal_b
        else:
            ew_key, ew_normal, ns_key, ns_normal = key_b, normal_b, key_a, normal_a
        x, y = cell_a[0] * TILE, cell_a[1] * TILE
        outer_col = x if ew_normal == WEST else x + 3
        inner_col = x + 1 if ew_normal == WEST else x + 2
        outer_row = y if ns_normal == NORTH else y + 3
        inner_row = y + 1 if ns_normal == NORTH else y + 2
        # Diagonal weave: each color takes one diagonal pair of the shared
        # 2x2 corner, so neither dominates.
        frame[outer_row, outer_col] = PORTAL_COLORS[ew_key]
        frame[inner_row, inner_col] = PORTAL_COLORS[ew_key]
        frame[outer_row, inner_col] = PORTAL_COLORS[ns_key]
        frame[inner_row, outer_col] = PORTAL_COLORS[ns_key]

    def _draw_player(self, frame: np.ndarray) -> None:
        x, y = self.player[0] * TILE, self.player[1] * TILE
        frame[y : y + TILE, x : x + TILE] = GREEN
        if self.carrying:
            frame[y + 1 : y + 3, x + 1 : x + 3] = YELLOW
        if not self.spec.gun:
            return
        gun = PORTAL_COLORS[self.armed]
        if self.facing == EAST:
            frame[y + 1 : y + 3, x + 3] = gun
        elif self.facing == WEST:
            frame[y + 1 : y + 3, x] = gun
        elif self.facing == NORTH:
            frame[y, x + 1 : x + 3] = gun
        else:
            frame[y + 3, x + 1 : x + 3] = gun

    def _draw_bolt(self, frame: np.ndarray, bolt: BoltAnim) -> None:
        if not bolt.path:
            return
        cell = bolt.path[min(bolt.idx, len(bolt.path) - 1)]
        x, y = cell[0] * TILE, cell[1] * TILE
        frame[y + 1 : y + 3, x + 1 : x + 3] = PORTAL_COLORS[bolt.color]
        frame[y + 1, x + 1] = WHITE

    def _draw_splash(self, frame: np.ndarray, splash: SplashAnim) -> None:
        x, y = splash.cell[0] * TILE, splash.cell[1] * TILE
        if splash.stage == 0:
            frame[y + 1 : y + 3, x + 1 : x + 3] = WHITE
            frame[y, x + 1 : x + 3] = BLUE
            frame[y + 3, x + 1 : x + 3] = BLUE
            frame[y + 1 : y + 3, x] = BLUE
            frame[y + 1 : y + 3, x + 3] = BLUE
        elif splash.stage == 1:
            self._draw_outline(frame, splash.cell, BLUE)
        else:
            for px, py in ((0, 0), (TILE - 1, 0), (0, TILE - 1), (TILE - 1, TILE - 1)):
                frame[y + py, x + px] = BLUE

    def _draw_homecoming(self, frame: np.ndarray, stage: int) -> None:
        """Final cutscene: the agent walks home across a quiet dark field,
        steps through the blue door, and the window lights up."""
        frame[:, :] = BLACK
        frame[40:42, :] = FLOOR
        frame[27:40, 44:60] = WHITE_WALL
        frame[24:27, 47:57] = DARK_WALL
        frame[30:32, 46:49] = DARK_WALL
        frame[31:40, 50:54] = DOOR
        home = stage >= 23
        if not home:
            frame[33:40, 51:53] = BLACK
            x = min(2 + stage * 2, 45)
            frame[36:40, x : x + 4] = GREEN
            gun = PORTAL_COLORS[self.armed]
            frame[37:39, x + 3] = gun
        else:
            frame[30:32, 46:49] = YELLOW
            if stage >= 26:
                frame[26, 52] = WHITE

    def _draw_hint(self, frame: np.ndarray) -> None:
        """One-time tutorial cue (first box pickup): pulse each booster pad, show a
        box parked beside it, then send a box gliding off in the launch direction —
        teaching that a box placed by a booster gets launched."""
        elapsed = HINT_FRAMES - self._hint_frames
        bright = (self._hint_frames // 2) % 2 == 0
        for cell, direction in self.pads.items():
            self._draw_outline(frame, cell, WHITE if bright else GREEN)
            if elapsed < HINT_FRAMES // 2:
                beside = (cell[0] - direction[0], cell[1] - direction[1])
                if beside in self.spec.floor:
                    self._draw_box(frame, beside)
            else:
                steps = (elapsed - HINT_FRAMES // 2) // 2 + 1
                launch = (cell[0] + direction[0] * steps, cell[1] + direction[1] * steps)
                if self._in_bounds(launch):
                    self._draw_box(frame, launch)

    def _draw_time_bar(self, frame: np.ndarray) -> None:
        frame[BAR_Y : BAR_Y + 2, BAR_X : BAR_X + BAR_WIDTH] = BLACK
        filled = max(0, min(BAR_WIDTH, round(BAR_WIDTH * self.remaining_steps / max(1, self.spec.budget))))
        if filled:
            frame[BAR_Y : BAR_Y + 2, BAR_X : BAR_X + filled] = GREEN


def _screen_sprite() -> Sprite:
    return Sprite(
        np.full((CANVAS, CANVAS), BLACK, dtype=np.int8),
        name="screen",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        collidable=False,
        tags=["screen"],
    )
