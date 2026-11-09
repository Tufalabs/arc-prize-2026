from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
BOARD_SIZE = 12
CELL_SIZE = 5
MARGIN = 2

WATER = 10
BOAT = 9
HARBOR = 14
BUOY_A = 11
BUOY_B = 12
RAIL = 2
RAIL_DARK = 3
GATE = 6
GATE_DARK = 13
WHITE = 0
RED = 8
BLACK = 5

DIRS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}


@dataclass(frozen=True)
class BuoySpec:
    id: str
    pos: tuple[int, int]
    rail: tuple[tuple[int, int], ...] = ()


@dataclass
class Wave:
    source: str
    origin: tuple[int, int]
    radius: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    buoys: tuple[BuoySpec, ...]
    boats: tuple[tuple[int, int], ...]
    harbors: tuple[tuple[int, int], ...]
    gates: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    selected: str | None = None
    budget: int = 60


def _hrail(x1: int, x2: int, y: int) -> tuple[tuple[int, int], ...]:
    return tuple((x, y) for x in range(x1, x2 + 1))


def _vrail(x: int, y1: int, y2: int) -> tuple[tuple[int, int], ...]:
    return tuple((x, y) for y in range(y1, y2 + 1))


LEVEL_SPECS = (
    LevelSpec("First Bell", (BuoySpec("A", (0, 6)),), ((1, 6),), ((11, 6),), selected="A", budget=60),
    LevelSpec(
        "Slide Before Ringing",
        (BuoySpec("M", (2, 11), _hrail(2, 6, 11)),),
        ((6, 10),),
        ((6, 0),),
        selected="M",
        budget=84,
    ),
    LevelSpec(
        "The First Wave Gate",
        (BuoySpec("M", (0, 11), ((0, 11), *_vrail(1, 6, 11))), BuoySpec("N", (11, 6))),
        ((2, 6), (1, 4)),
        ((9, 6), (1, 0)),
        (((6, 6), (7, 6)),),
        selected="N",
        budget=90,
    ),
    LevelSpec(
        "Cut the Wake",
        (BuoySpec("P", (1, 10), _vrail(1, 6, 10)), BuoySpec("X", (1, 0)), BuoySpec("Q", (11, 11), _hrail(6, 11, 11))),
        ((2, 6), (6, 10)),
        ((7, 6), (6, 0)),
        (((5, 6), (6, 6)), ((6, 5), (6, 4))),
        selected="P",
        budget=162,
    ),
    LevelSpec(
        "One Wake, Two Boats",
        (
            BuoySpec("K", (6, 10), (*_hrail(0, 6, 9), (6, 10))),
            BuoySpec("M", (3, 11), _vrail(3, 6, 11)),
            BuoySpec("N", (7, 11), _hrail(7, 11, 11)),
        ),
        ((4, 6), (3, 4), (7, 10)),
        ((9, 6), (3, 0), (11, 0)),
        (((6, 6), (7, 6)), ((3, 3), (3, 2)), ((9, 10), (10, 10)), ((11, 5), (11, 4))),
        selected="K",
        budget=276,
    ),
    LevelSpec(
        "Helpful Gate Trap",
        (
            BuoySpec("K", (11, 11), (*_hrail(5, 11, 11), (5, 10))),
            BuoySpec("P", (1, 11), _vrail(1, 6, 11)),
            BuoySpec("R", (7, 1)),
        ),
        ((2, 6), (1, 4), (11, 10), (6, 1)),
        ((10, 6), (1, 0), (11, 0), (0, 1)),
        (((5, 6), (6, 6)), ((10, 6), (11, 6)), ((1, 2), (1, 1)), ((11, 5), (11, 4)), ((3, 1), (2, 1))),
        selected="K",
        budget=312,
    ),
    LevelSpec(
        "Leave and Return",
        (
            BuoySpec("T", (11, 1), _hrail(7, 11, 1)),
            BuoySpec("L", (3, 7), _hrail(3, 7, 7)),
            BuoySpec("W", (1, 11), _vrail(1, 6, 11)),
            BuoySpec("F", (5, 11), _hrail(5, 10, 11)),
        ),
        ((7, 6), (2, 6), (6, 1), (10, 10)),
        ((7, 6), (11, 6), (0, 1), (10, 0)),
        (((7, 5), (7, 6)), ((6, 6), (7, 6)), ((3, 1), (2, 1)), ((10, 5), (10, 4))),
        selected="T",
        budget=354,
    ),
)


class BellBuoysHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BellBuoys | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        y = 63
        frame[y, 2:62] = BLACK
        filled = max(0, min(60, int(60 * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[y, 2 : 2 + filled] = HARBOR
        return frame


def _cell_rect(cell: tuple[int, int]) -> tuple[int, int]:
    return MARGIN + cell[0] * CELL_SIZE, MARGIN + cell[1] * CELL_SIZE


def _center(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = _cell_rect(cell)
    return x + 2, y + 2


def _action_id(raw: object) -> int:
    return int(getattr(raw, "value", raw))


def _norm_gate(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _solid_board() -> Sprite:
    return Sprite(np.full((VIEW_SIZE, VIEW_SIZE), WATER, dtype=np.int8), name="board", layer=0, collidable=False)


def _build_level(spec: LevelSpec) -> Level:
    click_sprites: list[Sprite] = []
    for buoy in spec.buoys:
        x, y = _cell_rect(buoy.pos)
        click_sprites.append(
            Sprite(
                np.full((CELL_SIZE, CELL_SIZE), WHITE, dtype=np.int8),
                name=f"click_{buoy.id}",
                x=x,
                y=y,
                layer=20,
                visible=False,
                collidable=True,
                tags=["sys_click", "buoy_click"],
            )
        )
    return Level([_solid_board(), *click_sprites], (VIEW_SIZE, VIEW_SIZE), {"spec": spec}, spec.name)


class BellBuoys(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = BellBuoysHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "bell_buoys",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, WATER, WATER, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board = self.current_level.get_sprites_by_name("board")[0]
        self.buoys = {b.id: b.pos for b in self.spec.buoys}
        self.rails = {b.id: set(b.rail) for b in self.spec.buoys}
        self.boats = list(self.spec.boats)
        self.harbors = set(self.spec.harbors)
        self.gates: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.selected_buoy = self.spec.selected
        self.waves: list[Wave] = []
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.flash_cells: set[tuple[int, int]] = set()
        self.cancel_cells: set[tuple[int, int]] = set()
        self.open_gates: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._sync_click_sprites()
        self._draw()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._draw()
            self.complete_action()
            return

        aid = _action_id(self.action.id)
        self.flash_cells = set()
        self.cancel_cells = set()
        self.open_gates = set()
        if aid == ACTION_CLICK:
            self._handle_click()
            self._draw()
            self.complete_action()
            return

        valid = self._apply_command(aid)
        if not valid:
            if self.selected_buoy and self.selected_buoy in self.buoys:
                self.flash_cells.add(self.buoys[self.selected_buoy])
            self._draw()
            self.complete_action()
            return

        self.remaining_steps -= 1
        if self.remaining_steps < 0:
            self._draw()
            self.lose()
            self.complete_action()
            return

        self._tick_waves()
        self._draw()
        if self._is_win():
            self.next_level()
            self.complete_action()
        else:
            self.complete_action()

    def _handle_click(self) -> None:
        data = self.action.data or {}
        cell = ((int(data.get("x", 0)) - MARGIN) // CELL_SIZE, (int(data.get("y", 0)) - MARGIN) // CELL_SIZE)
        for bid, pos in self.buoys.items():
            if pos == cell:
                self.selected_buoy = bid
                return
        self.selected_buoy = None

    def _apply_command(self, aid: int) -> bool:
        if aid in DIRS:
            if not self.selected_buoy or not self.rails.get(self.selected_buoy):
                return False
            dx, dy = DIRS[aid]
            src = self.buoys[self.selected_buoy]
            dest = (src[0] + dx, src[1] + dy)
            occupied_buoys = set(self.buoys.values()) - {src}
            if dest not in self.rails[self.selected_buoy] or dest in occupied_buoys or dest in self.boats:
                return False
            self.buoys[self.selected_buoy] = dest
            self._sync_click_sprites()
            return True
        if aid == ACTION_SPACE:
            if self.selected_buoy and not any(w.source == self.selected_buoy for w in self.waves):
                self.waves.append(Wave(self.selected_buoy, self.buoys[self.selected_buoy], 0))
            return True
        return False

    def _tick_waves(self) -> None:
        for wave in self.waves:
            wave.radius += 1
        self.waves = [w for w in self.waves if w.radius <= self._max_radius(w.origin)]

        cancel_sources: set[str] = set()
        for i, a in enumerate(self.waves):
            for b in self.waves[i + 1 :]:
                d = _dist(a.origin, b.origin)
                if abs(a.radius - b.radius) <= d <= a.radius + b.radius:
                    cancel_sources.add(a.source)
                    cancel_sources.add(b.source)
        if cancel_sources:
            self.cancel_cells = {cell for w in self.waves if w.source in cancel_sources for cell in self._wave_cells(w)}
            self.waves = [w for w in self.waves if w.source not in cancel_sources]

        touched = {cell for w in self.waves for cell in self._wave_cells(w)}
        self.open_gates = {gate for gate in self.gates if gate[0] in touched or gate[1] in touched}
        self._push_boats()

    def _push_boats(self) -> None:
        proposals: dict[int, tuple[int, int] | None] = {}
        start_boats = set(self.boats)
        for idx, boat in enumerate(self.boats):
            pushes: set[tuple[int, int]] = set()
            for wave in self.waves:
                ox, oy = wave.origin
                bx, by = boat
                if (bx == ox or by == oy) and _dist(boat, wave.origin) == wave.radius:
                    dx = 0 if bx == ox else (1 if bx > ox else -1)
                    dy = 0 if by == oy else (1 if by > oy else -1)
                    pushes.add((dx, dy))
            if len(pushes) != 1:
                continue
            dx, dy = next(iter(pushes))
            dest = (boat[0] + dx, boat[1] + dy)
            if self._can_boat_enter(boat, dest, start_boats):
                proposals[idx] = dest
            else:
                proposals[idx] = None
                self.flash_cells.add(boat)
        dest_counts: dict[tuple[int, int], int] = {}
        for dest in proposals.values():
            if dest is not None:
                dest_counts[dest] = dest_counts.get(dest, 0) + 1
        for idx, dest in proposals.items():
            if dest is not None and dest_counts[dest] == 1:
                self.boats[idx] = dest

    def _can_boat_enter(self, src: tuple[int, int], dest: tuple[int, int], start_boats: set[tuple[int, int]]) -> bool:
        if not (0 <= dest[0] < BOARD_SIZE and 0 <= dest[1] < BOARD_SIZE):
            return False
        if dest in self.buoys.values() or dest in start_boats:
            return False
        gate = _norm_gate(src, dest)
        return gate not in self.gates or gate in self.open_gates

    def _is_win(self) -> bool:
        return all(boat in self.harbors and self._is_moored(boat) for boat in self.boats)

    def _is_moored(self, boat: tuple[int, int]) -> bool:
        occupied = set(self.boats)
        for wave in self.waves:
            ox, oy = wave.origin
            bx, by = boat
            if bx != ox and by != oy:
                continue
            distance = _dist(boat, wave.origin)
            if wave.radius > distance:
                continue
            dx = 0 if bx == ox else (1 if bx > ox else -1)
            dy = 0 if by == oy else (1 if by > oy else -1)
            dest = (bx + dx, by + dy)
            if not (0 <= dest[0] < BOARD_SIZE and 0 <= dest[1] < BOARD_SIZE):
                continue
            if dest in self.buoys.values() or dest in occupied:
                continue
            return False
        return True

    def _max_radius(self, origin: tuple[int, int]) -> int:
        return max(_dist(origin, corner) for corner in ((0, 0), (11, 0), (0, 11), (11, 11)))

    def _wave_cells(self, wave: Wave) -> set[tuple[int, int]]:
        ox, oy = wave.origin
        return {
            (x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if abs(x - ox) + abs(y - oy) == wave.radius
        }

    def _sync_click_sprites(self) -> None:
        for sprite in self.current_level.get_sprites_by_tag("buoy_click"):
            bid = sprite.name.split("_", 1)[1]
            sprite.set_position(*_cell_rect(self.buoys[bid]))

    def _draw(self) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), WATER, dtype=np.int8)
        frame[0:VIEW_SIZE, 0:VIEW_SIZE] = BLACK
        frame[MARGIN : MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN : MARGIN + BOARD_SIZE * CELL_SIZE] = WATER

        for rails in self.rails.values():
            for cell in rails:
                self._draw_rail(frame, cell)
        for harbor in self.harbors:
            self._draw_harbor(frame, harbor)
        for gate in self.gates:
            self._draw_gate(frame, gate, gate in self.open_gates)
        for wave in self.waves:
            for cell in self._wave_cells(wave):
                self._draw_wave(frame, cell)
        for cell in self.cancel_cells:
            self._fill_cell(frame, cell, RED)
        for idx, boat in enumerate(self.boats):
            self._draw_boat(frame, boat, idx)
        for idx, (bid, pos) in enumerate(sorted(self.buoys.items())):
            self._draw_buoy(frame, pos, idx, bid == self.selected_buoy)
        for cell in self.flash_cells:
            self._outline_cell(frame, cell, RED)
        self.board.pixels = frame

    def _fill_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = _cell_rect(cell)
        frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = color

    def _outline_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = _cell_rect(cell)
        frame[y, x : x + CELL_SIZE] = color
        frame[y + CELL_SIZE - 1, x : x + CELL_SIZE] = color
        frame[y : y + CELL_SIZE, x] = color
        frame[y : y + CELL_SIZE, x + CELL_SIZE - 1] = color

    def _draw_rail(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _cell_rect(cell)
        frame[y + 2, x : x + CELL_SIZE] = RAIL
        frame[y + 1 : y + 4, x + 2] = RAIL_DARK

    def _draw_harbor(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _cell_rect(cell)
        frame[y + 1 : y + 5, x] = HARBOR
        frame[y + 4, x : x + 5] = HARBOR
        frame[y + 1 : y + 5, x + 4] = HARBOR

    def _draw_gate(self, frame: np.ndarray, gate: tuple[tuple[int, int], tuple[int, int]], open_gate: bool) -> None:
        a, b = gate
        color = HARBOR if open_gate else GATE
        dark = HARBOR if open_gate else GATE_DARK
        ax, ay = _cell_rect(a)
        bx, by = _cell_rect(b)
        if a[0] != b[0]:
            x = max(ax, bx)
            y = ay
            frame[y : y + CELL_SIZE, x - 1 : x + 1] = color
            frame[y + 2, x - 1 : x + 1] = dark
        else:
            x = ax
            y = max(ay, by)
            frame[y - 1 : y + 1, x : x + CELL_SIZE] = color
            frame[y - 1 : y + 1, x + 2] = dark

    def _draw_wave(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _cell_rect(cell)
        frame[y, x + 1 : x + 4] = WHITE
        frame[y + 4, x + 1 : x + 4] = WHITE
        frame[y + 1 : y + 4, x] = WHITE
        frame[y + 1 : y + 4, x + 4] = WHITE

    def _draw_boat(self, frame: np.ndarray, cell: tuple[int, int], idx: int) -> None:
        del idx
        x, y = _cell_rect(cell)
        color = HARBOR if cell in self.harbors and self._is_moored(cell) else (12 if cell in self.harbors else BOAT)
        frame[y + 1, x + 1 : x + 4] = color
        frame[y + 2 : y + 4, x : x + 5] = color
        frame[y + 4, x + 1 : x + 4] = BOAT

    def _draw_buoy(self, frame: np.ndarray, cell: tuple[int, int], idx: int, selected: bool) -> None:
        del idx
        self._fill_cell(frame, cell, BUOY_A)
        if selected:
            self._outline_cell(frame, cell, WHITE)
