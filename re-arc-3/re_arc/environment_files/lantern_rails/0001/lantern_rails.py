from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "lantern_rails-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

BOARD_ORIGIN = 2
CELL = 5
SIZE = 64

BG = 4
VOID = 5
GUIDE = 2
RAIL = 3
LIGHT = 11
SWITCH = 12
CART = 9
NOSE = 10
DEPOT = 14
SELECT = 6
RED = 8
RIVET = 1

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
ACTION_DIR = {ACTION_UP: "N", ACTION_DOWN: "S", ACTION_LEFT: "W", ACTION_RIGHT: "E"}


@dataclass(frozen=True)
class LanternSpec:
    name: str
    pos: tuple[int, int]
    beam: str
    guide: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CartSpec:
    name: str
    pos: tuple[int, int]
    heading: str


@dataclass(frozen=True)
class SwitchSpec:
    stem: str
    a: str
    b: str
    state: str


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rails: dict[tuple[int, int], str]
    depots: dict[tuple[int, int], str]
    switches: dict[tuple[int, int], SwitchSpec]
    rocks: tuple[tuple[int, int], ...]
    carts: tuple[CartSpec, ...]
    lanterns: tuple[LanternSpec, ...]
    selected: str | None
    budget: int


@dataclass
class CartState:
    name: str
    pos: tuple[int, int]
    heading: str
    parked: bool = False


@dataclass
class LanternState:
    name: str
    pos: tuple[int, int]
    beam: str
    guide: tuple[tuple[int, int], ...]


class RailsHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: LanternRails | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        x0, y0, width = 3, 61, 58
        frame[y0 : y0 + 2, x0 : x0 + width] = VOID
        filled = max(0, min(width, int(width * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[y0 : y0 + 2, x0 : x0 + filled] = LIGHT
        return frame


LEVEL_SPECS = ()


def _c(x: int, y: int) -> tuple[int, int]:
    return BOARD_ORIGIN + x * CELL, BOARD_ORIGIN + y * CELL


def _cell_from_click(x: int, y: int) -> tuple[int, int]:
    return (int(x) - BOARD_ORIGIN) // CELL, (int(y) - BOARD_ORIGIN) // CELL


def _line_h(rails: dict[tuple[int, int], str], xs: range, y: int) -> None:
    for x in xs:
        rails[(x, y)] = "H"


def _line_v(rails: dict[tuple[int, int], str], x: int, ys: range) -> None:
    for y in ys:
        rails[(x, y)] = "V"


def _level_specs() -> tuple[LevelSpec, ...]:
    specs: list[LevelSpec] = []

    rails = {}
    _line_h(rails, range(2, 5), 4)
    rails[(5, 4)] = "SW"
    rails[(5, 5)] = "V"
    specs.append(
        LevelSpec(
            "First Beam",
            rails,
            {(5, 6): "N"},
            {},
            (),
            (CartSpec("C1", (2, 4), "E"),),
            (LanternSpec("Lr", (0, 4), "E", ((0, 4), (0, 5), (0, 6))),),
            None,
            48,
        )
    )

    rails = {}
    _line_h(rails, range(2, 5), 3)
    rails[(5, 3)] = "SW"
    _line_v(rails, 5, range(4, 6))
    rails[(5, 6)] = "NE"
    rails[(6, 6)] = "H"
    specs.append(
        LevelSpec(
            "Shaft Light",
            rails,
            {(7, 6): "W"},
            {},
            ((1, 4), (1, 5)),
            (CartSpec("C1", (2, 3), "E"),),
            (
                LanternSpec("Lr", (0, 3), "E", tuple((0, y) for y in range(3, 7))),
                LanternSpec("Lc", (4, 0), "S", ((4, 0), (5, 0))),
            ),
            "Lr",
            84,
        )
    )

    rails = {}
    _line_h(rails, range(1, 5), 5)
    _line_h(rails, range(6, 10), 5)
    _line_v(rails, 5, range(1, 5))
    _line_v(rails, 5, range(6, 10))
    rails[(5, 5)] = "X"
    specs.append(
        LevelSpec(
            "Dark Crossing",
            rails,
            {(10, 5): "W", (5, 10): "N"},
            {},
            (),
            (CartSpec("Ch", (1, 5), "E"), CartSpec("Cv", (5, 1), "S")),
            (LanternSpec("Lr", (0, 5), "E", ((0, 5),)), LanternSpec("Lc", (5, 0), "S", ((4, 0), (5, 0)))),
            None,
            102,
        )
    )

    rails = {
        (2, 5): "NE",
        (3, 5): "H",
        (4, 5): "T",
        (4, 4): "V",
        (4, 3): "SW",
        (3, 3): "H",
        (2, 3): "SE",
        (2, 4): "V",
        (5, 5): "H",
        (6, 5): "SW",
        (6, 6): "V",
    }
    specs.append(
        LevelSpec(
            "Switch Loop",
            rails,
            {(6, 7): "N"},
            {(4, 5): SwitchSpec("W", "N", "E", "A")},
            ((1, 4),),
            (CartSpec("C1", (2, 5), "E"),),
            (
                LanternSpec("Lr", (0, 5), "E", tuple((0, y) for y in range(3, 6))),
                LanternSpec("Lc", (3, 0), "S", tuple((x, 0) for x in range(2, 7))),
            ),
            "Lr",
            180,
        )
    )

    rails = {
        (3, 5): "H",
        (4, 5): "H",
        (5, 5): "T",
        (5, 4): "V",
        (5, 3): "SE",
        (6, 3): "H",
        (7, 3): "H",
        (8, 3): "SW",
        (8, 4): "V",
        (8, 5): "V",
        (8, 6): "NE",
        (9, 6): "H",
        (5, 6): "V",
        (5, 7): "NE",
        (6, 7): "H",
        (7, 7): "SW",
        (7, 8): "V",
        (7, 9): "NE",
        (8, 9): "H",
        (9, 9): "H",
    }
    specs.append(
        LevelSpec(
            "Old Switch",
            rails,
            {(10, 6): "W", (10, 9): "W"},
            {(5, 5): SwitchSpec("W", "N", "S", "A")},
            ((6, 8),),
            (CartSpec("C1", (4, 5), "E"), CartSpec("C2", (3, 5), "E")),
            (
                LanternSpec("Lr", (0, 5), "E", tuple((0, y) for y in range(3, 10))),
                LanternSpec("Lc", (4, 0), "S", tuple((x, 0) for x in range(4, 9))),
                LanternSpec("Ls", (11, 5), "W", ((11, 5),)),
            ),
            "Lr",
            240,
        )
    )

    rails = {
        (1, 6): "H",
        (2, 6): "H",
        (3, 6): "H",
        (4, 6): "T",
        (4, 5): "SE",
        (5, 5): "H",
        (6, 5): "H",
        (7, 5): "T",
        (7, 4): "V",
        (7, 3): "SE",
        (8, 3): "NW",
        (8, 2): "SE",
        (9, 2): "H",
        (8, 5): "H",
        (9, 5): "NW",
        (9, 4): "SE",
        (4, 7): "V",
        (4, 8): "NE",
        (5, 8): "H",
        (6, 8): "H",
        (7, 8): "H",
        (8, 8): "SW",
        (8, 9): "NE",
        (9, 9): "H",
    }
    specs.append(
        LevelSpec(
            "Three Sorter",
            rails,
            {(10, 2): "W", (10, 4): "W", (10, 9): "W"},
            {(4, 6): SwitchSpec("W", "N", "S", "A"), (7, 5): SwitchSpec("W", "N", "E", "A")},
            (),
            (CartSpec("C1", (3, 6), "E"), CartSpec("C2", (2, 6), "E"), CartSpec("C3", (1, 6), "E")),
            (LanternSpec("Lr", (0, 6), "E", tuple((0, y) for y in range(2, 10))),),
            "Lr",
            276,
        )
    )

    rails = {
        (1, 6): "H",
        (2, 6): "H",
        (3, 6): "H",
        (4, 6): "T",
        (4, 5): "SE",
        (5, 5): "H",
        (6, 5): "H",
        (7, 5): "T",
        (7, 4): "V",
        (7, 3): "SW",
        (6, 3): "H",
        (5, 3): "H",
        (8, 5): "H",
        (9, 5): "SW",
        (9, 6): "V",
        (4, 7): "V",
        (4, 8): "NE",
        (5, 8): "H",
        (6, 8): "H",
        (7, 8): "SW",
        (7, 9): "V",
        (7, 10): "NE",
        (8, 10): "H",
        (9, 10): "H",
    }
    specs.append(
        LevelSpec(
            "Surgical Columns",
            rails,
            {(4, 3): "E", (9, 7): "N", (10, 10): "W"},
            {(4, 6): SwitchSpec("W", "N", "S", "A"), (7, 5): SwitchSpec("W", "N", "E", "A")},
            ((1, 9), (8, 6), (8, 7)),
            (CartSpec("C1", (3, 6), "E"), CartSpec("C2", (2, 6), "E"), CartSpec("C3", (1, 6), "E")),
            (
                LanternSpec("Lr", (0, 6), "E", tuple((0, y) for y in range(3, 11))),
                LanternSpec("Lc", (6, 0), "S", tuple((x, 0) for x in range(6, 10))),
            ),
            "Lr",
            102,
        )
    )

    rails = {
        (1, 6): "H",
        (2, 6): "H",
        (3, 6): "H",
        (4, 6): "T",
        (4, 5): "SE",
        (5, 5): "H",
        (6, 5): "X",
        (7, 5): "T",
        (7, 4): "V",
        (7, 3): "SW",
        (6, 3): "X",
        (5, 3): "H",
        (8, 5): "H",
        (9, 5): "SW",
        (9, 6): "V",
        (4, 7): "V",
        (4, 8): "NE",
        (5, 8): "H",
        (6, 8): "X",
        (7, 8): "SW",
        (7, 9): "V",
        (7, 10): "NE",
        (8, 10): "H",
        (9, 10): "H",
        (6, 1): "V",
        (6, 2): "V",
        (6, 4): "V",
        (6, 6): "V",
        (6, 7): "V",
        (6, 9): "V",
    }
    specs.append(
        LevelSpec(
            "Final Mine",
            rails,
            {(4, 3): "E", (9, 7): "N", (10, 10): "W", (6, 10): "N"},
            {(4, 6): SwitchSpec("W", "N", "S", "A"), (7, 5): SwitchSpec("W", "N", "E", "A")},
            ((1, 9), (8, 6), (8, 7)),
            (
                CartSpec("C1", (3, 6), "E"),
                CartSpec("C2", (2, 6), "E"),
                CartSpec("C3", (1, 6), "E"),
                CartSpec("Cv", (6, 1), "S"),
            ),
            (
                LanternSpec("Lr", (0, 6), "E", tuple((0, y) for y in range(3, 11))),
                LanternSpec("Lc", (6, 0), "S", tuple((x, 0) for x in range(5, 10))),
            ),
            "Lr",
            136,
        )
    )
    return tuple(specs)


LEVEL_SPECS = _level_specs()


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(np.full((SIZE, SIZE), BG, dtype=np.int8), name="board", layer=0, collidable=False)
    clicker = Sprite(
        np.full((SIZE, SIZE), -1, dtype=np.int8),
        name="clicks",
        layer=50,
        collidable=False,
        tags=["sys_click", "sys_every_pixel"],
    )
    return Level([board, clicker], (SIZE, SIZE), {"spec": spec}, spec.name)


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


class LanternRails(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = RailsHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, SIZE, SIZE, BG, VOID, [self._hud]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.board = self.current_level.get_sprites_by_name("board")[0]
        self.spec: LevelSpec = level.get_data("spec")
        self.rails = dict(self.spec.rails)
        self.depots = dict(self.spec.depots)
        self.switch_specs = dict(self.spec.switches)
        self.switch_state = {cell: (0 if sw.state == "A" else 1) for cell, sw in self.switch_specs.items()}
        self.rocks = set(self.spec.rocks)
        self.carts = [CartState(c.name, c.pos, c.heading) for c in self.spec.carts]
        self.lanterns = {
            lantern.name: LanternState(lantern.name, lantern.pos, lantern.beam, lantern.guide)
            for lantern in self.spec.lanterns
        }
        self.selected = self.spec.selected
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.lit: set[tuple[int, int]] = set()
        self.flash_cells: set[tuple[int, int]] = set()
        self._animation: list[
            tuple[list[tuple[str, tuple[int, int], tuple[int, int]]], set[tuple[int, int]], bool]
        ] = []
        self._pending_win = False
        self._sync()

    def step(self) -> None:
        if self._animation:
            self._finish_animation()
            return

        if self.action.id == GameAction.RESET:
            self._sync()
            self.complete_action()
            return

        aid = _action_id(self.action.id)
        if aid == ACTION_CLICK:
            self._click(self.action.data)
        elif aid in ACTION_DIR:
            self._move_lantern(ACTION_DIR[aid])
        elif aid == ACTION_SPACE:
            self._space()
            if self._animation:
                self._finish_animation()
                return
        self._sync()
        if not self._animation and not self._pending_win:
            self.complete_action()

    def _finish_animation(self) -> None:
        while self._animation:
            moves, toggles, final = self._animation.pop(0)
            self._sync(moves=moves, switch_overrides=toggles if final else None)
        self.flash_cells.clear()
        self._sync()
        if self._pending_win:
            self._pending_win = False
            self.complete_action()
            self.next_level()
        else:
            self.complete_action()

    def _click(self, data: dict[str, int]) -> None:
        cell = _cell_from_click(int(data.get("x", 0)), int(data.get("y", 0)))
        for lantern in self.lanterns.values():
            if lantern.pos == cell:
                self.selected = lantern.name
                return

    def _move_lantern(self, direction: str) -> None:
        if not self.selected or self.selected not in self.lanterns:
            return
        lantern = self.lanterns[self.selected]
        dx, dy = DIRS[direction]
        target = (lantern.pos[0] + dx, lantern.pos[1] + dy)
        occupied = {other.pos for other in self.lanterns.values() if other.name != lantern.name}
        if target in lantern.guide and target not in occupied:
            lantern.pos = target

    def _space(self) -> None:
        self.remaining_steps -= 1
        self._compute_lit()
        proposals: dict[int, tuple[int, int]] = {}
        for idx, cart in enumerate(self.carts):
            dest = self._proposal(cart)
            if dest is not None:
                proposals[idx] = dest
        movers, conflicts = self._resolve_occupancy(proposals)
        if conflicts:
            self.flash_cells = conflicts
            self._animation = [([], set(), False), ([], set(), True)]
            return
        if not movers:
            if self.remaining_steps <= 0:
                self.lose()
            return
        before = {cart.name: cart.pos for cart in self.carts}
        leaving_switches: set[tuple[int, int]] = set()
        for idx in movers:
            cart = self.carts[idx]
            if cart.pos in self.switch_specs:
                leaving_switches.add(cart.pos)
            cart.pos = proposals[idx]
        for cell in leaving_switches:
            self.switch_state[cell] = 1 - self.switch_state[cell]
        for idx in movers:
            self._arrive(self.carts[idx])
        moves = [(self.carts[idx].name, before[self.carts[idx].name], self.carts[idx].pos) for idx in movers]
        toggles = set(leaving_switches)
        self._pending_win = self._is_solved()
        self._animation = [(moves, set(), False), (moves, set(), False), (moves, set(), False), (moves, set(), False)]
        if toggles:
            self._animation.extend([(moves, toggles, False), (moves, toggles, True)])
        if self.remaining_steps <= 0 and not self._pending_win:
            self.lose()

    def _proposal(self, cart: CartState) -> tuple[int, int] | None:
        if cart.parked:
            return None
        dx, dy = DIRS[cart.heading]
        dest = (cart.pos[0] + dx, cart.pos[1] + dy)
        if dest not in self.lit:
            return None
        if not self._can_exit(cart.pos, cart.heading):
            return None
        if not self._accepts(dest, OPP[cart.heading]):
            return None
        return dest

    def _resolve_occupancy(self, proposals: dict[int, tuple[int, int]]) -> tuple[set[int], set[tuple[int, int]]]:
        occupied = {cart.pos: idx for idx, cart in enumerate(self.carts) if not cart.parked or cart.pos in self.depots}
        by_dest: dict[tuple[int, int], list[int]] = {}
        for idx, dest in proposals.items():
            by_dest.setdefault(dest, []).append(idx)
        conflicts = {dest for dest, idxs in by_dest.items() if len(idxs) > 1 and dest not in occupied}
        movers: set[int] = set()

        def can_move(idx: int, seen: set[int]) -> bool:
            if idx not in proposals or idx in seen:
                return False
            dest = proposals[idx]
            if dest in conflicts:
                return False
            occ = occupied.get(dest)
            if occ is None:
                return True
            if occ == idx:
                return False
            if self.carts[occ].parked:
                return False
            return can_move(occ, seen | {idx})

        for idx in proposals:
            if can_move(idx, set()):
                movers.add(idx)
        return movers, conflicts

    def _can_exit(self, cell: tuple[int, int], heading: str) -> bool:
        if cell in self.depots:
            return False
        piece = self.rails.get(cell)
        if piece == "T":
            sw = self.switch_specs[cell]
            active = sw.a if self.switch_state[cell] == 0 else sw.b
            return heading in {sw.stem, active}
        return piece is not None and heading in _connections(piece)

    def _accepts(self, cell: tuple[int, int], entry_side: str) -> bool:
        if cell in self.depots:
            return self.depots[cell] == entry_side and not any(c.parked and c.pos == cell for c in self.carts)
        piece = self.rails.get(cell)
        if piece == "T":
            sw = self.switch_specs[cell]
            active = sw.a if self.switch_state[cell] == 0 else sw.b
            return entry_side in {sw.stem, active}
        return piece is not None and entry_side in _connections(piece)

    def _arrive(self, cart: CartState) -> None:
        entry = OPP[cart.heading]
        if cart.pos in self.depots:
            cart.parked = True
            return
        piece = self.rails.get(cart.pos)
        if piece == "T":
            sw = self.switch_specs[cart.pos]
            active = sw.a if self.switch_state[cart.pos] == 0 else sw.b
            cart.heading = active if entry == sw.stem else sw.stem
        elif piece == "X":
            return
        else:
            conn = _connections(piece or "")
            cart.heading = next(direction for direction in conn if direction != entry)

    def _compute_lit(self) -> None:
        lit: set[tuple[int, int]] = set()
        targets = set(self.rails) | set(self.depots)
        for lantern in self.lanterns.values():
            dx, dy = DIRS[lantern.beam]
            x, y = lantern.pos[0] + dx, lantern.pos[1] + dy
            while 0 <= x < 12 and 0 <= y < 12:
                if (x, y) in self.rocks:
                    break
                if (x, y) in targets:
                    lit.add((x, y))
                x += dx
                y += dy
        self.lit = lit

    def _is_solved(self) -> bool:
        parked = [cart.pos for cart in self.carts if cart.parked]
        return len(parked) == len(self.depots) == len(self.carts) and set(parked) == set(self.depots)

    def _sync(
        self,
        *,
        moves: list[tuple[str, tuple[int, int], tuple[int, int]]] | None = None,
        switch_overrides: set[tuple[int, int]] | None = None,
    ) -> None:
        self._compute_lit()
        frame = np.full((SIZE, SIZE), BG, dtype=np.int8)
        frame[0:SIZE, 0] = VOID
        frame[0:SIZE, SIZE - 1] = VOID
        frame[0, 0:SIZE] = VOID
        frame[SIZE - 1, 0:SIZE] = VOID
        for rock in self.rocks:
            self._draw_cell(frame, rock, VOID)
        for lantern in self.lanterns.values():
            for node in lantern.guide:
                self._draw_guide(frame, node)
        for cell in sorted(set(self.rails) | set(self.depots)):
            self._draw_track(frame, cell, self._piece_at(cell), cell in self.lit, switch_overrides)
        for cell in self.flash_cells:
            self._draw_cell(frame, cell, RED)
        for lantern in self.lanterns.values():
            self._draw_lantern(frame, lantern, lantern.name == self.selected)
        cart_positions = {name: pos for name, pos, _dest in moves or []}
        for cart in self.carts:
            pos = cart_positions.get(cart.name, cart.pos)
            self._draw_cart(frame, pos, cart.heading, cart.parked)
        self.board.pixels = frame

    def _piece_at(self, cell: tuple[int, int]) -> str:
        if cell in self.depots:
            return f"D{self.depots[cell]}"
        return self.rails.get(cell, "")

    def _draw_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = _c(*cell)
        frame[y : y + CELL, x : x + CELL] = color

    def _draw_guide(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _c(*cell)
        frame[y + 2, x : x + CELL] = GUIDE
        frame[y : y + CELL, x + 2] = GUIDE

    def _draw_track(
        self,
        frame: np.ndarray,
        cell: tuple[int, int],
        piece: str,
        lit: bool,
        switch_overrides: set[tuple[int, int]] | None,
    ) -> None:
        x, y = _c(*cell)
        color = LIGHT if lit else RAIL
        if piece.startswith("D"):
            frame[y + 1 : y + 4, x + 1 : x + 4] = DEPOT
            frame[y + 2, x + 2] = color
            return
        if piece == "T":
            color = SWITCH
            sw = self.switch_specs[cell]
            active = sw.a if self.switch_state[cell] == 0 else sw.b
            if switch_overrides and cell in switch_overrides:
                active = sw.a if active == sw.b else sw.b
            dirs = (sw.stem, active)
        else:
            dirs = tuple(_connections(piece))
        frame[y + 2, x + 2] = color
        for direction in dirs:
            if direction == "N":
                frame[y : y + 3, x + 2] = color
            elif direction == "S":
                frame[y + 2 : y + 5, x + 2] = color
            elif direction == "W":
                frame[y + 2, x : x + 3] = color
            elif direction == "E":
                frame[y + 2, x + 2 : x + 5] = color

    def _draw_lantern(self, frame: np.ndarray, lantern: LanternState, selected: bool) -> None:
        x, y = _c(*lantern.pos)
        if selected:
            frame[y : y + CELL, x] = SELECT
            frame[y : y + CELL, x + CELL - 1] = SELECT
            frame[y, x : x + CELL] = SELECT
            frame[y + CELL - 1, x : x + CELL] = SELECT
        frame[y + 1 : y + 4, x + 1 : x + 4] = LIGHT
        nx, ny = DIRS[lantern.beam]
        frame[y + 2 + ny, x + 2 + nx] = RIVET

    def _draw_cart(self, frame: np.ndarray, cell: tuple[int, int], heading: str, parked: bool) -> None:
        x, y = _c(*cell)
        body = DEPOT if parked else CART
        frame[y + 1 : y + 4, x + 1 : x + 4] = body
        dx, dy = DIRS[heading]
        frame[y + 2 + dy, x + 2 + dx] = NOSE


def _connections(piece: str) -> tuple[str, ...]:
    if piece == "H":
        return ("W", "E")
    if piece == "V":
        return ("N", "S")
    if piece == "X":
        return ("W", "E", "N", "S")
    if piece in {"NE", "NW", "SE", "SW"}:
        return tuple(piece)
    return ()


def make_game(seed: int = 0) -> LanternRails:
    return LanternRails(seed=seed)
