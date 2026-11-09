"""EN18 -- "Electricity Net".

A NetSlide-style wiring puzzle. Tiles carry fixed wire segments; you cannot
rotate them, only slide whole rows / columns (cyclic, with wrap) using the
buttons around each tile group. Arrange wires into a connected net with no loose
ends; solving a group sends electricity along its wire.

Engine (generalised over levels):
  * A level is a set of GROUPS (each an independent NetSlide of arbitrary shape),
    WIRES between them, and a BULB.
  * Solving a group energises its outgoing wire (orange flow animation). A wire
    can either enable a gated line on another group (its buttons go grey->blue)
    or, for the final wire, light the bulb and complete the level.

Palette note: this repo's COLOR_MAP is inverted vs. standard ARC (white=0,
grays=1..4, black=5).

Levels:
  1. one 2x2 group (U), wire -> bulb (bottom-right).
  2. a 2x2 group (top-left, U rotated 90 deg CW) and a 2x2 group (top-right) each
     power one axis of a T-shaped group (centre; 3-wide bar + 1 stem cell); its
     buttons start grey and become usable once fed. Solving the T lights the bulb
     (bottom-left).
"""

from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
WHITE = 0  # frames + connecting wire (unenergised) + unconnected tile wire
GRAY = 2  # energy bar fill
GRAY_EMPTY = 4  # energy bar depleted track
BLACK = 5  # background / tile interior
WIRE = 11  # tile wire (yellow) -- shown once the tile connects to any neighbour
BALL = 12  # endpoint 1px terminal marker (orange)
BTN = 9  # enabled slide button (blue)
BTN_OFF = 3  # disabled / unpowered slide button (grey)
BTN_BELT = 15  # auto-belt line marker (purple); NOT clickable -- the belt advances
# one step whenever the player clicks a blue button anywhere
ROT_BTN = 6  # rotate bulb + pink rotate-spot background; clicking the bulb turns
# the tile on each pink spot 90deg CW
ORANGE = 12  # energised frame + connecting wire
BULB_OFF = 2  # unlit bulb (gray)
BULB_ON = 11  # lit bulb (yellow)
GREEN = 14  # teleport-tile background + green bulb (lit)

BLINK_TOTAL = 24  # frames of blinking before the teleport fires
BLINK_PERIOD = 4  # frames per on/off half-cycle

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
GRID = 64
ENERGY_ROWS = 2
BOARD_TOP = 2
TILE = 5
MARGIN = 4  # button margin around a group's core frame
FLOW_PX = 3  # connecting-wire pixels energised per flow frame
FLOW_HOLD_PX = 9  # extra progress held with the target lit before completing

OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
ROT_CW = {"U": "R", "R": "D", "D": "L", "L": "U"}  # 90 degrees clockwise


def _rotate_dirs(dirs, n=1):
    d = set(dirs)
    for _ in range(n % 4):
        d = {ROT_CW[x] for x in d}
    return frozenset(d)


def _ncell(cell, d):
    c, r = cell
    if d == "R":
        return (c + 1, r)
    if d == "L":
        return (c - 1, r)
    if d == "D":
        return (c, r + 1)
    return (c, r - 1)


def _adj_ok(cellset, placement, tiles) -> bool:
    """Loose-ends-free: every arm meets a matching arm on an in-shape neighbour."""
    for cell, tid in placement.items():
        for d in tiles[tid]:
            nb = _ncell(cell, d)
            if nb not in cellset or OPP[d] not in tiles[placement[nb]]:
                return False
    return True


def _any_solvable(cells, tiles) -> bool:
    """True iff SOME arrangement of this tile multiset is loose-ends-free."""
    cs = set(cells)
    ids = list(tiles.keys())
    for perm in itertools.permutations(ids):
        if _adj_ok(cs, {cell: perm[i] for i, cell in enumerate(cells)}, tiles):
            return True
    return False


def _tile_connected(cellset, placement, tiles, cell) -> bool:
    """True iff this cell's tile has >= 1 arm meeting a matching neighbour arm."""
    for d in tiles[placement[cell]]:
        nb = _ncell(cell, d)
        if nb in cellset and OPP[d] in tiles[placement[nb]]:
            return True
    return False


def _tile_pixels(tiles, tid, bg=BLACK, wire_col=WIRE) -> np.ndarray:
    img = np.full((TILE, TILE), bg, dtype=np.int8)
    dirs = tiles[tid]
    if "U" in dirs:
        img[0:2, 2] = wire_col
    if "D" in dirs:
        img[3:5, 2] = wire_col
    if "L" in dirs:
        img[2, 0:2] = wire_col
    if "R" in dirs:
        img[2, 3:5] = wire_col
    if dirs:
        img[2, 2] = wire_col
    if len(dirs) == 1:  # endpoint -> 1px orange terminal marker at the centre
        img[2, 2] = BALL
    return img


# ---------------------------------------------------------------------------
# Group: one NetSlide puzzle of arbitrary cell shape
# ---------------------------------------------------------------------------
class Group:
    def __init__(self, spec):
        self.id = spec["id"]
        self.bx, self.by = spec["block"]
        self.cols, self.rows = spec["cols"], spec["rows"]
        self.cells = [tuple(c) for c in spec["cells"]]
        self.cellset = set(self.cells)
        self.tiles = {k: frozenset(v) for k, v in spec["tiles"].items()}
        # cells and start are parallel by construction; strict=True fails fast on a bad level spec.
        self.placement = {cell: tid for cell, tid in zip(self.cells, spec["start"], strict=True)}

    # geometry -------------------------------------------------------------
    @property
    def core_x0(self):
        return self.bx + MARGIN

    @property
    def core_y0(self):
        return self.by + MARGIN

    @property
    def core_w(self):
        return self.cols * TILE + 2

    @property
    def core_h(self):
        return self.rows * TILE + 2

    def tile_origin(self, c, r):
        return self.core_x0 + 1 + c * TILE, self.core_y0 + 1 + r * TILE

    def frame_mask(self):
        """1px ring hugging the union of the (possibly non-rectangular) cells."""
        if getattr(self, "_ring", None) is None:
            mask = np.zeros((GRID, GRID), bool)
            for cell in self.cells:
                tx, ty = self.tile_origin(*cell)
                mask[ty : ty + TILE, tx : tx + TILE] = True
            dil = mask.copy()
            dil[1:, :] |= mask[:-1, :]
            dil[:-1, :] |= mask[1:, :]
            dil[:, 1:] |= mask[:, :-1]
            dil[:, :-1] |= mask[:, 1:]
            dil[1:, 1:] |= mask[:-1, :-1]
            dil[:-1, :-1] |= mask[1:, 1:]
            dil[1:, :-1] |= mask[:-1, 1:]
            dil[:-1, 1:] |= mask[1:, :-1]
            self._ring = dil & ~mask
        return self._ring

    def lines(self):
        """Slidable lines: every row/col with >= 2 active cells, ordered."""
        out = []
        for r in range(self.rows):
            lc = [(c, r) for c in range(self.cols) if (c, r) in self.cellset]
            if len(lc) >= 2:
                out.append(("row", r, lc))
        for c in range(self.cols):
            lc = [(c, r) for r in range(self.rows) if (c, r) in self.cellset]
            if len(lc) >= 2:
                out.append(("col", c, lc))
        return out

    def line_cells(self, axis, index):
        for a, i, lc in self.lines():
            if a == axis and i == index:
                return lc
        return []

    def line_buttons(self, axis, index):
        x1 = self.core_x0 + self.core_w - 1
        y1 = self.core_y0 + self.core_h - 1
        if axis == "row":
            ty = self.tile_origin(0, index)[1] + 1
            return [(self.core_x0 - 3, ty, 3, 3, "L"), (x1 + 1, ty, 3, 3, "R")]
        tx = self.tile_origin(index, 0)[0] + 1
        return [(tx, self.core_y0 - 3, 3, 3, "U"), (tx, y1 + 1, 3, 3, "D")]

    def button_center(self, axis, index, direction):
        for x0, y0, _w, _h, d in self.line_buttons(axis, index):
            if d == direction:
                return x0 + 1, y0 + 1
        return None

    # mechanics ------------------------------------------------------------
    def slide(self, lc, direction):
        ids = [self.placement[c] for c in lc]
        ids = [ids[-1], *ids[:-1]] if direction in ("R", "D") else [*ids[1:], ids[0]]
        for c, t in zip(lc, ids, strict=False):
            self.placement[c] = t

    def solved(self):
        return _adj_ok(self.cellset, self.placement, self.tiles)


# ---------------------------------------------------------------------------
# Level construction helpers
# ---------------------------------------------------------------------------
def _apply_list(cells, plist, lc, direction):
    idx = {cell: i for i, cell in enumerate(cells)}
    pos = [idx[c] for c in lc]
    ids = [plist[p] for p in pos]
    ids = [ids[-1], *ids[:-1]] if direction in ("R", "D") else [*ids[1:], ids[0]]
    out = list(plist)
    for p, t in zip(pos, ids, strict=False):
        out[p] = t
    return out


def _lines_of(cells, cols, rows):
    cs = set(cells)
    out = []
    for r in range(rows):
        lc = [(c, r) for c in range(cols) if (c, r) in cs]
        if len(lc) >= 2:
            out.append(("row", r, lc))
    for c in range(cols):
        lc = [(c, r) for r in range(rows) if (c, r) in cs]
        if len(lc) >= 2:
            out.append(("col", c, lc))
    return out


def _solve_len(cells, tiles, plist, lines, axes=None):
    cs = set(cells)
    use = lines if axes is None else [li for li in lines if li[0] in axes]

    def ok(pl):
        return _adj_ok(cs, {cell: pl[i] for i, cell in enumerate(cells)}, tiles)

    if ok(plist):
        return 0
    start = tuple(plist)
    seen = {start}
    q = deque([(start, 0)])
    while q:
        s, dist = q.popleft()
        for axis, _i, lc in use:
            for d in ("L", "R") if axis == "row" else ("U", "D"):
                nxt = tuple(_apply_list(cells, list(s), lc, d))
                if nxt in seen:
                    continue
                if ok(list(nxt)):
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return None


def _dirs_for(axis):
    return ("L", "R") if axis == "row" else ("U", "D")


def _scramble(cells, cols, rows, tiles, solved_list, seed, min_moves=2, need_both=False, slides=(5, 14)):
    lines = _lines_of(cells, cols, rows)
    rng = random.Random(seed)
    cs = set(cells)
    for _ in range(8000):
        p = list(solved_list)
        for _ in range(rng.randint(*slides)):
            axis, _i, lc = rng.choice(lines)
            d = rng.choice(("L", "R") if axis == "row" else ("U", "D"))
            p = _apply_list(cells, p, lc, d)
        if _adj_ok(cs, {cell: p[i] for i, cell in enumerate(cells)}, tiles):
            continue
        if (_solve_len(cells, tiles, p, lines) or 0) < min_moves:
            continue
        if need_both:
            if _solve_len(cells, tiles, p, lines, axes={"row"}) is not None:
                continue
            if _solve_len(cells, tiles, p, lines, axes={"col"}) is not None:
                continue
        return p
    raise RuntimeError("en18: scramble failed")


def _spec_from_start(gid, block, cols, rows, cells, dirs, start):
    return {
        "id": gid,
        "block": list(block),
        "cols": cols,
        "rows": rows,
        "cells": [list(c) for c in cells],
        "tiles": {str(i): list(dirs[i]) for i in range(len(cells))},
        "start": start,
    }


def _group_spec(gid, block, cols, rows, cells, dirs, seed, min_moves=2, need_both=False, slides=(5, 14)):
    """cells/dirs are parallel; tile-id = str(index)."""
    tiles = {str(i): frozenset(dirs[i]) for i in range(len(cells))}
    solved_list = [str(i) for i in range(len(cells))]
    start = _scramble(cells, cols, rows, tiles, solved_list, seed, min_moves, need_both, slides)
    return _spec_from_start(gid, block, cols, rows, cells, dirs, start)


def _scramble_belt(cells, cols, rows, tiles, solved_list, seed, belt, player, belt_dir, slides=(4, 9)):
    """Scramble for a belt grid: one line auto-advances (belt) every action; the
    player can shift the other line. Pick a start that needs BOTH lines and is
    solvable under that constraint."""
    lines = _lines_of(cells, cols, rows)
    belt_lc = next(lc for a, i, lc in lines if (a, i) == belt)
    player_lc = next(lc for a, i, lc in lines if (a, i) == player)
    cs = set(cells)
    rng = random.Random(seed)

    def ok(p):
        return _adj_ok(cs, {c: p[i] for i, c in enumerate(cells)}, tiles)

    def line_only(start, lc, axis):
        seen = {tuple(start)}
        q = deque([tuple(start)])
        while q:
            s = q.popleft()
            if ok(list(s)):
                return True
            for d in _dirs_for(axis):
                n = tuple(_apply_list(cells, list(s), lc, d))
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        return False

    def belt_solvable(start):
        # each action: optional player shift, then the belt advances one step
        seen = {tuple(start)}
        q = deque([tuple(start)])
        while q:
            s = q.popleft()
            for pm in (None, *_dirs_for(player[0])):
                p1 = list(s) if pm is None else _apply_list(cells, list(s), player_lc, pm)
                c = _apply_list(cells, p1, belt_lc, belt_dir)
                if ok(c):
                    return True
                tc = tuple(c)
                if tc not in seen:
                    seen.add(tc)
                    q.append(tc)
        return False

    for _ in range(8000):
        p = list(solved_list)
        for _ in range(rng.randint(*slides)):
            a, _i, lc = rng.choice(lines)
            p = _apply_list(cells, p, lc, rng.choice(_dirs_for(a)))
        if ok(p):
            continue
        if line_only(p, belt_lc, belt[0]) or line_only(p, player_lc, player[0]):
            continue  # must need both lines
        if not belt_solvable(p):
            continue  # must be solvable while the belt keeps moving
        return p
    raise RuntimeError("en18: belt scramble failed")


def _group_spec_belt(gid, block, cols, rows, cells, dirs, seed, belt, player, belt_dir):
    tiles = {str(i): frozenset(dirs[i]) for i in range(len(cells))}
    solved_list = [str(i) for i in range(len(cells))]
    start = _scramble_belt(cells, cols, rows, tiles, solved_list, seed, belt, player, belt_dir)
    return _spec_from_start(gid, block, cols, rows, cells, dirs, start)


def _scramble_free(cells, cols, rows, seed, slides=(6, 12)):
    """Random reachable arrangement (no solvability requirement)."""
    lines = _lines_of(cells, cols, rows)
    rng = random.Random(seed)
    p = [str(i) for i in range(len(cells))]
    for _ in range(rng.randint(*slides)):
        a, _i, lc = rng.choice(lines)
        p = _apply_list(cells, p, lc, rng.choice(_dirs_for(a)))
    return p


def _group_spec_green(gid, block, cols, rows, cells, dirs, green_cell, seed, slides=(6, 12)):
    """A grid with a fixed green TELEPORT SPOT (a background cell). Tiles slide
    over it; whichever tile is on the spot at teleport-time is swapped with the
    other grid's spot tile. `dirs` is the (unsolvable) initial multiset; the right
    tiles must be teleported in from the other grid to make it solvable."""
    tiles = {str(i): frozenset(dirs[i]) for i in range(len(cells))}
    assert not _any_solvable(cells, tiles), f"{gid}: green grid is solvable without a teleport"
    start = _scramble_free(cells, cols, rows, seed, slides)
    spec = _spec_from_start(gid, block, cols, rows, cells, dirs, start)
    spec["green_cell"] = list(green_cell)
    return spec


def _seg(a, b):
    return range(a, b + 1) if b >= a else range(a, b - 1, -1)


def _lpath(p0, p1, hfirst):
    (x0, y0), (x1, y1) = p0, p1
    pts = []
    if hfirst:
        pts += [(x, y0) for x in _seg(x0, x1)]
        pts += [(x1, y) for y in _seg(y0, y1) if y != y0]
    else:
        pts += [(x0, y) for y in _seg(y0, y1)]
        pts += [(x, y1) for x in _seg(x0, x1) if x != x0]
    return [list(p) for p in pts]


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------
def _build_level_1() -> Level:
    g = _group_spec(
        "M",
        (10, 22),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["D"], ["U", "R"], ["U", "L"]],
        seed=11,
        min_moves=3,
    )
    bulb = [54, 54]
    wires = [
        {"id": "wM", "src": "M", "path": _lpath((26, 32), tuple(bulb), True), "target": {"type": "bulb", "bulb": "b"}}
    ]
    data = {
        "groups": [g],
        "wires": wires,
        "bulbs": [{"id": "b", "pos": bulb, "on": BULB_ON, "role": "goal"}],
        "solve_order": ["M"],
        "energy": 30,
    }
    return _level_from_data("Level 1", data)


def _build_level_2() -> Level:
    # A: 2x2 U rotated 90 deg CW -- a cup opening to the RIGHT (endpoints on the
    # right column, back of the U down the left column). Top-left (top-right).
    A = _group_spec(
        "A",
        (3, 4),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["R", "D"], ["L"], ["U", "R"], ["L"]],
        seed=7,
        min_moves=3,
    )
    # B: 2x2 CIRCLE -- four corner elbows forming a closed loop (no endpoints),
    # so it reads as a ring rather than the same U as Level 1. Shuffled by seed.
    B = _group_spec(
        "B",
        (41, 4),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["R", "D"], ["L", "D"], ["U", "R"], ["U", "L"]],
        seed=11,
        min_moves=3,
    )
    # T-shape: 3-wide bar (row 0) + short 1-cell stem hanging from the centre col 1.
    T = _group_spec(
        "T",
        (18, 32),
        3,
        2,
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [["R"], ["L", "R", "D"], ["L"], ["U"]],
        seed=5,
        min_moves=2,
        need_both=True,
    )
    bulb = [8, 56]
    wires = [
        {
            "id": "wA",
            "src": "A",
            "path": _lpath((12, 20), (19, 39), False),
            "target": {"type": "line", "group": "T", "axis": "row", "index": 0},
        },
        {
            "id": "wB",
            "src": "B",
            "path": _lpath((50, 20), (30, 34), False),
            "target": {"type": "line", "group": "T", "axis": "col", "index": 1},
        },
        {"id": "wT", "src": "T", "path": _lpath((30, 51), tuple(bulb), False), "target": {"type": "bulb", "bulb": "b"}},
    ]
    data = {
        "groups": [A, B, T],
        "wires": wires,
        "bulbs": [{"id": "b", "pos": bulb, "on": BULB_ON, "role": "goal"}],
        "solve_order": ["A", "B", "T"],
        "energy": 90,
    }
    return _level_from_data("Level 2", data)


def _build_level_3() -> Level:
    # BL: 2x2 U (bottom-left); BR: 3x2 snake (bottom-right); L: corner (top-centre).
    # L's top row is an AUTO-BELT fed by BL (dotted wire, purple, moves every
    # action); its right column is the player-controlled line fed by BR.
    BL = _group_spec(
        "BL",
        (3, 40),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["D"], ["U", "R"], ["U", "L"]],
        seed=11,
        min_moves=3,
    )
    # 2x2 U (the old 3x2 snake with its leftmost column removed; the orphaned L arm
    # on the surviving top tile is dropped so the right two columns close into a U).
    BR = _group_spec(
        "BR",
        (37, 36),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["D"], ["U", "R"], ["U", "L"]],
        seed=4,
        min_moves=3,
    )
    # L: short 2-cell vertical (col 0) + 3-cell horizontal bottom row, corner (0,1).
    #   X o o
    #   X X X
    # col 0 (left, vertical, 2) is the auto-belt fed by BL; row 1 (bottom, 3) the
    # player line fed by BR. Bulb sits left of the vertical for a straight wire.
    lcells = [(0, 0), (0, 1), (1, 1), (2, 1)]
    ldirs = [["D"], ["U", "R"], ["L", "R"], ["L"]]
    L = _group_spec_belt("L", (24, 4), 3, 2, lcells, ldirs, seed=5, belt=("col", 0), player=("row", 1), belt_dir="D")
    bulb = [8, 11]
    wires = [
        {
            "id": "wBL",
            "src": "BL",
            "path": _lpath((12, 43), (31, 20), False),
            "style": "dotted",
            "target": {"type": "line", "group": "L", "axis": "col", "index": 0},
        },
        {
            "id": "wBR",
            "src": "BR",
            "path": _lpath((51, 39), (45, 16), False),
            "target": {"type": "line", "group": "L", "axis": "row", "index": 1},
        },
        {"id": "wL", "src": "L", "path": _lpath((27, 11), tuple(bulb), True), "target": {"type": "bulb", "bulb": "b"}},
    ]
    belts = [{"group": "L", "axis": "col", "index": 0, "direction": "D"}]
    data = {
        "groups": [BL, BR, L],
        "wires": wires,
        "bulbs": [{"id": "b", "pos": bulb, "on": BULB_ON, "role": "goal"}],
        "belts": belts,
        "solve_order": ["BL", "BR", "L"],
        "energy": 120,
    }
    return _level_from_data("Level 3", data)


def _build_level_4() -> Level:
    # R (2x2, top-centre) lights the GREEN bulb. P & Q (2x2, each with a green
    # TELEPORT tile) are locked until you click the lit green bulb: it blinks,
    # then the two green tiles swap between P and Q, unlocking them. Solved, P & Q
    # power the 4x3 CROSS (plus shape) -> final bulb.
    # R: a 2x2 square with one extra cell on the top-right -> 3 cells on top, 2 on
    # the bottom. Solving it lights the GREEN bulb.
    R = _group_spec(
        "R",
        (22, 3),
        3,
        2,
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        [["R", "D"], ["L", "R", "D"], ["L"], ["U", "R"], ["U", "L"]],
        seed=2,
        min_moves=3,
    )
    # P is a small L-tromino (a 2x2 square with the bottom-right block removed); Q
    # is a 2x2. Both start ACTIVE but UNSOLVABLE and each holds one shape the other
    # needs. The fixed green SPOT is cell (0,0): slide the right tile onto each
    # grid's spot, then teleport to swap them -- one swap makes both solvable.
    #   P solves to an L-net {RD,L,U}; it holds UL (which Q needs) instead of U.
    #   Q solves to a loop {RD,LD,UR,UL}; it holds U (which P needs) instead of UL.
    P = _group_spec_green(
        "P", (2, 22), 2, 2, [(0, 0), (1, 0), (0, 1)], [["R", "D"], ["L"], ["U", "L"]], green_cell=(0, 0), seed=3
    )
    Q = _group_spec_green(
        "Q",
        (44, 22),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["R", "D"], ["L", "D"], ["U", "R"], ["U"]],
        green_cell=(0, 0),
        seed=6,
    )
    # Cross: horizontal row 1 (3 wide) + SHORT vertical col 1 (2 tall: top arm +
    # centre). The vertical (col 1) is an AUTO-BELT fed by Q -- it advances on its
    # own each action; the player only clicks the horizontal row (fed by P).
    xcells = [(1, 0), (0, 1), (1, 1), (2, 1)]
    xdirs = [["D"], ["R"], ["U", "L", "R"], ["L"]]
    X = _group_spec_belt("X", (16, 38), 3, 2, xcells, xdirs, seed=5, belt=("col", 1), player=("row", 1), belt_dir="D")
    bulbs = [
        {"id": "green", "pos": [33, 28], "on": GREEN, "role": "teleport"},
        {"id": "final", "pos": [8, 57], "on": BULB_ON, "role": "goal"},
    ]
    wires = [
        {"id": "wR", "src": "R", "path": _lpath((33, 19), (33, 28), True), "target": {"type": "bulb", "bulb": "green"}},
        {
            "id": "wP",
            "src": "P",
            "path": _lpath((11, 38), (17, 49), False),
            "target": {"type": "line", "group": "X", "axis": "row", "index": 1},
        },
        {
            "id": "wQ",
            "src": "Q",
            "path": _lpath((53, 38), (28, 41), False),
            "style": "dotted",
            "target": {"type": "line", "group": "X", "axis": "col", "index": 1},
        },
        {"id": "wX", "src": "X", "path": _lpath((33, 54), (8, 57), False), "target": {"type": "bulb", "bulb": "final"}},
    ]
    belts = [{"group": "X", "axis": "col", "index": 1, "direction": "D"}]
    data = {
        "groups": [R, P, Q, X],
        "wires": wires,
        "bulbs": bulbs,
        "belts": belts,
        "teleport": {"trigger": "green", "groups": ["P", "Q"]},
        "solve_order": ["R", "P", "Q", "X"],
        "energy": 150,
    }
    return _level_from_data("Level 4", data)


def _build_level_5() -> Level:
    # L-grid (corner) has TWO auto-belts that cross at the corner: row 0 (belt_H,
    # powered by A=2x2) and col 2 (belt_V, powered by B=3x2). Each belt starts
    # only when its source solves, so the GAP between solving A and B sets their
    # relative phase -- time it so the L wires connect -> final bulb.
    # A and B start unsolvable: each holds 2 of the other's tiles. C (standalone
    # 2x2) lights the GREEN bulb, unlocking the teleport that swaps spot tiles
    # between A and B (two swaps fix both).
    # L is an L-tromino: 2-cell top row (belt_H) + 2-cell right col (belt_V),
    # sharing the corner (1,0).
    lcells = [(0, 0), (1, 0), (1, 1)]
    ldirs = [["R"], ["L", "D"], ["U"]]  # solved corner net
    lstart = [str(i) for i in range(len(lcells))]
    lstart = _apply_list(lcells, lstart, [(0, 0), (1, 0)], "R")  # scramble belt_H
    lstart = _apply_list(lcells, lstart, [(1, 0), (1, 1)], "D")  # scramble belt_V
    L = _spec_from_start("L", (24, 18), 2, 2, lcells, ldirs, lstart)

    A = _group_spec_green(
        "A",
        (1, 8),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["L", "D"], ["U", "R"], ["R", "D"]],  # {D,LD,UR,RD}; solves to U {D,D,UR,UL}
        green_cell=(0, 0),
        seed=3,
    )
    B = _group_spec_green(  # 2x2 (the old 2x3 with its bottom row cut); solves to a loop
        "B",
        (44, 6),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["U", "R"], ["U", "L"], ["U", "L"]],  # {D,UR,UL,UL}; solves to a circle after 2 swaps
        green_cell=(0, 0),
        seed=6,
    )
    C = _group_spec(
        "C",
        (4, 42),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["D"], ["D"], ["U", "R"], ["U", "L"]],
        seed=2,
        min_moves=3,
    )
    bulbs = [
        {"id": "green", "pos": [42, 53], "on": GREEN, "role": "teleport"},
        {"id": "final", "pos": [29, 41], "on": BULB_ON, "role": "goal"},
    ]
    wires = [
        {
            "id": "wA",
            "src": "A",
            "path": _lpath((16, 17), (28, 25), False),
            "style": "dotted",
            "target": {"type": "line", "group": "L", "axis": "row", "index": 0},
        },
        {
            "id": "wB",
            "src": "B",
            "path": _lpath((47, 12), (36, 22), True),
            "style": "dotted",
            "target": {"type": "line", "group": "L", "axis": "col", "index": 1},
        },
        {
            "id": "wC",
            "src": "C",
            "path": _lpath((20, 51), (42, 53), False),
            "target": {"type": "bulb", "bulb": "green"},
        },
        {"id": "wL", "src": "L", "path": _lpath((29, 29), (29, 41), True), "target": {"type": "bulb", "bulb": "final"}},
    ]
    belts = [
        {"group": "L", "axis": "row", "index": 0, "direction": "R"},
        {"group": "L", "axis": "col", "index": 1, "direction": "D"},
    ]
    data = {
        "groups": [L, A, B, C],
        "wires": wires,
        "bulbs": bulbs,
        "belts": belts,
        "teleport": {"trigger": "green", "groups": ["A", "B"]},
        "solve_order": ["C", "A", "B", "L"],
        "energy": 250,
    }
    return _level_from_data("Level 5", data)


def _build_level_6() -> Level:
    # Mechanic: ROTATION via a purple bulb. Solving A (a circle) lights the purple
    # rotate bulb; clicking it turns the tile on each PINK spot 90deg CW -- one spot
    # in B (top-right) and one in C (bottom-left). B's pink spot holds {R} (a
    # CW-rotated {D}); one click finishes its U -> lights the green bulb -> unlocks
    # the teleport between C and D. C holds a straight tile that can never connect;
    # the teleport swaps it for D's {L,D}, which on C's pink spot rotates to {U,L}
    # to finish C's U -> final bulb. (The bulb rotates BOTH spots each click, so
    # park a don't-care tile on the other spot while fixing one.)
    A = _group_spec(  # CIRCLE (four corner elbows) -> lights the purple rotate bulb
        "A",
        (2, 6),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["R", "D"], ["L", "D"], ["U", "R"], ["U", "L"]],
        seed=2,
        min_moves=3,
    )
    # B solved = U {D,D,UR,UL}; its pink spot (0,0) is where the wrong-oriented tile
    # must be rotated. Its tiles include one {R} (a CW-rotated {D}); the grid is
    # SHUFFLED, so you must both slide (bring {R} onto the pink spot, arrange the
    # rest) AND rotate it once ({R}->{D}). Slide-unsolvable as a multiset.
    B = _spec_from_start(
        "B",
        (34, 6),
        2,
        2,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [["R"], ["D"], ["U", "R"], ["U", "L"]],
        _scramble_free([(0, 0), (1, 0), (0, 1), (1, 1)], 2, 2, seed=4),
    )
    B["rotate_cell"] = [0, 0]
    # C & D are a MUTUAL teleport swap that fixes both. C solves to a U {D,D,UR,UL}
    # but holds an endpoint {L} where it needs {U,L}. D is a small L (vertical arm 2
    # + horizontal arm 2) that needs that {L} endpoint but holds a {L,D}. Put C's {L}
    # on C's green spot and D's {L,D} on D's green spot, then teleport: C gets {L,D}
    # (rotate it on the pink spot -> {U,L}), D gets {L}.
    c_cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    C = _group_spec_green("C", (3, 40), 2, 2, c_cells, [["D"], ["D"], ["U", "R"], ["L"]], green_cell=(0, 0), seed=7)
    C["rotate_cell"] = [1, 1]
    # D: small L -- vertical arm col 0 = (0,0),(0,1); horizontal arm row 1 = (0,1),
    # (1,1). Solved net (0,0)=D / (0,1)=UR / (1,1)=L. The VERTICAL slider works from
    # the start -- use it to bring the {L,D} tile (parked on col 0) onto the green
    # spot for the teleport. The HORIZONTAL slider is gated until C is solved; you
    # need it to finish arranging D -> final bulb. Start parks {L,D} at (0,1).
    d_cells = [(0, 0), (0, 1), (1, 1)]
    D = _spec_from_start(
        "D",
        (42, 35),
        2,
        2,
        d_cells,
        [["D"], ["U", "R"], ["L", "D"]],
        ["0", "2", "1"],  # (0,0)=D, (0,1)={L,D} (on the vertical arm), (1,1)=UR
    )
    D["green_cell"] = [0, 0]
    bulbs = [
        {"id": "rotate", "pos": [26, 13], "on": ROT_BTN, "role": "rotate"},
        {"id": "green", "pos": [28, 29], "on": GREEN, "role": "teleport"},
        {"id": "final", "pos": [33, 58], "on": BULB_ON, "role": "goal"},
    ]
    wires = [
        {
            "id": "wA",
            "src": "A",
            "path": _lpath((18, 13), (26, 13), True),
            "target": {"type": "bulb", "bulb": "rotate"},
        },
        {
            "id": "wB",
            "src": "B",
            "path": _lpath((40, 22), (28, 29), False),
            "target": {"type": "bulb", "bulb": "green"},
        },
        {  # C powers D's HORIZONTAL slider (row 1)
            "id": "wC",
            "src": "C",
            "path": _lpath((19, 50), (47, 50), True),
            "target": {"type": "line", "group": "D", "axis": "row", "index": 1},
        },
        {
            "id": "wD",
            "src": "D",
            "path": _lpath((52, 50), (33, 58), False),
            "target": {"type": "bulb", "bulb": "final"},
        },
    ]
    data = {
        "groups": [A, B, C, D],
        "wires": wires,
        "bulbs": bulbs,
        "teleport": {"trigger": "green", "groups": ["C", "D"]},
        "solve_order": ["A", "B", "C", "D"],
        "energy": 300,
    }
    return _level_from_data("Level 6", data)


def _level_from_data(name, data) -> Level:
    screen = Sprite(pixels=[[BLACK] * GRID for _ in range(GRID)], name="screen", collidable=False, layer=0)
    return Level(name=name, sprites=[screen], grid_size=(GRID, GRID), data=data)


LEVEL_BUILDERS = [_build_level_1, _build_level_2, _build_level_3, _build_level_4, _build_level_5, _build_level_6]


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class En18(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [build() for build in LEVEL_BUILDERS]
        camera = Camera(0, 0, GRID, GRID, BLACK, BLACK, [])
        super().__init__(
            "en18", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[6], seed=seed
        )

    # -- per-level runtime state (rebuilt from data so resets stay clean) --
    def on_set_level(self, level: Level) -> None:
        specs = level.get_data("groups")
        self._groups = {spec["id"]: Group(spec) for spec in specs}
        self._green = {s["id"]: tuple(s["green_cell"]) for s in specs if "green_cell" in s}  # gid -> green spot cell
        self._wires = [dict(w) for w in level.get_data("wires")]
        self._solve_order = list(level.get_data("solve_order"))
        self._gate = {}  # (gid, axis, index) -> wire index
        bulb_wire_of = {}  # bulb id -> wire index
        for wi, w in enumerate(self._wires):
            t = w["target"]
            if t["type"] == "line":
                self._gate[(t["group"], t["axis"], t["index"])] = wi
            else:
                bulb_wire_of[t["bulb"]] = wi
        # rotate spots: a fixed pink background cell; clicking the lit rotate bulb
        # turns whatever tile sits on each spot 90deg CW (gid -> cell).
        self._rotate_spots = {s["id"]: tuple(s["rotate_cell"]) for s in specs if "rotate_cell" in s}
        self._bulbs = []
        for b in level.get_data("bulbs"):
            self._bulbs.append(
                {
                    "id": b["id"],
                    "pos": tuple(b["pos"]),
                    "on": b.get("on", BULB_ON),
                    "role": b.get("role", "goal"),
                    "wire": bulb_wire_of.get(b["id"]),
                }
            )
        self._goal_wire = next((b["wire"] for b in self._bulbs if b["role"] == "goal"), None)
        self._belt = {}  # (gid, axis, index) -> auto direction
        for b in level.get_data("belts") or []:
            self._belt[(b["group"], b["axis"], b["index"])] = b["direction"]
        self._teleport = level.get_data("teleport")  # {trigger bulb id, groups:[g1,g2]} or None
        self._teleport_gated = set(level.get_data("teleport_gated") or [])
        self._teleported = False
        self._energized = set()
        self._anim = None
        self._anim_queue = []
        self._flow = None
        self._flow_queue = []
        self._blink = None
        self._energy_cap = int(level.get_data("energy"))
        self._energy = self._energy_cap
        self._handled_action_count = self._action_count
        self._paint()

    # -- queries -----------------------------------------------------------
    def _line_enabled(self, gid, axis, index):
        if gid in self._teleport_gated and not self._teleported:
            return False
        w = self._gate.get((gid, axis, index))
        return w is None or w in self._energized

    def _belt_active(self, gid, axis, index):
        return (gid, axis, index) in self._belt and self._line_enabled(gid, axis, index)

    def _line_clickable(self, gid, axis, index):
        return self._line_enabled(gid, axis, index) and (gid, axis, index) not in self._belt

    def _rotate_bulb(self):
        return next((b for b in self._bulbs if b["role"] == "rotate"), None)

    def _hit_rotate_bulb(self, gx, gy):
        """The lit purple rotate bulb, if clicked."""
        b = self._rotate_bulb()
        if b is None or b["wire"] is None or b["wire"] not in self._energized:
            return False
        cx, cy = b["pos"]
        return cx - 2 <= gx <= cx + 2 and cy - 2 <= gy <= cy + 2

    def _do_rotate_spots(self):
        # rotate the tile sitting on each marked pink spot 90deg CW
        for gid, cell in self._rotate_spots.items():
            g = self._groups[gid]
            tid = g.placement[cell]
            g.tiles[tid] = _rotate_dirs(g.tiles[tid], 1)

    def _group_powered(self, gid):
        for wi, w in enumerate(self._wires):
            if w["src"] == gid and (wi in self._energized or (self._flow and self._flow["wire"] == wi)):
                return True
        return False

    def _screen(self):
        return self.current_level.get_sprites_by_name("screen")[0]

    # -- rendering ---------------------------------------------------------
    def _paint(self):
        f = np.full((GRID, GRID), BLACK, dtype=np.int8)
        # energy bar
        fill = round(GRID * max(0, self._energy) / self._energy_cap)
        f[0:ENERGY_ROWS, :] = GRAY_EMPTY
        if fill > 0:
            f[0:ENERGY_ROWS, 0:fill] = GRAY
        # board frame
        f[BOARD_TOP, :] = WHITE
        f[GRID - 1, :] = WHITE
        f[BOARD_TOP:GRID, 0] = WHITE
        f[BOARD_TOP:GRID, GRID - 1] = WHITE
        # wires
        for wi, w in enumerate(self._wires):
            energ = wi in self._energized
            flowing = bool(self._flow) and self._flow["wire"] == wi
            prog = self._flow["f"] * FLOW_PX if flowing else (len(w["path"]) if energ else 0)
            dotted = w.get("style") == "dotted"
            n = len(w["path"])
            for idx, (x, y) in enumerate(w["path"]):
                # dotted: skip odd interior pixels, but keep both ends solid so
                # the wire visibly connects to its source and target button
                if dotted and idx % 2 == 1 and 1 < idx < n - 2:
                    continue
                f[y, x] = ORANGE if (energ or (flowing and idx < prog)) else WHITE
        # green flash while a teleport is firing
        green_vis = True
        if self._blink is not None:
            green_vis = (self._blink["f"] // BLINK_PERIOD) % 2 == 0
        # bulbs
        self._paint_bulbs(f, green_vis)
        # groups
        for gid, g in self._groups.items():
            self._paint_group(f, gid, g, green_vis)
        self._screen().pixels = f

    def _paint_bulbs(self, f, green_vis):
        for b in self._bulbs:
            cx, cy = b["pos"]
            lit = b["wire"] is not None and b["wire"] in self._energized
            col = b["on"] if lit else BULB_OFF
            if b["role"] == "teleport" and not green_vis:
                col = BULB_OFF  # flash off during the blink
            # 1px frame in the bulb's energized colour, so you can see what
            # colour it will turn before it lights
            f[cy - 3 : cy + 4, cx - 3 : cx + 4] = b["on"]
            f[cy - 2 : cy + 3, cx - 2 : cx + 3] = col  # filled 5x5 square

    def _paint_group(self, f, gid, g, green_vis=True):
        # core frame (1px outline hugging the cell shape)
        frame_col = ORANGE if self._group_powered(gid) else WHITE
        f[g.frame_mask()] = frame_col
        # buttons: grey (disabled) / blue (clickable player line) / purple (running
        # auto-belt -- NOT clickable; it advances one step on any blue-button click).
        # Grey while a belt is still unpowered.
        for axis, index, _lc in g.lines():
            if (gid, axis, index) in self._belt:
                col = BTN_BELT if self._belt_active(gid, axis, index) else BTN_OFF
            else:
                col = BTN if self._line_enabled(gid, axis, index) else BTN_OFF
            for bx0, by0, bw, bh, _d in g.line_buttons(axis, index):
                f[by0 : by0 + bh, bx0 : bx0 + bw] = col
        # green spot (teleport) and pink spot (rotate): FIXED background cells; the
        # tile on each shows that colour, and is drawn on top.
        gcell = self._green.get(gid)
        rcell = self._rotate_spots.get(gid)
        if gcell is not None and green_vis:
            gx, gy = g.tile_origin(*gcell)
            f[gy : gy + TILE, gx : gx + TILE] = GREEN
        if rcell is not None:
            rx, ry = g.tile_origin(*rcell)
            f[ry : ry + TILE, rx : rx + TILE] = ROT_BTN
        # tiles
        moving = set()
        if self._anim and self._anim["gid"] == gid:
            moving = self._paint_group_anim(f, g)
        for cell, tid in g.placement.items():
            if cell in moving:
                continue
            bg = GREEN if (cell == gcell and green_vis) else (ROT_BTN if cell == rcell else BLACK)
            wcol = WIRE if _tile_connected(g.cellset, g.placement, g.tiles, cell) else WHITE
            tx, ty = g.tile_origin(*cell)
            f[ty : ty + TILE, tx : tx + TILE] = _tile_pixels(g.tiles, tid, bg, wcol)

    def _paint_group_anim(self, f, g):
        a = self._anim
        lc = [tuple(c) for c in a["cells"]]
        n = len(lc)
        k = a["k"]
        tx, ty = g.tile_origin(*lc[0])
        if a["axis"] == "row":
            strip = np.full((TILE, n * TILE), BLACK, dtype=np.int8)
            for s, cell in enumerate(lc):
                strip[:, s * TILE : (s + 1) * TILE] = _tile_pixels(g.tiles, g.placement[cell], wire_col=WHITE)
            strip = np.roll(strip, k if a["direction"] == "R" else -k, axis=1)
            region = f[ty : ty + TILE, tx : tx + n * TILE]
        else:
            strip = np.full((n * TILE, TILE), BLACK, dtype=np.int8)
            for s, cell in enumerate(lc):
                strip[s * TILE : (s + 1) * TILE, :] = _tile_pixels(g.tiles, g.placement[cell], wire_col=WHITE)
            strip = np.roll(strip, k if a["direction"] == "D" else -k, axis=0)
            region = f[ty : ty + n * TILE, tx : tx + TILE]
        # black is transparent so the fixed green spot underneath shows through
        mask = strip != BLACK
        region[mask] = strip[mask]
        return set(lc)

    # -- action handling ---------------------------------------------------
    def _slide_desc(self, gid, axis, index, direction):
        lc = self._groups[gid].line_cells(axis, index)
        return {
            "gid": gid,
            "axis": axis,
            "index": index,
            "cells": [list(c) for c in lc],
            "direction": direction,
            "k": 0,
        }

    def _hit_clickable(self, gx, gy):
        for gid, g in self._groups.items():
            for axis, index, _lc in g.lines():
                if not self._line_clickable(gid, axis, index):
                    continue
                for bx0, by0, bw, bh, d in g.line_buttons(axis, index):
                    if bx0 <= gx < bx0 + bw and by0 <= gy < by0 + bh:
                        return self._slide_desc(gid, axis, index, d)
        return None

    def _hit_teleport_bulb(self, gx, gy):
        """The lit green teleport bulb, if clicked (repeatable -> toggles)."""
        if not self._teleport:
            return False
        b = next((b for b in self._bulbs if b["id"] == self._teleport["trigger"]), None)
        if b is None or b["wire"] not in self._energized:
            return False
        cx, cy = b["pos"]
        return cx - 2 <= gx <= cx + 2 and cy - 2 <= gy <= cy + 2

    def _do_teleport(self):
        g1, g2 = self._teleport["groups"]
        G1, G2 = self._groups[g1], self._groups[g2]
        t1 = G1.placement[self._green[g1]]  # tile currently ON g1's green spot
        t2 = G2.placement[self._green[g2]]
        G1.tiles[t1], G2.tiles[t2] = G2.tiles[t2], G1.tiles[t1]  # swap the two spot tiles
        self._teleported = not self._teleported

    def _begin_action(self):
        self._energy = max(0, self._energy - 1)
        data = self.action.data or {}
        gx, gy = int(data.get("x", -1)), int(data.get("y", -1))
        if self._hit_teleport_bulb(gx, gy):  # click lit green bulb -> blink -> teleport
            self._blink = {"f": 0}
            self._anim = None
            self._anim_queue = []
            return
        if self._hit_rotate_bulb(gx, gy):  # click lit pink bulb -> rotate the marked spots
            self._do_rotate_spots()
            self._anim = None
            self._anim_queue = []
            return
        queue = []
        hit = self._hit_clickable(gx, gy)  # player's own move (one blue line)
        if hit is not None:
            queue.append(hit)
        # belts tick ONLY alongside a player's blue-button slide. The purple belt
        # buttons are not clickable -- clicking one (or empty space) is a no-op.
        if hit is not None:
            for (gid, axis, index), direction in self._belt.items():
                if self._line_enabled(gid, axis, index):
                    queue.append(self._slide_desc(gid, axis, index, direction))
        self._anim = queue[0] if queue else None
        self._anim_queue = queue[1:]

    def _commit_slide(self):
        a = self._anim
        self._groups[a["gid"]].slide([tuple(c) for c in a["cells"]], a["direction"])
        self._anim = None

    def _update_power(self):
        """Power is DYNAMIC: a wire stays energised only while its source group is
        still solved. If the net breaks up (e.g. a belt slides a connected pattern
        apart, or the player un-solves a source), it unpowers -- which stops any
        belt it gated and unlights any bulb it fed. Re-solving the source re-powers
        it (with a fresh flow animation via _collect_flows). This lets you pause a
        belt by breaking its source and re-time it before reconnecting."""
        for wi, w in enumerate(self._wires):
            if wi in self._energized and not self._groups[w["src"]].solved():
                self._energized.discard(wi)

    def _collect_flows(self):
        for gid, g in self._groups.items():
            if not g.solved():
                continue
            for wi, w in enumerate(self._wires):
                if w["src"] == gid and wi not in self._energized and wi not in self._flow_queue:
                    self._flow_queue.append(wi)

    def _finish_action_or_flow(self):
        """After all slides commit: drop power from any broken net, start any
        solve-triggered flows, else finish."""
        self._update_power()
        self._collect_flows()
        if self._flow_queue:
            self._flow = {"wire": self._flow_queue.pop(0), "f": 0}
            self._paint()
            return
        self._paint()
        self.complete_action()
        if self._energy <= 0:
            self.lose()

    def step(self):
        if self._action_count != self._handled_action_count:
            self._handled_action_count = self._action_count
            self._begin_action()

        # Phase 0: teleport blink, then swap the green tiles
        if self._blink is not None:
            self._blink["f"] += 1
            self._paint()
            if self._blink["f"] >= BLINK_TOTAL:
                self._do_teleport()
                self._blink = None
                self._paint()
                self.complete_action()
            return

        # Phase 1: slide animations (player move + any active belts, in sequence)
        if self._anim is not None:
            self._anim["k"] += 1
            if self._anim["k"] >= TILE:
                self._commit_slide()
                if self._anim_queue:
                    self._anim = self._anim_queue.pop(0)
                    self._paint()
                    return
                self._finish_action_or_flow()
                return
            self._paint()
            return

        # Phase 2: energy-flow finale / gate energisation (possibly several)
        if self._flow is not None:
            self._flow["f"] += 1
            wi = self._flow["wire"]
            path_len = len(self._wires[wi]["path"])
            if self._flow["f"] * FLOW_PX >= path_len:
                self._energized.add(wi)  # latch on contact so hold frames show it
            self._paint()
            if self._flow["f"] * FLOW_PX >= path_len + FLOW_HOLD_PX:
                self._flow = None
                if self._flow_queue:
                    self._flow = {"wire": self._flow_queue.pop(0), "f": 0}
                    return
                self.complete_action()
                if self._goal_wire is not None and self._goal_wire in self._energized:
                    self.next_level()
            return

        # No animation: instant action (rotate / empty click) -- may solve a grid
        self._finish_action_or_flow()
