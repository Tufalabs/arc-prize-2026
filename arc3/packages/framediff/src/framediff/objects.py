"""Objects matched across two frames, and the moves they propose to the search.

A piece is a 4-connected component of one colour, of the start frame or the target, that touches a cell the
two frames disagree on: what did not change near anything that did proposes nothing. A start piece and a
target piece match when they have one colour and one shape, the target's turned or mirrored included; a
match whose piece did not move is the object left where it was. Within a group of identical pieces the
nearest are paired first, so two identical objects that moved are each matched to the place nearest it.

Each match that moved proposes a ``MaskedMove``: the piece's cells picked by colour in its box, cleared to the
colour the target shows most where it left, written where it went. Touching pieces of several colours that
moved by one offset are proposed together as well, as one object. A proposal is made only where the colour
mask picks out the object exactly: a box holding another cell of the object's colours would carry that too.

The proposals are candidates like any other: the search scores each by what it costs against the cells it
makes right, on the board as the earlier edits left it, and takes one only when it is the cheapest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from framediff.grid import GRID_SIZE, Grid
from framediff.transforms import WILDCARD, BlockCopy, MaskedMove

#: The largest piece matched: past it a component is a background or a wall, which moves rarely and costs a
#: search much to compare.
MAX_PIECE_CELLS = 400

#: The most pieces of one colour and shape that are paired: past it the group is a texture, not objects.
MAX_GROUP = 40

#: The most moves proposed for one pair of frames, nearest matches first.
MAX_PROPOSALS = 48

#: The four ways a cell touches another.
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

#: The eight ways a shape may be mirrored and turned, identity first, as (flip, rot): ``BlockCopy``'s order.
_VARIANTS: tuple[tuple[bool, int], ...] = tuple((flip, rot) for flip in (False, True) for rot in range(4))


@dataclass
class Piece:
    """One 4-connected component of one colour, its box half-open."""

    color: int
    cells: frozenset[tuple[int, int]]
    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """The piece's cells inside its box."""
        mask = np.zeros((self.row1 - self.row0, self.col1 - self.col0), dtype=bool)
        for row, col in self.cells:
            mask[row - self.row0, col - self.col0] = True
        return mask

    @property
    def key(self) -> tuple[int, tuple[int, ...], bytes]:
        """Colour and exact shape: what an unturned match shares."""
        mask = self.mask
        return self.color, mask.shape, np.packbits(mask).tobytes()

    @property
    def centre(self) -> tuple[int, int]:
        """The box's centre, doubled to stay whole."""
        return self.row0 + self.row1, self.col0 + self.col1


def pieces(frame: Grid, near: npt.NDArray[np.bool_]) -> list[Piece]:
    """The pieces of ``frame`` that hold a cell of ``near``, in reading order of the first such cell; wildcard
    cells belong to none, and a piece past ``MAX_PIECE_CELLS`` is left out."""
    seen = np.zeros(frame.shape, dtype=bool)
    found: list[Piece] = []
    for row, col in np.argwhere(near):
        if seen[row, col] or frame[row, col] == WILDCARD:
            continue
        color = int(frame[row, col])
        cells = {(int(row), int(col))}
        seen[row, col] = True
        stack = [(int(row), int(col))]
        while stack and len(cells) <= MAX_PIECE_CELLS:
            r, c = stack.pop()
            for dr, dc in _STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not seen[nr, nc] and frame[nr, nc] == color:
                    seen[nr, nc] = True
                    cells.add((nr, nc))
                    stack.append((nr, nc))
        if len(cells) > MAX_PIECE_CELLS:
            continue
        rows, cols = zip(*cells)
        found.append(Piece(color, frozenset(cells), min(rows), max(rows) + 1, min(cols), max(cols) + 1))
    return found


def _variant(mask: npt.NDArray[np.bool_], flip: bool, rot: int) -> npt.NDArray[np.bool_]:
    """The mask mirrored and turned as ``MaskedMove`` mirrors and turns an object."""
    mask = np.fliplr(mask) if flip else mask
    return np.rot90(mask, rot) if rot else mask


def _paired(starts: list[Piece], targets: list[Piece]) -> list[tuple[Piece, Piece]]:
    """Pieces paired nearest first, each used once."""
    distances = sorted(
        (abs(a.centre[0] - b.centre[0]) + abs(a.centre[1] - b.centre[1]), i, j)
        for i, a in enumerate(starts)
        for j, b in enumerate(targets)
    )
    used_start: set[int] = set()
    used_target: set[int] = set()
    pairs: list[tuple[Piece, Piece]] = []
    for _, i, j in distances:
        if i not in used_start and j not in used_target:
            used_start.add(i)
            used_target.add(j)
            pairs.append((starts[i], targets[j]))
    return pairs


@dataclass
class Match:
    """A start piece found again in the target, mirrored and turned by ``(flip, rot)``."""

    start: Piece
    target: Piece
    flip: bool = False
    rot: int = 0

    @property
    def offset(self) -> tuple[int, int]:
        """How far its box moved."""
        return self.target.row0 - self.start.row0, self.target.col0 - self.start.col0


def matches(start: Grid, target: Grid) -> list[Match]:
    """Every piece found again in the target at another place or turned, nearest first.

    Pieces of one colour and shape are paired nearest first, after those that stayed where they were are set
    aside; a start piece left over is then looked for among the target's leftovers of its colour turned or
    mirrored. A group past ``MAX_GROUP`` on either side is not paired."""
    live = start != WILDCARD
    near = (start != target) & live
    grown = near.copy()
    grown[1:] |= near[:-1]
    grown[:-1] |= near[1:]
    grown[:, 1:] |= near[:, :-1]
    grown[:, :-1] |= near[:, 1:]
    groups: dict[tuple[int, tuple[int, ...], bytes], tuple[list[Piece], list[Piece]]] = {}
    for side, frame in enumerate((start, target)):
        for piece in pieces(frame, grown):
            groups.setdefault(piece.key, ([], []))[side].append(piece)
    found: list[Match] = []
    left_start: list[Piece] = []
    left_target: list[Piece] = []
    for olds, news in groups.values():
        stayed = {(p.row0, p.col0) for p in olds} & {(p.row0, p.col0) for p in news}
        olds = [p for p in olds if (p.row0, p.col0) not in stayed]
        news = [p for p in news if (p.row0, p.col0) not in stayed]
        if len(olds) > MAX_GROUP or len(news) > MAX_GROUP:
            continue
        pairs = _paired(olds, news)
        found += [Match(a, b) for a, b in pairs]
        paired_start = {id(a) for a, _ in pairs}
        paired_target = {id(b) for _, b in pairs}
        left_start += [p for p in olds if id(p) not in paired_start]
        left_target += [p for p in news if id(p) not in paired_target]
    for old in left_start:
        shape = old.mask
        for new in sorted(
            left_target, key=lambda p: abs(p.centre[0] - old.centre[0]) + abs(p.centre[1] - old.centre[1])
        ):
            if new.color != old.color or len(new.cells) != len(old.cells):
                continue
            variant = next(
                ((flip, rot) for flip, rot in _VARIANTS[1:] if np.array_equal(_variant(shape, flip, rot), new.mask)),
                None,
            )
            if variant is not None:
                found.append(Match(old, new, *variant))
                left_target.remove(new)
                break
    return sorted(found, key=lambda m: abs(m.offset[0]) + abs(m.offset[1]))


def _clear(target: Grid, left: npt.NDArray[np.bool_]) -> int | None:
    """The colour the target shows most on the cells an object left, None when it left none the target asks a
    colour of (a wildcard of the target takes whatever is left there)."""
    colors = target[left]
    colors = colors[colors != WILDCARD]
    if not len(colors):
        return None
    values, counts = np.unique(colors, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _proposal(
    start: Grid, target: Grid, cells: frozenset[tuple[int, int]], colors: tuple[int, ...], match: Match
) -> MaskedMove | BlockCopy | None:
    """The masked move of ``cells`` by ``match``'s offset and turn, or None when the colour mask would pick other
    cells of its box too or the move would leave the frame. An object that fills its box is a plain move of the
    box: the mask would pick every cell of it."""
    rows, cols = zip(*cells)
    row0, row1, col0, col1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
    mask = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for row, col in cells:
        mask[row - row0, col - col0] = True
    if not np.array_equal(np.isin(start[row0:row1, col0:col1], colors), mask):
        return None
    placed = _variant(mask, match.flip, match.rot)
    to_row = row0 + match.offset[0] if not (match.flip or match.rot) else match.target.row0
    to_col = col0 + match.offset[1] if not (match.flip or match.rot) else match.target.col0
    height, width = placed.shape
    if not (0 <= to_row and to_row + height <= GRID_SIZE and 0 <= to_col and to_col + width <= GRID_SIZE):
        return None
    landed = np.zeros(start.shape, dtype=bool)
    landed[to_row : to_row + height, to_col : to_col + width] = placed
    left = np.zeros(start.shape, dtype=bool)
    left[row0:row1, col0:col1] = mask
    clear = _clear(target, left & ~landed)
    if mask.all():
        return BlockCopy(
            to_row,
            to_row + height,
            to_col,
            to_col + width,
            src_row=row0,
            src_col=col0,
            rot=match.rot,
            flip=match.flip,
            clear=clear,
        )
    return MaskedMove(
        to_row,
        to_row + height,
        to_col,
        to_col + width,
        src_row=row0,
        src_col=col0,
        colors=colors,
        rot=match.rot,
        flip=match.flip,
        clear=clear,
    )


def proposals(start: Grid, target: Grid, distance_cost: float = 0.0) -> list[MaskedMove | BlockCopy]:
    """The masked moves the matches propose, nearest first and at most ``MAX_PROPOSALS``: each piece that moved
    or turned, and each group of touching pieces that moved by one offset without turning, as one object; a plain
    move where the object fills its box."""
    found: list[MaskedMove | BlockCopy] = []
    plain: dict[tuple[int, int], list[Match]] = {}
    for match in matches(start, target):
        move = _proposal(start, target, match.start.cells, (match.start.color,), match)
        if move is not None:
            found.append(move)
        if not (match.flip or match.rot):
            plain.setdefault(match.offset, []).append(match)
    for group in plain.values():
        for joined in touching(group):
            if len(joined) < 2:
                continue
            cells = frozenset(cell for m in joined for cell in m.start.cells)
            colors = tuple(sorted({m.start.color for m in joined}))
            move = _proposal(start, target, cells, colors, joined[0])
            if move is not None:
                found.append(move)
    found.sort(key=lambda move: move.distance)
    return [move.priced(distance_cost) for move in found[:MAX_PROPOSALS]]


def touching(group: list[Match]) -> list[list[Match]]:
    """The matches of one offset in groups whose start pieces touch, each group in the order given."""
    owner: dict[tuple[int, int], int] = {}
    for index, match in enumerate(group):
        for cell in match.start.cells:
            owner[cell] = index
    parent = list(range(len(group)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, match in enumerate(group):
        for row, col in match.start.cells:
            for dr, dc in _STEPS:
                other = owner.get((row + dr, col + dc))
                if other is not None and other != index:
                    parent[root(other)] = root(index)
    joined: dict[int, list[Match]] = {}
    for index, match in enumerate(group):
        joined.setdefault(root(index), []).append(match)
    return list(joined.values())
