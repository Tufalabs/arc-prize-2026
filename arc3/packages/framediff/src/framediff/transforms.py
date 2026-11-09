"""Edits that turn one frame into another, and the numpy code that performs them.

Every transform exists twice: ``transform(grid)`` performs it here, and
``to_code()`` writes it as numpy that the model reads and the harness runs.
The code is the product, so ``to_code`` checks its own text against
``transform`` before returning it.

Emitted code is a block over three names: the frame the edits start from,
the frame they must reach, and the board being built.

    g = a.copy()
    g[27:30, 36:39] = 0  # cleared behind the move below
    g[27:30, 30:33] = a[27:30, 36:39]  # moved left 6 (colors: 0, 9)
    assert (g == b).all()

A plain copy or move says which way its content went, and the colours it
changes cells to (2026-10-08 and 2026-10-09): the slice reads the
other way round, and names no colour. Those are read from the frame the edit starts from
(``with_colors``), since an edit holds none.

Each transform contributes its own lines; the opening copy and the closing
assert belong to ``Transform.to_code``, so any transform on its own is a
complete block. ``NoOp`` is the one exception: nothing changed, so its block
is only ``assert (a == b).all()``, with no board at all. Rectangles are half-open, and the fields spell them exactly
as the slices they emit: ``row0:row1, col0:col1``.

The start frame may hold wildcards, cells of ``-1`` that stand for "any
colour": a model's render says so where it does not know. A wildcard stays
a wildcard through every edit here, whatever a transform writes on it, and
a block emitted for such a start resolves them from the target just before
its assert. The target never holds one.

A block may call two helpers, ``from_ascii_grid`` and ``bresenham_line``,
and takes both from where it runs: the REPL the model works in preloads
them, and ``block_namespace`` binds framediff's own everywhere else. Those
two and ``np`` are all a block needs. The REPL advertises
``from_ascii_grid`` but not ``bresenham_line``, so a block that draws a line
says, once, where it comes from.

Bit costs are declared per kind, not derived from an encoding. Their only
job is the ordering they set, which is what makes a search pick the edit a
person would name: a copy for a moved sprite, a fill for a cleared region,
a single cell for a single cell.
"""

from __future__ import annotations

import ast
import dataclasses
import keyword
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt

from framediff.drawing import bresenham_line
from framediff.grid import GRID_SIZE, NUM_COLORS, Grid, from_ascii_grid, to_ascii_grid

#: The comment a block that draws a line carries, once, ahead of its first
#: line: the REPL preloads ``bresenham_line`` without advertising it.
LINE_NOTE = "# bresenham_line is already loaded; see help(bresenham_line)"

#: The value of a cell in a start frame that stands for any colour.
WILDCARD = -1


def block_namespace(**frames: Grid) -> dict[str, Any]:
    """The names a block runs with outside the REPL: ``np``, the two helpers, and the frames passed."""
    return {"np": np, "from_ascii_grid": from_ascii_grid, "bresenham_line": bresenham_line, **frames}


class Transform(ABC):
    """An edit from one frame toward another: performed by ``transform``,
    written by ``to_code``, priced by ``bit_cost``."""

    def transform(self, grid: npt.ArrayLike) -> Grid:
        """``grid`` with this edit applied, as a new int8 array.

        ``grid`` is both the start frame the edit reads and the board's first
        value, as ``g = a.copy()`` has it; it is not modified. Its wildcard
        cells are wildcards in the result too.
        """
        start = np.array(grid)
        if start.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"a frame is {GRID_SIZE}x{GRID_SIZE}, got shape {start.shape}")
        if start.min() < WILDCARD or start.max() >= NUM_COLORS:
            raise ValueError(
                f"a frame holds colours 0-{NUM_COLORS - 1}, or {WILDCARD} for a wildcard, got {start.min()}..{start.max()}"
            )
        start = start.astype(np.int8)
        start.flags.writeable = False
        board = start.copy()
        self._apply(board, start)
        board[start == WILDCARD] = WILDCARD
        return board

    def to_code(self, a: str = "a", b: str = "b", g: str = "g", *, test: bool = True, wildcards: bool = False) -> str:
        """This edit as a numpy block that builds ``b`` from ``a`` on the board ``g``.

        ``a`` and ``b`` are the expressions the block runs against: a name, or
        a chain of attributes, subscripts and calls on one, such as
        ``history[-2].current_frame.grid``. ``g`` is the name the block
        assigns. ``wildcards`` says ``a`` holds wildcard cells, which the
        block then resolves from ``b`` before its assert. ``test``, on by
        default, runs the block on probe frames: it must reach what
        ``transform`` reaches and leave its start frame untouched.
        """
        _check_names(a, b, g)
        if test:
            self._self_test(wildcards)
        return "\n".join(self._block(a, b, g, wildcards))

    def priced(self, distance_cost: float) -> Transform:
        """This edit with every copy and move in it priced at ``distance_cost`` bits per cell it travels
        (``BlockCopy.distance_cost``); the edit itself is the same."""
        return self

    def bit_cost(self) -> float:
        """This edit's price in a description of the change (see the module docstring)."""
        cost = float(self._bit_cost())
        assert cost >= 0, f"{type(self).__name__} priced itself at {cost}"
        return cost

    def _self_test(self, wildcards: bool) -> None:
        """Run the block, under plain names, on every probe frame."""
        code = self.to_code(test=False, wildcards=wildcards)
        for probe in _probe_frames(wildcards):
            expected = self.transform(probe)
            start = probe.copy()
            namespace = block_namespace(a=start, b=expected)
            try:
                exec(code, namespace)
            except AssertionError as error:
                raise AssertionError(f"the emitted block does not reach transform()'s result:\n{code}") from error
            if not np.array_equal(start, probe):
                raise AssertionError(f"the emitted block writes to its start frame:\n{code}")

    def _block(self, a: str, b: str, g: str, wildcards: bool) -> list[str]:
        """The whole block: notes, the copy onto the board, this edit's lines, the wildcards resolved, the assert."""
        notes = list(dict.fromkeys(self._notes()))
        resolve = [f"{g}[{a} == {WILDCARD}] = {b}[{a} == {WILDCARD}]  # resolve wildcards"] if wildcards else []
        return [*notes, f"{g} = {a}.copy()", *self._code_lines(a, g), *resolve, f"assert ({g} == {b}).all()"]

    def _notes(self) -> list[str]:
        """Comment lines the block opens with for this edit; the block keeps one of each."""
        return []

    @abstractmethod
    def _apply(self, board: Grid, start: Grid) -> None:
        """Apply the edit to ``board`` in place; ``start`` is the read-only start frame."""

    @abstractmethod
    def _code_lines(self, a: str, g: str) -> list[str]:
        """The block's lines for this edit alone, with ``a`` the start frame and ``g`` the board."""

    @abstractmethod
    def _bit_cost(self) -> float:
        """This edit's declared cost."""


@dataclass
class NoOp(Transform):
    """Nothing changed: the frames are equal.

    Its block is the comparison alone, ``assert (a == b).all()``, or with
    wildcards in ``a`` ``assert ((a == -1) | (a == b)).all()``; there is no
    board to build. Inside a composite it adds nothing.
    """

    COST: ClassVar[float] = 0

    def _block(self, a: str, b: str, g: str, wildcards: bool) -> list[str]:
        """Only the comparison."""
        if wildcards:
            return [f"assert (({a} == {WILDCARD}) | ({a} == {b})).all()"]
        return [f"assert ({a} == {b}).all()"]

    def _apply(self, board: Grid, start: Grid) -> None:
        """Nothing to apply."""

    def _code_lines(self, a: str, g: str) -> list[str]:
        """No lines of its own."""
        return []

    def _bit_cost(self) -> float:
        """Free: nothing is described."""
        return self.COST


@dataclass
class SetPixel(Transform):
    """One cell set to one colour."""

    row: int
    col: int
    color: int

    COST: ClassVar[float] = 4

    def __post_init__(self) -> None:
        _coerce_ints(self, "row", "col", "color")
        _check_span(self.row, self.row + 1, "row")
        _check_span(self.col, self.col + 1, "col")
        _check_color(self.color, "color")

    def _apply(self, board: Grid, start: Grid) -> None:
        """Set the cell."""
        board[self.row, self.col] = self.color

    def _code_lines(self, a: str, g: str) -> list[str]:
        """One assignment."""
        return [f"{g}[{self.row}, {self.col}] = {self.color}"]

    def _bit_cost(self) -> float:
        """The declared cost of a single cell."""
        return self.COST


@dataclass
class _RectTransform(Transform):
    """An edit confined to the rectangle ``row0:row1, col0:col1``."""

    row0: int
    row1: int
    col0: int
    col1: int

    def __post_init__(self) -> None:
        _coerce_ints(self, "row0", "row1", "col0", "col1")
        _check_span(self.row0, self.row1, "row")
        _check_span(self.col0, self.col1, "col")

    @property
    def region(self) -> tuple[slice, slice]:
        """The rectangle as an index into a frame."""
        return slice(self.row0, self.row1), slice(self.col0, self.col1)

    @property
    def region_code(self) -> str:
        """The rectangle as the slices the block writes."""
        return f"{self.row0}:{self.row1}, {self.col0}:{self.col1}"


@dataclass
class FillRect(_RectTransform):
    """Every cell of the rectangle set to one colour."""

    color: int

    COST: ClassVar[float] = 6

    def __post_init__(self) -> None:
        super().__post_init__()
        _coerce_ints(self, "color")
        _check_color(self.color, "color")

    def _apply(self, board: Grid, start: Grid) -> None:
        """Fill the rectangle."""
        board[self.region] = self.color

    def _code_lines(self, a: str, g: str) -> list[str]:
        """One slice assignment."""
        return [f"{g}[{self.region_code}] = {self.color}"]

    def _bit_cost(self) -> float:
        """The declared cost of a fill, whatever its size."""
        return self.COST


@dataclass
class Recolour(_RectTransform):
    """Within the rectangle, every cell of colour ``old`` set to ``new``; other cells keep theirs."""

    old: int
    new: int

    COST: ClassVar[float] = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        _coerce_ints(self, "old", "new")
        _check_color(self.old, "old")
        _check_color(self.new, "new")
        if self.old == self.new:
            raise ValueError(f"a recolour changes a colour; old and new are both {self.old}")

    def _apply(self, board: Grid, start: Grid) -> None:
        """Recolour the matching cells of the board."""
        view = board[self.region]
        view[view == self.old] = self.new

    def _code_lines(self, a: str, g: str) -> list[str]:
        """A masked assignment into the rectangle's view of the board."""
        view = f"{g}[{self.region_code}]"
        return [f"{view}[{view} == {self.old}] = {self.new}"]

    def _bit_cost(self) -> float:
        """The declared cost of a recolour, whatever its size."""
        return self.COST


@dataclass
class BlockCopy(_RectTransform):
    """The rectangle filled from a block whose top-left cell is ``(src_row, src_col)``.

    The block is read from the start frame (``source='a'``) or from the board
    as the edits before this one left it (``source='g'``), mirrored left-right
    if ``flip``, then turned ``rot`` quarter turns counterclockwise: the order
    and senses of ``np.fliplr`` and ``np.rot90``, which is how the block is
    written. With an odd ``rot`` the block is the rectangle's shape
    transposed. A copy leaves the block where it was; with ``clear`` set it
    is a move, leaving that colour behind where the block was read, at no
    extra cost. The block's old place is cleared before its new place is
    written, so a move by less than the block's size comes out right.
    ``distance_cost`` adds that many bits per cell the block travels, rows
    plus columns from where it is read to where it is written. ``written``
    is the colours its line changes cells to, which a plain copy's or move's
    comment names: the edit holds no frame, so they are set from the one it
    is read from (:func:`with_colors`), and are no part of what the edit is.
    """

    src_row: int
    src_col: int
    rot: int = 0
    flip: bool = False
    source: Literal["a", "g"] = "a"
    clear: int | None = None
    distance_cost: float = 0.0
    written: tuple[int, ...] | None = field(default=None, compare=False)

    COST: ClassVar[float] = 10
    TURNED_COST: ClassVar[float] = 18

    def __post_init__(self) -> None:
        super().__post_init__()
        _coerce_ints(self, "src_row", "src_col", "rot")
        if not 0 <= self.rot < 4:
            raise ValueError(f"rot counts quarter turns 0-3, got {self.rot}")
        if not isinstance(self.flip, (bool, np.bool_)):  # pyright: ignore[reportUnnecessaryIsInstance]  # a search may hand a numpy bool, a caller anything
            raise TypeError(f"flip must be a bool, got {self.flip!r}")
        self.flip = bool(self.flip)
        if self.source not in ("a", "g"):
            raise ValueError(f"source is 'a' (the start frame) or 'g' (the board), got {self.source!r}")
        rows, cols = self.block_shape
        _check_span(self.src_row, self.src_row + rows, "source row")
        _check_span(self.src_col, self.src_col + cols, "source col")
        if self.distance_cost < 0:
            raise ValueError(f"distance_cost is bits per cell travelled, 0 or more, got {self.distance_cost!r}")
        if self.clear is not None:
            _coerce_ints(self, "clear")
            _check_color(self.clear, "clear")
            if self.source == "g":
                raise ValueError("a move reads the start frame; clearing behind a copy from the board is not supported")

    @property
    def source_code(self) -> str:
        """The block's old place as the slices the block writes."""
        rows, cols = self.block_shape
        return f"{self.src_row}:{self.src_row + rows}, {self.src_col}:{self.src_col + cols}"

    @property
    def block_shape(self) -> tuple[int, int]:
        """(rows, cols) of the block read, before it is mirrored and turned."""
        rows, cols = self.row1 - self.row0, self.col1 - self.col0
        return (cols, rows) if self.rot % 2 else (rows, cols)

    def priced(self, distance_cost: float) -> BlockCopy:
        """This copy at another price per cell travelled."""
        return dataclasses.replace(self, distance_cost=distance_cost)

    @property
    def distance(self) -> int:
        """How far the block travels: rows plus columns from its old top-left cell to its new one."""
        return abs(self.row0 - self.src_row) + abs(self.col0 - self.src_col)

    @property
    def turned(self) -> bool:
        """Whether the block is mirrored or turned on its way."""
        return self.flip or self.rot != 0

    def _apply(self, board: Grid, start: Grid) -> None:
        """Read the block, clear its old place if this is a move, mirror and turn it, write it into the rectangle."""
        rows, cols = self.block_shape
        reader = start if self.source == "a" else board
        block = reader[self.src_row : self.src_row + rows, self.src_col : self.src_col + cols]
        if self.clear is not None:
            board[self.src_row : self.src_row + rows, self.src_col : self.src_col + cols] = self.clear
        if self.flip:
            block = np.fliplr(block)
        if self.rot:
            block = np.rot90(block, self.rot)
        board[self.region] = block

    def _code_lines(self, a: str, g: str) -> list[str]:
        """One slice assignment, after the old place is cleared for a move; a plain copy or move says which way
        its content went, since the slice reads the other way round."""
        reader = a if self.source == "a" else g
        block = f"{reader}[{self.source_code}]"
        if self.flip:
            block = f"np.fliplr({block})"
        if self.rot:
            block = f"np.rot90({block}, {self.rot})"
        line = f"{g}[{self.region_code}] = {block}"
        if not self.turned:
            verb = "copied" if self.clear is None else "moved"
            line += f"  # {verb} {_direction(self.row0 - self.src_row, self.col0 - self.src_col)}"
            line += _colors_note(self.written)
        if self.clear is None:
            return [line]
        return [f"{g}[{self.source_code}] = {self.clear}  # cleared behind the move below", line]

    def _bit_cost(self) -> float:
        """The declared cost of a copy, and of a mirrored or turned one, and what it travels; a move costs the
        same."""
        return (self.TURNED_COST if self.turned else self.COST) + self.distance_cost * self.distance


#: What a masked move pays over a copy of its box: the line that picks the object's cells out of the box. A
#: first price, to be read against a run's corpus (2026-10-08).
MASK_COST = 4


@dataclass
class MaskedMove(_RectTransform):
    """An object moved: the cells of ``colors`` in the box whose top-left cell is ``(src_row, src_col)``, and only
    those, written into the rectangle, with ``clear`` left behind on the cells they leave.

    The cells are picked in the start frame by colour, so the edit is the object alone when its box holds no
    other cell of those colours; the search proposes it only then (``framediff.objects``). Mirrored and turned as
    ``BlockCopy`` mirrors and turns a block, for an object of one colour. Where a copy of the box would carry
    whatever lies around the object inside it, this carries the object and leaves the rest.
    """

    src_row: int
    src_col: int
    colors: tuple[int, ...] = ()
    rot: int = 0
    flip: bool = False
    clear: int | None = None
    distance_cost: float = 0.0
    written: tuple[int, ...] | None = field(default=None, compare=False)

    COST: ClassVar[float] = BlockCopy.COST + MASK_COST
    TURNED_COST: ClassVar[float] = BlockCopy.TURNED_COST + MASK_COST

    def __post_init__(self) -> None:
        super().__post_init__()
        _coerce_ints(self, "src_row", "src_col", "rot")
        colors = tuple(sorted({_as_int(color, "color") for color in self.colors}))
        if not colors:
            raise ValueError("a masked move names the colours of the object it moves")
        for color in colors:
            _check_color(color, "color")
        self.colors = colors
        if not 0 <= self.rot < 4:
            raise ValueError(f"rot counts quarter turns 0-3, got {self.rot}")
        self.flip = bool(self.flip)
        if self.turned and len(colors) > 1:
            raise ValueError("a mirrored or turned masked move moves an object of one colour")
        rows, cols = self.block_shape
        _check_span(self.src_row, self.src_row + rows, "source row")
        _check_span(self.src_col, self.src_col + cols, "source col")
        if self.clear is not None:
            _coerce_ints(self, "clear")
            _check_color(self.clear, "clear")
        if self.distance_cost < 0:
            raise ValueError(f"distance_cost is bits per cell travelled, 0 or more, got {self.distance_cost!r}")

    @property
    def block_shape(self) -> tuple[int, int]:
        """(rows, cols) of the box read, before it is mirrored and turned."""
        rows, cols = self.row1 - self.row0, self.col1 - self.col0
        return (cols, rows) if self.rot % 2 else (rows, cols)

    @property
    def source(self) -> tuple[slice, slice]:
        """The box the object is read from, as an index into a frame."""
        rows, cols = self.block_shape
        return slice(self.src_row, self.src_row + rows), slice(self.src_col, self.src_col + cols)

    @property
    def source_code(self) -> str:
        """The box as the slices the block reads."""
        rows, cols = self.block_shape
        return f"{self.src_row}:{self.src_row + rows}, {self.src_col}:{self.src_col + cols}"

    @property
    def turned(self) -> bool:
        """Whether the object is mirrored or turned on its way."""
        return self.flip or self.rot != 0

    @property
    def distance(self) -> int:
        """How far the box travels: rows plus columns from its old top-left cell to its new one."""
        return abs(self.row0 - self.src_row) + abs(self.col0 - self.src_col)

    def priced(self, distance_cost: float) -> MaskedMove:
        """This move at another price per cell travelled."""
        return dataclasses.replace(self, distance_cost=distance_cost)

    def _oriented(self, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """``array`` mirrored and turned the way the object is."""
        if self.flip:
            array = np.fliplr(array)
        return np.rot90(array, self.rot) if self.rot else array

    def _apply(self, board: Grid, start: Grid) -> None:
        """Pick the object's cells in the start frame, clear them, and write them, turned, into the rectangle."""
        block = start[self.source]
        mask = np.isin(block, self.colors)
        if self.clear is not None:
            board[self.source][mask] = self.clear
        placed = self._oriented(mask)
        board[self.region][placed] = self._oriented(block)[placed]

    def _code_lines(self, a: str, g: str) -> list[str]:
        """The line picking the object's cells, the clearing behind it, and the move, which says which way the
        object went unless it turned. The pick names the object by its colours alone: its box is in the slice."""
        box = f"{a}[{self.source_code}]"
        if len(self.colors) == 1:
            pick, what = f"{box} == {self.colors[0]}", f"the colour-{self.colors[0]} shape"
        else:
            pick, what = (
                f"np.isin({box}, {list(self.colors)})",
                f"the shape of colours {', '.join(map(str, self.colors))}",
            )
        lines = [f"mask = {pick}  # {what}"]
        if self.clear is not None:
            lines.append(f"{g}[{self.source_code}][mask] = {self.clear}  # cleared behind the move below")
        placed = "mask"
        if self.flip:
            placed = f"np.fliplr({placed})"
        if self.rot:
            placed = f"np.rot90({placed}, {self.rot})"
        value = str(self.colors[0]) if len(self.colors) == 1 else f"{box}[mask]"
        line = f"{g}[{self.region_code}][{placed}] = {value}"
        if not self.turned:
            line += f"  # moved {_direction(self.row0 - self.src_row, self.col0 - self.src_col)}"
            line += _colors_note(tuple(sorted(self.colors)) if self.written is None else self.written)
        return [*lines, line]

    def _bit_cost(self) -> float:
        """The declared cost of a masked move, and of a mirrored or turned one, and what it travels."""
        return (self.TURNED_COST if self.turned else self.COST) + self.distance_cost * self.distance


@dataclass
class Line(Transform):
    """A one-cell-wide straight line of one colour from ``start`` to ``end``,
    both (row, col) and both drawn.

    Drawn here and in the block by ``bresenham_line``, so ties are broken
    the way the games break them, and a line and its reverse can differ by a
    cell on some slopes.
    """

    start: tuple[int, int]
    end: tuple[int, int]
    color: int

    COST: ClassVar[float] = 12

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            point = tuple(getattr(self, name))
            if len(point) != 2:
                raise ValueError(f"{name} is a (row, col) pair, got {point!r}")
            row, col = (_as_int(value, name) for value in point)
            _check_span(row, row + 1, f"{name} row")
            _check_span(col, col + 1, f"{name} col")
            setattr(self, name, (row, col))
        _coerce_ints(self, "color")
        _check_color(self.color, "color")

    def cells(self) -> list[tuple[int, int]]:
        """The line's cells as (row, col), in row-major order."""
        scratch = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        bresenham_line(scratch, self.start, self.end, 1)
        return [(int(row), int(col)) for row, col in np.argwhere(scratch)]

    def _notes(self) -> list[str]:
        """Where ``bresenham_line`` comes from."""
        return [LINE_NOTE]

    def _apply(self, board: Grid, start: Grid) -> None:
        """Draw the line."""
        bresenham_line(board, self.start, self.end, self.color)

    def _code_lines(self, a: str, g: str) -> list[str]:
        """One call to ``bresenham_line``."""
        (r0, c0), (r1, c1) = self.start, self.end
        return [f"bresenham_line({g}, ({r0}, {c0}), ({r1}, {c1}), {self.color})"]

    def _bit_cost(self) -> float:
        """The declared cost of a line, whatever its length."""
        return self.COST


@dataclass
class AsciiGrid(Transform):
    """A rectangle written cell by cell, its top-left cell at ``(row0, col0)``.

    ``rows`` are the rectangle's lines in ascii-grid notation, one character
    per cell. The block writes them as a picture, one row per line, read by
    ``from_ascii_grid``.
    """

    row0: int
    col0: int
    rows: tuple[str, ...]

    COST: ClassVar[float] = 8
    CELL_COST: ClassVar[float] = 1

    def __post_init__(self) -> None:
        _coerce_ints(self, "row0", "col0")
        rows = tuple(self.rows)
        for i, row in enumerate(rows):
            if not isinstance(row, str) or row != row.strip():  # pyright: ignore[reportUnnecessaryIsInstance]  # callers pass whatever they split
                raise ValueError(f"row {i + 1} of the grid must be a string without surrounding whitespace")
        self.rows = rows
        height, width = self.values.shape
        _check_span(self.row0, self.row0 + height, "row")
        _check_span(self.col0, self.col0 + width, "col")

    @classmethod
    def from_array(cls, row0: int, col0: int, grid: npt.ArrayLike) -> AsciiGrid:
        """The paste that writes ``grid`` with its top-left cell at ``(row0, col0)``."""
        return cls(row0, col0, tuple(to_ascii_grid(grid).splitlines()))

    @property
    def values(self) -> Grid:
        """The rectangle's contents as an array."""
        return from_ascii_grid("\n".join(self.rows))

    def _apply(self, board: Grid, start: Grid) -> None:
        """Write the rectangle."""
        height, width = self.values.shape
        board[self.row0 : self.row0 + height, self.col0 : self.col0 + width] = self.values

    def _code_lines(self, a: str, g: str) -> list[str]:
        """A slice assignment from ``from_ascii_grid``, the rows at the left margin."""
        height, width = self.values.shape
        region = f"{self.row0}:{self.row0 + height}, {self.col0}:{self.col0 + width}"
        return [f'{g}[{region}] = from_ascii_grid("""', *self.rows, '""")']

    def _bit_cost(self) -> float:
        """The declared cost of a paste plus one per cell written."""
        height, width = self.values.shape
        return self.COST + self.CELL_COST * height * width


@dataclass
class Composite(Transform):
    """Transforms applied in order.

    Each reads the board as the ones before it left it; all of them read the
    same start frame. An empty composite leaves the frame as it is.
    """

    transforms: tuple[Transform, ...] = ()

    def __post_init__(self) -> None:
        transforms = tuple(self.transforms)
        for i, part in enumerate(transforms):
            if not isinstance(part, Transform):  # pyright: ignore[reportUnnecessaryIsInstance]  # callers pass lists of anything
                raise TypeError(f"part {i} of a composite must be a Transform, got {part!r}")
        self.transforms = transforms

    def priced(self, distance_cost: float) -> Composite:
        """The parts, each priced at ``distance_cost``."""
        return Composite(tuple(part.priced(distance_cost) for part in self.transforms))

    def _notes(self) -> list[str]:
        """The parts' notes, in the order the parts first need them."""
        return [note for part in self.transforms for note in part._notes()]

    def _apply(self, board: Grid, start: Grid) -> None:
        """Apply each part in order."""
        for part in self.transforms:
            part._apply(board, start)

    def _code_lines(self, a: str, g: str) -> list[str]:
        """Each part's lines, in order."""
        return [line for part in self.transforms for line in part._code_lines(a, g)]

    def _bit_cost(self) -> float:
        """The parts' costs, summed."""
        return sum((part.bit_cost() for part in self.transforms), 0.0)


def _colors_note(colors: tuple[int, ...] | None) -> str:
    """The colours a copy or move changes cells to, as its comment ends with them; nothing where they are not
    known or there are none."""
    return "" if not colors else f" (colors: {', '.join(map(str, colors))})"


#: A copy's or move's colours as :func:`_colors_note` writes them, for text that drops them.
COLORS_NOTE = re.compile(r" \(colors: [0-9, ]+\)")


def with_colors(transform: Transform, start: npt.ArrayLike) -> Transform:
    """``transform`` with each copy and move in it naming the colours its line changes cells to (``written``),
    replayed from ``start``: a cell its line writes counts where it held another colour just before that line,
    after the edits before it and, for a move, after its old place is cleared, which is a line of its own. A
    wildcard is no colour. The edit is otherwise the same, and equal to ``transform``."""
    frame = np.array(start, dtype=np.int8)
    frame.flags.writeable = False
    board = frame.copy()
    parts = transform.transforms if isinstance(transform, Composite) else (transform,)
    named: list[Transform] = []
    for part in parts:
        if isinstance(part, (BlockCopy, MaskedMove)):
            before = board.copy()
            if part.clear is not None:
                if isinstance(part, BlockCopy):
                    rows, cols = part.block_shape
                    before[part.src_row : part.src_row + rows, part.src_col : part.src_col + cols] = part.clear
                else:
                    before[part.source][np.isin(frame[part.source], part.colors)] = part.clear
            after = board.copy()
            part._apply(after, frame)  # pyright: ignore[reportPrivateUsage]
            changed = (after != before) & (after != WILDCARD)
            part = dataclasses.replace(part, written=tuple(sorted(set(after[changed].tolist()))))
        part._apply(board, frame)  # pyright: ignore[reportPrivateUsage]
        named.append(part)
    return Composite(tuple(named)) if isinstance(transform, Composite) else named[0]


def _probe_frames(wildcards: bool = False) -> list[Grid]:
    """The frames every block is self-tested on.

    A seeded random frame catches geometry: a wrong shift, turn or mirror
    changes the result almost everywhere. A banded frame puts every colour in
    every 4x4 window, so a recolour of a small rectangle finds its colour.
    With ``wildcards``, a third is the random frame with wildcards scattered
    over it, so the resolving line is exercised.
    """
    rows, cols = np.indices((GRID_SIZE, GRID_SIZE))
    banded = ((4 * rows + cols) % NUM_COLORS).astype(np.int8)
    rng = np.random.default_rng(0)
    scattered = rng.integers(0, NUM_COLORS, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    probes = [scattered, banded]
    if wildcards:
        holed = scattered.copy()
        holed[rng.random((GRID_SIZE, GRID_SIZE)) < 0.1] = WILDCARD
        probes.append(holed)
    return probes


def _check_names(a: str, b: str, g: str) -> None:
    """``a`` and ``b`` readable expressions, ``g`` an assignable name, all three different."""
    if not g.isidentifier() or keyword.iskeyword(g):
        raise ValueError(f"g={g!r} must be a plain name; the block assigns it")
    for role, name in (("a", a), ("b", b)):
        try:
            node = ast.parse(name, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"{role}={name!r} is not an expression") from error
        while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
            node = node.func if isinstance(node, ast.Call) else node.value
        if not isinstance(node, ast.Name):
            raise ValueError(f"{role}={name!r} must be a name, or attributes, subscripts and calls on one")
    if len({a, b, g}) != 3:
        raise ValueError(f"a, b and g must be three different names, got {a!r}, {b!r}, {g!r}")


def _as_int(value: object, what: str) -> int:
    """``value`` as a plain int, refusing bools and non-integers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return int(cast("int | np.integer[Any]", value))


def _coerce_ints(instance: object, *names: str) -> None:
    """Replace the named fields of a dataclass by plain ints, so emitted code spells numbers as numbers."""
    for name in names:
        setattr(instance, name, _as_int(getattr(instance, name), name))


def _check_span(start: int, stop: int, what: str) -> None:
    """A non-empty half-open span inside the frame."""
    if not 0 <= start < stop <= GRID_SIZE:
        raise ValueError(f"{what} span {start}:{stop} must be non-empty and inside 0:{GRID_SIZE}")


def _check_color(value: int, what: str) -> None:
    """A colour index."""
    if not 0 <= value < NUM_COLORS:
        raise ValueError(f"{what} must be a colour 0-{NUM_COLORS - 1}, got {value}")


def _direction(rows: int, cols: int) -> str:
    """An offset in the words a block annotates it with: ``down 5, left 3``, ``right 6``, ``in place``."""
    parts = [f"{'down' if rows > 0 else 'up'} {abs(rows)}"] if rows else []
    parts += [f"{'right' if cols > 0 else 'left'} {abs(cols)}"] if cols else []
    return ", ".join(parts) or "in place"
