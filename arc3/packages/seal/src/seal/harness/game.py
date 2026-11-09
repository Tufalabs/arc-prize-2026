"""The game as your code sees it: ``history``, its frames and moves, and the
functions that read them and act.

Call ``help()`` for an overview, and ``help(obj)`` on any name for details.
"""

# In this module, # comments are developer comments that the LLM will never see.
# Docstrings are the LLM's API reference, this module's own included.
# LLM will NOT have access to the source code.
#
# The seal's copy of the whale's surface: the same classes, the world-model
# names (``Unknown``, ``Turn.state``, the ``world_model.py`` namespace
# builder), plus the segmentation the seal offers on any board and the
# ``submit_action`` the model acts through. History is fed from the agent's
# state payload rather than from a ``Step``.
from __future__ import annotations

import base64
import copy
import inspect
import os
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ItemsView, Iterable, Self, ValuesView, cast, overload

import numpy as np
import numpy.typing as npt

from framediff import drawing as _framediff_drawing
from framediff import grid as _framediff_grid
from seal import module_names
from seal.frame_delta import kept_animation as _kept_animation
from seal.harness import PRESENTED as _PRESENTED
from seal.harness import present as _present
from seal.palette import ARC_COLOR_NAMES, ARC_HEX_CHARS, ASCII_SYMBOLS, AsciiLayout
from seal.segmentation import (
    segment_layer as _segment_layer_untyped,  # pyright: ignore[reportUnknownVariableType]  # typed below
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from contextlib import AbstractContextManager

#: The name this module presents to the model. Declared before any
#: definition below, because class creation reads the module's
#: ``__name__``: every class and function here therefore reports
#: ``runtime.game`` as its ``__module__``, which is what a repr, a
#: traceback, ``help()`` and a pickle all show. ``seal.harness.install()``
#: registers the name so those references resolve.
__name__ = "runtime.game"

_present(__spec__.name, __name__)  # before any definition below; see seal.harness

__all__ = [
    "help",
    "ascii_grid",
    "from_ascii_grid",
    "color_names",
    "ColorCounts",
    "DiffSparse",
    "Frame",
    "Turn",
    "History",
    "Action",
    "Node",
    "Segmentation",
    "segment",
    "Unknown",
    "WorldModelRequired",
    "WorldModelStop",
]

#: Colour index to name, the seal's own names (``seal.palette``).
color_names: dict[int, str] = dict(ARC_COLOR_NAMES)


class Locked:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Base class that supports locking instances to prevent modification.
    Locked instances raise AttributeError on setattr.
    Use copy.deepcopy() to get an unlocked, editable copy.
    """

    # _locked is managed via object.__setattr__ to avoid recursion.
    # Class-level default keeps ``getattr(self, '_locked', False)`` in
    # ``__setattr__`` a fast class-dict lookup — without it, unset instances
    # would fall through to our ``__getattr__`` + ``dir()`` every assignment.
    _locked = False
    #: How to use an instance that has no public attributes to list, said
    #: where a missing attribute would otherwise list none.
    _usage: str | None = None

    # Subclasses can list attribute names that are allowed to be set even when locked (e.g. caches)
    _mutable_when_locked: tuple[str, ...] = ()

    def __setattr__(self, name: str, value: object) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        if getattr(self, "_locked", False) and name not in self._mutable_when_locked:
            raise AttributeError("Game state is read-only. Use copy.deepcopy() to get an editable copy.")
        object.__setattr__(self, name, value)

    def _lock(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Freeze the instance: every later attribute write raises."""
        object.__setattr__(self, "_locked", True)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            object.__setattr__(result, k, copy.deepcopy(v, memo))
        # Unlock the copy
        object.__setattr__(result, "_locked", False)
        return result

    # Hide Locked internals from help() — only expose subclass members
    _hidden_from_dir = frozenset(
        {"_lock", "_mutable_when_locked", "_locked", "__deepcopy__", "__setattr__", "__dict__", "__weakref__"}
    )

    def __dir__(self) -> list[str]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return [name for name in super().__dir__() if name not in self._hidden_from_dir]

    def __getattr__(self, name: str) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        # Only fires when normal lookup fails — surface the meaningful
        # attributes so the LLM doesn't have to guess.
        public = [n for n in dir(self) if not n.startswith("_")]
        offer = f"Available: {', '.join(public)}." if public or self._usage is None else self._usage
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}. {offer}")


import builtins as _builtins  # noqa: E402

_help_sentinel = object()


def help(obj: object = _help_sentinel) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Show documentation for game objects. Shows docstring and properties with their descriptions."""
    if obj is _help_sentinel:
        # No argument: show overview
        print("Available objects: history, available_actions, submit_action,")
        if _has_segmentation:
            print("                   ascii_grid, from_ascii_grid, segment, color_names")
            print("Available types: Frame, Turn, History, Action, DiffSparse, ColorCounts, Segmentation, Node")
        else:
            print("                   ascii_grid, from_ascii_grid, color_names")
            print("Available types: Frame, Turn, History, Action, DiffSparse, ColorCounts")
        print("\nCall help(obj) on any of these for details.")
        return

    # A module of the surface, the package itself and the private store
    # included: the overview. Pydoc would print each module's docstring and
    # every class with its private bases, which the branches below mask.
    if isinstance(obj, types.ModuleType) and obj.__name__.split(".")[0] == _PRESENTED:
        help()
        return

    cls = obj if isinstance(obj, type) else type(obj)
    module = getattr(cls, "__module__", "")

    # For other non-game objects, fall back to builtin help. Matched on the
    # tail rather than the full name: this module is presented to the model as
    # `runtime.game`, so its own classes carry that `__module__` and a check
    # against the real name sends every one of them to pydoc — which prints
    # the MRO, private base classes and parameters the surface deliberately
    # masks, on text `design.md` counts as prompt text.
    if not module.endswith("runtime.game") and cls.__name__ not in __all__:
        _builtins.help(obj)
        return

    # For functions, just show the docstring
    if callable(obj) and not isinstance(obj, type):
        print(f"{obj.__name__}()")
        if obj.__doc__:
            print(obj.__doc__)
        return

    print(cls.__name__)
    if cls.__doc__:
        print(cls.__doc__)

    # Show properties (skip if docstring already has a Properties section)
    if not cls.__doc__ or "Properties:" not in cls.__doc__:
        props: list[tuple[str, str]] = []
        for name in sorted(dir(cls)):
            if name.startswith("_"):
                continue
            attr = getattr(cls, name, None)
            if isinstance(attr, property) and attr.fget and attr.fget.__doc__:
                props.append((name, attr.fget.__doc__.strip().split("\n")[0]))
        if props:
            print("\nProperties:")
            for name, doc in props:
                print(f"  .{name} — {doc}")

    # For non-type objects, show current repr
    if not isinstance(obj, type):
        print(f"\nCurrent value:\n{repr(obj)}")


# ignore: `arr` is deliberately left unannotated. ascii_grid is advertised to the
# model ("Call help(obj) on any of these for details") and help() falls through to
# builtins.help for plain functions, which renders the signature — so an annotation
# here would change a string the model reads. The first statement pins the type.
#: How ``ascii_grid`` lays a board out. Set once per kernel from the agent's
#: configuration (``_configure_ascii_grid``, in ``initial_code``); not a
#: parameter the model is offered, so a sweep over them changes what every
#: render looks like without the model choosing.
_ascii_row_numbers = True
_ascii_column_group: int | None = None
#: The characters a board prints in, by colour index: the alphabet the agent
#: chose (``SealAgent.ascii_symbols``), which framediff's code blocks write and
#: ``from_ascii_grid`` reads in too.
_ascii_chars = ARC_HEX_CHARS


def _configure_ascii_grid(  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    *, row_numbers: bool, column_group: int | None, symbols: str = "hex"
) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """The agent's layout and alphabet for every board this process prints or
    reads: ``ascii_grid``, ``from_ascii_grid`` and framediff's blocks. What
    ``help()`` says of the two functions follows the alphabet."""
    global _ascii_row_numbers, _ascii_column_group, _ascii_chars
    if column_group is not None and column_group < 1:
        raise ValueError(f"ascii column group must be at least 1; got {column_group}")
    if symbols not in ASCII_SYMBOLS:
        raise ValueError(f"ascii symbols must be one of {sorted(ASCII_SYMBOLS)}; got {symbols!r}")
    _ascii_row_numbers = row_numbers
    _ascii_column_group = column_group
    _ascii_chars = ASCII_SYMBOLS[symbols]
    _framediff_grid.set_alphabet(_ascii_chars)
    ascii_grid.__doc__ = _ASCII_GRID_DOCS[symbols]
    from_ascii_grid.__doc__ = _FROM_ASCII_GRID_DOCS[symbols]


def _ascii_layout_scope(layout: AsciiLayout) -> AbstractContextManager[None]:  # pyright: ignore[reportUnusedFunction]  # called from seal.py_model
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """:func:`_configure_ascii_grid` with ``layout`` for the span of a
    ``with``, and the process's own settings back after: for a process that
    runs more than one agent's model code, the agent's and the search's."""
    import contextlib

    @contextlib.contextmanager
    def _ctx() -> Generator[None, None, None]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        global _ascii_row_numbers, _ascii_column_group, _ascii_chars
        saved = (_ascii_row_numbers, _ascii_column_group, _ascii_chars, _framediff_grid.alphabet())
        saved_docs = (ascii_grid.__doc__, from_ascii_grid.__doc__)
        _configure_ascii_grid(row_numbers=layout.row_numbers, column_group=layout.column_group, symbols=layout.symbols)
        try:
            yield
        finally:
            _ascii_row_numbers, _ascii_column_group, _ascii_chars, alphabet = saved
            _framediff_grid.set_alphabet(alphabet)
            ascii_grid.__doc__, from_ascii_grid.__doc__ = saved_docs

    return _ctx()


def ascii_grid(arr) -> str:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Render a 2D integer array as a readable ascii grid.
    Accepts ndarray, list of lists, or anything convertible via np.array().
    Colors 0-9 display as digits, 10-15 as a-f. This is the ONLY place a
    colour stops being the integer it is: everything else -- `.grid`,
    `segment()`, a node's `color` -- carries the index itself.
    Each row may be prefixed with its index in the array you passed and a
    tab, so on a crop add the slice's start; ascii_grid(grid[::2, ::2])
    gives a quick overview.
    """
    arr = np.asarray(arr, dtype=np.int8)  # pyright: ignore[reportUnknownArgumentType]
    if arr.ndim != 2:
        raise ValueError(f"ascii_grid expects a 2D array, got {arr.ndim}D.")
    if np.any((arr < 0) | (arr > 15)):
        raise ValueError("ascii_grid supports colour indices 0-15.")
    if arr.size == 0:
        return "(empty grid)"
    lines: list[str] = []
    for index, row in enumerate(arr):
        cells = "".join(_ascii_chars[int(value)] for value in row)
        if _ascii_column_group is not None:
            group = _ascii_column_group
            cells = "\t".join(cells[start : start + group] for start in range(0, len(cells), group))
        # The file tools' shape -- number, tab, content -- so a numbered row
        # reads the way every other numbered line the model sees does.
        lines.append(f"{index}\t{cells}" if _ascii_row_numbers else cells)
    return "\n".join(lines)


# The parameters are unannotated for the same reason as ascii_grid's: help()
# renders the signature, and an annotation would change what the model reads.
# Both functions below wrap framediff's, which the harness's own code blocks
# are checked against, and re-raise a refusal from their own frame so that no
# framediff frame reaches a traceback the model reads.
def from_ascii_grid(text):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Read an ascii grid back into a 2-D NumPy array of color indices, dtype int8.
    The inverse of ascii_grid: one character per cell, digits for 0-9 and
    letters a-f for 10-15. What ascii_grid printed reads back as it is: the
    row numbers and tabs it adds are recognised and skipped. So does a board
    or a sprite you write as a picture, e.g.
        sprite = from_ascii_grid(\"\"\"
        0990
        9449
        0990
        \"\"\")
    gives a 3x4 array. Blank lines, spaces and indentation are ignored; every
    row must hold the same number of cells.
    """
    given = cast(object, text)
    if not isinstance(given, str):
        raise TypeError(f"from_ascii_grid expects the grid as a string, got {type(given).__name__}.")
    try:
        return _framediff_grid.from_ascii_grid(given)
    except ValueError as error:
        raise ValueError(str(error)) from None


#: What ``help()`` says of the two functions, by alphabet. Hex is each
#: function's own docstring; letters say the same of the letter alphabet.
_ASCII_GRID_DOCS: dict[str, str | None] = {
    "hex": ascii_grid.__doc__,
    "letters": """Render a 2D integer array as a readable ascii grid.
    Accepts ndarray, list of lists, or anything convertible via np.array().
    Each color displays as its letter from the color legend, e.g. W for 0
    (white) and b for 9 (blue). This is the ONLY place a colour stops being
    the integer it is: everything else -- `.grid`, `segment()`, a node's
    `color` -- carries the index itself.
    Each row may be prefixed with its index in the array you passed and a
    tab, so on a crop add the slice's start; ascii_grid(grid[::2, ::2])
    gives a quick overview.
    """,
}
_FROM_ASCII_GRID_DOCS: dict[str, str | None] = {
    "hex": from_ascii_grid.__doc__,
    "letters": """Read an ascii grid back into a 2-D NumPy array of color indices, dtype int8.
    The inverse of ascii_grid: one character per cell, each color's letter
    from the color legend. What ascii_grid printed reads back as it is: the
    row numbers and tabs it adds are recognised and skipped. So does a board
    or a sprite you write as a picture, e.g.
        sprite = from_ascii_grid(\"\"\"
        WbbW
        bccb
        WbbW
        \"\"\")
    gives a 3x4 array. Blank lines, spaces and indentation are ignored; every
    row must hold the same number of cells.
    """,
}


def bresenham_line(grid, start, end, color):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Draw a straight line of one color on a 2-D NumPy array, in place, and return None.

    bresenham_line(grid, start, end, color)
    - grid: a writable 2-D NumPy array. A frame's .grid is read-only; draw on
      a copy, e.g. g = history[-1].frame.grid.copy().
    - start, end: (row, col) cells inside grid. Both are drawn.
    - color: a color index 0-15. Every cell of the line is set to it; no
      other cell changes.

    The line is one cell wide and steps diagonally where it needs to. It
    covers max(|r1 - r0|, |c1 - c0|) + 1 cells: one per column if it spans
    more columns than rows, one per row otherwise. Horizontal, vertical and
    45-degree lines are the plain straight runs.

    Which cells a sloped line takes, exactly: with (r0, c0) = start,
    (r1, c1) = end, dr = |r1 - r0|, dc = |c1 - c0| and err = dc - dr, draw
    (r0, c0); until (r0, c0) == (r1, c1), let e2 = 2 * err, then if e2 > -dr
    subtract dr from err and step c0 one toward c1, and if e2 < dc add dc to
    err and step r0 one toward r1 (both can happen in one step), and draw
    (r0, c0). This is the integer Bresenham algorithm ARC games use to draw
    lines between cells.

    Direction matters where a line passes exactly between two cells:
    drawn on np.zeros((5, 9), dtype=np.int8) with color 3,
    bresenham_line(g, (0, 0), (4, 8), 3) gives   bresenham_line(g, (4, 8), (0, 0), 3) gives
        330000000                                     300000000
        003300000                                     033000000
        000033000                                     000330000
        000000330                                     000003300
        000000003                                     000000033
    so draw from the end the game draws from.
    """
    board = _drawable(cast(object, grid))
    first = _cell("start", cast(object, start), board.shape)
    last = _cell("end", cast(object, end), board.shape)
    given = cast(object, color)
    if not _is_int(given) or not 0 <= cast(int, given) <= 15:
        raise ValueError(f"bresenham_line: color must be a color index 0-15, got {given!r}.")
    _framediff_drawing.bresenham_line(board, first, last, int(cast(int, given)))


def _drawable(grid: object) -> npt.NDArray[Any]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``grid`` as the writable 2-D array bresenham_line draws on, or the refusal saying what it is instead."""
    if not isinstance(grid, np.ndarray):
        raise TypeError("bresenham_line draws on a 2-D NumPy array, e.g. g = history[-1].frame.grid.copy().")
    board = cast("npt.NDArray[Any]", grid)
    if board.ndim != 2:
        raise TypeError("bresenham_line draws on a 2-D NumPy array, e.g. g = history[-1].frame.grid.copy().")
    if not board.flags.writeable:
        raise ValueError("bresenham_line: this grid is read-only (a frame's .grid is); draw on a copy of it.")
    return board


def _cell(name: str, point: object, shape: tuple[int, ...]) -> tuple[int, int]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``point`` as a (row, col) cell inside a board of ``shape``, or the refusal naming ``name``."""
    items = list(cast("tuple[object, ...]", point)) if isinstance(point, (tuple, list)) else []
    if len(items) != 2 or not all(_is_int(value) for value in items):
        raise TypeError(f"bresenham_line: {name} must be a (row, col) pair of integers, got {point!r}.")
    row, col = (int(cast(int, value)) for value in items)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError(f"bresenham_line: {name} {(row, col)} is outside the {shape[0]}x{shape[1]} grid.")
    return row, col


def _is_int(value: object) -> bool:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """An integer the model may pass as a coordinate or color: Python's or NumPy's, not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


# Bare `dict`, NOT `dict[int, int]`: subclassing a parameterised generic
# sets `__orig_bases__` on the class, which the model sees in `dir()` --
# and it does call `dir()` on these. Caught by a replay mismatch.
class ColorCounts(dict):  # type: ignore[type-arg]  # see above
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Dict mapping color (int) to pixel count.
    Access count for a specific color: color_counts[3]
    """

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        if not self:
            return "ColorCounts(empty)"
        lines = [f"ColorCounts({len(self)} colors)"]
        counts = cast("ItemsView[int, int]", self.items())
        for c, count in sorted(counts, key=lambda kv: -kv[1]):
            name = color_names.get(c, "?")
            lines.append(f"  {c} {name}: {count}")
        return "\n".join(lines)


# Bare `dict` for the same reason as ColorCounts above: a parameterised
# base would add `__orig_bases__` to what `dir()` shows the model.
class DiffSparse(dict):  # type: ignore[type-arg]  # see above
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Sparse representation of changes between two frames.
    Keys are (old_color, new_color) tuples. Values are lists of (row, col) coordinates.
    Access coordinates for a specific change type: diff_sparse[(0, 3)]
    """

    # Subclasses dict so all dict operations work: .keys(), .items(), iteration, indexing, len()

    @property
    def total_pixels(self) -> int:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Total number of pixels that changed."""
        values = cast("ValuesView[list[tuple[int, int]]]", self.values())
        return sum(len(coords) for coords in values)

    @property
    def change_counts(self) -> dict[tuple[int, int], int]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Dict mapping each (old_color, new_color) pair to the number of pixels with that change."""
        # Not a ColorCounts — keys are tuples, not single colors
        changes = cast("ItemsView[tuple[int, int], list[tuple[int, int]]]", self.items())
        return {key: len(coords) for key, coords in changes}

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        if not self:
            return "DiffSparse(no changes)"
        total = self.total_pixels
        lines = [f"DiffSparse({len(self)} change types, {total} pixels total)"]
        changes = cast("ItemsView[tuple[int, int], list[tuple[int, int]]]", self.items())
        for (old, new), coords in sorted(changes, key=lambda kv: -len(kv[1])):
            old_name = color_names.get(old, "?")
            new_name = color_names.get(new, "?")
            lines.append(f"  ({old} {old_name} → {new} {new_name}): {len(coords)} pixels")
        first = cast("tuple[int, int]", next(iter(cast("Iterable[Any]", self))))
        lines.append(f"# explore: diff_sparse[{first}] for coordinates, .change_counts")
        return "\n".join(lines)


class Frame(Locked):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """A single frame of gameplay. Contains the grid state and optional diff to a previous frame.

    Properties:
    - grid: np.ndarray (64x64), dtype int8, values 0-15
    - color_counts: ColorCounts, each color's pixel count
    - diff: np.ndarray (64x64), values -15 to 15 (None if no previous frame)
    - diff_sparse: DiffSparse dict of {(old, new): [(row, col), ...]} (None if no previous frame)
    Use ascii_grid(frame.grid) to visualize (consider slicing frame.grid).
    Call segment(frame.grid) for its connected components.
    """

    _mutable_when_locked = ("_cached_diff_sparse",)

    # When a frame was observed is its turn's (Turn.step, Turn.level): a frame
    # is a board. The two still answer here, for code already written against
    # them, but are left out of what dir() and the missing-attribute hint offer.
    _hidden_from_dir = Locked._hidden_from_dir | {"step", "level"}

    def __init__(
        self,
        grid: npt.NDArray[Any],
        previous: npt.NDArray[Any] | None = None,
        diff_context: str | None = None,
        frame_type: str = "visible",
        step: int = 0,
        level: int = 0,
    ):
        # grid and previous are raw ndarrays, stored directly
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        self.step = step
        self.level = level
        self._grid = grid.astype(np.int8)
        self._grid.flags.writeable = False  # prevent in-place mutation
        self._previous = previous.astype(np.int8) if previous is not None else None
        if self._previous is not None:
            self._previous.flags.writeable = False
        self._diff_context = diff_context  # e.g. 'previous turn', 'previous animation frame'
        self._frame_type = frame_type  # 'visible' or 'animation'
        self._cached_diff_sparse: DiffSparse | None = None  # computed on first access

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        result = super().__deepcopy__(memo)
        # Make arrays writeable in the copy
        result._grid.flags.writeable = True
        if result._previous is not None:
            result._previous.flags.writeable = True
        result._cached_diff_sparse = None  # clear cache
        return result

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        n_colors = len(np.unique(self._grid))
        has_diff = self._previous is not None
        parts = [f"Frame({self._frame_type}, {n_colors} colors"]
        if has_diff:
            # cast: diff_sparse is None exactly when _previous is, i.e. never under has_diff.
            diff_sparse = cast(DiffSparse, self.diff_sparse)
            ctx = f" vs {self._diff_context}" if self._diff_context else ""
            parts.append(f", {diff_sparse.total_pixels} pixels changed{ctx}")
        parts.append(")")
        hints = ["ascii_grid(.grid)", ".color_counts"]
        if has_diff:
            hints.insert(1, ".diff_sparse")
        return "".join(parts) + "\n# explore: " + ", ".join(hints)

    @property
    def grid(self) -> npt.NDArray[np.int8]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """The frame's 64x64 grid state, dtype int8, values 0-15."""
        return self._grid

    @property
    def color_counts(self) -> ColorCounts:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Dict mapping each color present in the grid to its pixel count."""
        colors, counts = np.unique(self._grid, return_counts=True)
        return ColorCounts(zip((int(c) for c in colors), (int(n) for n in counts)))

    @property
    def diff(self) -> npt.NDArray[np.int8] | None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Pixel-wise difference to the previous frame (64x64), values -15 to 15.
        None if there is no previous frame.
        Prefer diff_sparse for analyzing changes.
        """
        if self._previous is None:
            return None
        return self._grid - self._previous

    @property
    def diff_sparse(self) -> "DiffSparse | None":
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Sparse representation of changes from the previous frame.
        Keys are (old_color, new_color) tuples. Values are lists of (row, col) coordinates.
        None if there is no previous frame.
        """
        if self._previous is None:
            return None
        if self._cached_diff_sparse is not None:
            return self._cached_diff_sparse
        mask = self._grid != self._previous
        rows, cols = np.where(mask)
        result = DiffSparse()
        for row, col in zip(rows, cols):
            old_val = int(self._previous[row, col])
            new_val = int(self._grid[row, col])
            key = (old_val, new_val)
            if key not in result:
                result[key] = []
            result[key].append((int(row), int(col)))
        self._cached_diff_sparse = result
        return result


@dataclass(frozen=True)
class Action:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """One move, in every place a move appears: ``history[i].action``, the
    ``move`` do_move is handed, what ``submit_action`` takes, and its
    result's ``executed_actions``.

    ``Action('UP')`` is a move; ``Action('MOUSE', row=r, col=c)`` is a
    click. A list of them steps a world model and plays the same moves:
    ``for move in plan: world_model.do_move(state, move)``, then
    ``submit_action(plan)``.

    Properties:
    - name: str — the action name, as in ``available_actions`` (e.g. 'LEFT', 'MOUSE').
    - row, col: int | None — 0-indexed grid coordinates (0..63) for
        MOUSE. ``row`` counts from the top, ``col``
        counts from the left, so the clicked pixel is
        ``frame.grid[row, col]``. Both None for simple actions
        (UP/DOWN/LEFT/RIGHT/...).

    Two Actions with the same name and coordinates are equal, so they
    work as dict keys and set members.
    """

    name: str
    row: int | None = None
    col: int | None = None

    def __post_init__(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Spell the name as the game does, and hold a click to its coordinates."""
        name = str(self.name).strip().upper()
        object.__setattr__(self, "name", name)
        if name == "MOUSE":
            if self.row is None or self.col is None:
                raise ValueError("Action('MOUSE') needs a row and a col: Action('MOUSE', row=r, col=c).")
            object.__setattr__(self, "row", int(self.row))
            object.__setattr__(self, "col", int(self.col))
        elif self.row is not None or self.col is not None:
            raise ValueError(f"Only MOUSE takes a row and a col; Action({name!r}) does not.")

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        if self.row is not None:
            return f"Action({self.name!r}, row={self.row}, col={self.col})"
        return f"Action({self.name!r})"


class Turn(Locked):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """A single turn of gameplay: the action taken, the resulting outcome, and any animation frames.

    Properties:
    - action: the Action taken this turn (None for the level's initial
        state). Use `action.name` for the action name; for MOUSE,
        `action.row` and `action.col` are 0-indexed grid coordinates
        (top-left = (0, 0)).
    - outcome: 'playing', 'won', or 'lost'
    - step: int, the game's action counter when this turn settled
    - level: int, the level it was played on (the same on every turn of a history)
    - frame: Frame showing the board this turn settled on
    - animation_frames: list of Frames showing the animation sequence (may be empty)
    """

    _mutable_when_locked = ("state", "predicted_frame")

    #: What a frame has and a turn does not: asked of a turn, the error points
    #: at the turn's frame.
    _FRAME_ATTRIBUTES = frozenset({"grid", "color_counts", "diff", "diff_sparse"})

    def __init__(
        self,
        action: Action | None,
        outcome: str,
        frame: Frame,
        animation_frames: list[Frame] | None = None,
        state: dict[str, Any] | None = None,
        predicted_frame: npt.NDArray[np.int8] | None = None,
        plan: str | None = None,
    ):
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        self.action = action
        self.outcome = outcome
        self.step = frame.step
        self.level = frame.level
        self.frame = frame
        self.animation_frames = animation_frames or []
        if _has_plan:
            self.plan = plan
        if _has_world_model:
            self.state = state
            self.predicted_frame = predicted_frame

    def __getattr__(self, name: str) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """A board attribute asked of the turn names where it is; anything
        else gets the usual list of what the turn has."""
        if name in self._FRAME_ATTRIBUTES:
            raise AttributeError(
                f"'Turn' object has no attribute {name!r}; the board is the turn's frame: .frame.{name}"
            )
        return super().__getattr__(name)

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        parts = [f"Turn(action={self.action!r}, outcome={self.outcome!r}"]
        if self.animation_frames:
            parts.append(f", {len(self.animation_frames)} animation frames")
        state = self.__dict__.get("state")
        if state is not None:
            parts.append(", state set")
        parts.append(")")
        hints = [".frame"]
        if self.animation_frames:
            hints.append(".animation_frames")
        if state is not None:
            hints.append(".state")
        if self.__dict__.get("predicted_frame") is not None:
            hints.append(".predicted_frame")
        if self.__dict__.get("plan") is not None:
            hints.append(".plan")
        return "".join(parts) + "\n# explore: " + ", ".join(hints)


#: What a ``Turn`` says about the attributes a world model fills, appended to
#: its docstring in a kernel that has one.
_TURN_WORLD_MODEL_DOC = """    - state: dict | None — world model's predicted state at the end of this turn.
        render(state) draws frame.grid on playing turns. Set by
        test_world_model() and by the check on every move under a passing
        world model; None until then, and past any turn the model did not
        predict.
    - predicted_frame: ndarray | None — the board your world model rendered
        for this turn, set alongside state (-1 where it declined to predict
        a cell). Compare it with frame.grid to see what it got wrong.
    """

#: What a ``Turn`` says about its ``plan``, under ``plan_argument``.
_TURN_PLAN_DOC = """    - plan: str | None — the plan you gave the python call that played this
        turn (None if you gave none).
    """

_TURN_DOC = Turn.__doc__ or ""

#: Whether this process's game has a world model, which decides whether a
#: ``Turn`` carries ``state`` and ``predicted_frame`` and how frozen module
#: data explains itself: set once per process by ``_configure_world_model``.
#: On by default, for the processes that only ever run a world model (the
#: test harness, the search child).
_has_world_model = True


#: Whether this process's turns carry a ``plan`` (``SealAgent.plan_argument``):
#: set once per process by ``_configure_plan``. Off by default.
_has_plan = False


def _turn_doc() -> str:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``Turn``'s docstring with the attributes this process's turns have."""
    doc = _TURN_DOC
    if _has_plan:
        doc = doc.removesuffix("    ") + _TURN_PLAN_DOC
    if _has_world_model:
        doc = doc.removesuffix("    ") + _TURN_WORLD_MODEL_DOC
    return doc


def _configure_world_model(*, enabled: bool) -> bool:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl) and seal.errors
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Whether this process's game has a world model: without one a ``Turn``
    has no ``state`` or ``predicted_frame``, so neither ``help(Turn)``,
    ``dir`` nor the missing-attribute hint names them. Returns the previous
    setting, for a caller that renders for another game and restores it."""
    global _has_world_model
    previous, _has_world_model = _has_world_model, enabled
    Turn.__doc__ = _turn_doc()
    return previous


def _configure_plan(*, enabled: bool) -> bool:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Whether this process's turns carry the ``plan`` of the call that played
    them, which decides whether ``help(Turn)``, ``dir`` and the hint name it.
    Returns the previous setting."""
    global _has_plan
    previous, _has_plan = _has_plan, enabled
    Turn.__doc__ = _turn_doc()
    return previous


_configure_world_model(enabled=True)


class History(Locked):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """This level's gameplay so far, structured by turns.

    Access turns by index: history[n], history[-1]
    Slice: history[a:b] is a History of those turns
    Iterate: for turn in history
    Length: len(history)

    It starts again at every new level -- levels of one game share their
    mechanics but not their board, so a turn from the old one describes
    something that is no longer there. What you want to keep across a level
    goes in a file or the journal.
    """

    _usage = (
        "history is this level's turns, not a list: index it (history[-1]), slice it (history[a:b]), "
        "iterate it, len() it; enumerate(history) gives each turn's index."
    )

    def __init__(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        self._turns: list[Turn] = []

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        n = len(self._turns)
        if n == 0:
            return "History(empty)"
        lines = [f"History({n} turns)"]
        # Show last few turns as summary
        show = self._turns[-3:] if n > 3 else self._turns
        start = max(0, n - 3)
        for i, turn in enumerate(show, start=start):
            anim = f", {len(turn.animation_frames)} anim" if turn.animation_frames else ""
            lines.append(f"  [{i}] action={turn.action!r}{anim}")
        if n > 3:
            lines.insert(1, f"  ... ({n - 3} earlier turns)")
        lines.append(f"# explore: history[-1], history[0], history[-1].frame")  # noqa: F541
        return "\n".join(lines)

    @overload
    def __getitem__(self, index: int) -> Turn: ...

    @overload
    def __getitem__(self, index: slice) -> "History": ...

    def __getitem__(self, index: int | slice) -> "Turn | History":
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """A turn by index, or a History of the turns a slice takes: the same
        turns, read-only as this one is."""
        if isinstance(index, slice):
            part = History()
            part._turns = self._turns[index]
            if self._locked:
                part._lock()
            return part
        return self._turns[index]

    def __len__(self) -> int:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return len(self._turns)

    def __iter__(self) -> "Iterator[Turn]":
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return iter(self._turns)

    def _append_turn(self, entry: dict[str, Any]) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Add a new turn from a host payload entry. Diffs are computed automatically."""
        if getattr(self, "_locked", False):
            raise RuntimeError("Cannot modify history.")

        # Extract structured action
        action = _move(entry) if entry.get("action") else None

        # Game outcome: win (level or game), lose (game over), or playing.
        # Already derived by the agent's observation bookkeeping.
        outcome = str(entry.get("outcome", "playing"))

        # Extract grids: the entry's frames, of which the last is the settled
        # board and the rest its animation.
        frames = [frame for frame in (_grid_from_payload(item) for item in entry.get("animation", [])) if frame]
        if not frames:
            return False
        grid, step, level = frames[-1]
        anim_grids = [frame[0] for frame in frames[:-1]]

        # Determine previous frame for diff
        prev_grid: npt.NDArray[np.int8] | None
        if self._turns:
            prev_grid = self._turns[-1].frame.grid
        else:
            prev_grid = None

        settled = Frame(
            grid, previous=prev_grid, diff_context="previous turn", frame_type="visible", step=step, level=level
        )

        # The animation frames the turn keeps (``kept_animation``: no repeats,
        # no trailing copies of the settled board), with chained diffs: the
        # first against the previous turn's settled frame, each later one
        # against the one before it.
        animation_frames: list[Frame] = []
        anim_prev = prev_grid
        for anim_grid in _kept_animation(prev_grid, anim_grids, grid, _same_grid):
            ctx = "previous turn" if not animation_frames else "previous animation frame"
            animation_frames.append(
                Frame(anim_grid, previous=anim_prev, diff_context=ctx, frame_type="animation", step=step, level=level)
            )
            anim_prev = anim_grid

        plan = entry.get("plan")
        turn = Turn(action, outcome, settled, animation_frames, plan=str(plan) if plan is not None else None)
        # Lock the turn and all its frames
        settled._lock()
        for af in animation_frames:
            af._lock()
        turn._lock()

        self._turns.append(turn)
        return True


class Unknown(Exception):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Raise this from do_move() if the move is valid but the outcome is unpredictable based on our knowledge so far."""

    pass


def _is_decline(error: BaseException) -> bool:  # pyright: ignore[reportUnusedFunction]  # called from seal.harness._repl_state, seal.harness.test and seal.model
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Whether ``error`` is the model declining a move: the preloaded
    ``Unknown``, or a class of its own by that name — a world model that
    defines one means the same thing, and should not read as a crash."""
    return isinstance(error, Unknown) or type(error).__name__ == "Unknown"


def _same_grid(first: npt.NDArray[np.int8], second: npt.NDArray[np.int8]) -> bool:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Whether two boards are the same, cell for cell."""
    return bool(np.array_equal(first, second))


def _grid_from_payload(payload: Any) -> tuple[npt.NDArray[np.int8], int, int] | None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """One frame payload as ``(grid, step, level)``, or None for a malformed one."""
    if not isinstance(payload, dict):
        return None
    payload = cast("dict[str, Any]", payload)
    b64 = payload.get("grid_b64") or ""
    rows, cols = (int(value) for value in payload.get("shape", [0, 0]))
    flat = np.frombuffer(base64.b64decode(b64) if b64 else b"", dtype=np.uint8)
    if flat.size != rows * cols:
        return None
    grid = flat.reshape(rows, cols).astype(np.int8)
    return grid, int(payload.get("step", 0)), int(payload.get("level", 0))


class Node(Locked):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """One 4-connected same-colour component of a board, from segment().

    Properties:
    - id: int -- its index in the Segmentation, reading order of the
        component's top-left cell.
    - color: int -- the palette index (not a character).
    - hash: str -- translation-invariant, so the same shape matches itself
        across frames or positions.
    - pixels: np.ndarray (N, 2) of [row, col] cells in reading order, so
        `frame.grid[node.pixels[:, 0], node.pixels[:, 1]]` reads them.
    - pixel_count: int
    - bbox: (row0, row1, col0, col1) -- the box around it, half-open:
        `grid[row0:row1, col0:col1]` is its box.
    - boundary: np.ndarray (M, 2) of [row, col] corner points of the outer
        perimeter, clockwise.
    - children: list[int] -- ids of the components this one encloses.
    """

    def __init__(
        self,
        id: int,
        color: int,
        hash: str,
        pixels: npt.NDArray[np.int64],
        pixel_count: int,
        bbox: tuple[int, int, int, int],
        boundary: npt.NDArray[np.int64],
        children: list[int],
    ):
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        self.id = id
        self.color = color
        self.hash = hash
        self.pixels = pixels
        self.pixel_count = pixel_count
        self.bbox = bbox
        self.boundary = boundary
        self.children = children

    def __getitem__(self, key: Any) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        # The dict form is the first thing a model tries on a node; the
        # answer names the form that works rather than the type that does not.
        raise TypeError(f"Node is not a dict: read node.{key} rather than node[{key!r}]. See help(Node).")

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        row0, row1, col0, col1 = self.bbox
        # The box as slices, as `.bbox` is: a range printed any other way reads off by one against it.
        parts = [
            f"Node(id={self.id}, color={self.color}, {self.pixel_count} px, rows {row0}:{row1}, cols {col0}:{col1}"
        ]
        if self.children:
            parts.append(f", {len(self.children)} children")
        parts.append(")")
        return "".join(parts) + "\n# explore: .bbox, .pixels, .boundary, .children, .hash"


class Segmentation:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """The connected components of one board, and how they touch.

    Properties:
    - nodes: list[Node] -- one 4-connected same-colour component each, in
        reading order of their top-left cell, which is also their `id`.
    - adjacency_list: list[[i, j]] -- id pairs of components sharing an edge.

    Index it (`seg[0]`), iterate it, and len() it to reach `nodes`;
    `seg.by_color(5)` filters. See help(Node) for what a node holds.
    """

    def __init__(self, nodes: list[Node], adjacency_list: list[list[int]]):
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        self.nodes = nodes
        self.adjacency_list = adjacency_list

    def __getitem__(self, index: int) -> Node:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return self.nodes[index]

    def __len__(self) -> int:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return len(self.nodes)

    def __iter__(self) -> "Iterator[Node]":
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        return iter(self.nodes)

    def by_color(self, color: int) -> list[Node]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Every node of one colour index."""
        return [node for node in self.nodes if node.color == int(color)]

    def __repr__(self) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        counts: dict[int, int] = {}
        for node in self.nodes:
            counts[node.color] = counts.get(node.color, 0) + 1
        summary = ", ".join(f"{color}:{count}" for color, count in sorted(counts.items()))
        return (
            f"Segmentation({len(self.nodes)} nodes; per colour {summary})"
            "\n# explore: seg[0], seg.by_color(5), seg.adjacency_list"
        )


# segment_layer indexes this per component, so a range of ints yields integer
# colours without touching the verbatim segmentation source.
_COLOR_INDICES = list(range(16))
# The verbatim source is unannotated by design (it is diffed against its
# upstream); the type it has is stated here instead.
segment_layer = cast("Callable[[list[list[int]], list[int]], dict[str, Any]]", _segment_layer_untyped)


# ignore: unannotated for the same reason as ascii_grid -- help() renders the signature.
def _node(found: dict[str, Any]) -> Node:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """One component of ``segment_layer``'s result as a Node."""
    pixels = np.asarray(found["pixels"], dtype=int).reshape(-1, 2)
    (row0, col0), (row1, col1) = pixels.min(axis=0), pixels.max(axis=0) + 1
    return Node(
        id=int(found["id"]),
        color=int(found["color"]),
        hash=str(found["hash"]),
        pixels=pixels,
        pixel_count=int(found["pixel_count"]),
        bbox=(int(row0), int(row1), int(col0), int(col1)),
        boundary=np.asarray(found["boundary"], dtype=int).reshape(-1, 2),
        children=[int(child) for child in found["children"]],
    )


def segment(grid) -> Segmentation:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Segment any 2D board into 4-connected same-colour components.

    Takes an ndarray, a list of lists, or a Frame's `.grid` -- a crop, a
    mask, or a board you predicted, not only a frame the game handed you.
    Returns a Segmentation; see help(Node) for what a node holds.
    """
    rows = np.asarray(grid, dtype=np.int8)  # pyright: ignore[reportUnknownArgumentType]
    if rows.ndim != 2:
        raise ValueError(f"segment expects a 2D array, got {rows.ndim}D.")
    result = segment_layer([[int(value) for value in row] for row in rows], _COLOR_INDICES)
    nodes = [_node(node) for node in cast("list[dict[str, Any]]", result["nodes"])]
    for node in nodes:
        node._lock()
    return Segmentation(nodes, cast("list[list[int]]", result["adjacency_list"]))


def _plain(value: Any) -> Any:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """A numpy scalar as the Python value it holds; anything else unchanged.

    The actions cross to the host as a restricted pickle that refuses numpy
    types, and a coordinate computed from a grid is a numpy integer. The host
    validates the value either way; this only keeps a valid one from being
    refused on the way."""
    return cast("np.generic[Any]", value).item() if isinstance(value, np.generic) else value


def _move(entry: dict[str, Any]) -> Action:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """A move as the host records it -- ``{'action': 'MOUSE', 'row': 4,
    'col': 7}`` -- as the Action the model reads."""
    row, col = entry.get("row"), entry.get("col")
    return Action(str(entry.get("action", "")), None if row is None else int(row), None if col is None else int(col))


def _with_actions(result: dict[str, Any]) -> dict[str, Any]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """A result dict from the host with its moves as Actions."""
    out = dict(result)
    executed = out.get("executed_actions")
    if isinstance(executed, list):
        out["executed_actions"] = [
            _move(cast("dict[str, Any]", entry)) if isinstance(entry, dict) else entry
            for entry in cast("list[Any]", executed)
        ]
    if isinstance(out.get("action"), dict):
        out["action"] = _move(cast("dict[str, Any]", out["action"]))
    return out


def _normalize_actions(actions: Any, function_name: str) -> list[dict[str, Any]]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """The model's argument to ``submit_action`` as a list of action dicts.
    Actions are the documented form; a bare name or a dict is read the same
    way, so a model writing either is not refused for it."""
    if isinstance(actions, (Action, str, dict)):
        items: list[Any] = [actions]
    elif isinstance(actions, (list, tuple)):
        items = list(cast("Iterable[Any]", actions))
    else:
        raise TypeError(f"{function_name}(actions) expects an Action or a list of Actions, e.g. [Action('UP')].")
    if not items:
        raise ValueError(f"{function_name}(actions) requires at least one action.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, Action):
            move: dict[str, Any] = {"action": item.name}
            if item.row is not None:
                move["row"], move["col"] = item.row, item.col
            normalized.append(move)
            continue
        if isinstance(item, str):
            action_name = item.strip()
            if not action_name:
                raise ValueError(f"Action {index} is empty.")
            normalized.append({"action": action_name})
            continue
        if isinstance(item, dict):
            item = cast("dict[str, Any]", item)
            action_name = str(item.get("action", "")).strip()
            if not action_name:
                raise ValueError(f"Action {index} is missing an `action` field.")
            entry: dict[str, Any] = {"action": action_name}
            if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
            if "row" in item:
                entry["row"] = _plain(item.get("row"))
            if "col" in item:
                entry["col"] = _plain(item.get("col"))
            normalized.append(entry)
            continue
        raise TypeError(f"Action {index}: expected an Action such as Action('UP'), got {type(item).__name__}.")
    return normalized


class WorldModelStop(Exception):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Raised by submit_action() when your tested world model stops the play:
    the game contradicted a move it predicted, or it declined to predict a
    move with Unknown and another move was about to run. The moves before the
    stop were executed and are on history; the
    rest of the cell does not run."""


class WorldModelRequired(Exception):
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Raised by submit_action() when moves need a passing test_world_model()
    and yours fails. Nothing was executed; the message shows how it failed."""


#: What a refused batch of actions ends with: the game's own actions, so a
#: model that wrote an unknown or malformed one sees what it can send. The
#: host's refusals end with the same (``SealPrompts.valid_actions_note``).
VALID_ACTIONS_NOTE = " This game's actions: {actions}."

#: What ``submit_action(..., plan=...)`` is told where the ``python`` tool
#: takes a ``plan``: the parameter is the tool call's.
PLAN_ON_SUBMIT_ACTION = (
    "`plan` is an argument of the `python` tool call itself, next to `code`, not of submit_action. "
    "Nothing was executed."
)

#: Stands for a ``plan`` the call did not give.
_NO_PLAN: Any = object()


def _make_submit_action(  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    submit: Callable[[list[dict[str, Any]]], Any],
    install: Callable[[dict[str, Any]], None],
    gate: Any = None,
    game_actions: Callable[[], list[str]] | None = None,
) -> Callable[[Any], dict[str, Any]]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """The model's ``submit_action``, over the store's host adapter.

    A factory rather than a module-level function holding the store: the
    function's globals are this module's, which the model may read, and the
    store stays out of them. ``install`` is what the reply's refreshed state
    goes through, so the new turns are live for the rest of the cell.

    ``gate`` is the store's world-model gate. Before a move it validates the
    model silently. A passing model runs the batch stepped: the host pauses
    after every move, the gate checks the new turn, and the batch continues
    or stops there. Without one the batch goes to the host as exploration,
    which the host runs or refuses (``mandatory``), or as the single batched
    call it is with the gate off (``optional``, and always under ``none``).

    ``game_actions`` names the game's actions, which a malformed batch's
    refusal ends with. A ``plan`` keyword, which models whose ``python`` tool
    takes one sometimes give here, is refused with where it belongs; the
    signature ``help()`` shows has no such parameter.
    """

    def _with_game_actions(error: Exception) -> Exception:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """``error`` ending with the game's actions, of its own type."""
        names = game_actions() if game_actions is not None else []
        if not names:
            return error
        return type(error)(f"{error}{VALID_ACTIONS_NOTE.format(actions=', '.join(names))}")

    def _received(reply: Any) -> dict[str, Any]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """One host reply: refusals raised, the refreshed state installed,
        the result dict handed back."""
        if not isinstance(reply, dict):
            raise RuntimeError("Invalid action response from host.")
        reply = cast("dict[str, Any]", reply)
        if reply.get("error"):
            raise ValueError(str(reply["error"]))
        if isinstance(reply.get("state"), dict):
            # Live execution: the host ran the actions and returned the
            # refreshed state, so the new turns are appended and
            # `history` is live for the rest of the cell.
            install(cast("dict[str, Any]", reply["state"]))
        res: Any = reply.get("result") if isinstance(reply.get("result"), dict) else (reply.get("ack") or {})
        return _with_actions(cast("dict[str, Any]", res))

    def _stopped(reply: Any) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Install the stopped batch's final state, and say how far it got."""
        final = _received(reply)
        executed = list(cast("list[Any]", final.get("executed_actions") or []))
        requested = int(final.get("requested_count") or len(executed))
        if len(executed) < requested:
            return f" The batch stopped after {len(executed)} of {requested} actions; the rest were not executed."
        return ""

    def submit_action(actions, plan=_NO_PLAN):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]  # help() renders the signature
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
        """Execute one or more game actions: an Action, or a list of them
        (``[Action('UP'), Action('MOUSE', row=r, col=c)]``). history is
        already updated when it returns."""
        if plan is not _NO_PLAN:
            if _has_plan:
                raise TypeError(PLAN_ON_SUBMIT_ACTION)
            raise TypeError("submit_action() got an unexpected keyword argument 'plan'")
        try:
            normalized_actions = _normalize_actions(actions, "submit_action")
        except (TypeError, ValueError) as error:
            raise _with_game_actions(error) from None
        if gate is None or gate.mode == "none":
            return _received(submit(normalized_actions))
        if gate.declined is not None:
            raise WorldModelStop(
                f"Your world model declined to predict an earlier move in this cell ({gate.declined}), "
                "so no further action runs here. Nothing was executed."
            )
        if gate.stopped:
            raise WorldModelStop(
                "Your world model mispredicted an earlier move in this cell, so no further action runs here. "
                "Nothing was executed."
            )
        failing = gate.before_first_move()
        if failing is not None and gate.mode == "optional":
            return _received(submit(normalized_actions))
        if failing is not None:
            reply = gate.host_exploring(normalized_actions)
            if isinstance(reply, dict) and cast("dict[str, Any]", reply).get("world_model_required"):
                raise WorldModelRequired(
                    f"{cast('dict[str, Any]', reply)['error']}\nNothing was executed. test_world_model() gives:\n{failing!r}"
                )
            return _received(reply)
        gate.pre_move()
        reply = gate.host_stepped(normalized_actions)
        while True:
            paused = isinstance(reply, dict) and bool(cast("dict[str, Any]", reply).get("paused"))
            res = _received(reply)
            if not paused:
                return res
            note = gate.after_move()
            remaining = int(cast("dict[str, Any]", reply).get("remaining") or 0)
            if note is not None:
                raise WorldModelStop(
                    f"Your world model mispredicted history[-1]: {note}"
                    + _stopped(gate.host_stop("world_model_mismatch", note))
                )
            if gate.declined is not None and remaining:
                raise WorldModelStop(
                    f"Your world model declined to predict history[-1]: {gate.declined}. "
                    "The move was executed, but nothing further runs in this cell."
                    + _stopped(gate.host_stop("world_model_unknown", gate.declined))
                )
            gate.pre_move()
            reply = gate.host_continue()

    # What help() and inspect show: the one parameter the model is told of.
    submit_action.__signature__ = inspect.Signature(  # pyright: ignore[reportFunctionMemberAccess]
        [inspect.Parameter("actions", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    return submit_action  # pyright: ignore[reportUnknownVariableType]  # the unannotated parameter, see above


def __dir__() -> list[str]:
    """The names the model should find here, not the ones we imported.

    ``dir()`` on a module lists its whole namespace by default, so without
    this the model reads back our imports — ``segment_layer``, ``np``,
    ``cast`` — names it cannot use and, in the first case, cannot even
    import. Answering with :data:`__all__` makes the question "what is on
    this surface?" get the surface as its answer.
    """
    return sorted(__all__)


#: Whether this process's model has ``segment()`` (``SealAgent.segmentation``): set once per process, at the
#: kernel's start, by ``_configure_segmentation``. On by default.
_has_segmentation = True

#: The surface segmentation adds.
_SEGMENTATION_NAMES = ("segment", "Segmentation", "Node")


def _configure_segmentation(*, enabled: bool) -> None:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Whether this process's model has ``segment()``, ``Segmentation`` and ``Node``. Off, they are not
    preloaded, neither ``help()`` nor ``dir`` names them, and they leave this module, so importing it does not
    reach them either. Set once, before the names are preloaded: off is not turned back on."""
    global _has_segmentation
    if enabled == _has_segmentation:
        return
    if enabled:
        raise RuntimeError("segmentation, once off, is not turned back on in this process")
    _has_segmentation = False
    for name in _SEGMENTATION_NAMES:
        __all__.remove(name)
        del globals()[name]


def _build_common_namespace() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Names the REPL __main__ namespace is populated with, from one source
    of truth, so the surface the LLM sees is the one this module documents."""
    return {
        "Frame": Frame,
        "Turn": Turn,
        "History": History,
        "Action": Action,
        "ColorCounts": ColorCounts,
        "DiffSparse": DiffSparse,
        **({"Node": Node, "Segmentation": Segmentation, "segment": segment} if _has_segmentation else {}),
        "ascii_grid": ascii_grid,
        "from_ascii_grid": from_ascii_grid,
        "bresenham_line": bresenham_line,
        "color_names": color_names,
        "help": help,
    }


def _preloaded_names(*, world_model: bool, required: bool) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl) and seal.harness._repl_state
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """The names every piece of the model's code starts with — a snippet, a
    module it wrote, and world_model.py alike: the standard modules the
    prompt names, numpy as ``np``, the surface, and the world model's
    exceptions when one is in use (``WorldModelRequired`` only where a
    passing model is ``required`` to act). What differs between those three
    is only what their loaders add on top: the moving state and the acting
    tools."""
    import collections
    import functools
    import itertools
    import json
    import math
    import re

    names: dict[str, Any] = {
        "np": np,
        "copy": copy,
        "json": json,
        "re": re,
        "math": math,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
        "os": os,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
        "deque": collections.deque,
    }
    names.update(_build_common_namespace())
    if world_model:
        names.update(Unknown=Unknown, WorldModelStop=WorldModelStop)
    if world_model and required:
        names["WorldModelRequired"] = WorldModelRequired
    return names


def _build_wm_namespace(  # pyright: ignore[reportUnusedFunction]  # called from seal.harness.test
    available_actions: list[str],
    file: str,
    *,
    history: History,
) -> dict[str, Any]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Build the namespace dict used to exec a world_model.py file.

    - Populates the preloaded names every piece of the model's code gets
      (:func:`_preloaded_names`) — not the acting tools or the live frame,
      which a replay must not reach.
    - Binds ``available_actions`` to a defensive copy of the caller's list,
      and ``history`` to the caller's history — module-level code may read
      the whole level, not just its first turn.
    - Sets ``__file__`` / ``__name__`` so tracebacks are meaningful and
      code can locate its own directory via ``Path(__file__).parent``.

    The returned dict is passed to ``exec(code, ns)`` by
    ``seal.harness.test._load_world_model``. Every call produces a fresh
    dict so different callers never share a namespace.
    """
    import builtins

    ns = _preloaded_names(world_model=True, required=True)
    ns["__file__"] = file
    ns["__name__"] = "world_model"
    ns["__builtins__"] = builtins.__dict__
    ns["available_actions"] = list(available_actions)
    ns["history"] = history
    return ns


#: Never evicted by the sibling-import eviction below.
#:
#: ``world_model`` genuinely lives in the model's working directory, so the
#: evictor would match it, and evicting it breaks pickle-by-reference of
#: the model's classes.
#:
#: ``__main__`` is here because the eviction resolves each module's
#: ``__file__`` with ``os.path.abspath``, which is CWD-RELATIVE, and this
#: can run with the cwd chdir'd into the model's own directory. A
#: ``__main__`` whose ``__file__`` is relative therefore resolves *inside*
#: that directory and gets evicted, after which ``multiprocessing`` spawn
#: dies on ``KeyError: '__main__'`` in ``get_preparation_data``, taking
#: every search with it.
_WM_PROTECTED_MODULES = frozenset({"world_model", "__main__"})

#: Set by the REPL kernel, where the model's modules are the objects its cells
#: and modules hold and must stay registered: a load there re-runs, in place,
#: any of them edited since it last ran, instead of dropping them all
#: (``seal.harness._repl_state.rerun_edited_modules``).
_rerun_edited_modules: Callable[[], None] | None = None


def _preloaded_import_scope(  # pyright: ignore[reportUnusedFunction]  # called from seal.py_model
    directory: str, names: dict[str, Any]
) -> AbstractContextManager[None]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """For its duration, a module imported from ``directory`` starts with
    ``names`` in its namespace, as the kernel's workspace loader gives the
    model's modules the names a cell starts with.

    Outside the kernel — the host's second test, the search child — a plain
    import would run a helper module the model wrote without them, and a
    helper that calls ``segment`` or ``Counter`` unimported, as it may in
    the REPL, would raise where the REPL passed it."""
    import contextlib
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import os
    import sys

    dir_abs = os.path.abspath(directory)
    preset = {key: value for key, value in names.items() if not (key.startswith("__") and key.endswith("__"))}

    class _Loader(importlib.machinery.SourceFileLoader):
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Runs a module with the preloaded names already in it."""

        def exec_module(self, module: Any) -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            """Seed the namespace, run the file, freeze its data — as the
            kernel's loader does."""
            from seal.harness import test as harness_test

            for key, value in preset.items():
                module.__dict__.setdefault(key, value)
            started = set(module.__dict__)
            super().exec_module(module)
            harness_test._freeze_module_globals(module.__dict__, started)

    class _Finder(importlib.abc.MetaPathFinder):
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Finds a top-level module that is a file in ``directory``."""

        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            """A spec for ``directory/<fullname>.py``, or None — also when
            Python already has a module by that name, as in the kernel."""
            if "." in fullname:
                return None
            candidate = os.path.join(dir_abs, fullname + ".py")
            if not os.path.isfile(candidate) or module_names.taken(fullname, dir_abs):
                return None
            return importlib.util.spec_from_file_location(fullname, candidate, loader=_Loader(fullname, candidate))

    @contextlib.contextmanager
    def _ctx() -> Generator[None, None, None]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        finder = _Finder()
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass

    return _ctx()


def _wm_sibling_import_path(  # pyright: ignore[reportUnusedFunction]  # called from seal.harness.test
    directory: str,
    names: dict[str, Any] | None = None,
) -> AbstractContextManager[None]:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Context manager that makes the model's own ``.py`` helper modules next
    to world_model.py importable during an ``exec`` of its source.

    The model may factor data into a helper module via the file tools (e.g.
    ``from base_grid import BASE_GRID``). In the REPL kernel the working
    directory is already cwd-on-sys.path, so that import resolves; loaders
    in other processes have no such entry, so ``directory`` goes on
    sys.path for the duration of the exec to match the REPL.

    Around the exec, every module imported from inside ``directory`` (except
    the protected ones) is evicted from sys.modules — on entry so this load
    re-reads the model's latest edit from disk rather than a stale cached
    version, and on exit so the per-game working-directory modules don't
    leak into unrelated code.

    In the REPL kernel nothing is evicted: the directory is importable
    already, and the model's modules are objects its cells and modules
    hold, which the kernel refreshes after every action only while they
    stay registered. Any edited since it last ran is re-run in place
    instead, so an edit still counts at once.

    Outside the kernel, ``names`` — the namespace world_model.py runs in —
    is what the model's other modules start with too
    (:func:`_preloaded_import_scope`), as the kernel's loader gives them."""
    import contextlib
    import os
    import sys

    @contextlib.contextmanager
    def _ctx() -> Generator[None, None, None]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        dir_abs = os.path.abspath(directory)
        if _rerun_edited_modules is not None:
            _rerun_edited_modules()
            yield
            return

        def _evict() -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            for name in list(sys.modules):
                if name in _WM_PROTECTED_MODULES:
                    continue
                f = getattr(sys.modules.get(name), "__file__", None)
                if f and os.path.abspath(f).startswith(dir_abs + os.sep):
                    del sys.modules[name]

        # Evicting sys.modules forces a re-import, and the re-import reads
        # the file on disk: bytecode is never written (``sys.dont_write_bytecode``
        # at package import), so no cached entry can be an edit behind.
        _evict()
        # Last, as in the kernel: a module Python already has keeps its name.
        sys.path.append(dir_abs)
        preloaded = contextlib.ExitStack()
        if names is not None:
            preloaded.enter_context(_preloaded_import_scope(dir_abs, names))
        try:
            with preloaded:
                yield
        finally:
            try:
                sys.path.remove(dir_abs)
            except ValueError:
                pass
            _evict()

    return _ctx()


_BLOCKED_NOTICE = (
    "[Source code for the runtime's own internals is not available. You can freely read "
    "files in your working directory and any files you create. Use help(obj) for "
    "docstrings on the preloaded objects.]"
)
_BLOCKED_DENIAL = "Source code access denied"

#: Shown when a read or a listing reaches outside the places the model's code
#: has any use for. Names none of them, for the reason the write notice gives.
_OUTSIDE_NOTICE = (
    "[Files outside your working directory are not available here, apart from the Python "
    "installation and its libraries. You can freely read files in your working directory "
    "and any files you create.]"
)
_OUTSIDE_DENIAL = "File access denied"

#: Shown when a write lands outside the REPL's own directory. It deliberately
#: does NOT name that directory: an absolute path here is cluster-specific —
#: it carries the SLURM job id and this checkout's name — so a reference
#: recording it reproduces on no other machine, and it tells the model a
#: harness repository exists at all. The path is also of no use to it: the
#: directory named is already its cwd, so every relative path it writes
#: resolves there. What it needs to know is *that* writes are confined.
_WRITE_NOTICE = (
    "[Writes outside your working directory are not allowed. You can freely "
    "create and modify files in your working directory (the one you start "
    "in, and where a relative path resolves). The write was discarded; the "
    "on-disk file is unchanged.]"
)


def _notify(notice: str) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Put a notice in the cell's output, and only there.

    A blocked read or write also happens outside a cell — rendering a
    traceback for the reply, say — where the kernel's own stdout is the
    protocol stream the host reads as JSON. A notice printed there is not
    output the model ever sees: it corrupts the reply and the host kills
    the kernel. The cell's stdout is a capture the worker redirects to, so
    printing only while redirected is what keeps the notice to the cell.
    """
    if sys.stdout is not sys.__stdout__:
        print(notice, flush=True)


class _PathRules:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Which paths the REPL may read, and where it may write.

    Resolved once, at the moment the kernel installs the patches, and shared
    by all three of them so every entry point answers the same question the
    same way. Protected directories come from the imported packages'
    ``__file__``; the tempdir is the kernel subprocess's spawn cwd, captured
    before the LLM can ``chdir`` away from it.
    """

    #: The packages whose source the model may not read: ours, and the game
    #: engine and the games, which run in the host process but are installed
    #: beside the libraries the model may read. framediff's is the
    #: implementation behind from_ascii_grid and bresenham_line, which their
    #: help() describes instead.
    PROTECTED = (
        "seal",
        "platypus",
        "taaf",
        "ullm",
        "framediff",
        "duck",
        "sweeper",
        "reference_agents",
        "arcengine",
        "arc_agi",
        "re_arc",
    )

    #: The system's own directories, readable like Python's: libraries read
    #: from them, and they hold nothing of this run or of any other.
    SYSTEM_ROOTS = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev")

    def __init__(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        import importlib.util

        self.home = os.path.realpath(os.path.expanduser("~"))
        self.blocked_dirs: list[str] = []
        for pkg_name in self.PROTECTED:
            # Located, not imported: the games have no business in the kernel.
            try:
                spec = importlib.util.find_spec(pkg_name)
            except (ImportError, ValueError):
                continue
            if spec is None:
                continue
            locations = list(spec.submodule_search_locations or [])
            if not locations and spec.origin:
                locations = [os.path.dirname(spec.origin)]
            self.blocked_dirs.extend(os.path.realpath(location) for location in locations)
        self.tempdir = os.path.realpath(os.getcwd())
        # Where reads may go: the working directory, the kernel's HOME, Python
        # and its libraries, and the system's own directories. Not the rest of
        # the job's directory, the other games' working directories beside
        # this one, other jobs, or shared storage holding earlier runs.
        python_roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, *sys.path}
        self.readable_roots = [self.tempdir, self.home, *self.SYSTEM_ROOTS]
        self.readable_roots.extend(os.path.realpath(root) for root in python_roots if root)

    @staticmethod
    def resolve(path: Any) -> str | None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """One argument as an absolute path, or None if it is not one."""
        try:
            # cast + ignore: `path` is whatever the patched entry point was handed,
            # so fspath/realpath cannot pick their str-vs-bytes overload here; every
            # call site compares the result against str paths.
            return cast(str, os.path.realpath(os.fspath(path)))  # pyright: ignore[reportUnknownArgumentType]
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _within(path: str, directory: str) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Whether ``path`` is ``directory`` or lies under it."""
        return path == directory or path.startswith(directory + os.sep)

    def is_blocked(self, path: Any) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Whether reading this path would hand the model our source."""
        resolved = self.resolve(path)
        if resolved is None:
            return False
        return any(self._within(resolved, d) for d in self.blocked_dirs)

    def is_readable(self, path: Any) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Whether a read or a listing here is allowed: under one of the
        readable roots, and not our source. A path that cannot be resolved is
        let through, as ``is_blocked`` does."""
        resolved = self.resolve(path)
        if resolved is None:
            return True
        if any(self._within(resolved, d) for d in self.blocked_dirs):
            return False
        return any(self._within(resolved, root) for root in self.readable_roots)

    def is_listable(self, path: Any) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Whether a directory listing here is allowed: under one of the
        readable roots. Our packages list like any library — names are not
        source, and ``importlib.metadata`` lists every ``sys.path`` entry,
        the worker's own directory inside ullm among them."""
        resolved = self.resolve(path)
        if resolved is None:
            return True
        return any(self._within(resolved, root) for root in self.readable_roots)

    def is_writable(self, path: Any) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Whether a write here is allowed: the REPL tempdir, or the kernel's
        scratch HOME — libraries write their caches under ``~``, and HOME
        points outside the working directory precisely so those writes do not
        land where the model's files are listed."""
        resolved = self.resolve(path)
        if resolved is None:
            return False
        if any(self._within(resolved, d) for d in self.blocked_dirs):
            # Protected source is never writable, wherever it happens to sit.
            # The HOME exemption is for library caches and takes HOME to be a
            # scratch directory outside the checkout — which the kernel's HOME
            # is, but a machine that keeps its checkout under HOME is not, and
            # there the exemption would otherwise reach the very code these
            # rules exist to protect.
            return False
        return self._within(resolved, self.tempdir) or self._within(resolved, self.home)

    def deny_read(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Tell the model why the read it just made came back empty."""
        _notify(_BLOCKED_NOTICE)

    def deny_write(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Tell the model why the write it just made went nowhere."""
        _notify(_WRITE_NOTICE)

    def deny_outside(self) -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Tell the model why the read or listing outside its places failed."""
        _notify(_OUTSIDE_NOTICE)


def _patch_inspect(rules: _PathRules) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``inspect``'s four source-reading entry points, each returning the
    denial sentinel for a blocked object rather than its source."""
    import inspect as _inspect

    orig_getsource = _inspect.getsource
    orig_getsourcelines = _inspect.getsourcelines
    orig_getsourcefile = _inspect.getsourcefile
    orig_getfile = _inspect.getfile

    def object_is_blocked(obj: Any) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        try:
            return rules.is_blocked(orig_getfile(obj))
        except (TypeError, OSError):
            return False

    def _getsource(obj: Any, *args: Any, **kwargs: Any) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if object_is_blocked(obj):
            rules.deny_read()
            return _BLOCKED_DENIAL
        return orig_getsource(obj, *args, **kwargs)

    def _getsourcelines(obj: Any, *args: Any, **kwargs: Any) -> tuple[list[str], int]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if object_is_blocked(obj):
            rules.deny_read()
            return ([_BLOCKED_DENIAL], 0)
        return orig_getsourcelines(obj, *args, **kwargs)

    def _getsourcefile(obj: Any, *args: Any, **kwargs: Any) -> str | None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if object_is_blocked(obj):
            rules.deny_read()
            return None
        return orig_getsourcefile(obj, *args, **kwargs)

    def _getfile(obj: Any, *args: Any, **kwargs: Any) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if object_is_blocked(obj):
            rules.deny_read()
            return _BLOCKED_DENIAL
        return orig_getfile(obj, *args, **kwargs)

    for patched in (_getsource, _getsourcelines, _getsourcefile, _getfile):
        patched._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_inspect.getsource, "_sandbox_patched", False):
        _inspect.getsource = _getsource
        _inspect.getsourcelines = _getsourcelines
        _inspect.getsourcefile = _getsourcefile
        _inspect.getfile = _getfile


def _patch_linecache(rules: _PathRules) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``linecache``, which is how ``inspect`` reads a file underneath.

    These return empty lines rather than the denial sentinel. Python's
    traceback formatter calls ``linecache.getline`` for every frame it
    shows, so returning the sentinel would put ``Source code access denied``
    where the source line goes — and the LLM reading that traceback takes it
    for the root error ("the runtime refused to load my module")
    and debugs a problem that does not exist. Empty is what linecache
    already answers for a file that isn't there: the traceback omits the
    source line and the real exception reads cleanly.
    """
    import linecache as _linecache

    orig_getline = _linecache.getline
    orig_getlines = _linecache.getlines

    def _getline(filename: str, lineno: int, *args: Any, **kwargs: Any) -> str:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if rules.is_blocked(filename):
            return ""
        return orig_getline(filename, lineno, *args, **kwargs)

    def _getlines(filename: str, *args: Any, **kwargs: Any) -> list[str]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if rules.is_blocked(filename):
            return []
        return orig_getlines(filename, *args, **kwargs)

    _getline._sandbox_patched = True  # type: ignore[attr-defined]
    _getlines._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_linecache.getline, "_sandbox_patched", False):
        _linecache.getline = _getline
        _linecache.getlines = _getlines


def _patch_open(rules: _PathRules) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """``builtins.open`` and ``io.open``, which is where ``pathlib`` lands.

    A blocked read and a write outside the tempdir both come back as an
    in-memory proxy holding the denial text: it supports read / readlines /
    iteration *and* write, so the caller's code runs to completion and the
    write is discarded on close rather than reaching disk.
    """
    import builtins as _builtins
    import io as _io

    def is_write_mode(mode: str) -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        # 'w'/'a'/'x' write/append/exclusive-create; '+' adds write to any read mode.
        return any(c in mode for c in ("w", "a", "x", "+"))

    def proxy(mode: str, denial: str = _BLOCKED_DENIAL) -> _io.BytesIO | _io.StringIO:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if "b" in mode:
            return _io.BytesIO(denial.encode("utf-8"))
        return _io.StringIO(denial)

    def make_patched_open(orig: Any) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        def _patched_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            # int (fd) bypasses — we can't resolve fds to paths reliably
            # and blocking them breaks too much legitimate code.
            if isinstance(file, int):
                return orig(file, mode, *args, **kwargs)
            try:
                # ignore: `file` is whatever open() was called with, so fspath cannot
                # pick its str-vs-bytes overload; both consumers below take Any.
                path_str = os.fspath(file)  # pyright: ignore[reportUnknownVariableType]
            except TypeError:
                return orig(file, mode, *args, **kwargs)
            # Write modes are kept to the tempdir. Checked before the blocked
            # -path test so a write to one of our paths gets the
            # write-specific notice, which is the more informative one.
            if is_write_mode(mode) and not rules.is_writable(path_str):
                rules.deny_write()
                return proxy(mode)
            if rules.is_blocked(path_str):
                rules.deny_read()
                return proxy(mode)
            if not rules.is_readable(path_str):
                rules.deny_outside()
                return proxy(mode, _OUTSIDE_DENIAL)
            return orig(file, mode, *args, **kwargs)

        _patched_open._sandbox_patched = True  # type: ignore[attr-defined]
        return _patched_open

    if not getattr(_builtins.open, "_sandbox_patched", False):
        _builtins.open = make_patched_open(_builtins.open)
    if not getattr(_io.open, "_sandbox_patched", False):
        _io.open = make_patched_open(_io.open)


def _patch_listing(rules: _PathRules) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
    """``os.listdir`` and ``os.scandir`` (under ``pathlib``, ``glob`` and
    ``os.walk``), refused outside the readable roots: listing the directories
    around the working directory names the other games' and the job's own.
    A refusal raises ``PermissionError`` after the notice, which is what a
    walk or a glob already skips over. The import system lists directories
    through ``posix`` directly, so imports are unaffected."""
    import errno

    def make_patched(orig: Any) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        def _patched(path: Any = ".") -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            if isinstance(path, int) or rules.is_listable(path):
                return orig(path)
            rules.deny_outside()
            raise PermissionError(errno.EACCES, "Permission denied", os.fsdecode(path))

        _patched._sandbox_guarded = True  # type: ignore[attr-defined]
        return _patched

    # A marker of its own: the name-order wrapper (_sort_directory_listings)
    # is installed first and carries _sandbox_patched, and this wraps it.
    if not getattr(os.listdir, "_sandbox_guarded", False):
        os.listdir = make_patched(os.listdir)
    if not getattr(os.scandir, "_sandbox_guarded", False):
        os.scandir = make_patched(os.scandir)


#: Raised when the model's own code touches the process environment. States
#: the rule rather than the mechanism, as the other notices do.
_ENVIRONMENT_REFUSAL = (
    "Environment variables are not available to your code: what it does may depend only on your "
    "files and `history`. Keep a setting in a file instead."
)


def _guard_environment(rules: _PathRules) -> None:
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Refuse the process environment to the model's own code — its cells and
    the modules in its working directory, ``world_model.py`` among them — and
    leave it to everything else.

    The environment is state outside the model's files that a cell can set
    and its code can read, and the processes its code runs in do not share
    it: a world model that read a variable a cell had set passed here and
    failed in the agent's process, which ended the game. Libraries and the
    harness read it as before (numpy's thread counts, ``tempfile``'s
    directory), so the check is on who is asking: the first caller outside
    this guard, ``os`` and the mapping machinery ``os`` delegates to — the
    path ``os.getenv``, ``.get``, ``in`` and ``.update`` all take. Reads are
    refused as well as writes, since the kernel's environment is not the
    agent's to begin with.
    """
    import _collections_abc

    # Frozen at startup, these two report a pseudo-file rather than their path.
    machinery = {__file__, os.__file__, _collections_abc.__file__, "<frozen os>", "<frozen _collections_abc>"}

    def model_is_asking() -> bool:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        frame = sys._getframe(1)  # pyright: ignore[reportPrivateUsage]
        while frame is not None and frame.f_code.co_filename in machinery:
            frame = frame.f_back
        if frame is None:
            return False
        filename = frame.f_code.co_filename
        if filename.startswith("<repl"):
            return True
        # Any other pseudo-file (``<frozen posixpath>``) is the library's, and
        # would otherwise resolve against the working directory.
        if not os.path.isabs(filename):
            return False
        resolved = rules.resolve(filename)
        return resolved is not None and os.path.dirname(resolved) == rules.tempdir

    def refuse_the_model() -> None:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        if model_is_asking():
            raise PermissionError(_ENVIRONMENT_REFUSAL)

    class _GuardedEnviron(os._Environ):  # type: ignore[type-arg]  # pyright: ignore[reportMissingTypeArgument]
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """``os.environ`` and ``os.environb``, answering everyone but the model."""

        _sandbox_patched = True

        def __getitem__(self, key: Any) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            return cast(Any, super().__getitem__(key))

        def __setitem__(self, key: Any, value: Any) -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            super().__setitem__(key, value)

        def __delitem__(self, key: Any) -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            super().__delitem__(key)

        def __iter__(self) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            return cast(Any, super().__iter__())

        def __len__(self) -> int:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            return super().__len__()

    def make_guarded(orig: Any) -> Any:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        def _guarded(*args: Any) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            refuse_the_model()
            return orig(*args)

        _guarded._sandbox_patched = True  # type: ignore[attr-defined]
        return _guarded

    # The class of the existing objects rather than new objects: a library
    # that took a reference to os.environ at import holds the same one.
    for store in (os.environ, getattr(os, "environb", None)):
        if store is not None and not getattr(store, "_sandbox_patched", False):
            store.__class__ = _GuardedEnviron
    # putenv/unsetenv change the process's environment without the mapping.
    if not getattr(os.putenv, "_sandbox_patched", False):
        os.putenv = make_guarded(os.putenv)
    if not getattr(os.unsetenv, "_sandbox_patched", False):
        os.unsetenv = make_guarded(os.unsetenv)


def _block_source_access() -> None:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Block reads of our package sources — and the game engine's and the
    games' — via inspect, linecache and open; keep reads and listings to the
    tempdir, the kernel's HOME, Python and its libraries and the system's
    directories; confine writes to the REPL tempdir; and keep the process
    environment from the model's own code. The tempdir itself, files the LLM
    creates, the standard library and numpy stay readable.

    Every trigger prints its notice to stdout so the LLM gets the same
    explanation regardless of which entry point it tried, and gets back a
    short sentinel that is safe under ``.splitlines()`` / substring search /
    iteration: there is no source content left for a follow-up operation to
    uncover.

    Idempotent, via a ``_sandbox_patched`` marker on each replacement.

    **Accident prevention, not a sandbox.** Only these entry points are
    patched: ``os.open`` is not, integer file descriptors are waved through
    by design, and ``shutil`` / ``subprocess`` are untouched — so a
    determined write outside the tempdir still lands. What it buys is that a
    model reaching for our source reads an explanation instead, and that a
    model stashing a file under a literal ``/tmp`` path does not collide
    with the REPL next door. ``_SOURCE_TRIPWIRE`` in ``GameREPL._execute``
    catches read bypasses as defence in depth.
    """
    rules = _PathRules()
    _patch_inspect(rules)
    _patch_linecache(rules)
    _patch_open(rules)
    _patch_listing(rules)
    _guard_environment(rules)


def _confine_filesystem() -> None:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Have the operating system hold the kernel to the places
    :func:`_block_source_access` patches Python to keep it to, for every way
    a file can be reached (``seal.landlock``): reads in the system's
    directories, Python's installation and its libraries, the working
    directory and the kernel's HOME; writes in the last two only. Our
    packages, the game engine and the games are unreadable even where they
    are installed beside the libraries.

    Called once everything of ours the kernel runs is imported: what is
    imported later cannot be read. Temporary files go under HOME, since the
    system's temporary directory is shared with every other process on the
    machine and is not writable here.
    """
    import tempfile

    from seal import landlock

    rules = _PathRules()
    scratch = os.path.join(rules.home, "tmp")
    os.makedirs(scratch, exist_ok=True)
    os.environ["TMPDIR"] = scratch
    tempfile.tempdir = None
    landlock.restrict(
        readable=[*rules.SYSTEM_ROOTS, sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix],
        writable=[rules.tempdir, rules.home, os.devnull],
        hidden=rules.blocked_dirs,
        # Listable, as the guard leaves them: ``importlib.metadata`` lists
        # every ``sys.path`` entry, our source directories among them, and
        # one it cannot list it opens as a zip file instead -- a read of our
        # source, which the guard answers with its notice, in the output of a
        # cell that only imported a library.
        listable=[entry for entry in sys.path if entry],
    )


def _configure_model_process() -> None:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl) and the search child
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Set up a process the model's code runs in: the kernel and the search
    child alike, so its code behaves the same in both. numpy's 1.x scalar
    repr (``5``, not ``np.int8(5)``: a key the model builds from one must
    not change between them), and directory listings in name order."""
    np.set_printoptions(legacy="1.25")
    _sort_directory_listings()


def _model_process_settings() -> AbstractContextManager[None]:  # pyright: ignore[reportUnusedFunction]  # called from seal.search
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """:func:`_configure_model_process` for the span of a ``with``, in a
    process the harness also runs in — the agent's, for the search's second
    test — and undone after."""
    import contextlib
    import os as _os

    @contextlib.contextmanager
    def _ctx() -> Generator[None, None, None]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        listdir, scandir = _os.listdir, _os.scandir
        with np.printoptions(legacy="1.25"):
            _sort_directory_listings()
            try:
                yield
            finally:
                _os.listdir, _os.scandir = listdir, scandir

    return _ctx()


def _sort_directory_listings() -> None:  # pyright: ignore[reportUnusedFunction]  # called from the REPL kernel's initial_code (seal.repl)
    # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
    """Make every directory listing the LLM can reach come back in name order.

    ``os.listdir`` and ``os.scandir`` return raw readdir order, which on
    ext4 with ``dir_index`` is a filename hash carrying a per-filesystem
    seed. The same tmpdir, built by the same code, therefore lists in a
    different order on a different machine — and the LLM does call
    ``os.listdir()`` in the REPL, so that order reaches it and a recorded
    run cannot be replayed elsewhere.

    Both are patched because they are separate implementations, and
    ``glob``, ``os.walk`` and ``pathlib.Path.iterdir`` all route through
    ``scandir``. Idempotent via ``_sandbox_patched``, like
    :func:`_block_source_access`.
    """
    import os as _os

    _orig_listdir = _os.listdir
    _orig_scandir = _os.scandir

    def _listdir(path: Any = None) -> list[Any]:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        return sorted(_orig_listdir(path))  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportCallIssue]

    _listdir._sandbox_patched = True  # type: ignore[attr-defined]

    class _SortedScandir:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        """Materialised scandir result in name order.

        Still a context manager and an iterator: ``os.walk`` and ``glob``
        both use ``with scandir(...) as it``, so returning a bare ``iter()``
        here would break them.
        """

        def __init__(self, entries: list[Any]) -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            self._it = iter(entries)

        def __iter__(self) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            return self._it

        def __next__(self) -> Any:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            return next(self._it)

        def __enter__(self) -> "_SortedScandir":
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            return self

        def __exit__(self, *exc: Any) -> bool:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            return False

        def close(self) -> None:
            # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
            pass

    def _scandir(path: Any = None) -> _SortedScandir:
        # __SANDBOX_SRC_TRIPWIRE_a4c1f70d__
        # Drain under the original's context manager so its directory fd is
        # released here rather than at some later GC.
        with _orig_scandir(path) as entries:  # pyright: ignore[reportUnknownArgumentType, reportCallIssue]
            return _SortedScandir(sorted(entries, key=lambda e: e.name))

    _scandir._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_os.listdir, "_sandbox_patched", False):
        _os.listdir = _listdir  # pyright: ignore[reportAttributeAccessIssue]
    if not getattr(_os.scandir, "_sandbox_patched", False):
        _os.scandir = _scandir  # pyright: ignore[reportAttributeAccessIssue]
