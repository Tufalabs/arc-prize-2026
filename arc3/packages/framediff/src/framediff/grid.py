"""Frames as framediff handles them, and their ascii-grid notation.

A frame is a 64x64 grid of colour indices 0-15. The notation writes one
character per cell from an alphabet of sixteen, ``0``-``9`` and ``a``-``f``
unless the process sets another (:func:`set_alphabet`): the board the seal's
``ascii_grid()`` prints, so a paste the model reads is in the form it already
reads frames in, and ``from_ascii_grid`` reads back what ``ascii_grid()``
printed in any of its layouts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Side of every frame. The engine's camera always renders this size.
GRID_SIZE = 64

#: Number of colours; a cell holds an index below it.
NUM_COLORS = 16

#: One character per colour, by index: the default alphabet.
HEX_CHARS = "0123456789abcdef"

Grid = npt.NDArray[np.int8]

#: The alphabet this process writes and reads boards in. A process setting
#: rather than an argument, since the code blocks framediff writes call
#: ``from_ascii_grid`` with nothing but the text: the process a block runs in
#: has to read it in the alphabet it was written in.
_alphabet = HEX_CHARS


def alphabet() -> str:
    """The alphabet this process writes and reads boards in, one character per colour by index."""
    return _alphabet


def set_alphabet(chars: str) -> None:
    """Write and read boards in ``chars`` from now on in this process: sixteen
    distinct characters, one per colour by index, none of them whitespace."""
    global _alphabet
    if len(chars) != NUM_COLORS or len(set(chars)) != NUM_COLORS or any(char.isspace() for char in chars):
        raise ValueError(f"an alphabet is {NUM_COLORS} distinct non-space characters, got {chars!r}")
    _alphabet = chars


def to_ascii_grid(grid: npt.ArrayLike) -> str:
    """``grid`` in ascii-grid notation, one line per row, nothing else on a line."""
    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"an ascii grid is two-dimensional, got shape {array.shape}")
    if array.size and (array.min() < 0 or array.max() >= NUM_COLORS):
        raise ValueError(f"an ascii grid holds colours 0-{NUM_COLORS - 1}, got {array.min()}..{array.max()}")
    lut = np.array(list(_alphabet))
    return "\n".join("".join(row) for row in lut[array.astype(np.int16)])


def from_ascii_grid(text: str) -> Grid:
    """The array an ascii grid shows, as int8 colour indices.

    One character per cell, from this process's alphabet. Blank lines, spaces and
    tabs are ignored, and so is a row-number column: when every line starts
    with a number and a tab, written without leading zeros, and the numbers
    count up by one, that column is the row index ``ascii_grid()`` prints,
    not part of the grid. Errors name the row and column of what they
    refuse.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("from_ascii_grid: the text holds no rows")
    fields = [line.strip().split("\t") for line in lines]
    if _numbered(fields):
        fields = [row[1:] for row in fields]
    rows = ["".join("".join(row).split()) for row in fields]
    for i, row in enumerate(rows):
        if len(row) != len(rows[0]):
            raise ValueError(f"from_ascii_grid: row {i + 1} has {len(row)} cells, row 1 has {len(rows[0])}")
        for j, char in enumerate(row):
            if char not in _alphabet:
                raise ValueError(
                    f"from_ascii_grid: row {i + 1}, column {j + 1} is {char!r}; a cell is one of {_alphabet!r}"
                )
    return np.array([[_alphabet.index(char) for char in row] for row in rows], dtype=np.int8)


def _numbered(fields: list[list[str]]) -> bool:
    """Whether every line's first tab-separated field is a row index, counting up by one from the first."""
    if any(len(row) < 2 or not row[0].isdecimal() or row[0] != str(int(row[0])) for row in fields):
        return False
    first = int(fields[0][0])
    return first < GRID_SIZE and all(int(row[0]) == first + i for i, row in enumerate(fields))
