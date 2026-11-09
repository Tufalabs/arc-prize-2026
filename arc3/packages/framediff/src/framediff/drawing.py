"""The line the ARC games draw between two cells.

The engine has no line primitive, so a game that draws a line hand-rolls
one; r11l's is the integer Bresenham below. ``Line`` draws with this
function, and so does the REPL's ``bresenham_line``, which wraps it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def bresenham_line(grid: npt.NDArray[np.int8], start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    """Set the cells of the straight line from ``start`` to ``end`` to ``color``, in place.

    ``start`` and ``end`` are (row, col) cells inside ``grid`` and are both
    drawn. The line is one cell wide and 8-connected: it covers
    ``max(|rows apart|, |cols apart|) + 1`` cells, one per row or per
    column, whichever it spans more of. Ties are broken the way r11l breaks
    them, so a line and its reverse can differ by a cell on some slopes.
    """
    # The column steps in the first branch and the row in the second, as in
    # r11l; swapping them moves the cells a tie picks.
    (r0, c0), (r1, c1) = start, end
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dc - dr
    while True:
        grid[r0, c0] = color
        if (r0, c0) == (r1, c1):
            return
        e2 = 2 * err
        if e2 > -dr:
            err, c0 = err - dr, c0 + sc
        if e2 < dc:
            err, r0 = err + dc, r0 + sr
