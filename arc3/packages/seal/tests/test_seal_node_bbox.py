"""A node's box: `.bbox` as half-open `(row0, row1, col0, col1)`, and its repr printing the same slices.

Models build a node's box by hand from its pixels or outline in two games of
three, and they learn a node from the prompt and its repr, not from help().
The attribute, the repr and a slice share one notation, so none of them can
read one off against another.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import numpy.typing as npt

from seal.harness import game

#: The kernel's ``segment``, typed: its argument is left untyped where the model reads its signature.
segment = cast("Callable[[npt.ArrayLike], game.Segmentation]", game.segment)


def test_the_box_slices_exactly_the_node_and_the_repr_prints_it() -> None:
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[2:5, 3] = 4
    grid[4, 3:6] = 4  # an L over rows 2-4, cols 3-5
    (node,) = segment(grid).by_color(4)

    assert node.bbox == (2, 5, 3, 6)
    row0, row1, col0, col1 = node.bbox
    box = grid[row0:row1, col0:col1]
    assert box.shape == (3, 3) and int((box == 4).sum()) == node.pixel_count
    assert "rows 2:5, cols 3:6" in repr(node)
    assert "# explore: .bbox" in repr(node)


def test_a_one_cell_node_has_a_one_cell_box() -> None:
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[3, 0] = 7
    (node,) = segment(grid).by_color(7)
    assert node.bbox == (3, 4, 0, 1)
