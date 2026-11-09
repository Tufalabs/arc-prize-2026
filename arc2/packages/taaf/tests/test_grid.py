"""Tests for taaf.grid — the value type everything else is built from, and
the only place grid shape and palette are enforced."""

from __future__ import annotations

import numpy as np
import pytest

import taaf.grid
from taaf.grid import Grid


def test_grid_rejects_out_of_palette_and_oversized():
    """Validation happens at construction, because a malformed grid reaching
    the scorer would compare unequal and read as a wrong answer, not a bug."""
    with pytest.raises(ValueError, match="0..9"):
        Grid(np.array([[10]], dtype=np.int8))
    with pytest.raises(ValueError, match="1..30"):
        Grid(np.zeros((31, 1), dtype=np.int8))
    with pytest.raises(ValueError, match="1..30"):
        Grid(np.zeros((0, 3), dtype=np.int8))
    with pytest.raises(ValueError, match="2-D"):
        Grid(np.zeros((2, 2, 2), dtype=np.int8))


def test_grid_copies_its_buffer():
    """A grid must not change under whoever holds it."""
    buf = np.zeros((2, 2), dtype=np.int8)
    grid = Grid(buf)
    buf[0, 0] = 5
    assert grid.data[0, 0] == 0


def test_grid_equality_is_by_value_and_hashable():
    """Scoring is grid equality, so it has to compare contents — and the
    dataclass-generated __eq__ would return an array here."""
    a = Grid(np.array([[1, 2]], dtype=np.int8))
    assert a == Grid(np.array([[1, 2]], dtype=np.int8))
    assert a != Grid(np.array([[2, 1]], dtype=np.int8))
    assert a != Grid(np.array([[1], [2]], dtype=np.int8))  # same cells, different shape
    assert len({a, Grid(np.array([[1, 2]], dtype=np.int8))}) == 1


def test_grid_list_round_trip():
    """JSON and submissions speak nested lists; the round-trip must be exact."""
    rows = [[0, 9], [3, 4], [5, 6]]
    assert Grid.from_lists(rows).to_lists() == rows


def test_from_lists_reports_bad_values_as_value_errors():
    """Submission validation catches ``ValueError``: a value outside the
    palette must arrive as one, not as int8 overflow or a ragged-array
    error of another type."""
    with pytest.raises(ValueError, match="0..9"):
        Grid.from_lists([[300]])
    with pytest.raises(ValueError):
        Grid.from_lists([[1, 2], [3]])


def test_palette_is_the_arc_ten():
    """The palette is ARC2's, not a slice of ARC3's sixteen — a mismatch
    would silently recolour every rendered answer sheet."""
    assert taaf.grid.N_COLORS == 10
    assert taaf.grid.ARC_COLORS[0] == "#000000"
    assert taaf.grid.ARC_COLORS[9] == "#870C25"
