"""Objects matched across two frames, proposed to the search as masked moves (``Decomposer.object_moves``).

A sprite that crosses a patterned floor cannot be copied with its box, which carries the floor along; a
masked move carries its own cells. What is pinned: the masked move's block in each form, which pieces are
matched (moved, turned, several colours as one object) and which are not proposed (a box the colour cannot
isolate), the search taking the object where it is the cheapest reading and keeping today's where it is not,
and the run's guarantee that a script is never costlier with the objects than without.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from framediff import GRID_SIZE, BlockCopy, Decomposer, Grid, MaskedMove, Source
from framediff.decompose import OPTIONS_OFF
from framediff.objects import proposals

#: The search with its options off, and with the objects alone on: what the objects change, apart from the others.
PLAIN = Decomposer(**OPTIONS_OFF)
OBJECTS = dataclasses.replace(PLAIN, object_moves=True)


def _floor(pattern: str) -> Grid:
    """A wall of 5 around a striped or plain floor at rows 20:30, cols 5:40."""
    grid = np.full((GRID_SIZE, GRID_SIZE), 5, dtype=np.int8)
    if pattern == "stripes":
        grid[20:30, 5:40] = np.where(np.arange(5, 40) % 2 == 0, 1, 2)
    else:
        grid[20:30, 5:40] = 1
    return grid


def _put(floor: Grid, shape: np.ndarray, row: int, col: int) -> Grid:
    """``floor`` with the non-zero cells of ``shape`` painted at ``(row, col)``."""
    grid = floor.copy()
    window = grid[row : row + shape.shape[0], col : col + shape.shape[1]]
    window[shape > 0] = shape[shape > 0]
    return grid


BIG_L = np.array([[6, 0, 0], [6, 0, 0], [6, 6, 6]], dtype=np.int8)


def test_a_masked_move_writes_the_object_alone() -> None:
    a = _put(_floor("stripes"), BIG_L, 24, 10)
    move = MaskedMove(25, 28, 11, 14, src_row=24, src_col=10, colors=(6,), clear=1)
    code = move.to_code("a", "b")
    assert "mask = a[24:27, 10:13] == 6  # the colour-6 shape\n" in code
    assert "g[25:28, 11:14][mask] = 6  # moved down 1, right 1" in code
    assert move.bit_cost() == 14
    result = move.transform(a)
    assert int((result == 6).sum()) == 5 and (result[25:28, 11:14][BIG_L > 0] == 6).all()


def test_several_colours_move_their_values_and_one_colour_may_turn() -> None:
    two = MaskedMove(24, 26, 12, 15, src_row=24, src_col=10, colors=(9, 6), clear=1)
    assert two.colors == (6, 9)
    assert "np.isin(a[24:26, 10:13], [6, 9])" in two.to_code("a", "b")
    assert "= a[24:26, 10:13][mask]  # moved right 2" in two.to_code("a", "b")
    turned = MaskedMove(24, 27, 10, 13, src_row=24, src_col=10, colors=(6,), rot=1)
    assert "g[24:27, 10:13][np.rot90(mask, 1)] = 6" in turned.to_code("a", "b")
    assert turned.bit_cost() == 22
    with pytest.raises(ValueError, match="one colour"):
        MaskedMove(24, 27, 10, 13, src_row=24, src_col=10, colors=(6, 9), rot=1)
    with pytest.raises(ValueError, match="names the colours"):
        MaskedMove(24, 27, 10, 13, src_row=24, src_col=10, colors=())


def test_matches_propose_the_moved_the_turned_and_the_joined() -> None:
    floor = _floor("plain")
    moved = [m for m in proposals(_put(floor, BIG_L, 24, 10), _put(floor, BIG_L, 25, 11)) if isinstance(m, MaskedMove)]
    assert any(m.colors == (6,) and (m.row0, m.col0) == (25, 11) and not m.turned for m in moved)
    turned = proposals(_put(floor, BIG_L, 24, 10), _put(floor, np.rot90(BIG_L).copy(), 24, 10))
    assert any(m.turned for m in turned)
    two = np.array([[6, 9, 0], [6, 6, 9]], dtype=np.int8)
    joined = proposals(_put(floor, two, 24, 10), _put(floor, two, 24, 12))
    assert any(isinstance(m, MaskedMove) and m.colors == (6, 9) for m in joined)


def test_a_wildcard_where_an_object_was_is_no_colour_to_leave() -> None:
    """A target that asks nothing of the cells an object left (wildcards there) has no colour to clear them
    with: the move leaves them as they are, rather than writing a wildcard."""
    floor = _floor("stripes")
    a = _put(floor, BIG_L, 24, 10)
    b = _put(floor, BIG_L, 24, 20)
    b[24:27, 10:13][BIG_L > 0] = -1
    moves = [m for m in proposals(a, b) if (m.src_row, m.src_col, m.row0, m.col0) == (24, 10, 24, 20)]
    assert moves and all(m.clear is None for m in moves)


def test_an_object_that_fills_its_box_is_a_plain_move() -> None:
    """A solid block crossing stripes: the mask would pick every cell of its box, so the box moves, at a plain
    move's cost."""
    floor = _floor("stripes")
    block = np.full((2, 3), 6, dtype=np.int8)
    found = proposals(_put(floor, block, 24, 10), _put(floor, block, 25, 13))
    (move,) = [m for m in found if (m.src_row, m.src_col, m.row0, m.col0) == (24, 10, 25, 13)]
    assert isinstance(move, BlockCopy) and move.block_shape == (2, 3)


def test_a_box_the_colour_cannot_isolate_is_not_proposed() -> None:
    """A second piece of the same colour inside the object's box would be moved with it: no proposal."""
    floor = _floor("plain")
    a = _put(floor, BIG_L, 24, 10)
    a[24, 12] = 6  # inside the L's box, not touching it
    b = _put(floor, BIG_L, 25, 11)
    b[24, 12] = 6
    assert not any(m.src_row == 24 and m.src_col == 10 for m in proposals(a, b))


@pytest.mark.parametrize(("rows", "cols"), [(1, 1), (0, 1)])
def test_an_object_crossing_stripes_is_one_masked_move(rows: int, cols: int) -> None:
    """Off the stripes' period the box cannot be copied with its floor; the object moves alone."""
    a = _put(_floor("stripes"), BIG_L, 24, 10)
    b = _put(_floor("stripes"), BIG_L, 24 + rows, 10 + cols)
    without = PLAIN.explain([Source("a", a)], b, "b")
    found = OBJECTS.explain([Source("a", a)], b, "b")
    assert found.cost is not None and without.cost is not None and found.cost < without.cost
    assert "mask = a[24:27, 10:13] == 6" in (found.code or "")


def test_a_floor_that_repeats_at_the_shift_keeps_its_copy() -> None:
    """Shifted by the stripes' period, a copy of the box is exact and cheaper than naming the object."""
    a = _put(_floor("stripes"), BIG_L, 24, 10)
    b = _put(_floor("stripes"), BIG_L, 24, 12)
    found = OBJECTS.explain([Source("a", a)], b, "b")
    assert found.cost == 10 and "mask" not in (found.code or "")


def test_with_objects_a_script_is_never_costlier() -> None:
    floors = [_floor("stripes"), _floor("plain")]
    frames = [_put(floor, BIG_L, row, col) for floor in floors for row, col in ((24, 10), (25, 11), (24, 13))]
    rng = np.random.default_rng(5)
    frames.append(rng.integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8))
    for a, b in itertools.permutations(frames, 2):
        with_objects = OBJECTS.decompose(a, b)
        assert with_objects.bit_cost() <= PLAIN.decompose(a, b).bit_cost()
        assert np.array_equal(with_objects.transform(a), b)
