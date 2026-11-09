"""``taaf.support.CompressedList``: the compressed per-step record."""

import pickle

import numpy as np

from taaf.support import CompressedList


def test_items_come_back_equal_and_fresh():
    """An item is stored as it was at ``append``: later changes to the
    appended object do not reach the record, and each read is a new object."""
    record: CompressedList[list[int]] = CompressedList()
    item = [1, 2, 3]
    record.append(item)
    item.append(4)
    assert record[0] == [1, 2, 3]
    assert record[0] is not record[0]
    assert record[:1] == [[1, 2, 3]]


def test_keep_last_serves_the_appended_object():
    """With ``keep_last`` the newest item is the appended object itself, the
    rest are decoded; the tail is dropped on pickling, leaving the record."""
    record: CompressedList[dict[str, int]] = CompressedList(keep_last=True)
    first, second = {"n": 1}, {"n": 2}
    record.append(first)
    record.append(second)
    assert record[-1] is second and record[1] is second
    assert record[0] == first and record[0] is not first
    restored = pickle.loads(pickle.dumps(record))
    assert restored[-1] == second and restored[-1] is not second
    assert restored == record


def test_equality_compares_the_items_a_reader_sees():
    """Two records are equal when their items are: equal dicts built in a
    different order pickle to different bytes, and a live tail changed after
    ``append`` is what a reader sees, not what was compressed."""
    assert CompressedList([{"x": 1, "y": 2}]) == CompressedList([{"y": 2, "x": 1}])
    tail = {"n": 1}
    live: CompressedList[dict[str, int]] = CompressedList(keep_last=True)
    live.append(tail)
    tail["n"] = 2
    assert live != CompressedList([{"n": 1}])
    assert live == CompressedList([{"n": 2}])


def test_items_stored_alike_compare_without_their_own_equality():
    """Identical stored bytes count as the same item, as identity does in a
    list, so a record of arrays, whose own ``==`` gives no single answer,
    still compares equal to itself and to a copy of itself."""
    record = CompressedList([np.zeros((2, 2)), np.ones((2, 2))])
    assert record == record
    assert record == pickle.loads(pickle.dumps(record))
