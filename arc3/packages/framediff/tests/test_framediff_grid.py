"""The ascii-grid alphabet: what a board prints in, and reads back in."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from framediff import grid

LETTERS = "WwgGcBMPRbSYOrNp"


@pytest.fixture(autouse=True)
def _restore_alphabet() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse
    before = grid.alphabet()
    yield
    grid.set_alphabet(before)


def test_a_board_round_trips_in_the_alphabet_set() -> None:
    board = np.arange(16, dtype=np.int8).reshape(2, 8)
    assert grid.to_ascii_grid(board) == "01234567\n89abcdef"
    grid.set_alphabet(LETTERS)
    assert grid.to_ascii_grid(board) == "WwgGcBMP\nRbSYOrNp"
    assert np.array_equal(grid.from_ascii_grid("WwgGcBMP\nRbSYOrNp"), board)


def test_a_board_in_another_alphabet_is_refused_by_name() -> None:
    """``b`` is 11 in hex and 9 in letters: a board is read in the one
    alphabet set, never guessed at."""
    grid.set_alphabet(LETTERS)
    with pytest.raises(ValueError, match="is '0'"):
        grid.from_ascii_grid("0b")


@pytest.mark.parametrize("chars", ["0123", "0123456789abcdee", "0123456789abcde "])
def test_an_alphabet_is_sixteen_distinct_visible_characters(chars: str) -> None:
    with pytest.raises(ValueError, match="16 distinct"):
        grid.set_alphabet(chars)


def test_a_paste_block_writes_and_reads_its_picture_in_the_alphabet_set() -> None:
    """The one block that carries a picture: in letters, it is written in
    letters and its ``from_ascii_grid`` rebuilds the frame it describes."""
    from framediff import AsciiGrid, block_namespace

    grid.set_alphabet(LETTERS)
    start = np.zeros((64, 64), dtype=np.int8)
    paste = AsciiGrid.from_array(40, 20, np.array([[0, 9, 1], [15, 2, 14]]))
    assert paste.rows == ("Wbw", "pgN")
    expected = paste.transform(start)
    namespace = block_namespace(a=start.copy(), b=expected)
    exec(paste.to_code(), namespace)
    assert np.array_equal(namespace["g"], expected)
