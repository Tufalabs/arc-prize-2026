"""The palette table and the shipped legend describe the same 16 colours.

``grid_utils`` is a verbatim upstream copy carrying the char legend as
pre-formatted prose, so the names table in ``palette`` is a second copy of
the same fact. This is what keeps them from drifting: a name edited in one
place and not the other fails here rather than reaching a prompt.
"""

from __future__ import annotations

from duck.grid_utils import ARC_COLOR_CHARS, ARC_COLOR_LEGEND
from duck.palette import ARC_COLOR_NAMES, ARC_HEX_CHARS, char_legend, numeric_legend


def test_the_names_table_reproduces_the_shipped_legend() -> None:
    """Byte-for-byte: the char arms must not move when numeric mode lands."""
    assert char_legend() == ARC_COLOR_LEGEND


def test_both_boards_cover_the_same_sixteen_indices() -> None:
    """A glyph board shorter than the palette would silently fold colours
    together at the end of the range."""
    assert sorted(ARC_COLOR_NAMES) == list(range(16))
    assert len(ARC_COLOR_CHARS) == len(ARC_HEX_CHARS) == 16
    assert len(set(ARC_HEX_CHARS)) == 16


def test_the_numeric_legend_states_the_mapping_where_it_breaks() -> None:
    """Glyph and index coincide below 10 and diverge above it, which is the
    one thing a model reading `.ascii` against `.array` has to know."""
    legend = numeric_legend()
    assert '9="9" blue' in legend
    assert '10="a" sky blue' in legend
    assert '15="f" purple' in legend
