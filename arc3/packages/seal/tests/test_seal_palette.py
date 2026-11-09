"""The palette table and the legend the model reads describe the same 16 colours.

The legend is the only place the seal ever spells a colour out: boards carry
indices, `ascii_grid()` renders them as hex, and nothing else translates. So the
table behind it is model-visible in full, and a hole in it — a missing index,
a repeated glyph — reaches a prompt rather than raising anywhere.
"""

from __future__ import annotations

from seal.palette import ARC_COLOR_NAMES, ARC_HEX_CHARS, numeric_legend


def test_the_glyph_board_covers_the_same_sixteen_indices() -> None:
    """A glyph board shorter than the palette would silently fold colours
    together at the end of the range."""
    assert sorted(ARC_COLOR_NAMES) == list(range(16))
    assert len(ARC_HEX_CHARS) == 16
    assert len(set(ARC_HEX_CHARS)) == 16


def test_the_numeric_legend_states_the_mapping_where_it_breaks() -> None:
    """Glyph and index coincide below 10 and diverge above it, which is the
    one thing a model reading `ascii_grid()` against `.grid` has to know."""
    legend = numeric_legend()
    assert '9="9" blue' in legend
    assert '10="a" sky blue' in legend
    assert '15="f" purple' in legend
