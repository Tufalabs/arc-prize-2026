"""The ARC palette as the model is shown it: indices, glyphs, and names.

Two renderings of one 16-colour palette. The character board
(``WwgGcBMPRbSYOrNp``) is what a char-mode frame prints; the hex board
(``0123456789abcdef``) is what a numeric-mode frame prints, so that a glyph
in ``.ascii`` and a value in ``.array`` name the same colour for 0-9 and
differ only past 9 — which is why the legend states the mapping for all
sixteen rather than leaving the break to be inferred.

The names live here as a table because three model-visible things need them:
the char legend, the numeric legend, and the palette image. ``grid_utils`` is
a verbatim upstream copy and holds the char legend as pre-formatted prose, so
this table cannot be its source — ``tests/test_duck_palette.py`` asserts the
two agree byte-for-byte instead.
"""

from __future__ import annotations

from duck.grid_utils import ARC_COLOR_CHARS

#: What a numeric-mode board prints. Digits where the glyph and the index
#: coincide, letters past 9 — the convention every hex dump uses.
ARC_HEX_CHARS = "0123456789abcdef"

#: Colour index to name. The duck's own names, which differ from the whale's
#: at the light end; the palette image is labelled from here so that it and
#: the legend cannot disagree inside one prompt.
ARC_COLOR_NAMES: dict[int, str] = {
    0: "white",
    1: "light gray",
    2: "gray",
    3: "dark gray",
    4: "charcoal",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "sky blue",
    11: "yellow",
    12: "orange",
    13: "dark red",
    14: "light green",
    15: "purple",
}


def char_legend() -> str:
    """The char-mode legend, as ``grid_utils`` spells it.

    Derived rather than quoted so the coupling check has something to
    compare; the shipped string stays the one in ``grid_utils``.
    """
    return ", ".join(f"{ARC_COLOR_CHARS[index]}={name}" for index, name in sorted(ARC_COLOR_NAMES.items()))


def numeric_legend() -> str:
    """The numeric-mode legend: index, the glyph ``.ascii`` prints, and name.

    Stated for all sixteen including where index and glyph coincide. The
    redundancy is the point — a legend that only covered 10-15 would leave
    the model to work out where the correspondence stops.
    """
    return ", ".join(f'{index}="{ARC_HEX_CHARS[index]}" {name}' for index, name in sorted(ARC_COLOR_NAMES.items()))
