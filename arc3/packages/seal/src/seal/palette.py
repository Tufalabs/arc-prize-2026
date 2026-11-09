"""The ARC palette as the model is shown it: indices, glyphs, and names.

A colour IS its index here. The hex board (``0123456789abcdef``) exists only
because a text board needs single-width cells, so a glyph in ``ascii_grid()``
and a value in ``.grid`` name the same colour for 0-9 and differ only past 9 —
which is why the legend states the mapping for all sixteen rather than
leaving the break to be inferred. The duck's letters (``WwgGcBMPRbSYOrNp``)
are the other alphabet a board can print in (``SealAgent.ascii_symbols``);
``.grid`` holds indices either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: What a board prints. Digits where the glyph and the index coincide,
#: letters past 9 — the convention every hex dump uses.
ARC_HEX_CHARS = "0123456789abcdef"

#: The duck's board letters, one per colour by index: an initial of the colour
#: name, its case telling apart two colours that share one.
ARC_LETTER_CHARS = "WwgGcBMPRbSYOrNp"

#: Which alphabet a board prints in: ``SealAgent.ascii_symbols``.
AsciiSymbols = Literal["hex", "letters"]

#: Each alphabet by its name.
ASCII_SYMBOLS: dict[str, str] = {"hex": ARC_HEX_CHARS, "letters": ARC_LETTER_CHARS}


@dataclass
class AsciiLayout:
    """How ``ascii_grid`` lays a board out and which characters it prints it
    in: what every process the model's code runs in -- the kernel, the
    search's own and the agent's for its test -- is configured with, so a
    board the model wrote reads back the same in each."""

    row_numbers: bool = True
    column_group: int | None = None
    symbols: AsciiSymbols = "hex"


#: Colour index to name. The seal's own names, which differ from the whale's
#: at the light end.
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


def numeric_legend(chars: str = ARC_HEX_CHARS) -> str:
    """The legend: index, the glyph ``ascii_grid()`` prints in ``chars``, and name.

    Stated for all sixteen including where index and glyph coincide. The
    redundancy is the point — a legend that only covered 10-15 would leave
    the model to work out where the correspondence stops.
    """
    return ", ".join(f'{index}="{chars[index]}" {name}' for index, name in sorted(ARC_COLOR_NAMES.items()))
