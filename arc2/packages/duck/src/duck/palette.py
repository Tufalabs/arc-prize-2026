"""The colour legend the model is shown.

A board of swatches, one per cell value, each labelled with the integer the
kernel holds and the colour's name. Without it the model sees pictures whose
colours it cannot name and arrays whose numbers it cannot picture, and has
to infer the correspondence from the images themselves — which it can only
do for colours the task happens to use.

Ported from the whale's `palette_legend.py`. Two differences, both because
ARC-AGI-2 has ten colours rather than sixteen: the board is 5x2 rather than
4x4, and there is no symbol row — `ascii_grid` renders every value as its
own digit here, so the integer *is* the symbol and a second label would say
the same thing twice.

RGB comes from `taaf.grid.ARC_PALETTE` and so do the names, so this module
hardcodes neither.

**Every pixel is part of the interface.** Replay references pin an image by
its exact bytes, so a changed coordinate, draw order or font produces a
different hash and fails every replay. Read the drawing below as pinned
text rather than as implementation — and see `duck.fonts` for why the font
is bundled rather than chosen.

Run as a script to write a PNG next to this module and look at it:

    python -m duck.palette
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

import taaf.grid
from duck import fonts

# Half the whale's cell size, and deliberately NOT tied to ``image_scale``.
# The legend is a fixed reference the model reads once, not part of the
# puzzle, so it does not move with the sweep variable — but at the whale's
# 128px cells it rendered 640x256, which is 47x the pixel area of the
# largest task image once those dropped to scale 4, and 89% of the prompt's
# whole image budget. An image with no pattern in it should not be the most
# prominent thing in a prompt about reading patterns.
_COLS = 5
_ROWS = 2
_CELL_W = 64
_CELL_H = 64
_NAME_BAND_H = 14
_COLOR_AREA_H = _CELL_H - _NAME_BAND_H

_BG = (30, 30, 30)
_BAND_BG = (20, 20, 20)
_BAND_FG = (220, 220, 220)

_INDEX_FONT_SIZE = 22
_NAME_FONT_SIZE = 8


def _rgb(hex_code: str) -> tuple[int, int, int]:
    """One ``#rrggbb`` as the byte triple PIL fills with."""
    return (int(hex_code[1:3], 16), int(hex_code[3:5], 16), int(hex_code[5:7], 16))


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Rec. 709 luma of a swatch, used only to pick a legible label colour."""
    red, green, blue = rgb
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _label_color(swatch: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light swatch, white on a dark one. Computed rather than
    tabulated so a palette change cannot leave a label unreadable."""
    return (0, 0, 0) if _relative_luminance(swatch) > 140 else (255, 255, 255)


def render_palette_legend() -> Image.Image:
    """The legend, as a PIL image."""
    width, height = _COLS * _CELL_W, _ROWS * _CELL_H
    image = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(image)
    index_font = fonts.load(_INDEX_FONT_SIZE, bold=True)
    name_font = fonts.load(_NAME_FONT_SIZE)

    for value, (hex_code, name) in enumerate(taaf.grid.ARC_PALETTE):
        row, col = divmod(value, _COLS)
        left, top = col * _CELL_W, row * _CELL_H
        swatch = _rgb(hex_code)
        draw.rectangle([left, top, left + _CELL_W - 1, top + _COLOR_AREA_H - 1], fill=swatch)
        draw.rectangle(
            [left, top + _COLOR_AREA_H, left + _CELL_W - 1, top + _CELL_H - 1],
            fill=_BAND_BG,
        )
        draw.text((left + 5, top + 3), str(value), font=index_font, fill=_label_color(swatch))
        draw.text(
            (left + _CELL_W // 2, top + _COLOR_AREA_H + _NAME_BAND_H // 2),
            name,
            font=name_font,
            fill=_BAND_FG,
            anchor="mm",
        )
    return image


def legend_array() -> npt.NDArray[np.uint8]:
    """The legend as the RGB array ``ullm.conv.ImageContent`` carries."""
    return np.asarray(render_palette_legend(), dtype=np.uint8)


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".png")
    render_palette_legend().save(out)
    print(f"wrote {out}")


__all__ = ["legend_array", "render_palette_legend"]
