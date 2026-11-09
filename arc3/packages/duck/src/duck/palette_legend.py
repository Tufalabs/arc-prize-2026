"""Render the ARC palette legend image shown to the model.

A 4x4 grid of colour swatches, each labelled with its integer index and the
character `.ascii` renders it as (drawn on the swatch in a contrasting
colour), and its colour name (in a band below). :mod:`duck.agent` attaches
the result to the first user prompt of each level when the frame API is
numeric and an image already travels — the legend explains indices, so it is
worth nothing to an arm that never shows one.

Labels come from :data:`duck.palette.ARC_COLOR_NAMES` and RGB values from
``taaf.game.ARC_COLORS``, so the image and the prompt's own legend cannot
disagree about what colour 3 is called.

**Every pixel here is part of the interface.** The committed references pin
an image by its exact bytes (``platypus.reference`` stores content blocks by
hash), so this module's output is pinned exactly as tightly as a prompt
template is: a changed coordinate, draw order or font produces a different
hash and fails every replay. Read the drawing below as pinned text, not as
implementation — and see :mod:`duck.fonts` for why the font is required
rather than chosen.

Run as a script to write a PNG next to this module and eyeball it:

    python -m duck.palette_legend
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from taaf.game import ARC_COLORS

from duck import fonts
from duck.palette import ARC_COLOR_NAMES, ARC_HEX_CHARS

_COLS = 4
_ROWS = 4
_CELL_W = 128
_CELL_H = 128
_NAME_BAND_H = 28
_COLOR_AREA_H = _CELL_H - _NAME_BAND_H

_BG = (30, 30, 30)
_BAND_BG = (20, 20, 20)
_BAND_FG = (220, 220, 220)

_INDEX_FONT_SIZE = 44
_SYMBOL_FONT_SIZE = 30
_NAME_FONT_SIZE = 15


def _to_rgb_byte(rgb_float: tuple[float, float, float]) -> tuple[int, int, int]:
    """``ARC_COLORS``' 0..1 floats as the 0..255 ints PIL fills with."""
    r, g, b = rgb_float
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _relative_luminance(rgb_byte: tuple[int, int, int]) -> float:
    """Rec. 709 luma of a swatch, used only to pick a legible label colour."""
    r, g, b = rgb_byte
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrasting_text_color(rgb_byte: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on a light swatch, white on a dark one."""
    return (0, 0, 0) if _relative_luminance(rgb_byte) > 140 else (255, 255, 255)


def render_palette_legend() -> Image.Image:
    """The palette legend as a PIL RGB image, 512x512.

    Every literal below — the offsets, the draw order, the bbox-based
    centring — is fenced (see the module docstring). Changing one is a
    change to what the model is shown, not a tidy-up.
    """
    width = _COLS * _CELL_W
    height = _ROWS * _CELL_H
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)

    index_font = fonts.load(_INDEX_FONT_SIZE, bold=True)
    symbol_font = fonts.load(_SYMBOL_FONT_SIZE, bold=True)
    name_font = fonts.load(_NAME_FONT_SIZE)

    for idx in range(16):
        col = idx % _COLS
        row = idx // _COLS
        x0 = col * _CELL_W
        y0 = row * _CELL_H

        rgb = _to_rgb_byte(ARC_COLORS[idx])
        # Color swatch
        draw.rectangle(
            [(x0, y0), (x0 + _CELL_W - 1, y0 + _COLOR_AREA_H - 1)],
            fill=rgb,
        )
        # Index top-left, `.ascii` glyph top-right — the same character for
        # 0-9, the letter a-f above that, which is the one place the two
        # renderings of a colour stop agreeing.
        text_color = _contrasting_text_color(rgb)
        draw.text((x0 + 10, y0 + 6), str(idx), fill=text_color, font=index_font)
        sym = ARC_HEX_CHARS[idx]
        sym_bbox = draw.textbbox((0, 0), sym, font=symbol_font)
        sym_w = sym_bbox[2] - sym_bbox[0]
        draw.text(
            (x0 + _CELL_W - 12 - sym_w, y0 + 14),
            sym,
            fill=text_color,
            font=symbol_font,
        )

        # Name band below swatch
        band_y0 = y0 + _COLOR_AREA_H
        draw.rectangle(
            [(x0, band_y0), (x0 + _CELL_W - 1, y0 + _CELL_H - 1)],
            fill=_BAND_BG,
        )
        name = ARC_COLOR_NAMES.get(idx, "?")
        bbox = draw.textbbox((0, 0), name, font=name_font)
        name_w = bbox[2] - bbox[0]
        name_h = bbox[3] - bbox[1]
        nx = x0 + (_CELL_W - name_w) // 2
        ny = band_y0 + (_NAME_BAND_H - name_h) // 2 - bbox[1]
        draw.text((nx, ny), name, fill=_BAND_FG, font=name_font)

    return img


#: The name-only legend's geometry, at ARC2's scale rather than this module's.
#:
#: The index legend above is 512x512 — sixteen 128px cells. ARC2 met the same
#: problem coming from the whale and wrote down why it shrank: at 128px cells
#: the legend rendered 640x256, "47x the pixel area of the largest task image
#: once those dropped to scale 4, and 89% of the prompt's whole image budget.
#: An image with no pattern in it should not be the most prominent thing in a
#: prompt about reading patterns." That holds here. At 64px cells and two rows
#: this is 512x128 — the same pixel area as a 64x64 board at the duck's
#: ``multimodal_upscale`` of 4, so the reference no longer outweighs the thing
#: being reasoned about.
#:
#: Eight columns rather than four rows keeps it a strip, which also stops it
#: reading as a board.
_NAME_COLS = 8
_NAME_ROWS = 2
_NAME_CELL_W = 64
_NAME_CELL_H = 64
_NAME_ONLY_BAND_H = 14
_NAME_ONLY_COLOR_AREA_H = _NAME_CELL_H - _NAME_ONLY_BAND_H
_NAME_ONLY_FONT_SIZE = 8


def render_color_name_legend() -> Image.Image:
    """The palette as colour NAMES only — no index, no `.ascii` glyph.

    A different question from :func:`render_palette_legend`. That one explains
    what a colour INDEX looks like, which is what a numeric board needs. This
    one ties a swatch to the name the prompt already uses, which is what an
    agent reading an image of a board needs whatever its board rendering is —
    so it is not gated on the frame API.

    **Every pixel here is part of the interface**, on the same terms as the
    legend above: the committed references pin the image by its bytes.
    """
    width = _NAME_COLS * _NAME_CELL_W
    height = _NAME_ROWS * _NAME_CELL_H
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    name_font = fonts.load(_NAME_ONLY_FONT_SIZE)

    for idx in range(16):
        row, col = divmod(idx, _NAME_COLS)
        x0 = col * _NAME_CELL_W
        y0 = row * _NAME_CELL_H

        draw.rectangle(
            [(x0, y0), (x0 + _NAME_CELL_W - 1, y0 + _NAME_ONLY_COLOR_AREA_H - 1)],
            fill=_to_rgb_byte(ARC_COLORS[idx]),
        )
        band_y0 = y0 + _NAME_ONLY_COLOR_AREA_H
        draw.rectangle(
            [(x0, band_y0), (x0 + _NAME_CELL_W - 1, y0 + _NAME_CELL_H - 1)],
            fill=_BAND_BG,
        )
        draw.text(
            (x0 + _NAME_CELL_W // 2, band_y0 + _NAME_ONLY_BAND_H // 2),
            ARC_COLOR_NAMES.get(idx, "?"),
            font=name_font,
            fill=_BAND_FG,
            anchor="mm",
        )
    return img


def main() -> None:
    """Write the legend next to this module so it can be looked at.

    A developer convenience — nothing reads the file. The legend the model
    sees is rendered in-process by :mod:`duck.agent`.
    """
    out_path = Path(__file__).parent / "palette_legend.png"
    img = render_palette_legend()
    img.save(out_path, "PNG")
    print(f"Wrote {out_path}  ({img.size[0]}x{img.size[1]})")
    print()
    print("Preview options:")
    print(f"  - In VSCode: click {out_path} (or Cmd/Ctrl+P → palette_legend.png)")
    print(f"  - Terminal:  xdg-open {out_path}")
    print("  - Notebook:  from duck.palette_legend import render_palette_legend")
    print("               render_palette_legend()")


if __name__ == "__main__":
    main()
