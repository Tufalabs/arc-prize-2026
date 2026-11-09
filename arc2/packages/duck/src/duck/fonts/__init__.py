"""The one font the duck draws with, shipped beside this module.

:mod:`duck.palette` renders the colour legend the model is shown, and
replay references pin an image by its exact bytes
(``platypus.reference``). Glyph shapes are therefore part of the interface,
and the font is not a preference. This branch has no references yet
(`roadmap.md`), so nothing pins these glyphs today — bundling now is what
makes the references worth pinning when they land.

**Bundled, not found.** Substituting Arial, Helvetica or Pillow's bitmap
default would not degrade the image, it would produce a *different* one, so
every replay on that machine would fail on a hash with nothing in the diff
to say why. Reading the installed system font has the same failure in a
quieter form: distributions package their own builds of DejaVu, so two
machines can disagree while both believe they have the right font. Carrying
the bytes removes the question — the glyphs are pinned by the repo, on a
laptop, a compute node and an offline Kaggle kernel alike.

``LICENSE_DEJAVU`` sits next to the ``.ttf`` files because the Bitstream
Vera license requires the notice to travel with the fonts. They are shipped
verbatim; modifying them under these names is what that license forbids.
"""

from __future__ import annotations

import io
from functools import lru_cache
from importlib import resources

from PIL import ImageFont


@lru_cache(maxsize=2)
def _font_bytes(name: str) -> bytes:
    """The file, read once per weight — each is ~0.7 MB and ``load`` is
    called per rendered image."""
    return (resources.files(__name__) / name).read_bytes()


def load(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu Sans at ``size``, bold or regular.

    Built from an in-memory copy rather than a path so the lookup cannot
    depend on where the package was installed from — a zipped install and a
    source tree resolve identically. The stream is kept alive by the font
    object that reads glyphs from it.
    """
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(io.BytesIO(_font_bytes(name)), size)
