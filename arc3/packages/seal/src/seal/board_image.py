"""A board as the model sees it: the frame in the ARC palette, upscaled, and with
``SealAgent.axis_labels`` its rows and columns labelled along the top and left
edges.

Every image of a board goes through here (the board image on a prompt, the
audit's two, and the diff image, ``seal.diff_image``), so a labelled board and
its diff image line up cell for cell. The labels are the whale's
(``whale.history_narrative`` at ``d5bd7a6a``), drawn in the bundled font
(``seal.fonts``): the committed references pin an image by its bytes, so the
glyphs are part of the interface.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import taaf.game
import ullm.conv

from platypus.llm import frame_image

# Axis-label config: tick every N native cells, with a padding margin on top
# and left of the rendered grid for labels. 10-cell spacing gives 7 labels
# per axis on a 64-cell grid (0, 10, 20, …, 60) — sparse but uncluttered.
AXIS_TICK_STEP = 10
_AXIS_LABEL_PAD = 32
# Below this upscale there is not enough room for legible labels, so
# ``SealAgent`` refuses labels on a smaller board image.
AXIS_MIN_SCALE = 4
_AXIS_FONT_SIZE = 14

# Dark-mode margin: dark slate background with white labels and ticks. The
# slate is deliberately off pure black so it's not confused with palette
# colour #5 (true black) by visual inspection of the image edge-region.
_AXIS_MARGIN_RGB: tuple[int, int, int] = (30, 30, 35)
_AXIS_LABEL_RGB: tuple[int, int, int] = (255, 255, 255)


def board_image(grid: npt.NDArray[np.int8], *, scale: int, axis_labels: bool) -> ullm.conv.ImageContent:
    """``grid`` in the palette, ``scale`` pixels a cell, labelled when ``axis_labels``."""
    image = frame_image(taaf.game.Frame(data=grid), scale=scale)
    if not axis_labels:
        return image
    rgb = add_axis_labels(np.asarray(image.data, dtype=np.uint8), grid_shape=grid.shape, scale=scale)
    return ullm.conv.ImageContent(data=rgb, alt_text=image.alt_text)


def add_axis_labels(
    arr: npt.NDArray[np.uint8],
    grid_shape: tuple[int, ...],
    scale: int,
    pad: int = _AXIS_LABEL_PAD,
    step: int = AXIS_TICK_STEP,
) -> npt.NDArray[np.uint8]:
    """Pad ``arr`` with a dark slate margin on top and left, then draw row/col
    axis ticks + numeric labels at every ``step`` native cells in white.
    Returns a new ndarray (the original image is preserved unchanged inside
    the padded canvas)."""
    # Inside the function: PIL and the bundled font cost an import a game
    # child pays only once a labelled board is drawn.
    from PIL import Image, ImageDraw

    from seal import fonts

    font = fonts.load(_AXIS_FONT_SIZE)
    h, w, _ = arr.shape
    canvas = np.empty((h + pad, w + pad, 3), dtype=np.uint8)
    canvas[:] = _AXIS_MARGIN_RGB  # broadcast fill
    canvas[pad:, pad:, :] = arr

    canvas_img = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(canvas_img)
    rows, cols = grid_shape
    fg = _AXIS_LABEL_RGB

    # Column labels (top) — label every ``step`` columns. Tick line below
    # the label pointing down into the grid, ending on the margin's last
    # pixel: one further, as the whale drew it, paints a pixel of the board.
    for c in range(0, cols, step):
        x = pad + c * scale + scale // 2
        draw.text((x, 4), str(c), fill=fg, font=font, anchor="mt")
        draw.line([(x, pad - 5), (x, pad - 1)], fill=fg, width=1)

    # Row labels (left) — label every ``step`` rows. Tick line right of
    # the label pointing into the grid.
    for r in range(0, rows, step):
        y = pad + r * scale + scale // 2
        draw.text((pad - 6, y), str(r), fill=fg, font=font, anchor="rm")
        draw.line([(pad - 5, y), (pad - 1, y)], fill=fg, width=1)

    return np.asarray(canvas_img, dtype=np.uint8)
