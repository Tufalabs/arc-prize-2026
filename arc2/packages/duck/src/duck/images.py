"""ARC grids as pictures and optional pair composites.

The renderer supplies whichever image presentation an agent enables. A task
may pair each training input beside its output for compact correspondence, or
send grids separately for direct text-image adjacency; other feature arms can
omit images and present exact digit rows instead. The kernel remains the
authoritative array surface in every arm.

Pixels come from `taaf.grid.ARC_COLORS`, the same palette
`platypus.llm.grid_image` uses, so a grid looks identical whichever of the
two rendered it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import ullm.conv

import taaf.grid

#: The ARC palette as RGB rows, indexed by cell value.
PALETTE: npt.NDArray[np.uint8] = np.array(
    [[int(taaf.grid.ARC_COLORS[i][j : j + 2], 16) for j in (1, 3, 5)] for i in range(taaf.grid.N_COLORS)],
    dtype=np.uint8,
)

#: What surrounds and separates grids. White, so it cannot be mistaken for a
#: cell: every palette colour is darker, and black is a real ARC colour that
#: a background would otherwise blend into.
BACKGROUND = np.array([255, 255, 255], dtype=np.uint8)

#: A dark separator that remains visible beside both black cells and the
#: white pair-image background.
GRIDLINE_COLOR = np.array([60, 60, 60], dtype=np.uint8)


def render(grid: taaf.grid.Grid, *, scale: int = 1, gridlines: bool = False) -> npt.NDArray[np.uint8]:
    """One grid as an RGB array, nearest-upscaled by ``scale``.

    Upscaling is pixel-exact rather than interpolated: cell boundaries stay
    hard, and a model asked to count cells can. Vision models resolve a raw
    30x30 poorly, which is what the scale is for. At scales of four or more,
    ``gridlines`` adds a one-pixel lattice around and between cells.
    """
    rgb = PALETTE[grid.data]
    if scale != 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    if gridlines and scale >= 4:
        rows, cols = grid.shape
        pixels = np.broadcast_to(
            GRIDLINE_COLOR,
            (rows * scale + rows + 1, cols * scale + cols + 1, 3),
        ).copy()
        flat_rows = np.arange(rows * scale)
        flat_cols = np.arange(cols * scale)
        row_index = 1 + flat_rows + flat_rows // scale
        col_index = 1 + flat_cols + flat_cols // scale
        pixels[np.ix_(row_index, col_index)] = rgb
        return pixels
    return rgb


def _paste(canvas: npt.NDArray[np.uint8], tile: npt.NDArray[np.uint8], top: int, left: int) -> None:
    """Place ``tile`` on ``canvas`` at ``(top, left)``."""
    canvas[top : top + tile.shape[0], left : left + tile.shape[1]] = tile


def side_by_side(
    grids: list[taaf.grid.Grid],
    *,
    scale: int = 8,
    gap: int = 8,
    margin: int = 8,
    gridlines: bool = False,
) -> npt.NDArray[np.uint8]:
    """Several grids in a row, vertically centred on a common background.

    Centred rather than top-aligned because an ARC pair often changes size,
    and a shared centre line makes a size change read as growth around the
    middle instead of as the output having moved down. ``gridlines`` is
    forwarded to each grid rendering.
    """
    tiles = [render(grid, scale=scale, gridlines=gridlines) for grid in grids]
    height = max(tile.shape[0] for tile in tiles) + 2 * margin
    width = sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1) + 2 * margin
    canvas = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    left = margin
    for tile in tiles:
        _paste(canvas, tile, top=(height - tile.shape[0]) // 2, left=left)
        left += tile.shape[1] + gap
    return canvas


# Qwen VL image processors round each side of an image up to a multiple of
# their patch size times merge size (32 on the checkpoints served here)
# before comparing the pixel count against their ceiling, so a ceiling taken
# from the raw render would shrink the very image it was sized for. Sides
# are rounded up to this instead: a multiple of every such factor up to it,
# and the margin costs a few hundred tokens of start-up profile, nothing
# the model sees.
PROCESSOR_SIDE_ROUNDING = 64


def _rounded_pixels(height: int, width: int) -> int:
    """The pixel count a processor compares against its ceiling: each side
    rounded up to :data:`PROCESSOR_SIDE_ROUNDING`."""
    rounding = PROCESSOR_SIDE_ROUNDING
    return -(-height // rounding) * rounding * (-(-width // rounding) * rounding)


def largest_image_pixels(*, scale: int, gridlines: bool) -> int:
    """The pixel ceiling an image processor needs so it never shrinks an
    image this agent sends.

    Two candidates, because the ceiling is a bound on every image and the
    grids are not always the larger: a pair of the largest grids ARC allows
    side by side, which is the widest render and grows with ``scale``, and
    the colour legend, whose size is fixed by its own layout, so that at
    small scales it is the legend that decides. Rendered rather than
    computed from the layout constants, so neither number can drift from
    what the agent actually sends.
    """
    # Inside the function: the legend's module draws with PIL and a bundled
    # font, which a game child has no reason to pay for on import.
    from duck import palette

    largest = taaf.grid.Grid(np.zeros((taaf.grid.MAX_SIDE, taaf.grid.MAX_SIDE), dtype=np.int8))
    height, width, _ = side_by_side([largest, largest], scale=scale, gridlines=gridlines).shape
    legend_height, legend_width, _ = palette.legend_array().shape
    return max(_rounded_pixels(height, width), _rounded_pixels(legend_height, legend_width))


def pair_image(
    question: taaf.grid.Grid,
    answer: taaf.grid.Grid,
    *,
    scale: int = 8,
    alt_text: str = "",
    gridlines: bool = False,
) -> ullm.conv.ImageContent:
    """One example pair: input on the left, output on the right."""
    return ullm.conv.ImageContent(
        data=side_by_side([question, answer], scale=scale, gridlines=gridlines),
        alt_text=alt_text,
    )


def question_image(
    question: taaf.grid.Grid,
    *,
    scale: int = 8,
    alt_text: str = "",
    gridlines: bool = False,
) -> ullm.conv.ImageContent:
    """One test input, rendered the same way a pair's halves are so that it
    reads as the same kind of object at the same size."""
    return ullm.conv.ImageContent(
        data=side_by_side([question], scale=scale, gridlines=gridlines),
        alt_text=alt_text,
    )


__all__ = [
    "BACKGROUND",
    "GRIDLINE_COLOR",
    "PALETTE",
    "PROCESSOR_SIDE_ROUNDING",
    "largest_image_pixels",
    "pair_image",
    "question_image",
    "render",
    "side_by_side",
]
