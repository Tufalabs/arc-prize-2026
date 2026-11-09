"""The diff image (``SealAgent.diff_image``): the cells a move changed, beside the board image.

The kernel decides what a move's frame is compared with and which cells differ
(``harness._repl_state.diff_image_after_call``); this draws them, from the same
palette as the board image.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import taaf.game
import ullm.conv

from platypus.llm import frame_image
from seal.board_image import add_axis_labels

#: The colour of every cell the move did not change. The palette's indices 0-5
#: are a grey ramp from white to black, so no grey, black or white is free; a
#: dark desaturated navy is no game colour.
UNCHANGED_RGB = (16, 24, 48)


def render_diff_image(
    frame: npt.NDArray[np.int8], changed: npt.NDArray[np.bool_], *, scale: int, axis_labels: bool
) -> ullm.conv.ImageContent:
    """``frame``'s changed cells in their own colour and the rest navy, upscaled
    and labelled as the board image is, so the two line up cell for cell."""
    rgb = np.array(frame_image(taaf.game.Frame(data=frame), scale=1).data, copy=True)
    rgb[~changed] = UNCHANGED_RGB
    if scale != 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    if axis_labels:
        rgb = add_axis_labels(rgb.astype(np.uint8), grid_shape=frame.shape, scale=scale)
    return ullm.conv.ImageContent(data=rgb, alt_text="")
