# pyright: basic
# Verbatim copy from ARC3-Inference (c7bf328); keep diffable, do not annotate.
"""Grid formatting helpers."""

from __future__ import annotations

from typing import Sequence

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
ARC_COLOR_LEGEND = (
    "W=white, w=light gray, g=gray, G=dark gray, c=charcoal, B=black, "
    "M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, "
    "r=dark red, N=light green, p=purple"
)


def format_grid_ascii(grid: Sequence[Sequence[int]]) -> str:
    """Render a colour-index grid as the newline-delimited letter board the
    model reads (`current_frame.ascii`)."""
    if not grid:
        return "(empty grid)"
    lines = []
    for row in grid:
        chars = []
        for v in row:
            chars.append(ARC_COLOR_CHARS[max(0, min(15, int(v)))])
        lines.append("".join(chars))
    return "\n".join(lines)
