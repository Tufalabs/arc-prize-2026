"""What changed between two frames, as a short numpy edit script.

``framediff.transforms`` holds the edits; each performs itself on a frame and
writes itself as numpy the model can read and run. ``framediff.decompose``
searches for a short script between two given frames, and chooses among
several start frames for one target. ``framediff.animation`` rebuilds an
animation frame by frame as one block.
"""

from framediff.animation import AnimationDecomposition, AnimationFrame
from framediff.decompose import Decomposer, Decomposition, Source, explain
from framediff.drawing import bresenham_line
from framediff.grid import GRID_SIZE, HEX_CHARS, NUM_COLORS, Grid, from_ascii_grid, to_ascii_grid
from framediff.transforms import (
    WILDCARD,
    AsciiGrid,
    BlockCopy,
    Composite,
    FillRect,
    Line,
    MaskedMove,
    NoOp,
    Recolour,
    SetPixel,
    Transform,
    block_namespace,
)

__all__ = [
    "AnimationDecomposition",
    "AnimationFrame",
    "GRID_SIZE",
    "HEX_CHARS",
    "NUM_COLORS",
    "WILDCARD",
    "AsciiGrid",
    "BlockCopy",
    "Composite",
    "Decomposer",
    "Decomposition",
    "FillRect",
    "Grid",
    "Line",
    "MaskedMove",
    "NoOp",
    "Recolour",
    "SetPixel",
    "Source",
    "Transform",
    "block_namespace",
    "bresenham_line",
    "explain",
    "from_ascii_grid",
    "to_ascii_grid",
]
