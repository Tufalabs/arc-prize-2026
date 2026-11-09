"""The ARC grid and its palette — the one value type everything else is
built from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import ListedColormap

    # Served by the module ``__getattr__`` below; declared so callers
    # (``taaf.diagnostics``) type-check against the real colormap.
    ARC_CMAP: ListedColormap
else:
    # Runtime stand-ins for the two matplotlib types the annotations name,
    # so ``typing.get_type_hints`` on this module and its methods resolves
    # without importing matplotlib — importing it is what this module
    # avoids, and a name that exists only for the type checker turns every
    # runtime consumer of the annotations (typeguard, schema and doc
    # tooling) into a NameError.
    Axes = Any
    ListedColormap = Any

# ARC's own ten, in the order the testing interface uses them. Indices are
# the cell values, so this table is the palette and the legend at once.
#
# One table rather than two parallel lists: a colour and its name are the
# same fact, and anything that shows the model a swatch has to show the name
# beside it. Split apart they are two literals that must agree.
ARC_PALETTE: tuple[tuple[str, str], ...] = (
    ("#000000", "black"),
    ("#0074D9", "blue"),
    ("#FF4136", "red"),
    ("#2ECC40", "green"),
    ("#FFDC00", "yellow"),
    ("#AAAAAA", "grey"),
    ("#F012BE", "fuchsia"),
    ("#FF851B", "orange"),
    ("#7FDBFF", "teal"),
    ("#870C25", "maroon"),
)
#: Hex colours by cell value.
ARC_COLORS: list[str] = [hex_code for hex_code, _ in ARC_PALETTE]
#: Colour names by cell value. Model-facing: these are the words a solver
#: reads on a legend and may use in its own reasoning, so they are part of
#: the interface rather than a convenience.
COLOR_NAMES: list[str] = [name for _, name in ARC_PALETTE]
MAX_SIDE = 30
N_COLORS = len(ARC_PALETTE)


def _arc_cmap() -> ListedColormap:
    """The palette as a matplotlib colormap, built on first use and then
    bound as the ordinary module global ``ARC_CMAP``.

    ``ARC_COLORS`` is the palette itself and costs nothing; the colormap
    is the matplotlib-shaped view of it that only a plotting caller
    wants, and importing matplotlib for it costs ~30 MB of private heap
    in every process that imports this module. Once built it is a
    normal global, so ``dir()``, star import and repeated access all see
    the same object.
    """
    cmap: ListedColormap | None = globals().get("ARC_CMAP")
    if cmap is None:
        import matplotlib.colors

        cmap = matplotlib.colors.ListedColormap(ARC_COLORS)
        globals()["ARC_CMAP"] = cmap
    return cmap


def __getattr__(name: str) -> ListedColormap:
    """``ARC_CMAP`` on first attribute access (PEP 562); bound after that."""
    if name == "ARC_CMAP":
        return _arc_cmap()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The public surface whether or not ``ARC_CMAP`` has been built yet."""
    return sorted({*globals(), "ARC_CMAP"})


# The module's own API. Listed so ``from taaf.grid import *`` reaches
# ``ARC_CMAP`` through ``__getattr__`` — a star import consults ``__all__``
# name by name — rather than only what happens to be bound already.
__all__ = [
    "ARC_CMAP",
    "ARC_COLORS",
    "ARC_PALETTE",
    "COLOR_NAMES",
    "MAX_SIDE",
    "N_COLORS",
    "Grid",
]


@dataclass(frozen=True, eq=False)
class Grid:
    """A 2-D int8 array of ARC colors, 1..30 on each side.

    Validated at construction rather than at use: a malformed grid reaching
    the scorer would compare unequal and read as a wrong answer instead of a
    bug. The buffer is copied, so a grid never changes under whoever holds
    it.
    """

    data: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise ValueError(f"Grid must be 2-D, got shape {arr.shape}")
        h, w = arr.shape
        if not (1 <= h <= MAX_SIDE and 1 <= w <= MAX_SIDE):
            raise ValueError(f"Grid must be 1..{MAX_SIDE} on each side, got {h}x{w}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Grid must hold integers, got dtype {arr.dtype}")
        if not np.all((arr >= 0) & (arr < N_COLORS)):
            raise ValueError(f"Grid values must be 0..{N_COLORS - 1}")
        object.__setattr__(self, "data", np.array(arr, dtype=np.int8, copy=True))

    @classmethod
    def from_lists(cls, rows: list[list[int]]) -> Grid:
        """Build from the nested-list form used by ARC's JSON and by
        submissions."""
        # No dtype here: an out-of-range value must fail the palette check
        # below as a ValueError, not overflow int8 first.
        return cls(np.array(rows))

    def to_lists(self) -> list[list[int]]:
        """The nested-list form, for JSON and submissions."""
        return [[int(v) for v in row] for row in self.data]

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        h, w = self.data.shape
        return (int(h), int(w))

    def __eq__(self, other: object) -> bool:
        # Hand-written: the generated __eq__ compares arrays with ==, which
        # yields an array rather than a bool.
        if not isinstance(other, Grid):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def __hash__(self) -> int:
        return hash((self.data.shape, self.data.tobytes()))

    def draw(self, ax: Axes | None = None) -> Axes:
        """Draw on ``ax``, or on a fresh axes sized to the grid."""
        import matplotlib.pyplot as plt

        if ax is None:
            h, w = self.data.shape
            scale = 3.0 / max(h, w)
            _fig, ax = plt.subplots(figsize=(w * scale, h * scale))
        ax.imshow(self.data, cmap=_arc_cmap(), vmin=0, vmax=N_COLORS - 1, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
