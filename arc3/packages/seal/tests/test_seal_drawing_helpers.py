"""``from_ascii_grid`` and ``bresenham_line`` as the model meets them in the REPL.

Both wrap framediff, whose blocks call them by name, so what matters here is
the seal side of that: the model can learn everything about them from
``help()`` because it cannot read their source, a mistake comes back as a
refusal naming the argument rather than as a numpy error from inside, and no
traceback shows a frame of framediff's. The advertised one is named in the
help overview; the other is preloaded without being named anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from seal.harness import game

sys.path.insert(0, str(Path(__file__).parent))
from test_seal_file_imports import _Kernel  # noqa: E402


@pytest.mark.parametrize(
    ("call", "complaint"),
    [
        (lambda: game.bresenham_line(np.zeros(5, dtype=np.int8), (0, 0), (0, 4), 1), "2-D NumPy array"),
        (lambda: game.bresenham_line(game.Frame(np.zeros((5, 5), dtype=np.int8)).grid, (0, 0), (4, 4), 1), "read-only"),
        (lambda: game.bresenham_line(np.zeros((5, 5), dtype=np.int8), (0, 0), (5, 1), 1), "outside the 5x5 grid"),
        (lambda: game.bresenham_line(np.zeros((5, 5), dtype=np.int8), (0, -1), (4, 4), 1), "outside the 5x5 grid"),
        (lambda: game.bresenham_line(np.zeros((5, 5), dtype=np.int8), (0,), (4, 4), 1), "pair of integers"),
        (lambda: game.bresenham_line(np.zeros((5, 5), dtype=np.int8), (0, 0), (4, 4), 16), "color index 0-15"),
        (lambda: game.from_ascii_grid(np.zeros((2, 2))), "as a string"),
        (lambda: game.from_ascii_grid("01\n0g"), "row 2, column 2"),
    ],
    ids=[
        "one_dimensional",
        "read_only_frame",
        "past_the_edge",
        "negative_index",
        "not_a_pair",
        "bad_color",
        "not_text",
        "bad_cell",
    ],
)
def test_a_misuse_is_refused_by_name(call: object, complaint: str) -> None:
    """A negative index would wrap and a frame would be edited in place, both silently; neither happens."""
    assert callable(call)
    with pytest.raises((TypeError, ValueError), match=complaint):
        call()


def test_bresenham_line_draws_on_the_grid_it_is_given() -> None:
    """In place, both ends included, and nothing else touched."""
    grid = np.full((5, 9), 5, dtype=np.int8)
    assert game.bresenham_line(grid, (0, 0), (4, 8), 3) is None
    assert game.ascii_grid(grid).splitlines()[0].endswith("335555555")
    assert int((grid == 3).sum()) == 9 and int((grid == 5).sum()) == 45 - 9


async def test_the_kernel_explains_both_and_shows_neither() -> None:
    """``help()`` is the whole documentation; source and framediff's frames stay out of reach."""
    kernel = _Kernel()
    try:
        overview = await kernel.run("help()")
        assert "from_ascii_grid" in overview
        assert "bresenham_line" not in overview

        described = await kernel.run("help(bresenham_line)")
        assert "Which cells a sloped line takes, exactly" in described
        assert "Direction matters" in described

        source = await kernel.run(
            "import inspect\n"
            "print(inspect.getsource(bresenham_line))\n"
            "print(inspect.getsource(bresenham_line.__globals__['_framediff_drawing'].bresenham_line))"
        )
        assert "def " not in source and source.count("Source code access denied") == 2

        refused = await kernel.run("from_ascii_grid('012\\n01')")
        assert "row 2 has 2 cells" in refused
        assert "framediff" not in refused and "grid.py" not in refused
    finally:
        await kernel.close()
