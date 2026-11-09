"""framediff's transforms: each edit performed in process and written as numpy must agree.

The emitted block is what the model reads and the harness runs, so the sweep
holds every kind of transform to the same contract — the block reaches what
``transform`` reaches, leaves its start frame alone, and spells itself the
same way every time — and the unit tests below pin the semantics a block
could get subtly wrong: which way a turn goes, what a copy from the board
reads, which cells a line takes, what ``from_ascii_grid`` reads back.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from framediff import (
    GRID_SIZE,
    AsciiGrid,
    BlockCopy,
    Composite,
    FillRect,
    Grid,
    Line,
    MaskedMove,
    NoOp,
    Recolour,
    SetPixel,
    Transform,
    block_namespace,
    from_ascii_grid,
    to_ascii_grid,
)
from framediff.transforms import with_colors


def _frame(seed: int) -> Grid:
    """A random frame; a wrong geometry shows almost everywhere on one."""
    return np.random.default_rng(seed).integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)


def _run(code: str, a: Grid, b: Grid) -> dict[str, Any]:
    """Execute a block against ``a`` and ``b`` bound under their default names."""
    namespace = block_namespace(a=a, b=b)
    exec(code, namespace)
    return namespace


CASES: dict[str, Transform] = {
    "no_op": NoOp(),
    "set_pixel": SetPixel(63, 58, 0),
    "fill": FillRect(27, 30, 24, 27, color=0),
    "recolour": Recolour(20, 44, 16, 48, old=0, new=2),
    "copy": BlockCopy(27, 30, 30, 33, src_row=27, src_col=36),
    "copy_turned": BlockCopy(10, 13, 10, 15, src_row=40, src_col=40, rot=1),
    "copy_mirrored": BlockCopy(10, 13, 10, 15, src_row=40, src_col=40, flip=True),
    "copy_mirrored_turned": BlockCopy(10, 13, 10, 15, src_row=40, src_col=40, rot=3, flip=True),
    "copy_from_board_overlapping": BlockCopy(10, 20, 12, 22, src_row=10, src_col=10, source="g"),
    "move": BlockCopy(27, 30, 30, 33, src_row=27, src_col=36, clear=0),
    "move_overlapping": BlockCopy(10, 20, 12, 22, src_row=10, src_col=10, clear=5),
    "move_turned": BlockCopy(10, 13, 10, 15, src_row=40, src_col=40, rot=1, flip=True, clear=2),
    "line": Line((5, 3), (20, 9), color=8),
    "paste": AsciiGrid(40, 20, ("0a1b", "f2e3")),
    "composite": Composite(
        (FillRect(0, 4, 0, 4, color=3), BlockCopy(10, 14, 10, 14, src_row=0, src_col=0, source="g"), SetPixel(1, 1, 9))
    ),
    "composite_with_lines": Composite(
        (Line((2, 3), (12, 40), color=8), AsciiGrid(4, 20, ("ee0ee",)), Line((50, 5), (40, 20), color=3))
    ),
    "empty_composite": Composite(()),
}


@pytest.mark.parametrize("name", CASES)
def test_the_block_reaches_what_transform_reaches(name: str) -> None:
    """On a frame the self-test never saw, the block rebuilds the result and its assert holds."""
    transform = CASES[name]
    start = _frame(7)
    expected = transform.transform(start)
    namespace = _run(transform.to_code(), start.copy(), expected)
    # A NoOp's block builds no board; the start frame is then what it vouches for.
    assert np.array_equal(namespace.get("g", namespace["a"]), expected)
    assert np.array_equal(namespace["a"], start), "the block wrote to its start frame"


@pytest.mark.parametrize("name", CASES)
def test_transform_leaves_its_input_alone(name: str) -> None:
    """``transform`` returns a new int8 frame and never writes the one it was given."""
    start = _frame(3)
    before = start.copy()
    result = CASES[name].transform(start)
    assert np.array_equal(start, before)
    assert result.dtype == np.int8 and result is not start


@pytest.mark.parametrize("name", CASES)
def test_emission_is_a_function_of_the_transform(name: str) -> None:
    """Equal transforms emit the same text, and numbers are spelled as numbers
    even when a search hands the fields numpy integers."""
    transform = CASES[name]
    assert transform.to_code() == transform.to_code()
    assert "np.int" not in transform.to_code()


def test_a_no_op_block_is_the_comparison_alone() -> None:
    """Nothing changed, so there is no board: one assert, which fails if the frames do differ."""
    code = NoOp().to_code(a="history[-2].current_frame.grid", b="current_frame.grid")
    assert code == "assert (history[-2].current_frame.grid == current_frame.grid).all()"
    start = _frame(1)
    changed = start.copy()
    changed[0, 0] = (changed[0, 0] + 1) % 16
    _run(NoOp().to_code(), start, start.copy())
    with pytest.raises(AssertionError):
        _run(NoOp().to_code(), start, changed)
    assert NoOp().bit_cost() == 0
    assert Composite((NoOp(), SetPixel(0, 0, 1))).to_code() == SetPixel(0, 0, 1).to_code()


def test_numpy_integers_in_fields_become_plain_numbers() -> None:
    """A search computes coordinates with numpy; the block must not print ``np.int64(27)``."""
    from_numpy = FillRect(np.int64(27), np.int64(30), np.int32(24), np.int16(27), color=np.int8(0))  # pyright: ignore[reportArgumentType]  # what a search hands over
    assert from_numpy == FillRect(27, 30, 24, 27, color=0)
    assert from_numpy.to_code() == FillRect(27, 30, 24, 27, color=0).to_code()


def test_the_block_names_the_callers_expressions() -> None:
    """The names are whatever the model holds the frames as, and the board is what the caller picks."""
    code = BlockCopy(27, 30, 30, 33, src_row=27, src_col=36).to_code(
        a="history[-2].current_frame.grid", b="current_frame.grid", g="board"
    )
    assert code.splitlines() == [
        "board = history[-2].current_frame.grid.copy()",
        "board[27:30, 30:33] = history[-2].current_frame.grid[27:30, 36:39]  # copied left 6",
        "assert (board == current_frame.grid).all()",
    ]


@pytest.mark.parametrize(
    ("names", "complaint"),
    [
        (("a + b", "b", "g"), "must be a name"),
        (("a", "b", "g[0]"), "plain name"),
        (("a", "b", "class"), "plain name"),
        (("frame", "frame", "g"), "three different names"),
        (("a", "b[", "g"), "not an expression"),
    ],
)
def test_names_the_block_cannot_use_are_refused(names: tuple[str, str, str], complaint: str) -> None:
    """A name the block could not read from, or assign, fails at emission and not in the kernel."""
    with pytest.raises(ValueError, match=complaint):
        SetPixel(0, 0, 1).to_code(*names)


def test_a_block_that_disagrees_with_its_transform_is_caught() -> None:
    """The self-test is what makes the text trustworthy; a transform whose code lies must not emit."""

    class Lying(SetPixel):
        """Writes a different colour than it sets."""

        def _code_lines(self, a: str, g: str) -> list[str]:
            """The wrong colour."""
            return [f"{g}[{self.row}, {self.col}] = {(self.color + 1) % 16}"]

    with pytest.raises(AssertionError, match="does not reach"):
        Lying(5, 5, 3).to_code()


def test_a_block_that_writes_its_start_frame_is_caught() -> None:
    """``a`` is the model's own frame; a block that edits it corrupts the model's history."""

    class Vandal(SetPixel):
        """Sets the cell on the start frame too."""

        def _code_lines(self, a: str, g: str) -> list[str]:
            """Both assignments."""
            return [f"{a}[{self.row}, {self.col}] = {self.color}", f"{g}[{self.row}, {self.col}] = {self.color}"]

    with pytest.raises(AssertionError, match="writes to its start frame"):
        Vandal(5, 5, 3).to_code()


@pytest.mark.parametrize(("rot", "flip"), [(1, False), (2, False), (3, False), (0, True), (1, True), (3, True)])
def test_a_turned_copy_mirrors_then_turns_like_numpy(rot: int, flip: bool) -> None:
    """``flip`` is ``np.fliplr`` and comes first; ``rot`` counts ``np.rot90`` quarter turns counterclockwise."""
    start = _frame(11)
    copy = BlockCopy(10, 13, 20, 25, src_row=40, src_col=30, rot=rot, flip=flip)
    rows, cols = copy.block_shape
    block = start[40 : 40 + rows, 30 : 30 + cols]
    expected = np.rot90(np.fliplr(block) if flip else block, rot)
    assert np.array_equal(copy.transform(start)[10:13, 20:25], expected)


def test_a_copy_from_the_board_reads_the_edits_before_it() -> None:
    """``source='g'`` sees what earlier parts wrote; ``source='a'`` sees the start frame."""
    start = np.full((GRID_SIZE, GRID_SIZE), 5, dtype=np.int8)
    fill = FillRect(0, 3, 0, 3, color=11)
    from_board = Composite((fill, BlockCopy(20, 23, 20, 23, src_row=0, src_col=0, source="g"))).transform(start)
    from_start = Composite((fill, BlockCopy(20, 23, 20, 23, src_row=0, src_col=0, source="a"))).transform(start)
    assert (from_board[20:23, 20:23] == 11).all()
    assert (from_start[20:23, 20:23] == 5).all()


def test_a_copy_or_move_names_the_colours_it_changes_cells_to() -> None:
    """A copy's comment ends with the colours its line changes cells to: what it writes where the board held
    another colour just before the line, after the edits before it and, for a move, after its old place is
    cleared. Floor carried onto floor is no change. A masked move with no frame to replay names its object's
    colours. The edit is the same, and equal to itself without them."""
    start = np.full((GRID_SIZE, GRID_SIZE), 5, dtype=np.int8)
    start[0, 0] = 9
    fill = FillRect(0, 2, 0, 1, color=11)
    move = BlockCopy(20, 23, 20, 23, src_row=0, src_col=0, clear=0)
    board_copy = BlockCopy(30, 32, 30, 32, src_row=0, src_col=0, source="g")
    named = with_colors(Composite((move, fill, board_copy)), start)
    assert named == Composite((move, fill, board_copy))
    lines = named.to_code("a", "b").splitlines()
    assert lines[2].endswith("# moved down 20, right 20 (colors: 9)")
    assert lines[4].endswith("# copied down 30, right 30 (colors: 0, 11)")
    assert "(col:" not in Composite((move, fill)).to_code("a", "b"), "nothing named where nothing was read"
    masked = MaskedMove(25, 28, 11, 14, src_row=24, src_col=10, colors=(9, 6), clear=1)
    assert masked.to_code("a", "b").splitlines()[-2].endswith("# moved down 1, right 1 (colors: 6, 9)")


def test_an_overlapping_copy_from_the_board_reads_the_block_before_writing() -> None:
    """A move by less than its own size reads the block as it was, not half-overwritten."""
    start = _frame(5)
    moved = BlockCopy(10, 20, 12, 22, src_row=10, src_col=10, source="g").transform(start)
    assert np.array_equal(moved[10:20, 12:22], start[10:20, 10:20])


def test_a_recolour_touches_only_its_colour_inside_its_rectangle() -> None:
    """Other colours inside, and the old colour outside, keep their cells."""
    start = _frame(9)
    result = Recolour(10, 30, 10, 30, old=4, new=12).transform(start)
    inside = start[10:30, 10:30]
    assert np.array_equal(result[10:30, 10:30], np.where(inside == 4, 12, inside))
    outside = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    outside[10:30, 10:30] = False
    assert np.array_equal(result[outside], start[outside])


def test_a_paste_writes_its_grid_as_a_picture() -> None:
    """The rows sit at the left margin, one per line, inside a ``from_ascii_grid`` call."""
    paste = AsciiGrid.from_array(40, 20, np.array([[0, 10, 1], [15, 2, 14]]))
    assert paste.rows == ("0a1", "f2e")
    assert paste.to_code().splitlines()[1:5] == ['g[40:42, 20:23] = from_ascii_grid("""', "0a1", "f2e", '""")']
    assert paste.bit_cost() == AsciiGrid.COST + AsciiGrid.CELL_COST * 6


def test_a_block_that_draws_lines_says_once_where_bresenham_line_comes_from() -> None:
    """The REPL does not advertise ``bresenham_line``, so the block points at its help, once, first."""
    lines = Composite(
        (Line((0, 0), (5, 9), color=1), FillRect(9, 12, 9, 12, color=2), Line((20, 20), (30, 2), color=3))
    )
    code = lines.to_code().splitlines()
    note = "# bresenham_line is already loaded; see help(bresenham_line)"
    assert code.count(note) == 1 and code[0] == note
    assert code[1:] == [
        "g = a.copy()",
        "bresenham_line(g, (0, 0), (5, 9), 1)",
        "g[9:12, 9:12] = 2",
        "bresenham_line(g, (20, 20), (30, 2), 3)",
        "assert (g == b).all()",
    ]
    assert "bresenham_line" not in FillRect(0, 3, 0, 3, color=1).to_code()


@pytest.mark.parametrize(
    "printed",
    [
        "0123\n4567\n89ab",
        "0\t0123\n1\t4567\n2\t89ab",
        "13\t0123\n14\t4567\n15\t89ab",
        "01\t23\n45\t67\n89\tab",
        "0\t01\t23\n1\t45\t67\n2\t89\tab",
        "\n    0123\n    4567\n\n    89ab\n",
    ],
    ids=["plain", "row_numbers", "row_numbers_of_a_crop", "column_groups", "both", "indented_with_blank_lines"],
)
def test_from_ascii_grid_reads_what_ascii_grid_prints(printed: str) -> None:
    """Every layout the REPL's ``ascii_grid()`` prints reads back to the same array."""
    expected = np.arange(12, dtype=np.int8).reshape(3, 4)
    assert np.array_equal(from_ascii_grid(printed), expected)
    assert np.array_equal(from_ascii_grid(to_ascii_grid(expected)), expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("11\t22\n11\t22", [[1, 1, 2, 2], [1, 1, 2, 2]]),
        ("0000\t5\n0001\t5", [[0, 0, 0, 0, 5], [0, 0, 0, 1, 5]]),
    ],
    ids=["does_not_count_up", "leading_zeros"],
)
def test_a_first_column_that_is_not_a_row_index_is_part_of_the_grid(text: str, expected: list[list[int]]) -> None:
    """Grouped cells that happen to be digits are cells: row indices count up by one and have no leading zeros."""
    assert from_ascii_grid(text).tolist() == expected


@pytest.mark.parametrize(
    ("text", "complaint"),
    [("", "no rows"), ("012\n01", "row 2 has 2 cells"), ("01g", "row 1, column 3 is 'g'")],
)
def test_from_ascii_grid_names_what_it_refuses(text: str, complaint: str) -> None:
    """The model calls it too, so a refusal says where."""
    with pytest.raises(ValueError, match=complaint):
        from_ascii_grid(text)


def _r11l_line_drawer() -> Callable[..., None]:
    """r11l's own line rasterizer, loaded from the game file."""
    re_arc = pytest.importorskip("re_arc")
    path = Path(re_arc.__file__).parent / "environment_files" / "r11l" / "495a7899" / "r11l.py"
    spec = importlib.util.spec_from_file_location("r11l_game", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for value in vars(module).values():
        if isinstance(value, type) and hasattr(value, "gfdlgriwmq"):
            return value.gfdlgriwmq  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
    raise AssertionError(f"no class in {path.name} carries the line rasterizer")


def test_a_line_takes_the_cells_r11l_draws() -> None:
    """Cell for cell against the game's own function, over many slopes and both directions."""
    draw = _r11l_line_drawer()
    rng = np.random.default_rng(0)
    for _ in range(300):
        r0, c0, r1, c1 = (int(v) for v in rng.integers(0, GRID_SIZE, size=4))
        frame = np.full((GRID_SIZE, GRID_SIZE), 5, dtype=np.int8)
        draw(None, frame, c0, r0, c1, r1, 8)
        theirs = {(int(r), int(c)) for r, c in zip(*np.nonzero(frame == 8))}
        assert set(Line((r0, c0), (r1, c1), color=8).cells()) == theirs, f"({r0}, {c0}) -> ({r1}, {c1})"


@pytest.mark.parametrize(
    ("build", "complaint"),
    [
        (lambda: FillRect(10, 10, 0, 5, color=1), "non-empty"),
        (lambda: FillRect(60, 65, 0, 5, color=1), "inside 0:64"),
        (lambda: SetPixel(0, 0, 16), "colour 0-15"),
        (lambda: Recolour(0, 5, 0, 5, old=3, new=3), "changes a colour"),
        (lambda: BlockCopy(0, 3, 0, 5, src_row=62, src_col=0, rot=1), "source row"),
        (lambda: BlockCopy(0, 3, 0, 5, src_row=0, src_col=0, rot=4), "quarter turns"),
        (lambda: BlockCopy(0, 3, 0, 5, src_row=0, src_col=0, source="b"), "source is"),  # pyright: ignore[reportArgumentType]
        (lambda: AsciiGrid(0, 0, ("012", "01")), "row 2 has 2 cells"),
        (lambda: AsciiGrid(0, 0, ("01g",)), "a cell is one of"),
        (lambda: AsciiGrid(0, 62, ("012",)), "inside 0:64"),
        (lambda: Line((0, 0), (64, 3), color=2), "end row"),
    ],
)
def test_a_transform_outside_the_frame_or_the_palette_is_refused(
    build: Callable[[], Transform], complaint: str
) -> None:
    """numpy would clip an out-of-frame slice silently, in process and in the block alike."""
    with pytest.raises(ValueError, match=complaint):
        build()


def test_a_composite_costs_its_parts() -> None:
    """A description of the change is priced as the sum of its edits."""
    parts = (FillRect(0, 3, 0, 3, color=1), BlockCopy(10, 13, 10, 13, src_row=0, src_col=0, rot=2), SetPixel(5, 5, 5))
    assert Composite(parts).bit_cost() == sum(part.bit_cost() for part in parts)


def test_a_move_clears_its_old_place_first() -> None:
    """A move by less than the block's size: the block lands whole and only the uncovered part of its old place
    is cleared, because the clear comes before the write; it costs what the copy costs."""
    start = _frame(2)
    move = BlockCopy(10, 20, 12, 22, src_row=10, src_col=10, clear=5)
    result = move.transform(start)
    assert np.array_equal(result[10:20, 12:22], start[10:20, 10:20])
    assert (result[10:20, 10:12] == 5).all()
    assert move.bit_cost() == BlockCopy(10, 20, 12, 22, src_row=10, src_col=10).bit_cost()
    lines = move.to_code().splitlines()
    assert lines[1] == "g[10:20, 10:20] = 5  # cleared behind the move below"
    assert lines[2] == "g[10:20, 12:22] = a[10:20, 10:20]  # moved right 2"


def test_a_move_reads_the_start_frame() -> None:
    """Clearing behind a copy from the board would read what was just cleared; it is refused."""
    with pytest.raises(ValueError, match="a move reads the start frame"):
        BlockCopy(10, 13, 10, 13, src_row=0, src_col=0, source="g", clear=0)


def test_a_copy_says_which_way_its_content_went() -> None:
    """The slice reads destination first; the comment names the direction the content travelled, in rows and cols."""
    cases = [
        (BlockCopy(20, 23, 20, 23, src_row=15, src_col=23), "# copied down 5, left 3"),
        (BlockCopy(10, 13, 40, 43, src_row=16, src_col=40, clear=0), "# moved up 6"),
        (BlockCopy(10, 13, 40, 43, src_row=10, src_col=34, clear=0), "# moved right 6"),
    ]
    for copy, comment in cases:
        assert copy.to_code().splitlines()[-2].endswith(comment), copy.to_code()
    assert "#" not in BlockCopy(10, 13, 10, 15, src_row=40, src_col=40, rot=1).to_code().splitlines()[1]
