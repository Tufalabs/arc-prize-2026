"""framediff's decomposer: the script it finds must rebuild the target, and reach for the edit the costs order first.

The search is greedy on declared costs, so the tests pin outcomes at the
cost table's boundaries (a moved sprite is a copy and a fill, a run of one
colour a fill and not a line, content nothing else describes a paste), the
exactness of the two fast searches it rests on (the correlation against a
brute-force count, the rectangle search against a brute-force scan), and
the contract every result keeps: the block runs, its assert holds, wildcard
cells cost nothing, and the same frames give the same block. They search with
the options off (``OPTIONS_OFF``), so they pin the plain search; each option's
own tests say what it changes.
"""

from __future__ import annotations

import dataclasses
import itertools
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import framediff.decompose as decompose_module
from framediff import (
    GRID_SIZE,
    WILDCARD,
    AsciiGrid,
    BlockCopy,
    Composite,
    Decomposer,
    FillRect,
    Grid,
    Line,
    NoOp,
    SetPixel,
    Source,
    Transform,
    block_namespace,
    explain,
    from_ascii_grid,
)
from framediff.decompose import OPTIONS_OFF, _best_rectangles, _Search


def _frames(name: str) -> dict[str, Grid]:
    """The named frames of one of the notebooks' recorded-frame files."""
    text = (Path(__file__).parents[1] / "notebooks" / f"{name}_frames.txt").read_text()
    return {
        name: from_ascii_grid(grid) for name, _, grid in (block.partition("\n") for block in text.strip().split("\n\n"))
    }


FRAMES = _frames("tu93")
LF52 = _frames("lf52")


def _plain(**changes: Any) -> Decomposer:
    """The search with its options off, ``changes`` over it."""
    return Decomposer(**{**OPTIONS_OFF, **changes})


def _blank(color: int = 0) -> Grid:
    """A frame of one colour."""
    return np.full((GRID_SIZE, GRID_SIZE), color, dtype=np.int8)


def _parts(transform: Transform) -> tuple[Transform, ...]:
    """The parts of a result, which is a composite unless nothing changed."""
    assert isinstance(transform, Composite)
    return transform.transforms


def _kinds(transform: Transform) -> list[str]:
    """The parts' kinds in order, a turned copy marked."""
    return [
        type(part).__name__ + ("*" if isinstance(part, BlockCopy) and part.turned else "") for part in _parts(transform)
    ]


def _check(a: Grid, b: Grid, transform: Transform) -> None:
    """The result rebuilds ``b`` from ``a`` in process, and its block does the same and passes its assert."""
    assert np.array_equal(transform.transform(a), b)
    namespace = block_namespace(a=a.copy(), b=b)
    exec(transform.to_code(), namespace)


def test_equal_frames_are_a_no_op() -> None:
    """Nothing to describe: a NoOp, not an empty composite."""
    start = FRAMES["level1_start"]
    assert _plain().decompose(start, start.copy()) == NoOp()


def test_one_cell_is_one_cell() -> None:
    """A single changed cell is set on its own; nothing cheaper describes it."""
    start = FRAMES["level1_start"]
    target = start.copy()
    target[40, 40] = 9
    result = _plain().decompose(start, target)
    assert _parts(result) == (SetPixel(40, 40, 9),)
    _check(start, target, result)


def test_a_moved_sprite_is_a_move() -> None:
    """tu93's first move: the player tile shifts six columns leaving white, and the bar loses a cell; 14."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    result = _plain().decompose(start, target)
    _check(start, target, result)
    assert result.bit_cost() == 14
    assert _parts(result)[0] == BlockCopy(15, 18, 21, 24, src_row=15, src_col=15, clear=0)


def test_the_guard_slide_is_short() -> None:
    """tu93's losing move, 28 cells: a strip shift, a cell and the bar."""
    start, target = FRAMES["level2_before_right"], FRAMES["level2_after_right"]
    result = _plain().decompose(start, target)
    _check(start, target, result)
    assert result.bit_cost() <= 30 and len(_parts(result)) <= 5


def test_a_move_is_one_transform() -> None:
    """A sprite moved by its own height inside a strip: one move at 10, not the strip turned in place at 18."""
    start = _blank(3)
    start[0:5, :] = 4
    start[5:7, 29:34] = 12
    start[7:10, 29:34] = 9
    target = _blank(3)
    target[0:5, :] = 4
    target[10:12, 29:34] = 12
    target[12:15, 29:34] = 9
    result = _plain().decompose(start, target)
    _check(start, target, result)
    assert result.bit_cost() == 10
    assert _parts(result) == (BlockCopy(10, 15, 29, 34, src_row=5, src_col=29, clear=3),)


def test_a_run_of_one_colour_is_a_fill_not_a_line() -> None:
    """A horizontal run costs 6 as a fill and 12 as a line."""
    start = _blank()
    target = start.copy()
    target[10, 5:40] = 8
    result = _plain().decompose(start, target)
    assert _kinds(result) == ["FillRect"]
    _check(start, target, result)


def test_a_diagonal_is_a_line() -> None:
    """A diagonal run of one colour is a line at 12; as a paste it would pay for its whole bounding box."""
    start = _blank()
    line = Line((5, 3), (30, 40), color=8)
    target = line.transform(start)
    result = _plain().decompose(start, target)
    assert _kinds(result) == ["Line"] and result.bit_cost() == 12
    _check(start, target, result)


def test_new_content_is_a_paste() -> None:
    """A 3x3 block of nine colours from nowhere: a paste at 17, not nine cells at 36."""
    start = _blank()
    target = start.copy()
    target[20:23, 20:23] = np.arange(1, 10, dtype=np.int8).reshape(3, 3)
    result = _plain().decompose(start, target)
    assert _kinds(result) == ["AsciiGrid"] and result.bit_cost() == 17
    _check(start, target, result)


def test_a_turned_block_is_a_turned_copy() -> None:
    """A random 3x5 block reappearing turned a quarter turn: a turned copy at 18, not a paste at 23."""
    start = _blank()
    start[2:5, 2:7] = np.random.default_rng(1).integers(1, 16, (3, 5))
    copy = BlockCopy(40, 45, 40, 43, src_row=2, src_col=2, rot=1)
    target = copy.transform(start)
    result = _plain().decompose(start, target)
    assert _kinds(result) == ["BlockCopy*"] and result.bit_cost() == 18
    _check(start, target, result)


def test_a_paste_is_carved() -> None:
    """A complex shape beside a filled rectangle: a fill and a paste of the shape, not one paste of both."""
    start = _blank()
    target = start.copy()
    target[10:16, 10:16] = np.random.default_rng(3).integers(1, 16, (6, 6))
    target[10:16, 16:30] = 7
    result = _plain().decompose(start, target)
    _check(start, target, result)
    assert result.bit_cost() <= FillRect.COST + AsciiGrid.COST + 36
    assert sorted(_kinds(result)) == ["AsciiGrid", "FillRect"]


def test_a_scroll_under_a_fixed_bar_is_one_copy() -> None:
    """The world shifts two columns while the bar at the bottom stays: one copy of the world, then the revealed strip."""
    rng = np.random.default_rng(5)
    start = rng.integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    start[60:64, :] = 11
    target = start.copy()
    target[0:60, 0:62] = start[0:60, 2:64]
    target[0:60, 62:64] = rng.integers(0, 16, (60, 2))
    result = _plain().decompose(start, target)
    _check(start, target, result)
    copies = [part for part in _parts(result) if isinstance(part, BlockCopy)]
    # The world copy may clear the strip it uncovers, which is free and is then part of the reveal.
    assert dataclasses.replace(copies[0], clear=None) == BlockCopy(0, 60, 0, 62, src_row=0, src_col=2)
    assert result.bit_cost() <= BlockCopy.COST + AsciiGrid.COST + 120


def test_wildcards_cost_nothing_and_are_resolved() -> None:
    """A start frame with wildcard cells: they are never wrong, the block resolves them, a NoOp compares around them."""
    start, target = FRAMES["level1_start"].copy(), FRAMES["level1_after_right"]
    start[63, :] = WILDCARD
    result = _plain().decompose(start, target)
    assert result.bit_cost() == 10, "the bar under the wildcards is not described"
    code = result.to_code(wildcards=True)
    assert "g[a == -1] = b[a == -1]  # resolve wildcards" in code.splitlines()
    namespace = block_namespace(a=start.copy(), b=target)
    exec(code, namespace)
    assert np.array_equal(namespace["g"], target)
    all_but_the_bar = target.copy()
    all_but_the_bar[63, :] = WILDCARD
    assert _plain().decompose(all_but_the_bar, target) == NoOp()
    assert (
        explain(all_but_the_bar, target, "predicted", "current")
        == "assert ((predicted == -1) | (predicted == current)).all()"
    )


def test_transforms_keep_wildcards() -> None:
    """Whatever an edit writes on a wildcard cell, the cell is a wildcard in the result."""
    start = _blank(3)
    start[10:12, 10:12] = WILDCARD
    result = FillRect(5, 20, 5, 20, color=9).transform(start)
    assert (result[10:12, 10:12] == WILDCARD).all() and result[5, 5] == 9


def test_a_wildcard_in_the_target_is_a_cell_any_value_suits() -> None:
    """A target's wildcard asks for nothing: a rectangle with one in it is one fill, not a fill and a cell left
    out. ``explain`` refuses one, since its block ends in an assert against the target."""
    target = _blank()
    target[10:14, 10:14] = 5
    target[11, 12] = WILDCARD
    assert _parts(_plain().decompose(_blank(), target)) == (FillRect(10, 14, 10, 14, color=5),)
    with pytest.raises(ValueError, match="target must hold colours 0-15"):
        _plain().explain([Source("start", _blank())], target, "cur")


def test_the_search_is_deterministic_and_its_blocks_run() -> None:
    """The same frames give the same block, on every pair of the recorded frames, and every block passes its assert."""
    for name_a, name_b in itertools.permutations(FRAMES, 2):
        a, b = FRAMES[name_a], FRAMES[name_b]
        first = explain(a, b, "before", "after", "board")
        assert first == explain(a, b, "before", "after", "board")
        namespace = block_namespace(before=a.copy(), after=b)
        exec(first, namespace)
        assert np.array_equal(namespace["board"], b)


def test_explain_writes_the_names_given() -> None:
    """The block is written against the caller's expressions."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    code = explain(start, target, "history[-2].current_frame.grid", "current_frame.grid")
    assert code.startswith("g = history[-2].current_frame.grid.copy()")
    assert code.endswith("assert (g == current_frame.grid).all()")


def test_explain_chooses_by_cost_plus_handicap() -> None:
    """The cheapest source wins once each carries its handicap; every source's own cost comes back, in order."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    near = target.copy()
    near[0, 0] = (int(target[0, 0]) + 1) % 16
    own_costs = [_plain().decompose(near, target).bit_cost(), _plain().decompose(start, target).bit_cost()]
    plain = _plain().explain([Source("near", near), Source("start", start)], target, "cur")
    assert plain.source is not None and plain.source.name == "near"
    assert plain.costs == own_costs and plain.cost == own_costs[0]
    handicapped = _plain().explain(
        [Source("near", near, handicap=own_costs[1] - own_costs[0] + 1), Source("start", start)], target, "cur"
    )
    assert handicapped.source is not None and handicapped.source.name == "start" and handicapped.costs == own_costs
    assert handicapped.code == explain(start, target, "start", "cur")


def test_explain_breaks_a_tie_toward_the_earlier_source() -> None:
    """Two sources at the same cost plus handicap: the first given wins."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    first = _plain().explain([Source("one", start), Source("two", start.copy())], target, "cur")
    assert first.source is not None and first.source.name == "one"


def test_explain_flags_a_script_over_budget_and_still_writes_it() -> None:
    """``max_bits`` flags the chosen script, on its cost without the handicap; the block is written either way."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    cost = _plain().decompose(start, target).bit_cost()
    decomposer = _plain()
    decomposer.max_bits = cost
    within = decomposer.explain([Source("start", start, handicap=1000)], target, "cur")
    assert not within.over_budget
    decomposer.max_bits = cost - 1
    over = decomposer.explain([Source("start", start)], target, "cur")
    assert over.over_budget and over.code == within.code


def test_explain_out_of_time_with_nothing_found_is_over_budget_with_no_script() -> None:
    """A call that reaches its deadline before any search comes back chooses nothing and says it timed out; the
    caller's deadline ends it before the decomposer's own limit, and a call within both is untouched."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    sources = [Source("start", start)]
    decomposer = _plain()
    decomposer.time_limit_s = 0
    late = decomposer.explain(sources, target, "cur")
    assert late.timed_out and late.over_budget
    assert late.source is None and late.transform is None and late.code is None and late.cost is None
    past = _plain().explain(sources, target, "cur", deadline=time.monotonic())
    assert past.timed_out and past.code is None
    in_time = _plain().explain(sources, target, "cur", deadline=time.monotonic() + 3600)
    assert not in_time.timed_out and in_time.code == explain(start, target, "start", "cur")


def test_explain_out_of_time_keeps_the_best_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clock passes the deadline once the first source's search is back: its script is chosen and written as
    any other, though the second source, one cell off the target and far cheaper, was never searched; the call
    says it timed out."""
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    near = target.copy()
    near[0, 0] = (near[0, 0] + 1) % 16
    whole = _plain().explain([Source("start", start)], target, "cur")
    now = [0.0]
    monkeypatch.setattr(decompose_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    search = Decomposer.decompose_within

    def then_late(self: Decomposer, a: Grid, b: Grid, give_up: Callable[[float], bool]) -> Transform | None:
        found = search(self, a, b, give_up)
        now[0] = 1e9
        return found

    monkeypatch.setattr(Decomposer, "decompose_within", then_late)
    late = _plain().explain([Source("start", start), Source("near", near)], target, "cur")
    assert late.timed_out and late.source is not None and late.source.name == "start"
    assert late.code == whole.code and late.costs == [whole.cost, None]
    assert late.over_budget == whole.over_budget


def test_explain_needs_a_source() -> None:
    """No source, no script."""
    with pytest.raises(ValueError, match="at least one source"):
        _plain().explain([], FRAMES["level1_start"], "cur")


def test_correlation_counts_exactly() -> None:
    """The FFT count of wrong cells a shift makes right equals the brute-force count, for every shift."""
    rng = np.random.default_rng(9)
    start = rng.integers(0, 4, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    target = rng.integers(0, 4, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    search = _Search(start, target, _plain())
    wrong = start != target
    counts = {shift[1]: shift[2] for shift in search.correlate(0, search.spectra[0], search.planes(wrong), 50)}
    for (row, col), count in counts.items():
        expected = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if wrong[r, c] and 0 <= r - row < GRID_SIZE and 0 <= c - col < GRID_SIZE:
                    expected += int(start[r - row, c - col] == target[r, c])
        assert count == expected


def test_best_rectangles_are_maximal() -> None:
    """The rectangle search finds the maximum sum a brute-force scan finds, on random maps."""
    rng = np.random.default_rng(11)
    maps = [rng.integers(-3, 3, (7, 9)).astype(np.int32) for _ in range(6)]
    for gain, found in zip(maps, _best_rectangles(maps)):
        best = max(
            int(gain[r0:r1, c0:c1].sum())
            for r0 in range(7)
            for r1 in range(r0 + 1, 8)
            for c0 in range(9)
            for c1 in range(c0 + 1, 10)
        )
        if best <= 0:
            assert found is None
        else:
            assert found is not None
            assert found.gain == best == int(gain[found.row0 : found.row1, found.col0 : found.col1].sum())


def test_a_frame_of_the_wrong_shape_is_refused() -> None:
    """Both frames are 64x64."""
    with pytest.raises(ValueError, match="a must be a 64x64 frame"):
        _plain().decompose(np.zeros((8, 8), dtype=np.int8), _blank())


def test_a_swap_leaves_the_more_common_colour_behind() -> None:
    """Two one-colour blocks trading places read as either one moving; the frame's background is what stays."""
    start = _blank(0)
    start[10:13, 10:13] = 9
    target = _blank(0)
    target[10:13, 16:19] = 9
    result = _plain().decompose(start, target)
    assert _parts(result) == (BlockCopy(10, 13, 16, 19, src_row=10, src_col=10, clear=0),)
    _check(start, target, result)
    # The same frames with the colours exchanged: on a blue field the white tile is the mover, blue is left behind.
    start, target = 9 - start, 9 - target
    result = _plain().decompose(start, target)
    assert _parts(result) == (BlockCopy(10, 13, 16, 19, src_row=10, src_col=10, clear=9),)
    _check(start, target, result)


def test_a_strip_copy_shrinks_to_a_move() -> None:
    """A sprite in a red column moves down by its own height: a copy of the four-row strip shifted by two
    writes the same cells as a move of the sprite leaving red, and the move is what is emitted."""
    start = _blank(0)
    start[20:40, 22:24] = 8
    start[30:32, 22:24] = 14
    target = _blank(0)
    target[20:40, 22:24] = 8
    target[32:34, 22:24] = 14
    result = _plain().decompose(start, target)
    assert _parts(result) == (BlockCopy(32, 34, 22, 24, src_row=30, src_col=22, clear=8),)
    _check(start, target, result)


@pytest.mark.parametrize(("rows", "cols"), [(0, -1), (-1, 0), (0, 1), (1, 0), (0, -2), (-2, 0)])
def test_priced_by_distance_a_small_move_is_the_whole_block_moving(rows: int, cols: int) -> None:
    """A one-colour block moved by less than its size reads two ways at one price: the block moving, and its
    trailing strip jumping across it. Priced by the distance travelled, the block moves, whichever way it went;
    unpriced, left and up read as the strip."""
    start = _blank(0)
    start[20:23, 20:23] = 5
    target = _blank(0)
    target[20 + rows : 23 + rows, 20 + cols : 23 + cols] = 5
    result = _plain(move_distance_cost=1 / 128).decompose(start, target)
    _check(start, target, result)
    (move,) = [part for part in _parts(result) if isinstance(part, BlockCopy)]
    assert move.block_shape == (3, 3) and move.distance == abs(rows) + abs(cols)
    assert result.bit_cost() == BlockCopy.COST + move.distance / 128
    if rows < 0 or cols < 0:
        (unpriced,) = [part for part in _parts(_plain().decompose(start, target)) if isinstance(part, BlockCopy)]
        assert unpriced.distance == 3, "the reading the price exists to replace"


def test_priced_by_distance_a_script_is_never_costlier_than_the_unpriced_search() -> None:
    """The price can send a greedy step elsewhere; the unpriced runs are made too, so the script that comes back
    costs no more, priced by distance, than the unpriced search's own script priced the same way."""
    rate = 1 / 128
    frames = list(FRAMES.values())
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    pairs = [*itertools.permutations(frames, 2), (frames[0], noise)]
    for start, target in pairs:
        priced = _plain(move_distance_cost=rate).decompose(start, target)
        unpriced = _plain().decompose(start, target).priced(rate)
        _check(start, target, priced)
        assert priced.bit_cost() <= unpriced.bit_cost() + 1e-9


def test_turned_copies_do_not_displace_plain_moves() -> None:
    """lf52: two identical sprites move six columns along maze corridors. With every variant in play the first
    step is one sprite turned onto the other's new place, which fixes the most cells and leads to 144; plain
    shifts alone reach 80, and the cheaper run is what is returned."""
    start, target = LF52["level2_before_right"], LF52["level2_after_right"]
    result = _plain().decompose(start, target)
    _check(start, target, result)
    assert result.bit_cost() <= 80
    assert "BlockCopy*" not in _kinds(result)


def test_a_decomposer_pickled_before_its_options_reads_back_with_them_off() -> None:
    """A run's solver is unpickled to replay it or read its corpus; a decomposer pickled before an option existed
    holds no value for it, and reads back with it off rather than with today's default."""
    old = Decomposer()
    for name in OPTIONS_OFF:
        delattr(old, name)
    restored = pickle.loads(pickle.dumps(old))
    assert {name: getattr(restored, name) for name in OPTIONS_OFF} == OPTIONS_OFF
    assert pickle.loads(pickle.dumps(Decomposer())) == Decomposer()


def test_a_settled_frame_looks_ahead_and_an_animation_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """``lookahead`` serves ``explain``; ``explain_animation`` searches with ``animation_lookahead`` in its place."""
    seen: list[int] = []
    setup = decompose_module._Search.__init__  # pyright: ignore[reportPrivateUsage]

    def spy(self: Any, start: Grid, target: Grid, options: Decomposer, *rest: Any, **named: Any) -> None:
        seen.append(options.lookahead)
        setup(self, start, target, options, *rest, **named)

    monkeypatch.setattr(decompose_module._Search, "__init__", spy)  # pyright: ignore[reportPrivateUsage]
    start, target = FRAMES["level1_start"], FRAMES["level1_after_right"]
    decomposer = Decomposer(lookahead=2, animation_lookahead=1)
    decomposer.explain([Source("start", start)], target, "cur")
    assert seen and set(seen) == {2}
    seen.clear()
    decomposer.explain_animation(start, [target], "before", "frames[{k}]")
    assert seen and set(seen) == {1}
