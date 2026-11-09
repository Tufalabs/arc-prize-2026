"""Runs of an animation's frames written as one loop (``Decomposer.animation_patterns``, ``framediff.patterns``).

What is pinned: each kind of run found and written (a blink as a periodic copy, a bar growing as a stepped
edit, a sprite crossing a striped floor as a shape sliding over its background), a stride's loop stepping
with it, nothing written for a run under its minimum or one the frames break, the block running under the
caller's names, and nothing changed with the option off.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import framediff.animation as animation_module
import framediff.patterns as patterns_module
from framediff import GRID_SIZE, Decomposer, Grid, block_namespace
from framediff.animation import AnimationFrame
from framediff.decompose import OPTIONS_OFF
from framediff.patterns import Names, Periodic, Sliding

#: The search with its options off, and with the patterns alone on: what the patterns change, apart from the others.
PLAIN = Decomposer(**OPTIONS_OFF)
PATTERNS = dataclasses.replace(PLAIN, animation_patterns=True)


def _blank(color: int = 0) -> Grid:
    """A frame of one colour."""
    return np.full((GRID_SIZE, GRID_SIZE), color, dtype=np.int8)


def _bar(rows: int) -> Grid:
    """A bar of colour 7 at cols 10:20, ``rows`` rows tall from row 40."""
    frame = _blank()
    frame[40 : 40 + rows, 10:20] = 7
    return frame


SPRITE = np.array([[6, 0, 0], [6, 0, 0], [6, 6, 6]], dtype=np.int8)


def _on_stripes(col: int) -> Grid:
    """The L-shaped sprite at row 24, ``col``, over a floor whose columns alternate colours 1 and 2."""
    frame = _blank(5)
    frame[20:30, 5:40] = np.where(np.arange(5, 40) % 2 == 0, 1, 2)
    window = frame[24:27, col : col + 3]
    window[SPRITE > 0] = 6
    return frame


def test_a_blink_is_one_periodic_loop() -> None:
    lit = _blank()
    lit[10:13, 10:13] = 4
    frames = [lit, _blank()] * 4
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert [(run.kind, run.frames) for run in found.runs] == [("periodic", [2, 3, 4, 5, 6, 7])]
    assert "for k in range(2, 8):\n    anim[k] = anim[k - 2].copy()" in found.code
    assert len(found.code) < len(PLAIN.explain_animation(_blank(), frames, "before", "frames[{k}]").code)


def test_a_bar_growing_a_row_a_frame_is_one_stepped_loop() -> None:
    frames = [_bar(rows) for rows in range(1, 9)]
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert [(run.kind, run.frames) for run in found.runs] == [("stepped", list(range(1, 8)))]
    assert "for k in range(1, 8):\n    g[k + 40:k + 41, 10:20] = 7\n    anim[k] = g.copy()" in found.code


def test_a_stride_steps_the_loop_with_it() -> None:
    frames = [_bar(rows) for rows in range(1, 13)]
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]", stride=2)
    (run,) = found.runs
    assert run.kind == "stepped" and run.frames == [2, 4, 6, 8, 10]
    assert "for k in range(2, 11, 2):" in found.code


def test_a_sprite_crossing_stripes_slides_over_its_background() -> None:
    """Over a floor that does not repeat at the step, each frame's own piece differs; as a shape over the
    background, the frames are one loop."""
    frames = [_on_stripes(11 + k) for k in range(8)]
    found = PATTERNS.explain_animation(_on_stripes(10), frames, "before", "frames[{k}]")
    assert [(run.kind, run.frames) for run in found.runs] == [("sliding", list(range(8)))]
    assert "shape = before[24:27, 10:13].copy()" in found.code
    assert "g[24:27, k + 11:k + 14][mask] = shape[mask]" in found.code
    assert found.built == list(range(8))


def test_a_stride_its_pieces_overrun_is_kept_when_its_runs_fit() -> None:
    """A long blink: its pieces pass the length limit, its one loop does not. The stride is judged with its
    runs before it is given up, so it is built every frame."""
    lit = _blank()
    lit[10:13, 10:13] = 4
    frames = [lit, _blank()] * 20
    plain = PLAIN.explain_animation(_blank(), frames, "before", "frames[{k}]", max_chars=None)
    looped = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]", max_chars=None)
    limit = len(looped.code) + 10
    assert len(plain.code) > limit
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]", max_chars=limit)
    assert found.stride == 1 and found.code == looped.code


def test_a_stride_no_run_brings_under_the_limit_is_given_up_before_it_is_built_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames nothing describes, each taken as it is (the search stubbed to say so at once): no run can bring
    stride 1 under a tight limit, so it is given up a few frames past the limit rather than searched to its
    end."""
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8) for _ in range(30)]
    searched: list[int] = []

    def as_it_is(*args: Any, **kwargs: Any) -> AnimationFrame:
        k: int = args[4]
        searched.append(k)
        return AnimationFrame(k, None, None, None, {})

    monkeypatch.setattr(animation_module, "_choose", as_it_is)
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]", max_chars=600)
    first_stride = searched[: searched.index(0, 1)]
    assert found.stride > 1 and len(first_stride) < 10


def test_a_sprite_easing_across_stripes_slides_along_a_path() -> None:
    """The sprite speeds up and slows down (ls20 in the reference run's corpus): no velocity, but one shape over
    a background that stays put, so the frames are one loop over the places it reaches."""
    cols = [11, 13, 17, 22, 24, 25]
    frames = [_on_stripes(col) for col in cols]
    found = PATTERNS.explain_animation(_on_stripes(10), frames, "before", "frames[{k}]")
    (run,) = found.runs
    assert run.kind == "sliding" and run.frames == list(range(6)) and cast("Sliding", run).velocity is None
    assert "# Frames 0-5: the shape below moving along a path over a background that stays put" in found.code
    assert f"for k, col in zip(range(6), {cols}):" in found.code
    assert "g[24:27, col:col + 3][mask] = shape[mask]" in found.code


def test_a_path_in_both_directions_lists_its_places() -> None:
    def frame(row: int, col: int) -> Grid:
        board = _blank(5)
        board[20:40, 5:40] = np.where(np.arange(5, 40) % 2 == 0, 1, 2)
        board[row : row + 3, col : col + 3][SPRITE > 0] = 6
        return board

    places = [(25, 11), (27, 12), (26, 15), (30, 16)]
    found = PATTERNS.explain_animation(frame(24, 10), [frame(*place) for place in places], "before", "frames[{k}]")
    (run,) = found.runs
    assert run.kind == "sliding" and cast("Sliding", run).places == tuple(places)
    assert f"for k, (row, col) in zip(range(4), {places}):" in found.code


def test_a_path_that_opens_at_one_velocity_offers_that_opening_too() -> None:
    """Four frames at one velocity, then a jump: the velocity run over the opening is offered beside the whole
    path, and the shorter block is chosen."""
    cols = [11, 12, 13, 14, 20]
    frames = [_on_stripes(col) for col in cols]
    start = _on_stripes(10)
    records = PLAIN.explain_animation(start, frames, "before", "frames[{k}]").frames
    runs = [run for _, _, run in patterns_module._sliding(start, frames, records)]  # pyright: ignore[reportPrivateUsage]
    assert sorted((len(run.frames), cast("Sliding", run).velocity) for run in runs) == [(4, (0, 1)), (5, None)]


def test_under_three_frames_nothing_is_a_run() -> None:
    frames = [_bar(rows) for rows in range(1, 4)]
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert not any(run.kind == "stepped" for run in found.runs)


def test_a_frame_that_breaks_the_step_ends_the_run() -> None:
    """A bar that grows a row a frame and then jumps: the run stops at the jump, and the block still holds."""
    frames = [_bar(rows) for rows in (1, 2, 3, 4, 5, 9, 10)]
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert [(run.kind, run.frames) for run in found.runs] == [("stepped", [1, 2, 3, 4])]


def test_a_background_that_changes_is_no_slide() -> None:
    """A second change under way beside the sprite: no frame is the background with the shape on top."""
    frames: list[Grid] = []
    for k in range(6):
        frame = _on_stripes(11 + k)
        frame[0, k] = 9
        frames.append(frame)
    found = PATTERNS.explain_animation(_on_stripes(10), frames, "before", "frames[{k}]")
    assert not any(run.kind == "sliding" for run in found.runs)


def test_a_box_holding_more_of_the_shapes_colour_starts_no_slide() -> None:
    """The loop picks the shape by colour inside its box; a still cell of that colour in the box, apart from the
    shape, would be picked too and the block would not hold. No slide starts from that box; one may start a
    frame later, once the box has left the cell behind."""

    def frame(col: int) -> Grid:
        board = _blank(5)
        board[20:30, 5:40] = np.where(np.arange(5, 40) % 2 == 0, 1, 2)
        board[24, 22] = 6  # still, inside the sprite's first box; the sprite moves away and never touches it
        window = board[24:27, col : col + 3]
        window[SPRITE > 0] = 6
        return board

    found = PATTERNS.explain_animation(frame(20), [frame(19 - k) for k in range(6)], "before", "frames[{k}]")
    assert all(cast("Sliding", run).start >= 0 for run in found.runs if run.kind == "sliding")


def test_the_loops_run_under_the_callers_names() -> None:
    frames = [_on_stripes(11 + k) for k in range(8)]
    before = _on_stripes(10)
    found = PATTERNS.explain_animation(before, frames, "history[3].frame.grid", "history[4].animation_frames[{k}].grid")
    history = [SimpleNamespace(frame=SimpleNamespace(grid=before)) for _ in range(4)]
    history.append(SimpleNamespace(animation_frames=[SimpleNamespace(grid=frame) for frame in frames]))
    namespace = block_namespace()
    namespace["history"] = history
    exec(found.code, namespace)
    anim = cast("dict[int, Grid]", namespace["anim"])
    assert all(np.array_equal(built, frames[k]) for k, built in anim.items())


def test_a_step_after_a_blink_starts_from_the_frame_before_it() -> None:
    """A periodic run fills the dict and leaves the board behind; the stepped run after it edits the board, which
    has to hold the frame before it first (escape_velocity in the reference run's corpus)."""
    lit = _blank()
    lit[10:13, 10:13] = 4
    growing: list[Grid] = []
    for rows in range(1, 9):
        frame = _bar(rows)
        frame[10:13, 10:13] = 4
        growing.append(frame)
    frames = [lit, _blank(), lit, _blank(), lit, *growing]
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert [(run.kind, run.frames[0]) for run in found.runs] == [("periodic", 2), ("stepped", 5)]
    assert not found.runs_dropped


def test_a_block_its_loops_break_is_written_without_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run is only a shorter way to write frames the pieces already build: should its loop be wrong, the block
    is written without runs, still holds, and says the runs were dropped."""
    lit = _blank()
    lit[10:13, 10:13] = 4
    frames = [lit, _blank()] * 4

    def wrong(self: Periodic, names: Names) -> list[str]:
        return ["", "for k in range(2, 8):", f"    {names.dict}[k] = {names.dict}[k - 1].copy()"]

    monkeypatch.setattr(Periodic, "lines", wrong)
    found = PATTERNS.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert found.runs_dropped and found.runs == [] and "for k in" not in found.code


def test_off_nothing_changes() -> None:
    frames = [_bar(rows) for rows in range(1, 9)]
    found = PLAIN.explain_animation(_blank(), frames, "before", "frames[{k}]")
    assert found.runs == [] and "for k in" not in found.code
