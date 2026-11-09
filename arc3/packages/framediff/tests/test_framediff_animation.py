"""``Decomposer.explain_animation``: one block that rebuilds an animation frame by frame.

What is pinned: the block's shape on an animation that uses every way a
frame can be built, which candidate wins (the cheapest; at a tie the most
recent, the frame before the move last), exact repeats from any distance,
the budget per frame, and that the block runs under the caller's names. The
search runs with its options off (``OPTIONS_OFF``), so these pin the plain
pieces; the options' own tests say what each changes.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import pytest

import framediff.animation as animation_module
import framediff.decompose as decompose_module
from framediff import GRID_SIZE, Decomposer, Grid, Transform, block_namespace
from framediff.animation import STRIDES, AnimationFrame
from framediff.decompose import OPTIONS_OFF


def _plain() -> Decomposer:
    """The search with its options off."""
    return Decomposer(**OPTIONS_OFF)


def _blank(color: int = 0) -> Grid:
    """A frame of one colour."""
    return np.full((GRID_SIZE, GRID_SIZE), color, dtype=np.int8)


def _animation() -> tuple[Grid, list[Grid]]:
    """A frame before the move and five animation frames: a square that flashes, a noise frame, a scroll of it."""
    before = _blank()
    first = before.copy()
    first[40:43, 40:43] = 5
    second = first.copy()
    second[40:43, 40:43] = 8
    second[12, 7] = 3
    noise = np.random.default_rng(0).integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    scrolled = noise.copy()
    scrolled[30:34] = noise[31:35]
    return before, [first, second, first.copy(), noise, scrolled]


def test_every_way_a_frame_is_built() -> None:
    """From the frame before, from the previous frame, the same as an earlier frame, and taken as it is."""
    before, frames = _animation()
    found = _plain().explain_animation(before, frames, "history[3].frame.grid", "history[4].animation_frames[{k}].grid")
    assert found.code == "\n".join(
        [
            "anim = {}",
            "",
            "# Frame 0: from history[3].frame.grid",
            "g = history[3].frame.grid.copy()",
            "g[40:43, 40:43] = 5",
            "anim[0] = g.copy()",
            "",
            "# Frame 1: from frame 0",
            "g[40:43, 40:43] = 8",
            "g[12, 7] = 3",
            "anim[1] = g.copy()",
            "",
            "# Frame 2: the same as frame 0",
            "g = anim[0].copy()",
            "anim[2] = g.copy()",
            "",
            "# Frame 3: no concise description, taken as it is",
            "g = history[4].animation_frames[3].grid.copy()",
            "anim[3] = g.copy()",
            "",
            "# Frame 4: from frame 3",
            "g[30:34, 0:64] = anim[3][31:35, 0:64]  # copied up 1 (colors: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)",
            "anim[4] = g.copy()",
            "",
            "assert all((frame == history[4].animation_frames[k].grid).all() for k, frame in anim.items())",
        ]
    )
    assert [frame.source for frame in found.frames] == [
        "history[3].frame.grid",
        "anim[0]",
        "anim[0]",
        None,
        "anim[3]",
    ]
    assert [frame.over_budget for frame in found.frames] == [False, False, False, True, False]


def test_out_of_time_with_nothing_found_every_frame_is_taken_as_it_is() -> None:
    """A call that reaches its deadline before any frame is built says so and takes every frame as it is; the
    block still runs."""
    before, frames = _animation()
    decomposer = _plain()
    decomposer.time_limit_s = 0
    late = decomposer.explain_animation(before, frames, "before", "frames[{k}]")
    assert late.timed_out and late.stride == 1
    assert [frame.over_budget for frame in late.frames] == [True] * len(frames)
    namespace = block_namespace(before=before.copy())
    namespace["frames"] = frames
    exec(late.code, namespace)
    past = _plain().explain_animation(before, frames, "before", "frames[{k}]", deadline=time.monotonic())
    assert past.timed_out
    assert not _plain().explain_animation(before, frames, "before", "frames[{k}]").timed_out


def test_out_of_time_the_frames_built_by_then_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clock passes the deadline once frame 1 is built: frames 0 and 1 keep their scripts, the frames after
    are taken as they are, and the block still runs and says it timed out."""
    before, frames = _animation()
    whole = _plain().explain_animation(before, frames, "before", "frames[{k}]")
    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(animation_module, "time", clock)
    monkeypatch.setattr(decompose_module, "time", clock)
    choose = animation_module._choose  # pyright: ignore[reportPrivateUsage]

    def then_late_after_frame_1(*args: Any, **kwargs: Any) -> AnimationFrame:
        record = choose(*args, **kwargs)
        if record.index == 1:
            now[0] = 1e9
        return record

    monkeypatch.setattr(animation_module, "_choose", then_late_after_frame_1)
    late = _plain().explain_animation(before, frames, "before", "frames[{k}]")
    assert late.timed_out and late.stride == 1
    assert late.frames[:2] == whole.frames[:2] and all(frame.transform is not None for frame in late.frames[:2])
    assert [frame.transform for frame in late.frames[2:]] == [None] * (len(frames) - 2)
    namespace = block_namespace(before=before.copy())
    namespace["frames"] = frames
    exec(late.code, namespace)


def test_out_of_time_the_block_kept_fits_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clock passes the deadline once frame 1 is built, under a limit the two built frames fit and the
    three after them, taken as they are, would not: those three are dropped from the block's end."""
    before, frames = _animation()
    prefix = _plain().explain_animation(before, frames[:2], "before", "frames[{k}]")
    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(animation_module, "time", clock)
    monkeypatch.setattr(decompose_module, "time", clock)
    choose = animation_module._choose  # pyright: ignore[reportPrivateUsage]

    def then_late_after_frame_1(*args: Any, **kwargs: Any) -> AnimationFrame:
        record = choose(*args, **kwargs)
        if record.index == 1:
            now[0] = 1e9
        return record

    monkeypatch.setattr(animation_module, "_choose", then_late_after_frame_1)
    late = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=len(prefix.code) + 50)
    assert late.timed_out and late.code == prefix.code and [frame.index for frame in late.frames] == [0, 1]


def test_the_block_runs_under_the_callers_names() -> None:
    """Run against objects the expressions reach, the block rebuilds every frame."""
    before, frames = _animation()
    found = _plain().explain_animation(before, frames, "history[3].frame.grid", "history[4].animation_frames[{k}].grid")
    history = [SimpleNamespace(frame=SimpleNamespace(grid=before)) for _ in range(4)]
    history.append(SimpleNamespace(animation_frames=[SimpleNamespace(grid=frame) for frame in frames]))
    namespace = block_namespace()
    namespace["history"] = history
    exec(found.code, namespace)
    anim = cast("dict[int, Grid]", namespace["anim"])
    assert list(anim) == list(range(len(frames)))
    assert all(np.array_equal(built, frames[k]) for k, built in anim.items())


def test_an_animation_can_start_from_its_own_first_frame() -> None:
    """With no frame before it, the animation's first frame is the start: frame 0 is that frame, and the rest
    build on it."""
    _, frames = _animation()
    found = _plain().explain_animation(
        frames[0], frames, "history[0].animation_frames[0].grid", "history[0].animation_frames[{k}].grid"
    )
    assert "# Frame 0: the same as history[0].animation_frames[0].grid" in found.code
    namespace = block_namespace()
    namespace["history"] = [SimpleNamespace(animation_frames=[SimpleNamespace(grid=frame) for frame in frames])]
    exec(found.code, namespace)
    anim = cast("dict[int, Grid]", namespace["anim"])
    assert all(np.array_equal(built, frames[k]) for k, built in anim.items())


def test_the_cheapest_wins_and_the_most_recent_breaks_a_tie() -> None:
    """A flicker between two boards: each frame is the same as the one two back, not the one four back, and a
    board back at the frame before the move is named as the animation frame it repeats, not as that frame."""
    before = _blank()
    lit = before.copy()
    lit[5, 5] = 9
    flicker = [lit, before.copy(), lit.copy(), before.copy(), lit.copy()]
    found = _plain().explain_animation(before, flicker, "before", "frames[{k}]")
    assert [frame.source for frame in found.frames] == ["before", "before", "anim[0]", "anim[1]", "anim[2]"]
    assert [frame.cost for frame in found.frames] == [4, 0, 0, 0, 0]
    assert "# Frame 4: the same as frame 2\ng = anim[2].copy()" in found.code


def test_the_frame_before_the_move_wins_when_it_is_cheapest() -> None:
    """A detailed board wiped and restored: the restored frame is the same as the frame before the move."""
    before = np.random.default_rng(1).integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    wiped = before.copy()
    wiped[20:30, 20:30] = 0
    back = _plain().explain_animation(before, [wiped, before.copy()], "before", "frames[{k}]")
    assert back.frames[1].source == "before" and back.frames[1].cost == 0
    assert "# Frame 1: the same as before\ng = before.copy()\nanim[1] = g.copy()" in back.code


def test_an_exact_repeat_of_any_earlier_frame_is_a_source() -> None:
    """A bar that grows and then shrinks back: the shrinking frames are the same as the growing ones they repeat,
    however far back, though an edit of the frame before would take one line; a near repeat that old is not a
    source."""
    before = _blank()
    growing: list[Grid] = []
    for step in range(5):
        frame = before.copy()
        frame[10 : 14 + 4 * step, 30:40] = 7
        growing.append(frame)
    frames = [*growing, *(frame.copy() for frame in reversed(growing[:-1]))]
    found = _plain().explain_animation(before, frames, "before", "frames[{k}]")
    assert [frame.source for frame in found.frames[5:]] == ["anim[3]", "anim[2]", "anim[1]", "anim[0]"]
    assert "# Frame 8: the same as frame 0\ng = anim[0].copy()\nanim[8] = g.copy()" in found.code
    near = growing[0].copy()
    near[60, 60] = 3
    nearly = _plain().explain_animation(before, [*growing, near], "before", "frames[{k}]")
    assert "anim[0]" not in nearly.frames[5].costs


def test_the_budget_is_per_frame() -> None:
    """Under a lower budget each frame whose script exceeds it is taken as it is; the frames under it keep theirs."""
    before, frames = _animation()
    decomposer = _plain()
    decomposer.animation_max_bits = 8
    found = decomposer.explain_animation(before, frames, "before", "frames[{k}]")
    assert [frame.over_budget for frame in found.frames] == [False, True, False, True, True]
    assert "# Frame 1: no concise description, taken as it is\ng = frames[1].copy()" in found.code


def _sliding(count: int) -> tuple[Grid, list[Grid]]:
    """A square sliding right four columns a frame, over a blank board."""
    before = _blank()
    frames: list[Grid] = []
    for step in range(count):
        frame = before.copy()
        frame[10:14, 4 * step : 4 * step + 4] = 5
        frames.append(frame)
    return before, frames


def test_a_stride_builds_every_strideth_frame_from_the_frames_built() -> None:
    """At stride 3 the block builds frames 0, 3, 6 and 9 under their own indices, and runs."""
    before, frames = _sliding(11)
    found = _plain().explain_animation(before, frames, "before", "frames[{k}]", stride=3)
    assert found.stride == 3 and [frame.index for frame in found.frames] == [0, 3, 6, 9]
    assert "anim[3] = g.copy()" in found.code and "anim[1]" not in found.code
    namespace = block_namespace(before=before)
    namespace["frames"] = frames
    exec(found.code, namespace)
    assert list(namespace["anim"]) == [0, 3, 6, 9]


def test_without_a_stride_the_first_that_fits_the_limit_is_taken() -> None:
    """Each stride in turn until the block fits; with no limit, stride 1."""
    before, frames = _sliding(12)
    whole = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=None)
    assert whole.stride == 1
    exact = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=len(whole.code))
    assert exact.stride == 1 and exact.code == whole.code
    short = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=len(whole.code) - 1)
    assert short.stride == STRIDES[1] and len(short.code) < len(whole.code)


def test_a_block_is_abandoned_when_it_passes_the_limit_not_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """At a limit a few frames long, stride 1 stops searching at the frame that passes it."""
    before, frames = _sliding(16)
    searched: list[int] = []
    original = Decomposer.decompose_within

    def counting(
        self: Decomposer, a: npt.ArrayLike, b: npt.ArrayLike, give_up: Callable[[float], bool]
    ) -> Transform | None:
        """Count the frames searched toward."""
        searched.append(next(k for k, frame in enumerate(frames) if np.array_equal(frame, b)))
        return original(self, a, b, give_up)

    monkeypatch.setattr(Decomposer, "decompose_within", counting)
    found = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=300)
    first_pass = searched[: searched.index(0, 1)] if searched.count(0) > 1 else searched
    assert found.stride > 1 and max(first_pass) < len(frames) - 1


def test_the_last_stride_is_written_whole_and_a_given_stride_ignores_the_limit() -> None:
    """A limit no stride fits: the last stride, whole. A stride given: that stride, whole, whatever the limit."""
    before, frames = _sliding(13)
    last = _plain().explain_animation(before, frames, "before", "frames[{k}]", max_chars=10)
    assert last.stride == STRIDES[-1] and len(last.code) > 10
    given = _plain().explain_animation(before, frames, "before", "frames[{k}]", stride=1, max_chars=10)
    assert given.stride == 1 and len(given.frames) == len(frames)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_a_stride_is_a_whole_number_of_frames(stride: object) -> None:
    """Zero, negative, fractional and boolean strides are refused."""
    with pytest.raises(ValueError, match="stride must be a whole number"):
        _plain().explain_animation(_blank(), [_blank(1)], "before", "frames[{k}]", stride=stride)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("frames", "frame_name", "names", "message"),
    [
        ([], "frames[{k}]", {}, "at least one frame"),
        ([_blank()], "frames[0]", {}, "must hold {k}"),
        ([_blank()], "frames[{k}]", {"list_name": "g"}, "must differ"),
        ([_blank()], "frames[{k}]", {"board_name": "k"}, "plain name other than 'k'"),
    ],
)
def test_bad_arguments_are_refused(frames: list[Grid], frame_name: str, names: dict[str, str], message: str) -> None:
    """No frames, a frame name without its index, and names the block cannot use."""
    with pytest.raises(ValueError, match=message):
        _plain().explain_animation(
            _blank(),
            frames,
            "before",
            frame_name,
            list_name=names.get("list_name", "anim"),
            board_name=names.get("board_name", "g"),
        )
