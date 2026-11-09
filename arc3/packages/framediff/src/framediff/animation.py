"""An animation explained frame by frame, as one block that rebuilds it.

``Decomposer.explain_animation`` takes the frame before the move and the
animation's frames, and writes a block that builds the animation's frames
into a dict by frame index, one piece per frame, and ends in one assert
over the dict::

    anim = {}

    # Frame 0: from history[3].frame.grid
    g = history[3].frame.grid.copy()
    g[40:43, 40:43] = 5
    anim[0] = g.copy()

    # Frame 1: from frame 0
    g[40:43, 40:43] = 8
    anim[1] = g.copy()

    # Frame 2: the same as frame 0
    g = anim[0].copy()
    anim[2] = g.copy()

    # Frame 3: no concise description, taken as it is
    g = history[4].animation_frames[3].grid.copy()
    anim[3] = g.copy()

    assert all((frame == history[4].animation_frames[k].grid).all() for k, frame in anim.items())

With a stride, only every stride-th frame is built (0, stride, 2 x
stride, ...), each from the frames built before it; the dict keeps the
frames' own indices. Without one, the stride is the first of ``STRIDES``
whose block stays within the character limit: a block is abandoned as soon
as it passes the limit, and the last stride tried is written whole.

The board holds the newest built frame throughout, so a frame built from
the one before it needs no line to start, and any other source starts with
a copy of it; a piece reads its source under the source's own name, never
the board it is editing. A frame that repeats an earlier built frame
exactly is a copy of it, the most recent such, or of the frame before the
move if it repeats only that: a copy costs nothing, so nothing is
searched. Any other frame's candidates are the built frames just before
it, most recent first, then the frame before the move; the cheapest script
within ``animation_max_bits`` wins, the earlier candidate at a tie. The
budget is per frame: a frame no candidate fits is taken as it is, and the
frames after it still build on it. A candidate's search is abandoned once
its script so far is over budget or no cheaper than the best so far, so a
frame nothing describes costs a bounded search, not a full one per
candidate.

Before the block is returned it is run under plain names against the
frames it was given, and every script in it has passed its own self-test.

One clock covers the whole call, every stride tried included: a call still
searching at its deadline (``Decomposer.deadline``) says it ``timed_out``
and keeps the best it found by then, at the stride under way: each frame
searched before the deadline as it was built, the frame under way built
from the best script its search had found, and the frames after it taken
as they are, as many as fit the stride's length limit. Runs are not looked
for then.
"""

from __future__ import annotations

import dataclasses
import keyword
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt

from framediff.grid import Grid
from framediff.patterns import MIN_RUN, Describe, Run, choose
from framediff.transforms import NoOp, Transform, _check_names, block_namespace, with_colors

if TYPE_CHECKING:
    from framediff.decompose import Decomposer

#: The loop variable of the closing assert, which the frame name template is written with.
_INDEX = "k"

#: How many built frames just before a frame are searched for an edit that builds it (2026-09-24);
#: an exact repeat is looked for among all built frames.
LOOKBACK = 3

#: The strides tried in turn when none is given, until a block stays within the character limit
#: (2026-09-25).
STRIDES = (1, 2, 3, 4, 6)

#: The character limit on a block whose stride is chosen: long enough for a short animation frame by
#: frame, short enough to sit in a prompt beside the frame delta (2026-09-25).
MAX_CHARS = 5000


@dataclass
class AnimationFrame:
    """How one animation frame is built, or that it is taken as it is."""

    #: The frame's index in the animation.
    index: int
    #: The chosen source's expression, the frame before the move or the dict entry of an earlier frame;
    #: None when the frame is taken as it is.
    source: str | None
    #: The index of the earlier animation frame it is built from; None for the frame before the move, or
    #: when the frame is taken as it is.
    earlier: int | None
    #: The chosen script; None when the frame is taken as it is.
    transform: Transform | None
    #: Each candidate searched, by expression, and its script's cost; None for a search abandoned over
    #: budget or once it could not win. An exact repeat is the one entry, at no cost; a candidate whose
    #: frame repeats an earlier candidate's is not searched.
    costs: dict[str, float | None]

    @property
    def over_budget(self) -> bool:
        """Whether no candidate's script fits the budget, so the frame is taken as it is."""
        return self.transform is None

    @property
    def cost(self) -> float | None:
        """The chosen script's cost; None when the frame is taken as it is."""
        return None if self.transform is None else self.transform.bit_cost()


@dataclass
class AnimationDecomposition:
    """The block that rebuilds an animation, the stride it was built at, and how each built frame is built."""

    code: str
    stride: int
    frames: list[AnimationFrame]
    #: Whether the call reached its deadline, and so took the frames it had not reached as they are.
    timed_out: bool = False
    #: The runs written as loops (``Decomposer.animation_patterns``), in order.
    runs: list[Run] = field(default_factory=lambda: list[Run]())
    #: Whether runs were chosen but the block holding them failed its check, and was written without them.
    runs_dropped: bool = False

    @property
    def built(self) -> list[int]:
        """The frames the block builds rather than takes as they are: by a piece of their own, or in a run."""
        in_runs = {index for run in self.runs for index in run.frames}
        return [frame.index for frame in self.frames if frame.transform is not None or frame.index in in_runs]


def explain_animation(
    decomposer: Decomposer,
    before: npt.ArrayLike,
    frames: Sequence[npt.ArrayLike],
    before_name: str,
    frame_name: str,
    *,
    stride: int | None = None,
    max_chars: int | None = MAX_CHARS,
    list_name: str = "anim",
    board_name: str = "g",
    deadline: float | None = None,
) -> AnimationDecomposition:
    """The block that rebuilds ``frames``, an animation that followed ``before``.

    ``before_name`` is the expression the block reads the frame before the
    move under; for an animation with no frame before it, its own first
    frame, which the block then takes as it is. ``frame_name`` is the expression of one animation frame with
    ``{k}`` where its index goes, such as
    ``history[4].animation_frames[{k}].grid``: a frame taken as it is is
    read under it, and the closing assert loops over it. ``stride`` builds
    every stride-th frame; None tries ``STRIDES`` in turn, each block
    abandoned once it passes ``max_chars`` characters, and writes the last
    whole whatever its length. With a stride given, the block is written
    whole and ``max_chars`` is not consulted. ``list_name`` is the dict the
    block builds and ``board_name`` the board it edits. ``deadline`` ends the
    call earlier than the decomposer's own limit (``Decomposer.deadline``).
    """
    from framediff.decompose import _frame

    if not frames:
        raise ValueError("an animation needs at least one frame")
    if stride is not None and (isinstance(stride, bool) or not isinstance(stride, int) or stride < 1):  # pyright: ignore[reportUnnecessaryIsInstance]  # callers pass what the model typed
        raise ValueError(f"stride must be a whole number of frames, 1 or more, got {stride!r}")
    if "{k}" not in frame_name:
        raise ValueError(f"frame_name must hold {{k}} where the frame index goes, got {frame_name!r}")
    for role, name in (("list_name", list_name), ("board_name", board_name)):
        if not name.isidentifier() or keyword.iskeyword(name) or name in (_INDEX, _FRAME):
            raise ValueError(f"{role}={name!r} must be a plain name other than {_INDEX!r} and {_FRAME!r}")
    if list_name == board_name:
        raise ValueError(f"list_name and board_name must differ, got {list_name!r} for both")
    # Checked as the closing assert reads it: the start may be one of the animation's own frames.
    _check_names(before_name, frame_name.format(k=_INDEX), board_name)
    start = _frame(before, "before", wildcards=False)
    grids = [_frame(frame, f"frame {k}", wildcards=False) for k, frame in enumerate(frames)]
    names = _Names(before_name, frame_name, list_name, board_name)
    end = decomposer.deadline(deadline)
    decomposer = dataclasses.replace(decomposer, lookahead=decomposer.animation_lookahead)

    describe = _describer(decomposer, end)
    strides = STRIDES if stride is None else (stride,)
    records: list[AnimationFrame] | None = None
    runs: list[Run] = []
    attempt = strides[0]
    limit: int | None = None
    timed_out = False
    for attempt in strides:
        limit = max_chars if stride is None and attempt != strides[-1] else None
        runs = []
        try:
            records = _build(decomposer, start, grids, attempt, names, limit, end, describe)
        except _OutOfTime as stopped:
            records, timed_out = stopped.records, True
        if records is not None and decomposer.animation_patterns and not timed_out:
            runs = _runs(start, grids, records, names, attempt, describe)
            if limit is not None and len(_write(records, names, runs)) > limit:
                records = None
        if records is not None:
            break
    assert records is not None, "the last stride is built without a limit"
    if timed_out and limit is not None:
        # The clock stopped a stride that had a limit: its block is kept as far as it fits, frames dropped from
        # its end, those taken as they are first.
        while len(records) > 1 and len(_write(records, names)) > limit:
            records.pop()

    plain_names = _Names("before", "frames[{k}]", list_name, board_name)
    dropped = False
    if runs and not _rebuilds(_write(records, plain_names, runs), start, grids):
        # A run is only ever a shorter way to write frames the pieces already build: a block its loops break is
        # written without them, and says so, rather than fail.
        runs, dropped = [], True
    plain = _write(records, plain_names, runs)
    if not _rebuilds(plain, start, grids):
        raise AssertionError(f"the animation block does not rebuild its frames:\n{plain}")
    return AnimationDecomposition(_write(records, names, runs), attempt, records, timed_out, runs, dropped)


def _rebuilds(plain: str, start: Grid, grids: list[Grid]) -> bool:
    """Whether a block written under plain names runs and rebuilds ``grids`` from ``start``."""
    namespace = block_namespace(before=start.copy())
    namespace["frames"] = grids
    try:
        exec(plain, namespace)
    except AssertionError:
        return False
    return True


class _OutOfTime(Exception):
    """A build reached the call's deadline; ``records`` holds every frame of its stride, those it had not
    reached taken as they are."""

    def __init__(self, records: list[AnimationFrame]) -> None:
        super().__init__("the build reached the call's deadline")
        self.records = records


#: The loop variable of the closing assert holding each built frame.
_FRAME = "frame"


@dataclass
class _Names:
    """The expressions and names a block is written against."""

    before: str
    frame: str
    dict: str
    board: str


def _build(
    decomposer: Decomposer,
    start: Grid,
    grids: list[Grid],
    stride: int,
    names: _Names,
    limit: int | None,
    end: float,
    describe: Describe | None = None,
) -> list[AnimationFrame] | None:
    """How every stride-th frame is built; None as soon as the block written so far passes ``limit``.
    Raises :class:`_OutOfTime` once the clock passes ``end``, with the frames built by then and the rest taken
    as they are."""
    records: list[AnimationFrame] = []
    notes: set[str] = set()
    # The block as _write joins it: notes and the opening line, each piece after a newline, then a blank line
    # and the closing assert.
    length = len(_opening(names)) + 2 + len(_closing(names))
    for k in range(0, len(grids), stride):
        target = grids[k]
        built = [record.index for record in records]
        record = _choose(decomposer, start, grids, target, k, built, names, end)
        records.append(record)
        if time.monotonic() >= end:
            # The frame under way keeps the best script its search found before the clock stopped it.
            raise _OutOfTime(
                records + [AnimationFrame(j, None, None, None, {}) for j in range(k + stride, len(grids), stride)]
            )
        if limit is not None:
            fresh = [] if record.transform is None else [n for n in record.transform._notes() if n not in notes]  # pyright: ignore[reportPrivateUsage]
            notes.update(fresh)
            length += sum(len(note) + 1 for note in fresh) + len("\n".join(_piece(record, built, names))) + 1
            if length > limit and not (
                decomposer.animation_patterns and _may_fit(start, grids, records, names, stride, limit, describe)
            ):
                return None
    return records


def _runs(
    start: Grid,
    grids: list[Grid],
    records: list[AnimationFrame],
    names: _Names,
    step: int,
    describe: Describe | None = None,
) -> list[Run]:
    """The runs that save the most characters over ``records`` (``patterns.choose``)."""
    return choose(start, grids, records, names, step, _piece_chars(records, names), describe).runs


def _describer(decomposer: Decomposer, end: float) -> Describe:
    """A sliding run's background built as a frame's piece is: the decomposer's script within a frame's budget,
    or None. Kept per call, since a stride judged before it is built whole asks for the same ones again."""
    known: dict[tuple[bytes, bytes], Transform | None] = {}

    def describe(source: Grid, target: Grid) -> Transform | None:
        key = (source.tobytes(), target.tobytes())
        if key not in known:
            found = decomposer.decompose_within(
                source, target, lambda cost: cost > decomposer.animation_max_bits or time.monotonic() >= end
            )
            known[key] = None if found is None else with_colors(found, source)
        return known[key]

    return describe


def _piece_chars(records: list[AnimationFrame], names: _Names) -> list[int]:
    """What each record's piece writes, a newline included."""
    built = [record.index for record in records]
    return [len("\n".join(_piece(record, built[:position], names))) + 1 for position, record in enumerate(records)]


def _may_fit(
    start: Grid,
    grids: list[Grid],
    records: list[AnimationFrame],
    names: _Names,
    step: int,
    limit: int,
    describe: Describe | None = None,
) -> bool:
    """Whether a stride whose pieces so far pass ``limit`` may still fit it once its runs are chosen: its frames
    so far, written with their best runs, within the limit once the pieces of the last ``MIN_RUN`` frames are
    taken off, which a run reaching past them could yet write as part of its loop. Judging the stride before it
    is built whole is what keeps a long animation's strides from each being searched in full."""
    tail = sum(_piece_chars(records, names)[-MIN_RUN:])
    return len(_write(records, names, _runs(start, grids, records, names, step, describe))) - tail <= limit


def _choose(
    decomposer: Decomposer,
    start: Grid,
    grids: list[Grid],
    target: Grid,
    k: int,
    built: list[int],
    names: _Names,
    end: float,
) -> AnimationFrame:
    """How frame ``k`` is built from the frames built before it (``built``, in order) or the frame before the move."""
    # A copy of an earlier frame costs nothing and nothing costs less, so an exact repeat is taken without
    # searching: the most recent built frame it repeats, else the frame before the move.
    same = next((earlier for earlier in reversed(built) if np.array_equal(grids[earlier], target)), None)
    if same is not None or np.array_equal(start, target):
        name = names.before if same is None else f"{names.dict}[{same}]"
        return AnimationFrame(k, name, same, NoOp(), {name: 0.0})
    # (earlier frame or None, expression, frame), in the order ties go.
    candidates: list[tuple[int | None, str, Grid]] = [
        (earlier, f"{names.dict}[{earlier}]", grids[earlier]) for earlier in reversed(built[-LOOKBACK:])
    ]
    candidates.append((None, names.before, start))
    best: tuple[float, int | None, str, Transform] | None = None
    costs: dict[str, float | None] = {}
    tried: list[Grid] = []
    for earlier, name, source in candidates:
        if any(np.array_equal(source, seen) for seen in tried):
            continue
        tried.append(source)
        bar = math.inf if best is None else best[0]

        def give_up(cost: float, bar: float = bar) -> bool:
            """Over budget, no cheaper than the best so far, or out of time."""
            return cost > decomposer.animation_max_bits or cost >= bar or time.monotonic() >= end

        transform = decomposer.decompose_within(source, target, give_up)
        costs[name] = None if transform is None else transform.bit_cost()
        if transform is not None:
            best = (transform.bit_cost(), earlier, name, transform)
    if best is None:
        return AnimationFrame(k, None, None, None, costs)
    _, earlier, name, transform = best
    if not isinstance(transform, NoOp):
        transform._self_test(wildcards=False)  # pyright: ignore[reportPrivateUsage]
    transform = with_colors(transform, start if earlier is None else grids[earlier])
    return AnimationFrame(k, name, earlier, transform, costs)


def _opening(names: _Names) -> str:
    """The line that opens a block."""
    return f"{names.dict} = {{}}"


def _closing(names: _Names) -> str:
    """The assert that closes a block, over every frame it built."""
    frame = names.frame.format(k=_INDEX)
    return f"assert all(({_FRAME} == {frame}).all() for {_INDEX}, {_FRAME} in {names.dict}.items())"


def _piece(record: AnimationFrame, built: list[int], names: _Names, synced: bool = True) -> list[str]:
    """One frame's lines, a blank one first; ``built`` holds the frames built before it, and ``synced`` says
    whether the board holds the last of them."""
    g, k = names.board, record.index
    lines = [""]
    if record.transform is None:
        lines += [f"# Frame {k}: no concise description, taken as it is", f"{g} = {names.frame.format(k=k)}.copy()"]
    else:
        source = names.before if record.earlier is None else f"{names.dict}[{record.earlier}]"
        said = names.before if record.earlier is None else f"frame {record.earlier}"
        same = isinstance(record.transform, NoOp)
        lines.append(f"# Frame {k}: {'the same as' if same else 'from'} {said}")
        if not built or record.earlier != built[-1] or not synced:
            lines.append(f"{g} = {source}.copy()")
        lines += record.transform._code_lines(source, g)  # pyright: ignore[reportPrivateUsage]
    lines.append(f"{names.dict}[{k}] = {g}.copy()")
    return lines


def _write(records: list[AnimationFrame], names: _Names, runs: Sequence[Run] = ()) -> str:
    """The block for the frames as ``records`` build them, written against ``names``, each of ``runs`` as its
    loop in place of its frames' pieces."""
    starting = {run.frames[0]: run for run in runs}
    drawn = {index for run in runs if run.kind == "sliding" for index in run.frames}
    notes = [
        note
        for record in records
        if record.transform is not None and record.index not in drawn
        for note in record.transform._notes()  # pyright: ignore[reportPrivateUsage]
    ]
    lines = [*dict.fromkeys(notes), _opening(names)]
    built: list[int] = []
    synced = True
    position = 0
    while position < len(records):
        run = starting.get(records[position].index)
        if run is not None:
            if run.needs_board and not synced:
                lines += ["", f"{names.board} = {names.dict}[{built[-1]}].copy()"]
            lines += run.lines(names)
            built += run.frames
            synced = run.synced
            position += len(run.frames)
            continue
        lines += _piece(records[position], built, names, synced)
        built.append(records[position].index)
        synced = True
        position += 1
    lines += ["", _closing(names)]
    return "\n".join(lines)
