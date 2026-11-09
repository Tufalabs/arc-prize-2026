"""A frame the model just got, explained as a short edit script from a frame it already knows.

Two events produce one: a new gameplay frame the model did not predict, or
predicted wrongly, and a ``test_world_model()`` that fails on a rendered
frame. Either way the target is compared with up to three baselines, the
model's own mispredicted render, the frame before, and the frame the level
opened with, and framediff's ``Decomposer.explain`` returns the cheapest as
a block of numpy. The block names the exact expressions the model holds the
frames under, and is run against them before it is shown: it must reach the
target, or the game ends.

Which is cheapest is decided with a light preference for the more
informative comparison: a handicap added to the previous frame's cost, and
a larger one to the level start's. And a block is only shown within the
decomposer's budget, which drops once the level is understood, that is once
a world model has passed on it: by then the model has read the frame
before, either through its own passing model or through a block, and only a
short delta is worth its tokens. Over budget the text says there is no
concise difference.

A frame that repeats an earlier frame of its level exactly is named as that
repeat (:func:`repeated_turn`): in place of a block when a RESET or an UNDO
brought it back, since the game restoring a board explains nothing about
the game, and after the block otherwise.

A move's animation is explained the same way, as one block that rebuilds
its frames (framediff's ``explain_animation``): on request, through
``describe_animation(i)`` in the REPL, and on its own for the longest
animation a call played when it is longer than every one shown on the
level so far. Which moves had animation frames can be said on the moves
themselves, with the count the kernel's history keeps
(:func:`kept_animation`).

Three switches on ``SealAgent`` choose among these, independently:
``settled_delta`` (the new frame, and the render a test fails on),
``animation_description`` (``none``; ``available`` on request; ``auto``, on
its own as well) and ``animation_marks`` (which moves had animation
frames). This module runs in the kernel, where the frames and the names
live; the host imports it for its types and for :func:`kept_animation`.
"""

from __future__ import annotations

import base64
import dataclasses
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from framediff import Decomposer, Source

#: How a move's animation frames are described as a block of code: not at all, on request
#: (``describe_animation``), or on request and, for a call's longest animation, on their own too.
AnimationDescription = Literal["none", "available", "auto"]

#: How long one framediff call may search on the seal, past framediff's own limit (2026-09-29): the
#: pilot's searches that reached the old limit ran a fraction of it on a quiet node, slowed by a node's memory
#: stalls rather than running away.
TIME_LIMIT_S = 300.0


def default_decomposer() -> Decomposer:
    """framediff's search as the seal ships it: framediff's defaults, with the seal's time limit."""
    from framediff import Decomposer

    return Decomposer(time_limit_s=TIME_LIMIT_S)


#: The decomposer's budget once a world model has passed on the level, in place of its ``max_bits``
#: (2026-09-17): by then the model has read the frame before, and only a short delta is worth its
#: tokens. The default of ``SealAgent.max_bits_understood`` and its kernel half.
MAX_BITS_UNDERSTOOD = 50

#: Added to each baseline's cost when choosing, by kind: the model's own mispredicted render says the most
#: about its misunderstanding, the level start the least. ``render`` is the model's render of a frame its
#: check passed within the pixel tolerance, the one baseline such a frame is described from.
HANDICAPS = {"prediction": 0, "previous": 10, "level_start": 20, "render": 0}

#: How each baseline is named to the model.
PHRASES = {
    "prediction": "your mispredicted frame",
    "previous": "the previous frame",
    "level_start": "the level start frame",
    "render": "your world model's render",
}


class FrameDeltaError(RuntimeError):
    """A block did not reach the frame it describes, the kernel could not say, or framediff ran out of time
    where that is to end the game: the game ends."""


def _out_of_time(what: str, seconds: float) -> FrameDeltaError:
    """The error for a decomposition that reached its deadline where a timeout ends the game."""
    return FrameDeltaError(f"framediff ran out of time on {what} after {seconds:.1f} s")


#: Why a record was made, for the statistics later: the turn had no render (``unpredicted``: no passing
#: model, or exploration), the model declined the move (``declined``), its render differs
#: (``mispredicted``), the frame is a game-over screen no model renders (``lost``), or the win
#: screen, which no model renders either, described for the reflection turn (``won``), a
#: ``test_world_model()`` failed on the turn's render (``test_failure``), the render matched
#: (``predicted``: no decomposition, the prompt says the frame is as predicted), or it matched only
#: under another energy value, which the move's check accepts (``energy_reconciled``: no
#: decomposition either), or it passed within the model's pixel tolerance (``tolerated``:
#: described from the render, after a line saying so).
REASONS = (
    "unpredicted",
    "declined",
    "mispredicted",
    "lost",
    "won",
    "test_failure",
    "predicted",
    "energy_reconciled",
    "tolerated",
)


@dataclass
class Delta:
    """One decomposition, shown to the model or skipped for its cost: what a span records."""

    event: str
    reason: str
    understood: bool
    turn_index: int
    target: str
    baselines: list[dict[str, Any]]
    #: The chosen baseline's kind, its block and the block's cost; None when the search ran out of time.
    chosen: str | None
    code: str | None
    cost: float | None
    limit: float
    shown: bool
    text: str
    #: Whether the search reached its deadline, which reads as a frame nothing describes within budget.
    timed_out: bool
    duration_s: float
    #: The frames the search was given, the target's and each baseline's by kind, packed (:func:`pack_grid`):
    #: the host keeps them on the span, so a run's decompositions can be searched again offline. The model's
    #: render exists nowhere else.
    grids: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """The record as plain data."""
        return dataclasses.asdict(self)


def baseline(kind: str, expression: str, frame: npt.NDArray[np.int8]) -> Source:
    """A baseline of one of the kinds in :data:`HANDICAPS`, read by the block under ``expression``."""
    from framediff import Source

    return Source(expression, frame, HANDICAPS[kind])


def describe(
    target: npt.NDArray[np.int8],
    target_name: str,
    baselines: dict[str, Source],
    *,
    understood: bool,
    decomposer: Decomposer,
    max_bits_understood: float,
    namespace: dict[str, Any],
    event: str,
    reason: str,
    turn_index: int,
    subject: str,
    deadline: float | None = None,
    crash_on_timeout: bool = False,
) -> Delta:
    """Explain ``target`` from the cheapest baseline, run the block, and word it for the model.

    ``baselines`` maps each kind to its source, in priority order: at an
    equal cost plus handicap the first wins. ``namespace`` holds what the
    block runs against, the names in the expressions and the helpers; the
    block is run on a copy of it before it is shown, and a block that does
    not reach the target raises :class:`FrameDeltaError`. ``subject`` opens
    the sentence: what the target is called to the model. ``event`` and
    ``reason`` (one of :data:`REASONS`) go on the record. ``decomposer``
    searches, and a block is shown within its ``max_bits``, or within
    ``max_bits_understood`` once the level is ``understood``.

    The search stops at framediff's own time limit, or at ``deadline`` if
    that comes first; then the frame reads as one nothing describes within
    budget, or with ``crash_on_timeout`` the call raises
    :class:`FrameDeltaError`.
    """
    if not baselines:
        raise ValueError("a delta needs at least one baseline")
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}, got {reason!r}")
    if understood:
        decomposer = dataclasses.replace(decomposer, max_bits=max_bits_understood)
    kinds, sources = list(baselines), list(baselines.values())
    started = time.monotonic()
    found = decomposer.explain(sources, target, target_name, deadline=deadline)
    duration = round(time.monotonic() - started, 3)
    if found.timed_out and crash_on_timeout:
        raise _out_of_time(target_name, duration)
    chosen = next((kind for kind, source in baselines.items() if source is found.source), None)
    shown = not found.over_budget
    if shown:
        assert found.source is not None and found.code is not None and chosen is not None
        try:
            exec(found.code, dict(namespace))
        except Exception as error:
            raise FrameDeltaError(
                f"the block explaining {target_name} from {found.source.name} did not run: {type(error).__name__}: {error}\n"
                f"{found.code}"
            ) from error
        text = shown_text(subject, chosen, found.code)
    else:
        text = skipped_text(subject, kinds)
    return Delta(
        event=event,
        reason=reason,
        understood=understood,
        turn_index=turn_index,
        target=target_name,
        baselines=[
            {"kind": kind, "expression": source.name, "cost": cost, "handicap": source.handicap}
            for kind, source, cost in zip(kinds, sources, found.costs)
        ],
        chosen=chosen,
        code=found.code,
        cost=found.cost,
        limit=decomposer.max_bits,
        shown=shown,
        text=text,
        timed_out=found.timed_out,
        duration_s=duration,
        grids={"target": pack_grid(target), **{kind: pack_grid(source.frame) for kind, source in baselines.items()}},
    )


def pack_grid(grid: npt.ArrayLike) -> dict[str, Any]:
    """A frame as plain data a record can carry out of the kernel: its shape, and its int8 cells compressed
    and base64-encoded. A render's wildcard cells stay -1."""
    array = np.ascontiguousarray(grid, dtype=np.int8)
    return {"shape": list(array.shape), "cells": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii")}


def unpack_grid(packed: dict[str, Any]) -> npt.NDArray[np.int8]:
    """The frame :func:`pack_grid` packed."""
    cells = zlib.decompress(base64.b64decode(packed["cells"]))
    return np.frombuffer(cells, dtype=np.int8).reshape(packed["shape"]).copy()


def predicted_text(wildcards: bool) -> str:
    """The line for a frame the model's render matched."""
    qualifier = " (on the cells it predicted)" if wildcards else ""
    return f"The current frame is as predicted by your world model{qualifier}."


def energy_reconciled_text(wildcards: bool) -> str:
    """The line for a frame the model's render matched once its energy was taken from the frame. No block
    and no values: the bar is the part a model should leave rough, and a diff of it invites the work."""
    qualifier = " (on the cells it predicted)" if wildcards else ""
    return f"The current frame is as predicted by your world model{qualifier}, once its energy is reconciled with the frame's."


def tolerated_text(pixels: int | None) -> str:
    """The line before the block for a frame the model's check passed within its pixel tolerance: how many
    cells the render missed, or, with ``None``, that its energy was reconciled with the frame's too."""
    if pixels is None:
        return (
            "The current frame passed your world model's check within your pixel tolerance, once its energy is "
            "reconciled with the frame's."
        )
    cells = "1 pixel" if pixels == 1 else f"{pixels} pixels"
    return f"The current frame passed your world model's check within your pixel tolerance; your render missed {cells}."


#: The moves that put back a board the level has already shown: a frame one of them brings back is named as
#: the repeat it is, in place of a block, since the game restoring a board is no move's effect to explain.
RESTORING = ("RESET", "UNDO")


def repeated_turn(grids: Sequence[npt.NDArray[np.int8]], index: int) -> int | None:
    """The earlier turn whose frame ``grids[index]`` repeats exactly, among a level's settled frames: the
    level's start when it is that, else the most recent. None when it repeats none, and when it repeats the
    frame just before it, which is a move that changed nothing rather than a board come back."""
    target = grids[index]
    if index < 2 or np.array_equal(target, grids[index - 1]):
        return None
    if np.array_equal(target, grids[0]):
        return 0
    return next((k for k in range(index - 2, 0, -1) if np.array_equal(target, grids[k])), None)


def repeat_text(index: int) -> str:
    """The line for a frame that repeats ``history[index]``'s exactly."""
    start = ", the level's start" if index == 0 else ""
    return f"The current frame is the same as `history[{index}].frame`{start}."


def shown_text(subject: str, kind: str, code: str) -> str:
    """The wording around a block the model reads. It says the block is exact: the model otherwise re-derives
    from pixels what the block already states."""
    return (
        f"{subject} can be most easily understood as a delta from {PHRASES[kind]} as follows "
        "(this is based on visual decomposition, without understanding the game). The code is exact: it was run "
        "against these frames and its assert holds, so there is no need to check it against the pixels.\n\n"
        f"```python\n{code}\n```"
    )


def skipped_text(subject: str, kinds: list[str]) -> str:
    """The wording when no block is concise enough to show."""
    phrases = [PHRASES[kind] for kind in kinds]
    joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " or " + phrases[-1]
    return (
        f"{subject} has no concise description as a delta from {joined} "
        "(this is based on visual decomposition, without understanding the game)."
    )


def matches(predicted: npt.NDArray[np.int8], actual: npt.NDArray[np.int8]) -> bool:
    """Whether a render matches a frame on every cell it predicts; a wildcard cell matches anything."""
    from framediff import WILDCARD

    live = predicted != WILDCARD
    return predicted.shape == actual.shape and bool(np.array_equal(predicted[live], actual[live]))


#: Where a block at a stride the model chose is cut off: nothing else bounds it (2026-09-25).
ANIMATION_CUT_CHARS = 10000

#: Around a text the harness prints into the model's cell that the output cap leaves whole: a requested
#: animation's block, which its own cut bounds, and which a cap sized for the model's own printing would
#: otherwise cut in the middle (2026-09-29). The REPL removes them. Private-use code points, which
#: nothing the model reads carries.
UNCAPPED_START, UNCAPPED_END = "\ue000", "\ue001"

_Item = TypeVar("_Item")


def kept_animation(
    before: _Item | None, frames: Sequence[_Item], settled: _Item, same: Callable[[_Item, _Item], bool]
) -> list[_Item]:
    """The animation frames a turn keeps: none equal to the frame kept before it, the first compared with the
    board before the move when there is one, and none at the end equal to the settled frame.

    Repeats carry the engine's timing, which the model does not use. The kernel's ``history`` keeps these
    frames and the host counts them to mark the moves, so ``same`` compares whatever the caller holds."""
    kept: list[_Item] = []
    previous = before
    for frame in frames:
        if previous is not None and same(frame, previous):
            continue
        kept.append(frame)
        previous = frame
    while kept and same(kept[-1], settled):
        kept.pop()
    return kept


def _ordinal(number: int) -> str:
    """``2nd``, ``3rd``, ``4th``, ``6th``."""
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def animation_start(index: int) -> str:
    """The frame ``history[index]``'s animation is built from, as the block names it: the turn before it, or
    for the game's opening turn, whose first observation can carry an animation with no turn before it, the
    animation's own first frame."""
    return "history[0].animation_frames[0].grid" if index == 0 else f"history[{index - 1}].frame.grid"


def describe_animation(
    history: Any,
    index: int,
    stride: int | None,
    *,
    decomposer: Decomposer,
    max_chars: int,
    cut_chars: int,
    namespace: dict[str, Any],
    reason: str,
    deadline: float | None = None,
    crash_on_timeout: bool = False,
) -> tuple[str, dict[str, Any]]:
    """The text that explains ``history[index]``'s animation, and the record a span keeps.

    ``index`` is a turn with animation frames, built from the turn before it;
    the game's opening turn, which has none before it, is built from its own
    first frame (:func:`animation_start`). Without
    a stride the block stays within ``max_chars`` by building every n-th
    frame, and says so; a stride given is used as asked. Either way a
    block longer than ``cut_chars`` is cut there, after it has been run
    whole against ``namespace``: a block that does not reach its frames
    raises :class:`FrameDeltaError`. ``reason`` (``automatic`` or
    ``requested``) goes on the record. A search that reaches framediff's
    time limit, or ``deadline`` first, takes every frame as it is, or with
    ``crash_on_timeout`` raises :class:`FrameDeltaError`.
    """
    turn = history[index]
    frames = [frame.grid for frame in turn.animation_frames]
    before = animation_start(index)
    started = time.monotonic()
    found = decomposer.explain_animation(
        frames[0] if index == 0 else history[index - 1].frame.grid,
        frames,
        before,
        f"history[{index}].animation_frames[{{k}}].grid",
        stride=stride,
        max_chars=max_chars if stride is None else None,
        deadline=deadline,
    )
    duration = round(time.monotonic() - started, 3)
    if found.timed_out and crash_on_timeout:
        raise _out_of_time(f"history[{index}]'s animation", duration)
    built = found.built
    subject = f"The animation of history[{index}] ({len(frames)} frames)"
    visual = "(this is based on visual decomposition, without understanding the game)"
    cut = False
    if not built:
        text = f"{subject} has no concise description {visual}."
    else:
        try:
            exec(found.code, dict(namespace))
        except Exception as error:
            raise FrameDeltaError(
                f"the block rebuilding history[{index}]'s animation did not run: {type(error).__name__}: {error}\n"
                f"{found.code}"
            ) from error
        code = found.code
        if len(code) > cut_chars:
            # At the blank line between two frames' pieces, so the last frame shown is whole.
            code = code[:cut_chars].rsplit("\n\n", 1)[0] + "\n\n# ... cut off"
            cut = True
        text = f"{subject} can be rebuilt as follows {visual}."
        if stride is None and found.stride > 1:
            text += (
                f" Only every {_ordinal(found.stride)} frame is built, to keep this under {max_chars:,} characters; "
                f"`describe_animation({index}, stride=1)` builds every frame."
            )
        if cut:
            text += (
                " The whole block was run against these frames and its assert holds, but it is cut off here at "
                f"{cut_chars:,} characters; a larger `stride` shows all of it."
            )
        else:
            text += (
                " The code is exact: it was run against these frames and its assert holds, so there is no need to "
                "check it against the pixels."
            )
        text += f"\n\n```python\n{code}\n```"
    record: dict[str, Any] = {
        "event": "animation",
        "reason": reason,
        "turn_index": index,
        "frames": len(frames),
        "stride": found.stride,
        "requested_stride": stride,
        "built": len(built),
        "as_is": len(found.frames) - len(built),
        "chars": len(found.code),
        "cut": cut,
        "shown": bool(built),
        "code": found.code,
        "text": text,
        "timed_out": found.timed_out,
        "duration_s": duration,
    }
    # Which runs the block wrote as loops (``Decomposer.animation_patterns``): per kind, as ``runs_<kind>``, how
    # many frames it covers. Fields of their own, since the record becomes a span's scalar values; only the
    # kinds written, so a record without runs reads as before.
    for kind in sorted({r.kind for r in found.runs}):
        record[f"runs_{kind}"] = sum(len(r.frames) for r in found.runs if r.kind == kind)
    if found.runs_dropped:
        record["runs_dropped"] = True
    return text, record
