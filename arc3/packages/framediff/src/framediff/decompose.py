"""The search for a short edit script from one frame to another.

``Decomposer.decompose(a, b)`` returns a ``Transform`` that turns ``a`` into
``b``: ``NoOp`` when they are equal, otherwise a ``Composite`` of the
transforms in ``framediff.transforms``. ``Decomposer.explain`` takes several
named sources, each with a handicap, decomposes each toward a named target,
and returns the cheapest source's script as a block of code, flagged when it
costs more than the decomposer's ``max_bits``. ``explain`` is the one-source
shorthand. ``Decomposer.explain_animation`` writes a block that rebuilds an
animation frame by frame (``framediff.animation``). Both give up once they
have searched for the decomposer's ``time_limit_s``, or past a deadline the
caller gives, and then say so (``timed_out``) with the outcome of a search
that found nothing within budget.

The search is greedy on one objective, the change in the script's total
cost with the single-cell fallback priced in:

    score = cost − PIXEL_COST · (cells made right − cells made wrong)

Each step takes the candidate with the lowest score, applies it to the
board, and stops when no candidate scores below zero, since a candidate at
zero or above is no better than setting its cells one by one. Whatever
remains wrong is then set cell by cell. The greedy runs twice, with plain
shifts only and with every mirrored and turned variant, and the cheaper
script is returned: a turned copy that fixes two things at once wins a
step and can lead a run to a costlier script than the plain moves it
displaced, and the cheaper of two runs is never costlier than either.
There are no tie rules: ties fall where the enumeration order below puts
them, and the whole search is a deterministic function of the two frames.

A paste is the one transform whose cost grows with its area, and a cell
inside it costs 1 where a cell set on its own costs 4, so a paste absorbs
whatever lies next to the content it is for: a filled rectangle beside a
complex shape, or on two unrelated frames, everything. Once taken it cannot
be undone. So a paste is carved before it is proposed: the cheapest other
transforms inside its rectangle are each tried, followed by pastes of what
they leave wrong, and the cheapest of those replaces the plain paste when
it costs less. Every paste that results is carved in turn. A candidate is
therefore a short list of transforms, and the greedy appends the list.

Candidates are found in two ways.

- **A gain map and its best rectangle.** A fill, a recolour, a paste and a
  block copy at a given shift each write one value per cell inside some
  rectangle, so their gain is a per-cell map, and the best rectangle is the
  maximum-sum subrectangle of that map (2D Kadane, exact). Every map is at
  most zero outside the cells that are still wrong, so the rectangle is
  searched inside their bounding box only.
- **Shifts by correlation.** A block copy's shift is shortlisted by how
  many wrong cells it would make right, which for every shift at once is
  the cross-correlation of each colour's plane in ``a`` with the same
  colour's plane of the still-wrong cells of ``b``, summed over colours and
  computed by FFT. A mirrored or turned copy is the same search on the
  mirrored or turned whole frame, because a turned sub-block is a sub-block
  of the turned frame. A sprite that moved also leaves a hole, which the
  fill search finds; the same correlation from the hole's content finds
  where it went. A copy is offered as a move, clearing its old place to
  the colour that makes the most of it right, whenever that gains.
- **Lines from components.** A connected component of wrong cells that
  ``b`` paints in one colour is tried as a line between its extreme cells,
  each way round, since a line and its reverse can differ by a cell.

A wildcard cell of ``a`` is never wrong, and nothing written on it counts
as fixed or broken; the block resolves it from ``b`` at the end.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, cast

import numpy as np
import numpy.typing as npt

from framediff.animation import MAX_CHARS, AnimationDecomposition, explain_animation
from framediff.grid import GRID_SIZE, NUM_COLORS, Grid
from framediff.objects import proposals
from framediff.transforms import (
    WILDCARD,
    AsciiGrid,
    BlockCopy,
    Composite,
    FillRect,
    Line,
    NoOp,
    Recolour,
    SetPixel,
    Transform,
    with_colors,
)

#: What a wrong cell costs to fix on its own: the objective's unit.
PIXEL_COST = SetPixel.COST

#: The FFT size: shifts run from −63 to 63, and this keeps them from wrapping onto each other.
_FFT_SIZE = 2 * GRID_SIZE

#: The gain a copy's map carries where its source would fall outside the frame: no rectangle may span it.
_OFF_FRAME = -(GRID_SIZE * GRID_SIZE * PIXEL_COST)

#: The eight ways a block may be mirrored and turned, identity first, as (flip, rot).
_VARIANTS: tuple[tuple[bool, int], ...] = tuple((flip, rot) for flip in (False, True) for rot in range(4))

_Map = npt.NDArray[np.int32]

#: A shortlisted shift: (variant, (rows down, cols right), wrong cells it would make right).
_Shift = tuple[int, tuple[int, int], int]

#: A candidate: the transforms it appends, in order.
_Parts = tuple[Transform, ...]


@dataclass
class Rectangle:
    """A half-open rectangle ``row0:row1, col0:col1`` and the gain of a map summed over it."""

    gain: int
    row0: int
    row1: int
    col0: int
    col1: int


#: Each of the search's options as it is off, plain greedy among them: a decomposer pickled or recorded before an
#: option existed searched without it, and reads back with it off.
OPTIONS_OFF: dict[str, Any] = {
    "move_distance_cost": 0.0,
    "object_moves": False,
    "animation_patterns": False,
    "lookahead": 1,
    "animation_lookahead": 1,
}


@dataclass
class Decomposer:
    """The search; ``decompose`` runs it.

    ``shifts_per_variant`` is how many shifts the correlation shortlists for
    each of the seven mirrored-and-turned variants of the start frame, per
    step, and ``plain_shifts`` how many for the plain variant, whose
    rectangle searches are cheap. One catches one mover; more catch several
    movers in one step's shortlist, and shifts whose matches are scattered
    rank above a mover's by count alone, so the list has to reach past
    them. ``hole_shifts`` is how many of the best fills are read as holes a
    sprite left, whose content is then looked for among the wrong cells.
    ``lookahead`` is how many of a step's best candidates are tried one
    step further before one is chosen; 1 is plain greedy;
    ``animation_lookahead`` takes its place in ``explain_animation``.
    ``carve_tries``
    is how many of the best transforms inside a paste's rectangle are tried
    as a way to shrink it. ``max_bits`` is the most a script ``explain``
    returns may cost before it is flagged as over budget, and
    ``animation_max_bits`` the most one frame's script in
    ``explain_animation`` may cost before the frame is taken as it is;
    ``decompose`` ignores both. ``time_limit_s`` is how long one ``explain``
    or ``explain_animation`` call may search, every source and stride
    included.
    """

    shifts_per_variant: int = 3
    plain_shifts: int = 6
    hole_shifts: int = 3
    # A settled frame is searched once a move, cheaply, and greedy's first step
    # can lead it to a costlier script than a smaller step would; one step of
    # lookahead finds the cheaper one at a modest cost. An animation searches
    # every frame of every stride tried, where the same lookahead buys less and
    # costs the long ones their time limit (framediff2's lookahead notebook on
    # the reference run's corpus; 2026-10-09).
    lookahead: int = 2
    animation_lookahead: int = 1
    carve_tries: int = 3
    # Where a script stops being a short description of the change: set with
    # the seal's first frame deltas (2026-09-17).
    max_bits: float = 150
    # An animation is read a frame at a time, so a frame is worth a short
    # piece or none (2026-09-24).
    animation_max_bits: float = 50
    # Well past the slowest frame and animation measured, so it stops only a
    # search that has run away (2026-09-25).
    time_limit_s: float = 180
    # A light cost per cell a copy or move travels (``BlockCopy.distance_cost``).
    # A one-colour block moved left or up by less than its size has two
    # readings that write the same cells at the same cost, the whole block
    # moving and its trailing strip jumping across it, and the enumeration
    # order picked the strip for those two directions; priced by distance, the
    # shorter move wins in every direction. Light enough to decide between
    # moves and nothing else; on the reference run's corpus no block came out
    # costlier for it (2026-10-08).
    move_distance_cost: float = 1 / 128
    # Objects matched across the two frames as candidates (``framediff.objects``):
    # a piece of one colour, or touching pieces of several, that moved or turned
    # is proposed as a ``MaskedMove`` of its own cells, which a copy of its box
    # cannot be where the box holds anything else. Searched in a run of its
    # own beside the two plain ones, so a script is never costlier than
    # without it (2026-10-08).
    object_moves: bool = True
    # Runs of an animation's frames that one loop explains, written as that loop
    # (``framediff.patterns``): a frame that repeats the one some frames back, the
    # same edit a step on each frame, one shape sliding over a background that
    # stays put. Chosen only where shorter than the frames' own pieces; a stride
    # whose pieces pass the length limit is judged by its frames so far with
    # their best runs before it is given up (2026-10-08).
    animation_patterns: bool = True

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore, an option the pickle predates read as off (:data:`OPTIONS_OFF`): a run's solver is unpickled
        to replay it or read its corpus, and its search ran without the options added since."""
        vars(self).update({**OPTIONS_OFF, **state})

    def deadline(self, deadline: float | None = None) -> float:
        """When a call starting now gives up, on the ``time.monotonic()`` clock: after ``time_limit_s``, or at
        ``deadline`` if that comes first."""
        own = time.monotonic() + self.time_limit_s
        return own if deadline is None else min(own, deadline)

    def decompose(self, a: npt.ArrayLike, b: npt.ArrayLike) -> Transform:
        """A transform turning ``a`` into ``b``: ``NoOp`` if they are equal, else a ``Composite``. Either frame may
        hold wildcards: a cell of ``a``'s stays one whatever is written on it, and any value suits a cell of
        ``b``'s."""
        found = self.decompose_within(a, b, lambda cost: False)
        assert found is not None
        return found

    def decompose_within(
        self, a: npt.ArrayLike, b: npt.ArrayLike, give_up: Callable[[float], bool]
    ) -> Transform | None:
        """``decompose``, abandoned once the script so far costs what ``give_up`` rejects: None then.

        A script's cost only grows as the search appends to it, so a caller
        that only wants a script under some bound loses nothing by stopping
        a run at it; a script that comes back is the one ``decompose``
        returns.
        """
        start, target = _frame(a, "a", wildcards=True), _frame(b, "b", wildcards=True)
        live = (start != WILDCARD) & (target != WILDCARD)
        if np.array_equal(start[live], target[live]):
            return None if give_up(0) else NoOp()
        # Two runs, plain shifts only and then every variant; the cheaper script wins, the plain one at a tie.
        # A turned copy that fixes two things at once wins a step and can lead the run to a costlier script
        # than the plain moves it displaced; the minimum of the two is never costlier than either. The
        # second run stops once it can no longer come in cheaper than the first.
        #
        # With a price on distance the runs are made with it and then without it, every script priced by
        # distance, the priced run's at a tie. The price decides between moves that tie, and a nudge it gives
        # an early step leads some runs to costlier scripts; the unpriced runs are today's search, so the
        # result is never costlier than today's script priced the same way.
        #
        # With object moves on, each rate adds a run with every variant and the matched objects too.
        best: Composite | None = None
        rates = (self.move_distance_cost, 0.0) if self.move_distance_cost else (0.0,)
        runs = ((False, False), (True, False), *(((True, True),) if self.object_moves else ()))
        for rate in rates:
            for turned, objects in runs:
                bar = math.inf if best is None else best.bit_cost()

                def stop(cost: float, bar: float = bar) -> bool:
                    return give_up(cost) or cost >= bar

                found = self._greedy(start, target, turned, rate, stop, objects=objects)
                if found is not None and rate != self.move_distance_cost:
                    found = found.priced(self.move_distance_cost)
                    if stop(found.bit_cost()):
                        found = None
                if found is not None:
                    best = found
        return best

    def explain(
        self,
        sources: Sequence[Source],
        target: npt.ArrayLike,
        target_name: str,
        intermediate_name: str = "g",
        *,
        deadline: float | None = None,
    ) -> Decomposition:
        """The cheapest source's script toward ``target``, as a block written against the names given.

        Each source is decomposed; the lowest cost plus handicap wins, the
        earlier source at a tie. The block resolves the chosen source's
        wildcard cells from the target when it has any. Only the chosen
        script is written as code, and it is written whatever it costs:
        ``over_budget`` says whether that cost exceeds ``max_bits``.

        A call still searching at its deadline (:meth:`deadline`) stops
        there and says it ``timed_out``, keeping the best it found by then:
        the cheapest script among the sources whose search had come back with
        one, each such script exact but perhaps costlier than a whole search
        would have found. With none, it comes back over budget with no script,
        what a frame nothing describes within budget would give.
        """
        if not sources:
            raise ValueError("explain needs at least one source")
        _frame(target, "target", wildcards=False)  # the block ends in an assert against it
        end = self.deadline(deadline)
        transforms = [
            self.decompose_within(source.frame, target, lambda cost: time.monotonic() >= end) for source in sources
        ]
        timed_out = time.monotonic() >= end
        assert timed_out or all(transform is not None for transform in transforms), "only the clock abandons a search"
        costs = [None if transform is None else transform.bit_cost() for transform in transforms]
        finished = [i for i, cost in enumerate(costs) if cost is not None]
        if not finished:
            return Decomposition(source=None, transform=None, code=None, over_budget=True, costs=costs, timed_out=True)
        index = min(finished, key=lambda i: cast("float", costs[i]) + sources[i].handicap)
        source = sources[index]
        transform = with_colors(cast("Transform", transforms[index]), source.frame)
        wildcards = bool((np.asarray(source.frame) == WILDCARD).any())
        return Decomposition(
            source=source,
            transform=transform,
            code=transform.to_code(source.name, target_name, intermediate_name, wildcards=wildcards),
            over_budget=transform.bit_cost() > self.max_bits,
            costs=costs,
            timed_out=timed_out,
        )

    def explain_animation(
        self,
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
        """The block that rebuilds ``frames``, an animation that followed ``before`` (``framediff.animation``)."""
        return explain_animation(
            self,
            before,
            frames,
            before_name,
            frame_name,
            stride=stride,
            max_chars=max_chars,
            list_name=list_name,
            board_name=board_name,
            deadline=deadline,
        )

    def _greedy(
        self,
        start: Grid,
        target: Grid,
        turned: bool,
        distance_cost: float,
        give_up: Callable[[float], bool],
        *,
        objects: bool = False,
    ) -> Composite | None:
        """One greedy run from ``start`` to ``target``, with or without the mirrored and turned variants and the
        matched objects, its copies priced at ``distance_cost``; None once the script so far costs what
        ``give_up`` rejects."""
        live = (start != WILDCARD) & (target != WILDCARD)
        search = _Search(start, target, self, turned, distance_cost, objects)
        board = start.copy()
        parts: list[Transform] = []
        cost = 0.0
        while ((board != target) & live).any():
            found = search.step(board)
            if found is None:
                break
            chosen, gain = found
            right_before = int(((board == target) & live).sum())
            for transform in chosen:
                transform._apply(board, start)
            right_after = int(((board == target) & live).sum())
            if right_after - right_before != gain:
                raise AssertionError(f"{chosen} was scored at gain {gain} but gained {right_after - right_before}")
            parts.extend(chosen)
            cost += sum(transform.bit_cost() for transform in chosen)
            if give_up(cost):
                return None
        for row, col in np.argwhere((board != target) & live):
            parts.append(SetPixel(int(row), int(col), int(target[row, col])))
        script = Composite(tuple(parts))
        return None if give_up(script.bit_cost()) else script


@dataclass(eq=False)
class Source:
    """A frame a target may be explained from; compared by identity, since its frame is an array.

    ``name`` is the expression the block reads it under: a name, or a chain
    of attributes, subscripts and calls on one. ``frame`` may hold wildcard
    cells. ``handicap`` is added to its cost when the sources are compared,
    so a less informative comparison has to be that much shorter to be
    chosen; it does not count against ``max_bits``.
    """

    name: str
    frame: npt.ArrayLike
    handicap: float = 0


@dataclass
class Decomposition:
    """What ``Decomposer.explain`` chose: the source, its script and the block, and every source's cost.

    A call that ran out of time chose nothing: ``source``, ``transform`` and ``code`` are None.
    """

    source: Source | None
    transform: Transform | None
    code: str | None
    over_budget: bool
    #: Each source's cost without its handicap, in the order the sources were given; None for a search
    #: the clock stopped.
    costs: list[float | None]
    #: Whether the call reached its deadline.
    timed_out: bool = False

    @property
    def cost(self) -> float | None:
        """The chosen script's cost; None when nothing was chosen."""
        return None if self.transform is None else self.transform.bit_cost()


def explain(
    a: npt.ArrayLike, b: npt.ArrayLike, name_a: str = "a", name_b: str = "b", intermediate_name: str = "g"
) -> str:
    """The block of code that builds ``b`` from ``a``, written against the names given."""
    decomposer = Decomposer()
    found = decomposer.explain([Source(name_a, a)], b, name_b, intermediate_name)
    if found.code is None:
        raise TimeoutError(f"no script from {name_a} to {name_b} within {decomposer.time_limit_s:g} s")
    return found.code


class _Search:
    """Everything about one (a, b) pair that does not change as the board is edited."""

    def __init__(
        self,
        start: Grid,
        target: Grid,
        options: Decomposer,
        turned: bool = True,
        distance_cost: float = 0.0,
        objects: bool = False,
    ) -> None:
        self.start, self.target = start, target
        self.live = (start != WILDCARD) & (target != WILDCARD)
        self.variant_ids = list(range(len(_VARIANTS))) if turned else [0]
        self.shifts_per_variant, self.plain_shifts = options.shifts_per_variant, options.plain_shifts
        self.hole_shifts = options.hole_shifts
        self.lookahead, self.carve_tries = max(options.lookahead, 1), options.carve_tries
        self.distance_cost = distance_cost
        self.object_moves = proposals(start, target, distance_cost) if objects else []
        self.colors = sorted(set(np.unique(start).tolist()) & set(np.unique(target).tolist()) - {WILDCARD})
        self.variants = [_variant(start, flip, rot) for flip, rot in _VARIANTS]
        # Each variant padded by a frame on every side with a value no cell holds, so a shifted view
        # reads that value wherever the source would fall outside the frame.
        self.padded = [np.pad(variant, GRID_SIZE, constant_values=-1) for variant in self.variants]
        self.spectra = {
            variant: {
                color: np.conj(np.fft.rfft2(self.variants[variant] == color, s=(_FFT_SIZE, _FFT_SIZE)))
                for color in self.colors
            }
            for variant in self.variant_ids
        }

    def step(self, board: Grid) -> tuple[_Parts, int] | None:
        """The transforms to apply next, with their gain, or None when nothing scores below zero.

        The best few candidates are each applied to a copy of the board and
        followed by the best candidate there, scored with the parent's shift
        shortlist since one edit hardly moves it; the lowest two-step total
        wins, ties to the parent's order.
        """
        shifts = self.shifts(self.wrong(board))
        ranked = self.ranked(board, shifts)
        if not ranked:
            return None
        chosen: tuple[float, _Parts, int] | None = None
        for score, parts, gain in ranked[: self.lookahead]:
            child = board.copy()
            for transform in parts:
                transform._apply(child, self.start)
            follow = self.ranked(child, shifts) if len(ranked) > 1 else []
            total = score + (follow[0][0] if follow else 0)
            if chosen is None or total < chosen[0]:
                chosen = (total, parts, gain)
        assert chosen is not None
        return chosen[1:]

    def ranked(
        self, board: Grid, shifts: list[_Shift], box: tuple[int, int, int, int] | None = None, paste: bool = True
    ) -> list[tuple[float, _Parts, int]]:
        """Every candidate scoring below zero as (score, parts, gain), best first, ties in enumeration order."""
        scored = [
            (_cost(parts) - PIXEL_COST * gain, parts, gain)
            for parts, gain in self.candidates(board, shifts, box, paste)
        ]
        return sorted((entry for entry in scored if entry[0] < 0), key=lambda entry: entry[0])

    def candidates(
        self, board: Grid, shifts: list[_Shift], box: tuple[int, int, int, int] | None = None, paste: bool = True
    ) -> Iterator[tuple[_Parts, int]]:
        """Every candidate on this board with its gain, in the order ties are settled by.

        ``box`` confines the candidates to a rectangle ``(row0, row1, col0,
        col1)``; by default they are confined to the bounding box of the
        wrong cells, outside which no map is positive. ``paste`` off leaves
        pastes out, which is how a paste's own rectangle is carved.
        """
        wrong = self.wrong(board)
        box_given = box is not None
        if box is None:
            if not wrong.any():
                return
            rows, cols = np.nonzero(wrong)
            box = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
        row0, row1, col0, col1 = box
        box_slices = slice(row0, row1), slice(col0, col1)
        wrong_box = wrong[box_slices]
        if not wrong_box.any():
            return
        live = self.live[box_slices].astype(np.int32)
        right = ((board == self.target)[box_slices]).astype(np.int32) * live
        target, board_box = self.target[box_slices], board[box_slices]

        kinds: list[tuple[str, tuple[int, ...]]] = []
        maps: list[_Map] = []
        target_wrong = target[wrong_box]
        for color, count in zip(*np.unique(target_wrong, return_counts=True)):
            if count >= _min_gain(FillRect.COST):
                kinds.append(("fill", (int(color),)))
                maps.append((target == color).astype(np.int32) * live - right)
        pairs = np.stack([board_box[wrong_box], target_wrong], axis=1)
        for (old, new), count in zip(*np.unique(pairs, axis=0, return_counts=True)):
            if count >= _min_gain(Recolour.COST):
                kinds.append(("recolour", (int(old), int(new))))
                is_old = (board_box == old).astype(np.int32) * live
                maps.append(is_old * ((target == new).astype(np.int32) - (target == old).astype(np.int32)))
        for variant, (shift_row, shift_col), fixed in shifts:
            flip, rot = _VARIANTS[variant]
            if fixed < _min_gain(BlockCopy.TURNED_COST if flip or rot else BlockCopy.COST):
                continue
            kinds.append(("copy", (variant, shift_row, shift_col)))
            maps.append(self.copy_map(variant, (shift_row, shift_col), box, live, right))
        if paste and int(wrong_box.sum()) * PIXEL_COST > AsciiGrid.COST:
            kinds.append(("paste", ()))
            maps.append(_paste_map(wrong_box))

        holes: list[tuple[float, Rectangle]] = []
        for (kind, args), rect in zip(kinds, _best_rectangles(maps)):
            if rect is None:
                continue
            rect = Rectangle(rect.gain, rect.row0 + row0, rect.row1 + row0, rect.col0 + col0, rect.col1 + col0)
            if kind == "fill":
                fill = FillRect(rect.row0, rect.row1, rect.col0, rect.col1, color=args[0])
                holes.append((fill.bit_cost() - PIXEL_COST * rect.gain, rect))
                yield (fill,), rect.gain
            elif kind == "recolour":
                yield (Recolour(rect.row0, rect.row1, rect.col0, rect.col1, old=args[0], new=args[1]),), rect.gain
            elif kind == "copy":
                yield self.moved(board, self.copy(args[0], (args[1], args[2]), rect))
            else:
                yield self.carved(board, rect, shifts)

        # A fill's rectangle may be the hole a sprite left; where the hole's content reappears among
        # the wrong cells is that sprite's shift, which the count over all wrong cells can miss when
        # background matches at many shifts.
        seen = {shift[:2] for shift in shifts}
        reappeared: list[_Shift] = []
        for _, hole in sorted(holes, key=lambda entry: entry[0])[: self.hole_shifts]:
            mask = np.zeros_like(wrong)
            mask[hole.row0 : hole.row1, hole.col0 : hole.col1] = True
            for shift in self.reappearances(mask & self.live, wrong, 2):
                if shift[:2] not in seen and shift[2] >= _min_gain(BlockCopy.COST):
                    seen.add(shift[:2])
                    reappeared.append(shift)
        extra = [self.copy_map(0, shift[1], box, live, right) for shift in reappeared]
        for shift, rect in zip(reappeared, _best_rectangles(extra)):
            if rect is not None:
                rect = Rectangle(rect.gain, rect.row0 + row0, rect.row1 + row0, rect.col0 + col0, rect.col1 + col0)
                yield self.moved(board, self.copy(0, shift[1], rect))
        confined = np.zeros_like(wrong)
        confined[box_slices] = wrong_box
        for line in self.lines(confined):
            gain = sum(
                int(self.target[cell] == line.color) - int(board[cell] == self.target[cell])
                for cell in line.cells()
                if self.live[cell]
            )
            yield (line,), int(gain)
        # The matched objects, whose gain is read off the board as the earlier edits left it; not inside a
        # paste's rectangle, where the paste's own cells are what is being priced.
        if box_given:
            return
        wrong_count = int(wrong.sum())
        for move in self.object_moves:
            gain = wrong_count - int(self.wrong(self.applied(board, move)).sum())
            if gain > 0:
                yield (move,), gain

    def copy_map(
        self, variant: int, shift: tuple[int, int], box: tuple[int, int, int, int], live: _Map, right: _Map
    ) -> _Map:
        """The gain per cell of ``box`` of copying the variant shifted by ``shift`` onto it; off-frame sources bar a cell."""
        row0, row1, col0, col1 = box
        shifted = self.padded[variant][
            GRID_SIZE + row0 - shift[0] : GRID_SIZE + row1 - shift[0],
            GRID_SIZE + col0 - shift[1] : GRID_SIZE + col1 - shift[1],
        ]
        return np.where(
            shifted < 0, _OFF_FRAME, (shifted == self.target[row0:row1, col0:col1]).astype(np.int32) * live - right
        )

    def reappearances(self, source: npt.NDArray[np.bool_], wrong: npt.NDArray[np.bool_], count: int) -> list[_Shift]:
        """The ``count`` plain shifts under which the most of the start frame's ``source`` cells land on wrong
        cells they make right, best first."""
        colors = sorted(set(np.unique(self.start[source]).tolist()) & set(self.colors))
        if not colors:
            return []
        product = sum(
            np.fft.rfft2((self.target == color) & wrong, s=(_FFT_SIZE, _FFT_SIZE))
            * np.conj(np.fft.rfft2((self.start == color) & source, s=(_FFT_SIZE, _FFT_SIZE)))
            for color in colors
        )
        counts = np.rint(np.fft.irfft2(product, s=(_FFT_SIZE, _FFT_SIZE))).astype(np.int64).ravel()
        order = np.argpartition(-counts, count - 1)[:count]
        shortlist = sorted(
            (-int(counts[i]), _signed_shift(int(i) // _FFT_SIZE), _signed_shift(int(i) % _FFT_SIZE)) for i in order
        )
        return [(0, (row, col), -negated) for negated, row, col in shortlist]

    def wrong(self, board: Grid) -> npt.NDArray[np.bool_]:
        """The cells of the board that differ from the target, a wildcard in neither."""
        return (board != self.target) & self.live

    def moved(self, board: Grid, copy: BlockCopy) -> tuple[_Parts, int]:
        """The copy as a move when leaving a colour behind gains: the colour that makes the most of the block's
        old place right, at no cost; otherwise the copy as it is. With its exact gain either way. A plain copy
        is first shrunk to the smallest move writing the same, and a one-colour swap read with the more common
        colour left behind."""
        before = int(self.wrong(board).sum())
        gain = before - int(self.wrong(self.applied(board, copy)).sum())
        copy, gain = self.shrunk(board, copy, gain)
        if copy.clear is not None:
            return (self.swapped(copy),), gain
        rows, cols = copy.block_shape
        source = slice(copy.src_row, copy.src_row + rows), slice(copy.src_col, copy.src_col + cols)
        wrong = self.wrong(self.applied(board, copy))
        target, wrong_source = self.target[source], wrong[source]
        if not wrong_source.any():
            return (copy,), gain
        colors, counts = np.unique(target[wrong_source], return_counts=True)
        color = int(colors[int(np.argmax(counts))])
        move = BlockCopy(
            copy.row0,
            copy.row1,
            copy.col0,
            copy.col1,
            copy.src_row,
            copy.src_col,
            copy.rot,
            copy.flip,
            clear=color,
            distance_cost=self.distance_cost,
        )
        move_gain = before - int(self.wrong(self.applied(board, move)).sum())
        if move_gain <= gain:
            return (copy,), gain
        return (self.swapped(move),), move_gain

    def applied(self, board: Grid, transform: Transform) -> Grid:
        """A copy of the board with the transform applied."""
        child = board.copy()
        transform._apply(child, self.start)
        return child

    def shrunk(self, board: Grid, copy: BlockCopy, gain: int) -> tuple[BlockCopy, int]:
        """A plain copy as the smallest move that writes the same, if one does.

        A copy of a strip shifted by less than its height writes what lay
        beyond a sprite onto the sprite's old place: the same cells a move
        of the sprite alone, leaving that colour behind, would write at the
        same cost. For each colour the copy writes, the bounding box of the
        other cells it writes is tried as such a move; the smallest whose
        gain is at least the copy's wins, and at equal area the copy stays.
        """
        if copy.turned:
            return copy, gain
        rows, cols = copy.block_shape
        written = self.start[copy.src_row : copy.src_row + rows, copy.src_col : copy.src_col + cols]
        best, best_gain, best_area = copy, gain, rows * cols
        for color in np.unique(written).tolist():
            if color == WILDCARD:
                continue
            others = np.argwhere(written != color)
            if not len(others):
                continue
            (top, left), (bottom, right) = others.min(axis=0), others.max(axis=0) + 1
            if (bottom - top) * (right - left) >= best_area:
                continue
            move = BlockCopy(
                copy.row0 + top,
                copy.row0 + bottom,
                copy.col0 + left,
                copy.col0 + right,
                src_row=copy.src_row + top,
                src_col=copy.src_col + left,
                clear=int(color),
                distance_cost=self.distance_cost,
            )
            move_gain = int(self.wrong(board).sum()) - int(self.wrong(self.applied(board, move)).sum())
            if move_gain >= best_gain:
                best, best_gain, best_area = move, move_gain, (bottom - top) * (right - left)
        return best, best_gain

    def swapped(self, move: BlockCopy) -> BlockCopy:
        """A move of a one-colour block onto a one-colour place reads either way; the reading that leaves the
        more common of the two colours behind, by their counts in the start frame.

        Both readings write the same cells to the same values at the same
        cost, so this settles a tie, nothing more. A turned move is left
        alone: a turn of a one-colour block never beats the plain move.
        """
        if move.turned or move.clear is None:
            return move
        rows, cols = move.block_shape
        block = self.start[move.src_row : move.src_row + rows, move.src_col : move.src_col + cols]
        landing = self.start[move.region]
        if (block != block.flat[0]).any() or (landing != move.clear).any():
            return move
        moving, behind = int(block.flat[0]), move.clear
        if int((self.start == moving).sum()) <= int((self.start == behind).sum()):
            return move
        return BlockCopy(
            move.src_row,
            move.src_row + rows,
            move.src_col,
            move.src_col + cols,
            move.row0,
            move.col0,
            clear=moving,
            distance_cost=self.distance_cost,
        )

    def carved(self, board: Grid, rect: Rectangle, shifts: list[_Shift]) -> tuple[_Parts, int]:
        """The paste of ``rect``, or something cheaper: other transforms inside it and pastes of what they leave.

        Each paste is tried against the best few other transforms confined
        to its rectangle, each followed by pastes of the cells still wrong
        (the maximum-sum rectangles of the paste map, while a paste beats
        single cells); the best score replaces it when below the paste's.
        Cells that would be left for single-cell edits stay wrong for the
        search to reach later, and are not counted in the gain. Every paste
        that results is tried the same way, so the parts come out in the
        order they apply.
        """
        working = board.copy()
        parts: list[Transform] = []
        pending = [rect]
        while pending:
            rect = pending.pop(0)
            region = slice(rect.row0, rect.row1), slice(rect.col0, rect.col1)
            box = (rect.row0, rect.row1, rect.col0, rect.col1)
            wrong_now = self.wrong(working)
            plain = AsciiGrid.COST + AsciiGrid.CELL_COST * _area(rect) - PIXEL_COST * int(wrong_now[region].sum())
            best: tuple[float, _Parts, list[Rectangle]] | None = None
            # The transforms tried are confined to the rectangle, but a move clears its old place outside
            # it, so the gain is read over the whole frame.
            for _, inner, _ in self.ranked(working, shifts, box, paste=False)[: self.carve_tries]:
                child = working.copy()
                for transform in inner:
                    transform._apply(child, self.start)
                wrong_child = self.wrong(child)
                covers, leftover = self.cover(child, box)
                gain = int(wrong_now.sum()) - int(wrong_child.sum()) + int(wrong_child[region].sum()) - leftover
                cost = _cost(inner) + sum(AsciiGrid.COST + AsciiGrid.CELL_COST * _area(cover) for cover in covers)
                score = cost - PIXEL_COST * gain
                if score < plain and (best is None or score < best[0]):
                    best = (score, inner, covers)
            if best is None:
                # A cell any value suits keeps what the board holds.
                wanted = np.where(self.target[region] == WILDCARD, working[region], self.target[region])
                paste = AsciiGrid.from_array(rect.row0, rect.col0, wanted)
                paste._apply(working, self.start)
                parts.append(paste)
                continue
            for transform in best[1]:
                transform._apply(working, self.start)
            parts.extend(best[1])
            pending = best[2] + pending
        return tuple(parts), int(self.wrong(board).sum()) - int(self.wrong(working).sum())

    def cover(self, board: Grid, box: tuple[int, int, int, int]) -> tuple[list[Rectangle], int]:
        """Pastes for the wrong cells of ``box`` while a paste beats single cells, and how many cells are left."""
        row0, row1, col0, col1 = box
        wrong = self.wrong(board)[row0:row1, col0:col1]
        covers: list[Rectangle] = []
        while wrong.any():
            (found,) = _best_rectangles([_paste_map(wrong)])
            if found is None or found.gain <= AsciiGrid.COST:
                break
            covers.append(
                Rectangle(found.gain, found.row0 + row0, found.row1 + row0, found.col0 + col0, found.col1 + col0)
            )
            wrong[found.row0 : found.row1, found.col0 : found.col1] = False
        return covers, int(wrong.sum())

    def shifts(self, wrong: npt.NDArray[np.bool_]) -> list[_Shift]:
        """The shifts worth a rectangle search, as (variant, shift, wrong cells it would make right).

        For every shift at once, the count is the cross-correlation of the
        variant's colour planes with the wrong cells' colour planes of the
        target, summed over colours: one FFT product per colour. Each
        variant contributes its best few shifts over all wrong cells, the
        plain variant more of them: background matches at many shifts by
        coincidence, scattered where no rectangle can use them, and only
        the rectangle search can tell those from a mover's shift, so the
        plain variant, whose maps are cheap, gets a longer list. A mover is
        one component of wrong cells, and a small one loses the count to
        larger changes; so for each of the largest components the plain
        variant also contributes the shifts that reproduce the most of the
        component's whole bounding box, wrong cells and right ones alike,
        which is where that box came from.
        """
        found: list[_Shift] = []
        if not self.colors:
            return found
        # The wrong cells' planes are the same for every variant, so they are transformed once.
        planes = self.planes(wrong)
        for variant, spectrum in self.spectra.items():
            count = self.plain_shifts if variant == 0 else self.shifts_per_variant
            found.extend(self.correlate(variant, spectrum, planes, count))
        components = sorted(_components(wrong), key=len, reverse=True)[: self.shifts_per_variant * 2]
        for cells in components:
            rows, cols = np.array(cells).T
            mask = np.zeros_like(wrong)
            mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
            found.extend(self.correlate(0, self.spectra[0], self.planes(mask & self.live), 2))
        return list(dict.fromkeys(found))

    def planes(self, mask: npt.NDArray[np.bool_]) -> dict[int, Any]:
        """The Fourier transform of the masked target cells of each colour, for ``correlate``."""
        return {color: np.fft.rfft2((self.target == color) & mask, s=(_FFT_SIZE, _FFT_SIZE)) for color in self.colors}

    def correlate(self, variant: int, spectrum: dict[int, Any], planes: dict[int, Any], count: int) -> list[_Shift]:
        """The ``count`` shifts of one variant that make the most of the masked target cells right, best first;
        ``planes`` is ``self.planes(mask)``."""
        product = sum(planes[color] * spectrum[color] for color in self.colors)
        counts = np.rint(np.fft.irfft2(product, s=(_FFT_SIZE, _FFT_SIZE))).astype(np.int64).ravel()
        order = np.argpartition(-counts, count - 1)[:count]
        shortlist = sorted(
            (-int(counts[i]), _signed_shift(int(i) // _FFT_SIZE), _signed_shift(int(i) % _FFT_SIZE)) for i in order
        )
        return [(variant, (row, col), -negated) for negated, row, col in shortlist]

    def copy(self, variant: int, shift: tuple[int, int], rect: Rectangle) -> BlockCopy:
        """The copy that fills ``rect`` from the variant of the start frame shifted by ``shift``."""
        flip, rot = _VARIANTS[variant]
        indices = _variant(np.arange(GRID_SIZE * GRID_SIZE).reshape(GRID_SIZE, GRID_SIZE), flip, rot)
        block = indices[rect.row0 - shift[0] : rect.row1 - shift[0], rect.col0 - shift[1] : rect.col1 - shift[1]]
        return BlockCopy(
            rect.row0,
            rect.row1,
            rect.col0,
            rect.col1,
            src_row=int(block.min()) // GRID_SIZE,
            src_col=int(block.min()) % GRID_SIZE,
            rot=rot,
            flip=flip,
            distance_cost=self.distance_cost,
        )

    def lines(self, wrong: npt.NDArray[np.bool_]) -> Iterator[Line]:
        """A line between the extreme cells of each one-coloured component of wrong cells, each way round."""
        for cells in _components(wrong):
            if len(cells) < _min_gain(Line.COST):
                continue
            colors = {int(self.target[cell]) for cell in cells}
            if len(colors) != 1:
                continue
            (color,) = colors
            by_row, by_col = sorted(cells), sorted(cells, key=lambda cell: (cell[1], cell[0]))
            ends = [(by_row[0], by_row[-1]), (by_col[0], by_col[-1])]
            for first, last in dict.fromkeys(ends):
                yield Line(first, last, color)
                yield Line(last, first, color)


def _frame(grid: npt.ArrayLike, name: str, *, wildcards: bool) -> Grid:
    """``grid`` as a read-only int8 frame, wildcards allowed or not."""
    frame = np.array(grid)
    if frame.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{name} must be a {GRID_SIZE}x{GRID_SIZE} frame, got shape {frame.shape}")
    lowest = WILDCARD if wildcards else 0
    if frame.size and (frame.min() < lowest or frame.max() >= NUM_COLORS):
        raise ValueError(f"{name} must hold colours {lowest}-{NUM_COLORS - 1}, got {frame.min()}..{frame.max()}")
    frame = frame.astype(np.int8)
    frame.flags.writeable = False
    return frame


def _variant(frame: npt.NDArray[np.integer], flip: bool, rot: int) -> npt.NDArray[np.integer]:
    """The frame mirrored and turned the way ``BlockCopy`` mirrors and turns a block."""
    variant = np.fliplr(frame) if flip else frame
    return np.rot90(variant, rot) if rot else variant


def _signed_shift(index: int) -> int:
    """A correlation index back to a shift: the upper half of the FFT range is negative."""
    return index - _FFT_SIZE if index >= _FFT_SIZE // 2 else index


def _cost(parts: _Parts) -> float:
    """The parts' costs, summed."""
    return sum(part.bit_cost() for part in parts)


def _area(rect: Rectangle) -> int:
    """How many cells the rectangle holds."""
    return (rect.row1 - rect.row0) * (rect.col1 - rect.col0)


def _paste_map(wrong: npt.NDArray[np.bool_]) -> _Map:
    """A paste's gain per cell, its per-cell cost taken off: a wrong cell is worth a single-cell edit."""
    return (PIXEL_COST * wrong - AsciiGrid.CELL_COST).astype(np.int32)


def _min_gain(cost: float) -> int:
    """The smallest gain at which a transform of this cost scores below zero."""
    return int(cost // PIXEL_COST) + 1


def _best_rectangles(maps: list[_Map]) -> list[Rectangle | None]:
    """The maximum-sum subrectangle of each map, or None for a map with no positive cell.

    The rectangle never needs a cell outside the bounding box of the map's
    positive cells, since cropping to that box drops only cells at or below
    zero; each map is searched in its own box, and maps that share a box are
    searched together.
    """
    found: list[Rectangle | None] = [None] * len(maps)
    boxes: dict[tuple[int, int, int, int], list[int]] = {}
    for k, gain in enumerate(maps):
        rows, cols = np.nonzero(gain > 0)
        if rows.size:
            box = (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)
            boxes.setdefault(box, []).append(k)
    for (row0, row1, col0, col1), members in boxes.items():
        stack = np.stack([maps[k][row0:row1, col0:col1] for k in members])
        for k, rect in zip(members, _kadane(stack)):
            found[k] = Rectangle(rect.gain, rect.row0 + row0, rect.row1 + row0, rect.col0 + col0, rect.col1 + col0)
    return found


def _kadane(maps: _Map) -> list[Rectangle]:
    """The maximum-sum subrectangle of each map in a ``(K, H, W)`` stack.

    Every span of rows is summed into a strip, and Kadane's run over the
    columns of all strips at once finds each strip's best span of columns.
    A run restarts when its sum drops to zero or below, and a better sum
    must be strictly better, so a rectangle never grows by zero-gain cells
    at either end.
    """
    count, height, width = maps.shape
    prefix = np.zeros((count, height + 1, width), dtype=np.int32)
    np.cumsum(maps, axis=1, out=prefix[:, 1:])
    top, bottom = np.triu_indices(height)
    strips = prefix[:, bottom + 1, :] - prefix[:, top, :]
    spans = strips.shape[1]
    best = np.full((count, spans), np.iinfo(np.int32).min, dtype=np.int32)
    best_start = np.zeros((count, spans), dtype=np.int32)
    best_stop = np.zeros((count, spans), dtype=np.int32)
    run = np.zeros((count, spans), dtype=np.int32)
    run_start = np.zeros((count, spans), dtype=np.int32)
    for col in range(width):
        gain = strips[:, :, col]
        restart = run <= 0
        run = np.where(restart, gain, run + gain)
        run_start = np.where(restart, col, run_start)
        better = run > best
        best = np.where(better, run, best)
        best_start = np.where(better, run_start, best_start)
        best_stop = np.where(better, col + 1, best_stop)
    return [
        Rectangle(
            int(best[k, span]),
            int(top[span]),
            int(bottom[span]) + 1,
            int(best_start[k, span]),
            int(best_stop[k, span]),
        )
        for k, span in enumerate(best.argmax(axis=1).tolist())
    ]


def _components(mask: npt.NDArray[np.bool_]) -> list[list[tuple[int, int]]]:
    """The 8-connected components of the true cells, each as sorted (row, col) cells, in order of first cell."""
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, col in np.argwhere(mask):
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack = [(int(row), int(col))]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr in range(max(r - 1, 0), min(r + 2, GRID_SIZE)):
                for nc in range(max(c - 1, 0), min(c + 2, GRID_SIZE)):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        components.append(sorted(cells))
    return components
