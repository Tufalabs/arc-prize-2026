"""Runs of an animation's frames that one statement explains, written as a loop in place of a piece each.

``framediff.animation`` builds an animation frame by frame, one piece each. With
``Decomposer.animation_patterns`` three kinds of run are then looked for, and each is written as one loop
where that is shorter than the pieces it replaces:

- **Periodic**: every frame of the run is the same as the frame ``d`` before it: a blink, a cycle. Found
  in the pieces, which already copy a frame that repeats a built one.
- **Stepped**: every frame of the run is built from the one before it by the same edit, its numbers moving by
  a fixed step a frame: a bar growing a row a frame, a sprite sliding over a plain floor. Found in the pieces'
  own lines, so it reaches every kind of edit the search writes.
- **Sliding**: one shape, of any outline and of one or more colours, moving over a background that stays
  put, for at least three frames: by one offset a frame, written as that velocity, or along any path, written
  as the list of its places. Found in the frames themselves: the shape is matched across two frames
  (``framediff.objects``) and followed frame by frame where the frames changed, and the background is what the
  frames show around it. Each frame is then the background with the shape on top, which reads the same however
  mixed the floor it crosses.

Which runs are written is chosen over the whole animation for the most characters saved; runs never overlap.
Every block is run against its frames before it is returned (``framediff.animation``), loops included.
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Protocol

import numpy as np
import numpy.typing as npt

from framediff.grid import GRID_SIZE, Grid
from framediff.objects import matches, touching
from framediff.transforms import COLORS_NOTE, WILDCARD, NoOp, Transform

if TYPE_CHECKING:
    from framediff.animation import AnimationFrame

#: The fewest frames a stepped or sliding run covers: two frames that repeat an edit could be chance.
MIN_RUN = 3

#: The fewest frames a periodic run covers.
MIN_PERIODIC = 2

#: A script from one frame to another within the budget a frame's piece has, or None: what builds a sliding
#: run's background from the frame before the run (``framediff.animation`` passes the decomposer's).
Describe = Callable[[Grid, Grid], Transform | None]

Kind = Literal["periodic", "stepped", "sliding"]


class Names(Protocol):
    """The names a block is written against (``framediff.animation``)."""

    before: str
    frame: str
    dict: str
    board: str


def linear(offset: int, slope: int) -> str:
    """``offset + slope * k`` as the loop writes it: ``k + 3``, ``2 * k - 1``, ``40 - k``, ``7``."""
    if slope == 0:
        return str(offset)
    term = "k" if abs(slope) == 1 else f"{abs(slope)} * k"
    if slope > 0:
        return term if offset == 0 else f"{term} + {offset}" if offset > 0 else f"{term} - {-offset}"
    return f"-{term}" if offset == 0 else f"{offset} - {term}"


def _range(first: int, last: int, step: int) -> str:
    """The loop's range over frames ``first`` to ``last``."""
    if step != 1:
        return f"range({first}, {last + 1}, {step})"
    return f"range({last + 1})" if first == 0 else f"range({first}, {last + 1})"


def _span(frames: list[int]) -> str:
    """The frames a run covers, as its comment names them."""
    return (
        f"Frames {frames[0]}-{frames[-1]}"
        if frames[-1] - frames[0] == len(frames) - 1
        else (f"Frames {frames[0]}-{frames[-1]}, every {frames[1] - frames[0]}")
    )


@dataclass
class Run:
    """A run of built frames one loop explains; ``lines`` writes it against a block's names."""

    kind: Kind
    frames: list[int]
    #: Whether the board holds the run's last frame after it, as a piece leaves it.
    synced: bool
    #: Whether the run's loop edits the board, which must then hold the frame before the run.
    needs_board: ClassVar[bool] = False

    def lines(self, names: Names) -> list[str]:  # pragma: no cover - every kind writes its own
        """The run's lines, a blank one first."""
        raise NotImplementedError


@dataclass
class Periodic(Run):
    """Every frame the same as the frame ``distance`` before it."""

    distance: int = 0
    step: int = 1

    def lines(self, names: Names) -> list[str]:
        """A comment and the loop that copies each frame from the one ``distance`` back."""
        return [
            "",
            f"# {_span(self.frames)} repeat: each the same as the frame {self.distance} before it",
            f"for k in {_range(self.frames[0], self.frames[-1], self.step)}:",
            f"    {names.dict}[k] = {names.dict}[k - {self.distance}].copy()",
        ]


@dataclass
class Stepped(Run):
    """Every frame built from the one before it by the same edit, each number a fixed step on: ``parts``
    is the edit's text around its numbers, and number ``i`` reads ``offsets[i] + slopes[i] * k``."""

    parts: tuple[str, ...] = ()
    offsets: tuple[int, ...] = ()
    slopes: tuple[int, ...] = ()
    step: int = 1

    needs_board: ClassVar[bool] = True

    def lines(self, names: Names) -> list[str]:
        """A comment and the loop that applies the edit once a frame."""
        text = self.parts[0] + "".join(
            linear(offset, slope) + part for offset, slope, part in zip(self.offsets, self.slopes, self.parts[1:])
        )
        return [
            "",
            f"# {_span(self.frames)}: each from the one before it by the same edit, a step on",
            f"for k in {_range(self.frames[0], self.frames[-1], self.step)}:",
            *(f"    {line}" for line in text.splitlines()),
            f"    {names.dict}[k] = {names.board}.copy()",
        ]


@dataclass
class Sliding(Run):
    """One shape moving from frame ``start`` (-1, the frame before the move) over a background that stays put, the
    top-left cell of its box at ``places[i]`` on the run's i-th frame: the shape is read in its box at ``start``,
    the background is ``start`` with what the shape stood on: built by ``background``, a script from ``start``
    like any frame's piece, else copied from frame ``shows``, which shows it."""

    start: int = -1
    shows: int = 0
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    colors: tuple[int, ...] = ()
    places: tuple[tuple[int, int], ...] = ()
    background: Transform | None = None
    #: Whether the shape fills its box, so that it is drawn as the box itself, with no mask to pick it out.
    full: bool = False

    @property
    def velocity(self) -> tuple[int, int] | None:
        """The offset the shape moves by on every frame, when it is one."""
        points = [(self.box[0], self.box[2]), *self.places]
        steps = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])}
        return next(iter(steps)) if len(steps) == 1 else None

    def lines(self, names: Names) -> list[str]:
        """The shape, its mask, the background, and the loop that draws the shape where it is on each frame: a
        step on from the last at one velocity, else at the places listed."""
        source = names.before if self.start < 0 else f"{names.dict}[{self.start}]"
        row0, row1, col0, col1 = self.box
        height, width = row1 - row0, col1 - col0
        box = f"{row0}:{row1}, {col0}:{col1}"
        pick = f"shape == {self.colors[0]}" if len(self.colors) == 1 else f"np.isin(shape, {list(self.colors)})"
        frames = _range(self.frames[0], self.frames[-1], 1)
        velocity = self.velocity
        if velocity is not None:
            rows, cols = velocity
            how = f"moving {_offset(rows, cols)} a frame"
            loop = f"for k in {frames}:"
            place = (
                f"{linear(row0 - rows * self.start, rows)}:{linear(row1 - rows * self.start, rows)}, "
                f"{linear(col0 - cols * self.start, cols)}:{linear(col1 - cols * self.start, cols)}"
            )
        else:
            how = "moving along a path"
            tops, lefts = [top for top, _ in self.places], [left for _, left in self.places]
            if len(set(tops)) == 1:
                loop = f"for k, col in zip({frames}, {lefts}):"
                place = f"{tops[0]}:{tops[0] + height}, col:col + {width}"
            elif len(set(lefts)) == 1:
                loop = f"for k, row in zip({frames}, {tops}):"
                place = f"row:row + {height}, {lefts[0]}:{lefts[0] + width}"
            else:
                loop = f"for k, (row, col) in zip({frames}, {list(self.places)}):"
                place = f"row:row + {height}, col:col + {width}"
        if self.background is not None:
            # The board without the shape, built from the frame before the run as any frame's piece is.
            background = [
                f"bg = {source}.copy()  # the board without the shape",
                *self.background._code_lines(source, "bg"),  # pyright: ignore[reportPrivateUsage]
            ]
        else:
            background = [
                f"bg = {source}.copy()",
                f"bg[{box}]{'' if self.full else '[mask]'} = {names.frame.format(k=self.shows)}[{box}]"
                f"{'' if self.full else '[mask]'}  # what it stood on, as frame {self.shows} shows it",
            ]
        return [
            "",
            f"# {_span(self.frames)}: the shape below {how} over a background that stays put",
            f"shape = {source}[{box}].copy()",
            *([] if self.full else [f"mask = {pick}"]),
            *background,
            loop,
            f"    {names.board} = bg.copy()",
            f"    {names.board}[{place}] = shape" if self.full else f"    {names.board}[{place}][mask] = shape[mask]",
            f"    {names.dict}[k] = {names.board}.copy()",
        ]


def _offset(rows: int, cols: int) -> str:
    """An offset as the comments say it: ``down 1, right 2``."""
    parts = [f"{'down' if rows > 0 else 'up'} {abs(rows)}"] if rows else []
    parts += [f"{'right' if cols > 0 else 'left'} {abs(cols)}"] if cols else []
    return ", ".join(parts)


@dataclass
class Choice:
    """The runs chosen for an animation, and what they save."""

    runs: list[Run] = field(default_factory=lambda: list[Run]())
    saved: int = 0


def _template(text: str) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """``text`` split around its integer literals: the text between them, and their values. None for a text
    holding a string literal over several lines, whose lines a loop would indent, or a number not a whole one."""
    if '"""' in text or "'''" in text:
        return None
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    parts: list[str] = []
    numbers: list[int] = []
    last = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.NUMBER:
                continue
            if not token.string.isdigit():
                return None
            begin = starts[token.start[0] - 1] + token.start[1]
            parts.append(text[last:begin])
            numbers.append(int(token.string))
            last = begin + len(token.string)
    except (tokenize.TokenError, IndentationError):
        return None
    parts.append(text[last:])
    return tuple(parts), tuple(numbers)


def _edit_text(record: AnimationFrame, names: Names) -> str | None:
    """The lines of a frame's own edit, against its source in the dict, or None when it has none; without the
    colours a copy names, which change from frame to frame where the edit does not, and which one loop's text
    could not say for every frame."""
    if record.transform is None or isinstance(record.transform, NoOp) or record.earlier is None:
        return None
    source = f"{names.dict}[{record.earlier}]"
    return COLORS_NOTE.sub("", "\n".join(record.transform._code_lines(source, names.board)))  # pyright: ignore[reportPrivateUsage]


def _repeats(record: AnimationFrame) -> int | None:
    """How many frames back the frame a record copies lies, when the record is a copy of a built frame."""
    if not isinstance(record.transform, NoOp) or record.earlier is None:
        return None
    return record.index - record.earlier


def _periodic(records: list[AnimationFrame], step: int) -> list[tuple[int, int, Run]]:
    """Every maximal periodic run, as (first position, last position, run)."""
    found: list[tuple[int, int, Run]] = []
    position = 0
    while position < len(records):
        distance = _repeats(records[position])
        if distance is None:
            position += 1
            continue
        end = position
        while end + 1 < len(records) and _repeats(records[end + 1]) == distance:
            end += 1
        if end - position + 1 >= MIN_PERIODIC:
            frames = [r.index for r in records[position : end + 1]]
            found.append((position, end, Periodic("periodic", frames, False, distance=distance, step=step)))
        position = end + 1
    return found


def _stepped(records: list[AnimationFrame], names: Names, step: int) -> list[tuple[int, int, Run]]:
    """Every maximal stepped run, as (first position, last position, run)."""
    templates: list[tuple[tuple[str, ...], tuple[int, ...]] | None] = []
    for position, record in enumerate(records):
        chained = position > 0 and record.earlier == records[position - 1].index
        text = _edit_text(record, names) if chained else None
        templates.append(None if text is None else _template(text))
    found: list[tuple[int, int, Run]] = []
    position = 0
    while position < len(records):
        first = templates[position]
        if first is None or position + 1 >= len(records) or templates[position + 1] is None:
            position += 1
            continue
        parts, numbers = first
        second = templates[position + 1]
        assert second is not None
        if second[0] != parts:
            position += 1
            continue
        deltas = tuple(b - a for a, b in zip(numbers, second[1]))
        end, current = position + 1, second
        while end + 1 < len(records):
            following = templates[end + 1]
            if following is None or following[0] != parts:
                break
            if tuple(b - a for a, b in zip(current[1], following[1])) != deltas:
                break
            end, current = end + 1, following
        if end - position + 1 >= MIN_RUN and all(delta % step == 0 for delta in deltas):
            slopes = tuple(delta // step for delta in deltas)
            k0 = records[position].index
            offsets = tuple(number - slope * k0 for number, slope in zip(numbers, slopes))
            frames = [r.index for r in records[position : end + 1]]
            run = Stepped("stepped", frames, True, parts=parts, offsets=offsets, slopes=slopes, step=step)
            found.append((position, end, run))
            position = end
        else:
            position += 1
    return found


def _slide(
    frames: list[Grid],
    start: int,
    cells: frozenset[tuple[int, int]],
    colors: tuple[int, ...],
    offset: tuple[int, int],
    describe: Describe | None = None,
) -> list[Sliding]:
    """The run from ``frames[start]`` on (an index into ``frames``, which leads with the frame before the move) in
    which the shape ``cells``, which moves by ``offset`` to the next frame, goes on moving over a background that
    stays put, followed for as long as each frame shows it somewhere it could have moved to; and, where that
    path opens with three or more frames at one velocity before it turns, that opening too, whose loop is
    shorter. Empty under ``MIN_RUN`` frames."""
    rows, cols = zip(*cells)
    row0, row1, col0, col1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
    mask = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for row, col in cells:
        mask[row - row0, col - col0] = True
    shape = frames[start][row0:row1, col0:col1].copy()
    if not np.array_equal(np.isin(shape, colors), mask):
        return []  # the loop picks the shape by its colours, which would pick other cells of its box too
    places: list[tuple[int, int]] = []
    here, step = (row0, col0), offset
    for index in range(start + 1, len(frames)):
        place = _place(frames[index - 1], frames[index], shape, mask, here, step)
        if place is None:
            break
        here, step = place, (place[0] - here[0], place[1] - here[1])
        places.append(place)
    found = [run for run in (_background(frames, start, (row0, col0), shape, mask, colors, places, describe),) if run]
    if found and found[0].velocity is None:
        opening = _opening(row0, col0, places)
        if opening >= MIN_RUN:
            run = _background(frames, start, (row0, col0), shape, mask, colors, places[:opening], describe)
            if run is not None:
                found.append(run)
    return found


def _opening(row0: int, col0: int, places: list[tuple[int, int]]) -> int:
    """How many of ``places`` the shape reaches at the velocity of its first step."""
    points = [(row0, col0), *places]
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    return next((i for i, step in enumerate(steps) if step != steps[0]), len(steps))


def _background(
    frames: list[Grid],
    start: int,
    origin: tuple[int, int],
    shape: Grid,
    mask: npt.NDArray[np.bool_],
    colors: tuple[int, ...],
    places: list[tuple[int, int]],
    describe: Describe | None = None,
) -> Sliding | None:
    """The run of the shape, its box at ``origin`` in ``frames[start]``, at ``places`` on the frames after it, its
    background the start frame with what the shape stood on as the latest frame of the run shows it that shows
    every cell of it the run ever uncovers (the cells it never leaves are under it on every frame); shortened
    until every frame of it is that background with the shape on top, and None under ``MIN_RUN`` frames."""
    row0, col0 = origin
    height, width = mask.shape
    stood = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    stood[row0 : row0 + height, col0 : col0 + width] = mask
    visible: list[npt.NDArray[np.bool_]] = []
    for top, left in places:
        covered = np.zeros_like(stood)
        covered[top : top + height, left : left + width] = mask
        visible.append(stood & ~covered)
    places = list(places)
    while len(places) >= MIN_RUN:
        last = start + len(places)
        needed = np.logical_or.reduce(visible[: len(places)])
        shows = next((i for i in range(len(places) - 1, -1, -1) if not (needed & ~visible[i]).any()), None)
        if shows is None:
            places.pop()
            continue
        background = frames[start].copy()
        background[row0 : row0 + height, col0 : col0 + width][mask] = frames[start + 1 + shows][
            row0 : row0 + height, col0 : col0 + width
        ][mask]
        if all(
            np.array_equal(_drawn(background, shape, mask, *place), frames[start + 1 + i])
            for i, place in enumerate(places)
        ):
            # What the background is to be, for a script to build: the cells the shape uncovers as a frame of the
            # run shows them, and those it never leaves wildcards, since the shape covers them on every frame.
            wanted = background.copy()
            wanted[stood & ~needed] = WILDCARD
            return Sliding(
                "sliding",
                list(range(start, last)),  # frame indices: frames[i] is animation frame i - 1
                True,
                start=start - 1,
                shows=start + shows,
                box=(row0, row0 + height, col0, col0 + width),
                colors=colors,
                places=tuple(places),
                background=None if describe is None else describe(frames[start], wanted),
                full=bool(mask.all()),
            )
        places.pop()
    return None


def _place(
    previous: Grid,
    frame: Grid,
    shape: Grid,
    mask: npt.NDArray[np.bool_],
    here: tuple[int, int],
    step: tuple[int, int],
) -> tuple[int, int] | None:
    """Where the shape, its box at ``here`` in ``previous``, is in ``frame``: where its last ``step`` takes it, else
    the nearest place that covers every cell that changed outside its old box. None where it stood still, or
    where more changed than one shape could cover (the background did not stay put)."""
    height, width = mask.shape

    def shows(top: int, left: int) -> bool:
        """Whether ``frame`` holds the shape with its box at ``(top, left)``."""
        inside = 0 <= top and top + height <= GRID_SIZE and 0 <= left and left + width <= GRID_SIZE
        return inside and bool(np.array_equal(frame[top : top + height, left : left + width][mask], shape[mask]))

    changed = frame != previous
    if not changed.any():
        return None
    guess = (here[0] + step[0], here[1] + step[1])
    if guess != here and shows(*guess):
        return guess
    outside = changed.copy()
    outside[here[0] : here[0] + height, here[1] : here[1] + width] = False
    if outside.any():
        rows, cols = np.nonzero(outside)
        if rows.max() - rows.min() >= height or cols.max() - cols.min() >= width:
            return None
        tops = range(int(rows.max()) - height + 1, int(rows.min()) + 1)
        lefts = range(int(cols.max()) - width + 1, int(cols.min()) + 1)
    else:
        tops = range(here[0] - height + 1, here[0] + height)
        lefts = range(here[1] - width + 1, here[1] + width)
    places = sorted(
        ((top, left) for top in tops for left in lefts if (top, left) != here),
        key=lambda place: (abs(place[0] - guess[0]) + abs(place[1] - guess[1]), place),
    )
    return next((place for place in places if shows(*place)), None)


def _drawn(background: Grid, shape: Grid, mask: npt.NDArray[np.bool_], top: int, left: int) -> Grid:
    """``background`` with the cells of ``shape`` under ``mask`` drawn at ``(top, left)``."""
    board = background.copy()
    board[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = shape[mask]
    return board


def _sliding(
    start: Grid, grids: list[Grid], records: list[AnimationFrame], describe: Describe | None = None
) -> list[tuple[int, int, Run]]:
    """Every sliding run over the built frames, as (first position, last position, run). The frames are built at
    stride 1, so a frame's position is its index; they may be the first frames only, when a stride is judged
    before it is built whole, and a run then ends at the last of them."""
    if [record.index for record in records] != list(range(len(records))):
        return []
    frames = [start, *grids[: len(records)]]
    found: list[tuple[int, int, Run]] = []
    covered = -1
    for index in range(len(frames) - MIN_RUN):
        if index < covered:
            continue
        best: list[Sliding] = []
        for cells, colors, offset in movers(frames[index], frames[index + 1]):
            runs = _slide(frames, index, cells, colors, offset, describe)
            if runs and (not best or len(runs[0].frames) > len(best[0].frames)):
                best = runs
        found += [(run.frames[0], run.frames[-1], run) for run in best]
        if best:
            covered = best[0].frames[-1] + 1
    return found


def movers(start: Grid, target: Grid) -> list[tuple[frozenset[tuple[int, int]], tuple[int, ...], tuple[int, int]]]:
    """Every shape that moved without turning from ``start`` to ``target``: each matched piece, and each group of
    touching pieces that moved by one offset, as (cells in ``start``, colours, offset)."""
    plain = [m for m in matches(start, target) if not (m.flip or m.rot)]
    found: list[tuple[frozenset[tuple[int, int]], tuple[int, ...], tuple[int, int]]] = [
        (m.start.cells, (m.start.color,), m.offset) for m in plain
    ]
    by_offset: dict[tuple[int, int], list[Any]] = {}
    for match in plain:
        by_offset.setdefault(match.offset, []).append(match)
    for offset, group in by_offset.items():
        for joined in touching(group):
            if len(joined) > 1:
                cells = frozenset(cell for m in joined for cell in m.start.cells)
                found.append((cells, tuple(sorted({m.start.color for m in joined})), offset))
    return found


def choose(
    start: Grid,
    grids: list[Grid],
    records: list[AnimationFrame],
    names: Names,
    step: int,
    piece_chars: list[int],
    describe: Describe | None = None,
) -> Choice:
    """The runs that save the most characters together, none overlapping: ``piece_chars[i]`` is what the piece
    of ``records[i]`` writes. ``describe`` builds a sliding run's background (``Describe``); without it, the
    background is copied from a frame of the run that shows it."""
    candidates = _periodic(records, step) + _stepped(records, names, step)
    if step == 1:
        candidates += _sliding(start, grids, records, describe)
    saving: dict[int, list[tuple[int, int, Run]]] = {}
    for first, last, run in candidates:
        gain = sum(piece_chars[first : last + 1]) - len("\n".join(run.lines(names))) - 1
        if gain > 0:
            saving.setdefault(last, []).append((first, gain, run))
    # Most characters saved over the first i positions, and the runs that save them.
    best: list[tuple[int, list[Run]]] = [(0, [])]
    for position in range(len(records)):
        value, runs = best[-1]
        for first, gain, run in saving.get(position, []):
            before, chosen = best[first]
            if before + gain > value:
                value, runs = before + gain, [*chosen, run]
        best.append((value, runs))
    value, runs = best[-1]
    return Choice(runs, value)
