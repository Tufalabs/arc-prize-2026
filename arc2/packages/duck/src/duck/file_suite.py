"""The file-editing tools the model is given, over one :class:`Workspace`.

Five tools over the model's working directory: a numbered view that takes a
range and lists the directory, a grep-style search, an exact-substring
replacement that reports where the change landed and explains a failed match,
whole-file create, and a one-step undo. The same directory is the kernel's, so
a file written here is a module a snippet can import.

Every string in this module is prompt text. So is every tool name, every
description, every parameter schema, and the order :func:`file_tools` lists
them in. A change to any of it is a change to what the model reads, with a
reference regeneration behind it.

This is the duck's own surface, deliberately not shared with any other
agent's equivalent: for a prompt surface the code IS the bytes, and each
agent's committed references pin its own — so a shared implementation would
let one agent's wording change force another's regeneration. Keep them
separate; they are free to diverge.
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import os
import time
from dataclasses import dataclass, field
from typing import Any

import regex
import ullm.conv
import ullm.tool

from duck import module_names
from duck.workspace import MAX_READ_BYTES, Change, Entry, NotText, PathNotAllowed, Workspace, flat_note

#: Lines shown either side of a change. Enough to place it, few enough that
#: an edit's result is not a second copy of the file.
CONTEXT_LINES = 4

#: A whole view stops here. A safety valve, not a ration: a truncated view the
#: model wanted costs a second call, so it pays for the file twice.
VIEW_CAP = 2000

#: Most lines an echoed region shows. Without it the echo is bounded only by
#: the size of the change, so a large replacement answers with a large slice
#: of the file — the thing this surface exists to stop doing.
MAX_ECHO_LINES = 40

#: Characters of one line any rendering shows. Every other bound here counts
#: lines, and a line has no length limit — one minified JSON line defeats all
#: of them at once. Capping the line rather than the answer keeps every line
#: and its number visible, which a cap on the finished text does not.
MAX_LINE_CHARS = 400

#: A ceiling on the one knob the model still sets, so it cannot ask for a
#: context window wider than the file.
MAX_CONTEXT_LINES = 10

#: Hits reported before the answer says how many more there were. Fixed rather
#: than model-settable: a cap the model can raise is a cap that does not bound
#: anything.
MATCH_CAP = 50

#: Replacements at most this many lines long are echoed verbatim inside the
#: frame. The redundancy that justified the frame is dominated by large
#: replacements; a one-line edit's confirmation costs about its own length,
#: and it is exactly the case where seeing the merged line catches a splice
#: mistake.
FILL_ECHO_LINES = 2

#: An ``old_string`` longer than this gets no near-miss scan. The scan
#: compares it against every same-sized window of the file, so its cost grows
#: with both, and a quotation that large has no one near miss worth showing.
NEAR_MISS_MAX_CHARS = 2_000

#: How long the model's own pattern may run before the search gives up.
#: A backtracking pattern is superlinear in the subject, so a line only a
#: little longer than the last turns a fast search into an unbounded one --
#: and this runs on the event loop, where nothing else in that game's
#: process proceeds meanwhile. `re` cannot be interrupted, so the search
#: uses a matcher that honours a deadline rather than one wrapped in a
#: timeout that could never fire.
SEARCH_TIMEOUT_S = 10.0

#: Below this ratio, a failed match has no near miss worth showing — printing
#: the least-bad window of a file the model did not mean is worse than saying
#: nothing.
NEAR_MISS_RATIO = 0.6


def _reason(error: OSError) -> str:
    """Why an OS call failed, without the path it failed on.

    ``str(error)`` renders the filename, which is the resolved location — host
    specific, absent from the model's world, and different on every run.
    """
    return error.strerror or type(error).__name__


def not_text_message(error: NotText) -> str:
    """What the model is told about a file no tool can read.

    The failure it replaces — a ``UnicodeDecodeError``, which is not an
    ``OSError`` — escaped every tool's error handling.
    """
    return f"'{error.path}' is not a text file ({error.size} bytes). The file tools read and write text only."


@dataclass
class WorkspaceTool(ullm.tool.Tool):
    """A tool that acts on the model's working directory.

    One choke point for what every tool needs and none should implement twice:
    a refused path is explained the same way whichever tool was called.
    Subclasses return **text**, and override :meth:`_act` (path-taking) or
    :meth:`_answer` (not).
    """

    #: Defaulted only because the base class is a dataclass; every instance is
    #: constructed with one.
    workspace: Workspace | None = None
    #: Whether ``undo_edit`` is in the suite this tool belongs to. Read by the
    #: tools that edit, whose recovery hint names it: a hint that sends the
    #: model to a tool it was not given costs it a wasted call. Set by
    #: :func:`file_tools` from the same gate that decides the suite.
    undo_edit_offered: bool = True

    def _require_workspace(self) -> Workspace:
        """The workspace this tool acts on."""
        assert self.workspace is not None, "file tool used before it was given a workspace"
        return self.workspace

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Run the tool.

        Deliberately no budget applied here — the duck's ``tool_output_tokens``
        governs the python tool and does not reach these. A cap on the finished
        answer is a band-aid over a tool that produces too much: it lands
        mid-sentence and leaves the model with a mangled version of a question
        it asked correctly. What bounds these answers is their construction —
        every rendering is capped in lines, every line in characters, and the
        two knobs the model sets have ceilings.
        """
        try:
            text = await self._answer(arguments)
        except PathNotAllowed as error:
            # Caught here rather than beside the path that raised it: the tools
            # that take no single path override `_answer`, and a refusal
            # escaping one of those would end the game instead of answering.
            self.is_error = True
            text = self._require_workspace().denial(error.path)
        # The flat contract's enforcement half runs after every call. File
        # tools cannot create directories themselves, so this fires only for
        # what slipped in from outside — and says so, because deleting
        # something the model wrote without telling it is worse than the
        # directory was.
        removed = self._require_workspace().sweep()
        if removed:
            text += flat_note(removed)
        return ullm.conv.StringContent(text=text)

    async def _answer(self, arguments: dict[str, Any]) -> str:
        """Act on the path this tool was given."""
        return await self._act(str(arguments["path"]), arguments)

    async def _act(self, path: str, arguments: dict[str, Any]) -> str:
        """Do this tool's work on ``path``. ``path`` is the model's own text
        and is the only form any message may name."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rendering. One implementation, so a line looks the same whichever tool
# produced it — a view, an edit's result and a search hit are all the same
# shape, and the model learns it once.
# ---------------------------------------------------------------------------


def clipped(line: str) -> str:
    """One line, short enough to be worth reading.

    A line is unbounded, so every line-counting cap elsewhere is defeated by a
    single long one. Marked with what was left off, so the model can tell a
    clipped line from a short one and reach for Python if it needs the rest.
    """
    if len(line) <= MAX_LINE_CHARS:
        return line
    return f"{line[:MAX_LINE_CHARS]}... (+{len(line) - MAX_LINE_CHARS} more characters on this line)"


def numbered(lines: list[str], first: int, width: int) -> str:
    """``lines`` prefixed with their numbers, starting at ``first``.

    A tab separates the gutter from the line, not spaces. With spaces the
    boundary is ambiguous — the gutter is as wide as the file is long, and a
    line's own indentation continues straight on from it, so quoting a line
    back means guessing how many of those spaces were ours. Measured: 78% of
    failed edits in the first run were text that was right and indentation
    that was not. A tab is also what the tools models are trained against
    use, and what their descriptions tell them to strip.
    """
    return "\n".join(f"{first + offset:>{width}}\t{clipped(line)}" for offset, line in enumerate(lines))


def _width(total: int) -> int:
    """How wide a line number is for a file of ``total`` lines."""
    return len(str(max(1, total)))


def region(content: str, first: int, last: int, *, context: int = CONTEXT_LINES) -> str:
    """Lines ``first``..``last`` of ``content`` with context, numbered and capped.

    An empty result says so rather than rendering a range: a file emptied by
    the change has no lines to quote, and "Lines 1-0 now read:" followed by
    nothing is a worse answer than a sentence.
    """
    lines = content.splitlines()
    if not lines:
        return "The file is now empty."
    low = max(1, first - context)
    high = min(len(lines), last + context)
    shown = lines[low - 1 : high]
    tail = ""
    if len(shown) > MAX_ECHO_LINES:
        hidden = len(shown) - MAX_ECHO_LINES
        shown = shown[:MAX_ECHO_LINES]
        high = low + MAX_ECHO_LINES - 1
        tail = f"\n... and {hidden} more changed line{'s' if hidden != 1 else ''}."
    heading = f"Line {low} now reads:" if low == high else f"Lines {low}-{high} now read:"
    return f"{heading}\n\n" + numbered(shown, low, _width(len(lines))) + tail


def framed(content: str, start: int, end: int, *, context: int = CONTEXT_LINES) -> str:
    """Where a change landed: its neighbours always, the change itself only
    when it is short.

    What an edit tells the model that it does not already know is *where* its
    text landed and what now sits either side of it. The text itself it wrote
    one message ago: measured over a full run, 80% of an echo's characters
    were the model's own ``new_string`` handed straight back — dominated by
    large replacements. So a replacement up to :data:`FILL_ECHO_LINES` is
    shown verbatim (its confirmation is cheap, and the merged line is where a
    splice mistake shows), and a longer one is named rather than repeated.
    """
    lines = content.splitlines()
    if not lines:
        return "The file is now empty."
    first = content.count("\n", 0, start) + 1
    if end <= start and content[start : start + 1] == "\n":
        # A deletion that quoted the leading newline leaves `start` on the
        # terminator of the line before the join, so the text that went was
        # the line after it -- where deleting "l3\n" instead lands directly on
        # it. Both spellings remove the same line and must say the same thing.
        first += 1
    last = content.count("\n", 0, max(start, end - 1)) + 1
    changed = 0 if end <= start else last - first + 1
    # Three wide at least: the `...` marker sits in the gutter, and a gutter
    # narrower than the marker misaligns every line around it.
    width = max(_width(len(lines)), 3)

    above = lines[max(0, first - 1 - context) : first - 1]
    below = lines[first - 1 + changed : first - 1 + changed + context]
    parts: list[str] = []
    if above:
        parts.append(numbered(above, first - len(above), width))
    if not changed:
        parts.append(f"{'...':>{width}}\t(removed)")
        where = f"the removed text was at line {first}"
    elif changed <= FILL_ECHO_LINES:
        parts.append(numbered(lines[first - 1 : first - 1 + changed], first, width))
        where = (
            f"your replacement is at line {first}" if first == last else f"your replacement is at lines {first}-{last}"
        )
    else:
        parts.append(f"{'...':>{width}}\t(your {_lines_word(changed)})")
        where = f"your replacement is at lines {first}-{last}"
    if below:
        parts.append(numbered(below, first + changed, width))
    return f"{where}:\n\n" + "\n".join(parts)


def _joined(numbers: list[int]) -> str:
    """``[1, 2, 3]`` as ``1, 2 and 3`` — for a sentence, not a list."""
    if len(numbers) == 1:
        return str(numbers[0])
    return ", ".join(str(n) for n in numbers[:-1]) + f" and {numbers[-1]}"


def _lines_of(content: str, needle: str) -> list[int]:
    """Every 1-based line where ``needle`` starts.

    Non-overlapping, like ``str.count`` and ``str.replace`` — the messages
    these numbers appear in quote those counts, and two rules would let the
    sentence disagree with itself.
    """
    found: list[int] = []
    if not needle:
        return found
    at = content.find(needle)
    while at != -1:
        found.append(content.count("\n", 0, at) + 1)
        at = content.find(needle, at + len(needle))
    return found


def _positions_of(content: str, needle: str) -> list[int]:
    """Every character offset where ``needle`` starts, non-overlapping."""
    found: list[int] = []
    if not needle:
        return found
    at = content.find(needle)
    while at != -1:
        found.append(at)
        at = content.find(needle, at + len(needle))
    return found


def _some_lines(numbers: list[int], cap: int = 10) -> str:
    """``numbers`` for a sentence, capped — a common ``old_string`` can occur
    hundreds of times, and every bound elsewhere is by construction."""
    if len(numbers) <= cap:
        return _joined(numbers)
    head = ", ".join(str(n) for n in numbers[:cap])
    return f"{head}, and {len(numbers) - cap} more"


def parse_note(path: str, content: str, *, undo_edit: bool = True) -> str:
    """A warning when ``content`` is not valid Python, or ``""``.

    ``undo_edit`` is whether that tool is offered: the way out the note
    suggests has to be one the model has.

    Appended rather than blocking. Rejecting the write would break a
    legitimate sequence of edits whose intermediate states do not parse, make
    an edit's success depend on a property of the whole file the model was not
    reasoning about, and cost most where files are largest — a rejected
    whole-file write has to be regenerated in full.
    """
    if not path.endswith(".py"):
        return ""
    try:
        ast.parse(content)
    except SyntaxError as error:
        where = f" at line {error.lineno}" if error.lineno else ""
        way_out = "fix it or call undo_edit()" if undo_edit else "fix it"
        return (
            f"\n\nNote: {path} does not parse yet — SyntaxError{where}: {error.msg}. "
            f"If you are part-way through a sequence of edits, carry on; otherwise {way_out}."
        )
    return ""


def near_miss(content: str, old_string: str) -> str:
    """The closest thing in ``content`` to ``old_string``, or ``""``.

    A bare "not found" leaves the model nothing to act on but a re-read, and
    re-reading is what this surface exists to avoid. The common cause is
    indentation, so that case is named rather than left to be spotted.
    """
    if len(old_string) > NEAR_MISS_MAX_CHARS:
        return ""
    wanted = old_string.splitlines() or [old_string]
    lines = content.splitlines()
    if not lines:
        return ""
    best_ratio, best_at = 0.0, 0
    # One matcher, quotation cached as seq2; the cheap ratios gate the real
    # one, so the scan stays linear-ish over the file instead of quadratic.
    matcher = difflib.SequenceMatcher(None, "", "\n".join(wanted))
    for start in range(max(1, len(lines) - len(wanted) + 1)):
        matcher.set_seq1("\n".join(lines[start : start + len(wanted)]))
        if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_at = ratio, start
    if best_ratio < NEAR_MISS_RATIO:
        return ""
    window = lines[best_at : best_at + len(wanted)]
    indent_only = [line.strip() for line in window] == [line.strip() for line in wanted]
    # Bounded by the same rule as an echoed region: `old_string` is the
    # model's, so without this a quotation of any size comes straight back.
    hidden = max(0, len(window) - MAX_ECHO_LINES)
    shown = numbered(window[:MAX_ECHO_LINES], best_at + 1, _width(len(lines)))
    if hidden:
        shown += f"\n... and {hidden} more line{'s' if hidden != 1 else ''}."
    if indent_only:
        # The number prefix contributes leading space of its own, which is the
        # one thing this message is about — so say what differs rather than
        # leaving it to be seen.
        return f"\n\nClosest match, at line {best_at + 1} — same text, indented differently. The file has:\n\n{shown}"
    return f"\n\nClosest match, at line {best_at + 1}:\n\n{shown}"


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


@dataclass
class ViewFileTool(WorkspaceTool):
    """Show a file with line numbers, or list the directory."""

    name: str = "view_file"
    description: str = (
        "View a file in your working directory. Each line is prefixed with "
        "its number and a tab. Pass `view_range` to see part of it, or `.` "
        "to list the directory."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to view, or `.` to list the working directory"},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    # Exactly two, enforced here: the schema is checked before
                    # the tool runs, and a one-element range would otherwise
                    # index past the end — which raises, and a raise out of a
                    # tool is not handed to the model, it ends the game.
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "First and last line to show, 1-based and inclusive; -1 as the last means end of file",
                },
            },
            "required": ["path"],
        }
    )

    async def _act(self, path: str, arguments: dict[str, Any]) -> str:
        """Render the file, the requested range, or the working directory."""
        workspace = self._require_workspace()
        if path in (".", "./", ""):
            return directory_listing(workspace)
        try:
            oversized = _too_large(workspace, path)
            if oversized is not None:
                self.is_error = True
                return oversized
            content = workspace.read(path)
        except IsADirectoryError:
            # Reachable only inside the call that created the directory — the
            # sweep removes it before any later call could look.
            self.is_error = True
            return (
                f"'{path}' is a directory. Your working directory is flat — "
                "keep every file directly in it, with no subdirectories."
            )
        except FileNotFoundError:
            self.is_error = True
            return f"File '{path}' does not exist."
        except NotText as error:
            self.is_error = True
            return not_text_message(error)
        except OSError as error:
            self.is_error = True
            return f"Error reading '{path}': {_reason(error)}."

        lines = content.splitlines()
        total = len(lines)
        width = _width(total)
        window = arguments.get("view_range")
        if window is None:
            if total > VIEW_CAP:
                head = numbered(lines[:VIEW_CAP], 1, width)
                return (
                    f"{path} ({_lines_word(total)}) — showing lines 1-{VIEW_CAP}. "
                    f"Pass view_range to see the rest.\n{head}"
                )
            if not lines:
                return f"{path} is empty."
            return f"{path} ({_lines_word(total)})\n{numbered(lines, 1, width)}"

        first, last = int(window[0]), int(window[1])
        last = total if last == -1 else last
        # Two different mistakes, and calling a backwards range "outside" sends
        # the model looking at the file's length for a problem that is in its
        # own argument.
        if last < first:
            self.is_error = True
            return f"view_range [{window[0]}, {window[1]}] runs backwards: the first line must not come after the last."
        if first < 1:
            # The likely mistake is 0-based counting, so name the convention
            # rather than the file's length.
            self.is_error = True
            return f"view_range [{window[0]}, {window[1]}] starts before the file: lines are numbered from 1."
        if first > total:
            self.is_error = True
            return f"view_range [{window[0]}, {window[1]}] is outside '{path}', which has {_lines_word(total)}."
        last = min(last, total)
        if last - first + 1 > VIEW_CAP:
            # The same valve as a whole view: an explicit range must not be
            # the one route around it.
            last = first + VIEW_CAP - 1
            shown = numbered(lines[first - 1 : last], first, width)
            return (
                f"{path}, lines {first}-{last} of {total} — showing the first {VIEW_CAP}; "
                f"pass a narrower view_range for the rest.\n{shown}"
            )
        shown = numbered(lines[first - 1 : last], first, width)
        return f"{path}, lines {first}-{last} of {total}\n{shown}"


def file_rows(workspace: Workspace, entries: list[Entry] | None = None) -> str:
    """One line per file in the working directory: name, and length.

    Just the rows. The sentence above them differs by where they appear — a
    view answers "what is here", the level-start prompt says "this came with
    you" — but the rows are one rendering, so the model reads the same shape
    both times.

    A file that is not text is listed by size; directories never appear (the
    working directory is flat by contract, and the sweep makes that true).

    ``entries`` lets a caller that already has the listing pass it in.
    Fetching one is not cheap — :meth:`Workspace.entries` reads every text
    file through to count its lines — so a caller that needs both the count
    and the rows must not ask twice.
    """
    if entries is None:
        entries = workspace.entries()
    if not entries:
        return "  (no files yet)"
    name_width = max(len(entry.name) for entry in entries)
    measures = [
        f"{_lines_word(entry.lines)}" if entry.lines is not None else f"{entry.size} bytes" for entry in entries
    ]
    measure_width = max(len(measure) for measure in measures)
    return "\n".join(
        f"  {entry.name:<{name_width}}  {measure:>{measure_width}}" for entry, measure in zip(entries, measures)
    )


def directory_listing(workspace: Workspace) -> str:
    """What the working directory holds, as ``view_file('.')`` answers it."""
    entries = workspace.entries()
    return f"Working directory — {_files(len(entries))}\n{file_rows(workspace, entries)}"


@dataclass
class SearchFilesTool(WorkspaceTool):
    """Find text without printing whole files."""

    name: str = "search_files"
    description: str = (
        "Find text in your working directory without printing whole files. The pattern is a regular expression."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for"},
                "files": {"type": "string", "description": "Which files to search, as a glob (default: all files)"},
                "context_lines": {"type": "integer", "description": "Lines of context around each hit (default 2)"},
            },
            "required": ["pattern"],
        }
    )

    async def _answer(self, arguments: dict[str, Any]) -> str:
        """Search takes no path, so it does not use the base class's resolution."""
        workspace = self._require_workspace()
        pattern = str(arguments["pattern"])
        glob = str(arguments.get("files") or "*")
        # Clamped, not trusted: a context wider than the file turns every hit
        # into a whole-file view, and an unbounded cap turns a common word
        # into the corpus.
        requested = arguments.get("context_lines")
        context = min(max(0, 2 if requested is None else int(requested)), MAX_CONTEXT_LINES)
        cap = MATCH_CAP
        # One deadline for the whole call, not per line: a pattern that is
        # merely slow on every line of a large workspace stalls the game just
        # as effectively as one that is catastrophic on a single line.
        deadline = time.monotonic() + SEARCH_TIMEOUT_S
        try:
            expression = regex.compile(pattern)
        except regex.error as error:
            self.is_error = True
            return f"'{pattern}' is not a valid regular expression: {error}."

        # Names only: a search reads the files it keeps, and a line count
        # for the ones it discards is a read whose answer is never shown.
        candidates = [path.name for path in workspace.paths() if fnmatch.fnmatch(path.name, glob)]
        blocks: list[str] = []
        total = 0
        shown = 0
        searched = 0
        matched_files: list[str] = []
        for name in candidates:
            if shown >= cap:
                # The answer is already full. Reading on would buy a larger
                # total in the header at the price of reading the rest of
                # the workspace, so the count says what was searched.
                break
            try:
                if workspace.size(name) > MAX_READ_BYTES:
                    continue  # too large to read is not a file that was searched
                lines = workspace.read(name).splitlines()
            except (OSError, PathNotAllowed, NotText):
                continue  # a file no tool can read is not a file that was searched
            searched += 1
            try:
                hits = [
                    index for index, line in enumerate(lines) if expression.search(line, timeout=_remaining(deadline))
                ]
            except TimeoutError:
                self.is_error = True
                return (
                    f"'{pattern}' took longer than {SEARCH_TIMEOUT_S:.0f}s to search and was stopped, so "
                    "no results are available. A pattern with nested repetition such as `(a+)+` can take "
                    "time that grows explosively with line length; rewrite it more specifically, or search "
                    "for a plain substring."
                )
            if not hits:
                continue
            matched_files.append(name)
            total += len(hits)
            room = cap - shown
            if room <= 0:
                continue
            taken = hits[:room]
            shown += len(taken)
            blocks.append(f"{name}\n" + _grouped(lines, taken, context))

        if not total:
            return f"No matches for '{pattern}' in {_files(searched)} ({glob})."
        where = matched_files[0] if len(matched_files) == 1 else _files(len(matched_files))
        header = f"{_matches(total)} for '{pattern}' in {where} ({_files(searched)} searched)."
        if total > shown:
            header = (
                f"{_matches(total)} for '{pattern}' in {where} — showing the first {shown}. "
                "Narrow the pattern to see the rest."
            )
        return header + "\n\n" + "\n\n".join(blocks)


def _too_large(
    workspace: Workspace, path: str, remedy: str = "Read it from Python instead, a piece at a time."
) -> str | None:
    """Why ``path`` will not be read, or None; ``remedy`` closes the refusal
    with what the model can do instead.

    Checked before the read rather than after: the point is to not spend
    the memory, so a refusal computed from the finished string would be
    the failure it is meant to prevent.
    """
    try:
        size = workspace.size(path)
    except OSError:
        return None  # let the read report it, in its own words
    if size <= MAX_READ_BYTES:
        return None
    return f"'{path}' is {size} bytes, larger than the {MAX_READ_BYTES} the file tools will read. {remedy}"


def _remaining(deadline: float) -> float:
    """Seconds left before the search deadline, never zero.

    ``regex`` reads a non-positive timeout as no deadline at all, so an
    exhausted budget has to arrive as a tiny positive number for the next
    match to raise rather than run unbounded.
    """
    return max(1e-6, deadline - time.monotonic())


def _lines_word(count: int) -> str:
    """``3 lines`` / ``1 line``."""
    return f"{count} line{'s' if count != 1 else ''}"


def _files(count: int) -> str:
    """``3 files`` / ``1 file``."""
    return f"{count} file{'s' if count != 1 else ''}"


def _matches(count: int) -> str:
    """``3 matches`` / ``1 match`` — of lines, as grep counts them."""
    return f"{count} match{'es' if count != 1 else ''}"


def _grouped(lines: list[str], hits: list[int], context: int) -> str:
    """Matching lines with context, overlapping windows merged.

    Follows the convention every search tool uses, because that is the one the
    model has read most: a matching line is marked ``:`` and a context line
    ``-``, and runs that do not touch are separated by ``--``. Merging is not
    cosmetic — without it a file smaller than twice the context is printed once
    per hit, so two matches render the whole file twice and the count above
    reads as if it were wrong.
    """
    width = _width(len(lines))
    marked = set(hits)
    spans: list[list[int]] = []
    for hit in hits:
        low, high = max(0, hit - context), min(len(lines) - 1, hit + context)
        if spans and low <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], high)
        else:
            spans.append([low, high])
    groups = [
        "\n".join(f"{n + 1:>{width}}{':' if n in marked else '-'}{clipped(lines[n])}" for n in range(low, high + 1))
        for low, high in spans
    ]
    return "\n--\n".join(groups)


@dataclass
class StrReplaceTool(WorkspaceTool):
    """Replace one exact, unique substring."""

    name: str = "str_replace"
    description: str = (
        "Replace an exact substring in a file in your working directory. `old_string` "
        "must match the file exactly, so strip the line number and tab that a view "
        "puts in front of each line. It must appear exactly once unless you pass "
        "`replace_all`. You are shown where the change landed."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace, matched verbatim including indentation",
                },
                "new_string": {"type": "string", "description": "Text to put in its place"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
            },
            "required": ["path", "old_string", "new_string"],
        }
    )

    async def _act(self, path: str, arguments: dict[str, Any]) -> str:
        """Apply one replacement, and show what the file now says."""
        workspace = self._require_workspace()
        old_string = str(arguments["old_string"])
        new_string = self._repaired_text(arguments["new_string"])
        oversized = _too_large(workspace, path, "Edit it from Python instead.")
        if oversized is not None:
            self.is_error = True
            return oversized
        try:
            before = workspace.read(path)
        except FileNotFoundError:
            self.is_error = True
            return f"File '{path}' does not exist."
        except NotText as error:
            self.is_error = True
            return not_text_message(error)
        except OSError as error:
            self.is_error = True
            return f"Error reading '{path}': {_reason(error)}."

        outcome = replace_in(before, path, old_string, new_string, bool(arguments.get("replace_all")))
        if outcome.refusal is not None:
            self.is_error = True
            return outcome.refusal
        assert outcome.after is not None
        try:
            workspace.apply(path, before, outcome.after)
        except OSError as error:
            self.is_error = True
            return f"Error writing '{path}': {_reason(error)}."
        return outcome.report + parse_note(path, outcome.after, undo_edit=self.undo_edit_offered)


@dataclass(frozen=True)
class Replacement:
    """What one replacement would do: the new text and what to say about it,
    or the refusal that stopped it."""

    after: str | None
    report: str
    refusal: str | None = None


def replace_in(before: str, path: str, old_string: str, new_string: str, replace_all: bool) -> Replacement:
    """Work out one replacement against ``before``, without writing anything.

    Separate from the tool because deciding and writing are different jobs:
    every refusal this can answer with is a pure function of the file and the
    arguments, so the rules are readable and testable without a workspace.
    """
    if not old_string:
        # `"" in x` is everywhere at once: counting it reads the file length,
        # and replacing it would interleave new_string between every
        # character. Refused outright rather than diagnosed.
        return Replacement(
            None,
            "",
            "old_string is empty — provide the exact text to replace, or use "
            "create_file to rewrite the whole file.\n\nNothing was changed.",
        )
    count = before.count(old_string)
    if count == 0:
        miss = near_miss(before, old_string)
        if miss:
            return Replacement(None, "", f"old_string not found in '{path}'.{miss}\n\nNothing was changed.")
        return Replacement(
            None,
            "",
            f"old_string not found in '{path}', and no similar text was found. The file may have changed "
            f"since you last saw it — view_file('{path}') to see it as it is now.\n\nNothing was changed.",
        )
    if count > 1 and not replace_all:
        at = _some_lines(_lines_of(before, old_string))
        return Replacement(
            None,
            "",
            f"old_string appears {count} times in '{path}', at lines {at}; it must be unique. "
            f"Include surrounding lines to pick one, or pass replace_all=true to change all {count}."
            "\n\nNothing was changed.",
        )
    if replace_all and count > 1:
        # Locations in the file as it now is: each replacement shifts what
        # follows by the length difference, so the numbers are carried across
        # rather than read off the old text.
        drift = len(new_string) - len(old_string)
        after_all = before.replace(old_string, new_string)
        at_lines = [
            after_all.count("\n", 0, position + index * drift) + 1
            for index, position in enumerate(_positions_of(before, old_string))
        ]
        return Replacement(
            after_all,
            f"Edited {path}: {count} occurrences of old_string replaced, at lines {_some_lines(at_lines)}.",
        )
    start = before.index(old_string)
    after = before[:start] + new_string + before[start + len(old_string) :]
    return Replacement(after, f"Edited {path} — " + framed(after, start, start + len(new_string)))


@dataclass
class CreateFileTool(WorkspaceTool):
    """Create a file, or replace one entirely."""

    name: str = "create_file"
    description: str = "Create a file in your working directory, or replace one entirely."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to create"},
                "content": {"type": "string", "description": "What the file should contain"},
            },
            "required": ["path", "content"],
        }
    )

    async def _act(self, path: str, arguments: dict[str, Any]) -> str:
        """Write the whole file."""
        workspace = self._require_workspace()
        stem, suffix = os.path.splitext(path)
        if suffix == ".py" and module_names.taken(stem, workspace.directory):
            self.is_error = True
            return (
                f"'{path}' was not created: `{stem}` is already a module Python can import (the standard "
                f"library's or an installed package's), so `import {stem}` would never reach your file. "
                "Choose another name."
            )
        content = self._repaired_text(arguments["content"])
        # Replacing a file reads it first, for undo: one too large to read is
        # refused, as one that is not text is below.
        oversized = _too_large(workspace, path, "Replace it from Python instead.")
        if oversized is not None:
            self.is_error = True
            return oversized
        try:
            before = workspace.read(path)
        except NotText as error:
            # No `before` means no undo, and overwriting something unreadable
            # is not a thing to do silently.
            self.is_error = True
            return not_text_message(error)
        except (FileNotFoundError, IsADirectoryError, PathNotAllowed, OSError):
            before = None
        try:
            workspace.apply(path, before, content)
        except OSError as error:
            self.is_error = True
            return f"Error writing '{path}': {_reason(error)}."
        lines = len(content.splitlines())
        if before is None:
            report = f"Created {path} (empty)." if not lines else f"Created {path} ({_lines_word(lines)})."
        else:
            report = f"Replaced {path} — was {_lines_word(len(before.splitlines()))}, now {lines}."
        return report + parse_note(path, content, undo_edit=self.undo_edit_offered)


@dataclass
class UndoEditTool(WorkspaceTool):
    """Take back the most recent change, and only that one."""

    name: str = "undo_edit"
    description: str = (
        "Undo the most recent change you made to a file. One step only: calling it again, before you "
        "make another edit, finds nothing to take back."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to undo; omit for the most recent change"},
            },
            "required": [],
        }
    )

    async def _answer(self, arguments: dict[str, Any]) -> str:
        """Revert one change, or say why it could not be."""
        workspace = self._require_workspace()
        path = arguments.get("path")
        change = workspace.last_change(str(path) if path else None)
        if change is None:
            where = f" to '{path}'" if path else ""
            self.is_error = True
            return f"Nothing to undo — no file changes{where} to take back."
        if workspace.changed_outside(change):
            self.is_error = True
            return (
                f"{change.path} has changed since that edit — "
                "something other than the file tools wrote it. Nothing was undone. "
                f"Use view_file('{change.path}') to see it as it is now."
            )
        try:
            workspace.revert(change)
        except OSError as error:
            self.is_error = True
            return f"Error writing: {_reason(error)}."

        if change.before is None:
            return f"Undid the creation of {change.path}; the file is gone."
        first, last = _changed_span(change.after, change.before)
        return f"Undid the last change to {change.path}. " + region(change.before, first, last)


def _changed_span(before: str, after: str) -> tuple[int, int]:
    """The 1-based line range where ``after`` differs from ``before``."""
    old = before.splitlines()
    new = after.splitlines()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    changed = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changed:
        return 1, len(new)
    return min(op[3] for op in changed) + 1, max(max(op[4] for op in changed), 1)


#: The suite in the order the model is shown it, each paired with the agent
#: field that gates it (``None`` for the three that are not optional).
#:
#: Order is prompt text, and it is a claim about what to reach for first:
#: look, search, then change. One table rather than a list per caller, because
#: the tools offered and the tools NAMED in the system prompt have to be the
#: same set — a prompt that advertises a tool the model was not given is a
#: prompt that lies, and two hand-kept lists drift the first time one is
#: edited.
_SUITE: tuple[tuple[type[WorkspaceTool], str | None], ...] = (
    (ViewFileTool, None),
    (SearchFilesTool, "search_files"),
    (StrReplaceTool, None),
    (CreateFileTool, None),
    (UndoEditTool, "undo_edit"),
)


def _offered(*, search_files: bool, undo_edit: bool) -> list[type[WorkspaceTool]]:
    """The classes an agent with these gates offers, in suite order."""
    gates = {"search_files": search_files, "undo_edit": undo_edit}
    return [cls for cls, gate in _SUITE if gate is None or gates[gate]]


def file_tools(workspace: Workspace, *, search_files: bool = True, undo_edit: bool = True) -> list[ullm.tool.Tool]:
    """The suite, in the order the model is shown it.

    The two gated tools default to offered, so a caller that does not mention
    them gets the whole suite.
    """
    return [
        cls(workspace=workspace, undo_edit_offered=undo_edit)
        for cls in _offered(search_files=search_files, undo_edit=undo_edit)
    ]


def file_tool_names(*, search_files: bool = True, undo_edit: bool = True) -> tuple[str, ...]:
    """The names of the tools :func:`file_tools` would build, same order.

    Needs no workspace, because a prompt has to name the suite before a kernel
    exists to hold one. Read off the classes rather than written out, so the
    prompt cannot name a tool the builder does not build.
    """
    return tuple(cls.name for cls in _offered(search_files=search_files, undo_edit=undo_edit))


__all__ = [
    "Change",
    "CreateFileTool",
    "Entry",
    "SearchFilesTool",
    "StrReplaceTool",
    "UndoEditTool",
    "ViewFileTool",
    "WorkspaceTool",
    "directory_listing",
    "file_rows",
    "file_tool_names",
    "file_tools",
]
