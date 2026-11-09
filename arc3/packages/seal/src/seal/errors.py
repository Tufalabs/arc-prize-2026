"""What a Python error looks like to the model.

Errors reach the model as the output of a REPL cell it ran, or in a
``TestResult`` rendered while replaying history (``harness.test``), and
every one comes through here.

What the model should read is its own code and its own call chain, one line
saying what the harness was doing when it called in, and nothing else. Three
things are therefore taken out, each because the model cannot act on it:

* **Our frames.** Every frame *above* the model is collapsed into a single
  synthetic boundary line. The rule is direction, not filename: code that
  *called* the model is ours and says nothing it can use, while code the
  model *called* — numpy, this package's own public methods — is as
  informative as any library's and stays. Without this, the
  name of whichever harness function happens to hold the ``try``, its line
  number and its source text are all prompt material, which freezes our
  decomposition; and the model reads a chain that starts inside a file it is
  forbidden to open.
* **Machine-specific paths.** The model's own files live under a temp
  directory drawn fresh per kernel, and ours under wherever this checkout
  happens to be. A prompt naming either reproduces on no other machine — and
  the temp directory's ``mkdtemp`` segment not even on the same one. Paths
  are shortened wherever they appear, inside exception *messages* as well as
  in ``File "..."`` lines.
* **Object addresses.** ``<function f at 0x7f…>`` is per-process, so a
  recording quoting one reproduces nowhere.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import os
import re
from typing import Generator

#: Files whose frames belong to the model: what it typed, and the world model
#: it wrote. Its other modules are only ever reached from a cell, so their
#: frames sit below a ``<repl>`` frame and are kept by direction;
#: ``world_model.py`` is also run by ``test_world_model()`` and the move gate,
#: from inside the harness, where nothing of the model's sits above it.
MODEL_FILES = ("world_model.py", "<repl>")

#: Files that are ours, wherever they appear. Direction alone cannot find
#: these: the housekeeping we prepend to every cell runs *inside* ``<repl>``
#: and calls down into the harness, so our frames turn up below a frame that
#: looks like the model's. They are named rather than inferred.
#: ``python_worker.py`` is ullm's: a timeout that lands inside a host call
#: shows the kernel's own frames below ``submit_action``.
HARNESS_FILES = (
    "test.py",
    "game.py",
    "_repl_state.py",
    "segmentation.py",
    "errors.py",
    "python_worker.py",
    "<harness-setup>",
)

#: This package's own directory: a frame from a file under it is ours,
#: whatever the file is called — the search's model wrapper and planner
#: included, which carry no name in :data:`HARNESS_FILES`.
_SEAL_DIR = os.path.dirname(os.path.abspath(__file__))

#: The model's working directory for tracebacks rendered in this context,
#: when a caller does not pass one (:func:`model_directory`).
_MODEL_DIR: contextvars.ContextVar[str | None] = contextvars.ContextVar("seal_model_dir", default=None)

#: The pseudo-file the boundary line names. Follows Python's own convention
#: for a frame with no readable source (``<string>``, ``<stdin>``), which is
#: exactly true here: the model cannot open the harness.
HARNESS_FILE = "<harness>"

_FRAME_START = re.compile(r'^\s*File "([^"]+)", line \d+, in ')
#: One line only: an unanchored ``[^"]+`` would run through newlines, and when
#: the output cap has elided the middle of a traceback that means pairing the
#: cut head's ``File "`` with the first quote anywhere in the tail — stitching
#: the two cut ends (elision marker included) into one fabricated frame.
_TRACEBACK_PATH = re.compile(r'File "([^"\n]+)"')
#: A path inside an exception message rather than a frame, e.g.
#: ``cannot import name 'X' from 'solver' (/tmp/ullm_python_ab12/…)``, or
#: quoted, as a module's default repr has it.
_MESSAGE_PATH = re.compile(r"(?<![\w])(/[\w./+-]+\.py)\b")
#: One of our files named in a message rather than a frame. CPython writes
#: the defining file of a module it could not import a name from, and for
#: ``__main__`` that is the kernel's own: ``cannot import name 'segment'
#: from '__main__' (python_worker.py)``. Shortening cannot help — it is
#: already a basename — so the name is replaced with what our frames say.
_MESSAGE_HARNESS_FILE = re.compile(
    r"\((" + "|".join(re.escape(name) for name in HARNESS_FILES if name.endswith(".py")) + r")\)"
)
#: The tail of a default object repr; the address is per-process.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")
#: The two halves of a ``File "..."`` header the output cap cut through. The
#: cap elides the *middle* of long text and runs before this module does, so a
#: cut through a traceback leaves a head whose quote never closes — a path
#: fragment no complete-header regex can see — and possibly a tail that is
#: the back half of a different header, its own path fragment in front.
_DANGLING_FRAME = re.compile(r'^(\s*File ")([^"\n]*)$', re.MULTILINE)
_DANGLING_TAIL = re.compile(r'^[^"\n]*/[^"\n]*", line (\d+), in ([\w.<>]+)$', re.MULTILINE)
#: The name CPython gives a cell that failed to compile. The model's other
#: frames say ``<repl>``, and a SyntaxError is as much its own code as any.
_COMPILE_FILE = re.compile(r'^(\s*File ")<unknown>(", line \d+)$', re.MULTILINE)
#: An attribute the model guessed at, and the class it guessed on.
_ATTRIBUTE_ERROR = re.compile(r"AttributeError: '([\w.]+)' object has no attribute '(\w+)'")


def shorten_path(path: str) -> str:
    """One file, as the model should see it: the basename, keeping the parent
    for ``__init__.py`` so ``numpy/__init__.py`` and ``torch/__init__.py`` do
    not collide in a multi-frame stack. Pseudo-files (``<repl>``, ``<frozen
    importlib._bootstrap>``) pass through untouched."""
    base = os.path.basename(path)
    if base != "__init__.py":
        return base
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{base}" if parent else base


def _frame_file(header: str) -> str | None:
    """The file a ``File "..."`` header names, shortened, or None."""
    match = _TRACEBACK_PATH.search(header)
    return None if match is None else shorten_path(match.group(1))


def _raw_path(header: str) -> str | None:
    """The path a ``File "..."`` header names, as written."""
    match = _TRACEBACK_PATH.search(header)
    return None if match is None else match.group(1)


def _is_model_frame(header: str, model_dir: str | None) -> bool:
    """Whether a ``File "..."`` header names something the model wrote: one
    of :data:`MODEL_FILES`, or any file in its working directory — a helper
    module ``world_model.py`` imports its ``do_move`` from, say, which the
    harness calls with no frame of the model's above it."""
    if _frame_file(header) in MODEL_FILES:
        return True
    raw = _raw_path(header)
    return (
        model_dir is not None
        and raw is not None
        and os.path.isabs(raw)
        and os.path.dirname(os.path.abspath(raw)) == model_dir
    )


def _is_harness_frame(header: str) -> bool:
    """Whether a ``File "..."`` header names one of ours: by name, or by
    lying under this package."""
    if _frame_file(header) in HARNESS_FILES:
        return True
    raw = _raw_path(header)
    return raw is not None and os.path.isabs(raw) and os.path.abspath(raw).startswith(_SEAL_DIR + os.sep)


@contextlib.contextmanager
def model_directory(directory: str) -> Generator[None, None, None]:
    """Render the tracebacks of this block with ``directory`` as the model's
    working directory, for callers that do not pass one to :func:`for_model`."""
    token = _MODEL_DIR.set(os.path.abspath(directory))
    try:
        yield
    finally:
        _MODEL_DIR.reset(token)


def _public_callable(header: str) -> str | None:
    """The function name in a frame header, if it is one the model may call.

    A private name is ours by construction, so a frame carrying one is
    dropped rather than summarised — the model can neither call it nor read
    it. A public one is on the surface it programs against, so naming it
    tells it which of its own calls went wrong.
    """
    match = re.search(r", line \d+, in ([\w.]+)$", header.rstrip())
    if match is None:
        return None
    name = match.group(1)
    return None if name.startswith("_") else name


def _blocks(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """The text split into frames and the prose between them.

    A frame is its ``File "..."`` header plus the indented source and caret
    lines under it; everything else — the ``Traceback (most recent call
    last):`` header, the exception line, the ``During handling…`` separator
    between chained exceptions — is prose that passes through untouched.
    """
    out: list[tuple[str | None, list[str]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _FRAME_START.match(line):
            block = [line]
            index += 1
            while index < len(lines) and lines[index].startswith("    "):
                block.append(lines[index])
                index += 1
            out.append((line, block))
            continue
        out.append((None, [line]))
        index += 1
    return out


def clean_text(text: str) -> str:
    """Paths shortened and addresses masked, anywhere in rendered text.

    Applied on its own to text that is not a traceback — an exception
    message, a warning — and as the second half of :func:`for_model`.
    """
    text = _TRACEBACK_PATH.sub(lambda m: f'File "{shorten_path(m.group(1))}"', text)
    text = _MESSAGE_PATH.sub(lambda m: shorten_path(m.group(1)), text)
    text = _MESSAGE_HARNESS_FILE.sub(f"({HARNESS_FILE})", text)
    # After the complete headers, so only the cut ones are left to match. A
    # fragment is half a path at best, so it is named for what it is rather
    # than shortened as if it were whole.
    text = _DANGLING_FRAME.sub(r"\1<truncated>", text)
    text = _DANGLING_TAIL.sub(r'  File "<truncated>", line \1, in \2', text)
    text = _COMPILE_FILE.sub(r"\1<repl>\2", text)
    return _OBJECT_ADDRESS.sub(">", text)


#: The sentences CPython puts between chained tracebacks. Each segment is a
#: traceback in its own right, so each is normalised on its own — a harness
#: frame in the second one sits *after* the first one's model frames, and a
#: single pass over the whole text would keep it.
_CHAIN_JOINS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)


def _normalise_segment(text: str, *, where: str, doing: str | None, model_dir: str | None) -> str:
    """One traceback: our frames above the model replaced by the boundary."""
    blocks = _blocks(text.splitlines())
    first_model = next(
        (i for i, (header, _) in enumerate(blocks) if header is not None and _is_model_frame(header, model_dir)),
        None,
    )

    # With no model frame at all, direction cannot say, so the files do. With
    # one of ours among them, every frame is above the model — the call into
    # it failed before entry: here, the housekeeping prepended to the cell
    # raised before the model's code ran. Keeping those frames would show it
    # our file, our line, our function name and the source text of our call,
    # which is precisely what the boundary exists to prevent. The exception
    # line survives and carries the whole diagnosis; when the fault really is
    # ours, the raw traceback is in the run's own spans. With none of ours,
    # the segment is the first half of a chain raised inside a library the
    # model called — ``inspect`` re-raising an ``IndexError`` as ``OSError``
    # — and it is kept as the library's, which is what it is.
    has_harness_frame = any(header is not None and _is_harness_frame(header) for header, _ in blocks)
    if first_model is not None:
        above = first_model
    else:
        above = len(blocks) if has_harness_frame else 0
    dropped = [i for i, (header, _) in enumerate(blocks) if header is not None and i < above]
    # Only when something was actually replaced. A REPL cell that raises in
    # the model's own code has no frame above it, and announcing a boundary
    # there would describe a call that never happened.
    boundary: list[str] = []
    if dropped:
        boundary.append(f'  File "{HARNESS_FILE}", in {where}')
        if doing:
            boundary.append(f"    {doing}")
    # Where the boundary goes: immediately before the model's own frame, or —
    # when there is none — in place of the first frame collapsed.
    insert_at = first_model if first_model is not None else (dropped[0] if dropped else None)

    out: list[str] = []
    for index, (header, block) in enumerate(blocks):
        if index == insert_at:
            out.extend(boundary)
        if index in dropped:
            continue  # a frame above the model: ours, and it says nothing usable
        if header is not None and _is_harness_frame(header):
            # Ours wherever it sits. Below the model it is a call the model
            # made, so the public name it called is worth keeping — but never
            # our file, our line, or the source text CPython echoes from it.
            called = _public_callable(header)
            if called is not None:
                out.append(f'  File "{HARNESS_FILE}", in {called}')
            continue
        out.extend(block)
    return "\n".join(out)


def for_model(
    text: str,
    *,
    where: str = "harness",
    doing: str | None = None,
    model_dir: str | None = None,
    world_model: bool | None = None,
) -> str:
    """A rendered traceback as the model should read it.

    ``where`` and ``doing`` describe, in the harness's own vocabulary, what it
    was doing when it called into the model's code. Together they become the
    boundary frame, which is worth more to the model than the source text of
    ours that it replaces. ``model_dir`` is the model's working directory: a
    frame from any file there is the model's own. Left out, it is the one
    :func:`model_directory` set around the call, if any. ``world_model`` says
    whether the game has one, for the surface the hint documents; left out,
    it is this process's setting.
    """
    lines = text.splitlines()
    segments: list[list[str]] = [[]]
    joins: list[str] = []
    for line in lines:
        if line.strip() in _CHAIN_JOINS:
            joins.append(line)
            segments.append([])
            continue
        segments[-1].append(line)

    directory = os.path.abspath(model_dir) if model_dir is not None else _MODEL_DIR.get()
    cleaned = [
        _normalise_segment("\n".join(segment), where=where, doing=doing, model_dir=directory) for segment in segments
    ]
    out = cleaned[0]
    for join, segment in zip(joins, cleaned[1:]):
        out += "\n" + join + "\n" + segment
    out = clean_text(out)
    hint = surface_hint(out, world_model=world_model)
    return out if hint is None else f"{out}\n\n{hint}"


def surface_hint(text: str, *, world_model: bool | None = None) -> str | None:
    """What the model should have reached for, when it guessed at *our* API.

    Renders the harness's own ``help()`` for the class it guessed on, which
    is the same thing it would read had it asked — so the hint cannot drift
    from the documentation, and it carries what introspection cannot: the
    properties ``Turn`` and ``History`` set in ``__init__``, and none of the
    ``dict`` mutators that ``dir()`` reports for ``DiffSparse``.

    Only for classes on the surface we hand it. The model's own mistakes and
    a library's complaints are its business, and answering everything would
    be noise.
    """
    match = _ATTRIBUTE_ERROR.search(text)
    if match is None:
        return None
    name, attribute = match.group(1).rsplit(".", 1)[-1], match.group(2)
    # Idempotent: the tool applies :func:`for_model` twice to one cell's
    # output (before the cap cuts, and once more on the way out), and a
    # second pass over a hinted text must not hint it again.
    if f"Hint: {name} has no {attribute!r}" in text:
        return None

    import seal.harness.game as surface

    if name not in surface.__all__:
        return None
    target = getattr(surface, name, None)
    if not isinstance(target, type):
        return None

    # ``help`` prints rather than returns, so its output is captured. This
    # runs in the tool wrapper rather than the kernel, where nothing else
    # prints — a returning ``help_text`` would be tidier and is worth doing
    # if this module is ever refactored.
    buffer = io.StringIO()
    previous = None if world_model is None else surface._configure_world_model(enabled=world_model)  # pyright: ignore[reportPrivateUsage]
    try:
        with contextlib.redirect_stdout(buffer):
            surface.help(target)
    finally:
        if previous is not None:
            surface._configure_world_model(enabled=previous)  # pyright: ignore[reportPrivateUsage]
    rendered = buffer.getvalue().strip()
    if not rendered:
        return None
    return f"Hint: {name} has no {attribute!r}. Its documented surface:\n\n{rendered}"
