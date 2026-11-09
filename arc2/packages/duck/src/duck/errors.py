"""Rendered kernel output as the model should read it.

What the model should read is its own code and its own call chain, and
nothing it cannot act on. Three things are taken out, at
the one seam where kernel output becomes a tool result:

* **Our frames.** ``check``, ``show`` and ``ascii_grid`` are compiled under
  :data:`HARNESS_SETUP_FILE`, and the sandbox's patches and the working
  directory's import machinery are files of this package, so their frames
  are ours. Every frame *above* the model's first is collapsed into one
  boundary line; one *below* it, a call the model made, keeps only the public
  name it called, never our file, our line or the source CPython echoes from
  it. A library the model called stays: which of its own calls went wrong is
  the most useful thing a traceback says.
* **Machine-specific paths.** A traceback out of the kernel carries absolute
  paths — a temp directory under a SLURM job id, a checkout name, a Kaggle
  mount — none of which the model can act on and all of which cost tokens.
  They are shortened wherever they appear, in messages as well as in
  ``File "..."`` lines.
* **Object addresses.** Per-process, so an otherwise identical run's output
  would differ.
"""

from __future__ import annotations

import os
import re

#: The pseudo-file the kernel's own functions are compiled under: no readable
#: source, which is exactly true, since the model cannot open the harness.
HARNESS_SETUP_FILE = "<harness-setup>"

#: Files whose frames belong to the model: what it typed. Its modules are
#: files in its working directory, recognised by place (``model_dir``).
MODEL_FILES = ("<repl>",)

#: Files that are ours by name, wherever they appear. ``python_worker.py`` is
#: ullm's: a timeout that lands inside a host call shows the kernel's own
#: frames below ``check`` or ``show``. Anything under this package is ours too.
HARNESS_FILES = (HARNESS_SETUP_FILE, "python_worker.py")

#: This package's own directory: a frame from a file under it is ours,
#: whatever the file is called.
_DUCK_DIR = os.path.dirname(os.path.abspath(__file__))

#: The pseudo-file the boundary line names, after Python's own convention for
#: a frame with no readable source (``<string>``, ``<stdin>``).
HARNESS_FILE = "<harness>"

_FRAME_START = re.compile(r'^\s*File "([^"]+)", line \d+, in ')
#: A path inside a ``File "..."`` traceback header. Anchored to one line: an
#: unanchored ``[^"]+`` runs through newlines, so where the output cap has
#: elided the middle of a traceback it pairs the cut head's ``File "`` with
#: the first quote in the tail and stitches two cut ends into one fabricated
#: frame.
_TRACEBACK_PATH = re.compile(r'File "([^"\n]+)"')
#: A path in an exception message rather than a frame, e.g. ``cannot import
#: name 'x' from 'y' (/tmp/ullm_python_ab12/...)``, or quoted, as a module's
#: default repr has it.
_MESSAGE_PATH = re.compile(r"(?<![\w])(/[\w./+-]+\.py)\b")
#: One of our files named in a message rather than a frame. CPython writes the
#: defining file of a module it could not import a name from, and for
#: ``__main__`` that is the kernel worker's: ``cannot import name 'x' from
#: '__main__' (python_worker.py)``. Already a basename, so it is replaced with
#: what our frames say.
_MESSAGE_HARNESS_FILE = re.compile(
    r"\((" + "|".join(re.escape(name) for name in HARNESS_FILES if name.endswith(".py")) + r")\)"
)
#: The tail of a default object repr; the address is per-process.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")
#: The two halves of a ``File "..."`` header an output cap cut through: a head
#: whose quote never closes, a path fragment no complete-header regex can see,
#: and possibly a tail that is the back half of another header.
_DANGLING_FRAME = re.compile(r'^(\s*File ")([^"\n]*)$', re.MULTILINE)
_DANGLING_TAIL = re.compile(r'^[^"\n]*/[^"\n]*", line (\d+), in ([\w.<>]+)$', re.MULTILINE)

#: The sentences CPython puts between chained tracebacks. Each segment is a
#: traceback in its own right, so each is normalised on its own — a harness
#: frame in the second one sits *after* the first one's model frames, and a
#: single pass over the whole text would keep it.
_CHAIN_JOINS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)


def shorten_path(path: str) -> str:
    """One file as the model should see it: the basename, keeping the parent
    for ``__init__.py`` so ``numpy/__init__.py`` and ``pandas/__init__.py``
    do not collide in a multi-frame stack. Pseudo-files (``<repl>``,
    ``<frozen importlib._bootstrap>``) pass through untouched."""
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
    """Whether a ``File "..."`` header names something the model wrote: a
    cell, or any file in its working directory."""
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
    return raw is not None and os.path.isabs(raw) and os.path.abspath(raw).startswith(_DUCK_DIR + os.sep)


def _public_callable(header: str) -> str | None:
    """The function name in a frame header, if it is one the model may call.

    A private name is ours by construction, so a frame carrying one is dropped
    rather than summarised — the model can neither call it nor read it. A
    public one is on the surface it programs against, so naming it tells it
    which of its own calls went wrong.
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
    last):`` header, the exception line — is prose that passes through
    untouched.
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
    """Paths shortened and addresses masked, anywhere in rendered text: on
    its own for text that is not a traceback, and as the second half of
    :func:`for_model`."""
    text = _TRACEBACK_PATH.sub(lambda m: f'File "{shorten_path(m.group(1))}"', text)
    text = _MESSAGE_PATH.sub(lambda m: shorten_path(m.group(1)), text)
    text = _MESSAGE_HARNESS_FILE.sub(f"({HARNESS_FILE})", text)
    # After the complete headers, so only the cut ones are left to match. A
    # fragment is half a path at best, so it is named for what it is rather
    # than shortened as if it were whole.
    text = _DANGLING_FRAME.sub(r"\1<truncated>", text)
    text = _DANGLING_TAIL.sub(r'  File "<truncated>", line \1, in \2', text)
    return _OBJECT_ADDRESS.sub(">", text)


def _normalise_segment(text: str, *, where: str, model_dir: str | None) -> str:
    """One traceback: our frames above the model replaced by the boundary,
    and ours below it reduced to the public name the model called."""
    blocks = _blocks(text.splitlines())
    first_model = next(
        (i for i, (header, _) in enumerate(blocks) if header is not None and _is_model_frame(header, model_dir)),
        None,
    )
    # With no model frame at all, direction cannot say, so the files do. With
    # one of ours among them, every frame is above the model: the call into
    # it failed before entry, and the exception line carries the diagnosis.
    # With none of ours, the segment is the first half of a chain raised
    # inside a library the model called -- ``inspect`` re-raising an
    # ``IndexError`` as ``OSError`` -- and is kept as the library's.
    has_harness_frame = any(header is not None and _is_harness_frame(header) for header, _ in blocks)
    if first_model is not None:
        above = first_model
    else:
        above = len(blocks) if has_harness_frame else 0
    dropped = [i for i, (header, _) in enumerate(blocks) if header is not None and i < above]
    # Only when something was actually replaced: a cell that raises in the
    # model's own code has nothing above it, and announcing a boundary there
    # would describe a call that never happened.
    boundary = [f'  File "{HARNESS_FILE}", in {where}'] if dropped else []
    insert_at = first_model if first_model is not None else (dropped[0] if dropped else None)

    out: list[str] = []
    for index, (header, block) in enumerate(blocks):
        if index == insert_at:
            out.extend(boundary)
        if index in dropped:
            continue
        if header is not None and _is_harness_frame(header):
            called = _public_callable(header)
            if called is not None:
                out.append(f'  File "{HARNESS_FILE}", in {called}')
            continue
        out.extend(block)
    return "\n".join(out)


def for_model(text: str, *, where: str = "harness", model_dir: str | None = None) -> str:
    """Kernel output as the model should read it: each traceback in it
    normalised (:func:`_normalise_segment`), then :func:`clean_text`.

    ``where`` names, in the harness's own vocabulary, what it was doing when
    it called into the model's code; it becomes the boundary frame.
    ``model_dir`` is the model's working directory: a frame from any file
    there is the model's own.
    """
    segments: list[list[str]] = [[]]
    joins: list[str] = []
    for line in text.splitlines():
        if line.strip() in _CHAIN_JOINS:
            joins.append(line)
            segments.append([])
            continue
        segments[-1].append(line)
    directory = os.path.realpath(model_dir) if model_dir is not None else None
    cleaned = [_normalise_segment("\n".join(segment), where=where, model_dir=directory) for segment in segments]
    out = cleaned[0]
    for join, segment in zip(joins, cleaned[1:], strict=True):
        out += "\n" + join + "\n" + segment
    return clean_text(out)


__all__ = ["HARNESS_FILE", "HARNESS_SETUP_FILE", "clean_text", "for_model", "shorten_path"]
