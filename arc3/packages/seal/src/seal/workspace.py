"""The model's working directory: what may be touched, and what it holds.

One object owns the directory the model's files live in, so every tool asks
the same question of the same rule instead of carrying its own copy. It
resolves paths, reads and writes, and lists what is there.

**Resolved paths do not leave.** Every method takes the path as the model
wrote it and refuses rather than resolving one, so a tool has nothing
host-specific to put in a message even by accident. The hazard this closes
structurally: an ``OSError``'s filename IS the resolved path, so any handler
that interpolates the exception has already leaked it.

The directory belongs to the python tool's kernel and is read off it at call
time rather than copied: an unpickled or deepcopied tool arrives with a fresh
one, and a workspace holding the old string would resolve into a directory
nobody is using.

The same directory is the kernel's working directory, so a file written here
is a module the model's snippets can import — the second half of what makes
these tools worth having.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seal.repl import GameREPL


class NotText(Exception):
    """A file that is not text, so no tool can show or edit it.

    Reading it as text would put replacement characters into a prompt, or
    raise a ``UnicodeDecodeError`` — which is not an ``OSError``, and so
    escapes every tool's error handling entirely.
    """

    def __init__(self, path: str, size: int) -> None:
        super().__init__(path)
        self.path = path
        self.size = size


class PathNotAllowed(Exception):
    """A path the workspace does not permit.

    Carries the path **as the model wrote it**, which is the only form any
    message about it may use.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path


@dataclass(frozen=True)
class Change:
    """One undoable act on one file. ``before`` is None when the file did not
    exist, so undoing a creation removes it rather than leaving an empty one."""

    path: str
    before: str | None
    after: str


@dataclass(frozen=True)
class Entry:
    """One file in the working directory. ``lines`` is None when it is not
    text, which is what the listing shows instead of a line count."""

    name: str
    size: int
    lines: int | None


@dataclass
class Workspace:
    """The directory the model's files live in, and the rule for reaching them.

    Containment is the rule: a path must resolve inside the directory. What it
    is called is not — ``allowed_suffixes`` defaults to None, meaning any name,
    because a snippet can already write whatever it likes here (numpy's
    ``save`` and ``savetxt`` reach the real filesystem), and a workspace that
    LISTED those files while refusing to open them would advertise a directory
    the model does not have. Narrowing it is a configuration a caller can ask
    for; the suffix rule is then a claim about what the tools are FOR, never
    about what they are allowed to reach.

    ``always_allowed`` names files permitted whatever their suffix — inert
    while the suffix rule is off, and empty regardless, since the seal's
    harness writes no files of its own.

    ``changes`` is what undo walks back. It is cleared when the model executes
    an action against the game: the board has moved, so the file state that
    produced it is history rather than something to take back.
    """

    repl: "GameREPL"
    always_allowed: tuple[str, ...] = ()
    allowed_suffixes: tuple[str, ...] | None = None
    changes: list[Change] = field(default_factory=list[Change])

    @property
    def directory(self) -> str:
        """Where the model's files are, right now.

        Read off the tool on every call — see the module docstring for why a
        stored copy goes stale.
        """
        tmpdir = self.repl._tmpdir  # pyright: ignore[reportPrivateUsage] — the tool owns it, this only reads it
        assert tmpdir is not None, "the python tool's working directory has been cleared"
        return tmpdir

    def resolve(self, path: str) -> str:
        """``path`` as a real location, or raise :class:`PathNotAllowed`.

        Kept private to this module by convention: callers pass model-facing
        paths and let the workspace do the reading and writing.
        """
        directory = self.directory
        # An always-allowed name skips the flat rule and the suffix rule, but
        # not containment: it is a name we promised to accept, not a promise
        # that whatever now sits under it is inside the workspace. A snippet
        # can leave a symlink there, and following one would put every read
        # and write the tools do outside the directory they are scoped to.
        named = path in self.always_allowed
        if not named and (os.sep in path or (os.altsep and os.altsep in path)):
            # The working directory is flat by contract: the listing and the
            # search are both one level deep, and importing a nested file
            # needs a package the model did not write, so a file down there
            # would be silently second-class.
            raise PathNotAllowed(path)
        try:
            base = os.path.realpath(directory)
            real = os.path.realpath(os.path.join(directory, path))
        except ValueError:
            # A null byte, or anything else the OS will not accept as a name.
            # ValueError is not an OSError, so without this it leaves the tool
            # — and a raise out of a tool ends the game rather than reaching
            # the model.
            raise PathNotAllowed(path) from None
        within = real.startswith(base + os.sep)
        allowed_suffix = named or not self.allowed_suffixes or real.endswith(self.allowed_suffixes)
        if within and allowed_suffix:
            return real
        raise PathNotAllowed(path)

    def denial(self, path: str) -> str:
        """Why ``path`` was refused, and what the model may use instead."""
        if os.sep in path or (os.altsep and os.altsep in path):
            return (
                f"'{path}' is not allowed: your working directory is flat — "
                "keep every file directly in it, with no subdirectories."
            )
        if self.allowed_suffixes is None:
            return (
                f"File '{path}' is not allowed. You may read and write any file in your "
                "working directory, but not outside it."
            )
        kinds = ", ".join(self.allowed_suffixes)
        allowed = f" Always allowed: {list(self.always_allowed)}." if self.always_allowed else ""
        return f"File '{path}' is not allowed. You may read and write {kinds} files in your working directory.{allowed}"

    def read(self, path: str) -> str:
        """The file's text.

        Raises ``PathNotAllowed``, ``FileNotFoundError``, ``OSError`` or
        :class:`NotText`; never returns a location.
        """
        real = self.resolve(path)
        try:
            with open(real) as handle:
                return handle.read()
        except UnicodeDecodeError:
            raise NotText(path, os.path.getsize(real)) from None

    def write(self, path: str, content: str) -> None:
        """Replace the file's text. Raises as :meth:`read` does."""
        with open(self.resolve(path), "w") as handle:
            handle.write(content)

    def size(self, path: str) -> int:
        """The file's size in bytes, without opening it.

        What a caller checks before deciding to read: the caps on a rendered
        answer bound what the model is SHOWN, and a file large enough to
        exhaust the process on the way to a small answer never reaches them.
        """
        return os.path.getsize(self.resolve(path))

    def exists(self, path: str) -> bool:
        """Whether the file is there. Raises ``PathNotAllowed`` if it could
        not be looked at in the first place."""
        return os.path.isfile(self.resolve(path))

    def apply(self, path: str, before: str | None, after: str) -> None:
        """Write one file, recording what was there so undo can put it back."""
        self.write(path, after)
        self.changes.append(Change(path=path, before=before, after=after))

    def last_change(self, path: str | None = None) -> Change | None:
        """The most recent change, to ``path`` if one is named."""
        for change in reversed(self.changes):
            if path is None or change.path == path:
                return change
        return None

    def changed_outside(self, change: Change) -> bool:
        """Whether ``change``'s file has moved since it was made.

        A snippet can write the model's files itself, and undoing onto work
        made that way would discard it silently — which is worse than
        refusing.
        """
        try:
            return self.read(change.path) != change.after
        except (OSError, PathNotAllowed, NotText):
            return True

    def revert(self, change: Change) -> None:
        """Put ``change`` back, and end the undo.

        Undo reaches exactly one step. Reverting clears the whole record, so a
        second undo before a real edit finds nothing and says so, rather than
        walking further back through work the model is no longer discussing.
        One step is what the tool is for — taking back the edit just made —
        and a deeper stack invites a model to unwind a file to a state it
        cannot see and has not reasoned about.

        Clearing rather than removing this entry is also what keeps the record
        honest: ``Change`` is frozen and compares by value, so a model that
        edits A to B and back to A leaves two equal entries, and removing "the
        one just reverted" by equality would take the wrong one and leave a
        record whose ``after`` no longer matches the file.
        """
        if change.before is None:
            os.remove(self.resolve(change.path))
        else:
            self.write(change.path, change.before)
        self.changes.clear()

    def sweep(self) -> list[str]:
        """Remove every directory from the working directory; return their names.

        The flat contract's enforcement half: the model can still create a
        directory from a snippet, and one that lingered would be second-class
        everywhere that matters — the listing and the search are both one
        level deep. Called after every tool call, and the caller reports what
        was removed: silent deletion of something the model wrote is worse
        than the directory was.
        """
        return sweep_directory(self.directory)

    def forget_changes(self) -> None:
        """Drop the change log, which is what bounds how far undo reaches."""
        self.changes.clear()

    def paths(self, path: str | None = None) -> list[pathlib.Path]:
        """The files a tool may open here, name-sorted, without opening any.

        What :meth:`entries` needs before it starts reading, and all a caller
        needs when it only wants names — globbing a search, say, where a line
        count is work whose answer is never shown.

        Sorted because raw directory order is a per-filesystem hash, and what
        is built from this reaches the model: leaving it would make two
        machines describe the same directory differently. Directories are
        skipped (the working directory is flat by contract, and any directory
        here is transient), and so is a symlink resolving outside — the tools
        refuse to open one, so listing it would advertise a file and then
        refuse every read of it.
        """
        root = pathlib.Path(self.directory if path is None else self.resolve(path))
        base = os.path.realpath(self.directory)
        return [
            entry
            for entry in sorted(root.iterdir())
            if entry.is_file() and os.path.realpath(entry).startswith(base + os.sep)
        ]

    def entries(self, path: str | None = None) -> list[Entry]:
        """Every file in the working directory, or in one inside it, measured.

        :meth:`paths` decides what is listed; this adds the measurements, and
        that is the expensive half — a line count means reading the file
        through. A caller that wants names only should ask :meth:`paths`.

        A file that is not text is listed with its size and no line count — a
        listing that hides the array the model stashed is worse than one that
        admits it cannot read it.
        """
        found: list[Entry] = []
        for entry in self.paths(path):
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            try:
                lines: int | None = len(entry.read_text().splitlines())
            except (OSError, UnicodeDecodeError):
                lines = None
            found.append(Entry(name=entry.name, size=size, lines=lines))
        return found


def sweep_directory(directory: str) -> list[str]:
    """Remove ``directory``'s subdirectories; return the removed names.

    A symlink is unlinked, never followed — following one would delete
    whatever it points at, which may be outside the workspace entirely.
    """
    removed: list[str] = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.islink(full):
            if os.path.isdir(full):
                os.unlink(full)
                removed.append(name + "/")
            continue
        if os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)
            removed.append(name + "/")
    return removed


def flat_note(removed: list[str]) -> str:
    """What the model is told when the sweep removed something.

    Appended to the tool result that triggered the sweep, so the teaching
    happens where the act happened rather than pages earlier in the prompt.
    """
    names = ", ".join(f"'{name}'" for name in removed)
    what = "directory" if len(removed) == 1 else "directories"
    return (
        f"\n\nNote: removed {what} {names} — your working directory is flat; "
        "keep every file directly in it, with no subdirectories."
    )
