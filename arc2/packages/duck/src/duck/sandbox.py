"""Accident prevention inside the model's kernel.

Lifted from the whale's `harness/game.py`, which is where this mechanism was
debugged. It runs in the kernel subprocess, installed once from the prelude,
and does two things: keeps the model's reads out of our package sources, and
confines its writes to its own working directory.

**Accident prevention, not a sandbox.** Only these entry points are patched.
`os.open` is not, integer file descriptors are waved through by design, and
`shutil` / `subprocess` are untouched — so a determined write outside the
tempdir still lands. What it buys is that a model reaching for our source
reads an explanation instead of finding the harness, and that a model
stashing a file under a literal `/tmp` path does not collide with the kernel
next door.

The ARC-AGI-2 task file is *not* blocked, and neither is the Kaggle
competition mount. The answers to the questions live there, so a model that
went looking could read them — the same exposure the ARC3 stack has, where
the game files are also on disk, and left open on the same grounds: the
model does not reach for them and the traces would show it if one did. It is
recorded here so nobody reads this module as a guarantee it does not make.
"""

from __future__ import annotations

import os
from typing import Any, Self, cast

#: Packages whose source the model may not read. Resolved HOST-side by
#: :func:`host_blocked_dirs` and passed in, because importing them is exactly
#: what this must not do: the kernel is an isolated worker, and pulling the
#: agent stack into it costs far longer than a kernel is allowed to take to
#: start. The kernel therefore imports nothing but the standard library.
PROTECTED = ("duck", "platypus", "taaf", "ullm")

#: If this appears in kernel output, the model got at something it should not
#: have and the whole output is discarded (`python_tool.DuckPythonTool`).
#: Every function below carries it so that reading this source trips it.
SOURCE_TRIPWIRE = "__DUCK_SRC_TRIPWIRE_4c1e9f27__"

#: Shown when a read is refused.
BLOCKED_NOTICE = (
    "[Source code for the harness that runs this task is not available. You "
    "can freely read files in your working directory and any files you "
    "create. Only the harness implementation is blocked. Use help(obj) for "
    "docstrings and signatures.]"
)
BLOCKED_DENIAL = "Source code access denied"

#: Shown when a write lands outside the kernel's own directory. It
#: deliberately does NOT name that directory: an absolute path here is
#: machine-specific and would tell the model a harness checkout exists at
#: all, while being of no use to it — the directory named is already its cwd,
#: so every relative path it writes resolves there.
WRITE_NOTICE = (
    "[Writes outside your working directory are not allowed. You can freely "
    "create and modify files in your working directory (the one you start "
    "in, and where a relative path resolves). The write was discarded; the "
    "on-disk file is unchanged.]"
)


class PathRules:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of harness internals.
    """Which paths the kernel may read, and where it may write.

    Resolved once, when the kernel installs the patches, and shared by all of
    them so every entry point answers the same question the same way.
    ``blocked_dirs`` is resolved host-side (:func:`host_blocked_dirs`); the
    tempdir is the kernel subprocess's spawn cwd, captured before the model
    can ``chdir`` away from it.
    """

    def __init__(self, blocked_dirs: list[str]) -> None:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        self.home = os.path.realpath(os.path.expanduser("~"))
        self.blocked_dirs = [os.path.realpath(d) for d in blocked_dirs]
        self.tempdir = os.path.realpath(os.getcwd())

    @staticmethod
    def resolve(path: Any) -> str | None:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """One argument as an absolute path, or None if it is not one."""
        try:
            # cast + ignore: `path` is whatever the patched entry point was
            # handed, so fspath/realpath cannot pick their str-vs-bytes
            # overload; every call site compares against str paths.
            return cast(str, os.path.realpath(os.fspath(path)))  # pyright: ignore[reportUnknownArgumentType]
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _within(path: str, directory: str) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Whether ``path`` is ``directory`` or sits under it.

        The separator matters: a plain prefix test would put ``/tmp/kernel2``
        inside ``/tmp/kernel``.
        """
        return path == directory or path.startswith(directory + os.sep)

    def is_blocked(self, path: Any) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Whether reading this path would hand the model our source."""
        resolved = self.resolve(path)
        if resolved is None:
            return False
        return any(self._within(resolved, d) for d in self.blocked_dirs)

    def is_writable(self, path: Any) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Whether a write here is allowed: the kernel's tempdir, or its
        scratch HOME — libraries write caches under ``~``, and HOME points
        outside the working directory precisely so those writes do not land
        where the model's own files are listed."""
        resolved = self.resolve(path)
        if resolved is None:
            return False
        if any(self._within(resolved, d) for d in self.blocked_dirs):
            # Protected source is never writable, wherever it sits. The HOME
            # exemption takes HOME to be a scratch directory outside the
            # checkout — which the kernel's is, but a machine keeping its
            # checkout under HOME is not, and there the exemption would
            # otherwise reach the very code these rules protect.
            return False
        return self._within(resolved, self.tempdir) or self._within(resolved, self.home)

    def deny_read(self) -> None:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Tell the model why the read it just made came back empty."""
        print(BLOCKED_NOTICE, flush=True)

    def deny_write(self) -> None:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Tell the model why the write it just made went nowhere."""
        print(WRITE_NOTICE, flush=True)


def host_blocked_dirs() -> list[str]:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Where :data:`PROTECTED` lives, resolved in the host process.

    Called where those packages are already imported, so it costs nothing;
    the result is baked into the kernel's startup code as literals.
    """
    dirs: list[str] = []
    for name in PROTECTED:
        try:
            module = __import__(name)
        except ImportError:
            continue
        path = getattr(module, "__file__", None)
        if path:
            dirs.append(os.path.realpath(os.path.dirname(path)))
    return dirs


def _patch_inspect(rules: PathRules) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """``inspect``'s four source-reading entry points, each returning the
    denial sentinel for a blocked object rather than its source."""
    import inspect as _inspect

    orig_getsource = _inspect.getsource
    orig_getsourcelines = _inspect.getsourcelines
    orig_getsourcefile = _inspect.getsourcefile
    orig_getfile = _inspect.getfile

    def object_is_blocked(obj: Any) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        try:
            return rules.is_blocked(orig_getfile(obj))
        except (TypeError, OSError):
            return False

    def _getsource(obj: Any, *args: Any, **kwargs: Any) -> str:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if object_is_blocked(obj):
            rules.deny_read()
            return BLOCKED_DENIAL
        return orig_getsource(obj, *args, **kwargs)

    def _getsourcelines(obj: Any, *args: Any, **kwargs: Any) -> tuple[list[str], int]:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if object_is_blocked(obj):
            rules.deny_read()
            return ([BLOCKED_DENIAL], 0)
        return orig_getsourcelines(obj, *args, **kwargs)

    def _getsourcefile(obj: Any, *args: Any, **kwargs: Any) -> str | None:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if object_is_blocked(obj):
            rules.deny_read()
            return None
        return orig_getsourcefile(obj, *args, **kwargs)

    def _getfile(obj: Any, *args: Any, **kwargs: Any) -> str:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if object_is_blocked(obj):
            rules.deny_read()
            return BLOCKED_DENIAL
        return orig_getfile(obj, *args, **kwargs)

    for patched in (_getsource, _getsourcelines, _getsourcefile, _getfile):
        patched._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_inspect.getsource, "_sandbox_patched", False):
        _inspect.getsource = _getsource
        _inspect.getsourcelines = _getsourcelines
        _inspect.getsourcefile = _getsourcefile
        _inspect.getfile = _getfile


def _patch_linecache(rules: PathRules) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """``linecache``, which is how ``inspect`` reads a file underneath.

    These return empty lines rather than the denial sentinel. Python's
    traceback formatter calls ``linecache.getline`` for every frame it shows,
    so returning the sentinel would put ``Source code access denied`` where
    the source line goes — and a model reading that traceback takes it for
    the root error and debugs a problem that does not exist. Empty is what
    linecache already answers for a file that is not there: the traceback
    omits the source line and the real exception reads cleanly.
    """
    import linecache as _linecache

    orig_getline = _linecache.getline
    orig_getlines = _linecache.getlines

    def _getline(filename: str, lineno: int, *args: Any, **kwargs: Any) -> str:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if rules.is_blocked(filename):
            return ""
        return orig_getline(filename, lineno, *args, **kwargs)

    def _getlines(filename: str, *args: Any, **kwargs: Any) -> list[str]:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if rules.is_blocked(filename):
            return []
        return orig_getlines(filename, *args, **kwargs)

    _getline._sandbox_patched = True  # type: ignore[attr-defined]
    _getlines._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_linecache.getline, "_sandbox_patched", False):
        _linecache.getline = _getline
        _linecache.getlines = _getlines


def _patch_open(rules: PathRules) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """``builtins.open`` and ``io.open``, which is where ``pathlib`` lands.

    A blocked read and a write outside the tempdir both come back as an
    in-memory proxy holding the denial text: it supports read / readlines /
    iteration *and* write, so the caller's code runs to completion and the
    write is discarded on close rather than reaching disk.
    """
    import builtins as _builtins
    import io as _io

    def is_write_mode(mode: str) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        # 'w'/'a'/'x' write/append/exclusive-create; '+' adds write to any read mode.
        return any(c in mode for c in ("w", "a", "x", "+"))

    def proxy(mode: str) -> _io.BytesIO | _io.StringIO:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        if "b" in mode:
            return _io.BytesIO(BLOCKED_DENIAL.encode("utf-8"))
        return _io.StringIO(BLOCKED_DENIAL)

    def make_patched_open(orig: Any) -> Any:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        def _patched_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            # int (fd) bypasses — fds cannot be resolved to paths reliably
            # and blocking them breaks too much legitimate code.
            if isinstance(file, int):
                return orig(file, mode, *args, **kwargs)
            try:
                # ignore: `file` is whatever open() was called with, so fspath
                # cannot pick its str-vs-bytes overload; both consumers take Any.
                path_str = os.fspath(file)  # pyright: ignore[reportUnknownVariableType]
            except TypeError:
                return orig(file, mode, *args, **kwargs)
            # Write modes are kept to the tempdir. Checked before the
            # blocked-path test so a write to one of our paths gets the
            # write-specific notice, which is the more informative one.
            if is_write_mode(mode) and not rules.is_writable(path_str):
                rules.deny_write()
                return proxy(mode)
            if rules.is_blocked(path_str):
                rules.deny_read()
                return proxy(mode)
            return orig(file, mode, *args, **kwargs)

        _patched_open._sandbox_patched = True  # type: ignore[attr-defined]
        return _patched_open

    if not getattr(_builtins.open, "_sandbox_patched", False):
        _builtins.open = make_patched_open(_builtins.open)
    if not getattr(_io.open, "_sandbox_patched", False):
        _io.open = make_patched_open(_io.open)


def block_source_access(blocked_dirs: list[str]) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Block reads of our package sources via inspect, linecache and open,
    and confine writes to the kernel's tempdir. Everything else — the tempdir
    itself, files the model creates, the standard library, numpy — stays
    readable.

    Every trigger prints its notice to stdout so the model gets the same
    explanation whichever entry point it tried, and gets back a short
    sentinel that is safe under ``.splitlines()`` / substring search /
    iteration: there is no source content left for a follow-up operation to
    uncover. Idempotent, via a ``_sandbox_patched`` marker on each
    replacement.
    """
    rules = PathRules(blocked_dirs)
    _patch_inspect(rules)
    _patch_linecache(rules)
    _patch_open(rules)


def sort_directory_listings() -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Make every directory listing the model can reach come back in name
    order.

    ``os.listdir`` and ``os.scandir`` return raw readdir order, which on ext4
    with ``dir_index`` is a filename hash carrying a per-filesystem seed. The
    same tempdir, built by the same code, therefore lists differently on a
    different machine — and the model does call ``os.listdir()``, so that
    order reaches it and a recorded run cannot be replayed elsewhere.

    Both are patched because they are separate implementations, and ``glob``,
    ``os.walk`` and ``pathlib.Path.iterdir`` all route through ``scandir``.
    """
    import os as _os

    _orig_listdir = _os.listdir
    _orig_scandir = _os.scandir

    def _listdir(path: Any = None) -> list[Any]:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        return sorted(_orig_listdir(path))  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportCallIssue]

    _listdir._sandbox_patched = True  # type: ignore[attr-defined]

    class _SortedScandir:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """Materialised scandir result in name order.

        Still a context manager and an iterator: ``os.walk`` and ``glob`` both
        use ``with scandir(...) as it``, so returning a bare ``iter()`` would
        break them.
        """

        def __init__(self, entries: list[Any]):
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            self._entries = entries
            self._iter = iter(entries)

        def __iter__(self) -> Any:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            return self._iter

        def __next__(self) -> Any:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            return next(self._iter)

        def __enter__(self) -> Self:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            return self

        def __exit__(self, *exc: object) -> None:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            return None

        def close(self) -> None:
            # __DUCK_SRC_TRIPWIRE_4c1e9f27__
            return None

    def _scandir(path: Any = ".") -> Any:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        with _orig_scandir(path) as it:  # pyright: ignore[reportUnknownArgumentType]
            entries = sorted(it, key=lambda e: e.name)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
        return _SortedScandir(entries)  # pyright: ignore[reportUnknownArgumentType]

    _scandir._sandbox_patched = True  # type: ignore[attr-defined]

    if not getattr(_os.listdir, "_sandbox_patched", False):
        _os.listdir = _listdir  # pyright: ignore[reportAttributeAccessIssue]
    if not getattr(_os.scandir, "_sandbox_patched", False):
        _os.scandir = _scandir  # pyright: ignore[reportAttributeAccessIssue]


def install(blocked_dirs: list[str]) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Everything the kernel installs at startup, in one call from the
    prelude. ``blocked_dirs`` comes from :func:`host_blocked_dirs`."""
    block_source_access(blocked_dirs)
    sort_directory_listings()


__all__ = [
    "BLOCKED_DENIAL",
    "BLOCKED_NOTICE",
    "PROTECTED",
    "SOURCE_TRIPWIRE",
    "WRITE_NOTICE",
    "PathRules",
    "block_source_access",
    "host_blocked_dirs",
    "install",
    "sort_directory_listings",
]
