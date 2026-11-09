# __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.
"""The kernel's filesystem, confined by the operating system (Linux Landlock).

The guard in ``seal.harness.game`` patches the Python entry points a model
reads and writes through; ``os.open``, a shell-out or a C extension go around
it, and the long runs' models did exactly that to read the games' source.
Landlock is enforced by Linux itself, on this process and on everything it
starts, and cannot be lifted once applied, so nothing the model runs gets
past it.

Landlock can only grant. Each rule allows a set of accesses beneath one
directory (or on one file), and anything no rule allows is refused. So
"this tree except these directories" is not one rule but many: the tree's
entries are granted one by one, skipping the hidden ones, descending only
into the directories that contain one (:func:`restrict`). The directories
on the way down stay listable — Python's import system lists them — so a
hidden directory's name shows, but nothing in it can be read.

Our own code keeps running because it is imported before the restriction:
Landlock checks a file when it is opened, not what is already in memory.
An import of ours after it fails.
"""
# __SANDBOX_SRC_TRIPWIRE_a4c1f70d__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of sandbox internals.

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable

# x86_64 and aarch64 share these (both take them from the generic table).
_SYS_CREATE_RULESET = 444
_SYS_ADD_RULE = 445
_SYS_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1 << 0
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12
REFER = 1 << 13  # ABI 2: link or rename into another directory
TRUNCATE = 1 << 14  # ABI 3

#: What can be granted on a file, as opposed to beneath a directory.
_FILE_ACCESS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE

#: Reading and running, but nothing that changes the filesystem.
READ = READ_FILE | READ_DIR | EXECUTE


class _RulesetAttr(ctypes.Structure):
    """``struct landlock_ruleset_attr``, its first field only: the filesystem
    accesses this ruleset handles. Network rules are left unhandled (the
    kernel already runs without a network)."""

    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    """``struct landlock_path_beneath_attr`` (packed)."""

    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = ctypes.CDLL(None, use_errno=True)


def abi_version() -> int:
    """The Landlock ABI this kernel offers; 0 when it offers none (an older
    kernel, Landlock not among the active security modules, or a seccomp
    filter that refuses the call)."""
    version = _libc.syscall(_SYS_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_RULESET_VERSION))
    return max(int(version), 0)


def handled_access(abi: int) -> int:
    """Every filesystem access ``abi`` can restrict, so none is left
    unrestricted by omission."""
    access = (MAKE_SYM << 1) - 1
    if abi >= 2:
        access |= REFER
    if abi >= 3:
        access |= TRUNCATE
    return access


def _within(path: str, directory: str) -> bool:
    """Whether ``path`` is ``directory`` or lies under it."""
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


class _Ruleset:
    """An open Landlock ruleset, rules added one path at a time."""

    def __init__(self, handled: int) -> None:
        self.handled = handled
        attr = _RulesetAttr(handled_access_fs=handled)
        fd = _libc.syscall(_SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), 0)
        if fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"landlock_create_ruleset: {os.strerror(errno)}")
        self.fd = int(fd)
        # Directories already split into their entries: a symlink back up the
        # tree would otherwise split the same directory forever.
        self._split: set[str] = set()

    def allow(self, path: str, access: int) -> None:
        """Grant ``access`` beneath ``path`` (on it, for a file). A path that
        does not exist is skipped: the rules name places a machine may lack."""
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except FileNotFoundError:
            return
        try:
            if not os.path.isdir(path):
                access &= _FILE_ACCESS
            access &= self.handled
            if not access:
                return
            attr = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
            if _libc.syscall(_SYS_ADD_RULE, self.fd, _RULE_PATH_BENEATH, ctypes.byref(attr), 0) != 0:
                errno = ctypes.get_errno()
                raise OSError(errno, f"landlock_add_rule {path}: {os.strerror(errno)}")
        finally:
            os.close(fd)

    def allow_except(self, root: str, access: int, hidden: list[str]) -> None:
        """Grant ``access`` beneath ``root`` but not beneath any of ``hidden``:
        ``root`` whole if nothing hidden lies under it, else each of its
        entries the same way, and ``root`` itself listable only."""
        inside = [h for h in hidden if _within(h, root)]
        if not inside:
            self.allow(root, access)
            return
        if root in inside or root in self._split:
            return
        self._split.add(root)
        self.allow(root, READ_DIR & access)
        with os.scandir(root) as entries:
            children = [entry.path for entry in entries]
        for child in children:
            # A symlink out of the tree grants nothing here: where it leads is
            # granted by a rule of its own, or not at all.
            target = os.path.realpath(child)
            if _within(target, root):
                self.allow_except(target, access, inside)

    def enforce(self) -> None:
        """Apply the ruleset to this process and everything it starts."""
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"prctl(PR_SET_NO_NEW_PRIVS): {os.strerror(errno)}")
        if _libc.syscall(_SYS_RESTRICT_SELF, self.fd, 0) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"landlock_restrict_self: {os.strerror(errno)}")
        os.close(self.fd)


def restrict(
    *, readable: Iterable[str], writable: Iterable[str], hidden: Iterable[str], listable: Iterable[str] = ()
) -> None:
    """Confine this process and everything it starts, for good: reads (and
    running programs) beneath ``readable``, everything beneath ``writable``,
    directory listings alone beneath ``listable``, and nothing else at all —
    except that nothing beneath ``hidden`` can be read, wherever it sits. A
    path that does not exist is skipped.

    Raises ``OSError`` where Landlock is not available: the caller asked for
    a confinement, and running without one is not the same process.
    """
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is not available on this machine (an older kernel, or not an active security module)")
    hidden_paths = sorted({os.path.realpath(path) for path in hidden})
    ruleset = _Ruleset(handled_access(abi))
    for root in sorted({os.path.realpath(path) for path in readable}):
        ruleset.allow_except(root, READ, hidden_paths)
    for root in sorted({os.path.realpath(path) for path in writable}):
        ruleset.allow_except(root, ruleset.handled, hidden_paths)
    for root in sorted({os.path.realpath(path) for path in listable}):
        ruleset.allow(root, READ_DIR)
    ruleset.enforce()
