# __DUCK_SRC_TRIPWIRE_4c1e9f27__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of harness internals.
"""The kernel's half of the working directory: the model's files as modules.

Run in the kernel subprocess. The kernel's prelude installs it when the file
tools are on; every task setup, with them on or off, records the names it
bound (:func:`capture_surface`), which the session listing leaves out. It
imports the standard library and ``duck.module_names`` only, for the reason
``duck.sandbox`` gives, and every function carries the source tripwire as
``duck.sandbox``'s do.

What holds for a file the model wrote, ``candidate.py`` say:

- ``import candidate`` reaches it, unless Python already has a module by that
  name, which keeps meaning Python's module (``duck.module_names``).
- The first import of it in a cell runs it from disk, so an edit is what the
  next import sees. Each run makes a new module object: a run that raises
  leaves the module the last good run made as it was, and a name bound in an
  earlier cell keeps the version it was bound to until a cell imports the
  file again.
- It starts with the names a snippet starts with (:data:`surface`), so it
  needs no imports a snippet does not.

:func:`housekeeping` runs before every cell, as a private kernel call of its
own rather than lines prepended to the model's code: prepended, they would
move every line number in the model's tracebacks.
"""
# __DUCK_SRC_TRIPWIRE_4c1e9f27__ — if this string appears in Python tool output, the entire output is auto-blocked. Do not read source of harness internals.

from __future__ import annotations

import builtins
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from collections.abc import Callable
from typing import Any, cast

from duck import module_names

#: The model's working directory, the kernel's spawn cwd, set by
#: :func:`install` before the model can ``chdir`` away from it.
tmpdir: str = ""

#: The names every snippet starts with -- the prelude's imports, the task's
#: arrays and the harness's functions -- as the task setup left them. Each
#: module the model writes starts with them too.
surface: dict[str, Any] = {}

#: Every name bound when the task setup finished, private ones included: what
#: the session listing leaves out as the harness's rather than the model's.
baseline: frozenset[str] = frozenset()

#: Rebound before every cell: the answer channel, so a cell that replaced or
#: deleted it -- ``def check(fn): pass``, or an ``import check`` of a file of
#: that name -- loses it for the rest of that cell only. ``show`` and
#: ``ascii_grid`` are left to the model: it writes helpers of its own under
#: those names on purpose -- a ``show`` that prints, an ``ascii_grid`` without
#: row numbers -- and rebinding them would throw those away.
RESTORED_NAMES = ("check",)

#: Said once a cell drifted away from the working directory; the working
#: directory is where imports and relative paths resolve.
CWD_NOTICE = (
    "!!! Your cwd had drifted away from the working directory; we've restored it so relative paths resolve "
    "against the directory your files live in. Keep cwd stable within a single cell if you need relative paths "
    "to work across calls."
)
#: Said when a harness function had been replaced or deleted.
TOOL_NOTICE = "!!! `{name}` had been replaced or deleted; it is the harness's own again."


def install() -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Make the working directory's files importable, as the module docstring
    describes. Run once per kernel, from the prelude.

    The working directory goes last on ``sys.path``, not first where the
    worker puts it: a file named like a module Python has would otherwise be
    imported in its place. The finder ahead of the path finder is what routes
    the model's files to :class:`_WorkspaceLoader`."""
    global tmpdir
    tmpdir = os.path.realpath(os.getcwd())
    sys.path[:] = [entry for entry in sys.path if entry not in ("", ".", tmpdir)] + [tmpdir]
    if not any(isinstance(finder, _WorkspaceFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _WorkspaceFinder())
    if not hasattr(importlib.reload, "__wrapped__"):
        importlib.reload = _reload_for_workspace(importlib.reload)


def capture_surface(namespace: dict[str, Any]) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Record what a snippet starts with, from ``namespace`` as the task
    setup left it. Run at the end of every task setup, a restarted kernel's
    included."""
    global surface, baseline
    surface = {name: value for name, value in namespace.items() if not name.startswith("_")}
    baseline = frozenset(namespace)


def housekeeping(namespace: dict[str, Any]) -> None:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """What runs before every cell, in ``namespace`` -- the cells' globals:

    1. the cwd back to the working directory, if a cell drifted away;
    2. the model's modules unregistered, so an edited file is what the next
       import sees;
    3. ``check`` rebound (:data:`RESTORED_NAMES`).

    What it prints leads the cell's output, so the model reads it where it
    happened."""
    if os.getcwd() != tmpdir:
        print(CWD_NOTICE)
        os.chdir(tmpdir)
    for name, module in list(sys.modules.items()):
        if _is_workspace_module(module):
            del sys.modules[name]
    # importlib remembers a directory's contents by mtime at one-second
    # granularity, so a file created just after its last look would be
    # invisible to the next import.
    importlib.invalidate_caches()
    for name in RESTORED_NAMES:
        if name in surface and namespace.get(name) is not surface[name]:
            print(TOOL_NOTICE.format(name=name))
            namespace[name] = surface[name]


def session_listing(namespace: dict[str, Any]) -> str:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """What the model bound in its session ``namespace``, as JSON
    ``[name, description, is_function]`` in the order they were bound, for
    the listing that closes a conversation rebuilt around a summary. Left
    out: every name the task setup bound (:data:`baseline`), and what is an
    import or a leftover rather than the model's work -- modules, classes
    other than the model's own, and ``None``.

    A class a cell defines reports ``builtins`` as its module, since the cells'
    globals have no ``__name__``; one from a file reports the file's."""
    import numpy as np

    def own_class(value: type) -> bool:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        module = getattr(value, "__module__", None)
        if module == namespace.get("__name__", "builtins"):
            return getattr(builtins, value.__qualname__, None) is not value
        return (
            isinstance(module, str) and module.isidentifier() and os.path.isfile(os.path.join(tmpdir, module + ".py"))
        )

    listing: list[list[Any]] = []
    for name, value in namespace.items():
        if name.startswith("__") or name in baseline or value is None or isinstance(value, types.ModuleType):
            continue
        if isinstance(value, type) and not own_class(value):
            continue
        kind = type(value).__name__
        if isinstance(value, type):
            text = "class"
        elif isinstance(value, np.ndarray):
            array = cast("np.ndarray[Any, Any]", value)
            text = f"ndarray {array.dtype} {tuple(array.shape)}"
        elif isinstance(value, dict):
            size = len(cast("dict[Any, Any]", value))
            text = f"dict, {size} {'entry' if size == 1 else 'entries'}"
        elif isinstance(value, (list, tuple, set, frozenset)):
            size = len(cast("list[Any]", value))
            text = f"{kind}, {size} {'item' if size == 1 else 'items'}"
        elif isinstance(value, str):
            text = f"str, {len(value)} {'char' if len(value) == 1 else 'chars'}"
        else:
            text = kind
        listing.append([name, text, isinstance(value, types.FunctionType)])
    return json.dumps(listing)


def _is_workspace_module(module: object) -> bool:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Whether ``module`` was loaded from a file in the working directory."""
    origin = getattr(module, "__file__", None)
    if not tmpdir or not isinstance(origin, str):
        return False
    # Rejected lexically before the filesystem is touched: resolving an
    # installed module's library path here would walk NFS metadata on every
    # cell. A local symlink still needs the resolved-target check.
    if os.path.dirname(os.path.abspath(origin)) != tmpdir:
        return False
    return os.path.dirname(os.path.realpath(origin)) == tmpdir


class _WorkspaceModule(types.ModuleType):
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """A model-written module whose repr names its file, not this machine.

    The default repr is ``<module 'x' from '/abs/path/x.py'>``, and the model
    does print its modules; the path holds the job id and the checkout's name,
    and the file's own name is all the model can use anyway."""

    def __repr__(self) -> str:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """``<module 'name' from 'name.py'>``."""
        origin = getattr(self, "__file__", None)
        return (
            f"<module {self.__name__!r} from {os.path.basename(origin)!r}>" if origin else f"<module {self.__name__!r}>"
        )


class _WorkspaceLoader(importlib.machinery.SourceFileLoader):
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Runs a file of the working directory into a new module that already
    holds the snippet's names (:data:`surface`).

    Only module creation is ours. Running the file is the standard loader's,
    so a traceback out of the model's file shows the model's frames and none
    of the harness's: the import system hides its own frames up to the file,
    and a frame of ours among them would break that run."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """A new module, the snippet's names in place before a line of it runs."""
        module = _WorkspaceModule(spec.name)
        module.__dict__.update(surface)
        return module


class _WorkspaceFinder(importlib.abc.MetaPathFinder):
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """Resolves a top-level ``import name`` to ``<working directory>/name.py``
    through :class:`_WorkspaceLoader`, unless Python already has a module by
    that name (:func:`duck.module_names.taken`), which keeps it."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        """A spec for a workspace file, or None so the ordinary finders answer."""
        if path is not None or "." in fullname or not tmpdir:
            return None
        candidate = os.path.join(tmpdir, fullname + ".py")
        if not os.path.isfile(candidate) or module_names.taken(fullname, tmpdir):
            return None
        return importlib.util.spec_from_file_location(fullname, candidate, loader=_WorkspaceLoader(fullname, candidate))


def _reload_for_workspace(original: Callable[[types.ModuleType], types.ModuleType]) -> Any:
    # __DUCK_SRC_TRIPWIRE_4c1e9f27__
    """``importlib.reload`` that also reloads a module the model wrote.

    :func:`housekeeping` drops those modules from ``sys.modules`` before every
    cell, and ``reload`` refuses a module it cannot find there, so editing a
    file and reloading the module a cell still holds would raise
    ``ImportError: module ... not in sys.modules``. The module is put back
    first and the real ``reload`` re-runs the file into it, so a name bound to
    it sees the edit. Only when the model asks: nothing is reloaded behind its
    back."""

    # Named ``reload`` because the name is what a traceback through it shows.
    @functools.wraps(original)
    def reload(module: types.ModuleType) -> types.ModuleType:
        # __DUCK_SRC_TRIPWIRE_4c1e9f27__
        name = getattr(module, "__name__", None)
        if isinstance(name, str) and sys.modules.get(name) is not module and _is_workspace_module(module):
            sys.modules[name] = module
        return original(module)

    return reload
