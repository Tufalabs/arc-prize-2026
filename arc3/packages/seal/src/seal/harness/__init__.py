"""The surface the model programs against, under the name it sees.

The model writes ``from runtime.game import Action``, ``import runtime`` and
``dir(runtime.game)`` in its own REPL code, so these names are contract
rather than implementation detail. ``runtime`` rather than the whale's
``harness``: both packages register their surface in ``sys.modules`` of
whatever process imports them, so in one that imports both (the whale lives
on its own branch, ``d5bd7a6a``) neither answers for the other.

Each submodule declares the name it presents by setting ``__name__`` above
its definitions and calling :func:`present`. Class creation reads the
module's ``__name__``, so every class and function below that line reports
the presented module for free — no walking the namespace afterwards, and
nothing added later can quietly keep the internal name. A repr, a traceback,
``help()`` and a pickle then all name something the model can import.

:func:`present` also registers the name in ``sys.modules`` immediately,
which is load-bearing rather than tidy: ``@dataclass`` looks its own module
up by ``__module__`` while the class is being created, so the name has to
resolve before the rest of the module body runs. One module object under two
keys, never a copy — a copy would drift the moment either side changed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
import typing

# No bytecode caches, anywhere seal code loads model files: the kernel (its
# spawn env says the same), the host-side validate, and the search child. The
# working directory the model lists must hold only what the runtime and the
# model put there, and a __pycache__ that is never written also cannot serve a
# stale entry: a model file edited twice inside one
# filesystem timestamp cannot be served from a cache that does not exist.
# Set at import so every process that can load a model file gets it
# before any such load; the venv's own modules are compiled ahead of time and
# unaffected.
sys.dont_write_bytecode = True

#: The name the model imports the surface by.
PRESENTED = "runtime"

#: What the model reads of the package itself (``runtime.__doc__``). This
#: module's own docstring is ours, about how the surface is presented.
_PRESENTED_DOC = "The game's Python surface, in ``runtime.game``. Call ``help()`` for an overview."

_package = sys.modules.get(PRESENTED)
if _package is None:
    _package = types.ModuleType(PRESENTED)
    _package.__doc__ = _PRESENTED_DOC
    # A package, so ``import runtime.game`` is allowed to look for a
    # submodule at all — without ``__path__`` the import system rejects the
    # dotted name outright and never reaches a finder. Empty because nothing
    # is found on disk under this name: _PresentedFinder answers instead.
    _package.__path__ = []  # type: ignore[attr-defined]
    sys.modules[PRESENTED] = _package


def present(real_name: str, presented_name: str) -> None:
    """Register a submodule under the name it presents to the model.

    ``real_name`` is the module's import-system name (``__spec__.name``),
    which is the key it is already in ``sys.modules`` under; the module is
    added under ``presented_name`` as well, and hung off the ``runtime``
    package so ``import runtime.game`` resolves.
    """
    module = sys.modules[real_name]
    sys.modules[presented_name] = module
    leaf = presented_name.split(".", 1)[1]
    # A private submodule is registered — its own names have to resolve — but
    # is not hung off the package, where it would be one attribute access from
    # anything holding ``runtime``. ``_repl_state`` is the case: it exists as a
    # separate module precisely to stay off the model's discovery path, and an
    # attribute on the presented package is that path.
    if not leaf.startswith("_"):
        setattr(_package, leaf, module)


def install() -> None:
    """Import every presented submodule, so all the names resolve.

    Called in the REPL kernel, and by :class:`_PresentedFinder` on the first
    import of a presented name anywhere else.
    """
    import importlib

    # By name rather than by statement: the import is for its side effect —
    # each module registers itself under the name it presents — and a plain
    # import statement reads as unused to every checker that looks.
    for module in ("_repl_state", "game", "test"):
        importlib.import_module(f"seal.harness.{module}")


class _AlreadyImported(importlib.abc.Loader):
    """A loader for a module that is already loaded under another name.

    The import system insists on a loader, but there is nothing to load: the
    module object exists and ran its body when it was imported for real.
    """

    def __init__(self, module: types.ModuleType) -> None:
        self._module = module
        self._spec = module.__spec__

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        """The one that already exists — never a second copy of it."""
        return self._module

    def exec_module(self, module: types.ModuleType) -> None:
        """Nothing to execute; restore the spec ``module_from_spec`` just
        overwrote, so the module keeps naming where it really came from."""
        module.__spec__ = self._spec


class _PresentedFinder(importlib.abc.MetaPathFinder):
    """Makes ``runtime.game`` importable in a process that has not asked for
    the surface, which is how a pickle of model state reads back.

    A pickle records its classes by ``__module__``, and the model receives —
    and may stash in its own state — objects whose module is ``runtime.game``.
    Reading one back calls ``__import__("runtime.game")``, so anything
    unpickling model state needs the name to resolve: an offline repro from a
    diagnostics artifact, a checkpoint restore, a notebook. Registering this
    finder from ``seal/__init__.py`` costs one light module import, where
    calling :func:`install` eagerly would pull the whole surface — numpy
    included — into a process that only wants the agent's presets.
    """

    def find_spec(
        self,
        fullname: str,
        path: "typing.Sequence[str] | None" = None,
        target: types.ModuleType | None = None,
    ) -> "importlib.machinery.ModuleSpec | None":
        """A spec for any presented name, once the real modules are loaded."""
        if fullname != PRESENTED and not fullname.startswith(PRESENTED + "."):
            return None
        install()
        module = sys.modules.get(fullname)
        if module is None:
            return None  # a presented name that no submodule claims
        return importlib.util.spec_from_loader(fullname, _AlreadyImported(module), is_package=fullname == PRESENTED)


def install_finder() -> None:
    """Make the presented names importable, without importing them.

    Idempotent: called from ``seal/__init__.py``, which any number of
    modules may import.
    """
    if not any(isinstance(finder, _PresentedFinder) for finder in sys.meta_path):
        sys.meta_path.append(_PresentedFinder())
