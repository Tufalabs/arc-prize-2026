"""The concrete :class:`~seal.model.WorldModel` over model-written
``world_model.py`` source.

Everything the model produced is carried as bytes, not as a path: the source
itself, plus the sibling files it wrote (any helper module, any pickle it
stashed). A temp directory is materialised from them on every load, so the
same object works in the process that built it and in a search subprocess it
was pickled into, where the model's working directory is not its own.

The load is the one ``test_world_model()`` runs — the namespace from
``_build_wm_namespace`` with this level's ``history``, the module globals
frozen after the exec — so a search predicts what the test checked.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import sys
import tempfile
import types
import weakref
from typing import Any, Generator

import numpy as np
import numpy.typing as npt

from platypus.contract import Action
from seal import errors
from seal.action_names import model_move
from seal.harness import game as harness_game
from seal.harness import test as harness_test
from seal.model import DISTANCE_FN_NAME, RENDER_FN_NAME, ActiveScope, Prediction, WorldModel
from seal.palette import AsciiLayout


class PyWorldModel(WorldModel):
    """World model backed by an exec'd ``world_model.py``.

    Stores the files and construction data so it can rebuild its namespace
    from scratch — needed for pickle (subprocess search) and deepcopy.
    ``files`` holds every file of the working directory; they are written back
    out to a fresh temp directory before the exec, and that directory is the
    model's cwd whenever it runs.
    """

    def __init__(
        self,
        files: dict[str, bytes],
        turns: list[dict[str, Any]],
        available_actions: list[str],
        *,
        ascii_layout: AsciiLayout | None = None,
    ) -> None:
        super().__init__()
        self._files = dict(files)
        #: The kernel's board layout and alphabet, in force whenever the
        #: model's code runs here -- in this process and in the search's, where
        #: unpickling runs ``world_model.py`` again before anything else could
        #: configure it: a sprite it reads with ``from_ascii_grid`` at load has
        #: to read in the alphabet it was written in.
        self._ascii_layout = ascii_layout or AsciiLayout()
        #: The level's turns as the kernel receives them. Kept as payloads
        #: rather than a built ``History``, which is what pickles plainly.
        self._turns = list(turns)
        self._available_actions = list(available_actions)
        self._ns: dict[str, Any] = {}
        self._names: dict[str, Any] = {}
        self._dir = _materialise(self._files, self)
        self._load()

    @property
    def directory(self) -> str:
        """The materialised working directory this instance loads from."""
        return self._dir

    def history(self) -> Any:
        """A fresh ``History`` of the level, as the model's code reads it."""
        history = harness_game.History()
        for turn in self._turns:
            history._append_turn(turn)  # pyright: ignore[reportPrivateUsage]
        return history

    @property
    def files(self) -> dict[str, bytes]:
        """The model's own files, as written. The source of truth for the
        materialised directory — mutating the directory does not change these."""
        return self._files

    def _load(self) -> None:
        """Exec ``world_model.py`` into a fresh module. Runs in the main process
        and again in the search child, so the model is self-contained across the
        process boundary rather than depending on a directory that may be gone.
        """
        wm_path = os.path.join(self._dir, "world_model.py")
        # A registered module rather than a bare dict, and left registered:
        # classes the model defines can end up inside ``state``, and pickling
        # a state — the search's dedup key, the trace shipped to the child —
        # looks the class up as ``sys.modules['world_model'].Foo``. The child
        # unpickles the model first, so it is registered before the trace.
        mod = types.ModuleType("world_model")
        with self._environment():
            self._names = harness_test.run_world_model(  # the one runner every load shares
                mod, wm_path, self.history(), self._available_actions, keep_registered=True
            )
        self._ns = mod.__dict__

    @contextlib.contextmanager
    def _environment(self) -> Generator[None, None, None]:
        """Chdir into the materialised directory, so relative file I/O in
        model-written code resolves, with the directory on ``sys.path`` so a
        sibling module the model imports from inside a function resolves too.
        Entered once per outermost ``active()``, so the search pays for it once
        and not per node."""
        try:
            old_cwd = os.getcwd()
        except OSError:
            old_cwd = None
        os.chdir(self._dir)
        # Last, as in the kernel: a module Python already has keeps its name.
        sys.path.append(self._dir)
        # A module the model imports inside a function starts with the same
        # names as one it imports at load; a traceback through any of its
        # files reads as the model's own.
        preloaded = contextlib.ExitStack()
        preloaded.enter_context(errors.model_directory(self._dir))
        preloaded.enter_context(harness_game._ascii_layout_scope(self._ascii_layout))  # pyright: ignore[reportPrivateUsage]
        if self._names:
            preloaded.enter_context(harness_game._preloaded_import_scope(self._dir, self._names))  # pyright: ignore[reportPrivateUsage]
        try:
            with preloaded:
                yield
        finally:
            try:
                sys.path.remove(self._dir)
            except ValueError:
                pass
            self._forget_imports()
            if old_cwd is not None:
                try:
                    os.chdir(old_cwd)
                except OSError:
                    pass

    def _forget_imports(self) -> None:
        """Take the modules imported from this copy of the files out of
        ``sys.modules``. A helper the model imports inside a function is
        imported by the calls, after the load's own cleanup; left registered,
        it is what the next version's import of that name finds first, and
        the next test runs this version's helper."""
        prefix = os.path.abspath(self._dir) + os.sep
        protected = harness_game._WM_PROTECTED_MODULES  # pyright: ignore[reportPrivateUsage]
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name not in protected and path and os.path.abspath(path).startswith(prefix):
                del sys.modules[name]

    def _initial_state(self) -> dict[str, Any]:
        """The model's own ``initial_state()``."""
        return self._ns["initial_state"]()

    def _describe_move(self, action: Action) -> str:
        """The move as the model knows it, ``Action('UP')``."""
        return repr(harness_game._move(model_move(action)))  # pyright: ignore[reportPrivateUsage]

    def _step(self, state: dict[str, Any], action: Action) -> Prediction:
        """Run the model's ``do_move`` on the move as the model knows it. What it
        raises is the base class's to map; the outcome is read back from the
        state there."""
        self._ns["do_move"](state, harness_game._move(model_move(action)))  # pyright: ignore[reportPrivateUsage]
        return Prediction(kind="claim")

    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        """The board this state draws. Shape is not checked here — the replay
        is what holds the model to the contract."""
        return np.asarray(self._ns[RENDER_FN_NAME](state), dtype=np.int8)

    def _distance_to_win(self, state: dict[str, Any]) -> Any:
        """The model's heuristic, or ``None`` when it defined none. A heuristic
        that raises still raises, into the planner's auto-disable path."""
        fn = self._ns.get(DISTANCE_FN_NAME)
        if fn is None:
            return None
        return fn(state)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "files": self._files,
            "turns": self._turns,
            "available_actions": self._available_actions,
            "ascii_layout": self._ascii_layout,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._files = state["files"]
        self._turns = state["turns"]
        self._available_actions = state["available_actions"]
        self._ascii_layout = state["ascii_layout"]
        self._active_depth = 0
        self._active_scope = ActiveScope(self)
        self._ns = {}
        self._names = {}
        self._dir = _materialise(self._files, self)
        self._load()


def read_model_files(directory: str) -> dict[str, bytes]:
    """Every regular file in ``directory``, as bytes — the model's working
    directory captured for :class:`PyWorldModel`.

    Flat by design: the working directory has no subdirectories (the REPL
    sweeps them), and anything nested would not be importable as a sibling
    module anyway.
    """
    root = pathlib.Path(directory)
    return {entry.name: entry.read_bytes() for entry in sorted(root.iterdir()) if entry.is_file()}


def _materialise(files: dict[str, bytes], owner: object) -> str:
    """Write ``files`` into a fresh temp directory and return its path.

    The directory is removed when ``owner`` is collected — one directory per
    model instance, so a deepcopy or an unpickle in a search subprocess gets
    its own and nothing is shared through the filesystem. A search's process
    killed at its deadline collects nothing, so there the directory is made
    under the search's scratch directory (its ``TMPDIR``), which the parent
    removes (``search_process.run_search``).
    """
    directory = tempfile.mkdtemp(prefix="seal-model-")
    weakref.finalize(owner, shutil.rmtree, directory, True)
    for name, data in files.items():
        with open(os.path.join(directory, name), "wb") as f:
            f.write(data)
    return directory
