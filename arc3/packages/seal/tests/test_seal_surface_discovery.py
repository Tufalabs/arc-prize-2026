"""What the model can find of the runtime by looking rather than by being told.

The presented ``runtime`` package exists so a repr, a traceback and a pickle
name something the model can import. That same presentation is a discovery
path: the whale found it used over 125 games — private names from ``dir()``
on a presented module, and the private store's own module repr handing over
its real name and an absolute path. A path is the half that costs more than
tidiness: it names a temp directory that does not reproduce even on the same
machine, so a reference recording one is unreplayable.

The whale's tests, over the seal's presentation.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

import seal.harness
import seal.harness.game


def _presented(name: str) -> ModuleType:
    """One presented module, as the REPL kernel reaches it.

    Imported by name rather than by statement: these names exist only once
    ``install()`` has registered them, so a static checker is right that
    ``import runtime`` resolves to nothing on disk.
    """
    seal.harness.install()
    return importlib.import_module(name)


def test_the_private_store_is_not_hung_off_the_package() -> None:
    """``runtime._repl_state`` must not resolve by attribute.

    It is a separate module *because* it should stay off the model's discovery
    path, and an attribute on the presented package is that path — one access
    from anything holding ``runtime``.
    """
    runtime = _presented("runtime")

    assert not hasattr(runtime, "_repl_state")
    assert hasattr(runtime, "game"), "the public surface still has to be reachable"


def test_the_private_store_never_names_its_file() -> None:
    """Whichever handle reaches it, its repr identifies no machine.

    The default module repr is ``<module 'x.y' from '/abs/path'>``. The
    bootstrap binds this module inside the REPL, so that path can reach the
    model by a route no attribute check closes.
    """
    store = _presented("runtime._repl_state")

    rendered = repr(store)

    assert "/" not in rendered, rendered
    assert "seal" not in rendered, rendered
    assert "private store" in rendered


def test_runtime_game_answers_dir_with_its_surface() -> None:
    """The surface and nothing else: not the imports the module needs, and
    not the patches and factories the bootstrap calls."""
    surface = _presented("runtime.game")

    assert dir(surface) == sorted(seal.harness.game.__all__)
    for private in ("_build_common_namespace", "_make_submit_action", "_PathRules", "segment_layer", "np"):
        assert private not in dir(surface)


def test_the_surface_does_not_collide_with_the_whales() -> None:
    """Both packages present a surface in ``sys.modules`` of whatever process
    imports them, so one that imports both (the whale lives on its own branch)
    needs the seal's name to be its own, or whichever registered last answers
    for the other."""
    assert seal.harness.PRESENTED != "harness"
    assert seal.harness.game.Frame.__module__ == "runtime.game"


def _help_output(capsys: pytest.CaptureFixture[str], *args: object) -> str:
    """What the surface's ``help`` prints for ``args``."""
    seal.harness.game.help(*args)
    return capsys.readouterr().out


def test_help_on_a_module_of_the_surface_is_the_overview(capsys: pytest.CaptureFixture[str]) -> None:
    """``help(runtime.game)`` and its siblings print what ``help()`` does.
    Pydoc's page would print each module's docstring and every class with the
    private base and the method resolution order that ``help(Frame)`` masks."""
    overview = _help_output(capsys)
    assert "Available types:" in overview
    for name in ("runtime", "runtime.game", "runtime.test", "runtime._repl_state"):
        assert _help_output(capsys, _presented(name)) == overview, name


def test_the_surface_documents_only_the_surface() -> None:
    """What the model reads of the presented modules themselves names nothing
    of how they are built."""
    for name in ("runtime", "runtime.game", "runtime.test"):
        doc = _presented(name).__doc__ or ""
        assert doc, name
        for internal in ("seal", "whale", "pytest", "sys.modules"):
            assert internal not in doc, (name, internal)
