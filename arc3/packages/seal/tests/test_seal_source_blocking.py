"""What the REPL may read, and where it may write.

``_block_source_access`` runs inside the kernel subprocess, patching
``inspect``, ``linecache`` and ``open`` so the model cannot read our source
and cannot write outside its own tempdir. Ported with the patches from the
whale, whose tests these are.

The patches are global to the interpreter, so each test installs them, does
its one thing, and restores. The rules object is separated from them
deliberately, and most of the interesting cases are on it.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import inspect
import io
import linecache
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from seal.harness import game


@pytest.fixture
def restore_patches() -> Iterator[None]:
    """Everything ``_block_source_access`` replaces, put back afterwards."""
    saved = (
        inspect.getsource,
        inspect.getsourcelines,
        inspect.getsourcefile,
        inspect.getfile,
        linecache.getline,
        linecache.getlines,
        builtins.open,
        io.open,
        os.listdir,
        os.scandir,
    )
    yield
    (
        inspect.getsource,
        inspect.getsourcelines,
        inspect.getsourcefile,
        inspect.getfile,
        linecache.getline,
        linecache.getlines,
        builtins.open,
        io.open,
        os.listdir,
        os.scandir,
    ) = saved


def test_our_source_is_blocked_and_everything_else_is_not() -> None:
    """The rule itself: our packages, and nothing else.

    numpy is the case worth pinning — the model calls into it constantly and
    reading it is as legitimate as reading any library's source.
    """
    rules = game._PathRules()

    assert rules.is_blocked(game.__file__)
    assert rules.is_blocked(str(Path(game.__file__).parent / "_repl_state.py"))
    assert not rules.is_blocked(np.__file__)
    assert not rules.is_blocked("/usr/lib/python3/json/decoder.py")
    # Not a path at all — an object handed to a patched entry point.
    assert not rules.is_blocked(object())


def test_the_engine_and_the_games_are_blocked_without_being_imported() -> None:
    """The games run in the host, but they are installed beside the libraries
    the model reads: a game's own source would hand over the level. Located,
    not imported, so a kernel does not load them to find them."""
    games = importlib.util.find_spec("re_arc")
    engine = importlib.util.find_spec("arcengine")
    if games is None or engine is None:
        pytest.skip("the game packages are not installed here")
    loaded = "re_arc" in sys.modules
    rules = game._PathRules()

    assert ("re_arc" in sys.modules) == loaded
    assert rules.is_blocked(os.path.join(next(iter(games.submodule_search_locations or [])), "environment_files"))
    assert rules.is_blocked(str(engine.origin))


def test_reads_stay_within_the_working_directory_python_and_the_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the directories beside the working directory, where the other
    games' are, nor shared storage holding earlier runs; Python, its
    libraries and the system's directories stay readable."""
    monkeypatch.chdir(tmp_path)
    rules = game._PathRules()

    assert rules.is_readable(str(tmp_path / "notes.txt"))
    assert rules.is_readable(np.__file__)
    assert rules.is_readable("/usr/lib/python3/json/decoder.py")
    assert rules.is_readable(os.path.expanduser("~/.cache/x"))
    assert not rules.is_readable(str(tmp_path.parent / "another_game" / "solver.py"))
    assert not rules.is_readable("/data/shared/research/arc_3_results")
    assert not rules.is_readable(game.__file__)


def test_open_and_listing_refuse_what_is_not_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_patches: None
) -> None:
    """A read outside comes back as the denial; a listing raises the error a
    walk or a glob already skips; the working directory lists as before."""
    monkeypatch.chdir(tmp_path)
    (tmp_path.parent / "beside.txt").write_text("another game's")
    (tmp_path / "mine.txt").write_text("mine")
    rules = game._PathRules()
    game._patch_open(rules)
    game._patch_listing(rules)

    with open(tmp_path.parent / "beside.txt") as f:
        assert f.read() == game._OUTSIDE_DENIAL
    with pytest.raises(PermissionError):
        os.listdir(tmp_path.parent)
    assert list(os.walk(tmp_path.parent)) == []
    assert os.listdir(tmp_path) == ["mine.txt"]
    assert os.listdir(os.path.dirname(np.__file__))
    # Our packages list like any library: names are not source.
    assert os.listdir(os.path.dirname(game.__file__))


def test_package_metadata_lookups_pass_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_patches: None
) -> None:
    """``importlib.metadata`` lists every ``sys.path`` entry and opens one as
    a zip when it cannot list it; scipy asks it at import. The kernel's own
    directory is inside a protected package, and a refusal there put two
    notices in front of the output of any cell importing scipy."""
    import importlib.metadata

    monkeypatch.chdir(tmp_path)
    rules = game._PathRules()

    def refused() -> None:
        raise AssertionError("a metadata lookup was refused")

    monkeypatch.setattr(rules, "deny_read", refused)
    monkeypatch.setattr(rules, "deny_outside", refused)
    game._patch_open(rules)
    game._patch_listing(rules)
    monkeypatch.syspath_prepend(os.path.dirname(game.__file__))

    assert importlib.metadata.version("numpy")
    assert list(importlib.metadata.distributions())


def test_writes_are_confined_to_the_repl_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tempdir is the kernel's spawn cwd, captured when the patches go in
    — so a later ``chdir`` does not move the boundary."""
    monkeypatch.chdir(tmp_path)
    rules = game._PathRules()

    assert rules.is_writable(str(tmp_path / "solver.py"))
    assert rules.is_writable(str(tmp_path / "nested" / "deep.pkl"))
    assert not rules.is_writable(str(tmp_path.parent / "elsewhere.py"))
    assert not rules.is_writable(game.__file__)

    # chdir away: the captured tempdir is what still counts.
    monkeypatch.chdir(tmp_path.parent)
    assert rules.is_writable(str(tmp_path / "solver.py"))


def test_inspect_returns_the_denial_rather_than_our_source(restore_patches: None) -> None:
    """What the model gets when it asks for a harness function's source."""
    rules = game._PathRules()
    game._patch_inspect(rules)

    assert inspect.getsource(game.Turn) == game._BLOCKED_DENIAL
    assert inspect.getsourcefile(game.Turn) is None
    assert inspect.getsourcelines(game.Turn) == ([game._BLOCKED_DENIAL], 0)
    # A library it may legitimately read still answers normally.
    assert "def " in inspect.getsource(np.take)


def test_a_blocked_file_reads_as_absent_to_linecache(restore_patches: None) -> None:
    """Empty, not the denial sentinel.

    The traceback formatter calls ``getline`` for every frame it renders. The
    sentinel there would replace the source line with "Source code access
    denied", which the model reads as the error itself and then debugs a
    problem that does not exist.
    """
    rules = game._PathRules()
    game._patch_linecache(rules)

    assert linecache.getline(game.__file__, 1) == ""
    assert linecache.getlines(game.__file__) == []
    assert game._BLOCKED_DENIAL not in linecache.getline(np.__file__, 1)


def test_open_denies_our_source_and_discards_a_write_outside_the_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_patches: None
) -> None:
    """Both halves of the ``open`` patch, and that the discarded write really
    does leave the on-disk file alone."""
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "untouched.txt"
    outside.write_text("original")

    real_open = builtins.open
    rules = game._PathRules()
    game._patch_open(rules)

    with open(game.__file__) as f:
        assert f.read() == game._BLOCKED_DENIAL
    with open(game.__file__, "rb") as f:
        assert f.read() == game._BLOCKED_DENIAL.encode()

    with open(outside, "w") as f:
        f.write("clobbered")
    with real_open(outside) as f:
        assert f.read() == "original"

    # Its own directory stays fully usable, which is the point of confining
    # rather than forbidding.
    mine = tmp_path / "solver.py"
    with open(mine, "w") as f:
        f.write("def solve(frame): ...")
    assert mine.read_text() == "def solve(frame): ..."


def test_the_write_notice_names_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The notice reaches the model's context, so it must reproduce anywhere.

    The tempdir carries the SLURM job id and this checkout's name on the
    cluster — a reference recording it replays on no other machine, and it
    tells the model a harness repository exists. The directory is the
    model's own cwd, so naming it would add nothing.
    """
    monkeypatch.chdir(tmp_path)
    rules = game._PathRules()
    rules.deny_write()

    notice = capsys.readouterr().out
    assert "working directory" in notice
    assert str(tmp_path) not in notice
    assert "/" not in notice, notice


def test_installing_twice_leaves_one_layer_of_patches(restore_patches: None) -> None:
    """A reused kernel or a re-import must not wrap the wrapper; each layer
    would be one more frame in the same check."""
    rules = game._PathRules()
    game._patch_inspect(rules)
    first = inspect.getsource
    game._patch_inspect(rules)

    assert inspect.getsource is first


def test_our_source_stays_unwritable_when_home_contains_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HOME exemption must not reach the code these rules protect.

    Writes under HOME are allowed because libraries cache there, and the
    kernel's HOME is a scratch directory outside the checkout. A machine that
    keeps its checkout under HOME breaks that assumption — a CI runner does,
    with the workspace under ``/home/runner`` — and the exemption would then
    hand out write access to our own source. Nothing in a real run reaches
    this, since the kernel's HOME is created under the system temp directory;
    it is pinned because the assumption is invisible where it holds.
    """
    container = Path(game.__file__).parents[2]  # the directory holding the packages
    monkeypatch.setenv("HOME", str(container))
    rules = game._PathRules()

    assert not rules.is_writable(game.__file__), "the HOME exemption reached our source"
    assert rules.is_writable(str(container / "cache" / "fontconfig")), "the exemption stopped working entirely"


def test_a_blocked_read_speaks_only_into_a_cell(capfd: pytest.CaptureFixture[str]) -> None:
    """The notice belongs in the cell's output and nowhere else.

    The same rules run outside a cell — rendering a traceback for the
    reply — where the kernel's stdout is the protocol stream the host reads
    as JSON. A notice printed there is not output the model sees: it
    corrupts the reply, and the host kills the kernel as it did once in the
    baseline run. The worker redirects stdout for a cell, so being
    redirected is what tells the two apart.
    """
    rules = game._PathRules()

    with contextlib.redirect_stdout(io.StringIO()) as inside:
        rules.deny_read()
        rules.deny_write()
    assert game._BLOCKED_NOTICE in inside.getvalue()
    assert game._WRITE_NOTICE in inside.getvalue()

    # Outside a cell, where the kernel's stdout is the protocol stream:
    # ``capfd`` captures the file descriptor, leaving the stdout OBJECT the
    # real one, which is what the guard reads.
    capfd.readouterr()
    with contextlib.redirect_stdout(sys.__stdout__):
        rules.deny_read()
        rules.deny_write()
    assert capfd.readouterr().out == ""


@pytest.fixture
def restore_environment() -> Iterator[None]:
    """The environment guard's changes, put back afterwards: the class of the
    two stores and the two functions that bypass them."""
    saved = (type(os.environ), type(os.environb), os.putenv, os.unsetenv)
    yield
    environ_class, environb_class, os.putenv, os.unsetenv = saved
    os.environ.__class__ = environ_class
    os.environb.__class__ = environb_class


def _as_if_from(filename: str, source: str) -> dict[str, object]:
    """Run ``source`` with its frames reporting ``filename``: a cell's
    ``<repl-N>``, a file in the working directory, or a library's path."""
    namespace: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "source",
    [
        "value = os.environ['PATH']",
        "value = os.environ.get('PATH')",
        "value = os.getenv('PATH')",
        "value = 'PATH' in os.environ",
        "value = list(os.environ)",
        "value = dict(os.environ)",
        "os.environ['SEAL_TEST_FLAG'] = '1'",
        "os.environ.update(SEAL_TEST_FLAG='1')",
        "os.putenv('SEAL_TEST_FLAG', '1')",
    ],
)
def test_the_models_code_cannot_touch_the_environment(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_environment: None
) -> None:
    """Refused from a cell and from a module in the working directory alike,
    by every route to the store, reads as well as writes."""
    monkeypatch.chdir(tmp_path)
    game._guard_environment(game._PathRules())  # pyright: ignore[reportPrivateUsage]
    for filename in ("<repl-3>", str(tmp_path / "world_model.py")):
        with pytest.raises(PermissionError, match="Environment variables are not available"):
            _as_if_from(filename, "import os\n" + source)
    assert "SEAL_TEST_FLAG" not in os.environ


def test_libraries_still_read_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_environment: None
) -> None:
    """The guard asks who is calling: a library reads and writes as before,
    also when the model's code is what called the library."""
    monkeypatch.chdir(tmp_path)
    game._guard_environment(game._PathRules())  # pyright: ignore[reportPrivateUsage]
    library = str(tmp_path.parent / "site-packages" / "somelib.py")
    namespace = _as_if_from(library, "import os\nvalue = os.environ.get('PATH')\nos.environ['SEAL_TEST_FLAG'] = '1'\n")
    assert namespace["value"] == os.environ["PATH"]
    del os.environ["SEAL_TEST_FLAG"]
    namespace = _as_if_from("<repl-4>", "import os.path\nvalue = os.path.expanduser('~')\n")
    assert namespace["value"] != "~"


async def test_a_cell_setting_a_variable_reads_the_refusal() -> None:
    """End to end in a real kernel: the cell's own frames are what the guard
    recognises, and the refusal arrives as the model's error with none of the
    guard's frames in it."""
    from seal.repl import GameREPL

    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="none")
    tool.host_functions = {
        "get_state": lambda: {"level": 1, "turns": [], "available_actions": []},
        "action": lambda normalized: {"ack": {"queued": len(normalized)}},  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    }
    try:
        out = (await tool._execute({"code": "import os\nos.environ['LM'] = '1'\n"})).to_string()  # pyright: ignore[reportPrivateUsage]
        assert "Environment variables are not available to your code" in out, out
        assert "game.py" not in out, out
        out = (
            await tool._execute({"code": "import numpy, tempfile\nprint('ok', bool(tempfile.gettempdir()))\n"})
        ).to_string()  # pyright: ignore[reportPrivateUsage]
        assert "ok True" in out, out
    finally:
        await tool.clear_state()
