"""Where a path is allowed to land, and where the answer is read from.

The workspace is the one place the duck's file rule lives, and it is a rule
about where a path *lands*, not how it is spelled: a snippet can leave a
symlink behind, so a name that looks local is not a promise that what sits
under it still is.

The kernel's sandbox is accident prevention rather than a boundary
(``duck.sandbox``), and a kernel can already reach the filesystem. What is
under test is that the TOOLS keep to the directory they claim to be scoped to,
and that they ask the live kernel where that directory is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
from duck.workspace import NotText, PathNotAllowed, Workspace


class _StubTool:
    """Only what :class:`Workspace` reads: where the files are."""

    def __init__(self, directory: Path) -> None:
        self._tmpdir = str(directory)


def _workspace(directory: Path) -> Workspace:
    """A workspace over ``directory``, configured as the duck ships it."""
    return Workspace(repl=cast("Any", _StubTool(directory)))


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("../escape.py", id="out_of_the_directory"),
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("sub/nested.py", id="subdirectory"),
        pytest.param("bad\0name.py", id="null_byte"),
    ],
)
def test_a_path_that_does_not_belong_here_is_refused(tmp_path: Path, path: str) -> None:
    """Every refusal is a ``PathNotAllowed`` the tools can answer with.

    The null byte earns its row: it raises ``ValueError``, which is not an
    ``OSError``, so without the trap it leaves the tool entirely — and a raise
    out of a tool ends the game instead of reaching the model.
    """
    with pytest.raises(PathNotAllowed):
        _workspace(tmp_path).resolve(path)


def test_any_name_inside_the_directory_is_reachable(tmp_path: Path) -> None:
    """Containment is the rule; the name is not.

    A snippet writes here directly — ``np.save`` and ``np.savetxt`` reach the
    real filesystem — so a workspace that refused anything but ``.py`` would
    list files it then declined to open, which describes a directory the model
    does not have.
    """
    workspace = _workspace(tmp_path)
    for name in ("helper.py", "notes.txt", "scores.npy", "no_suffix"):
        assert workspace.resolve(name).startswith(str(tmp_path))


def test_a_narrowed_workspace_still_refuses_on_the_suffix(tmp_path: Path) -> None:
    """The suffix rule is a configuration, not a dead field: a caller that
    wants the tools scoped to source files gets that, and gets a refusal that
    says which kinds are allowed."""
    workspace = Workspace(repl=cast("Any", _StubTool(tmp_path)), allowed_suffixes=(".py",))

    with pytest.raises(PathNotAllowed):
        workspace.resolve("notes.txt")
    assert "You may read and write .py files" in workspace.denial("notes.txt")


def test_a_symlink_out_of_the_workspace_is_refused_and_unlisted(tmp_path: Path) -> None:
    """A local-looking name pointed outside.

    Refusing the read is half of it: a listing that advertised the file would
    also report the TARGET's size, describing something from outside the
    directory it claims to be listing, and every later read of it would fail.
    """
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "real.py").write_text("X = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n" * 100)
    os.symlink(outside, directory / "escape.py")

    workspace = _workspace(directory)

    with pytest.raises(PathNotAllowed):
        workspace.resolve("escape.py")
    assert [entry.name for entry in workspace.entries()] == ["real.py"]


def test_the_directory_is_read_off_the_tool_every_time(tmp_path: Path) -> None:
    """A restored agent's kernel arrives with a FRESH working directory, and a
    workspace holding the old string would resolve into one nobody is using —
    silently, since both paths exist and both are writable."""
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    tool = _StubTool(first)
    workspace = Workspace(repl=cast("Any", tool))

    assert workspace.resolve("helper.py").startswith(str(first))
    tool._tmpdir = str(second)
    assert workspace.resolve("helper.py").startswith(str(second))


def test_a_refusal_never_names_a_resolved_path(tmp_path: Path) -> None:
    """What the model is told about a bad path is the path IT wrote.

    A resolved location is host-specific, absent from the model's world and
    different on every run, so leaking one into a prompt both confuses the
    model and makes the run unreproducible.
    """
    workspace = _workspace(tmp_path)
    for path in ("../escape.py", "sub/nested.py", "/etc/passwd"):
        denial = workspace.denial(path)
        assert path in denial
        assert str(tmp_path) not in denial


def test_undo_reaches_exactly_one_step(tmp_path: Path) -> None:
    """Reverting clears the whole log, so a second undo before another edit
    finds nothing — rather than walking back through work the model has
    stopped discussing."""
    workspace = _workspace(tmp_path)
    workspace.apply("helper.py", None, "X = 1\n")
    workspace.apply("helper.py", "X = 1\n", "X = 2\n")

    change = workspace.last_change()
    assert change is not None
    workspace.revert(change)

    assert workspace.read("helper.py") == "X = 1\n"
    assert workspace.last_change() is None


def test_an_undo_onto_work_a_snippet_made_is_refusable(tmp_path: Path) -> None:
    """``changed_outside`` is what lets the tool refuse instead of discarding:
    the model writes its own files from snippets, and an undo that silently
    overwrote that work would be worse than no undo at all."""
    workspace = _workspace(tmp_path)
    workspace.apply("helper.py", None, "X = 1\n")
    change = workspace.last_change()
    assert change is not None
    assert not workspace.changed_outside(change)

    workspace.write("helper.py", "X = 1\nY = 2\n")  # as a snippet would
    assert workspace.changed_outside(change)


def test_a_file_that_is_not_text_is_refused_as_such(tmp_path: Path) -> None:
    """Decoding it would put replacement characters into a prompt, and the
    ``UnicodeDecodeError`` it raises instead is not an ``OSError`` — so
    without a trap of its own it escapes every tool's handling and ends the
    game. Reached in practice by ``np.save``, whose output is binary and
    lands right beside the model's source files."""
    tmp_path.joinpath("blob.py").write_bytes(b"\xff\xfe\x00")
    workspace = _workspace(tmp_path)

    with pytest.raises(NotText) as raised:
        workspace.read("blob.py")
    assert raised.value.path == "blob.py"
    assert raised.value.size == 3

    # Listed all the same: a listing that hides what it cannot read
    # describes a directory the model does not have.
    (entry,) = workspace.entries()
    assert (entry.name, entry.lines, entry.size) == ("blob.py", None, 3)


def test_a_directory_left_behind_is_swept_and_reported(tmp_path: Path) -> None:
    """The flat contract's enforcement half. The file tools cannot make a
    directory, but a snippet can, and one that lingered would be second-class
    everywhere — invisible to the listing and to the search alike. Reported
    rather than removed quietly: deleting what the model wrote without saying
    so is worse than the directory was."""
    directory = tmp_path / "work"
    directory.mkdir()
    (directory / "keep.py").write_text("X = 1\n")
    (directory / "scratch").mkdir()
    (directory / "scratch" / "inner.py").write_text("Y = 2\n")

    workspace = _workspace(directory)

    assert workspace.sweep() == ["scratch/"]
    assert [entry.name for entry in workspace.entries()] == ["keep.py"]
