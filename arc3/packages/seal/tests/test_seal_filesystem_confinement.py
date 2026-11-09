"""The kernel's files, confined by the operating system (``seal.landlock``).

The source guard patches Python, so these rows reach for files the ways that
walk around it — ``os.open``, a shell-out — which are the ways the long runs'
models read the games' code. Each runs in a real kernel: a confinement
cannot be lifted, so it cannot be tried in the test's own process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import seal.harness.game
from seal.repl import GameREPL


def _state() -> dict[str, Any]:
    """A minimal runtime-state payload: these rows only touch files."""
    return {"level": 1, "turns": [], "available_actions": []}


def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
    """Never called: no row acts, but the prelude needs the host call to exist."""
    return {"ack": {"queued": len(normalized)}}


def _a_game_file() -> Path:
    """One file of the games' own package."""
    spec = importlib.util.find_spec("re_arc")
    assert spec is not None and spec.submodule_search_locations
    root = Path(next(iter(spec.submodule_search_locations)))
    return next(path for path in sorted(root.rglob("*.py")) if path.stat().st_size)


async def _run(tool: GameREPL, code: str) -> str:
    """One snippet, as the model's tool call would run it."""
    return (await tool._execute({"code": code})).to_string()  # pyright: ignore[reportPrivateUsage] — the tool call itself


def _reader(path: Path) -> str:
    """A cell reading ``path`` past the source guard, two ways."""
    return (
        "import os, subprocess\n"
        "try:\n"
        f"    os.read(os.open({str(path)!r}, os.O_RDONLY), 64)\n"
        "    print('os.open: read')\n"
        "except PermissionError:\n"
        "    print('os.open: refused')\n"
        f"cat = subprocess.run(['cat', {str(path)!r}], capture_output=True)\n"
        "print('cat:', 'read' if cat.returncode == 0 else 'refused')\n"
    )


async def test_the_games_our_code_and_other_directories_cannot_be_reached() -> None:
    """The games and our source, though installed or kept beside what the
    model may read, and a directory beside the working directory — another
    game's — can neither be read nor changed, however the model's code
    reaches for them; its own directory, its temporary files, an import of a
    library no cell has used yet and a scan of the installed distributions
    work as before, without the source guard's notice."""
    tool = GameREPL(timeout=60.0, reset_before_each_call=False, use_world_model="none")
    tool.host_functions = {"get_state": _state, "action": _action}
    try:
        await _run(tool, "pass")
        directory = Path(tool._tmpdir or "")  # pyright: ignore[reportPrivateUsage] — the test writes beside where it runs
        beside = directory.parent / f"{directory.name}_beside"
        beside.mkdir()
        (beside / "solver.py").write_text("another game's\n")

        games = await _run(tool, _reader(_a_game_file()))
        ours = await _run(tool, _reader(Path(seal.harness.game.__file__)))
        other = await _run(tool, _reader(beside / "solver.py"))
        await _run(tool, f"import subprocess\nsubprocess.run(['rm', '-rf', {str(beside)!r}])\n")
        own = await _run(
            tool,
            "import subprocess, tempfile\n"
            "subprocess.run('echo mine > notes.txt', shell=True)\n"
            "print('notes:', open('notes.txt').read().strip())\n"
            "with tempfile.NamedTemporaryFile('w+') as f:\n"
            "    f.write('scratch'); f.flush(); f.seek(0); print('temp:', f.read())\n"
            "import scipy.signal\n"
            "print('scipy:', scipy.signal.__name__)\n"
            # Lists every sys.path entry, our source directories among them.
            "import importlib.metadata\n"
            "print('distributions:', len(list(importlib.metadata.distributions())) > 0)\n",
        )
    finally:
        await tool.clear_state()

    for out in (games, ours, other):
        assert "os.open: refused" in out and "cat: refused" in out, out
    assert (beside / "solver.py").read_text() == "another game's\n"
    assert "notes: mine" in own and "temp: scratch" in own and "scipy: scipy.signal" in own, own
    assert "distributions: True" in own and "not available" not in own, own


async def test_off_the_kernel_is_not_confined() -> None:
    """``confine_filesystem`` off is the kernel as it was: the source guard
    alone, which ``os.open`` walks around."""
    tool = GameREPL(timeout=60.0, reset_before_each_call=False, use_world_model="none", confine_filesystem=False)
    tool.host_functions = {"get_state": _state, "action": _action}
    try:
        out = await _run(tool, _reader(_a_game_file()))
    finally:
        await tool.clear_state()

    assert "os.open: read" in out and "cat: read" in out, out
