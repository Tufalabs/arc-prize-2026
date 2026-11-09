"""The second half of the file tools: importing what the model wrote.

A file in the kernel's working directory is a module, so a candidate kept in
``candidate.py`` is one ``import`` away rather than retyped into every
snippet. For that to be true rather than nearly true, an edited file has to
be what the next import sees, a file that does not run must not cost the
version that did, a reset must not take the files with it, and a file named
after a module Python already has must not stand in for it. What runs before
every cell keeps the answer channel and the working directory where the model
left them only by accident: ``check`` and the cwd come back, with a word.

The flag off, none of it runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from duck.config import CheckMode
from duck.python_tool import DuckPythonTool

import taaf.game
import taaf.grid


def _grid(rows: list[list[int]]) -> taaf.grid.Grid:
    return taaf.grid.Grid(np.array(rows, dtype=np.int8))


_VIEW = taaf.game.GameView(game_id="files", given=((_grid([[1, 2]]), _grid([[2, 1]])),), questions=(_grid([[3, 4]]),))


class _Kernel:
    """A started duck kernel and the directory its files live in."""

    def __init__(self, tool: DuckPythonTool) -> None:
        self.tool = tool

    @property
    def directory(self) -> Path:
        assert self.tool._tmpdir is not None
        return Path(self.tool._tmpdir)

    async def run(self, code: str) -> str:
        """One snippet, as the model's tool call runs it."""
        return (await self.tool._execute({"code": code})).to_string()


async def _started(*, file_tools: bool = True) -> _Kernel:
    tool = DuckPythonTool(timeout=60.0, check_mode=CheckMode.ENABLED, file_tools=file_tools)
    await tool.setup_task(_VIEW)
    return _Kernel(tool)


@pytest.fixture
async def kernel() -> AsyncIterator[_Kernel]:
    started = await _started()
    try:
        yield started
    finally:
        await started.tool.clear_state()


async def test_the_candidate_imports_checks_and_answers(kernel: _Kernel) -> None:
    kernel.directory.joinpath("candidate.py").write_text("def fn(grid):\n    return grid[:, ::-1]\n")
    out = await kernel.run(
        "import candidate\ncheck(candidate.fn)\nfor i, grid in enumerate(test_in):\n"
        "    test_out[i][0] = candidate.fn(grid)\nprint(test_out[0][0].tolist())\n"
    )
    assert "[[4, 3]]" in out, out


async def test_an_edited_file_is_what_the_next_import_sees(kernel: _Kernel) -> None:
    """Silent otherwise: the model edits a file, imports it again, gets the
    cached module and reasons at length about code it has already replaced."""
    kernel.directory.joinpath("candidate.py").write_text("AREA = 64\n")
    assert "64" in await kernel.run("import candidate\nprint(candidate.AREA)\n")
    kernel.directory.joinpath("candidate.py").write_text("AREA = 100\n")
    assert "100" in await kernel.run("import candidate\nprint(candidate.AREA)\n")


async def test_a_name_bound_in_an_earlier_call_keeps_its_version_until_the_next_import(kernel: _Kernel) -> None:
    """Each import of an edited file makes a new module, which is what the
    prompt says: the example imports the module in the call that uses it."""
    kernel.directory.joinpath("candidate.py").write_text("VALUE = 1\n")
    await kernel.run("import candidate as held\nfrom candidate import VALUE\n")
    kernel.directory.joinpath("candidate.py").write_text("VALUE = 2\n")
    out = await kernel.run("import candidate\nprint(held.VALUE, VALUE, candidate.VALUE)\n")
    assert "1 1 2" in out, out


async def test_reloading_a_held_module_picks_up_the_edit(kernel: _Kernel) -> None:
    """``reload`` looks in ``sys.modules``, which the refresh before every
    cell empties of the model's modules, so the kernel's ``reload`` puts the
    module back first; the name held sees the edit in the same object."""
    kernel.directory.joinpath("candidate.py").write_text("AREA = 64\n")
    await kernel.run("import candidate\nheld = candidate\n")
    kernel.directory.joinpath("candidate.py").write_text("AREA = 144\n")
    out = await kernel.run("import importlib\nimportlib.reload(held)\nprint(held.AREA)\n")
    assert "144" in out, out


async def test_a_file_that_raises_leaves_the_version_that_ran(kernel: _Kernel) -> None:
    kernel.directory.joinpath("candidate.py").write_text("VALUE = 1\n")
    await kernel.run("import candidate\n")
    kernel.directory.joinpath("candidate.py").write_text("VALUE = 2\nraise ValueError('half typed')\n")
    out = await kernel.run(
        "try:\n    import candidate as again\nexcept ValueError as error:\n    print('raised', error)\n"
        "print('kept', candidate.VALUE)\n"
    )
    assert "raised half typed" in out and "kept 1" in out, out


async def test_a_module_starts_with_the_names_a_snippet_does(kernel: _Kernel) -> None:
    kernel.directory.joinpath("candidate.py").write_text(
        "FIRST = train_in[0].tolist()\ndef fn(grid):\n    return np.fliplr(grid), Counter([1, 1])\n"
    )
    out = await kernel.run(
        "import candidate\nprint(candidate.FIRST, candidate.fn(test_in[0]), callable(candidate.check))\n"
    )
    assert "[[1, 2]]" in out and "Counter({1: 2})" in out and out.rstrip().endswith("True"), out


async def test_a_traceback_out_of_a_module_shows_the_models_frames_only(kernel: _Kernel) -> None:
    """The loader only makes the module, so none of its frames sits between
    the import system's and the model's file, and the import system hides
    its own."""
    kernel.directory.joinpath("candidate.py").write_text("def fn(grid):\n    raise ValueError('boom')\n")
    out = await kernel.run("import candidate\ncandidate.fn(test_in[0])\n")
    assert 'File "candidate.py", line 2, in fn' in out, out
    assert "kernel_workspace" not in out and "importlib" not in out, out
    kernel.directory.joinpath("candidate.py").write_text("x = 1 +\n")
    out = await kernel.run("import candidate\n")
    assert "candidate.py" in out and "SyntaxError" in out and "kernel_workspace" not in out, out


async def test_a_module_does_not_name_this_machine(kernel: _Kernel) -> None:
    kernel.directory.joinpath("candidate.py").write_text("X = 1\n")
    out = await kernel.run("import candidate\nprint(candidate)\n")
    assert "<module 'candidate' from 'candidate.py'>" in out, out


async def test_files_outlive_the_kernel_they_were_written_from(kernel: _Kernel) -> None:
    """A kernel that dies takes the session with it, not the work: the
    candidate in its file is still there for the restarted kernel."""
    kernel.directory.joinpath("candidate.py").write_text("AREA = 64\n")
    await kernel.run("carried = 1\nimport candidate\n")
    kernel.tool.reset()
    assert "NameError" in await kernel.run("print(carried)\n")
    assert "64" in await kernel.run("import candidate\nprint(candidate.AREA)\n")


async def test_a_file_named_like_one_of_pythons_modules_does_not_stand_in_for_it(kernel: _Kernel) -> None:
    for name in ("inspect", "wave", "boardscan"):
        kernel.directory.joinpath(f"{name}.py").write_text("MINE = True\n")
    kernel.tool.reset()
    out = await kernel.run(
        "import inspect, wave, boardscan\n"
        "print(hasattr(inspect, 'MINE'), hasattr(wave, 'MINE'), hasattr(wave, 'open'), boardscan.MINE)\n"
    )
    assert "False False True True" in out, out


async def test_check_comes_back_after_a_cell_replaced_it(kernel: _Kernel) -> None:
    """The answer channel lost to a stray ``def check`` or ``import check``
    is lost for that cell alone."""
    assert "fake" in await kernel.run("def check(fn):\n    return 'fake'\nprint(check(None))\n")
    out = await kernel.run("print(check.__name__, check(lambda g: g[:, ::-1]).ok)\n")
    assert "`check` had been replaced or deleted" in out and "check True" in out, out
    out = await kernel.run("print(1)\n")
    assert "had been replaced" not in out, "said once, when it happened"


async def test_the_models_own_show_and_ascii_grid_stay_its_own(kernel: _Kernel) -> None:
    """A model writes helpers under these names on purpose; only the answer
    channel is put back."""
    await kernel.run("def show(grid, label=''):\n    print('mine', label)\ndef ascii_grid(arr):\n    return 'mine too'")
    out = await kernel.run("show(test_in[0], 'x')\nprint(ascii_grid(test_in[0]))")
    assert "mine x" in out and "mine too" in out and "had been replaced" not in out, out


async def test_the_cwd_comes_back_after_a_cell_left_it(kernel: _Kernel) -> None:
    kernel.directory.joinpath("candidate.py").write_text("X = 1\n")
    await kernel.run("import os\nos.chdir('/')\n")
    out = await kernel.run("import os, candidate\nprint(sorted(os.listdir('.')), candidate.X)\n")
    assert "cwd had drifted away" in out and "['candidate.py'] 1" in out, out


async def test_a_directory_a_cell_makes_is_swept_and_said_so(kernel: _Kernel) -> None:
    out = await kernel.run("import os\nos.mkdir('sub')\n")
    assert "removed directory 'sub/'" in out and not kernel.directory.joinpath("sub").exists(), out


async def test_the_session_listing_names_the_models_work_not_the_harness(kernel: _Kernel) -> None:
    kernel.directory.joinpath("shapes.py").write_text("class Box:\n    pass\n")
    await kernel.run(
        "from shapes import Box\nclass Local:\n    pass\ndef solve(g):\n    return g\n"
        "seen = {1: 2}\nscratch = np.zeros((2, 3), dtype=np.int8)\nnothing = None\nalias = int\n"
    )
    listing = await kernel.tool.session_objects()
    assert listing == [
        ("Box", "class", False),
        ("Local", "class", False),
        ("solve", "function", True),
        ("seen", "dict, 1 entry", False),
        ("scratch", "ndarray int8 (2, 3)", False),
    ]


async def test_the_session_listing_leaves_out_the_harness_without_the_flag_too() -> None:
    """The listing closes a summary with the file tools off as well, as on the
    shipped second take, so the setup records the names it bound either way."""
    kernel = await _started(file_tools=False)
    try:
        await kernel.run("def solve(g):\n    return g\n")
        assert await kernel.tool.session_objects() == [("solve", "function", True)]
    finally:
        await kernel.tool.clear_state()


async def test_without_the_flag_nothing_is_restored_or_swept() -> None:
    kernel = await _started(file_tools=False)
    try:
        await kernel.run("def check(fn):\n    return 'fake'\n")
        out = await kernel.run("import os\nos.mkdir('sub')\nprint(check(None))\n")
        assert "fake" in out and "removed directory" not in out, out
        assert kernel.directory.joinpath("sub").is_dir()
    finally:
        await kernel.tool.clear_state()
