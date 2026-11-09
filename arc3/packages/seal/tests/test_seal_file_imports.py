"""The second productive path: importing what the model wrote.

The file tools are half of the surface. The other half is that a file in the
working directory is a MODULE — the kernel runs there, so a helper the model
wrote is one ``import`` away, and the analysis it spent a turn building is
reusable instead of retyped.

Two things have to hold for that to be true rather than nearly true: an
edited file has to be what the next import sees, and a reset must not take
the files with it. A file named after a module Python already has is the
exception: that name keeps meaning Python's module.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Literal

import numpy as np

from seal.repl import GameREPL


def _state() -> dict[str, Any]:
    """A minimal runtime-state payload: these rows only care about imports."""
    return {"level": 1, "turns": [], "available_actions": []}


class _Kernel:
    """A persistent seal kernel and the directory its files live in."""

    def __init__(self, use_world_model: Literal["none", "mandatory"] = "none") -> None:
        self.tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model=use_world_model)

        # No row here executes an action; the acting host call is present
        # only because the prelude requires one to exist.
        def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
            return {"ack": {"queued": len(normalized)}}

        self.tool.host_functions = {
            "get_state": _state,
            "action": _action,
        }

    @property
    def directory(self) -> Path:
        """Where a file has to be written to be importable."""
        assert self.tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        return Path(self.tool._tmpdir)  # pyright: ignore[reportPrivateUsage] — same

    async def run(self, code: str) -> str:
        """One snippet, as the model's tool call would run it."""
        content = await self.tool._execute({"code": code})
        return content.to_string()

    async def close(self) -> None:
        """Drop the kernel and its directory."""
        await self.tool.clear_state()


async def test_a_file_the_model_wrote_is_importable() -> None:
    """The whole point of the surface: work written to a file comes back as a
    module instead of being retyped into the next snippet."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        out = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in out, out
    finally:
        await kernel.close()


async def test_an_edited_file_is_what_the_next_import_sees() -> None:
    """The failure this closes is silent: without eviction the model edits a
    file, imports it again, gets the cached module and reasons at length about
    code it has already replaced."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        first = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in first, first

        kernel.directory.joinpath("helper.py").write_text("AREA = 100\n")
        second = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "100" in second, second
    finally:
        await kernel.close()


async def test_reloading_a_held_module_picks_up_the_edit() -> None:
    """Edit the file, then ``importlib.reload`` the module an earlier cell
    imported: the usual way to pick up an edit. The per-cell refresh drops the
    model's modules from ``sys.modules``, which is where ``reload`` looks, so
    the kernel's ``reload`` puts the module back first. The name the model
    holds sees the edit, in the same object."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        first = await kernel.run("import helper\nheld = helper\nprint(helper.AREA)\n")
        assert "64" in first, first
        kernel.directory.joinpath("helper.py").write_text("AREA = 144\n")
        second = await kernel.run("import importlib\nimportlib.reload(helper)\nprint(helper.AREA, helper is held)\n")
        assert "144 True" in second, second
    finally:
        await kernel.close()


async def test_reloading_a_held_module_after_another_module_imported_it() -> None:
    """A module the model wrote imports another one; reloading the first
    installs a fresh copy of the second under its name, and reloading the copy
    an earlier cell still holds must still work."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("rules.py").write_text("STEP = 1\n")
        kernel.directory.joinpath("planner.py").write_text("import rules\n")
        first = await kernel.run("import rules\nheld = rules\nprint(rules.STEP)\n")
        assert "1" in first, first
        kernel.directory.joinpath("rules.py").write_text("STEP = 2\n")
        second = await kernel.run(
            "import importlib, planner\nimportlib.reload(planner)\nimportlib.reload(rules)\nprint(rules.STEP, rules is held)\n"
        )
        assert "2 True" in second, second
    finally:
        await kernel.close()


async def test_files_outlive_the_kernel_they_were_written_from() -> None:
    """A reset drops definitions, not work. Every persistence mode resets the
    kernel somewhere, and one that also wiped the directory would delete the
    model's files without ever saying so."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        await kernel.run("carried = 1\nimport helper\n")
        kernel.tool.reset()
        out = await kernel.run("print(carried)\n")
        assert "NameError" in out, f"the reset must drop kernel state; got {out[:300]}"
        out = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in out, out
    finally:
        await kernel.close()


async def test_a_module_the_model_wrote_follows_an_action() -> None:
    """A module starts with ``history`` and the cell's tools, and ``history``
    moves when the module itself acts, so the line after ``submit_action``
    reads the new board."""
    import base64

    grids = [b"\x00" * 4, b"\x05" * 4]
    frames = [
        {"grid_b64": base64.b64encode(g).decode(), "shape": [2, 2], "step": i, "level": 1} for i, g in enumerate(grids)
    ]
    turns = [{"action": "", "outcome": "playing", "animation": [frames[0]]}]

    def _state() -> dict[str, Any]:
        return {"level": 1, "turns": list(turns), "available_actions": ["UP"]}

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        turns.append({"action": normalized[0]["action"], "outcome": "playing", "animation": [frames[1]]})
        return {"result": {"executed": True}, "state": _state()}

    kernel = _Kernel()
    kernel.tool.host_functions = {"get_state": _state, "action": _action}
    try:
        kernel.directory.joinpath("helper.py").write_text(
            "AT_IMPORT = int(history[-1].frame.grid[0, 0])\n"
            "def act_and_read():\n"
            "    before = int(history[-1].frame.grid[0, 0])\n"
            "    submit_action(['UP'])\n"
            "    return before, int(history[-1].frame.grid[0, 0]), len(history), len(segment(history[-1].frame.grid))\n"
        )
        out = await kernel.run(
            "import helper\nprint(helper.AT_IMPORT, helper.act_and_read(), int(history[-1].frame.grid[0, 0]))\n"
        )
        assert "0 (0, 5, 2, 1) 5" in out, out
    finally:
        await kernel.close()


async def test_a_module_the_model_wrote_does_not_name_this_machine() -> None:
    """Its repr and its traceback frames name the file; the directory it sits
    in holds the job id and the checkout name, which the model has no use
    for and which no other machine has."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        out = await kernel.run("import helper\nprint(repr(helper))\n")
        assert out == "<module 'helper' from 'helper.py'>", out
    finally:
        await kernel.close()


async def test_the_world_model_is_bound_in_every_cell() -> None:
    """``world_model`` is a name the model never has to import: the cell
    prelude loads the file from disk, so its module-level code re-runs over
    the history so far and the edit made in the previous cell is what this
    one calls."""
    kernel = _Kernel(use_world_model="mandatory")
    try:
        kernel.directory.joinpath("world_model.py").write_text("MARK = 'first'\nSEEN = len(history)\n")
        out = await kernel.run("print(world_model.MARK, world_model.SEEN)\nwm = world_model\n")
        assert "first 0" in out, out

        kernel.directory.joinpath("world_model.py").write_text("MARK = 'second'\n")
        out = await kernel.run("print(world_model.MARK, wm.MARK, wm is world_model)")
        assert "second second True" in out, out
    finally:
        await kernel.close()


async def test_a_world_model_that_does_not_load_says_so() -> None:
    """A file that raises on import would otherwise leave the model calling
    the previous version without knowing it — the cell still runs, and it
    says which version it is running."""
    kernel = _Kernel(use_world_model="mandatory")
    try:
        kernel.directory.joinpath("world_model.py").write_text("MARK = 'first'\n")
        await kernel.run("pass")

        kernel.directory.joinpath("world_model.py").write_text("MARK = 'second'\nraise ValueError('half typed')\n")
        out = await kernel.run("print(world_model.MARK)\nimport world_model as wm\nprint('imported', wm.MARK)\n")

        assert "half typed" in out and "NOT reflected" in out, out
        assert "first\nimported first" in out, out
    finally:
        await kernel.close()


def _frame(fill: int, step: int) -> dict[str, Any]:
    """One settled board of a single colour, as the host sends it."""
    grid = np.full((64, 64), fill, dtype=np.uint8)
    return {"grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"), "shape": [64, 64], "step": step, "level": 1}


async def test_a_module_follows_an_action_after_a_world_model_check() -> None:
    """A world-model check loads world_model.py in the middle of a cell. The
    modules the cell imported before it are still the ones refreshed after
    the next action, so a helper that reads the board after a move reads the
    new one."""
    turns: list[dict[str, Any]] = [{"action": "", "outcome": "playing", "animation": [_frame(0, 0)]}]

    def get_state() -> dict[str, Any]:
        return {"level": 1, "turns": list(turns), "available_actions": ["UP"]}

    def explore(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        turns.append({"action": "UP", "outcome": "playing", "animation": [_frame(7, len(turns))]})
        return {"result": {"executed": True}, "state": get_state()}

    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="mandatory")
    tool.host_functions = {"get_state": get_state, "action": explore, "action_exploring": explore}
    try:
        await tool._execute({"code": "pass"})
        directory = Path(tool._tmpdir or "")  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        directory.joinpath("world_model.py").write_text("VERSION = 1\n")
        directory.joinpath("helper.py").write_text("def fill():\n    return int(history[-1].frame.grid[0, 0])\n")
        code = (
            "import helper\n"
            "print('before', helper.fill())\n"
            "test_world_model()\n"
            "submit_action([Action('UP')])\n"
            "print('after', helper.fill())\n"
        )
        out = (await tool._execute({"code": code})).to_string()
    finally:
        await tool.clear_state()

    assert "before 0" in out and "after 7" in out, out


async def test_a_name_bound_in_an_earlier_call_follows_the_file() -> None:
    """A file runs into the same module every time, so an alias kept from an
    earlier call is the module the next import gives, at the latest version
    — not a second copy of an older one."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("VALUE = 1\n")
        await kernel.run("import helper as h\n")
        kernel.directory.joinpath("helper.py").write_text("VALUE = 2\n")
        out = await kernel.run("import helper\nprint(h is helper, h.VALUE)\n")
        assert "True 2" in out, out
    finally:
        await kernel.close()


async def test_a_file_that_raises_keeps_its_previous_version() -> None:
    """A half-typed edit raises where it is imported, and the module goes on
    holding the version that last ran whole, not what the broken file got
    through before it raised."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("VALUE = 1\n")
        await kernel.run("import helper as h\n")
        kernel.directory.joinpath("helper.py").write_text("VALUE = 2\nraise ValueError('half typed')\n")
        out = await kernel.run(
            "try:\n    import helper\nexcept ValueError as error:\n    print('raised', error)\nprint('kept', h.VALUE)\n"
        )
        assert "raised half typed" in out and "kept 1" in out, out
    finally:
        await kernel.close()


async def test_every_piece_of_the_models_code_starts_with_the_same_names() -> None:
    """A snippet, a module the model writes and world_model.py start with one
    set of names; world_model.py goes without the ones that act, because the
    replay runs it too."""
    names = (
        "np",
        "json",
        "Counter",
        "Action",
        "Unknown",
        "history",
        "available_actions",
        "submit_action",
        "test_world_model",
        "world_model",
    )
    probe = f"SEEN = [name for name in {names!r} if name in globals()]\n"
    kernel = _Kernel(use_world_model="mandatory")
    try:
        kernel.directory.joinpath("world_model.py").write_text(probe)
        kernel.directory.joinpath("helper.py").write_text(probe)
        out = await kernel.run(
            probe
            + "import helper\nprint('snippet', SEEN)\nprint('helper', helper.SEEN)\nprint('world_model', world_model.SEEN)\n"
        )
    finally:
        await kernel.close()

    everything = list(names)
    assert f"snippet {everything}" in out and f"helper {everything}" in out, out
    replayed = [name for name in names if name not in ("submit_action", "test_world_model", "world_model")]
    assert f"world_model {replayed}" in out, out


async def test_an_edited_module_counts_in_the_same_calls_world_model_check() -> None:
    """world_model.py imports a module the cell has already imported; the
    cell edits that module, and the check it runs next sees the edit."""
    kernel = _Kernel(use_world_model="mandatory")
    try:
        kernel.directory.joinpath("rules.py").write_text("STEP = 1\n")
        kernel.directory.joinpath("world_model.py").write_text(
            "from rules import STEP\nassert STEP == 1, 'STEP is ' + str(STEP)\n"
        )
        out = await kernel.run("import rules\nopen('rules.py', 'w').write('STEP = 2\\n')\nprint(test_world_model())\n")
        assert "STEP is 2" in out, out
    finally:
        await kernel.close()


async def test_a_module_the_model_wrote_is_read_only_after_it_loads() -> None:
    """Nothing a cell stores in a module reaches a test that loads the files
    afresh, so storing it fails at once: module-level data is frozen, and the
    module refuses an assignment from outside. A new call's import runs the
    file again and the data is whole. Without a world model, the refusal
    says so in terms that game has: no ``do_move``, ``render`` or ``state``."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("table.py").write_text("SEEN = {}\nLIMIT = 3\n")
        out = await kernel.run("import table\ntable.SEEN['a'] = 1\n")
        assert "read-only" in out and "the next `python` call that imports it" in out, out
        assert "do_move" not in out and "`state`" not in out, out
        out = await kernel.run("import table\ntable.LIMIT = 4\n")
        assert "module 'table' is read-only after it loads" in out, out
        out = await kernel.run("import table\nprint(table.SEEN, table.LIMIT)\n")
        assert "{} 3" in out, out
    finally:
        await kernel.close()


async def test_a_world_model_that_does_not_load_says_so_once() -> None:
    """The warning at the top of a call is said once per failure: the same
    traceback again is not repeated, and a different one is."""
    kernel = _Kernel(use_world_model="mandatory")
    try:
        kernel.directory.joinpath("world_model.py").write_text("raise ValueError('first')\n")
        first = await kernel.run("print('one')\n")
        again = await kernel.run("print('two')\n")
        kernel.directory.joinpath("world_model.py").write_text("raise ValueError('second')\n")
        changed = await kernel.run("print('three')\n")
    finally:
        await kernel.close()
    assert "world_model could not be (re)loaded" in first and "ValueError: first" in first, first
    assert "could not be (re)loaded" not in again, again
    assert "ValueError: second" in changed, changed


async def test_a_cell_sees_the_surface_not_the_harness_plumbing() -> None:
    """The harness reaches its store through the import system, not a name, so
    a snippet listing its globals finds none of it, and shadowing a name the
    harness once used breaks nothing in the next call."""
    kernel = _Kernel(use_world_model="mandatory")
    try:
        out = await kernel.run("print(sorted(n for n in ('runtime', '_sg', '_rs') if n in globals()))\n_rs = 0\n")
        after = await kernel.run("print(len(history), callable(submit_action))\n")
    finally:
        await kernel.close()
    assert "[]" in out, out
    assert "0 True" in after or "1 True" in after, after


async def test_a_file_named_like_one_of_pythons_modules_does_not_stand_in_for_it() -> None:
    """A model file named after a module Python already has — the standard
    library's or an installed package's — is never what ``import`` reaches:
    not when a fresh kernel starts in its directory, where the harness's own
    imports would otherwise pick it up, and not from a cell. The model's
    other modules import as before."""
    kernel = _Kernel()
    try:
        await kernel.run("pass")
        for name in ("inspect", "wave", "boardscan"):
            kernel.directory.joinpath(f"{name}.py").write_text("MINE = True\n")
        # A fresh kernel in the same directory, as at a new level.
        kernel.tool.reset(keep_tmpdir_contents=True)
        out = await kernel.run(
            "import inspect, wave, boardscan\n"
            "print(hasattr(inspect, 'MINE'), hasattr(wave, 'MINE'), hasattr(wave, 'open'), boardscan.MINE)\n"
        )
        assert "False False True True" in out, out
    finally:
        await kernel.close()
