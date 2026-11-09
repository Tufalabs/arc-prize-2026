"""The two places the search chain ends the game instead of reporting.

The chain runs the harness three times over one model — the REPL's test, a
test in the search's own process, and the search — and replays a found path
in the REPL. When two of those disagree, the seal is wrong rather than the model, and
a report built on either side would tell the model something false. So the
game ends, loudly. The scripted reference cannot reach these (a model does not
make two harnesses disagree), so they are pinned here directly.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seal.palette import AsciiLayout
from seal.planner import NoPlan, PlanDiagnostics
from seal.repl import GameREPL
from seal.search import ModelJob, Provenance, SearchHarnessError, SearchReport, make_planner, search_level
from seal.search_process import SearchResult, run_search
from seal.state import FramePayload, TimelineEntry

#: A one-colour board that never changes: passes on any history of such boards.
STILL_MODEL = """\
import numpy as np


def initial_state():
    return {'outcome': 'playing', 'energy': 2000}


def do_move(state, action):
    state['energy'] -= 1


def render(state):
    return np.full((8, 8), 3, dtype=np.int8)
"""


def _frame(fill: int) -> FramePayload:
    """An 8x8 board of one colour, as the host records it."""
    grid = np.full((8, 8), fill, dtype=np.uint8)
    return FramePayload(grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"), shape=[8, 8], step=0, level=1)


def _turn(fill: int) -> dict[str, Any]:
    """The seed turn of a level on that board, as the kernel receives it."""
    return {"action": "", "outcome": "playing", "animation": [_frame(fill).payload()]}


def test_the_retest_after_a_call_describes_no_frame() -> None:
    """The search tests the model again after the call. That test is the
    harness's, not the model's: a frame delta from it would reach the model as
    if it had asked, a call late — and a block that failed to reproduce its
    frame would end the game over a test nobody ran."""
    from seal.harness import _repl_state as store

    seen: list[bool] = []

    class _Passed:
        """A TestResult that passed."""

        def __bool__(self) -> bool:
            """Falsy, as a passing result is."""
            return False

    def test_world_model() -> _Passed:
        """Stand in for the REPL's own, recording whether the settled delta was on."""
        seen.append(store.settled_delta)
        return _Passed()

    previous, store.settled_delta = store.settled_delta, True
    store.wm_gate.explicit_test_passed = True
    store.current_test_world_model = test_world_model
    try:
        assert store.search_retest() == "pass"
    finally:
        store.settled_delta = previous
        store.current_test_world_model = None
    assert seen == [False], "the retest ran with the settled delta on"
    assert store.settled_delta == previous, "the settled delta was not restored"


async def test_a_model_the_host_process_fails_ends_the_game(tmp_path: Path) -> None:
    """Reached only when the REPL passed the same files; here the search
    process's test is handed a history the model does not draw, which is how
    it would look."""
    (tmp_path / "world_model.py").write_text(STILL_MODEL)
    entry = TimelineEntry(action=None, frame=_frame(5), animation=[_frame(5)], level=1)
    with pytest.raises(SearchHarnessError, match="the search's process's test of the same files failed"):
        await search_level(
            directory=str(tmp_path),
            entries=[entry],
            turns=[_turn(5)],
            model_actions=["UP"],
            game_actions=["UP"],
            level=1,
            seconds=5.0,
            ascii_layout=AsciiLayout(),
            memory_bytes=2 * 2**30,
        )


def _search(directory: Path, *, seconds: float = 30.0, memory_bytes: int = 2 * 2**30) -> Any:
    """``search_level`` over ``directory``'s model on a one-turn history of
    ``STILL_MODEL``'s board."""
    entry = TimelineEntry(action=None, frame=_frame(3), animation=[_frame(3)], level=1)
    return search_level(
        directory=str(directory),
        entries=[entry],
        turns=[_turn(3)],
        model_actions=["UP"],
        game_actions=["UP"],
        level=1,
        seconds=seconds,
        ascii_layout=AsciiLayout(),
        memory_bytes=memory_bytes,
    )


async def test_a_model_slow_to_start_never_holds_the_games_process(tmp_path: Path) -> None:
    """The model's code runs only in the search's process, so a model that
    takes its time to load and test holds that process alone, and the game
    can stop waiting for it at any moment."""
    slow = STILL_MODEL.replace("def initial_state():\n", "def initial_state():\n    __import__('time').sleep(30)\n")
    (tmp_path / "world_model.py").write_text(slow)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_search(tmp_path, seconds=60.0), timeout=2.0)
    assert time.monotonic() - started < 10.0


async def test_a_model_too_large_to_load_fails_the_search_not_the_game(tmp_path: Path) -> None:
    """The search's memory cap is on before the model's code first runs: one
    that allocates far past it as it loads takes the search's process down,
    and the model is told the search failed. Bounded all the same, so a
    regression that loads the model outside the cap fails this test rather
    than the machine."""
    ballast = "\n_ballast = [bytearray(64 * 2**20) for _ in range(32)]\n"
    (tmp_path / "world_model.py").write_text(STILL_MODEL + ballast)
    report = await _search(tmp_path, memory_bytes=256 * 2**20)
    assert report.result.outcome in ("error", "died"), report.result.describe()
    assert "MemoryError" in (report.result.error or "") or report.result.outcome == "died"
    assert "not complete" in report.text()


async def test_running_out_of_memory_in_the_search_test_fails_the_search_not_the_game(tmp_path: Path) -> None:
    """Under the search's cap a ``MemoryError`` is its budget, even where the
    test catches it, as it catches whatever the model's code raises: the model
    is told the search failed, and the game plays on."""
    raising = STILL_MODEL.replace("def initial_state():\n", "def initial_state():\n    raise MemoryError\n")
    (tmp_path / "world_model.py").write_text(raising)
    report = await _search(tmp_path)
    assert report.result.outcome == "error", report.result.describe()
    assert "MemoryError" in (report.result.error or "")


async def test_a_killed_or_cancelled_search_leaves_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search killed at its deadline, or abandoned by the game, runs no
    cleanup of its own: the copy of the model's files it made, and the logs
    it wrote, go with it all the same."""
    work, temp = tmp_path / "work", tmp_path / "tmp"
    work.mkdir()
    temp.mkdir()
    # The temp directory this process and the search's both use, so a file
    # either leaves behind lands where the test looks.
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.setenv("TMPDIR", str(temp))
    hanging = STILL_MODEL.replace("    state['energy'] -= 1\n", "    __import__('time').sleep(60)\n")
    (work / "world_model.py").write_text(hanging)
    entry = TimelineEntry(action=None, frame=_frame(3), animation=[_frame(3)], level=1)
    job = ModelJob(
        directory=str(work),
        entries=[entry],
        turns=[_turn(3)],
        model_actions=["UP"],
        ascii_layout=AsciiLayout(),
        matching={},
    )
    result = await run_search(make_planner(60.0, 2 * 2**30), job, ["ACTION1"], seed=0, hard_timeout=3.0)
    assert result.outcome == "timeout", result.describe()
    assert list(temp.iterdir()) == []
    with pytest.raises(TimeoutError):
        search = run_search(make_planner(60.0, 2 * 2**30), job, ["ACTION1"], seed=0, hard_timeout=60.0)
        await asyncio.wait_for(search, timeout=3.0)
    assert list(temp.iterdir()) == []


async def test_a_path_the_repl_does_not_reproduce_is_refused() -> None:
    """The REPL steps the path through its own model and compares every board
    with the search's; a different one raises instead of binding the names. So
    does a path searched on files other than the ones the REPL has."""
    from seal.harness import test as harness_test

    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="mandatory")
    state: dict[str, Any] = {"level": 1, "turns": [_turn(3)], "available_actions": ["UP"]}

    def get_state() -> dict[str, Any]:
        return state

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool.host_functions = {"get_state": get_state, "action": action}
    try:
        assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        Path(tool._tmpdir, "world_model.py").write_text(STILL_MODEL)  # pyright: ignore[reportPrivateUsage]
        content = await tool._execute({"code": "print(test_world_model())"})  # pyright: ignore[reportPrivateUsage]
        assert "ALL PASSED" in content.to_string()

        digests = harness_test.file_digests(harness_test.working_files(tool._tmpdir))  # pyright: ignore[reportPrivateUsage]

        def install(fill: int, files: dict[str, str] = digests) -> str:
            payload = {
                "actions": [{"action": "UP"}],
                "turns": [{"action": "UP", "outcome": "playing", "frame": _frame(fill).payload()}],
                "digests": files,
            }
            return (
                "import runtime._repl_state as _h\n"
                f"_h.install_suggestion(globals(), {payload!r})\n"
                "print(len(suggested_history), suggested_actions)"
            )

        output, failed = await tool.run_harness(install(4), timeout=30.0)
        assert failed and "SearchMismatch" in output and "board differs" in output
        output, failed = await tool.run_harness(install(3, {**digests, "notes.txt": "0" * 64}), timeout=30.0)
        assert failed and "SearchMismatch" in output and "not the ones the search read" in output
        output, failed = await tool.run_harness(install(3), timeout=30.0)
        assert not failed and "2 [Action('UP')]" in output
    finally:
        await tool.clear_state()


#: A model split in two, the helper importing the model back: loads the same
#: way in the kernel and off it.
CIRCULAR_MODEL = """\
import rules


def initial_state():
    return {'outcome': 'playing', 'energy': 2000}


def do_move(state, action):
    state['energy'] -= 1


def render(state):
    return np.full((8, 8), rules.FILL, dtype=np.int8)
"""

#: The helper: uses a preloaded name unimported, and imports world_model back.
CIRCULAR_RULES = """\
import world_model

FILL = 3
SIZES = Counter([FILL])
LOADING = world_model.__name__
"""


async def test_the_models_own_tests_are_collected_and_the_retest_is_not() -> None:
    """Each explicit ``test_world_model()`` leaves one record for the host's
    ``seal/test`` span, collected once; the harness's retest after the call
    is not the model's, and leaves none."""
    import json

    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="mandatory")
    state: dict[str, Any] = {"level": 1, "turns": [_turn(3)], "available_actions": ["UP"]}

    def get_state() -> dict[str, Any]:
        return state

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool.host_functions = {"get_state": get_state, "action": action}
    collect = "import runtime._repl_state as _h\nprint(_h.after_call())\ndel _h"
    try:
        assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        Path(tool._tmpdir, "world_model.py").write_text(STILL_MODEL.replace("3, dtype", "4, dtype"))  # pyright: ignore[reportPrivateUsage]
        await tool._execute({"code": "test_world_model()"})  # pyright: ignore[reportPrivateUsage]
        Path(tool._tmpdir, "world_model.py").write_text(STILL_MODEL)  # pyright: ignore[reportPrivateUsage]
        # Strict energy as the model's setter leaves it, set here without the
        # host the setter would call.
        await tool.run_harness(
            "import runtime._repl_state as _h\n_h.matching['strict_energy'] = True\ndel _h", timeout=30.0
        )
        await tool._execute({"code": "test_world_model()"})  # pyright: ignore[reportPrivateUsage]
        output, failed = await tool.run_harness(collect, timeout=30.0)
        assert not failed, output
        failing, passing = json.loads(output.strip().splitlines()[-1])["tests"]
        assert (failing["passed"], failing["category"], failing["turn_index"]) == (False, "render", 0), failing
        assert passing == {
            "passed": True,
            "category": None,
            "turn_index": None,
            "turns": 1,
            "strict_energy": True,
            "pixel_tolerance": 0,
            "pixels_tolerated": 0,
        }

        output, failed = await tool.run_harness(
            "import runtime._repl_state as _h\nprint(_h.search_retest())\ndel _h", timeout=30.0
        )
        assert not failed, output
        output, failed = await tool.run_harness(collect, timeout=30.0)
        assert not failed and json.loads(output.strip().splitlines()[-1])["tests"] == [], output
    finally:
        await tool.clear_state()


def test_the_model_loads_off_the_kernel_as_it_does_in_it(tmp_path: Path) -> None:
    """One runner for world_model.py: its helpers start with the same names,
    a helper that imports the model back gets the module being loaded, and
    the helper's data is read-only — in the agent process as in the kernel."""
    from seal.harness import game as harness_game
    from seal.harness import test as harness_test
    from seal.py_model import PyWorldModel

    (tmp_path / "world_model.py").write_text(CIRCULAR_MODEL)
    (tmp_path / "rules.py").write_text(CIRCULAR_RULES)
    history = harness_game.History()
    history._append_turn(_turn(3))  # pyright: ignore[reportPrivateUsage]
    result = harness_test.validate_world_model(str(tmp_path / "world_model.py"), history, ["UP"])
    assert not result, repr(result)
    model = PyWorldModel(
        {name: (tmp_path / name).read_bytes() for name in ("world_model.py", "rules.py")}, [_turn(3)], ["UP"]
    )
    assert int(model.render(model.initial_state())[0, 0]) == 3


#: A model whose helper is imported inside a function, so the import runs at
#: each call rather than at load.
LATE_IMPORT_MODEL = """\
def initial_state():
    return {'outcome': 'playing', 'energy': 2000}


def do_move(state, action):
    state['energy'] -= 1


def render(state):
    from late_fill import FILL
    return np.full((8, 8), FILL, dtype=np.int8)
"""


def test_a_helper_edited_between_two_tests_is_the_edited_one(tmp_path: Path) -> None:
    """The agent process tests and searches every version of the model's
    files, each from its own directory. A helper the model imports inside a
    function was imported by the previous version's search; the next test
    must run the helper as the files now say, as the REPL does, or it fails
    a model the REPL passed and the game ends."""
    import sys

    from seal.harness import game as harness_game
    from seal.harness import test as harness_test
    from seal.py_model import PyWorldModel

    def files(fill: int) -> dict[str, bytes]:
        """The model with its helper at ``fill``."""
        return {"world_model.py": LATE_IMPORT_MODEL.encode(), "late_fill.py": f"FILL = {fill}\n".encode()}

    try:
        before = PyWorldModel(files(3), [_turn(3)], ["UP"])
        assert int(before.render(before.initial_state())[0, 0]) == 3
        edited = tmp_path / "edited"
        edited.mkdir()
        for name, data in files(4).items():
            (edited / name).write_bytes(data)
        history = harness_game.History()
        history._append_turn(_turn(4))  # pyright: ignore[reportPrivateUsage]
        result = harness_test.validate_world_model(str(edited / "world_model.py"), history, ["UP"])
        assert not result, repr(result)
        after = PyWorldModel(files(4), [_turn(4)], ["UP"])
        assert int(after.render(after.initial_state())[0, 0]) == 4
    finally:
        sys.modules.pop("late_fill", None)


#: A cell that slides right along the top row, rendered as the state's own
#: array: what the model hands back is the board it goes on mutating.
SLIDING_MODEL = """\
import numpy as np


def initial_state():
    g = np.zeros((8, 8), dtype=np.int8)
    g[0, 0] = 3
    return {'outcome': 'playing', 'energy': 2000, 'grid': g}


def do_move(state, action):
    state['energy'] -= 1
    g = state['grid']
    c = int(np.argmax(g[0]))
    g[0, c] = 0
    g[0, min(c + 1, 7)] = 3


def render(state):
    return state['grid']
"""


def _slid(column: int, action: str = "") -> dict[str, Any]:
    """A turn whose board has the cell at ``column`` of the top row."""
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[0, column] = 3
    frame = FramePayload(grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"), shape=[8, 8], step=0, level=1)
    return {"action": action, "outcome": "playing", "animation": [frame.payload()]}


def test_a_walked_plan_keeps_each_moves_own_board() -> None:
    """The search walks a found path through one state, and a render that
    returns the state's own array would leave every move showing the last
    board -- which the REPL's replay then refuses at the first move."""
    from platypus.contract import Action
    from seal.py_model import PyWorldModel
    from seal.search_process import predict_turns

    model = PyWorldModel({"world_model.py": SLIDING_MODEL.encode()}, [_slid(0)], ["RIGHT"])
    turns, reason = predict_turns(model, model.initial_state(), [Action("ACTION4")] * 3)
    assert reason is None
    assert [int(np.argmax(turn.frame[0])) for turn in turns if turn.frame is not None] == [1, 2, 3]


async def test_a_replayed_history_keeps_each_turns_own_predicted_board() -> None:
    """``test_world_model()`` replays the level through one state, keeping each
    turn's render as its ``predicted_frame``; the per-move check keeps the
    move's the same way. Neither may follow the state on to later moves."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="mandatory")
    state: dict[str, Any] = {
        "level": 1,
        "turns": [_slid(0), _slid(1, "RIGHT"), _slid(2, "RIGHT")],
        "available_actions": ["RIGHT"],
    }

    def get_state() -> dict[str, Any]:
        return state

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool.host_functions = {"get_state": get_state, "action": action}
    try:
        assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        Path(tool._tmpdir, "world_model.py").write_text(SLIDING_MODEL)  # pyright: ignore[reportPrivateUsage]
        content = await tool._execute(  # pyright: ignore[reportPrivateUsage]
            {"code": "print(test_world_model())\nprint([int(t.predicted_frame[0].argmax()) for t in history])"}
        )
        assert "ALL PASSED" in content.to_string(), content.to_string()
        assert content.to_string().strip().endswith("[0, 1, 2]"), content.to_string()
    finally:
        await tool.clear_state()


def test_the_model_process_settings_hold_for_a_with_and_are_undone() -> None:
    """The kernel's settings — numpy's 1.x scalar repr, directory listings in
    name order — for the span of the search's second test, and not after."""
    import os

    from seal.harness import game as harness_game

    listdir = os.listdir
    with harness_game._model_process_settings():  # pyright: ignore[reportPrivateUsage]
        assert repr(np.int8(5)) == "5"
        assert os.listdir is not listdir
    assert repr(np.int8(5)) != "5"
    assert os.listdir is listdir


async def test_a_harness_cells_output_comes_back_whole() -> None:
    """The agent parses what a harness cell prints — the frame delta's JSON
    line, the search's verdict — so it is not cut at the model's output cap
    (two games in the first A/B crashed on a cut JSON line), while the
    model's own call still is."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="none")
    tool.host_functions = {"get_state": lambda: {"level": 1, "turns": [], "available_actions": []}}
    try:
        await tool._execute({"code": "pass"})  # pyright: ignore[reportPrivateUsage]
        long = "print('x' * 50000)"
        output, failed = await tool.run_harness(long, timeout=30.0)
        assert not failed and len(output.strip()) == 50000
        content = await tool._execute({"code": long})  # pyright: ignore[reportPrivateUsage]
        assert "truncated" in content.to_string()
    finally:
        await tool.clear_state()


def test_a_search_killed_inside_the_models_code_names_the_call() -> None:
    """A search the parent had to kill says which call into the model it was
    stuck in, so a model whose ``do_move`` never returns for some move can
    find it; a child that died says where it died."""
    provenance = Provenance(level=1, history_length=1, digests={}, mouse=False)
    stuck = "do_move(state, Action('UP'))"
    killed = SearchReport(provenance, SearchResult(outcome="timeout", error="killed", stuck_in=stuck)).text()
    assert f"It was stopped inside {stuck} in your world model, a call that had not returned." in killed
    died = SearchReport(provenance, SearchResult(outcome="died", error="exited", stuck_in=stuck)).text()
    assert f"It died inside {stuck} in your world model." in died
    # Killed in the search's own work: nothing to name.
    assert "stopped inside" not in SearchReport(provenance, SearchResult(outcome="timeout", error="killed")).text()


def _exhausted(*, mouse: bool, over_cap: int = 0) -> str:
    """The report of a search that searched everything and found nothing."""
    provenance = Provenance(level=1, history_length=1, digests={}, mouse=mouse)
    diagnostics = PlanDiagnostics(exit_reason="exhausted", most_click_regions=140, boards_over_click_cap=over_cap)
    value = NoPlan(tag="exhausted", reason="", detail={}, diagnostics=diagnostics)
    return SearchReport(provenance, SearchResult(outcome="ok", value=value)).text()


def test_a_complete_search_in_a_mouse_game_is_complete_for_its_clicks() -> None:
    """In a game with MOUSE, a search that searched everything searched only
    what its clicks reach -- one per same-colour region -- and says so where
    it says it is complete, not after; it says whether every region of every
    board got a click. Without MOUSE the claim stands unqualified."""
    clicked_all = _exhausted(mouse=True)
    assert "Search space: complete for the clicks tried — every state they reach was searched." in clicked_all
    assert (
        "The states those clicks reach were fully searched, and no win or unknown state is among them." in clicked_all
    )
    assert "on every region of its board" in clicked_all
    capped = _exhausted(mouse=True, over_cap=3)
    assert "on its 100 largest regions only" in capped
    assert (
        "Some searched boards had more regions than that, up to 140, and their other regions were never clicked."
        in capped
    )
    keyboard = _exhausted(mouse=False)
    assert "Search space: complete — every reachable state was searched." in keyboard
    assert "The state space was fully searched, and no win or unknown state exists." in keyboard
    assert "MOUSE" not in keyboard
