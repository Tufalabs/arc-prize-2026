"""What the duck's file suite says back, on every path a model can reach.

A successful session only takes the successful paths, and every other branch
is model-visible text that nothing else pins. So this one is driven to its
corners on purpose:

- a whole view, a ranged view, a view of the directory, a range past the end,
  and a file that is not there;
- a search that hits across two files, one that finds nothing, and a pattern
  that is not a regular expression;
- a replacement that lands, one whose ``old_string`` differs only in
  indentation, an ambiguous one, and one with nothing like it in the file;
- an undo that reverts, an undo that removes the file it created, an undo with
  nothing left to take back, and — the duck's own scoping — an undo after an
  action has been executed, which finds the log cleared;
- a whole-file write that leaves the file unparseable, which warns and applies
  anyway;
- a path outside the directory, a path into a subdirectory, and a null byte;
- any file, not only ``.py`` — a snippet writes beside the model's source,
  and one of those files is binary, so it is named and sized rather than
  decoded into a prompt;
- importing a helper the model wrote, and importing it again after an edit —
  which must see the edit rather than the module it already loaded — and
  `importlib.reload` of it after a further edit, which the kernel's drop of the
  model's modules before each call would otherwise leave with nothing to reload;
- the allowlisted modules, preloaded as `np` is, without clobbering a variable
  of the model's own that shares a name with one;
- the tracebacks of the model's own code, which is what it reads a cause off:
  an error inside a helper shows the helper's file and line under the snippet's,
  each frame with its source line; one inside a function an EARLIER call
  defined shows that call's line, not the current snippet's line of the same
  number; a module whose own top level raises on import, and one that does not
  parse, are shown at their file and line, with no path of the host's; a
  runaway recursion is counted rather than listed; and a very long line is cut;
- what the wide sandbox (on, as it ships) lets a snippet reach:
  `time`, `pickle`, `scipy`, `open` for its own data files, and every
  exception class, so an `except KeyError:` catches rather than raising
  NameError in its place; and `inspect.getsource` of a function an earlier
  call defined.

What is deliberately NOT here. Rendering rules that are pure functions of
their input — the view cap, a clipped line, a long replacement named rather
than echoed — live in
``packages/duck/tests/test_duck_file_suite_rendering.py``: a reference
re-sends the conversation on every call, so it grows quadratically and is the
wrong instrument for driving a formatter to its edges. And one trap has no
path to it from the duck's kernel at all — a directory inside the flat
workspace — because a snippet cannot create one: numpy reaches the
filesystem but will not make a directory to write into. It is covered in
``packages/duck/tests/test_duck_workspace.py``, which is where insurance
against a wider runtime belongs.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_duck_scripted_file_suite'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from duck import DuckAgent
from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from test_duck_scripted_reference import CornerGame

RNG_SEED = 20260824

#: Small enough to quote whole, and shaped for the edits below: ``ROWS`` twice
#: so one replacement is ambiguous, and unindented so another can miss on
#: indentation alone.
HELPER = "ROWS = 8\nCOLS = 8\nAREA = ROWS * COLS\n"

#: A second file, so a search can report hits in more than one.
LEVEL_DATA = "ROWS = 4\nSTART = (0, 0)\n"

#: A module as models write them: one function takes the frame it needs, the
#: other reads ``current_frame``, which only a snippet has.
BOARD = "def height(frame):\n    return frame.shape[0]\n\n\ndef width():\n    return current_frame.shape[1]\n"

#: A module whose top level raises as it is imported, on its second line.
BOOM = "X = 1\nY = undefined_name\n"

#: A module that does not parse, imported anyway.
BROKEN = "def f(:\n    pass\n"

#: Defined by one call, raising on ITS third line when a later one-line call
#: runs it: a traceback that looked the line up in the current snippet would
#: find nothing there.
EARLIER = "def first_pixel(frame):\n    grid = frame.array\n    return grid[99][0]\n"

#: A snippet whose failing line is far longer than a traceback quotes.
LONG_LINE = "print('" + "a" * 300 + "'[1000])\n"

#: The modules and builtins models reached for and were refused, used as they used them.
REACH = (
    "import time, pickle, scipy.ndimage\n"
    "labels, count = scipy.ndimage.label(current_frame.array == current_frame.array[0][0])\n"
    "with open('state.pkl', 'wb') as handle:\n"
    "    pickle.dump({'count': int(count)}, handle)\n"
    "with open('state.pkl', 'rb') as handle:\n"
    "    print('reloaded', pickle.load(handle)['count'] == count, time.time() > 0)\n"
    "try:\n"
    "    {}['missing']\n"
    "except KeyError:\n"
    "    print('caught KeyError')\n"
)

#: A whole-file write that does not parse. It is applied anyway, with a note:
#: rejecting it would break a legitimate sequence of edits whose intermediate
#: states do not parse.
HELPER_BROKEN = "ROWS = 8\nCOLS = (8\n"


def _view(path: str, view_range: list[int] | None = None) -> ullm.conv.LLMResponse:
    """The scripted model looking at a file."""
    arguments: dict[str, Any] = {"path": path}
    if view_range is not None:
        arguments["view_range"] = view_range
    return tool_call("view_file", arguments)


def _search(**arguments: Any) -> ullm.conv.LLMResponse:
    """The scripted model searching."""
    return tool_call("search_files", arguments)


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model replacing a substring."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _undo(path: str | None = None) -> ullm.conv.LLMResponse:
    """The scripted model taking back its last change."""
    return tool_call("undo_edit", {} if path is None else {"path": path})


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call, in order.

    Everything before the first ``submit_action`` happens inside one decision
    move: the turn ends when the model acts, so the file work has to come
    first for the whole suite to be exercised against one conversation.
    """
    return [
        # --- nothing written yet: an empty directory, and nothing to undo ---
        _view("."),
        _undo(),
        # --- looking ------------------------------------------------------
        _create("helper.py", HELPER),
        _view("helper.py"),
        _view("helper.py", [1, 2]),
        _view("helper.py", [90, 95]),
        _view("ghost.py"),
        # --- searching ----------------------------------------------------
        _create("level_data.py", LEVEL_DATA),
        _search(pattern="ROWS"),
        _search(pattern="NOTHING_LIKE_THIS"),
        _search(pattern="ROWS ("),
        # --- replacing: one that lands, three that do not -----------------
        _replace("helper.py", "COLS = 8", "COLS = 12"),
        _replace("helper.py", "    ROWS = 8", "    ROWS = 9"),
        _replace("helper.py", "ROWS", "R"),
        _replace("helper.py", "PLAYER = None", "PLAYER = 0"),
        # --- undo: one step, and the file it created ----------------------
        _undo(),
        _undo(),
        _create("scratch.py", "TEMP = 1\n"),
        _undo(),
        # --- a write that does not parse: warned, and applied -------------
        _create("helper.py", HELPER_BROKEN),
        _undo(),
        # --- paths that are not the model's to reach ----------------------
        _view("../escape.py"),
        _view("sub/nested.py"),
        _view("bad\0name.py"),
        # --- any name, and one the tools cannot read ----------------------
        _create("notes.txt", "Board is 8x8. UP advances level 1.\n"),
        _python("np.save('scores.npy', np.arange(3))\n"),
        _view("scores.npy"),
        _view("."),
        # --- the other half of the surface: the files are modules ---------
        _python("import helper\nprint('AREA', helper.AREA)\n"),
        _replace("helper.py", "COLS = 8", "COLS = 12"),
        _python("import helper\nprint('COLS', helper.COLS)\n"),
        _replace("helper.py", "COLS = 12", "COLS = 16"),
        _python("import importlib\nimportlib.reload(helper)\nprint('RELOADED COLS', helper.COLS)\n"),
        _python("time = 'mine'\n"),
        _python("print('PRELOADED', collections.Counter('aab')['a'], scipy.ndimage.label(np.eye(3))[1], time)\n"),
        _create("board.py", BOARD),
        _python("import board\nprint('HEIGHT', board.height(current_frame))\nprint('WIDTH', board.width())\n"),
        _python(EARLIER),
        _python("print(first_pixel(current_frame))\n"),
        _python("import inspect\nprint('SOURCE<' + inspect.getsource(first_pixel) + '>')\n"),
        _create("boom.py", BOOM),
        _python("import boom\n"),
        _create("broken.py", BROKEN),
        _python("import broken\n"),
        _python("def down(n):\n    return down(n + 1)\n\n\ndown(0)\n"),
        _python(LONG_LINE),
        _python(REACH),
        # --- act: this is what clears the undo log ------------------------
        _python("submit_action(['UP', 'UP'])\n"),
        # --- the next move: the edit above is no longer undoable ----------
        _undo(),
        _python("submit_action(['DOWN', 'DOWN'])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The duck with the file suite on, over the two-level corner game."""
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        frame_api="ergonomic",
        file_tools="full",
        # The two gated tools, which the shipped duck withdraws: this is the
        # case that exercises them, and the reduced suite is pinned by the
        # prompt goldens instead.
        expose_tool_search_files=True,
        expose_tool_undo_edit=True,
        completion_cutoff_notice=True,
        expose_undo=True,
        # The whole suite is exercised inside the first decision move, so the
        # cap has to clear the longest one rather than a typical one, with
        # room for a corner to be added without silently truncating the run.
        max_llm_calls_per_move=60,
        # What the reach step, `inspect.getsource` and `importlib.reload` use.
        wide_sandbox=True,
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


FILE_SUITE = ReferenceCase(
    name="duck_scripted_file_suite",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_file_suite_replays_and_covers_its_corners(tmp_path: Path) -> None:
    """The committed reference replays bit-exact, and every steered corner is
    still in it — so a future regeneration cannot quietly lose one."""
    (ref,) = await replay_case(FILE_SUITE, tmp_path)
    text = ref.all_request_text()

    # Outcome: the file work cost nothing in play — the game is still won with
    # exactly the steered actions.
    assert ref.action_ids() == ["ACTION1", "ACTION1", "ACTION2", "ACTION2"]
    assert ref.expected["state"] == "won"

    # Looking.
    assert "Working directory — 0 files" in text
    assert "1\tROWS = 8" in text
    assert "helper.py, lines 1-2 of 3" in text
    assert "is outside 'helper.py', which has 3 lines" in text
    assert "File 'ghost.py' does not exist." in text

    # Searching.
    assert "3 matches for 'ROWS' in 2 files (2 files searched)." in text
    assert "No matches for 'NOTHING_LIKE_THIS' in 2 files (*)." in text
    assert "is not a valid regular expression" in text

    # Replacing: what landed, and the three ways it can fail.
    assert "Edited helper.py — your replacement is at line 2:" in text
    assert "Closest match, at line 1 — same text, indented differently." in text
    assert "old_string appears 2 times in 'helper.py', at lines 1 and 3; it must be unique." in text
    assert "and no similar text was found" in text

    # Undo: one step, the creation it removes, and running out.
    assert "Undid the last change to helper.py. Lines 1-3 now read:" in text
    assert "Undid the creation of scratch.py; the file is gone." in text
    assert "Nothing to undo — no file changes have been made since your last action." in text

    # A write that does not parse is warned about, not refused.
    assert "helper.py does not parse yet — SyntaxError at line 2: '(' was never closed." in text

    # Paths that are not the model's to reach, each refused in its own terms
    # and none of them naming a resolved location.
    assert "'sub/nested.py' is not allowed: your working directory is flat" in text
    assert "File 'bad\0name.py' is not allowed. You may read and write any file" in text

    # Any name is reachable, and a binary one is named and sized rather than
    # decoded — the model writes both from the same directory.
    assert "Created notes.txt (1 line)." in text
    assert "'scores.npy' is not a text file" in text

    # The files are modules: the helper imports, and the edit is what the
    # second import sees rather than the module already loaded.
    assert "AREA 64" in text
    assert "COLS 12" in text
    # A reload after another edit, of the module a previous call imported.
    assert "RELOADED COLS 16" in text
    # The modules a snippet may import are there without the import, and a
    # variable of the model's own named like one of them stays its own.
    assert "PRELOADED 2 3 mine" in text

    # Tracebacks of the model's own code: each frame of it with its source
    # line, and no path of the host's anywhere. (Quotes and newlines are
    # escaped: a traceback travels inside the tool result's JSON.)
    def frame(name: str, line: int, function: str, source: str) -> str:
        return f'  File \\"{name}\\", line {line}, in {function}\\n    {source}\\n'

    assert "ullm_python_" not in text
    # A helper's error under the snippet line that called it.
    assert "HEIGHT 64" in text
    assert (
        frame("<python_tool>", 3, "<module>", "print('WIDTH', board.width())")
        + frame("board.py", 6, "width", "return current_frame.shape[1]")
        + "NameError: name 'current_frame' is not defined"
    ) in text
    # A function an earlier call defined, at ITS line.
    assert (
        frame("<python_tool>", 1, "<module>", "print(first_pixel(current_frame))")
        + frame("<python_tool, earlier call>", 3, "first_pixel", "return grid[99][0]")
        + "IndexError"
    ) in text
    # A module whose top level raises during the import.
    assert (
        frame("<python_tool>", 1, "<module>", "import boom")
        + frame("boom.py", 2, "<module>", "Y = undefined_name")
        + "NameError: name 'undefined_name' is not defined"
    ) in text
    # A module that does not parse: its file and line, not the host's path.
    assert (
        frame("<python_tool>", 1, "<module>", "import broken")
        + '  File \\"broken.py\\", line 1\\n    def f(:\\n'
        + "SyntaxError: invalid syntax"
    ) in text
    # A runaway recursion: three frames shown, the rest counted.
    assert (frame("<python_tool>", 2, "down", "return down(n + 1)") * 3 + "  [Previous line repeated") in text
    assert "RecursionError: maximum recursion depth exceeded" in text
    # What a snippet may reach now: the modules, open, and the exception classes.
    assert "reloaded True True\\ncaught KeyError" in text
    # `inspect` finds the source of a function an earlier call defined.
    assert "SOURCE<def first_pixel(frame):\\n    grid = frame.array\\n    return grid[99][0]>" in text
    # A long line, cut.
    assert "    print('" + "a" * 193 + " ...\\nIndexError: string index out of range" in text

    # The listing after all of it: everything the model wrote, whichever tool
    # or snippet wrote it.
    assert "Working directory — 4 files" in text


async def test_regenerate_duck_scripted_file_suite(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. The fresh reference is validated by
    immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(FILE_SUITE, backend=backend, tmp_path=tmp_path)
