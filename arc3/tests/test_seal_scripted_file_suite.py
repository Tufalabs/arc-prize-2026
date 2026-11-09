"""What the seal's file suite says back, on every path a model can reach.

A successful session only takes the successful paths, and every other branch
is model-visible text that nothing else pins. So this one is driven to its
corners on purpose:

- a whole view, a ranged view, a view of the directory, a range past the end,
  and a file that is not there;
- a replacement that lands, one whose ``old_string`` differs only in
  indentation, an ambiguous one, and one with nothing like it in the file;
- a whole-file write that leaves the file unparseable, which warns and applies
  anyway;
- a path outside the directory, a path into a subdirectory, and a null byte;
- any file, not only ``.py`` — a snippet writes beside the model's source,
  and one of those files is binary, so it is named and sized rather than
  decoded into a prompt;
- importing a helper the model wrote, and importing it again after an edit —
  which must see the edit rather than the module it already loaded.

What is deliberately NOT here. Rendering rules that are pure functions of
their input — the view cap, a clipped line, a long replacement named rather
than echoed — live in
``packages/seal/tests/test_seal_file_suite_rendering.py``: a reference
re-sends the conversation on every call, so it grows quadratically and is the
wrong instrument for driving a formatter to its edges. And one trap has no
path to it from the seal's kernel at all — a directory inside the flat
workspace — because a snippet cannot create one: numpy reaches the
filesystem but will not make a directory to write into. It is covered in
``packages/seal/tests/test_seal_workspace.py``, which is where insurance
against a wider runtime belongs.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_file_suite'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv

from platypus import LLMAgentSolver, RunConfig
from reference_support import (
    ReferenceCase,
    ReferenceRun,
    regenerate_case,
    replay_case,
    require_regeneration,
    tool_call,
)
from seal import SealAgent
from test_seal_scripted_reference import CornerGame

RNG_SEED = 20260824

#: Small enough to quote whole, and shaped for the edits below: ``ROWS`` twice
#: so one replacement is ambiguous, and unindented so another can miss on
#: indentation alone.
HELPER = "ROWS = 8\nCOLS = 8\nAREA = ROWS * COLS\n"

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


def _replace(path: str, old: str, new: str) -> ullm.conv.LLMResponse:
    """The scripted model replacing a substring."""
    return tool_call("str_replace", {"path": path, "old_string": old, "new_string": new})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


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
        # --- nothing written yet: an empty directory ------------------------
        _view("."),
        # --- looking ------------------------------------------------------
        _create("helper.py", HELPER),
        _view("helper.py"),
        _view("helper.py", [1, 2]),
        _view("helper.py", [90, 95]),
        _view("ghost.py"),
        # --- replacing: one that lands, three that do not -----------------
        _replace("helper.py", "COLS = 8", "COLS = 12"),
        _replace("helper.py", "    ROWS = 8", "    ROWS = 9"),
        _replace("helper.py", "ROWS", "R"),
        _replace("helper.py", "PLAYER = None", "PLAYER = 0"),
        # --- a write that does not parse: warned, and applied -------------
        _create("helper.py", HELPER_BROKEN),
        _create("helper.py", HELPER),
        # --- paths that are not the model's to reach ----------------------
        _view("../escape.py"),
        _view("sub/nested.py"),
        _view("bad\0name.py"),
        # --- a module name Python already has: no import would reach it ---
        _create("queue.py", "PENDING = []\n"),
        # --- any name, and one the tools cannot read ----------------------
        _create("notes.txt", "Board is 8x8. UP advances level 1.\n"),
        _python("np.save('scores.npy', np.arange(3))\n"),
        _view("scores.npy"),
        _view("."),
        # --- the other half of the surface: the files are modules ---------
        _python("import helper\nprint('AREA', helper.AREA)\n"),
        _replace("helper.py", "COLS = 8", "COLS = 12"),
        _python("import helper\nprint('COLS', helper.COLS)\n"),
        # --- act ----------------------------------------------------------
        _python("submit_action([Action('UP')] * 2)\n"),
        _python("submit_action([Action('DOWN')] * 2)\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the two-level corner game."""
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        # No side turns: a reflection or an audit would take the script's
        # next replies, and none of the corners here is theirs.
        level_reflection="none",
        strategy_audit="none",
        # The whole suite is exercised inside the first decision move, so the
        # cap has to clear the longest one rather than a typical one, with
        # room for a corner to be added without silently truncating the run.
        max_llm_calls_per_move=40,
        # Acts without a world model, which the default gate refuses.
        use_world_model="none",
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


FILE_SUITE = ReferenceCase(
    name="seal_scripted_file_suite",
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

    # Replacing: what landed, and the three ways it can fail.
    assert "Edited helper.py — your replacement is at line 2:" in text
    assert "Closest match, at line 1 — same text, indented differently." in text
    assert "old_string appears 2 times in 'helper.py', at lines 1 and 3; it must be unique." in text
    assert "and no similar text was found" in text

    # A write that does not parse is warned about, not refused.
    assert "helper.py does not parse yet — SyntaxError at line 2: '(' was never closed." in text

    # Paths that are not the model's to reach, each refused in its own terms
    # and none of them naming a resolved location.
    assert "'sub/nested.py' is not allowed: your working directory is flat" in text
    assert "File 'bad\0name.py' is not allowed. You may read and write any file" in text
    # A name Python already has is refused, and says why: the file would
    # exist, and `import queue` would still mean Python's module.
    assert "'queue.py' was not created: `queue` is already a module Python can import" in text

    # Any name is reachable, and a binary one is named and sized rather than
    # decoded — the model writes both from the same directory.
    assert "Created notes.txt (1 line)." in text
    assert "'scores.npy' is not a text file" in text

    # The files are modules: the helper imports, and the edit is what the
    # second import sees rather than the module already loaded.
    assert "AREA 64" in text
    assert "COLS 12" in text

    # The listing after all of it: everything the model wrote, whichever tool
    # or snippet wrote it.
    assert "Working directory — 3 files" in text


async def test_regenerate_seal_scripted_file_suite(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. The fresh reference is validated by
    immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(FILE_SUITE, backend=backend, tmp_path=tmp_path)
