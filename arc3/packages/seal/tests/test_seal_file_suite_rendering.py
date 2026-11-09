"""How the file suite renders what it found — the rules a reference cannot pin.

Every bound in this surface is a bound on CONSTRUCTION rather than on the
finished text: a view is capped in lines, a line in characters, an echoed
region in lines, a search in hits. That is what makes the answers safe to send
without a budget over them, and it is also what makes them fiddly — each rule
has an edge, and a reference is the wrong instrument for driving a formatter
to one (it re-sends the conversation on every call, so it grows quadratically
in exactly the cases worth testing).

The search's time bound is here for the same reason: it takes ten seconds
of real work to reach, which a reference cannot afford to record.

The behaviour these renderings sit inside is covered end to end by
``tests/test_seal_scripted_file_suite.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from seal.file_suite import (
    FILL_ECHO_LINES,
    MAX_ECHO_LINES,
    MAX_LINE_CHARS,
    MAX_READ_BYTES,
    SEARCH_TIMEOUT_S,
    VIEW_CAP,
    SearchFilesTool,
    ViewFileTool,
    clipped,
    file_rows,
    file_tools,
    framed,
    near_miss,
    numbered,
    region,
    replace_in,
)
from seal.workspace import Workspace


class _StubTool:
    """Only what :class:`Workspace` reads: where the files are."""

    def __init__(self, directory: Path) -> None:
        self._tmpdir = str(directory)


def _workspace(directory: Path) -> Workspace:
    """A workspace over ``directory``, configured as the seal ships it."""
    return Workspace(repl=cast("Any", _StubTool(directory)))


def test_a_line_is_clipped_and_says_so() -> None:
    """Every other bound here counts lines, and a line has no length limit —
    one minified line would defeat all of them at once. The marker matters as
    much as the cut: without it the model cannot tell a clipped line from a
    short one, and reasons about the half it was shown as if it were whole."""
    line = "x" * (MAX_LINE_CHARS + 25)
    shown = clipped(line)
    assert shown.startswith("x" * MAX_LINE_CHARS)
    assert "+25 more characters on this line" in shown
    assert clipped("short") == "short"


def test_the_gutter_is_a_tab_and_the_numbers_line_up() -> None:
    """A tab, not spaces: the gutter is as wide as the file is long, and a
    line's own indentation runs straight on from it, so with spaces the model
    has to guess how many of them were ours when it quotes a line back — and
    an ``old_string`` that guesses wrong fails to match."""
    assert numbered(["a", "b"], 9, 2) == " 9\ta\n10\tb"


def test_an_echoed_region_stops_and_says_how_much_it_left() -> None:
    """A region is bounded by the size of the change unless something bounds
    it, so a large edit would answer with a large slice of the file — the
    thing this surface exists to stop doing."""
    content = "\n".join(f"line {index}" for index in range(200))
    shown = region(content, 1, 200)
    assert shown.count("\n\n") == 1  # heading, then the lines
    body = shown.split("\n\n", 1)[1]
    assert len(body.splitlines()) == MAX_ECHO_LINES + 1  # the lines, plus the tail note
    assert "more changed line" in body


def test_a_long_replacement_is_named_rather_than_repeated() -> None:
    """What an edit tells the model that it does not already know is WHERE its
    text landed and what now sits either side. The text itself the model wrote
    one message ago, so handing back a long one is paying twice for it."""
    content = "\n".join(f"line {index}" for index in range(20))
    long_change = framed(content, 0, len(content))
    assert "(your " in long_change
    assert "line 5" not in long_change.split("(your ", 1)[0]

    short = "one\ntwo\nthree\n"
    short_change = framed(short, 0, 3)
    assert "your replacement is at line 1" in short_change
    assert "\tone" in short_change, f"a change of {FILL_ECHO_LINES} lines or fewer is shown"


def test_a_failed_match_that_differs_only_in_indentation_says_so() -> None:
    """A bare "not found" leaves the model nothing to act on but a re-read,
    and re-reading is what this surface exists to avoid. Indentation is the
    common cause, and the number gutter contributes leading space of its own —
    so the difference has to be named, not left to be seen."""
    content = "def go():\n    ROWS = 8\n    return ROWS\n"
    assert "same text, indented differently" in near_miss(content, "ROWS = 8")
    assert near_miss(content, "COMPLETELY = unrelated + text + here") == ""


def test_replace_all_reports_where_every_change_landed() -> None:
    """The line numbers are read off the file as it NOW is: each replacement
    shifts what follows it, so numbers taken from the old text would point the
    model at the wrong lines in the file it is about to view."""
    before = "ROWS = 8\nCOLS = 8\nAREA = ROWS * COLS\n"
    outcome = replace_in(before, "helper.py", "ROWS", "ROW_COUNT", replace_all=True)
    assert outcome.refusal is None
    assert outcome.after == "ROW_COUNT = 8\nCOLS = 8\nAREA = ROW_COUNT * COLS\n"
    assert "2 occurrences of old_string replaced, at lines 1 and 3." in outcome.report


def test_an_empty_old_string_is_refused_rather_than_diagnosed() -> None:
    """``"" in x`` is true everywhere at once: counting it reads the file
    length, and replacing it would interleave the new text between every
    character of the file."""
    outcome = replace_in("X = 1\n", "helper.py", "", "Y", replace_all=False)
    assert outcome.after is None
    assert outcome.refusal is not None
    assert "old_string is empty" in outcome.refusal


def test_a_file_no_tool_can_read_is_listed_by_size(tmp_path: Path) -> None:
    """A listing that hid the file would be describing a directory the model
    cannot act on. It has no line count to give, so it gives what it has."""
    tmp_path.joinpath("helper.py").write_text("X = 1\n")
    tmp_path.joinpath("blob.py").write_bytes(b"\xff\xfe\x00")

    rows = file_rows(_workspace(tmp_path))

    assert rows == "  blob.py    3 bytes\n  helper.py   1 line"


async def test_a_pattern_that_will_not_finish_is_stopped_and_explained(tmp_path: Path) -> None:
    """The model writes the pattern, and a backtracking one is superlinear in
    the subject — so a line a little longer than the last turns a fast search
    into one that never returns, on the event loop of that game's process.

    ``re`` cannot be interrupted at all, which is why the search runs on an
    engine that honours a deadline; a timeout wrapped around ``re`` would
    never fire. The answer names the shape of the problem, because a model
    told only "timed out" will reasonably retry the same pattern.
    """
    tmp_path.joinpath("subject.py").write_text("a" * 40 + "b\n")
    tool = SearchFilesTool(workspace=_workspace(tmp_path))

    started = time.monotonic()
    answer = (await tool._execute({"pattern": "(a|a)*$"})).to_string()
    elapsed = time.monotonic() - started

    assert tool.is_error
    assert f"took longer than {SEARCH_TIMEOUT_S:.0f}s" in answer
    assert "nested repetition" in answer
    # Bounded by the deadline, not merely finishing eventually.
    assert elapsed < SEARCH_TIMEOUT_S * 2, elapsed


async def test_a_file_too_large_to_read_is_refused_before_it_is_read(tmp_path: Path) -> None:
    """The rendering caps bound what the model is shown; they do not bound the
    work done to get there. A snippet can write a file of any size and nothing
    quotas the workspace, so a small answer could still cost the process.

    Refused from the size on disk, so the memory is never spent — a check made
    after the read would be the failure it exists to prevent. The model is
    pointed at Python, which can take the file a piece at a time.
    """
    big = tmp_path / "huge.py"
    big.write_bytes(b"x" * (MAX_READ_BYTES + 1))
    tool = ViewFileTool(workspace=_workspace(tmp_path))

    answer = (await tool._execute({"path": "huge.py"})).to_string()

    assert tool.is_error
    assert "larger than" in answer
    assert "Read it from Python instead" in answer


async def test_search_skips_what_it_cannot_afford_and_stops_at_its_cap(tmp_path: Path) -> None:
    """Two bounds on the work, not just the answer: a file too large to read is
    not searched, and once the hit cap is met the remaining files are not read
    at all. Without the second, a common pattern reads the whole workspace to
    build hits it has already decided not to show."""
    tmp_path.joinpath("small.py").write_text("needle\n")
    tmp_path.joinpath("huge.py").write_bytes(b"needle\n" * ((MAX_READ_BYTES // 7) + 1))
    tool = SearchFilesTool(workspace=_workspace(tmp_path))

    answer = (await tool._execute({"pattern": "needle"})).to_string()

    # The oversized file is skipped, so it is neither searched nor counted.
    assert "huge.py" not in answer
    assert "small.py" in answer
    assert "1 file searched" in answer or "1 file" in answer


async def test_a_whole_view_stops_at_the_cap_and_says_how_to_go_on(tmp_path: Path) -> None:
    """A safety valve, not a ration: the model is told the file is longer and
    how to ask for the rest, because a truncation it cannot see costs it a
    wrong conclusion rather than a second call."""
    tmp_path.joinpath("big.py").write_text("".join(f"X{index} = 1\n" for index in range(VIEW_CAP + 50)))
    tool = ViewFileTool(workspace=_workspace(tmp_path))

    answer = (await tool._execute({"path": "big.py"})).to_string()

    assert f"showing lines 1-{VIEW_CAP}" in answer
    assert "Pass view_range to see the rest." in answer
    assert f"X{VIEW_CAP} = 1" not in answer


async def test_the_syntax_note_offers_undo_only_where_undo_is_offered(tmp_path: Path) -> None:
    """Withdrawing ``undo_edit`` withdraws it from the suite's advice too. A
    hint that sends the model to a tool it was not given costs it a call that
    answers "Unknown tool" -- in exactly the arm meant to measure its absence."""
    by_name = {tool.name: tool for tool in file_tools(_workspace(tmp_path), undo_edit=False)}
    assert "undo_edit" not in by_name

    created = (await by_name["create_file"]._execute({"path": "helper.py", "content": "def broken(\n"})).to_string()
    replaced = (
        await by_name["str_replace"]._execute({"path": "helper.py", "old_string": "broken(", "new_string": "broken(:"})
    ).to_string()

    for answer in (created, replaced):
        assert "does not parse yet" in answer
        assert "otherwise fix it." in answer
        assert "undo_edit" not in answer

    # The full suite still points at the tool it does offer.
    full = {tool.name: tool for tool in file_tools(_workspace(tmp_path))}
    answer = (await full["create_file"]._execute({"path": "other.py", "content": "def broken(\n"})).to_string()
    assert "otherwise fix it or call undo_edit()." in answer
