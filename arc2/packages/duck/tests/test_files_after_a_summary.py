"""What a summary says about the model's files and ``python`` session, and
what the conversation rebuilt around it shows of them.

With the file tools and the listing on, the request says the session and the
files are kept and that ``candidate.py`` will be shown, so it does not ask for
the candidate's source; the rebuilt history closes the summary on the listing,
with the file in full. A request that promises the file and a history that does
not show it would lose the candidate outright, so both halves are read off the
same run.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import ullm.conv
from duck import DuckAgent
from duck.prompts import DuckPrompts
from test_context_summarization import (
    INSTRUCTION,
    SUMMARY,
    _CountingScript,
    _history,
    _response,
    _run_block,
    _summarizing_agent,
)

import taaf.game
import taaf.grid

CANDIDATE = "def fn(grid):\n    return grid[:, ::-1]\n"


def _grid(rows: list[list[int]]) -> taaf.grid.Grid:
    return taaf.grid.Grid(np.array(rows, dtype=np.int8))


async def _agent(
    *, file_tools: Literal["none", "full"] = "full", listing: bool = True, candidate: str | None = CANDIDATE
) -> DuckAgent:
    """A summarizing duck whose kernel has run, holds ``solve`` and, unless
    ``candidate`` is None, has ``candidate.py`` in its working directory."""
    target = _summarizing_agent(summary_tail="full")
    target.file_tools = file_tools
    target.add_files_and_repl_to_summary = listing
    target.prompts = DuckPrompts(thorough_summary=True)
    repl = target._ensure_repl()
    view = taaf.game.GameView(
        game_id="files", given=((_grid([[1, 2]]), _grid([[2, 1]])),), questions=(_grid([[3, 4]]),)
    )
    await repl.setup_task(view)
    await repl._execute({"code": "def solve(grid):\n    return grid\nscratch = np.zeros((2, 3), dtype=np.int8)"})
    if candidate is not None:
        assert repl._tmpdir is not None
        with open(f"{repl._tmpdir}/candidate.py", "w") as handle:
            handle.write(candidate)
    _history(target)
    return target


async def _summarize(target: DuckAgent, summary: str = SUMMARY) -> _CountingScript:
    backend = _CountingScript(
        responses=[_response(summary), _response("DONE")],
        ordinary_count=target._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
        wrapper_marker=summary,
    )
    await _run_block(target, backend)
    return backend


async def test_the_candidate_is_shown_after_the_summary_rather_than_restated_in_it() -> None:
    target = await _agent()
    backend = await _summarize(target)

    summary_request, ordinary_request = backend.seen_requests
    request = summary_request.user_texts[-1]
    assert INSTRUCTION in request
    assert "thorough summary" in request and "At most" not in request
    assert "your `python` session (its variables and functions) and your files will not" in request
    assert "`candidate.py` will be shown in full after compaction, so do not restate it" in request
    assert "SOURCE CODE verbatim" not in request
    assert "A listing of your `python` session's objects and your files will be provided" in request
    assert summary_request.tools == ordinary_request.tools == ("python", "view_file", "str_replace", "create_file")

    wrapper_index = next(
        index
        for index, message in enumerate(target.messages)
        if isinstance(message, ullm.conv.UserPrompt) and SUMMARY in message.content.to_string()
    )
    closing = target.messages[wrapper_index + 1]
    assert isinstance(closing, ullm.conv.UserPrompt)
    text = closing.content.to_string()
    assert text.startswith("[context compacted — your `python` session and files are intact; continue from here]")
    assert "Your files:\n- `candidate.py` (" in text
    assert "- `solve`: function" in text and "- `scratch`: ndarray int8 (2, 3)" in text
    assert text.endswith(f"Your `candidate.py`:\n```python\n{CANDIDATE.rstrip()}\n```")
    assert text in ordinary_request.user_texts, "the next request carries the closing message"


async def test_without_the_file_the_summary_carries_the_candidates_source() -> None:
    target = await _agent(candidate=None)
    backend = await _summarize(target)

    request = backend.seen_requests[0].user_texts[-1]
    assert "Your current best candidate fn — include its SOURCE CODE verbatim." in request
    assert "candidate.py" not in request
    closings = [
        message.content.to_string()
        for message in target.messages
        if isinstance(message, ullm.conv.UserPrompt)
        and message.content.to_string().startswith("[context compacted — your")
    ]
    (closing,) = closings
    assert "Your files:\n(none)" in closing and "Your `candidate.py`" not in closing


async def test_the_listing_without_the_file_tools_names_the_session_alone() -> None:
    target = await _agent(file_tools="none")
    backend = await _summarize(target)

    request = backend.seen_requests[0].user_texts[-1]
    assert "your `python` session (its variables and functions) will not" in request
    assert "files" not in request
    assert "SOURCE CODE verbatim" in request, "without the file tools the file is not shown"
    (closing,) = [
        message.content.to_string()
        for message in target.messages
        if isinstance(message, ullm.conv.UserPrompt)
        and message.content.to_string().startswith("[context compacted — your")
    ]
    assert closing.startswith("[context compacted — your `python` session is intact; continue from here]")
    assert "Your files:" not in closing and "- `solve`: function" in closing


async def test_without_the_listing_nothing_closes_the_rebuilt_history() -> None:
    target = await _agent(listing=False)
    backend = await _summarize(target)

    request = backend.seen_requests[0].user_texts[-1]
    assert "SOURCE CODE verbatim" in request and "listing" not in request
    assert not any(
        isinstance(message, ullm.conv.UserPrompt)
        and message.content.to_string().startswith("[context compacted — your")
        for message in target.messages
    )


async def test_a_second_thorough_summary_asks_for_what_still_holds_of_the_first() -> None:
    target = await _agent()
    first = await _summarize(target)
    assert "previous summary" not in first.seen_requests[0].user_texts[-1]
    second = await _summarize(target, "SECOND SUMMARY")
    assert (
        "take over anything that is still relevant from the previous summary" in second.seen_requests[0].user_texts[-1]
    )
