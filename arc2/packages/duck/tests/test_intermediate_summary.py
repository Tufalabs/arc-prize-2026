"""Summaries that leave the history whole (``intermediate_summary_prompt_lengths``):
a request passing one of the lengths is held, a turn of its own asks for a
summary, which stays in the history as served, and the next block opens on the
message to go on. A compaction runs alone where both fall due, and re-arms the
lengths; its request then says the earlier summaries go too."""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backend
import ullm.conv
from duck import DuckAgent
from duck.prompts import DuckPrompts
from test_context_summarization import (
    SUMMARY,
    THINK_CLOSE,
    _CountingScript,
    _history,
    _response,
    _run_block,
    _summarizing_agent,
)

INTERMEDIATE = DuckPrompts().intermediate_summary_head
CONTINUE = DuckPrompts().intermediate_summary_continue
EARLIER = "your earlier summaries included"


def _agent(*lengths: int) -> DuckAgent:
    target = _summarizing_agent()
    target.intermediate_summary_prompt_lengths = lengths
    target._validate_coherence()  # pyright: ignore[reportPrivateUsage]
    _history(target)
    return target


def _script(*responses: Any, ordinary: int, compacted: int = 500) -> _CountingScript:
    return _CountingScript(
        responses=list(responses), ordinary_count=ordinary, compacted_count=compacted, summary_count=500
    )


def _texts(target: DuckAgent) -> list[str]:
    return [element.to_string() for element in target.messages]


async def test_a_length_passed_holds_the_request_for_a_summary_kept_in_the_history() -> None:
    target = _agent(1_000)
    backend = _script(_response("INTERMEDIATE NOTES", completion_tokens=5), ordinary=1_500)

    calls = await _run_block(target, backend)

    assert calls == 1
    (request,) = backend.seen_requests
    assert request.user_texts[-1].startswith(INTERMEDIATE)
    assert request.user_texts[-2] == "CURRENT REQUEST", "the block's opening prompt goes out with the request"
    assert request.response_injection == THINK_CLOSE
    assert request.max_tokens == target.reply_cap_tokens
    # Kept as served: the request, then the summary; the message to go on opens the next block.
    texts = _texts(target)
    assert INTERMEDIATE in texts[-2]
    assert "INTERMEDIATE NOTES" in texts[-1]
    assert target._pending_tool_feedback == [[ullm.conv.StringContent(text=CONTINUE)]]  # pyright: ignore[reportPrivateUsage]
    assert target._intermediate_lengths_passed == (1_000,)  # pyright: ignore[reportPrivateUsage]
    assert target._intermediate_summaries_in_cycle == 1  # pyright: ignore[reportPrivateUsage]
    assert target.context_summaries == 0, "nothing was compacted"


async def test_each_length_is_asked_once_a_cycle_and_the_history_goes_on() -> None:
    target = _agent(1_000)
    backend = _script(_response("INTERMEDIATE NOTES"), _response("DONE"), ordinary=1_500)

    await _run_block(target, backend)
    await _run_block(target, backend)

    first, second = backend.seen_requests
    assert first.user_texts[-1].startswith(INTERMEDIATE)
    assert second.user_texts[-1] == "CURRENT REQUEST"
    assert any("INTERMEDIATE NOTES" in text for text in second.element_texts), "the summary stays in the history"


async def test_lengths_passed_together_get_one_summary() -> None:
    target = _agent(500, 1_000, 2_000)
    backend = _script(_response("INTERMEDIATE NOTES"), _response("DONE"), ordinary=1_500)

    await _run_block(target, backend)
    await _run_block(target, backend)

    assert [request.user_texts[-1].startswith(INTERMEDIATE) for request in backend.seen_requests] == [True, False]
    assert target._intermediate_lengths_passed == (500, 1_000)  # pyright: ignore[reportPrivateUsage]


async def test_a_compaction_due_at_once_runs_alone_and_rearms_the_lengths() -> None:
    target = _agent(1_000)
    backend = _script(_response(SUMMARY), _response("DONE"), ordinary=target._context_budget_tokens() + 1)  # pyright: ignore[reportPrivateUsage]
    target._intermediate_lengths_passed = (1_000,)  # pyright: ignore[reportPrivateUsage]

    await _run_block(target, backend)

    summary, ordinary = backend.seen_requests
    assert summary.summary and not ordinary.summary
    assert not any(text.startswith(INTERMEDIATE) for request in backend.seen_requests for text in request.user_texts)
    # The rebuilt request (500 tokens) is under the length, so it is asked again.
    assert target._intermediate_lengths_passed == ()  # pyright: ignore[reportPrivateUsage]


async def test_a_rebuilt_request_past_a_length_does_not_ask_straight_away() -> None:
    target = _agent(1_000)
    backend = _script(
        _response(SUMMARY),
        _response("DONE"),
        ordinary=target._context_budget_tokens() + 1,  # pyright: ignore[reportPrivateUsage]
        compacted=1_500,
    )

    await _run_block(target, backend)

    assert [request.summary for request in backend.seen_requests] == [True, False]
    assert target._intermediate_lengths_passed == (1_000,)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("held", [0, 1])
async def test_the_compaction_request_says_the_earlier_summaries_go_too(held: int) -> None:
    target = _agent(1_000)
    target._intermediate_summaries_in_cycle = held  # pyright: ignore[reportPrivateUsage]
    backend = _script(_response(SUMMARY), _response("DONE"), ordinary=target._context_budget_tokens() + 1)  # pyright: ignore[reportPrivateUsage]

    await _run_block(target, backend)

    request = backend.seen_requests[0]
    assert request.summary
    assert (EARLIER in request.user_texts[-1]) is bool(held)
    assert target._intermediate_summaries_in_cycle == 0  # pyright: ignore[reportPrivateUsage]


async def test_an_unusable_intermediate_summary_costs_the_game_nothing_more() -> None:
    target = _agent(1_000)
    tool_call = ullm.conv.LLMResponse(
        final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": "1"}),
        token_usage=ullm.backend.TokenUsage(completion_tokens=3, num_requests=1),
    )
    backend = _script(tool_call, tool_call, ordinary=1_500)

    calls = await _run_block(target, backend)

    assert calls == 2
    assert all(request.user_texts[-1].startswith(INTERMEDIATE) for request in backend.seen_requests)
    assert not any(INTERMEDIATE in text for text in _texts(target)), "nothing of the refused turn is kept"
    assert target._pending_tool_feedback == []  # pyright: ignore[reportPrivateUsage]
    assert target._intermediate_summaries_in_cycle == 0  # pyright: ignore[reportPrivateUsage]


def test_without_lengths_nothing_is_ever_due() -> None:
    target = _summarizing_agent()
    assert not target._intermediate_summary_due(10**6)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"intermediate_summary_prompt_lengths": (0,)}, "positive token counts"),
        ({"intermediate_summary_prompt_lengths": (1_000,), "compaction_strategy": "prune"}, "summarize"),
    ],
)
def test_the_lengths_are_checked_before_the_engine_starts(overrides: dict[str, Any], match: str) -> None:
    target = _summarizing_agent()
    for name, value in overrides.items():
        setattr(target, name, value)
    with pytest.raises(ValueError, match=match):
        target._validate_coherence()  # pyright: ignore[reportPrivateUsage]


def test_the_request_names_no_size_no_thorough_mode_and_no_source() -> None:
    target = _summarizing_agent()
    target.prompts = DuckPrompts(thorough_summary=True)
    text = target._prompts().intermediate_summary_prompt(target._prompt_modes())  # pyright: ignore[reportPrivateUsage]
    assert text.startswith(INTERMEDIATE)
    assert "At most" not in text and "thorough" not in text and "SOURCE CODE" not in text
    assert "discarded" not in text and "after compaction" not in text
    assert [line[:2] for line in text.splitlines() if line[:1].isdigit()] == ["1.", "2.", "3.", "4."]
