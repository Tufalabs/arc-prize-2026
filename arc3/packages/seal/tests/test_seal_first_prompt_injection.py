"""``inject_text_on_first_prompt`` rides the game's first model call,
``inject_text_on_later_prompts`` every call after it; both armed on the
block's live params before every call."""

from __future__ import annotations

import pickle

import pytest
import ullm.backend
import ullm.conv

from seal.agent import SealAgent


def _agent(first: str, later: str = "") -> SealAgent:
    agent = SealAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        inject_text_on_first_prompt=first,
        inject_text_on_later_prompts=later,
    )
    agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=10)  # pyright: ignore[reportPrivateUsage]
    return agent


def _injection_by_call(agent: SealAgent, calls: int) -> list[str | None]:
    """What each of ``calls`` successive calls would inject, armed as the
    before-round hook arms it after counting the call."""
    seen: list[str | None] = []
    for _ in range(calls):
        agent.response_step += 1
        agent._arm_prompt_injection()  # pyright: ignore[reportPrivateUsage]
        seen.append(agent._live_params.response_injection)  # pyright: ignore[reportPrivateUsage]
    return seen


def test_only_the_first_call_of_the_game_is_injected() -> None:
    assert _injection_by_call(_agent("Okay, so"), 3) == ["Okay, so", None, None]


def test_later_calls_take_their_own_text() -> None:
    assert _injection_by_call(_agent("Okay, so", "We need"), 3) == ["Okay, so", "We need", "We need"]


def test_later_text_alone_leaves_the_opening_bare() -> None:
    assert _injection_by_call(_agent("", "We need"), 2) == [None, "We need"]


def test_nothing_injects_nothing() -> None:
    assert _injection_by_call(_agent(""), 2) == [None, None]


def test_a_resumed_game_does_not_open_again() -> None:
    """``response_step`` pickles with the game: a second block of a game
    that already made its first call gets the later text, not the opening."""
    agent = _agent("Okay, so", "We need")
    agent.response_step = 1
    assert _injection_by_call(agent, 1) == ["We need"]


def test_context_schedule_rearms_on_rewrite_without_counting_consuming_it() -> None:
    agent = _agent("Opening")
    agent.prompt_injection_schedule = "context"
    agent.response_step = 1
    agent._arm_prompt_injection()
    agent._arm_prompt_injection()
    assert agent._live_params.response_injection == "Opening"
    agent._record_prompt_injection_epoch()
    agent.response_step += 1
    agent._arm_prompt_injection()
    assert agent._live_params.response_injection is None
    agent._mark_history_rewritten()
    assert agent._live_params.response_injection == "Opening"
    agent._arm_prompt_injection()
    assert agent._live_params.response_injection == "Opening"
    agent._record_prompt_injection_epoch()
    restored = pickle.loads(pickle.dumps(agent))
    assert _injection_by_call(restored, 1) == [None]
    restored._mark_history_rewritten()
    assert restored._live_params.response_injection == "Opening"


@pytest.mark.asyncio
async def test_context_rewrite_in_trim_is_included_in_exact_prompt_bound() -> None:
    agent = _agent("Opening")
    agent.prompt_injection_schedule = "context"
    agent.token_native_context = True
    agent._record_prompt_injection_epoch()
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="play"))]
    agent._live_conversation = ullm.conv.Conversation(elements=agent.messages)
    observed: list[str | None] = []

    class CountingBackend:
        async def token_count(self, params: ullm.backend.GenericSamplingParameters, conversation: object) -> int:
            observed.append(params.response_injection)
            return 10 + len(params.response_injection or "")

    agent._live_backend = CountingBackend()

    async def rewrite(*args: object, **kwargs: object) -> list[ullm.conv.ConversationElement]:
        assert agent._live_params.response_injection is None
        agent._mark_history_rewritten()
        assert await agent._measured_prompt_tokens(agent.messages) == 17
        return agent.messages

    agent._trim_messages_for_context = rewrite
    await agent._trim_live_messages()
    assert agent._pending_bounds == [17]
    assert observed == ["Opening", "Opening"]
    assert agent._live_params.response_injection == "Opening"
