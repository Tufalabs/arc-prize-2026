"""Whole-block eviction, and what the provable bound weighs.

Eviction drops user-led blocks from the oldest end. It is the half of a trim
that cannot touch the current turn — see test_context_pruning.py for the
other half — and on this rung it is the rarer half, since a game is usually
one block.

The bound is the decision ruler wherever no engine bill applies (cold start,
and the rounds after a rewrite): text at its character count, framing and
images at their constants. What it does and does not weigh decides when a
trim fires, so it is behaviour.
"""

from __future__ import annotations

import numpy as np
import ullm.conv
from duck import DuckAgent


def _block(user_chars: int, reply_chars: int, tag: str) -> list[ullm.conv.ConversationElement]:
    turn = ullm.conv.ResponseTurn(status="processed")
    turn.elements.append(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.StringContent(text="r" * reply_chars), status="processed", stop_reason="stop"
        )
    )
    return [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=tag + "u" * user_chars)), turn]


async def test_eviction_drops_whole_oldest_blocks_and_keeps_last() -> None:
    agent = DuckAgent(reply_cap_tokens=None, context_window_tokens=6000, evict_turns=2, compact_to_tokens=2048)
    messages: list[ullm.conv.ConversationElement] = []
    for index in range(6):
        messages.extend(_block(2000, 1000, f"tag{index}-"))
    trimmed = await agent._trim_messages_for_context(list(messages))

    kept = [e.content.to_string()[:5] for e in trimmed if isinstance(e, ullm.conv.UserPrompt)]
    assert kept == sorted(kept), "order preserved"
    assert kept[-1] == "tag5-", "the newest block is never the one evicted"
    assert len(kept) < 6, "nothing was evicted at all"
    assert isinstance(trimmed[0], ullm.conv.UserPrompt), "a conversation must start at a user-led block"


async def test_the_last_block_survives_a_target_it_cannot_reach() -> None:
    """Eviction stops at the last block rather than emptying the
    conversation, even when the target is below what one block plus the
    system prompt can ever weigh."""
    agent = DuckAgent(reply_cap_tokens=None, context_window_tokens=5000, compact_to_tokens=1024)
    messages: list[ullm.conv.ConversationElement] = []
    for index in range(6):
        messages.extend(_block(2000, 1000, f"tag{index}-"))
    trimmed = await agent._trim_messages_for_context(list(messages))
    assert sum(1 for e in trimmed if isinstance(e, ullm.conv.UserPrompt)) == 1


async def test_eviction_chunk_leaves_headroom() -> None:
    """``evict_turns`` forces extra blocks out once eviction starts, so a
    re-trim is once per that many blocks rather than every round — each trim
    shifts the prefix and busts the KV cache behind it."""
    agent = DuckAgent(reply_cap_tokens=None, context_window_tokens=20000, evict_turns=3, compact_to_tokens=8192)
    messages: list[ullm.conv.ConversationElement] = []
    for index in range(8):
        messages.extend(_block(6000, 2000, f"tag{index}-"))
    trimmed = await agent._trim_messages_for_context(messages)
    dropped = 8 - sum(1 for e in trimmed if isinstance(e, ullm.conv.UserPrompt))
    assert dropped >= 3


def test_an_image_is_charged_its_engine_cost_not_its_bytes() -> None:
    """An image carries no text, so the bound renders it as nothing and
    charges it separately at its engine-side cost — the same for a noisy
    image and a flat one of the same size, since the engine sees cells, not
    bytes."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    noise = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    image = ullm.conv.ImageContent(data=noise)
    flat = ullm.conv.ImageContent(data=np.zeros((64, 64, 3), dtype=np.uint8))
    assert agent._content_request_text(image) == ""
    assert agent._image_bound_tokens(image) == agent._image_bound_tokens(flat) > 0


def test_the_legend_is_weighed_like_any_other_image() -> None:
    """It leads every task prompt, so if it were free in the bound the trim
    would fire late by exactly its weight."""
    from duck import palette

    legend = ullm.conv.ImageContent(data=palette.legend_array())
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    with_legend = ullm.conv.UserPrompt(
        content=ullm.conv.ListContent(elements=[ullm.conv.StringContent(text="look"), legend])
    )
    bare = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="look"))
    difference = agent._bounded_prompt_tokens([with_legend]) - agent._bounded_prompt_tokens([bare])
    assert difference == agent._image_bound_tokens(legend) > 0


def test_reasoning_counts_in_the_bound() -> None:
    """Reasoning is re-sent for all turns, so it weighs in the bound as the
    input text it returns as."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    turn = ullm.conv.ResponseTurn(status="processed")
    turn.elements.append(
        ullm.conv.LLMResponse(
            reasoning_content=ullm.conv.StringContent(text="THINKING-" * 100),
            final_content=ullm.conv.StringContent(text="short answer"),
            status="processed",
        )
    )
    with_reasoning = agent._element_request_dicts(turn)
    assert "THINKING-" in str(with_reasoning)


def test_the_bound_mirrors_what_is_sent() -> None:
    """System prompt and tool schema are in the pseudo-request: they sit in
    front of every request and cannot be evicted, so a bound omitting them
    reads the prompt as smaller than the engine will."""
    payload = DuckAgent(
        reply_cap_tokens=None,
    )._bound_request_payload([])
    assert payload["messages"][0]["role"] == "system"
    assert payload["tools"][0]["function"]["name"] == "python"
    assert payload["tool_choice"] == "auto"
