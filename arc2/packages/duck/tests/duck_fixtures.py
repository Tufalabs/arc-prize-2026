"""Conversation fixtures for the context-budget tests.

The compaction and pruning suites drive the same machinery from opposite
ends — one from the trigger, one from what happens once it fires — so they
share the shapes rather than each keeping a copy.

The constants are DERIVED, not copied from upstream: the trim arithmetic is
relative to the fixed prefix (system prompt plus tool schemas), which the
decision ruler charges at its character count — ~7.9k here with the file
tools against upstream's ~12.4k — so upstream's window would leave every
fixture comfortably under budget and quietly test nothing. Measured with the
ruler this branch ships (``_bounded_prompt_tokens``): an empty conversation is
7886, one round 8795, the 12-round single block 18956, the 4x6 multi-block
29851 and its last block alone 13345.
"""

from __future__ import annotations

from typing import Any

import ullm.conv
from duck import DuckAgent

#: Budget = WINDOW - 512 (reply reserve) - 512 (margin) = 16384. Chosen so the
#: single-block fixture below (18956) actually crosses it while one round
#: (8795) does not.
WINDOW = 17408

#: The compaction target. Above the ~7.9k floor an emptied turn leaves (the
#: fixed prefix plus the user prompt itself) so a trim can land on it, and
#: below the 13345 that block eviction alone reaches on the multi-block
#: fixture — otherwise the pruning half of a trim would never be reached and
#: the tests that exist to exercise it would pass vacuously.
TARGET = 11000


def turn(rounds: int) -> ullm.conv.ResponseTurn:
    """``rounds`` complete rounds: a response and the tool result answering it."""
    elements: list[Any] = []
    for i in range(rounds):
        elements.append(ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=f"thinking {i} " * 40)))
        elements.append(
            ullm.conv.ToolResult(
                call_id=f"c{i}", tool_name="python", content=ullm.conv.StringContent(text=f"result {i} " * 40)
            )
        )
    return ullm.conv.ResponseTurn(elements=elements, status="intermediate")


def agent(**kwargs: Any) -> DuckAgent:
    """An agent sized to the constants above, with the target set unless the
    caller is testing the target itself."""
    kwargs.setdefault("reply_cap_tokens", None)
    kwargs.setdefault("compact_to_tokens", TARGET)
    return DuckAgent(context_window_tokens=WINDOW, **kwargs)


def single_block(target: DuckAgent, rounds: int = 12) -> None:
    """ONE user-led block: the shape block eviction cannot touch, and the
    ordinary shape here — a game is usually one long turn."""
    target.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), turn(rounds)]


def many_blocks(target: DuckAgent, blocks: int = 4, rounds: int = 6) -> None:
    """SEVERAL user-led blocks: eviction has room to work, runs out of blocks
    before it runs out of tokens, and leaves the prompt parked between the
    target and the budget."""
    messages: list[Any] = []
    for index in range(blocks):
        messages.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=f"round {index}")))
        messages.append(turn(rounds))
    target.messages = messages


def current_turn(target: DuckAgent) -> ullm.conv.ResponseTurn:
    """The turn being built, which is always the last element."""
    tail = target.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn)
    return tail


def opening_prompt(target: DuckAgent) -> ullm.conv.UserPrompt:
    """The user prompt that opened the conversation."""
    head = target.messages[0]
    assert isinstance(head, ullm.conv.UserPrompt)
    return head


__all__ = [
    "TARGET",
    "WINDOW",
    "agent",
    "current_turn",
    "many_blocks",
    "opening_prompt",
    "single_block",
    "turn",
]
