"""Round pruning: drop the CURRENT turn's oldest rounds when whole-block
eviction cannot get under budget.

Block eviction never touches the current turn, but the current turn is what
grows — every round appends a response plus its tool results to one block,
with past reasoning re-sent each time. So a long turn outruns it, the prompt
goes out over the model's window, and without pruning the engine rejects it
and the nudge path wipes the whole conversation back to a fresh state prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.conv

from duck import DuckAgent, prompts

# The system prompt and tool schema are counted and cannot be evicted
# (~12.4k decision-ruler tokens before any conversation exists — the ruler
# charges text its character count), so a test window must clear that floor
# or every case degenerates into the impossible-budget branch.
WINDOW = 20480  # budget = 20480 - 512 (reply reserve) - 512 (margin) = 19456
# Between the ~12.4k floor and the budget, with room for the headroom
# rounds (evict_turns) on top of the floor, so pruning has a target it can
# land on while leaving the turn non-empty (~909 decision tokens per
# fixture round).
TARGET = 15500


def _turn(rounds: int) -> ullm.conv.ResponseTurn:
    """`rounds` complete rounds: a response and the tool result answering it."""
    elements: list[Any] = []
    for i in range(rounds):
        elements.append(ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=f"thinking {i} " * 40)))
        elements.append(
            ullm.conv.ToolResult(
                call_id=f"c{i}", tool_name="python", content=ullm.conv.StringContent(text=f"result {i} " * 40)
            )
        )
    return ullm.conv.ResponseTurn(elements=elements, status="intermediate")


def _current_turn(agent: DuckAgent) -> ullm.conv.ResponseTurn:
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    return turn


def _opening_prompt(agent: DuckAgent) -> ullm.conv.UserPrompt:
    prompt = agent.messages[0]
    assert isinstance(prompt, ullm.conv.UserPrompt)
    return prompt


def _agent() -> DuckAgent:
    # No image: keeps the legacy estimator's arithmetic easy to reason about.
    # No file suite: its schemas and prompt lines are a fixed cost the
    # compactor can never reclaim, and at this deliberately small window
    # that floor is a large share of the room the sawtooth has. What these
    # tests measure is the sawtooth, not the fixed cost of a surface -- the
    # shipped window has room for both.
    return DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        multimodal_context="",
        context_window_tokens=WINDOW,
        compact_to_tokens=TARGET,
        file_tools="none",
    )


def _loaded(agent: DuckAgent, rounds: int = 12) -> None:
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), _turn(rounds)]


async def test_rounds_are_pruned_when_blocks_cannot_help() -> None:
    agent = _agent()
    _loaded(agent)
    before = len(_current_turn(agent).elements)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    assert len(_current_turn(agent).elements) < before
    assert agent._decision_prompt_tokens(agent.messages) <= agent._context_budget_tokens()


async def test_pruning_keeps_rounds_whole() -> None:
    """A tool_use whose result went missing is malformed input on the next
    request, so a response and its results leave together."""
    agent = _agent()
    _loaded(agent)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    remaining = _current_turn(agent).elements
    # Unguarded: pruning the turn away entirely would satisfy "no orphan
    # result" vacuously, and that is a different bug, not a pass.
    assert remaining, "pruning emptied the turn — nothing left to keep whole"
    assert isinstance(remaining[0], ullm.conv.LLMResponse), "a turn must not start with an orphan result"
    # Rounds leave in pairs, so what survives is still whole rounds.
    assert len(remaining) % 2 == 0, [type(e).__name__ for e in remaining]


async def test_the_model_is_told_what_it_lost() -> None:
    """Pruning is silent PARTIAL loss mid-turn — unlike the nudge path's
    wholesale reset — so the model would keep reasoning from work it no
    longer has."""
    agent = _agent()
    _loaded(agent)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    assert prompts.DuckPrompts().rounds_pruned_marker in _opening_prompt(agent).content.to_string()


async def test_the_note_is_rewritten_not_stacked() -> None:
    """Pruning fires repeatedly within a turn; the note reports the running
    total once rather than accumulating a copy per pass. It is also prompt
    text, so stacking it would eat the budget it exists to describe."""
    agent = _agent()
    _loaded(agent)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    _current_turn(agent).elements.extend(_turn(rounds=12).elements)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    text = _opening_prompt(agent).content.to_string()
    assert text.count(prompts.DuckPrompts().rounds_pruned_marker) == 1, text


async def test_an_impossible_budget_raises_rather_than_overflowing() -> None:
    """Nothing droppable is left and it still does not fit: the user prompt
    alone exceeds the budget. Sending anyway just hands the engine a prompt
    it must reject, so the failure belongs here where the cause is legible."""
    agent = _agent()
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="huge " * 5000))
    agent.messages = [prompt, _turn(rounds=8)]
    with pytest.raises(RuntimeError, match="context budget"):
        await agent._trim_messages_for_context(agent.messages, enforce=True)
    # Pruning pared the turn away but NEVER touched the user prompt — at worst
    # it leaves [user prompt] and raises rather than dropping it.
    assert prompt in agent.messages
    assert not any(isinstance(e, ullm.conv.ResponseTurn) and e.elements for e in agent.messages)


async def test_retention_trim_neither_prunes_nor_raises() -> None:
    """The post-turn retention pass is not about to send anything, so it has
    no business dropping a turn's rounds or failing the game."""
    agent = _agent()
    _loaded(agent, rounds=4)
    kept = len(_current_turn(agent).elements)
    await agent._trim_messages_for_context(agent.messages)  # enforce defaults False
    assert len(_current_turn(agent).elements) == kept


async def test_pruning_leaves_headroom_to_avoid_thrash() -> None:
    """Dropping the bare minimum leaves the prompt at the budget edge, so the
    next round re-crosses and prunes again — and each prune shifts the prefix
    after the user turn, busting the turn body's KV cache. Pruning drops at
    least evict_turns rounds so re-prune is once per evict_turns rounds, not
    every round (same intent as block eviction's chunking)."""
    agent = _agent()  # evict_turns defaults to 2
    _loaded(agent, rounds=12)
    budget = agent._context_budget_tokens()
    await agent._prune_current_turn(agent.messages, budget)
    # After pruning fires, the prompt sits BELOW budget by at least ~one round,
    # so appending one more round does not immediately re-cross.
    assert agent._decision_prompt_tokens(agent.messages) <= budget
    _current_turn(agent).elements.extend(_turn(rounds=1).elements)
    assert agent._decision_prompt_tokens(agent.messages) <= budget, (
        "one round after pruning already exceeds budget — no headroom, will thrash"
    )


async def test_no_headroom_pruning_when_already_under_budget() -> None:
    """Headroom applies only when pruning was actually needed; a turn that
    already fits must be left completely alone."""
    agent = _agent()
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), _turn(rounds=1)]
    before = len(_current_turn(agent).elements)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    assert len(_current_turn(agent).elements) == before  # untouched


async def test_pruning_drops_the_pruned_rounds_pending_bounds() -> None:
    """Round pruning and the post-turn usage check interact: pruning removes
    responses that already had a pre-call bound recorded, so those bounds must
    go too. Otherwise the post-turn pairing check sees more bounds than
    responses and hard-errors, failing the move.

    _pending_bounds is 1:1 with the turn's responses at prune time, oldest
    first; pruning removes the oldest rounds, so the front bounds are dropped.
    """
    agent = _agent()
    turn = _turn(rounds=12)
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), turn]
    # One recorded pre-call bound per existing response, as before_round would.
    responses_before = sum(1 for e in turn.elements if isinstance(e, ullm.conv.LLMResponse))
    agent._pending_bounds.extend(range(responses_before))

    await agent._trim_messages_for_context(agent.messages, enforce=True)

    responses_after = sum(1 for e in _current_turn(agent).elements if isinstance(e, ullm.conv.LLMResponse))
    assert len(agent._pending_bounds) == responses_after, (
        f"{len(agent._pending_bounds)} bounds left for {responses_after} surviving responses — "
        f"the pairing the usage check asserts is broken"
    )
    # The surviving bounds are the NEWEST — the pruned (oldest) rounds' bounds went.
    assert agent._pending_bounds == list(range(responses_before - responses_after, responses_before))


async def test_pruned_rounds_are_handed_to_the_active_turn_recorder() -> None:
    """Every current-turn element leaves through the scoped callback or survives."""
    agent = _agent()
    _loaded(agent, rounds=12)
    before = list(_current_turn(agent).elements)
    recorded: list[list[ullm.conv.ConversationElement]] = []

    await agent._trim_messages_for_context(
        agent.messages,
        enforce=True,
        record_pruned_round=lambda elements: recorded.append(elements),
    )

    assert recorded, "pruning ran but handed no removed rounds to diagnostics"
    handed = [element for round_elements in recorded for element in round_elements]
    assert handed + list(_current_turn(agent).elements) == before
    assert all(isinstance(round_elements[0], ullm.conv.LLMResponse) for round_elements in recorded)


if __name__ == "__main__":
    pytest.main([__file__])


async def test_a_trim_declares_the_history_rewrite() -> None:
    """Eviction and pruning rewrite history the model was already sent, so
    they bump the agent's history epoch (ullm.continuation): the next call
    renders afresh by declaration rather than tripping exact history."""
    agent = _agent()
    _loaded(agent)
    before = agent.history_epoch
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    # One declaration per trim, however many blocks and rounds it removed:
    # the epoch is a version of sent history, not a count of edits.
    assert agent.history_epoch == before + 1


async def test_the_declaration_precedes_the_trims_first_edit() -> None:
    """A firing trim rewrites history the model was already sent, and the
    next request continues the served ids — so the rewrite is declared on
    the LIVE conversation before the trim's first edit (the memory-snapshot
    refresh that mutates the system prefix), not after the loops. A
    shortened history at the old epoch is exactly the drift the exact-
    history promise refuses."""
    agent = _agent()
    _loaded(agent)
    live = ullm.conv.Conversation(elements=agent.messages, history_epoch=agent.history_epoch)
    agent._live_conversation = live
    before = agent.history_epoch
    epochs_at_refresh: list[int] = []
    refresh = agent._refresh_memory_prompt_snapshot

    def spying_refresh(retained_elements: list[ullm.conv.ConversationElement] | None = None) -> None:
        epochs_at_refresh.append(live.history_epoch)
        refresh(retained_elements)

    agent._refresh_memory_prompt_snapshot = spying_refresh  # pyright: ignore[reportAttributeAccessIssue] — spy on the trim's first mutation
    await agent._trim_messages_for_context(agent.messages, enforce=False)
    assert agent.history_epoch == before + 1 and live.history_epoch == before + 1
    assert epochs_at_refresh, "fixture: the trim must have fired and refreshed the snapshot"
    assert all(epoch > before for epoch in epochs_at_refresh), (
        f"the trim edited the conversation before declaring the rewrite: {epochs_at_refresh}"
    )
