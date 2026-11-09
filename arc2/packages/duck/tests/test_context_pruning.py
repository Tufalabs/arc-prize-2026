"""Round pruning: drop the CURRENT turn's oldest rounds when whole-block
eviction cannot get under budget.

Block eviction never touches the current turn, but the current turn is what
grows — every round appends a response plus its tool results to one block,
with past reasoning re-sent each time. So a long turn outruns it, the prompt
goes out over the model's window, and without pruning the engine rejects it.

Here this is the normal case rather than the exception: a game is one call
into the agent, so most conversations are a single block and eviction has
nothing to evict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import ullm.conv
from duck import prompts
from duck_fixtures import agent, current_turn, many_blocks, opening_prompt, single_block, turn


async def test_rounds_are_pruned_when_blocks_cannot_help() -> None:
    pruning = agent()
    single_block(pruning)
    before = len(current_turn(pruning).elements)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    assert len(current_turn(pruning).elements) < before
    assert pruning._decision_prompt_tokens(pruning.messages) <= pruning._context_budget_tokens()


async def test_pruning_keeps_rounds_whole() -> None:
    """A tool call whose result went missing is malformed input on the next
    request, so a response and its results leave together."""
    pruning = agent()
    single_block(pruning)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    remaining = current_turn(pruning).elements
    # Unguarded: pruning the turn away entirely would satisfy "no orphan
    # result" vacuously, and that is a different bug, not a pass.
    assert remaining, "pruning emptied the turn — nothing left to keep whole"
    assert isinstance(remaining[0], ullm.conv.LLMResponse), "a turn must not start with an orphan result"
    assert len(remaining) % 2 == 0, [type(e).__name__ for e in remaining]


async def test_the_model_is_told_what_it_lost() -> None:
    """Pruning is silent PARTIAL loss mid-turn — unlike a wholesale reset — so
    the model would otherwise keep reasoning from work it no longer has."""
    pruning = agent()
    single_block(pruning)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    assert prompts.DuckPrompts().rounds_pruned_marker in opening_prompt(pruning).content.to_string()


async def test_the_note_is_rewritten_not_stacked() -> None:
    """Pruning fires repeatedly within a turn; the note reports the running
    total once rather than accumulating a copy per pass. It is also prompt text,
    so stacking it would eat the budget it exists to describe."""
    pruning = agent()
    single_block(pruning)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    current_turn(pruning).elements.extend(turn(rounds=12).elements)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    text = opening_prompt(pruning).content.to_string()
    assert text.count(prompts.DuckPrompts().rounds_pruned_marker) == 1, text


async def test_an_impossible_budget_raises_rather_than_overflowing() -> None:
    """Nothing droppable is left and it still does not fit: the user prompt
    alone exceeds the budget. Sending anyway just hands the engine a prompt it
    must reject, so the failure belongs here where the cause is legible."""
    doomed = agent()
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="huge " * 5000))
    doomed.messages = [prompt, turn(rounds=8)]
    with pytest.raises(RuntimeError, match="context budget"):
        await doomed._trim_messages_for_context(doomed.messages, enforce=True)
    # Pruning pared the turn away but NEVER touched the user prompt — at worst
    # it leaves [user prompt] and raises rather than dropping it.
    assert prompt in doomed.messages
    assert not any(isinstance(e, ullm.conv.ResponseTurn) and e.elements for e in doomed.messages)


async def test_retention_trim_neither_prunes_nor_raises() -> None:
    """The post-turn retention pass is not about to send anything, so it has no
    business dropping a turn's rounds or failing the game."""
    retaining = agent()
    single_block(retaining, rounds=4)
    kept = len(current_turn(retaining).elements)
    await retaining._trim_messages_for_context(retaining.messages)  # enforce defaults False
    assert len(current_turn(retaining).elements) == kept


async def test_pruning_leaves_headroom_to_avoid_thrash() -> None:
    """Dropping the bare minimum leaves the prompt at the budget edge, so the
    next round re-crosses and prunes again — and each prune shifts the prefix
    after the user turn, busting the turn body's KV cache. Pruning drops at
    least ``evict_turns`` rounds so a re-prune is once per that many rounds
    rather than every round."""
    thrashing = agent()  # default evict_turns headroom
    single_block(thrashing, rounds=12)
    budget = thrashing._context_budget_tokens()
    await thrashing._prune_current_turn(thrashing.messages, budget, dropped_into=[])
    assert thrashing._decision_prompt_tokens(thrashing.messages) <= budget
    current_turn(thrashing).elements.extend(turn(rounds=1).elements)
    assert thrashing._decision_prompt_tokens(thrashing.messages) <= budget, (
        "one round after pruning already exceeds budget — no headroom, will thrash"
    )


async def test_no_headroom_pruning_when_already_under_budget() -> None:
    """Headroom applies only when pruning was actually needed; a turn that
    already fits must be left completely alone."""
    idle = agent()
    single_block(idle, rounds=1)
    before = len(current_turn(idle).elements)
    await idle._trim_messages_for_context(idle.messages, enforce=True)
    assert len(current_turn(idle).elements) == before


async def test_pruning_drops_the_pruned_rounds_pending_bounds() -> None:
    """Round pruning and the post-turn usage check interact: pruning removes
    responses that already had a pre-call bound recorded, so those bounds must
    leave the live pairing too, or it sees more bounds than responses.

    ``_pending_bounds`` is 1:1 with the turn's responses at prune time, oldest
    first; pruning removes the oldest rounds, so the front bounds are moved —
    to the deferred queue with their responses, not away.
    """
    counting = agent()
    single_block(counting, rounds=12)
    built = current_turn(counting)
    responses_before = sum(1 for e in built.elements if isinstance(e, ullm.conv.LLMResponse))
    counting._pending_bounds.extend(range(responses_before))

    await counting._trim_messages_for_context(counting.messages, enforce=True)

    responses_after = sum(1 for e in current_turn(counting).elements if isinstance(e, ullm.conv.LLMResponse))
    assert len(counting._pending_bounds) == responses_after, (
        f"{len(counting._pending_bounds)} bounds left for {responses_after} surviving responses"
    )
    # The surviving bounds are the NEWEST — the pruned (oldest) rounds' bounds went.
    assert counting._pending_bounds == list(range(responses_before - responses_after, responses_before))
    assert len(counting._deferred_usage_checks) == responses_before - responses_after


async def test_a_compaction_is_logged_as_a_turn_of_its_own(tmp_path: Path) -> None:
    """A trim that prunes leaves a child llm/turn span named "compacted":
    the opening prompt as it read then plus the dropped rounds, as a
    conversation artifact the viewer already knows how to render, with the
    dropped rounds' response events and token spend on it. Without this the
    turn is written down once at its close, and rounds pruned mid-turn --
    the normal case here, a game being one long turn -- would vanish from
    both the transcript and the turn's token total."""
    from platypus.diagnostics import BlockStore, SpanLogger, build_span_tree, read_events

    pruning = agent()
    single_block(pruning, rounds=12)
    before = list(current_turn(pruning).elements)
    logger = SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks"))
    with logger.span("llm/turn", "analyzer"):
        await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    logger.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    (segment,) = turn.children
    assert (segment.kind, segment.name) == ("llm/turn", "compacted")

    responses = [e for e in segment.events if e["kind"] == "llm/response"]
    dropped_rounds = 12 - len(current_turn(pruning).elements) // 2
    assert len(responses) == dropped_rounds
    assert segment.kv["rounds_dropped"] == dropped_rounds
    # One trim, so every dropped round left at the same point in the turn's
    # life: all 12 of its rounds had run.
    assert segment.kv["dropped_after_round"] == 12

    # The artifact is a full conversation: the opening prompt as the model
    # read it, then the dropped rounds -- nothing lost, nothing invented.
    blocks = BlockStore(tmp_path / "blocks")
    (artifact,) = [a for a in segment.artifacts if a["name"] == "conversation"]
    structure = blocks.load(artifact["block"], "json")
    kinds = [element["kind"] for element in structure["elements"]]
    assert kinds == ["user", "turn"]
    segment_elements = structure["elements"][1]["elements"]
    assert len(segment_elements) == len(before) - len(current_turn(pruning).elements)
    assert segment_elements[0]["kind"] == "response"


async def test_consecutive_feedback_messages_are_one_eviction_and_diagnostic_block(
    tmp_path: Path,
) -> None:
    """Show then failed-check user messages stay together through eviction,
    pruning notes, and the archived conversation for dropped rounds."""
    from platypus.diagnostics import BlockStore, SpanLogger, build_span_tree, read_events

    pruning = agent()
    old_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="old block"))
    show_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="show feedback"))
    check_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="check feedback"))
    pruning.messages = [old_prompt, turn(rounds=6), show_prompt, check_prompt, turn(rounds=12)]
    assert pruning._block_bounds(pruning.messages) == [0, 2]

    logger = SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks"))
    with logger.span("llm/turn", "analyzer"):
        await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    logger.close()

    assert pruning.messages[:2] == [show_prompt, check_prompt]
    marker = prompts.DuckPrompts().rounds_pruned_marker
    assert marker not in show_prompt.content.to_string()
    assert marker in check_prompt.content.to_string()
    (root,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    (segment,) = root.children
    blocks = BlockStore(tmp_path / "blocks")
    (artifact,) = [item for item in segment.artifacts if item["name"] == "conversation"]
    structure = blocks.load(artifact["block"], "json")
    assert [element["kind"] for element in structure["elements"]] == ["user", "user", "turn"]


async def test_a_null_stream_compaction_records_nothing_and_changes_nothing() -> None:
    """The same trim outside any span is inert -- pruning still happens,
    nothing is written anywhere. Replay references depend on this."""
    pruning = agent()
    single_block(pruning, rounds=12)
    before = len(current_turn(pruning).elements)
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    assert len(current_turn(pruning).elements) < before


async def test_headroom_never_takes_the_last_round() -> None:
    """Headroom pruning is an optimization on top of an achieved fit, so it
    must not be the thing that empties a turn: the last round is the model's
    freshest work, and losing it costs more than the re-prune it saves. Only
    an actual inability to fit may clear the turn."""
    pruning = agent()
    single_block(pruning, rounds=3)
    # A target one round + prompt fits under: the must-fit loop stops with
    # exactly one round left, and headroom then wants more.
    one_round = pruning._decision_prompt_tokens([pruning.messages[0], turn(rounds=1)])
    await pruning._prune_current_turn(pruning.messages, one_round, dropped_into=[])
    assert len(current_turn(pruning).elements) == 2, "headroom emptied the turn"


async def test_a_trim_declares_the_history_rewrite() -> None:
    """Eviction and pruning rewrite history the model was already sent, so
    they bump the agent's history epoch (ullm.continuation): the next call
    renders afresh by declaration rather than tripping exact history."""
    pruning = agent()
    single_block(pruning)
    before = pruning.history_epoch
    await pruning._trim_messages_for_context(pruning.messages, enforce=True)
    # One declaration per trim, however many blocks and rounds it removed:
    # the epoch is a version of sent history, not a count of edits.
    assert pruning.history_epoch == before + 1


async def test_the_declaration_precedes_the_trims_own_recounts() -> None:
    """A firing trim re-counts the prompt after every block it drops; under
    exact history each count continues the served ids, so the rewrite must
    be declared before the first edit — a count over a shortened history at
    the old epoch is exactly the drift the promise refuses. Token-native
    mode is what routes the recounts through the backend at all."""
    pruning = agent(token_native_context=True)
    many_blocks(pruning)
    live = ullm.conv.Conversation(elements=pruning.messages, history_epoch=pruning.history_epoch)
    seen: list[tuple[int, int]] = []

    class _Counter:
        async def token_count(self, params: Any, conversation: ullm.conv.Conversation) -> int:
            seen.append((len(conversation.elements), conversation.history_epoch))
            return 10**6 if len(conversation.elements) > 2 else 100  # over budget until two blocks are gone

    pruning._live_backend = _Counter()
    pruning._live_params = object()
    pruning._live_conversation = live
    before = pruning.history_epoch
    await pruning._trim_messages_for_context(pruning.messages, enforce=False)
    assert pruning.history_epoch == before + 1
    first_len = seen[0][0]
    for length, epoch in seen[1:]:
        if length < first_len:
            assert epoch > before, f"counted a shortened history ({length} < {first_len}) at the old epoch: {seen}"


def test_the_agent_absorbs_declarations_the_manager_made() -> None:
    """The live conversation can be ahead of the agent: the turn manager
    declares its own rollbacks on it. A declaration from the agent's side
    then moves past whichever is newer, so neither counter ever goes
    backwards and both agree afterwards."""
    pruning = agent()
    pruning.history_epoch = 2
    live = ullm.conv.Conversation(history_epoch=5)
    pruning._live_conversation = live
    pruning._mark_history_rewritten()
    assert pruning.history_epoch == live.history_epoch == 6
