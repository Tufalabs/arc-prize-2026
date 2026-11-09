"""``keep_task_prompt``: eviction never takes the puzzle itself.

The task prompt — the legend and the pictures — is always the
conversation's first element, and every re-entry opens a new block behind
it. Plain eviction removes blocks from the oldest end, so the puzzle is
the first thing a re-entered game loses when it compacts. Behind the flag,
the first block's turn is evicted like any other but its prompt stays, and
later evictions take the block behind it.
"""

from __future__ import annotations

from pathlib import Path

import ullm.conv
from duck_fixtures import agent, many_blocks

from platypus.diagnostics import BlockStore, SpanLogger, build_span_tree, read_events


async def test_the_task_prompt_survives_eviction() -> None:
    keeping = agent(keep_task_prompt=True)
    many_blocks(keeping)
    task_prompt = keeping.messages[0]
    await keeping._trim_messages_for_context(keeping.messages, enforce=True)
    assert keeping.messages[0] is task_prompt, "the puzzle left the conversation"
    assert keeping._decision_prompt_tokens(keeping.messages) <= keeping._context_budget_tokens()


async def test_the_task_prompts_own_turn_goes_first_and_the_head_stays_a_bare_prompt() -> None:
    """What is kept is the pictures, not the first turn's reasoning: the
    block's tail leaves before any later block does, so the model then
    reads the puzzle and, directly after it, the newest prompt."""
    keeping = agent(keep_task_prompt=True)
    many_blocks(keeping)
    await keeping._trim_messages_for_context(keeping.messages, enforce=True)
    assert isinstance(keeping.messages[0], ullm.conv.UserPrompt)
    assert isinstance(keeping.messages[1], ullm.conv.UserPrompt), "the task prompt's turn should have gone first"


async def test_the_flag_off_evicts_the_task_prompt_as_before() -> None:
    """Evict-everything is the baseline the sweep compares against, so it
    has to stay reachable byte for byte."""
    plain = agent()
    many_blocks(plain)
    task_prompt = plain.messages[0]
    await plain._trim_messages_for_context(plain.messages, enforce=True)
    assert plain.messages[0] is not task_prompt


async def test_the_compaction_record_counts_blocks_that_left(tmp_path: Path) -> None:
    """Keeping the task prompt evicts its turn without the block leaving, and
    the record must not count that as a block: ``blocks_evicted`` is what
    the trim removed from the conversation's block structure, so it equals
    the blocks before minus the blocks after."""
    keeping = agent(keep_task_prompt=True)
    many_blocks(keeping)
    before = len(keeping._block_bounds(keeping.messages))
    diag = SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks"))
    with diag.span("duck/analyze"):
        await keeping._trim_messages_for_context(keeping.messages, enforce=True)
    diag.close()
    (span,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    after = len(keeping._block_bounds(keeping.messages))
    assert span.kv["compacted"] is True
    assert span.kv["blocks_evicted"] == before - after
