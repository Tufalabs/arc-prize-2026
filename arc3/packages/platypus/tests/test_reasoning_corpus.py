"""``platypus.reasoning_corpus``: reasoning text read back from a game's diagnostics.

The property that matters is completeness under compaction: a round that
context compaction removed before the turn closed survives only in the
``discarded_rounds`` artifact, and a reader of the surviving conversation
alone loses it -- in generation order, since the opening read takes the
first text and an evicted opening is stored after the rounds that outlived
it. Text long enough to leave the artifact and live in the
block store is the other path that has to hold, since that is where real
reasoning lives. Order is the harder half: a discarded round can be older
than everything kept (compaction) or sit between kept ones (a response a
lifted per-turn cap threw away), and the reader has to place both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import ullm.backend
import ullm.conv
import ullm.manager
import ullm.reasoning_register
import ullm.tool
from ullm.backends.stub import StubBackend

from platypus import reasoning_corpus
from platypus.diagnostics import BlockStore, SpanLogger
from platypus.llm import DiagnosticTurnManager

# Past capture's inline limit, so it is stored as a block and read back through the resolver.
EVICTED = "We need find the exit first. Need check every door before moving. " * 4
KEPT = "Let me think about what the corridor is actually doing here, and whether the door matters at all. " * 3


@dataclass
class _EchoTool(ullm.tool.Tool):
    """The smallest tool that lets a turn run a second round."""

    name: str = "echo"
    description: str = "echoes text"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    )

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        return ullm.conv.StringContent(text=str(arguments.get("text", "")))


def _game_dir(tmp_path: Path) -> Path:
    """A ``games/<run_id>/`` layout with the job's block store two levels up."""
    game = tmp_path / "games" / "g1"
    game.mkdir(parents=True)
    return game


async def _play(game: Path, script: list[Any], *, prune_first_round: bool) -> None:
    """One diagnostic turn over a scripted backend, optionally compacting its first round away."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[_EchoTool()])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=StubBackend.from_script(*script)),
        conversation=conversation,
        diag=SpanLogger(game / "spans.jsonl", BlockStore(game.parent.parent / "diag_blocks")),
        label="play",
    )
    rounds = 0

    async def prune() -> None:
        """Compact the first round out once the second is about to start."""
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            # The first round is its response and the tool result it produced.
            turn = manager.conversation.last_turn
            doomed = turn.elements[:2]
            manager.record_context_pruned_round(doomed)
            del turn.elements[:2]

    if prune_first_round:
        manager.before_round = prune
    await manager.run()
    manager.diag.close()


@pytest.mark.asyncio
async def test_reasoning_survives_compaction_and_the_block_store(tmp_path: Path) -> None:
    """Both rounds' reasoning comes back, the evicted one from discarded_rounds."""
    game = _game_dir(tmp_path)
    first = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=EVICTED),
        final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "hi"}),
    )
    second = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=KEPT), final_content=ullm.conv.StringContent(text="done")
    )
    await _play(game, [first, second], prune_first_round=True)

    texts = reasoning_corpus.game_reasoning(game)
    # Generation order, each round exactly once however many artifacts carry
    # it: the evicted round was written first and the span stores it last.
    assert texts == [EVICTED.strip(), KEPT.strip()]
    # What the opening read is: the register is scored on the game's FIRST
    # reasoning, so a surviving later round standing in for an evicted
    # opening misreads the game and silently mis-fits the opening model.
    assert reasoning_corpus.first_reasoning(game) == EVICTED.strip()


@pytest.mark.asyncio
async def test_register_scores_the_game_as_one_text(tmp_path: Path) -> None:
    """The game score is the classifier over everything the game reasoned, and reads the register."""
    game = _game_dir(tmp_path)
    blanked_style = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=EVICTED), final_content=ullm.conv.StringContent(text="ok")
    )
    await _play(game, [blanked_style], prune_first_round=False)
    score = reasoning_corpus.game_register(game)
    assert score == ullm.reasoning_register.blanked_score(reasoning_corpus.game_reasoning(game))
    assert score > 0.5


def test_missing_game_is_empty_not_an_error(tmp_path: Path) -> None:
    """A game directory with no stream yet scores as uninformative."""
    game = _game_dir(tmp_path)
    assert reasoning_corpus.game_reasoning(game) == []
    assert reasoning_corpus.game_register(game) == 0.5


def _capped(
    reasoning: str, content: ullm.conv.Content, completion: int, stop: ullm.conv.StopReason
) -> ullm.conv.LLMResponse:
    """A scripted response with exact usage, so the per-turn cap can bite."""
    return ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=reasoning),
        final_content=content,
        stop_reason=stop,
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=10, cached_tokens=2, completion_tokens=completion, num_requests=1
        ),
    )


@pytest.mark.asyncio
async def test_a_cap_lifted_retry_keeps_its_place(tmp_path: Path) -> None:
    """A response the lifted per-turn cap discarded was generated AFTER the
    kept responses before it, so it comes back between them -- and the
    opening stays the opening. Storage order would put it first."""
    game = _game_dir(tmp_path)

    def call(text: str) -> ullm.conv.ToolCallContent:
        """Ask the echo tool for another round."""
        return ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": text})

    texts = [
        "Opening thought about the exit.",
        "Second thought about the door.",
        "Third, cut off by the cap.",
        "Fourth, after the cap lifted.",
    ]
    backend = StubBackend.from_script(
        _capped(texts[0], call("one"), 60, "tool_use"),
        _capped(texts[1], call("two"), 60, "tool_use"),
        _capped(texts[2], call("discarded"), 30, "max_tokens"),
        _capped(texts[3], ullm.conv.StringContent(text="done"), 60, "stop"),
    )
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[_EchoTool()])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(
            backend=backend, sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=100)
        ),
        conversation=conversation,
        diag=SpanLogger(game / "spans.jsonl", BlockStore(game.parent.parent / "diag_blocks")),
        label="play",
    )
    manager.max_generated_tokens_per_turn = 150
    manager.per_turn_cap_lifted = lambda: backend.total_usage.completion_tokens >= 150
    await manager.run()
    manager.diag.close()

    assert reasoning_corpus.game_reasoning(game) == texts
    assert reasoning_corpus.first_reasoning(game) == texts[0]
