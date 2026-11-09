"""``platypus.reasoning_corpus``: reasoning text read back from a game's diagnostics.

The property that matters is completeness under compaction: a round that
context compaction removed from the live prompt survives only in the
``conversation`` artifact of the turn that generated it, and a reader of
the newest conversation alone loses it -- in generation order, since the
opening read takes the first text and the evicted round is the oldest.
Text long enough to leave the artifact and live in the block store is the
other path that has to hold, since that is where real reasoning lives.
Order is the harder half: a response a lifted per-turn cap discarded
leaves an event between the kept ones but no text of its own here, and the
reader has to keep the kept ones in their place.
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
from duck import DuckAgent
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


def _conversation() -> ullm.conv.Conversation:
    """A conversation with the echo tool, opened by one user prompt."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[_EchoTool()])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    return conversation


async def _play(game: Path, script: list[Any], *, evict_first_turn: bool) -> None:
    """One diagnostic turn per scripted response over ONE growing conversation,
    each turn re-prompted; with ``evict_first_turn`` the second turn's
    before-round hook compacts the first turn out of the live prompt the
    way the duck does -- by deleting it -- so its round survives only in the
    first turn's own artifact."""
    conversation = _conversation()
    diag = SpanLogger(game / "spans.jsonl", BlockStore(game.parent.parent / "diag_blocks"))
    for index, response in enumerate(script):
        if index:
            conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="more")))
        manager = DiagnosticTurnManager(
            response_manager=ullm.manager.ResponseManager(backend=StubBackend.from_script(response)),
            conversation=conversation,
            diag=diag,
            label="play",
        )

        async def evict() -> None:
            """Drop the first user-led block: its prompt, its turn."""
            del conversation.elements[:2]

        if evict_first_turn and index == 1:
            manager.before_round = evict
        await manager.run()
    diag.close()


@pytest.mark.asyncio
async def test_reasoning_survives_compaction_and_the_block_store(tmp_path: Path) -> None:
    """Both turns' reasoning comes back, the evicted one from the artifact of
    the turn that generated it."""
    game = _game_dir(tmp_path)
    first = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=EVICTED), final_content=ullm.conv.StringContent(text="hi")
    )
    second = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=KEPT), final_content=ullm.conv.StringContent(text="done")
    )
    await _play(game, [first, second], evict_first_turn=True)

    texts = reasoning_corpus.game_reasoning(game)
    # Generation order, each round exactly once however many artifacts carry it.
    assert texts == [EVICTED.strip(), KEPT.strip()]
    # What the opening read is: the register is scored on the game's FIRST
    # reasoning, so a surviving later round standing in for an evicted
    # opening misreads the game and silently mis-fits the opening model.
    assert reasoning_corpus.first_reasoning(game) == EVICTED.strip()


@pytest.mark.asyncio
async def test_a_round_pruned_mid_turn_keeps_its_place(tmp_path: Path) -> None:
    """The duck records rounds it prunes from the live turn as a child span
    with its own conversation. Those rounds are older than everything the
    parent still shows, so the reader takes the child in its place -- read
    parent-first, the game's opening would be its second response."""
    game = _game_dir(tmp_path)
    first = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=EVICTED),
        final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "one"}),
    )
    second = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=KEPT), final_content=ullm.conv.StringContent(text="done")
    )
    conversation = _conversation()
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=StubBackend.from_script(first, second)),
        conversation=conversation,
        diag=SpanLogger(game / "spans.jsonl", BlockStore(game.parent.parent / "diag_blocks")),
        label="play",
    )
    duck = DuckAgent(
        reply_cap_tokens=None,
    )
    duck._live_conversation = conversation  # pyright: ignore[reportPrivateUsage]
    rounds = 0

    async def prune() -> None:
        """Before the second round, drop the first one the way the duck does."""
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            turn = conversation.elements[-1]
            assert isinstance(turn, ullm.conv.ResponseTurn)
            dropped = list(turn.elements[:2])  # the response and the tool result it produced
            del turn.elements[:2]
            duck._record_compacted_rounds(  # pyright: ignore[reportPrivateUsage]
                dropped, [ullm.conv.StringContent(text="go")], dropped_after_round=1
            )

    manager.before_round = prune
    await manager.run()
    manager.diag.close()

    assert reasoning_corpus.game_reasoning(game) == [EVICTED.strip(), KEPT.strip()]
    assert reasoning_corpus.first_reasoning(game) == EVICTED.strip()


@pytest.mark.asyncio
async def test_register_scores_the_game_as_one_text(tmp_path: Path) -> None:
    """The game score is the classifier over everything the game reasoned, and reads the register."""
    game = _game_dir(tmp_path)
    blanked_style = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=EVICTED), final_content=ullm.conv.StringContent(text="ok")
    )
    await _play(game, [blanked_style], evict_first_turn=False)
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
async def test_a_cap_lifted_retry_leaves_the_kept_order_intact(tmp_path: Path) -> None:
    """A response the lifted per-turn cap discarded is recorded here as a
    child span with events and no text, so it cannot come back -- but the
    kept responses around it do, in order, and the opening stays the
    opening."""
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

    assert reasoning_corpus.game_reasoning(game) == [texts[0], texts[1], texts[3]]
    assert reasoning_corpus.first_reasoning(game) == texts[0]
