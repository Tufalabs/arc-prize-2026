"""I7 replay boundaries from Duck's actual journal-compaction diagnostics."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import pytest
import ullm.backend
import ullm.conv

from duck.agent import DuckAgent
from platypus.diagnostics import BlockStore, Span, SpanLogger, blocks_dir, game_stream_path, read_events
from platypus.llm import DiagnosticTurnManager, DiscardedRound
from platypus.viewer_data import load_game_full


async def _record_journal_run(
    root: Path,
    *,
    legacy: bool,
    twice: bool,
    pending: bool = False,
    cap_retry: bool = False,
    no_op_trims: bool = False,
) -> Path:
    """Record real Duck evictions and serialized ULLM responses, without inference."""
    store = BlockStore(blocks_dir(root))
    with contextlib.closing(SpanLogger(root / "run_spans.jsonl", store)) as run_logger:
        with run_logger.span("infra/run", "journal-test"):
            pass
    stream = game_stream_path(root / "games" / "game123")
    logger = SpanLogger(stream, store)
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=20480,
        compact_to_tokens=15500,
        file_tools="none",
    )
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text=agent._system_prompt_text())),
        elements=agent.messages,
    )
    agent._live_conversation = conversation
    created: list[ullm.conv.LLMResponse] = []
    dropped: list[list[ullm.conv.ConversationElement]] = []
    discarded_retry: list[ullm.conv.LLMResponse] = []

    def response(text: str) -> ullm.conv.LLMResponse:
        item = ullm.conv.LLMResponse(reasoning_content=ullm.conv.StringContent(text=text), status="processed")
        created.append(item)
        return item

    def capture(span: Span, observed: list[ullm.conv.LLMResponse]) -> None:
        manager = DiagnosticTurnManager(conversation=conversation, diag=logger)
        manager._recorded_elements = list(observed)
        manager._discarded_rounds = [
            DiscardedRound(elements=list(elements), reason="context_compaction") for elements in dropped
        ]
        if discarded_retry:
            manager._discarded_rounds.append(
                DiscardedRound(elements=list(discarded_retry), reason="per_turn_cap_lifted")
            )
        manager._record_turn(span)

    with contextlib.closing(logger), logger.span("infra/game", "ls20-test") as game:
        game.artifact("game_metadata", {"number_of_levels": 2, "base_actions_per_level": [2, 2]}, codec="json")
        with game.span("infra/move", "move 0") as move:
            for i in range(2):
                with move.span("llm/turn", "analyzer") as turn:
                    agent.messages.extend(
                        [
                            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=f"old user {i}")),
                            ullm.conv.ResponseTurn(elements=[response(f"old response {i} " + "detail " * 900)]),
                        ]
                    )
                    capture(turn, created[-1:])
            with move.span("llm/turn", "analyzer") as turn:
                current = ullm.conv.ResponseTurn(elements=[response("before first compaction")])
                agent.messages.extend(
                    [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="current user")), current]
                )
                if cap_retry:
                    discarded_retry.append(response("discarded cap retry"))
                agent.response_step = len(created) + 1
                await agent._trim_messages_for_context(agent.messages, enforce=True)
                assert agent._last_trim_fired
                first_after = agent._decision_prompt_tokens(agent.messages)
                second_after: int | None = None
                after_first = response("after first compaction " + ("detail " * 1500 if twice else ""))
                current.elements.append(after_first)
                after_second: ullm.conv.LLMResponse | None = None
                if twice:
                    if no_op_trims:
                        agent.response_step = len(created)
                        await agent._trim_messages_for_context(agent.messages)
                        assert agent._last_trim_fired and not dropped
                    agent.response_step = len(created) + 1
                    await agent._trim_messages_for_context(
                        agent.messages,
                        enforce=True,
                        record_pruned_round=lambda elements: dropped.append(list(elements)),
                    )
                    assert agent._last_trim_fired and dropped
                    second_after = agent._decision_prompt_tokens(agent.messages)
                    after_second = response("after second compaction" + (" detail" * 1500 if no_op_trims else ""))
                    current.elements.append(after_second)
                    if no_op_trims:
                        agent.response_step = len(created)
                        await agent._trim_messages_for_context(agent.messages)
                        assert agent._last_trim_fired
                for item in created:
                    item.token_usage = ullm.backend.TokenUsage(prompt_tokens=123, completion_tokens=10)
                if legacy:
                    after_first.token_usage = ullm.backend.TokenUsage(prompt_tokens=first_after, completion_tokens=10)
                    if twice:
                        assert second_after is not None and after_second is not None
                        after_second.token_usage = ullm.backend.TokenUsage(
                            prompt_tokens=second_after, completion_tokens=10
                        )
                if not pending:
                    capture(turn, created[2:])
    if legacy:
        events = read_events(stream)
        for event in events:
            if event.get("e") == "kv" and event["values"].get("compacted"):
                for key in (
                    "compaction_before_response",
                    "compaction_system_prompt",
                    "context_before",
                    "history_evicted",
                ):
                    event["values"].pop(key, None)
        stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    return root


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("twice", [False, True])
async def test_journal_boundaries_preserve_round_order_and_multiple_trims(
    tmp_path: Path, legacy: bool, twice: bool
) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=legacy, twice=twice)
    payload = load_game_full(root, None, 0)
    sections = [section for event in payload["events"] for section in event.get("transcript_sections", [])]
    order = [
        section["label"] if section["kind"] == "boundary" else section["content"][:24]
        for section in sections
        if section["kind"] == "boundary" or section["label"] == "THINKING"
    ]
    assert order[2:4] == ["before first compaction", "COMPACTION #1"]
    assert order[4].startswith("after first compaction")
    if twice:
        assert order[-2:] == ["COMPACTION #2", "after second compaction"]
    assert len(payload["contexts"]) == (3 if twice else 2)
    boundaries = [s for s in sections if s["kind"] == "boundary"]
    assert all(s["compaction"]["mode"] == "journal" and s["compaction"]["boundary_located"] for s in boundaries)
    assert all("No summary was generated" in s["content"] and "FAILED" not in s["content"] for s in boundaries)
    assert len([s for s in sections if s["label"] == "THINKING"]) == (5 if twice else 4)


@pytest.mark.asyncio
async def test_live_journal_marker_does_not_wait_for_the_turn_artifact(tmp_path: Path) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=True, twice=False, pending=True)
    payload = load_game_full(root, None, 0)
    boundaries = [s for e in payload["events"] for s in e.get("transcript_sections", []) if s["kind"] == "boundary"]
    assert len(boundaries) == 1
    assert boundaries[0]["compaction"]["pending_transcript"]
    assert "still being recorded" in boundaries[0]["content"]


@pytest.mark.asyncio
async def test_discarded_cap_retry_does_not_shift_journal_boundary(tmp_path: Path) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=False, twice=True, cap_retry=True)
    payload = load_game_full(root, None, 0)
    sections = [s for e in payload["events"] for s in e.get("transcript_sections", [])]
    order = [
        s["label"] if s["kind"] == "boundary" else s["content"]
        for s in sections
        if s["kind"] == "boundary" or s["label"] == "THINKING"
    ]
    assert order[2:4] == ["before first compaction", "COMPACTION #1"]
    assert order[4].startswith("after first compaction")
    assert order[-2:] == ["COMPACTION #2", "after second compaction"]
    assert not any("discarded cap retry" in text for text in order)


@pytest.mark.asyncio
async def test_legacy_estimated_tokens_leave_boundary_position_explicitly_unknown(tmp_path: Path) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=False, twice=False)
    stream = game_stream_path(root / "games" / "game123")
    events = read_events(stream)
    for event in events:
        if event.get("e") == "kv" and event["values"].get("compacted"):
            event["values"].pop("compaction_before_response")
    stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    payload = load_game_full(root, None, 0)
    boundaries = [s for e in payload["events"] for s in e.get("transcript_sections", []) if s["kind"] == "boundary"]
    assert len(boundaries) == 1
    assert not boundaries[0]["compaction"]["boundary_located"]
    assert "does not identify the exact response boundary" in boundaries[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_eviction", [False, True])
async def test_no_op_trims_do_not_add_journal_boundaries(tmp_path: Path, explicit_eviction: bool) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=False, twice=True, no_op_trims=True)
    stream = game_stream_path(root / "games" / "game123")
    events = read_events(stream)
    trims = [e["values"] for e in events if e.get("e") == "kv" and e["values"].get("compacted")]
    assert len(trims) == 4
    assert [v["blocks_evicted"] for v in trims] == [2, 0, 0, 0]
    assert trims[2]["rounds_pruned"] > 0
    assert trims[3]["rounds_pruned"] == trims[2]["rounds_pruned"]
    if not explicit_eviction:
        for values in trims:
            values.pop("history_evicted", None)
        stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    payload = load_game_full(root, None, 0)
    order = [
        s["label"] if s["kind"] == "boundary" else s["content"][:24]
        for e in payload["events"]
        for s in e.get("transcript_sections", [])
        if s["kind"] == "boundary" or s["label"] == "THINKING"
    ]
    assert order[2:4] == ["before first compaction", "COMPACTION #1"]
    assert order[-2:] == ["COMPACTION #2", "after second compaction "]
    assert len(payload["contexts"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_artifacts", [1, 2])
@pytest.mark.parametrize("malformed", [False, True])
async def test_missing_turn_artifacts_do_not_shift_later_journal_boundaries(
    tmp_path: Path, missing_artifacts: int, malformed: bool
) -> None:
    root = await _record_journal_run(tmp_path / "run", legacy=False, twice=True, cap_retry=True)
    stream = game_stream_path(root / "games" / "game123")
    events = read_events(stream)
    store = BlockStore(blocks_dir(root))
    omitted = 0
    retained: list[dict[str, Any]] = []
    for event in events:
        if event.get("e") == "artifact" and event.get("name") == "conversation" and omitted < missing_artifacts:
            omitted += 1
            if malformed:
                event["block"] = store.store([], "json")
                retained.append(event)
        else:
            retained.append(event)
    assert omitted == missing_artifacts
    stream.write_text("".join(json.dumps(event) + "\n" for event in retained))
    payload = load_game_full(root, None, 0)
    sections = [s for e in payload["events"] for s in e.get("transcript_sections", [])]
    order = [
        s["label"] if s["kind"] == "boundary" else s["content"][:24]
        for s in sections
        if s["kind"] == "boundary" or s["label"] == "THINKING"
    ]
    assert order[2 - missing_artifacts : 4 - missing_artifacts] == ["before first compaction", "COMPACTION #1"]
    assert order[-2:] == ["COMPACTION #2", "after second compaction"]
    assert all(s["compaction"]["boundary_located"] for s in sections if s["kind"] == "boundary")
