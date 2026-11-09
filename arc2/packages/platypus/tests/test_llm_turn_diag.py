"""``DiagnosticTurnManager``: the llm/turn span — per-response events,
token attribution, kernel-exec events, conversation artifact."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import ullm.backend
import ullm.conv
import ullm.manager
import ullm.tool
from ullm.backends.stub import StubBackend

from platypus.chunks import ViewContext, chunk_for
from platypus.diagnostics import NULL_SPAN, BlockStore, SpanLogger, build_span_tree, current_span, read_events
from platypus.llm import DiagnosticTurnManager, check_continuation_prefill


@dataclass
class _EchoTool(ullm.tool.Tool):
    name: str = "echo"
    description: str = "echoes text"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )
    execution_records: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        self.execution_records.append({"t": 1234.5, "duration_s": 0.25, "is_error": False, "fresh_kernel": True})
        return ullm.conv.StringContent(text=arguments["text"])

    def pop_execution_records(self) -> list[dict[str, Any]]:
        records = self.execution_records
        self.execution_records = []
        return records


def _turn_manager(tmp_path: Path, backend: StubBackend, tools: list[ullm.tool.Tool]) -> DiagnosticTurnManager:
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=tools)
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    return DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=backend),
        conversation=conversation,
        diag=SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks")),
        label="play",
    )


async def test_llm_turn_span_records_responses_kernels_and_conversation(tmp_path: Path) -> None:
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "hi"})),
        "final answer",
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert (turn.kind, turn.name, turn.outcome) == ("llm/turn", "play", "ok")

    responses = [e for e in turn.events if e["kind"] == "llm/response"]
    kernels = [e for e in turn.events if e["kind"] == "kernel/exec"]
    assert len(responses) == 2 and len(kernels) == 1
    assert responses[0]["values"]["n_tool_calls"] == 1
    assert responses[1]["values"]["stop_reason"] == "stop"
    assert kernels[0]["t"] == 1234.5 and kernels[0]["values"]["duration_s"] == 0.25

    # The span's token attribution equals what the backend reports.
    usage = backend.total_usage
    assert turn.self_generated == usage.completion_tokens > 0
    assert turn.self_uncached == usage.prompt_tokens - usage.cached_tokens > 0

    # The conversation artifact rehydrates: system prompt, user prompt,
    # the tool round and the final response, with bookkeeping.
    blocks = BlockStore(tmp_path / "blocks")
    (conversation_artifact,) = [a for a in turn.artifacts if a["name"] == "conversation"]
    structure = blocks.load(conversation_artifact["block"], "json")
    assert structure["system"]["tools"][0]["name"] == "echo"
    kinds = [element["kind"] for element in structure["elements"]]
    assert kinds == ["user", "turn"]
    turn_elements = structure["elements"][1]["elements"]
    assert [element["kind"] for element in turn_elements] == ["response", "tool_result", "response"]
    assert turn_elements[0]["usage"]["completion_tokens"] > 0
    assert turn_elements[2]["final"] == {"c": "str", "text": "final answer"}


async def test_llm_turn_span_on_midturn_crash(tmp_path: Path) -> None:
    """A backend failure mid-turn still yields a closed llm/turn span with
    outcome error and the completed responses recorded."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "hi"})),
        on_exhausted="raise",  # the second LLM call blows up
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    with pytest.raises(Exception, match="exhausted"):
        await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert turn.closed and turn.outcome == "error"
    assert len([e for e in turn.events if e["kind"] == "llm/response"]) == 1
    assert turn.self_generated == backend.total_usage.completion_tokens > 0
    assert any(a["name"] == "conversation" for a in turn.artifacts)


async def test_null_stream_is_plain_turn_manager() -> None:
    backend = StubBackend.from_script("just text")
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=backend), conversation=conversation
    )
    assert not manager.diag.active  # no stream_path: every diag call is a no-op
    await manager.run()
    assert conversation.last_turn.last_response.final_content.to_string() == "just text"
    # The turn ran to completion and left no span behind — instrumented code
    # takes the same path whether diagnostics are on or off.
    assert current_span() is NULL_SPAN


async def test_episode_timeout_during_writer_cleanup_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import platypus.live

    manager = _turn_manager(tmp_path, StubBackend.from_script("unused"), [])

    async def play() -> None:
        async with asyncio.timeout(None) as timer:
            writer_started = asyncio.Event()

            async def writer() -> None:
                try:
                    writer_started.set()
                    await asyncio.Future()
                finally:
                    timer.reschedule(asyncio.get_running_loop().time())
                    await asyncio.sleep(0.01)

            async def turn_body(self: Any) -> None:
                await writer_started.wait()

            def start_writer(*args: Any, **kwargs: Any) -> asyncio.Task[None]:
                return asyncio.create_task(writer())

            monkeypatch.setattr(platypus.live, "start_live_writer", start_writer)
            monkeypatch.setattr(ullm.manager.TurnManager, "_run", turn_body)
            await manager.run()

    # A child task isolates timeout cancellation from pytest's own task.
    with pytest.raises(TimeoutError):
        await asyncio.create_task(play())
    manager.diag.close()
    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert turn.closed
    assert any(artifact["name"] == "conversation" for artifact in turn.artifacts)


@dataclass
class _FailingTool(ullm.tool.Tool):
    """A tool whose call fails the way a kernel error does: an ordinary
    result the model reads, flagged so the record can count it."""

    name: str = "boom"
    description: str = "always fails"
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        del arguments
        self.is_error = True
        return ullm.conv.StringContent(text="Traceback ...")


async def test_every_model_visible_tool_result_is_recorded_with_its_verdict(tmp_path: Path) -> None:
    """Whether a turn's tool calls are WORKING is otherwise invisible without
    reading each result by hand — the response events count calls made, not
    calls that answered. Metrics only: the text is already in the turn's
    conversation artifact."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="boom", arguments={})),
        "gave up on the tool",
    )
    manager = _turn_manager(tmp_path, backend, [_FailingTool()])
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    results = [e for e in turn.events if e["kind"] == "tool/result"]
    assert [(e["values"]["tool_name"], e["values"]["is_error"]) for e in results] == [("boom", True)]


async def test_a_round_the_manager_discards_still_carries_its_spend(tmp_path: Path) -> None:
    """ullm re-runs a round truncated by a per-turn cap that has since
    lifted, and deletes the truncated one from the turn. Those tokens were
    generated and billed, so summing what SURVIVES the turn would lose them
    — the same hole compaction opened. The discarded round becomes a child
    span, and the parent's inclusive total is what the backend reports."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.StringContent(text="cut off mid-thought"), stop_reason="max_tokens"
        ),
        "the whole answer, second time",
    )
    manager = _turn_manager(tmp_path, backend, [])
    manager.max_generated_tokens_per_turn = 10_000
    # False while the round is in flight, True once it has answered: the cap
    # retiring mid-round is the only thing that triggers the discard.
    calls = iter([False])
    manager.per_turn_cap_lifted = lambda: next(calls, True)
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    (discarded,) = turn.children
    assert (discarded.kind, discarded.name) == ("llm/turn", "per_turn_cap_lifted")
    assert discarded.self_generated > 0, "the truncated round generated tokens and they were billed"

    usage = backend.total_usage
    assert turn.self_generated < usage.completion_tokens, "the surviving turn alone cannot account for the spend"
    assert turn.self_generated + discarded.self_generated == usage.completion_tokens


@dataclass
class _InjectingStub(StubBackend):
    """A scripted backend that accepts a response injection, so a turn over
    it records one the way a vLLM host would."""

    def supports_response_injection(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_llm_turn_names_the_injected_start_of_a_response(tmp_path: Path) -> None:
    """The response event, the conversation artifact, the chunk header and
    the rendered conversation all carry what was injected."""
    backend = _InjectingStub.from_script("final answer")
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="s"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    blocks = BlockStore(tmp_path / "blocks")
    manager = DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(
            backend=backend,
            sampling_params=ullm.backend.GenericSamplingParameters(max_tokens=10, response_injection="Okay,"),
        ),
        conversation=conversation,
        diag=SpanLogger(tmp_path / "spans.jsonl", blocks),
        label="play",
    )
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    (response,) = [e for e in turn.events if e["kind"] == "llm/response"]
    assert response["values"]["response_injection"] == "Okay,"
    view = ViewContext(blocks)
    (artifact,) = [a for a in turn.artifacts if a["name"] == "conversation"]
    recorded = [
        element
        for turn_element in view.load_block(artifact["block"])["elements"]
        if turn_element["kind"] == "turn"
        for element in turn_element["elements"]
        if element["kind"] == "response"
    ]
    assert [element["response_injection"] for element in recorded] == ["Okay,"]
    chunk = chunk_for(turn, view)
    assert "1 injected" in chunk.header_extra()
    assert "ullm-injected" in chunk.body_html()


@dataclass
class _RewritingTool(ullm.tool.Tool):
    """Declares a rewrite of the history already sent, as a compaction does."""

    name: str = "rewrite"
    description: str = "declares a history rewrite"
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    # A plain function (not the conversation itself, which holds this tool).
    declare_rewrite: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        assert self.declare_rewrite is not None
        self.declare_rewrite()
        return ullm.conv.StringContent(text="rewritten")


@pytest.mark.asyncio
async def test_each_response_carries_the_history_epoch_it_was_sent_in(tmp_path: Path) -> None:
    """Consecutive calls in one epoch extend each other's history exactly; a
    rewrite declared between two calls moves the next one to the next epoch."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "hi"})),
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="rewrite", arguments={})),
        "final answer",
    )
    rewriting = _RewritingTool()
    manager = _turn_manager(tmp_path, backend, [_EchoTool(), rewriting])
    conversation = manager.conversation
    rewriting.declare_rewrite = lambda: conversation.mark_history_rewritten()
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    epochs = [e["values"]["history_epoch"] for e in turn.events if e["kind"] == "llm/response"]
    assert epochs == [0, 0, 1]


def _billed(prompt: int, cached: int, completion: int) -> ullm.conv.LLMResponse:
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="r"))
    response.token_usage = ullm.backend.TokenUsage(
        prompt_tokens=prompt, cached_tokens=cached, completion_tokens=completion
    )
    return response


def test_a_continuation_that_re_prefills_its_history_is_announced(capsys: pytest.CaptureFixture[str]) -> None:
    """A continuation should prefill only what it adds -- its prompt less the
    previous call's prompt and reply; more than 200 tokens over is announced.
    A rewrite (a new epoch) and a history not extended exactly are not the
    cache's to answer for."""
    conversation = ullm.conv.Conversation()
    check_continuation_prefill(conversation, _billed(1000, 0, 200), where="g play")
    # Adds 50 (1250 - 1000 - 200); 60 uncached is one hash unit's slack.
    check_continuation_prefill(conversation, _billed(1250, 1190, 100), where="g play")
    assert capsys.readouterr().out == ""
    # Adds 50 (1400 - 1250 - 100) with 400 uncached: 350 re-prefilled.
    check_continuation_prefill(conversation, _billed(1400, 1000, 100), where="g play")
    assert capsys.readouterr().out == (
        "[recompute] g play: 400 uncached prompt tokens where the call adds 50 "
        "(prompt 1400, previous prompt 1250 + reply 100): 350 re-prefilled\n"
    )
    # Shorter than the previous prompt and reply: not an exact extension.
    check_continuation_prefill(conversation, _billed(1450, 0, 100), where="g play")
    conversation.mark_history_rewritten()
    check_continuation_prefill(conversation, _billed(900, 0, 100), where="g play")
    assert capsys.readouterr().out == ""
