"""``DiagnosticTurnManager``: llm/turn response/tool/kernel events,
token attribution, and the conversation artifact."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import pytest
import ullm.backend
import ullm.conv
import ullm.manager
import ullm.tool
from ullm.backends.stub import StubBackend

import platypus.llm
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


def _turn_manager(
    tmp_path: Path,
    backend: ullm.backend.Backend,
    tools: list[ullm.tool.Tool],
    *,
    diag: SpanLogger | None = None,
    label: str = "play",
) -> DiagnosticTurnManager:
    """Build one diagnostic turn over a scripted backend."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=tools)
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    return DiagnosticTurnManager(
        response_manager=ullm.manager.ResponseManager(backend=backend),
        conversation=conversation,
        diag=diag or SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks")),
        label=label,
    )


@pytest.mark.asyncio
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
    tool_results = [e for e in turn.events if e["kind"] == "tool/result"]
    assert len(responses) == 2 and len(kernels) == 1
    assert len(tool_results) == 1
    assert tool_results[0]["values"] == {"tool_name": "echo", "is_error": False}
    assert responses[0]["values"]["n_tool_calls"] == 1
    assert responses[1]["values"]["stop_reason"] == "stop"
    assert kernels[0]["t"] == 1234.5 and kernels[0]["values"]["duration_s"] == 0.25

    # The span's token attribution equals what the backend reports (I1's
    # per-span granularity).
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


@pytest.mark.asyncio
async def test_llm_turn_span_records_preexecution_tool_errors(tmp_path: Path) -> None:
    """Tool-result metrics include schema failures that never reach a kernel."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={})),
        "recovered",
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    tool_results = [e for e in turn.events if e["kind"] == "tool/result"]
    assert len(tool_results) == 1
    assert tool_results[0]["values"] == {"tool_name": "echo", "is_error": True}
    assert not [e for e in turn.events if e["kind"] == "kernel/exec"]


@pytest.mark.asyncio
async def test_llm_turn_span_keeps_metrics_for_pruned_rounds(tmp_path: Path) -> None:
    """Live compaction cannot erase response usage or tool-error metrics."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={})),
        "recovered",
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    rounds = 0

    async def prune_first_round() -> None:
        nonlocal rounds
        rounds += 1
        if rounds != 2:
            return
        turn = manager.conversation.last_turn
        doomed = turn.elements[:2]
        manager.record_context_pruned_round(doomed)
        del turn.elements[:2]

    manager.before_round = prune_first_round
    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    responses = [e for e in turn.events if e["kind"] == "llm/response"]
    tool_results = [e for e in turn.events if e["kind"] == "tool/result"]
    assert [e["values"].get("discarded_reason") for e in responses] == ["context_compaction", None]
    assert responses[0]["values"]["available_until_round"] == 1
    assert tool_results[0]["values"] == {
        "tool_name": "echo",
        "is_error": True,
        "discarded_reason": "context_compaction",
        "available_until_round": 1,
    }
    assert turn.self_generated == backend.total_usage.completion_tokens
    assert turn.self_uncached == backend.total_usage.prompt_tokens - backend.total_usage.cached_tokens

    blocks = BlockStore(tmp_path / "blocks")
    (artifact,) = [a for a in turn.artifacts if a["name"] == "discarded_rounds"]
    discarded = blocks.load(artifact["block"], "json")
    assert discarded["records"] == [{"reason": "context_compaction", "available_until_round": 1}]
    (discarded_turn,) = discarded["conversation"]["elements"]
    assert [element["kind"] for element in discarded_turn["elements"]] == ["response", "tool_result"]
    html = chunk_for(turn, ViewContext(blocks)).body_html()
    assert "1 discarded round" in html
    assert "responses completed at removal" in html
    assert "tool call: echo" in html


@pytest.mark.asyncio
async def test_discard_artifact_failure_keeps_primary_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Optional discarded content cannot suppress the surviving transcript."""
    backend = StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="echo", arguments={})),
        "recovered",
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    rounds = 0

    async def prune_first_round() -> None:
        nonlocal rounds
        rounds += 1
        if rounds == 2:
            doomed = manager.conversation.last_turn.elements[:2]
            manager.record_context_pruned_round(doomed)
            del manager.conversation.last_turn.elements[:2]

    original = platypus.llm.serialize_conversation_blocks

    def reject_discarded(
        conversation: ullm.conv.Conversation,
        block_store: BlockStore,
    ) -> Any:
        if conversation is not manager.conversation:
            raise ValueError("discarded content cannot be serialized")
        return original(conversation, block_store)

    manager.before_round = prune_first_round
    monkeypatch.setattr(platypus.llm, "serialize_conversation_blocks", reject_discarded)
    await manager.run()
    manager.diag.close()
    capsys.readouterr()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert [artifact["name"] for artifact in turn.artifacts] == ["conversation"]


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@dataclass
class _BarrierStubBackend(StubBackend):
    """Hold two calls in flight so shared-backend accounting overlaps."""

    arrivals: int = 0
    both_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Wait until two managers have started, then release both calls."""
        del sampling_params
        self.arrivals += 1
        if self.arrivals == 2:
            self.both_arrived.set()
        await self.release.wait()
        yield llm_input


@dataclass
class _CancellableBackend(ullm.backend.Backend):
    """Stream five attributable tokens, then wait to be cancelled."""

    reached_wait: asyncio.Event = field(default_factory=asyncio.Event)

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> str:
        """Return a fixed input; only cancellation accounting is under test."""
        del sampling_params, turn, conversation, send_back_reasoning
        return "prompt"

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[int, None]:
        """Emit five chunks, then remain in flight until the task is cancelled."""
        del sampling_params, llm_input
        for chunk in range(1, 6):
            yield chunk
            await asyncio.sleep(0)
        self.reached_wait.set()
        await asyncio.Event().wait()

    def _process_partial_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        intermediate: ullm.conv.LLMResponse,
    ) -> None:
        """Expose one character per streamed token in the partial response."""
        del sampling_params
        intermediate.final_content = ullm.conv.StringContent(text="x" * int(llm_output))

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Reject completion; this backend exists only for cancellation."""
        del sampling_params, llm_output, conversation, turn, element
        raise AssertionError("cancellable test backend unexpectedly completed")

    def _usage_from_partial(self, llm_output: Any) -> ullm.backend.TokenUsage | None:
        """Attribute the latest chunk as completion usage."""
        if llm_output is None:
            return None
        return ullm.backend.TokenUsage(prompt_tokens=10, completion_tokens=int(llm_output), num_requests=1)


@pytest.mark.asyncio
async def test_overlapping_turns_on_one_backend_are_accounted_once_each(tmp_path: Path) -> None:
    """A shared backend's cumulative counter cannot leak sibling usage."""
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="done"),
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=11,
            cached_tokens=3,
            completion_tokens=7,
            num_requests=1,
        ),
    )
    backend = _BarrierStubBackend(responses=[response], on_exhausted="repeat_last")
    diag = SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks"))
    managers = [
        _turn_manager(tmp_path, backend, [], diag=diag, label="first"),
        _turn_manager(tmp_path, backend, [], diag=diag, label="second"),
    ]

    tasks = [asyncio.create_task(manager.run()) for manager in managers]
    await asyncio.wait_for(backend.both_arrived.wait(), timeout=1)
    backend.release.set()
    await asyncio.gather(*tasks)
    diag.close()

    turns = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert len(turns) == 2
    assert sorted(turn.self_generated for turn in turns) == [7, 7]
    assert sorted(turn.self_uncached for turn in turns) == [8, 8]
    assert sum(turn.self_generated for turn in turns) == backend.total_usage.completion_tokens == 14


@pytest.mark.asyncio
async def test_cancelled_partial_response_keeps_request_scoped_usage(tmp_path: Path) -> None:
    """Cancellation usage reaches the owning span without a backend delta."""
    backend = _CancellableBackend()
    manager = _turn_manager(tmp_path, backend, [])
    task = asyncio.create_task(manager.run())
    await asyncio.wait_for(backend.reached_wait.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert turn.outcome == "cancelled"
    assert turn.self_generated == backend.total_usage.completion_tokens == 5
    assert turn.self_uncached == 10
    (response,) = [event for event in turn.events if event["kind"] == "llm/response"]
    assert response["values"]["completion_tokens"] == 5
    assert response["values"]["incomplete"] is True


def _used_response(
    content: ullm.conv.Content,
    completion_tokens: int,
    *,
    stop_reason: ullm.conv.StopReason,
) -> ullm.conv.LLMResponse:
    """Build a scripted response with exact usage for cap-retry accounting."""
    return ullm.conv.LLMResponse(
        final_content=content,
        stop_reason=stop_reason,
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=10,
            cached_tokens=2,
            completion_tokens=completion_tokens,
            num_requests=1,
        ),
    )


@pytest.mark.asyncio
async def test_inherited_observer_drops_activity_after_parent_close(tmp_path: Path) -> None:
    """A child task cannot recurse into or mutate a diagnostic turn after close."""
    backend = StubBackend.from_script(
        _used_response(ullm.conv.StringContent(text="outer answer"), 5, stop_reason="stop"),
        _used_response(ullm.conv.StringContent(text="late answer"), 7, stop_reason="stop"),
    )
    manager = _turn_manager(tmp_path, backend, [])
    release = asyncio.Event()
    late_task: asyncio.Task[None] | None = None

    async def run_late_nested_turn() -> None:
        """Run a plain manager from a task that inherited the parent's observer."""
        await release.wait()
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="late system")),
            elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="late prompt"))],
        )
        nested = ullm.manager.TurnManager(
            response_manager=ullm.manager.ResponseManager(backend=backend),
            conversation=conversation,
        )
        await nested.run()

    async def spawn_late_nested_turn() -> None:
        """Create the child while the diagnostic observer is installed."""
        nonlocal late_task
        if late_task is None:
            late_task = asyncio.create_task(run_late_nested_turn())

    manager.before_round = spawn_late_nested_turn
    await manager.run()
    assert late_task is not None
    release.set()
    await late_task
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    responses = [event for event in turn.events if event["kind"] == "llm/response"]
    assert [event["values"]["completion_tokens"] for event in responses] == [5]
    assert turn.self_generated == 5
    assert backend.total_usage.completion_tokens == 12


@pytest.mark.asyncio
async def test_inherited_observer_cannot_leak_into_a_rearmed_run(tmp_path: Path) -> None:
    """Each run rejects callbacks inherited from an earlier generation."""
    backend = StubBackend.from_script(
        _used_response(ullm.conv.StringContent(text="first answer"), 5, stop_reason="stop"),
        _used_response(ullm.conv.StringContent(text="late answer"), 7, stop_reason="stop"),
        _used_response(ullm.conv.StringContent(text="second answer"), 11, stop_reason="stop"),
    )
    manager = _turn_manager(tmp_path, backend, [])
    release_late = asyncio.Event()
    late_task: asyncio.Task[None] | None = None

    async def run_late_nested_turn() -> None:
        """Run with the observer context copied from the first generation."""
        await release_late.wait()
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="late system")),
            elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="late prompt"))],
        )
        nested = ullm.manager.TurnManager(
            response_manager=ullm.manager.ResponseManager(backend=backend),
            conversation=conversation,
        )
        await nested.run()

    async def spawn_late_nested_turn() -> None:
        """Create the late child under the first run's observer context."""
        nonlocal late_task
        if late_task is None:
            late_task = asyncio.create_task(run_late_nested_turn())

    manager.before_round = spawn_late_nested_turn
    await manager.run()
    assert late_task is not None

    manager.rearm()
    manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="again")))
    second_started = asyncio.Event()
    release_second = asyncio.Event()

    async def hold_second_run() -> None:
        """Keep generation two active while the generation-one child finishes."""
        second_started.set()
        await release_second.wait()

    manager.before_round = hold_second_run
    second_task = asyncio.create_task(manager.run())
    await asyncio.wait_for(second_started.wait(), timeout=1)
    release_late.set()
    await late_task
    release_second.set()
    await second_task
    manager.diag.close()

    first, second = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    first_responses = [event for event in first.events if event["kind"] == "llm/response"]
    second_responses = [event for event in second.events if event["kind"] == "llm/response"]
    assert [event["values"]["completion_tokens"] for event in first_responses] == [5]
    assert [event["values"]["completion_tokens"] for event in second_responses] == [11]
    assert first.self_generated == 5
    assert second.self_generated == 11
    assert backend.total_usage.completion_tokens == 23


@dataclass
class _NestedTurnTool(ullm.tool.Tool):
    """Run a plain nested TurnManager over the outer turn's shared backend."""

    name: str = "nested"
    description: str = "runs one nested LLM turn"
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    backend: ullm.backend.Backend = field(default_factory=ullm.backend.Backend)
    use_null_diagnostic: bool = False

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Run and return one nested plain-manager response."""
        del arguments
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="nested system")),
            elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="nested prompt"))],
        )
        manager_type: type[ullm.manager.TurnManager] = (
            DiagnosticTurnManager if self.use_null_diagnostic else ullm.manager.TurnManager
        )
        manager = manager_type(
            response_manager=ullm.manager.ResponseManager(backend=self.backend),
            conversation=conversation,
        )
        await manager.run()
        return ullm.conv.StringContent(text=conversation.last_turn.last_response.final_content.to_string())


@pytest.mark.asyncio
@pytest.mark.parametrize("use_null_diagnostic", [False, True])
async def test_unrecorded_nested_turn_is_attributed_to_active_diagnostic_parent(
    tmp_path: Path,
    use_null_diagnostic: bool,
) -> None:
    """Built-in-style subagent calls cannot disappear from clean-run totals."""
    outer_call = ullm.conv.ToolCallContent(tool_name="nested", arguments={})
    backend = StubBackend.from_script(
        _used_response(outer_call, 5, stop_reason="tool_use"),
        _used_response(ullm.conv.StringContent(text="nested answer"), 7, stop_reason="stop"),
        _used_response(ullm.conv.StringContent(text="outer answer"), 11, stop_reason="stop"),
    )
    manager = _turn_manager(
        tmp_path,
        backend,
        [_NestedTurnTool(backend=backend, use_null_diagnostic=use_null_diagnostic)],
    )

    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    responses = [event for event in turn.events if event["kind"] == "llm/response"]
    assert [event["values"]["completion_tokens"] for event in responses] == [5, 7, 11]
    assert turn.self_generated == backend.total_usage.completion_tokens == 23


@pytest.mark.asyncio
async def test_cap_lift_retry_is_recorded_but_not_put_back_in_conversation(tmp_path: Path) -> None:
    """A paid response retried after the cap lifts remains observable."""

    def call(text: str) -> ullm.conv.ToolCallContent:
        """Ask the echo tool for another round."""
        return ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": text})

    backend = StubBackend.from_script(
        _used_response(call("one"), 60, stop_reason="tool_use"),
        _used_response(call("two"), 60, stop_reason="tool_use"),
        _used_response(call("discarded"), 30, stop_reason="max_tokens"),
        _used_response(ullm.conv.StringContent(text="done"), 60, stop_reason="stop"),
    )
    manager = _turn_manager(tmp_path, backend, [_EchoTool()])
    manager.response_manager.sampling_params = ullm.backend.GenericSamplingParameters(
        temperature=0.0,
        max_tokens=100,
    )
    manager.max_generated_tokens_per_turn = 150
    manager.per_turn_cap_lifted = lambda: backend.total_usage.completion_tokens >= 150

    await manager.run()
    manager.diag.close()

    (turn,) = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    responses = [event for event in turn.events if event["kind"] == "llm/response"]
    assert [event["values"]["completion_tokens"] for event in responses] == [60, 60, 30, 60]
    discarded = [event for event in responses if event["values"].get("discarded_reason")]
    assert len(discarded) == 1
    assert discarded[0]["values"]["discarded_reason"] == "per_turn_cap_lifted"
    assert discarded[0]["values"]["stop_reason"] == "max_tokens"
    assert discarded[0]["values"]["truncated_by_per_turn_cap"] is True
    assert turn.self_generated == backend.total_usage.completion_tokens == 210

    blocks = BlockStore(tmp_path / "blocks")
    (conversation_artifact,) = [a for a in turn.artifacts if a["name"] == "conversation"]
    conversation = blocks.load(conversation_artifact["block"], "json")
    surviving = conversation["elements"][-1]["elements"]
    surviving_responses = [element for element in surviving if element["kind"] == "response"]
    assert [element["usage"]["completion_tokens"] for element in surviving_responses] == [60, 60, 60]
    assert all(element["stop_reason"] != "max_tokens" for element in surviving_responses)

    (discarded_artifact,) = [a for a in turn.artifacts if a["name"] == "discarded_rounds"]
    discarded_payload = blocks.load(discarded_artifact["block"], "json")
    assert discarded_payload["records"] == [{"reason": "per_turn_cap_lifted"}]
    (discarded_turn,) = discarded_payload["conversation"]["elements"]
    assert discarded_turn["elements"][0]["usage"]["completion_tokens"] == 30


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
