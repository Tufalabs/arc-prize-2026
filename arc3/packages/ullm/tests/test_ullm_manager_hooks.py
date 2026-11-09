"""TurnManager hooks: before_round, after_response, after_tool_result,
stop_when (serial + skip synthesis), and skipped-result synthesis on truncated
responses."""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tool


@dataclass
class RecordingTool(ullm.tool.Tool):
    name: str = "echo"
    description: str = "echoes"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    )
    executed: list[str] = field(default_factory=list)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        self.executed.append(str(arguments.get("text", "")))
        return ullm.conv.StringContent(text=f"echo:{arguments.get('text')}")


def _tool_call_response(*texts: str, stop_reason: str | None = None) -> ullm.conv.LLMResponse:
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": t}) for t in texts]
        )
    )
    if stop_reason is not None:
        response.stop_reason = stop_reason  # type: ignore[assignment]
    return response


def _manager(
    script: Sequence[ullm.conv.LLMResponse | str], tool: ullm.tool.Tool, **kwargs: Any
) -> ullm.manager.TurnManager:
    backend = ullm.backends.stub.StubBackend.from_script(*script, on_exhausted="raise")
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[tool])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    response_manager = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=100),
    )
    return ullm.manager.TurnManager(response_manager=response_manager, conversation=conversation, **kwargs)


def _tool_results(manager: ullm.manager.TurnManager) -> list[ullm.conv.ToolResult]:
    turn = manager.conversation.elements[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    return [e for e in turn.elements if isinstance(e, ullm.conv.ToolResult)]


async def test_stop_when_runs_serially_and_skips_the_rest() -> None:
    tool = RecordingTool()
    stop_after = {"count": 0}

    def stop_when(response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> str | None:
        stop_after["count"] += 1
        return "SKIPPED-BY-HOOK" if stop_after["count"] >= 2 else None

    manager = _manager([_tool_call_response("a", "b", "c")], tool, stop_when=stop_when)
    await manager.run()
    assert tool.executed == ["a", "b"]  # serial, stopped after the second
    results = _tool_results(manager)
    assert len(results) == 3  # every tool_use got a result
    assert [r.is_error for r in results] == [False, False, True]
    assert results[2].content.to_string() == "SKIPPED-BY-HOOK"


async def test_stop_when_replacement_takes_effect_within_the_round() -> None:
    """A caller may replace the public hook between serial tool results."""
    tool = RecordingTool()

    def replacement(response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> str:
        return "STOPPED-BY-REPLACEMENT"

    def initial(response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> None:
        manager.stop_when = replacement

    manager = _manager([_tool_call_response("a", "b", "c")], tool, stop_when=initial)
    await manager.run()

    assert tool.executed == ["a", "b"]
    results = _tool_results(manager)
    assert [result.is_error for result in results] == [False, False, True]
    assert results[2].content.to_string() == "STOPPED-BY-REPLACEMENT"


async def test_before_round_fires_before_every_call() -> None:
    tool = RecordingTool()
    rounds: list[int] = []
    manager = _manager(
        [_tool_call_response("a"), "done"],  # tool round + closing text round
        tool,
        before_round=lambda: rounds.append(len(tool.executed)),
    )
    await manager.run()
    assert rounds == [0, 1]  # before call 1 (nothing run) and before call 2


async def test_after_response_fires_for_a_response_stop_when_never_sees() -> None:
    """Per-response bookkeeping cannot ride ``stop_when``.

    ``stop_when`` fires per ToolResult, so the closing round — a response
    with no tool call — never reaches it, and a caller recording something
    about every served response (which state the engine served it from, say)
    would silently miss it.
    """
    tool = RecordingTool()
    seen: list[int] = []
    stopped: list[str] = []

    def stop_when(response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> None:
        stopped.append(result.content.to_string())

    def after_response(response: ullm.conv.LLMResponse) -> None:
        seen.append(len(response.final_content.tool_calls()))

    manager = _manager(
        [_tool_call_response("a"), "done"],  # tool round + closing text round
        tool,
        after_response=after_response,
        stop_when=stop_when,
    )
    await manager.run()
    assert seen == [1, 0]  # both rounds, the tool-call-less one included
    assert stopped == ["echo:a"]  # only the round that produced a result


@pytest.mark.parametrize("serial", [False, True])
async def test_after_tool_result_replaces_every_handled_result(serial: bool) -> None:
    """What the hook returns is what the turn records and ``stop_when`` sees,
    for a call that ran and for one refused before it could (an unknown tool),
    on both the parallel and the serial path."""
    tool = RecordingTool()
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.ToolCallContent(tool_name="echo", arguments={"text": "a"}),
                ullm.conv.ToolCallContent(tool_name="missing", arguments={}),
            ]
        )
    )
    stopped: list[str] = []

    def stop_when(response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> None:
        stopped.append(result.content.to_string())

    def after_tool_result(result: ullm.conv.ToolResult) -> ullm.conv.ToolResult:
        return dataclasses.replace(result, content=ullm.conv.StringContent(text=f"{result.content.to_string()} +note"))

    manager = _manager(
        [response, "done"],
        tool,
        after_tool_result=after_tool_result,
        stop_when=stop_when if serial else None,
    )
    await manager.run()
    texts = [result.content.to_string() for result in _tool_results(manager)]
    assert texts == ["echo:a +note", "Unknown tool: 'missing' +note"]
    assert stopped == (texts if serial else [])


async def test_truncated_response_synthesizes_skipped_results() -> None:
    """A cut-off response leaves its tool calls parsed but unexecuted, and
    every one of them still gets a result.

    Not politeness: a ``tool_use`` with no matching result is malformed input
    on the very next request, which Anthropic rejects with a 400. The calls
    themselves are never run — a truncated call's arguments cannot be trusted
    — so the result says exactly that.
    """
    tool = RecordingTool()
    manager = _manager([_tool_call_response("a", "b", stop_reason="max_tokens")], tool)
    await manager.run()
    assert tool.executed == []  # truncated tool calls are never executed
    results = _tool_results(manager)
    assert len(results) == 2
    assert all(r.is_error for r in results)
    assert "truncated (stop_reason=max_tokens)" in results[0].content.to_string()


async def test_unusable_unshare_fails_loudly_off_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silently dropping network isolation is a security downgrade; only the
    # Kaggle container (no user namespaces, no egress) may fall back.
    import pytest
    import ullm.tools.python as utp

    monkeypatch.setattr(utp, "_unshare_usable_cache", False)
    tool = utp.PythonTool(timeout=10.0)
    try:
        tool.arguments = {"code": "print(1)"}
        with pytest.raises(RuntimeError, match="isolate_network"):
            await tool.run()
    finally:
        await tool.clear_state()
