"""Training penalties count malformed calls, excluding execution and budget stops."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tool


@dataclass
class ParsedBackend(ullm.backends.stub.StubBackend):
    """Replay parser-produced metadata on the retained response."""

    def _fill_into(
        self,
        src: ullm.conv.LLMResponse,
        element: ullm.conv.LLMResponse,
        conversation: ullm.conv.Conversation,
        injection: Optional[str] = None,
    ) -> None:
        """Keep parse evidence alongside the sampled text and tool calls."""
        super()._fill_into(src, element, conversation, injection)
        element.malformed_error = src.malformed_error
        element.malformed_call_count = src.malformed_call_count
        element.generation_request_id = f"puzzle#{self._call_index}"


@dataclass
class PythonTool(ullm.tool.Tool):
    """Use Python's real compiler and exception types without a kernel sandbox."""

    name: str = "python"
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }
    )
    executions: int = 0

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Report Python exceptions through the tool's ordinary error channel."""
        self.executions += 1
        try:
            exec(arguments["code"])
        except (SyntaxError, ValueError) as exc:
            self.is_error = True
            return ullm.conv.StringContent(text=f"{type(exc).__name__}: {exc}")
        return ullm.conv.StringContent(text="ok")


def _manager(
    backend: ParsedBackend, tool: PythonTool, *, episode_cap: Optional[int] = None
) -> tuple[ullm.manager.TurnManager, list[tuple[int, Optional[str]]]]:
    """Run a single-attempt training turn and record journal-facing metadata."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(tools=[tool]),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="puzzle"))],
    )
    observed: list[tuple[int, Optional[str]]] = []
    response = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=1, top_p=1, max_tokens=4096),
        max_retries=0,
        perturb_retry_prompt=False,
        preserve_failed_draws=True,
    )
    manager = ullm.manager.TurnManager(
        response_manager=response,
        conversation=conversation,
        classify_tool_calls=True,
        max_generated_tokens_per_turn=episode_cap,
        after_response=lambda item: observed.append((item.malformed_call_count, item.truncation_cap_source)),
    )
    return manager, observed


@pytest.mark.parametrize("code", ["if broken syntax", "raise ValueError('ordinary exception')"])
async def test_multiple_invalid_calls_count_once_each_but_python_exceptions_do_not(code: str) -> None:
    calls: list[ullm.conv.Content] = [
        ullm.conv.ToolCallContent(tool_name="missing", arguments={}),
        ullm.conv.ToolCallContent(tool_name="python", arguments={"code": 19}),
        ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code}),
    ]
    backend = ParsedBackend(
        responses=[
            ullm.conv.LLMResponse(final_content=ullm.conv.ListContent(elements=calls)),
            ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="ordinary reply")),
        ]
    )
    tool = PythonTool()
    manager, observed = _manager(backend, tool)
    await manager.run()
    assert observed == [(2, None), (0, None)]
    results = [item for item in manager.conversation.last_turn.elements if isinstance(item, ullm.conv.ToolResult)]
    assert [item.malformed_call_count for item in results] == [1, 1, 0]
    assert "Unknown tool" in results[0].content.to_string()
    assert "Invalid arguments" in results[1].content.to_string()
    assert tool.executions == 1
    assert backend._call_index == 2
    assert manager.conversation.history_epoch == 0


@pytest.mark.parametrize(
    "episode_cap, expected", [(None, (1, "response")), (4096, (0, "episode")), (512, (0, "episode"))]
)
async def test_incomplete_call_penalty_distinguishes_response_and_episode_caps(
    episode_cap: Optional[int], expected: tuple[int, str]
) -> None:
    raw = '<tool_call>{"name":"python","arguments":'
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=raw),
        malformed_error="incomplete tool-call JSON",
        malformed_call_count=1,
        stop_reason="max_tokens",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=10, completion_tokens=episode_cap or 4096),
    )
    backend = ParsedBackend(responses=[response])
    manager, observed = _manager(backend, PythonTool(), episode_cap=episode_cap)
    await manager.run()
    assert observed == [expected]
    assert backend._call_index == 1
    retained = manager.conversation.last_turn.elements[0]
    assert isinstance(retained, ullm.conv.LLMResponse)
    assert retained.final_content.to_string() == raw
    feedback = [item for item in manager.conversation.last_turn.elements if isinstance(item, ullm.conv.UserPrompt)]
    assert len(feedback) == expected[0]
    if feedback:
        assert "Malformed tool call" in feedback[0].content.to_string()


async def test_unparseable_response_counts_once_and_retains_raw_draw() -> None:
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="<tool_call>broken calls"),
        malformed_error="cannot recover individual tool calls",
        malformed_call_count=1,
        stop_reason="malformed",
    )
    manager, observed = _manager(ParsedBackend(responses=[response]), PythonTool())
    await manager.run()
    assert observed == [(1, None)]
    retained = manager.conversation.last_turn.elements[0]
    assert isinstance(retained, ullm.conv.LLMResponse)
    assert retained.final_content.to_string() == response.final_content.to_string()
    assert "Malformed tool call" in manager.conversation.to_string()


async def test_empty_retained_reply_counts_as_one_malformed_call() -> None:
    """A kept draw with no text and no tool call costs what a malformed tool call costs."""
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.EmptyContent(),
        malformed_error="Model produced no final content (no text response, no tool call).",
        stop_reason="malformed",
    )
    manager, observed = _manager(ParsedBackend(responses=[response]), PythonTool())
    await manager.run()
    assert observed == [(1, None)]
    retained = manager.conversation.last_turn.elements[0]
    assert isinstance(retained, ullm.conv.LLMResponse)
    assert retained.malformed_call_count == 1
    feedback = [item for item in manager.conversation.last_turn.elements if isinstance(item, ullm.conv.UserPrompt)]
    assert len(feedback) == 1
    assert "Malformed reply: no text and no tool call" in feedback[0].content.to_string()


async def test_empty_capped_reply_is_not_a_malformed_call() -> None:
    """Reasoning that spent the whole budget keeps its budget stop, unpenalized."""
    response = ullm.conv.LLMResponse(final_content=ullm.conv.EmptyContent(), stop_reason="max_tokens")
    manager, observed = _manager(ParsedBackend(responses=[response]), PythonTool())
    await manager.run()
    assert observed == [(0, "response")]
    assert not [item for item in manager.conversation.last_turn.elements if isinstance(item, ullm.conv.UserPrompt)]


async def test_plain_capped_text_is_not_a_malformed_tool_call() -> None:
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="I am still reasoning"), stop_reason="max_tokens"
    )
    manager, observed = _manager(ParsedBackend(responses=[response]), PythonTool())
    await manager.run()
    assert observed == [(0, "response")]


@pytest.mark.parametrize(
    "arguments, parse_error",
    [({"wrong_key": "print('must not execute')"}, None), ({"_raw": "print('must not execute')"}, "invalid JSON")],
)
async def test_repairable_invalid_calls_are_reported_without_execution(
    arguments: dict[str, Any], parse_error: Optional[str]
) -> None:
    call = ullm.conv.ToolCallContent(tool_name="python", arguments=arguments, malformed_error=parse_error)
    backend = ParsedBackend(
        responses=[
            ullm.conv.LLMResponse(final_content=call),
            ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="ordinary reply")),
        ]
    )
    tool = PythonTool()
    manager, observed = _manager(backend, tool)
    await manager.run()
    assert observed == [(1, None), (0, None)]
    assert tool.executions == 0
    result = manager.conversation.last_turn.elements[1]
    assert isinstance(result, ullm.conv.ToolResult)
    assert result.is_error and result.malformed_call_count == 1
    assert result.content.to_string()
    original = manager.conversation.last_turn.elements[0]
    assert isinstance(original, ullm.conv.LLMResponse)
    assert original.final_content.tool_calls()[0].arguments == arguments


async def test_standalone_tool_argument_repair_is_preserved() -> None:
    tool = PythonTool(arguments={"wrong_key": "pass"})
    await tool.run()
    assert not tool.is_error
    assert tool.executions == 1


@dataclass
class CancelledBackend(ParsedBackend):
    """Cancel while a completed sampled draw is being handed to its worker."""

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Keep the draw, then reproduce wall-time cancellation at handoff."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        raise asyncio.CancelledError


async def test_episode_cancellation_keeps_pending_draw_without_penalty_or_retry() -> None:
    backend = CancelledBackend(
        responses=[ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="<tool_call>{"))]
    )
    manager, observed = _manager(backend, PythonTool())
    with pytest.raises(asyncio.CancelledError):
        await manager.run()
    assert observed == []
    assert backend._call_index == 1
    assert manager.conversation.last_turn.last_response.malformed_call_count == 0
    assert manager.conversation.last_turn.last_response.final_content.to_string() == "<tool_call>{"
