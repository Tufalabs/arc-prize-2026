"""OpenAIServer request construction: reasoning passback, chat_template_kwargs,
tool schema shape, and tool-call markup salvage. No network — _prepare_input
and _process_output are pure."""

from typing import Any

import openai.types.chat
import pytest
import ullm.backend
import ullm.backends.oai
import ullm.conv
import ullm.tool


def _chat_completion(
    content: str, *, reasoning: str | None = None, tool_calls: list[dict] | None = None
) -> openai.types.chat.ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning  # vLLM extension field; SDK models allow extras
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "choices": [{"finish_reason": "stop", "index": 0, "message": message}],
            "created": 0,
            "model": "m",
            "object": "chat.completion",
        }
    )


def _server(**kwargs: Any) -> ullm.backends.oai.OpenAIServer:
    return ullm.backends.oai.OpenAIServer(base_url="http://127.0.0.1:1/v1", api_key="unused", model="m", **kwargs)


def _tool() -> ullm.tool.Tool:
    return ullm.tool.Tool(
        name="python",
        description="run code",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    )


def _conversation_with_history() -> tuple[ullm.conv.Conversation, ullm.conv.ResponseTurn]:
    """System + one completed turn (tool call w/ reasoning + tool result) + a fresh turn."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[_tool()])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text="THOUGHTS about the move"),
        final_content=ullm.conv.ToolCallContent(call_id="c1", tool_name="python", arguments={"code": "1+1"}),
    )
    response._model_contents = _chat_completion("", reasoning="THOUGHTS about the move")
    past = ullm.conv.ResponseTurn()
    past.elements.append(response)
    past.elements.append(
        ullm.conv.ToolResult(call_id="c1", tool_name="python", content=ullm.conv.StringContent(text="2"))
    )
    conversation.elements.append(past)
    turn = ullm.conv.ResponseTurn()
    conversation.elements.append(turn)
    return conversation, turn


def _sampling(**kwargs: Any) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(temperature=0.5, **kwargs)


def test_reasoning_rides_as_assistant_field():
    conversation, turn = _conversation_with_history()
    server = _server(reasoning_history_field="reasoning")
    kwargs = server._prepare_input(_sampling(), turn, conversation, "all")
    assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["reasoning"] == "THOUGHTS about the move"
    # content must stay reasoning-free (the field is the vehicle, not the text)
    assert "THOUGHTS" not in (assistant[0]["content"] or "")
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "python"


def test_reasoning_free_response_survives_passback():
    """A response with no reasoning (reasoning_content=EmptyContent) must not
    crash request construction when the reasoning field is enabled — caught
    live by the gpu matrix (content_to_string raises on EmptyContent)."""
    conversation, turn = _conversation_with_history()
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="plain answer, no thinking"))
    response._model_contents = _chat_completion("plain answer, no thinking")
    past = ullm.conv.ResponseTurn()
    past.elements.append(response)
    conversation.elements.insert(-1, past)
    kwargs = _server(reasoning_history_field="reasoning")._prepare_input(_sampling(), turn, conversation, "all")
    plain = [m for m in kwargs["messages"] if m.get("content") == "plain answer, no thinking"]
    assert len(plain) == 1 and "reasoning" not in plain[0]


def test_reasoning_field_absent_when_not_sending_reasoning():
    conversation, turn = _conversation_with_history()
    server = _server(reasoning_history_field="reasoning")
    kwargs = server._prepare_input(_sampling(), turn, conversation, "none")
    assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"]
    assert "reasoning" not in assistant[0]


def test_send_back_reasoning_needs_a_vehicle():
    conversation, turn = _conversation_with_history()
    with pytest.raises(ValueError, match="reasoning_history_field"):
        _server()._prepare_input(_sampling(), turn, conversation, "all")
    # either vehicle suffices
    _server(reasoning_history_field="reasoning")._prepare_input(_sampling(), turn, conversation, "all")
    _server(parse_think_tags=True)._prepare_input(_sampling(), turn, conversation, "all")


def test_chat_template_kwargs_merge_into_extra_body():
    conversation, turn = _conversation_with_history()
    server = _server(
        reasoning_history_field="reasoning", chat_template_kwargs={"preserve_thinking": True, "enable_thinking": True}
    )
    kwargs = server._prepare_input(_sampling(extra_params={"top_k": 20}), turn, conversation, "all")
    assert kwargs["extra_body"]["top_k"] == 20
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": True}


def test_named_effort_rides_verbatim_and_leaves_template_kwargs_alone():
    """A named level becomes the reasoning_effort API param, untranslated —
    the client cannot know the far side's vocabulary, so the server's own
    rejection is the error surface. The configured thinking stance is not
    touched: with a level named, thinking stays whatever the config says."""
    conversation, turn = _conversation_with_history()
    conversation.system_prompt.reasoning_effort = "xhigh"
    server = _server(reasoning_history_field="reasoning", chat_template_kwargs={"enable_thinking": True})
    kwargs = server._prepare_input(_sampling(), turn, conversation, "all")
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_none_effort_disables_thinking_per_request():
    """None means no thinking, on this path like every other. Omitting the
    param used to leave a static ``enable_thinking: True`` (or the template's
    own default) in charge — thinking ON with the effort text missing, the
    inversion of what the caller asked. The per-request override wins over
    the configured stance, and no reasoning_effort param is sent."""
    conversation, turn = _conversation_with_history()
    conversation.system_prompt.reasoning_effort = None
    server = _server(reasoning_history_field="reasoning", chat_template_kwargs={"enable_thinking": True})
    kwargs = server._prepare_input(_sampling(), turn, conversation, "all")
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    # Even with no configured template kwargs the request must say it: the
    # template's default is thinking ON, so silence would invert None too.
    bare = _server(reasoning_history_field="reasoning")
    kwargs = bare._prepare_input(_sampling(), turn, conversation, "all")
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_none_effort_with_the_flag_off_touches_nothing():
    """`native_reasoning_effort=False` opts out of the whole native surface;
    the request must not grow template kwargs the config never set."""
    conversation, turn = _conversation_with_history()
    conversation.system_prompt.reasoning_effort = None
    server = _server(reasoning_history_field="reasoning", native_reasoning_effort=False)
    kwargs = server._prepare_input(_sampling(), turn, conversation, "all")
    assert "reasoning_effort" not in kwargs
    assert "chat_template_kwargs" not in kwargs.get("extra_body", {})


def test_tool_schema_strict_and_auto_choice():
    conversation, turn = _conversation_with_history()
    kwargs = _server(reasoning_history_field="reasoning")._prepare_input(_sampling(), turn, conversation, "all")
    assert kwargs["tool_choice"] == "auto"
    # strict is required CLIENT-side: the SDK streaming helper refuses to
    # auto-parse tool calls from non-strict tools (crashed a full run).
    assert kwargs["tools"][0]["function"]["strict"] is True


_MARKUP = "<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>"


def _process(
    server: ullm.backends.oai.OpenAIServer, completion: openai.types.chat.ChatCompletion
) -> ullm.conv.LLMResponse:
    conversation, turn = _conversation_with_history()
    element = ullm.conv.LLMResponse()
    server._process_output(_sampling(), completion, conversation, turn, element)
    return element


def test_markup_salvage_from_content():
    element = _process(_server(), _chat_completion(f"lead text {_MARKUP} trail"))
    calls = element.final_content.tool_calls()
    assert len(calls) == 1
    assert calls[0].tool_name == "python"
    assert calls[0].arguments == {"code": "print(1)"}
    # the broken markup is stripped from the surviving text
    text = element.final_content.to_string()
    assert "<tool_call>" not in text


def test_markup_salvage_from_reasoning():
    element = _process(_server(), _chat_completion("", reasoning=f"thinking... {_MARKUP}"))
    calls = element.final_content.tool_calls()
    assert len(calls) == 1
    assert calls[0].arguments == {"code": "print(1)"}


def test_structured_tool_calls_not_overridden_by_salvage():
    completion = _chat_completion(
        "",
        tool_calls=[
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "python", "arguments": '{"code": "real"}'},
            }
        ],
        reasoning=_MARKUP,
    )
    element = _process(_server(), completion)
    calls = element.final_content.tool_calls()
    assert len(calls) == 1
    assert calls[0].arguments == {"code": "real"}


def test_the_token_ids_field_is_not_injected_on_the_weaker_trace_grade() -> None:
    """``return_token_ids`` is a vLLM extension; a hosted API may reject the
    unknown field outright, and on the weaker grade the ids would go unread
    anyway. Tracing on the full grade still injects it; the weaker grade must
    send a request indistinguishable from an untraced one."""
    conversation, turn = _conversation_with_history()

    full = _server(reasoning_history_field="reasoning")
    full.trace_recorder = object()  # type: ignore[assignment]
    kwargs = full._prepare_input(_sampling(), turn, conversation, "none")
    assert kwargs["extra_body"]["return_token_ids"] is True

    weak = _server(reasoning_history_field="reasoning")
    weak.trace_recorder = object()  # type: ignore[assignment]
    weak._trace_no_token_ids = True
    kwargs = weak._prepare_input(_sampling(), turn, conversation, "none")
    assert "return_token_ids" not in kwargs.get("extra_body", {})


def test_only_a_positive_route_absence_downgrades_the_trace_grade() -> None:
    """A transient failure must not arm the weaker grade for the whole run.

    Downgrading is only allowed when the probe positively shows there is no
    vLLM surface — the route refused as such, or answering with something
    that is not the route. Auth failures, rate limits, server errors and
    dead sockets say nothing about the surface and must propagate, or a
    setup problem silently becomes weaker training data.
    """
    import json as _json

    import httpx
    import ullm.tracing
    from ullm.backends import oai

    def status_error(code: int) -> httpx.HTTPStatusError:
        request = httpx.Request("GET", "http://host/v1/models")
        return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))

    downgrades = oai.OpenAIServer._probe_shows_no_vllm_surface
    assert downgrades(ullm.tracing.TraceReconstructionError("no 'tokens'"))
    assert downgrades(_json.JSONDecodeError("html page", "<html>", 0))
    assert downgrades(status_error(404))
    assert downgrades(status_error(400))
    assert not downgrades(status_error(401))
    assert not downgrades(status_error(429))
    assert not downgrades(status_error(500))
    assert not downgrades(ConnectionError("refused"))
    assert not downgrades(TimeoutError("slow"))
