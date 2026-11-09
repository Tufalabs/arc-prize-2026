"""U5: ARC-2 parser errors keep sampled draws and distinguish tool syntax from prose."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.conv
import ullm.tool

pytest.importorskip("ullm.backends.vllmx", reason="requires pinned vLLM parsers")
import vllm.tool_parsers
import vllm.tool_parsers.hermes_tool_parser
from test_training_malformed_calls import ParsedBackend, PythonTool, _manager
from test_vllmx_engine_contract import TOK, _params
from test_vllmx_engine_contract import _backend as _engine_backend
from test_vllmx_training_policy import _backend, _conversation, _output


class Parser:
    """Return controlled recovered calls or a parser-level failure."""

    def __init__(self, arguments: list[str] | None = None, *, fail: bool = False) -> None:
        self.arguments = arguments or []
        self.fail = fail

    def extract_tool_calls(self, text: str, request: Any) -> Any:
        """Expose the native parser's recovered-call schema."""
        if self.fail:
            raise ValueError("invalid tool-call JSON")
        return SimpleNamespace(
            tools_called=bool(self.arguments),
            content=None if self.arguments else text,
            tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name="python", arguments=value)) for value in self.arguments
            ],
        )


def parse(text: str, parser: Any, *, enabled: bool = True, capped: bool = False) -> ullm.conv.LLMResponse:
    """Parse one response while retaining its complete raw engine evidence."""
    backend = _backend(None, accept_malformed_output=True, classify_tool_calls=enabled)
    backend._tool_parser = parser
    conversation = _conversation()
    conversation.system_prompt.tools = [
        ullm.tool.Tool(name="python", description="Execute Python", parameters={"type": "object"})
    ]
    result = ullm.conv.LLMResponse()
    output = _output(text, finish_reason="length" if capped else "stop")
    output.request_id = "slot#nonce"
    backend._parse_output(output, conversation, result)
    assert result._model_contents is output
    assert result.generation_request_id == "slot#nonce"
    return result


def _hermes() -> Any:
    """Use the installed parser that produces the backend's argument JSON."""
    return vllm.tool_parsers.hermes_tool_parser.Hermes2ProToolParser(_backend(None).engine.get_tokenizer())


@pytest.mark.parametrize("code", ['print("<tool_call>")', "print('a \"quote\" <tool_call>')"])
@pytest.mark.parametrize("incomplete_tail", [False, True])
def test_tag_literals_do_not_count_as_unrecovered_calls(code: str, incomplete_tail: bool) -> None:
    text = "<tool_call>" + json.dumps({"name": "python", "arguments": {"code": code}}) + "</tool_call>"
    if incomplete_tail:
        text += "<tool_call>"
    result = parse(text, _hermes())
    assert result.malformed_call_count == int(incomplete_tail)
    if not incomplete_tail:
        assert result.final_content.tool_calls()[0].arguments == {"code": code}
        assert result.malformed_error is None


@pytest.mark.parametrize("incomplete_tail", [False, True])
def test_xml_parameter_literals_do_not_hide_incomplete_calls(
    monkeypatch: pytest.MonkeyPatch, incomplete_tail: bool
) -> None:
    tokenizer = _backend(None).engine.get_tokenizer()
    monkeypatch.setattr(tokenizer, "get_vocab", lambda: tokenizer.specials, raising=False)
    parser = vllm.tool_parsers.ToolParserManager.get_tool_parser("qwen3_coder")(tokenizer)
    text = (
        "<tool_call>\n<function=python>\n<parameter=code>\n# <tool_call>\npass\n</parameter>\n</function>\n</tool_call>"
    )
    if incomplete_tail:
        text += "<tool_call>"
    result = parse(text, parser)
    assert result.malformed_call_count == int(incomplete_tail)
    assert len(result.final_content.tool_calls()) == 1
    assert "# <tool_call>" in result.final_content.tool_calls()[0].arguments["code"]


@pytest.mark.parametrize("arguments", [[], None, 7, "pass"])
async def test_nonobject_arguments_to_optional_tool_are_counted_without_execution(arguments: Any) -> None:
    text = "<tool_call>" + json.dumps({"name": "python", "arguments": arguments}) + "</tool_call>"
    response = parse(text, _hermes())
    tool = PythonTool(
        auto_correct_arguments=False,
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
    )
    backend = ParsedBackend(
        responses=[response, ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="done"))]
    )
    manager, observed = _manager(backend, tool)
    await manager.run()
    assert observed == [(1, None), (0, None)]
    assert tool.executions == 0
    result = manager.conversation.last_turn.elements[1]
    assert isinstance(result, ullm.conv.ToolResult)
    assert result.is_error and result.malformed_call_count == 1
    assert "expected a JSON object" in result.content.to_string()
    retained = manager.conversation.last_turn.elements[0]
    assert isinstance(retained, ullm.conv.LLMResponse)
    assert retained.final_content.tool_calls()[0].arguments == {"_raw": arguments}


def test_nonobject_argument_marker_is_opt_in() -> None:
    result = parse('<tool_call>{"name":"python","arguments":[]}</tool_call>', _hermes(), enabled=False)
    call = result.final_content.tool_calls()[0]
    assert call.arguments == {"_raw": []}
    assert call.malformed_error is None


@pytest.mark.parametrize("text", ["ordinary prose", "", "I need another approach"])
def test_prose_is_not_a_malformed_call(text: str) -> None:
    """Tool-free replies carry no malformed-call penalty."""
    result = parse(text, Parser())
    assert result.malformed_call_count == 0


def test_unrecoverable_tool_response_counts_once() -> None:
    """A parser-wide failure with multiple apparent calls counts once."""
    result = parse("<tool_call>bad</tool_call><tool_call>bad", Parser(fail=True))
    assert result.malformed_call_count == 1
    assert "bad" in result.final_content.to_string()


def test_recovered_json_errors_are_individually_marked() -> None:
    """Recovered calls retain separate invalid-argument markers for the manager."""
    result = parse("<tool_call>bad</tool_call><tool_call>bad</tool_call>", Parser(["{", "{bad"]))
    calls = result.final_content.tool_calls()
    assert len(calls) == 2
    assert all(call.malformed_error for call in calls)
    assert result.malformed_call_count == 0


def test_incomplete_remainder_keeps_valid_recovered_calls() -> None:
    """A response cap does not discard a preceding valid recovered call."""
    result = parse("<tool_call>ok</tool_call><tool_call>", Parser(['{"code":"1+1"}']), capped=True)
    assert len(result.final_content.tool_calls()) == 1
    assert result.final_content.tool_calls()[0].malformed_error is None
    assert result.malformed_call_count == 1
    assert result.stop_reason == "max_tokens"


def test_new_detection_is_opt_in() -> None:
    """ARC-3's existing parser classification is unchanged by ARC-2 validation."""
    result = parse("<tool_call>bad", Parser(), enabled=False)
    assert result.malformed_call_count == 0
    assert result.malformed_error is None


async def test_manager_feedback_renders_in_history_and_continues_original_sampled_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = "reason\n</think>\n\n<tool_call>broken<|im_end|>"
    backend, requests, _ = await _engine_backend(
        monkeypatch, [sampled, "reason\n</think>\n\nanswer<|im_end|>"], classify_tool_calls=True
    )
    backend.trace_recorder = None
    backend._tool_parser = Parser()
    manager, observed = _manager(backend, PythonTool())
    manager.response_manager.sampling_params = _params()
    try:
        await manager.run()
        conversation = manager.conversation
        first_turn = conversation.last_turn
        response, feedback = first_turn.elements
        assert isinstance(response, ullm.conv.LLMResponse)
        assert isinstance(feedback, ullm.conv.UserPrompt)
        feedback_text = feedback.content.to_string()
        assert "Malformed tool call" in feedback_text
        assert observed == [(1, None)]
        messages, images = backend._history_messages(first_turn, conversation, "all")
        assert images == []
        assert messages[-2]["content"] == "<tool_call>broken"
        assert messages[-1] == {"role": "user", "content": feedback_text}

        conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="continue")))
        conversation.status = "ready"
        manager.rearm()
        await manager.run()
        served = requests[0]["prompt"] + TOK.encode(sampled)
        assert requests[1]["prompt"][: len(served)] == served
        delta = TOK.decode(requests[1]["prompt"][len(served) :])
        assert "<|im_start|>user\n" + feedback_text + "<|im_end|>" in delta
        assert delta.index(feedback_text) < delta.index("continue")
        assert conversation.last_turn.last_response._served.continued is True
        assert response._model_contents.outputs[0].token_ids == TOK.encode(sampled)
        assert conversation.history_epoch == 0
    finally:
        await backend.close()
