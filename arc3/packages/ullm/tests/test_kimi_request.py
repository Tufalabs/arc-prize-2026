"""KimiServer's request shape, its retry window and its trace extras: a past
assistant message the API would refuse as empty, the overload retries, and the
per-token alternatives a traced call asks for and keeps. No network —
_prepare_input and the trace record are pure and the backoff is arithmetic."""

import json
from typing import Any

import openai.types.chat
import ullm.backend
import ullm.backends.kimi
import ullm.backends.oai
import ullm.conv
import ullm.tracing

#: Two output tokens as the API returns them, the top alternatives cut to two.
_LOGPROBS = [
    {
        "token": "391",
        "bytes": [51, 57, 49],
        "logprob": -0.01,
        "top_logprobs": [
            {"token": "391", "bytes": [51, 57, 49], "logprob": -0.01},
            {"token": "39", "bytes": [51, 57], "logprob": -4.6},
        ],
    },
    {
        "token": ".",
        "bytes": [46],
        "logprob": -0.2,
        "top_logprobs": [{"token": ".", "bytes": [46], "logprob": -0.2}],
    },
]


def _server() -> ullm.backends.kimi.KimiServer:
    return ullm.backends.kimi.KimiServer(model="kimi-k3.1-pro", api_key="test-key")


def _completion(
    content: str, *, reasoning: str | None = None, logprobs: list[dict] | None = None
) -> openai.types.chat.ChatCompletion:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    choice: dict[str, Any] = {"finish_reason": "stop", "index": 0, "message": message}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs, "refusal": None}
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "choices": [choice],
            "created": 0,
            "model": "kimi-k3.1-pro",
            "object": "chat.completion",
        }
    )


def _conversation(past: ullm.conv.LLMResponse) -> tuple[ullm.conv.Conversation, ullm.conv.ResponseTurn]:
    """System, a user prompt, one past reply, a nudge, and a fresh turn."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    done = ullm.conv.ResponseTurn()
    done.elements.append(past)
    conversation.elements.append(done)
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="make a call")))
    turn = ullm.conv.ResponseTurn()
    conversation.elements.append(turn)
    return conversation, turn


def _assistant_messages(past: ullm.conv.LLMResponse) -> list[dict]:
    conversation, turn = _conversation(past)
    kwargs = _server()._prepare_input(ullm.backend.GenericSamplingParameters(), turn, conversation, "all")
    return [m for m in kwargs["messages"] if m["role"] == "assistant"]


def test_an_empty_past_reply_is_sent_with_a_placeholder() -> None:
    """A reply with only reasoning (or nothing) would be refused by the API
    as an empty assistant message; it goes with the placeholder text, its
    reasoning still replayed."""
    past = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text="thinking"), final_content=ullm.conv.StringContent(text="")
    )
    past._model_contents = _completion("", reasoning="thinking")
    (assistant,) = _assistant_messages(past)
    assert assistant["content"] == "(no text)"
    assert assistant["reasoning_content"] == "thinking"


def test_a_past_reply_with_text_is_sent_as_it_was() -> None:
    past = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="the answer"))
    past._model_contents = _completion("the answer")
    (assistant,) = _assistant_messages(past)
    assert assistant["content"] == "the answer"


def test_overload_retries_last_twelve_hours() -> None:
    """An overload of hours is waited out by time, not by a count whose
    length depends on how long each refusal takes to come back."""
    assert _server().transient_retry_window_s == 12 * 3600


def test_a_traced_call_asks_for_the_top_logprobs() -> None:
    """The alternatives are asked for only when a trace keeps them."""
    past = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="hi"))
    past._model_contents = _completion("hi")
    conversation, turn = _conversation(past)
    server = _server()
    untraced = server._prepare_input(ullm.backend.GenericSamplingParameters(), turn, conversation, "all")
    assert "logprobs" not in untraced and "top_logprobs" not in untraced
    # Traced at the hosted-API grade, as the capability probe leaves an API.
    server.trace_recorder = object()  # any recorder: the request only checks that one is set
    server._trace_no_token_ids = True
    traced = server._prepare_input(ullm.backend.GenericSamplingParameters(), turn, conversation, "all")
    assert traced["logprobs"] is True and traced["top_logprobs"] == 20


def test_every_call_asks_for_the_hour_long_cache() -> None:
    """Five minutes lose a game's cache to any retry wait past them; None
    leaves the API its own five minutes."""
    past = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="hi"))
    past._model_contents = _completion("hi")
    conversation, turn = _conversation(past)
    server = _server()
    kwargs = server._prepare_input(ullm.backend.GenericSamplingParameters(), turn, conversation, "all")
    assert kwargs["extra_body"]["prompt_cache_options"] == {"mode": "implicit", "ttl": "1h"}
    server.prompt_cache_ttl = None
    kwargs = server._prepare_input(ullm.backend.GenericSamplingParameters(), turn, conversation, "all")
    assert "prompt_cache_options" not in kwargs.get("extra_body", {})


def test_the_trace_step_keeps_the_logprobs_as_returned() -> None:
    """The API's per-token alternatives land in the step's extra, packed while
    the collector holds the step and unpacked to exactly what the API
    returned: TAAF's own logprobs field aligns to output_token_ids, which an
    API trace has none of, so it stays None."""
    element = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="391."))
    element._model_contents = _completion("391.", logprobs=_LOGPROBS)
    record = _server()._build_trace_step_no_token_ids(
        run_id="run",
        sampling_params=ullm.backend.GenericSamplingParameters(),
        llm_input={"messages": [{"role": "user", "content": "17 * 23?"}]},
        element=element,
    )
    packed = record.extra["teacher_logprobs"]
    assert isinstance(packed, ullm.tracing.PackedJson)
    assert packed.unpack() == _LOGPROBS
    assert ullm.tracing.unpack_extra(record.extra)["teacher_logprobs"] == _LOGPROBS
    assert record.logprobs is None and record.output_token_ids == []


def test_the_conversation_keeps_no_logprobs() -> None:
    """Once the trace step has them, the alternatives leave the completion the
    conversation keeps (and the game receives whole); the reply itself does
    not change."""
    element = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="391."))
    element._model_contents = _completion("391.", logprobs=_LOGPROBS)
    _server()._shed_served_ids(element)
    assert element._model_contents.choices[0].logprobs is None
    assert element._model_contents.choices[0].message.content == "391."


def test_packed_json_is_lossless() -> None:
    """Floats at full precision, ints, unicode and byte lists come back equal,
    so a trace written from the unpacked value is the one written from the
    original."""
    value = [
        {"token": " caf\u00e9", "bytes": [32, 99, 97, 102, 195, 169], "logprob": -1.0000000000000002e-05},
        {"token": "\u2192", "bytes": [226, 134, 146], "logprob": -0.1 + -0.2, "top_logprobs": []},
        {"token": "x", "bytes": [120], "logprob": -9999999.123456789, "top_logprobs": [{"token": "y", "logprob": 0.0}]},
    ]
    unpacked = ullm.tracing.PackedJson.pack(value).unpack()
    assert unpacked == value
    assert json.dumps(unpacked) == json.dumps(value)


def test_a_step_without_logprobs_carries_none() -> None:
    element = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="391"))
    element._model_contents = _completion("391")
    record = _server()._build_trace_step_no_token_ids(
        run_id="run",
        sampling_params=ullm.backend.GenericSamplingParameters(),
        llm_input={"messages": [{"role": "user", "content": "17 * 23?"}]},
        element=element,
    )
    assert "teacher_logprobs" not in record.extra
