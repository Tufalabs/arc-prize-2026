"""Tracing against a hosted API, where no token-id witness exists.

A vLLM server proves each step by re-tokenising the stored messages and
detokenising the output ids. OpenRouter offers neither route, but SFT on image
steps discards token ids anyway (ARC3-Training's ``token_exact`` mode rejects
media steps, so vision trains through ``retemplate``, which reads ``messages``
+ ``output_text`` + ``output_message.tool_calls`` + ``media/``). These cover
that the weaker capability is produced rather than refused, and that the record
it yields is exactly what the trainer needs — including the tool call, which is
where a tool-calling agent's ACTION lives.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import taaf.trace
from ullm.backends import oai


def _server(**kw: Any) -> oai.OpenAIServer:
    return oai.OpenAIServer(
        base_url=kw.pop("base_url", "https://openrouter.ai/api/v1"),
        api_key="test-key",
        model=kw.pop("model", "moonshotai/kimi-k3"),
        streaming=False,
        **kw,
    )


def _completion(
    *, content: str | None, tool_calls: list[Any] | None = None, prompt_tokens: int = 4000, cached: int = 1024
) -> Any:
    """A non-streaming completion shaped like the SDK's parsed object, with no
    ``prompt_token_ids``/``token_ids`` — i.e. what a hosted API returns."""
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls or [], reasoning="thinking hard")
    choice = types.SimpleNamespace(message=message, model_extra={})
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
    )
    return types.SimpleNamespace(choices=[choice], model_extra={}, usage=usage)


def _tool_call(code: str = "print(current_frame.level)") -> Any:
    return types.SimpleNamespace(
        id="python_0",
        function=types.SimpleNamespace(name="python", arguments=f'{{"code": {code!r}}}'),
    )


# --- capability ---------------------------------------------------------------


def test_hosted_api_is_trace_capable_at_the_weaker_grade() -> None:
    """The probe failing must not mean "no traces": it means no token ids."""
    cap = _server()._trace_capability_no_token_ids()
    assert cap is not None
    assert cap.provider_name == "openrouter"
    assert cap.validation_method == "platypus.oai.no_token_witness.v1"
    assert cap.provider_model_id == "moonshotai/kimi-k3"
    assert cap.model_id == "moonshotai/kimi-k3"
    # Unknown rather than invented: the provider publishes no weights identity.
    assert cap.model_revision == ""
    assert cap.model_weight_hash == ""


def test_non_openrouter_host_is_named_from_the_url() -> None:
    cap = _server(base_url="https://api.openai.com/v1")._trace_capability_no_token_ids()
    assert cap.provider_name == "api.openai.com"


def test_capability_arms_the_degraded_step_builder() -> None:
    server = _server()
    assert server._trace_no_token_ids is False
    server._trace_capability_no_token_ids()
    assert server._trace_no_token_ids is True
    # The drift guard must stay disarmed: there is no served-weights identity
    # to re-resolve, and _trace_check_identity_drift no-ops on None.
    assert server._trace_identity is None


# --- the record --------------------------------------------------------------


def _build(completion: Any, messages: list[dict[str, Any]] | None = None) -> Any:
    server = _server()
    server._trace_capability_no_token_ids()
    # A SimpleNamespace stands in for the LLMResponse the builder reads.
    element: Any = types.SimpleNamespace(_model_contents=completion, stop_reason="tool_use")
    llm_input = {"model": "moonshotai/kimi-k3", "messages": messages or [{"role": "user", "content": "go"}]}
    return server._build_trace_step_no_token_ids(
        run_id="run-1",
        sampling_params=oai.ullm.backend.GenericSamplingParameters(temperature=0.6),
        llm_input=llm_input,
        element=element,
    )


def test_record_carries_no_token_ids_and_says_so() -> None:
    rec = _build(_completion(content="I will move left."))
    assert rec.input_token_ids == []
    assert rec.output_token_ids == []
    assert rec.input_token_reconstruction_status == "unavailable"
    assert rec.output_token_reconstruction_status == "unavailable"


def test_statuses_are_accepted_by_taaf() -> None:
    """The whole approach rests on taaf permitting these, so assert it here
    rather than trusting a docstring."""
    rec = _build(_completion(content="x"))
    assert (
        rec.input_token_reconstruction_status
        in taaf.trace._ACCEPTED_STATUSES_BY_FIELD["input_token_reconstruction_status"]
    )
    assert rec.output_token_reconstruction_status in taaf.trace.ACCEPTED_RECONSTRUCTION_STATUSES


def test_the_action_survives_as_a_tool_call() -> None:
    """The bug this guards: a completion whose action lives only in tool_calls.
    Dropping it trains the model to narrate and never act."""
    rec = _build(_completion(content=None, tool_calls=[_tool_call()]))
    calls = rec.output_message["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["python"]
    assert "current_frame.level" in calls[0]["function"]["arguments"]


def test_empty_prose_with_a_tool_call_is_not_padded() -> None:
    """A tool-call-only step is a real action; output_text stays honestly empty
    rather than being filled with a synthetic rendering no provider emitted."""
    rec = _build(_completion(content=None, tool_calls=[_tool_call()]))
    assert rec.output_text == ""
    assert rec.output_message["tool_calls"]  # the action is still recorded


def test_reasoning_is_kept_on_the_template_contract_key() -> None:
    rec = _build(_completion(content="prose"))
    assert rec.output_message["reasoning_content"] == "thinking hard"


def test_uncached_input_comes_from_usage_not_from_ids() -> None:
    """len(prompt_ids) would be 0 here, making the old expression negative."""
    rec = _build(_completion(content="x", prompt_tokens=4000, cached=1024))
    assert rec.uncached_input_tokens == 2976


def test_uncached_input_never_negative_when_usage_is_odd() -> None:
    rec = _build(_completion(content="x", prompt_tokens=0, cached=512))
    assert rec.uncached_input_tokens == 0


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([{"role": "user", "content": "text only"}], "not_applicable"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
            "unavailable",
        ),
    ],
)
def test_media_status_reflects_whether_the_call_had_images(messages: list[dict[str, Any]], expected: str) -> None:
    rec = _build(_completion(content="x"), messages=messages)
    assert rec.media_reconstruction_status == expected


def test_request_payload_round_trips_the_messages() -> None:
    """The collector derives payload hashes and request_args from this, and
    retemplate reads the prompt out of it."""
    msgs = [{"role": "user", "content": "go"}]
    rec = _build(_completion(content="x"), messages=msgs)
    assert rec.messages == msgs
    assert rec.request_payload["messages"] == msgs
    # The malformed marker is present on every record kind (None here).
    assert rec.extra == {"stop_reason": "tool_use", "malformed": None}


# --- the continuation bootstrap needs no filesystem ---------------------------


@pytest.mark.asyncio
async def test_continuation_probe_resolves_the_tokenizer_without_the_export_file(monkeypatch: Any) -> None:
    """A remote training server serves an adapter whose runtime export lives
    on ITS disk. Continuation needs only the server's ids and tokenizer
    identity, resolved from /v1/models; the export-file dereference stays
    with tracing (U7), which is where a client-reachable path is a contract."""
    server = _server(base_url="http://train.example:8000/v1", model="policy-lora")

    async def tokenize(*args: Any, **kwargs: Any) -> list[int]:
        return [1, 2, 3]

    async def create(**kwargs: Any) -> Any:
        return types.SimpleNamespace(
            model_extra={"prompt_token_ids": [1, 2, 3]},
            choices=[types.SimpleNamespace(model_extra={"token_ids": [4]})],
        )

    class _Client:
        async def request(self, method: str, url: str, *, json: Any = None) -> Any:
            assert method == "GET"
            assert json is None
            assert url.endswith("/v1/models")
            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "data": [{"id": "policy-lora", "root": "/server-only/run/adapter", "parent": "Qwen/Base"}]
                },
            )

    monkeypatch.setattr(server, "_trace_tokenize", tokenize)
    monkeypatch.setattr(server, "_server_client", lambda: _Client())
    server.__dict__["_client"] = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    await server.probe_continuation()
    assert server.supports_token_continuation()
    assert server._trace_hf_tokenizer_id == "Qwen/Base"
    # The trace-grade probe still insists on the export file: that contract is unchanged.
    with pytest.raises(FileNotFoundError, match="runtime_adapter_summary.json"):
        await server.trace_capability()
