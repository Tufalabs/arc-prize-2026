"""A reply with no final content can be accepted as the model's turn
(``Backend.accept_malformed_output``) instead of being re-sampled: it stays
in the conversation marked ``malformed``, is counted, and its trace extra
carries the reason. One call can ask the same of its own reply
(``GenericSamplingParameters.keep_empty_reply``), through a proxy too."""

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import pytest
import ullm.backend
import ullm.conv
import ullm.tracing

pytestmark = pytest.mark.asyncio


@dataclass
class _EmptyReplyBackend(ullm.backend.Backend):
    """Streams one chunk and leaves the final content empty."""

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        return "prompt"

    async def _generate(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        yield 1

    def _process_partial_output(self, sampling_params: Any, llm_output: Any, intermediate: Any) -> None:
        intermediate.final_content = ullm.conv.EmptyContent()

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        element.token_usage = ullm.backend.TokenUsage(prompt_tokens=10, completion_tokens=3, num_requests=1)

    def _usage_from_partial(self, llm_output: Any) -> Optional[ullm.backend.TokenUsage]:
        return None


def _conversation() -> ullm.conv.Conversation:
    conv = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="s"), tools=[]),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="u"))],
        status="intermediate",
    )
    conv.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
    return conv


async def _respond(
    backend: ullm.backend.Backend, params: ullm.backend.GenericSamplingParameters | None = None
) -> ullm.conv.Conversation:
    conversation = _conversation()
    async for _ in backend.respond(params or ullm.backend.GenericSamplingParameters(), conversation):
        pass
    return conversation


def _only_response(conversation: ullm.conv.Conversation) -> ullm.conv.LLMResponse:
    turn = conversation.elements[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    (element,) = [e for e in turn.elements if isinstance(e, ullm.conv.LLMResponse)]
    return element


async def test_empty_reply_is_rejected_by_default() -> None:
    with pytest.raises(ullm.backend.MalformedOutputError, match="no final content"):
        await _respond(_EmptyReplyBackend())


async def test_accepted_empty_reply_stays_as_the_models_turn() -> None:
    backend = _EmptyReplyBackend(accept_malformed_output=True)
    conversation = await _respond(backend)
    turn = conversation.elements[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    (element,) = [e for e in turn.elements if isinstance(e, ullm.conv.LLMResponse)]
    assert element.stop_reason == "malformed"
    assert element.malformed_error is not None and "no final content" in element.malformed_error
    assert ullm.backend._is_empty_content(element.final_content)
    # Counted like any completed reply.
    assert backend.total_usage.completion_tokens == 3
    extra = ullm.tracing.step_extra(element, continued=False, history_epoch=0, exact_history=True)
    assert extra["malformed"] == element.malformed_error
    assert extra["stop_reason"] == "malformed"


async def test_accept_flag_is_configuration_not_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag travels with the backend's configuration; the ambient
    environment of the constructing process never decides it."""
    monkeypatch.setenv("ULLM_ACCEPT_MALFORMED_OUTPUT", "1")
    assert _EmptyReplyBackend().accept_malformed_output is False
    assert _EmptyReplyBackend(accept_malformed_output=True).accept_malformed_output is True


async def test_hosted_api_trace_record_carries_the_malformed_marker() -> None:
    """The no-token-ids trace builder (hosted APIs) exposes ``extra["malformed"]``
    exactly like the token-witness path, so a consumer penalizing accepted
    empty replies sees it on every record kind."""
    from types import SimpleNamespace

    from ullm.backends import oai

    backend = oai.OpenAIServer(base_url="http://127.0.0.1:9/v1", api_key="unused", model="m")
    element = ullm.conv.LLMResponse()
    element.stop_reason = "malformed"
    element.malformed_error = "Model produced no final content (test)"
    element._model_contents = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None, reasoning=None))],
        usage=None,
    )
    record = backend._build_trace_step_no_token_ids(
        run_id="run-1",
        sampling_params=ullm.backend.GenericSamplingParameters(),
        llm_input={"model": "m", "messages": [{"role": "user", "content": "u"}]},
        element=element,
    )
    assert record.extra["malformed"] == element.malformed_error
    assert record.extra["stop_reason"] == "malformed"
    assert record.output_text == ""


async def test_one_call_can_keep_its_empty_reply() -> None:
    """Kept for the call that asks, marked as the backend-wide policy marks
    it, and for that call only: the backend still rejects the next one."""
    backend = _EmptyReplyBackend()
    element = _only_response(await _respond(backend, ullm.backend.GenericSamplingParameters(keep_empty_reply=True)))
    assert element.stop_reason == "malformed"
    assert element.malformed_error is not None and "no final content" in element.malformed_error
    assert backend.total_usage.completion_tokens == 3
    with pytest.raises(ullm.backend.MalformedOutputError, match="no final content"):
        await _respond(backend)


async def test_a_proxied_call_keeps_its_empty_reply_on_a_rejecting_host() -> None:
    """The request carries the ask to the host, whose backend rejects empty
    replies otherwise; the client receives the kept reply, marked."""
    import os
    import tempfile

    from ullm.backends import proxy

    with tempfile.TemporaryDirectory() as directory:
        socket_path = os.path.join(directory, "backend.sock")
        handle = await proxy.start_backend_server(_EmptyReplyBackend(), socket_path)
        client = proxy.ProxyBackend(socket_path=socket_path)
        try:
            kept = _only_response(await _respond(client, ullm.backend.GenericSamplingParameters(keep_empty_reply=True)))
            with pytest.raises(ullm.backend.MalformedOutputError, match="no final content"):
                await _respond(client)
        finally:
            await client.close()
            await handle.close()
    assert kept.stop_reason == "malformed"
    assert kept.malformed_error is not None and "no final content" in kept.malformed_error
