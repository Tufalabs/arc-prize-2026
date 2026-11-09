"""Token-native serving on OpenAIServer (U5): the request IS the id ledger.

Under exact history every call after the first extends the served ids on
``/v1/completions``; these tests pin the three properties that make that a
ledger rather than a cache: the engine's account of a continued call must
equal the sent ids exactly (``check_continuation_served`` — raised, never
absorbed), ``token_count`` weighs the very id list ``respond`` then sends
(no ``/tokenize``, no second render), and the delta between them is
tokenized exactly once (the continuation memo).

The integration-shaped run drives two real ``respond`` calls through a fake
OpenAI client — a chat completion that returns engine ids, then a continued
completion that echoes what it was sent — the wire shape a vLLM 0.25.1
server produces with ``return_token_ids``.
"""

from __future__ import annotations

from typing import Any

import openai.types
import openai.types.chat
import pytest
import ullm.backend
import ullm.conv
from ullm.backends import oai

EOS = "<|im_end|>"


class _StubTokenizer:
    """A Qwen-shaped template over a character tokenizer (the
    test_continuation fixture), with an encode-call ledger so once-only
    tokenization is observable."""

    specials = {EOS: 1000, "<|im_start|>": 1001, "<think>": 1002, "</think>": 1003}
    eos_token = EOS
    eos_token_id = 1000
    is_fast = True

    def __len__(self) -> int:
        return 0x110000

    def convert_ids_to_tokens(self, ids: list[int], **kwargs: Any) -> list[str]:
        """Expose the same characters to the engine's incremental decoder."""
        return [self.decode([token]) for token in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join the character pieces without introducing spaces."""
        return "".join(tokens)

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, **kwargs: Any) -> str:
        out = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                body = "<think>\n" + (message.get("reasoning_content") or "") + "\n</think>\n\n"
                body += message.get("content") or ""
                out += "<|im_start|>assistant\n" + body + EOS + "\n"
            else:
                out += "<|im_start|>" + role + "\n" + str(message.get("content")) + EOS + "\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n"
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded.append(text)
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special, sid in self.specials.items():
                if text.startswith(special, i):
                    ids.append(sid)
                    i += len(special)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        by_id = {sid: special for special, sid in self.specials.items()}
        return "".join(by_id.get(i, chr(i)) for i in ids)


# --- the invariant, in isolation ------------------------------------------


def _completion(*, prompt_token_ids: list[int] | None, prompt_tokens: int | None, text: str = "x") -> Any:
    body: dict[str, Any] = {
        "id": "c",
        "object": "text_completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
    }
    if prompt_token_ids is not None:
        body["choices"][0]["prompt_token_ids"] = prompt_token_ids
    if prompt_tokens is not None:
        body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}
    return openai.types.Completion.model_validate(body)


def test_an_agreeing_engine_passes_the_served_check() -> None:
    oai.check_continuation_served([1, 2, 3], _completion(prompt_token_ids=[1, 2, 3], prompt_tokens=3))


def test_a_different_echo_is_a_loud_violation() -> None:
    """The prompt was ids; there is no rendering in between, so an echo that
    differs means the serving stack changed the sequence under us."""
    with pytest.raises(RuntimeError, match="token-native invariant violated"):
        oai.check_continuation_served([1, 2, 3], _completion(prompt_token_ids=[1, 2, 4], prompt_tokens=3))


def test_a_different_bill_is_a_loud_violation() -> None:
    with pytest.raises(RuntimeError, match="token-native invariant violated"):
        oai.check_continuation_served([1, 2, 3], _completion(prompt_token_ids=None, prompt_tokens=7))


def test_a_response_without_the_extras_has_nothing_to_check() -> None:
    """No echo and no usage: nothing billed to compare. The witness path
    raises separately under exact history when the ids are missing."""
    oai.check_continuation_served([1, 2, 3], _completion(prompt_token_ids=None, prompt_tokens=None))


# --- the integration-shaped run -------------------------------------------


class _FakeChat:
    """``chat.completions.create`` for the fresh first call: renders nothing
    (the ids are asserted opaque), returns engine ids and an exact bill."""

    def __init__(self, tokenizer: _StubTokenizer) -> None:
        self._tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        prompt_ids = list(range(200, 230))  # engine-owned: the client never derives them
        output_ids = self._tokenizer.encode("\n</think>\n\nok") + [self._tokenizer.eos_token_id]
        return openai.types.chat.ChatCompletion.model_validate(
            {
                "id": "chat",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                        "token_ids": output_ids,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(output_ids),
                    "total_tokens": len(prompt_ids) + len(output_ids),
                },
                "prompt_token_ids": prompt_ids,
            }
        )


class _FakeCompletions:
    """``completions.create`` for continued calls: echoes the sent ids and
    bills exactly their length, as vLLM does for an id prompt. ``bill_skew``
    misbills to stand in for a serving stack that changed the sequence."""

    def __init__(self, tokenizer: _StubTokenizer, bill_skew: int = 0) -> None:
        self._tokenizer = tokenizer
        self.bill_skew = bill_skew
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        sent = [int(t) for t in kwargs["prompt"]]
        output_ids = self._tokenizer.encode("\n</think>\n\ndone") + [self._tokenizer.eos_token_id]
        return openai.types.Completion.model_validate(
            {
                "id": "c",
                "object": "text_completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "text": "\n</think>\n\ndone" + EOS,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "token_ids": output_ids,
                        "prompt_token_ids": sent,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(sent) + self.bill_skew,
                    "completion_tokens": len(output_ids),
                    "total_tokens": len(sent) + self.bill_skew + len(output_ids),
                },
            }
        )


class _Api:
    """The two-route shape of the SDK client the backend drives."""

    def __init__(self, chat: _FakeChat, completions: _FakeCompletions) -> None:
        self.chat = type("_Chat", (), {"completions": chat})()
        self.completions = completions


def _no_tokenize_client() -> Any:
    class _Refuse:
        async def post(self, url: str, json: Any = None) -> Any:  # noqa: A002
            raise AssertionError(f"token-native serving must not touch /tokenize, but posted to {url}")

    return _Refuse()


def _server(tokenizer: _StubTokenizer, bill_skew: int = 0) -> tuple[oai.OpenAIServer, _FakeChat, _FakeCompletions]:
    server = oai.OpenAIServer(
        base_url="http://vllm.local/v1",
        api_key="k",
        model="m",
        streaming=False,
        send_back_reasoning="all",
        reasoning_history_field="reasoning",  # the production wiring: vLLM's parser fills it
        exact_history=True,
    )
    chat = _FakeChat(tokenizer)
    completions = _FakeCompletions(tokenizer, bill_skew=bill_skew)
    server._client = _Api(chat, completions)  # pyright: ignore[reportAttributeAccessIssue] — the transient SDK client seam
    server._trace_hf_tokenizer_id = "stub"
    server._trace_hf_tokenizer = tokenizer
    server._server_client = lambda: _no_tokenize_client()  # type: ignore[method-assign]
    return server, chat, completions


def _conversation() -> ullm.conv.Conversation:
    return ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(text="sys"), reasoning_effort=None, tools=[]
        ),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hi")),
            ullm.conv.ResponseTurn(elements=[], status="intermediate"),
        ],
        status="intermediate",
    )


_PARAMS = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=64, send_back_reasoning="all")


async def _first_answer(server: oai.OpenAIServer, conversation: ullm.conv.Conversation) -> ullm.conv.LLMResponse:
    async for _ in server.respond(_PARAMS, conversation):
        pass
    answer = conversation.last_turn.last_response
    assert answer is not None and answer._served is not None, "the first call must leave a witness"
    return answer


def _extend(conversation: ullm.conv.Conversation) -> None:
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="again")))
    conversation.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))


async def test_the_request_is_the_ledger_and_is_weighed_before_it_is_sent() -> None:
    """token_count answers the exact id list respond() then sends: same
    length, ids extending the served history append-only, no /tokenize
    anywhere, and the delta between the calls tokenized exactly once."""
    tokenizer = _StubTokenizer()
    server, chat, completions = _server(tokenizer)
    conversation = _conversation()
    answer = await _first_answer(server, conversation)
    witness = answer._served
    served = [*witness.prefix_ids, *witness.output_ids]

    _extend(conversation)
    encodes_before = len(tokenizer.encoded)
    counted = await server.token_count(_PARAMS, conversation)
    encodes_after_count = len(tokenizer.encoded)
    assert counted is not None and counted > len(served), "the count must be the ledger plus the new delta"
    assert encodes_after_count == encodes_before + 1, "weighing the request tokenizes the delta once"

    async for _ in server.respond(_PARAMS, conversation):
        pass
    # The only encode since the count is the fake engine producing its own
    # output ids — the delta the count weighed was NOT tokenized again.
    assert tokenizer.encoded[encodes_after_count:] == ["\n</think>\n\ndone"], (
        "sending the weighed request must not re-tokenize the delta"
    )
    (call,) = completions.calls
    sent = [int(t) for t in call["prompt"]]
    assert len(sent) == counted, "the count and the sent request are the same id list"
    assert sent[: len(served)] == [int(t) for t in served], "append-only: the served history is a prefix"
    assert len(chat.calls) == 1, "only the first call renders; everything after extends ids"


async def test_an_exhausted_connection_names_the_route_and_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that closes without responding surfaces as a bare
    RemoteProtocolError; after the retries are spent, the raised error must
    carry the route, the request size and the elapsed time — the exact
    context that separates a read-timeout on the path from bad request
    content."""
    import httpx

    tokenizer = _StubTokenizer()
    server, _chat, completions = _server(tokenizer)
    conversation = _conversation()
    await _first_answer(server, conversation)
    _extend(conversation)

    async def _closed(**kwargs: Any) -> Any:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    monkeypatch.setattr(completions, "create", _closed)
    monkeypatch.setattr(oai, "_CONNECTION_RETRY_DELAYS_S", ())  # one attempt; the test needs no real backoff
    with pytest.raises(httpx.RemoteProtocolError) as excinfo:
        async for _ in server.respond(_PARAMS, conversation):
            pass
    notes = "\n".join(getattr(excinfo.value, "__notes__", []))
    assert "/v1/completions (id prompt)" in notes
    assert "prompt ids" in notes and "read-timeout" in notes


async def test_a_misbilling_engine_fails_the_continued_call_loudly() -> None:
    tokenizer = _StubTokenizer()
    server, _chat, _completions = _server(tokenizer, bill_skew=3)
    conversation = _conversation()
    await _first_answer(server, conversation)
    _extend(conversation)
    with pytest.raises(RuntimeError, match="token-native invariant violated"):
        async for _ in server.respond(_PARAMS, conversation):
            pass


if __name__ == "__main__":
    pytest.main([__file__])
