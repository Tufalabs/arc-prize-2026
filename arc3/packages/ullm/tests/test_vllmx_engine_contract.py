"""Shared engine path: U5 exact history and text evidence, U7 policy identity.

Malformed draws, stop tokens and buffered UTF-8 retain the sampled sequence.
"""

from __future__ import annotations

import json
import pickle
import types
from typing import Any

import pytest
import transformers
import ullm.backend
import ullm.conv
import ullm.tracing

vllmx = pytest.importorskip("ullm.backends.vllmx")


class _Tokenizer:
    is_fast = True
    eos_token = "<|im_end|>"
    eos_token_id = 1000
    bos_token_id = None
    pad_token_id = 1000
    chat_template = "test-qwen"
    special_tokens_map = {"eos_token": eos_token}
    specials = {eos_token: 1000, "<|im_start|>": 1001, "<think>": 1002, "</think>": 1003}
    backend_tokenizer = types.SimpleNamespace(
        to_str=lambda: json.dumps(
            {
                "model": {"type": "test", "vocab": {"ordinary": 65, "<|im_end|>": 1000}},
                "normalizer": None,
                "decoder": {"type": "test"},
            }
        )
    )

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        ids = []
        while text:
            for token, token_id in self.specials.items():
                if text.startswith(token):
                    ids.append(token_id)
                    text = text[len(token) :]
                    break
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        del kwargs
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse.get(i, chr(i)) for i in ids)

    def __len__(self) -> int:
        return 0x110000

    def convert_ids_to_tokens(self, ids: list[int], **kwargs: Any) -> list[str]:
        return [self.decode([token]) for token in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def apply_chat_template(
        self, messages: Any, *, add_generation_prompt: bool, tokenize: bool = True, **kwargs: Any
    ) -> Any:
        del kwargs
        text = ""
        for message in messages:
            role = message["role"]
            body = message.get("content") or ""
            if role == "assistant":
                body = "<think>\n" + (message.get("reasoning_content") or "") + "\n</think>\n\n" + body
            text += "<|im_start|>" + role + "\n" + body + self.eos_token + "\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return self.encode(text) if tokenize else text


class _Parser:
    end_token_id = 1003

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str, str]:
        del request
        if text.startswith("broken-parser"):
            raise ValueError("malformed reasoning")
        parts = text.split("</think>", 1)
        return parts[0], parts[1] if len(parts) == 2 else ""

    def count_reasoning_tokens(self, ids: Any) -> int:
        del ids
        return 0


TOK = _Tokenizer()
ADAPTER = {"name": "policy", "id": 4, "path": "/immutable/adapter-4", "server_instance_id": "server-1"}


def _params(injection: str | None = None) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(
        temperature=1.0,
        top_p=1.0,
        max_tokens=128,
        send_back_reasoning="all",
        response_injection=injection,
        extra_params={"top_k": -1, "presence_penalty": 0.0, "skip_special_tokens": False, "logprobs": 0},
    )


def _conversation() -> ullm.conv.Conversation:
    return ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys")),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")),
            ullm.conv.ResponseTurn(status="intermediate"),
        ],
        status="intermediate",
    )


class _Engine:
    remote_engine = True
    metadata = {"server_instance_id": "server-1"}

    async def connect(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def allows_generation_replay(self) -> bool:
        return False

    def __init__(self, tokenizer: Any, outputs: list[Any], requests: list[Any]) -> None:
        self.tokenizer, self.samples, self.requests = tokenizer, outputs, requests
        self.model_config = types.SimpleNamespace(
            model="test-base", revision="", tokenizer="test-tokenizer", tokenizer_revision=""
        )
        self.lora_request = types.SimpleNamespace(lora_name="policy", lora_int_id=4, lora_path="/immutable/adapter-4")
        self.aborted = []

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any = None,
        request_context: Any = None,
    ) -> Any:
        self.requests.append(
            {
                "prompt": list(prompt["prompt_token_ids"]),
                "user": request_id,
                "context": request_context,
                "params": sampling_params,
            }
        )
        sampled = self.samples.pop(0)
        ids = self.tokenizer.encode(sampled) if isinstance(sampled, str) else sampled["output_token_ids"]
        stopped = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        finish = "stop" if stopped else "length"
        text = self.tokenizer.decode(ids[:-1] if stopped else ids)
        if isinstance(sampled, dict):
            text = sampled["text"]
            finish = sampled.get("finish_reason", finish)
        output = types.SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=list(prompt["prompt_token_ids"]),
            lora_request=lora_request,
            adapter=ADAPTER,
            num_cached_tokens=0,
            finished=True,
            outputs=[
                types.SimpleNamespace(
                    text=text,
                    token_ids=ids,
                    finish_reason=finish,
                    stop_reason=sampled.get("stop_reason") if isinstance(sampled, dict) else None,
                    logprobs=[{t: types.SimpleNamespace(logprob=-0.5, rank=None)} for t in ids],
                )
            ],
        )
        if isinstance(sampled, dict):
            for name, value in sampled.get("mutations", {}).items():
                setattr(
                    output
                    if name in ("request_id", "prompt_token_ids", "lora_request", "finished")
                    else output.outputs[0],
                    name,
                    value,
                )
        yield output


async def _backend(monkeypatch: Any, outputs: list[Any], *, tokenizer: Any = TOK, **kwargs: Any) -> Any:
    requests, records = [], []
    backend = vllmx.VLLM(
        engine=_Engine(tokenizer, outputs, requests),
        exact_history=True,
        accept_malformed_output=True,
        verify_engine_output=True,
        token_count_evidence=True,
        require_explicit_sampling=True,
        **kwargs,
    )
    backend._tokenizer_setup_done = True
    backend._reasoning_parser = backend._reasoning_parser_no_thinking = _Parser()
    backend.trace_recorder = records.append
    return backend, requests, records


async def _respond(backend: Any, params: Any, conversation: Any) -> Any:
    async for _ in backend.respond(params, conversation, run_id="run"):
        pass
    return conversation.last_turn.last_response


@pytest.mark.parametrize(
    "first", ["!!!!!!!!!!!!!!!!!!!!", "broken-parser content", "think\n</think>\n\nanswer<|im_end|>"]
)
async def test_every_sampled_reply_is_retained_and_next_prompt_extends_it(monkeypatch: Any, first: str) -> None:
    backend, requests, records = await _backend(monkeypatch, [first, "done\n</think>\n\nanswer<|im_end|>"])
    conv = _conversation()
    params = _params()
    count = await backend._token_count(params, conv)
    first_response = await _respond(backend, params, conv)
    assert first_response._served is not None
    assert records[0].output_token_ids == TOK.encode(first)
    assert records[0].logprobs == [
        {"token_id": token, "logprob": -0.5, "rank": None, "top": [[token, -0.5, None]]} for token in TOK.encode(first)
    ]
    assert records[0].extra["adapter"] == ADAPTER
    assert count == len(requests[0]["prompt"])
    if first != "think\n</think>\n\nanswer<|im_end|>":
        assert first_response.stop_reason == "max_tokens"
    conv.elements.extend(
        [
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="next")),
            ullm.conv.ResponseTurn(status="intermediate"),
        ]
    )
    conv.status = "intermediate"
    await _respond(backend, params, conv)
    previous = requests[0]["prompt"] + TOK.encode(first)
    assert requests[1]["prompt"][: len(previous)] == previous
    assert requests[0]["user"] != requests[1]["user"]
    assert records[0].extra["continued"] is False
    assert records[1].extra["continued"] is True
    await backend.close()


@pytest.mark.parametrize("text", ["different answer", "answer!", "answe", "", "answer<|im_end|>"])
async def test_valid_but_inconsistent_server_text_is_fatal_before_parsing_or_progress(
    monkeypatch: Any, text: str
) -> None:
    backend, requests, records = await _backend(
        monkeypatch, [{"text": text, "output_token_ids": TOK.encode("answer<|im_end|>")}]
    )

    def progress(run_id: str) -> dict[str, Any]:
        return {"score": 0}

    def provider(run_id: str, previous: str | None) -> dict[str, Any]:
        return progress(run_id)

    backend.request_context_provider = provider
    try:
        with pytest.raises(ullm.tracing.TraceReconstructionError, match="text does not match"):
            await _respond(backend, _params(), _conversation())
        assert len(requests) == 1 and not records
        assert not backend._completed_requests
    finally:
        await backend.close()


@pytest.mark.parametrize("injection", [None, "prefix "])
async def test_engine_stop_text_parses_without_losing_terminal_sampled_token(
    monkeypatch: Any, injection: str | None
) -> None:
    detokenizers = pytest.importorskip("vllm.v1.engine.detokenizer")

    class Detokenizer(detokenizers.BaseIncrementalDetokenizer):
        def decode_next(self, next_token_id: int) -> str:
            return TOK.decode([next_token_id])

    producer = Detokenizer(
        types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(
                stop=None,
                min_tokens=0,
                include_stop_str_in_output=False,
            )
        )
    )
    output_ids = TOK.encode("reason\n</think>\n\nanswer<|im_end|>")
    producer.update(output_ids, stop_terminated=True)
    engine_text = producer.get_next_output_text(finished=True, delta=False)
    assert producer.output_token_ids == output_ids
    assert engine_text == "reason\n</think>\n\nanswer"
    assert engine_text != TOK.decode(output_ids)
    backend, requests, records = await _backend(
        monkeypatch,
        [
            {"text": engine_text, "output_token_ids": output_ids},
            "next\n</think>\n\nanswer<|im_end|>",
        ],
    )
    conv = _conversation()
    try:
        first = await _respond(backend, _params(injection), conv)
        assert first.final_content.to_string() == "answer"
        assert first.reasoning_content.to_string() == (injection or "") + "reason"
        assert first._model_contents.outputs[0].text == (injection or "") + engine_text
        assert records[0].output_token_ids == output_ids
        assert len(records[0].logprobs) == len(output_ids)
        conv.elements.extend(
            [
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="continue")),
                ullm.conv.ResponseTurn(status="intermediate"),
            ]
        )
        conv.status = "intermediate"
        await _respond(backend, _params(injection), conv)
        served = requests[0]["prompt"] + output_ids
        assert requests[1]["prompt"][: len(served)] == served
        assert records[1].extra["continued"] is True
    finally:
        await backend.close()


async def test_incrementally_buffered_utf8_is_retained_without_text_reconstruction(monkeypatch: Any) -> None:
    import tokenizers

    vocab = {chr(i): 10 + i for i in range(128)}
    vocab.update({"<0xE2>": 0, "<0x82>": 1, "<0xAC>": 2, "[UNK]": 3, **TOK.specials})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = tokenizers.decoders.Sequence([tokenizers.decoders.ByteFallback(), tokenizers.decoders.Fuse()])
    stream = tokenizers.decoders.DecodeStream(ids=[], skip_special_tokens=False)
    # vLLM's fast detokenizer keeps incomplete UTF-8 buffered; decoding the
    # same output in isolation produces replacement text instead.
    engine_text = stream.step(tokenizer, 0) or ""
    assert engine_text == "" and tokenizer.decode([0]) == "�"
    wrapper = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<think>", "</think>"],
        chat_template="test-qwen",
    )
    wrapper.apply_chat_template = types.MethodType(_Tokenizer.apply_chat_template, wrapper)
    backend, requests, records = await _backend(
        monkeypatch,
        [
            {"text": engine_text, "output_token_ids": [0]},
            {"text": "€", "output_token_ids": [0, 1, 2]},
        ],
        tokenizer=wrapper,
    )
    conv = _conversation()
    try:
        first = await _respond(backend, _params(), conv)
        assert first.stop_reason == "max_tokens"
        assert first._model_contents.outputs[0].text == ""
        assert records[0].output_token_ids == [0]
        assert records[0].logprobs[0]["logprob"] == -0.5
        conv.elements.extend(
            [
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="continue")),
                ullm.conv.ResponseTurn(status="intermediate"),
            ]
        )
        conv.status = "intermediate"
        await _respond(backend, _params(), conv)
        served = requests[0]["prompt"] + [0]
        assert requests[1]["prompt"][: len(served)] == served
    finally:
        await backend.close()


@pytest.mark.parametrize("error_type", [ValueError])
async def test_parser_shape_failures_preserve_truncation_and_continuation(monkeypatch: Any, error_type: Any) -> None:
    text = "broken-parser content"
    backend, requests, records = await _backend(
        monkeypatch, [{"text": text, "output_token_ids": TOK.encode(text), "finish_reason": "length"}, "done"]
    )

    class BrokenParser(_Parser):
        def extract_reasoning(self, text: str, request: Any = None) -> tuple[str, str]:
            if text.startswith("broken-parser"):
                raise error_type("bad model text")
            return super().extract_reasoning(text, request)

    backend._reasoning_parser = backend._reasoning_parser_no_thinking = BrokenParser()
    conv = _conversation()
    response = await _respond(backend, _params(), conv)
    assert response.stop_reason == "max_tokens"
    assert response.malformed_error
    assert records[0].extra["malformed"] == response.malformed_error
    conv.elements.extend(
        [
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="continue")),
            ullm.conv.ResponseTurn(status="intermediate"),
        ]
    )
    conv.status = "intermediate"
    await _respond(backend, _params(), conv)
    previous = requests[0]["prompt"] + TOK.encode(text)
    assert requests[1]["prompt"][: len(previous)] == previous
    await backend.close()


@pytest.mark.parametrize(
    "error_type", [RuntimeError, OSError, TypeError, AssertionError, KeyError, IndexError, AttributeError]
)
async def test_parser_infrastructure_failures_remain_fatal(monkeypatch: Any, error_type: Any) -> None:
    backend, requests, records = await _backend(monkeypatch, ["sampled"])

    class FailedParser(_Parser):
        def extract_reasoning(self, text: str, request: Any = None) -> tuple[str, str]:
            raise error_type("infrastructure failed")

    backend._reasoning_parser = backend._reasoning_parser_no_thinking = FailedParser()
    with pytest.raises(error_type, match="infrastructure failed"):
        await _respond(backend, _params(), _conversation())
    assert len(requests) == 1 and not records
    await backend.close()


@pytest.mark.parametrize(
    "mutation",
    [
        {"request_id": "another#draw"},
        {"finished": False},
        {"prompt_token_ids": [999]},
        {"token_ids": [-1]},
        {"token_ids": [True]},
        {"token_ids": [1.5]},
        {"logprobs": []},
        {"logprobs": None},
        {"logprobs": [{65: types.SimpleNamespace(logprob=float("nan"), rank=None)}]},
        {"logprobs": [{65: types.SimpleNamespace(logprob=0.1, rank=None)}]},
        {"lora_request": None},
    ],
)
async def test_invalid_engine_evidence_never_becomes_a_turn(monkeypatch: Any, mutation: Any) -> None:
    backend, requests, records = await _backend(
        monkeypatch, [{"text": "A", "output_token_ids": [65], "mutations": mutation}]
    )
    conv = _conversation()
    with pytest.raises(ullm.tracing.TraceReconstructionError):
        await _respond(backend, _params(), conv)
    assert len(requests) == 1 and not records
    assert backend._completed_requests == {}
    assert conv.last_turn.last_response._served is None
    await backend.close()


async def test_context_is_causal_and_never_enters_sampling_or_token_measurements(monkeypatch: Any) -> None:
    backend, requests, records = await _backend(monkeypatch, ["one", "two"])
    observed = []

    def context(run_id: str, previous: str | None) -> dict[str, Any]:
        observed.append((run_id, previous))
        return {"feedback": {"after_request": previous, "score": len(observed)}}

    backend.request_context_provider = context
    params = _params()
    params.request_context = {"reset": {"epoch": 0}}
    conv = _conversation()
    for _ in range(3):
        count = await backend.token_count(params, conv)
        assert isinstance(count, ullm.backend.TokenCount)
        restored = pickle.loads(pickle.dumps(count))
        assert int(restored) == len(restored.prompt_token_ids)
    assert observed == []
    await _respond(backend, params, conv)
    conv.elements.extend(
        [
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="next")),
            ullm.conv.ResponseTurn(status="intermediate"),
        ]
    )
    conv.status = "intermediate"
    await _respond(backend, params, conv)
    assert observed == [("run", None), ("run", requests[0]["user"])]
    assert requests[0]["context"] == {"reset": {"epoch": 0}, "feedback": {"after_request": None, "score": 1}}
    assert records[1].request_payload["request_context"] == requests[1]["context"]
    assert not hasattr(requests[0]["params"], "request_context")
    assert "feedback" not in TOK.decode(requests[0]["prompt"])
    restored = pickle.loads(pickle.dumps(backend))
    assert restored.request_context_provider is None and restored._completed_requests == {}
    assert restored._reasoning_parser is None
    assert restored._reasoning_lock is not backend._reasoning_lock
    await backend.close()


async def test_context_provider_cannot_override_explicit_metadata(monkeypatch: Any) -> None:
    backend, requests, _ = await _backend(monkeypatch, [])

    def provider(run: str, previous: str | None) -> dict[str, Any]:
        return {"same": "provider"}

    backend.request_context_provider = provider
    params = _params()
    params.request_context = {"same": "explicit"}
    with pytest.raises(ValueError, match="duplicates"):
        await _respond(backend, params, _conversation())
    assert requests == []
    await backend.close()


async def test_injection_is_exact_prompt_evidence_and_never_sampled_credit(monkeypatch: Any) -> None:
    backend, requests, records = await _backend(monkeypatch, ["reason\n</think>\n\nanswer<|im_end|>"])
    conv = _conversation()
    await _respond(backend, _params("prefix "), conv)
    assert TOK.decode(requests[0]["prompt"]).endswith("prefix ")
    assert records[0].output_token_ids == TOK.encode("reason\n</think>\n\nanswer<|im_end|>")
    assert len(records[0].logprobs) == len(records[0].output_token_ids)
    assert records[0].input_token_reconstruction_status == "ok"
    assert records[0].extra["response_injection"] == "prefix "
    await backend.close()


async def test_remote_trace_capability_identifies_its_validation_path(monkeypatch: Any) -> None:
    backend, _, _ = await _backend(monkeypatch, [])
    capability = await backend.trace_capability()
    assert capability is not None
    assert capability.api_kind == "engine_proxy"
    assert capability.validation_method == "platypus.vllmx.engine_proxy_roundtrip.v1"
    await backend.close()
