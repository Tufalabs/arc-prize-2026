"""What an RL trainer needs from the in-process vLLM backend (U5, U7):
``accept_malformed_output`` keeps a degenerate or unparseable draw as the
model's turn instead of re-sampling it, ``require_explicit_sampling``
refuses a distribution the engine would default, a LoRA adapter rides
every generation and is recorded per trace step under the base identity,
and a non-finite sampled logprob fails the trace rather than a gradient."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.backend
import ullm.conv
import ullm.tracing

vllmx = pytest.importorskip("ullm.backends.vllmx", reason="needs the vllm extra (linux)")

EOS = "<|im_end|>"


class _Tokenizer:
    """A Qwen-shaped template over a character tokenizer: special tokens are
    single ids, everything else one id per character."""

    specials = {EOS: 1000, "<|im_start|>": 1001, "<think>": 1002, "</think>": 1003}
    eos_token = EOS
    eos_token_id = 1000

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool, tokenize: bool = True, **kwargs: Any
    ) -> Any:
        del kwargs
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
        return out if tokenize is False else self.encode(out)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
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
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse.get(int(i), chr(int(i))) for i in ids)


class _ThinkParser:
    """vLLM's qwen3 parser shape: ``</think>`` closes the reasoning span."""

    end_token_id = 1003

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if "</think>" not in text:
            return text, ""
        reasoning, content = text.split("</think>", 1)
        return reasoning, content

    def count_reasoning_tokens(self, token_ids: list[int]) -> int:
        del token_ids
        return 0


class _RefusingParser(_ThinkParser):
    """A parser that cannot shape the text at all."""

    def __init__(self, error: type[Exception]) -> None:
        self.error = error

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del text, request
        raise self.error("no think block")


class _FakeEngine:
    """What the render, parse and trace paths touch: the tokenizer and the
    two config handles the capability reads."""

    def __init__(self, tokenizer: _Tokenizer, *, lora: bool = False) -> None:
        self._tokenizer = tokenizer
        self.model_config = SimpleNamespace(
            model="base/model", tokenizer="base/model", revision="r1", tokenizer_revision="r1"
        )
        self.vllm_config = SimpleNamespace(lora_config=object() if lora else None)

    def get_tokenizer(self) -> _Tokenizer:
        return self._tokenizer


TOK = _Tokenizer()
MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]


def _backend(parser: Any, **fields: Any) -> Any:
    backend = vllmx.VLLM(engine=_FakeEngine(TOK), **fields)  # type: ignore[arg-type]
    backend._tokenizer_setup_done = True
    backend._reasoning_parser = parser
    backend._reasoning_parser_no_thinking = parser
    return backend


def _conversation() -> ullm.conv.Conversation:
    return ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="S")),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q"))],
    )


def _output(text: str, *, finish_reason: str = "stop", lora_request: Any = None) -> SimpleNamespace:
    """A RequestOutput as the engine echoes it: the rendered prompt ids and one completion."""
    return SimpleNamespace(
        prompt_token_ids=TOK.apply_chat_template(MESSAGES, add_generation_prompt=True),
        outputs=[SimpleNamespace(text=text, token_ids=TOK.encode(text), finish_reason=finish_reason, logprobs=None)],
        num_cached_tokens=0,
        lora_request=lora_request,
    )


def _params(*, max_tokens: int | None = 10) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(temperature=1.0, top_p=1.0, max_tokens=max_tokens)


# --- accept_malformed_output ------------------------------------------------


def test_an_unparseable_reply_is_rejected_unless_accepted() -> None:
    sampled = "no closing tag" + EOS
    with pytest.raises(ValueError, match="no think block"):
        _backend(_RefusingParser(ValueError))._parse_output(_output(sampled), _conversation(), ullm.conv.LLMResponse())
    element = ullm.conv.LLMResponse()
    _backend(_RefusingParser(ValueError), accept_malformed_output=True)._parse_output(
        _output(sampled), _conversation(), element
    )
    assert element.stop_reason == "malformed"
    assert element.malformed_error is not None and "no think block" in element.malformed_error
    # The model's turn, kept whole: what it sampled is what history re-serves.
    assert element.final_content.to_string() == "no closing tag"
    assert element.status == "processed"


def test_an_accepted_parse_failure_keeps_the_budget_stop() -> None:
    element = ullm.conv.LLMResponse()
    _backend(_RefusingParser(ValueError), accept_malformed_output=True)._parse_output(
        _output("cut", finish_reason="length"), _conversation(), element
    )
    assert element.stop_reason == "max_tokens"
    assert element.malformed_error is not None


def test_a_pipeline_fault_propagates_whatever_the_policy_says() -> None:
    with pytest.raises(RuntimeError, match="no think block"):
        _backend(_RefusingParser(RuntimeError), accept_malformed_output=True)._parse_output(
            _output("x"), _conversation(), ullm.conv.LLMResponse()
        )


def test_a_degenerate_tail_is_rejected_unless_accepted() -> None:
    sampled = "so\n</think>\n\n" + "!" * 12
    with pytest.raises(ullm.backend.DegenerateOutputError):
        _backend(_ThinkParser())._parse_output(_output(sampled), _conversation(), ullm.conv.LLMResponse())
    element = ullm.conv.LLMResponse()
    _backend(_ThinkParser(), accept_malformed_output=True)._parse_output(_output(sampled), _conversation(), element)
    assert element.stop_reason == "stop"
    assert element.malformed_error is None
    # Repetition remains an ordinary, unmarked model turn, with its parsed split.
    assert element.reasoning_content.to_string() == "so"
    assert element.final_content.to_string() == "!" * 12


def test_history_serves_text_the_parser_cannot_split_whole() -> None:
    whole = SimpleNamespace(outputs=[SimpleNamespace(text="raw reply")])
    assert (
        vllmx._vllm_assistant_final_text(whole, _RefusingParser(ValueError), accept_malformed_output=True)
        == "raw reply"
    )
    split = SimpleNamespace(outputs=[SimpleNamespace(text="r\n</think>\n\nfinal")])
    assert vllmx._vllm_assistant_final_text(split, _ThinkParser()) == "final"


# --- require_explicit_sampling ------------------------------------------------


def test_explicit_sampling_refuses_an_engine_default() -> None:
    backend = _backend(_ThinkParser(), require_explicit_sampling=True)
    turn = ullm.conv.ResponseTurn()
    with pytest.raises(ValueError, match="top_p"):
        backend._prepare_input(
            ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=10), turn, _conversation(), "all"
        )
    # Stated coordinates pass; no request cap is a statement, not an omission.
    backend._prepare_input(_params(max_tokens=None), turn, _conversation(), "all")
    # Off, the engine's defaults are accepted as before.
    _backend(_ThinkParser())._prepare_input(
        ullm.backend.GenericSamplingParameters(max_tokens=10), turn, _conversation(), "all"
    )


# --- LoRA: per-step adapter identity ------------------------------------------


async def test_trace_capability_names_the_base_under_lora() -> None:
    backend = vllmx.VLLM(engine=_FakeEngine(TOK, lora=True))  # type: ignore[arg-type]
    capability = await backend.trace_capability()
    assert capability is not None
    assert capability.model_id == "base/model" and capability.model_revision == "r1"


async def test_the_trace_step_records_the_adapter_the_engine_reports() -> None:
    backend = _backend(_ThinkParser())
    adapter = SimpleNamespace(lora_name="policy-w3", lora_int_id=3, lora_path="/adapters/w3")
    for lora_request, expected in (
        (adapter, {"name": "policy-w3", "id": 3, "path": "/adapters/w3", "server_instance_id": None}),
        (None, {"name": "base/model", "id": 0, "path": None, "server_instance_id": None}),
    ):
        output = _output("so\n</think>\n\nanswer" + EOS, lora_request=lora_request)
        conversation = _conversation()
        element = ullm.conv.LLMResponse()
        backend._parse_output(output, conversation, element)
        record = await backend._build_trace_step(
            run_id="run",
            sampling_params=_params(),
            llm_input={
                "engine": {"type": "token", "prompt_token_ids": output.prompt_token_ids},
                "trace_ctx": {"messages": MESSAGES, "images": [], "template_kwargs": {}, "response_injection": None},
            },
            element=element,
            conversation=conversation,
        )
        assert record.extra.get("adapter") == expected
        assert record.output_token_ids == output.outputs[0].token_ids


async def test_generation_runs_under_the_configured_adapter() -> None:
    import vllm.lora.request

    class _Engine(_FakeEngine):
        def __init__(self) -> None:
            super().__init__(TOK)
            self.calls: list[dict[str, Any]] = []

        async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> Any:
            del prompt, params, request_id
            self.calls.append(kwargs)
            yield _output("so\n</think>\n\nok" + EOS, lora_request=kwargs.get("lora_request"))

    engine = _Engine()
    backend = vllmx.VLLM(engine=engine)  # type: ignore[arg-type]
    prompt = {"engine": {"type": "token", "prompt_token_ids": [1]}}
    adapter = vllm.lora.request.LoRARequest("policy-w3", 3, "/adapters/w3")
    backend.lora_request = adapter
    async for _ in backend._generate(_params(), prompt):
        pass
    # Swapped between windows: the next draw runs under the base weights,
    # and an engine without the keyword never sees it.
    backend.lora_request = None
    async for _ in backend._generate(_params(), prompt):
        pass
    assert engine.calls == [{"lora_request": adapter}, {}]


# --- logprobs ------------------------------------------------------------------


def test_a_non_finite_sampled_logprob_fails_the_trace() -> None:
    broken = SimpleNamespace(logprobs=[{7: SimpleNamespace(logprob=math.nan, rank=1)}])
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="non-finite"):
        ullm.tracing.logprobs_from_vllm(broken, [7])
    sound = SimpleNamespace(logprobs=[{7: SimpleNamespace(logprob=-0.5, rank=1)}])
    assert ullm.tracing.logprobs_from_vllm(sound, [7]) == [
        {"token_id": 7, "logprob": -0.5, "rank": 1, "top": [[7, -0.5, 1]]}
    ]


@pytest.mark.parametrize(
    "name,value", [("temperature", 0.25), ("top_p", 0.5), ("max_tokens", 1), ("seed", 9), ("stop", ["x"])]
)
async def test_explicit_sampling_rejects_shadowed_fields_before_generation(name: str, value: Any) -> None:
    backend = _backend(_ThinkParser(), require_explicit_sampling=True)
    params = _params()
    params.extra_params[name] = value
    with pytest.raises(ValueError, match="overrides named fields"):
        async for _ in backend._generate(params, {"engine": {"prompt_token_ids": [1]}}):
            pytest.fail("a conflicting distribution reached generation")


@pytest.mark.parametrize("returned_adapter", [None, "other"])
async def test_adapter_mismatch_aborts_before_any_output_is_accepted(returned_adapter: str | None) -> None:
    import vllm.lora.request

    class _Engine(_FakeEngine):
        def __init__(self) -> None:
            super().__init__(TOK)
            self.aborted: list[str] = []

        async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> Any:
            del prompt, params, request_id, kwargs
            wrong = None if returned_adapter is None else vllm.lora.request.LoRARequest("other", 4, "/other")
            yield _output("ok", lora_request=wrong)

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    engine = _Engine()
    backend = vllmx.VLLM(engine=engine, lora_request=vllm.lora.request.LoRARequest("policy", 3, "/policy"))  # type: ignore[arg-type]
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="different adapter"):
        async for _ in backend._generate(_params(), {"engine": {"prompt_token_ids": [1]}}):
            pytest.fail("an unverified adapter's output reached the agent")
    assert len(engine.aborted) == 1


async def test_an_inflight_request_keeps_its_adapter_when_the_host_field_changes() -> None:
    import vllm.lora.request

    class _Engine(_FakeEngine):
        async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> Any:
            del prompt, params, request_id
            yield _output("first", lora_request=kwargs["lora_request"])
            yield _output("second", lora_request=kwargs["lora_request"])

    adapter = vllm.lora.request.LoRARequest("policy", 3, "/policy")
    backend = vllmx.VLLM(engine=_Engine(TOK), lora_request=adapter)  # type: ignore[arg-type]
    texts: list[str] = []
    async for output in backend._generate(_params(), {"engine": {"prompt_token_ids": [1]}}):
        texts.append(output.outputs[0].text)
        backend.lora_request = None
    assert texts == ["first", "second"]


async def test_accepting_backend_does_not_abort_a_repetitive_generation() -> None:
    class _Engine(_FakeEngine):
        async def generate(self, prompt: Any, params: Any, request_id: str) -> Any:
            del prompt, params, request_id
            yield _output("!" * 12, finish_reason="length")

        async def abort(self, request_id: str) -> None:
            pytest.fail(f"accepted malformed generation was aborted: {request_id}")

    backend = vllmx.VLLM(engine=_Engine(TOK), accept_malformed_output=True)  # type: ignore[arg-type]
    outputs = [output async for output in backend._generate(_params(), {"engine": {"prompt_token_ids": [1]}})]
    assert len(outputs) == 1 and outputs[0].outputs[0].token_ids == TOK.encode("!" * 12)


@pytest.mark.parametrize("error_type", [ValueError, TypeError, AssertionError, KeyError, IndexError, AttributeError])
def test_acceptance_never_swallows_failures_outside_the_parser(error_type: type[Exception]) -> None:
    class BrokenAccountingParser(_ThinkParser):
        def count_reasoning_tokens(self, token_ids: list[int]) -> int:
            raise error_type("accounting failed")

    backend = _backend(BrokenAccountingParser(), accept_malformed_output=True)
    with pytest.raises(error_type, match="accounting failed"):
        backend._parse_output(_output("thought</think>answer"), _conversation(), ullm.conv.LLMResponse())


def test_history_parsing_obeys_the_same_failure_policy() -> None:
    output = SimpleNamespace(outputs=[SimpleNamespace(text="raw reply")])
    with pytest.raises(ValueError, match="no think block"):
        vllmx._vllm_assistant_final_text(output, _RefusingParser(ValueError))
    with pytest.raises(KeyError, match="no think block"):
        vllmx._vllm_assistant_final_text(output, _RefusingParser(KeyError), accept_malformed_output=True)


async def test_inprocess_context_is_refused_before_the_engine_can_generate() -> None:
    # A real AsyncLLMEngine has no request_context keyword. It must never see one.
    backend = _backend(_ThinkParser())
    params = _params()
    params.request_context = {"score": 1}
    with pytest.raises(ValueError, match="request_context requires"):
        async for _ in backend._generate(params, {"engine": {"prompt_token_ids": [1]}}):
            pytest.fail("unsupported application context reached the engine")


@pytest.mark.parametrize("skip_special_tokens", [None, True])
async def test_verified_parsing_refuses_sampling_that_strips_parser_markers(skip_special_tokens: bool | None) -> None:
    backend = _backend(_ThinkParser(), verify_engine_output=True)
    params = _params()
    if skip_special_tokens is not None:
        params.extra_params["skip_special_tokens"] = skip_special_tokens
    with pytest.raises(ValueError, match="skip_special_tokens=False"):
        async for _ in backend._generate(params, {"engine": {"prompt_token_ids": [1]}}):
            pytest.fail("a draw without verified parser markers reached the engine")


def test_inprocess_pickle_refusal_explains_the_supported_transport() -> None:
    import pickle

    with pytest.raises(TypeError, match="serve the backend through ProxyBackend"):
        pickle.dumps(_backend(_ThinkParser()))
