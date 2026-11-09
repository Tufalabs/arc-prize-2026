"""Answer sampling (``GenericSamplingParameters.after_reasoning``): the
reasoning is sampled with the call's own settings, the answer — what the
model writes once its reasoning closes — with the answer's. The in-process
vLLM backend serves such a reply as two requests, the reasoning stopped at
its end token and the answer continuing the same ids, and hands back one
reply. Driven through ``respond`` over a fake engine that serves what vLLM
serves: a stop token's id kept, its text left out.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
import ullm.backend
import ullm.capture
import ullm.conv

vllmx = pytest.importorskip("ullm.backends.vllmx", reason="needs the vllm extra (linux)")

EOS = "<|im_end|>"
THINK, UNTHINK = "<think>", "</think>"


class _Tokenizer:
    """A Qwen-shaped template over a character tokenizer: special tokens are
    single ids, everything else one id per character. The generation prompt
    opens a think block, or closes it at once when thinking is off."""

    specials = {EOS: 1000, "<|im_start|>": 1001, THINK: 1002, UNTHINK: 1003}
    eos_token = EOS
    eos_token_id = 1000

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = True,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        out = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                body = f"{THINK}\n" + (message.get("reasoning_content") or "") + f"\n{UNTHINK}\n\n"
                out += "<|im_start|>assistant\n" + body + (message.get("content") or "") + EOS + "\n"
            else:
                out += "<|im_start|>" + role + "\n" + str(message.get("content")) + EOS + "\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + (f"{THINK}\n" if enable_thinking else f"{THINK}\n\n{UNTHINK}\n\n")
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

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        inverse = {v: k for k, v in self.specials.items()}
        return "".join(inverse.get(int(i), chr(int(i))) for i in ids)


TOK = _Tokenizer()
END_ID = TOK.specials[UNTHINK]


class _ThinkParser:
    """vLLM's qwen3 parser in the two shapes the backend reads: where the
    prompt leaves the reasoning, and the split of a reply's text."""

    reasoning_end_str = UNTHINK

    def is_reasoning_end(self, ids: list[int]) -> bool:
        for token in reversed(ids):
            if token == END_ID:
                return True
            if token == TOK.specials[THINK]:
                return False
        return False

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if UNTHINK not in text:
            return text, ""
        reasoning, content = text.split(UNTHINK, 1)
        return reasoning, content

    def count_reasoning_tokens(self, token_ids: list[int]) -> int:
        del token_ids
        return 0  # the positional fallback in _count_reasoning_tokens answers


class _NoThinkParser(_ThinkParser):
    """The same parser built for a request rendered without thinking: an
    output with no end token is all content."""

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        return (None, text) if UNTHINK not in text else super().extract_reasoning(text, request)


class _Engine:
    """Serves a reasoning, then an answer, the way vLLM serves them: a request
    stopping at the reasoning's end token keeps the token's id and leaves its
    text out. ``reasoning_closes`` False runs the reasoning to the cap."""

    def __init__(self, reasoning: str = "Add them.", answer: str = "42", *, reasoning_closes: bool = True) -> None:
        self.reasoning, self.answer, self.reasoning_closes = reasoning, answer, reasoning_closes
        self.requests: list[tuple[list[int], Any, str]] = []

    def get_tokenizer(self) -> _Tokenizer:
        return TOK

    async def abort(self, request_id: str) -> None:
        del request_id

    async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> AsyncIterator[Any]:
        del kwargs
        prompt_ids = list(prompt["prompt_token_ids"])
        self.requests.append((prompt_ids, params, request_id))
        stops = set(params.stop_token_ids or [])
        opened = not _ThinkParser().is_reasoning_end(prompt_ids)
        if opened and not self.reasoning_closes:
            ids, finish, stop = TOK.encode(self.reasoning)[: params.max_tokens], "length", None
        elif opened:
            ids = TOK.encode(self.reasoning) + [END_ID]
            finish, stop = "stop", END_ID if END_ID in stops else None
            if stop is None:  # no cut: the reply samples straight on into the answer
                ids += TOK.encode("\n\n" + self.answer) + [TOK.eos_token_id]
        else:
            ids, finish, stop = TOK.encode(self.answer) + [TOK.eos_token_id], "stop", None
        if params.max_tokens is not None and len(ids) > params.max_tokens:
            ids, finish, stop = ids[: params.max_tokens], "length", None
        text_ids = ids[:-1] if stop is not None else ids
        yield SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            lora_request=None,
            finished=True,
            # Everything but the last 5 prompt tokens came from the cache.
            num_cached_tokens=max(0, len(prompt_ids) - 5),
            outputs=[
                SimpleNamespace(
                    text=TOK.decode([t for t in text_ids if t != TOK.eos_token_id]),
                    token_ids=ids,
                    finish_reason=finish,
                    stop_reason=stop,
                    logprobs=None,
                    cumulative_logprob=None,
                )
            ],
        )


def _backend(engine: _Engine, **fields: Any) -> Any:
    """A ``VLLM`` over the fake engine with only the fields the path under test reads."""
    backend = vllmx.VLLM(engine=engine)
    backend.__dict__.update(
        {
            "chat_template_kwargs": {},
            "native_reasoning_effort": False,
            "reasoning_effort_prompts": None,
            "_tokenizer_setup_done": True,
            "_reasoning_parser": _ThinkParser(),
            "_reasoning_parser_no_thinking": _NoThinkParser(),
            "_tool_parser": None,
            **fields,
        }
    )
    return backend


def _params(answer: ullm.backend.AnswerSampling | None = None) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(
        temperature=0.6, top_p=1.0, max_tokens=100, extra_params={"top_k": 194}, after_reasoning=answer
    )


ANSWER = ullm.backend.AnswerSampling(temperature=0.3, extra_params={"top_k": 20})


def _conversation() -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="S")),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q"))],
    )
    conversation.conversation_id = "c"
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    conversation.status = "intermediate"
    return conversation


async def _respond(backend: Any, params: ullm.backend.GenericSamplingParameters) -> ullm.conv.LLMResponse:
    conversation = _conversation()
    async for _ in backend.respond(params, conversation, run_id="r"):
        pass
    return conversation.last_turn.last_response


# --- the split ------------------------------------------------------------


async def test_the_answer_is_sampled_apart_from_the_reasoning() -> None:
    """Two requests: the reasoning at the call's settings, stopped at its end
    token; then the answer at the answer's, sent the same ids plus the
    reasoning, with the cap the reasoning left, on the same prefix-cache
    thread. One reply comes back, as a single request would have given it."""
    engine = _Engine()
    response = await _respond(_backend(engine), _params(ANSWER))
    (prompt, first, first_id), (continued, second, second_id) = engine.requests
    reasoning_ids = TOK.encode("Add them.") + [END_ID]
    assert (first.temperature, first.top_p, first.top_k) == (0.6, 1.0, 194) and END_ID in first.stop_token_ids
    assert (second.temperature, second.top_p, second.top_k) == (0.3, 1.0, 20) and END_ID not in second.stop_token_ids
    assert continued == prompt + reasoning_ids
    assert second.max_tokens == 100 - len(reasoning_ids)
    assert second.extra_args == first.extra_args and first.extra_args["prefix_cache_context"] == "r/c"
    assert second_id == f"{first_id}:answer"
    assert response.reasoning_content.to_string() == "Add them."
    assert response.final_content.to_string() == "42"
    assert response.stop_reason == "stop"
    usage = response.token_usage
    assert usage is not None
    assert usage.prompt_tokens == len(prompt)
    assert usage.completion_tokens == len(reasoning_ids) + len(TOK.encode("42")) + 1
    # The cache hits the reasoning request had, less the 5 the answer request
    # had to prefill again: prompt minus cached is the prefill the reply cost.
    assert usage.cached_tokens == (len(prompt) - 5) - 5


async def test_a_prompt_that_closed_the_reasoning_is_one_request_at_the_answers_settings() -> None:
    """A template rendering no thinking closes the think block in the prompt:
    everything the model writes is answer."""
    engine = _Engine()
    response = await _respond(_backend(engine, chat_template_kwargs={"enable_thinking": False}), _params(ANSWER))
    ((_, params, _),) = engine.requests
    assert (params.temperature, params.top_k) == (0.3, 20) and END_ID not in params.stop_token_ids
    assert response.final_content.to_string() == "42"


async def test_without_answer_sampling_the_request_is_unchanged() -> None:
    engine = _Engine()
    response = await _respond(_backend(engine), _params())
    ((_, params, _),) = engine.requests
    assert (params.temperature, params.top_k, params.stop_token_ids) == (0.6, 194, [])
    assert (response.reasoning_content.to_string(), response.final_content.to_string()) == ("Add them.", "42")


async def test_settings_that_change_nothing_are_one_request() -> None:
    """The answer's settings equal to the call's sample the reply alike: no cut."""
    engine = _Engine()
    same = ullm.backend.AnswerSampling(temperature=0.6, extra_params={"top_k": 194})
    await _respond(_backend(engine), _params(same))
    ((_, params, _),) = engine.requests
    assert params.stop_token_ids == []


async def test_a_reasoning_the_cap_cuts_off_is_the_whole_reply() -> None:
    engine = _Engine(reasoning="x" * 200, reasoning_closes=False)
    response = await _respond(_backend(engine), _params(ANSWER))
    assert len(engine.requests) == 1
    assert response.stop_reason == "max_tokens"
    assert response.reasoning_content.to_string() == "x" * 100


async def test_a_reasoning_that_spends_the_whole_cap_is_a_capped_reply() -> None:
    """vLLM finishes a request on its stop token before it checks the cap, so
    a reasoning closing on the last token allowed comes back stopped. One
    request, for which the end token is no stop, would have been capped
    there: so is the reply, and no answer request is sent."""
    reasoning = "y" * 99  # plus the end token: exactly the cap of 100
    engine = _Engine(reasoning=reasoning)
    response = await _respond(_backend(engine), _params(ANSWER))
    assert len(engine.requests) == 1
    assert response.stop_reason == "max_tokens"
    assert response.reasoning_content.to_string() == reasoning
    assert response._model_contents.outputs[0].text == reasoning + UNTHINK


# --- refusals and records -------------------------------------------------


def test_answer_sampling_takes_only_per_token_shaping() -> None:
    """Bounds, seeds, bans and penalties that count the request's own output
    stay the call's: across two requests none of them would mean what it
    means for one reply."""
    for key in ("max_tokens", "stop", "stop_token_ids", "seed", "min_tokens", "logit_bias", "presence_penalty"):
        with pytest.raises(ValueError, match="after_reasoning cannot set"):
            ullm.backend.AnswerSampling(extra_params={key: 1})
    ullm.backend.AnswerSampling(extra_params={"top_k": 20, "min_p": 0.05})


async def test_a_backend_that_cannot_refuses_before_appending() -> None:
    conversation = _conversation()
    with pytest.raises(ValueError, match="does not support after_reasoning"):
        async for _ in ullm.backend.DummyBackend().respond(_params(ANSWER), conversation):
            pass
    assert conversation.last_turn.elements == []


async def test_a_traced_call_is_refused_before_appending() -> None:
    """A trace step records one sampling for the whole reply, so it cannot
    describe a split one yet."""

    def record(step: Any) -> None:
        del step

    engine = _Engine()
    backend = _backend(engine)
    backend.__dict__["trace_recorder"] = record
    conversation = _conversation()
    with pytest.raises(ValueError, match="not traced yet"):
        async for _ in backend.respond(_params(ANSWER), conversation, run_id="r"):
            pass
    assert conversation.last_turn.elements == [] and engine.requests == []


async def test_a_minimum_is_refused_where_the_reply_splits() -> None:
    """vLLM masks every stop token until a request's ``min_tokens``, the
    reasoning's end included, and the answer request would start the minimum
    again: two requests cannot hold it. A prompt that closed the reasoning is
    one request, where the minimum bounds the reply as it always does."""
    params = replace(_params(ANSWER), extra_params={"top_k": 194, "min_tokens": 8})
    engine = _Engine()
    conversation = _conversation()
    with pytest.raises(ValueError, match="min_tokens"):
        async for _ in _backend(engine).respond(params, conversation, run_id="r"):
            pass
    assert conversation.last_turn.elements == [] and engine.requests == []

    closed = _Engine()
    await _respond(_backend(closed, chat_template_kwargs={"enable_thinking": False}), params)
    ((_, served, _),) = closed.requests
    assert (served.min_tokens, served.temperature) == (8, 0.3)


async def test_a_call_stopping_at_the_reasoning_end_is_refused() -> None:
    """A call that stops at the reasoning's end asks for no answer, so
    sampling one apart contradicts it: refused, whether the stop names the end
    token or its text and whatever the prompt's shape, before anything joins
    the conversation."""
    stops = (
        replace(_params(ANSWER), extra_params={"top_k": 194, "stop_token_ids": [END_ID]}),
        replace(_params(ANSWER), stop_sequences=[UNTHINK]),
    )
    for params in stops:
        for fields in ({}, {"chat_template_kwargs": {"enable_thinking": False}}):
            engine = _Engine()
            conversation = _conversation()
            with pytest.raises(ValueError, match="stops at the reasoning's end"):
                async for _ in _backend(engine, **fields).respond(params, conversation, run_id="r"):
                    pass
            assert conversation.last_turn.elements == [] and engine.requests == []


def test_the_capture_record_carries_answer_sampling_only_when_set() -> None:
    """Replays compare the canonical request verbatim: a stream recorded before
    the field existed must still compare equal."""
    plain = ullm.capture._sampling_to_json(_params())
    assert "after_reasoning" not in plain
    split = ullm.capture._sampling_to_json(_params(ANSWER))
    assert split == {**plain, "after_reasoning": {"temperature": 0.3, "top_p": None, "extra_params": {"top_k": 20}}}
    assert ullm.capture._sampling_to_json(replace(_params(ANSWER), after_reasoning=None)) == plain
