"""Response injection (``GenericSamplingParameters.response_injection``,
requirements U1's request surface and U5's nesting invariant): the text is
served as ids after the generation prompt, the answer starts with it, the
witness carries it so the next call continues by tokens, and every backend
that cannot serve it refuses the call before touching the conversation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import ullm.backend
import ullm.capture
import ullm.continuation as cont
import ullm.conv
import ullm.html_render
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
    """vLLM's qwen3 parser shape: no opening tag in the output (the prompt
    pre-filled it), ``</think>`` closes the reasoning span."""

    end_token_id = 1003

    def extract_reasoning(self, text: str, request: Any = None) -> tuple[str | None, str | None]:
        del request
        if "</think>" not in text:
            return text, ""
        reasoning, content = text.split("</think>", 1)
        return reasoning, content

    def count_reasoning_tokens(self, token_ids: list[int]) -> int:
        del token_ids
        return 0  # the positional fallback in _count_reasoning_tokens answers


class _FakeEngine:
    """The serving tokenizer and model identity used by rendering and traces."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.model_config = SimpleNamespace(
            model="test-base", tokenizer="test-tokenizer", revision="r1", tokenizer_revision="r1"
        )

    def get_tokenizer(self) -> _Tokenizer:
        """The character tokenizer above."""
        return self._tokenizer


TOK = _Tokenizer()
PREV = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]


def _params(injection: str | None) -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=10, response_injection=injection)


def _conversation() -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="S")),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q"))],
    )
    return conversation


def _intermediate() -> ullm.conv.Conversation:
    """A conversation ready for ``respond``: the trailing turn is being generated into."""
    conversation = _conversation()
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    conversation.status = "intermediate"
    return conversation


def _vllm(**fields: Any) -> Any:
    """A ``VLLM`` over the fake engine with only the fields the path under test reads."""
    backend = vllmx.VLLM(engine=_FakeEngine(TOK))
    backend.__dict__.update(
        engine=_FakeEngine(TOK),
        chat_template_kwargs={},
        native_reasoning_effort=False,
        reasoning_effort_prompts=None,
        _tokenizer_setup_done=True,
        _reasoning_parser=_ThinkParser(),
        _reasoning_parser_no_thinking=None,
        _tool_parser=None,
        **fields,
    )
    return backend


def _render(messages: list[dict[str, Any]], *, generation: bool) -> str:
    return str(TOK.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False))


# --- serving --------------------------------------------------------------


def test_injection_is_served_as_its_own_ids_after_the_rendered_prompt() -> None:
    """The prompt is the template's ids followed by the injection's own
    encoding — never a re-tokenisation of text + text across the seam."""
    backend = _vllm()
    turn = ullm.conv.ResponseTurn()
    base = backend._prepare_input(_params(None), turn, _conversation(), "all")["engine"]["prompt_token_ids"]
    injected = backend._prepare_input(_params("Okay,"), turn, _conversation(), "all")["engine"]["prompt_token_ids"]
    assert TOK.decode(base).endswith("<think>\n")
    assert injected == [*base, *TOK.encode("Okay,")]
    # Empty means none: the prompt is untouched.
    assert backend._prepare_input(_params(""), turn, _conversation(), "all")["engine"]["prompt_token_ids"] == base


def test_the_answer_starts_with_the_injection_and_the_ids_stay_sampled() -> None:
    """The parsed response, the raw completion text every history re-render
    reads, and the element all carry the injection; usage and the reasoning
    count are over the sampled ids alone."""
    backend = _vllm()
    sampled = " so\n</think>\n\nanswer" + EOS
    sampled_ids = TOK.encode(sampled)
    output = SimpleNamespace(
        prompt_token_ids=TOK.encode(_render(PREV, generation=True) + "Okay,"),
        outputs=[SimpleNamespace(text=sampled, token_ids=list(sampled_ids), finish_reason="stop")],
        num_cached_tokens=0,
    )
    element = ullm.conv.LLMResponse()
    backend._parse_output(output, _conversation(), element, "Okay,")
    assert element.reasoning_content.to_string() == "Okay, so"
    assert element.final_content.to_string() == "answer"
    assert element._model_contents.outputs[0].text == "Okay," + sampled
    assert element._model_contents.outputs[0].token_ids == sampled_ids
    assert element.token_usage is not None
    assert element.token_usage.completion_tokens == len(sampled_ids)
    assert element.token_usage.reasoning_tokens == len(TOK.encode(" so\n"))


# --- continuation ---------------------------------------------------------


def _conversation_with(witness: cont.ServedTurn) -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation()
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="q")))
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="answer"))
    response.status = "processed"
    response._served = witness
    turn = ullm.conv.ResponseTurn()
    turn.elements.append(response)
    conversation.elements.append(turn)
    return conversation


def test_the_witness_matches_the_answer_with_its_injection() -> None:
    """A budget-stopped answer inside the think block tells the two delta
    readings apart: matched with the injection, the delta is the structure
    the answer never reached (close, terminator, framing); matched without
    it, the sampled text would not be found and the close would be lost."""
    head = _render(PREV, generation=True)
    sampled = " so"  # cut inside the think block, no terminator
    prompt_ids = TOK.encode(head) + TOK.encode("Okay,")
    answered_message = {"role": "assistant", "reasoning_content": "Okay, so", "content": ""}
    answered = _render(PREV + [answered_message], generation=False)
    witness = cont.witness(
        TOK,
        epoch=0,
        prompt_ids=prompt_ids,
        output_ids=TOK.encode(sampled),
        served_before=None,
        head_render=head,
        answered_render=answered,
        injected="Okay,",
    )
    assert witness is not None and witness.injected == "Okay," and witness.ended_with_eos is False
    history = cont.served_history(_conversation_with(witness))
    assert history is not None
    served, last = history
    assert served == prompt_ids + TOK.encode(sampled)
    new = _render(PREV + [answered_message, {"role": "user", "content": "r"}], generation=True)
    continued = cont.continue_prompt(TOK, served, last, new)
    assert continued is not None
    assert continued.prompt_ids == served + continued.delta_ids
    # The close the answer never reached, then the terminator and the framing
    # newline the render puts after every answer, then the new turn.
    assert TOK.decode(continued.delta_ids) == "\n</think>\n\n" + EOS + new[len(answered) - 1 :]


def test_leave_witness_hands_the_injection_to_the_witness() -> None:
    backend = _vllm(exact_history=True, send_back_reasoning="all")
    sampled = " so\n</think>\n\nanswer" + EOS
    conversation = _intermediate()
    turn = conversation.last_turn
    element = ullm.conv.LLMResponse()
    turn.elements.append(element)
    output = SimpleNamespace(
        prompt_token_ids=TOK.encode(_render(PREV, generation=True) + "Okay,"),
        outputs=[SimpleNamespace(text=sampled, token_ids=TOK.encode(sampled), finish_reason="stop")],
        num_cached_tokens=0,
    )
    backend._process_output(_params("Okay,"), output, conversation, turn, element)
    assert isinstance(element._served, cont.ServedTurn)
    assert element._served.injected == "Okay,"
    assert element._served.prefix_ids == tuple(output.prompt_token_ids)


# --- refusals -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_backend_that_cannot_inject_refuses_before_appending() -> None:
    """A backend that cannot serve the text refuses the call rather than
    dropping it, and refuses before the conversation is touched, so a run that
    asked for a register it will not get stops instead of scoring."""
    backend = ullm.backend.DummyBackend()
    conversation = _intermediate()
    with pytest.raises(ValueError, match="response_injection"):
        async for _ in backend.respond(_params("x"), conversation):
            pass
    assert conversation.last_turn.elements == []
    with pytest.raises(ValueError, match="response_injection"):
        await backend.token_count(_params("x"), conversation)


@pytest.mark.asyncio
async def test_a_traced_step_is_token_faithful_and_marked() -> None:
    """The roundtrip re-renders the template and appends the injection's
    ids, the output render is compared without the injection, and the
    step's ``extra`` names what was injected."""
    backend = _vllm(exact_history=False, send_back_reasoning="all", mm_processor_kwargs={})
    conversation = _intermediate()
    turn = conversation.last_turn
    llm_input = backend._prepare_input(_params("Okay,"), turn, conversation, "all")
    sampled = " so\n</think>\n\nanswer" + EOS
    output = SimpleNamespace(
        prompt_token_ids=list(llm_input["engine"]["prompt_token_ids"]),
        outputs=[SimpleNamespace(text=sampled, token_ids=TOK.encode(sampled), finish_reason="stop", logprobs=None)],
        num_cached_tokens=0,
    )
    element = ullm.conv.LLMResponse()
    turn.elements.append(element)
    backend._process_output(_params("Okay,"), output, conversation, turn, element)
    step = await backend._build_trace_step(
        run_id="g", sampling_params=_params("Okay,"), llm_input=llm_input, element=element, conversation=conversation
    )
    assert step.input_token_ids[-len(TOK.encode("Okay,")) :] == TOK.encode("Okay,")
    assert step.output_token_ids == TOK.encode(sampled)
    assert step.output_message is not None and step.output_message["reasoning_content"] == "Okay, so"
    assert step.output_token_reconstruction_status == "ok"
    assert step.extra["response_injection"] == "Okay,"
    assert step.extra["adapter"] == {"name": "test-base", "id": 0, "path": None, "server_instance_id": None}
    assert step.sampling_params["response_injection"] == "Okay,"


def test_output_render_view_drops_only_a_prefix_the_text_starts_with() -> None:
    without = ullm.tracing.output_message_without_injection
    assert without({"role": "assistant", "reasoning_content": "Okay, so", "content": "a"}, "Okay,") == {
        "role": "assistant",
        "reasoning_content": " so",
        "content": "a",
    }
    assert without({"role": "assistant", "content": "Okay, a"}, "Okay,") == {"role": "assistant", "content": " a"}
    parsed_away = {"role": "assistant", "content": "a"}
    assert without(parsed_away, "Okay,") == parsed_away
    assert without(parsed_away, None) is parsed_away


# --- capture and rendering ------------------------------------------------


def test_the_injection_is_part_of_the_request_identity() -> None:
    """Replay compares canonical requests: an injected request must not
    match a recording made without one."""
    blocks: dict[str, Any] = {}
    plain = ullm.capture.canonical_dumps(ullm.capture.canonicalize_request(_params(None), _intermediate(), blocks))
    injected = ullm.capture.canonical_dumps(ullm.capture.canonicalize_request(_params("x"), _intermediate(), blocks))
    assert plain != injected and '"response_injection":"x"' in injected


def test_html_names_the_injected_start_of_the_answer() -> None:
    conversation = _conversation()
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text="<b>Okay</b>, so"),
        final_content=ullm.conv.StringContent(text="answer"),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=3, completion_tokens=2),
    )
    response.response_injection = "<b>Okay</b>,"
    response.status = "processed"
    conversation.elements.append(ullm.conv.ResponseTurn(elements=[response]))
    blocks: dict[str, Any] = {}
    structure = ullm.capture.serialize_conversation(conversation, blocks)

    def text_of(ref: Any) -> str:
        return ref if isinstance(ref, str) else str(blocks[ref["$block"]])

    html = ullm.html_render.conversation_html(structure, text_of=text_of, image_src=str, image_label=None)
    assert '<div class="ullm-injected">' in html
    assert "<code>&lt;b&gt;Okay&lt;/b&gt;,</code>" in html
    response.response_injection = None
    plain = ullm.html_render.conversation_html(
        ullm.capture.serialize_conversation(conversation, blocks), text_of=text_of, image_src=str, image_label=None
    )
    assert "ullm-injected" not in plain
