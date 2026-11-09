"""Exact history refuses what it cannot witness on EVERY call, the first one
included (requirements U5): an image in the prompt or stripped past reasoning
raises before any history is looked up, so the promise never lapses into
silent fresh renders. Also: the served prompt ids are shed from a response's
raw completion once witnessed, so the conversation stays linear in its
history. No engine, no server."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import openai.types.chat
import pytest
import ullm.conv
import ullm.tracing
from ullm.backends import oai
from ullm.continuation import HistoryDriftError

vllmx = pytest.importorskip("ullm.backends.vllmx")


def _vllm(*, exact_history: bool) -> Any:
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    backend.__dict__.update(exact_history=exact_history, _reasoning_parser=None, _tool_parser=None)
    return backend


def _first_call() -> ullm.conv.Conversation:
    conversation = ullm.conv.Conversation()
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    conversation.elements.append(ullm.conv.ResponseTurn())
    return conversation


def test_vllmx_continues_images_so_a_first_call_with_one_just_renders_afresh() -> None:
    """The in-process engine continues images by ids: the first call (no
    witness yet) renders afresh under exact history without complaint."""
    assert _vllm(exact_history=True)._continuation_input(_first_call(), [], [_pixels(0)], [object()], {}, "all") is None


def test_the_delta_encoder_refuses_misaligned_images() -> None:
    """A delta whose placeholders do not match the images that arrived since
    the last answer would expand into a prompt the engine rejects or, worse,
    misattributes — refused before it is sent."""
    backend = _vllm(exact_history=False)
    backend.__dict__.update(_processor_setup_done=True, _processor=SimpleNamespace(image_token="<|image_pad|>"))
    encode = backend._delta_encoder([object(), object()])
    with pytest.raises(HistoryDriftError, match="placeholder"):
        encode("\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>r")


def test_vllmx_refuses_stripped_reasoning_on_the_first_call_under_exact_history() -> None:
    with pytest.raises(HistoryDriftError, match="send_back_reasoning"):
        _vllm(exact_history=True)._continuation_input(_first_call(), [], [], [], {}, "none")


def test_vllmx_renders_afresh_without_the_promise() -> None:
    assert _vllm(exact_history=False)._continuation_input(_first_call(), [], [], [], {}, "none") is None


def _oai(*, exact_history: bool) -> oai.OpenAIServer:
    server = oai.OpenAIServer(base_url="http://127.0.0.1:1/v1", api_key="unused", model="m", streaming=False)
    server.__dict__.update(exact_history=exact_history, _trace_hf_tokenizer_id="tok")  # as the probe leaves it
    return server


_IMAGE_MESSAGES = [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}
]


def test_oai_refuses_an_image_on_the_first_call_under_exact_history() -> None:
    with pytest.raises(HistoryDriftError, match="image"):
        _oai(exact_history=True)._continuation_input({"messages": _IMAGE_MESSAGES}, _first_call(), "all")


def test_oai_refuses_an_unprobed_server_under_exact_history() -> None:
    server = _oai(exact_history=True)
    server.__dict__.update(_trace_hf_tokenizer_id="")
    with pytest.raises(HistoryDriftError, match="probed vLLM server"):
        server._continuation_input({"messages": [{"role": "user", "content": "q"}]}, _first_call(), "all")


def test_continued_tool_call_answer_has_a_parsed_trace_view() -> None:
    """A /v1/completions answer is plain text; the parsed element — not the
    lifted wire adapter — is the structured view a continued step records."""
    completion = SimpleNamespace(
        model_dump=lambda: {
            "id": "c",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "text": "hm\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n1+1\n</parameter>\n</function>\n</tool_call><|im_end|>",
                    "finish_reason": "stop",
                    "token_ids": [1, 2],
                    "prompt_token_ids": [9],
                }
            ],
            "usage": None,
        }
    )
    lifted = oai._chat_completion_from_completion(completion, think_seeded=True, eos="<|im_end|>")
    element = ullm.conv.LLMResponse()
    _oai(exact_history=False)._parse_output(lifted, element)
    parsed = ullm.tracing.output_message_from_element(element)
    assert parsed is not None and parsed["tool_calls"] and parsed["tool_calls"][0]["function"]["name"] == "python"
    assert "<tool_call>" in (lifted.choices[0].message.content or "")  # the wire adapter keeps the raw markup


def test_served_prompt_ids_are_shed_from_raw_completions() -> None:
    """Every response's raw completion carries the whole prompt it was
    sampled after; once witnessed, that is shed so a conversation of n
    answers pickles linear in its history, not in the sum of its prompts."""
    conversation = ullm.conv.Conversation()
    turn = ullm.conv.ResponseTurn()
    conversation.elements.append(turn)
    backend = _vllm(exact_history=False)
    for k in range(1, 4):
        element = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="a"))
        element._model_contents = SimpleNamespace(
            prompt_token_ids=list(range(4000 * k)), outputs=[SimpleNamespace(token_ids=[1, 2, 3], text="a")]
        )
        turn.elements.append(element)
    before = len(pickle.dumps(conversation))
    for element in turn.elements:
        backend._shed_served_ids(element)
    after = len(pickle.dumps(conversation))
    assert all(e._model_contents.prompt_token_ids is None for e in turn.elements)
    assert all(e._model_contents.outputs[0].token_ids == [1, 2, 3] for e in turn.elements)
    assert after < before / 4

    chat = openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "prompt_token_ids": list(range(100)),
        }
    )
    element = ullm.conv.LLMResponse()
    element._model_contents = chat
    _oai(exact_history=False)._shed_served_ids(element)
    assert "prompt_token_ids" not in (chat.model_extra or {})


def _pixels(value: int) -> ullm.conv.ImageContent:
    """A small image, every pixel ``value``."""
    return ullm.conv.ImageContent(data=np.full((2, 2, 3), value, dtype=np.uint8))


def _witnessed_with(fingerprints: tuple[str, ...]) -> ullm.conv.Conversation:
    """A conversation whose last answer was witnessed after these served images."""
    from ullm.continuation import ServedTurn

    conversation = _first_call()
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="a"))
    response.status = "processed"
    response._served = ServedTurn(
        epoch=0,
        prefix_ids=(1, 2),
        output_ids=(3,),
        continued=False,
        head_len=1,
        answered_len=2,
        answered_sha="x",
        ended_with_eos=True,
        image_fingerprints=fingerprints,
    )
    conversation.last_turn.elements.append(response)
    return conversation


def test_a_served_image_swapped_for_same_size_pixels_is_refused_under_exact_history() -> None:
    """The render and the placeholder count cannot see the swap; the engine
    would get the new pixels under the old ids. The fingerprint can."""
    original, swapped = _pixels(0), _pixels(255)
    conversation = _witnessed_with((original.fingerprint(),))
    with pytest.raises(HistoryDriftError, match="already served has changed"):
        _vllm(exact_history=True)._continuation_input(conversation, [], [swapped], [object()], {}, "all")


def test_a_served_image_swapped_for_same_size_pixels_renders_afresh_without_the_promise() -> None:
    original, swapped = _pixels(0), _pixels(255)
    conversation = _witnessed_with((original.fingerprint(),))
    assert _vllm(exact_history=False)._continuation_input(conversation, [], [swapped], [object()], {}, "all") is None
