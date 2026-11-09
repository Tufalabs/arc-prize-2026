"""U5: fresh local image draws use text; continuations preserve engine-served IDs."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
import ullm.conv
from test_vllmx_engine_contract import _params, _Parser, _respond, _Tokenizer

vllmx = pytest.importorskip("ullm.backends.vllmx")


class ImageTokenizer(_Tokenizer):
    """Render real HF image messages using the test model's three-token image grid."""

    specials = {**_Tokenizer.specials, "<|image_pad|>": 1004}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.specials[token]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        flattened = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(item["text"] if item["type"] == "text" else "<|image_pad|>" for item in content)
            flattened.append({**message, "content": content})
        return super().apply_chat_template(flattened, **kwargs)


class ImageProcessor:
    """Expand placeholders while recording which images the host processes."""

    image_token = "<|image_pad|>"

    def __init__(self, tokenizer: ImageTokenizer) -> None:
        self.tokenizer = tokenizer
        self.processed: list[int] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, *, text: list[str], images: list[Any], **kwargs: Any) -> dict[str, Any]:
        self.processed.append(len(images))
        assert text[0].count(self.image_token) == len(images)
        return {"input_ids": [self.tokenizer.encode(text[0].replace(self.image_token, self.image_token * 3))]}


class ImageEngine:
    """An in-process engine that expands text or collapsed-token image prompts."""

    def __init__(self) -> None:
        self.tokenizer = ImageTokenizer()
        self.model_config = types.SimpleNamespace(
            get_multimodal_config=lambda: types.SimpleNamespace(mm_processor_kwargs={})
        )
        self.prompts: list[Any] = []
        self.served: list[list[int]] = []

    def get_tokenizer(self) -> ImageTokenizer:
        return self.tokenizer

    async def abort(self, request_id: str) -> None:
        pytest.fail(f"completed image draw unexpectedly aborted: {request_id}")

    async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        ids = self.tokenizer.encode(prompt["prompt"]) if "prompt" in prompt else prompt["prompt_token_ids"]
        # A prompt whose media was prepared remotely carries its placeholders expanded already.
        expanded = (
            list(ids)
            if "media" in prompt
            else [token for item in ids for token in ([1004] * 3 if item == 1004 else [item])]
        )
        self.served.append(expanded)
        text = "reason\n</think>\n\nanswer"
        yield types.SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=expanded,
            lora_request=None,
            finished=True,
            num_cached_tokens=0,
            outputs=[types.SimpleNamespace(text=text, token_ids=self.tokenizer.encode(text), finish_reason="stop")],
        )


def image_backend() -> tuple[Any, ImageEngine, ImageProcessor]:
    engine = ImageEngine()
    backend = vllmx.VLLM(engine=engine, exact_history=True, accept_malformed_output=True)
    backend._tokenizer_setup_done = backend._processor_setup_done = True
    backend._reasoning_parser = backend._reasoning_parser_no_thinking = _Parser()
    processor = ImageProcessor(engine.tokenizer)
    backend._processor = processor
    return backend, engine, processor


def image_conversation() -> ullm.conv.Conversation:
    return ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys")),
        elements=[
            ullm.conv.UserPrompt(content=ullm.conv.ImageContent(data=np.zeros((32, 32, 3), dtype=np.uint8))),
            ullm.conv.ResponseTurn(status="intermediate"),
        ],
        status="intermediate",
    )


@pytest.mark.parametrize("injection", [None, "prefix "])
async def test_local_image_draws_continue_served_ids_and_render_fresh_after_rewrite(injection: str | None) -> None:
    backend, engine, processor = image_backend()
    conversation = image_conversation()
    params = _params(injection)
    try:
        first = await _respond(backend, params, conversation)
        assert ("prompt" in engine.prompts[0]) is (injection is None)
        assert processor.processed == ([] if injection is None else [1])
        assert first._served is not None
        assert engine.served[0].count(1004) == 3

        conversation.elements.extend(
            [
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="next")),
                ullm.conv.ResponseTurn(status="intermediate"),
            ]
        )
        conversation.status = "intermediate"
        await _respond(backend, params, conversation)
        served = engine.served[0] + list(first._served.output_ids)
        assert "prompt" not in engine.prompts[1]
        assert engine.served[1][: len(served)] == served

        conversation.elements[:] = image_conversation().elements
        conversation.mark_history_rewritten()
        conversation.status = "intermediate"
        await _respond(backend, params, conversation)
        assert ("prompt" in engine.prompts[2]) is (injection is None)
        assert engine.served[2] == engine.served[0]
    finally:
        await backend.close()


@pytest.mark.parametrize("rewritten", [False, True])
def test_remote_image_preparation_receives_expanded_tokens_even_for_fresh_history(rewritten: bool) -> None:
    backend, engine, processor = image_backend()
    prepared: list[Any] = []

    def prepare_media(prompt: Any) -> dict[str, str]:
        prepared.append(prompt)
        return {"schema": "prepared-media"}

    engine.prepare_media = prepare_media  # type: ignore[attr-defined]
    conversation = image_conversation()
    if rewritten:
        conversation.mark_history_rewritten()
    result = backend._prepare_input(_params(), conversation.last_turn, conversation, "all")
    assert processor.processed == [1]
    assert prepared[0]["prompt_token_ids"].count(1004) == 3
    assert "prompt" not in prepared[0]
    assert result["engine"] == {
        "type": "token",
        "prompt_token_ids": prepared[0]["prompt_token_ids"],
        "media": {"schema": "prepared-media"},
    }
