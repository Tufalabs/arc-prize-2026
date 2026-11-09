"""An id prompt with images reaches the in-process engine in a shape it
processes: vLLM drops ``multi_modal_data`` beside a ``type: token`` input,
so the prompt goes untyped with each image's placeholder run collapsed, and
the prompt the engine served is held to the expanded ids. No engine."""

from __future__ import annotations

import types
from typing import Any

import pytest
import ullm.backend
import ullm.tracing
from ullm.continuation import HistoryDriftError

vllmx = pytest.importorskip("ullm.backends.vllmx")

PAD = 7


class _Tokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|image_pad|>"
        return PAD


class _Engine:
    """In process: no ``remote_engine``. Serves the collapsed prompt with its
    placeholders expanded back by ``expand``."""

    def __init__(self, expand: int) -> None:
        self.expand, self.prompts = expand, []

    def get_tokenizer(self) -> Any:
        return _Tokenizer()

    async def abort(self, request_id: str) -> None:
        pass

    async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        served = [
            t
            for token in prompt.get("prompt_token_ids", [])
            for t in ([PAD] * self.expand if token == PAD else [token])
        ]
        yield types.SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=served,
            lora_request=None,
            finished=True,
            outputs=[types.SimpleNamespace(text="ok", token_ids=[1, 2], finish_reason="stop", stop_reason=None)],
        )


def _backend(engine: Any) -> Any:
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    backend.__dict__.update(
        engine=engine,
        _processor_setup_done=True,
        _processor=types.SimpleNamespace(image_token="<|image_pad|>"),
        require_explicit_sampling=False,
        verify_engine_output=False,
        accept_malformed_output=True,
        lora_request=None,
        _request_counter=0,
        _completed_requests={},
    )
    return backend


EXPANDED = [1, 9, PAD, PAD, PAD, 8, 2, 9, PAD, PAD, PAD, 8, 3]
IMAGES = {"image": [object(), object()]}


def test_placeholder_runs_collapse_to_one_token_each() -> None:
    assert vllmx._collapse_image_placeholders(EXPANDED, PAD, 2) == [1, 9, PAD, 8, 2, 9, PAD, 8, 3]


def test_runs_that_do_not_match_the_images_are_refused() -> None:
    with pytest.raises(HistoryDriftError, match="placeholder"):
        vllmx._collapse_image_placeholders(EXPANDED, PAD, 3)


def test_an_id_prompt_with_images_goes_untyped_and_collapsed() -> None:
    backend = _backend(_Engine(expand=3))
    built = {
        "type": "token",
        "prompt_token_ids": EXPANDED,
        "multi_modal_data": IMAGES,
        "mm_processor_kwargs": {"max_pixels": 64},
    }
    prompt = backend._engine_prompt(built, remote=False)
    assert "type" not in prompt
    assert prompt["prompt_token_ids"] == [1, 9, PAD, 8, 2, 9, PAD, 8, 3]
    assert prompt["multi_modal_data"] is IMAGES and prompt["mm_processor_kwargs"] == {"max_pixels": 64}
    assert built["prompt_token_ids"] == EXPANDED  # the built prompt, witnessed and counted, keeps the expansions


@pytest.mark.parametrize(
    "built, remote",
    [
        ({"type": "token", "prompt_token_ids": [1, 2, 3]}, False),
        ({"prompt": "text <|image_pad|>", "multi_modal_data": IMAGES}, False),
        ({"type": "token", "prompt_token_ids": EXPANDED, "multi_modal_data": IMAGES}, True),
    ],
)
def test_other_prompts_reach_the_engine_as_built(built: dict[str, Any], remote: bool) -> None:
    assert _backend(_Engine(expand=3))._engine_prompt(built, remote=remote) is built


def _params() -> ullm.backend.GenericSamplingParameters:
    return ullm.backend.GenericSamplingParameters(temperature=1.0, top_p=1.0, max_tokens=8)


async def _draw(backend: Any) -> list[Any]:
    llm_input = {"type": "token", "prompt_token_ids": EXPANDED, "multi_modal_data": IMAGES}
    return [output async for output in backend._generate(_params(), {"engine": llm_input})]


async def test_the_engine_serves_the_expanded_ids_it_was_built_with() -> None:
    engine = _Engine(expand=3)
    outputs = await _draw(_backend(engine))
    assert outputs[0].prompt_token_ids == EXPANDED
    assert "type" not in engine.prompts[0]


async def test_a_different_expansion_is_fatal() -> None:
    with pytest.raises(ullm.tracing.TraceReconstructionError, match="expanded the image placeholders"):
        await _draw(_backend(_Engine(expand=2)))
