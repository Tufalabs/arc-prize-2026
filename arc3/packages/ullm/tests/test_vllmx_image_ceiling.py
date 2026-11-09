"""An image the engine's processor would shrink is refused, not shrunk.

A processor resizes past its pixel ceiling instead of raising, so a host
whose render outgrows the ceiling sends a picture the model reads smaller
than it was drawn, with nothing in the response to say so. The check hands
the processor, built as vLLM builds the engine's, a blank image of each
rendered size and refuses where what comes back is smaller than rounding
alone would make it. Upscaling a small image is not a shrink and passes.

No engine: a real Qwen3-VL processor saved to disk with a word-level
tokenizer, at the Flash-Next checkpoint's patch and merge sizes. Which of
two statements holds below was measured first through vLLM's own processing
context on that checkpoint; these cases pin the check to those readings.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

import pytest

vllmx = pytest.importorskip("ullm.backends.vllmx")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

#: The checkpoint's floor and ceiling, in pixels; each side rounds to 32.
CHECKPOINT_FLOOR = 65536
CHECKPOINT_CEILING = 1048576
LOW_CEILING = 262144
HIGH_CEILING = 4194304


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A processor-only checkpoint directory ``AutoProcessor`` loads as it
    would a real one."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    specials = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
    vocab = {"[UNK]": 0, **{token: index + 1 for index, token in enumerate(specials)}}
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        additional_special_tokens=specials,
    )
    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, size={"shortest_edge": CHECKPOINT_FLOOR, "longest_edge": CHECKPOINT_CEILING}
    )
    path = tmp_path_factory.mktemp("checkpoint")
    Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor()
    ).save_pretrained(path)
    return path


def _backend(checkpoint: pathlib.Path, stated: dict[str, Any], engine_stated: dict[str, Any] | None = None) -> Any:
    """A VLLM backend whose engine is just a model path and its stated options."""
    backend = vllmx.VLLM.__new__(vllmx.VLLM)
    multimodal_config = types.SimpleNamespace(mm_processor_kwargs=engine_stated or {})
    backend.__dict__.update(
        engine=types.SimpleNamespace(
            model_config=types.SimpleNamespace(model=str(checkpoint), get_multimodal_config=lambda: multimodal_config)
        ),
        _processor_setup_done=False,
        _processor=None,
        _processed_image_sizes={},
        mm_processor_kwargs=stated,
    )
    return backend


def _image(side: int) -> Any:
    """Only the two sides are read, so a stand-in carries just those."""
    return types.SimpleNamespace(width=side, height=side)


def _size(shortest: int, longest: int) -> dict[str, int]:
    return {"shortest_edge": shortest, "longest_edge": longest}


def test_the_checkpoints_ceiling_holds_where_nothing_else_is_stated(checkpoint: pathlib.Path) -> None:
    backend = _backend(checkpoint, {})
    backend._refuse_images_the_processor_would_shrink([_image(1024)])
    with pytest.raises(ValueError, match="would resize to 1024x1024"):
        backend._refuse_images_the_processor_would_shrink([_image(1536)])


def test_images_within_the_ceiling_pass_and_a_small_one_upscales(checkpoint: pathlib.Path) -> None:
    backend = _backend(checkpoint, {}, engine_stated={"max_pixels": LOW_CEILING})
    # At the ceiling exactly, off the rounding grid under it, and far below
    # the floor, where the rule enlarges rather than shrinks.
    backend._refuse_images_the_processor_would_shrink([_image(512), _image(500), _image(28)])


def test_a_size_ceiling_is_read_whether_it_is_raised_or_lowered(checkpoint: pathlib.Path) -> None:
    """``size`` is a per-call option: stated above the checkpoint's ceiling it
    lets a larger render through, and below it it catches a shrink the
    checkpoint's own ceiling would not."""
    raised = _backend(checkpoint, {"size": _size(CHECKPOINT_FLOOR, HIGH_CEILING)})
    raised._refuse_images_the_processor_would_shrink([_image(1536)])

    lowered = _backend(checkpoint, {"size": _size(CHECKPOINT_FLOOR, LOW_CEILING)})
    with pytest.raises(ValueError, match="would resize to 512x512"):
        lowered._refuse_images_the_processor_would_shrink([_image(1024)])


def test_a_lone_max_pixels_holds_from_the_engine_or_the_request(checkpoint: pathlib.Path) -> None:
    """The ARC2 duck states its ceiling this way. The processor ignores a lone
    ``max_pixels`` per call; it holds because vLLM hands it to construction."""
    for engine_stated, stated in (({"max_pixels": LOW_CEILING}, {}), ({}, {"max_pixels": LOW_CEILING})):
        backend = _backend(checkpoint, stated, engine_stated=engine_stated)
        with pytest.raises(ValueError, match="would resize to 512x512"):
            backend._refuse_images_the_processor_would_shrink([_image(1024)])


def test_a_per_call_size_outranks_a_constructed_max_pixels_wherever_each_is_stated(
    checkpoint: pathlib.Path,
) -> None:
    """Not the engine-then-request order: ``size`` reaches the call and
    ``max_pixels`` the construction, and the call wins in both arrangements."""
    engine_lowers = _backend(checkpoint, {"size": _size(CHECKPOINT_FLOOR, HIGH_CEILING)}, {"max_pixels": LOW_CEILING})
    engine_lowers._refuse_images_the_processor_would_shrink([_image(1024)])

    request_raises = _backend(checkpoint, {"max_pixels": HIGH_CEILING}, {"size": _size(CHECKPOINT_FLOOR, LOW_CEILING)})
    with pytest.raises(ValueError, match="would resize to 512x512"):
        request_raises._refuse_images_the_processor_would_shrink([_image(1024)])


def test_a_processor_without_the_rule_is_left_alone(checkpoint: pathlib.Path) -> None:
    """The check refuses what it can prove: a processor whose module states no
    resize rule is not ours to judge, and a run against one is not stopped."""
    backend = _backend(checkpoint, {"max_pixels": 8})
    backend.__dict__.update(
        _processor_setup_done=True,
        _processor=types.SimpleNamespace(image_processor=types.SimpleNamespace(patch_size=14, merge_size=2)),
    )
    backend._refuse_images_the_processor_would_shrink([_image(560)])


def test_an_engine_supplied_processor_keeps_its_image_ceiling(checkpoint: pathlib.Path) -> None:
    processor = transformers.AutoProcessor.from_pretrained(str(checkpoint), max_pixels=LOW_CEILING)
    backend = _backend(checkpoint, {})
    backend.engine.get_processor = lambda: processor
    with pytest.raises(ValueError, match="would resize to 512x512"):
        backend._refuse_images_the_processor_would_shrink([_image(1024)])
    assert backend._processor is processor


def test_processor_construction_keeps_the_model_revision(
    checkpoint: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[dict[str, Any]] = []
    load = transformers.AutoProcessor.from_pretrained

    def record_load(*args: Any, **kwargs: Any) -> Any:
        loaded.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", record_load)
    backend = _backend(checkpoint, {"max_pixels": LOW_CEILING})
    backend.engine.model_config.revision = "pinned-revision"
    with pytest.raises(ValueError, match="would resize to 512x512"):
        backend._refuse_images_the_processor_would_shrink([_image(1024)])
    assert len(loaded) == 2
    assert all(options["revision"] == "pinned-revision" for options in loaded)
    assert loaded[-1]["max_pixels"] == LOW_CEILING
