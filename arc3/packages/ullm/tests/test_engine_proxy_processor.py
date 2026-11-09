"""U5: remote processors reconstruct and verify advertised image configuration."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import PIL.Image
import pytest
import tokenizers
import torch
import transformers
import ullm.backends.engine_proxy
import ullm.media


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A processor-only checkpoint directory ``AutoProcessor`` loads as it
    would a real one."""
    specials = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
    vocab = {"[UNK]": 0, **{token: index + 1 for index, token in enumerate(specials)}}
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        additional_special_tokens=specials,
    )
    image_processor = transformers.Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 1048576}
    )
    path = tmp_path_factory.mktemp("checkpoint")
    transformers.Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=transformers.Qwen3VLVideoProcessor()
    ).save_pretrained(path)
    return path


@pytest.mark.parametrize(
    "stated",
    [{}, {"max_pixels": 262144}, {"size": {"shortest_edge": 65536, "longest_edge": 262144}}],
)
def test_remote_processor_reconstructs_the_advertised_identity(
    checkpoint: pathlib.Path, stated: dict[str, Any]
) -> None:
    """U5: remote rendering uses the advertised processor's construction options."""
    identity_args: dict[str, Any] = dict(
        model=str(checkpoint),
        revision="pinned-revision",
        mm_processor_kwargs=stated,
        model_architecture="Qwen3_5ForConditionalGeneration",
    )
    server_processor = transformers.AutoProcessor.from_pretrained(str(checkpoint), **stated)
    client = ullm.backends.engine_proxy.EngineProxy(base_url="http://unused", model="policy")
    client.metadata["processing"] = ullm.media.processing_identity(server_processor, **identity_args)
    processor = client.get_processor()
    assert ullm.media.processing_identity(processor, **identity_args) == client.metadata["processing"]
    assert client.get_processor() is processor


def test_remote_processor_still_rejects_a_different_configuration(checkpoint: pathlib.Path) -> None:
    """U5: matching overrides alone cannot authorize a different processor."""
    stated = {"max_pixels": 262144}
    server_processor = transformers.AutoProcessor.from_pretrained(str(checkpoint), **stated)
    server_processor.image_processor.merge_size = 4
    client = ullm.backends.engine_proxy.EngineProxy(base_url="http://unused", model="policy")
    client.metadata["processing"] = ullm.media.processing_identity(
        server_processor,
        model=str(checkpoint),
        revision=None,
        mm_processor_kwargs=stated,
        model_architecture="Qwen3_5ForConditionalGeneration",
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="local image processing differs from engine metadata"):
            client.get_processor()


@pytest.fixture(scope="module")
def image_request(checkpoint: pathlib.Path) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build native tokens and tensors with the actual advertised processor."""
    stated = {"min_pixels": 65536, "max_pixels": 262144}
    processor = transformers.AutoProcessor.from_pretrained(str(checkpoint), **stated)
    identity = ullm.media.processing_identity(
        processor,
        model=str(checkpoint),
        revision=None,
        mm_processor_kwargs=stated,
        model_architecture="Qwen3_5ForConditionalGeneration",
    )
    images = [PIL.Image.new("RGB", (384, 256), "red"), PIL.Image.new("RGB", (256, 512), "blue")]
    rendered = processor(
        text="before<|vision_start|><|image_pad|><|vision_end|>between<|vision_start|><|image_pad|><|vision_end|>after",
        images=images,
        return_tensors="pt",
        **stated,
    )
    prompt = {"prompt_token_ids": rendered["input_ids"][0].tolist(), "multi_modal_data": {"image": images}}
    return processor, identity, prompt, dict(rendered)


@pytest.mark.parametrize("overrides", [None, {}, {"max_pixels": 262144}, {"min_pixels": 65536, "max_pixels": 262144}])
def test_prepare_media_inherits_options_and_reproduces_native_tensors(
    image_request: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]], overrides: dict[str, Any] | None
) -> None:
    processor, identity, prompt, rendered = image_request
    client = ullm.backends.engine_proxy.EngineProxy(base_url="http://unused", model="policy")
    client.metadata["processing"] = identity
    prompt = dict(prompt)
    if overrides is not None:
        prompt["mm_processor_kwargs"] = overrides
    media = client.prepare_media(prompt)
    tensors, observed = ullm.media.reproduce_media(prompt["prompt_token_ids"], media, processor, identity)
    assert observed == {key: value for key, value in media.items() if key not in {"version", "images"}}
    assert len(media["layout"]) == 2
    for key, tensor in tensors.items():
        assert torch.equal(tensor, rendered[key])
    assert [ullm.media.decode_image(item).tobytes() for item in media["images"]] == [
        image.tobytes() for image in prompt["multi_modal_data"]["image"]
    ]


@pytest.mark.parametrize("overrides", [{"max_pixels": 65536}, {"do_resize": False}])
def test_prepare_media_rejects_effective_option_changes(
    image_request: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]], overrides: dict[str, Any]
) -> None:
    _, identity, prompt, _ = image_request
    client = ullm.backends.engine_proxy.EngineProxy(base_url="http://unused", model="policy")
    client.metadata["processing"] = identity
    with pytest.raises(ValueError, match="overrides differ from engine metadata"):
        client.prepare_media({**prompt, "mm_processor_kwargs": overrides})


@pytest.mark.parametrize(
    "tamper", ["pixels", "dimensions", "order", "layout", "pixel_values", "position_ids", "architecture", "algorithm"]
)
def test_reproduce_media_rejects_changed_images_layout_or_evidence(
    image_request: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]], tamper: str
) -> None:
    processor, identity, prompt, _ = image_request
    media = ullm.media.build_media(prompt["prompt_token_ids"], prompt["multi_modal_data"]["image"], processor, identity)
    media = copy.deepcopy(media)
    if tamper in {"pixels", "dimensions"}:
        replacement = prompt["multi_modal_data"]["image"][0].copy()
        if tamper == "pixels":
            replacement.putpixel((0, 0), (0, 255, 0))
        else:
            replacement = replacement.resize((256, 256))
        media["images"][0] = ullm.media.encode_image(replacement)
        media["image_hashes"][0] = ullm.media.pixel_hash(replacement)
    elif tamper == "order":
        media["images"].reverse()
        media["image_hashes"].reverse()
    elif tamper == "layout":
        media["layout"][0]["offset"] += 1
    elif tamper == "architecture":
        media["processing"]["model_architecture"] = "Qwen3_5MoeForConditionalGeneration"
    elif tamper == "algorithm":
        media["processing"]["position_algorithm"] = "flat"
    else:
        media["tensors"][tamper]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ullm.media.reproduce_media(prompt["prompt_token_ids"], media, processor, identity)


@pytest.mark.parametrize("architecture", ["Qwen3_5ForCausalLM", "Qwen3_8FlashForConditionalGeneration", "UnknownModel"])
def test_processing_identity_rejects_unsupported_position_architectures(
    checkpoint: pathlib.Path, architecture: str
) -> None:
    processor = transformers.AutoProcessor.from_pretrained(str(checkpoint))
    with pytest.raises(ValueError, match="unsupported image position model architecture"):
        ullm.media.processing_identity(
            processor, model=str(checkpoint), revision=None, mm_processor_kwargs={}, model_architecture=architecture
        )


def test_remote_processor_does_not_cache_an_unsupported_identity(
    image_request: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, identity, _, _ = image_request
    client = ullm.backends.engine_proxy.EngineProxy(base_url="http://unused", model="policy")
    client.metadata["processing"] = {**identity, "model_architecture": "UnknownModel"}
    for _ in range(2):
        with pytest.raises(ValueError, match="unsupported image position model architecture"):
            client.get_processor()
