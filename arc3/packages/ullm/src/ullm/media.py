"""Lossless image transport and deterministic Qwen image-processing evidence.

The payload contains pixels and hashes, never sampled text or processed tensors.
Heavy processing dependencies are loaded only when media is actually present.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from typing import Any


def _digest(value: Any) -> str:
    """Hash a JSON value with an unambiguous canonical representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def pixel_hash(image: Any) -> str:
    """Bind image mode and dimensions as well as every pixel byte."""
    header = json.dumps([image.mode, *image.size], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes() + image.convert("RGBA").tobytes()).hexdigest()


def encode_image(image: Any) -> dict[str, Any]:
    """Encode a lossless PNG and verify that its pixels round-trip exactly."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    value = {
        "png_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "pixel_sha256": pixel_hash(image),
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
    }
    decode_image(value)
    return value


def decode_image(value: Any) -> Any:
    """Reject corrupt bytes, metadata or same-size pixel substitutions."""
    import PIL.Image

    if not isinstance(value, dict) or set(value) != {"png_base64", "sha256", "pixel_sha256", "mode", "width", "height"}:
        raise ValueError("invalid lossless image envelope")
    raw = base64.b64decode(value["png_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("image PNG hash differs")
    image = PIL.Image.open(io.BytesIO(raw))
    image.load()
    if image.format != "PNG" or (image.mode, image.width, image.height) != (
        value["mode"],
        value["width"],
        value["height"],
    ):
        raise ValueError("image PNG geometry or mode differs")
    if pixel_hash(image) != value["pixel_sha256"]:
        raise ValueError("image pixels differ")
    return image


def _position_algorithm(model_architecture: str) -> str:
    """Refuse architectures whose image position calculation is not verified."""
    if model_architecture not in {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        # Flash-Next: the same text/image M-RoPE walk (temporal merge 1,
        # advance by max(h, w) // merge), checked against Transformers'
        # Qwen4ExpModel.get_rope_index in tests/test_media_positions.py.
        "Qwen4ExpForConditionalGeneration",
    }:
        raise ValueError(f"unsupported image position model architecture: {model_architecture}")
    return "qwen3-vl-image-mrope-v1"


def processing_identity(
    processor: Any,
    *,
    model: str,
    revision: str | None,
    mm_processor_kwargs: dict[str, Any],
    model_architecture: str,
) -> dict[str, Any]:
    """Bind processing to the serving engine's resolved model architecture."""
    algorithm = _position_algorithm(model_architecture)
    config = dict(processor.image_processor.to_dict())
    config.pop("_name_or_path", None)
    return {
        "model": model,
        "revision": revision,
        "model_architecture": model_architecture,
        "transformers": importlib.metadata.version("transformers"),
        "pillow": importlib.metadata.version("pillow"),
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision") if importlib.util.find_spec("torchvision") else None,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor_config_sha256": _digest(config),
        "mm_processor_kwargs": dict(mm_processor_kwargs),
        "image_token_id": int(processor.image_token_id),
        "spatial_merge_size": int(processor.image_processor.merge_size),
        "position_algorithm": algorithm,
    }


def tensor_evidence(tensor: Any) -> dict[str, Any]:
    """Hash exact contiguous CPU tensor bytes with their dtype and shape."""
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def image_layout(prompt_ids: list[int], grids: Any, processing: dict[str, Any]) -> list[dict[str, Any]]:
    """Match each ordered image to its already-expanded native placeholder run."""
    token = processing["image_token_id"]
    merge = processing["spatial_merge_size"]
    spans = []
    cursor = 0
    for grid in grids.tolist():
        if len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid) or grid[0] != 1:
            raise ValueError("ARC image grid must have one temporal frame and positive integer dimensions")
        if grid[1] % merge or grid[2] % merge:
            raise ValueError("image grid cannot be spatially merged")
        length = grid[0] * grid[1] * grid[2] // (merge * merge)
        try:
            offset = prompt_ids.index(token, cursor)
        except ValueError as error:
            raise ValueError("image has no native placeholder span") from error
        if prompt_ids[offset : offset + length] != [token] * length:
            raise ValueError("image native placeholder expansion differs from its grid")
        spans.append({"offset": offset, "length": length, "grid_thw": grid})
        cursor = offset + length
    if token in prompt_ids[cursor:]:
        raise ValueError("native image placeholders have no corresponding image")
    return spans


def position_inputs(length: int, layout: list[dict[str, Any]], processing: dict[str, Any]) -> dict[str, Any]:
    """Reproduce Qwen image M-RoPE positions directly from native token layout."""
    import torch

    if processing["position_algorithm"] != _position_algorithm(processing["model_architecture"]):
        raise ValueError("image position algorithm differs from model architecture")
    positions = torch.zeros((3, length), dtype=torch.int64)
    types = torch.zeros(length, dtype=torch.int64)
    cursor = position = 0
    merge = processing["spatial_merge_size"]
    for span in layout:
        offset, count = span["offset"], span["length"]
        if offset < cursor or offset + count > length:
            raise ValueError("image placeholder spans overlap or exceed native prompt")
        text_length = offset - cursor
        positions[:, cursor:offset] = torch.arange(text_length) + position
        position += text_length
        _, height, width = span["grid_thw"]
        grid = torch.meshgrid(
            torch.arange(1), torch.arange(height // merge), torch.arange(width // merge), indexing="ij"
        )
        positions[:, offset : offset + count] = torch.stack(grid).reshape(3, -1) + position
        types[offset : offset + count] = 1
        position += max(height, width) // merge
        cursor = offset + count
    positions[:, cursor:] = torch.arange(length - cursor) + position
    return {"position_ids": positions, "mm_token_type_ids": types}


def process_images(processor: Any, images: list[Any], processing: dict[str, Any]) -> dict[str, Any]:
    """Run only pinned image processing, leaving all sampled token IDs untouched."""
    kwargs = dict(processing["mm_processor_kwargs"])
    kwargs.pop("truncation", None)
    kwargs.pop("max_length", None)
    kwargs.pop("return_tensors", None)
    result = processor.image_processor(images=images, return_tensors="pt", **kwargs)
    if set(result) != {"pixel_values", "image_grid_thw"}:
        raise ValueError(f"unsupported image processor fields: {sorted(result)}")
    return dict(result)


def media_evidence(
    prompt_ids: list[int], images: list[dict[str, Any]], processing: dict[str, Any], tensors: dict[str, Any]
) -> dict[str, Any]:
    """Bind native layout and position inputs to the actual image tensors."""
    layout = image_layout(prompt_ids, tensors["image_grid_thw"], processing)
    if len(layout) != len(images):
        raise ValueError("processed image count differs from ordered image bytes")
    positions = position_inputs(len(prompt_ids), layout, processing)
    return {
        "processing": processing,
        "image_hashes": [item["pixel_sha256"] for item in images],
        "prompt_length": len(prompt_ids),
        "layout": layout,
        "tensors": {name: tensor_evidence(value) for name, value in {**tensors, **positions}.items()},
    }


def build_media(prompt_ids: list[int], images: list[Any], processor: Any, processing: dict[str, Any]) -> dict[str, Any]:
    """Build a complete media request with deterministic processing evidence."""
    encoded = [encode_image(image) for image in images]
    tensors = process_images(processor, images, processing)
    evidence = media_evidence(prompt_ids, encoded, processing, tensors)
    return {"version": 1, "images": encoded, **evidence}


def validate_media(media: Any, prompt_ids: list[int] | None = None) -> None:
    """Validate a request structure before costly processing or engine admission."""
    fields = {"version", "images", "processing", "image_hashes", "prompt_length", "layout", "tensors"}
    if not isinstance(media, dict) or set(media) != fields or media["version"] != 1:
        raise ValueError("invalid native media envelope")
    if not isinstance(media["images"], list) or not media["images"]:
        raise ValueError("native media requires at least one image")
    images = [decode_image(item) for item in media["images"]]
    if [pixel_hash(image) for image in images] != media["image_hashes"]:
        raise ValueError("ordered native image hashes differ")
    if prompt_ids is not None and media["prompt_length"] != len(prompt_ids):
        raise ValueError("media evidence belongs to another prompt length")


def reproduce_media(
    prompt_ids: list[int], media: dict[str, Any], processor: Any, processing: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reprocess image bytes and require every declared tensor and layout hash."""
    validate_media(media, prompt_ids)
    if media["processing"] != processing:
        raise ValueError("image processing identity differs")
    tensors = process_images(processor, [decode_image(item) for item in media["images"]], processing)
    observed = media_evidence(prompt_ids, media["images"], processing, tensors)
    expected = {name: media[name] for name in observed}
    if observed != expected:
        raise ValueError("processed image tensors, positions or native layout differ")
    return tensors, observed
