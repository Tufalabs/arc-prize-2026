"""Shared dataset helpers for the SFT pipeline.

The only input format is the TAAF raw trace; see
`sft/taaf_adapter.py` for the loader. This module
holds the format-agnostic pieces the adapter builds on:

- `_resolve_image`      : image content source (`path`/`url`/`base64`) -> PIL
- `_project_messages`   : inline image parts -> value-less placeholders + PIL list
- `records_to_dataset`  : TRL-ready record dicts -> `datasets.Dataset`
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_PART_TYPE = "image"
MEDIA_SOURCE_KEYS = ("path", "url", "base64")


def _resolve_image(part: dict[str, Any], *, base_dir: Path) -> Image.Image:
    """Resolve a native image content part to an RGB PIL image."""
    if "path" in part:
        path = Path(part["path"])
        if not path.is_absolute():
            path = base_dir / path
        with Image.open(path) as image:
            return image.convert("RGB")
    elif "url" in part:
        import urllib.request

        with urllib.request.urlopen(part["url"], timeout=30) as response:  # noqa: S310
            payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    elif "base64" in part:
        payload = base64.b64decode(part["base64"], validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    else:
        raise ValueError(
            f"image part must carry one of {MEDIA_SOURCE_KEYS}, got keys {sorted(part)}"
        )


def _project_messages(
    messages: list[dict[str, Any]], *, base_dir: Path, images_out: list[Image.Image]
) -> list[dict[str, Any]]:
    """Replace inline image sources with placeholders; append PIL images in order.

    Content is normalized to the list-of-parts form (a bare string becomes a
    single `{"type": "text", "text": ...}` part). This keeps the `content`
    column homogeneous so Arrow/`datasets` can build a consistent schema, and is
    the shape modern multimodal chat templates expect.
    """
    projected: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": "" if content is None else str(content)}]
        new_content: list[dict[str, Any]] = []
        for part in content:
            is_inline_image = (
                isinstance(part, dict)
                and part.get("type") == IMAGE_PART_TYPE
                and any(key in part for key in MEDIA_SOURCE_KEYS)
            )
            if is_inline_image:
                images_out.append(_resolve_image(part, base_dir=base_dir))
                new_content.append({"type": IMAGE_PART_TYPE})
            else:
                new_content.append(part)
        projected.append({**message, "content": new_content})
    return projected


def records_to_dataset(records: list[dict[str, Any]]):
    """Wrap TRL-ready records into a `datasets.Dataset`.

    Casts a PIL `images` column to the datasets Image feature when present.
    Imported lazily so callers can build records without `datasets` installed.
    """
    from datasets import Dataset, Image as HFImage, Sequence

    has_images = any(record.get("images") for record in records)
    dataset = Dataset.from_list(records)
    if has_images:
        features = dataset.features.copy()
        features["images"] = Sequence(HFImage())
        dataset = dataset.cast(features)
    return dataset
