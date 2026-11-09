"""Reproduce witnessed image inputs for native policy forwards without retokenizing."""

from __future__ import annotations

from typing import Any


def checkpoint_source(training: Any) -> tuple[str, str | None, bool]:
    """Where the tokenizer, image processor and model config load from.

    Returns ``(path_or_id, revision, local_files_only)``. Flash trains from a
    local checkpoint directory (``training.flash.checkpoint_root``); its
    ``model_name`` is the served alias, not a Hub repository. Other modes keep
    the Hub id and pinned revision.
    """
    if getattr(training, "weight_mode", "") == "flash":
        return str(training.flash.checkpoint_root), None, True
    revision = str(getattr(training, "model_revision", "") or "").strip() or None
    return str(training.model_name), revision, False


def resolve_processing_identity(
    model_name: str, revision: str | None, *, local_files_only: bool = False
) -> dict[str, Any]:
    """Pin the image-processing stack a checkpoint's media inputs are bound to.

    ullm binds image processing to the serving engine's resolved model
    architecture, so the identity names the same single architecture.
    """
    import transformers
    from ullm.media import processing_identity

    processor = transformers.AutoProcessor.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    architectures = list(
        getattr(
            transformers.AutoConfig.from_pretrained(
                model_name,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=True,
            ),
            "architectures",
            None,
        )
        or []
    )
    if len(architectures) != 1:
        raise ValueError(
            "Multimodal training requires a model config naming exactly one "
            f"architecture; got {architectures}."
        )
    return processing_identity(
        processor,
        model=model_name,
        revision=revision,
        mm_processor_kwargs={},
        model_architecture=architectures[0],
    )


def _default_identity(config: Any) -> dict[str, Any]:
    """The identity of a harness that names no facts of its own."""
    return {
        "environment": config.inference.environment,
        "advantage_estimator": config.training.advantage_estimator,
        "games": sorted(config.inference.games or [config.inference.game]),
    }


def build_checkpoint_identity(
    config: Any, *, support: Any, manifest: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Combine the harness identity with the media identity its inputs need.

    The harness adapter names its facts (dataset, prompting); whether an
    image-processing stack is pinned follows from its declared media, not
    from the harness name. Rollout augmentation, when enabled, is recorded
    for every harness: a checkpoint resumes only under the augmentation it
    was trained with, or none.
    """
    identity = support.training_identity(config, manifest)
    augmentation = config.inference.augmentation.identity()
    if augmentation is not None:
        identity = dict(identity or _default_identity(config))
        identity["rollout_augmentation"] = augmentation
    media = (
        support.media_requirements(config.inference)
        if hasattr(support, "media_requirements")
        else support.MEDIA
    )
    if media.get("images"):
        if identity is None:
            identity = _default_identity(config)
        source, revision, local_files_only = checkpoint_source(config.training)
        identity["processing_stack"] = resolve_processing_identity(
            source, revision, **({"local_files_only": True} if local_files_only else {})
        )
    return identity


def load_processing_stack(identity: dict[str, Any], *, local_files_only: bool = True):
    """Load the serving revision and reject incompatible image implementations."""
    import transformers
    from ullm.media import processing_identity

    processor = transformers.AutoProcessor.from_pretrained(
        identity["model"],
        revision=identity["revision"],
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    actual = processing_identity(
        processor,
        model=identity["model"],
        revision=identity["revision"],
        mm_processor_kwargs=identity["mm_processor_kwargs"],
        model_architecture=identity["model_architecture"],
    )
    if actual != identity:
        raise ValueError("training image-processing stack differs from serving")
    return processor


def prepare_training_media(
    input_ids: list[list[int]],
    media: list[dict | None],
    evidence: list[dict | None],
    processor,
) -> dict[str, Any]:
    """Reproduce pixels and positions for a right-padded mixed image/text batch."""
    import torch
    from ullm.media import position_inputs, processing_identity, reproduce_media

    if len(input_ids) != len(media) or len(media) != len(evidence):
        raise ValueError("media batch and native token batch lengths differ")
    if not any(item is not None for item in media):
        if any(item is not None for item in evidence):
            raise ValueError("text sample carries unexpected media evidence")
        return {}
    longest = max(map(len, input_ids))
    positions = torch.zeros((3, len(input_ids), longest), dtype=torch.int64)
    token_types = torch.zeros((len(input_ids), longest), dtype=torch.int64)
    pixels, grids = [], []
    for index, (ids, item, witness) in enumerate(
        zip(input_ids, media, evidence, strict=True)
    ):
        if item is None:
            if witness is not None:
                raise ValueError("text sample carries unexpected media evidence")
            positions[:, index, : len(ids)] = torch.arange(len(ids))
            continue
        declared = item["processing"]
        actual = processing_identity(
            processor,
            model=declared["model"],
            revision=declared["revision"],
            mm_processor_kwargs=declared["mm_processor_kwargs"],
            model_architecture=declared["model_architecture"],
        )
        prompt_length = item["prompt_length"]
        if prompt_length > len(ids):
            raise ValueError("sample truncated the witnessed multimodal prompt")
        tensors, observed = reproduce_media(
            ids[:prompt_length], item, processor, actual
        )
        if observed != witness:
            raise ValueError("training media differs from consumed server evidence")
        full_positions = position_inputs(len(ids), observed["layout"], actual)
        positions[:, index, : len(ids)] = full_positions["position_ids"]
        token_types[index, : len(ids)] = full_positions["mm_token_type_ids"]
        pixels.append(tensors["pixel_values"])
        grids.append(tensors["image_grid_thw"])
    return {
        "pixel_values": torch.cat(pixels),
        "image_grid_thw": torch.cat(grids),
        "mm_token_type_ids": token_types,
        "position_ids": positions,
    }
