"""Distillation recipe adapter for the shared Flash model stack.

Reusable loading, attention, expert, GDN, and pointwise code lives under
``training.flash``. This module preserves the historical API and keeps the
qualified trainable-set/frozen-teacher assertions at recipe level.
"""

from __future__ import annotations

import os
from pathlib import Path

from training.flash.model import (
    FlashModelBundle,
    HostEmbedding,
    assign,
    build_flash_model,
    frozen_fingerprint,
    router_forward,
)
from training.flash.reader import SOURCE, dequant_nvfp4
from training.flash.reader import Reader as _Reader
from training.flash.head_loss import ChunkedCE, completion_loss
from training.flash.policies import TrainabilityPolicy


class Reader(_Reader):
    """Apply distillation's optional scratch-stage contract to the core reader."""

    def __init__(self, root=SOURCE):
        root = Path(root)
        shard_root = None
        if os.environ.get("FLASH_STAGED_STUDENT") and str(
            root.resolve()
        ) == os.environ.get("FLASH_CANONICAL_STUDENT"):
            from sft.distillation.staging import validate_stage

            shard_root = Path(os.environ["FLASH_STAGED_STUDENT"])
            validate_stage(root, shard_root)
        super().__init__(root, shard_root=shard_root)


def build_model(
    root,
    distributed=True,
    device="cuda",
    teacher=False,
    offload_every=1,
    attention_backend="sdpa",
    qsa_kernel=None,
    reuse_qsa_selection=False,
    retain_qsa_attention=None,
    expert_optimization="weighted_permutation",
    fuse_norm_tail=True,
    expert_lora=None,
    train_routers=True,
    **kernel_options,
):
    """Build router-only or expert-LoRA distillation with explicit trainables.

    ``kernel_options`` are the remaining ``build_flash_model`` execution
    choices of a resolved kernel stack (training/flash/kernel_mode.py); the
    defaults above and the builder's own are the reference stack.
    """

    from training.flash.expert_lora import normalize_lora

    expert_lora = normalize_lora(expert_lora)
    if type(train_routers) is not bool:
        raise ValueError("train_routers must be boolean")
    if teacher and expert_lora:
        raise ValueError("teacher must not have expert LoRA")
    if not teacher and not train_routers and expert_lora is None:
        raise ValueError("training requires routers or expert LoRA")
    trainability = (
        TrainabilityPolicy.frozen_reference()
        if teacher
        else TrainabilityPolicy(
            mode="routers" if train_routers else "experts",
            expert_lora=expert_lora,
        )
    )
    bundle = build_flash_model(
        root,
        role="teacher" if teacher else "student",
        trainability=trainability,
        reader_factory=Reader,
        distributed=distributed,
        device=device,
        offload_every=offload_every,
        attention_backend=attention_backend,
        qsa_kernel=qsa_kernel,
        reuse_qsa_selection=reuse_qsa_selection,
        retain_qsa_attention=retain_qsa_attention,
        expert_optimization=expert_optimization,
        fuse_norm_tail=fuse_norm_tail,
        **kernel_options,
    )
    routers = bundle.parameter_groups["routers"]
    expected_router_parameters = (
        0
        if teacher or not train_routers
        else bundle.model.config.num_hidden_layers
        * bundle.model.config.num_experts
        * bundle.model.config.hidden_size
    )
    if (
        sum(parameter.numel() for parameter in routers.values())
        != expected_router_parameters
    ):
        raise AssertionError("distillation router parameter set changed")
    trainable = {
        id(parameter)
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    }
    adapters = bundle.parameter_groups.get("expert_lora", {})
    if expert_lora and len(adapters) != bundle.model.config.num_hidden_layers * 6:
        raise AssertionError("expert LoRA parameter census changed")
    # Preserve the historical checkpoint field names; the trainer mapping now
    # includes adapters as well as routers, in deterministic layer order.
    parameters = {}
    for i in range(bundle.model.config.num_hidden_layers):
        prefix = f"layers.{i}."
        parameters.update({k: v for k, v in routers.items() if k.startswith(prefix)})
        parameters.update({k: v for k, v in adapters.items() if k.startswith(prefix)})
    if trainable != {id(parameter) for parameter in parameters.values()}:
        raise AssertionError("distillation trainable parameter set changed")
    return bundle.model, bundle.head, parameters


__all__ = [
    "SOURCE",
    "ChunkedCE",
    "FlashModelBundle",
    "HostEmbedding",
    "Reader",
    "assign",
    "build_model",
    "completion_loss",
    "dequant_nvfp4",
    "frozen_fingerprint",
    "router_forward",
]
