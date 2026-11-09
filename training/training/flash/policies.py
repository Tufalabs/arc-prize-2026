"""Explicit construction policies for the shared Flash model stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from torch import nn

from .expert_lora import normalize_lora

LayerConfigurator = Callable[[nn.Module, int], dict[str, nn.Parameter]]
TrainablesValidator = Callable[[dict[str, nn.Parameter]], None]


@dataclass(frozen=True)
class TrainabilityPolicy:
    """Describe trainability without coupling the model to a trainer.

    ``configure_layer`` is the extension seam for adapters.  It runs after a
    layer is loaded and before activation-checkpoint/FSDP wrapping, and returns
    the parameters that the distribution policy must leave replicated.
    ``validate_trainables`` optionally checks the model-wide result, keyed by
    qualified parameter name; custom trainability must train something.
    ``train_routers`` adds the FP32 routers to custom trainability as their
    own ``routers`` group (mode ``routers`` trains them already). ``expert_lora``
    with custom trainability adds routed-expert adapters beside the custom ones.
    """

    mode: Literal["routers", "frozen", "custom", "experts"] = "routers"
    router_backend: Literal["fp32_reference", "stock"] = "fp32_reference"
    configure_layer: LayerConfigurator | None = None
    validate_trainables: TrainablesValidator | None = None
    training: bool = True
    expert_lora: dict | None = None
    train_routers: bool = False

    def __post_init__(self):
        if self.mode not in ("routers", "frozen", "custom", "experts"):
            raise ValueError("Unknown trainability mode")
        lora = normalize_lora(self.expert_lora)
        object.__setattr__(self, "expert_lora", lora)
        if self.mode == "experts" and lora is None:
            raise ValueError("expert-only training requires expert_lora")
        if lora is not None and self.mode not in ("routers", "experts", "custom"):
            raise ValueError("expert_lora requires routers, experts or custom trainability")
        if self.mode == "custom" and self.configure_layer is None:
            raise ValueError("custom trainability requires configure_layer")
        if self.mode != "custom" and (self.configure_layer is not None or self.validate_trainables is not None):
            raise ValueError("configure_layer and validate_trainables are only valid for custom trainability")
        if type(self.train_routers) is not bool:
            raise ValueError("train_routers must be boolean")
        if self.train_routers and self.mode != "custom":
            raise ValueError("train_routers extends custom trainability; mode 'routers' already trains routers")
        if self.routers_trainable and self.router_backend != "fp32_reference":
            raise ValueError("router training requires the FP32 reference backend")

    @property
    def routers_trainable(self):
        return self.mode == "routers" or self.train_routers

    @classmethod
    def router_only(cls):
        return cls(mode="routers", router_backend="fp32_reference", training=True)

    @classmethod
    def frozen_reference(cls):
        return cls(mode="frozen", router_backend="fp32_reference", training=False)


@dataclass(frozen=True)
class DistributionPolicy:
    """FSDP wrapping choices, independent of process-count recipes.

    A distributed build shards every frozen weight with FSDP, the routed
    experts as BF16 parameters (NVFP4 in the checkpoint, expanded at load).
    Opt-in ``expert_storage="nvfp4_sharded"`` keeps them packed instead (sharded
    the same way, expanded per forward; ``nvfp4_experts.py``), with the
    expansion ``nvfp4_materialization`` "eager" (PyTorch) or "fused" (Triton).
    """

    reshard_after_forward: bool | int = True
    forward_prefetch: int = 0
    backward_prefetch: int = 0
    expert_storage: Literal["bf16", "nvfp4_sharded"] = "bf16"
    nvfp4_materialization: Literal["eager", "fused"] = "eager"

    def __post_init__(self):
        if type(self.reshard_after_forward) not in (bool, int) or (
            type(self.reshard_after_forward) is int and self.reshard_after_forward <= 0
        ):
            raise ValueError("reshard_after_forward must be boolean or positive shard count")
        if any(type(value) is not int or value < 0
               for value in (self.forward_prefetch, self.backward_prefetch)):
            raise ValueError("FSDP prefetch counts must be nonnegative integers")
        if self.expert_storage not in ("bf16", "nvfp4_sharded"):
            raise ValueError(f"expert_storage={self.expert_storage!r}; allowed values are 'bf16', 'nvfp4_sharded'")
        if self.nvfp4_materialization not in ("eager", "fused"):
            raise ValueError(
                f"nvfp4_materialization={self.nvfp4_materialization!r}; allowed values are 'eager', 'fused'"
            )
        if self.nvfp4_materialization != "eager" and self.expert_storage != "nvfp4_sharded":
            raise ValueError("nvfp4_materialization applies only to expert_storage='nvfp4_sharded'")
