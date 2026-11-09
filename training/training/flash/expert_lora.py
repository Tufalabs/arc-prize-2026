"""Replicated FP32 LoRA masters for routed experts; frozen BF16 base GEMMs."""

import math
import torch
from torch import nn


def normalize_lora(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"rank", "alpha", "dropout"}:
        raise ValueError("expert_lora requires exactly rank, alpha, dropout")
    rank, alpha = value["rank"], value["alpha"]
    if type(rank) is not int or not 1 <= rank <= 128:
        raise ValueError("expert LoRA rank must be 1..128")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("expert LoRA alpha must be finite and positive")
    if value["dropout"] != 0:
        raise ValueError("Only deterministic zero-dropout expert LoRA is supported")
    return dict(rank=rank, alpha=float(alpha), dropout=0.0)


def install_expert_lora(module, config):
    config = normalize_lora(config)
    if config is None:
        raise ValueError("expert LoRA configuration is required")
    if hasattr(module, "_expert_lora"):
        raise ValueError("expert LoRA is already installed")
    if hasattr(module, "expert_shapes"):
        # Opt-in packed NVFP4 storage (nvfp4_experts.py): the BF16 shapes it expands to.
        (e, twice_i, h), _ = module.expert_shapes()
        device = module.gate_packed.device
    else:
        e, twice_i, h = module.gate_up_proj.shape
        device = module.gate_up_proj.device
        if twice_i % 2 or module.down_proj.shape != (e, h, twice_i // 2):
            raise ValueError("incompatible expert dimensions")
        if module.gate_up_proj.requires_grad or module.down_proj.requires_grad:
            raise ValueError("expert LoRA requires frozen base weights")
    rank = config["rank"]
    module._expert_lora = config
    for name, in_dim, out_dim in [
        ("gate", h, twice_i // 2),
        ("up", h, twice_i // 2),
        ("down", twice_i // 2, h),
    ]:
        # Separate RNG stream: installing adapters must not perturb base model RNG.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            a = torch.empty(e, rank, in_dim, device=device, dtype=torch.float32)
            nn.init.uniform_(a, -1 / math.sqrt(in_dim), 1 / math.sqrt(in_dim))
        b = torch.zeros(e, out_dim, rank, device=a.device, dtype=torch.float32)
        module.register_parameter("lora_" + name + "_A", nn.Parameter(a))
        module.register_parameter("lora_" + name + "_B", nn.Parameter(b))
    return {
        name: p
        for name, p in module.named_parameters(recurse=False)
        if name.startswith("lora_")
    }


def _unscaled_delta(module, name, inputs, offsets, grouped_linear):
    from .expert_lora_shard import adapter_weight

    # The parameters themselves, or (opt-in expert_lora_sharded) their gathered slices.
    a = adapter_weight(module, "lora_" + name + "_A").to(inputs.dtype)
    b = adapter_weight(module, "lora_" + name + "_B").to(inputs.dtype)
    # Pad rank to a multiple of16 for grouped BF16 GEMM alignment. Padding is
    # constant zero and does not add trainable directions or change alpha/r.
    pad = (-a.shape[-2]) % 16
    if pad:
        a = torch.nn.functional.pad(a, (0, 0, 0, pad))
        b = torch.nn.functional.pad(b, (0, pad))
    low = grouped_linear(inputs, a, offsets)
    return grouped_linear(low, b, offsets)


def lora_delta(module, name, inputs, offsets, grouped_linear):
    return _unscaled_delta(module, name, inputs, offsets, grouped_linear) * (
        module._expert_lora["alpha"] / module._expert_lora["rank"]
    )


class _AddIntoColumns(torch.autograd.Function):
    """``target[:, lo:hi] += delta`` in place; the gradient passes through uncopied.

    An in-place add into a view makes autograd rebase ``target`` onto CopySlices,
    whose backward copies the whole gradient of ``target`` and clones the slice;
    here ``target`` gets the incoming gradient itself and ``delta`` a view of it,
    exactly what ``target + cat(...)`` (AddBackward, CatBackward) hands both.
    """

    @staticmethod
    def forward(ctx, target, delta, lo, hi):
        target[:, lo:hi].add_(delta)
        ctx.mark_dirty(target)
        ctx.columns = (lo, hi)
        return target

    @staticmethod
    def backward(ctx, grad):
        lo, hi = ctx.columns
        return grad, grad[:, lo:hi], None, None


def add_lora_delta(target, module, name, inputs, offsets, grouped_linear, columns=None):
    """Opt-in expert_lora_inplace_delta: ``target += lora_delta(...)`` in place.

    The same products, scale and BF16 add as ``target + lora_delta(...)``, so the
    values and gradients are identical; only one delta-sized temporary is live
    instead of the delta, its scaled copy, a concatenation and the new sum. Safe
    under autograd because a grouped GEMM saves its inputs, not its output.
    ``columns=(lo, hi)`` adds into ``target[:, lo:hi]`` without a view's
    CopySlices backward (_AddIntoColumns); returns the updated ``target``.
    Ported from the distillation trainer (c381e82a, 625a5380).
    """
    delta = _unscaled_delta(module, name, inputs, offsets, grouped_linear)
    delta = delta.mul_(module._expert_lora["alpha"] / module._expert_lora["rank"])
    if columns is None:
        return target.add_(delta)
    return _AddIntoColumns.apply(target, delta, *columns)


def validate_expert_lora_inplace_delta(value, expert_lora):
    if type(value) is not bool:
        raise ValueError("expert_lora_inplace_delta must be boolean")
    if value and not expert_lora:
        raise ValueError("expert_lora_inplace_delta requires routed expert LoRA")


def enable_inplace_delta(model):
    """Switch every routed-expert adapter of ``model`` to add_lora_delta."""
    modules = [m for m in model.modules() if getattr(m, "_expert_lora", None)]
    if not modules:
        raise ValueError("expert_lora_inplace_delta found no routed expert LoRA")
    for m in modules:
        m._expert_lora_inplace_delta = True
    return len(modules)


def merge_projection(weight, a, b, config):
    """Merge one expert projection into its frozen execution weight for export."""
    config = normalize_lora(config)
    if (
        config is None
        or a.shape != (config["rank"], weight.shape[-1])
        or b.shape != (weight.shape[-2], config["rank"])
    ):
        raise ValueError("LoRA merge dimensions do not match configuration")
    result = (
        weight.float() + (b.float() @ a.float()) * (config["alpha"] / config["rank"])
    ).to(weight.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite merged expert weight")
    return result
