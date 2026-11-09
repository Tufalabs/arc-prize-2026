"""Token-weighted accumulation with an explicit, resumable data microbatch cursor."""

import hashlib
import math
import re

import torch


def microbatch_count(cursor, accumulation, total=None):
    if (
        type(cursor) is not int
        or cursor < 0
        or type(accumulation) is not int
        or accumulation < 1
    ):
        raise ValueError("invalid accumulation/cursor")
    if total is None:
        return accumulation
    if type(total) is not int or total <= cursor:
        raise ValueError("data budget already consumed")
    return min(accumulation, total - cursor)


def remaining_updates(cursor, total, accumulation):
    microbatch_count(cursor, accumulation, total)
    return math.ceil((total - cursor) / accumulation)


# The routed experts' frozen tensors under the retired NVFP4 storage, and the
# BF16 execution weights the build now expands them to at load.
_NVFP4_EXPERT_TENSORS = frozenset(
    f"{projection}_{part}"
    for projection in ("gate", "up", "down")
    for part in ("packed", "scale", "outer")
)
_BF16_EXPERT_TENSORS = frozenset({"gate_up_proj", "down_proj"})
# Each retired tensor's per-expert source in the student checkpoint.
_NVFP4_SOURCE = {
    "packed": ".weight",
    "scale": ".weight_scale",
    "outer": ".weight_scale_2",
}


def nvfp4_expert_digests(
    reader, parents, num_experts, distribution="replicated", rank=0, world=1
):
    """This rank's frozen-fingerprint hashes of the packed experts of
    ``parents`` (``layers.<i>....mlp.experts``) under the retired NVFP4
    storage, recomputed from the student checkpoint ``reader`` reads.

    That storage stacked each projection's packed weights, block scales and
    outer scales over a layer's experts, and ``frozen_fingerprint`` hashed
    each stack's raw bytes on every rank: the whole stack (a buffer) under
    ``distribution="replicated"``, the rank's FSDP2 shard (the ``torch.chunk``
    of the expert dimension) under ``"fsdp"``. Hashing the experts'
    checkpoint tensors in order hashes the same bytes without building the
    stacks. Replicated hashes are the same on every rank, so with ``world >
    1`` (in the process group) the ranks split the layers and gather the
    hashes: each rank reads ``1/world`` of the packed experts either way.
    """
    experts = range(num_experts)
    if distribution == "fsdp":
        size = -(-num_experts // world)
        experts = experts[rank * size : (rank + 1) * size]
    elif distribution != "replicated":
        raise ValueError(f"unknown NVFP4 expert distribution {distribution!r}")
    elif world > 1:
        import torch.distributed as dist

        shares = [None] * world
        dist.all_gather_object(
            shares, nvfp4_expert_digests(reader, parents[rank::world], num_experts)
        )
        return {name: digest for share in shares for name, digest in share.items()}
    leaves = sorted(_NVFP4_EXPERT_TENSORS)
    digests = {}
    for parent in parents:
        match = re.fullmatch(r"layers\.(\d+)\.(?:.+\.)?mlp\.experts", parent)
        if match is None:
            raise ValueError(f"not a routed-expert module: {parent}")
        hashes = {leaf: hashlib.sha256() for leaf in leaves}
        for expert in experts:
            prefix = f"model.language_model.layers.{match[1]}.mlp.experts.{expert}."
            for leaf, digest in hashes.items():
                projection, part = leaf.split("_")
                tensor = reader.tensor(
                    f"{prefix}{projection}_proj{_NVFP4_SOURCE[part]}"
                )
                digest.update(
                    tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
                )
        digests |= {
            f"{parent}.{leaf}": digest.hexdigest() for leaf, digest in hashes.items()
        }
    return digests


def _split_experts(fingerprint, leaves):
    experts, others = {}, {}
    for name, digest in fingerprint.items():
        parent, _, leaf = name.rpartition(".")
        if parent.endswith("mlp.experts") and leaf in leaves:
            experts.setdefault(parent, {})[leaf] = digest
        else:
            others[name] = digest
    return experts, others


def validate_resume_fingerprint(manifest, saved, current, nvfp4_source):
    """Check this rank's frozen-tensor hashes against the checkpoint's.

    A checkpoint saved with NVFP4 expert storage hashed the packed experts,
    which the build now expands to BF16 at load. For such a checkpoint
    ``nvfp4_source(parents)`` hashes the student checkpoint's packed experts
    as that storage did on this rank (``nvfp4_expert_digests``); each must
    equal its saved hash, the BF16 experts must have the saved layout, and
    every other frozen tensor must match by hash. The next checkpoint records
    the BF16 fingerprint, which later resumes compare exactly.
    """
    if manifest.get("expert_storage", "bf16") != "nvfp4":
        if saved != current:
            raise ValueError("frozen student differs from the checkpoint's")
        return
    old_experts, old_others = _split_experts(saved, _NVFP4_EXPERT_TENSORS)
    new_experts, new_others = _split_experts(current, _BF16_EXPERT_TENSORS)
    if (
        old_others != new_others
        or not old_experts
        or set(old_experts) != set(new_experts)
        or any(set(leaves) != _NVFP4_EXPERT_TENSORS for leaves in old_experts.values())
        or any(set(leaves) != _BF16_EXPERT_TENSORS for leaves in new_experts.values())
    ):
        raise ValueError("frozen student differs from the checkpoint's")
    source = nvfp4_source(sorted(old_experts))
    changed = [
        f"{parent}.{leaf}"
        for parent, leaves in sorted(old_experts.items())
        for leaf, digest in sorted(leaves.items())
        if source.get(f"{parent}.{leaf}") != digest
    ]
    if changed:
        raise ValueError(
            "frozen student differs from the checkpoint's: the student "
            f"checkpoint's packed experts no longer hash as saved ({changed[0]}"
            + (f" and {len(changed) - 1} more)" if len(changed) > 1 else ")")
        )


def checkpoint_cursor(saved):
    if "data_microbatch_cursor" in saved:
        value = saved["data_microbatch_cursor"]
    elif saved["manifest"].get("gradient_accumulation", 1) == 1:
        value = saved["step"]
    else:
        raise ValueError("accumulated checkpoint lacks data cursor")
    if type(value) is not int or value < saved["step"]:
        raise ValueError("invalid checkpoint data cursor")
    return value


def validate_resume_manifest(saved, current, allow_accumulation_change=False):
    from .config import drop_retired_placement

    # A checkpoint saved with NVFP4 expert storage resumes with the BF16
    # experts, which compute bitwise the same; the replicated layout cannot.
    old = drop_retired_placement(saved, "the checkpoint manifest")
    new = dict(current)
    from .batching import normalize_batching

    # Missing checkpoint metadata predates length bucketing and means random.
    old["batching"] = normalize_batching(old.get("batching") or {"policy": "random"})
    new["batching"] = normalize_batching(new.get("batching") or {"policy": "random"})
    for manifest in (old, new):
        packing = manifest.pop("sequence_packing", None)
        if isinstance(packing, dict) and packing.get("enabled"):
            raise ValueError("cannot resume sequence-packed training")
    old.pop("gradient_noise", None)  # obsolete diagnostics, never training state
    old.setdefault("qsa_selector_policy", "guarded_fp32_top513_v1")
    new.setdefault("qsa_selector_policy", "guarded_fp32_top513_v1")
    old.setdefault("expert_optimization", "none")
    new.setdefault("expert_optimization", "none")
    from training.flash.expert_lora import normalize_lora

    old["expert_lora"] = normalize_lora(old.get("expert_lora"))
    new["expert_lora"] = normalize_lora(new.get("expert_lora"))
    old.setdefault("train_routers", True)
    new.setdefault("train_routers", True)
    old.setdefault("fuse_norm_tail", False)
    new.setdefault("fuse_norm_tail", False)
    old.setdefault("retain_qsa_attention", False)
    new.setdefault("retain_qsa_attention", False)
    old.setdefault("reuse_qsa_selection", False)
    new.setdefault("reuse_qsa_selection", False)
    old.setdefault("qsa_kernel", "flex")
    new.setdefault("qsa_kernel", "flex")
    old.setdefault("attention_backend", "sdpa")
    new.setdefault("attention_backend", "sdpa")
    old.setdefault("loss_weighting", "token")
    new.setdefault("loss_weighting", "token")
    from training.flash.kernel_mode import KERNEL_FIELDS, PRESETS

    # Manifests from before kernel_mode ran the reference stack; the fields
    # recorded before it keep their own historical defaults above.
    for manifest in (old, new):
        manifest.setdefault("kernel_mode", "reference")
        for name in KERNEL_FIELDS:
            if name != "selector_policy":
                manifest.setdefault(name, PRESETS["reference"][name])
    old.setdefault("gradient_accumulation", 1)
    new.setdefault("gradient_accumulation", 1)
    changed = old["gradient_accumulation"] != new["gradient_accumulation"]
    if changed and not allow_accumulation_change:
        raise ValueError("accumulation change requires explicit authorization")
    if allow_accumulation_change:
        old["gradient_accumulation"] = new["gradient_accumulation"]
    if old != new:
        raise ValueError("resume training manifest mismatch")
    return changed


class RouterGradientAccumulator:
    """Keep each microbatch's gradient available for noise moments, then sum it."""

    def __init__(self):
        self.buffers = None

    @torch.no_grad()
    def add(self, routers):
        for name, p in routers.items():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("Missing/nonfinite router gradient: " + name)
        if self.buffers is None:
            # Own the existing gradient storage, no extra clone for GA=1.
            self.buffers = {name: p.grad.detach() for name, p in routers.items()}
        else:
            if list(self.buffers) != list(routers):
                raise ValueError("router gradient census changed")
            for name, p in routers.items():
                self.buffers[name].add_(p.grad)
        for p in routers.values():
            p.grad = None

    def assign(self, routers):
        if self.buffers is None or list(self.buffers) != list(routers):
            raise ValueError("missing accumulated router gradients")
        for name, p in routers.items():
            p.grad = self.buffers[name]
        self.buffers = None
