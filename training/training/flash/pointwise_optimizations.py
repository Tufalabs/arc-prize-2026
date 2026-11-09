"""Opt-in pointwise optimizations for ordinary (non-EP) Flash routing.

The expert path follows Transformers' Apache-2.0 grouped_mm_experts_forward,
without expert-parallel sentinel handling. Only install on this distillation
model's ordinary top-k router, which always emits real local expert indices.
"""

import torch
from transformers.integrations.moe import _grouped_linear
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts, Qwen4ExpTextRMSNorm

from .gated_pointwise import swiglu
from .norm_tail import grouped_norm_forward
from .expert_epilogue import weighted_permutation
from .expert_lora import add_lora_delta, lora_delta
from .moe_permutation import _combine_with_inverse, _gather_with_inverse, _inverse
from .options import EXPERT_OPTIMIZATIONS

OPTIMIZED_EXPERT_PATHS = tuple(name for name in EXPERT_OPTIMIZATIONS if name != "none")


def expert_forward_with_weights(
    self, hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj,
    *, optimization, grouped_linear=_grouped_linear,
):
    """Optimized expert forward; ``optimization="none"`` belongs to the native module."""
    if optimization not in OPTIMIZED_EXPERT_PATHS:
        raise ValueError(f"expert_forward_with_weights: optimization={optimization!r}; allowed values are {OPTIMIZED_EXPERT_PATHS}")
    count, hidden_dim = hidden_states.shape
    top_k = top_k_index.shape[-1]
    expert_ids, permutation = torch.sort(top_k_index.reshape(-1))
    if optimization == "fused_permutation":
        selected, inverse = _gather_with_inverse(hidden_states, permutation, top_k)
    else:
        selected = hidden_states[permutation // top_k]
    weights = (
        None if optimization == "fused_permutation"
        else top_k_weights.reshape(-1)[permutation]
    )
    histogram_input = (
        expert_ids.float()
        if hidden_states.device.type in ("cpu", "mps")
        else expert_ids.int()
    )
    counts = torch.histc(
        histogram_input, bins=self.num_experts, min=0, max=self.num_experts - 1
    )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    up = grouped_linear(selected, gate_up_proj, offsets)
    # Opt-in expert_lora_inplace_delta: each delta added in place, one at a time (expert_lora.add_lora_delta).
    inplace = getattr(self, "_expert_lora_inplace_delta", False)
    # Pre-existing expert-LoRA configurators also support native HF modules.
    if getattr(self, "_expert_lora", None) and inplace:
        half = up.shape[-1] // 2
        for name, columns in (("gate", (0, half)), ("up", (half, 2 * half))):
            up = add_lora_delta(up, self, name, selected, offsets, grouped_linear, columns)
    elif getattr(self, "_expert_lora", None):
        up = up + torch.cat(
            [
                lora_delta(self, name, selected, offsets, grouped_linear)
                for name in ("gate", "up")
            ],
            dim=-1,
        )
    activated = self._apply_gate(up)
    projected = grouped_linear(activated, down_proj, offsets)
    if getattr(self, "_expert_lora", None) and inplace:
        projected = add_lora_delta(projected, self, "down", activated, offsets, grouped_linear)
    elif getattr(self, "_expert_lora", None):
        projected = projected + lora_delta(
            self, "down", activated, offsets, grouped_linear
        )
    if optimization == "fused_permutation":
        return _combine_with_inverse(projected, top_k_weights, permutation, inverse).to(hidden_states.dtype)
    inverse = _inverse(permutation)
    if optimization == "weighted_permutation":
        weighted = weighted_permutation(projected, weights, permutation, inverse)
    else:
        weighted = (projected * weights.unsqueeze(-1))[inverse]
    return weighted.view(count, top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


class FrozenGroupedRMSNorm(Qwen4ExpTextRMSNorm):
    """Explicit frozen grouped-norm adapter used at model construction."""

    def forward(self, x):
        if self.weight.requires_grad:
            raise ValueError("Norm tail optimization requires frozen norm weights")
        return grouped_norm_forward(self, x)


class FlashExperts(Qwen4ExpTextExperts):
    """Explicit execution choices for the builder's ordinary grouped experts."""

    def __init__(self, config, *, optimization="none", fuse_gate=False):
        super().__init__(config)
        self.optimization = optimization
        self.grouped_linear = _grouped_linear
        self.fuse_gate = fuse_gate

    def _apply_gate(self, value):
        if self.fuse_gate:
            return swiglu(value)
        return super()._apply_gate(value)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.optimization == "none":
            return super().forward(hidden_states, top_k_index, top_k_weights)
        return expert_forward_with_weights(
            self, hidden_states, top_k_index, top_k_weights,
            self.gate_up_proj, self.down_proj,
            grouped_linear=self.grouped_linear, optimization=self.optimization,
        )
