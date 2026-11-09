"""Opt-in sharded NVFP4 storage for the routed experts (``expert_storage="nvfp4_sharded"``).

The default build expands the checkpoint's NVFP4 experts to BF16 parameters once at load. This storage keeps the
checkpoint's packed form instead: E2M1 nibbles (uint8), one E4M3 scale per 16 weights and one FP32 scale per
projection, as frozen parameters, so a distributed build's per-layer FSDP shards them across ranks like every other
frozen weight (about 4.5 bits per weight, so roughly 3.6x less expert memory per rank than sharded BF16). Each
forward, and each activation-checkpoint replay, expands the layer's gathered experts to the same BF16 values the
default storage holds (``reader.dequant_nvfp4``, or the fused Triton expansion, bitwise equal) and runs the
ordinary optimized expert path on them, so the arithmetic after the expansion is the default's.
"""

from __future__ import annotations

import torch
from torch import nn

from .pointwise_optimizations import OPTIMIZED_EXPERT_PATHS, expert_forward_with_weights
from .reader import dequant_nvfp4

PROJECTIONS = ("gate", "up", "down")
PARTS = ("packed", "scale", "outer")
MATERIALIZATIONS = ("eager", "fused")


def _stack(reader, names, suffix, device):
    values = [reader.tensor(name + suffix) for name in names]
    if not all(v.shape == values[0].shape and v.dtype == values[0].dtype for v in values):
        raise ValueError(f"inconsistent NVFP4 expert tensors for {suffix}")
    return torch.stack(values).to(device)


class NVFP4Experts(nn.Module):
    """Frozen packed routed experts; one layer expanded to BF16 per forward."""

    def __init__(self, tensors, *, optimization, fuse_gate=False, materialization="eager"):
        super().__init__()
        if optimization not in OPTIMIZED_EXPERT_PATHS:
            raise ValueError(
                f"NVFP4Experts: optimization={optimization!r}; allowed values are {OPTIMIZED_EXPERT_PATHS}"
            )
        if materialization not in MATERIALIZATIONS:
            raise ValueError(
                f"NVFP4Experts: materialization={materialization!r}; allowed values are {MATERIALIZATIONS}"
            )
        names = {f"{p}_{part}" for p in PROJECTIONS for part in PARTS}
        if set(tensors) != names:
            raise ValueError(f"NVFP4Experts needs exactly {sorted(names)}")
        if len({t.device for t in tensors.values()}) != 1:
            raise ValueError("NVFP4 expert tensors must share one device")
        experts = tensors["gate_packed"].shape[0]
        if not all(t.shape[0] == experts for t in tensors.values()):
            raise ValueError("NVFP4 expert count mismatch")
        gate, up, down = (tensors[p + "_packed"] for p in PROJECTIONS)
        if any(t.dtype != torch.uint8 for t in (gate, up, down)) or up.shape != gate.shape:
            raise ValueError("NVFP4 experts need uint8 gate/up/down weights of matching shapes")
        intermediate, hidden = gate.shape[1], gate.shape[2] * 2
        if down.shape != (experts, hidden, intermediate // 2):
            raise ValueError("incompatible NVFP4 expert dimensions")
        for name in sorted(names):
            # Frozen parameters, not buffers, so that FSDP shards them along the expert dimension.
            self.register_parameter(name, nn.Parameter(tensors[name], requires_grad=False))
        self.num_experts, self.hidden_dim, self.intermediate_dim = experts, hidden, intermediate
        self.optimization, self.fuse_gate, self.materialization = optimization, fuse_gate, materialization
        self.has_bias, self.has_gate, self.is_transposed = False, True, False
        from transformers.integrations.moe import _grouped_linear

        self.grouped_linear = _grouped_linear

    @classmethod
    def from_reader(cls, reader, layer_prefix, num_experts, device, **options):
        tensors = {}
        for projection in PROJECTIONS:
            names = [f"{layer_prefix}mlp.experts.{e}.{projection}_proj" for e in range(num_experts)]
            for part, suffix in zip(PARTS, (".weight", ".weight_scale", ".weight_scale_2")):
                tensors[f"{projection}_{part}"] = _stack(reader, names, suffix, device)
        return cls(tensors, **options)

    def expert_shapes(self):
        """(experts, 2 x intermediate, hidden) and (experts, hidden, intermediate), as the BF16 storage's."""
        e, i, h = self.num_experts, self.intermediate_dim, self.hidden_dim
        return (e, 2 * i, h), (e, h, i)

    def materialize(self):
        """The layer's BF16 execution weights: gate and up stacked as gate_up, and down."""
        if self.materialization == "fused":
            from .nvfp4_materialization import materialize_fused

            return materialize_fused(self)
        gate, up, down = (
            dequant_nvfp4(*(getattr(self, f"{p}_{part}") for part in PARTS)) for p in PROJECTIONS
        )
        return torch.cat((gate, up), dim=-2), down

    def _apply_gate(self, value):
        if self.fuse_gate:
            from .gated_pointwise import swiglu

            return swiglu(value)
        gate, up = value.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        gate_up, down = self.materialize()
        return expert_forward_with_weights(
            self, hidden_states, top_k_index, top_k_weights, gate_up, down,
            grouped_linear=self.grouped_linear, optimization=self.optimization,
        )
