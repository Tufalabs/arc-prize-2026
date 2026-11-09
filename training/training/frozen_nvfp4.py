"""Frozen ModelOpt NVFP4 experts with activation and router-weight gradients.

The checkpoint stores two E2M1 values per byte, one E4M3 scale per 16
weights, and one FP32 global scale per projection. This module implements
the weight-only W4A16 interpretation used by the rollout Marlin backend.
Checkpoint ``input_scale`` entries are intentionally unused: activations
are BF16, not dynamically quantized to FP4.

Packed weights and scales are buffers, never optimizer parameters. A custom
linear autograd function re-dequantizes in backward for dX and saves only
the packed buffers, avoiding retention of every dense expert matrix until
backward. This is a correctness implementation using PyTorch GEMMs; it is
not an optimized quantized training kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dequantize_nvfp4(
    weight: Tensor,
    weight_scale: Tensor,
    weight_scale_2: Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Decode checkpoint-native row-major NVFP4; no Marlin repacking/swizzle."""
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    nibbles = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2)
    signs = torch.where((nibbles & 0x08) != 0, -1.0, 1.0)
    decoded = values[(nibbles & 0x07).long()] * signs
    blocks = decoded.reshape(weight.shape[0], -1, 16)
    combined_scale = weight_scale.float() * weight_scale_2.float()
    scaled = blocks * combined_scale.unsqueeze(-1)
    return scaled.flatten(-2).to(dtype=dtype)


class _FrozenNVFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, weight_scale, weight_scale_2, compute_dtype):
        ctx.input_dtype = inputs.dtype
        ctx.compute_dtype = compute_dtype
        ctx.save_for_backward(weight, weight_scale, weight_scale_2)
        dense_weight = dequantize_nvfp4(
            weight, weight_scale, weight_scale_2, dtype=compute_dtype
        )
        # Explicit arithmetic keeps the forward and dX reference consistent
        # even when the caller uses autocast for the rest of Flash.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.to(compute_dtype), dense_weight)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        weight, weight_scale, weight_scale_2 = ctx.saved_tensors
        dense_weight = dequantize_nvfp4(
            weight, weight_scale, weight_scale_2, dtype=ctx.compute_dtype
        )
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            grad_input = torch.matmul(grad_output.to(ctx.compute_dtype), dense_weight)
        # Deliberately no dW or scale gradients, even if called incorrectly
        # with a floating scale tensor that originally required gradients.
        return grad_input.to(ctx.input_dtype), None, None, None, None


class FrozenNVFP4Linear(nn.Module):
    """A fixed quantized projection with trainable upstream activations."""

    def __init__(
        self,
        weight: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.uint8 or weight.ndim != 2:
            raise ValueError("NVFP4 weight must be a rank-two uint8 packed tensor")
        if weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % 8:
            raise ValueError("NVFP4 projection input width must be divisible by 16")
        if weight_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("NVFP4 block scales must be E4M3 float8")
        if tuple(weight_scale.shape) != (weight.shape[0], weight.shape[1] // 8):
            raise ValueError("NVFP4 block scale shape does not match packed weights")
        if weight_scale_2.dtype != torch.float32 or weight_scale_2.numel() != 1:
            raise ValueError("NVFP4 global scale must be one FP32 value")
        if len({weight.device, weight_scale.device, weight_scale_2.device}) != 1:
            raise ValueError("NVFP4 buffers must be on the same device")
        if compute_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("NVFP4 compute dtype must be floating point")
        if not bool(torch.isfinite(weight_scale.float()).all()) or not bool(
            torch.isfinite(weight_scale_2).all()
        ):
            raise ValueError("NVFP4 scales must be finite")
        self.in_features = weight.shape[1] * 2
        self.out_features = weight.shape[0]
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", weight.detach().contiguous())
        self.register_buffer("weight_scale", weight_scale.detach().contiguous())
        self.register_buffer(
            "weight_scale_2", weight_scale_2.detach().reshape(()).contiguous()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError("input hidden dimension does not match frozen projection")
        return _FrozenNVFP4Linear.apply(
            inputs,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            self.compute_dtype,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, frozen_nvfp4=True"


class FrozenNVFP4Expert(nn.Module):
    """One frozen SiLU-gated expert, with checkpoint-compatible module names."""

    def __init__(
        self,
        gate_proj: FrozenNVFP4Linear,
        up_proj: FrozenNVFP4Linear,
        down_proj: FrozenNVFP4Linear,
    ) -> None:
        super().__init__()
        if (gate_proj.in_features, gate_proj.out_features) != (
            up_proj.in_features,
            up_proj.out_features,
        ) or (down_proj.in_features, down_proj.out_features) != (
            gate_proj.out_features,
            gate_proj.in_features,
        ):
            raise ValueError("frozen expert projection shapes are inconsistent")
        if (
            len(
                {
                    gate_proj.compute_dtype,
                    up_proj.compute_dtype,
                    down_proj.compute_dtype,
                }
            )
            != 1
        ):
            raise ValueError("frozen expert projections must share their compute dtype")
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class FrozenNVFP4Experts(nn.Module):
    """HF-compatible sparse expert collection; execute selected token/expert pairs.

    Children are named ``0``, ``1``, etc., matching the individual-expert
    ModelOpt checkpoint layout. A long token batch can touch every expert,
    but each token still executes only its supplied top-k selections.
    """

    def __init__(self, experts: Sequence[FrozenNVFP4Expert]) -> None:
        super().__init__()
        if not experts:
            raise ValueError("at least one frozen expert is required")
        self.num_experts = len(experts)
        self.hidden_dim = experts[0].gate_proj.in_features
        self.intermediate_dim = experts[0].gate_proj.out_features
        for index, expert in enumerate(experts):
            if (expert.gate_proj.in_features, expert.gate_proj.out_features) != (
                self.hidden_dim,
                self.intermediate_dim,
            ):
                raise ValueError("all frozen experts must have matching dimensions")
            self.add_module(str(index), expert)

    def __getitem__(self, index: int) -> FrozenNVFP4Expert:
        return self._modules[str(index)]

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError("expert input must be [tokens, hidden_size]")
        if (
            top_k_index.ndim != 2
            or top_k_index.shape != top_k_weights.shape
            or top_k_index.shape[0] != hidden_states.shape[0]
            or top_k_index.shape[1] == 0
            or top_k_index.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "routing IDs and weights must have matching [tokens, top_k] shapes"
            )
        if bool(((top_k_index < 0) | (top_k_index >= self.num_experts)).any()):
            raise ValueError("selected expert ID out of range")
        # No one-hot [experts, top_k, tokens] allocation. Selection itself is
        # discrete, while selected weights and expert inputs stay differentiable.
        # The accumulator is updated IN PLACE in fp32: an out-of-place
        # index_add would keep one full [tokens, hidden] copy per expert alive
        # for backward (512 copies of a 32k-token sequence is ~86 GB), while
        # the in-place add saves only the token index. fp32 accumulation also
        # matches the serving reduction more closely than a bf16 running sum.
        accumulator = torch.zeros(
            hidden_states.shape, dtype=torch.float32, device=hidden_states.device
        )
        for expert_id in top_k_index.unique(sorted=True).tolist():
            token_index, slot = torch.where(top_k_index == expert_id)
            expert_output = self[expert_id](hidden_states[token_index])
            weighted = expert_output * top_k_weights[token_index, slot, None]
            accumulator.index_add_(0, token_index, weighted.to(torch.float32))
        return accumulator.to(hidden_states.dtype)
