"""QSA-specific adaptation of FLA's selected-attention online-softmax algorithm.

FLA source: fla/ops/nsa/parallel.py, fla-core 0.5.1 (MIT).
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li and contributors.
Changes: physical sparse reductions, padded GQA heads, exact dense-prefix/tail
policy, and deterministic reverse adjacency for dK/dV. Physical reduction kernels
live in physical_qsa_backward.py; masked holes and contraction order are preserved.
Opt-in indexed QSA backend; default execution remains FlexAttention.
Requires GPU qualification before production use.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
from torch.nn.attention.flex_attention import (
    _apply_kernel_options,
    _identity,
    flex_attention_hop,
)

from .physical_qsa_backward import physical_dkv, physical_dq


@torch.compile(dynamic=True)
def _flex_delta(out, grad):
    # Flex fixes the head dimension before lowering this FP32 reduction. Keep
    # the same fixed reduction tree when this helper is compiled independently.
    torch._dynamo.mark_static(out, -1)
    torch._dynamo.mark_static(grad, -1)
    return (out * grad).sum(dim=-1).float().contiguous()


def _validate_sparse_inputs(q, k, v, indices, *, dtypes=(torch.bfloat16,), op="Sparse QSA"):
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[0] != 1
        or q.dtype not in dtypes
        or q.device.type != "cuda"
        or k.shape != v.shape
        or k.shape[0] != 1
        or q.shape[2:] != k.shape[2:]
        or q.shape[1] != 24
        or k.shape[1] != 2
        or k.device != q.device
        or v.device != q.device
        or indices.device != q.device
        or q.shape[-1] != 256
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or indices.dtype != torch.int32
        or indices.shape != (q.shape[2], 512)
    ):
        raise ValueError(
            f"{op} requires CUDA batch-one {dtypes} head256 GQA24:2 and 512 int32 blocks; "
            f"got {q.device}, Q/K/V dtypes {q.dtype}/{k.dtype}/{v.dtype}"
        )


@torch.compile(dynamic=True, fullgraph=True)
def _raw_flex(q, k, v, mask, scale):
    # Pinned Torch 2.13 internal HOP: retain its base-2 normalization directly.
    # The public API scales LSE to natural logarithms, losing FP32 low bits.
    for x in (q, k, v):
        torch._dynamo.mark_static(x, -3)
        torch._dynamo.mark_static(x, -1)
    options = _apply_kernel_options(
        q,
        k,
        v,
        True,
        {
            "BACKEND": "TRITON",
            "BLOCK_M": 32,
            "BLOCK_N": 32,
        },
    )
    out, lse, _ = flex_attention_hop(
        q, k, v, _identity, mask.as_tuple(), scale, options
    )
    return out, lse


class _HybridAttention(torch.autograd.Function):
    """Preserve Flex forward, use indexed sparse backward on identical activations."""

    @staticmethod
    def forward(ctx, q, k, v, indices, mask, scale):
        out, lse = _raw_flex(q, k, v, mask, scale)
        lse = lse.contiguous()
        ctx.save_for_backward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            indices.contiguous(),
            out.contiguous(),
            lse,
        )
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, indices, out, lse = ctx.saved_tensors
        grad = grad.contiguous()
        delta = _flex_delta(out, grad)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        sorted_indices = indices.sort(dim=-1).values.contiguous()
        physical_dq[(q.shape[2], k.shape[1])](
            q, k, v, sorted_indices, lse, grad, delta, dq,
            N=q.shape[2], D=q.shape[3], G=q.shape[1] // k.shape[1],
            SCALE=ctx.scale, num_warps=4, num_stages=2,
        )
        ordered, offsets = _reverse_adjacency(indices)
        physical_dkv[(triton.cdiv(q.shape[2], 4), k.shape[1])](
            q, k, v, lse, grad, delta, ordered, offsets, dk, dv,
            N=q.shape[2], D=q.shape[3], G=q.shape[1] // k.shape[1],
            SCALE=ctx.scale, num_warps=4, num_stages=2,
        )
        return dq, dk, dv, None, None, None


@torch.compile(dynamic=True, fullgraph=True)
def _compiled_hybrid(q, k, v, indices, mask, scale):
    return _HybridAttention.apply(q, k, v, indices, mask, scale)


def hybrid_attention(q, k, v, indices, mask, scale):
    _validate_sparse_inputs(q, k, v, indices)
    # Dynamic sequence dimensions must not turn the fixed attention scale into
    # an unbacked scalar in the pinned Torch 2.13 FlexAttention lowering.
    with torch._dynamo.config.patch(specialize_float=True):
        return _compiled_hybrid(q, k, v, indices, mask, scale)


@torch.compile(dynamic=True)
def _adjacency_keys(indices):
    n, s = indices.shape
    t = torch.arange(n, device=indices.device)[:, None]
    slot = torch.arange(s + 1, device=indices.device)[None, :]
    extended = torch.cat((indices, ((t + 1) // 4).to(torch.int32)), dim=1)
    keys = torch.where(t < s * 4 + 3, slot, extended)
    valid = (keys >= 0) & (keys * 4 <= t)
    # Invalid entries go to a sentinel bucket after every real key block.
    return torch.where(valid, keys, (n + 3) // 4).to(torch.int32).flatten()


def _reverse_adjacency(indices):
    keys = _adjacency_keys(indices)
    # Stable key sort preserves original query order within each block.
    order = torch.argsort(keys, stable=True)
    query_ids = (order // (indices.shape[1] + 1)).to(torch.int32)
    counts = torch.zeros(
        (indices.shape[0] + 3) // 4 + 1, dtype=torch.int64, device=keys.device
    )
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys, dtype=torch.int64))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return query_ids, offsets
