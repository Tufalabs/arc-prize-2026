"""Indexer score kernels with explicit FP32 accumulation and reduction."""

import torch
import triton
import triton.language as tl

from .dtype_contract import floating_inputs


@triton.jit
def _fp32_heads(DOT, S, G, H: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    offset = row * H * G + col
    total = tl.maximum(tl.load(DOT + offset, col < G, 0), 0)
    for head in range(1, H):
        total = total + tl.maximum(tl.load(DOT + offset + head * G, col < G, 0), 0)
    tl.store(S + row * G + col, total * SCALE, col < G)


def fp32_head_reduction(dot, heads, dim):
    floating_inputs("fp32_head_reduction", dot)
    if dot.dtype != torch.float32:
        raise TypeError(f"fp32_head_reduction: requires FP32 dot products, got {dot.dtype}")
    dot = dot.contiguous()
    result = dot.new_empty((dot.shape[0] // heads, dot.shape[1]))
    _fp32_heads[(result.shape[0], triton.cdiv(result.shape[1], 256))](
        dot, result, result.shape[1], heads, dim ** -0.5, 256,
        enable_fp_fusion=False,
    )
    return result


@triton.jit
def _indexer_scores(Q, K, S, N, G,
                    H: tl.constexpr, D: tl.constexpr,
                    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
                    KS0: tl.constexpr, KS1: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    d = tl.arange(0, BD)
    q = tl.load(Q + (row // H)[:, None] * QS0 + (row % H)[:, None] * QS1
                + d[None, :] * QS2, (row[:, None] < N * H) & (d[None, :] < D), 0)
    k = tl.load(K + col[None, :] * KS0 + d[:, None] * KS1,
                (col[None, :] < G) & (d[:, None] < D), 0)
    dot = tl.dot(q, k, out_dtype=tl.float32)
    score = tl.sum(tl.maximum(dot, 0).reshape(BM // H, H, BN), 1) / tl.sqrt(float(D))
    query = tl.program_id(0) * (BM // H) + tl.arange(0, BM // H)
    tl.store(S + query[:, None] * G + col[None, :], score,
             (query[:, None] < N) & (col[None, :] < G))


def scores(query, keys):
    floating_inputs("BF16 indexer scoring", query, keys, same_dtype=True)
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"BF16/FP16 indexer scoring: requires BF16/FP16 inputs, got {query.dtype}")
    if query.ndim != 3 or keys.ndim != 2 or query.shape[-1] != keys.shape[-1]:
        raise ValueError(f"bf16_scores: expected query [tokens, heads, dim] and keys [blocks, dim]; got {query.shape}, {keys.shape}")
    n, heads, dim = query.shape
    if not query.is_cuda or heads not in (1, 2, 4, 8) or not 0 < dim <= 256:
        raise ValueError("BF16 indexer scoring requires CUDA, 1/2/4/8 heads and dim <= 256")
    result = torch.empty((n, keys.shape[0]), device=query.device, dtype=torch.float32)
    _indexer_scores[(triton.cdiv(n * heads, 64), triton.cdiv(keys.shape[0], 128))](
        query, keys, result, n, keys.shape[0], heads, dim,
        *query.stride(), *keys.stride(), 64, 128, triton.next_power_of_2(dim),
    )
    return result
