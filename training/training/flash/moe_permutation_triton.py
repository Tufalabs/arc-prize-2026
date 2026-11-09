"""Coalesced MoE copies/reductions with static channel and top-k indexing."""

import torch
import triton
import triton.language as tl

from .dtype_contract import low_precision_inputs, index_inputs


@triton.jit
def _moe_gather(X, P, Y, SIZE, D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    row = tl.load(P + i // D, i < SIZE, 0) // K
    value = tl.load(X + row * D + i % D, i < SIZE, 0)
    tl.store(Y + i, value, i < SIZE)


@triton.jit
def _moe_gather_backward(G, INV, DX, SIZE, D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for expert in range(K):
        row = tl.load(INV + (i // D) * K + expert, i < SIZE, 0)
        total += tl.load(G + row * D + i % D, i < SIZE, 0).to(tl.float32)
    tl.store(DX + i, total, i < SIZE)


@triton.jit
def _moe_weighted_scatter(X, W, INV, Y, SIZE, D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for expert in range(K):
        slot = (i // D) * K + expert
        row = tl.load(INV + slot, i < SIZE, 0)
        x = tl.load(X + row * D + i % D, i < SIZE, 0)
        w = tl.load(W + slot, i < SIZE, 0)
        # Eager rounds the product before its top-k reduction.
        product = (x.to(tl.float32) * w.to(tl.float32)).to(x.dtype)
        total += product.to(tl.float32)
    tl.store(Y + i, total, i < SIZE)


@triton.jit
def _moe_weighted_scatter_backward(G, X, W, P, DX, DW,
                                   D: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    slot = tl.load(P + row)
    g = tl.load(G + (slot // K) * D + col, col < D, 0)
    x = tl.load(X + row * D + col, col < D, 0)
    w = tl.load(W + slot)
    tl.store(DX + row * D + col, g.to(tl.float32) * w.to(tl.float32), col < D)
    product = (g.to(tl.float32) * x.to(tl.float32)).to(x.dtype)
    tl.store(DW + slot, tl.sum(product.to(tl.float32), 0))


def gather(x, permutation, top_k):
    low_precision_inputs("gather_tokens", x)
    index_inputs("gather_tokens", x, permutation)
    x = x.contiguous()
    permutation = permutation.contiguous()
    y = x.new_empty((permutation.numel(), x.shape[-1]))
    _moe_gather[(triton.cdiv(y.numel(), 1024),)](
        x, permutation, y, y.numel(), x.shape[-1], top_k, 1024,
    )
    return y


def gather_backward(gradient, inverse, count, top_k):
    low_precision_inputs("gather_tokens backward", gradient)
    index_inputs("gather_tokens backward", gradient, inverse)
    gradient = gradient.contiguous()
    inverse = inverse.contiguous()
    dx = gradient.new_empty((count, gradient.shape[-1]))
    _moe_gather_backward[(triton.cdiv(dx.numel(), 256),)](
        gradient, inverse, dx, dx.numel(), dx.shape[-1], top_k, 256,
    )
    return dx


def combine(x, weights, inverse):
    low_precision_inputs("combine_experts", x, weights, same_dtype=True)
    index_inputs("combine_experts", x, inverse)
    x, weights = x.contiguous(), weights.contiguous()
    inverse = inverse.contiguous()
    y = x.new_empty((weights.shape[0], x.shape[-1]))
    _moe_weighted_scatter[(triton.cdiv(y.numel(), 256),)](
        x, weights, inverse, y, y.numel(), x.shape[-1], weights.shape[-1], 256,
    )
    return y


def combine_backward(gradient, x, weights, permutation):
    low_precision_inputs("combine_experts backward", x, weights, gradient, same_dtype=True)
    index_inputs("combine_experts backward", x, permutation)
    gradient, x, weights = gradient.contiguous(), x.contiguous(), weights.contiguous()
    permutation = permutation.contiguous()
    dx, dw = torch.empty_like(x), torch.empty_like(weights)
    _moe_weighted_scatter_backward[(x.shape[0],)](
        gradient, x, weights, permutation, dx, dw,
        x.shape[-1], weights.shape[-1], triton.next_power_of_2(x.shape[-1]),
    )
    return dx, dw
