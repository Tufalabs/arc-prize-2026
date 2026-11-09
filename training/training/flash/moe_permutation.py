"""Gather and weighted reduction with explicit, atomics-free backwards."""

import torch
from torch.autograd.function import once_differentiable

from . import moe_permutation_triton as kernels
from .dtype_contract import floating_inputs, index_inputs


def _validate_permutation(op, x, permutation, count):
    floating_inputs(op, x, cuda=False)
    index_inputs(op, x, permutation)
    if x.ndim != 2 or permutation.ndim != 1 or permutation.numel() != count:
        raise ValueError(f"{op}: x.shape={x.shape}, permutation.shape={permutation.shape}; requires rank-2 rows and {count} rank-1 indices")


def _validate_combine(x, weights, permutation):
    floating_inputs("combine_experts", x, weights, cuda=False)
    _validate_permutation("combine_experts", x, permutation, weights.numel())
    if weights.ndim != 2 or weights.shape[-1] <= 0 or x.shape[0] != weights.numel():
        raise ValueError(f"combine_experts: x.shape={x.shape}, weights.shape={weights.shape}; requires rank-2 weights and one expert row per weight")


def _inverse(permutation):
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def _gather(x, permutation, top_k):
    return x[permutation // top_k]


def _gather_backward(gradient, inverse, count, top_k):
    return gradient[inverse].view(count, top_k, gradient.shape[-1]).sum(1)


def _combine(x, weights, inverse):
    count, top_k = weights.shape
    routed = x[inverse].view(count, top_k, x.shape[-1])
    return (routed * weights[..., None]).sum(1)


def _combine_backward(gradient, x, weights, permutation, top_k):
    upstream = gradient[permutation // top_k]
    dx = upstream * weights.flatten()[permutation, None]
    # Match eager BF16 multiplication rounding before the FP32 reduction.
    dw_sorted = (upstream * x).sum(-1)
    return dx, dw_sorted


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, permutation, top_k):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError(f"gather_tokens: top_k={top_k!r}; requires a positive integer")
        _validate_permutation("gather_tokens", x, permutation, x.shape[0] * top_k if x.ndim else 0)
        if x.is_cuda and x.dtype == torch.float32:
            raise TypeError("gather_tokens: FP32 activations are unsupported")
        inverse = _inverse(permutation)
        ctx.save_for_backward(inverse)
        ctx.mark_non_differentiable(inverse)
        ctx.count, ctx.top_k = x.shape[0], top_k
        if x.is_cuda:
            return kernels.gather(x, permutation, top_k), inverse
        return _gather(x, permutation, top_k), inverse

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient, _inverse_gradient):
        (inverse,) = ctx.saved_tensors
        call = _gather_backward
        if gradient.is_cuda:
            call = kernels.gather_backward
        return call(gradient, inverse, ctx.count, ctx.top_k), None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, permutation, inverse):
        _validate_combine(x, weights, permutation)
        ctx.save_for_backward(x, weights, permutation, inverse)
        if x.is_cuda and x.dtype == weights.dtype and x.dtype != torch.float32:
            return kernels.combine(x, weights, inverse)
        return _combine(x, weights, inverse)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weights, permutation, inverse = ctx.saved_tensors
        if x.is_cuda and x.dtype == weights.dtype and x.dtype != torch.float32:
            dx, dw = kernels.combine_backward(gradient, x, weights, permutation)
            return dx, dw, None, None
        dx, dw = _combine_backward(gradient, x, weights, permutation, weights.shape[-1])
        return dx, dw[inverse].view_as(weights), None, None


@torch.compiler.disable
def gather_tokens(x, permutation, top_k):
    return _Gather.apply(x, permutation, top_k)[0]


@torch.compiler.disable
def combine_experts(x, weights, permutation):
    # Validate before deriving the inverse, which assumes a rank-1 permutation.
    _validate_combine(x, weights, permutation)
    return _Combine.apply(x, weights, permutation, _inverse(permutation))


@torch.compiler.disable
def _gather_with_inverse(x, permutation, top_k):
    return _Gather.apply(x, permutation, top_k)


@torch.compiler.disable
def _combine_with_inverse(x, weights, permutation, inverse):
    return _Combine.apply(x, weights, permutation, inverse)
