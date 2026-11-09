"""Frozen-head chunked cross-entropy, with optional logit retention for backward.

ChunkedCE recomputes logits in backward; CachedHeadCE keeps them from forward
and releases them before decoder backward. Both match eager autograd of
summed cross-entropy, including ignore_index=-100 and autocast.
"""

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


def validate_head_inputs(op, hidden, weight, targets, chunk):
    if hidden.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError(f"{op}: expected hidden/weight rank 2 and targets rank 1; got {hidden.shape}, {weight.shape}, {targets.shape}")
    if hidden.shape[0] != targets.shape[0] or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"{op}: incompatible hidden={hidden.shape}, weight={weight.shape}, targets={targets.shape}")
    if targets.dtype != torch.int64:
        raise TypeError(f"{op}: targets dtype={targets.dtype}; allowed value is torch.int64")
    if type(chunk) is not int or chunk <= 0:
        raise ValueError(f"{op}: chunk={chunk!r}; requires a positive integer")
    for name, value in (("hidden", hidden), ("weight", weight)):
        if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"{op}: unsupported {name} dtype {value.dtype}")
    if hidden.device != weight.device or targets.device != hidden.device:
        raise ValueError(f"{op}: hidden, weight and targets must share a device")
    if hidden.dtype != weight.dtype and not torch.is_autocast_enabled(hidden.device.type):
        raise TypeError(f"{op}: mixed hidden/weight dtypes require autocast")
    if weight.requires_grad or not targets.numel():
        raise ValueError(f"{op}: requires frozen weights and nonempty targets")


def cross_entropy_gradient(logits, targets):
    """d(summed cross-entropy)/d(logits) for FP32 logits; ignore_index -100 rows are zero."""
    dlogits = logits.softmax(-1)
    ignored = targets == -100
    rows = torch.arange(len(targets), device=logits.device)
    dlogits[rows, targets.masked_fill(ignored, 0)] -= 1
    return dlogits.masked_fill_(ignored[:, None], 0)


class CachedHeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        validate_head_inputs("CachedHeadCE", hidden, weight, targets, chunk)
        logits = []
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, len(targets), chunk):
            block = F.linear(hidden[start:start + chunk], weight)
            logits.append(block)
            total += F.cross_entropy(block.float(), targets[start:start + chunk], reduction="sum")
        ctx.save_for_backward(weight, targets, *logits)
        ctx.chunk = chunk
        ctx.shape, ctx.dtype = hidden.shape, hidden.dtype
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        weight, targets, *logits = ctx.saved_tensors
        out = torch.empty(ctx.shape, device=weight.device, dtype=ctx.dtype)
        for i, block in enumerate(logits):
            start, stop = i * ctx.chunk, min((i + 1) * ctx.chunk, len(targets))
            dlogits = cross_entropy_gradient(block.float(), targets[start:stop])
            with torch.autocast(weight.device.type, enabled=False):
                out[start:stop] = (dlogits * gradient).to(block.dtype) @ weight.to(block.dtype)
        return out, None, None, None


class ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        validate_head_inputs("ChunkedCE", hidden, weight, targets, chunk)
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk = chunk
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for start in range(0, len(targets), chunk):
            logits = torch.nn.functional.linear(
                hidden[start : start + chunk], weight
            )
            ctx.logit_dtype = logits.dtype
            total += torch.nn.functional.cross_entropy(
                logits.float(), targets[start : start + chunk], reduction="sum"
            )
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        hidden, weight, targets = ctx.saved_tensors
        out = torch.empty_like(hidden)
        for start in range(0, len(targets), ctx.chunk):
            stop = min(start + ctx.chunk, len(targets))
            with torch.autocast(hidden.device.type, enabled=False):
                logits = torch.nn.functional.linear(
                    hidden[start:stop].to(ctx.logit_dtype), weight.to(ctx.logit_dtype)
                ).float()
            dlogits = cross_entropy_gradient(logits, targets[start:stop])
            with torch.autocast(hidden.device.type, enabled=False):
                out[start:stop] = (dlogits * grad).to(ctx.logit_dtype) @ weight.to(ctx.logit_dtype)
        return out, None, None, None


def completion_loss(hidden, head, ids, prompt_len, chunk=128, *, cache_logits=False):
    # Logit at p-1 predicts the first completion token p. Prompt never supervised.
    h = hidden[0, prompt_len - 1 : -1].contiguous()
    targets = ids[0, prompt_len:]
    if cache_logits:
        return CachedHeadCE.apply(h, head.weight, targets, chunk)
    return ChunkedCE.apply(h, head.weight, targets, chunk)
