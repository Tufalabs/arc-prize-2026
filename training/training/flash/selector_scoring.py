"""Indexer-only scoring policies; attention masks and top-k semantics stay intact."""

import math

import torch

from . import selector_triton as kernels
from .dtype_contract import floating_inputs


def fp32_scores(query, keys):
    floating_inputs("fp32_scores", query, keys, cuda=False)
    with torch.autocast(query.device.type, enabled=False):
        n, heads, dim = query.shape
        dot = query.float().reshape(-1, dim) @ keys.float().T
        return dot.reshape(n, heads, keys.shape[0]).transpose(1, 2).relu().sum(-1) / math.sqrt(dim)


def _ordered_fp32_scores(query, keys):
    with torch.autocast(query.device.type, enabled=False):
        n, heads, dim = query.shape
        dot = query.float().reshape(-1, dim) @ keys.float().T
        dot = dot.reshape(n, heads, keys.shape[0])
        # Eager reduces this strided head dimension from left to right. A Triton
        # tree reduction changes near-tie ordering even with FP32 dot products.
        total = dot[:, 0].relu()
        for head in range(1, heads):
            total = total + dot[:, head].relu()
        return total / math.sqrt(dim)


@torch.compiler.disable
def compiled_fp32_scores(query, keys):
    floating_inputs("compiled_fp32_scores", query, keys, cuda=False)
    with torch.autocast(query.device.type, enabled=False):
        if not query.is_cuda:
            return _ordered_fp32_scores(query, keys)
        _, heads, dim = query.shape
        dot = query.float().reshape(-1, dim) @ keys.float().T
        return kernels.fp32_head_reduction(dot, heads, dim)


@torch.compiler.disable
def bf16_scores(query, keys):
    if query.dtype not in (torch.bfloat16, torch.float16) or keys.dtype != query.dtype:
        raise ValueError("BF16/FP16 selector requires matching BF16/FP16 query and keys")
    return kernels.scores(query, keys)
