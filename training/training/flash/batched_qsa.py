"""Opt-in Qwen4Exp QSA prefill indexer; main attention/autograd are unchanged.

Specializes batch-one contiguous causal BF16 prefill without a cache. Other
layouts dispatch to the original bound method. No parameters or buffers change.
"""

from __future__ import annotations

import weakref

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)

from .decision_replay import reject_replay
from .options import SELECTOR_POLICIES
from .selector_scoring import bf16_scores, compiled_fp32_scores, fp32_scores

SELECTOR_POLICY = "batched_fp32_top513_v1"


def _causal_layout(module, x, mask, cache):
    if (
        cache is not None
        or x.ndim != 3
        or x.shape[0] != 1
        or x.dtype != torch.bfloat16
        or module.index_kv_heads != 1
        or mask is None
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (1, 1, x.shape[1], x.shape[1])
        or mask.device != x.device
        or torch.backends.cuda.matmul.allow_tf32
    ):
        return False
    try:
        version = mask._version
    except RuntimeError:  # Inference tensors have no mutation version counter.
        version = None
    previous = module._qsa_layout_cache
    if (
        version is not None
        and previous
        and previous[0]() is mask
        and previous[1] == version
    ):
        return True
    n = x.shape[1]
    keys = torch.arange(n, device=x.device)
    for start in range(0, n, 1024):
        queries = torch.arange(start, min(start + 1024, n), device=x.device)
        if not torch.equal(
            mask[0, 0, start : start + 1024], queries[:, None] >= keys[None, :]
        ):
            return False
    module._qsa_layout_cache = (
        (weakref.ref(mask), version) if version is not None else None
    )
    return True


@torch.no_grad()
def _batched(module, x, position_embeddings, mask, return_blocks=False):
    n = x.shape[1]
    ratio, budget = module.compress_ratio, module.token_budget
    k = module.block_topk
    tile = module._qsa_tile
    policy = module.selector_policy
    stats = {
        "path": "batched",
        "sequence": n,
        "tile": tile,
        "guarded_rows": 0,
        "selector_policy": module.selector_policy,
    }
    module._qsa_last_stats = stats
    # Until there are k+1 complete blocks, every visible token is selected.
    prefix = min(n, (k + 1) * ratio - 1)
    if return_blocks:
        selected = torch.full((n, k), -1, device=x.device, dtype=torch.int32)
    else:
        selected = torch.zeros((n, n + 1), device=x.device, dtype=torch.bool)
        selected[:prefix, :n].copy_(mask[0, 0, :prefix])
    if prefix == n:
        return selected if return_blocks else selected[:, :n][None, None]

    cos, sin = position_embeddings
    qk = module.index_qk_proj(x)
    q, raw = qk.split(
        [
            module.index_n_heads * module.index_head_dim,
            module.index_kv_heads * module.index_head_dim,
        ],
        dim=-1,
    )
    q = module.q_layernorm(q.reshape(1, n, module.index_n_heads, module.index_head_dim))
    q = apply_rotary_pos_emb(q, cos=cos[:, -n:], sin=sin[:, -n:], unsqueeze_dim=2)[0]
    raw = raw.reshape(n, module.index_head_dim)
    groups = n // ratio
    # Match reference arithmetic: FP32 mean, cast back, normalize, then RoPE.
    pooled = (
        raw[: groups * ratio].reshape(groups, ratio, -1).float().mean(1).to(raw.dtype)
    )
    pooled = module.k_layernorm(pooled)
    starts = torch.arange(groups, device=x.device) * ratio
    block_keys = apply_rotary_pos_emb(
        pooled[:, None],
        cos=cos[0].index_select(0, starts),
        sin=sin[0].index_select(0, starts),
    ).squeeze(1)
    score = {
        SELECTOR_POLICY: fp32_scores,
        "compiled_fp32_top513_v1": compiled_fp32_scores,
        "bf16_dot_fp32_top513_v1": bf16_scores,
    }[policy]
    keys = block_keys if score is bf16_scores else block_keys.float()
    offsets = torch.arange(ratio, device=x.device)
    tail_offsets = torch.arange(ratio - 1, device=x.device)
    for start in range(prefix, n, tile):
        end = min(start + tile, n)
        positions = torch.arange(start, end, device=x.device)
        complete = (positions + 1) // ratio
        max_groups = end // ratio
        scores = score(q[start:end], keys[:max_groups])
        scores.masked_fill_(
            torch.arange(max_groups, device=x.device)[None] >= complete[:, None],
            -torch.inf,
        )
        # Keep the qualified batched top-(k+1) membership; no per-query
        # reference rescore, near-tie guard or host synchronization.
        _, block_ids = scores.topk(k + 1, dim=-1)
        chosen = block_ids[:, :k].clone()
        if return_blocks:
            selected[start:end].copy_(chosen.to(torch.int32))
        else:
            tokens = (chosen[:, :, None] * ratio + offsets).reshape(end - start, budget)
            selected[start:end].scatter_(1, tokens, True)
            tail = complete[:, None] * ratio + tail_offsets
            tail = torch.where(tail <= positions[:, None], tail, n)
            selected[start:end].scatter_(1, tail, True)
    return selected if return_blocks else selected[:, :n][None, None]


@torch.no_grad()
def batched_documents(module, x, position_embeddings, lengths):
    """Run the qualified selector independently for each aligned document."""
    if x.shape[0] != 1 or sum(lengths) != x.shape[1] or any(n <= 0 for n in lengths):
        raise ValueError("Invalid packed QSA document lengths")
    ratio, k = module.compress_ratio, module.block_topk
    if any(n % ratio for n in lengths):
        raise ValueError("Packed QSA documents must align to the compression ratio")
    selected = []
    start = 0
    for length in lengths:
        stop = start + length
        local_positions = tuple(value[:, start:stop] for value in position_embeddings)
        local = _batched(
            module, x[:, start:stop], local_positions, None, return_blocks=True
        )
        local = torch.where(local >= 0, local + start // ratio, local)
        selected.append(local)
        start = stop
    result = torch.cat(selected)
    if result.shape != (x.shape[1], k):
        raise RuntimeError("Packed selector shape mismatch")
    module._qsa_last_stats = {
        "path": "batched_documents",
        "sequence": x.shape[1],
        "documents": len(lengths),
        "selector_policy": module.selector_policy,
    }
    return result


class BatchedQSAIndexer(Qwen4ExpTextQSAIndexer):
    """Explicit QSA adapter with a native fallback and no method mutation."""

    def __init__(self, config, layer_idx, *, tile=256, selector_policy=SELECTOR_POLICY):
        super().__init__(config, layer_idx)
        if tile <= 0 or self.token_budget % self.compress_ratio:
            raise ValueError("Positive tile and an integral block budget required")
        self.selector_policy = selector_policy
        if self.selector_policy not in SELECTOR_POLICIES:
            raise ValueError("Unknown Flash selector_policy")
        self._qsa_tile = tile
        self._qsa_layout_cache = None
        self._qsa_last_stats = {}

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
    ):
        reject_replay("BatchedQSAIndexer with attention_backend='sdpa'", "use flex_qsa")
        if _causal_layout(self, hidden_states, attention_mask, past_key_values):
            return _batched(self, hidden_states, position_embeddings, attention_mask)
        self._qsa_last_stats = {"path": "original_fallback"}
        return super().forward(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )
