"""The head's QSA attention for training, with multi-step (draft) queries.

How the served drafter attends (``vllm/models/qwen4_exp/nvidia``):

- **Pass 0** runs the head over every position the target verified. Its
  attention is ordinary QSA: each query scores the complete four-token
  blocks it can see with the indexer, keeps the top 512, and attends to
  their tokens plus its own open block.
- **Draft step j >= 1** runs one query per request at position t + j (t is
  the anchor: the last position pass 0 covered). The drafter's config does
  not set ``index_share_for_mtp_iteration``, so the step runs its own
  indexer over the cache, which then holds pass 0's keys up to t and the
  earlier drafts' keys at t+1 .. t+j.

Training runs every anchor at once. Step j's attention is two parts merged by
their log-sum-exps: its own sparse selection over pass 0's keys up to its
anchor, and the draft keys at t+1 .. t+j (its own included), which sit in
its open block or the one before. One approximation: a four-token block that
straddles t is served as a candidate block pooled from pass-0 and draft
keys; here its pass-0 part is always attended and its draft part always
attended as the diagonal. Parity against the served drafter measures what
that costs.

The selection and the mask are our, from arc-training's router
distillation (``sft/distillation/batched_qsa.py`` and ``flex_qsa.py`` on
``flash-router-distillation-production``), generalized from "a query
sees up to itself" to "a query sees up to its row's limit": FP32 batched
block scores and top-k, bitset membership, a conservative 128 x 128 tile
union, and the compiled FlexAttention call with his B200 head-256 tiling.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

QUERY_TILE = 128
KEY_TILE = 128
_compiled: Any = None


def fake_fp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` ``[..., dim]`` as a ``fp8_per_token_head`` KV cache holds it: one
    scale per row, absmax / 448 floored at 1e-6, values stored in FP8 e4m3
    (vLLM's ``_reshape_cache_per_token_head``). The served drafter's cache
    follows the target's dtype, so its attention reads keys and values this
    way. Straight-through: the backward sees the identity."""
    detached = x.detach().float()
    scale = (detached.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-6)
    stored = (detached * (1.0 / scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return x + (stored.to(x.dtype) - x).detach()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial rotary on the leading ``cos.shape[-1]`` dims of ``x``
    ``[tokens, heads, dim]``; ``cos``/``sin`` are ``[tokens, rotary]``."""
    rotary = cos.shape[-1]
    cos, sin = cos[:, None, :], sin[:, None, :]
    x_rope, x_pass = x[..., :rotary], x[..., rotary:]
    half = rotary // 2
    rotated = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
    return torch.cat((x_rope * cos + rotated * sin, x_pass), dim=-1)


@torch.no_grad()
def block_keys(indexer: Any, raw: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The indexer's pooled key per complete four-token block, as the
    reference computes it: FP32 mean, cast back, normalize, then rope at the
    block's first position. ``raw`` is ``[tokens, index_head_dim]``."""
    ratio = indexer.compress_ratio
    groups = raw.shape[0] // ratio
    pooled = raw[: groups * ratio].reshape(groups, ratio, -1).float().mean(1).to(raw.dtype)
    pooled = indexer.k_layernorm(pooled)
    starts = torch.arange(groups, device=raw.device) * ratio
    return rope(pooled[:, None], cos.index_select(0, starts), sin.index_select(0, starts)).squeeze(1)


def indexer_projections(
    indexer: Any, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roped indexer queries ``[tokens, heads, dim]`` and raw keys ``[tokens, dim]``."""
    n = hidden.shape[0]
    qk = indexer.index_qk_proj(hidden)
    q, raw = qk.split(
        [indexer.index_n_heads * indexer.index_head_dim, indexer.index_kv_heads * indexer.index_head_dim], dim=-1
    )
    q = indexer.q_layernorm(q.reshape(n, indexer.index_n_heads, indexer.index_head_dim))
    return rope(q, cos, sin), raw.reshape(n, indexer.index_head_dim)


@torch.no_grad()
def select_blocks(
    indexer: Any, queries: torch.Tensor, keys: torch.Tensor, limits: torch.Tensor, tile: int = 256
) -> torch.Tensor:
    """Each row's top-k complete blocks among those whose tokens all lie at
    or before its limit, as int32 block ids (``-1`` where every visible block
    is kept anyway: those rows are in the dense prefix)."""
    ratio, k = indexer.compress_ratio, indexer.block_topk
    n = queries.shape[0]
    chosen = torch.full((n, k), -1, device=queries.device, dtype=torch.int32)
    prefix = (k + 1) * ratio - 1
    rows = torch.nonzero(limits >= prefix).flatten()
    if rows.numel() == 0:
        return chosen
    keys32 = keys.float()
    scale = math.sqrt(indexer.index_head_dim)
    for start in range(int(rows[0]), n, tile):
        end = min(start + tile, n)
        complete = (limits[start:end] + 1) // ratio
        max_groups = int(complete.max())
        query = queries[start:end].float()
        dot = query.reshape(-1, indexer.index_head_dim) @ keys32[:max_groups].T
        dot = dot.reshape(end - start, indexer.index_n_heads, max_groups).transpose(1, 2)
        scores = dot.relu().sum(-1) / scale
        scores.masked_fill_(torch.arange(max_groups, device=queries.device)[None] >= complete[:, None], -torch.inf)
        top = scores.topk(k, dim=-1)
        ids = torch.where(top.values > -torch.inf, top.indices, -1)
        dense = complete[:, None] <= k
        chosen[start:end] = torch.where(dense, -1, ids).to(torch.int32)
    return chosen


@torch.no_grad()
def selection_mask(chosen: torch.Tensor, limits: torch.Tensor, n_kv: int, ratio: int, budget: int) -> BlockMask:
    """Bitset membership plus a conservative tile union, without a dense
    mask: a query sees its chosen blocks, every token when its limit is in
    the dense prefix, and its limit's open block; never past its limit."""
    n_q = chosen.shape[0]
    nq, nk = (n_q + QUERY_TILE - 1) // QUERY_TILE, (n_kv + KEY_TILE - 1) // KEY_TILE
    bits = torch.zeros((nq * QUERY_TILE, nk), dtype=torch.int32, device=chosen.device)
    valid = chosen >= 0
    ids = chosen.clamp_min(0).long()
    values = (torch.ones_like(ids) << (ids % 32)).to(torch.int32)
    # Distinct ids per row: scatter_add is a bitwise OR, sign bit included.
    bits[:n_q].scatter_add_(1, ids // 32, torch.where(valid, values, 0))
    prefix = budget + ratio - 1
    present = bits.view(nq, QUERY_TILE, nk).ne(0).any(1)
    padded = torch.full((nq * QUERY_TILE,), -1, dtype=limits.dtype, device=limits.device)
    padded[:n_q] = limits
    tile_limits = padded.view(nq, QUERY_TILE)
    lo = torch.where(tile_limits >= 0, tile_limits, tile_limits.max()).min(1).values[:, None]
    hi = tile_limits.max(1).values[:, None]
    kstart = torch.arange(nk, device=chosen.device)[None, :] * KEY_TILE
    present |= (lo < prefix) & (kstart <= hi.clamp_max(prefix - 1))
    present |= (kstart <= hi) & (kstart + KEY_TILE > ((lo + 1) // ratio) * ratio)
    counts = present.sum(-1).to(torch.int32)[None, None]
    indices = torch.argsort(present.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)[None, None]
    limits = limits.contiguous()

    def mask_mod(b: Any, h: Any, q: Any, kv: Any) -> Any:
        # Kernels can evaluate padded indices: clamp loads, then mask them out.
        qc = q.clamp(0, n_q - 1)
        limit = limits[qc]
        word = bits[qc, (kv // KEY_TILE).clamp(0, nk - 1)]
        selected = ((word >> ((kv // ratio) % 32)) & 1) != 0
        tail = kv >= ((limit + 1) // ratio) * ratio
        return (q < n_q) & (kv < n_kv) & (kv <= limit) & ((limit < prefix) | selected | tail)

    return BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=(QUERY_TILE, KEY_TILE), mask_mod=mask_mod, seq_lengths=(n_q, n_kv)
    )


def sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: BlockMask, scale: float, compiled: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` ``[1, heads, n_q, dim]``, ``k``/``v`` ``[1, kv_heads, n_kv, dim]``;
    returns the output and its natural-log log-sum-exp ``[1, heads, n_q]``."""
    global _compiled
    function: Any = flex_attention
    options: dict[str, Any] | None = None
    if compiled:
        if _compiled is None:
            _compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)
        function = _compiled
        # B200/head_dim 256: keep forward tiling, tune the backward.
        options = {"BACKEND": "TRITON", "BLOCK_M": 32, "BLOCK_N": 32, "bwd_num_warps": 8, "bwd_num_stages": 2}
    out, aux = function(
        q, k, v, block_mask=mask, scale=scale, enable_gqa=True, return_aux=AuxRequest(lse=True), kernel_options=options
    )
    return out, aux.lse


def merge_diagonal(
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    diagonal: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> torch.Tensor:
    """Fold per-row extra keys into an attention output. ``out`` ``[n, heads,
    dim]`` with ``lse`` ``[n, heads]`` over the shared context; ``q`` ``[n,
    heads, dim]``; each diagonal entry is one extra key and value per row,
    ``[n, kv_heads, dim]``, attended by that row alone."""
    heads, kv_heads = q.shape[1], diagonal[0][0].shape[1]
    group = heads // kv_heads
    scores = [lse.float()]
    values = []
    for key, value in diagonal:
        key = key.repeat_interleave(group, dim=1)
        scores.append((q.float() * key.float()).sum(-1) * scale)
        values.append(value.repeat_interleave(group, dim=1).float())
    stacked = torch.stack(scores, dim=0)  # [1 + d, n, heads]
    weights = torch.softmax(stacked, dim=0)
    merged = weights[0][..., None] * out.float()
    for weight, value in zip(weights[1:], values):
        merged = merged + weight[..., None] * value
    return merged.to(out.dtype)
