"""Vectorised QSA index selection for Flash-Next training.

Transformers 5.16.1's ``Qwen4ExpTextQSAIndexer.forward`` selects each query's
attended key blocks in a Python loop over every query position: at 32k tokens
that is 32k iterations per attention layer, twice per step under activation
checkpointing, and it dominated the trainer probe's forward time. This module
computes the same selection with batched tensor ops (pooled block keys once
per layer, all query/block scores at once, top-k per query, the same boolean
mask) and installs it by swapping the indexer's class, so parameter names,
weights and the attention layer around it are untouched.

Semantics reproduced exactly, for the unpadded causal mask training uses
(query ``i`` sees tokens ``0..i``; in a chunked prefill with a cache, a
chunk's queries sit at the cached length and see every earlier token): block ``j`` (tokens ``4j..4j+3``) is
complete for query ``i`` iff ``4(j+1) <= i+1``; the query scores every
complete block (relu of the per-head dot products, summed over heads, scaled
by ``1/sqrt(d)``), keeps its top ``budget/4`` blocks, and always sees the
incomplete tail block. Top-k ties can order differently from the reference;
``verify_fast_qsa_indexer`` measures that on real inputs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _reference_indexer_class():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer

    return Qwen4ExpTextQSAIndexer


def _fast_forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    if attention_mask is None or attention_mask.dtype != torch.bool:
        raise RuntimeError("the training QSA indexer requires the boolean (sdpa) causal mask")
    batch_size, seq_length, _ = hidden_states.shape
    device = hidden_states.device

    ratio = int(self.compress_ratio)
    head_dim = int(self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]
    hidden_shape = (batch_size, seq_length, -1, head_dim)

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [self.index_n_heads * head_dim, self.index_kv_heads * head_dim],
        dim=-1,
    )
    q = self.q_layernorm(q.reshape(*hidden_shape))
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    raw_keys = token_k.reshape(*hidden_shape).squeeze(2)
    # A chunked prefill (the routing replay) appends this chunk's keys to the
    # cached ones, as the reference does; queries then sit at the offset.
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
    kv_length = raw_keys.shape[1]
    offset = kv_length - seq_length
    if attention_mask.shape != (batch_size, 1, seq_length, kv_length):
        raise RuntimeError(f"unexpected attention mask shape {tuple(attention_mask.shape)}")
    causal = torch.ones(seq_length, kv_length, dtype=torch.bool, device=device).tril_(offset)
    if not torch.equal(attention_mask[:, 0], causal.expand(batch_size, seq_length, kv_length)):
        raise RuntimeError("the training QSA indexer requires an unpadded causal mask")

    num_blocks = kv_length // ratio
    positions = torch.arange(seq_length, device=device) + offset
    complete_blocks = (positions + 1) // ratio  # blocks complete for each query
    block_starts = torch.arange(num_blocks, device=device) * ratio
    if num_blocks > 0:
        pooled = raw_keys[:, : num_blocks * ratio].reshape(batch_size, num_blocks, ratio, head_dim)
        pooled = pooled.float().mean(dim=2).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(2),
            cos=full_cos[:, block_starts, :],
            sin=full_sin[:, block_starts, :],
            unsqueeze_dim=2,
        ).squeeze(2)
        keep = min(int(self.block_topk), num_blocks)
        valid = torch.arange(num_blocks, device=device)[None, :] < complete_blocks[:, None]
        selected_blocks = []
        for start in range(0, seq_length, 4096):
            stop = min(seq_length, start + 4096)
            scores = torch.einsum("bihd,bjd->bihj", q[:, start:stop].float(), block_keys.float())
            scores = torch.relu(scores).sum(dim=2) / math.sqrt(head_dim)
            scores = scores.masked_fill(~valid[start:stop][None], float("-inf"))
            top = scores.topk(keep, dim=-1)
            block_index = top.indices.masked_fill(torch.isinf(top.values), -1)
            selected_blocks.append(block_index)
        blocks = torch.cat(selected_blocks, dim=1)  # [batch, seq, keep], -1 = none
        tokens = blocks[..., None] * ratio + torch.arange(ratio, device=device)
        tokens = torch.where(blocks[..., None] >= 0, tokens, -1).flatten(-2)
    else:
        tokens = torch.full((batch_size, seq_length, 0), -1, dtype=torch.long, device=device)
    tail = complete_blocks[:, None] * ratio + torch.arange(ratio, device=device)[None, :]
    tail = torch.where(tail <= positions[:, None], tail, -1).expand(batch_size, -1, -1)
    selected_token_indices = torch.cat([tokens, tail], dim=-1)

    kv_length = attention_mask.shape[-1]
    scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
    selected_token_mask = torch.zeros(
        (batch_size, seq_length, kv_length + 1), device=device, dtype=torch.bool
    )
    return selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)


def install_fast_qsa_indexer(model: nn.Module) -> int:
    """Swap every attention layer's indexer to the vectorised forward; returns the count."""
    reference = _reference_indexer_class()
    fast = type("FastQSAIndexer", (reference,), {"forward": _fast_forward})
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    count = 0
    for layer in base.model.layers:
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is None:
            continue
        if type(indexer) is not reference:
            raise RuntimeError(f"unexpected indexer class {type(indexer).__name__}")
        indexer.__class__ = fast
        count += 1
    if count == 0:
        raise RuntimeError("no QSA indexer found to replace")
    return count


@torch.no_grad()
def verify_fast_qsa_indexer(model: nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    """Run the reference loop and the vectorised selection on the same captured
    inputs of every attention layer; report exact agreement and the mask entries
    that differ (top-k ties). Call after ``install_fast_qsa_indexer``."""
    reference = _reference_indexer_class()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    captured: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
    handles = []
    for layer in base.model.layers:
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is None:
            continue

        def hook(module, args, kwargs, _captured=captured):
            _captured.append((module, args, kwargs))

        handles.append(indexer.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError("no indexer inputs were captured")
    layers = []
    for module, args, kwargs in captured:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fast_mask = _fast_forward(module, *args, **kwargs)
            reference_mask = reference.forward(module, *args, **kwargs)
        differing = int((fast_mask != reference_mask).sum())
        selected = int(reference_mask.sum())
        layers.append({"differing_entries": differing, "selected_entries": selected})
    return {
        "tokens": int(input_ids.shape[1]),
        "layers_checked": len(layers),
        "exact_layers": sum(1 for item in layers if item["differing_entries"] == 0),
        "max_differing_entries": max(item["differing_entries"] for item in layers),
        "max_differing_fraction": max(
            item["differing_entries"] / max(1, item["selected_entries"]) for item in layers
        ),
    }
