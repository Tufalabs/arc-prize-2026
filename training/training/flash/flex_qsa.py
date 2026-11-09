"""Batched QSA selections with a compiled sparse-attention training backend.

Qualified scope: causal BF16 training without KV cache/dropout. Public padded
batches and explicit packed documents are converted to one document-isolated
token stream before entering the attention layers. Only Qwen4ExpTextAttention
is replaced. DeltaNet is untouched. The indexer's FP32 batched ranking is
shared with the SDPA backend; near-tie rescoring is disabled.
"""

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    apply_rotary_pos_emb,
)

from .batched_qsa import BatchedQSAIndexer, _batched, batched_documents
from .decision_replay import qsa_blocks
from .group_qsa import attention as group_attention
from .indexed_qsa import hybrid_attention as indexed_attention
from .physical_qsa import hybrid_attention as physical_attention
from .qsa_selection import select_blocks
from .qsa_policy import validate_qsa_kernel
from .qsa_retention import retain_attention

QUERY_TILE = 128
KEY_TILE = 128
_compiled = None
_events = None


def begin_metrics():
    global _events
    _events = []


def finish_metrics():
    """Called after the worker synchronizes; includes checkpoint re-execution."""
    global _events
    events, _events = _events, None
    if not events:
        return {}
    masks = [r for r in events if r["stage"] >= 2 and "live" in r]
    result = {
        "calls_including_recompute": len(events),
        "attention_core_computations": sum(
            r.get("core_reused") is False for r in events
        ),
        "attention_core_reuses": sum(r.get("core_reused") is True for r in events),
        "attention_retained_storage_bytes": sum(
            r.get("retained_bytes", 0) for r in events if r.get("core_reused") is False
        ),
        "selection_computations": sum(
            r.get("selection_reused") is False for r in events
        ),
        "selection_reuses": sum(r.get("selection_reused") is True for r in events),
        "selection_original_bytes": sum(
            r.get("selection_original_bytes", 0) for r in events
        ),
        "completed_calls_including_recompute": sum(r["stage"] == 3 for r in events),
        "selection_ms": sum(
            r["events"][0].elapsed_time(r["events"][1])
            for r in events
            if r["stage"] >= 1
        ),
        "mask_build_ms": sum(
            r["events"][1].elapsed_time(r["events"][2]) for r in masks
        ),
        "completed_forward_after_mask_ms": sum(
            r["events"][2].elapsed_time(r["events"][3])
            for r in events
            if r["stage"] == 3
        ),
    }
    if masks:
        live = sum(int(r["live"].item()) for r in masks)
        result["tile_skip_fraction"] = 1 - live / sum(r["total"] for r in masks)
        result["qsa_extra_tile_skip_fraction"] = 1 - live / sum(
            r["causal"] for r in masks
        )
    return result


@torch.no_grad()
def selection_metadata(
    chosen, ratio, budget, query_tile=QUERY_TILE, packed_lengths=None
):
    """Bitset membership + conservative tile union, without a dense NxN mask.

    Each int32 stores 32 four-token QSA blocks (128 keys). Distinct top-k
    indices mean scatter_add acts as bitwise OR, including the sign bit.
    Prefix and incomplete tail membership are evaluated analytically.
    """
    n = chosen.shape[0]
    if ratio != 4 or budget % ratio or chosen.shape[1] != budget // ratio:
        raise ValueError(
            "flex_qsa requires four-token QSA blocks and exact top-k shape"
        )
    if query_tile != QUERY_TILE:
        raise ValueError("unqualified query tile")
    nq, nk = (n + query_tile - 1) // query_tile, (n + KEY_TILE - 1) // KEY_TILE
    bits = torch.zeros((nq * query_tile, nk), dtype=torch.int32, device=chosen.device)
    valid = chosen >= 0
    ids = chosen.clamp_min(0).long()
    values = (torch.ones_like(ids) << (ids % 32)).to(torch.int32)
    bits[:n].scatter_add_(1, ids // 32, torch.where(valid, values, 0))
    present = bits.view(nq, query_tile, nk).ne(0).any(1)
    qstart = torch.arange(nq, device=chosen.device)[:, None] * query_tile
    qend = (qstart + query_tile - 1).clamp_max(n - 1)
    kstart = torch.arange(nk, device=chosen.device)[None, :] * KEY_TILE
    packed = packed_lengths is not None
    if packed_lengths is None:
        packed_lengths = (n,)
    packed_lengths = tuple(int(value) for value in packed_lengths)
    if (
        sum(packed_lengths) != n
        or any(
            value <= 0 or (packed and value % query_tile) for value in packed_lengths
        )
        or query_tile != KEY_TILE
    ):
        raise ValueError("Packed documents must be positive whole FlexAttention tiles")
    tile_document = torch.empty(nq, dtype=torch.long, device=chosen.device)
    token_document = torch.empty(n, dtype=torch.long, device=chosen.device)
    document_starts = torch.empty(
        len(packed_lengths), dtype=torch.long, device=chosen.device
    )
    document_lengths = torch.tensor(
        packed_lengths, dtype=torch.long, device=chosen.device
    )
    prefix_lengths = document_lengths.clamp_max(budget + ratio - 1)
    start = 0
    for document, length in enumerate(packed_lengths):
        stop = start + length
        tile_document[start // query_tile : (stop + query_tile - 1) // query_tile] = (
            document
        )
        token_document[start:stop] = document
        document_starts[document] = start
        prefix = min(length, budget + ratio - 1)
        qslice = slice(start // query_tile, (stop + query_tile - 1) // query_tile)
        qlocal_start = qstart[qslice] - start
        qlocal_end = qend[qslice] - start
        klocal_start = kstart - start
        same = tile_document[qslice, None] == tile_document[None, :]
        # Prefix is dense causal; the partial current four-token block is included.
        present[qslice] |= (
            same
            & (qlocal_start < prefix)
            & (klocal_start <= qlocal_end.clamp_max(prefix - 1))
        )
        present[qslice] |= (
            same
            & (klocal_start <= qlocal_end)
            & (klocal_start + KEY_TILE > (qlocal_start // ratio) * ratio)
        )
        start = stop
    same_tiles = tile_document[:, None] == tile_document[None, :]
    present &= same_tiles
    counts = present.sum(-1).to(torch.int32)[None, None]
    indices = torch.argsort(
        present.to(torch.int8), dim=-1, descending=True, stable=True
    ).to(torch.int32)[None, None]

    def mask_mod(b, h, q, kv):
        # Kernels can evaluate padded indices: clamp loads, then mask them out.
        word = bits[q.clamp(0, n - 1), (kv // KEY_TILE).clamp(0, nk - 1)]
        selected = ((word >> ((kv // ratio) % 32)) & 1) != 0
        safe_q, safe_kv = q.clamp(0, n - 1), kv.clamp(0, n - 1)
        qdoc, kvdoc = token_document[safe_q], token_document[safe_kv]
        qlocal = q - document_starts[qdoc]
        kvlocal = kv - document_starts[kvdoc]
        prefix = prefix_lengths[qdoc]
        tail = kvlocal >= ((qlocal + 1) // ratio) * ratio
        return (
            (q < n)
            & (kv < n)
            & (qdoc == kvdoc)
            & (kvlocal <= qlocal)
            & ((qlocal < prefix) | selected | tail)
        )

    mask = BlockMask.from_kv_blocks(
        counts,
        indices,
        BLOCK_SIZE=(query_tile, KEY_TILE),
        mask_mod=mask_mod,
        seq_lengths=(n, n),
    )
    return mask, present.sum(), present.numel()


def sparse_attention(q, k, v, mask, scale):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)
    return _compiled(
        q,
        k,
        v,
        block_mask=mask,
        scale=scale,
        enable_gqa=True,
        # B200/head_dim256: preserve forward tiling and tune backward execution.
        kernel_options={
            "BACKEND": "TRITON",
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "bwd_num_warps": 8,
            "bwd_num_stages": 2,
        },
    )


def attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    packed_lengths = kwargs.pop("packed_lengths", None)

    kernel = validate_qsa_kernel(
        self.qsa_kernel,
        packed=packed_lengths is not None and len(packed_lengths) != 1,
    )
    if (
        past_key_values is not None
        or attention_mask is not None
        or hidden_states.shape[0] != 1
        or hidden_states.dtype != torch.bfloat16
        or self.attention_dropout != 0
        or self.indexer.index_kv_heads != 1
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError(
            "flex_qsa requires one internally packed causal BF16 token stream without cache/dropout/TF32"
        )
    events = (
        [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        if _events is not None
        else None
    )
    if events:
        events[0].record()
        record = {"events": events, "stage": 0}
        _events.append(record)

    def compute_selection():
        return (
            batched_documents(
                self.indexer, hidden_states, position_embeddings, packed_lengths
            )
            if packed_lengths is not None
            else _batched(
                self.indexer,
                hidden_states,
                position_embeddings,
                None,
                return_blocks=True,
            )
        )

    chosen, reused = select_blocks(
        self.indexer,
        hidden_states,
        lambda: qsa_blocks(self.indexer, hidden_states, compute_selection, packed_lengths),
    )
    if events:
        events[1].record()
        record["stage"] = 1
        record["selection_reused"] = reused
        record["selection_original_bytes"] = (
            chosen.numel() * chosen.element_size() if not reused else 0
        )
    if kernel != "group_sparse":
        mask, live, total = selection_metadata(
            chosen,
            self.indexer.compress_ratio,
            self.indexer.token_budget,
            packed_lengths=packed_lengths,
        )
    if events:
        events[2].record()
        record["stage"] = 2
        if kernel != "group_sparse":
            nq = (hidden_states.shape[1] + QUERY_TILE - 1) // QUERY_TILE
            record.update(live=live.detach(), total=total, causal=nq * (nq + 1) // 2)
    shape = hidden_states.shape[:-1]
    hshape = (*shape, -1, self.head_dim)
    query, gate = torch.chunk(
        self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*shape, -1)
    query = self.q_norm(query.reshape(hshape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
    cos, sin = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    def attention_core():
        if kernel == "group_sparse":
            return group_attention(query, key, value, chosen, self.scaling, packed_lengths)
        if kernel == "flex":
            return sparse_attention(query, key, value, mask, self.scaling)
        sparse = indexed_attention if kernel == "indexed_backward" else physical_attention
        return sparse(query, key, value, chosen, mask, self.scaling)

    result, core_reused, retained_bytes = retain_attention(
        self, (query, key, value), attention_core
    )
    if events:
        record.update(core_reused=core_reused, retained_bytes=retained_bytes)
    result = result.transpose(1, 2).reshape(*shape, -1).contiguous()
    result = self.o_proj(result * torch.sigmoid(gate))
    if events:
        events[3].record()
        record["stage"] = 3
    return result, None


class FlexQSAAttention(Qwen4ExpTextAttention):
    """Constructor-installed sparse-attention adapter."""

    def __init__(self, config, layer_idx, *, qsa_kernel=None, selector_policy="batched_fp32_top513_v1"):
        super().__init__(config, layer_idx)
        self.qsa_kernel = qsa_kernel
        self.indexer = BatchedQSAIndexer(config, layer_idx, selector_policy=selector_policy)
        if qsa_kernel == "group_sparse" and (self.indexer.compress_ratio != 4 or self.indexer.block_topk != 512):
            raise ValueError(
                f"qsa_kernel='group_sparse': indexer compress_ratio={self.indexer.compress_ratio}, "
                f"block_topk={self.indexer.block_topk}; the group kernel is specialized for "
                "4-token blocks and 512 selected blocks (dense prefix 2051)"
            )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        return attention_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
