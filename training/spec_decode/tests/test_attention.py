"""The training attention against Transformers' dense QSA, on a toy config.

Pass 0 must equal ``Qwen4ExpTextAttention.forward`` (its per-query indexer
loop and dense masks). A draft step must equal a brute-force attention over
its anchor's selected pass-0 tokens plus the draft keys.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from drafter import attention as qsa

# Uncompiled FlexAttention is the CPU reference path; it says so.
pytestmark = pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")


def _toy():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        layer_types=["full_attention"],
        num_hidden_layers=1,
        ple_layer_ids=[],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    attn = m.Qwen4ExpTextAttention(config, 0).to(torch.float32)
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.2)
    rotary = m.Qwen4ExpTextRotaryEmbedding(config)
    return m, config, attn, rotary


def _projections(attn, x, cos, sin):
    n = x.shape[0]
    query, gate = torch.chunk(attn.q_proj(x).view(n, -1, 2 * attn.head_dim), 2, dim=-1)
    query = qsa.rope(attn.q_norm(query), cos, sin)
    key = qsa.rope(attn.k_norm(attn.k_proj(x).view(n, -1, attn.head_dim)), cos, sin)
    value = attn.v_proj(x).view(n, -1, attn.head_dim)
    return query, gate, key, value


@torch.no_grad()
def test_pass_zero_matches_transformers():
    m, config, attn, rotary = _toy()
    n = 45
    x = torch.randn(1, n, config.hidden_size)
    positions = torch.arange(n)[None]
    cos, sin = rotary(x, positions)
    mask = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    reference, _ = attn(x, (cos, sin), attention_mask=mask)

    cos, sin = cos[0], sin[0]
    hidden = x[0]
    query, gate, key, value = _projections(attn, hidden, cos, sin)
    index_q, raw = qsa.indexer_projections(attn.indexer, hidden, cos, sin)
    blocks = qsa.block_keys(attn.indexer, raw, cos, sin)
    limits = torch.arange(n)
    chosen = qsa.select_blocks(attn.indexer, index_q, blocks, limits)
    block_mask = qsa.selection_mask(chosen, limits, n, 4, 8)
    out, lse = qsa.sparse_attention(
        query.transpose(0, 1)[None], key.transpose(0, 1)[None], value.transpose(0, 1)[None], block_mask, attn.scaling, compiled=False
    )
    out = out[0].transpose(0, 1).reshape(n, -1)
    ours = attn.o_proj(out * torch.sigmoid(gate.reshape(n, -1)))
    torch.testing.assert_close(ours, reference[0], atol=2e-5, rtol=2e-5)

    # The log-sum-exp is natural-log, over the same tokens.
    scores = torch.einsum("qhd,khd->hqk", query, key.repeat_interleave(2, dim=1)) * attn.scaling
    allowed = torch.zeros(n, n, dtype=torch.bool)
    for q in range(n):
        allowed[q] = _visible(chosen[q], q, n)
    expected = scores.masked_fill(~allowed[None], -torch.inf).logsumexp(-1)
    torch.testing.assert_close(lse[0], expected, atol=2e-5, rtol=2e-5)


def _visible(chosen_row, limit, n, ratio=4, budget=8):
    """Tokens a query with this context limit attends: brute force."""
    allowed = torch.zeros(n, dtype=torch.bool)
    if limit < budget + ratio - 1:
        allowed[: limit + 1] = True
        return allowed
    for block in chosen_row.tolist():
        if block >= 0:
            allowed[block * ratio : block * ratio + ratio] = True
    allowed[((limit + 1) // ratio) * ratio : limit + 1] = True
    return allowed


@torch.no_grad()
def test_draft_step_matches_brute_force():
    m, config, attn, rotary = _toy()
    n0, step = 45, 2
    torch.manual_seed(1)
    hidden0 = torch.randn(n0, config.hidden_size)
    cos0, sin0 = (t[0] for t in rotary(hidden0[None], torch.arange(n0)[None]))
    _, _, key0, value0 = _projections(attn, hidden0, cos0, sin0)
    _, raw0 = qsa.indexer_projections(attn.indexer, hidden0, cos0, sin0)
    blocks = qsa.block_keys(attn.indexer, raw0, cos0, sin0)

    rows = n0 - step
    anchors = torch.arange(rows)
    hidden = torch.randn(rows, config.hidden_size)
    cos, sin = (t[0] for t in rotary(hidden[None], (anchors + step)[None]))
    query, _, key, value = _projections(attn, hidden, cos, sin)
    earlier = [(torch.randn(rows + 1, 2, attn.head_dim), torch.randn(rows + 1, 2, attn.head_dim))]
    index_q, _ = qsa.indexer_projections(attn.indexer, hidden, cos, sin)
    chosen = qsa.select_blocks(attn.indexer, index_q, blocks, anchors)
    block_mask = qsa.selection_mask(chosen, anchors, n0, 4, 8)
    out, lse = qsa.sparse_attention(
        query.transpose(0, 1)[None], key0.transpose(0, 1)[None], value0.transpose(0, 1)[None], block_mask, attn.scaling, compiled=False
    )
    diagonal = [(k[:rows], v[:rows]) for k, v in earlier] + [(key, value)]
    merged = qsa.merge_diagonal(out[0].transpose(0, 1), lse[0].transpose(0, 1), query, diagonal, attn.scaling)

    for t in range(rows):
        allowed = _visible(chosen[t], t, n0)
        keys = torch.cat([key0[allowed]] + [k[t : t + 1] for k, _ in diagonal]).repeat_interleave(2, dim=1)
        values = torch.cat([value0[allowed]] + [v[t : t + 1] for _, v in diagonal]).repeat_interleave(2, dim=1)
        weights = F.softmax(torch.einsum("hd,khd->hk", query[t], keys) * attn.scaling, dim=-1)
        expected = torch.einsum("hk,khd->hd", weights, values)
        torch.testing.assert_close(merged[t], expected, atol=2e-5, rtol=2e-5)
    # Selection scored against pass-0 blocks complete at the anchor only.
    for t in range(rows):
        ids = chosen[t][chosen[t] >= 0]
        assert bool((ids < (t + 1) // 4).all())
