"""Experimental physical32 sparse forward, derived from the attributed FLA/QSA
kernel in training.flash.indexed_qsa; same MIT attribution/license applies.
Default QSA forward with the physical-reduction backward shared by indexed QSA.
"""

import torch
import triton
import triton.language as tl

from .indexed_qsa import _HybridAttention, _validate_sparse_inputs


@triton.jit(do_not_specialize=["N", "WORDS"])
def physical_forward(
    Q,
    K,
    V,
    INDICES,
    COUNTS,
    BITS,
    OUTPUT,
    L,
    N,
    _HQ: tl.constexpr,
    _HK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    SCALE: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    WORDS,
    FP32_PV: tl.constexpr,
    SEPARATE_SCALE: tl.constexpr,
    NATURAL_EXP: tl.constexpr,
):
    t, h = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, BG)
    d = tl.arange(0, BD)
    heads = h * G + g
    q = tl.load(
        Q + (heads[:, None] * N + t) * D + d[None, :],
        mask=(g[:, None] < G) & (d[None, :] < D),
        other=0,
    )
    m = tl.full((BG,), -float("inf"), tl.float32)
    z = tl.zeros((BG,), tl.float32)
    acc = tl.zeros((BG, BD), tl.float32)
    count = tl.load(COUNTS + t)
    for start in range(count):
        tile = tl.load(INDICES + t * (S + 1) + start)
        pos = tile * BT + tl.arange(0, BT)
        word = tl.load(BITS + t * WORDS + pos // 128, mask=pos < N, other=0)
        selected = ((word >> ((pos // 4) % 32)) & 1) != 0
        valid = (
            (pos < N)
            & (pos <= t)
            & ((t < 2051) | selected | (pos >= ((t + 1) // 4) * 4))
        )
        k = tl.load(
            K + (h * N + pos[None, :]) * D + d[:, None],
            mask=valid[None, :] & (d[:, None] < D),
            other=0,
        )
        v = tl.load(
            V + (h * N + pos[:, None]) * D + d[None, :],
            mask=valid[:, None] & (d[None, :] < D),
            other=0,
        )
        score = tl.dot(q, k)
        if NATURAL_EXP:
            score = score * SCALE
        elif SEPARATE_SCALE:
            score = (score * SCALE) * 1.4426950408889634
        else:
            score = score * (SCALE * 1.4426950408889634)
        score = tl.where(valid[None, :], score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, 1))
        if NATURAL_EXP:
            p = tl.exp(score - new_m[:, None])
            alpha = tl.exp(m - new_m)
        else:
            p = tl.exp2(score - new_m[:, None])
            alpha = tl.exp2(m - new_m)
        if FP32_PV:
            acc = acc * alpha[:, None] + tl.dot(
                p, v.to(tl.float32), input_precision="ieee"
            )
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        z = z * alpha + tl.sum(p, 1)
        m = new_m
    out = acc / z[:, None]
    tl.store(
        OUTPUT + (heads[:, None] * N + t) * D + d[None, :],
        out,
        mask=(g[:, None] < G) & (d[None, :] < D),
    )
    if NATURAL_EXP:
        norm = (m + tl.log(z)) * 1.4426950408889634
    else:
        norm = m + tl.log2(z)
    tl.store(L + heads * N + t, norm, mask=g < G)


def prepare_physical(chosen, bt):
    n = len(chosen)
    rows = torch.arange(n, device=chosen.device)
    ids = torch.cat(
        [chosen.long() // (bt // 4), (((rows + 1) // 4) * 4 // bt)[:, None]], 1
    )
    ids = ids.sort(-1).values
    unique = torch.cat(
        [
            torch.ones((n, 1), device=chosen.device, dtype=torch.bool),
            ids[:, 1:] != ids[:, :-1],
        ],
        1,
    )
    valid = unique & (ids >= 0) & (ids * bt <= rows[:, None])
    ids = torch.where(valid, ids, n + 1).sort(-1).values
    slots = torch.arange(513, device=chosen.device)[None, :]
    ids = torch.where(
        rows[:, None] < 2051,
        torch.where(slots * bt <= rows[:, None], slots, n + 1),
        ids,
    )
    counts = (ids <= n).sum(-1).to(torch.int32).contiguous()
    words = (n + 127) // 128
    bits = torch.zeros((n, words), device=chosen.device, dtype=torch.int32)
    c = chosen.long()
    value = (torch.ones_like(c) << (c.clamp_min(0) % 32)).to(torch.int32)
    bits.scatter_add_(1, c.clamp_min(0) // 32, torch.where(c >= 0, value, 0))
    return ids.to(torch.int32).contiguous(), counts, bits


class _PhysicalAttention(_HybridAttention):
    @staticmethod
    def forward(ctx, q, k, v, indices, mask, scale):
        q, k, v, indices = [x.contiguous() for x in (q, k, v, indices)]
        for x in (q, k, v):
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        torch._dynamo.mark_static(indices, -1)
        n = q.shape[2]
        ids, counts, bits = prepare_physical(indices, 32)
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[1], n), device=q.device, dtype=torch.float32)
        physical_forward[(n, k.shape[1])](
            q,
            k,
            v,
            ids,
            counts,
            bits,
            out,
            lse,
            N=n,
            _HQ=24,
            _HK=2,
            D=256,
            S=512,
            SCALE=scale,
            G=12,
            BG=16,
            BD=256,
            BT=32,
            WORDS=bits.shape[1],
            FP32_PV=False,
            SEPARATE_SCALE=False,
            NATURAL_EXP=False,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, indices, out, lse)
        ctx.scale = scale
        return out


@torch.compile(dynamic=True, fullgraph=True)
def _compiled(q, k, v, indices, mask, scale):
    return _PhysicalAttention.apply(q, k, v, indices, mask, scale)


def hybrid_attention(q, k, v, indices, mask, scale):
    _validate_sparse_inputs(q, k, v, indices)
    with torch._dynamo.config.patch(specialize_float=True):
        return _compiled(q, k, v, indices, mask, scale)
