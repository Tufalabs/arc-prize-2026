"""QSA backward preserving physical reduction order and Flex masking arithmetic.

Derived from the FLA/QSA kernels attributed in training.flash.indexed_qsa (MIT).
Each program processes selected physical 64-token tiles in increasing order,
retaining masked holes so tensor-core reductions match the reference. Masking
before exp2 also preserves the reference floating-point contraction boundaries.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def physical_dkv(
    Q,
    K,
    V,
    L,
    DO,
    DELTA,
    IDS,
    OFFSETS,
    DK,
    DV,
    N,
    D: tl.constexpr,
    G: tl.constexpr,
    SCALE: tl.constexpr,
):
    BM: tl.constexpr = 16
    BT: tl.constexpr = 64
    block, h = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BM)
    pos = block * 4 + rows
    d = tl.arange(0, 256)
    kp = (h * N + pos[:, None]) * D + d[None, :]
    km = (rows[:, None] < 4) & (pos[:, None] < N)
    k = tl.load(K + kp, mask=km, other=0)
    v = tl.load(V + kp, mask=km, other=0)
    begin = tl.load(OFFSETS + block)
    end = tl.load(OFFSETS + block + 1)
    dk = tl.zeros((BM, 256), tl.float32)
    dv = tl.zeros((BM, 256), tl.float32)
    for g in range(G):
        head = h * G + g
        start = begin
        while start < end:
            j = start + tl.arange(0, BT)
            selected = tl.load(IDS + j, mask=j < end, other=N + BT)
            base = tl.load(IDS + start) // BT * BT
            in_tile = (j < end) & (selected < base + BT)
            bits = tl.sum(
                tl.where(
                    in_tile,
                    tl.full((BT,), 1, tl.uint64) << (selected % BT).to(tl.uint64),
                    0,
                ),
                0,
            )
            consumed = tl.sum(in_tile.to(tl.int32), 0)
            t = base + tl.arange(0, BT)
            active = ((bits >> tl.arange(0, BT).to(tl.uint64)) & 1) != 0
            ptr = (head * N + t[:, None]) * D + d[None, :]
            q = tl.load(Q + ptr, mask=(active[:, None] & (t[:, None] < N)), other=0)
            do = tl.load(DO + ptr, mask=(active[:, None] & (t[:, None] < N)), other=0)
            lse = tl.load(L + head * N + t, mask=active & (t < N), other=0)
            delta = tl.load(DELTA + head * N + t, mask=active & (t < N), other=0)
            scores = tl.dot(k, tl.trans(q)) * SCALE
            valid = (
                (rows[:, None] < 4)
                & (pos[:, None] < N)
                & (pos[:, None] <= t[None, :])
                & active[None, :]
                & (t[None, :] < N)
            )
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.exp2(scores * 1.4426950408889634 - lse[None, :])
            p = tl.where(valid, p, 0)
            dv += tl.dot(p.to(do.dtype), do)
            dp = tl.dot(v, tl.trans(do))
            ds = (p * (dp - delta[None, :])).to(q.dtype)
            dk += tl.dot(ds, q)
            start += consumed
    tl.store(DK + kp, dk * SCALE, mask=km)
    tl.store(DV + kp, dv, mask=km)


@triton.jit(do_not_specialize=["N"])
def physical_dq(
    Q,
    K,
    V,
    INDICES,
    L,
    DO,
    DELTA,
    DQ,
    N,
    D: tl.constexpr,
    G: tl.constexpr,
    SCALE: tl.constexpr,
):
    BM: tl.constexpr = 16
    BT: tl.constexpr = 64
    t, h = tl.program_id(0), tl.program_id(1)
    g = tl.arange(0, BM)
    d = tl.arange(0, 256)
    heads = h * G + g
    ptr = (heads[:, None] * N + t) * D + d[None, :]
    q = tl.load(Q + ptr, mask=g[:, None] < G, other=0)
    do = tl.load(DO + ptr, mask=g[:, None] < G, other=0)
    lse = tl.load(L + heads * N + t, mask=g < G, other=0)
    delta = tl.load(DELTA + heads * N + t, mask=g < G, other=0)
    dense = t < 2051
    count = tl.where(dense, (t + 4) // 4, 512 + ((t + 1) % 4 != 0).to(tl.int32))
    cursor = 0
    dq = tl.zeros((BM, 256), tl.float32)
    while cursor < count:
        ids = cursor + tl.arange(0, BT // 4)
        blocks = tl.load(INDICES + t * 512 + ids, mask=ids < 512, other=0)
        blocks = tl.where(ids == 512, (t + 1) // 4, blocks)
        blocks = tl.where(dense, ids, blocks)
        first = tl.min(tl.where(ids < count, blocks, 2147483647), 0)
        base = first // (BT // 4) * (BT // 4)
        in_tile = (ids < count) & (blocks < base + BT // 4)
        bits = tl.sum(tl.where(in_tile, 1 << (blocks % (BT // 4)), 0), 0)
        consumed = tl.sum(in_tile.to(tl.int32), 0)
        pos = base * 4 + tl.arange(0, BT)
        valid = (((bits >> ((pos // 4) % (BT // 4))) & 1) != 0) & (pos <= t) & (pos < N)
        kp = (h * N + pos[:, None]) * D + d[None, :]
        k = tl.load(K + kp, mask=valid[:, None], other=0)
        v = tl.load(V + kp, mask=valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(valid[None, :] & (g[:, None] < G), scores, -float("inf"))
        p = tl.exp2(scores * 1.4426950408889634 - lse[:, None])
        p = tl.where(valid[None, :] & (g[:, None] < G), p, 0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(q.dtype)
        dq += tl.dot(ds, k)
        cursor += consumed
    tl.store(DQ + ptr, dq * SCALE, mask=g[:, None] < G)
