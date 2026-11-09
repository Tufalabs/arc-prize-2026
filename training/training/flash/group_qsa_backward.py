"""Deterministic group-packed QSA dQ and delta kernels; one writer per element.

Derived from the FLA/QSA algorithms attributed in indexed_qsa.py (MIT).
dK/dV live in group_qsa_wave.py; floating-point atomics are absent.
"""

import triton
import triton.language as tl


@triton.jit
def group_delta(OUT, DO, DELTA, N: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    d = tl.arange(0, 256)
    o = tl.load(OUT + row[:, None] * 256 + d[None, :], row[:, None] < 24 * N, 0)
    do = tl.load(DO + row[:, None] * 256 + d[None, :], row[:, None] < 24 * N, 0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(DELTA + row, delta, row < 24 * N)


@triton.jit
def group_dq(
    Q, K, V, DO, LSE, DELTA, IDS, MEMBERS, COUNTS, STARTS, ENDS, KEYS, KEY_ENDS, DQ,
    N: tl.constexpr, SCALE: tl.constexpr, BM: tl.constexpr = 128,
    BN: tl.constexpr = 64, HEADS: tl.constexpr = 12, USE_TMA: tl.constexpr = False,
    G: tl.constexpr = 8,
):
    group, part = tl.program_id(0), tl.program_id(1)
    head = part // (12 // HEADS)
    start, end = tl.load(STARTS + group), tl.load(ENDS + group)
    m = tl.arange(0, BM)
    query = start + m // HEADS
    h = head * 12 + (part % (12 // HEADS)) * HEADS + m % HEADS
    active = (m < G * HEADS) & (query < end)
    d = tl.arange(0, 256)
    qp = (h[:, None] * N + query[:, None]) * 256 + d[None, :]
    if USE_TMA:
        qrows = tl.where(active, h * N + query, 24 * N)
        q = Q.gather(qrows, 0)
        do = DO.gather(qrows, 0)
    else:
        q = tl.load(Q + qp, active[:, None], other=0)
        do = tl.load(DO + qp, active[:, None], other=0)
    lse = tl.load(LSE + h * N + query, active, other=0)
    delta = tl.load(DELTA + h * N + query, active, other=0)
    dq = tl.zeros((BM, 256), tl.float32)
    count = tl.load(COUNTS + group)
    for i in range(tl.cdiv(count * 4, BN)):
        slot = i * (BN // 4) + tl.arange(0, BN) // 4
        block = tl.load(IDS + group * 4104 + slot, slot < count, other=0)
        bits = tl.load(MEMBERS + group * 4104 + slot, slot < count, other=0)
        pos = tl.load(KEYS + block) + tl.arange(0, BN) % 4
        kend = tl.load(KEY_ENDS + block)
        valid_key = (slot < count) & (pos < kend)
        kp = (head * N + pos[:, None]) * 256 + d[None, :]
        if USE_TMA:
            krows = tl.where(valid_key, head * N + pos, 2 * N)
            k = K.gather(krows, 0)
            v = V.gather(krows, 0)
        else:
            k = tl.load(K + kp, valid_key[:, None], other=0)
            v = tl.load(V + kp, valid_key[:, None], other=0)
        valid = (active[:, None] & valid_key[None, :] & (pos[None, :] <= query[:, None])
                 & (((bits[None, :] >> (m[:, None] // HEADS)) & 1) != 0))
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        p = tl.exp2(tl.where(valid, score - lse[:, None], -float("inf")))
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None])).to(q.dtype)
        dq += tl.dot(ds, k)
    tl.store(DQ + qp, dq * SCALE, active[:, None])
