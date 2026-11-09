"""Experimental bounded partial-gradient waves with deterministic reduction.

The tensor-core pass packs each query group's sparse keys. A second pass owns
one physical key block and reduces that wave's partials in increasing group
order. Waves run sequentially; no floating-point atomic operations are used.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .group_qsa_metadata import reverse_groups
from .dtype_contract import low_precision_inputs

PARTIAL_DTYPE = torch.float16
PARTIAL_WARPS = 8


@triton.jit
def group_wave_partials(
    Q, K, V, DO, LSE, DELTA, IDS, MEMBERS, COUNTS, STARTS, ENDS, KEYS, KEY_ENDS,
    PK, PV, SK, SV, WAVE_START, GROUPS: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr,
    G: tl.constexpr = 8, BM: tl.constexpr = 128, USE_TMA: tl.constexpr = True,
    SCALED: tl.constexpr = False,
):
    local_group, head = tl.program_id(0), tl.program_id(1)
    group = WAVE_START + local_group
    if group < GROUPS:
        count = tl.load(COUNTS + group)
        start, end = tl.load(STARTS + group), tl.load(ENDS + group)
        m, r, d = tl.arange(0, BM), tl.arange(0, 64), tl.arange(0, 256)
        query, h = start + m // 12, head * 12 + m % 12
        active = (m < G * 12) & (query < end)
        if USE_TMA:
            qrows = tl.where(active, h * N + query, 24 * N)
            q, do = Q.gather(qrows, 0), DO.gather(qrows, 0)
        else:
            qp = (h[:, None] * N + query[:, None]) * 256 + d[None, :]
            q, do = tl.load(Q + qp, active[:, None], 0), tl.load(DO + qp, active[:, None], 0)
        lse = tl.load(LSE + h * N + query, active, 0)
        delta = tl.load(DELTA + h * N + query, active, 0)
        for chunk in range(tl.cdiv(count, 16)):
            slot = chunk * 16 + r // 4
            block = tl.load(IDS + group * 4104 + slot, slot < count, other=0)
            bits = tl.load(MEMBERS + group * 4104 + slot, slot < count, other=0)
            pos = tl.load(KEYS + block) + r % 4
            kend = tl.load(KEY_ENDS + block)
            valid_key = (slot < count) & (pos < kend)
            if USE_TMA:
                krows = tl.where(valid_key, head * N + pos, 2 * N)
                k, v = K.gather(krows, 0), V.gather(krows, 0)
            else:
                kp = (head * N + pos[:, None]) * 256 + d[None, :]
                k, v = tl.load(K + kp, valid_key[:, None], 0), tl.load(V + kp, valid_key[:, None], 0)
            valid = (valid_key[:, None] & active[None, :] & (pos[:, None] <= query[None, :])
                     & (((bits[:, None] >> (m[None, :] // 12)) & 1) != 0))
            score = tl.dot(k, tl.trans(q)) * (SCALE * 1.4426950408889634)
            p = tl.exp2(tl.where(valid, score - lse[None, :], -float("inf")))
            dp = tl.dot(v, tl.trans(do))
            ds = (p * (dp - delta[None, :])).to(q.dtype)
            dk, dv = tl.dot(ds, q), tl.dot(p.to(do.dtype), do)
            target = (((local_group * 2 + head) * 16416 + chunk * 64 + r[:, None]).to(tl.int64) * 256
                      + d[None, :])
            if SCALED:
                # Power-of-two row scales preserve FP16 range without clipping.
                # Map each maximum into [4096, 8192), leaving overflow margin.
                # Only elements below 2^-26 of their row maximum turn subnormal and
                # lose relative precision, so FP16 partials stay finer than BF16 output.
                ek = (tl.max(tl.abs(dk), 1).to(tl.int32, bitcast=True) >> 23) & 255
                ev = (tl.max(tl.abs(dv), 1).to(tl.int32, bitcast=True) >> 23) & 255
                sk = (tl.maximum(ek - 12, 1) << 23).to(tl.float32, bitcast=True)
                sv = (tl.maximum(ev - 12, 1) << 23).to(tl.float32, bitcast=True)
                row = ((local_group * 2 + head) * 16416 + chunk * 64 + r).to(tl.int64)
                tl.store(SK + row, sk, slot < count)
                tl.store(SV + row, sv, slot < count)
                dk, dv = dk / sk[:, None], dv / sv[:, None]
            tl.store(PK + target, dk, (slot < count)[:, None])
            tl.store(PV + target, dv, (slot < count)[:, None])


@triton.jit
def group_wave_reduce(
    PK, PV, SK, SV, ORDER, OFFSETS, KEYS, KEY_ENDS, DK, DV, WAVE_START,
    WAVE: tl.constexpr, N: tl.constexpr, SCALED: tl.constexpr = False,
):
    block, head = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(OFFSETS + block), tl.load(OFFSETS + block + 1)
    lo, hi = begin, end
    while lo < hi:
        middle = (lo + hi) // 2
        group = tl.load(ORDER + middle) // 4104
        before = group < WAVE_START
        lo, hi = tl.where(before, middle + 1, lo), tl.where(before, hi, middle)
    first = lo
    hi = end
    while lo < hi:
        middle = (lo + hi) // 2
        group = tl.load(ORDER + middle) // 4104
        before = group < WAVE_START + WAVE
        lo, hi = tl.where(before, middle + 1, lo), tl.where(before, hi, middle)
    if first < lo:
        r, d = tl.arange(0, 4), tl.arange(0, 256)
        pos = tl.load(KEYS + block) + r
        valid = pos < tl.load(KEY_ENDS + block)
        target = (head * N + pos[:, None]) * 256 + d[None, :]
        dk, dv = tl.load(DK + target, valid[:, None], 0), tl.load(DV + target, valid[:, None], 0)
        for edge_index in range(first, lo):
            edge = tl.load(ORDER + edge_index)
            local_group, slot = edge // 4104 - WAVE_START, edge % 4104
            source = (((local_group * 2 + head) * 16416 + slot * 4 + r[:, None]).to(tl.int64) * 256
                      + d[None, :])
            pk, pv = tl.load(PK + source).to(tl.float32), tl.load(PV + source).to(tl.float32)
            if SCALED:
                row = ((local_group * 2 + head) * 16416 + slot * 4 + r).to(tl.int64)
                pk *= tl.load(SK + row)[:, None]
                pv *= tl.load(SV + row)[:, None]
            dk += pk
            dv += pv
        tl.store(DK + target, dk, valid[:, None])
        tl.store(DV + target, dv, valid[:, None])


@triton.jit
def group_wave_finish(AK, AV, DK, DV, SIZE: tl.constexpr, SCALE: tl.constexpr):
    x = tl.program_id(0) * 512 + tl.arange(0, 512)
    k = tl.load(AK + x, x < SIZE, 0)
    v = tl.load(AV + x, x < SIZE, 0)
    tl.store(DK + x, k * SCALE, x < SIZE)
    tl.store(DV + x, v, x < SIZE)


def backward_kv(q, k, v, grad, lse, delta, ids, members, counts, starts, ends,
                keys, key_ends, dk, dv, scale, group_tokens, wave):
    low_precision_inputs("group QSA backward_kv", q, k, v, grad, dk, dv, same_dtype=True)
    n, groups = q.shape[2], len(starts)
    wave = min(wave, groups)
    order, offsets = reverse_groups(ids, counts, len(keys))
    # FP16 activations need FP32 partials: another FP16 rounding can exceed
    # the eager FP16 error budget, even when row scaling prevents overflow.
    partial_dtype = torch.float32 if q.dtype == torch.float16 else PARTIAL_DTYPE
    partial_k = torch.empty((wave, 2, 16416, 256), device=q.device, dtype=partial_dtype)
    partial_v = torch.empty_like(partial_k)
    scaled = partial_dtype == torch.float16
    scale_k = torch.empty(partial_k.shape[:-1], device=q.device, dtype=torch.float32) if scaled else partial_k
    scale_v = torch.empty_like(scale_k) if scaled else partial_v
    accum_k = torch.zeros_like(k, dtype=torch.float32)
    accum_v = torch.zeros_like(v, dtype=torch.float32)
    use_tma = torch.cuda.get_device_capability(q.device)[0] == 10
    inputs = (q, k, v, grad)
    if use_tma:
        inputs = tuple(TensorDescriptor(x, [x.shape[1] * n, 256],
                                        [256, 1], [1, 256]) for x in inputs)
    for start in range(0, groups, wave):
        group_wave_partials[(min(wave, groups - start), 2)](
            *inputs, lse, delta, ids, members, counts, starts, ends, keys, key_ends,
            partial_k, partial_v, scale_k, scale_v, start, GROUPS=groups, N=n, SCALE=scale,
            G=group_tokens, BM=triton.next_power_of_2(group_tokens * 12), USE_TMA=use_tma, SCALED=scaled,
            num_warps=PARTIAL_WARPS, num_stages=1,
        )
        group_wave_reduce[(len(keys), 2)](
            partial_k, partial_v, scale_k, scale_v, order, offsets, keys, key_ends, accum_k, accum_v,
            start, WAVE=wave, N=n, SCALED=scaled, num_warps=4,
        )
    group_wave_finish[(triton.cdiv(k.numel(), 512),)](
        accum_k, accum_v, dk, dv, SIZE=k.numel(), SCALE=scale,
    )
