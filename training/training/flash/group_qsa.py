"""Experimental exact group-packed QSA, with linear-size sparse metadata.

Online softmax follows the FLA/QSA algorithm attributed in indexed_qsa.py
(FLA 0.5.1, MIT; Songlin Yang, Yu Zhang, Zhiyuan Li and contributors).
Unlike physical_qsa, reductions pack keys and query heads into tensor-core tiles.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .group_qsa_backward import group_delta, group_dq
from .group_qsa_wave import backward_kv

from .group_qsa_metadata import GROUP, SLOTS, group_layout
from .indexed_qsa import _validate_sparse_inputs

# Internal launch parameters; group_tune.py measures alternatives before changes.
# The kernels are specialized for 4-token blocks, 512 selected blocks (dense
# prefix 2051) and eight-query groups; FlexQSAAttention rejects other indexers.
DQ_HEADS, DQ_WARPS = 12, 8
DQ_BN, DQ_STAGES = 32, 2
BACKWARD_TMA = True
DKV_WAVE_GROUPS = 512


@triton.jit
def _segment_or(a, x, b, y):
    return b, tl.where(a == b, x | y, y)


@triton.jit
def group_union(CHOSEN, STARTS, ENDS, DOCS, OFFSETS, IDS, MEMBERS, COUNTS, G: tl.constexpr = 8):
    group = tl.program_id(0)
    start = tl.load(STARTS + group)
    end = tl.minimum(tl.load(ENDS + group), start + G)
    document = tl.load(DOCS + group)
    offset = tl.load(OFFSETS + group)
    r = tl.arange(0, G * 512)
    lane, slot = r // 512, r % 512
    query = start + lane
    local = query - document
    block = tl.load(CHOSEN + query * 512 + slot, query < end, other=-1)
    block = tl.where(local < 2051, offset + slot, block)
    valid = ((query < end) & (block >= offset) & (block <= offset + local // 4))
    key = tl.sort(tl.where(valid, block * 8 + lane, 2147483647), descending=False)
    blocks = key // 8
    bits = tl.where(key != 2147483647, 1 << (key % 8), 0)
    _, combined = tl.associative_scan((blocks, bits), 0, _segment_or)
    next_block = tl.gather(blocks, tl.minimum(r + 1, G * 512 - 1), 0)
    last = (key != 2147483647) & ((r == G * 512 - 1) | (blocks != next_block))
    target = tl.cumsum(last.to(tl.int32), 0) - 1
    count = tl.sum(last.to(tl.int32), 0)
    # There are only two possible tail blocks in an aligned eight-token group.
    # Insert them after sorting 4096 selections instead of sorting 8192 slots.
    tail0 = offset + (start - document) // 4
    tail1 = tail0 + 1
    lanes = tl.arange(0, 8)
    tail_bits = tl.where((start + lanes < end) & (lanes % 4 != 3), 1 << lanes, 0)
    bits0 = tl.sum(tl.where(lanes < 4, tail_bits, 0), 0)
    bits1 = tl.sum(tl.where(lanes >= 4, tail_bits, 0), 0)
    present0 = tl.sum((last & (blocks == tail0)).to(tl.int32), 0) != 0
    present1 = tl.sum((last & (blocks == tail1)).to(tl.int32), 0) != 0
    add0, add1 = (~present0) & (bits0 != 0), (~present1) & (bits1 != 0)
    combined = tl.where(blocks == tail0, combined | bits0, combined)
    combined = tl.where(blocks == tail1, combined | bits1, combined)
    target += (add0 & (blocks == tail1)).to(tl.int32)
    tl.store(IDS + group * 4104 + target, blocks, last)
    tl.store(MEMBERS + group * 4104 + target, combined, last)
    slot0 = count - present1.to(tl.int32)
    slot1 = count + add0.to(tl.int32)
    tl.store(IDS + group * 4104 + slot0, tail0, add0)
    tl.store(MEMBERS + group * 4104 + slot0, bits0, add0)
    tl.store(IDS + group * 4104 + slot1, tail1, add1)
    tl.store(MEMBERS + group * 4104 + slot1, bits1, add1)
    tl.store(COUNTS + group, count + add0.to(tl.int32) + add1.to(tl.int32))


@triton.jit
def group_forward(
    Q, K, V, IDS, MEMBERS, COUNTS, STARTS, ENDS, KEYS, KEY_ENDS, OUT, LSE,
    N: tl.constexpr, SCALE: tl.constexpr, BM: tl.constexpr = 128,
    BN: tl.constexpr = 64, USE_TMA: tl.constexpr = False, G: tl.constexpr = 8,
):
    group, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(STARTS + group), tl.load(ENDS + group)
    m = tl.arange(0, BM)
    query, h = start + m // 12, head * 12 + m % 12
    active = (m < G * 12) & (query < end)
    d = tl.arange(0, 256)
    qp = (h[:, None] * N + query[:, None]) * 256 + d[None, :]
    if USE_TMA:
        q = Q.gather(tl.where(active, h * N + query, 24 * N), 0)
    else:
        q = tl.load(Q + qp, active[:, None], other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 256), tl.float32)
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
            rows = tl.where(valid_key, head * N + pos, 2 * N)
            k = K.gather(rows, 0)
            v = V.gather(rows, 0)
        else:
            k = tl.load(K + kp, valid_key[:, None], other=0)
            v = tl.load(V + kp, valid_key[:, None], other=0)
        valid = (active[:, None] & valid_key[None, :] & (pos[None, :] <= query[:, None])
                 & (((bits[None, :] >> (m[:, None] // 12)) & 1) != 0))
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(new_max == -float("inf"), 0., new_max)
        p = tl.exp2(score - safe_max[:, None])
        alpha = tl.exp2(maximum - safe_max)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        total = total * alpha + tl.sum(p, 1)
        maximum = new_max
    tl.store(OUT + qp, acc / tl.where(total > 0, total, 1.)[:, None], active[:, None])
    tl.store(LSE + h * N + query, maximum + tl.log2(total), active)


def prepare_groups(chosen, packed_lengths=None):
    if not chosen.is_cuda or chosen.dtype != torch.int32 or chosen.ndim != 2 or chosen.shape[1] != 512:
        raise ValueError("group QSA prepare_groups: requires CUDA [tokens, 512] int32 selections")
    chosen = chosen.contiguous()
    layout = group_layout(len(chosen), chosen.device, packed_lengths)
    shape = (len(layout.starts), SLOTS)
    ids = torch.empty(shape, device=chosen.device, dtype=torch.int32)
    members = torch.empty_like(ids)
    counts = torch.empty(shape[0], device=chosen.device, dtype=torch.int32)
    group_union[(shape[0],)](
        chosen, layout.starts, layout.ends, layout.document_starts,
        layout.block_offsets, ids, members, counts, G=GROUP, num_warps=8,
    )
    return layout, ids, members, counts


def forward(q, k, v, chosen, scale, packed_lengths=None):
    """Forward oracle/benchmark entry point; LSE uses base-2 logarithms."""
    with torch.no_grad():
        return attention(q, k, v, chosen, scale, packed_lengths, return_lse=True)



class _GroupAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, chosen, scale, lengths):
        _validate_sparse_inputs(q, k, v, chosen, dtypes=(torch.bfloat16, torch.float16), op="group QSA")
        q, k, v, chosen = (x.contiguous() for x in (q, k, v, chosen))
        layout, ids, members, counts = prepare_groups(chosen, lengths)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[1:3], device=q.device, dtype=torch.float32)
        use_tma = torch.cuda.get_device_capability(q.device)[0] == 10
        inputs = (q, k, v)
        if use_tma:
            inputs = tuple(TensorDescriptor(x, [x.shape[1] * q.shape[2], 256],
                                            [256, 1], [1, 256]) for x in inputs)
        group_forward[(len(layout.starts), 2)](
            *inputs, ids, members, counts, layout.starts, layout.ends,
            layout.key_starts, layout.key_ends, out, lse, N=q.shape[2], SCALE=scale,
            USE_TMA=use_tma, G=GROUP, BM=triton.next_power_of_2(GROUP * 12),
            num_warps=8, num_stages=2,
        )
        ctx.save_for_backward(
            q, k, v, out, lse, ids, members, counts,
            layout.starts, layout.ends, layout.key_starts, layout.key_ends,
        )
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _grad_lse):
        q, k, v, out, lse, ids, members, counts, starts, ends, keys, key_ends = ctx.saved_tensors
        grad = grad.contiguous()
        n = q.shape[2]
        delta = torch.empty_like(lse)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        group_delta[(triton.cdiv(24 * n, 16),)](out, grad, delta, N=n)
        inputs = (q, k, v, grad)
        use_tma = BACKWARD_TMA and torch.cuda.get_device_capability(q.device)[0] == 10
        if use_tma:
            inputs = tuple(TensorDescriptor(x, [x.shape[1] * n, 256],
                                            [256, 1], [1, 256]) for x in inputs)
        group_dq[(len(starts), 2 * (12 // DQ_HEADS))](
            *inputs, lse, delta, ids, members, counts, starts, ends, keys,
            key_ends, dq, N=n, SCALE=ctx.scale, HEADS=DQ_HEADS,
            BM=triton.next_power_of_2(GROUP * DQ_HEADS), USE_TMA=use_tma, G=GROUP,
            BN=DQ_BN if use_tma else 64, num_warps=DQ_WARPS, num_stages=DQ_STAGES if use_tma else 1,
        )
        backward_kv(q, k, v, grad, lse, delta, ids, members, counts, starts, ends,
                    keys, key_ends, dk, dv, ctx.scale, GROUP, DKV_WAVE_GROUPS)
        return dq, dk, dv, None, None, None


def attention(q, k, v, chosen, scale, packed_lengths=None, *, return_lse=False):
    """Exact QSA with deterministic gradients and optional diagnostic base-2 LSE."""
    out, lse = _GroupAttention.apply(q, k, v, chosen, scale, packed_lengths)
    return (out, lse) if return_lse else out
