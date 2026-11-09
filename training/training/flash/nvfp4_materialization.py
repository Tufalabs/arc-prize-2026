"""Fused NVFP4 expansion into existing BF16 expert execution layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand(
    P,
    S,
    OUTER,
    Y,
    ERROR,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    OUTER_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < N
    byte = tl.load(P + x // 2, mask=valid, other=0).to(tl.int32)
    code = (byte >> ((x % 2) * 4)) & 15
    mag = code & 7
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    # Unary negation can lower to 0 - value and lose E2M1's negative zero.
    # Set the IEEE sign bit directly before either scale multiplication.
    sign = (code & 8).to(tl.uint32) << 28
    value = (value.to(tl.uint32, bitcast=True) ^ sign).to(tl.float32, bitcast=True)
    scale = tl.load(S + x // 16, mask=valid, other=0.0).to(tl.float32)
    expert = x // (ROWS * K)
    outer = tl.load(
        OUTER + tl.where(OUTER_SCALAR, 0, expert), mask=valid, other=0.0
    ).to(tl.float32)
    value = ((value * scale) * outer).to(tl.bfloat16)
    row = (x // K) % ROWS
    target = (expert * OUT_ROWS + ROW_OFFSET + row) * K + x % K
    tl.store(Y + target, value, mask=valid)
    bad = valid & (
        (value.to(tl.float32) != value.to(tl.float32))
        | (tl.abs(value.to(tl.float32)) == float("inf"))
    )
    if tl.sum(bad.to(tl.int32), 0) > 0:
        tl.atomic_or(ERROR, 1)


def expand_into(packed, scale, outer, output, error, *, row_offset=0):
    if packed.device.type != "cuda" or packed.dtype != torch.uint8:
        raise ValueError("Fused NVFP4 expansion requires CUDA packed uint8")
    if packed.ndim != 3 or not all(
        x.is_contiguous() for x in (packed, scale, outer, output)
    ):
        raise ValueError("Fused NVFP4 expansion requires contiguous expert matrices")
    e, rows, half_k = packed.shape
    k = half_k * 2
    if scale.shape != (e, rows, k // 16) or k % 16:
        raise ValueError("NVFP4 packed/scale shapes do not agree")
    if outer.numel() not in (1, e) or output.shape[0] != e or output.shape[2] != k:
        raise ValueError("Fused NVFP4 outer scales or output shape do not agree")
    if (
        output.dtype != torch.bfloat16
        or row_offset < 0
        or row_offset + rows > output.shape[1]
    ):
        raise ValueError("Fused NVFP4 output rows do not agree")
    if any(x.device != packed.device for x in (scale, outer, output, error)):
        raise ValueError("Fused NVFP4 tensors must share one device")
    _expand[(triton.cdiv(e * rows * k, 1024),)](
        packed,
        scale,
        outer,
        output,
        error,
        N=e * rows * k,
        K=k,
        ROWS=rows,
        OUT_ROWS=output.shape[1],
        ROW_OFFSET=row_offset,
        OUTER_SCALAR=outer.numel() == 1,
        BLOCK=1024,
        enable_fp_fusion=False,
    )


def materialize_fused(module, expert_slice=slice(None)):
    """Write gate/up directly into combined storage; check all projections once."""
    packed = module.gate_packed[expert_slice]
    down_packed = module.down_packed[expert_slice]
    e, rows, half_k = packed.shape
    gate_up = torch.empty(
        (e, rows * 2, half_k * 2),
        device=packed.device,
        dtype=torch.bfloat16,
    )
    down = torch.empty(
        (*down_packed.shape[:-1], down_packed.shape[-1] * 2),
        device=gate_up.device,
        dtype=torch.bfloat16,
    )
    error = torch.zeros((), device=gate_up.device, dtype=torch.int32)
    for name, target, offset in (
        ("gate", gate_up, 0),
        ("up", gate_up, rows),
        ("down", down, 0),
    ):
        expand_into(
            getattr(module, name + "_packed")[expert_slice],
            getattr(module, name + "_scale")[expert_slice],
            getattr(module, name + "_outer")[expert_slice],
            target,
            error,
            row_offset=offset,
        )
    if error.item():
        raise ValueError("Nonfinite NVFP4 expansion")
    return gate_up, down
