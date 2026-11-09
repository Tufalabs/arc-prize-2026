"""BF16 rounding-aware GDN norm and MLP gate kernels."""

import torch
import triton
import triton.language as tl

from .dtype_contract import floating_inputs, low_precision_inputs


@triton.jit
def _gdn_norm_gate(X, Z, W, Y, R, N, D: tl.constexpr, EPS,
                   B: tl.constexpr, ROWS: tl.constexpr, SERVING: tl.constexpr, SIGMOID: tl.constexpr,
                   PRODUCT_DTYPE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, B)
    mask = (row[:, None] < N) & (col[None, :] < D)
    offset = row[:, None] * D + col[None, :]
    x = tl.load(X + offset, mask, 0)
    z = tl.load(Z + offset, mask, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    xf = x.to(tl.float32)
    variance = tl.sum(xf * xf, 1) / D + EPS
    r = tl.rsqrt(variance)
    normed = xf * r[:, None]
    if not SERVING:
        normed = normed.to(x.dtype).to(tl.float32)
    scaled = normed * w[None, :]
    if not SERVING:
        scaled = scaled.to(PRODUCT_DTYPE).to(tl.float32)
    silu = tl.sigmoid(z) if SIGMOID else z * tl.sigmoid(z)
    tl.store(Y + offset, scaled * silu, mask)
    tl.store(R + row, r, row < N)


@triton.jit
def _gdn_norm_gate_backward(X, Z, W, R, G, DX, DZ, N, D: tl.constexpr,
                            B: tl.constexpr, ROWS: tl.constexpr, SERVING: tl.constexpr, SIGMOID: tl.constexpr,
                            PRODUCT_DTYPE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, B)
    mask = (row[:, None] < N) & (col[None, :] < D)
    offset = row[:, None] * D + col[None, :]
    x = tl.load(X + offset, mask, 0)
    z = tl.load(Z + offset, mask, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    g = tl.load(G + offset, mask, 0).to(tl.float32)
    r = tl.load(R + row, row < N, 0)
    xf = x.to(tl.float32)
    normed = xf * r[:, None]
    if not SERVING:
        normed = normed.to(x.dtype).to(tl.float32)
    scaled = normed * w[None, :]
    if not SERVING:
        scaled = scaled.to(PRODUCT_DTYPE).to(tl.float32)
    sigmoid = 1 / (1 + tl.exp(-z))
    silu = sigmoid if SIGMOID else z * sigmoid
    derivative = sigmoid * (1 - sigmoid) if SIGMOID else sigmoid * (1 + z * (1 - sigmoid))
    dz = (g * scaled) * derivative
    dscaled = g * silu
    if not SERVING:
        dscaled = dscaled.to(PRODUCT_DTYPE).to(tl.float32)
    dnorm = dscaled * w[None, :]
    if not SERVING:
        dnorm = dnorm.to(x.dtype).to(tl.float32)
    dx = r[:, None] * (dnorm - xf * (tl.sum(dnorm * xf, 1) / D)[:, None] * r[:, None] * r[:, None])
    tl.store(DX + offset, dx, mask)
    tl.store(DZ + offset, dz, mask)


def norm_gate(x, z, weight, eps, serving=False, sigmoid=False):
    low_precision_inputs("norm_gate", x, z, weight)
    if x.shape != z.shape or weight.numel() != x.shape[-1]:
        raise ValueError("norm_gate: gate shape and norm weight width must match the activation")
    x, z, weight = x.contiguous(), z.contiguous(), weight.contiguous()
    product_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
                     torch.float32: tl.float32}[torch.promote_types(x.dtype, weight.dtype)]
    y = torch.empty_like(x)
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    r = torch.empty(n, dtype=torch.float32, device=x.device)
    rows = 16
    if serving:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        rows = min(triton.next_power_of_2(triton.cdiv(n, 2 * sms)), 4)
    _gdn_norm_gate[(triton.cdiv(n, rows),)](
        x, z, weight, y, r, n, d, eps, triton.next_power_of_2(d), rows, serving, sigmoid, product_dtype,
        num_warps=1 if serving else 4, enable_fp_fusion=serving)
    return y, r


def norm_gate_backward(x, z, weight, r, gradient, serving=False, sigmoid=False):
    low_precision_inputs("norm_gate_backward", x, z, weight, gradient)
    x, z, weight = x.contiguous(), z.contiguous(), weight.contiguous()
    product_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
                     torch.float32: tl.float32}[torch.promote_types(x.dtype, weight.dtype)]
    dx, dz = torch.empty_like(x), torch.empty_like(z)
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    _gdn_norm_gate_backward[(triton.cdiv(n, 16),)](
        x, z, weight, r, gradient.contiguous(), dx, dz, n, d,
        triton.next_power_of_2(d), 16, serving, sigmoid, product_dtype, enable_fp_fusion=False)
    return dx, dz


@triton.jit
def _mlp_swiglu(GATE, UP, Y, SIZE, D: tl.constexpr, PACKED: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    offset = (i // D) * (2 * D if PACKED else D) + i % D
    x = tl.load(GATE + offset, i < SIZE, 0)
    up = tl.load(UP + offset, i < SIZE, 0).to(tl.float32)
    xf = x.to(tl.float32)
    silu = (xf / (1 + tl.exp(-xf))).to(x.dtype).to(tl.float32)
    tl.store(Y + i, silu * up, i < SIZE)


@triton.jit
def _mlp_swiglu_backward(GATE, UP, G, DGATE, DUP, SIZE, D: tl.constexpr,
                         PACKED: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    offset = (i // D) * (2 * D if PACKED else D) + i % D
    x = tl.load(GATE + offset, i < SIZE, 0)
    up = tl.load(UP + offset, i < SIZE, 0).to(tl.float32)
    g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
    xf = x.to(tl.float32)
    s = 1 / (1 + tl.exp(-xf))
    silu = (xf * s).to(x.dtype).to(tl.float32)
    ds = (g * up).to(x.dtype).to(tl.float32)
    dx = ds * (s * (1 + xf * (1 - s)))
    tl.store(DGATE + offset, dx, i < SIZE)
    tl.store(DUP + offset, g * silu, i < SIZE)


def swiglu(gate, up, packed):
    low_precision_inputs("swiglu", gate, *(() if packed else (up,)), same_dtype=True)
    if (packed and gate.shape[-1] % 2) or (not packed and gate.shape != up.shape):
        raise ValueError("swiglu: requires equal gate/up shapes or an even packed width")
    gate = gate.contiguous()
    up = None if packed else up.contiguous()
    d = gate.shape[-1] // 2 if packed else gate.shape[-1]
    y = gate.new_empty((*gate.shape[:-1], d))
    other = gate[..., d:] if packed else up
    _mlp_swiglu[(triton.cdiv(y.numel(), 512),)](
        gate, other, y, y.numel(), d, packed, 512, enable_fp_fusion=False)
    return y


def swiglu_backward(gate, up, gradient, packed):
    low_precision_inputs("swiglu_backward", gate, gradient, *(() if packed else (up,)), same_dtype=True)
    gate = gate.contiguous()
    up = None if packed else up.contiguous()
    d = gate.shape[-1] // 2 if packed else gate.shape[-1]
    dx = torch.empty_like(gate)
    du = dx[..., d:] if packed else torch.empty_like(up)
    other = gate[..., d:] if packed else up
    _mlp_swiglu_backward[(triton.cdiv(gradient.numel(), 512),)](
        gate, other, gradient.contiguous(), dx, du, gradient.numel(), d, packed, 512,
        enable_fp_fusion=False)
    return dx, None if packed else du


@triton.jit
def _gdn_pack_qkv(X, Q, K, V, SIZE, H: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
                  S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    row, col = i // (HV * D), i % (HV * D)
    source = (col // D // (HV // H)) * D + col % D
    q = tl.load(X + row * S0 + source * S1, i < SIZE, 0)
    k = tl.load(X + row * S0 + (H * D + source) * S1, i < SIZE, 0)
    v = tl.load(X + row * S0 + (2 * H * D + col) * S1, i < SIZE, 0)
    tl.store(Q + i, q, i < SIZE)
    tl.store(K + i, k, i < SIZE)
    tl.store(V + i, v, i < SIZE)


@triton.jit
def _gdn_pack_qkv_backward(GQ, GK, GV, DX, SIZE, H: tl.constexpr, HV: tl.constexpr,
                           D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    width: tl.constexpr = (2 * H + HV) * D
    row, col = i // width, i % width
    isq, isk = col < H * D, (col >= H * D) & (col < 2 * H * D)
    source = row * HV * D + ((col % (H * D)) // D * (HV // H)) * D + col % D
    total = tl.full((B,), 0, tl.float32)
    for repeat in range(HV // H):
        q = tl.load(GQ + source + repeat * D, (i < SIZE) & isq, 0).to(tl.float32)
        k = tl.load(GK + source + repeat * D, (i < SIZE) & isk, 0).to(tl.float32)
        total += q + k
    value = tl.load(GV + row * HV * D + col - 2 * H * D,
                    (i < SIZE) & (~isq) & (~isk), 0).to(tl.float32)
    tl.store(DX + i, total + value, i < SIZE)


def pack_qkv(x, heads, value_heads, dim):
    floating_inputs("pack_qkv", x)
    if heads <= 0 or value_heads % heads or x.shape[-1] != (2 * heads + value_heads) * dim:
        raise ValueError("pack_qkv: requires repeated heads and matching packed width")
    shape = (*x.shape[:-1], value_heads, dim)
    q, k, v = (x.new_empty(shape) for _ in range(3))
    flat = x.reshape(-1, x.shape[-1])
    _gdn_pack_qkv[(triton.cdiv(q.numel(), 512),)](
        flat, q, k, v, q.numel(), heads, value_heads, dim, *flat.stride(), 512)
    return q, k, v


def pack_qkv_backward(gq, gk, gv, heads):
    floating_inputs("pack_qkv_backward", gq, gk, gv, same_dtype=True)
    value_heads, dim = gq.shape[-2:]
    dx = gq.new_empty((*gq.shape[:-2], (2 * heads + value_heads) * dim))
    _gdn_pack_qkv_backward[(triton.cdiv(dx.numel(), 512),)](
        gq.contiguous(), gk.contiguous(), gv.contiguous(), dx, dx.numel(),
        heads, value_heads, dim, 512, enable_fp_fusion=False)
    return dx


@triton.jit
def _qsa_projection_norm(X, W, Y, R, N, T, H: tl.constexpr, D: tl.constexpr,
                         S0, S1: tl.constexpr, S2: tl.constexpr,
                         EPS: tl.constexpr, B: tl.constexpr, BATCH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    col = tl.arange(0, B)
    if BATCH == 1:
        offset = row // H * S1 + row % H * S2
    else:
        offset = (row // (T * H)) * S0 + (row // H % T) * S1 + row % H * S2
    mask = (row[:, None] < N) & (col[None, :] < D)
    x = tl.load(X + offset[:, None] + col[None, :], mask, 0).to(tl.float32)
    gain = 1 + tl.load(W + col, col < D, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    tl.store(Y + row[:, None] * D + col[None, :], (x * r[:, None]) * gain[None, :], mask)
    tl.store(R + row, r, row < N)


@triton.jit
def _qsa_projection_norm_backward(X, W, R, G, DX, N, T, H: tl.constexpr, D: tl.constexpr,
                                  S0, S1: tl.constexpr, S2: tl.constexpr, B: tl.constexpr, BATCH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    col = tl.arange(0, B)
    if BATCH == 1:
        offset = row // H * S1 + row % H * S2
    else:
        offset = (row // (T * H)) * S0 + (row // H % T) * S1 + row % H * S2
    flat = row[:, None] * D + col[None, :]
    mask = (row[:, None] < N) & (col[None, :] < D)
    x = tl.load(X + offset[:, None] + col[None, :], mask, 0).to(tl.float32)
    gain = 1 + tl.load(W + col, col < D, 0).to(tl.float32)
    g = tl.load(G + flat, mask, 0).to(tl.float32) * gain[None, :]
    r = tl.load(R + row, row < N, 0)
    dx = r[:, None] * (g - x * (tl.sum(g * x, 1) / D)[:, None] * r[:, None] * r[:, None])
    tl.store(DX + flat, dx, mask)


def projection_norm(x, weight, eps):
    low_precision_inputs("projection_norm", x, weight)
    if x.ndim != 4 or weight.numel() != x.shape[-1]:
        raise ValueError("projection_norm: requires [batch, tokens, heads, dim] and matching weight")
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    batch, tokens, heads, dim = x.shape
    n = batch * tokens * heads
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    r = torch.empty(n, device=x.device, dtype=torch.float32)
    _qsa_projection_norm[(triton.cdiv(n, 4),)](
        x, weight, y, r, n, tokens, heads, dim, *x.stride()[:3], eps,
        triton.next_power_of_2(dim), batch, num_warps=4, enable_fp_fusion=False)
    return y, r


def projection_norm_backward(x, weight, r, gradient):
    low_precision_inputs("projection_norm_backward", x, weight, gradient)
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    batch, tokens, heads, dim = x.shape
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _qsa_projection_norm_backward[(triton.cdiv(r.numel(), 4),)](
        x, weight, r, gradient.contiguous(), dx, batch * tokens * heads,
        tokens, heads, dim, *x.stride()[:3], triton.next_power_of_2(dim), batch,
        num_warps=4, enable_fp_fusion=False)
    return dx


@triton.jit
def _lora_scaled_add(X, A, Y, N, SCALE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    scaled = (a * SCALE).to(x.dtype).to(tl.float32)
    tl.store(Y + i, x.to(tl.float32) + scaled, i < N)


def scaled_add(base, adapter, scale):
    floating_inputs("scaled_add", base, adapter, same_dtype=True)
    if base.shape != adapter.shape:
        raise ValueError("scaled_add: requires equal base/adapter shapes")
    y = base.new_empty(base.shape)
    _lora_scaled_add[(triton.cdiv(y.numel(), 512),)](
        base.contiguous(), adapter.contiguous(), y, y.numel(), scale, 512, enable_fp_fusion=False)
    return y
