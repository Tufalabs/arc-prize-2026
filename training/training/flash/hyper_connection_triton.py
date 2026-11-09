"""Token-local hyper-connection kernels; all reductions are deterministic."""

import torch
import triton
import triton.language as tl

from .dtype_contract import low_precision_inputs


def _group_count(op, groups):
    if type(groups) is not int or groups <= 0:
        raise ValueError(f"{op}: groups={groups!r}; requires a positive integer")


def _groups(op, x, groups):
    _group_count(op, groups)
    if x.ndim < 2 or x.shape[-1] <= 0 or x.shape[-1] % groups:
        raise ValueError(f"{op}: x.shape={x.shape} must have rank >= 2 and a positive width divisible by groups={groups}")


def _norm_shapes(op, x, weight, groups):
    _groups(op, x, groups)
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"{op}: weight.shape={weight.shape}; requires ({x.shape[-1]},)")


def _same_shape(op, x, other):
    if x.shape != other.shape:
        raise ValueError(f"{op}: mismatched shapes {x.shape}, {other.shape}")


def _inject_shapes(op, x, h, weights):
    if h.ndim < 2 or weights.ndim != h.ndim or x.ndim != h.ndim:
        raise ValueError(f"{op}: expected matching ranks >= 2; got {x.shape}, {h.shape}, {weights.shape}")
    groups = weights.shape[-1]
    _groups(op, x, groups)
    if x.shape[:-1] != h.shape[:-1] or x.shape[:-1] != weights.shape[:-1] or x.shape[-1] != groups * h.shape[-1]:
        raise ValueError(f"{op}: incompatible token dimensions or stream widths: x={x.shape}, h={h.shape}, weights={weights.shape}")


def _stats_shape(op, x, r, groups):
    if r.shape != (x.numel() // (x.shape[-1] // groups),) or r.dtype != torch.float32 or r.device != x.device:
        raise ValueError(f"{op}: invalid statistics shape/dtype/device: {r.shape}, {r.dtype}, {r.device}")


@triton.jit
def _hc_norm(X, W, Y, R, D: tl.constexpr, C: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + (row % C) * D + col, col < D, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = x * r
    if SERVING:
        y += y * w
    else:
        y *= 1 + w
    tl.store(Y + row * D + col, y, col < D)
    tl.store(R + row, r)


@triton.jit
def _hc_norm_backward(X, W, R, G, DX, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + (row % C) * D + col, col < D, 0).to(tl.float32) + 1
    g = tl.load(G + row * D + col, col < D, 0).to(tl.float32) * w
    r = tl.load(R + row)
    dx = r * (g - x * (tl.sum(g * x, 0) / D) * r * r)
    tl.store(DX + row * D + col, dx, col < D)


def norm(x, weight, groups, eps, serving=False):
    _norm_shapes("HC norm", x, weight, groups)
    low_precision_inputs("HC norm", x, weight)
    x, weight = x.contiguous(), weight.contiguous()
    d = x.shape[-1] // groups
    y = torch.empty_like(x)
    r = torch.empty(x.numel() // d, device=x.device, dtype=torch.float32)
    _hc_norm[(r.numel(),)](x, weight, y, r, d, groups, eps,
                           triton.next_power_of_2(d), SERVING=serving, enable_fp_fusion=serving)
    return y, r


def norm_backward(x, weight, r, gradient, groups):
    _norm_shapes("HC norm_backward", x, weight, groups)
    _same_shape("HC norm_backward", x, gradient)
    _stats_shape("HC norm_backward", x, r, groups)
    low_precision_inputs("HC norm_backward", x, weight, gradient)
    x, weight = x.contiguous(), weight.contiguous()
    dx = torch.empty_like(x)
    d = x.shape[-1] // groups
    _hc_norm_backward[(r.numel(),)](x, weight, r, gradient.contiguous(), dx, d, groups,
                                    triton.next_power_of_2(d), enable_fp_fusion=False)
    return dx


@triton.jit
def _hc_activation(X, Y, SIZE, C: tl.constexpr, INJECT: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + i, i < SIZE, 0)
    z = x.to(tl.float32) / C
    if not SERVING:
        z = z.to(x.dtype).to(tl.float32)
    s = 1 / (1 + tl.exp(-z))
    if INJECT:
        y = 2 * (s if SERVING else s.to(x.dtype).to(tl.float32))
    else:
        y = z * s
    tl.store(Y + i, y, i < SIZE)


@triton.jit
def _hc_activation_backward(X, G, DX, SIZE, C: tl.constexpr, INJECT: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + i, i < SIZE, 0)
    g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
    z = x.to(tl.float32) / C
    if not SERVING:
        z = z.to(x.dtype).to(tl.float32)
    s = 1 / (1 + tl.exp(-z))
    if SERVING:
        grad = g * (2 * s * (1 - s) if INJECT else s * (1 + z * (1 - s)))
    elif INJECT:
        s = s.to(x.dtype).to(tl.float32)
        grad = ((g * 2).to(x.dtype).to(tl.float32) * s * (1 - s)).to(x.dtype).to(tl.float32)
    else:
        grad = (g * (s * (1 + z * (1 - s)))).to(x.dtype).to(tl.float32)
    tl.store(DX + i, grad / C, i < SIZE)


def activation(x, groups, inject, serving=False):
    _group_count("HC activation", groups)
    low_precision_inputs("HC activation", x)
    x = x.contiguous()
    y = torch.empty_like(x)
    _hc_activation[(triton.cdiv(x.numel(), 512),)](
        x, y, x.numel(), groups, inject, 512, SERVING=serving, enable_fp_fusion=serving)
    return y


def activation_backward(x, gradient, groups, inject, serving=False):
    _group_count("HC activation_backward", groups)
    _same_shape("HC activation_backward", x, gradient)
    low_precision_inputs("HC activation_backward", x, gradient)
    x = x.contiguous()
    dx = torch.empty_like(x)
    _hc_activation_backward[(triton.cdiv(x.numel(), 512),)](
        x, gradient.contiguous(), dx, x.numel(), groups, inject, 512, SERVING=serving, enable_fp_fusion=False)
    return dx


@triton.jit
def _hc_mix(X, U, Y, SIZE, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for stream in range(C):
        offset = (i // D) * C * D + stream * D + i % D
        x = tl.load(X + offset, i < SIZE, 0)
        u = tl.load(U + offset, i < SIZE, 0).to(tl.float32)
        s = tl.sigmoid(u)
        if SERVING:
            total += s * x.to(tl.float32)
        else:
            s = s.to(x.dtype).to(tl.float32)
            total += (s * x.to(tl.float32)).to(x.dtype).to(tl.float32)
    tl.store(Y + i, total / C, i < SIZE)


@triton.jit
def _hc_mix_backward(X, U, G, DX, DU, SIZE, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + i, i < SIZE, 0)
    u = tl.load(U + i, i < SIZE, 0).to(tl.float32)
    g = tl.load(G + (i // (C * D)) * D + i % D, i < SIZE, 0).to(tl.float32)
    g = g / C
    s = tl.sigmoid(u)
    if not SERVING:
        g = g.to(x.dtype).to(tl.float32)
        s = s.to(x.dtype).to(tl.float32)
    tl.store(DX + i, g * s, i < SIZE)
    ds = g * x.to(tl.float32)
    if not SERVING:
        ds = ds.to(x.dtype).to(tl.float32)
    tl.store(DU + i, ds * s * (1 - s), i < SIZE)


def mix(x, up, groups, serving=False):
    _groups("HC mix", x, groups)
    _same_shape("HC mix", x, up)
    low_precision_inputs("HC mix", x, up, same_dtype=True)
    x, up = x.contiguous(), up.contiguous()
    d = x.shape[-1] // groups
    y = x.new_empty((*x.shape[:-1], d))
    _hc_mix[(triton.cdiv(y.numel(), 256),)](
        x, up, y, y.numel(), d, groups, 256, SERVING=serving, enable_fp_fusion=serving)
    return y


def mix_backward(x, up, gradient, groups, serving=False):
    _groups("HC mix_backward", x, groups)
    _same_shape("HC mix_backward", x, up)
    if gradient.shape != (*x.shape[:-1], x.shape[-1] // groups):
        raise ValueError(f"HC mix_backward: gradient.shape={gradient.shape} is incompatible with x={x.shape}, groups={groups}")
    low_precision_inputs("HC mix_backward", x, up, gradient, same_dtype=True)
    x, up = x.contiguous(), up.contiguous()
    dx, du = torch.empty_like(x), torch.empty_like(up)
    _hc_mix_backward[(triton.cdiv(x.numel(), 512),)](
        x, up, gradient.contiguous(), dx, du, x.numel(), x.shape[-1] // groups,
        groups, 512, SERVING=serving, enable_fp_fusion=False)
    return dx, du


@triton.jit
def _hc_inject(X, H, W, Y, SIZE, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    token = i // (C * D)
    x = tl.load(X + i, i < SIZE, 0)
    h = tl.load(H + token * D + i % D, i < SIZE, 0).to(tl.float32)
    w = tl.load(W + i // D, i < SIZE, 0).to(tl.float32)
    if SERVING:
        w = 2.0 * tl.sigmoid(w / C)
        product = h * w
    else:
        product = (h * w).to(x.dtype).to(tl.float32)
    tl.store(Y + i, x.to(tl.float32) + product, i < SIZE)


@triton.jit
def _hc_inject_backward_h(G, W, DH, SIZE, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for stream in range(C):
        slot = (i // D) * C + stream
        g = tl.load(G + slot * D + i % D, i < SIZE, 0)
        w = tl.load(W + slot, i < SIZE, 0).to(tl.float32)
        if SERVING:
            total += g.to(tl.float32) * (2.0 * tl.sigmoid(w / C))
        else:
            total += (g.to(tl.float32) * w).to(g.dtype).to(tl.float32)
    tl.store(DH + i, total, i < SIZE)


@triton.jit
def _hc_inject_backward_w(G, H, W, DW, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    g = tl.load(G + row * D + col, col < D, 0)
    h = tl.load(H + (row // C) * D + col, col < D, 0).to(tl.float32)
    product = g.to(tl.float32) * h
    if SERVING:
        s = tl.sigmoid(tl.load(W + row).to(tl.float32) / C)
        total = tl.sum(product, 0) * (2.0 * s * (1 - s) / C)
    else:
        total = tl.sum(product.to(g.dtype).to(tl.float32), 0)
    tl.store(DW + row, total)


@triton.jit
def _hc_serving_inject(X, H, W, Y, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    stream = tl.arange(0, triton.next_power_of_2(C))
    mask = (stream[:, None] < C) & (col[None, :] < D)
    offset = row * C * D + stream[:, None] * D + col[None, :]
    res = tl.load(X + offset, mask, 0).to(tl.float32)
    h = tl.load(H + row * D + col, col < D, 0).to(tl.float32)
    logits = tl.load(W + row * C + stream, stream < C, 0).to(tl.float32)
    weights = 2.0 * tl.sigmoid(logits / C)
    product = h[None, :] * weights[:, None]
    tl.store(Y + offset, res + product, mask)


@triton.jit
def _hc_serving_inject_backward_h(G, W, DH, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    total = tl.full((B,), 0, tl.float32)
    for stream in tl.static_range(C):
        logits = tl.load(W + row * C + stream).to(tl.float32)
        weight = 2.0 * tl.sigmoid(logits / C)
        g = tl.load(G + (row * C + stream) * D + col, col < D, 0).to(tl.float32)
        total += g * weight
    tl.store(DH + row * D + col, total, col < D)


def inject(x, h, weights, serving=False):
    _inject_shapes("HC inject", x, h, weights)
    low_precision_inputs("HC inject", x, h, weights, same_dtype=True)
    x, h, weights = x.contiguous(), h.contiguous(), weights.contiguous()
    y = torch.empty_like(x)
    if serving:
        d, c = h.shape[-1], weights.shape[-1]
        _hc_serving_inject[(h.numel() // d, triton.cdiv(d, 512))](x, h, weights, y, d, c, 512)
        return y
    _hc_inject[(triton.cdiv(x.numel(), 512),)](
        x, h, weights, y, x.numel(), h.shape[-1], weights.shape[-1], 512,
        SERVING=serving, enable_fp_fusion=serving)
    return y


def inject_backward(gradient, h, weights, serving=False):
    _inject_shapes("HC inject_backward", gradient, h, weights)
    low_precision_inputs("HC inject_backward", gradient, h, weights, same_dtype=True)
    h, weights = h.contiguous(), weights.contiguous()
    gradient = gradient.contiguous()
    dh, dw = torch.empty_like(h), torch.empty_like(weights)
    d, c = h.shape[-1], weights.shape[-1]
    if serving:
        _hc_serving_inject_backward_h[(h.numel() // d, triton.cdiv(d, 256))](
            gradient, weights, dh, d, c, 256, enable_fp_fusion=False)
    else:
        _hc_inject_backward_h[(triton.cdiv(h.numel(), 256),)](
            gradient, weights, dh, h.numel(), d, c, 256, SERVING=False, enable_fp_fusion=False)
    _hc_inject_backward_w[(weights.numel(),)](
        gradient, h, weights, dw, d, c, triton.next_power_of_2(d), SERVING=serving, enable_fp_fusion=False)
    return gradient, dh, dw


@triton.jit
def _hc_inject_norm(X, H, W, GAIN, Y, Z, R, D: tl.constexpr, C: tl.constexpr,
                    EPS: tl.constexpr, B: tl.constexpr, SERVING: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if SERVING:
        col = tl.arange(0, B // 512)[:, None] * 512 + tl.arange(0, 512)[None, :]
    else:
        col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, 0)
    h = tl.load(H + (row // C) * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + row).to(tl.float32)
    if SERVING:
        w = 2.0 * tl.sigmoid(w / C)
        product = h * w
    else:
        product = (h * w).to(x.dtype).to(tl.float32)
    y = (x.to(tl.float32) + product).to(x.dtype).to(tl.float32)
    y = tl.where(col < D, y, 0)
    if SERVING:
        squares = tl.sum(tl.sum(y * y, 1), 0)
    else:
        squares = tl.sum(y * y, 0)
    r = tl.rsqrt(squares / D + EPS)
    tl.store(Y + row * D + col, y, col < D)
    # Match the serving kernel's late affine load to reduce register pressure
    # across the grouped statistics reduction at large token counts.
    gain = tl.load(GAIN + (row % C) * D + col, col < D, 0).to(tl.float32)
    z = y * r
    if SERVING:
        z += z * gain
    else:
        z *= 1 + gain
    tl.store(Z + row * D + col, z, col < D)
    tl.store(R + row, r)


@triton.jit
def _hc_inject_norm_backward(Y, GAIN, R, GY, GZ, DX,
                             D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    y = tl.load(Y + row * D + col, col < D, 0).to(tl.float32)
    gain = tl.load(GAIN + (row % C) * D + col, col < D, 0).to(tl.float32) + 1
    g = tl.load(GZ + row * D + col, col < D, 0).to(tl.float32) * gain
    r = tl.load(R + row)
    dy = r * (g - y * (tl.sum(g * y, 0) / D) * r * r)
    direct = tl.load(GY + row * D + col, col < D, 0)
    total = dy.to(direct.dtype).to(tl.float32) + direct.to(tl.float32)
    tl.store(DX + row * D + col, total, col < D)


def inject_norm(x, h, weights, gain, eps, serving=False):
    _inject_shapes("HC inject_norm", x, h, weights)
    _norm_shapes("HC inject_norm", x, gain, weights.shape[-1])
    low_precision_inputs("HC inject_norm", x, h, weights, same_dtype=True)
    low_precision_inputs("HC inject_norm gain", x, gain)
    x, h, weights, gain = x.contiguous(), h.contiguous(), weights.contiguous(), gain.contiguous()
    d, c = h.shape[-1], weights.shape[-1]
    y, z = torch.empty_like(x), torch.empty_like(x)
    r = torch.empty(weights.numel(), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(d)
    if serving:
        # vLLM reduces whole 512-wide tiles, padding the tile count to a power of two.
        block = max(block, 512)
    _hc_inject_norm[(r.numel(),)](
        x, h, weights, gain, y, z, r, d, c, eps, block,
        SERVING=serving, enable_fp_fusion=serving)
    return y, z, r


def inject_norm_backward(y, gain, r, gy, gz, groups):
    _norm_shapes("HC inject_norm_backward", y, gain, groups)
    _same_shape("HC inject_norm_backward", y, gy)
    _same_shape("HC inject_norm_backward", y, gz)
    _stats_shape("HC inject_norm_backward", y, r, groups)
    low_precision_inputs("HC inject_norm_backward", y, gain, gy, gz)
    y, gain = y.contiguous(), gain.contiguous()
    dx = torch.empty_like(y)
    d = y.shape[-1] // groups
    _hc_inject_norm_backward[(r.numel(),)](
        y, gain, r, gy.contiguous(), gz.contiguous(), dx, d, groups,
        triton.next_power_of_2(d), enable_fp_fusion=False)
    return dx


@triton.jit
def _hc_norm_branches_backward(X, W, R, GMIX, GDOWN, GBLOCK, GRES, DX,
                               D: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
                               B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    offset = row * D + col
    x = tl.load(X + offset, col < D, 0).to(tl.float32)
    direct = tl.load(GRES + offset, col < D, 0)
    g = tl.load(GMIX + offset, col < D, 0).to(tl.float32)
    g += tl.load(GDOWN + offset, col < D, 0).to(tl.float32)
    if BLOCK:
        g += tl.load(GBLOCK + offset, col < D, 0).to(tl.float32)
    g = g.to(direct.dtype).to(tl.float32)
    gain = tl.load(W + (row % C) * D + col, col < D, 0).to(tl.float32) + 1
    g *= gain
    r = tl.load(R + row)
    dx = r * (g - x * (tl.sum(g * x, 0) / D) * r * r)
    dx = dx.to(direct.dtype).to(tl.float32) + direct.to(tl.float32)
    tl.store(DX + offset, dx, col < D)


def norm_branches_backward(x, weight, r, gmix, gdown, gblock, gres, groups):
    _norm_shapes("HC norm_branches_backward", x, weight, groups)
    _stats_shape("HC norm_branches_backward", x, r, groups)
    for gradient in (gmix, gdown, gres, *((gblock,) if gblock is not None else ())):
        _same_shape("HC norm_branches_backward", x, gradient)
    low_precision_inputs("HC norm_branches_backward", x, weight, r, gmix, gdown, gres)
    x, weight = x.contiguous(), weight.contiguous()
    gmix, gdown = gmix.contiguous(), gdown.contiguous()
    gblock = None if gblock is None else gblock.contiguous()
    dx = torch.empty_like(x)
    d = x.shape[-1] // groups
    _hc_norm_branches_backward[(r.numel(),)](
        x, weight, r, gmix, gdown, gblock if gblock is not None else gmix,
        gres.contiguous(), dx, d, groups, gblock is not None,
        triton.next_power_of_2(d), enable_fp_fusion=False)
    return dx
