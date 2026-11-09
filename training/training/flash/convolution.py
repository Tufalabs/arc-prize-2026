"""Opt-in cache-free convolution paths, including packed documents."""

import torch
import torch.nn.functional as F


@torch.compiler.disable
def fla_convolution(x, weight, bias, activation, cu_seqlens):
    # FLA ships only in the Linux flash-distillation extra; CI and macOS import
    # this module without it.
    from fla.modules.convolution import causal_conv1d

    output, _ = causal_conv1d(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
        backend="triton",
    )
    return output.transpose(1, 2)


def ple_convolution(x, weight, dilation, positions):
    """Depthwise FP32 accumulation, BF16 conv rounding, then SiLU.

    positions contains each token's position within its document, so a tap
    never reads across a packed boundary. The four taps compile into one
    pointwise kernel, independent of the number of documents.
    """
    total = torch.zeros_like(x, dtype=torch.float32)
    token = torch.arange(x.shape[1], device=x.device)
    for tap in range(weight.shape[-1]):
        offset = (weight.shape[-1] - 1 - tap) * dilation
        value = x[:, (token - offset).clamp_min(0)].float()
        value = torch.where((positions >= offset)[None, :, None], value, 0)
        total = total + value * weight[:, 0, tap].float()
    return F.silu(total.to(x.dtype))


compiled_ple_convolution = torch.compile(
    ple_convolution,
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
