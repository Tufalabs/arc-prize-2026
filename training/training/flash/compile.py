"""Opt-in decoder compilation with explicit eager kernel boundaries."""

import torch
from transformers.integrations.moe import _grouped_linear
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from .dense_lora import LoRALinear


def compile_decoder(layer, precision="reference"):
    """Compile the decoder's tensor glue, keeping stateful kernels opaque.

    The outer FlashDecoderLayer.forward owns packed-document context. QSA
    selection/retention and host n-gram lookup remain eager, so checkpoint
    replay sees the same context managers as the uncompiled path.
    """
    if precision not in ("reference", "fused"):
        raise ValueError("Unknown compile_precision")
    if precision == "reference":
        for module in layer.modules():
            if isinstance(module, (torch.nn.Linear, LoRALinear,
                                   qwen.Qwen4ExpTextRMSNorm, qwen.Qwen4ExpTextRMSNormGated,
                                   qwen.Qwen4ExpTextTopKRouter)):
                module.forward = torch.compiler.disable(module.forward)

    if hasattr(layer, "linear_attn"):
        gdn = layer.linear_attn
        gdn.reference_gates = precision == "reference"
        gdn._flash_chunk_rule = torch.compiler.disable(gdn._flash_chunk_rule)
        gdn._convolution = torch.compiler.disable(gdn._convolution)
    if hasattr(layer, "self_attn"):
        layer.self_attn.forward = torch.compiler.disable(layer.self_attn.forward)
    experts = layer.mlp.experts
    experts.grouped_linear = torch.compiler.disable(_grouped_linear)
    if experts.optimization == "none":
        experts.forward = torch.compiler.disable(experts.forward)
    if layer.ple is not None:
        layer.ple.ple_embedding.forward = torch.compiler.disable(layer.ple.ple_embedding.forward)
        layer.ple._short_conv = torch.compiler.disable(layer.ple._short_conv)
    layer.compiled_forward = torch.compile(
        layer._eager_forward(),
        dynamic=True,
        fullgraph=False,
        options={"emulate_precision_casts": True},
    )
