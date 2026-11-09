from __future__ import annotations

from importlib import metadata
from types import SimpleNamespace

import torch


EXPECTED_FA4_VERSION = "4.0.0b19"


def _fa4_compatible_input(tensor: torch.Tensor) -> torch.Tensor:
    """Make the BSHD view consumed by FA4 contiguous without changing API layout."""
    bshd = tensor.transpose(1, 2)
    if bshd.is_contiguous():
        return tensor
    return bshd.contiguous().transpose(1, 2)


def register_fa4_contiguous_inputs() -> None:
    """Keep Qwen's gated, strided query view from being misread by FA4."""
    from transformers import AttentionInterface

    original = AttentionInterface()["flash_attention_4"]
    if getattr(original, "_arc3_contiguous_inputs", False):
        return

    def contiguous_flash_attention_forward(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ):
        # Qwen's q_proj returns Q as half of a [Q, gate] view. Its last
        # dimension is contiguous but its head stride includes the gate half;
        # flash-attn-4 b19 assumes the complete BSHD view is contiguous.
        return original(
            module,
            _fa4_compatible_input(query),
            _fa4_compatible_input(key),
            _fa4_compatible_input(value),
            attention_mask,
            **kwargs,
        )

    contiguous_flash_attention_forward._arc3_contiguous_inputs = True
    AttentionInterface.register("flash_attention_4", contiguous_flash_attention_forward)


def require_fa4_distribution() -> str:
    try:
        installed_version = metadata.version("flash-attn-4")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("flash-attn-4 is not installed") from exc
    if installed_version != EXPECTED_FA4_VERSION:
        raise RuntimeError(
            "unexpected flash-attn-4 version: "
            f"expected {EXPECTED_FA4_VERSION}, found {installed_version}"
        )
    return installed_version


def verify_fa4(*, sequence_length: int = 128) -> str:
    installed_version = require_fa4_distribution()

    from transformers import AttentionInterface

    if "flash_attention_4" not in AttentionInterface().valid_keys():
        raise RuntimeError(
            "the installed Transformers build does not register flash_attention_4"
        )
    register_fa4_contiguous_inputs()

    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as exc:
        raise RuntimeError("flash_attn.cute is not importable") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("FA4 verification requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        raise RuntimeError(
            "FA4 requires Hopper or Blackwell; "
            f"found compute capability {capability[0]}.{capability[1]}"
        )

    # Match Qwen3.6-27B's GQA geometry: 24 query heads, 4 KV heads, D=256.
    shapes = ((1, sequence_length, 24, 256), (1, sequence_length, 4, 256))
    query_and_gate = torch.randn(
        (*shapes[0][:-1], shapes[0][-1] * 2),
        device=device,
        dtype=torch.bfloat16,
    )
    query = query_and_gate.chunk(2, dim=-1)[0].detach().requires_grad_(True)
    key = torch.randn(
        shapes[1], device=device, dtype=torch.bfloat16, requires_grad=True
    )
    value = torch.randn(
        shapes[1], device=device, dtype=torch.bfloat16, requires_grad=True
    )
    torch.cuda.reset_peak_memory_stats(device)
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_4"),
        is_causal=True,
        layer_idx=0,
    )
    output = AttentionInterface()["flash_attention_4"](
        module,
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None,
    )[0]
    reference = flash_attn_func(
        query.contiguous(), key.contiguous(), value.contiguous(), causal=True
    )
    if isinstance(reference, tuple):
        reference = reference[0]
    max_difference = (output.float() - reference.float()).abs().max().item()
    if max_difference > 0.02:
        raise RuntimeError(
            "FA4 Transformers integration failed the strided-Q parity check: "
            f"max_difference={max_difference:.6g}"
        )
    output.float().square().mean().backward()
    torch.cuda.synchronize(device)
    if any(
        tensor.grad is None or not torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    ):
        raise RuntimeError("FA4 backward produced missing or non-finite gradients")

    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    device_name = torch.cuda.get_device_name(device)
    return (
        f"FA4 ready: version={installed_version} device={device_name} "
        f"capability={capability[0]}.{capability[1]} "
        f"strided_q_max_diff={max_difference:.6g} peak_mib={peak_mib:.1f}"
    )
