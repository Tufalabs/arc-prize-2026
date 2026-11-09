"""Checkpoint tensor access and native NVFP4 expansion, without the model stack.

The exporter and other CPU tools import this on login nodes pinned to
Transformers 5.14.1, so it depends only on torch and safetensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

SOURCE = Path(
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform"
)


class Reader:
    """Read checkpoint tensors from a canonical or explicitly staged root."""

    def __init__(self, root=SOURCE, *, shard_root=None):
        self.root = Path(root)
        self.shard_root = Path(shard_root) if shard_root is not None else self.root
        self.mapping = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles = {}

    def tensor(self, name):
        shard = self.mapping[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                self.shard_root / shard, framework="pt", device="cpu"
            )
        return self.handles[shard].get_tensor(name)

    def dequant(self, prefix, device):
        return dequant_nvfp4(
            self.tensor(prefix + ".weight").to(device),
            self.tensor(prefix + ".weight_scale").to(device),
            self.tensor(prefix + ".weight_scale_2").to(device),
        )


def dequant_nvfp4(packed, scale, outer):
    """Expand NVFP4 nibbles to BF16, including batched expert dimensions."""
    if packed.dtype != torch.uint8:
        raise TypeError("NVFP4 weights must be packed uint8")
    if packed.shape[:-1] != scale.shape[:-1] or packed.shape[-1] * 2 != (
        scale.shape[-1] * 16
    ):
        raise ValueError("NVFP4 packed/scale shapes do not agree")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    outer = outer.float()
    while outer.ndim < codes.ndim:
        outer = outer.unsqueeze(-1)
    result = lut[codes.long()] * scale.float().repeat_interleave(16, -1) * outer
    result = result.bfloat16()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite NVFP4 expansion")
    return result
