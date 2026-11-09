"""Trained Flash routers in an adapter checkpoint: one canonical BF16 file and its hash.

With ``training.flash.train_routers`` every adapter directory the learner
writes carries its routers (training/FLASH_RL.md, "Trainable routers"), part
of the training state a resume or warm load reads back. The file is the
canonical safetensors form of training/flash_safetensors.py, which the merged
policy weights share: HF tensor names in layer order, ``{"format": "pt"}``
metadata and the BF16 cast of the FP32 masters, which is exactly what the
learner routes with, so its sha256 names the routers themselves. Servers
receive the routers inside the merged policy weights
(training/flash_policy_weights.py), not from this file.

CPU-only and dependency-free beyond torch, which only the tensor functions
import.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from training import flash_safetensors

if TYPE_CHECKING:
    import torch

ROUTER_FILE = "routers.safetensors"
ROUTER_MANIFEST = "routers.json"
_HF_NAME = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight")
_HEADER_METADATA = {"format": "pt"}


def hf_router_name(layer: int) -> str:
    return f"model.language_model.layers.{int(layer)}.mlp.gate.weight"


def _layer(name: str) -> int:
    match = _HF_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"not a Flash router tensor name: {name!r}")
    return int(match[1])


def _in_file_order(tensors: Mapping[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """The routers of layers 0..n-1 in that order, all of one shape."""
    if not tensors:
        raise ValueError("no router tensors to serialize")
    layers = {_layer(name): name for name in tensors}
    if sorted(layers) != list(range(len(layers))):
        raise ValueError(f"router layers must be 0..n-1, got {sorted(layers)}")
    ordered = [(layers[layer], tensors[layers[layer]]) for layer in range(len(layers))]
    shape = tuple(ordered[0][1].shape)
    for name, value in ordered:
        if tuple(value.shape) != shape:
            raise ValueError(f"router {name} has shape {tuple(value.shape)}, expected {shape}")
    return ordered


def serialize_routers(tensors: Mapping[str, torch.Tensor]) -> bytes:
    """The canonical file of CPU routers: layers 0..n-1, BF16 matrices of one shape, fixed header."""
    return flash_safetensors.serialize(_in_file_order(tensors), _HEADER_METADATA, kind="router")


def parse_routers(data: bytes | bytearray) -> dict[str, torch.Tensor]:
    """Inverse of serialize_routers; anything it would not have written is refused."""
    _, tensors = flash_safetensors.parse(data, source="router file")
    for name in tensors:
        _layer(name)
    if serialize_routers(tensors) != data:
        raise ValueError("router file is not in canonical form")
    return tensors


def write_router_publication(directory: Path, tensors: Mapping[str, torch.Tensor], *, iteration: int) -> dict:
    """Write ROUTER_FILE (hashed as it is written, made durable) and its manifest into an adapter directory."""
    sha256 = flash_safetensors.write_file(Path(directory) / ROUTER_FILE, _in_file_order(tensors), _HEADER_METADATA,
                                          kind="router")
    manifest = {
        "file": ROUTER_FILE,
        "sha256": sha256,
        "iteration": int(iteration),
        "tensors": len(tensors),
    }
    (Path(directory) / ROUTER_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_router_manifest(directory: Path) -> dict | None:
    """A publication's router manifest, or None when it publishes no routers."""
    directory = Path(directory)
    path = directory / ROUTER_MANIFEST
    if not path.is_file():
        if (directory / ROUTER_FILE).exists():
            raise ValueError(f"{directory / ROUTER_FILE} has no {ROUTER_MANIFEST}")
        return None
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {"file", "sha256", "iteration", "tensors"}
            or manifest["file"] != ROUTER_FILE or type(manifest["iteration"]) is not int
            or type(manifest["tensors"]) is not int or manifest["tensors"] <= 0):
        raise ValueError(f"{path} is not a router manifest")
    flash_safetensors.check_sha256(manifest["sha256"], "router")
    if not (directory / ROUTER_FILE).is_file():
        raise ValueError(f"{path} names a missing {ROUTER_FILE}")
    return manifest


def load_router_file(path: Path, sha256: str) -> dict[str, torch.Tensor]:
    """Read the file once, check its hash, then parse those same bytes."""
    return parse_routers(flash_safetensors.read_verified(Path(path), sha256, what="router"))
