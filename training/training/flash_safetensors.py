"""Canonical BF16 safetensors files: the one writer and strict reader Flash publications share.

The merged policy weights (training/flash_policy_weights.py) and an adapter's
trained routers (training/flash_routers.py) are written byte for byte the same
way, so a file's sha256 names its content: an 8-byte little-endian header
length; a compact JSON header holding ``__metadata__`` and then each tensor, in
the caller's order, as ``{"dtype": "BF16", "shape": [rows, columns],
"data_offsets": [start, end]}``, padded with spaces to a multiple of 8 bytes;
then the tensors' bytes back to back in the same order. The reader parses the
bytes that were hashed (``read_verified`` reads a file once while hashing it)
and accepts only BF16 matrices whose extents tile the data exactly; names and
metadata are the caller's to check.

CPU-only and dependency-free beyond torch, which only the tensor functions
import: the trainer's control plane validates hashes without it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_SHA256 = re.compile(r"[0-9a-f]{64}")
_READ_CHUNK = 64 * 1024 * 1024


def check_sha256(value, what: str) -> str:
    """``value`` if it is 64 lowercase hex digits; ``what`` names it in the refusal."""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{what} sha256 must be 64 lowercase hex digits")
    return value


def _header(tensors: Sequence[tuple[str, torch.Tensor]], metadata: Mapping[str, str], kind: str) -> bytes:
    import torch

    header: dict = {"__metadata__": dict(metadata)}
    offset = 0
    for name, value in tensors:
        if value.dtype != torch.bfloat16 or value.ndim != 2:
            raise ValueError(f"{kind} {name} must be a BF16 matrix")
        size = 2 * value.numel()
        header[name] = {"dtype": "BF16", "shape": list(value.shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _payload(name: str, value: torch.Tensor, kind: str) -> memoryview:
    import torch

    value = value.detach()
    if value.device.type != "cpu":
        raise ValueError(f"{kind} {name} must be on the CPU to be written")
    return memoryview(value.contiguous().view(torch.uint8).numpy()).cast("B")


def serialize(tensors: Sequence[tuple[str, torch.Tensor]], metadata: Mapping[str, str], *, kind: str) -> bytes:
    """The canonical file of CPU BF16 matrices ``(name, tensor)`` in this order, in memory.

    ``kind`` names a tensor in refusals ("policy weight", "router").
    """
    return b"".join([_header(tensors, metadata, kind), *(_payload(name, value, kind) for name, value in tensors)])


def write_file(path: Path, tensors: Sequence[tuple[str, torch.Tensor]], metadata: Mapping[str, str], *,
               kind: str) -> str:
    """Write ``serialize``'s bytes to ``path``, hashed as they are written and made durable; their sha256."""
    head = _header(tensors, metadata, kind)
    digest = hashlib.sha256(head)
    with Path(path).open("wb") as handle:
        handle.write(head)
        for name, value in tensors:
            raw = _payload(name, value, kind)
            digest.update(raw)
            handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def read_verified(path: Path, sha256: str, *, what: str) -> bytearray:
    """The whole file, read once and hashed as it is read; refused unless its sha256 is ``sha256``."""
    expected = check_sha256(sha256, what)
    path = Path(path)
    size = path.stat().st_size
    data = bytearray(size)
    view = memoryview(data)
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb", buffering=0) as handle:
        while offset < size:
            count = handle.readinto(view[offset : offset + _READ_CHUNK])
            if not count:
                raise ValueError(f"{path} shrank while it was read")
            digest.update(view[offset : offset + count])
            offset += count
        if handle.read(1):
            raise ValueError(f"{path} grew while it was read")
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{path} has sha256 {actual}, expected {expected}")
    return data


def parse(data: bytes | bytearray, *, source) -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    """``(metadata, tensors)`` of a file's bytes: zero-copy BF16 matrices in header order.

    Refused unless the header is a JSON object with string ``__metadata__`` and
    every other entry is exactly a BF16 dtype, a positive 2-D shape and data
    offsets of its size, the extents tiling the data after the header with no
    gap or overlap. Immutable ``bytes`` are copied once so the views are writable.
    """
    import torch

    if not isinstance(data, bytearray):
        data = bytearray(data)
    if len(data) < 8:
        raise ValueError(f"{source} is truncated")
    (length,) = struct.unpack("<Q", bytes(data[:8]))
    if length > len(data) - 8:
        raise ValueError(f"{source} header is truncated")
    try:
        header = json.loads(bytes(data[8 : 8 + length]))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} header is not JSON") from error
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        raise ValueError(f"{source} has no metadata")
    metadata = header.pop("__metadata__")
    if any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError(f"{source} metadata must map names to strings")
    body = 8 + length
    layouts = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"} or entry["dtype"] != "BF16":
            raise ValueError(f"{source}: {name} must be stored as BF16")
        shape, offsets = entry["shape"], entry["data_offsets"]
        if (
            not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
            or not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets)
            or offsets[1] - offsets[0] != 2 * shape[0] * shape[1]
        ):
            raise ValueError(f"{source}: {name} has an invalid shape or extent")
        layouts[name] = (shape, offsets)
    position = 0
    for (start, end), name in sorted((offsets, name) for name, (_, offsets) in layouts.items()):
        if start != position:
            raise ValueError(f"{source}: {name} does not follow the previous tensor")
        position = end
    if body + position != len(data):
        raise ValueError(f"{source}: tensor data does not cover the file exactly")
    tensors = {
        name: torch.frombuffer(data, dtype=torch.bfloat16, count=shape[0] * shape[1],
                               offset=body + offsets[0]).view(shape[0], shape[1])
        for name, (shape, offsets) in layouts.items()
    }
    return metadata, tensors
