"""Re-shard a checkpoint into small shards, in the order the engine loads them.

The engine's loader reads one shard at a time, in natural file-name order
(``safetensors_weights_iterator``), and on a Kaggle kernel the bounded
prefetch (``ullm.kaggle_prefetch``) reads whole shards a window ahead of it.
The e307 checkpoint's ten shards are 10.7 GB each, so a 24 GiB window holds
two, and a decoder layer's tensors are spread over two of them (layer 0 in
shards 1 and 4). Here:

1. every tensor outside the decoder, the PLE table and the MTP block (the
   embeddings, the LM head, the vision tower, the hyper-connection mixer)
   comes first, packed to ``--pack-gib``;
2. then every decoder layer is a shard of its own, in layer order (~0.93 GiB
   on e307), its PLE projections with it;
3. then the PLE table's pieces, in piece order, packed to ``--pack-gib``;
4. then the MTP block, packed alike and last: every cell drafts from its
   own drafter checkpoint and the loader reads it for nothing, so it sits
   apart; ``--drop-mtp`` leaves it out.

The prune tool writes few large shards because the cluster's NFS serves big
sequential files well; this layout is for a kernel's loader and its prefetch.

Bytes are copied, not re-encoded: each tensor's dtype, shape and data are the
source's, written in safetensors' own layout (a header padded to 8 bytes,
then the tensors by element size, largest first, so each starts aligned).
The output is read back and compared tensor by tensor against the source
(a digest per tensor, taken as it is copied). Every other file is copied
verbatim, and ``resharding.json`` records the source and the layout.

Usage: reshard_checkpoint.py --src DIR --out DIR [--pack-gib 4.5] [--drop-mtp]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX = "model.safetensors.index.json"
CHUNK = 64 << 20
ITEMSIZE = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
PLE_TABLE_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")


@dataclass
class Tensor:
    """One tensor of the source: where its bytes are, and their digest once copied."""

    name: str
    dtype: str
    shape: list[int]
    source: Path
    offset: int  # absolute, in the source file
    nbytes: int
    digest: str = ""


def read_header(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
    """A shard's tensors, its metadata, and where its data starts."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    metadata = header.pop("__metadata__", {})
    return header, metadata, 8 + length


def section(name: str) -> tuple[int, int]:
    """Where a tensor goes: (section, key within it). Sections in load order:
    0 the rest, 1 a decoder layer (keyed by its index), 2 the PLE table
    (keyed by piece), 3 the MTP block."""
    if name.startswith("mtp."):
        return 3, 0
    if piece := PLE_TABLE_RE.search(name):
        return 2, int(piece.group(1))
    if layer := LAYER_RE.match(name):
        return 1, int(layer.group(1))
    return 0, 0


def plan(tensors: list[Tensor], pack: int) -> list[list[Tensor]]:
    """The output shards, in load order: one per decoder layer, the other
    sections packed in order up to ``pack`` bytes."""
    ordered = sorted(tensors, key=lambda t: (*section(t.name), t.name))
    shards: list[list[Tensor]] = []
    current: list[Tensor] = []
    current_key: tuple[int, int] | None = None
    size = 0
    for tensor in ordered:
        sec, key = section(tensor.name)
        group = (sec, key) if sec == 1 else (sec, -1)
        if current and (group != current_key or (sec != 1 and size + tensor.nbytes > pack)):
            shards.append(current)
            current, size = [], 0
        current.append(tensor)
        current_key = group
        size += tensor.nbytes
    if current:
        shards.append(current)
    return shards


def write_shard(path: Path, tensors: list[Tensor], metadata: dict[str, str]) -> int:
    """Write ``tensors`` as one safetensors file, copying each one's bytes
    from its source and recording their digest. Returns the data size."""
    tensors = sorted(tensors, key=lambda t: (-ITEMSIZE[t.dtype], t.name))
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for t in tensors:
        header[t.name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.nbytes]}
        offset += t.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for t in tensors:
            digest = hashlib.blake2b(digest_size=16)
            with open(t.source, "rb") as src:
                src.seek(t.offset)
                left = t.nbytes
                while left:
                    block = src.read(min(CHUNK, left))
                    if not block:
                        raise RuntimeError(f"{t.source} ends inside {t.name}")
                    digest.update(block)
                    out.write(block)
                    left -= len(block)
            t.digest = digest.hexdigest()
    return offset


def verify(out: Path, shards: dict[str, list[Tensor]]) -> int:
    """Read every written shard back and compare each tensor's dtype, shape
    and bytes with what was copied into it."""
    checked = 0
    for name, tensors in shards.items():
        header, _, start = read_header(out / name)
        expected = {t.name: t for t in tensors}
        if set(header) != set(expected):
            raise RuntimeError(f"{name}: tensor names differ from the plan")
        with open(out / name, "rb") as f:
            for tensor_name, entry in header.items():
                t = expected[tensor_name]
                begin, end = (int(x) for x in entry["data_offsets"])
                if entry["dtype"] != t.dtype or entry["shape"] != t.shape or end - begin != t.nbytes:
                    raise RuntimeError(f"{name}: {tensor_name} header differs from the source's")
                f.seek(start + begin)
                digest = hashlib.blake2b(digest_size=16)
                left = t.nbytes
                while left:
                    block = f.read(min(CHUNK, left))
                    digest.update(block)
                    left -= len(block)
                if digest.hexdigest() != t.digest:
                    raise RuntimeError(f"{name}: {tensor_name} bytes differ from the source's")
                checked += 1
    return checked


def main() -> None:
    """Plan the shards, copy every tensor, write the index, verify against the source."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pack-gib", type=float, default=4.5)
    ap.add_argument("--drop-mtp", action="store_true", help="leave the MTP block (section 3) out")
    args = ap.parse_args()
    start = time.monotonic()

    index = json.loads((args.src / INDEX).read_text())
    tensors: list[Tensor] = []
    metadata: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        header, metadata, data_start = read_header(args.src / shard)
        for name, entry in header.items():
            if index["weight_map"].get(name) != shard:
                raise RuntimeError(f"{name} is in {shard} but the index says {index['weight_map'].get(name)}")
            begin, end = (int(x) for x in entry["data_offsets"])
            dtype = str(entry["dtype"])
            if dtype not in ITEMSIZE:
                raise RuntimeError(f"{name}: dtype {dtype} has no known element size")
            shape = [int(d) for d in entry["shape"]]
            tensors.append(Tensor(name, dtype, shape, args.src / shard, data_start + begin, end - begin))
    if len(tensors) != len(index["weight_map"]):
        raise RuntimeError("the index names tensors the shards do not hold")
    expected = set(index["weight_map"])
    if args.drop_mtp:
        tensors = [t for t in tensors if section(t.name)[0] != 3]
        expected = {t.name for t in tensors}

    layout = plan(tensors, int(args.pack_gib * 2**30))
    count = len(layout)
    names = [f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)]
    args.out.mkdir(parents=True, exist_ok=False)
    weight_map: dict[str, str] = {}
    total = 0
    for name, shard_tensors in zip(names, layout, strict=True):
        size = write_shard(args.out / name, shard_tensors, metadata)
        total += size
        for t in shard_tensors:
            weight_map[t.name] = name
        sections = sorted({section(t.name)[0] for t in shard_tensors})
        print(f"  {name}: {len(shard_tensors)} tensors, {size / 2**30:.2f} GiB, sections {sections}", flush=True)

    out_index = {"metadata": {**index.get("metadata", {}), "total_size": total}, "weight_map": weight_map}
    (args.out / INDEX).write_text(json.dumps(out_index, indent=2))
    for path in sorted(args.src.iterdir()):
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != INDEX:
            shutil.copy2(path, args.out / path.name)

    checked = verify(args.out, dict(zip(names, layout, strict=True)))
    if checked != len(tensors) or set(weight_map) != expected:
        raise RuntimeError(f"verified {checked} tensors of {len(tensors)}")
    source_bytes = sum(t.nbytes for t in tensors)
    if total != source_bytes:
        raise RuntimeError(f"wrote {total} bytes of tensor data, the source holds {source_bytes}")

    record = {
        "source": str(args.src),
        "source_index_sha256": hashlib.sha256((args.src / INDEX).read_bytes()).hexdigest(),
        "rule": "rest, then one shard per decoder layer, then the PLE table, then MTP; packed to pack_gib",
        "pack_gib": args.pack_gib,
        "mtp_dropped": args.drop_mtp,
        "tensors": len(tensors),
        "tensor_bytes": total,
        "shards": [
            {"file": name, "bytes": sum(t.nbytes for t in ts), "sections": sorted({section(t.name)[0] for t in ts})}
            for name, ts in zip(names, layout, strict=True)
        ],
    }
    (args.out / "resharding.json").write_text(json.dumps(record, indent=1))
    print(
        f"{count} shards, {total / 2**30:.2f} GiB, {checked} tensors verified against the source, "
        f"{time.monotonic() - start:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
