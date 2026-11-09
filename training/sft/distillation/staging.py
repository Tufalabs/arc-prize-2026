"""Stage safetensor shards on local scratch with independent copy verification.

Model provenance remains the canonical source path. A loader may use the returned
local directory only for Reader shard access, after validate_stage succeeds.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def identity(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def inventory(source):
    index = source / "model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    for name in names:
        if Path(name).name != name:
            raise ValueError("Shard must be a basename")
    return index, names


def copy_verified(source, destination):
    before = identity(source)
    h = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with source.open("rb") as src, temporary.open("xb") as dst:
        for b in iter(lambda: src.read(8 * 1024**2), b""):
            h.update(b)
            dst.write(b)
        dst.flush()
        os.fsync(dst.fileno())
    if identity(source) != before or sha(temporary) != h.hexdigest():
        raise ValueError("Source changed or copied payload mismatch")
    temporary.replace(destination)
    return {"source_identity": before, "sha256": h.hexdigest(), "bytes": before[0]}


def stage(source, destination, workers=8):
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    index, names = inventory(source)
    index_hash = sha(index)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            pool.map(
                lambda name: copy_verified(source / name, destination / name), names
            )
        )
    if sha(index) != index_hash:
        raise ValueError("Source index changed during staging")
    manifest = {
        "source": str(source),
        "index_sha256": index_hash,
        "shards": dict(zip(names, records)),
    }
    (destination / "STAGED.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_stage(source, destination, full=False):
    source = source.resolve()
    manifest = json.loads((destination / "STAGED.json").read_text())
    index, names = inventory(source)
    if manifest["source"] != str(source) or manifest["index_sha256"] != sha(index):
        raise ValueError("Stage provenance differs")
    if sorted(manifest["shards"]) != names:
        raise ValueError("Stage shard inventory differs")
    for name, record in manifest["shards"].items():
        if identity(source / name) != record["source_identity"]:
            raise ValueError("Source shard changed")
        local = destination / name
        if local.stat().st_size != record["bytes"] or (
            full and sha(local) != record["sha256"]
        ):
            raise ValueError("Local shard corrupted")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.destination)
    validate_stage(args.source, args.destination, full=True)
    print(
        json.dumps(
            {
                "status": "staged_and_verified",
                "source": str(args.source),
                "destination": str(args.destination),
            }
        )
    )
