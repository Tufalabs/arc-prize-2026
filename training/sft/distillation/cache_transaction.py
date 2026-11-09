"""Commit rank snapshots only after every rank is durable; reject corruption."""

import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def initialize(root, identity):
    path = Path(root) / "cache-identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Cache identity differs")
    else:
        if (
            any(Path(root).glob("*.safetensors"))
            or (Path(root) / "cache-progress.json").exists()
        ):
            raise ValueError("Unidentified partial cache cannot be adopted")
        atomic_json(path, identity)


def committed(root, identity):
    initialize(root, identity)
    path = Path(root) / "cache-progress.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state["identity"] != identity or len(state["states"]) != identity["world_size"]:
        raise ValueError("Invalid cache commit identity or rank count")
    for rank, record in enumerate(state["states"]):
        name = f"state-b{state['batches']}-rank{rank}.pt"
        if record["name"] != name or digest(Path(root) / name) != record["sha256"]:
            raise ValueError("Committed rank snapshot corrupted")
    return state


def commit(root, identity, batches):
    old = committed(root, identity)
    expected = 1 if old is None else old["batches"] + 1
    if batches != expected:
        raise ValueError("Non-consecutive cache commit")
    states = []
    for rank in range(identity["world_size"]):
        path = Path(root) / f"state-b{batches}-rank{rank}.pt"
        states.append({"name": path.name, "sha256": digest(path)})
    value = {"identity": identity, "batches": batches, "states": states}
    atomic_json(Path(root) / "cache-progress.json", value)
    return value


def atomic_torch_save(value, path):
    import torch

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
