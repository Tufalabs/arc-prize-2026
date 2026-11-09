"""Cut routed experts out of the Flash-Next NVFP4 checkpoint, per a blank plan.

What a physical prune is, on this checkpoint: every decoder layer's MoE block
has a router ``mlp.gate.weight`` of shape ``[num_experts, hidden]`` and, per
expert, twelve tensors under ``mlp.experts.<i>.`` (three projections, each a
packed NVFP4 weight with its block scales and two scalars). Removing an
expert is deleting its twelve tensors and its router row; the survivors are
renumbered to a contiguous ``0..k-1`` because the loader iterates
``experts.<i>`` for ``i in range(config.num_experts)``. Nothing else in the
checkpoint has an expert dimension (verified: the only other 512-wide
tensors are attention projections), and the shared expert and its gate are
untouched. ``text_config.num_experts`` is set to the survivor count.

The plan's kept lists are ascending, so a survivor's new index is its rank
among the kept, and the router row slice ``weight[kept]`` lands in that same
order -- the two renumberings agree by construction.

Streams shard by shard through safetensors' zero-copy reader and writes few
large shards: the output lives on NFS and is read by every worker, and a
handful of big sequential files is what that serves well. Copies serving
configs and tokenizer/processor files, and writes ``pruning.json`` with
the plan's provenance beside it. Anything MTP is copied untouched: the MTP
block keeps its 512 experts, declared as ``text_config.mtp_num_experts``
so the draft model sizes its MoE by them (a fork reading; upstream would
index the decoder's count into the block's rows) -- flagged in the summary.

Usage: prune_flashnext_experts.py --src SNAPSHOT --plan PLAN.json --out DIR [--shard-gib 10]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from safetensors import safe_open

# safetensors ships no type information, so its writer and the handle its
# reader yields are both Unknown to the checker. Named once, here.
from safetensors.torch import save_file as _raw_save_file  # pyright: ignore[reportUnknownVariableType]

_save_file = cast("Callable[..., None]", _raw_save_file)

# Preserve serving metadata, including ModelOpt's companion configuration.
# Conversion markers, metrics and reports describe the source, not this prune.
COPIED_FILES = (
    "generation_config.json", "hf_quant_config.json", "chat_template.jinja",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json",
)


def save_file(tensors: dict[str, torch.Tensor], path: str, metadata: dict[str, str]) -> None:
    """Write weights with the same umask-derived mode as the serving configs."""
    _save_file(tensors, path, metadata=metadata)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(path, 0o666 & ~umask)


class _Reader(Protocol):
    """The part of ``safetensors.safe_open``'s handle this tool uses."""

    def keys(self) -> Iterable[str]:
        """Every tensor name in the shard."""
        ...

    def get_tensor(self, name: str) -> torch.Tensor:
        """Read one tensor whole."""
        ...

    def get_slice(self, name: str) -> Any:
        """A lazy handle whose shape can be read without loading the data."""
        ...


@contextlib.contextmanager
def open_shard(path: Path) -> Generator[_Reader]:
    """Open a safetensors shard for reading, with the handle's type named."""
    handle: Any
    with safe_open(str(path), framework="pt") as handle:  # pyright: ignore[reportUnknownVariableType] — same
        yield cast(_Reader, handle)


EXPERT_RE = re.compile(r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.)experts\.(?P<idx>\d+)\.(?P<rest>.+)$")
GATE_RE = re.compile(r"^.*\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
DECODER = "language_model.layers."


def main() -> None:
    """Cut the plan's experts out of the source checkpoint and verify the result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-gib", type=float, default=10.0)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    kept = {int(k): list(v) for k, v in plan["kept"].items()}
    assert all(v == sorted(v) for v in kept.values()), "kept lists must be ascending"
    new_index = {lay: {old: new for new, old in enumerate(v)} for lay, v in kept.items()}
    n_kept = {len(v) for v in kept.values()}
    assert len(n_kept) == 1, f"uniform plans only, got per-layer counts {sorted(n_kept)}"
    num_kept = n_kept.pop()
    num_src = int(plan["num_experts"])

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    args.out.mkdir(parents=True, exist_ok=False)

    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes: int = 0
    out_files: list[Path] = []
    weight_map: dict[str, str] = {}
    total: int = 0
    dropped = 0
    renamed = 0
    gates = 0
    mtp_untouched = 0
    limit = int(args.shard_gib * 2**30)

    def flush() -> None:
        nonlocal buffer, buffer_bytes
        if not buffer:
            return
        name = f"model-{len(out_files) + 1:05d}.safetensors"
        save_file(buffer, str(args.out / name), {"format": "pt"})
        for k in buffer:
            weight_map[k] = name
        out_files.append(args.out / name)
        buffer, buffer_bytes = {}, 0

    for shard in shards:
        with open_shard(args.src / shard) as f:
            for key in f.keys():
                m = EXPERT_RE.match(key)
                if m and DECODER in key:
                    lay, idx = int(m["layer"]), int(m["idx"])
                    if idx not in new_index[lay]:
                        dropped += 1
                        continue
                    new_key = f"{m['prefix']}experts.{new_index[lay][idx]}.{m['rest']}"
                    tensor = f.get_tensor(key)
                    renamed += 1
                elif (g := GATE_RE.match(key)) and DECODER in key:
                    lay = int(g["layer"])
                    tensor = f.get_tensor(key)
                    assert tensor.shape[0] == num_src, (key, tensor.shape)
                    tensor = tensor[torch.tensor(kept[lay])].contiguous()
                    new_key = key
                    gates += 1
                else:
                    if ".mlp.experts." in key or key.endswith(".mlp.gate.weight"):
                        mtp_untouched += 1
                    tensor = f.get_tensor(key)
                    new_key = key
                buffer[new_key] = tensor
                buffer_bytes += tensor.numel() * tensor.element_size()
                total += tensor.numel() * tensor.element_size()
                if buffer_bytes >= limit:
                    flush()
        print(f"  {shard}: done (out shards so far {len(out_files)})", flush=True)
    flush()

    # Final names carry the shard count, as the Hub layout does.
    n = len(out_files)
    final = {}
    for i, p in enumerate(out_files, 1):
        q = args.out / f"model-{i:05d}-of-{n:05d}.safetensors"
        p.rename(q)
        final[p.name] = q.name
    weight_map = {k: final[v] for k, v in weight_map.items()}
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )

    for name in COPIED_FILES:
        p = args.src / name
        if p.is_file():
            shutil.copy2(p, args.out / p.name)
    cfg = json.loads((args.src / "config.json").read_text())
    assert cfg["text_config"]["num_experts"] == num_src
    cfg["text_config"]["num_experts"] = num_kept
    if mtp_untouched:
        # The MTP block keeps every source expert; the draft model reads this
        # count for it, where the decoder's count would index past its rows.
        cfg["text_config"]["mtp_num_experts"] = int(cfg["text_config"].get("mtp_num_experts") or num_src)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    config_sha = hashlib.sha256((args.out / "config.json").read_bytes()).hexdigest()
    (args.out / "pruning.json").write_text(json.dumps({**plan, "config_sha256": config_sha, "shards": n}, indent=1))

    # Verify by re-reading what was written, not by trusting the counters.
    seen_experts: dict[int, set[int]] = {}
    gate_shapes: dict[int, tuple[int, ...]] = {}
    n_tensors = 0
    for q in sorted(args.out.glob("model-*.safetensors")):
        with open_shard(q) as f:
            for key in f.keys():
                n_tensors += 1
                m = EXPERT_RE.match(key)
                if m and DECODER in key:
                    seen_experts.setdefault(int(m["layer"]), set()).add(int(m["idx"]))
                elif (g := GATE_RE.match(key)) and DECODER in key:
                    gate_shapes[int(g["layer"])] = tuple(f.get_slice(key).get_shape())
    bad = [lay for lay, s in seen_experts.items() if s != set(range(num_kept))]
    bad_gates = [lay for lay, s in gate_shapes.items() if s[0] != num_kept]
    assert not bad and not bad_gates, (bad, bad_gates)
    assert len(seen_experts) == len(kept) == len(gate_shapes)
    print(
        f"wrote {n} shards, {total / 2**30:.2f} GiB, {n_tensors} tensors | experts dropped {dropped}, kept+renumbered {renamed}, "
        f"routers sliced {gates} | non-decoder MoE tensors copied untouched: {mtp_untouched} | config sha256 {config_sha}"
    )


if __name__ == "__main__":
    main()
