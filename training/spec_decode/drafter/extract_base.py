"""Write a Flash-Next checkpoint's MTP drafter as a checkpoint of its own.

When a speculative config names no separate model, vLLM loads the drafter
from the target's checkpoint by iterating every shard and keeping the
tensors it recognises: the MTP block, the head and the embedding, 33
tensors, found by reading the whole checkpoint a second time while the
target is already resident. Beside the pinned PLE table that second pass
is what a 160 GB allocation cannot cover (one step killed at 162 GB RSS).

This writes those tensors as one file into a sibling directory with the
config and tokenizer files beside it, for ``speculative_config["model"]``.
The head keeps every expert it has; a pruned checkpoint that left its MTP
block whole declares the block's count as ``text_config.mtp_num_experts``,
and where the config lacks the key it is written from the block's own
router rows, which is what the fork's drafter reads.

Usage: python -m drafter.extract_base --src CHECKPOINT [--out DIR]   (default: <CHECKPOINT>-drafter)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from drafter.files import save_file

# What the drafter loads besides ``mtp.*``: the head and the embedding it
# shares with the target (``mtp_use_dedicated_embeddings`` is false).
DRAFTER_SHARED_KEYS = ("lm_head.weight", "model.language_model.embed_tokens.weight")
MTP_ROUTER_KEY = "mtp.layers.0.mlp.gate.weight"
# What serving reads beside the weights, and the record of the target's prune.
# A converted or pruned source also holds its own conversion's markers,
# metrics and reports, which describe that checkpoint and not the head, and a
# ModelOpt record (hf_quant_config.json) the head's own quantization outdates.
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "pruning.json",
)


def main() -> None:
    """Gather the drafter's tensors into one file; copy the config and tokenizer files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="the checkpoint the target serves from")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.src.with_name(args.src.name + "-drafter")

    weight_map: dict[str, str] = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = [k for k in weight_map if k.startswith("mtp.")] + list(DRAFTER_SHARED_KEYS)
    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        by_shard.setdefault(weight_map[key], []).append(key)
    out.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        # safetensors ships no type information; the handle is Any by choice.
        reader = cast("Any", safe_open(args.src / shard, "pt"))
        with reader as f:
            for key in keys:
                tensors[key] = cast("torch.Tensor", f.get_tensor(key))
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})

    for name in COPIED_FILES:
        if (args.src / name).is_file():
            shutil.copy2(args.src / name, out / name)
    cfg = json.loads((out / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    mtp_experts = int(tensors[MTP_ROUTER_KEY].shape[0])
    if int(text.get("mtp_num_experts") or 0) != mtp_experts:
        text["mtp_num_experts"] = mtp_experts
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"drafter: {len(tensors)} tensors, {size / 2**30:.2f} GiB, {mtp_experts} experts -> {out}")


if __name__ == "__main__":
    main()
