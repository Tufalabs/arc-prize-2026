"""Offline acceptance of heads on cached sets, side by side.

Each head (``base``: the drafter checkpoint as shipped; a drafter directory,
NVFP4 experts included; else a trained ``head.safetensors``) runs the
trainer's evaluation over each set: per draft step, KL, acceptance on raw
logits and as the served sampler verifies (``accept_served``), per harness.
``--nvfp4 NAME`` rounds that head's routed experts to NVFP4 and back first,
as Marlin would serve them quantized (``drafter.nvfp4``).

``--min-position N`` scores only the positions from N of each chain on. A
head trained on windows (``DRAFTER_MAX_TOKENS``) attends, past the first
window, over less than the served head does, which holds the whole chain:
the same positions scored with ``DRAFTER_MAX_TOKENS`` set and unset measure
what that costs (unset needs the whole chain on the card, which evaluation,
with no backward, has room for).

    python -m drafter.evaluate OUT_JSON --set NAME=CACHE_DIR:DATASET_DIR [...] --head NAME=PATH|base [...] [--nvfp4 NAME] [--steps 3] [--min-position N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch


def main(argv: Optional[list[str]] = None) -> None:
    from drafter import nvfp4
    from drafter.head import LabelHead, load_head, load_trained
    from drafter.train import Trainer, evaluate, load_chains, scored_from

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--set", action="append", required=True, dest="sets")
    parser.add_argument("--head", action="append", required=True, dest="heads")
    parser.add_argument("--nvfp4", action="append", default=[], help="a head name to evaluate with NVFP4 experts")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--min-position", type=int, default=0, help="score only positions from this one of a chain on")
    parser.add_argument("--eager-attention", action="store_true")
    parser.add_argument("--kv-fp8", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.checkpoint_steps, args.ce_weight = False, 0.0

    started = time.monotonic()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    label = LabelHead(device=device)
    sets = {}
    for spec in args.sets:
        name, paths = spec.split("=", 1)
        cache, dataset = paths.split(":", 1)
        chains = load_chains(Path(cache), Path(dataset))
        if args.min_position:
            kept = [scored_from(chain, args.min_position) for chain in chains]
            chains = [chain for chain in kept if chain is not None]
            print(f"[evaluate] {name}: {len(chains)} pieces reach position {args.min_position:,}", flush=True)
        sets[name] = chains
    results: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.heads:
        name, path = spec.split("=", 1)
        if path != "base" and Path(path).is_dir():
            head = load_head(Path(path), device=device)
        else:
            head = load_head(device=device)
            if path != "base":
                load_trained(head, Path(path))
        if name in args.nvfp4:
            nvfp4.fake_quantize(head)
        trainer = Trainer(head, label, args, optimize=False)
        results[name] = {set_name: evaluate(trainer, chains, device) for set_name, chains in sets.items()}
        del trainer, head
        torch.cuda.empty_cache()
        print(f"[evaluate] {name} done at {time.monotonic() - started:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for set_name in sets:
        for group in ("all", "arc2", "arc3"):
            for name in ("t0.6", "served"):
                row = []
                for head_name in results:
                    values = [
                        results[head_name][set_name].get(f"{group}/drafts_{name}/pos{s}") for s in range(args.steps)
                    ]
                    row.append(f"{head_name} " + "/".join("-" if v is None else f"{v:.3f}" for v in values))
                print(f"[evaluate] {set_name} {group:4s} per draft {name:7s} " + "   ".join(row), flush=True)
        for group in ("all", "arc2", "arc3"):
            for key in ("ceiling_served", "accept_served", "accept_t0.6", "reach_t0.6", "reach_served", "kl"):
                row = []
                for head_name in results:
                    values = [results[head_name][set_name].get(f"{group}/step{s}/{key}") for s in range(args.steps)]
                    row.append(f"{head_name} " + "/".join("-" if v is None else f"{v:.3f}" for v in values))
                print(f"[evaluate] {set_name} {group:4s} {key:14s} " + "   ".join(row), flush=True)


if __name__ == "__main__":
    main()
