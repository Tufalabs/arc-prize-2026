"""The trainer's head against the served drafter, on identical inputs.

A cache extracted with ``--drafter DIR`` holds, beside the target's stream,
that drafter's pass-0 single stream at every position as vLLM computed it
(``draft_single``). Pass 0 of the trainer's head over the same chain -- the
same stream, the same next tokens, the same positions -- must reproduce it:
compared per slot as the single stream's relative error, and as the draft
distributions at T=0.6 (KL, overlap sum min(q_vllm, q_ours), top-1
agreement). The last position is left out: vLLM's drafter reads the token it
just sampled there, which the chain does not hold.

    python -m drafter.parity CACHE_DIR DATASET_DIR [--head PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def main(argv: Optional[list[str]] = None) -> None:
    from safetensors.torch import load_file

    from drafter.head import LabelHead, load_head, load_trained
    from drafter.train import Trainer, load, load_chains

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--head", type=Path, default=None, help="trained head.safetensors; default the checkpoint's")
    parser.add_argument("--eager-attention", action="store_true")
    parser.add_argument("--kv-fp8", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.steps = 1
    args.checkpoint_steps, args.ce_weight = False, 0.0

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads(sorted(args.cache.glob("manifest*.json"))[0].read_text())
    print(f"[parity] cache drafter: {manifest.get('drafter')}; head: {args.head or 'checkpoint'}", flush=True)
    label = LabelHead(device=device)
    head = load_head(device=device)
    if args.head is not None:
        load_trained(head, args.head)
    trainer = Trainer(head, label, args, optimize=False)
    report = {}
    for data in load_chains(args.cache, args.dataset):
        chain = load(data, device)
        served = load_file(str(data.path))["draft_single"].to(device)[:-1]
        keep: dict[str, torch.Tensor] = {}
        trainer.chain_loss(chain, train=False, keep=keep)
        ours = torch.cat([head.sample_stream(keep["multi"][i : i + 4096]) for i in range(0, len(keep["multi"]), 4096)])
        error = (ours.float() - served.float()).norm(dim=-1) / served.float().norm(dim=-1)
        sums = {"kl": 0.0, "overlap": 0.0, "top1": 0.0}
        for start in range(0, len(ours), 256):
            q_served = F.linear(served[start : start + 256], label.lm_head).float() / 0.6
            q_ours = F.linear(ours[start : start + 256], label.lm_head).float() / 0.6
            log_served, log_ours = q_served.log_softmax(-1), q_ours.log_softmax(-1)
            sums["kl"] += float((log_served.exp() * (log_served - log_ours)).sum())
            sums["overlap"] += float(torch.minimum(log_served.exp(), log_ours.exp()).sum())
            sums["top1"] += float((q_served.argmax(-1) == q_ours.argmax(-1)).sum())
        rows = len(ours)
        report[data.chain_id] = {
            "rows": rows,
            "rel_error_median": float(error.median()),
            "rel_error_p99": float(error.quantile(0.99)),
            **{key: value / rows for key, value in sums.items()},
        }
        print(f"[parity] {data.chain_id}: {report[data.chain_id]}", flush=True)
    (args.cache / f"parity_{args.head.parent.name if args.head else 'checkpoint'}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
