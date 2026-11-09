"""Probe pruning plans on held-out high-scoring rollouts (GPU, Flash runtime).

Usage:
    ARC3_RUN_DIR=<out> sbatch scripts/flash_dev.sbatch scripts/prune_probe_plans.py \\
        --checkpoint <512-expert dir> --stats <route stats dir> --plans <plans dir> \\
        --out <out> --holdout <regex> [--min-score 0] [--shard K --shards N]

For every held-out rollout that scored above ``--min-score``, replays its
chains as served, images included (``training.flash_route_stats.ServedReplay``),
once with the full model and once per plan (the plan's pruned experts
masked out of every router, which routes as the pruned checkpoint would) and
records the summed log-probability of the replies the rollout actually
sampled. A plan that keeps what those winning rollouts relied on loses little
log-likelihood on them. Writes ``<out>/<run>.json`` per rollout (resumable) and,
from shard 0, nothing more: ``--summarize`` over the directory prints the
per-plan mean drop in log-probability per reply token.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(out: Path) -> None:
    runs = [json.loads(path.read_text()) for path in sorted(out.glob("*.json")) if path.name != "summary.json"]
    if not runs:
        raise SystemExit("no probed rollouts")
    plans = sorted({name for run in runs for name in run["plans"]} - {"full"})
    table = {}
    for name in plans:
        drops = [
            (run["plans"]["full"]["sum_logprob"] - run["plans"][name]["sum_logprob"]) / run["plans"]["full"]["tokens"]
            for run in runs
            if name in run["plans"]
        ]
        table[name] = {"rollouts": len(drops), "mean_nll_increase_per_token": sum(drops) / len(drops)}
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    for name, row in sorted(table.items(), key=lambda item: item[1]["mean_nll_increase_per_token"]):
        print(f"{name:28s} rollouts={row['rollouts']:3d} nll_increase/token={row['mean_nll_increase_per_token']:.5f}")


@torch.no_grad()
def reply_logprob(replay, chain, replies, images, chunk_tokens) -> tuple[float, int]:
    """Summed log-probability of a chain's replies, the chain replayed as served (its images included)."""
    from training.flash_route_stats import reply_targets

    model = replay.model
    total, count = 0.0, 0

    def on_chunk(start, hidden):
        nonlocal total, count
        pairs = reply_targets(start, hidden.shape[1], len(chain), replies)
        for low in range(0, len(pairs), 2048):
            part = pairs[low : low + 2048]
            rows = torch.tensor([row for row, _ in part], device=hidden.device)
            targets = torch.tensor([chain[position] for _, position in part], device=hidden.device)
            logits = model.lm_head(hidden[0, rows]).float()
            total += float((logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)).sum())
            count += len(part)

    replay(chain, images, chunk_tokens=chunk_tokens, on_chunk=on_chunk)
    return total, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-chain-tokens",
        type=int,
        default=0,
        help="score only each chain's first N tokens (0: all); bounds the 1 + plans passes per rollout",
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.out)
        return

    from training.flash_model import load_flash_text_model
    from training.flash_qsa_train import install_fast_qsa_indexer
    from training.flash_route_stats import RouterMask, ServedReplay, trace_chains, truncate_chain

    holdout = re.compile(args.holdout)
    runs = []
    for path in sorted(args.stats.glob("*.json")):
        if path.name.startswith("verify_"):
            continue
        record = json.loads(path.read_text())
        if holdout.search(record["source"]) and float(record["score"]) > args.min_score:
            runs.append(record)
    mine = [run for index, run in enumerate(runs) if index % args.shards == args.shard]
    plans = {path.stem: json.loads(path.read_text()) for path in sorted(args.plans.glob("*_k*.json"))}
    print(f"{len(runs)} held-out rollouts above {args.min_score}; shard replays {len(mine)} x {1 + len(plans)} passes", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_flash_text_model(args.checkpoint, device="cuda:0", progress=lambda m: print(m, flush=True))
    install_fast_qsa_indexer(model)
    mask = RouterMask(model)
    replay = ServedReplay(model, args.checkpoint)
    for run in mine:
        target = args.out / f"{run['run_id']}.json"
        if target.exists():
            continue
        chain_list = trace_chains(run["steps"])
        if args.max_chain_tokens:
            chain_list = [
                truncate_chain(chain, replies, images, args.max_chain_tokens, replay.image_token_id)
                for chain, replies, images in chain_list
            ]
        result = {}
        for name, plan in [("full", None), *plans.items()]:
            mask.set(plan)
            total, count = 0.0, 0
            for chain, replies, images in chain_list:
                part_total, part_count = reply_logprob(replay, chain, replies, images, args.chunk_tokens)
                total, count = total + part_total, count + part_count
            result[name] = {"sum_logprob": total, "tokens": count}
            print(f"{run['run_id']} {run['game_id']} {name}: {total / max(1, count):.5f}/token", flush=True)
        mask.set(None)
        record = {**{k: run[k] for k in ("run_id", "game_id", "score", "source")}, "max_chain_tokens": args.max_chain_tokens}
        target.write_text(json.dumps({**record, "plans": result}, indent=1))


if __name__ == "__main__":
    main()
