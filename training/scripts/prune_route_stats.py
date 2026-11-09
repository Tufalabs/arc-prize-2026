"""Replay scored Flash rollouts through the unpruned model and record routing.

Usage (one GPU per shard, inside the Flash runtime):
    ARC3_RUN_DIR=<out> sbatch scripts/flash_dev.sbatch scripts/prune_route_stats.py \\
        --checkpoint <512-expert NVFP4 dir> --rollouts <root of eval roots> --out <out> \\
        --shard K --shards N [--verify-tokens 12288]

Discovers every rollout under ``--rollouts`` (``*/*/official_eval_rollouts/
cycle_*/server_*/window_*/platypus``: ``benchmark.json`` for the ARC score,
``traces/<run>/steps.jsonl`` for the token ids) and in each ``--recording``
(a TAAF recording such as the deployed cells' long runs, ``DIR:PASSES`` for at
most that many passes per game), replays each rollout's chains as served,
their images through the vision tower (``training.flash_route_stats.ServedReplay``),
in chunks, and writes ``<out>/<run>.npz`` with
the per-layer, per-expert sums plus ``<out>/<run>.json`` (game, score, chain
lengths). Rollouts already written are skipped, so a shard resumes. Helpers
(``--shards 1 --claim --reverse``) join shards already running: they share
every rollout through exclusive claim files, highest run id first. With
``--verify-tokens`` it first checks, on one real chain, that the chunked replay
matches the served reply log-probs as well as a one-shot forward does and
yields the same per-expert aggregates (``verify_chunked``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.flash_model import load_flash_text_model  # noqa: E402
from training.flash_qsa_train import install_fast_qsa_indexer  # noqa: E402
from training.flash_route_stats import (  # noqa: E402
    RouteStats,
    ServedReplay,
    aggregate_agreement,
    chains_with_logprobs,
    install_route_recorder,
    replay_chain,
    trace_chains,
)


def discover(root: Path) -> list[dict]:
    """Every scored rollout with its trace, sorted by run id."""
    rollouts = []
    for window in sorted(root.glob("*/*/official_eval_rollouts/cycle_*/server_*/window_*/platypus")):
        benchmark = window / "benchmark.json"
        if not benchmark.exists():
            continue
        for run in json.loads(benchmark.read_text())["game_runs"]:
            steps = window / "traces" / run["run_id"] / "steps.jsonl"
            if not steps.exists():
                raise FileNotFoundError(f"scored rollout without its trace: {steps}")
            rollouts.append(
                {
                    "run_id": run["run_id"],
                    "game_id": str(run["game_id"]).split("#", 1)[0],
                    "score": float(run.get("final_score") or 0.0),
                    "levels_completed": int(run.get("levels_completed") or 0),
                    "steps": str(steps),
                    "source": str(window.relative_to(root)),
                }
            )
    return sorted(rollouts, key=lambda item: item["run_id"])


def _read_benchmark(recording: Path) -> dict | None:
    """A recording's ``benchmark.json``, or None while there is none or TAAF is
    rewriting it (it saves every ten minutes during a run and at its end)."""
    try:
        return json.loads((recording / "benchmark.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def recording_done(recording: Path) -> bool:
    """Whether a recording's run has ended (TAAF stamps ``end_time``)."""
    benchmark = _read_benchmark(recording)
    return bool(benchmark and benchmark.get("end_time"))


def discover_recording(recording: Path, passes: int | None = None, *, finished_only: bool = False) -> list[dict]:
    """A TAAF recording's rollouts (``benchmark.json`` beside ``traces/``, as
    the deployed cells' long runs and traffic recordings keep them), at most
    ``passes`` per game in the recording's order, so every game is covered.

    With ``finished_only`` the recording may still be running: its rollouts
    are the traces TAAF has sealed (``_SUCCESS``, written with the trace when
    a game ends), labelled from ``benchmark.json`` once one is saved."""
    import re

    benchmark = _read_benchmark(recording)
    if benchmark is None and not finished_only:
        raise FileNotFoundError(f"{recording}: no readable benchmark.json")
    runs = [run for run in (benchmark or {}).get("game_runs", []) if run.get("run_id")]
    if finished_only:
        by_id = {run["run_id"]: run for run in runs}
        traces = recording / "traces"
        sealed = (
            sorted(p.name for p in traces.iterdir() if (p / "_SUCCESS").exists() and (p / "steps.jsonl").exists())
            if traces.is_dir()
            else []
        )
        runs = [by_id.get(run_id, {"run_id": run_id}) for run_id in sealed]
    taken: dict[str, int] = {}
    rollouts = []
    for run in runs:
        # A clone of an official-110 entry (``ar25-0c556536_3``) is its game.
        game = re.sub(r"_\d+$", "", str(run["game_id"]).split("#", 1)[0]) if run.get("game_id") else ""
        if passes is not None and game and taken.get(game, 0) >= passes:
            continue
        steps = recording / "traces" / run["run_id"] / "steps.jsonl"
        if not steps.exists():
            raise FileNotFoundError(f"scored rollout without its trace: {steps}")
        taken[game] = taken.get(game, 0) + 1
        rollouts.append(
            {
                "run_id": run["run_id"],
                "game_id": game,
                "score": float(run.get("final_score") or 0.0),
                "levels_completed": int(run.get("levels_completed") or 0),
                "steps": str(steps),
                "source": recording.name,
            }
        )
    return sorted(rollouts, key=lambda item: item["run_id"])


def take_claim(path: Path, owner: str) -> bool:
    """Whether ``owner`` holds the rollout claimed by ``path``: it created the
    claim just now (exclusive create), or holds it from before a requeue."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        try:
            return path.read_text().strip() == owner
        except FileNotFoundError:
            return False
    with os.fdopen(descriptor, "w") as handle:
        handle.write(owner)
    return True


def assigned(run_id: str, shard: int, shards: int) -> bool:
    """Whether ``shard`` of ``shards`` replays ``run_id``: by a hash of the id,
    so the assignment holds while a followed recording grows."""
    return int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % shards == shard


def _recording_arg(value: str) -> tuple[Path, int | None]:
    """``DIR`` or ``DIR:PASSES``."""
    head, sep, tail = value.rpartition(":")
    if sep and tail.isdigit():
        return Path(head), int(tail)
    return Path(value), None


@torch.no_grad()
def _reply_logprobs(model, ids: list[int], targets: dict[int, float], pieces) -> dict[int, float]:
    """The replayed log-probability of each target token from ``(start,
    hidden)`` chunks (a target at p is predicted by the hidden state at p-1)."""
    values = {}
    for start, hidden in pieces:
        positions = [p for p in targets if start <= p - 1 < start + hidden.shape[1]]
        for low in range(0, len(positions), 1024):
            part = positions[low : low + 1024]
            rows = torch.tensor([p - 1 - start for p in part], device=hidden.device)
            chosen = torch.tensor([ids[p] for p in part], device=hidden.device)
            logits = model.lm_head(hidden[0, rows]).float()
            lp = logits.gather(-1, chosen[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            values.update(zip(part, lp.tolist()))
    return values


@torch.no_grad()
def verify_chunked(model, stats: RouteStats, ids: list[int], targets: dict[int, float], chunk_tokens: int) -> dict:
    """Chunked replay against one forward over the same ids, both scored
    against the serving engine's own reply log-probs, plus how alike their
    per-expert aggregates are. Chunking changes kernel shapes, so bf16
    rounding (and with it, borderline top-k choices) differs from the first
    layer on; what must hold is that the chunked replay is as faithful to the
    served model as one forward, and that pruning would read the same
    experts off both."""
    device = next(model.parameters()).device
    stats.reset()
    hidden = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state
    whole = stats.arrays()
    whole_lp = _reply_logprobs(model, ids, targets, [(0, hidden)])
    del hidden
    stats.reset()
    pieces = []
    replay_chain(model, ids, chunk_tokens=chunk_tokens, stats=stats, on_chunk=lambda s, h: pieces.append((s, h)))
    chunked = stats.arrays()
    chunked_lp = _reply_logprobs(model, ids, targets, pieces)
    stats.reset()
    common = sorted(set(whole_lp) & set(chunked_lp))
    if not common:
        raise RuntimeError("verification chain has no reply tokens with engine log-probs")

    def error(values):
        return sum(abs(values[p] - targets[p]) for p in common) / len(common)

    return {
        "tokens": len(ids),
        "chunk_tokens": chunk_tokens,
        "reply_tokens": len(common),
        "whole_lp_error": error(whole_lp),
        "chunked_lp_error": error(chunked_lp),
        **aggregate_agreement(whole, chunked, keep=307),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollouts", type=Path, default=None, help="root of eval roots (*/*/official_eval_rollouts/...)")
    parser.add_argument(
        "--recording",
        type=_recording_arg,
        action="append",
        default=[],
        help="a TAAF recording dir (benchmark.json + traces/), optionally DIR:PASSES per game; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--verify-tokens", type=int, default=0)
    parser.add_argument(
        "--max-lp-ratio",
        type=float,
        default=1.1,
        help="chunked replay's engine log-prob error may exceed one forward's by this factor (+0.01)",
    )
    parser.add_argument("--min-count-correlation", type=float, default=0.95)
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="accepted for older commands and ignored: a chain's images always go through the vision tower",
    )
    parser.add_argument(
        "--follow",
        type=float,
        default=0.0,
        help="seconds between rescans: replay each --recording's games as they finish, until every one has ended",
    )
    parser.add_argument(
        "--claim",
        action="store_true",
        help="take each rollout by an exclusive <out>/<run>.claim first, so helpers share one pool without repeats",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="replay the highest run ids first: a helper meets the forward shards instead of trailing them",
    )
    args = parser.parse_args()
    # Requeued jobs keep their id, so a requeued helper takes back its claims.
    worker = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    if not 0 <= args.shard < args.shards:
        raise ValueError("shard must be in [0, shards)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rollouts is None and not args.recording:
        raise ValueError("give --rollouts or at least one --recording")
    if args.follow and args.rollouts is not None:
        raise ValueError("--follow follows recordings; eval roots are complete when read")

    def pending() -> list[dict]:
        """This shard's rollouts not yet replayed."""
        rollouts = discover(args.rollouts) if args.rollouts is not None else []
        for recording, passes in args.recording:
            rollouts.extend(discover_recording(recording, passes, finished_only=bool(args.follow)))
        return sorted(
            (
                item
                for item in rollouts
                if assigned(item["run_id"], args.shard, args.shards)
                and not (args.out / f"{item['run_id']}.npz").exists()
            ),
            key=lambda item: item["run_id"],
            reverse=args.reverse,
        )

    mine = pending()
    print(f"shard {args.shard}/{args.shards}: {len(mine)} rollouts to replay now"
          f"{' (following)' if args.follow else ''}", flush=True)

    model = load_flash_text_model(args.checkpoint, device="cuda:0", progress=lambda m: print(m, flush=True))
    install_fast_qsa_indexer(model)
    layers, experts = len(model.model.layers), int(model.config.num_experts)

    stats = RouteStats.zeros(layers, experts, "cuda:0")
    install_route_recorder(model, stats)
    replay = ServedReplay(model, args.checkpoint)
    if args.verify_tokens and mine:
        ids, targets = max(chains_with_logprobs(mine[0]["steps"]), key=lambda chain: len(chain[0]))
        ids = ids[: args.verify_tokens]
        if replay.image_token_id in ids:
            # The one-forward reference is text-only.
            raise ValueError(f"--verify-tokens needs a chain that opens with {len(ids):,} tokens free of images")
        targets = {p: v for p, v in targets.items() if p < len(ids)}
        report = verify_chunked(model, stats, ids, targets, min(args.chunk_tokens, max(1024, len(ids) // 3)))
        print(f"chunked-vs-whole replay: {report}", flush=True)
        (args.out / f"verify_shard{args.shard}.json").write_text(json.dumps(report, indent=2))
        if (
            report["chunked_lp_error"] > args.max_lp_ratio * report["whole_lp_error"] + 0.01
            or report["min_count_correlation"] < args.min_count_correlation
        ):
            raise RuntimeError(f"chunked replay is less faithful than one forward: {report}")

    def replay_item(item: dict) -> None:
        target = args.out / f"{item['run_id']}.npz"
        if target.exists():
            return
        if args.claim and not take_claim(args.out / f"{item['run_id']}.claim", worker):
            return
        started = time.perf_counter()
        stats.reset()
        lengths = []
        images_before = replay.images
        for chain, _, image_paths in trace_chains(item["steps"]):
            replay(chain, image_paths, chunk_tokens=args.chunk_tokens, stats=stats, what=item["run_id"])
            lengths.append(len(chain))
        arrays = stats.arrays()
        # Per worker: a claimless shard may replay a rollout a helper holds.
        partial = target.with_suffix(f".{worker}.tmp.npz")
        np.savez_compressed(partial, sum_gnorm=arrays["sum_gnorm"], sum_g=arrays["sum_g"], count=arrays["count"])
        meta = {
            **item,
            "tokens": arrays["tokens"],
            "chains": lengths,
            "seconds": time.perf_counter() - started,
            # What a plan made of these statistics rests on (scripts/prune_plans.py).
            "checkpoint": str(args.checkpoint),
            "multimodal": True,  # every chain replayed as served (ServedReplay)
            "images": replay.images - images_before,
        }
        (args.out / f"{item['run_id']}.json").write_text(json.dumps(meta, indent=2))
        os.replace(partial, target)
        print(f"{item['run_id']} {item['game_id']} score={item['score']:.2f} tokens={arrays['tokens']} "
              f"chains={len(lengths)} {meta['seconds']:.0f}s", flush=True)

    while True:
        for item in mine:
            replay_item(item)
        if not args.follow:
            break
        mine = pending()
        if mine:
            continue
        if all(recording_done(recording) for recording, _ in args.recording):
            # A game sealed as the run ended is picked up by one last scan.
            mine = pending()
            if not mine:
                break
            continue
        time.sleep(args.follow)
        mine = pending()


if __name__ == "__main__":
    main()
