"""Test a training run's checkpoints as they land, beside the run.

The shipped head is tested once (as update 0), then every checkpoint listed
in ``RUN/checkpoints/index.jsonl`` that has no result yet: the trainer's
offline evaluation on each ``--set`` (per draft step, KL and acceptance at
T=0.6 and as the deployed sampler verifies, per harness), over FP8
per-token-head K/V as served. Results go to ``RUN/test.jsonl`` and to
TensorBoard under ``RUN/tb/test`` -- a run of its own beside the trainer's,
on the same update axis. Exits once ``RUN/DONE`` exists and every listed
checkpoint is tested, or after ``--hours``.

Online tests (the served engine) are separate and on demand:
``drafter.export`` a checkpoint, then ``drafter.online`` or a replay with
``--drafter``.

    python -m drafter.watch RUN_DIR --set NAME=CACHE:DATASET [...] [--steps 3] [--poll 60] [--hours 12]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch


def main(argv: Optional[list[str]] = None) -> None:
    from torch.utils.tensorboard import SummaryWriter

    from drafter.head import LabelHead, load_head, load_trained
    from drafter.train import Trainer, evaluate, load_chains

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--set", action="append", required=True, dest="sets")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--kv-fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--hours", type=float, default=12.0)
    args = parser.parse_args(argv)
    args.checkpoint_steps, args.ce_weight, args.eager_attention = False, 0.0, False

    started = time.monotonic()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    sets = {}
    for spec in args.sets:
        name, paths = spec.split("=", 1)
        cache, dataset = paths.split(":", 1)
        sets[name] = load_chains(Path(cache), Path(dataset))
    label = LabelHead(device=device)
    head = load_head(device=device)
    trainer = Trainer(head, label, args, optimize=False)
    results_path = args.run / "test.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["update"] for line in results_path.read_text().splitlines() if line}
    writer = SummaryWriter(str(args.run / "tb" / "test"))

    def test(update: int, path: Optional[str]) -> None:
        tick = time.monotonic()
        if path is not None:
            load_trained(head, Path(path))
        row: dict = {"update": update, "checkpoint": path}
        for name, chains in sets.items():
            result = evaluate(trainer, chains, device)
            row[name] = result
            for key, value in result.items():
                if isinstance(value, float) and math.isfinite(value):
                    writer.add_scalar(f"test/{name}/{key}", value, update)
        writer.flush()
        with results_path.open("a") as out:
            out.write(json.dumps(row) + "\n")
        done.add(update)
        brief = {
            name: " ".join(f"{row[name].get(f'all/step{s}/accept_t0.6', float('nan')):.3f}" for s in range(args.steps))
            for name in sets
        }
        print(f"[watch] update {update}: acceptance at T=0.6 per step {brief} ({time.monotonic() - tick:.0f}s)", flush=True)

    if 0 not in done:
        test(0, None)
    index = args.run / "checkpoints" / "index.jsonl"
    while time.monotonic() - started < args.hours * 3600:
        listed = [json.loads(line) for line in index.read_text().splitlines() if line] if index.exists() else []
        pending = list({entry["update"]: entry for entry in listed if entry["update"] not in done}.values())
        for entry in pending:
            test(entry["update"], entry["path"])
        if (args.run / "DONE").exists() and not pending:
            print("[watch] the run is done and every checkpoint is tested", flush=True)
            break
        if not pending:
            time.sleep(args.poll)
    writer.close()


if __name__ == "__main__":
    main()
