"""Teacher-forced exact completion hidden states; no student resident on GPUs."""

import argparse
import datetime
import fcntl
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sft.distillation.cache_transaction import (
    atomic_torch_save,
    commit,
    committed,
    initialize,
)
from sft.distillation.common import (
    Dataset,
    dump,
    row_key,
    sha,
    teacher_identity,
)
from sft.distillation.flash_model import build_model, frozen_fingerprint
from sft.distillation.routing import RouteRecorder, attach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=30000)
    args = ap.parse_args()
    attempt_started = time.monotonic()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device("cuda", rank),
    )
    world = dist.get_world_size()
    if world < 1:
        raise ValueError("teacher cache requires at least one distributed rank")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = teacher_identity(args.teacher)
    dataset = Dataset(args.data)
    identity = {
        "schema": "resumable-teacher-cache-v1",
        "teacher_identity": source,
        "dataset_sha256": sha(args.data),
        "rows": len(dataset),
        "world_size": world,
    }
    cache_lock = None
    if rank == 0:
        args.cache.mkdir(parents=True, exist_ok=True)
        cache_lock = (args.cache / "writer.lock").open("a")
        fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(args.cache, identity)
    dist.barrier()
    progress = committed(args.cache, identity)
    start_batch = 0 if progress is None else progress["batches"]
    total_batches = math.ceil(len(dataset) / world)
    if (args.cache / "manifest.json").exists():
        raise ValueError("Complete cache already exists; validate and reuse it")
    model, head, routers = build_model(args.teacher, teacher=True)
    assert not routers and not any(p.requires_grad for p in model.parameters())
    before = frozen_fingerprint(model, head)
    if rank == 0 and progress is None:
        save_file(
            {"weight": head.weight.detach().cpu().contiguous()},
            args.cache / "head.safetensors",
        )
    dist.barrier()
    head_sha = sha(args.cache / "head.safetensors")
    recorder = RouteRecorder(48, 307, "cuda")
    attach(model, recorder)
    total_counts, total_mass = (
        torch.zeros_like(recorder.counts),
        torch.zeros_like(recorder.mass),
    )
    entries = {}
    if progress is not None:
        state = torch.load(
            args.cache / progress["states"][rank]["name"],
            map_location="cpu",
            weights_only=True,
        )
        assert state["identity"] == identity and state["batches"] == start_batch
        assert state["frozen"] == before and state["head_sha256"] == head_sha
        entries = state["entries"]
        expected_indices = {
            str(i) for i in range(rank, min(start_batch * world, len(dataset)), world)
        }
        assert set(entries) == expected_indices
        for index, record in entries.items():
            assert row_key(dataset[int(index)]) == record["row_sha256"]
            assert (
                sha(args.cache / f"{int(index):08d}.safetensors")
                == record["file_sha256"]
            )
        total_counts.copy_(state["counts"].to("cuda"))
        total_mass.copy_(state["mass"].to("cuda"))
    started = time.monotonic()
    completed_rows = len(entries)
    completed_batches = start_batch
    for batch in range(start_batch, total_batches):
        index = batch * world + rank
        row = dataset[min(index, len(dataset) - 1)]
        prompt = len(row["input_token_ids"])
        ids = torch.tensor(
            [row["input_token_ids"] + row["output_token_ids"]], device="cuda"
        )
        recorder.begin(prompt)
        recorder.enabled = index < len(dataset)
        with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            hidden = model(input_ids=ids, use_cache=False).last_hidden_state
            selected = hidden[0, prompt - 1 : -1].detach().cpu().contiguous()
        recorder.finish()
        total_counts.add_(recorder.counts)
        total_mass.add_(recorder.mass)
        assert torch.isfinite(selected).all()
        if index < len(dataset):
            key = row_key(row)
            path = args.cache / f"{index:08d}.safetensors"
            tmp = path.with_suffix(".tmp")
            save_file({"hidden": selected}, tmp, metadata={"row_sha256": key})
            tmp.replace(path)
            entries[str(index)] = {
                "row_sha256": key,
                "file_sha256": sha(path),
                "completion_tokens": len(selected),
            }
        if index < len(dataset):
            completed_rows += 1
            with (args.cache / f"progress-rank{rank}.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "completed_rows": completed_rows,
                            "total_rows": len(dataset),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
        del hidden, selected, ids
        completed_batches = batch + 1
        atomic_torch_save(
            {
                "identity": identity,
                "batches": completed_batches,
                "frozen": before,
                "head_sha256": head_sha,
                "entries": entries,
                "counts": total_counts.cpu(),
                "mass": total_mass.cpu(),
            },
            args.cache / f"state-b{completed_batches}-rank{rank}.pt",
        )
        dist.barrier()
        if rank == 0:
            commit(args.cache, identity, completed_batches)
        dist.barrier()
        # Keep the current committed generation; all readers have crossed the barrier.
        if completed_batches > 1:
            previous = args.cache / f"state-b{completed_batches - 1}-rank{rank}.pt"
            previous.unlink(missing_ok=True)
        stop = torch.tensor(
            int(time.monotonic() - attempt_started >= args.max_seconds), device="cuda"
        )
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() and completed_batches < total_batches:
            break
    assert frozen_fingerprint(model, head) == before
    if completed_batches < total_batches:
        if rank == 0:
            dump(
                args.cache / "PARTIAL.json",
                {
                    "batches": completed_batches,
                    "total_batches": total_batches,
                    "resume_from": "cache-progress.json",
                    "supervisor_exit_code": 75,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )
        dist.barrier()
        dist.destroy_process_group()
        raise SystemExit(75)
    recorder.counts.copy_(total_counts)
    recorder.mass.copy_(total_mass)
    usage = recorder.report()
    if rank == 0:
        dump(args.cache / "teacher-expert-usage.json", usage)
    assert frozen_fingerprint(model, head) == before
    dump(
        args.cache / f"rank{rank}.json", {"entries": entries, "frozen_unchanged": True}
    )
    dist.barrier()
    if rank == 0:
        (args.cache / "PARTIAL.json").unlink(missing_ok=True)
        combined = {}
        for i in range(world):
            combined.update(
                json.loads((args.cache / f"rank{i}.json").read_text())["entries"]
            )
        assert len(combined) == len(dataset)
        dump(
            args.cache / "manifest.json",
            {
                "format": "full-vocabulary-kl-via-bf16-hidden-v1",
                "teacher_identity": source,
                "dataset_sha256": sha(args.data),
                "tokenizer_sha256": source["tokenizer_sha256"],
                "head_sha256": sha(args.cache / "head.safetensors"),
                "rows": len(dataset),
                "entries": combined,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
