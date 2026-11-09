"""Guarded Slurm submission of bounded, resumable distillation segments."""

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from .accumulation import checkpoint_cursor, remaining_updates
from .cache_transaction import atomic_json, digest
from .config import load_config, validate_inputs


def endpoint(start, total, segment=100):
    if not 0 <= start < total or segment < 1:
        raise ValueError("Invalid segment budget")
    return min(start + segment, total)


def read(path):
    return json.loads(Path(path).read_text())


def assert_passed(result, steps, world_size=4):
    if read(result / "supervisor-exit.json")["exit_code"] != 0:
        raise ValueError("Previous supervisor failed")
    student = result / "student"
    if read(student / "TRAINING_PASSED.json")["steps"] != steps:
        raise ValueError("Previous segment did not complete its budget")
    for rank in range(world_size):
        final = read(student / f"final-rank{rank}.json")
        if not (
            final["status"] == "passed"
            and final["frozen_unchanged"]
            and final["router_replicas_equal"]
            and final["completed_steps"] == steps
        ):
            raise ValueError("Previous segment failed rank acceptance")
        metrics = [
            json.loads(line)
            for line in (student / f"metrics-rank{rank}.jsonl").read_text().splitlines()
        ]
        if (
            not metrics
            or metrics[-1]["step"] != steps
            or not all(
                math.isfinite(row[k])
                for row in metrics
                for k in ("loss_per_token", "router_grad_norm")
            )
        ):
            raise ValueError("Previous segment has incomplete/nonfinite metrics")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--segment-steps", type=int, default=100)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Explicit initial checkpoint for a new pinned run/ledger",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_inputs(config)
    repo = Path(__file__).resolve().parents[2]
    # parents[2] is repository root: repo/sft/distillation/launch.py.
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Commit the worktree before preparing a pinned launch")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = args.run_dir / "attempts.json"
        attempts = read(record) if record.exists() else []
        start = 0
        if config.get("resume"):
            raise ValueError(
                "Use the attempt ledger for segmented resumes, not base-config resume"
            )
        if args.resume_checkpoint:
            if attempts:
                raise ValueError(
                    "initial resume checkpoint only allowed for a new ledger"
                )
            import torch

            saved = torch.load(
                args.resume_checkpoint, map_location="cpu", weights_only=True
            )
            start = saved["step"]
            cursor = checkpoint_cursor(saved)
            if (
                len(saved["rng"]) != config["world_size"]
                or len(saved["frozen"]) != config["world_size"]
                or not saved["optimizer"]["state"]
            ):
                raise ValueError("incomplete initial checkpoint")
            if not all(torch.isfinite(p).all() for p in saved["routers"].values()):
                raise ValueError("nonfinite initial checkpoint")
            if config.get("data_microbatches") is not None:
                expected_end = start + remaining_updates(
                    cursor,
                    config["data_microbatches"],
                    config.get("gradient_accumulation", 1),
                )
                if expected_end != config.get(
                    "production_total_steps", config["steps"]
                ):
                    raise ValueError(
                        "update target does not match remaining data budget"
                    )
            config["resume"] = str(args.resume_checkpoint.resolve())
            config["resume_sha256"] = digest(args.resume_checkpoint)
            del saved
        if attempts and attempts[-1]["status"] == "submission_intent":
            raise ValueError("Uncertain previous submission; inspect before retry")
        submitted = [row for row in attempts if row["status"] == "submitted"]
        if submitted:
            previous = submitted[-1]
            if previous["status"] != "submitted":
                raise ValueError("Uncertain previous submission; inspect before retry")
            if previous["base_config_sha256"] != digest(args.config):
                raise ValueError("Base configuration changed during the run")
            if previous["git_commit"] != revision:
                raise ValueError("Implementation changed during the run")
            state = subprocess.check_output(
                [
                    "sacct",
                    "-j",
                    str(previous["job_id"]),
                    "-X",
                    "-n",
                    "-P",
                    "--format=State,ExitCode",
                ],
                text=True,
            ).strip()
            if state != "COMPLETED|0:0":
                raise ValueError(
                    f"Previous allocation is not successfully complete: {state}"
                )
            start = previous["end_step"]
            result = Path(previous["result"])
            assert_passed(result, start, config["world_size"])
            config["resume"] = str(result / "student" / f"checkpoint-step{start}.pt")
            if not Path(config["resume"]).is_file():
                raise ValueError("Previous checkpoint is missing")
            config["resume_sha256"] = digest(Path(config["resume"]))
        total = config.get("production_total_steps", config["steps"])
        config["steps"] = endpoint(start, total, args.segment_steps)
        config["wandb"]["name"] = f"router-updates{start + 1}-{config['steps']}"
        attempt = args.run_dir / f"attempt{len(attempts) + 1:03d}"
        snapshot, result = attempt / "config.json", attempt / "training"
        command = [
            "sbatch",
            "--parsable",
            f"--gpus=b200:{config['world_size']}",
            "--output=" + str(args.run_dir / "slurm-%j.out"),
            str(repo / "slurm/sft_distillation.sbatch"),
            str(repo),
            revision,
            str(snapshot),
            str(result),
            hashlib.sha256(
                (json.dumps(config, sort_keys=True) + "\n").encode()
            ).hexdigest(),
        ]
        scheduler = config["scheduler"]
        for flag in ("account", "partition", "qos"):
            if scheduler[flag] is not None:
                command.insert(2, f"--{flag}={scheduler[flag]}")
        report = {
            "start_step": start,
            "end_step": config["steps"],
            "total_steps": total,
            "git_commit": revision,
            "initial_resume": config.get("resume"),
            "resume_sha256": config.get("resume_sha256"),
            "gradient_accumulation": config.get("gradient_accumulation", 1),
            "data_microbatches": config.get("data_microbatches"),
            "command": command,
        }
        if not args.submit:
            print(json.dumps(report, indent=2))
            return
        subprocess.run(command[:1] + ["--test-only"] + command[1:], check=True)
        attempt.mkdir(exist_ok=False)
        atomic_json(snapshot, config)
        snapshot.chmod(0o444)
        if digest(snapshot) != command[-1]:
            raise ValueError("Serialized configuration hash mismatch")
        row = dict(
            status="submission_intent",
            **report,
            result=str(result),
            config=str(snapshot),
            config_sha256=digest(snapshot),
            base_config_sha256=digest(args.config),
        )
        attempts.append(row)
        atomic_json(record, attempts)
        try:
            response = subprocess.check_output(command, text=True)
        except subprocess.CalledProcessError as error:
            row.update(status="rejected", returncode=error.returncode)
            atomic_json(record, attempts)
            shutil.rmtree(attempt)
            raise
        job = response.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("Unrecognized submission response; inspect intent ledger")
        row.update(status="submitted", job_id=job)
        atomic_json(record, attempts)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
