"""Configuration-based supervisor for cached-teacher router distillation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config, validate_inputs


def worker_command(config, result):
    result = Path(result)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=" + str(config.get("world_size", 4)),
        "--module",
        "sft.distillation.worker",
        "--probe-config",
        str(result / "probe-config.json"),
        "--student",
        config["student"],
        "--data",
        config["data"],
        "--cache",
        config["cache"],
        "--out",
        str(result / "student"),
    ]
    for key in (
        "steps",
        "lr",
        "temperature",
        "ce_weight",
        "chunk",
        "save_every",
        "seed",
    ):
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    from .config import kernel_stack

    stack = kernel_stack(config)
    command.extend(["--kernel-mode", stack.pop("kernel_mode")])
    command.extend(["--qsa-kernel", stack.pop("qsa_kernel")])
    command.extend(["--loss-weighting", config.get("loss_weighting", "example")])
    command.extend(["--attention-backend", stack.pop("attention_backend")])
    command.extend(["--expert-optimization", stack.pop("expert_optimization")])
    command.extend(["--expert-lora", json.dumps(config.get("expert_lora"))])
    command.append(
        "--train-routers" if config.get("train_routers", True) else "--no-train-routers"
    )
    command.append(
        "--fuse-norm-tail" if stack.pop("fuse_norm_tail") else "--no-fuse-norm-tail"
    )
    # The rest of the resolved stack, explicit, so the worker builds exactly it.
    command.extend(["--kernel-options", json.dumps(stack, sort_keys=True)])
    from training.flash.offload import resolve_qsa_retention

    retain = resolve_qsa_retention(
        config.get("attention_backend", "sdpa"),
        config.get("reuse_qsa_selection", False),
        config.get("retain_qsa_attention"),
    )
    command.append("--retain-qsa-attention" if retain else "--no-retain-qsa-attention")
    if config.get("reuse_qsa_selection", False):
        command.append("--reuse-qsa-selection")
    command.extend(["--offload-every", str(config.get("offload_every", 1))])
    command.extend(
        ["--gradient-accumulation", str(config.get("gradient_accumulation", 1))]
    )
    if config.get("data_microbatches") is not None:
        command.extend(["--data-microbatches", str(config["data_microbatches"])])
    if config.get("allow_accumulation_change", False):
        command.append("--allow-accumulation-change")
    if config.get("resume_sha256"):
        command.extend(["--resume-sha256", config["resume_sha256"]])
    if config.get("resume"):
        command.extend(["--resume", config["resume"]])
    command.extend(["--batching-config", json.dumps(config.get("batching", {}))])
    return command


def source_identity():
    package = Path(__file__).resolve().parent
    result = {
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.iterdir())
            if p.is_file()
        }
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(package), "rev-parse", "HEAD"], text=True
        ).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "-C", str(package), "status", "--porcelain", "--", "."], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = os.getenv("ARC_DISTILLATION_GIT_COMMIT")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--gpu-telemetry", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the worker command; no GPU, W&B, or output writes",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = validate_inputs(config)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="validated",
                    **report,
                    steps=config["steps"],
                    command=worker_command(config, args.result or Path("RESULT")),
                ),
                indent=2,
            )
        )
        return
    if args.result is None:
        parser.error("--result is required for training")
    from .common import dump
    from .preflight import audit
    from .staging import stage
    from .wandb_logging import WandbLogger

    # This audit performs no model forward or CUDA allocation.
    preflight = {
        "teacher": audit(Path(config["teacher"]), 307),
        "student": audit(Path(config["student"]), report["student_experts"]),
    }
    args.result.mkdir(parents=True, exist_ok=False)
    dump(args.result / "run-config.json", config)
    dump(args.result / "source-identity.json", source_identity())
    dump(args.result / "preflight.json", preflight)
    dump(args.result / "probe-config.json", config["probe"])
    if args.stage_dir:
        stage(Path(config["student"]), args.stage_dir)
        os.environ["FLASH_CANONICAL_STUDENT"] = config["student"]
        os.environ["FLASH_STAGED_STUDENT"] = str(args.stage_dir)
    from .host_memory import HostMemoryMonitor

    host_monitor = HostMemoryMonitor(args.result / "host-memory.jsonl")
    host_monitor.start()
    logger = WandbLogger(config, args.result)
    exit_code = 1
    monitor = None
    telemetry = None
    try:
        if args.gpu_telemetry:
            telemetry = (args.result / "gpu-utilization.csv").open("w")
            command = [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv",
                "-l",
                "5",
            ]
            if os.getenv("SLURM_JOB_GPUS"):
                command.append("--id=" + os.environ["SLURM_JOB_GPUS"])
            monitor = subprocess.Popen(command, stdout=telemetry)
        logger.run_process(worker_command(config, args.result), phase="student")
        exit_code = 0
    finally:
        host_monitor.stop()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=15)
        if telemetry is not None:
            telemetry.close()
        logger.finish(exit_code)
        dump(args.result / "supervisor-exit.json", {"exit_code": exit_code})


if __name__ == "__main__":
    main()
