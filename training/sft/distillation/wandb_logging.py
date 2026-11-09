"""One W&B writer in the supervisor; GPU workers only write local evidence."""

import csv
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from sft.distillation.common import dump
from sft.distillation.compute_metrics import SCHEMA as COMPUTE_SCHEMA
from sft.distillation.compute_metrics import aggregate_compute
from sft.distillation.expert_metrics import ExpertUsageWindow
from sft.distillation.gradient_noise import wandb_metrics


class Tail:
    def __init__(self, path):
        self.path, self.offset = Path(path), 0

    def lines(self):
        if not self.path.exists():
            return []
        lines = []
        with self.path.open() as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):
                    break
                lines.append(line)
                self.offset = f.tell()
        return lines


def aggregate_step(rows):
    if not rows:
        raise ValueError("Need at least one rank at an optimizer step")
    step = rows[0]["step"]
    if any(r["step"] != step for r in rows):
        raise ValueError("Need all ranks at the same optimizer step")
    seconds = max(r["seconds"] for r in rows)
    metrics = {
        "train/step": step,
        "train/loss_per_completion_token": rows[0]["loss_per_token"],
        "train/router_grad_norm": rows[0]["router_grad_norm"],
        "train/step_seconds": seconds,
        "train/completion_tokens_per_second": rows[0]["global_completion_tokens"]
        / seconds,
        "train/input_tokens_per_second": sum(r["sequence_length"] for r in rows)
        / seconds,
        "train/global_completion_tokens": rows[0]["global_completion_tokens"],
    }
    if "objective_loss" in rows[0]:
        metrics["train/objective_loss"] = rows[0]["objective_loss"]
        metrics["train/loss_per_example"] = rows[0]["loss_per_example"]
        metrics["train/global_examples"] = rows[0]["global_examples"]
    for key in (
        "expert_lora_grad_norm",
        "trainable_grad_norm",
        "trainable_update_ratio",
    ):
        if key in rows[0]:
            metrics["train/" + key] = rows[0][key]
    metrics.update(aggregate_compute(rows))
    throughput = metrics.get("train/useful_model_flops_per_second_estimate")
    if throughput is not None:
        metrics["train/reference_bf16_mfu_percent"] = throughput / 9e15 * 100

    for rank, row in enumerate(rows):
        for key, value in row.get("attention", {}).items():
            metrics[f"attention/rank{rank}/{key}"] = value

    if rows[0].get("router_update_ratio") is not None:
        metrics["train/router_update_ratio"] = rows[0]["router_update_ratio"]
    if "global_gradient_noise" in rows[0]:
        metrics.update(wandb_metrics(rows[0]["global_gradient_noise"]))
    for rank, row in enumerate(rows):
        for key in [
            "sequence_length",
            "seconds",
            "peak_allocated_gib",
            "peak_reserved_gib",
        ]:
            metrics[f"train/rank{rank}/{key}"] = row[key]
    return metrics


def probe_payload(report):
    # Quality/ablation diagnostics remain; detailed routing diagnostics stay local.
    payload = {"probe/step": report["step"]}
    for key in (
        "baseline_kl",
        "baseline_ce",
        "teacher_top1_agreement",
        "advisory_ready",
        "window_kl_range",
        "window_evaluations",
        "ablation_budget_configured",
        "seconds",
    ):
        value = report.get(key)
        if isinstance(value, (int, float, bool)):
            payload["probe/" + key] = value
    for criterion, values in report["criteria"].items():
        for key, value in values.items():
            if key.startswith("ablation_") and isinstance(value, (int, float, bool)):
                payload[f"probe/{criterion}/{key}"] = value
    layers = report["criteria"]["frequency"]["layers"]
    overlaps = [layer["overlap_previous"] for layer in layers]
    if overlaps and all(value is not None for value in overlaps):
        payload["experts/pruning_candidate_stability"] = sum(overlaps) / len(overlaps)
    return payload, []


class WandbLogger:
    def __init__(self, config, result):
        self.config, self.result = config, Path(result)
        self.settings = config.get("wandb", {})
        self.mode = self.settings.get("mode", "online")
        if self.mode == "online" and not self.settings.get("entity"):
            raise ValueError(
                "Set an explicit W&B entity before online logging; no default team fallback"
            )
        import wandb

        self.wandb = wandb
        self.started = time.monotonic()
        self.interval = int(self.settings.get("expert_log_every", 10))
        if self.interval < 1:
            raise ValueError("expert_log_every must be positive")
        self.world_size = config.get("world_size", 4)
        self.tails = [
            Tail(self.result / "student" / f"metrics-rank{i}.jsonl")
            for i in range(self.world_size)
        ]
        self.teacher_tails = [
            Tail(Path(config["cache"]) / f"progress-rank{i}.jsonl") for i in range(4)
        ]
        self.host_tail = Tail(self.result / "host-memory.jsonl")
        self.pinned_tails = [
            Tail(self.result / "student" / f"pinned-memory-rank{i}.jsonl")
            for i in range(self.world_size)
        ]
        self.gpu_tail = Tail(self.result / "gpu-utilization.csv")
        self.gpu_header = None
        self.gpu_epoch = None
        self.pending, self.last_step = {}, 0
        self.expert_window = ExpertUsageWindow(config.get("probe", {}))
        self.probe_seen = set()
        public_config = {
            k: config[k]
            for k in [
                "teacher",
                "student",
                "data_sha256",
                "steps",
                "production_total_steps",
                "lr",
                "temperature",
                "ce_weight",
                "chunk",
                "offload_every",
                "save_every",
                "seed",
                "resume",
                "gradient_accumulation",
                "data_microbatches",
                "allow_accumulation_change",
                "expert_optimization",
                "world_size",
                "fuse_norm_tail",
                "probe",
            ]
            if k in config
        }
        from .config import kernel_stack

        public_config["kernel_stack"] = kernel_stack(config)
        public_config["validation_only"] = bool(config.get("validation_only", False))
        public_config["teacher_cache_reused"] = (
            Path(config["cache"]) / "manifest.json"
        ).exists()
        public_config.update(
            world_size=self.world_size,
            sequences_per_gpu=1,
            gradient_accumulation=config.get("gradient_accumulation", 1),
            global_sequences_per_update=self.world_size
            * config.get("gradient_accumulation", 1),
            slurm_job_id=os.getenv("SLURM_JOB_ID"),
            attention_backend=config.get("attention_backend", "sdpa"),
            reference_bf16_peak_flops_per_second=9e15,
            logging_schema=3,
            compute_metrics_schema=COMPUTE_SCHEMA,
        )
        if config.get("resume"):
            parent = Path(config["resume"]).parent.parent / "wandb-run.json"
            if parent.exists():
                public_config["parent_wandb_run"] = json.loads(parent.read_text()).get(
                    "url"
                )
        self.result.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=self.settings.get("project", "flash-router-distillation"),
            entity=self.settings.get("entity"),
            group=self.settings.get("group", "e307-router-distillation"),
            name=self.settings.get("name")
            or f"distill-{os.getenv('SLURM_JOB_ID', self.result.name)}",
            mode=self.mode,
            dir=str(self.result),
            config=public_config,
            force=self.mode == "online",
            settings=wandb.Settings(
                console="off", disable_git=True, disable_code=True, x_disable_stats=True
            ),
        )
        self.run.define_metric("probe/step")
        self.run.define_metric("probe/*", step_metric="probe/step")
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("experts/*", step_metric="train/step")
        self.run.define_metric("attention/*", step_metric="train/step")
        self.run.define_metric(
            "experts/pruning_candidate_stability", step_metric="probe/step"
        )
        self.run.define_metric("system/elapsed_seconds")
        self.run.define_metric("gpu/*", step_metric="system/elapsed_seconds")
        dump(
            self.result / "wandb-run.json",
            {
                "id": self.run.id,
                "url": self.run.url,
                "project": self.run.project,
                "entity": self.run.entity,
                "mode": self.mode,
                "resume_policy": "new linked run per attempt; original optimizer step preserved",
            },
        )

    def poll(self, phase=None):
        for line in self.host_tail.lines():
            row = json.loads(line)
            payload = {
                f"host/{k}": row[k] / 2**30
                for k in ("current_bytes", "peak_bytes", "limit_bytes")
                if k in row
            }
            if payload:
                self.run.log(
                    {k.replace("_bytes", "_gib"): v for k, v in payload.items()}
                )
        for rank, tail in enumerate(self.pinned_tails):
            for line in tail.lines():
                row = json.loads(line)
                self.run.log(
                    {
                        f"host/rank{rank}/{row['phase']}/{k}": v
                        for k, v in row.items()
                        if k not in ("timestamp", "step", "phase")
                    }
                )
        for path in sorted(
            (self.result / "student").glob("probe-step*.json"),
            key=lambda p: int(p.stem.removeprefix("probe-step")),
        ):
            if path.name not in self.probe_seen:
                payload, _ = probe_payload(json.loads(path.read_text()))
                self.run.log(payload)
                self.probe_seen.add(path.name)
        for rank, tail in enumerate(self.tails):
            for line in tail.lines():
                row = json.loads(line)
                self.pending.setdefault(row["step"], {})[rank] = row
        world_size = getattr(self, "world_size", len(self.tails))
        for step in sorted(self.pending):
            rows = self.pending[step]
            usage = self.result / "student" / f"expert-usage-step{step}.json"
            if len(rows) < world_size or not usage.exists():
                break
            self.run.log(aggregate_step([rows[i] for i in range(world_size)]))
            self.expert_window.add(json.loads(usage.read_text()))
            if (
                self.expert_window.steps >= self.interval
                or step == self.config["steps"]
            ):
                self.run.log({"train/step": step, **self.expert_window.flush()})
            self.last_step = step
            del self.pending[step]
        for line in self.gpu_tail.lines():
            values = next(csv.reader([line], skipinitialspace=True))
            if self.gpu_header is None:
                self.gpu_header = values
                continue
            if len(values) != 8:
                continue
            gpu = values[
                2
            ].strip()  # Physical UUID; no confusion with rank/local indices.
            timestamp = (
                datetime.datetime.strptime(values[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                .astimezone()
                .timestamp()
            )
            if self.gpu_epoch is None:
                self.gpu_epoch = timestamp
            payload = {"system/elapsed_seconds": timestamp - self.gpu_epoch}
            for key, raw in zip(
                [
                    "utilization_pct",
                    "memory_utilization_pct",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_watts",
                ],
                values[3:],
                strict=True,
            ):
                try:
                    payload[f"gpu/{gpu}/{key}"] = float(raw.split()[0])
                except ValueError:
                    continue
            self.run.log(payload)
        if phase == "teacher-cache":
            for rank, tail in enumerate(self.teacher_tails):
                for line in tail.lines():
                    row = json.loads(line)
                    self.run.log({f"teacher/rank{rank}/{k}": v for k, v in row.items()})

    def run_process(self, command, phase):
        self.run.summary["phase"] = phase
        with subprocess.Popen(command) as process:
            try:
                while True:
                    code = process.poll()
                    self.poll(phase)
                    if code is not None:
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                        return
                    time.sleep(5)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise

    def finish(self, exit_code):
        try:
            self.poll()
            self.run.summary.update(
                {
                    "exit_code": exit_code,
                    "last_optimizer_step": self.last_step,
                    "status": "passed"
                    if exit_code == 0
                    else ("partial" if exit_code == 75 else "failed"),
                }
            )
            artifact = self.wandb.Artifact(
                f"distill-evidence-{self.run.id}", type="training-evidence"
            )
            # Compact reports only: no token histories, teacher tensors or model checkpoints.
            for name in [
                "run-config.json",
                "preflight.json",
                "gpu-utilization.csv",
                "host-memory.jsonl",
                "student/manifest.json",
                "student/TRAINING_PASSED.json",
                "student/expert-teacher-ids.json",
            ]:
                path = self.result / name
                if path.is_file() and name != "run-config.json":
                    artifact.add_file(str(path), name=name)
            for path in (self.result / "student").glob("metrics-rank*.jsonl"):
                artifact.add_file(str(path), name="student/" + path.name)
            for path in (self.result / "student").glob("pinned-memory-rank*.jsonl"):
                artifact.add_file(str(path), name="student/" + path.name)
            for path in (self.result / "student").glob("probe-step*.json"):
                artifact.add_file(str(path), name="student/" + path.name)
            self.run.log_artifact(artifact)
        finally:
            self.run.finish(exit_code=exit_code)
