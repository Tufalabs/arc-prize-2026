from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


class NullMemoryLogger:
    def log(
        self,
        tag: str,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        return None

    @contextmanager
    def region(
        self,
        name: str,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        yield


class CUDAMemoryLogger:
    def __init__(
        self,
        output_dir: str | Path,
        device: int | torch.device | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        sync: bool = True,
        base_metadata: dict[str, Any] | None = None,
        log_every_n_steps: int = 1,
        max_logged_steps: int | None = None,
        reset_peak_stats_per_region: bool = False,
        log_gpu_utilization: bool = False,
        gpu_utilization_sample_interval_seconds: float = 1.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = _resolve_rank(rank)
        self.local_rank = _resolve_local_rank(local_rank)
        self.world_size = _resolve_world_size(world_size)
        self.device = _resolve_device(device, self.local_rank)
        self.sync = bool(sync)
        self.base_metadata = dict(base_metadata or {})
        self.hostname = str(self.base_metadata.pop("hostname", socket.gethostname()))
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.max_logged_steps = max_logged_steps
        self.reset_peak_stats_per_region = bool(reset_peak_stats_per_region)
        self.log_gpu_utilization = bool(log_gpu_utilization)
        self.gpu_utilization_sample_interval_seconds = max(
            0.0, float(gpu_utilization_sample_interval_seconds)
        )
        self._gpu_utilization_cache: dict[str, Any] | None = None
        self._gpu_utilization_cache_time = 0.0
        self.path = self.output_dir / f"mem_rank{self.rank}.jsonl"

    def log(
        self,
        tag: str,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._should_log_step(step):
            return
        self._write_row(self._row(tag=tag, step=step, extra=extra))

    @contextmanager
    def region(
        self,
        name: str,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self._should_log_step(step):
            yield
            return
        started_at = time.perf_counter()
        start = self._row(tag=f"{name}:start", step=step, extra=extra)
        start["region"] = name
        self._write_row(start)
        if self.reset_peak_stats_per_region:
            self._reset_peak_memory_stats()
        try:
            yield
        finally:
            ended_at = time.perf_counter()
            end = self._row(tag=f"{name}:end", step=step, extra=extra)
            end["region"] = name
            self._write_row(end)
            summary = self._summary_row(
                name=name,
                step=step,
                start=start,
                end=end,
                duration_s=ended_at - started_at,
                extra=extra,
            )
            self._write_row(summary)

    def _should_log_step(self, step: int | None) -> bool:
        if step is None:
            return True
        if self.max_logged_steps is not None and int(step) >= self.max_logged_steps:
            return False
        return int(step) % self.log_every_n_steps == 0

    def _row(
        self,
        *,
        tag: str,
        step: int | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.base_metadata)
        row.update(
            {
                "time": time.time(),
                "step": step,
                "tag": tag,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
                "hostname": self.hostname,
                "device": _device_label(self.device),
            }
        )
        row.update(self._measure())
        if extra:
            row.update(extra)
        return row

    def _summary_row(
        self,
        *,
        name: str,
        step: int | None,
        start: dict[str, Any],
        end: dict[str, Any],
        duration_s: float,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = dict(self.base_metadata)
        row.update(
            {
                "time": time.time(),
                "step": step,
                "tag": f"{name}:summary",
                "region": name,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
                "hostname": self.hostname,
                "device": _device_label(self.device),
                "duration_s": duration_s,
                "start_allocated_mb": start["allocated_mb"],
                "end_allocated_mb": end["allocated_mb"],
                "delta_allocated_mb": end["allocated_mb"] - start["allocated_mb"],
                "peak_allocated_mb": end["max_allocated_mb"],
                "start_reserved_mb": start["reserved_mb"],
                "end_reserved_mb": end["reserved_mb"],
                "delta_reserved_mb": end["reserved_mb"] - start["reserved_mb"],
                "peak_reserved_mb": end["max_reserved_mb"],
            }
        )
        row.update(self._measure())
        if extra:
            row.update(extra)
        return row

    def _measure(self) -> dict[str, float]:
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "max_reserved_mb": 0.0,
            }
        if self.sync:
            torch.cuda.synchronize(self.device)
        measurements = {
            "allocated_mb": float(torch.cuda.memory_allocated(self.device))
            / (1024.0**2),
            "reserved_mb": float(torch.cuda.memory_reserved(self.device)) / (1024.0**2),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated(self.device))
            / (1024.0**2),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved(self.device))
            / (1024.0**2),
        }
        measurements.update(self._measure_gpu_utilization())
        return measurements

    def _measure_gpu_utilization(self) -> dict[str, Any]:
        if not self.log_gpu_utilization:
            return {}
        now = time.monotonic()
        if (
            self._gpu_utilization_cache is not None
            and now - self._gpu_utilization_cache_time
            < self.gpu_utilization_sample_interval_seconds
        ):
            return dict(self._gpu_utilization_cache)
        measurements = _query_nvidia_smi(self.device)
        self._gpu_utilization_cache = measurements
        self._gpu_utilization_cache_time = now
        return dict(measurements)

    def _write_row(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _reset_peak_memory_stats(self) -> None:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


def build_memory_logger(
    config: Any,
    run_dir: str | Path,
    *,
    base_metadata: dict[str, Any] | None = None,
    device: int | torch.device | None = None,
) -> CUDAMemoryLogger | NullMemoryLogger:
    if not bool(getattr(config, "enabled", False)):
        return NullMemoryLogger()
    rank = _resolve_rank(None)
    ranks = getattr(config, "ranks", [0])
    if not _rank_enabled(rank, ranks):
        return NullMemoryLogger()
    output_subdir = str(getattr(config, "output_subdir", "memory") or "memory")
    return CUDAMemoryLogger(
        output_dir=Path(run_dir) / output_subdir,
        device=device,
        rank=rank,
        local_rank=_resolve_local_rank(None),
        world_size=_resolve_world_size(None),
        sync=bool(getattr(config, "sync", False)),
        base_metadata=base_metadata,
        log_every_n_steps=max(1, int(getattr(config, "log_every_n_steps", 1))),
        max_logged_steps=getattr(config, "max_logged_steps", None),
        reset_peak_stats_per_region=bool(
            getattr(config, "reset_peak_stats_per_region", False)
        ),
        log_gpu_utilization=bool(getattr(config, "log_gpu_utilization", False)),
        gpu_utilization_sample_interval_seconds=max(
            0.0,
            float(getattr(config, "gpu_utilization_sample_interval_seconds", 1.0)),
        ),
    )


def _rank_enabled(rank: int, ranks: Any) -> bool:
    if isinstance(ranks, str):
        return ranks.strip().lower() == "all"
    if isinstance(ranks, int):
        return rank == ranks
    if isinstance(ranks, (list, tuple, set)):
        return rank in {int(item) for item in ranks}
    return rank == 0


def _resolve_rank(rank: int | None) -> int:
    if rank is not None:
        return int(rank)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0") or 0)


def _resolve_local_rank(local_rank: int | None) -> int:
    if local_rank is not None:
        return int(local_rank)
    return int(os.environ.get("LOCAL_RANK", "0") or 0)


def _resolve_world_size(world_size: int | None) -> int:
    if world_size is not None:
        return int(world_size)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def _resolve_device(device: int | torch.device | None, local_rank: int) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _device_label(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return str(device)


def _query_nvidia_smi(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    device_index = 0 if device.index is None else int(device.index)
    fields = [
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
    ]
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"gpu_utilization_error": str(exc)}
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(fields):
        return {"gpu_utilization_error": f"unexpected nvidia-smi output: {line}"}
    parsed = [_parse_float(value) for value in values]
    return {
        "gpu_utilization_percent": parsed[0],
        "gpu_memory_utilization_percent": parsed[1],
        "gpu_memory_used_mb": parsed[2],
        "gpu_memory_total_mb": parsed[3],
        "gpu_power_draw_w": parsed[4],
        "gpu_sm_clock_mhz": parsed[5],
        "gpu_memory_clock_mhz": parsed[6],
        "gpu_temperature_c": parsed[7],
    }


def _parse_float(value: str) -> float | None:
    if value in {"", "N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
