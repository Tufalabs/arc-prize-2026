from __future__ import annotations

import json
from types import SimpleNamespace

from training.config import MemoryProfilerConfig
import training.memory_profile.logger as memory_logger
from training.memory_profile import (
    CUDAMemoryLogger,
    NullMemoryLogger,
    build_memory_logger,
)
from training.memory_profile.report import generate_report


def test_cuda_memory_logger_writes_point_and_region_rows(tmp_path):
    logger = CUDAMemoryLogger(
        output_dir=tmp_path,
        device="cpu",
        rank=0,
        local_rank=0,
        world_size=1,
        sync=False,
        base_metadata={"run_id": "test-run"},
    )

    logger.log("step_start", step=2, extra={"batch_size": 4})
    with logger.region("forward", step=2, extra={"phase": "train"}):
        pass

    rows = [
        json.loads(line)
        for line in (tmp_path / "mem_rank0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["tag"] for row in rows] == [
        "step_start",
        "forward:start",
        "forward:end",
        "forward:summary",
    ]
    for row in rows:
        assert row["rank"] == 0
        assert row["local_rank"] == 0
        assert row["world_size"] == 1
        assert row["hostname"]
        assert row["device"] == "cpu"
        assert row["allocated_mb"] == 0.0
        assert row["reserved_mb"] == 0.0
        assert row["max_allocated_mb"] == 0.0
        assert row["max_reserved_mb"] == 0.0
    summary = rows[-1]
    assert summary["region"] == "forward"
    assert "duration_s" in summary
    assert summary["delta_allocated_mb"] == 0.0
    assert summary["peak_allocated_mb"] == 0.0


def test_cuda_memory_logger_respects_max_logged_steps(tmp_path):
    logger = CUDAMemoryLogger(
        output_dir=tmp_path,
        device="cpu",
        rank=0,
        local_rank=0,
        world_size=1,
        sync=False,
        max_logged_steps=2,
    )

    logger.log("step_start", step=0)
    logger.log("step_start", step=1)
    logger.log("step_start", step=2)

    rows = [
        json.loads(line)
        for line in (tmp_path / "mem_rank0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["step"] for row in rows] == [0, 1]


def test_build_memory_logger_respects_disabled_and_rank_filter(tmp_path, monkeypatch):
    assert isinstance(
        build_memory_logger(MemoryProfilerConfig(enabled=False), tmp_path),
        NullMemoryLogger,
    )

    monkeypatch.setenv("RANK", "1")
    assert isinstance(
        build_memory_logger(MemoryProfilerConfig(enabled=True, ranks=[0]), tmp_path),
        NullMemoryLogger,
    )
    assert isinstance(
        build_memory_logger(MemoryProfilerConfig(enabled=True, ranks="all"), tmp_path),
        CUDAMemoryLogger,
    )


def test_memory_profiler_config_from_dict():
    config = MemoryProfilerConfig.from_dict(
        {
            "enabled": True,
            "ranks": "all",
            "sync": True,
            "log_every_n_steps": 0,
            "max_logged_steps": 20,
            "reset_peak_stats_per_region": True,
            "log_gpu_utilization": True,
            "gpu_utilization_sample_interval_seconds": 2.5,
            "output_subdir": "cuda_memory",
            "report": {"enabled": False, "format": "html"},
        }
    )

    assert config.enabled is True
    assert config.ranks == "all"
    assert config.sync is True
    assert config.log_every_n_steps == 1
    assert config.max_logged_steps == 20
    assert config.reset_peak_stats_per_region is True
    assert config.log_gpu_utilization is True
    assert config.gpu_utilization_sample_interval_seconds == 2.5
    assert config.output_subdir == "cuda_memory"
    assert config.report.enabled is False


def test_query_nvidia_smi_parses_gpu_utilization(monkeypatch):
    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(stdout="97, 44, 120000, 183000, 612.5, 1800, 3200, 61\n")

    monkeypatch.setattr(memory_logger.subprocess, "run", fake_run)

    result = memory_logger._query_nvidia_smi(memory_logger.torch.device("cuda:1"))

    assert result == {
        "gpu_utilization_percent": 97.0,
        "gpu_memory_utilization_percent": 44.0,
        "gpu_memory_used_mb": 120000.0,
        "gpu_memory_total_mb": 183000.0,
        "gpu_power_draw_w": 612.5,
        "gpu_sm_clock_mhz": 1800.0,
        "gpu_memory_clock_mhz": 3200.0,
        "gpu_temperature_c": 61.0,
    }


def test_generate_report_writes_html_and_csvs(tmp_path):
    rows = [
        {
            "time": 1.0,
            "step": 0,
            "tag": "step_start",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "hostname": "host",
            "device": "cuda:0",
            "allocated_mb": 1024.0,
            "reserved_mb": 2048.0,
            "max_allocated_mb": 1024.0,
            "max_reserved_mb": 2048.0,
            "gpu_utilization_percent": 50.0,
            "gpu_memory_utilization_percent": 40.0,
            "gpu_power_draw_w": 500.0,
        },
        {
            "time": 1.75,
            "step": 0,
            "tag": "step_end",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "hostname": "host",
            "device": "cuda:0",
            "allocated_mb": 1280.0,
            "reserved_mb": 2048.0,
            "max_allocated_mb": 2048.0,
            "max_reserved_mb": 4096.0,
            "microbatch": 1,
            "microbatch_size": 4,
            "seq_len": 128,
            "target_tokens": 64,
            "completion_tokens": 16,
            "gpu_utilization_percent": 60.0,
            "gpu_memory_utilization_percent": 50.0,
            "gpu_power_draw_w": 550.0,
        },
        {
            "time": 2.0,
            "step": 1,
            "tag": "forward:summary",
            "region": "forward",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "hostname": "host",
            "device": "cuda:0",
            "allocated_mb": 1536.0,
            "reserved_mb": 2048.0,
            "max_allocated_mb": 3072.0,
            "max_reserved_mb": 4096.0,
            "duration_s": 0.5,
            "start_allocated_mb": 1024.0,
            "end_allocated_mb": 1536.0,
            "delta_allocated_mb": 512.0,
            "peak_allocated_mb": 3072.0,
            "start_reserved_mb": 2048.0,
            "end_reserved_mb": 2048.0,
            "delta_reserved_mb": 0.0,
            "peak_reserved_mb": 4096.0,
            "gpu_utilization_percent": 90.0,
            "gpu_memory_utilization_percent": 80.0,
            "gpu_power_draw_w": 700.0,
        },
        {
            "time": 3.0,
            "step": 1,
            "tag": "backward:summary",
            "region": "backward",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "hostname": "host",
            "device": "cuda:0",
            "allocated_mb": 1024.0,
            "reserved_mb": 2048.0,
            "max_allocated_mb": 3072.0,
            "max_reserved_mb": 4096.0,
            "duration_s": 0.5,
            "start_allocated_mb": 1536.0,
            "end_allocated_mb": 1024.0,
            "delta_allocated_mb": -512.0,
            "peak_allocated_mb": 3072.0,
            "start_reserved_mb": 2048.0,
            "end_reserved_mb": 2048.0,
            "delta_reserved_mb": 0.0,
            "peak_reserved_mb": 4096.0,
            "gpu_utilization_percent": 70.0,
            "gpu_memory_utilization_percent": 60.0,
            "gpu_power_draw_w": 600.0,
        },
    ]
    with (tmp_path / "mem_rank0.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output = generate_report(tmp_path)

    html = output.read_text(encoding="utf-8")
    assert "Memory Over Time By Rank" in html
    assert "GPU Utilization Over Time" in html
    assert "Power Draw Over Time" in html
    assert "Step Duration" in html
    assert "0.75s" in html
    assert "Region Duration By Step" in html
    assert "forward" in html
    assert "backward" in html
    assert "Peak Memory By Rank" not in html
    assert "Peak Memory By Region" in html
    assert "Memory Delta By Region" in html
    assert "Leak Trend Summary" in html
    assert "-0.50 GB" in html
    assert (tmp_path / "memory_summary.csv").exists()
    assert (tmp_path / "memory_region_summary.csv").exists()
    assert (tmp_path / "step_duration_summary.csv").exists()
    assert (tmp_path / "step_region_duration_summary.csv").exists()
    summary_csv = (tmp_path / "memory_summary.csv").read_text(encoding="utf-8")
    assert "mean_gpu_utilization_percent" in summary_csv
    assert "67.5" in summary_csv
    step_csv = (tmp_path / "step_duration_summary.csv").read_text(encoding="utf-8")
    assert "duration_s" in step_csv
    assert "0.75" in step_csv
    step_region_csv = (tmp_path / "step_region_duration_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "region" in step_region_csv
    assert "forward" in step_region_csv
    assert "backward" in step_region_csv
