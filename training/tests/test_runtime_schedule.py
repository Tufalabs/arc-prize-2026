"""Tests for inference.max_runtime_minutes_schedule (train-window runtime ramp)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from test_inference import _minimal_experiment_config_payload
from training.config import ExperimentConfig
from training.inference import RolloutCollector
from training.run_artifacts import (
    _RUN_LAUNCH_TIME_FILENAME,
    _RUN_LAUNCH_TIME_FORMAT,
    read_run_launch_time,
    record_run_launch_time,
)

_SCHEDULE = [[0, 5], [12, 7], [24, 9], [36, 11], [48, 13], [60, 15]]
_LAUNCH_CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/ppo_5gpu_arc3_officials_platypus.json"
)


def _schedule_config(tmp_path: Path, schedule) -> ExperimentConfig:
    payload = _minimal_experiment_config_payload(
        inference={"max_runtime_minutes_schedule": schedule}
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExperimentConfig.load(path)


def _collector(tmp_path: Path, run_root: Path | None) -> RolloutCollector:
    return RolloutCollector(
        _schedule_config(tmp_path, _SCHEDULE),
        repo_root=Path("/nonexistent-repo-root"),
        run_root=run_root,
    )


def _run_root_launched_hours_ago(tmp_path: Path, hours: float) -> Path:
    root = tmp_path / "run_root"
    root.mkdir(exist_ok=True)
    launched = datetime.now() - timedelta(hours=hours)
    (root / _RUN_LAUNCH_TIME_FILENAME).write_text(
        launched.strftime(_RUN_LAUNCH_TIME_FORMAT) + "\n", encoding="utf-8"
    )
    return root


def test_launch_experiment_config_parses_schedule_and_critic() -> None:
    # Smoke-load the real launch config: schedule plus the new training keys.
    config = ExperimentConfig.load(_LAUNCH_CONFIG_PATH)
    assert config.inference.max_runtime_minutes_schedule == [
        (0.0, 5.0),
        (24.0, 15.0),
    ]
    assert config.training.value_baseline == "hidden_critic"
    assert config.training.pinned_adapter_interval == 25
    assert config.inference.agent_reasoning_effort == "xhigh"


def test_agent_reasoning_effort_validation(tmp_path) -> None:
    payload = _minimal_experiment_config_payload(
        inference={"agent_reasoning_effort": "MEDIUM"}
    )
    path = tmp_path / "effort.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert ExperimentConfig.load(path).inference.agent_reasoning_effort == "medium"

    payload["inference"]["agent_reasoning_effort"] = "maximum"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="agent_reasoning_effort"):
        ExperimentConfig.load(path)


def test_record_run_launch_time_is_write_once(tmp_path) -> None:
    record_run_launch_time(tmp_path)
    first = read_run_launch_time(tmp_path)
    assert first is not None
    # A later leg must keep the original anchor.
    (tmp_path / _RUN_LAUNCH_TIME_FILENAME).write_text(
        "20200101_000000\n", encoding="utf-8"
    )
    record_run_launch_time(tmp_path)
    assert read_run_launch_time(tmp_path) == datetime(2020, 1, 1)
    assert read_run_launch_time(None) is None
    assert read_run_launch_time(tmp_path / "missing") is None


def test_schedule_interpolates_by_elapsed_hours(tmp_path) -> None:
    # The ramp is piecewise-linear: 13h sits 1/12 of the way from the
    # (12h, 7min) point to the (24h, 9min) point.
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 13.0))
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(
        7.0 + (13.0 - 12.0) / 12.0 * 2.0, abs=0.02
    )
    # Halfway through the first span: 6h -> 6 minutes.
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 6.0))
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(
        6.0, abs=0.02
    )
    # 0h elapsed -> first control point.
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 0.001))
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(
        5.0, abs=0.02
    )
    # Exactly on a control point -> its value.
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 48.0))
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(
        13.0, abs=0.02
    )
    # Far beyond the last point -> final value holds.
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 1000.0))
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(15.0)


def test_schedule_never_applies_to_eval_windows(tmp_path) -> None:
    collector = _collector(tmp_path, _run_root_launched_hours_ago(tmp_path, 1000.0))
    assert collector._effective_max_runtime_minutes("eval") == pytest.approx(5.0)


def test_schedule_falls_back_without_launch_marker(tmp_path, caplog) -> None:
    unmarked = tmp_path / "unmarked_run_root"
    unmarked.mkdir()
    collector = _collector(tmp_path, unmarked)
    with caplog.at_level("WARNING"):
        assert collector._effective_max_runtime_minutes("train") == pytest.approx(5.0)
        assert collector._effective_max_runtime_minutes("train") == pytest.approx(5.0)
    warnings = [
        record
        for record in caplog.records
        if "max_runtime_minutes_schedule" in record.getMessage()
    ]
    assert len(warnings) == 1  # warn once, not per window

    collector = _collector(tmp_path, None)
    assert collector._effective_max_runtime_minutes("train") == pytest.approx(5.0)


@pytest.mark.parametrize(
    "schedule, message",
    [
        ([[0, 5], [0, 7]], "strictly increasing"),
        ([[-1, 5]], ">= 0"),
        ([[0, 0]], "> 0"),
        ([[0]], "pairs"),
        ("12:5", "list"),
    ],
)
def test_schedule_validation_rejects_malformed_entries(
    tmp_path, schedule, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _schedule_config(tmp_path, schedule)
