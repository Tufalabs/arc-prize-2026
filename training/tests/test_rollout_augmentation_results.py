"""The augmentation travels trainer -> worker -> result -> trajectory, and back
to the trainer's audit, for both harnesses."""

from __future__ import annotations

import inspect
import io
import json
from pathlib import Path

import pytest

from harness_support import augmentation
from harness_support.agent_settings import record_agent_settings
from harness_support.augmentation import (
    augmentation_seed,
    sample_arc2,
    sample_arc3,
    verify_rollouts,
)
from test_arc2_integration import result_payload
from test_full_game_slots import _spec, _worker
from test_platypus_rollout import _window as arc3_window
from training.platypus_rollout import build_window_trajectories
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
)


def _params(environment, window=5):
    return {
        "version": 1,
        "environment": environment,
        "run_seed": 17,
        "window": window,
        "dihedral": environment == "arc2",
        "colors": True,
    }


def _arc3_record(params, game):
    seed = augmentation_seed(params["run_seed"], params["window"], game)
    return sample_arc3(seed).record(puzzle=game, params=params)


def _journal(job_dir, slot, record):
    root = Path(job_dir) / augmentation.JOURNAL_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{slot}.json").write_text(json.dumps(record))


def _post(worker, spec, monkeypatch):
    captured = []

    def capture(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(b'{"accepted": 2}')

    monkeypatch.setattr(worker.urllib.request, "urlopen", capture)
    worker._post_results(spec)
    return captured[0]["runs"]


def _results_spec(tmp_path):
    spec = _spec(tmp_path)
    spec.update(
        games=["ar25", "ls20"],
        results_url="http://unused.invalid/results",
        max_runtime_s_per_game=60,
    )
    (Path(tmp_path) / "benchmark.json").write_text(
        json.dumps(
            {
                "game_runs": [
                    {"run_id": "first", "game_id": "ar25#s0", "state": "gave_up"},
                    {"run_id": "second", "game_id": "ls20#s1", "state": "gave_up"},
                ]
            }
        ),
        encoding="utf-8",
    )
    # What build_benchmark records for the window's agent.
    record_agent_settings(Path(tmp_path), {"max_tokens_per_level": [90_000]})
    return spec


def test_the_worker_reports_what_each_slot_was_built_with(tmp_path, monkeypatch):
    worker = _worker()
    spec = _results_spec(tmp_path)
    assert [run["augmentation"] for run in _post(worker, spec, monkeypatch)] == [None, None]
    spec["augmentation"] = _params("arc3")
    records = [_arc3_record(spec["augmentation"], game) for game in ("ar25", "ls20")]
    _journal(tmp_path, "ar25#s0", records[0])
    with pytest.raises(RuntimeError, match="'ls20#s1' without an augmentation"):
        _post(worker, spec, monkeypatch)
    _journal(tmp_path, "ls20#s1", records[1])
    runs = _post(worker, spec, monkeypatch)
    assert [run["augmentation"] for run in runs] == records


def test_arc2_trajectories_carry_the_record_the_trainer_rederives():
    params = _params("arc2")
    seed = augmentation_seed(17, 5, "puzzle")
    record = sample_arc2(seed, [0, 1, 3]).record(puzzle="puzzle", params=params)
    run = RunResult.from_payload({**result_payload(), "augmentation": record})
    capture = StepCapture(
        "abc", "first", [1, 2], [3, 4], [-0.1, -0.2], "stop", 0,
        native_token_prompt=True, adapter=run.behavior_policy_snapshot,
    )
    window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
    rollouts = build_window_trajectories(window, policy_iteration=1, max_runtime_seconds=900.0)
    (rollout,) = rollouts
    assert rollout.metadata["augmentation"] == record
    assert rollout.metadata["arc2_outcome"]["augmentation"] == record
    verify_rollouts(rollouts, params)
    with pytest.raises(ValueError, match="unaugmented window"):
        verify_rollouts(rollouts, None)
    with pytest.raises(ValueError, match="differs from its window"):
        verify_rollouts(rollouts, _params("arc2", window=6))


def test_arc3_trajectories_carry_the_record_the_trainer_rederives():
    params = _params("arc3")
    window = arc3_window()
    (run,) = window.results.values()
    record = _arc3_record(params, run.game_id)
    run.augmentation = record
    rollouts = build_window_trajectories(
        window, policy_iteration=7, max_runtime_seconds=300.0, game_ids=["pathfinder"]
    )
    assert rollouts[0].metadata["augmentation"] == record
    verify_rollouts(rollouts, params)
    run.augmentation = None
    plain = build_window_trajectories(window, policy_iteration=7, max_runtime_seconds=300.0)
    assert plain[0].metadata["augmentation"] is None
    verify_rollouts(plain, None)
    with pytest.raises(ValueError, match="reported no augmentation"):
        verify_rollouts(plain, params)


def test_the_gateway_admits_only_an_object_or_null():
    assert RunResult.from_payload(result_payload()).augmentation is None
    with pytest.raises(GatewayError, match="augmentation must be an object or null"):
        RunResult.from_payload({**result_payload(), "augmentation": [1, 2]})


def test_the_collector_sends_the_window_augmentation_and_audits_its_rollouts():
    """Train windows send the setting with the run seed and schedule index;
    every collected window's rollouts are checked against it."""
    from training.inference import RolloutCollector

    source = inspect.getsource(RolloutCollector._collect_window_platypus)
    assert '"augmentation": augmentation.window_spec(' in source
    for argument in (
        "inference_config.augmentation",
        "environment=inference_config.environment",
        "run_seed=self.config.training.seed",
        "window=schedule_index",
        "kind=window_kind",
    ):
        assert argument in source
    assert 'augmentation.verify_rollouts(rollouts, spec["augmentation"])' in source
