from collections import Counter, deque
import json
import logging
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.config import (
    ExperimentConfig,
    InferenceConfig,
    LoraSettings,
    TrainingConfig,
)
from training.contracts import RolloutTrajectory
from training.cli import (
    SynchronousRolloutContractError,
    _mean_game_return_history,
    _new_game_return_history,
    _load_training_run_state,
    _initial_sample_budget_state,
    _resolve_resume_state,
    _restore_game_return_history,
    _rollout_status_scalars,
    _save_config_snapshots,
    _policy_lag_windows,
    _update_game_arc_score_history,
    _update_game_levels_history,
    _update_game_return_history,
    _update_sample_budget_state,
    _validate_synchronous_rollout_window,
    _write_per_game_metric_group,
    _write_training_tensorboard_scalars,
    _write_training_run_state,
)
from training.inference import (
    CollectedWindow,
    ContinuousRolloutWorker,
    RolloutCollector,
    _platypus_window_outcomes,
    _rollout_status_counts,
    _wait_for_adapter_ready,
)


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        num_iterations=1,
        parallel_rollouts=2,
        rollout_buffer_size=1,
        inference=InferenceConfig(
            base_url="http://127.0.0.1:1234",
            model_id="local-qwen",
            game="pathfinder-0001",
            games=["pathfinder-0001"],
            max_runtime_minutes=10.0,
            platypus_venv="/tmp/platypus-venv",
        ),
        training=TrainingConfig(
            model_name="Qwen/Qwen3.5-9B",
            output_dir="runs",
            device="cpu",
            seed=7,
            max_sequence_tokens=512,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            epochs_per_window=1,
            learning_rate=1e-5,
            weight_decay=0.0,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            max_grad_norm=1.0,
            disable_dropout=True,
            bf16=False,
            gradient_checkpointing=False,
            lora=LoraSettings(
                r=8,
                alpha=16,
                dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        ),
    )


def _minimal_experiment_config_payload(
    *,
    parallel_rollouts: int = 6,
    top_level: dict | None = None,
    inference: dict | None = None,
    training: dict | None = None,
) -> dict:
    payload = {
        "num_iterations": 1,
        "parallel_rollouts": parallel_rollouts,
        "rollout_buffer_size": 1,
        "inference": {
            "agent_max_completion_tokens": 4096,
            "base_url": "http://127.0.0.1:1234",
            "model_id": "local-qwen",
            "games": ["pathfinder-0001"],
            "max_runtime_minutes": 5.0,
            "platypus_venv": "/tmp/platypus-venv",
        },
        "training": {
            "model_name": "Qwen/Qwen3.5-9B",
            "value_baseline": "hidden_critic",
            "output_dir": "runs",
            "device": "cpu",
            "seed": 7,
            "max_sequence_tokens": 512,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs_per_window": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "max_grad_norm": 1.0,
            "bf16": False,
            "gradient_checkpointing": False,
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": ["q_proj"],
            },
        },
    }
    payload.update(top_level or {})
    payload["inference"].update(inference or {})
    payload["training"].update(training or {})
    return payload


def _write_experiment_config(path: Path, **overrides) -> None:
    path.write_text(
        json.dumps(_minimal_experiment_config_payload(**overrides)),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("section_path", "unknown_field", "reported_section"),
    [
        ((), "num_iteratons", "config"),
        (("inference",), "agent_temperatur", "inference"),
        (("training",), "learnng_rate", "training"),
        (("training", "lora"), "rank", "training.lora"),
        (("training", "sample_budget"), "target_ratio", "training.sample_budget"),
        (
            ("training", "entropy_control"),
            "minimum_coef",
            "training.entropy_control",
        ),
        (("metrics",), "per_game", "metrics"),
        (("sampling",), "history_size", "sampling"),
        (("official_eval",), "cycle_seconds", "official_eval"),
        (("memory_profiler",), "log_interval", "memory_profiler"),
        (
            ("memory_profiler", "report"),
            "file_format",
            "memory_profiler.report",
        ),
        (("viewer",), "host", "viewer"),
        (("inference",), "_comment", "inference"),
        (("training",), "quantization_mode", "training"),
    ],
)
def test_experiment_config_rejects_unknown_fields_in_every_section(
    tmp_path: Path,
    section_path: tuple[str, ...],
    unknown_field: str,
    reported_section: str,
) -> None:
    payload = _minimal_experiment_config_payload()
    section = payload
    for name in section_path:
        section = section.setdefault(name, {})
    section[unknown_field] = "typo"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(reported_section)}\.{re.escape(unknown_field)}",
    ):
        ExperimentConfig.load(config_path)


def test_experiment_config_rejects_non_object_nested_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = _minimal_experiment_config_payload(training={"lora": []})
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training.lora must be a JSON object"):
        ExperimentConfig.load(config_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_experiment_config_rejects_non_finite_json_numbers(
    tmp_path: Path,
    constant: str,
) -> None:
    config_path = tmp_path / "config.json"
    payload = _minimal_experiment_config_payload()
    payload["inference"]["agent_temperature"] = f"__{constant}__"
    serialized = json.dumps(payload).replace(f'"__{constant}__"', constant)
    config_path.write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite numeric value"):
        ExperimentConfig.load(config_path)


def test_runtime_fingerprint_drift_warns_instead_of_failing_by_default(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = RolloutCollector(repo_root=tmp_path, config=_experiment_config())
    monkeypatch.setenv(
        "ARC3_RUNTIME_FINGERPRINT_PATH", str(tmp_path / "runtime_fingerprint.json")
    )
    monkeypatch.delenv("ARC3_STRICT_RUNTIME_FINGERPRINT", raising=False)

    def fail_fingerprint_check(_path: str) -> None:
        raise RuntimeError("changed")

    monkeypatch.setattr(
        "training.inference.verify_runtime_fingerprint",
        fail_fingerprint_check,
    )
    caplog.set_level(logging.WARNING)

    collector._verify_runtime_fingerprint()
    collector._verify_runtime_fingerprint()

    warnings = [
        record
        for record in caplog.records
        if "runtime Python environment fingerprint changed" in record.message
    ]
    assert len(warnings) == 1


def test_runtime_fingerprint_strict_mode_still_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = RolloutCollector(repo_root=tmp_path, config=_experiment_config())
    monkeypatch.setenv(
        "ARC3_RUNTIME_FINGERPRINT_PATH", str(tmp_path / "runtime_fingerprint.json")
    )
    monkeypatch.setenv("ARC3_STRICT_RUNTIME_FINGERPRINT", "1")

    def fail_fingerprint_check(_path: str) -> None:
        raise RuntimeError("changed")

    monkeypatch.setattr(
        "training.inference.verify_runtime_fingerprint",
        fail_fingerprint_check,
    )

    with pytest.raises(RuntimeError, match="changed"):
        collector._verify_runtime_fingerprint()


def test_runtime_lora_policy_refresh_marks_new_policy_for_sync(tmp_path) -> None:
    config = _experiment_config()
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )

    first_policy = tmp_path / "policy-0"
    second_policy = tmp_path / "policy-1"

    assert worker._should_sync(first_policy, 0) is True

    worker._active_policy_path = first_policy
    worker._active_policy_iteration = 0

    assert worker._should_sync(first_policy, 1) is True
    assert worker._should_sync(second_policy, 1) is True


def test_initial_lora_and_first_update_both_sync(tmp_path) -> None:
    worker = ContinuousRolloutWorker(
        _experiment_config(), repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )
    initial = tmp_path / "initial-lora"
    first = tmp_path / "policy-0"
    assert worker._should_sync(initial, -1)
    worker._active_policy_path = initial
    worker._active_policy_iteration = -1
    assert not worker._should_sync(initial, -1)
    assert worker._should_sync(first, 0)


def test_production_collector_rejects_adapter_free_engine_before_dispatch(
    tmp_path, monkeypatch
):
    from training.rollout_gateway import CaptureGateway, GatewayError

    config = ExperimentConfig.load(
        Path(__file__).resolve().parents[1] / "configs/ppo_2gpu_arc3_games25_platypus.json"
    )
    collector = RolloutCollector(config=config, repo_root=tmp_path)
    gateway = CaptureGateway(upstream_base_url="http://unused.invalid", port=0)
    monkeypatch.setattr(collector, "_platypus_gateway", lambda: gateway)
    monkeypatch.setattr(
        collector,
        "adapter_identity",
        lambda model: {
            "name": model,
            "id": 0,
            "path": None,
            "server_instance_id": "base",
        },
    )
    with pytest.raises(GatewayError, match="require a loaded LoRA"):
        collector._collect_window_platypus(
            window_id="initial",
            window_dir=tmp_path / "window",
            slots=[],
            per_rollout_runtime_seconds=900,
            policy_iteration=-1,
            schedule_index=0,
            model_id="base",
            collection_started_at=0,
        )
    assert not (tmp_path / "window").exists()


def test_fixed_policy_lag_grows_after_the_initial_same_policy_window() -> None:
    assert (
        _policy_lag_windows(
            iteration_index=0,
            rollout_policy_iteration=-1,
        )
        == 0
    )
    assert (
        _policy_lag_windows(
            iteration_index=1,
            rollout_policy_iteration=-1,
        )
        == 1
    )
    assert (
        _policy_lag_windows(
            iteration_index=2,
            rollout_policy_iteration=-1,
        )
        == 2
    )


def test_single_window_rollout_buffer_caps_overlap_to_one_window(tmp_path) -> None:
    config = _experiment_config()
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )

    assert worker._buffer.maxsize == 1


def test_training_window_exposes_conservative_rollout_eta(
    monkeypatch, tmp_path
) -> None:
    import training.inference as training_inference

    config = _experiment_config()
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )
    clock = {"now": 100.0}
    observed: dict[str, float | None] = {}
    monkeypatch.setattr(training_inference.time, "monotonic", lambda: clock["now"])

    def fake_collect_window(**kwargs) -> CollectedWindow:
        observed["seconds_remaining"] = worker.seconds_until_window_ready()
        clock["now"] += 2.0
        return CollectedWindow(
            window_id="window-eta",
            window_dir=tmp_path / "rollouts" / "window-eta",
            rollouts=[],
            policy_iteration=-1,
        )

    monkeypatch.setattr(worker, "_collect_window_across_servers", fake_collect_window)

    worker._active_policy_path = Path("/test/initialized-adapter")
    worker._active_policy_iteration = -1
    monkeypatch.setattr(worker, "_sync_pending_policy_if_needed", lambda: None)
    worker._collect_training_window()

    assert observed["seconds_remaining"] == pytest.approx(600.0)
    assert worker.seconds_until_window_ready() == pytest.approx(0.0)


def test_continuous_worker_keeps_durable_state_on_run_root(tmp_path) -> None:
    """With rollout windows on node-local scratch, durable per-run state stays
    on run_root while server logs follow the durable run root."""
    config = _experiment_config()
    run_root = tmp_path / "run"
    scratch_windows_root = tmp_path / "scratch" / "rollout_windows" / "run"
    worker = ContinuousRolloutWorker(
        config,
        repo_root=tmp_path,
        output_root=scratch_windows_root / "rollouts",
        run_root=run_root,
    )

    assert worker.run_root == run_root
    assert worker.server_log_path == run_root / "inference_server.log"
    assert worker.server_log_paths == [run_root / "inference_server.log"]


def test_base_model_id_prefers_explicit_inference_model_id(tmp_path) -> None:
    config = _experiment_config()
    config.inference.model_id = "served-base-model"
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )

    assert worker._base_model_id() == "served-base-model"


def test_rollout_slots_use_explicit_scheduled_seeds(tmp_path) -> None:
    config = _experiment_config()
    collector = RolloutCollector(config, repo_root=tmp_path)

    slots = collector._rollout_slots(
        policy_iteration=0,
        scheduled_games=["a-0001", "b-0001"],
        scheduled_rollout_seeds=[42, 43],
    )

    assert [slot.game_id for slot in slots] == ["a-0001", "b-0001"]
    assert [slot.rollout_seed for slot in slots] == ["42", "43"]


def test_rollout_seed_override_takes_precedence_over_scheduled_seeds(
    monkeypatch, tmp_path
) -> None:
    config = _experiment_config()
    collector = RolloutCollector(config, repo_root=tmp_path)
    monkeypatch.setenv("ARC3_ROLLOUT_SEED", "99")

    slots = collector._rollout_slots(
        policy_iteration=0,
        scheduled_games=["a-0001", "b-0001"],
        scheduled_rollout_seeds=[42, 43],
    )

    assert [slot.rollout_seed for slot in slots] == ["99", "99"]


def test_rollout_slots_sample_training_games_with_replacement(
    tmp_path,
) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 8
    config.inference.games = ["pathfinder-0001", "identify_the_agent-0002"]
    collector = RolloutCollector(config, repo_root=tmp_path)

    slots = collector._rollout_slots(policy_iteration=3)

    game_counts = Counter(slot.game_id for slot in slots)
    assert len(slots) == 8
    assert set(game_counts) <= set(config.inference.games)
    assert max(game_counts.values()) > 1
    assert all(slot.rollout_seed is None for slot in slots)


def test_rollout_slots_do_not_force_full_training_game_coverage_within_window(
    tmp_path,
) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 4
    config.inference.games = [
        "rotate_pattern-0002",
        "taps-0001",
        "identify_the_agent-0002",
        "stack_equalizer-0001",
        "pathfinder-0001",
        "match_sliders-0001",
    ]
    collector = RolloutCollector(config, repo_root=tmp_path)

    slots = collector._rollout_slots(policy_iteration=0)
    game_counts = Counter(slot.game_id for slot in slots)

    assert sum(game_counts.values()) == 4
    assert set(game_counts) <= set(config.inference.games)
    assert len(game_counts) < len(config.inference.games)


def test_rollout_slots_allow_training_repeats_within_each_window(tmp_path) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 8
    config.inference.games = [
        "game-0000",
        "game-0001",
    ]
    collector = RolloutCollector(config, repo_root=tmp_path)

    windows: list[list[str]] = []
    for policy_iteration in range(-1, 4):
        slots = collector._rollout_slots(policy_iteration=policy_iteration)
        windows.append([slot.game_id for slot in slots])

    assert all(len(window) == 8 for window in windows)
    assert all(len(set(window)) <= 2 for window in windows)
    assert all(max(Counter(window).values()) > 1 for window in windows)
    assert all(set(window) <= set(config.inference.games) for window in windows)
    assert any(
        set(left) & set(right)
        for left, right in zip(windows[:-1], windows[1:], strict=True)
    )


def test_rollout_slots_are_deterministic_for_fixed_schedule_index(tmp_path) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 2
    config.inference.games = [
        "game-0000",
        "game-0001",
        "game-0002",
    ]
    collector = RolloutCollector(config, repo_root=tmp_path)

    first = [
        slot.game_id
        for slot in collector._rollout_slots(policy_iteration=-1, schedule_index=3)
    ]
    second = [
        slot.game_id
        for slot in collector._rollout_slots(policy_iteration=-1, schedule_index=3)
    ]

    assert first == second
    assert len(first) == 2
    assert set(first) <= set(config.inference.games)


def test_rollout_slots_replacement_sampling_is_deterministic(tmp_path) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 6
    config.inference.games = [
        "rotate_pattern-0002",
        "taps-0001",
        "identify_the_agent-0002",
        "stack_equalizer-0001",
    ]
    collector = RolloutCollector(config, repo_root=tmp_path)

    first_window = collector._rollout_slots(policy_iteration=-1)
    second_window = collector._rollout_slots(policy_iteration=0)
    windows = [
        [slot.game_id for slot in first_window],
        [slot.game_id for slot in second_window],
    ]

    for window in windows:
        game_counts = Counter(window)
        assert len(window) == 6
        assert set(game_counts) <= set(config.inference.games)
        assert max(game_counts.values()) > 1
    assert first_window == collector._rollout_slots(policy_iteration=-1)


def test_experiment_config_allows_multiple_training_games(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_experiment_config(
        config_path,
        parallel_rollouts=4,
        inference={"games": ["pathfinder-0001", "identify_the_agent-0002"]},
    )

    config = ExperimentConfig.load(config_path)

    assert config.inference.games == ["pathfinder-0001", "identify_the_agent-0002"]
    assert config.inference.game == "pathfinder-0001"


def test_rollout_window_logs_both_scoring_methods() -> None:
    # Rollout and eval must expose the SAME summarize_rollouts scalar set
    # (incl. dispersion: return_std/min/max), independent of reward_mode.
    # Rollout starts with the full as_scalars() dict, overrides avg_return,
    # and only keeps the history-based avg_arc_score for weighted sampling;
    # eval writes the full set with the "eval" prefix.
    training_source = Path("training/cli.py").read_text(encoding="utf-8")
    eval_source = Path("training/official_eval.py").read_text(encoding="utf-8")
    assert "dict(rollout_metrics.as_scalars())" in training_source
    assert '"rollout/avg_arc_score"' in training_source  # history-based override
    assert 'metrics.as_scalars(scalar_prefix="eval")' in eval_source
    from training.metrics import summarize_rollouts

    scalars = summarize_rollouts([]).as_scalars()
    assert "rollout/return_std" in scalars
    assert "rollout/avg_levels_completed_pct" in scalars


def test_failed_native_adapter_load_is_not_retried(monkeypatch, tmp_path):
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=503, text="uncertain engine update")

    monkeypatch.setattr("training.inference.requests.post", post)
    with pytest.raises(RuntimeError, match="Failed to load"):
        collector.load_runtime_adapter(adapter_name="policy", adapter_path=tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:1234/adapters/load"
    assert calls[0][1]["json"] == {"name": "policy", "path": str(tmp_path)}


def _runtime_export(adapter_dir: Path, modules=("q_proj", "v_proj"), tensors=8) -> Path:
    """The summary training/runtime_lora.py writes beside a runtime export."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "runtime_adapter_summary.json").write_text(
        json.dumps(
            {"runtime_target_modules": list(modules), "converted_tensor_count": tensors}
        ),
        encoding="utf-8",
    )
    return adapter_dir


# The gateway's census of that export: q/v merged into the packed qkv_proj
# layer, each in two layers (four modules, eight tensors).
_CENSUS = {
    "applied": {"q_proj": "qkv_proj", "v_proj": "qkv_proj"},
    "modules": {"q_proj": 2, "v_proj": 2},
}


def test_load_runtime_adapter_uses_server_api_key_env(monkeypatch, tmp_path) -> None:
    config = _experiment_config()
    monkeypatch.setenv("SERVER_API_KEY", "secret-token")
    collector = RolloutCollector(config, repo_root=tmp_path)
    adapter_dir = _runtime_export(tmp_path / "adapter")
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "adapter": {
                    "name": "policy",
                    "id": 1,
                    "path": str(adapter_dir),
                    "server_instance_id": "test-engine",
                },
                "lora_census": _CENSUS,
            }

    def fake_post(url, json, headers, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["json"] = dict(json)
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("training.inference.requests.post", fake_post)

    collector.load_runtime_adapter(adapter_name="policy", adapter_path=adapter_dir)

    assert captured["url"] == "http://127.0.0.1:1234/adapters/load"
    assert captured["headers"] == {"Authorization": "Bearer secret-token"}
    assert collector.loaded_runtime_adapter_census == _CENSUS


@pytest.mark.parametrize(
    ("census", "export", "match"),
    [
        # A gateway that returns no census, or a malformed one.
        (None, {}, "no valid LoRA module census"),
        ({"applied": {"q_proj": "qkv_proj"}}, {}, "no valid LoRA module census"),
        (_CENSUS | {"modules": {"q_proj": 2, "v_proj": 0}}, {}, "no valid"),
        (_CENSUS | {"applied": {"q_proj": "qkv_proj", "v_proj": ""}}, {}, "no valid"),
        # An exported module the engine does not apply.
        (
            {"applied": {"q_proj": "qkv_proj"}, "modules": {"q_proj": 4}},
            {},
            r"missing \['v_proj'\], unexpected \[\]",
        ),
        # A module the export never wrote.
        (_CENSUS, {"modules": ("q_proj",), "tensors": 4}, r"unexpected \['v_proj'\]"),
        # Every target, but not every exported tensor.
        (_CENSUS, {"tensors": 12}, r"\(4 modules\).*\(12 tensors\)"),
    ],
)
def test_load_requires_a_census_applying_exactly_the_exported_modules(
    monkeypatch, tmp_path, census, export, match
):
    """The engine must apply every module the runtime export wrote and no
    other; anything else is fatal before the adapter serves a rollout."""
    adapter_dir = _runtime_export(tmp_path / "adapter", **export)
    identity = {
        "name": "policy",
        "id": 3,
        "path": str(adapter_dir),
        "server_instance_id": "engine-a",
    }
    monkeypatch.setattr(
        "training.inference.requests.post",
        lambda *a, **k: SimpleNamespace(
            status_code=200, json=lambda: {"adapter": identity, "lora_census": census}
        ),
    )
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    with pytest.raises(RuntimeError, match=match):
        collector.load_runtime_adapter(adapter_name="policy", adapter_path=adapter_dir)
    assert getattr(collector, "_loaded_runtime_adapter_identity", None) is None


def test_load_without_the_runtime_export_summary_is_fatal(monkeypatch, tmp_path):
    identity = {
        "name": "policy",
        "id": 3,
        "path": str(tmp_path),
        "server_instance_id": "engine-a",
    }
    monkeypatch.setattr(
        "training.inference.requests.post",
        lambda *a, **k: SimpleNamespace(
            status_code=200, json=lambda: {"adapter": identity, "lora_census": _CENSUS}
        ),
    )
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="runtime LoRA export summary"):
        collector.load_runtime_adapter(adapter_name="policy", adapter_path=tmp_path)


def test_wait_for_adapter_ready_times_out_when_missing(tmp_path, monkeypatch) -> None:
    # The production timeout is 30s; the timeout SEMANTICS (poll until
    # deadline, then raise naming the missing ready file) are what matters,
    # so shrink the deadline to keep the suite fast.
    monkeypatch.setattr("training.inference._ADAPTER_READY_TIMEOUT", 0.3)
    monkeypatch.setattr("training.inference._ADAPTER_READY_POLL", 0.05)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(RuntimeError, match="adapter_config.json"):
        _wait_for_adapter_ready(adapter_dir)


def test_wait_for_adapter_ready_accepts_existing_adapter(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    _wait_for_adapter_ready(adapter_dir)


def test_experiment_config_honors_inference_base_url_env_override(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.json"
    _write_experiment_config(config_path, parallel_rollouts=1)

    monkeypatch.setenv("ARC3_INFERENCE_BASE_URL", "http://127.0.0.1:24567")

    config = ExperimentConfig.load(config_path)

    assert config.inference.base_url == "http://127.0.0.1:24567"


def test_rollout_return_history_state_round_trips(tmp_path) -> None:
    run_root = tmp_path / "run"
    adapter_dir = run_root / "checkpoints" / "adapter_iter_000003"
    runtime_adapter_dir = adapter_dir / "runtime_vllm"
    trainer_state_path = run_root / "checkpoints" / "trainer_state_iter_000003.pt"
    runtime_adapter_dir.mkdir(parents=True)
    trainer_state_path.write_bytes(b"state")
    train_result = SimpleNamespace(
        adapter_dir=adapter_dir,
        runtime_adapter_dir=runtime_adapter_dir,
        trainer_state_path=trainer_state_path,
        optimizer_steps=2,
        advantage_group_mean_reward=0.25,
        advantage_group_std_reward=0.1,
    )
    trainer = SimpleNamespace(optimizer_step_count=17)
    rollout_return_history = _new_game_return_history(
        ["easy_taps-0001", "pathfinder-0001", "memory-0001"],
        history_size=2,
    )
    _update_game_return_history(
        rollout_return_history,
        history_size=2,
        rollouts=[
            RolloutTrajectory(
                trajectory_id="easy-a",
                task_id="easy_taps-0001",
                steps=[],
                trajectory_reward=10.0,
                metadata={},
            ),
            RolloutTrajectory(
                trajectory_id="easy-b",
                task_id="easy_taps-0001",
                steps=[],
                trajectory_reward=30.0,
                metadata={},
            ),
            RolloutTrajectory(
                trajectory_id="pathfinder",
                task_id="pathfinder-0001",
                steps=[],
                trajectory_reward=6.0,
                metadata={},
            ),
        ],
    )
    _write_training_run_state(
        run_root=run_root,
        config_path="configs/ppo_8gpu_official.json",
        config=_experiment_config(),
        iteration_metrics={"iteration": 3, "avg_return": 0.5},
        train_result=train_result,
        trainer=trainer,
        rollout_return_history=rollout_return_history,
    )

    state = _load_training_run_state(run_root)

    assert state is not None
    restored = _restore_game_return_history(
        ["easy_taps-0001", "pathfinder-0001", "memory-0001"],
        history_size=2,
        payload=state["rollout_return_history"],
    )
    avg_return, observed_games, configured_games = _mean_game_return_history(
        restored,
        ["easy_taps-0001", "pathfinder-0001", "memory-0001"],
    )

    assert avg_return == pytest.approx(((10.0 + 30.0) / 2 + 6.0 + 0.0) / 3)
    assert observed_games == 2
    assert configured_games == 3


def test_levels_history_supports_official_subset_mean(tmp_path) -> None:
    history = _new_game_return_history(
        ["bp35-0a0ad940", "anchor_drift-0001"],
        history_size=4,
    )
    _update_game_levels_history(
        history,
        history_size=4,
        rollouts=[
            RolloutTrajectory(
                trajectory_id="official-a",
                task_id="bp35-0a0ad940",
                steps=[],
                trajectory_reward=0.0,
                metadata={"levels_completed": 3},
            ),
            RolloutTrajectory(
                trajectory_id="official-b",
                task_id="bp35-0a0ad940",
                steps=[],
                trajectory_reward=0.0,
                metadata={"levels_completed": 1},
            ),
            RolloutTrajectory(
                trajectory_id="port-a",
                task_id="anchor_drift-0001",
                steps=[],
                trajectory_reward=0.0,
                metadata={},
            ),
        ],
    )

    official_avg, observed_games, configured_games = _mean_game_return_history(
        history, ["bp35-0a0ad940"]
    )

    assert official_avg == pytest.approx(2.0)
    assert observed_games == 1
    assert configured_games == 1


def test_resume_state_requires_explicit_run_dir(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    state = {
        "version": 1,
        "adapter_dir": "checkpoints/adapter_iter_000003",
        "runtime_adapter_dir": "checkpoints/adapter_iter_000003/runtime_vllm",
        "trainer_state_path": "checkpoints/trainer_state_iter_000003.pt",
    }
    (run_root / "training_state.json").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume was not explicitly requested"):
        _resolve_resume_state(run_root, requested_run_dir=False)

    assert _resolve_resume_state(run_root, requested_run_dir=True) == state


def test_existing_checkpoint_artifacts_without_resume_state_reject_implicit_run_dir(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "checkpoints" / "adapter_iter_000003").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Refusing to start a fresh experiment"):
        _resolve_resume_state(run_root, requested_run_dir=False)


def test_effective_config_exposes_resolved_games_to_dedicated_eval(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    source_config_path = tmp_path / "source.json"
    source_config_path.write_text("{}", encoding="utf-8")
    config = _experiment_config()
    config.training.value_baseline = "hidden_critic"
    config.inference.games = ["train-a-0001", "train-b-0001"]
    config.official_eval.enabled = True
    config.official_eval.games = ["eval-a-0001", "eval-b-0001"]
    config.official_eval.server_gpus = ["3"]

    effective_config_path = _save_config_snapshots(
        run_root, str(source_config_path), config
    )
    payload = json.loads(effective_config_path.read_text(encoding="utf-8"))

    assert effective_config_path == run_root / "effective_config.json"
    assert payload["inference"]["games"] == ["train-a-0001", "train-b-0001"]
    assert "game" not in payload["inference"]
    assert "eval_games" not in payload["inference"]
    assert payload["official_eval"]["games"] == ["eval-a-0001", "eval-b-0001"]
    reloaded = ExperimentConfig.load(effective_config_path)
    assert reloaded.inference.games == ["train-a-0001", "train-b-0001"]
    assert reloaded.official_eval.games == ["eval-a-0001", "eval-b-0001"]


def test_per_game_metric_group_logs_sampled_games_only() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}

        def add_scalar(self, name: str, value: float, iteration_index: int) -> None:
            self.scalars[name] = (float(value), iteration_index)

    rollouts = [
        RolloutTrajectory(
            trajectory_id="pathfinder-a",
            task_id="pathfinder-0001",
            steps=[],
            trajectory_reward=1.0,
            metadata={"levels_completed": 1, "final_arc_score": 20.0},
        ),
        RolloutTrajectory(
            trajectory_id="pathfinder-b",
            task_id="pathfinder-0001",
            steps=[],
            trajectory_reward=3.0,
            metadata={"levels_completed": 2, "final_arc_score": 30.0},
        ),
        RolloutTrajectory(
            trajectory_id="maze",
            task_id="maze/odd id",
            steps=[],
            trajectory_reward=4.0,
            metadata={"levels_completed": 3, "final_arc_score": 40.0},
        ),
    ]
    writer = RecordingWriter()

    metrics = _write_per_game_metric_group(
        writer,
        prefix="rollout_by_game",
        iteration_index=9,
        rollouts=rollouts,
    )

    assert writer.scalars["rollout_by_game/pathfinder-0001/avg_return"] == (2.0, 9)
    assert writer.scalars["rollout_by_game/pathfinder-0001/avg_levels_completed"] == (
        1.5,
        9,
    )
    assert writer.scalars["rollout_by_game/pathfinder-0001/avg_arc_score"] == (
        25.0,
        9,
    )
    assert (
        "rollout_by_game/pathfinder-0001/avg_levels_completed_pct" not in writer.scalars
    )
    assert "rollout_by_game/pathfinder-0001/max_levels_completed" not in writer.scalars
    assert "rollout_by_game/pathfinder-0001/retained_rollouts" not in writer.scalars
    assert writer.scalars["rollout_by_game/maze_odd_id/avg_return"] == (4.0, 9)
    assert metrics["pathfinder-0001"]["avg_return"] == 2.0
    assert metrics["pathfinder-0001"]["avg_arc_score"] == 25.0
    assert "retained_rollouts" not in metrics["pathfinder-0001"]


def test_rollout_return_history_averages_games_equally_with_missing_games_zero() -> (
    None
):
    history = _new_game_return_history(
        ["easy_taps-0001", "pathfinder-0001", "memory-0001"],
        history_size=2,
    )

    _update_game_return_history(
        history,
        history_size=2,
        rollouts=[
            RolloutTrajectory(
                trajectory_id="easy-a",
                task_id="easy_taps-0001",
                steps=[],
                trajectory_reward=10.0,
                metadata={},
            ),
            RolloutTrajectory(
                trajectory_id="easy-b",
                task_id="easy_taps-0001",
                steps=[],
                trajectory_reward=30.0,
                metadata={},
            ),
            RolloutTrajectory(
                trajectory_id="pathfinder",
                task_id="pathfinder-0001",
                steps=[],
                trajectory_reward=6.0,
                metadata={},
            ),
        ],
    )

    avg_return, observed_games, configured_games = _mean_game_return_history(
        history,
        ["easy_taps-0001", "pathfinder-0001", "memory-0001"],
    )

    assert avg_return == pytest.approx(((10.0 + 30.0) / 2 + 6.0 + 0.0) / 3)
    assert observed_games == 2
    assert configured_games == 3


def test_rollout_arc_score_history_uses_observed_values_not_zero_padding() -> None:
    history = _new_game_return_history(["pathfinder-0001"], history_size=64)

    _update_game_arc_score_history(
        history,
        history_size=64,
        rollouts=[
            RolloutTrajectory(
                trajectory_id="pathfinder-a",
                task_id="pathfinder-0001",
                steps=[],
                trajectory_reward=0.0,
                metadata={"final_arc_score": 25.0},
            ),
            RolloutTrajectory(
                trajectory_id="pathfinder-b",
                task_id="pathfinder-0001",
                steps=[],
                trajectory_reward=0.0,
                metadata={"final_arc_score": 75.0},
            ),
        ],
    )

    avg_arc_score, observed_games, configured_games = _mean_game_return_history(
        history,
        ["pathfinder-0001"],
    )

    assert avg_arc_score == pytest.approx(50.0)
    assert observed_games == 1
    assert configured_games == 1


@pytest.mark.parametrize(
    ("scores_by_game", "expected", "observed"),
    [
        ({}, 0.0, 0),
        ({"game-a": [60.0]}, 60.0, 1),
        ({"game-a": [0.0]}, 0.0, 1),
        ({"game-a": [25.0, 75.0], "game-b": [20.0]}, 35.0, 2),
    ],
)
def test_arc_score_history_excludes_unsampled_games(
    scores_by_game, expected, observed
) -> None:
    game_ids = ["game-a", "game-b", "game-c"]
    history = _new_game_return_history(game_ids, history_size=8)
    for game_id, scores in scores_by_game.items():
        history[game_id].extend(scores)
    # Scores for games outside the configured split must not contribute.
    history["other-game"] = deque([100.0])

    average, observed_games, configured_games = _mean_game_return_history(
        history, game_ids, include_unobserved=False
    )

    assert average == pytest.approx(expected)
    assert observed_games == observed
    assert configured_games == 3


def test_training_tensorboard_scalars_use_transition_reward_names() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.scalars: dict[str, tuple[float, int]] = {}
            self.names: list[str] = []

        def add_scalar(self, name: str, value: float, iteration_index: int) -> None:
            self.scalars[name] = (float(value), iteration_index)
            self.names.append(name)

    writer = RecordingWriter()
    train_result = SimpleNamespace(
        sample_count=10,
        raw_sample_count=20,
        eligible_sample_count=16,
        filtered_zero_signal_sample_count=4,
        subsampled_sample_count=6,
        sample_budget_fraction=0.75,
        sample_budget_step_multiple=4,
        final_loss=0.25,
        mean_entropy=1.25,
        entropy_loss=-0.00125,
        entropy_coef_used=0.001,
        zero_signal_denominator_scale=1.25,
        probe_entropy=0.9,
        entropy_to_policy_grad_ratio=0.05,
        final_learning_rate=1e-6,
        optimizer_steps=3,
        mean_ratio=1.01,
        mean_clip_fraction=0.02,
        mean_grad_norm=0.3,
        mean_return_to_go=12.5,
        return_to_go_positive_fraction_kept=0.7,
        advantage_positive_fraction_kept=0.6,
        transition_reward_positive_fraction_raw=0.15,
        transition_reward_sum_raw=33.0,
        baseline_loss=1.5,
        baseline_mean_value=4.0,
        baseline_mean_target=5.0,
        baseline_explained_variance=0.25,
        baseline_post_update_mean_value=4.5,
        baseline_post_update_mean_absolute_error=1.25,
        baseline_post_update_explained_variance=0.5,
        baseline_mean_raw_advantage=1.0,
        baseline_std_raw_advantage=2.0,
        critic_metrics={"per_game/ar25/pre_update/bias": -4.5},
        avg_prompt_tokens_per_sample=1024.0,
        total_env_steps=123,
        avg_actions_per_completed_level=7.5,
        sample_build_seconds=0.5,
        optimization_seconds=100.0,
        optimizer_cycle_seconds=10.0,
        processed_tokens=300,
        forward_time_s=2.5,
        backward_time_s=5.0,
        policy_export_seconds=0.25,
        total_train_window_seconds=11.0,
        cuda_optimization_peak_delta_gb=40.0,
        cuda_optimization_persistent_delta_gb=1.0,
        extra_cycles=2,
        extra_optimizer_steps=2,
        extra_phase_seconds=1.5,
        rollout_logprob_parity_token_count=512,
        rollout_logprob_parity_sampler_policy_iteration=7,
        rollout_logprob_parity_trainer_policy_iteration=8,
        rollout_logprob_parity_collection_lag=1,
        rollout_logprob_parity_checked_policy_lag=0,
        rollout_logprob_parity_seconds=0.75,
    )

    _write_training_tensorboard_scalars(
        writer,
        train_result=train_result,
        iteration_index=9,
    )

    assert len(writer.names) == len(set(writer.names))
    assert writer.scalars["train/z_timing/optimization_seconds"] == (100.0, 9)
    assert writer.scalars["train/z_timing/optimizer_cycle_seconds"] == (10.0, 9)
    assert writer.scalars["train/tokens_per_s"] == (30.0, 9)
    assert writer.scalars["train/forward_time_s"] == (2.5, 9)
    assert writer.scalars["train/backward_time_s"] == (5.0, 9)
    assert writer.scalars["train/forward_backward_time_s"] == (7.5, 9)
    train_result.optimizer_cycle_seconds = 0.0
    _write_training_tensorboard_scalars(
        writer, train_result=train_result, iteration_index=9
    )
    assert writer.scalars["train/z_timing/optimization_seconds"] == (100.0, 9)
    assert writer.scalars["train/z_timing/optimizer_cycle_seconds"] == (0.0, 9)
    assert writer.scalars["train/tokens_per_s"] == (0.0, 9)
    assert writer.scalars["train/sample_count"] == (10.0, 9)
    assert writer.scalars["train/critic/per_game/ar25/pre_update/bias"] == (-4.5, 9)
    assert "train/level_baseline/per_game/ar25/pre_update/bias" not in writer.scalars
    assert writer.scalars["train/logprob_parity/token_count"] == (512.0, 9)
    assert writer.scalars["train/logprob_parity/sampler_policy_iteration"] == (7.0, 9)
    assert writer.scalars["train/logprob_parity/trainer_policy_iteration"] == (8.0, 9)
    assert writer.scalars["train/logprob_parity/collection_lag"] == (1.0, 9)
    assert writer.scalars["train/logprob_parity/checked_policy_lag"] == (0.0, 9)
    assert writer.scalars["train/logprob_parity/seconds"] == (0.75, 9)
    train_result.rollout_logprob_parity_sampler_policy_iteration = None
    writer.scalars.clear()
    _write_training_tensorboard_scalars(
        writer, train_result=train_result, iteration_index=9
    )
    assert "train/logprob_parity/sampler_policy_iteration" not in writer.scalars
    assert writer.scalars["train/samples/filtered_zero_signal"] == (4.0, 9)
    assert writer.scalars["train/sample_budget_fraction"] == (0.75, 9)
    assert writer.scalars["train/final_loss"] == (0.25, 9)
    assert writer.scalars["train/mean_entropy"] == (1.25, 9)
    assert writer.scalars["train/entropy_loss"] == (-0.00125, 9)
    assert writer.scalars["train/zero_signal_denominator_scale"] == (1.25, 9)
    assert writer.scalars["train/probe_entropy"] == (0.9, 9)
    assert writer.scalars["train/entropy_to_policy_grad_ratio"] == (0.05, 9)
    assert writer.scalars["train/baseline_loss"] == (1.5, 9)
    assert writer.scalars["train/baseline_normalized_mse"] == (1.5, 9)
    assert writer.scalars["train/baseline_mean_value"] == (4.0, 9)
    assert writer.scalars["train/baseline_mean_target"] == (5.0, 9)
    assert writer.scalars["train/baseline_explained_variance"] == (0.25, 9)
    assert writer.scalars["train/baseline_post_update_mean_value"] == (4.5, 9)
    assert writer.scalars["train/baseline_post_update_mean_absolute_error"] == (
        1.25,
        9,
    )
    assert writer.scalars["train/baseline_post_update_explained_variance"] == (
        0.5,
        9,
    )
    assert writer.scalars["train/z_timing/train_window_seconds"] == (11.0, 9)
    assert "rollout/policy_iteration" not in writer.scalars
    assert "trainer/samples/filtered_zero_signal" not in writer.scalars
    assert "z_memory/trainer_model_allocated_gb" not in writer.scalars
    assert "z_memory/trainer_optimization_peak_delta_gb" not in writer.scalars
    assert "train/reward_positive_fraction" not in writer.scalars
    assert "train/reward_nonzero_fraction" not in writer.scalars
    assert "train/positive_reward_sample_count_kept" not in writer.scalars
    assert "train/immediate_reward_positive_fraction_kept" not in writer.scalars
    assert "train/transition_reward_positive_fraction_raw" not in writer.scalars
    assert "train/return_to_go_positive_fraction_kept" not in writer.scalars


def test_training_config_defaults_include_conservative_rl_settings() -> None:
    config = TrainingConfig.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-9B",
            "output_dir": "runs",
            "lora": {"target_modules": ["q_proj"]},
        }
    )

    assert config.learning_rate == 3e-6
    assert config.clip_ratio_low == 0.2
    # Absent/null means the upper clip is disabled (issue #28): symmetric
    # clipping's clip-high term is the entropy-draining side.
    assert config.clip_ratio_high is None
    assert config.format_break_penalty == pytest.approx(0.0)
    assert config.tool_error_penalty == pytest.approx(0.0)
    assert config.loss_aggregation == "token_mean"
    assert config.reference_kl_coef == pytest.approx(0.0)
    assert config.stop_on_collapse is False
    assert config.warmup_steps == 0
    assert config.cleanup_rollout_artifacts is False
    assert config.debug_rollout_examples_per_window == 0
    assert config.sample_budget.mode == "auto"
    assert config.sample_budget.initial_fraction == pytest.approx(1.0)
    assert config.sample_budget.target_train_to_collection_ratio == pytest.approx(0.95)
    assert config.state_critic_initial_value == pytest.approx(0.0)
    assert config.state_critic_learning_rate == pytest.approx(0.001)
    assert config.state_critic_target_scale == pytest.approx(100.0)
    assert config.gae_gamma == pytest.approx(1.0)
    assert config.gae_lambda == pytest.approx(1.0)
    assert config.advantage_clip == pytest.approx(0.0)


def test_training_config_accepts_state_critic_settings() -> None:
    config = TrainingConfig.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-9B",
            "value_baseline": "hidden_critic",
            "output_dir": "runs",
            "state_critic_learning_rate": 0.002,
            "state_critic_epochs_per_window": 2,
            "state_critic_initial_value": 2.5,
            "state_critic_target_scale": 50.0,
            "gae_gamma": 0.9,
            "gae_lambda": 0.8,
            "advantage_clip": 3.5,
            "lora": {"target_modules": ["q_proj"]},
        }
    )

    assert config.state_critic_learning_rate == pytest.approx(0.002)
    assert config.state_critic_epochs_per_window == 2
    assert config.state_critic_initial_value == pytest.approx(2.5)
    assert config.state_critic_target_scale == pytest.approx(50.0)
    assert config.gae_gamma == pytest.approx(0.9)
    assert config.gae_lambda == pytest.approx(0.8)
    assert config.advantage_clip == pytest.approx(3.5)


def test_training_config_can_disable_state_critic_value_prior() -> None:
    config = TrainingConfig.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-9B",
            "value_baseline": "hidden_critic",
            "output_dir": "runs",
            "state_critic_initial_value": None,
            "lora": {"target_modules": ["q_proj"]},
        }
    )

    assert config.state_critic_initial_value is None


def test_sample_budget_auto_state_ignores_first_process_optimization_warmup() -> None:
    config = _experiment_config()
    config.training.sample_budget.target_train_to_collection_ratio = 0.95
    config.training.sample_budget.adjustment_rate = 0.25
    state = _initial_sample_budget_state(config, resume_state=None)
    train_result = SimpleNamespace(
        sample_count=64,
        optimizer_steps=8,
        eligible_sample_count=128,
        sample_budget_fraction=1.0,
        total_train_window_seconds=200.0,
        budget_train_seconds=200.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=200.0,
        policy_lag_windows=0,
        is_first_process_optimization_pass=True,
    )

    assert updated["last_observed_train_to_collection_ratio"] is None
    assert updated["next_fraction"] == pytest.approx(1.0)
    assert updated["last_update_reason"] == "first_process_optimization_warmup"


def test_sample_budget_ignores_first_post_resume_optimization_warmup() -> None:
    config = _experiment_config()
    config.training.sample_budget.target_train_to_collection_ratio = 0.95
    config.training.sample_budget.adjustment_rate = 0.25
    state = _initial_sample_budget_state(
        config,
        resume_state={
            "sample_budget_state": {
                "next_fraction": 0.7,
            }
        },
    )
    train_result = SimpleNamespace(
        sample_count=64,
        optimizer_steps=8,
        eligible_sample_count=128,
        sample_budget_fraction=0.7,
        total_train_window_seconds=200.0,
        budget_train_seconds=200.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=200.0,
        policy_lag_windows=1,
        is_first_process_optimization_pass=True,
    )

    assert updated["last_observed_train_to_collection_ratio"] is None
    assert updated["next_fraction"] == pytest.approx(0.7)
    assert updated["last_update_reason"] == "first_process_optimization_warmup"


def test_sample_budget_auto_state_smooths_toward_target_ratio_from_second_iteration() -> (
    None
):
    config = _experiment_config()
    config.training.sample_budget.target_train_to_collection_ratio = 0.95
    config.training.sample_budget.adjustment_rate = 0.25
    state = _initial_sample_budget_state(config, resume_state=None)
    train_result = SimpleNamespace(
        sample_count=64,
        optimizer_steps=8,
        eligible_sample_count=128,
        sample_budget_fraction=1.0,
        total_train_window_seconds=200.0,
        budget_train_seconds=200.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=200.0,
        policy_lag_windows=1,
        is_first_process_optimization_pass=False,
    )

    assert updated["last_observed_train_to_collection_ratio"] == pytest.approx(2.0)
    assert updated["next_fraction"] == pytest.approx(0.86875)
    assert updated["last_update_reason"] == "timing_update"


def test_sample_budget_ignores_policy_lag_and_raises_when_trainer_is_fast() -> None:
    # No policy-lag cap: lag is structural in the one-wave async pipeline
    # (a 30-min episode spans more than one iteration), so it must not drag
    # the fraction down. And pure timing feedback must RAISE the fraction
    # when the trainer finishes before the wave fills (ratio < target).
    config = _experiment_config()
    config.training.sample_budget.target_train_to_collection_ratio = 1.0
    config.training.sample_budget.adjustment_rate = 0.25
    config.training.sample_budget.min_fraction = 0.0
    state = _initial_sample_budget_state(config, resume_state=None)
    state["current_fraction"] = 0.5
    train_result = SimpleNamespace(
        sample_count=64,
        optimizer_steps=8,
        eligible_sample_count=128,
        sample_budget_fraction=0.5,
        total_train_window_seconds=50.0,
        budget_train_seconds=50.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=50.0,
        policy_lag_windows=5,  # must be reported, never enforced
        is_first_process_optimization_pass=False,
    )

    # ratio 0.5 -> raw 1.0 -> smoothed 0.5 + 0.25*(1.0-0.5) = 0.625: the
    # old lag cap (current/5 = 0.1) would have forced this DOWN instead.
    assert updated["next_fraction"] == pytest.approx(0.625)
    assert updated["last_policy_lag_windows"] == 5
    assert updated["last_update_reason"] == "timing_update"


def test_sample_budget_auto_state_ignores_zero_sample_windows() -> None:
    config = _experiment_config()
    state = _initial_sample_budget_state(config, resume_state=None)
    train_result = SimpleNamespace(
        sample_count=0,
        optimizer_steps=0,
        eligible_sample_count=0,
        sample_budget_fraction=1.0,
        total_train_window_seconds=10.0,
        budget_train_seconds=10.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=10.0,
        policy_lag_windows=1,
        is_first_process_optimization_pass=False,
    )

    assert updated["next_fraction"] == pytest.approx(1.0)
    assert updated["last_update_reason"] == "no_optimizer_timing_signal"


def test_sample_budget_auto_state_ignores_parity_skipped_windows() -> None:
    # A parity-rejected window returns the REAL shape sample_count>0 with
    # optimizer_steps=0 and a near-instant train window. Feeding that into
    # the timing loop would read as spare trainer capacity and raise the
    # coverage fraction on a guardrail skip.
    config = _experiment_config()
    config.training.sample_budget.target_train_to_collection_ratio = 0.95
    state = _initial_sample_budget_state(config, resume_state=None)
    state["current_fraction"] = 0.7
    train_result = SimpleNamespace(
        sample_count=64,
        optimizer_steps=0,
        eligible_sample_count=128,
        sample_budget_fraction=0.7,
        total_train_window_seconds=5.0,
        budget_train_seconds=5.0,
    )

    updated = _update_sample_budget_state(
        config=config,
        state=state,
        train_result=train_result,
        rollout_iteration_seconds=100.0,
        trainer_iteration_seconds=5.0,
        policy_lag_windows=1,
        is_first_process_optimization_pass=False,
    )

    assert updated["next_fraction"] == pytest.approx(0.7)
    assert updated["last_update_reason"] == "no_optimizer_timing_signal"


def test_experiment_config_rejects_multi_window_rollout_buffer(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    _write_experiment_config(
        config_path,
        parallel_rollouts=2,
        top_level={"rollout_buffer_size": 2},
    )

    with pytest.raises(ValueError, match="rollout_buffer_size is fixed at 1"):
        ExperimentConfig.load(config_path)


def test_rollout_status_counts_tracks_context_exhausted() -> None:
    def rollout(reason: str) -> RolloutTrajectory:
        return RolloutTrajectory(
            trajectory_id=f"r-{reason}",
            task_id="pathfinder-0001",
            steps=[],
            termination_reason=reason,
        )

    counts = _rollout_status_counts(
        [
            rollout("COMPLETED_RUN"),
            rollout("TIMEOUT"),
            rollout("CONTEXT_EXHAUSTED"),
            rollout("CONTEXT_EXHAUSTED"),
            rollout("ERROR"),
            rollout("CRASHED"),
            rollout("crashed"),
            rollout("GAME_OVER"),
        ]
    )

    assert counts == {
        "wins": 1,
        "timeouts": 1,
        "context_exhausted": 2,
        "crashed": 2,
        "errors": 1,
        "other": 1,
    }


def test_rollout_status_scalars_report_crashed_beside_context_exhausted() -> None:
    def rollout(reason: str) -> RolloutTrajectory:
        return RolloutTrajectory(
            trajectory_id=f"r-{reason}", task_id="puzzle", steps=[], termination_reason=reason
        )

    scalars = _rollout_status_scalars(
        [rollout("COMPLETED_RUN"), rollout("CRASHED"), rollout("CONTEXT_EXHAUSTED"), rollout("TIMEOUT")]
    )
    assert scalars == {
        "rollout/context_exhausted_count": 1.0,
        "rollout/context_exhausted_fraction": 0.25,
        "rollout/crashed_count": 1.0,
        "rollout/crashed_fraction": 0.25,
    }
    assert _rollout_status_scalars([])["rollout/crashed_fraction"] == 0.0
    # The iteration record and the window carry the count too.
    source = Path("training/cli.py").read_text(encoding="utf-8")
    assert '"crashed_rollouts": window.crashed_rollouts' in source
    assert "_rollout_status_scalars(window.rollouts)" in source


def _window_rollout(index: int, status: str, note: str | None = None) -> RolloutTrajectory:
    """A collected rollout as the trajectory builders stamp its status."""
    return RolloutTrajectory(
        trajectory_id=f"run{index}-slot{index}",
        task_id="puzzle",
        steps=[],
        trajectory_reward=0.5 if status == "CRASHED" else 1.0,
        termination_reason=status,
        metadata={
            "status": status,
            "run_id": f"run{index}",
            "run_total_actions": 1,
            "crash_note": note,
            "crash_note_tail": note,
        },
    )


def test_a_crashed_game_is_counted_logged_and_does_not_stop_the_window(
    tmp_path, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="training.inference")
    rollouts = [
        _window_rollout(0, "COMPLETED_RUN"),
        _window_rollout(
            1,
            "CRASHED",
            "RuntimeError: kernel housekeeping failed: TypeError: 'NoneType' "
            "object is not subscriptable",
        ),
        _window_rollout(2, "TIMEOUT"),
    ]
    outcomes = _platypus_window_outcomes(
        "w1", rollouts, log_path=tmp_path / "worker.log"
    )
    assert outcomes == {"zero_action": 0, "errors": 0, "crashed": 1}
    messages = [record.getMessage() for record in caplog.records]
    assert (
        "platypus window w1 complete: rollouts=3 zero_action=0 errors=0 crashed=1"
        in messages
    )
    (warning,) = [m for m in messages if "crashed and was scored" in m]
    assert "run1" in warning and "kernel housekeeping failed" in warning


@pytest.mark.parametrize(
    "statuses", [["CRASHED", "CRASHED"], ["CRASHED", "ERROR"], ["ERROR"]]
)
def test_a_window_where_every_run_crashed_is_still_refused(tmp_path, statuses) -> None:
    (tmp_path / "worker.log").write_text("agent error in every game\n")
    rollouts = [
        _window_rollout(index, status) for index, status in enumerate(statuses)
    ]
    with pytest.raises(RuntimeError, match="every run crashed") as raised:
        _platypus_window_outcomes("w2", rollouts, log_path=tmp_path / "worker.log")
    assert "agent error in every game" in str(raised.value)


def test_window_available_peeks_queue_and_error() -> None:
    import queue as queue_module

    worker = ContinuousRolloutWorker.__new__(ContinuousRolloutWorker)
    worker._buffer = queue_module.Queue()
    worker._error = None
    assert not worker.window_available()

    worker._buffer.put(object())
    assert worker.window_available()

    failed = ContinuousRolloutWorker.__new__(ContinuousRolloutWorker)
    failed._buffer = queue_module.Queue()
    failed._error = RuntimeError("collection failed")
    assert failed.window_available()


def test_synchronous_rollout_window_rejects_lag_beyond_one() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="trajectory-0",
        task_id="game-0",
        steps=[],
        metadata={"rollout_policy_iteration": 3},
    )
    window = CollectedWindow(
        window_id="window-0",
        window_dir=Path("window-0"),
        rollouts=[rollout],
        policy_iteration=3,
    )

    _validate_synchronous_rollout_window(window, policy_lag_windows=1)
    with pytest.raises(
        SynchronousRolloutContractError,
        match="Training is stopping.*one-window overlap contract",
    ):
        _validate_synchronous_rollout_window(window, policy_lag_windows=2)


def test_synchronous_rollout_window_rejects_mixed_policy_stamps() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="trajectory-mixed",
        task_id="game-0",
        steps=[],
        metadata={"rollout_policy_iteration": 2},
    )
    window = CollectedWindow(
        window_id="window-mixed",
        window_dir=Path("window-mixed"),
        rollouts=[rollout],
        policy_iteration=3,
    )

    with pytest.raises(
        SynchronousRolloutContractError,
        match="mixed policy iterations",
    ):
        _validate_synchronous_rollout_window(window, policy_lag_windows=1)


def test_worker_collect_window_calls_bind_to_the_collector_signature() -> None:
    """Every kwarg the rollout worker passes to collect_window must exist on
    RolloutCollector.collect_window — a removed parameter must not survive
    at a call site (the stop_event removal crashed r22/r23 at window 1)."""
    import ast
    import inspect
    from pathlib import Path

    import training.inference as inference_module

    accepted = set(inspect.signature(RolloutCollector.collect_window).parameters)
    source = Path(inference_module.__file__).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect_window"
    ]
    assert calls, "expected collect_window call sites in training/inference.py"
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords if keyword.arg}
        unknown = passed - accepted
        assert not unknown, (
            f"collect_window call at line {call.lineno} passes {unknown}"
        )


def _load_worker_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "platypus_rollout_worker.py"
    )
    spec = importlib.util.spec_from_file_location("platypus_rollout_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_serve_consumes_specs_in_order_and_reports_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _load_worker_module()
    seen: list[str] = []

    async def fake_run(spec):
        seen.append(spec["window_id"])
        if spec["window_id"] == "w2":
            raise RuntimeError("boom")

    monkeypatch.setattr(worker, "_run", fake_run)
    monkeypatch.setattr(worker, "_post_results", lambda spec: None)
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    for window_id in ("w2", "w1"):
        (serve_dir / f"{window_id}.spec.json").write_text(
            json.dumps({"window_id": window_id, "results_url": "http://x"}),
            encoding="utf-8",
        )
    (serve_dir / "STOP").touch()

    assert worker.serve(serve_dir, poll_seconds=0.01) == 0

    # Name order, each spec consumed exactly once, statuses reported.
    assert seen == ["w1", "w2"]
    assert not list(serve_dir.glob("*.spec.json"))
    assert json.loads((serve_dir / "w1.done.json").read_text())["status"] == "ok"
    failed = json.loads((serve_dir / "w2.done.json").read_text())
    assert failed["status"] == "error" and "boom" in failed["error"]


def test_wait_for_window_done_outcomes(tmp_path: Path) -> None:
    import time as _time

    from training.inference import _wait_for_window_done

    class _Proc:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    done = tmp_path / "w.done.json"
    # Done file wins even when the process has exited.
    done.write_text("{}")
    assert (
        _wait_for_window_done(
            _Proc(0), done, deadline=_time.monotonic() + 5, poll_seconds=0.01
        )
        == "done"
    )
    done.unlink()
    assert (
        _wait_for_window_done(
            _Proc(1), done, deadline=_time.monotonic() + 5, poll_seconds=0.01
        )
        == "exited"
    )
    assert (
        _wait_for_window_done(
            _Proc(None), done, deadline=_time.monotonic() - 1, poll_seconds=0.01
        )
        == "watchdog"
    )


_LOADED_POLICY = {
    "name": "policy",
    "id": 3,
    "path": "/runs/adapter_iter_000003/runtime_vllm",
    "server_instance_id": "engine-a",
}


def _warm_collector(tmp_path, monkeypatch, *, status_code=200, adapter=None):
    """A collector without a server log whose warm-up answers with ``adapter``."""
    captured: dict[str, object] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return SimpleNamespace(
            status_code=status_code,
            text="no such model",
            json=lambda: {"outputs": [{"token_ids": [5]}], "adapter": adapter},
        )

    monkeypatch.setattr("training.inference.requests.post", fake_post)
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    assert collector.server_log_path is None
    return collector, captured


def test_warm_runtime_adapter_uses_trainer_control_plane(monkeypatch, tmp_path) -> None:
    collector, captured = _warm_collector(
        tmp_path, monkeypatch, adapter=dict(_LOADED_POLICY)
    )

    seconds = collector.warm_runtime_adapter(
        adapter_name="policy", expected_identity=dict(_LOADED_POLICY)
    )

    assert seconds >= 0.0
    assert captured["url"] == "http://127.0.0.1:1234/adapters/warm"
    assert captured["json"] == {"name": "policy"}


def test_warm_runtime_adapter_raises_when_alias_does_not_serve(
    monkeypatch, tmp_path
) -> None:
    collector, _ = _warm_collector(tmp_path, monkeypatch, status_code=404)

    with pytest.raises(RuntimeError, match="did not serve a warm-up"):
        collector.warm_runtime_adapter(
            adapter_name="policy", expected_identity=dict(_LOADED_POLICY)
        )


@pytest.mark.parametrize(
    ("served", "match"),
    [
        (_LOADED_POLICY | {"id": 2}, "served its warm-up draw with adapter"),
        (_LOADED_POLICY | {"path": "/runs/other"}, "served its warm-up draw"),
        (_LOADED_POLICY | {"server_instance_id": "engine-b"}, "served its warm-up"),
        (None, "invalid adapter identity"),
    ],
)
def test_warm_up_identity_must_equal_the_loaded_adapter(
    monkeypatch, tmp_path, served, match
) -> None:
    """The load response and the warm-up draw must name one adapter."""
    collector, _ = _warm_collector(tmp_path, monkeypatch, adapter=served)

    with pytest.raises(RuntimeError, match=match):
        collector.warm_runtime_adapter(
            adapter_name="policy", expected_identity=dict(_LOADED_POLICY)
        )


def test_platypus_window_seed_ignores_window_id_for_scheduled_seeds() -> None:
    from training.inference import _platypus_window_seed

    # Official-eval / explicit seeds replay the same layouts every cycle:
    # numeric seeds pass through, text seeds hash stably, and neither is
    # perturbed by the window id.
    assert _platypus_window_seed("20260826T090000Z", "12345") == 12345
    assert _platypus_window_seed("20260826T100000Z", "12345") == 12345
    assert _platypus_window_seed("w1", "layout-a") == _platypus_window_seed(
        "w2", "layout-a"
    )
    # Unseeded training windows draw fresh layouts per window.
    assert _platypus_window_seed("w1", None) != _platypus_window_seed("w2", None)


def test_platypus_collector_id_separates_servers() -> None:
    from training.inference import RolloutCollector

    def collector_with_base_url(base_url: str) -> RolloutCollector:
        collector = object.__new__(RolloutCollector)
        config = _experiment_config()
        config.inference.base_url = base_url
        collector.config = config
        return collector

    first = collector_with_base_url("http://127.0.0.1:50702")
    second = collector_with_base_url("http://127.0.0.1:50703")
    assert first._collector_id() == "port-50702"
    assert second._collector_id() == "port-50703"
    assert first._collector_id() != second._collector_id()


def test_derive_server_base_url_follows_the_port_stride_contract() -> None:
    from training.inference import SERVER_PORT_STRIDE, derive_server_base_url

    base = "http://127.0.0.1:50000"
    # Index 0 keeps the configured URL byte-identically; higher indices are
    # exactly what the job script computes when launching each server.
    assert derive_server_base_url(base, 0) == base
    assert derive_server_base_url(base, 1) == "http://127.0.0.1:50100"
    assert derive_server_base_url(base, 3) == "http://127.0.0.1:50300"
    assert SERVER_PORT_STRIDE == 100
    with pytest.raises(ValueError, match="no explicit port"):
        derive_server_base_url("http://localhost", 1)
    with pytest.raises(ValueError, match="server_index"):
        derive_server_base_url(base, -1)


def test_per_server_configs_use_pre_launched_ports_not_free_ports() -> None:
    from training.inference import _per_server_configs

    config = _experiment_config()
    config.inference.base_url = "http://127.0.0.1:50000"
    config.inference.server_gpus = ["1", "2", "3"]
    assert [
        server_config.inference.base_url
        for server_config in _per_server_configs(config)
    ] == [
        "http://127.0.0.1:50000",
        "http://127.0.0.1:50100",
        "http://127.0.0.1:50200",
    ]

    config.inference.server_gpus = []
    assert [
        server_config.inference.base_url
        for server_config in _per_server_configs(config)
    ] == ["http://127.0.0.1:50000"]


def test_split_slots_into_balanced_chunks_deals_complete_passes() -> None:
    from training.game_sampling import sample_training_games
    from training.inference import split_slots_into_balanced_chunks

    # The shipped officials recipe: 25 games x 2 passes over 2 servers.
    games = [f"official-{index:02d}" for index in range(25)]
    window = sample_training_games(games, slot_count=50, seed=42, schedule_index=0)
    chunks = split_slots_into_balanced_chunks(window, 2)

    assert sorted(index for chunk in chunks for index in chunk) == list(range(50))
    for chunk in chunks:
        # Each server holds one whole copy of the game multiset.
        assert Counter(window[index] for index in chunk) == Counter(games)


def test_split_slots_into_balanced_chunks_partitions_distinct_games() -> None:
    from training.inference import split_slots_into_balanced_chunks

    window = [f"game-{index}" for index in range(5)]
    chunks = split_slots_into_balanced_chunks(window, 2)

    assert sorted(index for chunk in chunks for index in chunk) == list(range(5))
    for chunk in chunks:
        counts = Counter(window[index] for index in chunk)
        assert set(counts.values()) <= {1}


def test_split_slots_into_balanced_chunks_supports_uneven_repeat_counts() -> None:
    from training.inference import split_slots_into_balanced_chunks

    window = ["a", "a", "a", "b", "b", "b"]
    chunks = split_slots_into_balanced_chunks(window, 2)
    assert sorted(index for chunk in chunks for index in chunk) == list(range(6))
    assert [len(chunk) for chunk in chunks] == [3, 3]
    assert all(
        Counter(window[index] for index in chunk)["a"] in (1, 2) for chunk in chunks
    )
    # One game repeated once: its copies go to different servers.
    chunks = split_slots_into_balanced_chunks(["a", "a", "b"], 2)
    assert sorted(len(chunk) for chunk in chunks) == [1, 2]
    for chunk in chunks:
        child = [["a", "a", "b"][index] for index in chunk]
        assert len(child) == len(set(child))


def test_continuous_worker_accepts_uneven_repeats_across_servers(
    tmp_path,
) -> None:
    config = _experiment_config()
    config.parallel_rollouts = 6  # 3 copies of each of 2 games
    config.inference.games = ["game-a", "game-b"]
    config.inference.game = "game-a"
    config.inference.base_url = "http://127.0.0.1:50000"
    config.inference.server_gpus = ["1", "2"]

    ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )

    config.parallel_rollouts = 4  # 2 copies each -> one pass per server
    ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )


def test_platypus_slot_seeds_carry_ordered_per_slot_seeds() -> None:
    from training.inference import _platypus_slot_seeds, _RolloutSlot

    slots = [
        _RolloutSlot(slot_index=0, game_id="game-a", rollout_seed="11"),
        _RolloutSlot(slot_index=1, game_id="game-b", rollout_seed="22"),
        _RolloutSlot(slot_index=2, game_id="game-a", rollout_seed="11"),
    ]
    assert _platypus_slot_seeds(slots) == [11, 22, 11]

    text_slots = [_RolloutSlot(slot_index=0, game_id="game-a", rollout_seed="layout-a")]
    assert _platypus_slot_seeds(text_slots) == _platypus_slot_seeds(text_slots)

    unseeded = [_RolloutSlot(slot_index=0, game_id="game-a", rollout_seed=None)]
    assert _platypus_slot_seeds(unseeded) is None


def test_platypus_slot_seeds_reject_scheduling_bugs() -> None:
    from training.inference import _platypus_slot_seeds, _RolloutSlot

    copies = [
        _RolloutSlot(slot_index=0, game_id="game-a", rollout_seed="11"),
        _RolloutSlot(slot_index=1, game_id="game-a", rollout_seed="12"),
    ]
    assert _platypus_slot_seeds(copies) == [11, 12]

    mixed = [
        _RolloutSlot(slot_index=0, game_id="game-a", rollout_seed="11"),
        _RolloutSlot(slot_index=1, game_id="game-b", rollout_seed=None),
    ]
    with pytest.raises(RuntimeError, match="unseeded"):
        _platypus_slot_seeds(mixed)


def test_worker_pins_distinct_seeds_by_external_slot_id() -> None:
    from training.rollout_slots import _rng_seed_per_game

    assert _rng_seed_per_game(
        ["game-a#s0", "game-b#s1", "game-a#s2", "game-b#s3"], [11, 22, 12, 23]
    ) == {"game-a#s0": 11, "game-b#s1": 22, "game-a#s2": 12, "game-b#s3": 23}
    with pytest.raises(ValueError, match="conflicting model RNG seeds"):
        _rng_seed_per_game(["game-a", "game-a"], [11, 12])
    with pytest.raises(ValueError, match="one seed per slot"):
        _rng_seed_per_game(["game-a"], [11, 12])


def test_prefix_cache_parser_ignores_created_timestamp_series() -> None:
    from training.inference import parse_prefix_cache_counters

    # Real payload shape from vLLM 0.25.1: the *_created series are unix
    # timestamps (~1.788e9) that used to be summed into both sides of the
    # ratio, pinning the reported hit rate at ~0.999 regardless of actual
    # cache behavior.
    payload = "\n".join(
        [
            "# HELP vllm:prefix_cache_queries_total ...",
            'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1.0564492e+07',
            'vllm:prefix_cache_queries_created{engine="0",model_name="m"} 1.7878354681353874e+09',
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 9.593808e+06',
            'vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.7878354681354046e+09',
            'vllm:external_prefix_cache_queries_total{engine="0",model_name="m"} 0.0',
            'vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 0.0',
        ]
    )
    counters = parse_prefix_cache_counters(payload)
    assert counters is not None
    hits, queries = counters
    assert hits == pytest.approx(9.593808e6)
    assert queries == pytest.approx(1.0564492e7)
    # The vLLM server's own log reported 90.8% at this snapshot.
    assert hits / queries == pytest.approx(0.908, abs=0.001)


def test_prefix_cache_parser_returns_none_without_queries() -> None:
    from training.inference import parse_prefix_cache_counters

    assert parse_prefix_cache_counters("") is None
    assert (
        parse_prefix_cache_counters(
            'vllm:prefix_cache_queries_created{engine="0"} 1.7e+09'
        )
        is None
    )


def test_tb_gap_collapse_rewrites_large_gaps(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    from training.tb_gap_collapse import collapse_run_dir

    run = tmp_path / "run"
    run.mkdir()
    writer = EventFileWriter(str(run))
    # Two segments of a run: steps 0-2 at 600s cadence, a 5-hour queue
    # wait, then steps 3-4.
    for step, wall in (
        (0, 1000.0),
        (1, 1600.0),
        (2, 2200.0),
        (3, 20200.0),
        (4, 20800.0),
    ):
        event = event_pb2.Event(wall_time=wall, step=step)
        event.summary.value.append(
            summary_pb2.Summary.Value(tag="score", simple_value=float(step))
        )
        writer.add_event(event)
    writer.close()

    collapsed = collapse_run_dir(run, gap_threshold=2400.0)
    assert collapsed == 1
    times = []
    for name in sorted(p.name for p in run.iterdir()):
        if "tfevents" not in name:
            continue
        for event in EventFileLoader(str(run / name)).Load():
            if event.HasField("summary"):
                times.append(event.wall_time)
    times.sort()
    deltas = [b - a for a, b in zip(times, times[1:])]
    # The 18,000s gap became the run's median cadence (600s).
    assert max(deltas) == pytest.approx(600.0)
    assert len(times) == 5

    # Idempotent: a second pass finds nothing to collapse.
    assert collapse_run_dir(run, gap_threshold=2400.0) == 0


def test_worker_spec_carries_the_malformed_reply_policy(tmp_path: Path) -> None:
    """Both knobs ride the spec (not the environment) so one configured
    agent describes the experiment on every host."""
    import inspect

    from training import inference as inference_module

    source = inspect.getsource(
        inference_module.RolloutCollector._collect_window_platypus
    )
    assert '"malformed_retries": 0' in source
    assert '"accept_malformed_output": True' in source
    assert '"terminate_on_empty_reply"' in source
    # The worker scores with the trainer's coefficients; the gateway re-derives
    # with the same ones (SamplingPolicy).
    training = "self.config.training"
    assert f'"max_episode_penalty": {training}.max_episode_penalty' in source
    assert f"max_episode_penalty={training}.max_episode_penalty" in source
    assert "ULLM_ACCEPT_MALFORMED_OUTPUT" not in inspect.getsource(inference_module)


def test_window_with_one_repeated_game_splits_into_unique_child_windows() -> None:
    """50 slots over 49 games on two servers: every game once, one game
    twice, its copies on different servers, 25 + 25."""
    from training.game_sampling import sample_training_games
    from training.inference import split_slots_into_balanced_chunks

    games = [f"g{i:02d}" for i in range(49)]
    window = sample_training_games(
        games,
        slot_count=50,
        seed=3,
        schedule_index=11,
        mode="balanced",
        games_per_window=0,
    )
    counts = {game: window.count(game) for game in set(window)}
    assert sorted(counts.values()) == [1] * 48 + [2]
    chunks = split_slots_into_balanced_chunks(window, 2)
    assert sorted(len(chunk) for chunk in chunks) == [25, 25]
    for chunk in chunks:
        child = [window[index] for index in chunk]
        assert len(child) == len(set(child))
    # The doubled game changes from window to window.
    doubled = set()
    for index in range(30):
        w = sample_training_games(
            games,
            slot_count=50,
            seed=3,
            schedule_index=index,
            mode="balanced",
            games_per_window=0,
        )
        doubled.add(next(g for g in set(w) if w.count(g) == 2))
    assert len(doubled) > 1
    # Unique slot IDs also permit extra copies within one child window.
    chunks = split_slots_into_balanced_chunks(["a", "a", "a", "b"], 2)
    assert [len(chunk) for chunk in chunks] == [2, 2]


@pytest.mark.parametrize("top_k", [None, -1, 0, 20])
def test_worker_sampling_overrides_inherited_probability_transforms(tmp_path, top_k):
    presets = pytest.importorskip("duck.presets")
    baseline = presets.analyzer_sampling_profiles()["play"]
    benchmark = _load_worker_module()._build_benchmark(
        {
            "games": ["ar25-0c556536"],
            "max_concurrent": 1,
            "context_window_tokens": 32768,
            "compact_to_tokens": 16384,
            "compaction_trigger_tokens": 28160,
            "safety_margin_tokens": 512,
            "malformed_retries": 0,
            "accept_malformed_output": True,
            "base_url": "http://127.0.0.1:1",
            "served_model": "policy",
            "max_completion_tokens": 4096,
            "max_summary_tokens": 0,
            "summary_forbidden_token_ids": [],
            "max_served_sequence_tokens": 32768,
            "job_dir": str(tmp_path),
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": top_k},
        }
    )
    agent = benchmark.solver.agent
    profile = agent._sampling(SimpleNamespace(rng=random.Random(7)))
    assert profile.temperature == profile.top_p == 1.0
    assert profile.max_tokens == 4096 and profile.send_back_reasoning == "all"
    assert profile.extra_params == {
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_p": 0.0,
        "min_tokens": 0,
        "top_k": top_k if top_k and top_k > 0 else -1,
        "ignore_eos": False,
        "skip_special_tokens": False,
        "logprobs": 0,
    }
    # The worker pins its own profile; the harness's deployment baseline is
    # left untouched (since sweeps 12/13 it samples without a presence
    # penalty or nucleus, so there is no longer a 1.5 to override).
    assert presets.analyzer_sampling_profiles()["play"] == baseline


@pytest.mark.parametrize(
    "extra",
    [
        {"continued": True, "history_epoch": 3, "exact_history": True},
        {"continued": "false", "history_epoch": "3", "exact_history": "true"},
        {},
    ],
)
def test_worker_step_reports_preserve_continuation_evidence(
    tmp_path: Path, extra: dict
) -> None:
    import json

    record = {
        "provider_validation": {"request_args": {"request_id": "run#nonce"}},
        "output_token_ids": [7, 8],
        "action_index": 2,
        "extra": extra,
    }
    (tmp_path / "steps.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (report,) = _load_worker_module()._step_reports(tmp_path)
    assert report["nonce"] == "nonce"
    for field in ("continued", "history_epoch", "exact_history"):
        # The trainer must receive absent and malformed declarations as-is,
        # so it can reject them rather than accepting a fabricated boundary.
        assert report[field] == extra.get(field)
        assert type(report[field]) is type(extra.get(field))


def test_training_collection_uses_distinct_reproducible_slot_seeds(
    tmp_path, monkeypatch
):
    config = _experiment_config()
    config.inference.server_gpus = []
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )
    games = ["game-a"] * 4 + ["game-b"]
    windows = []
    monkeypatch.setattr(worker, "_sample_training_games", lambda **kwargs: games)
    monkeypatch.setattr(worker, "_sync_pending_policy_if_needed", lambda: None)
    monkeypatch.setattr(
        worker,
        "_collect_window_across_servers",
        lambda **kwargs: (windows.append(kwargs), SimpleNamespace(policy_iteration=-1))[
            1
        ],
    )
    worker._active_policy_path = Path("/test/initialized-adapter")
    worker._active_policy_iteration = -1
    monkeypatch.setattr(worker, "_sync_pending_policy_if_needed", lambda: None)
    worker._collect_training_window()
    worker._active_policy_path = Path("/test/initialized-adapter")
    worker._active_policy_iteration = -1
    monkeypatch.setattr(worker, "_sync_pending_policy_if_needed", lambda: None)
    worker._collect_training_window()
    worker._next_schedule_index = 0
    worker._active_policy_path = Path("/test/initialized-adapter")
    worker._active_policy_iteration = -1
    monkeypatch.setattr(worker, "_sync_pending_policy_if_needed", lambda: None)
    worker._collect_training_window()
    first, second, replay = [window["scheduled_rollout_seeds"] for window in windows]
    assert len(set(first)) == len(games)
    assert not set(first) & set(second)
    assert first == replay
    assert all(window["scheduled_games"] == games for window in windows)


def test_native_adapter_identity_pins_current_engine_instance(monkeypatch, tmp_path):
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    identity = {
        "name": "policy",
        "id": 17,
        "path": "/runtime/17",
        "server_instance_id": "engine-a",
    }
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"adapter": dict(identity)}
        )

    monkeypatch.setattr("training.inference.requests.get", get)
    assert collector.adapter_identity("policy") == identity
    assert calls[0][0] == "http://127.0.0.1:1234/metadata"
    assert calls[0][1]["params"] == {"model": "policy"}
    identity["server_instance_id"] = "engine-b"
    assert collector.adapter_identity("policy")["server_instance_id"] == "engine-b"


@pytest.mark.parametrize(
    "invalid",
    [
        {"name": "other"},
        {"id": True},
        {"id": -1},
        {"id": 0},
        {"path": "relative"},
        {"path": None},
        {"server_instance_id": ""},
        {"extra": 1},
    ],
)
def test_native_adapter_identity_rejects_incomplete_or_wrong_provenance(
    monkeypatch, tmp_path, invalid
):
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    identity = {
        "name": "policy",
        "id": 17,
        "path": "/runtime/17",
        "server_instance_id": "engine-a",
    } | invalid
    monkeypatch.setattr(
        "training.inference.requests.get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"adapter": identity}
        ),
    )
    with pytest.raises(RuntimeError, match="invalid adapter identity"):
        collector.adapter_identity("policy")


@pytest.mark.parametrize(
    "changed", [{"id": 18}, {"path": "/different"}, {"server_instance_id": "engine-b"}]
)
def test_window_rejects_adapter_changed_after_successful_load(
    monkeypatch, tmp_path, changed
):
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    _runtime_export(tmp_path)
    identity = {
        "name": "policy",
        "id": 17,
        "path": str(tmp_path),
        "server_instance_id": "engine-a",
    }
    monkeypatch.setattr(
        "training.inference.requests.post",
        lambda *a, **k: SimpleNamespace(
            status_code=200,
            json=lambda: {"adapter": dict(identity), "lora_census": _CENSUS},
        ),
    )
    collector.load_runtime_adapter(adapter_name="policy", adapter_path=tmp_path)
    monkeypatch.setattr(
        "training.inference.requests.get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"adapter": identity | changed}
        ),
    )
    with pytest.raises(RuntimeError, match="changed after"):
        collector.adapter_identity("policy")


def test_load_rejects_success_response_for_wrong_adapter_path(monkeypatch, tmp_path):
    collector = RolloutCollector(_experiment_config(), repo_root=tmp_path)
    identity = {
        "name": "policy",
        "id": 17,
        "path": "/wrong",
        "server_instance_id": "engine-a",
    }
    monkeypatch.setattr(
        "training.inference.requests.post",
        lambda *a, **k: SimpleNamespace(
            status_code=200, json=lambda: {"adapter": identity}
        ),
    )
    with pytest.raises(RuntimeError, match="different adapter path"):
        collector.load_runtime_adapter(adapter_name="policy", adapter_path=tmp_path)


def test_overlapped_collection_stops_after_the_window_limit(tmp_path, monkeypatch) -> None:
    import threading

    config = _experiment_config()
    config.training.rollout_schedule = "overlapped"
    worker = ContinuousRolloutWorker(
        config, repo_root=tmp_path, output_root=tmp_path / "rollouts"
    )
    monkeypatch.setattr(worker, "_ensure_server_ready", lambda: None)
    collected: list[str] = []

    def collect_window(**kwargs):  # noqa: ANN003
        collected.append(f"window-{len(collected)}")
        return CollectedWindow(
            window_id=collected[-1],
            window_dir=tmp_path / collected[-1],
            rollouts=[],
            policy_iteration=-1,
            schedule_index=len(collected) - 1,
        )

    monkeypatch.setattr(worker, "_collect_training_window", collect_window)
    with pytest.raises(ValueError):
        worker.limit_windows(-1)
    # Two iterations remain: exactly two windows, then the thread ends on its
    # own instead of starting a third that nothing would consume.
    worker.limit_windows(2)
    thread = threading.Thread(target=worker._run, daemon=True)
    thread.start()
    assert worker.next_window(timeout=10).window_id == "window-0"
    assert worker.next_window(timeout=10).window_id == "window-1"
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert collected == ["window-0", "window-1"]
    assert worker._error is None
