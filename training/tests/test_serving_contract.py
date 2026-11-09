"""Trainer rejects bad serving configuration even if the harness hides it."""

import json
import time
from types import SimpleNamespace

import pytest

from training.config import SpeculativeDecodingConfig
from training.inference import (
    ContinuousRolloutWorker,
    RolloutCollector,
    _wait_for_window_done,
)
from training.rollout_gateway import GatewayError


def test_attach_checks_actual_server_runtime(monkeypatch):
    collector = object.__new__(RolloutCollector)
    collector.config = SimpleNamespace(
        inference=SimpleNamespace(
            base_url="http://local", speculative_decoding=SpeculativeDecodingConfig()
        ),
        training=SimpleNamespace(score_centering_top_k=32),
    )
    monkeypatch.setattr(collector, "_server_is_ready", lambda: True)
    monkeypatch.setattr(
        collector, "_server_auth_headers", lambda: {"Authorization": "test"}
    )
    seen = []

    def reject_runtime(base_url, *, headers, score_centering_top_k, speculative):
        assert score_centering_top_k == 32
        assert speculative is None  # the config does not ask for drafting
        seen.append((base_url, headers))
        raise RuntimeError("inherited top_k=20")

    monkeypatch.setattr("training.vllm_runtime.validate_server_runtime", reject_runtime)
    with pytest.raises(RuntimeError, match="top_k=20"):
        collector.ensure_server()
    assert seen == [("http://local", {"Authorization": "test"})]


@pytest.mark.parametrize("reported_done", [False, True])
def test_contract_failure_beats_worker_success_or_watchdog(tmp_path, reported_done):
    done = tmp_path / "window.done.json"
    if reported_done:
        done.write_text('{"status": "ok"}')

    def reject_window():
        raise GatewayError("presence_penalty must be zero")

    with pytest.raises(GatewayError, match="presence_penalty"):
        _wait_for_window_done(
            SimpleNamespace(poll=lambda: None),
            done,
            deadline=time.monotonic() - 1,
            check_health=reject_window,
        )


def test_next_window_rechecks_health_before_consuming_queued_data():
    worker = object.__new__(ContinuousRolloutWorker)
    worker._error = None
    worker.run_root = None
    calls = []

    def dequeue(*, timeout):
        calls.append("dequeued")
        return object()

    worker._buffer = SimpleNamespace(get=dequeue)

    def check_health():
        if calls:
            raise GatewayError("eval sampler violated the contract")

    with pytest.raises(GatewayError, match="eval sampler"):
        worker.next_window(check_health=check_health)


def test_eval_gateway_notifies_shared_trainer_failure_root(tmp_path, monkeypatch):
    from training.run_failure import FatalTrainingError, raise_if_run_failed

    collector = object.__new__(RolloutCollector)
    collector.run_root = tmp_path / "official_eval_worker"
    collector.failure_root = tmp_path
    collector.config = SimpleNamespace(
        inference=SimpleNamespace(base_url="http://127.0.0.1:1234")
    )

    def gateway_factory(**kwargs):
        return SimpleNamespace(start=lambda: None, base_url="http://gateway", **kwargs)

    monkeypatch.setattr("training.rollout_gateway.CaptureGateway", gateway_factory)
    gateway = collector._platypus_gateway()
    gateway.on_contract_violation(GatewayError("eval top_k violation"))
    with pytest.raises(FatalTrainingError, match="eval top_k"):
        raise_if_run_failed(tmp_path)
    assert not collector.run_root.exists()


@pytest.mark.parametrize("real_violation", [False, True])
def test_multi_server_shutdown_preserves_real_failures(tmp_path, real_violation):
    import threading

    def interrupted(**kwargs):
        raise InterruptedError("deliberate shutdown")

    def violated(**kwargs):
        raise GatewayError("top_k violation")

    worker = object.__new__(ContinuousRolloutWorker)
    worker.collectors = [
        SimpleNamespace(collect_window=interrupted),
        SimpleNamespace(collect_window=violated if real_violation else interrupted),
    ]
    worker._active_model_id = "policy"
    worker._active_policy_iteration = 0
    worker._stop_event = threading.Event()
    worker._stop_event.set()
    expected = RuntimeError if real_violation else InterruptedError
    with pytest.raises(expected) as error:
        worker._collect_window_across_servers(
            output_root=tmp_path,
            schedule_index=0,
            scheduled_games=["a", "b"],
            scheduled_rollout_seeds=[42, 43],
            window_kind="train",
        )
    if real_violation:
        assert isinstance(error.value.__cause__, GatewayError)


def test_next_window_observes_failure_while_waiting():
    import queue

    worker = object.__new__(ContinuousRolloutWorker)
    worker._error = None
    worker.run_root = None
    waits = []

    def dequeue(*, timeout):
        waits.append(timeout)
        raise queue.Empty

    worker._buffer = SimpleNamespace(get=dequeue)

    def check_health():
        if waits:
            raise GatewayError("evaluation failed")

    with pytest.raises(GatewayError, match="evaluation failed"):
        worker.next_window(check_health=check_health)
    assert waits == [1.0]


@pytest.mark.parametrize(
    "concurrency, slot_count, waves", [(None, 1, 1), (16, 32, 2), (16, 120, 8)]
)
@pytest.mark.parametrize("window_kind", ["train", "eval"])
def test_worker_watchdog_is_fatal_without_retry_or_partial_training(
    tmp_path, monkeypatch, concurrency, slot_count, waves, window_kind
):
    from training.config import DEFAULT_MALFORMED_CALL_PENALTY, InferenceConfig
    from training.rollout_gateway import WindowCapture
    from training.run_failure import FatalTrainingError, raise_if_run_failed

    collector = object.__new__(RolloutCollector)
    collector.run_root = collector.failure_root = tmp_path
    collector.config = SimpleNamespace(
        inference=InferenceConfig(
            base_url="http://127.0.0.1:1234",
            model_id="model",
            game="ar25",
            games=["ar25"],
            max_runtime_minutes=15,
            max_concurrent_episodes=concurrency,
            agent_context_window=32768,
            agent_max_completion_tokens=4096,
            agent_compact_to_tokens=16384,
        ),
        training=SimpleNamespace(
            model_name="model",
            model_revision="revision",
            weight_mode="lora",
            malformed_call_penalty=DEFAULT_MALFORMED_CALL_PENALTY,
            max_episode_penalty=None,
            score_centering_top_k=32,
            seed=7,
        ),
    )
    calls = []
    gateway = SimpleNamespace(
        base_url="http://127.0.0.1:6234",
        results_url="http://127.0.0.1:6234/rl/results",
        set_policy=lambda policy: setattr(gateway, "_policy", policy),
        drain_window=lambda: calls.append("drain") or WindowCapture(),
    )
    worker = SimpleNamespace(
        serve_dir=tmp_path,
        log_path=tmp_path / "worker.log",
        process=SimpleNamespace(pid=123, poll=lambda: None),
    )
    worker.log_path.write_text("worker did not finish its window\n")
    slot = SimpleNamespace(game_id="ar25", rollout_seed=42)
    monkeypatch.setattr(collector, "_platypus_gateway", lambda: gateway)
    monkeypatch.setattr(
        collector,
        "adapter_identity",
        lambda model: {
            "name": model,
            "id": 0,
            "path": None,
            "server_instance_id": "test-native",
        },
    )
    monkeypatch.setattr(
        collector,
        "_ensure_platypus_worker",
        lambda root: calls.append("launch") or worker,
    )
    monkeypatch.setattr(
        collector, "_close_platypus_worker", lambda worker: calls.append("close")
    )
    monkeypatch.setattr(collector, "_verify_runtime_fingerprint", lambda: None)
    monkeypatch.setattr(collector, "_effective_max_runtime_minutes", lambda kind: 15)
    monkeypatch.setattr(
        collector, "_rollout_slots", lambda **kwargs: [slot] * slot_count
    )
    monkeypatch.setattr("training.inference.time.monotonic", lambda: 100.0)

    def wait(*args, **kwargs):
        assert kwargs["deadline"] == 100.0 + waves * 900.0 + 900.0
        return "watchdog"

    monkeypatch.setattr("training.inference._wait_for_window_done", wait)
    with pytest.raises(RuntimeError, match="worker watchdog"):
        collector.collect_window(
            output_root=tmp_path / "rollouts", model_id="model", window_kind=window_kind
        )
    assert gateway._policy.score_centering_top_k == (
        32 if window_kind == "train" else 0
    )
    assert calls == ["drain", "launch", "close"]
    assert len(list(tmp_path.glob("*.spec.json"))) == 1
    spec = json.loads(next(tmp_path.glob("*.spec.json")).read_text())
    assert spec["max_concurrent"] == min(concurrency or slot_count, slot_count)
    assert len(spec["games"]) == slot_count
    with pytest.raises(FatalTrainingError, match="worker watchdog"):
        raise_if_run_failed(tmp_path)


@pytest.mark.parametrize("real_violation", [False, True])
def test_deliberate_shutdown_does_not_hide_contract_violations(
    tmp_path, monkeypatch, real_violation
):
    from training.run_failure import FatalTrainingError, raise_if_run_failed

    collector = object.__new__(RolloutCollector)
    collector.run_root = collector.failure_root = tmp_path
    collector._stopping_window = True
    from training.config import InferenceConfig

    collector.config = SimpleNamespace(
        inference=InferenceConfig(
            base_url="http://local",
            model_id="model",
            game="g",
            games=["g"],
            max_runtime_minutes=15,
        )
    )
    monkeypatch.setattr(collector, "_verify_runtime_fingerprint", lambda: None)
    monkeypatch.setattr(collector, "_effective_max_runtime_minutes", lambda kind: 15)
    monkeypatch.setattr(collector, "_rollout_slots", lambda **kwargs: [])

    def stopped(**kwargs):
        if real_violation:
            raise GatewayError("presence_penalty violation during shutdown")
        raise InterruptedError("rollout collection was deliberately stopped")

    monkeypatch.setattr(collector, "_collect_window_platypus", stopped)
    if real_violation:
        with pytest.raises(GatewayError, match="presence_penalty"):
            collector.collect_window(output_root=tmp_path / "rollouts")
        with pytest.raises(FatalTrainingError, match="presence_penalty"):
            raise_if_run_failed(tmp_path)
    else:
        with pytest.raises(InterruptedError, match="deliberately stopped"):
            collector.collect_window(output_root=tmp_path / "rollouts")
        raise_if_run_failed(tmp_path)
        assert not (tmp_path / "STOP_AUTO_RELAUNCH").exists()


@pytest.mark.parametrize("window_kind", ["train", "eval"])
def test_programmatic_training_sampling_is_validated_before_dispatch(
    tmp_path, monkeypatch, window_kind
):
    from training.config import InferenceConfig
    from training.run_failure import FatalTrainingError, raise_if_run_failed

    collector = object.__new__(RolloutCollector)
    collector.run_root = collector.failure_root = tmp_path
    collector.config = SimpleNamespace(
        inference=InferenceConfig(
            base_url="http://local",
            model_id="model",
            game="g",
            games=["g"],
            max_runtime_minutes=15,
            agent_temperature=0.6,
        )
    )
    dispatched = []
    monkeypatch.setattr(collector, "_verify_runtime_fingerprint", lambda: None)
    monkeypatch.setattr(collector, "_effective_max_runtime_minutes", lambda kind: 15)
    monkeypatch.setattr(collector, "_rollout_slots", lambda **kwargs: [])
    monkeypatch.setattr(
        collector,
        "_collect_window_platypus",
        lambda **kwargs: dispatched.append(kwargs),
    )
    if window_kind == "train":
        with pytest.raises(ValueError, match="agent_temperature"):
            collector.collect_window(output_root=tmp_path, window_kind=window_kind)
        assert not dispatched
        with pytest.raises(FatalTrainingError, match="agent_temperature"):
            raise_if_run_failed(tmp_path)
    else:
        collector.collect_window(output_root=tmp_path, window_kind=window_kind)
        assert len(dispatched) == 1
        raise_if_run_failed(tmp_path)
