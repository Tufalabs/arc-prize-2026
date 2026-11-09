"""Collection waits for the trained policy rather than an empty buffer slot."""

from pathlib import Path
import queue
import threading

import pytest

from training.config import ExperimentConfig
from training.inference import CollectedWindow, ContinuousRolloutWorker


@pytest.fixture
def worker(tmp_path, monkeypatch):
    # Exercise the real scheduler and collection handoff without starting a
    # serving process. Only the server/model calls and replay IO are replaced.
    path = Path(__file__).resolve().parents[1] / "configs/dppo_2gpu_arc3_ar25.json"
    config = ExperimentConfig.load(path)
    config.training.rollout_schedule = "synchronous"
    config.parallel_rollouts = 2
    config.inference.server_gpus = ["0"]
    instance = ContinuousRolloutWorker(
        config,
        repo_root=tmp_path,
        output_root=tmp_path / "rollouts",
        run_root=tmp_path / "run",
    )
    monkeypatch.setattr(instance, "_ensure_server_ready", lambda: None)
    calls = []
    syncs = []
    waiting = queue.Queue()
    wait = instance._policy_ready.wait

    def observe_wait(*args, **kwargs):
        waiting.put(True)
        return wait(*args, **kwargs)

    def sync_policy(policy_path, policy_iteration):
        syncs.append((policy_path, policy_iteration))
        instance._active_policy_path = policy_path
        instance._active_policy_iteration = policy_iteration
        instance._active_model_id = "runtime-lora"

    def collect_window(**kwargs):
        if instance.config.training.weight_mode in ("full_weight", "flash"):
            assert instance._active_policy_path is None
            assert instance._active_model_id == instance.config.inference.model_id
        else:
            assert instance._active_policy_path is not None
        calls.append((instance._active_policy_iteration, kwargs["schedule_index"]))
        return CollectedWindow(
            window_id=f"window-{len(calls)}",
            window_dir=kwargs["output_root"] / f"window-{len(calls)}",
            rollouts=[],
            policy_iteration=instance._active_policy_iteration,
            schedule_index=kwargs["schedule_index"],
        )

    monkeypatch.setattr(instance._policy_ready, "wait", observe_wait)
    monkeypatch.setattr(instance, "_sync_policy", sync_policy)
    monkeypatch.setattr(instance, "_collect_window_across_servers", collect_window)
    instance.test_calls = calls
    instance.test_syncs = syncs
    instance.test_waiting = waiting
    try:
        yield instance
    finally:
        instance.stop()


def test_synchronous_waits_for_initial_lora_and_every_trained_policy(worker, tmp_path):
    worker.start()
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == []
    assert worker.seconds_until_window_ready() is None

    initial = tmp_path / "init_adapter"
    worker.publish_policy(initial, policy_iteration=-1)
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == [(-1, 0)]
    assert worker.test_syncs == [(initial, -1)]
    assert worker.window_available()

    # Consuming the full window does not release another generation wave.
    window = worker.next_window(timeout=2)
    assert window.policy_iteration == -1
    assert not worker.window_available()
    assert worker.seconds_until_window_ready() is None
    assert worker.test_calls == [(-1, 0)]

    # A duplicate notification must not regenerate on the same weights.
    worker.publish_policy(initial, policy_iteration=-1)
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == [(-1, 0)]

    trained = tmp_path / "adapter_iter_000000"
    worker.publish_policy(trained, policy_iteration=0)
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == [(-1, 0), (0, 1)]
    assert worker.next_window(timeout=2).policy_iteration == 0
    assert worker.test_syncs == [(initial, -1), (trained, 0)]
    assert not worker.window_available()


def test_synchronous_resume_collects_the_published_checkpoint_once(worker, tmp_path):
    worker.set_next_schedule_index(62)
    worker.publish_policy(tmp_path / "adapter_iter_000061", policy_iteration=61)
    worker.start()
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == [(61, 62)]
    assert worker.next_window(timeout=2).schedule_index == 62
    worker.publish_policy(tmp_path / "adapter_iter_000062", policy_iteration=62)
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == [(61, 62), (62, 63)]
    assert worker.next_window(timeout=2).policy_iteration == 62


@pytest.fixture(params=["full_weight", "flash"])
def full_weight_worker(worker, request):
    # Flash's merged policy weights reach the servers the same way: over NCCL
    # from the trainer at a boundary, never by path.
    worker.config.training.weight_mode = request.param
    # Server startup is stubbed by the worker fixture; publication still goes
    # through the real boundary and scheduler rather than an adapter path.
    worker._server_started = True
    return worker


@pytest.mark.parametrize("schedule", ["synchronous", "overlapped"])
@pytest.mark.parametrize("initial_iteration", [-1, 61])
def test_full_weight_collection_requires_successful_publication(
    full_weight_worker, schedule, initial_iteration
):
    worker = full_weight_worker
    worker.config.training.rollout_schedule = schedule
    worker.set_next_schedule_index(initial_iteration + 1)
    with pytest.raises(RuntimeError, match="Rollout collection requires"):
        worker._collect_training_window()
    assert worker.test_calls == []

    published = []
    for iteration in (initial_iteration, initial_iteration + 1):
        assert worker.activate_policy_at_boundary(
            policy_iteration=iteration, activate=lambda: published.append(iteration)
        )
        window = worker._collect_training_window()
        assert window.policy_iteration == iteration
        assert window.schedule_index == iteration + 1
    assert published == [initial_iteration, initial_iteration + 1]
    assert worker.test_calls == [
        (initial_iteration, initial_iteration + 1),
        (initial_iteration + 1, initial_iteration + 2),
    ]
    assert worker.test_syncs == []


@pytest.mark.parametrize("initial_iteration", [-1, 61])
def test_synchronous_full_weight_publication_releases_one_wave(
    full_weight_worker, initial_iteration
):
    worker = full_weight_worker
    worker.set_next_schedule_index(initial_iteration + 1)
    worker.start()
    worker.test_waiting.get(timeout=2)
    assert worker.test_calls == []

    for iteration in (initial_iteration, initial_iteration + 1):
        assert worker.activate_policy_at_boundary(
            policy_iteration=iteration, activate=lambda: None
        )
        worker.test_waiting.get(timeout=2)
        assert worker.next_window(timeout=2).policy_iteration == iteration
        assert not worker.window_available()
        # Publishing the same version again must not release another wave.
        assert worker.activate_policy_at_boundary(
            policy_iteration=iteration, activate=lambda: None
        )
        worker.test_waiting.get(timeout=2)
        assert not worker.window_available()
    assert worker.test_calls == [
        (initial_iteration, initial_iteration + 1),
        (initial_iteration + 1, initial_iteration + 2),
    ]
    assert worker.test_syncs == []


def test_failed_full_weight_publication_wakes_waiter_without_collecting(
    full_weight_worker,
):
    worker = full_weight_worker
    worker.start()
    worker.test_waiting.get(timeout=2)

    def fail_publication():
        raise RuntimeError("weight transfer failed")

    with pytest.raises(RuntimeError, match="weight transfer failed"):
        worker.activate_policy_at_boundary(policy_iteration=-1, activate=fail_publication)
    worker._thread.join(timeout=2)
    assert not worker._thread.is_alive()
    with pytest.raises(RuntimeError, match="FATAL: background rollout"):
        worker.next_window(timeout=2)
    assert worker.test_calls == []


def test_skipped_optimizer_step_can_publish_unchanged_adapter_as_next_version(
    worker, tmp_path
):
    adapter = tmp_path / "unchanged_adapter"
    worker.publish_policy(adapter, policy_iteration=0)
    worker.start()
    worker.test_waiting.get(timeout=2)
    assert worker.next_window(timeout=2).policy_iteration == 0
    # Version counts completed windows, independently of optimizer step count
    # or whether zero advantages left every adapter tensor unchanged.
    worker.publish_policy(adapter, policy_iteration=1)
    worker.test_waiting.get(timeout=2)
    assert worker.next_window(timeout=2).policy_iteration == 1
    assert worker.test_syncs == [(adapter, 0), (adapter, 1)]
    assert worker.test_calls == [(0, 0), (1, 1)]


@pytest.mark.parametrize("publish_initial", [False, True])
def test_stop_wakes_synchronous_policy_wait_without_more_draws(
    worker, tmp_path, publish_initial
):
    if publish_initial:
        worker.publish_policy(tmp_path / "initial", policy_iteration=-1)
    worker.start()
    worker.test_waiting.get(timeout=2)
    expected = [(-1, 0)] if publish_initial else []
    assert worker.test_calls == expected
    worker.stop()
    assert not worker._thread.is_alive()
    assert worker.test_calls == expected
    assert worker._error is None
    assert not (worker.run_root / "run_failure.json").exists()


@pytest.mark.parametrize("violation", ["identity", "backwards", "skip", "prefetch"])
def test_invalid_policy_publication_is_fatal_even_if_caller_swallows_it(
    worker, tmp_path, violation
):
    initial = tmp_path / "initial"
    worker.publish_policy(initial, policy_iteration=4)
    worker.start()
    worker.test_waiting.get(timeout=2)
    if violation != "prefetch":
        worker.next_window(timeout=2)
    version = {"identity": 4, "backwards": 3, "skip": 6, "prefetch": 5}[violation]
    with pytest.raises(RuntimeError, match="Synchronous rollout policy"):
        worker.publish_policy(tmp_path / "bad", policy_iteration=version)
    worker._thread.join(timeout=2)
    assert not worker._thread.is_alive()
    with pytest.raises(RuntimeError, match="FATAL: background rollout"):
        worker.raise_if_failed()
    assert worker.test_calls == [(4, 0)]
    assert (worker.run_root / "STOP_AUTO_RELAUNCH").exists()


def test_policy_cannot_change_during_generation(worker, tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    collect = worker._collect_window_across_servers

    def slow_collection(**kwargs):
        entered.set()
        assert release.wait(2)
        return collect(**kwargs)

    monkeypatch.setattr(worker, "_collect_window_across_servers", slow_collection)
    worker.publish_policy(tmp_path / "initial", policy_iteration=-1)
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(RuntimeError, match="publication requires"):
            worker.publish_policy(tmp_path / "next", policy_iteration=0)
    finally:
        release.set()
    worker._thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="FATAL: background rollout"):
        worker.next_window(timeout=2)
    assert worker.test_calls == [(-1, 0)]


@pytest.mark.parametrize("schedule", ["synchronous", "overlapped"])
@pytest.mark.parametrize("change", ["response_version", "active_identity"])
def test_collection_rejects_policy_drift(
    worker, tmp_path, monkeypatch, change, schedule
):
    worker.config.training.rollout_schedule = schedule
    collect = worker._collect_window_across_servers

    def drifted_collection(**kwargs):
        window = collect(**kwargs)
        if change == "response_version":
            window.policy_iteration = 100
        else:
            worker._active_policy_path = tmp_path / "unexpected"
        return window

    monkeypatch.setattr(worker, "_collect_window_across_servers", drifted_collection)
    worker.publish_policy(tmp_path / "initial", policy_iteration=-1)
    worker.start()
    worker._thread.join(timeout=2)
    assert not worker._thread.is_alive()
    with pytest.raises(RuntimeError, match="changed policy within a window"):
        worker.next_window(timeout=2)
    assert not worker._buffer.qsize()
    assert (worker.run_root / "STOP_AUTO_RELAUNCH").exists()


@pytest.mark.parametrize("schedule", ["synchronous", "overlapped"])
def test_direct_collection_cannot_fall_back_to_base_model(worker, schedule):
    worker.config.training.rollout_schedule = schedule
    with pytest.raises(RuntimeError, match="Rollout collection requires"):
        worker._collect_training_window()
    assert worker.test_calls == []


def test_overlapped_schedule_still_prefetches_when_consuming_a_window(worker, tmp_path):
    worker.config.training.rollout_schedule = "overlapped"
    worker.publish_policy(tmp_path / "initial", policy_iteration=-1)
    worker.start()
    assert worker.next_window(timeout=2).policy_iteration == -1
    assert worker.next_window(timeout=2).policy_iteration == -1
    assert worker.test_calls[:2] == [(-1, 0), (-1, 1)]


def test_overlapped_publication_changes_only_the_next_wave(
    worker, tmp_path, monkeypatch
):
    worker.config.training.rollout_schedule = "overlapped"
    entered, release = threading.Event(), threading.Event()
    original = worker._collect_window_across_servers

    def blocked(**kwargs):
        if kwargs["schedule_index"] == 0:
            entered.set()
            assert release.wait(2)
        return original(**kwargs)

    monkeypatch.setattr(worker, "_collect_window_across_servers", blocked)
    initial, updated = tmp_path / "initial", tmp_path / "updated"
    worker.publish_policy(initial, policy_iteration=-1)
    worker.start()
    assert entered.wait(2)
    try:
        worker.publish_policy(updated, policy_iteration=0)
        assert worker._active_policy_path == initial
    finally:
        release.set()
    assert worker.next_window(timeout=2).policy_iteration == -1
    assert worker.next_window(timeout=2).policy_iteration == 0
    assert worker.test_syncs[:2] == [(initial, -1), (updated, 0)]


@pytest.mark.parametrize("version", [4, 3, 6])
def test_overlapped_publication_rejects_identity_changes_and_version_skips(
    worker, tmp_path, version
):
    worker.config.training.rollout_schedule = "overlapped"
    worker.publish_policy(tmp_path / "initial", policy_iteration=4)
    with pytest.raises(RuntimeError, match="Rollout policy"):
        worker.publish_policy(tmp_path / "different", policy_iteration=version)
    with pytest.raises(RuntimeError, match="FATAL: background rollout"):
        worker.raise_if_failed()


def test_lora_policies_never_publish_at_a_boundary(worker):
    with pytest.raises(RuntimeError, match="requires full_weight or flash mode"):
        worker.activate_policy_at_boundary(policy_iteration=-1, activate=lambda: None)
