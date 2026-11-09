from __future__ import annotations

import json
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_run_failure import cli_harness  # noqa: F401

from training import cli
from training import policy_training_backend as backend_module
from training import official_eval_policy as eval_policy
from training.policy_training_backend import (
    FullWeightPolicyTrainingBackend,
    TrainingResume,
)
from training.policy_sync import WeightManifest


def _config(*, weight_mode: str, checkpoint_every: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_iterations=5,
        official_eval=SimpleNamespace(
            enabled=False,
            startup_timeout_seconds=30.0 * 60.0,
        ),
        inference=SimpleNamespace(
            server_enable_lora=weight_mode == "lora",
            model_id="Qwen/Qwen3.6-27B",
        ),
        training=SimpleNamespace(
            weight_mode=weight_mode,
            init_adapter_dir=None,
            init_trainer_state_path=None,
            model_name="Qwen/Qwen3.6-27B",
            model_revision="revision-a",
            full_weight=SimpleNamespace(
                checkpoint_every_iterations=checkpoint_every,
                checkpoint_keep_last=6,
            ),
        ),
    )


class _FakeRuntime:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.trainer = SimpleNamespace(optimizer_step_count=9)

    def create_trainer(self, config, **kwargs):  # noqa: ANN001, ANN003
        self.events.append(("create", kwargs["checkpoint_dir"]))
        return self.trainer

    def publication_manifest(self) -> WeightManifest:
        self.events.append("manifest")
        return WeightManifest(("weight",), ("bfloat16",), ((2, 2),))

    def publish(self, send) -> None:  # noqa: ANN001
        self.events.append("publish")
        send(iter([("weight", torch.ones(2, 2))]))

    def train_window(self, rollouts, **kwargs):  # noqa: ANN001, ANN003
        self.events.append(("train", list(rollouts), kwargs))
        return "result"

    def checkpoint(self, *, root: Path, name: str, **kwargs) -> Future[Path]:  # noqa: ANN003
        path = root / name
        path.mkdir(parents=True)
        self.events.append(("checkpoint", name, kwargs))
        future: Future[Path] = Future()
        future.set_result(path)
        return future

    def close(self) -> None:
        self.events.append("close")


def test_full_weight_resume_and_periodic_checkpoint_state(tmp_path: Path) -> None:
    events: list[object] = []
    runtime = _FakeRuntime(events)
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=2), runtime
    )
    with pytest.raises(RuntimeError, match="version 2 full_weight"):
        backend.resolve_resume(tmp_path, {"version": 1})
    with pytest.raises(RuntimeError, match="iteration fields are inconsistent"):
        backend.resolve_resume(
            tmp_path,
            {
                "version": 2,
                "weight_mode": "full_weight",
                "next_iteration": 3,
                "last_completed_iteration": 1,
                "policy_iteration": 1,
                "checkpoint_path": "unused",
            },
        )

    checkpoint = tmp_path / "checkpoints" / "full_weight_iter_000000"
    checkpoint.mkdir(parents=True)
    resume = backend.resolve_resume(
        tmp_path,
        {
            "version": 2,
            "weight_mode": "full_weight",
            "next_iteration": 2,
            "last_completed_iteration": 1,
            "policy_iteration": 1,
            "next_schedule_index": 4,
            "checkpoint_path": "checkpoints/full_weight_iter_000000",
        },
    )
    assert resume.full_weight_checkpoint == checkpoint

    first_state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=object(),
        common_run_state={"next_iteration": 1},
    )
    assert first_state is not None
    assert (
        first_state["checkpoint_path"]
        == "checkpoints/full_weight_iter_000000_retry_001"
    )
    assert (
        backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=2,
            train_result=object(),
            common_run_state={"next_iteration": 3},
        )
        is None
    )
    state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=1,
        train_result=object(),
        common_run_state={"next_iteration": 2},
    )
    assert state == {
        "version": 2,
        "weight_mode": "full_weight",
        "next_iteration": 2,
        "checkpoint_path": "checkpoints/full_weight_iter_000001",
        "policy_iteration": 1,
    }
    assert not (
        {"adapter_dir", "runtime_adapter_dir", "trainer_state_path"} & set(state)
    )
    retry_state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=1,
        train_result=object(),
        common_run_state={"next_iteration": 2},
    )
    assert retry_state is not None
    assert retry_state["checkpoint_path"] == (
        "checkpoints/full_weight_iter_000001_retry_001"
    )


def test_full_weight_checkpoint_upload_overlaps_following_iterations(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    class _DelayedRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__(events)
            self.checkpoint_future: Future[Path] = Future()
            self.checkpoint_path: Path | None = None

        def checkpoint(self, *, root: Path, name: str, **kwargs) -> Future[Path]:  # noqa: ANN003
            self.checkpoint_path = root / name
            self.events.append(("checkpoint", name, kwargs))
            return self.checkpoint_future

    runtime = _DelayedRuntime()
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=20), runtime
    )

    assert (
        backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=0,
            train_result=object(),
            common_run_state={"next_iteration": 1},
        )
        is None
    )
    assert (
        backend.train_window(["next-window"], iteration_index=1, entropy_coef=0.0)
        == "result"
    )
    assert (
        backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=1,
            train_result=object(),
            common_run_state={"next_iteration": 2},
        )
        is None
    )

    assert runtime.checkpoint_path is not None
    runtime.checkpoint_path.mkdir(parents=True)
    runtime.checkpoint_future.set_result(runtime.checkpoint_path)
    durable_state = json.loads(
        (tmp_path / "training_state.json").read_text(encoding="utf-8")
    )
    assert durable_state["next_iteration"] == 1
    assert durable_state["policy_iteration"] == 0
    state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=2,
        train_result=object(),
        common_run_state={"next_iteration": 3},
    )

    assert state is not None
    assert state["next_iteration"] == 1
    assert state["policy_iteration"] == 0
    assert state["checkpoint_path"] == "checkpoints/full_weight_iter_000000"


def test_full_weight_checkpoint_does_not_return_stale_state_while_newer_pending(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    class _QueuedRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__(events)
            self.checkpoint_futures: list[Future[Path]] = []

        def checkpoint(self, *, root: Path, name: str, **kwargs) -> Future[Path]:  # noqa: ANN003
            path = root / name
            path.mkdir(parents=True)
            future: Future[Path] = Future()
            self.checkpoint_futures.append(future)
            self.events.append(("checkpoint", name, kwargs))
            return future

    runtime = _QueuedRuntime()
    config = _config(weight_mode="full_weight", checkpoint_every=1)
    config.training.full_weight.checkpoint_keep_last = 1
    backend = FullWeightPolicyTrainingBackend(config, runtime)

    assert (
        backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=0,
            train_result=None,
            common_run_state={"next_iteration": 1},
        )
        is None
    )
    first_path = tmp_path / "checkpoints" / "full_weight_iter_000000"
    runtime.checkpoint_futures[0].set_result(first_path)

    state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=1,
        train_result=None,
        common_run_state={"next_iteration": 2},
    )

    assert state is None
    assert (
        json.loads((tmp_path / "training_state.json").read_text())["next_iteration"]
        == 1
    )
    second_path = tmp_path / "checkpoints" / "full_weight_iter_000001"
    runtime.checkpoint_futures[1].set_result(second_path)
    durable_state = json.loads((tmp_path / "training_state.json").read_text())
    assert durable_state["next_iteration"] == 2
    assert durable_state["checkpoint_path"] == ("checkpoints/full_weight_iter_000001")
    assert not first_path.exists()


def test_full_weight_forced_finalize_reuses_current_checkpoint(tmp_path: Path) -> None:
    events: list[object] = []
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=1),
        _FakeRuntime(events),
    )
    backend.checkpoint_latest_iteration()
    assert events == []

    committed = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=None,
        common_run_state={"next_iteration": 1},
    )
    forced = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=None,
        common_run_state={"next_iteration": 1},
        force_checkpoint=True,
    )

    checkpoint_events = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "checkpoint"
    ]
    assert committed is not None
    assert forced is None
    backend.checkpoint_latest_iteration()
    assert len(events) == 1
    assert len(checkpoint_events) == 1


def test_full_weight_forced_checkpoint_off_cadence_is_durable(
    tmp_path: Path,
) -> None:
    """Segment yield: force_checkpoint writes and commits a checkpoint at an
    iteration the cadence would skip, so a clean pre-timeout exit loses no
    completed iterations."""
    events: list[object] = []
    config = _config(weight_mode="full_weight", checkpoint_every=1000)
    config.num_iterations = 2000
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime(events))

    unforced = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=6,
        train_result=None,
        common_run_state={"next_iteration": 7},
    )
    assert unforced is None
    assert not any(
        isinstance(event, tuple) and event[0] == "checkpoint" for event in events
    )

    committed = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=7,
        train_result=None,
        common_run_state={"next_iteration": 8},
        force_checkpoint=True,
    )

    checkpoint_events = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "checkpoint"
    ]
    assert [event[1] for event in checkpoint_events] == ["full_weight_iter_000007"]
    assert committed is not None
    assert committed["weight_mode"] == "full_weight"
    assert committed["next_iteration"] == 8
    assert committed["policy_iteration"] == 7
    assert (tmp_path / "checkpoints" / "full_weight_iter_000007").is_dir()


def test_full_weight_checkpoint_commit_prunes_superseded_directories(
    tmp_path: Path,
) -> None:
    config = _config(weight_mode="full_weight", checkpoint_every=1)
    config.training.full_weight.checkpoint_keep_last = 2
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime([]))

    for iteration in range(4):
        state = backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=iteration,
            train_result=None,
            common_run_state={"next_iteration": iteration + 1},
        )
        assert state is not None

    checkpoints = sorted(
        path.name for path in (tmp_path / "checkpoints").glob("full_weight_iter_*")
    )
    assert checkpoints == ["full_weight_iter_000002", "full_weight_iter_000003"]
    run_state = json.loads((tmp_path / "training_state.json").read_text())
    assert run_state["checkpoint_path"] == "checkpoints/full_weight_iter_000003"


def test_checkpoint_retention_keeps_best_and_inflight_eval_after_delayed_upload(
    tmp_path,
):
    def record(iteration, score, cycle):
        eval_policy.record_full_weight_eval_result(
            tmp_path,
            {
                "iteration": iteration,
                "cycle_index": cycle,
                "avg_arc_score": score,
                "completed_at": "now",
            },
        )

    root = tmp_path / "checkpoints"
    for i in range(6):
        (root / f"full_weight_iter_{i:06d}").mkdir(parents=True)
    record(0, 0.08, 1)
    record(1, 0.07, 2)
    record(1, 0.08, 3)  # A tie keeps the first best.
    ready = tmp_path / "official_eval_policy_ready.json"
    ready.write_text(json.dumps({"policy_iteration": 2}))
    upload = Future()
    committed = FullWeightPolicyTrainingBackend._commit_checkpoint_when_durable(
        upload,
        run_root=tmp_path,
        state={"policy_iteration": 5},
        keep_last=2,
    )
    upload.set_result(root / "full_weight_iter_000005")
    committed.result()
    assert sorted(p.name for p in root.iterdir()) == [
        f"full_weight_iter_{i:06d}" for i in [0, 2, 4, 5]
    ]
    record(2, 0.1, 20)  # Eval cycle 20 evaluated model iteration 2.
    ready.write_text(json.dumps({"policy_iteration": 5}))
    backend_module._prune_full_weight_checkpoints(
        root,
        keep_last=2,
        protected_checkpoint=root / "full_weight_iter_000005",
    )
    assert not (root / "full_weight_iter_000000").exists()
    assert (root / "full_weight_iter_000002").is_dir()
    best = json.loads((tmp_path / "best_eval.json").read_text())
    assert best["iteration"] == 2 and best["cycle_index"] == 20


def test_retention_reads_active_pin_before_best_during_eval_transition(
    tmp_path, monkeypatch
):
    ready = tmp_path / "official_eval_policy_ready.json"
    best = tmp_path / "best_eval.json"
    ready.write_text('{"policy_iteration": 2}')
    best.write_text('{"iteration": 0}')
    original_read = eval_policy._read_json

    def read_then_transition(path):
        result = original_read(path)
        if path == ready:
            # Eval finishes and requests the next policy between these reads.
            best.write_text('{"iteration": 2}')
            ready.write_text('{"policy_iteration": 5}')
        return result

    monkeypatch.setattr(eval_policy, "_read_json", read_then_transition)
    assert eval_policy.protected_full_weight_eval_iterations(tmp_path) == {2}


def test_evaluated_policy_gets_checkpoint_even_between_scheduled_saves(
    tmp_path, monkeypatch
):
    events = []
    _fake_transport(monkeypatch, events)
    config = _config(weight_mode="full_weight", checkpoint_every=20)
    config.num_iterations = 100
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime(events))
    backend.create_trainer(
        game_vocab=["game"], memory_logger=None, resume=TrainingResume()
    )
    backend._eval_inbox = eval_policy.FullWeightEvalRequestInbox(run_root=tmp_path)
    backend._service_eval_request(
        eval_policy.FullWeightEvalRequest("eval-7", 2, ("http://eval:8100",)),
        policy_iteration=7,
    )
    backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=7,
        train_result=None,
        common_run_state={"next_iteration": 8},
    )
    assert (tmp_path / "checkpoints/full_weight_iter_000007").is_dir()


def test_full_weight_checkpoint_prune_failure_does_not_fail_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(weight_mode="full_weight", checkpoint_every=1)
    config.training.full_weight.checkpoint_keep_last = 1
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime([]))

    def fail_prune(_path: Path) -> None:
        raise OSError("shared filesystem is busy")

    monkeypatch.setattr(
        "training.policy_training_backend.shutil.rmtree",
        fail_prune,
    )
    for iteration in range(2):
        state = backend.finalize_iteration(
            run_root=tmp_path,
            iteration_index=iteration,
            train_result=None,
            common_run_state={"next_iteration": iteration + 1},
        )
        assert state is not None

    run_state = json.loads((tmp_path / "training_state.json").read_text())
    assert run_state["checkpoint_path"] == "checkpoints/full_weight_iter_000001"
    assert "retention will retry" in caplog.text


def test_full_weight_checkpoint_listing_failure_does_not_fail_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(weight_mode="full_weight", checkpoint_every=1)
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime([]))

    def fail_retention(*_args, **_kwargs) -> None:
        raise OSError("stale shared filesystem mount")

    monkeypatch.setattr(
        "training.policy_training_backend._prune_full_weight_checkpoints",
        fail_retention,
    )
    state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=None,
        common_run_state={"next_iteration": 1},
    )

    assert state is not None
    run_state = json.loads((tmp_path / "training_state.json").read_text())
    assert run_state["checkpoint_path"] == "checkpoints/full_weight_iter_000000"
    assert "retention pass failed" in caplog.text


class _Collector:
    def __init__(self, events):
        self.events = events
        self.collectors = [
            SimpleNamespace(
                config=SimpleNamespace(
                    inference=SimpleNamespace(
                        base_url="http://rollout:8000",
                        model_id="Qwen/Qwen3.6-27B",
                    )
                ),
                _server_auth_headers=lambda: {"Authorization": "Bearer test"},
                pin_full_weight_identity=lambda identity: events.append(
                    ("pin", identity)
                ),
            )
        ]

    def prepare_servers(self):
        self.events.append("prepare")

    def activate_policy_at_boundary(
        self, *, policy_iteration, activate, deadline_epoch=None
    ):
        self.events.append(("boundary", policy_iteration))
        activate()
        return True

    def start(self):
        self.events.append("first-draw")


def _fake_transport(monkeypatch, events):
    class Endpoint:
        def send_weights(self, parameters):
            events.append(("send", tuple(name for name, tensor in parameters)))

        def close(self):
            pass

    class Client:
        def __init__(self, url, **kwargs):
            self.url = url
            self.adapter_identity = None

    class Transfer:
        def __init__(self, clients, *, trainer):
            self.clients = list(clients)
            self.urls = tuple(client.url for client in clients)

        def transfer(self, manifest, *, send_weights, weight_version):
            events.append(("transfer", self.urls, weight_version))
            send_weights()
            for client in self.clients:
                client.adapter_identity = {
                    "name": "Qwen/Qwen3.6-27B",
                    "id": 0,
                    "path": None,
                    "server_instance_id": client.url,
                    "kind": "full_weight",
                    "weight_version": weight_version,
                    "manifest_sha256": manifest.sha256,
                }

    monkeypatch.setattr(backend_module, "VllmNcclTrainerEndpoint", Endpoint)
    monkeypatch.setattr(backend_module, "VllmControlClient", Client)
    monkeypatch.setattr(backend_module, "VllmNcclWeightTransfer", Transfer)


@pytest.mark.parametrize("initial_iteration", [-1, 7])
def test_fresh_and_resumed_weights_reach_rollout_and_eval_before_first_draw(
    monkeypatch,
    tmp_path,
    initial_iteration,
):
    events = []
    _fake_transport(monkeypatch, events)
    config = _config(weight_mode="full_weight")
    config.official_eval.enabled = True
    backend = FullWeightPolicyTrainingBackend(config, _FakeRuntime(events))
    checkpoint = None
    if initial_iteration >= 0:
        checkpoint = (
            tmp_path / "checkpoints" / f"full_weight_iter_{initial_iteration:06d}"
        )
        checkpoint.mkdir(parents=True)
    resume = TrainingResume(
        last_completed_iteration=initial_iteration,
        full_weight_checkpoint=checkpoint,
    )
    backend.create_trainer(game_vocab=["game"], memory_logger=None, resume=resume)
    request = {
        "version": eval_policy._PROTOCOL_VERSION,
        "request_id": "eval-start",
        "cycle_index": 0,
        "server_urls": ["http://eval:8100"],
    }
    (tmp_path / "official_eval_policy_request.json").write_text(json.dumps(request))
    backend.prepare_rollouts(_Collector(events), resume=resume, run_root=tmp_path)
    transfers = [
        event for event in events if isinstance(event, tuple) and event[0] == "transfer"
    ]
    assert [event[1] for event in transfers] == [
        ("http://rollout:8000",),
        ("http://eval:8100",),
    ]
    assert transfers[0][2] == transfers[1][2]
    assert events[0] == ("create", checkpoint)
    assert events[-1] == "first-draw"
    ready = json.loads((tmp_path / "official_eval_policy_ready.json").read_text())
    assert ready["policy_iteration"] == initial_iteration
    assert ready["request_id"] == "eval-start"
    pinned = [
        event[1] for event in events if isinstance(event, tuple) and event[0] == "pin"
    ]
    assert pinned[0]["weight_version"] == transfers[0][2]
    assert pinned[0]["server_instance_id"] == "http://rollout:8000"
    assert ready["publication_identities"][0]["weight_version"] == transfers[0][2]
    assert (
        ready["publication_identities"][0]["server_instance_id"] == "http://eval:8100"
    )
    assert not (tmp_path / "checkpoints/.full_weight_in_progress").exists()


def test_eval_cycle_requests_gate_transfer_and_repeat_policy_only_acknowledges(
    monkeypatch,
    tmp_path,
):
    events = []
    _fake_transport(monkeypatch, events)
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight"), _FakeRuntime(events)
    )
    backend.create_trainer(
        game_vocab=["game"], memory_logger=None, resume=TrainingResume()
    )
    backend._eval_inbox = eval_policy.FullWeightEvalRequestInbox(run_root=tmp_path)
    provider = eval_policy.FullWeightEvalPolicyProvider(
        run_root=tmp_path,
        base_model_id="Qwen/Qwen3.6-27B",
        server_urls=["http://eval:8100"],
        poll_seconds=0,
    )
    policy_iteration = -1
    monkeypatch.setattr(
        eval_policy.time,
        "sleep",
        lambda _: backend._service_pending_eval_request(
            policy_iteration=policy_iteration,
        ),
    )
    assert provider.acquire(cycle_index=0).iteration == -1
    assert provider.acquire(cycle_index=1).iteration == -1
    assert (
        len(
            [
                event
                for event in events
                if isinstance(event, tuple) and event[0] == "transfer"
            ]
        )
        == 1
    )
    backend.train_window(["rollout"], iteration_index=0)
    policy_iteration = 0
    # Training and publication hooks do not mutate eval while its cycle runs.
    backend._service_pending_eval_request(policy_iteration=0)
    assert (
        len(
            [
                event
                for event in events
                if isinstance(event, tuple) and event[0] == "transfer"
            ]
        )
        == 1
    )
    assert provider.acquire(cycle_index=2).iteration == 0
    transfers = [
        event for event in events if isinstance(event, tuple) and event[0] == "transfer"
    ]
    assert len(transfers) == 2
    assert transfers[0][2] != transfers[1][2]
    backend._service_pending_eval_request(policy_iteration=0)
    assert (
        len(
            [
                event
                for event in events
                if isinstance(event, tuple) and event[0] == "transfer"
            ]
        )
        == 2
    )


def test_failed_eval_transfer_never_acknowledges_or_advances_version(
    monkeypatch, tmp_path
):
    events = []
    _fake_transport(monkeypatch, events)
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight"), _FakeRuntime(events)
    )
    backend.create_trainer(
        game_vocab=["game"], memory_logger=None, resume=TrainingResume()
    )
    backend._eval_inbox = eval_policy.FullWeightEvalRequestInbox(run_root=tmp_path)
    backend._eval_server_urls = ("http://eval:8100",)
    backend._eval_endpoint = object()

    def fail(*args, **kwargs):
        raise RuntimeError("transfer failed")

    backend._eval_transfer = SimpleNamespace(transfer=fail)
    request = eval_policy.FullWeightEvalRequest("request", 1, backend._eval_server_urls)
    with pytest.raises(RuntimeError, match="transfer failed"):
        backend._service_eval_request(request, policy_iteration=3)
    assert not (tmp_path / "official_eval_policy_ready.json").exists()
    assert backend._eval_weight_version is None


def test_full_weight_cli_writes_resume_state_only_after_checkpoint_is_durable(tmp_path):
    config = _config(weight_mode="full_weight")
    config.sampling = SimpleNamespace(return_history_size=4)
    future = Future()
    runtime = _FakeRuntime([])
    runtime.checkpoint = lambda **kwargs: future
    backend = FullWeightPolicyTrainingBackend(config, runtime)
    result = SimpleNamespace(
        optimizer_steps=1,
        advantage_group_mean_reward=0.5,
        advantage_group_std_reward=0.5,
    )
    cli._write_training_run_state(
        run_root=tmp_path,
        config_path="config.json",
        config=config,
        iteration_metrics={"iteration": 0, "next_schedule_index": 2},
        train_result=result,
        trainer=runtime.trainer,
        full_backend=backend,
        sample_budget_state={"current_fraction": 1.0},
    )
    state_path = tmp_path / "training_state.json"
    assert not state_path.exists()
    checkpoint = tmp_path / "checkpoints/full_weight_iter_000000"
    checkpoint.mkdir(parents=True)
    future.set_result(checkpoint)
    state = json.loads(state_path.read_text())
    assert state["version"] == 2
    assert state["weight_mode"] == "full_weight"
    assert state["next_schedule_index"] == 2
    assert state["next_iteration"] == state["policy_iteration"] + 1 == 1
    assert not {
        "adapter_dir",
        "runtime_adapter_dir",
        "trainer_state_path",
    }.intersection(state)
    assert backend.resolve_resume(tmp_path, state).full_weight_checkpoint == checkpoint


def test_full_weight_bench_rejects_before_backend_or_gpu_initialization(monkeypatch):
    config = _config(weight_mode="full_weight")
    monkeypatch.setattr(cli.ExperimentConfig, "load", lambda _: config)
    monkeypatch.setenv("ARC3_BENCH_SYNTHETIC_WINDOWS", "1")
    # The shared synthetic-benchmark refusal precedes the full-weight one;
    # either way no backend or GPU work may start.
    with pytest.raises(ValueError, match="(?i)mandatory policy-parity|full.weight"):
        cli.run_training(
            "unused.json", backend_factory=lambda _: pytest.fail("backend initialized")
        )


def test_final_publication_services_eval_without_waiting_for_unused_rollouts(
    monkeypatch,
):
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight"), _FakeRuntime([])
    )
    serviced = []
    monkeypatch.setattr(
        backend,
        "_service_pending_eval_request",
        lambda **kwargs: serviced.append(kwargs),
    )
    collector = SimpleNamespace(
        activate_policy_at_boundary=lambda **kwargs: pytest.fail(
            "unused rollout publication"
        )
    )
    assert backend.publish_after_train(
        collector, train_result=None, iteration_index=4, should_publish=False
    )
    assert serviced == [{"policy_iteration": 4}]


def test_publication_timing_excludes_only_boundary_wait(monkeypatch):
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight"), _FakeRuntime([])
    )
    clock = [100.0]
    monkeypatch.setattr(backend_module.time, "perf_counter", lambda: clock[0])

    def transfer():
        clock[0] += 20.0

    def boundary(**kwargs):
        clock[0] += 900.0
        kwargs["activate"]()
        return True

    monkeypatch.setattr(backend, "_publish_current_policy", transfer)
    monkeypatch.setattr(
        backend, "_service_pending_eval_request", lambda **_: transfer()
    )
    collector = SimpleNamespace(activate_policy_at_boundary=boundary)
    assert backend.publish_after_train(
        collector, train_result=None, iteration_index=1, should_publish=True
    )
    assert backend.publication_wait_seconds == 900.0
    assert clock[0] - 100.0 - backend.publication_wait_seconds == 40.0
    assert backend.publish_after_train(
        collector, train_result=None, iteration_index=2, should_publish=False
    )
    assert backend.publication_wait_seconds == 0.0


@pytest.mark.parametrize("num_iterations", [1, 2])
def test_cli_full_weight_lifecycle_and_final_publication(request, num_iterations):
    from dataclasses import MISSING, fields
    from unittest.mock import Mock
    from training.ppo import WindowTrainResult

    h = request.getfixturevalue("cli_harness")
    monkeypatch = request.getfixturevalue("monkeypatch")
    clock = [0.0]
    monkeypatch.setattr(h.cli.time, "perf_counter", lambda: clock[0])
    timings = []
    original_update = h.cli._update_sample_budget_state

    def update_budget(**kwargs):
        timings.append(kwargs["trainer_iteration_seconds"])
        return original_update(**kwargs)

    monkeypatch.setattr(h.cli, "_update_sample_budget_state", update_budget)
    config = h.cli.ExperimentConfig.load("unused.json")
    config.training.weight_mode = "full_weight"
    config.num_iterations = num_iterations
    h.trainer.optimizer_step_count = 1
    h.collector.set_next_schedule_index = Mock()
    base_window = h.collector.next_window.return_value
    h.collector.next_window.side_effect = [
        SimpleNamespace(
            **{
                **{
                    field.name: getattr(base_window, field.name)
                    for field in fields(base_window)
                },
                "window_id": f"window-{index}",
                "window_dir": h.root / f"window-{index}",
                "failed_processes": 0,
                "artifact_rollouts": 0,
                "schedule_index": index,
            }
        )
        for index in range(num_iterations)
    ]
    payload = {
        field.name: 0
        for field in fields(WindowTrainResult)
        if field.default is MISSING and field.default_factory is MISSING
    }
    payload.update(
        adapter_dir=None,
        runtime_adapter_dir=None,
        trainer_state_path=None,
        optimizer_steps=1,
        sample_budget_fraction=1.0,
        mean_ratio=1.0,
        mean_advantage_by_level={},
    )
    result = WindowTrainResult(**payload)
    events = []
    stale = h.root / "official_eval_policy_ready.json"
    stale.write_text('{"request_id":"stale"}')

    def create(**kwargs):
        events.append("create")
        h.spawn.assert_not_called()
        return h.trainer

    def prepare(collector, **kwargs):
        assert not stale.exists()
        assert h.spawn.call_count == 1
        events.append("prepare")
        collector.start()

    def train(rollouts, **kwargs):
        events.append("train")
        clock[0] += 100.0
        kwargs["check_health"]()
        return result

    def publish(*args, **kwargs):
        events.append(("publish", kwargs["should_publish"]))
        backend.publication_wait_seconds = 900.0 if kwargs["should_publish"] else 0.0
        clock[0] += backend.publication_wait_seconds

    backend = SimpleNamespace(
        resolve_resume=lambda *args: TrainingResume(),
        create_trainer=create,
        prepare_rollouts=prepare,
        train_window=train,
        publish_after_train=publish,
        finalize_iteration=lambda **kwargs: events.append(
            ("checkpoint", kwargs["iteration_index"])
        ),
        shutdown=lambda **kwargs: events.append(("shutdown", kwargs["graceful"])),
    )
    h.cli.run_training("unused.json", backend_factory=lambda _: backend)
    assert timings == [100.0] * num_iterations
    assert events[:2] == ["create", "prepare"]
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "publish"
    ] == [("publish", index + 1 < num_iterations) for index in range(num_iterations)]
    assert events[-1] == ("shutdown", True)
    h.collector.start.assert_called_once()
    h.trainer_factory.assert_not_called()
    h.collector.publish_policy.assert_not_called()


def _idle_full_weight_backend(h, events: list[object]) -> SimpleNamespace:  # noqa: ANN001
    """A backend that reaches the iteration loop without training anything."""
    from unittest.mock import Mock

    config = h.cli.ExperimentConfig.load("unused.json")
    config.training.weight_mode = "full_weight"
    h.collector.set_next_schedule_index = Mock()
    return SimpleNamespace(
        resolve_resume=lambda *args: TrainingResume(),
        create_trainer=lambda **kwargs: h.trainer,
        prepare_rollouts=lambda collector, **kwargs: collector.start(),
        train_window=lambda *args, **kwargs: pytest.fail("trained"),
        publish_after_train=lambda *args, **kwargs: pytest.fail("published"),
        finalize_iteration=lambda **kwargs: pytest.fail("checkpointed"),
        checkpoint_latest_iteration=lambda: None,
        last_checkpoint_seconds=None,
        shutdown=lambda **kwargs: events.append(("shutdown", kwargs["graceful"])),
    )


def test_cli_scheduled_shutdown_shuts_the_full_weight_backend_down_gracefully(
    request,
):
    """The reviewer's case: TrainingShutdown must not skip the upload wait."""
    import os
    import signal

    from training.shutdown import TrainingShutdown, training_sigterm_handler

    h = request.getfixturevalue("cli_harness")
    events: list[object] = []
    backend = _idle_full_weight_backend(h, events)

    def next_window(**kwargs):  # noqa: ANN003
        # The scheduled TERM lands while the trainer waits for a window; the
        # health check at the boundary turns it into the deferred shutdown.
        os.kill(os.getpid(), signal.SIGTERM)
        kwargs["check_health"]()
        pytest.fail("the deferred shutdown was not raised at the boundary")

    h.collector.next_window.side_effect = next_window
    with training_sigterm_handler():
        with pytest.raises(TrainingShutdown):
            h.cli.run_training("unused.json", backend_factory=lambda _: backend)
    assert events == [("shutdown", True)]
    assert not (h.root / "run_failure.json").exists()
    assert not (h.root / "STOP_AUTO_RELAUNCH").exists()
    h.collector.stop.assert_called_once()


def test_cli_segment_deadline_yields_before_an_unfinishable_iteration(
    request, monkeypatch
):
    from training.segment_budget import DEADLINE_ENV
    from training.shutdown import SegmentYield

    h = request.getfixturevalue("cli_harness")
    events: list[object] = []
    backend = _idle_full_weight_backend(h, events)
    # The deadline has already passed: no iteration can finish before it.
    monkeypatch.setenv(DEADLINE_ENV, str(h.cli.time.time() - 1.0))
    with pytest.raises(SegmentYield):
        h.cli.run_training("unused.json", backend_factory=lambda _: backend)
    h.collector.next_window.assert_not_called()
    assert events == [("shutdown", True)]
    yielded = json.loads((h.root / "segment_yield.json").read_text())
    assert yielded["next_iteration"] == 0
    assert yielded["remaining_seconds"] < 0
    assert not (h.root / "run_failure.json").exists()


@pytest.mark.parametrize("pending_previous", [False, True])
@pytest.mark.parametrize("upload_fails", [False, True])
def test_clock_yield_checkpoints_latest_iteration_before_reporting_success(
    request, monkeypatch, pending_previous, upload_fails
):
    from dataclasses import MISSING, fields, replace
    from queue import Queue
    from training.cli import _resolve_resume_state
    from training.ppo import WindowTrainResult
    from training.segment_budget import SegmentBudget
    from training.shutdown import SegmentYield, shutdown_requested

    h = request.getfixturevalue("cli_harness")
    config = h.cli.ExperimentConfig.load("unused.json")
    config.training.weight_mode = "full_weight"
    config.training.full_weight.checkpoint_every_iterations = 20
    config.num_iterations = 4
    events = []
    uploads = Queue()
    yielding = threading.Event()
    upload_errors = []

    class _UploadRuntime(_FakeRuntime):
        def checkpoint(self, *, root, name, **kwargs):
            iteration = kwargs["iteration_index"]
            if iteration == 0 and not pending_previous:
                return super().checkpoint(root=root, name=name, **kwargs)
            path = root / name
            path.mkdir(parents=True)
            self.events.append(("checkpoint", name, kwargs))
            future = Future()
            uploads.put((iteration, path, future))
            return future

    def finish_uploads():
        future = None
        try:
            assert yielding.wait(timeout=5)
            for _ in range(1 + int(pending_previous)):
                iteration, path, future = uploads.get(timeout=5)
                assert not (h.root / "segment_yield.json").exists()
                if upload_fails and iteration == 1:
                    future.set_exception(OSError("checkpoint upload failed"))
                else:
                    future.set_result(path)
        except BaseException as error:
            upload_errors.append(error)
            if future is not None and not future.done():
                future.set_exception(error)

    runtime = _UploadRuntime(events)
    runtime.trainer = h.trainer
    backend = FullWeightPolicyTrainingBackend(config, runtime)
    h.collector.set_next_schedule_index = lambda _: None
    base = h.collector.next_window.return_value
    h.collector.next_window.side_effect = [
        replace(base, window_id=f"window-{i}", schedule_index=i)
        for i in range(config.num_iterations)
    ]
    payload = {
        field.name: 0
        for field in fields(WindowTrainResult)
        if field.default is MISSING and field.default_factory is MISSING
    }
    payload.update(
        adapter_dir=None,
        runtime_adapter_dir=None,
        trainer_state_path=None,
        optimizer_steps=1,
        sample_budget_fraction=1.0,
        mean_ratio=1.0,
        mean_advantage_by_level={},
    )
    result = WindowTrainResult(**payload)
    clock = [0.0]
    budget = SegmentBudget(
        deadline_epoch=250.0, margin_seconds=0, clock=lambda: clock[0]
    )
    monkeypatch.setattr(
        SegmentBudget, "from_environment", classmethod(lambda cls: budget)
    )
    monkeypatch.setattr(h.cli.time, "perf_counter", lambda: clock[0])
    should_yield = budget.should_yield

    def check_budget():
        decision = should_yield()
        if decision:
            assert h.trainer.optimizer_step_count == 2
            assert not shutdown_requested()
            assert (backend._pending_checkpoint is not None) == pending_previous
            yielding.set()
        return decision

    monkeypatch.setattr(budget, "should_yield", check_budget)

    def prepare(collector, *, run_root, **kwargs):
        backend._run_root = run_root
        collector.start()

    def train(_rollouts, **kwargs):
        kwargs["check_health"]()
        clock[0] += 100.0
        h.trainer.optimizer_step_count += 1
        return result

    monkeypatch.setattr(backend, "prepare_rollouts", prepare)
    monkeypatch.setattr(backend, "publish_after_train", lambda *_a, **_kw: True)
    monkeypatch.setattr(runtime, "train_window", train)
    uploader = threading.Thread(target=finish_uploads, daemon=True)
    uploader.start()
    expected = OSError if upload_fails else SegmentYield
    with pytest.raises(expected):
        h.cli.run_training("unused.json", backend_factory=lambda _: backend)
    uploader.join(timeout=5)
    assert not uploader.is_alive()
    assert upload_errors == []
    state = _resolve_resume_state(h.root, requested_run_dir=True)
    if upload_fails:
        assert not (h.root / "segment_yield.json").exists()
        assert state["next_iteration"] == 1
        assert (h.root / "STOP_AUTO_RELAUNCH").exists()
        assert "checkpoint upload failed" in (h.root / "run_failure.json").read_text()
        return
    yielded = json.loads((h.root / "segment_yield.json").read_text())
    resume = backend.resolve_resume(h.root, state)
    assert yielded["next_iteration"] == state["next_iteration"] == 2
    assert resume.start_iteration == resume.next_schedule_index == 2
    assert resume.full_weight_checkpoint.name == "full_weight_iter_000001"
    assert state["optimizer_step_count"] == h.trainer.optimizer_step_count == 2
    assert events[-1] == "close"
    assert not (h.root / "run_failure.json").exists()


@pytest.mark.parametrize("signal_iteration", [0, 1])
def test_term_during_full_weight_update_commits_before_shutdown_and_resumes(
    request, signal_iteration
):
    """Cover the first checkpoint and an update outside the normal cadence."""
    import os
    import signal
    from dataclasses import MISSING, fields, replace
    from training.cli import _resolve_resume_state
    from training.ppo import WindowTrainResult
    from training.shutdown import (
        TrainingShutdown,
        defer_training_shutdown,
        training_sigterm_handler,
    )

    h = request.getfixturevalue("cli_harness")
    monkeypatch = request.getfixturevalue("monkeypatch")
    config = h.cli.ExperimentConfig.load("unused.json")
    config.training.weight_mode = "full_weight"
    config.training.full_weight.checkpoint_every_iterations = 1000
    config.num_iterations = 3
    events = []
    runtime = _FakeRuntime(events)
    runtime.trainer = h.trainer
    backend = FullWeightPolicyTrainingBackend(config, runtime)
    h.collector.set_next_schedule_index = lambda _: None
    base = h.collector.next_window.return_value
    h.collector.next_window.side_effect = [
        replace(base, window_id=f"window-{i}", schedule_index=i)
        for i in range(config.num_iterations)
    ]
    payload = {
        field.name: 0
        for field in fields(WindowTrainResult)
        if field.default is MISSING and field.default_factory is MISSING
    }
    payload.update(
        adapter_dir=None,
        runtime_adapter_dir=None,
        trainer_state_path=None,
        optimizer_steps=1,
        sample_budget_fraction=1.0,
        mean_ratio=1.0,
        mean_advantage_by_level={},
    )
    result = WindowTrainResult(**payload)

    def prepare(collector, *, run_root, **kwargs):
        backend._run_root = run_root
        collector.start()

    @defer_training_shutdown()
    def train(_rollouts, **kwargs):
        if kwargs["iteration_index"] == signal_iteration:
            os.kill(os.getpid(), signal.SIGTERM)
        kwargs["check_health"]()
        h.trainer.optimizer_step_count += 1
        return result

    monkeypatch.setattr(backend, "prepare_rollouts", prepare)
    monkeypatch.setattr(backend, "publish_after_train", lambda *_a, **_kw: True)
    monkeypatch.setattr(runtime, "train_window", train)
    with training_sigterm_handler(), pytest.raises(TrainingShutdown):
        h.cli.run_training("unused.json", backend_factory=lambda _: backend)
    assert events[-1] == "close"
    state = _resolve_resume_state(h.root, requested_run_dir=True)
    resume = backend.resolve_resume(h.root, state)
    assert resume.start_iteration == signal_iteration + 1
    assert resume.full_weight_checkpoint.is_dir()
    assert state["optimizer_step_count"] == signal_iteration + 1
    assert not (h.root / "run_failure.json").exists()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("legacy_protocol", "protocol"),
        ("missing_identity", "lacks publication identities"),
        ("wrong_count", "lacks publication identities"),
        ("different_version", "different policies"),
        ("different_model", "invalid adapter identity"),
        ("different_cycle", "different cycle"),
        ("unpublished_base", "different policies"),
    ],
)
def test_eval_acknowledgement_requires_matching_per_server_publications(
    tmp_path, defect, message
):
    identity = {
        "name": "model",
        "id": 0,
        "path": None,
        "server_instance_id": "server-one",
        "kind": "full_weight",
        "weight_version": "a" * 32,
        "manifest_sha256": "b" * 64,
    }

    def acknowledge():
        request = json.loads(
            (tmp_path / "official_eval_policy_request.json").read_text()
        )
        ready = {
            "version": eval_policy._PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "cycle_index": 2,
            "policy_iteration": 5,
            "publication_identities": [
                dict(identity),
                {**identity, "server_instance_id": "server-two"},
            ],
        }
        if defect == "legacy_protocol":
            ready["version"] = 1
        elif defect == "missing_identity":
            del ready["publication_identities"]
        elif defect == "wrong_count":
            ready["publication_identities"].pop()
        elif defect == "different_version":
            ready["publication_identities"][1]["weight_version"] = "c" * 32
        elif defect == "different_model":
            ready["publication_identities"][1]["name"] = "other-model"
        elif defect == "different_cycle":
            ready["cycle_index"] = 1
        elif defect == "unpublished_base":
            ready["publication_identities"] = [
                {
                    key: value
                    for key, value in item.items()
                    if key in {"name", "id", "path", "server_instance_id"}
                }
                for item in ready["publication_identities"]
            ]
        (tmp_path / "official_eval_policy_ready.json").write_text(json.dumps(ready))

    provider = eval_policy.FullWeightEvalPolicyProvider(
        run_root=tmp_path,
        base_model_id="model",
        server_urls=["http://one", "http://two"],
        liveness_check=acknowledge,
    )
    with pytest.raises(RuntimeError, match=message):
        provider.acquire(cycle_index=2)


def test_supervised_eval_wait_has_no_update_deadline_and_stops_on_trainer_failure(
    monkeypatch, tmp_path
):
    checks = []

    def require_alive():
        checks.append(None)
        if len(checks) == 3:
            raise RuntimeError("supervising trainer exited")

    provider = eval_policy.FullWeightEvalPolicyProvider(
        run_root=tmp_path,
        base_model_id="model",
        server_urls=["http://eval"],
        request_timeout_seconds=None,
        liveness_check=require_alive,
    )
    provider._has_acquired_policy = True
    monkeypatch.setattr(eval_policy.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        eval_policy.time, "monotonic", lambda: pytest.fail("unexpected update deadline")
    )
    with pytest.raises(RuntimeError, match="supervising trainer exited"):
        provider.acquire(cycle_index=2)
    assert len(checks) == 3


def test_unbounded_eval_wait_requires_trainer_liveness(tmp_path):
    with pytest.raises(ValueError, match="liveness check or timeout"):
        eval_policy.FullWeightEvalPolicyProvider(
            run_root=tmp_path,
            base_model_id="model",
            server_urls=["http://eval"],
            request_timeout_seconds=None,
        )


@pytest.mark.parametrize("flush_before_shutdown", [False, True])
def test_graceful_shutdown_drains_the_pending_upload_and_resume_continues_from_it(
    tmp_path: Path,
    flush_before_shutdown: bool,
) -> None:
    """A scheduled stop commits the in-flight checkpoint before the peers exit."""
    events: list[object] = []
    pending: list[Future[Path]] = []

    class _SlowUploadRuntime(_FakeRuntime):
        def checkpoint(self, *, root: Path, name: str, **kwargs) -> Future[Path]:  # noqa: ANN003
            path = root / name
            path.mkdir(parents=True)
            self.events.append(("checkpoint", name))
            future: Future[Path] = Future()
            pending.append(future)
            return future

    runtime = _SlowUploadRuntime(events)
    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=1), runtime
    )
    state = backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=None,
        common_run_state={
            "next_iteration": 1,
            "last_completed_iteration": 0,
            "next_schedule_index": 1,
        },
    )
    assert state is None  # the upload is still in flight
    assert not (tmp_path / "training_state.json").exists()
    assert backend.last_checkpoint_seconds is None

    drained = threading.Event()

    def finish_upload() -> None:
        drained.wait(timeout=5)
        pending[0].set_result(tmp_path / "checkpoints" / "full_weight_iter_000000")

    uploader = threading.Thread(target=finish_upload)
    uploader.start()
    drained.set()
    if flush_before_shutdown:
        backend.checkpoint_latest_iteration()
    backend.shutdown(graceful=True)
    uploader.join(timeout=5)

    assert events[-1] == "close"
    assert sum(
        isinstance(event, tuple) and event[0] == "checkpoint" for event in events
    ) == 1
    committed = json.loads((tmp_path / "training_state.json").read_text())
    assert committed["checkpoint_path"] == "checkpoints/full_weight_iter_000000"
    assert committed["next_iteration"] == 1
    assert backend.last_checkpoint_seconds is not None
    assert backend.last_checkpoint_seconds >= 0.0

    resumed = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=1), _FakeRuntime([])
    ).resolve_resume(tmp_path, committed)
    assert resumed.start_iteration == 1
    assert resumed.last_completed_iteration == 0
    assert (
        resumed.full_weight_checkpoint
        == tmp_path / "checkpoints" / "full_weight_iter_000000"
    )


def test_non_graceful_shutdown_leaves_the_pending_upload_and_state_untouched(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    class _NeverFinishesRuntime(_FakeRuntime):
        def checkpoint(self, *, root: Path, name: str, **kwargs) -> Future[Path]:  # noqa: ANN003
            (root / name).mkdir(parents=True)
            return Future()

    backend = FullWeightPolicyTrainingBackend(
        _config(weight_mode="full_weight", checkpoint_every=1),
        _NeverFinishesRuntime(events),
    )
    backend.finalize_iteration(
        run_root=tmp_path,
        iteration_index=0,
        train_result=None,
        common_run_state={"next_iteration": 1, "last_completed_iteration": 0},
    )
    backend.shutdown(graceful=False)
    assert "close" not in events
    assert not (tmp_path / "training_state.json").exists()
