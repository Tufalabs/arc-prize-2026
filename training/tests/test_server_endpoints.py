from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from training.inference import RolloutCollector, sync_collector_runtime_adapter
from training.server_endpoints import (
    collector_server_log_paths,
    configure_server_log_paths,
    server_ready,
)


def _config(train_count, eval_count):
    return SimpleNamespace(
        inference=SimpleNamespace(
            server_gpus=[str(i) for i in range(train_count)], server_log_paths=[]
        ),
        official_eval=SimpleNamespace(
            enabled=bool(eval_count),
            server_gpus=[str(10 + i) for i in range(eval_count)],
            server_log_paths=[],
        ),
    )


@pytest.mark.parametrize("train_count,eval_count", [(1, 0), (1, 1), (2, 1), (2, 2)])
def test_launcher_and_collectors_use_same_durable_logs(
    tmp_path, train_count, eval_count
):
    cfg = _config(train_count, eval_count)
    payload = {
        "inference": vars(cfg.inference).copy(),
        "official_eval": vars(cfg.official_eval).copy(),
    }
    configure_server_log_paths(payload, tmp_path)
    total = train_count + eval_count
    expected = [
        tmp_path
        / ("inference_server.log" if total == 1 else f"inference_server_{i}.log")
        for i in range(total)
    ]
    for evaluation in (False, True):
        key = "official_eval" if evaluation else "inference"
        wanted = expected[train_count:] if evaluation else expected[:train_count]
        # Old configs derive the correct durable paths, independently of rollout scratch.
        assert (
            collector_server_log_paths(cfg, tmp_path, evaluation=evaluation) == wanted
        )
        getattr(cfg, key).server_log_paths = payload[key]["server_log_paths"]
        assert (
            collector_server_log_paths(
                cfg, tmp_path / "unused-scratch", evaluation=evaluation
            )
            == wanted
        )


@pytest.mark.parametrize("paths", [["relative.log"], ["/a.log", "/b.log"]])
def test_explicit_log_mapping_rejects_wrong_count_or_relative_path(tmp_path, paths):
    config = _config(1, 0)
    config.inference.server_log_paths = paths
    with pytest.raises(ValueError, match="one absolute path per server"):
        collector_server_log_paths(config, tmp_path)


def test_bearer_readiness_real_http_and_subprocess_keeps_key_out_of_output(monkeypatch):
    key = "endpoint-test-secret"
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            self.send_response(
                200 if self.headers.get("Authorization") == "Bearer " + key else 401
            )
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/models"
    try:
        monkeypatch.setenv("SERVER_API_KEY", key)
        assert server_ready(url)
        assert not server_ready(url, authenticate=False)
        # This is the exact env-only command used by both single-node launchers.
        result = subprocess.run(
            [sys.executable, "-m", "training.server_endpoints", url],
            env=dict(os.environ),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert key not in result.stdout + result.stderr
        assert calls == [
            ("/models", "Bearer " + key),
            ("/models", None),
            ("/models", "Bearer " + key),
        ]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_single_node_launcher_uses_authenticated_helper_and_records_logs():
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts/platypus_debug_smoke.sbatch"
    ).read_text()
    assert "-m training.server_endpoints" in launcher
    assert "until curl -sf" not in launcher
    assert "configure_server_log_paths(cfg, Path(path).parent)" in launcher
    assert "configure_server_log_paths(cfg, Path(dst).parent)" in launcher


def test_collectors_pin_explicit_paths_instead_of_scratch(tmp_path):
    from training.config import ExperimentConfig, InferenceConfig, TrainingConfig
    from training.inference import ContinuousRolloutWorker

    paths = [str(tmp_path / "durable" / f"engine-{i}.log") for i in range(2)]
    config = ExperimentConfig(
        num_iterations=1,
        parallel_rollouts=2,
        rollout_buffer_size=1,
        inference=InferenceConfig(
            base_url="http://127.0.0.1:50000",
            model_id="qwen",
            game="tu93-0768757b",
            games=["tu93-0768757b"],
            max_runtime_minutes=1,
            server_gpus=["0", "1"],
            server_log_paths=paths,
        ),
        training=TrainingConfig.from_dict(
            {"model_name": "qwen", "output_dir": str(tmp_path)}
        ),
    )
    worker = ContinuousRolloutWorker(
        config,
        repo_root=tmp_path,
        output_root=tmp_path / "scratch" / "rollouts",
        run_root=tmp_path / "durable",
    )
    assert [c.server_log_path for c in worker.collectors] == [
        Path(path) for path in paths
    ]
    assert [c.config.inference.base_url for c in worker.collectors] == [
        "http://127.0.0.1:50000",
        "http://127.0.0.1:50100",
    ]


class _NativeControlPlane:
    """The trainer-facing native routes of one engine, served over real HTTP."""

    def __init__(self):
        self.instance = "engine-a"
        self.adapters = {}
        self.next_id = 1
        self.calls = []
        # Faults: aliases /models leaves out, and identity fields the
        # warm-up draw reports differently from the loaded adapter.
        self.unlisted = set()
        self.warm_changes = {}
        # The LoRA census the load reports: q_proj applied through qkv_proj.
        self.census = {"applied": {"q_proj": "qkv_proj"}, "modules": {"q_proj": 1}}
        # Merged policy weights published over /weights/*: the checkpoint's until one.
        self.policy = None

    def base(self):
        identity = {
            "name": "base",
            "id": 0,
            "path": None,
            "server_instance_id": self.instance,
        }
        if self.policy is not None:
            identity |= {
                "kind": "policy_weights",
                "policy_sha256": self.policy["sha256"],
                "policy_iteration": self.policy["iteration"],
            }
        return identity

    def respond(self, path, body):
        self.calls.append((path.partition("?")[0], (body or {}).get("name")))
        if path == "/policy_weights/info":
            return 200, {
                "sha256": (self.policy or {}).get("sha256"),
                "iteration": (self.policy or {}).get("iteration"),
                "tensors": 0,
                "server_instance_id": self.instance,
            }
        if path.startswith("/metadata"):
            return 200, {"adapter": self.base()}
        if path == "/models":
            listed = [
                item
                for name, item in self.adapters.items()
                if name not in self.unlisted
            ]
            return 200, {
                "models": [self.base()] + listed,
                "max_model_len": 4096,
                "server_instance_id": self.instance,
            }
        if path == "/adapters/load":
            adapter = {
                "name": body["name"],
                "id": self.next_id,
                "path": body["path"],
                "server_instance_id": self.instance,
            }
            self.next_id += 1
            self.adapters[body["name"]] = adapter
            return 200, {"adapter": adapter, "lora_census": self.census}
        if path == "/adapters/unload":
            return 200, {"removed": self.adapters.pop(body["name"], None) is not None}
        if path == "/adapters/warm":
            served = self.base() if body["name"] == "base" else self.adapters.get(body["name"])
            if served is None:
                return 400, {"error": "unknown model"}
            # The gateway's serialized draw: the serving adapter's identity.
            return 200, {
                "outputs": [{"token_ids": [5]}],
                "adapter": served | self.warm_changes,
            }
        return 404, {"error": path}


@pytest.fixture
def native_control_plane(tmp_path, monkeypatch):
    import json

    from training.config import ExperimentConfig, InferenceConfig, TrainingConfig

    monkeypatch.delenv("SERVER_API_KEY", raising=False)
    log_path = tmp_path / "server.log"
    log_path.write_text("")
    plane = _NativeControlPlane()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body):
            status, payload = plane.respond(self.path, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(None)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self._send(json.loads(self.rfile.read(length)))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = ExperimentConfig(
        num_iterations=1,
        parallel_rollouts=1,
        rollout_buffer_size=1,
        inference=InferenceConfig(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model_id="base",
            game="tu93-0768757b",
            games=["tu93-0768757b"],
            max_runtime_minutes=1,
        ),
        training=TrainingConfig.from_dict(
            {"model_name": "qwen", "output_dir": str(tmp_path)}
        ),
    )
    plane.collector = RolloutCollector(
        config, repo_root=tmp_path, server_log_path=log_path
    )
    try:
        yield plane
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _publish(tmp_path, iteration):
    """A runtime LoRA adapter export (the 27B LoRA path)."""
    import json

    runtime = (
        tmp_path / "checkpoints" / f"adapter_iter_{iteration:06d}" / "runtime_vllm"
    )
    runtime.mkdir(parents=True)
    (runtime / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]})
    )
    (runtime / "adapter_model.safetensors").write_bytes(b"tensors")
    (runtime / "runtime_adapter_summary.json").write_text(
        json.dumps({"runtime_target_modules": ["q_proj"], "converted_tensor_count": 2})
    )
    return runtime


def _publish_policy(plane, iteration, *, routers=False):
    """A Flash publication the server verified and resumed over /weights/*: its label and identity."""
    from policy_weight_fixtures import make_publication, checkpoint_tensors
    from training.flash_policy_weights import manifest_sha256

    manifest, _ = make_publication(checkpoint_tensors(), iteration=iteration, routers=routers, seed=iteration)
    digest = manifest_sha256(manifest)
    plane.policy = {"sha256": digest, "iteration": iteration}
    label = {"kind": "policy_weights", "policy_sha256": digest, "policy_iteration": iteration}
    return label, plane.base()


@pytest.fixture
def flash_plane(native_control_plane):
    native_control_plane.collector.config.training.weight_mode = "flash"
    return native_control_plane


@pytest.mark.parametrize("log", [None, "absent.log"])
def test_adapter_sync_needs_no_server_log(native_control_plane, tmp_path, log):
    """HTTP alone confirms the load; a server log is never required."""
    plane = native_control_plane
    plane.collector.server_log_path = None if log is None else tmp_path / log
    runtime = _publish(tmp_path, 7)
    sync_collector_runtime_adapter(
        plane.collector, policy_path=runtime, policy_iteration=7
    )
    assert plane.calls == [
        ("/adapters/load", "policy"),
        ("/models", None),
        ("/adapters/warm", "policy"),
    ]
    assert plane.collector._loaded_runtime_adapter_identity == plane.adapters["policy"]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"id": 99}, "served its warm-up draw with adapter"),
        ({"path": "/elsewhere/runtime_vllm"}, "served its warm-up draw with adapter"),
        ({"server_instance_id": "engine-b"}, "served its warm-up draw with adapter"),
        ({"name": "policy_previous"}, "invalid adapter identity"),
    ],
)
def test_warm_up_under_another_adapter_identity_is_fatal(
    native_control_plane, tmp_path, changes, match
):
    """The load reported one adapter; a warm-up draw served under any other
    identity (a stale id, another path, a restarted server) stops the run
    before any rollout."""
    plane = native_control_plane
    plane.warm_changes = changes
    runtime = _publish(tmp_path, 8)
    with pytest.raises(RuntimeError, match=match):
        sync_collector_runtime_adapter(
            plane.collector, policy_path=runtime, policy_iteration=8
        )
    assert plane.calls[-1] == ("/adapters/warm", "policy")


def test_census_missing_an_exported_module_is_fatal(native_control_plane, tmp_path):
    """The server applied a module set other than the export's: the sync stops
    at the load, before the alias is used for a warm-up or any rollout."""
    plane = native_control_plane
    plane.census = {"applied": {"k_proj": "qkv_proj"}, "modules": {"k_proj": 1}}
    runtime = _publish(tmp_path, 10)
    with pytest.raises(RuntimeError, match=r"missing \['q_proj'\]"):
        sync_collector_runtime_adapter(
            plane.collector, policy_path=runtime, policy_iteration=10
        )
    assert plane.calls == [("/adapters/load", "policy")]


def test_adapter_sync_logs_the_census_once(native_control_plane, tmp_path, caplog):
    plane = native_control_plane
    runtime = _publish(tmp_path, 11)
    with caplog.at_level("INFO", logger="training.inference"):
        sync_collector_runtime_adapter(
            plane.collector, policy_path=runtime, policy_iteration=11
        )
    logged = [r.getMessage() for r in caplog.records if "q_proj->qkv_proj x1" in r.getMessage()]
    assert len(logged) == 1 and "trainer iteration 11" in logged[0]


def test_alias_missing_from_models_is_fatal_and_quotes_the_log(
    native_control_plane, tmp_path
):
    plane = native_control_plane
    plane.unlisted = {"policy"}
    plane.collector.server_log_path.write_text("engine: alias table diverged\n")
    runtime = _publish(tmp_path, 9)
    with pytest.raises(RuntimeError, match=r"not listed by \S+/models") as raised:
        sync_collector_runtime_adapter(
            plane.collector, policy_path=runtime, policy_iteration=9
        )
    assert "engine: alias table diverged" in str(raised.value)
    assert plane.calls == [("/adapters/load", "policy"), ("/models", None)]


@pytest.mark.parametrize("routers", [False, True])
def test_confirmation_checks_info_and_warms_the_base_name(flash_plane, routers):
    """After the NCCL publication: the resumed identity names the label on this
    instance, /policy_weights/info agrees, and the warm-up draw under the base
    name carries the publication's identity."""
    plane = flash_plane
    label, resumed = _publish_policy(plane, 10, routers=routers)
    plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
    assert plane.calls == [("/policy_weights/info", None), ("/adapters/warm", "base")]
    assert plane.collector._loaded_policy_identity == plane.base()
    # Every window start checks the publication and pins it for the draws.
    plane.calls.clear()
    plane.collector.verify_policy_weights()
    assert plane.collector.adapter_identity("base")["policy_sha256"] == label["policy_sha256"]
    assert plane.calls == [("/policy_weights/info", None), ("/metadata", None)]


@pytest.mark.parametrize(
    "changes",
    [{"policy_sha256": "d" * 64}, {"policy_iteration": 11}, {"id": 3}, "no_label"],
)
def test_a_resume_naming_another_publication_is_fatal(flash_plane, changes):
    plane = flash_plane
    label, resumed = _publish_policy(plane, 10)
    resumed = {key: value for key, value in resumed.items() if key not in label} if changes == "no_label" else (
        resumed | changes)
    with pytest.raises(RuntimeError, match="not the published policy weights|invalid adapter identity"):
        plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
    assert plane.calls == []
    assert getattr(plane.collector, "_applied_policy_weights", None) is None


@pytest.mark.parametrize("change", ["weights", "restart"])
def test_window_start_policy_mismatch_is_fatal(flash_plane, change):
    plane = flash_plane
    label, resumed = _publish_policy(plane, 12)
    plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
    if change == "weights":
        plane.policy = {"sha256": "e" * 64, "iteration": 12}
    else:
        # A restarted server serves the checkpoint's weights on a new instance.
        plane.instance, plane.policy = "engine-b", None
    with pytest.raises(RuntimeError, match="Fatal policy weights mismatch"):
        plane.collector.verify_policy_weights()
    with pytest.raises(RuntimeError, match="not the applied merged policy"):
        plane.collector.adapter_identity("base")


@pytest.mark.parametrize(
    "changes",
    [{"policy_sha256": "e" * 64}, {"policy_iteration": 3}, {"server_instance_id": "engine-b"}],
)
def test_warm_up_under_another_publication_is_fatal(flash_plane, changes):
    plane = flash_plane
    plane.warm_changes = changes
    label, resumed = _publish_policy(plane, 13)
    with pytest.raises(RuntimeError, match="not with the published policy weights"):
        plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
    assert plane.calls[-1] == ("/adapters/warm", "base")
    with pytest.raises(RuntimeError, match="not the applied merged policy"):
        plane.collector.adapter_identity("base")


def test_a_server_restarted_between_publications_is_fatal(flash_plane):
    plane = flash_plane
    label, resumed = _publish_policy(plane, 14)
    plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
    plane.instance = "engine-b"
    label, resumed = _publish_policy(plane, 15)
    with pytest.raises(RuntimeError, match="restarted"):
        plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")


def test_flash_never_publishes_by_path(flash_plane, tmp_path):
    plane = flash_plane
    with pytest.raises(RuntimeError, match="over NCCL at a rollout boundary, never by path"):
        sync_collector_runtime_adapter(plane.collector, policy_path=_publish(tmp_path, 16), policy_iteration=16)
    assert plane.calls == []


def test_rollout_worker_takes_flash_policies_at_the_boundary(flash_plane, tmp_path):
    """The trainer's NCCL publication runs between windows, under the collection lock,
    and marks the policy ready for the next window under the base name."""
    from training.inference import ContinuousRolloutWorker

    plane = flash_plane
    worker = ContinuousRolloutWorker(
        plane.collector.config, repo_root=tmp_path, output_root=tmp_path / "rollouts", run_root=tmp_path
    )
    worker.collectors = [plane.collector]
    worker._server_started = True
    published = []

    def activate():
        assert worker._collection_lock.locked()
        label, resumed = _publish_policy(plane, 18)
        plane.collector.confirm_policy_weights(resumed, label=label, model_id="base")
        published.append(label)

    assert worker.activate_policy_at_boundary(policy_iteration=18, activate=activate)
    assert worker._active_model_id == "base" and worker._published_policy_ready
    assert (worker._active_policy_path, worker._active_policy_iteration) == (None, 18)
    assert plane.collector._applied_policy_weights["sha256"] == published[0]["policy_sha256"]
