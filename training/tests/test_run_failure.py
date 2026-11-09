"""Fatal errors cross process boundaries and stop training without retries."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.run_failure import (
    RUN_FAILURE_FILENAME,
    FatalTrainingError,
    raise_if_run_failed,
    record_run_failure,
)


def test_first_failure_survives_secondary_errors_and_disables_relaunch(tmp_path):
    try:
        raise RuntimeError("presence_penalty must be zero")
    except RuntimeError as error:
        record_run_failure(tmp_path, error)
    record_run_failure(tmp_path, ValueError("secondary teardown error"))
    record = json.loads((tmp_path / RUN_FAILURE_FILENAME).read_text())
    assert record["message"] == "presence_penalty must be zero"
    assert "test_first_failure" in record["traceback"]
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()
    assert not list(tmp_path.glob(".run_failure.*"))
    with pytest.raises(FatalTrainingError, match="presence_penalty must be zero"):
        raise_if_run_failed(tmp_path)


def test_concurrent_processes_publish_one_complete_first_error(tmp_path):
    command = (
        "from pathlib import Path; import sys; "
        "from training.run_failure import record_run_failure; "
        "record_run_failure(Path(sys.argv[1]), RuntimeError(sys.argv[2]))"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", command, str(tmp_path), f"worker-{index}"]
        )
        for index in range(4)
    ]
    for process in processes:
        assert process.wait(timeout=15) == 0
    path = tmp_path / RUN_FAILURE_FILENAME
    first_record = path.read_text()
    record = json.loads(first_record)
    assert record["message"] in {f"worker-{index}" for index in range(4)}
    assert record["error_type"] == "RuntimeError"
    record_run_failure(tmp_path, RuntimeError("later failure"))
    assert path.read_text() == first_record
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()
    assert not list(tmp_path.glob(".run_failure.*"))


@pytest.mark.parametrize("contents", ["{", "[]", '{"message":"missing type"}'])
def test_invalid_failure_record_cannot_be_treated_as_healthy(tmp_path, contents):
    (tmp_path / RUN_FAILURE_FILENAME).write_text(contents)
    with pytest.raises(FatalTrainingError, match="Invalid fatal run record"):
        raise_if_run_failed(tmp_path)


def test_unreadable_failure_record_cannot_be_treated_as_healthy(tmp_path):
    (tmp_path / RUN_FAILURE_FILENAME).mkdir()
    with pytest.raises(FatalTrainingError, match="Cannot read the fatal run record"):
        raise_if_run_failed(tmp_path)


def test_no_failure_record_and_optional_root_are_healthy(tmp_path):
    raise_if_run_failed(tmp_path)
    raise_if_run_failed(None)
    record_run_failure(None, RuntimeError("standalone collector"))
    assert not list(tmp_path.iterdir())


@pytest.fixture
def cli_harness(tmp_path, monkeypatch):
    """Use the real training control flow with inert model/process boundaries."""
    import training.cli as cli

    config = cli.ExperimentConfig.load(
        Path(cli.__file__).resolve().parents[1]
        / "configs/ppo_2gpu_arc3_games25_platypus.json"
    )
    config.num_iterations = 1
    config.training.init_adapter_dir = None
    config.training.init_trainer_state_path = None
    config.metrics.log_per_game = False
    config.memory_profiler.enabled = False
    config.official_eval.enabled = True
    for name in ("ARC3_RUN_DIR", "ARC3_BENCH_SYNTHETIC_WINDOWS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli.ExperimentConfig, "load", lambda path: config)
    monkeypatch.setattr(
        cli,
        "setup_experiment_directory",
        lambda path: (tmp_path, tmp_path / "training.log", tmp_path / "tensorboard"),
    )
    for name in (
        "_configure_hf_cache_defaults",
        "_configure_fast_kernel_defaults",
        "_set_seed",
        "setup_logging_for_experiment",
        "record_run_launch_time",
        "configure_game_sampling",
        "publish_game_nonzero_rates",
        "stage_out_rollout_scratch",
    ):
        monkeypatch.setattr(cli, name, Mock())
    monkeypatch.setattr(cli, "_resolve_training_device", lambda config: "cpu")
    monkeypatch.setattr(cli, "_resolve_resume_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli, "_save_config_snapshots", lambda *args: tmp_path / "config.json"
    )
    monkeypatch.setattr(cli, "resolve_rollout_windows_root", lambda path: path)
    monkeypatch.setattr(cli, "build_memory_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "SummaryWriter", Mock(return_value=Mock()))

    evaluator = SimpleNamespace(pid=12345, returncode=None)
    evaluator.poll = lambda: evaluator.returncode
    evaluator.terminate = lambda: setattr(evaluator, "returncode", -15)
    evaluator.wait = lambda **kwargs: evaluator.returncode
    evaluator.kill = lambda: setattr(evaluator, "returncode", -9)
    spawn = Mock(return_value=evaluator)
    monkeypatch.setattr(cli.subprocess, "Popen", spawn)
    trainer = SimpleNamespace(
        train_window=Mock(),
        shutdown_workers=Mock(),
        optimizer_step_count=0,
        _save_training_state=Mock(return_value=tmp_path / "trainer_state.pt"),
    )
    initial_runtime = tmp_path / "checkpoints" / "adapter_iter_-00001" / "runtime_vllm"

    def export_initial(output_dir, iteration_index):
        assert output_dir == tmp_path / "checkpoints"
        assert iteration_index == -1
        spawn.assert_not_called()
        initial_runtime.mkdir(parents=True, exist_ok=True)
        (initial_runtime / "adapter_config.json").write_text('{"r": 16}')
        return initial_runtime.parent, initial_runtime, 0.0

    trainer._export_policy_snapshot = Mock(side_effect=export_initial)
    trainer_factory = Mock(return_value=trainer)
    monkeypatch.setattr(cli, "PPOTrainer", trainer_factory)
    collector = SimpleNamespace(
        raise_if_failed=Mock(),
        start=Mock(),
        stop=Mock(),
        next_window=Mock(),
        server_log_path=None,
        server_log_paths=[],
        prefix_cache_hit_rates=lambda: {},
        windowed_prefix_cache_hit_rates=lambda: {},
        publish_policy=Mock(),
        window_available=lambda: False,
        seconds_until_window_ready=lambda: None,
    )
    from training.inference import CollectedWindow

    collector.next_window.return_value = CollectedWindow(
        window_id="window-0",
        window_dir=tmp_path / "window-0",
        rollouts=[],
        policy_iteration=-1,
        schedule_index=0,
        collection_duration_seconds=1.0,
        launched_rollouts=0,
        zero_action_rollouts=0,
    )
    monkeypatch.setattr(
        cli, "_build_rollout_collector", lambda *args, **kwargs: collector
    )
    return SimpleNamespace(
        cli=cli,
        root=tmp_path,
        evaluator=evaluator,
        spawn=spawn,
        trainer=trainer,
        trainer_factory=trainer_factory,
        collector=collector,
        initial_runtime=initial_runtime,
    )


@pytest.mark.parametrize("returncode", [0, 17])
def test_dead_eval_worker_is_fatal_and_never_restarted(cli_harness, returncode):
    h = cli_harness
    h.evaluator.returncode = returncode
    (h.root / "official_eval.stdout.log").write_text("presence_penalty must be zero\n")
    with pytest.raises(FatalTrainingError, match="presence_penalty must be zero"):
        h.cli.run_training("unused.json")
    assert h.spawn.call_count == 1
    h.trainer_factory.assert_called_once()
    h.collector.publish_policy.assert_called_once_with(
        h.initial_runtime, policy_iteration=-1
    )
    h.collector.next_window.assert_not_called()
    assert (h.root / "STOP_AUTO_RELAUNCH").exists()


def test_existing_failure_prevents_any_worker_or_model_start(cli_harness):
    h = cli_harness
    record_run_failure(h.root, RuntimeError("original serving failure"))
    with pytest.raises(FatalTrainingError, match="original serving failure"):
        h.cli.run_training("unused.json")
    h.spawn.assert_not_called()
    h.trainer_factory.assert_not_called()


def test_cli_passes_image_identity_to_ordinary_ppo_trainer(cli_harness, monkeypatch):
    h = cli_harness
    config = h.cli.ExperimentConfig.load("unused.json")
    config.inference.agent_multimodal_context = "current_grid"
    h.collector.start.side_effect = lambda: setattr(config, "num_iterations", 0)
    stack = {"sha256": "image-processor"}
    monkeypatch.setattr("training.multimodal.resolve_processing_identity", lambda *a: stack)
    h.cli.run_training("unused.json")
    identity = h.trainer_factory.call_args.kwargs["checkpoint_identity"]
    assert identity["environment"] == "arc3"
    assert identity["multimodal_context"] == "current_grid"
    assert identity["processing_stack"] == stack


def test_train_window_receives_live_fatal_health_check(cli_harness):
    h = cli_harness

    def fail_during_training(*args, check_health, **kwargs):
        check_health()
        record_run_failure(h.root, RuntimeError("evaluation failed during training"))
        check_health()
        pytest.fail("health callback ignored the fatal record")

    h.trainer.train_window.side_effect = fail_during_training
    with pytest.raises(FatalTrainingError, match="evaluation failed during training"):
        h.cli.run_training("unused.json")
    h.trainer.train_window.assert_called_once()
    h.collector.publish_policy.assert_called_once_with(
        h.initial_runtime, policy_iteration=-1
    )
    h.trainer.shutdown_workers.assert_called_once()
    # The initial-policy publication records a resumable state before the
    # first window; the failed window must not advance it.
    state = json.loads((h.root / "training_state.json").read_text())
    assert state["last_completed_iteration"] == -1
    assert state["next_iteration"] == 0


@pytest.mark.parametrize("warm_start", [False, True])
def test_initial_trainer_adapter_is_published_before_collection_and_eval(
    cli_harness, warm_start
):
    from training.official_eval import _await_published_policy

    h = cli_harness
    config = h.cli.ExperimentConfig.load("unused.json")
    seed_adapter = h.root / "seed-adapter" if warm_start else None
    config.training.init_adapter_dir = str(seed_adapter) if warm_start else None

    def assert_published():
        assert _await_published_policy(h.root) == (-1, h.initial_runtime)
        h.collector.publish_policy.assert_called_once_with(
            h.initial_runtime, policy_iteration=-1
        )
        assert h.trainer_factory.call_args.kwargs["resume_adapter_dir"] == seed_adapter

    def spawn(*args, **kwargs):
        assert_published()
        h.collector.start.assert_not_called()
        return h.evaluator

    def start():
        assert_published()
        config.num_iterations = 0

    h.spawn.side_effect = spawn
    h.collector.start.side_effect = start
    h.cli.run_training("unused.json")
    h.trainer._export_policy_snapshot.assert_called_once_with(
        h.root / "checkpoints", -1
    )
    h.trainer.train_window.assert_not_called()
    h.spawn.assert_called_once()


@pytest.mark.parametrize("failure", ["export", "incomplete", "publication"])
def test_initial_adapter_failure_prevents_collection_and_eval(
    cli_harness, monkeypatch, failure
):
    h = cli_harness
    if failure == "export":
        h.trainer._export_policy_snapshot.side_effect = OSError("initial export failed")
    elif failure == "incomplete":
        h.trainer._export_policy_snapshot.side_effect = None
        h.trainer._export_policy_snapshot.return_value = (
            h.initial_runtime.parent,
            h.initial_runtime,
            0.0,
        )
    else:
        write_json_atomic = h.cli._write_json_atomic

        def reject_initial_publication(path, payload):
            if path.name == "initial_policy.json":
                raise OSError("initial publication failed")
            write_json_atomic(path, payload)

        monkeypatch.setattr(h.cli, "_write_json_atomic", reject_initial_publication)
    with pytest.raises((OSError, RuntimeError), match="initial"):
        h.cli.run_training("unused.json")
    assert not (h.root / "initial_policy.json").exists()
    h.spawn.assert_not_called()
    h.collector.start.assert_not_called()
    h.collector.publish_policy.assert_not_called()
    h.trainer.train_window.assert_not_called()


@pytest.mark.parametrize("failure", [None, "export", "publication"])
def test_resume_publishes_regenerated_adapter_before_eval_starts(
    cli_harness, monkeypatch, failure
):
    from training.official_eval import _latest_published_policy

    h = cli_harness
    config = h.cli.ExperimentConfig.load("unused.json")
    config.num_iterations = 5
    # Finish the CPU fixture after startup, before requesting a training window.
    h.collector.start.side_effect = lambda: setattr(config, "num_iterations", 4)
    checkpoint = h.root / "checkpoints" / "adapter_iter_000003"
    checkpoint.mkdir(parents=True)
    trainer_state = checkpoint / "trainer_state.pt"
    trainer_state.touch()
    new_export = h.root / "this-job" / "runtime_vllm"
    state = {
        "version": 1,
        "next_iteration": 4,
        "last_completed_iteration": 3,
        "next_schedule_index": 6,
        "adapter_dir": str(checkpoint),
        "runtime_adapter_dir": str(h.root / "previous-job" / "runtime_vllm"),
        "trainer_state_path": str(trainer_state),
        "updated_at": "2026-09-07T00:00:00Z",
        "optimizer_steps": 17,
    }
    state_path = h.root / "training_state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(h.cli, "_resolve_resume_state", lambda *args, **kwargs: state)
    h.collector.set_next_schedule_index = Mock()

    def export(adapter_dir):
        h.spawn.assert_not_called()
        assert adapter_dir == checkpoint
        if failure == "export":
            raise OSError("runtime export failed")
        new_export.mkdir(parents=True)
        return new_export

    h.trainer._save_runtime_adapter = Mock(side_effect=export)
    if failure == "publication":
        write_json_atomic = h.cli._write_json_atomic

        def fail_publication(path, payload):
            if path == state_path:
                raise OSError("runtime publication failed")
            write_json_atomic(path, payload)

        monkeypatch.setattr(h.cli, "_write_json_atomic", fail_publication)

    def spawn(*args, **kwargs):
        assert _latest_published_policy(h.root) == (3, new_export)
        assert json.loads(state_path.read_text()) == {
            **state,
            "runtime_adapter_dir": "this-job/runtime_vllm",
        }
        return h.evaluator

    h.spawn.side_effect = spawn
    if failure is None:
        h.cli.run_training("unused.json")
        h.spawn.assert_called_once()
        h.collector.publish_policy.assert_called_once_with(
            new_export, policy_iteration=3
        )
        h.collector.set_next_schedule_index.assert_called_once_with(6)
    else:
        with pytest.raises(OSError, match=f"runtime {failure} failed"):
            h.cli.run_training("unused.json")
        h.spawn.assert_not_called()
        h.collector.publish_policy.assert_not_called()
        assert json.loads(state_path.read_text()) == state
        assert (h.root / "STOP_AUTO_RELAUNCH").exists()
    h.trainer.train_window.assert_not_called()


@pytest.mark.parametrize("eval_exits", [False, True])
def test_failure_is_checked_while_waiting_for_rollouts(cli_harness, eval_exits):
    h = cli_harness

    def wait_for_window(*, check_health):
        if eval_exits:
            h.evaluator.returncode = 17
            (h.root / "official_eval.stdout.log").write_text("bad serving settings\n")
        else:
            # The eval process can still be waiting for sibling cleanup.
            record_run_failure(h.root, RuntimeError("bad serving settings"))
        check_health()
        pytest.fail("fatal failure did not interrupt rollout waiting")

    h.collector.next_window.side_effect = wait_for_window
    with pytest.raises(FatalTrainingError, match="bad serving settings"):
        h.cli.run_training("unused.json")
    assert h.spawn.call_count == 1
    h.trainer.train_window.assert_not_called()
    h.collector.publish_policy.assert_called_once_with(
        h.initial_runtime, policy_iteration=-1
    )


@pytest.mark.parametrize(
    "callback", ["next_window_ready", "next_window_seconds_remaining", None]
)
def test_optimizer_health_checks_prevent_reuse_and_policy_publication(
    cli_harness, callback
):
    h = cli_harness

    def optimize(*args, **kwargs):
        record_run_failure(h.root, RuntimeError("serving failed during optimization"))
        if callback is not None:
            kwargs[callback]()
            pytest.fail("extra optimization callback ignored fatal run failure")
        # Even the ordinary optimizer pass must stop immediately on return,
        # before reading its result or publishing another policy.
        return SimpleNamespace()

    h.trainer.train_window.side_effect = optimize
    with pytest.raises(FatalTrainingError, match="serving failed during optimization"):
        h.cli.run_training("unused.json")
    assert h.trainer.train_window.call_count == 1
    h.collector.publish_policy.assert_called_once_with(
        h.initial_runtime, policy_iteration=-1
    )
    assert (h.root / "STOP_AUTO_RELAUNCH").exists()


def test_new_failure_after_window_receipt_prevents_optimizer_entry(cli_harness):
    h = cli_harness

    def metrics_after_window_receipt():
        record_run_failure(h.root, RuntimeError("serving failed before optimization"))
        return {}

    h.collector.prefix_cache_hit_rates = metrics_after_window_receipt
    with pytest.raises(FatalTrainingError, match="serving failed before optimization"):
        h.cli.run_training("unused.json")
    h.trainer.train_window.assert_not_called()
    h.collector.publish_policy.assert_called_once_with(
        h.initial_runtime, policy_iteration=-1
    )


@pytest.mark.parametrize(
    ("sampling_mode", "arc_score"),
    [
        ("balanced", 60.0),
        ("nonzero_rate", 60.0),
        ("nonzero_rate", 0.0),
        ("nonzero_rate", None),
    ],
)
@pytest.mark.parametrize("artifact_mode", ["cleanup", "shared", "scratch"])
def test_complete_iteration_publishes_new_adapter_and_persists_learner_metrics(
    cli_harness,
    sampling_mode,
    arc_score,
    artifact_mode,
    monkeypatch,
):
    from dataclasses import MISSING, fields
    from training.ppo import WindowTrainResult
    from training.contracts import RolloutTrajectory

    h = cli_harness
    config = h.cli.ExperimentConfig.load("unused.json")
    config.training.cleanup_rollout_artifacts = artifact_mode == "cleanup"
    archiver = Mock()
    archiver_factory = Mock(return_value=archiver)
    monkeypatch.setattr(h.cli, "RolloutArchiver", archiver_factory)
    if artifact_mode == "scratch":
        monkeypatch.setattr(
            h.cli, "resolve_rollout_windows_root", lambda path: h.root / "scratch" / path.name
        )
    window_dir = h.root / ("scratch" if artifact_mode == "scratch" else "shared") / "window"
    window_dir.mkdir(parents=True)
    (window_dir / "trace.json").write_text("training input")
    h.collector.next_window.return_value.window_dir = window_dir
    config.sampling.game_sampling_mode = sampling_mode
    if sampling_mode == "nonzero_rate":
        config.sampling.game_sampling_games_per_window = 1
    if arc_score is not None:
        h.collector.next_window.return_value.rollouts = [
            RolloutTrajectory(
                trajectory_id="sampled-game",
                task_id=config.inference.games[0],
                steps=[],
                metadata={"final_arc_score": arc_score, "rollout_policy_iteration": -1},
            )
        ]
    # A real result object prevents mocks accepting stale learner field names.
    values = {
        field.name: 0
        for field in fields(WindowTrainResult)
        if field.default is MISSING and field.default_factory is MISSING
    }
    values.update(
        adapter_dir=h.root / "checkpoints" / "adapter_iter_000000",
        runtime_adapter_dir=h.root
        / "checkpoints"
        / "adapter_iter_000000"
        / "runtime_vllm",
        trainer_state_path=h.root / "checkpoints" / "trainer_state_iter_000000.pt",
        mean_advantage_by_level={},
        baseline_loss=1.25,
        baseline_mean_value=12.0,
        rollout_logprob_parity_mean_abs=0.01,
    )
    result = WindowTrainResult(**values)

    def train(*args, **kwargs):
        assert (window_dir / "trace.json").read_text() == "training input"
        archiver.submit.assert_not_called()
        return result

    h.trainer.train_window.side_effect = train
    h.cli.run_training("unused.json")
    if artifact_mode == "scratch":
        archiver_factory.assert_called_once_with(h.root)
        archiver.submit.assert_called_once_with(window_dir)
        assert h.cli.stage_out_rollout_scratch.call_args.kwargs["archiver"] is archiver
    else:
        archiver_factory.assert_not_called()
        archiver.submit.assert_not_called()
    assert window_dir.exists() == (artifact_mode != "cleanup")
    scalars = {
        call.args[0]: call.args[1]
        for call in h.cli.SummaryWriter.return_value.add_scalar.call_args_list
    }
    if sampling_mode == "nonzero_rate" and arc_score is not None:
        assert scalars["rollout/avg_arc_score"] == pytest.approx(arc_score)
    else:
        assert "rollout/avg_arc_score" not in scalars
    assert "rollout/avg_arc_score_sampled" in scalars
    if arc_score is not None:
        assert scalars["rollout/avg_arc_score_sampled"] == pytest.approx(arc_score)
    state = json.loads((h.root / "training_state.json").read_text())
    assert state["next_iteration"] == 1
    assert state["latest_iteration_metrics"]["baseline_loss"] == 1.25
    assert state["latest_iteration_metrics"]["rollout_logprob_parity_mean_abs"] == 0.01
    h.collector.publish_policy.assert_called_with(
        result.runtime_adapter_dir, policy_iteration=0
    )
    h.trainer.shutdown_workers.assert_called_once()
    assert not (h.root / "STOP_AUTO_RELAUNCH").exists()


_TEARDOWN_502 = (
    "rollout serving contract violated: native vLLM generation failed: HTTP 502: "
    '{"error": "rollout gateway: upstream unreachable after 1 attempt(s); '
    "generation retries are forbidden: Cannot connect to host 127.0.0.1:53913 "
    "ssl:default [Connect call failed ('127.0.0.1', 53913)]\"}"
)


def test_teardown_error_during_scheduled_shutdown_is_not_latched(tmp_path, monkeypatch):
    """A stopped upstream during a scheduled shutdown is the shutdown, not a failure."""
    import training.shutdown as shutdown
    from training.rollout_gateway import GatewayError

    monkeypatch.setattr(shutdown, "_shutdown_requested", True)
    record_run_failure(tmp_path, GatewayError(_TEARDOWN_502))
    assert not (tmp_path / "run_failure.json").exists()
    assert not (tmp_path / "STOP_AUTO_RELAUNCH").exists()
    raise_if_run_failed(tmp_path)


def test_teardown_error_without_shutdown_request_is_latched(tmp_path):
    """The same 502 mid-run is a real serving failure."""
    from training.rollout_gateway import GatewayError

    record_run_failure(tmp_path, GatewayError(_TEARDOWN_502))
    assert (tmp_path / "run_failure.json").exists()
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()


def test_unrelated_error_during_scheduled_shutdown_is_latched(tmp_path, monkeypatch):
    """A pending TERM must not hide a genuine error."""
    import training.shutdown as shutdown

    monkeypatch.setattr(shutdown, "_shutdown_requested", True)
    record_run_failure(tmp_path, RuntimeError("CUDA error: an illegal memory access"))
    assert (tmp_path / "run_failure.json").exists()


def test_is_teardown_error_follows_the_cause_chain():
    from training.run_failure import is_teardown_error

    try:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as inner:
            raise RuntimeError("generation failed") from inner
    except RuntimeError as wrapped:
        assert is_teardown_error(wrapped)
    assert is_teardown_error(RuntimeError(_TEARDOWN_502))
    assert not is_teardown_error(RuntimeError("Same-checkpoint parity check failed"))
    assert not is_teardown_error(ValueError("Incomplete GRPO window"))


def test_scheduled_shutdown_teardown_error_ends_the_run_cleanly(
    cli_harness, monkeypatch
):
    """TERM arrives mid-window, the servers stop, the collector's 502 surfaces:
    the run ends as a scheduled shutdown with no latch and stays resumable."""
    import training.shutdown as shutdown
    from training.rollout_gateway import GatewayError
    from training.shutdown import TrainingShutdown

    h = cli_harness

    def servers_gone_during_collection(*args, **kwargs):
        monkeypatch.setattr(shutdown, "_shutdown_requested", True)
        raise GatewayError(_TEARDOWN_502)

    h.trainer.train_window.side_effect = servers_gone_during_collection
    with pytest.raises(TrainingShutdown):
        h.cli.run_training("unused.json")
    assert not (h.root / "run_failure.json").exists()
    assert not (h.root / "STOP_AUTO_RELAUNCH").exists()
    h.trainer.shutdown_workers.assert_called_once()


def test_segment_deadline_yield_ends_a_lora_segment_with_the_continuation_code(
    cli_harness, monkeypatch
):
    """The CLI exits 75 after an ordinary teardown; the run wants a continuation."""
    import sys

    from training.segment_budget import DEADLINE_ENV
    from training.shutdown import SEGMENT_YIELD_EXIT_CODE

    h = cli_harness
    monkeypatch.setenv(DEADLINE_ENV, str(h.cli.time.time() - 1.0))
    monkeypatch.setattr(sys, "argv", ["training.cli", "--config", "unused.json"])
    with pytest.raises(SystemExit) as exit_info:
        h.cli.main()
    assert exit_info.value.code == SEGMENT_YIELD_EXIT_CODE
    h.collector.next_window.assert_not_called()
    h.trainer.shutdown_workers.assert_called_once()
    h.collector.stop.assert_called_once()
    assert (h.root / "segment_yield.json").exists()
    assert not (h.root / "run_failure.json").exists()
    assert not (h.root / "STOP_AUTO_RELAUNCH").exists()
