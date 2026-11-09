from datetime import datetime as real_datetime
from pathlib import Path

import pytest

from training import run_artifacts


class _FrozenDatetime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 3, 16, 12, 34, 56)


def test_setup_experiment_directory_creates_run_dir_and_git_info(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(run_artifacts, "datetime", _FrozenDatetime)
    monkeypatch.setattr(
        run_artifacts,
        "get_git_info",
        lambda: (
            "[root]\npath: /repo\nbranch: main\ncommit: abc123\n\nstatus:\n(clean)\n\ndiff:\n(no diff)"
        ),
    )

    run_dir, log_file, tensorboard_dir = run_artifacts.setup_experiment_directory(
        tmp_path
    )

    assert run_dir == tmp_path / "20260316_123456"
    assert run_dir.is_dir()
    assert log_file == run_dir / "logs.log"
    assert tensorboard_dir == run_dir / "tensorboard"
    assert tensorboard_dir.is_dir()
    assert (run_dir / "git_info.txt").read_text(encoding="utf-8") == (
        "[root]\npath: /repo\nbranch: main\ncommit: abc123\n\nstatus:\n(clean)\n\ndiff:\n(no diff)"
    )


def test_setup_experiment_directory_avoids_same_second_collision(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(run_artifacts, "datetime", _FrozenDatetime)
    monkeypatch.setattr(run_artifacts, "get_git_info", lambda: "[root]\ncommit: abc123")

    first_run_dir, _, _ = run_artifacts.setup_experiment_directory(tmp_path)
    second_run_dir, _, _ = run_artifacts.setup_experiment_directory(tmp_path)

    assert first_run_dir == tmp_path / "20260316_123456"
    assert second_run_dir == tmp_path / "20260316_123456_01"
    assert second_run_dir.is_dir()


def test_default_shared_storage_root_uses_new_cluster_personal_data(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SHARED_STORAGE_ROOT", raising=False)
    monkeypatch.setenv("CLUSTER_USER", "alice")

    assert run_artifacts.default_shared_storage_root() == Path("/data/personal/alice")


def test_get_git_info_uses_original_repo_root_from_frozen_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    original_repo = tmp_path / "live_repo"
    original_repo.mkdir(parents=True)
    commands: list[tuple[list[str], Path]] = []

    def fake_run_git_command(args: list[str], *, cwd: Path) -> str:
        assert "submodule" not in args
        commands.append((args, cwd))
        return "value"

    monkeypatch.setenv("ARC3_ORIGINAL_REPO_ROOT", str(original_repo))
    monkeypatch.setattr(run_artifacts, "_run_git_command", fake_run_git_command)

    git_info = run_artifacts.get_git_info()

    assert f"path: {original_repo}" in git_info
    assert commands == [
        (["rev-parse", "HEAD"], original_repo),
        (["rev-parse", "--abbrev-ref", "HEAD"], original_repo),
        (["status", "--short"], original_repo),
        (["diff"], original_repo),
    ]


def test_resolve_rollout_windows_root_without_scratch_keeps_run_tree(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ARC3_ROLLOUT_SCRATCH_DIR", raising=False)
    monkeypatch.delenv("SCRATCH_DIR", raising=False)

    default_root = Path("/data/personal/user/training_runs/20260722_010203/rollouts")

    assert run_artifacts.resolve_rollout_windows_root(default_root) == default_root


def test_resolve_rollout_windows_root_maps_to_job_scratch(monkeypatch) -> None:
    monkeypatch.delenv("ARC3_ROLLOUT_SCRATCH_DIR", raising=False)
    monkeypatch.setenv("SCRATCH_DIR", "/scratch/jobs/12345")

    default_root = Path(
        "/data/personal/user/training_runs/20260722_010203/eval_rollouts"
    )

    assert run_artifacts.resolve_rollout_windows_root(default_root) == Path(
        "/scratch/jobs/12345/rollout_windows/20260722_010203/eval_rollouts"
    )


def test_resolve_rollout_windows_root_override_beats_job_scratch(monkeypatch) -> None:
    monkeypatch.setenv("ARC3_ROLLOUT_SCRATCH_DIR", "/raid/scratch/user")
    monkeypatch.setenv("SCRATCH_DIR", "/scratch/jobs/12345")

    default_root = Path("/data/personal/user/training_runs/20260722_010203/rollouts")

    assert run_artifacts.resolve_rollout_windows_root(default_root) == Path(
        "/raid/scratch/user/rollout_windows/20260722_010203/rollouts"
    )


def test_read_text_tail_reads_only_requested_complete_lines(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        ("old-prefix-" * 20_000) + "\nkept-one\nkept-two\nkept-three\n",
        encoding="utf-8",
    )

    assert (
        run_artifacts.read_text_tail(
            log_path,
            lines=2,
            chunk_bytes=32,
        )
        == "kept-two\nkept-three"
    )


def test_read_text_tail_handles_file_without_trailing_newline(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text("old\nkept-one\nkept-two", encoding="utf-8")

    assert run_artifacts.read_text_tail(log_path, lines=2) == "kept-one\nkept-two"


def test_stage_out_rollout_scratch_moves_artifacts_and_appends_logs(tmp_path) -> None:
    run_root = tmp_path / "runs" / "20260722_010203"
    scratch_root = tmp_path / "scratch" / "rollout_windows" / run_root.name
    rollouts_root = scratch_root / "rollouts"
    eval_rollouts_root = scratch_root / "eval_rollouts"
    (rollouts_root / "window_1").mkdir(parents=True)
    (rollouts_root / "window_1" / "rollout.log").write_text(
        "rollout\n", encoding="utf-8"
    )
    (eval_rollouts_root / "window_2").mkdir(parents=True)
    (eval_rollouts_root / "window_2" / "rollout.log").write_text(
        "eval\n", encoding="utf-8"
    )
    run_root.mkdir(parents=True)
    (run_root / "rollouts/window_previous").mkdir(parents=True)
    (run_root / "rollouts/window_previous/rollout.log").write_text(
        "previous rollout\n", encoding="utf-8"
    )
    permanent_log = run_root / "inference_server.log"
    permanent_log.write_text("previous job\n", encoding="utf-8")
    scratch_log = scratch_root / "inference_server.log"
    scratch_log.write_text("current job\n", encoding="utf-8")

    run_artifacts.stage_out_rollout_scratch(
        run_root=run_root,
        rollout_roots=[rollouts_root, eval_rollouts_root],
        server_log_paths=[scratch_log],
        cleanup_rollout_artifacts=False,
    )

    assert (run_root / "rollouts/window_1/rollout.log").read_text(
        encoding="utf-8"
    ) == "rollout\n"
    assert (run_root / "eval_rollouts/window_2/rollout.log").read_text(
        encoding="utf-8"
    ) == "eval\n"
    assert (run_root / "rollouts/window_previous/rollout.log").read_text(
        encoding="utf-8"
    ) == "previous rollout\n"
    assert permanent_log.read_text(encoding="utf-8") == ("previous job\ncurrent job\n")
    assert not rollouts_root.exists()
    assert not eval_rollouts_root.exists()
    assert not scratch_log.exists()


def test_stage_out_rollout_scratch_only_moves_logs_when_cleanup_enabled(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    scratch_root = tmp_path / "scratch"
    rollouts_root = scratch_root / "rollouts"
    rollouts_root.mkdir(parents=True)
    artifact = rollouts_root / "rollout.log"
    artifact.write_text("keep on scratch\n", encoding="utf-8")
    scratch_log = scratch_root / "inference_server.log"
    scratch_log.write_text("keep on scratch\n", encoding="utf-8")

    run_artifacts.stage_out_rollout_scratch(
        run_root=run_root,
        rollout_roots=[rollouts_root],
        server_log_paths=[scratch_log],
        cleanup_rollout_artifacts=True,
    )

    assert artifact.exists()
    assert not scratch_log.exists()
    assert (run_root / "inference_server.log").read_text(encoding="utf-8") == (
        "keep on scratch\n"
    )


def test_failed_rollouts_are_archived_without_copying_the_tree_again(tmp_path):
    import tarfile

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_failure.json").write_text('{"message": "primary failure"}')
    source = tmp_path / "scratch" / "rollouts"
    source.mkdir(parents=True)
    (source / "trace.json").write_text('{"output_token_ids": [248046]}')

    run_artifacts.stage_out_rollout_scratch(
        run_root=run_root,
        rollout_roots=[source],
        server_log_paths=[],
        cleanup_rollout_artifacts=False,
    )

    with tarfile.open(run_root / "failed_rollouts.tar.gz") as archive:
        member = archive.extractfile("rollouts/trace.json")
        assert member is not None
        assert member.read() == (source / "trace.json").read_bytes()
    assert not (run_root / "rollouts").exists()
    assert not list(run_root.glob("*.partial"))
    assert source.exists()


@pytest.mark.parametrize("child_behavior", ["stall", "fail"])
def test_failed_archiver_is_bounded_and_preserves_primary_error_and_logs(
    tmp_path, monkeypatch, child_behavior
):
    import subprocess
    import sys
    import time

    run_root = tmp_path / "run"
    run_root.mkdir()
    failure = run_root / "run_failure.json"
    failure.write_text('{"message": "primary failure"}')
    source = tmp_path / "scratch" / "rollouts"
    source.mkdir(parents=True)
    server_log = source.parent / "server.log"
    server_log.write_text("original engine failure\n")
    processes = []
    popen = subprocess.Popen

    def start_archiver(args):
        code = (
            "import pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).write_text('partial archive'); "
            + ("time.sleep(60)" if child_behavior == "stall" else "sys.exit(7)")
        )
        process = popen([sys.executable, "-c", code, args[-1]])
        processes.append(process)
        # Hand the archiver over once it has written its partial archive: the
        # 0.5 s budget then bounds the stall or the failure, not an interpreter
        # start-up, which can take longer than that on a loaded node.
        partial = Path(args[-1])
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if partial.exists() and partial.read_text() == "partial archive":
                break
            time.sleep(0.01)
        return process

    monkeypatch.setattr(run_artifacts.subprocess, "Popen", start_archiver)
    monkeypatch.setattr(run_artifacts, "_FAILURE_ARCHIVE_TIMEOUT_SECONDS", 0.5)
    run_artifacts.stage_out_rollout_scratch(
        run_root=run_root,
        rollout_roots=[source],
        server_log_paths=[server_log],
        cleanup_rollout_artifacts=False,
    )

    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert (
        run_root / "failed_rollouts.tar.gz.partial"
    ).read_text() == "partial archive"
    assert not (run_root / "failed_rollouts.tar.gz").exists()
    assert not (run_root / "rollouts").exists()
    assert failure.read_text() == '{"message": "primary failure"}'
    assert (run_root / "server.log").read_text() == "original engine failure\n"


@pytest.mark.parametrize("previous_suffix", ["", ".partial"])
def test_failure_archiving_preserves_previous_diagnostics(tmp_path, previous_suffix):
    import tarfile

    source = tmp_path / "rollouts"
    source.mkdir()
    (source / "trace.txt").write_text("current evidence")
    destination = tmp_path / "failed_rollouts.tar.gz"
    previous = tmp_path / (destination.name + previous_suffix)
    previous.write_text("previous evidence")

    run_artifacts._archive_failed_rollouts(source, destination)

    assert previous.read_text() == "previous evidence"
    archives = [p for p in tmp_path.glob("failed_rollouts*.gz") if p != previous]
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        member = archive.extractfile("rollouts/trace.txt")
        assert member is not None
        assert member.read() == b"current evidence"


def test_resolve_runtime_adapter_root_maps_to_job_scratch(monkeypatch) -> None:
    monkeypatch.delenv("ARC3_ROLLOUT_SCRATCH_DIR", raising=False)
    monkeypatch.setenv("SCRATCH_DIR", "/scratch/jobs/123")
    checkpoints = Path("/data/runs/20260825_run/checkpoints")
    assert run_artifacts.resolve_runtime_adapter_root(checkpoints) == Path(
        "/scratch/jobs/123/runtime_adapters/20260825_run"
    )
    monkeypatch.delenv("SCRATCH_DIR")
    assert run_artifacts.resolve_runtime_adapter_root(checkpoints) is None


def test_atomic_json_syncs_file_before_replace_and_directory_after(tmp_path, monkeypatch):
    import json
    import os
    import stat

    from training.cli import _write_json_atomic

    assert _write_json_atomic is run_artifacts.write_json_atomic
    path = tmp_path / "training_state.json"
    events = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def fsync(fd):
        is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
        if not is_dir:
            # The buffered JSON must have reached the file before fsync.
            assert json.loads(next(tmp_path.glob("*.tmp")).read_text()) == {"iteration": 2}
        events.append("directory" if is_dir else "file")
        real_fsync(fd)

    def replace(source, target):
        events.append("replace")
        return real_replace(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "replace", replace)
    _write_json_atomic(path, {"iteration": 2})
    assert events == ["file", "replace", "directory"]
    assert json.loads(path.read_text()) == {"iteration": 2}
