from __future__ import annotations

import getpass
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from training.rollout_archive import RolloutArchiver


log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUNS_SYMLINK = _PROJECT_ROOT / "runs"
_FAILURE_ARCHIVE_TIMEOUT_SECONDS = 30.0


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace JSON and make the file and directory entry durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staging_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        staging_path.unlink(missing_ok=True)


def cluster_username() -> str:
    return (
        os.environ.get("CLUSTER_USER")
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or getpass.getuser()
    )


def default_shared_storage_root() -> Path:
    override = os.environ.get("SHARED_STORAGE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path("/data/personal") / cluster_username()


def _runs_target() -> Path:
    return default_shared_storage_root() / "training_runs"


def rollout_scratch_root() -> Path | None:
    """Node-local scratch root for transient rollout artifacts, or None.

    ``ARC3_ROLLOUT_SCRATCH_DIR`` overrides; otherwise the Slurm job scratch
    exported by the job script (``SCRATCH_DIR``) is used. None means no
    scratch is configured (local dev, tests) and rollout artifacts stay in
    the run tree.
    """
    for name in ("ARC3_ROLLOUT_SCRATCH_DIR", "SCRATCH_DIR"):
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw)
    return None


def resolve_runtime_adapter_root(checkpoints_root: Path) -> Path | None:
    """Node-local home for the vLLM-format adapter exports, or None.

    The trainer publishes an adapter every iteration and the server on the
    same node resolves it on the next window's first requests (config,
    safetensors, tokenizer probes). Served from the run's NFS checkpoint
    tree, that first-request resolution cost ~55 s per window on a busy
    NFS; on job scratch it is local disk. The peft checkpoint itself stays
    in ``checkpoints_root`` (NFS) for resume, which re-exports when the
    scratch copy is gone. Keyed by run name so concurrent runs sharing a
    scratch dir cannot collide.
    """
    scratch = rollout_scratch_root()
    if scratch is None:
        return None
    return scratch / "runtime_adapters" / checkpoints_root.parent.name


def resolve_rollout_windows_root(default_root: Path) -> Path:
    """Node-local home for a run's transient rollout window/episode trees.

    Rollout children rewrite viewer/prompt/runtime-state files continuously
    while an episode runs, and every window is deleted after its rollouts are
    exported, so these trees are working files, not results. On shared NFS
    that churn is the dominant fs-op load of a run; placing it under the job's
    node-local scratch keeps NFS traffic to the planned stage-out (viewer
    export, replay buffer, checkpoints). Everything that reads these files
    (the collector, viewer export) runs on the same node as the rollout
    children.

    ``default_root`` must be a direct child of the run root (e.g.
    ``<run_root>/rollouts``); the mapping keeps both names so concurrent runs
    sharing a scratch dir cannot collide. Returns ``default_root`` unchanged
    when no scratch dir is configured.
    """
    scratch = rollout_scratch_root()
    if scratch is None:
        return default_root
    return scratch / "rollout_windows" / default_root.parent.name / default_root.name


def read_text_tail(path: Path, *, lines: int, chunk_bytes: int = 64 * 1024) -> str:
    """Read only the final ``lines`` of a text file.

    Server logs grow for the lifetime of a run. Reading them from the end
    keeps per-window snapshots and error reporting proportional to the tail
    size instead of the total run length.
    """
    line_count = max(1, int(lines))
    chunk_size = max(1, int(chunk_bytes))
    chunks: list[bytes] = []
    newline_count = 0

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newline_count <= line_count:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    content = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return "\n".join(content.splitlines()[-line_count:]).strip()


def _append_and_remove(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.move(str(source), str(destination))
        return

    source_size = source.stat().st_size
    if source_size:
        with destination.open("ab+") as output:
            output.seek(0, os.SEEK_END)
            if output.tell():
                output.seek(-1, os.SEEK_END)
                if output.read(1) != b"\n":
                    output.seek(0, os.SEEK_END)
                    output.write(b"\n")
            output.seek(0, os.SEEK_END)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
    source.unlink()


def _archive_failed_rollouts(source: Path, destination: Path) -> None:
    """Bound diagnostic compression and publish only a complete archive."""
    if (
        destination.exists()
        or destination.with_name(destination.name + ".partial").exists()
    ):
        destination = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.gz")
    partial = destination.with_name(destination.name + ".partial")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, tarfile; source = pathlib.Path(sys.argv[1]); "
            "archive = tarfile.open(sys.argv[2], 'w:gz', compresslevel=1); "
            "archive.add(source, arcname=source.name); archive.close()",
            str(source),
            str(partial),
        ],
    )
    try:
        returncode = process.wait(timeout=_FAILURE_ARCHIVE_TIMEOUT_SECONDS)
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            log.error(
                "diagnostic archiver did not exit after SIGKILL: pid=%s", process.pid
            )
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)
    partial.replace(destination)
    log.info("archived failed rollout diagnostics to %s", destination)


def stage_out_rollout_scratch(
    *,
    run_root: Path,
    rollout_roots: list[Path],
    server_log_paths: list[Path],
    cleanup_rollout_artifacts: bool,
    archiver: RolloutArchiver | None = None,
) -> None:
    """Move retained rollout trees and server logs from scratch to the run."""
    failed = (run_root / "run_failure.json").exists()
    archive_error: Exception | None = None
    if archiver is not None:
        try:
            archiver.close(rollout_roots)
        except Exception as error:
            # Still stage engine logs and preserve any original training error.
            archive_error = error
    elif failed:
        for source_root in rollout_roots:
            if source_root.exists() and source_root != run_root / source_root.name:
                destination = run_root / f"failed_{source_root.name}.tar.gz"
                try:
                    _archive_failed_rollouts(source_root, destination)
                except (OSError, subprocess.SubprocessError):
                    log.exception(
                        "failed to archive rollout diagnostics: %s", source_root
                    )
    if archiver is None and not failed and not cleanup_rollout_artifacts:
        for source_root in rollout_roots:
            destination_root = run_root / source_root.name
            if source_root == destination_root or not source_root.exists():
                continue
            destination_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
            shutil.rmtree(source_root)
            log.info("staged retained rollout artifacts to %s", destination_root)

    for source_path in server_log_paths:
        destination_path = run_root / source_path.name
        if source_path == destination_path or not source_path.exists():
            continue
        _append_and_remove(source_path, destination_path)
        log.info("staged vLLM server log to %s", destination_path)
    if archive_error is not None:
        raise archive_error


def _run_git_command(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _format_repo_git_info(name: str, repo_dir: Path) -> str:
    commit = _run_git_command(["rev-parse", "HEAD"], cwd=repo_dir)
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)
    try:
        status = _run_git_command(["status", "--short"], cwd=repo_dir)
    except subprocess.CalledProcessError:
        status = ""
    try:
        diff = _run_git_command(["diff"], cwd=repo_dir)
    except subprocess.CalledProcessError:
        diff = ""

    return "\n".join(
        [
            f"[{name}]",
            f"path: {repo_dir}",
            f"branch: {branch}",
            f"commit: {commit}",
            "",
            "status:",
            status or "(clean)",
            "",
            "diff:",
            diff or "(no diff)",
        ]
    )


def _git_info_repo_root() -> Path:
    original_root = str(os.environ.get("ARC3_ORIGINAL_REPO_ROOT", "") or "").strip()
    if original_root:
        candidate = Path(original_root)
        if candidate.exists():
            return candidate
    return _PROJECT_ROOT


def get_git_info() -> str:
    repo_root = _git_info_repo_root()
    return _format_repo_git_info("root", repo_root)


def save_git_info(base_dir: Path) -> Path:
    path = base_dir / "git_info.txt"
    try:
        path.write_text(get_git_info(), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to capture git info: %s", exc)
        path.write_text(f"git info unavailable: {exc}\n", encoding="utf-8")
    return path


def ensure_output_root(base_output_dir: str | Path = "runs") -> Path:
    output_root = Path(base_output_dir)
    if output_root == _RUNS_SYMLINK:
        runs_target = _runs_target()
        runs_target.mkdir(parents=True, exist_ok=True)
        if output_root.is_symlink():
            current_target = output_root.resolve(strict=False)
            if current_target != runs_target:
                output_root.unlink()
                output_root.symlink_to(runs_target, target_is_directory=True)
        elif output_root.exists():
            raise RuntimeError(
                f"Expected {_RUNS_SYMLINK} to be a symlink to {runs_target}, "
                f"but found an existing non-symlink path."
            )
        else:
            output_root.symlink_to(runs_target, target_is_directory=True)
        return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


_RUN_LAUNCH_TIME_FILENAME = "run_launched_at"
_RUN_LAUNCH_TIME_FORMAT = "%Y%m%d_%H%M%S"


def record_run_launch_time(run_root: Path) -> None:
    """Persist the wall-clock at which training FIRST started in this run dir.

    Written once and never overwritten, so relaunch legs and resumes keep the
    original anchor. This is deliberately not the run-directory timestamp:
    that is minted at sbatch submission, and a long-pending job would
    otherwise start life already hours into any wall-clock schedule.
    """
    marker = run_root / _RUN_LAUNCH_TIME_FILENAME
    if marker.exists():
        return
    marker.write_text(
        datetime.now().strftime(_RUN_LAUNCH_TIME_FORMAT) + "\n", encoding="utf-8"
    )


def read_run_launch_time(run_root: Path | None) -> datetime | None:
    """Return the run's recorded launch wall-clock, if any.

    Reads the marker written by ``record_run_launch_time``. Returns None when
    the run root is unknown or carries no (parseable) marker — callers fall
    back to schedule-less behavior.
    """
    if run_root is None:
        return None
    marker = run_root / _RUN_LAUNCH_TIME_FILENAME
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return datetime.strptime(text, _RUN_LAUNCH_TIME_FORMAT)
    except ValueError:
        return None


def setup_experiment_directory(
    base_output_dir: str | Path = "runs",
) -> tuple[Path, Path, Path]:
    requested_run_dir = os.environ.get("ARC3_RUN_DIR", "").strip()
    requested_timestamp = os.environ.get("RUN_TIMESTAMP", "").strip()

    if requested_run_dir:
        base_dir = Path(requested_run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = requested_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = ensure_output_root(base_output_dir)
        for attempt in range(1000):
            suffix = "" if attempt == 0 else f"_{attempt:02d}"
            base_dir = output_root / f"{timestamp}{suffix}"
            try:
                base_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(
                f"Could not create a unique run directory under {output_root} for {timestamp}."
            )

    tensorboard_dir = base_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(base_dir)

    log_file = base_dir / "logs.log"
    return base_dir, log_file, tensorboard_dir


def setup_logging_for_experiment(
    log_file_path: str | Path | None, fmt: str, *, append: bool = False
) -> Path | None:
    log_path = Path(log_file_path) if log_file_path is not None else None
    root_logger = logging.getLogger()
    log_level = root_logger.level
    if log_level == logging.NOTSET or log_level > logging.INFO:
        log_level = logging.INFO
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt)

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(
            log_path, mode="a" if append else "w", encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    return log_path
