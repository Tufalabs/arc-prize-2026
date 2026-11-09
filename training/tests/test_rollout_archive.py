import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from training import rollout_archive as module
from training import run_artifacts


def _window(root, name="window_0"):
    source = root / name
    source.mkdir(parents=True)
    (source / "trace.json").write_text('{"token_ids": [1, 2, 3]}')
    return source


def _wait_for_nonempty(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return
        time.sleep(0.01)
    pytest.fail(f"archive subprocess did not write {path}")


def test_background_process_archives_exact_bytes_and_final_remainders(
    tmp_path, monkeypatch
):
    root = tmp_path / "scratch" / "rollouts"
    source = _window(root)
    payload = bytes(range(256)) * 100
    (source / "tokens.bin").write_bytes(payload)
    run = tmp_path / "run"
    popen = subprocess.Popen
    launches = []

    def launch(command, **kwargs):
        launches.append((command, kwargs["env"]))
        return popen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    archiver = module.RolloutArchiver(run)
    assert archiver.submit(source)
    # A partial/unconsumed window is never submitted during training.
    leftover = _window(root, "window_partial")
    archiver.close([root])
    archives = list((run / "rollout_archives" / "rollouts").glob("*.tar.gz"))
    assert len(archives) == 2
    window_archive = next(
        path for path in archives if path.name.startswith("window_0.")
    )
    with tarfile.open(window_archive) as archive:
        assert archive.extractfile("window_0/tokens.bin").read() == payload
        assert json.load(archive.extractfile("window_0/trace.json")) == {
            "token_ids": [1, 2, 3]
        }
    with tarfile.open(
        next(p for p in archives if p.name.startswith("remaining."))
    ) as archive:
        assert archive.extractfile("rollouts/window_partial/trace.json").read() == (
            b'{"token_ids": [1, 2, 3]}'
        )
    assert not source.exists() and not leftover.exists()
    assert len(launches) == 2
    for command, environment in launches:
        assert command[1] == str(Path(module.__file__).resolve())
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["NVIDIA_VISIBLE_DEVICES"] == "void"
        assert environment["OMP_NUM_THREADS"] == "1"
    assert not list(run.rglob("*.partial"))
    assert not list((tmp_path / "scratch").rglob(".rollout-archive-*"))


def test_slow_writer_does_not_block_submit_and_queue_is_bounded(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []

    def blocked_archive(self, source, **kwargs):
        calls.append((source, threading.get_ident()))
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=5)
        shutil.rmtree(source)
        return True

    monkeypatch.setattr(module.RolloutArchiver, "_archive", blocked_archive)
    root = tmp_path / "scratch" / "rollouts"
    sources = [_window(root, f"window_{i}") for i in range(module._QUEUE_SIZE + 2)]
    archiver = module.RolloutArchiver(tmp_path / "run")
    try:
        archiver.submit(sources[0])
        assert started.wait(timeout=3)
        before = time.monotonic()
        for source in sources[1:-1]:
            assert archiver.submit(source)
        assert not archiver.submit(sources[-1])
        assert time.monotonic() - before < 0.5
        assert sources[0].exists() and sources[-1].exists()
        assert len(calls) == 1  # no second writer while the first is blocked
    finally:
        release.set()
        archiver.close([root])
    assert not root.exists()  # overflow recovered by final sweep
    assert all(thread_id != threading.get_ident() for _, thread_id in calls)


@pytest.mark.parametrize("failure", ["copy", "fsync", "rename"])
def test_failed_publish_keeps_source_and_hides_partial_archive(
    tmp_path, monkeypatch, failure
):
    source = _window(tmp_path / "scratch" / "rollouts")
    destination = tmp_path / "nfs" / "window.tar.gz"

    def fail(*args, **kwargs):
        raise OSError("NFS unavailable")

    if failure == "copy":

        def fail_copy(local, partial, rate):
            partial.write_bytes(b"incomplete gzip")
            fail()

        monkeypatch.setattr(module, "_copy_throttled", fail_copy)
    elif failure == "fsync":
        monkeypatch.setattr(module.os, "fsync", fail)
    else:
        monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="NFS unavailable"):
        module.archive_and_publish(source, destination, 8.0)
    assert (source / "trace.json").exists()
    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial"))
    assert not list(source.parent.glob(".rollout-archive-*"))


def test_publication_is_durable_before_source_removal(tmp_path, monkeypatch):
    source = _window(tmp_path / "scratch")
    destination = tmp_path / "nfs" / "window.tar.gz"
    syncs = []
    fsync, rmtree = os.fsync, shutil.rmtree

    def synced(fd):
        syncs.append(fd)
        assert source.exists()
        fsync(fd)

    def remove(path):
        assert len(syncs) == 2  # file contents, then published directory entry
        with tarfile.open(destination) as archive:
            assert archive.extractfile("window_0/trace.json") is not None
        rmtree(path)

    monkeypatch.setattr(module.os, "fsync", synced)
    monkeypatch.setattr(module.shutil, "rmtree", remove)
    module.archive_and_publish(source, destination, 8.0)
    assert not source.exists()


@pytest.mark.parametrize("nfs_delay", [0.0, 3.0])
def test_transfer_rate_is_paced_in_small_chunks(tmp_path, monkeypatch, nfs_delay):
    source, destination = tmp_path / "local", tmp_path / "remote"
    payload = b"x" * (module._CHUNK_BYTES * 4 + 17)
    source.write_bytes(payload)
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds + nfs_delay

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    module._copy_throttled(source, destination, 1.0)
    assert destination.read_bytes() == payload
    assert sum(sleeps) == pytest.approx(len(payload) / (1024 * 1024))
    assert max(sleeps) <= 0.25


def test_storage_failure_retries_on_shutdown_without_failing_submit(
    tmp_path, monkeypatch
):
    source = _window(tmp_path / "scratch" / "rollouts")
    run = tmp_path / "run"
    # A regular file where the NFS directory should be simulates a failed mount.
    run.write_text("unavailable")
    archiver = module.RolloutArchiver(run)
    assert archiver.submit(source)
    with pytest.raises(OSError, match="stage-out incomplete"):
        archiver.close([source.parent])
    assert source.exists()
    assert not archiver._thread.is_alive()


def test_shutdown_cancels_stalled_child_without_copytree_fallback(
    tmp_path, monkeypatch
):
    source = _window(tmp_path / "scratch" / "rollouts")
    popen = subprocess.Popen
    processes = []
    started = threading.Event()

    def stall(command, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(module.subprocess, "Popen", stall)
    archiver = module.RolloutArchiver(tmp_path / "run")
    archiver.shutdown_seconds = 0.1
    archiver.submit(source)
    assert started.wait(timeout=3)
    before = time.monotonic()
    with pytest.raises(TimeoutError, match="unsaved data remains"):
        archiver.close([source.parent])
    assert time.monotonic() - before < 3
    assert len(processes) == 1 and processes[0].poll() is not None
    assert source.exists()
    assert not archiver._thread.is_alive()


def test_shutdown_reaps_real_worker_and_reclaims_both_temporary_files(
    tmp_path, monkeypatch
):
    source = _window(tmp_path / "scratch" / "rollouts")
    (source / "tokens.bin").write_bytes(os.urandom(1024 * 1024))
    popen = subprocess.Popen
    started = threading.Event()
    tasks = []

    def launch(command, **kwargs):
        local = Path(command[command.index("--local-archive") + 1])
        destination = Path(command[3])
        partial = destination.with_name(f".{destination.name}.partial")
        process = popen(command, **kwargs)
        tasks.append((process, local, partial))
        started.set()
        return process

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    archiver = module.RolloutArchiver(tmp_path / "run")
    archiver.mib_per_second = 0.5
    archiver.shutdown_seconds = 0.01
    try:
        archiver.submit(source)
        assert started.wait(timeout=3)
        process, local, partial = tasks[0]
        _wait_for_nonempty(partial)
        assert local.stat().st_size > 0
        assert local.stat().st_mode & 0o777 == 0o600
        with pytest.raises(TimeoutError, match="unsaved data remains"):
            archiver.close([source.parent])
        assert process.poll() is not None
        assert not archiver._thread.is_alive()
        assert not local.exists() and not partial.exists()
        assert (source / "tokens.bin").stat().st_size == 1024 * 1024
    finally:
        if archiver._thread.is_alive():
            with pytest.raises((OSError, TimeoutError)):
                archiver.close([source.parent])


def test_repeated_remainder_timeouts_reclaim_archives_outside_rollout_root(
    tmp_path, monkeypatch
):
    root = tmp_path / "scratch" / "rollouts"
    source = _window(root)
    (source / "tokens.bin").write_bytes(os.urandom(1024 * 1024))
    popen = subprocess.Popen
    tasks = []

    def launch(command, **kwargs):
        local = Path(command[command.index("--local-archive") + 1])
        destination = Path(command[3])
        partial = destination.with_name(f".{destination.name}.partial")
        process = popen(command, **kwargs)
        wait = process.wait

        def expire_after_first_chunk(timeout):
            if timeout == module._TRANSFER_TIMEOUT_SECONDS:
                # Start the short, real process timeout after packing and the
                # first remote chunk, independently of CPU scheduling latency.
                _wait_for_nonempty(partial)
                return wait(timeout=0.01)
            return wait(timeout=timeout)

        monkeypatch.setattr(process, "wait", expire_after_first_chunk)
        tasks.append((process, local, partial))
        return process

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    for _ in range(2):
        archiver = module.RolloutArchiver(tmp_path / "run")
        archiver.mib_per_second = 0.5
        archiver.shutdown_seconds = 10
        # No queued windows: this exercises the actual final remainder sweep.
        with pytest.raises(OSError, match="stage-out incomplete"):
            archiver.close([root])
        assert not archiver._thread.is_alive()
        assert source.exists()
    assert len(tasks) == 2
    for process, local, partial in tasks:
        assert local.parent == root.parent  # outside the swept rollout root
        assert process.poll() is not None
        assert not local.exists() and not partial.exists()
    assert not list((tmp_path / "scratch").rglob(".rollout-archive-*"))
    assert not list((tmp_path / "run").rglob("*.partial"))


def test_many_files_produce_one_remote_archive(tmp_path, monkeypatch):
    source = _window(tmp_path / "scratch" / "rollouts")
    for index in range(500):
        (source / f"step_{index:04d}.json").write_text("rollout evidence " * 20)
    copy = module._copy_throttled
    copies = []

    def record_copy(local, remote, rate):
        assert source.exists()
        copies.append(remote)
        copy(local, remote, rate)

    monkeypatch.setattr(module, "_copy_throttled", record_copy)
    destination = tmp_path / "nfs" / "window.tar.gz"
    module.archive_and_publish(source, destination, 8.0)
    assert len(copies) == 1
    assert list(destination.parent.iterdir()) == [destination]
    with tarfile.open(destination) as archive:
        assert len(archive.getmembers()) == 502  # directory + 501 source files


def test_unreapable_writer_stops_dispatch_instead_of_starting_another(
    tmp_path, monkeypatch
):
    source = _window(tmp_path / "scratch" / "rollouts")
    launches = []
    temporaries = []

    class Unreapable:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("blocked NFS syscall", timeout)

        def kill(self):
            pass

    def launch(command, **kwargs):
        launches.append(command)
        local = Path(command[command.index("--local-archive") + 1])
        destination = Path(command[3])
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.partial")
        for path in (local, partial):
            path.write_bytes(b"writer still owns this file")
            temporaries.append(path)
        return Unreapable()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    archiver = module.RolloutArchiver(tmp_path / "run")
    archiver.submit(source)
    with pytest.raises(OSError, match="stage-out incomplete"):
        archiver.close([source.parent])
    assert len(launches) == 1
    assert source.exists()
    assert all(path.exists() for path in temporaries)


def test_background_diagnostics_do_not_take_training_file_handler_lock(tmp_path):
    class UnavailableNfsLog(logging.Handler):
        def emit(self, record):
            if record.name == module.__name__:
                raise AssertionError("background worker used the shared NFS logger")

    handler = UnavailableNfsLog()
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    source = _window(tmp_path / "scratch" / "rollouts")
    try:
        archiver = module.RolloutArchiver(tmp_path / "run")
        archiver.submit(source)
        archiver.close([source.parent])
        assert not source.exists()
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)


def test_failed_shutdown_still_stages_server_logs(tmp_path):
    class FailedArchiver:
        def close(self, roots):
            raise TimeoutError("NFS stalled")

    run = tmp_path / "run"
    source = _window(tmp_path / "scratch" / "rollouts")
    server_log = tmp_path / "scratch" / "server.log"
    server_log.write_text("engine diagnostics\n")
    with pytest.raises(TimeoutError, match="NFS stalled"):
        run_artifacts.stage_out_rollout_scratch(
            run_root=run,
            rollout_roots=[source.parent],
            server_log_paths=[server_log],
            cleanup_rollout_artifacts=False,
            archiver=FailedArchiver(),
        )
    assert source.exists()
    assert (run / "server.log").read_text() == "engine diagnostics\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CPU affinity/niceness")
def test_worker_imports_no_gpu_libraries_and_limits_its_own_cpu():
    code = (
        "import runpy, sys, os, json; "
        f"m = runpy.run_path({module.__file__!r}); "
        "m['_limit_worker_resources'](); "
        "print(json.dumps({'affinity': len(os.sched_getaffinity(0)), "
        "'nice': os.nice(0), 'torch': 'torch' in sys.modules, "
        "'numpy': 'numpy' in sys.modules}))"
    )
    parent_affinity = os.sched_getaffinity(0)
    parent_nice = os.nice(0)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {
        "affinity": 1,
        "nice": 19,
        "torch": False,
        "numpy": False,
    }
    assert os.sched_getaffinity(0) == parent_affinity
    assert os.nice(0) == parent_nice


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_transfer_limit_is_rejected_before_starting_worker(monkeypatch, value):
    monkeypatch.setenv("ARC3_ROLLOUT_ARCHIVE_MIB_PER_SECOND", value)
    with pytest.raises(ValueError, match="finite positive"):
        module.RolloutArchiver(Path("run"))
