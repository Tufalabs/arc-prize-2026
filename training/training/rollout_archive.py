"""CPU-only, rate-limited stage-out of windows the trainer no longer reads.

The training thread only enqueues paths. A bounded dispatcher starts one
stdlib-only subprocess at a time, keeping tar/gzip and NFS I/O off its GIL.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

# The trainer's root logger includes an NFS FileHandler. Sharing that handler
# from this thread could hold its lock across a stalled write and block the
# next training log message. Use an independent stderr handler instead.
log = logging.Logger(__name__, level=logging.INFO)
log.addHandler(logging.StreamHandler())
_QUEUE_SIZE = 8
_TRANSFER_TIMEOUT_SECONDS = 600.0
_CHUNK_BYTES = 256 * 1024


def _positive_environment_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class RolloutArchiver:
    """Best-effort background saves; shutdown retries all remaining scratch.

    Queue pressure and storage failures never stall or fail an optimization
    iteration. A clean shutdown reports unsuccessful retention to its caller.
    """

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.mib_per_second = _positive_environment_float(
            "ARC3_ROLLOUT_ARCHIVE_MIB_PER_SECOND", 8.0
        )
        self.shutdown_seconds = _positive_environment_float(
            "ARC3_ROLLOUT_ARCHIVE_SHUTDOWN_SECONDS", 120.0
        )
        self._queue: queue.Queue[Path] = queue.Queue(maxsize=_QUEUE_SIZE)
        self._closing = threading.Event()
        self._cancel = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._roots: list[Path] = []
        self._failed = False
        self._overflow = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="rollout-archive", daemon=True
        )
        self._thread.start()

    def submit(self, window_dir: Path) -> bool:
        """Called only after all training/viewer reads of this window finish."""
        if self._closing.is_set():
            return False
        try:
            self._queue.put_nowait(window_dir)
        except queue.Full:
            # Even warning handlers may write to NFS; report from the dispatcher.
            self._overflow.set()
            return False
        return True

    def close(self, rollout_roots: list[Path]) -> None:
        """Drain after collectors stop, including skipped/failed/partial windows.

        No synchronous tree-copy fallback: it could race a timed-out writer
        and restore the original unbounded NFS metadata traffic.
        """
        self._roots = rollout_roots
        self._closing.set()
        self._thread.join(timeout=self.shutdown_seconds)
        if self._thread.is_alive():
            self._cancel.set()
            with self._process_lock:
                if self._process is not None:
                    self._process.kill()
            self._thread.join(timeout=2.0)
            raise TimeoutError(
                "rollout archive shutdown exceeded "
                f"{self.shutdown_seconds:g}s; unsaved data remains at {rollout_roots}"
            )
        if self._failed:
            raise OSError(
                f"rollout stage-out incomplete; inspect scratch at {rollout_roots}"
            )

    def _archive(self, source: Path, *, remainder: bool = False) -> bool:
        if self._cancel.is_set():
            return False
        if not source.exists():
            return True
        bucket = source.name if remainder else source.parent.name
        name = "remaining" if remainder else source.name
        destination = (
            self.run_root
            / "rollout_archives"
            / bucket
            / (f"{name}.{uuid.uuid4().hex}.tar.gz")
        )
        # Own both temporary names before starting the worker. SIGKILL skips
        # the child's finally block, so the dispatcher must be able to reclaim
        # them even when the local archive sits outside a remainder root.
        local_archive = source.parent / f".rollout-archive-{uuid.uuid4().hex}"
        partial = destination.with_name(f".{destination.name}.partial")
        # Launch a script directly so no training package/model initialization
        # can introduce CUDA, torch, or a native thread pool in the worker.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(source),
            str(destination),
            str(self.mib_per_second),
            "--local-archive",
            str(local_archive),
            "--parent-pid",
            str(os.getpid()),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        reaped = False
        returncode = None
        try:
            with self._process_lock:
                if self._cancel.is_set():
                    return False
                self._process = subprocess.Popen(command, env=environment)
                process = self._process
            try:
                returncode = process.wait(timeout=_TRANSFER_TIMEOUT_SECONDS)
                reaped = True
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                    reaped = True
                except subprocess.TimeoutExpired:
                    # An uninterruptible NFS syscall can outlive SIGKILL.
                    # Never launch another writer while this one still exists.
                    self._failed = True
                    self._cancel.set()
                    raise
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            log.info("archived retained rollout artifacts to %s", destination)
            return True
        except (OSError, subprocess.SubprocessError):
            log.exception("rollout archive failed; keeping scratch at %s", source)
            return False
        finally:
            if reaped and returncode != 0:
                # Keep NFS cleanup on the background dispatcher, never in
                # close() or submit(). Do not unlink under an unreaped writer.
                for temporary in (local_archive, partial):
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        log.warning(
                            "failed to reclaim archive temporary %s",
                            temporary,
                            exc_info=True,
                        )
            with self._process_lock:
                self._process = None

    def _run(self) -> None:
        try:
            while not self._cancel.is_set():
                if self._overflow.is_set():
                    self._overflow.clear()
                    log.warning(
                        "rollout archive queue full; overflow remains on scratch "
                        "for final stage-out"
                    )
                try:
                    source = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                if not self._archive(source) and not self._closing.is_set():
                    # One spaced retry in the background; the final sweep will
                    # retry any remaining data without an unbounded retry queue.
                    if not self._cancel.wait(5.0):
                        self._archive(source)
            if not self._cancel.is_set():
                for root in self._roots:
                    if root == self.run_root / root.name or not root.exists():
                        continue
                    # Successful windows have been removed. Save every residual
                    # file in one archive, including incomplete collector windows.
                    if not any(root.iterdir()):
                        root.rmdir()
                        continue
                    if not self._archive(root, remainder=True):
                        self._failed = True
        except Exception:
            self._failed = True
            log.exception("rollout archive dispatcher failed")


def _limit_worker_resources() -> None:
    """One low-priority CPU; these changes apply only to this subprocess."""
    if hasattr(os, "nice"):
        os.nice(19)
    if hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {max(available)})
    ionice = shutil.which("ionice")
    if ionice:
        subprocess.run(
            [ionice, "-c", "3", "-p", str(os.getpid())],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _exit_with_parent(parent_pid: int) -> None:
    """Do not leave a writer racing a resumed job if its trainer is killed."""
    if sys.platform == "linux":
        import ctypes
        import signal

        # PR_SET_PDEATHSIG is inherited neither across fork nor this exec;
        # install it here and check the race where the parent already exited.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "cannot set archive parent-death signal")
    if os.getppid() != parent_pid:
        raise SystemExit("archive owner exited before worker startup")


def _copy_throttled(source: Path, destination: Path, mib_per_second: float) -> None:
    rate = mib_per_second * 1024 * 1024
    next_write_at = time.monotonic()
    with source.open("rb") as reader, destination.open("xb", buffering=0) as writer:
        while chunk := reader.read(_CHUNK_BYTES):
            now = time.monotonic()
            # Do not accumulate bandwidth credit while NFS stalls: catching up
            # afterward could send a large burst into an already busy server.
            next_write_at = max(next_write_at, now) + len(chunk) / rate
            delay = next_write_at - now
            if delay > 0:
                time.sleep(delay)
            remaining = memoryview(chunk)
            while remaining:
                written = writer.write(remaining)
                if not written:
                    raise OSError("archive write made no progress")
                remaining = remaining[written:]
        os.fsync(writer.fileno())


def archive_and_publish(
    source: Path,
    destination: Path,
    mib_per_second: float,
    *,
    local_archive: Path | None = None,
) -> None:
    """Pack on scratch, publish a durable archive, then remove its source."""
    if local_archive is None:
        local_archive = source.parent / f".rollout-archive-{uuid.uuid4().hex}"
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        descriptor = os.open(local_archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as archive_file:
            with tarfile.open(
                fileobj=archive_file, mode="w:gz", compresslevel=1
            ) as archive:
                archive.add(source, arcname=source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_throttled(local_archive, partial, mib_per_second)
        partial.replace(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        shutil.rmtree(source)
    finally:
        local_archive.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mib_per_second", type=float)
    parser.add_argument("--local-archive", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    _exit_with_parent(args.parent_pid)
    _limit_worker_resources()
    archive_and_publish(
        args.source,
        args.destination,
        args.mib_per_second,
        local_archive=args.local_archive,
    )


if __name__ == "__main__":
    main()
