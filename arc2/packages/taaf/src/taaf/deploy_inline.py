"""``InlineTarget`` — run a benchmark in the current process.

Graceful stop just leans on asyncio: cancelling the task
awaiting ``Benchmark.run`` (notebook cell interrupt, Ctrl-C under
``asyncio.run``, explicit ``task.cancel()``) propagates straight into
``Benchmark.run``'s teardown — no custom signal handler.

There is no per-run venv: an inline run is the launcher's own process
and environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import taaf.benchmark
import taaf.deploy

# Even though inline has no external kill, we still apply the same
# 10-minute buffer to ``max_runtime_s`` so solvers see consistent
# deadline semantics across targets.
_SOFT_DEADLINE_BUFFER_S = 600.0


@dataclass
class InlineHandle(taaf.deploy.DeploymentHandle):
    """Handle to an inline deployment. By construction ``deploy()``
    blocks until completion, so the handle is always in the done state
    by the time the caller receives it.

    Fields:

    - ``benchmark``: the populated ``Benchmark`` instance.
    """

    benchmark: taaf.benchmark.Benchmark
    _done: bool = False

    def wait(self) -> taaf.benchmark.Benchmark:
        """The benchmark; an inline run is already finished."""
        return self.benchmark

    def stop(self) -> None:
        """No-op: an inline run cancels through the task, not the handle."""
        # In-process cancellation is via task.cancel() during ``deploy``;
        # post-deploy there's nothing left to stop.

    @property
    def is_done(self) -> bool:
        """Whether the inline run finished."""
        return self._done

    @classmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> InlineHandle:
        """Reattach by loading ``benchmark.json``. Inline runs can't
        outlive their Python process, so an extant ``deploy_meta.json``
        always describes a finished run.
        """
        del meta
        json_path = job_dir / "benchmark.json"
        if not json_path.is_file():
            raise RuntimeError(
                f"no benchmark.json in {job_dir} — inline deploy() probably crashed before Benchmark.run completed."
            )
        bm = taaf.benchmark.Benchmark.from_json(json_path)
        return cls(job_dir=job_dir, benchmark=bm, _done=True)


@dataclass
class InlineTarget(taaf.deploy.DeploymentTarget):
    """Run the benchmark in the current process.

    Fields:

    - ``max_runtime_s``: self-imposed soft-deadline budget passed to
      ``Benchmark.run``. Inline has no external kill — the
      budget is purely defensive against misbehaving solvers.
    """

    max_runtime_s: float = 3600.0

    async def deploy(self, benchmark: taaf.benchmark.Benchmark) -> InlineHandle:
        """Run the benchmark in this process, with its own job directory,
        source snapshot and output capture."""
        if benchmark.job_dir is None:
            raise ValueError("InlineTarget requires benchmark.job_dir to be set.")
        job_dir = benchmark.job_dir
        taaf.deploy.check_job_dir_unused(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        taaf.deploy.snapshot_editable_sources(job_dir / "src")
        # Persist launcher-side git overview. Worker venvs install from
        # snapshots that exclude .git, so this file is the only durable
        # record of the real per-repo state at run start.
        taaf.deploy.write_git_status(job_dir)
        # Written before the run starts so even an early crash
        # leaves a reattachable record. job_id=None: inline has no
        # external scheduler.
        taaf.deploy.write_deploy_meta(
            job_dir=job_dir, target=self, handle_class=InlineHandle, benchmark=benchmark, job_id=None
        )

        # A blind 10-minute buffer on a short max_runtime_s would land
        # in the past and cancel the run before it started. Clamp to
        # at most half the budget.
        _budget = max(1.0, self.max_runtime_s)
        _buffer = min(_SOFT_DEADLINE_BUFFER_S, _budget / 2)
        soft_end_time = datetime.now() + timedelta(seconds=_budget - _buffer)

        handle = InlineHandle(job_dir=job_dir, benchmark=benchmark)

        with _tee_to_file(job_dir / "stdout.log"):
            print(taaf.deploy.format_preamble(benchmark))
            print(f"deploy.inline: job_dir       = {job_dir.absolute()}")
            print(f"deploy.inline: soft_end_time = {soft_end_time.isoformat()}")
            print("---")
            try:
                await benchmark.run(soft_end_time=soft_end_time, runtime_environment=self)
            except asyncio.CancelledError:
                # Benchmark.run re-raises after teardown when the awaiting
                # task was cancelled. Teardown already wrote artifacts;
                # swallow here so the caller sees a clean handle.
                print("deploy.inline: cancelled; teardown completed cleanly.")

        handle._done = True
        return handle


@contextlib.contextmanager
def _tee_to_file(log_path: Path):
    """Tee stdout + stderr to ``log_path`` while keeping the real
    streams — the log has to land in ``job_dir``; we keep the originals so the run
    remains interactive. Line-buffered so a crash leaves a readable
    tail. Truncate-on-open matches Slurm's ``#SBATCH --output``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


class _Tee:
    """Minimal write-only tee. Not a full TextIO; just what print() needs."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        """Write to every stream; returns the last stream's count."""
        n = 0
        for s in self._streams:
            n = s.write(data)
        return n

    def flush(self) -> None:
        """Flush every stream."""
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        """True if any underlying stream is a terminal."""
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)
