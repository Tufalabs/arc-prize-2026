"""``TufaSlurmTarget`` — deploy a benchmark as a Slurm job on the Tufa
cluster (R2.23 + R2.37 + R2.38 + cluster slice of R2.31–R2.36/R2.39).

Lifecycle:

1. ``deploy()`` snapshots sources (R2.35), pickles the benchmark to
   ``job_dir/benchmark_initial.pkl``, renders ``run_in_worker.py`` and
   the sbatch script, submits via ``sbatch --parsable``, returns
   immediately with a ``TufaSlurmHandle``.
2. The sbatch script creates a per-run ``.venv`` from the bundled
   sources (R2.37), then runs ``run_in_worker.py`` (in the optional
   container, R2.38).
3. The worker derives ``soft_end_time`` from ``SLURM_JOB_END_TIME`` and
   ``max_runtime_s`` (R2.32) and awaits ``Benchmark.run``.
4. ``handle.wait()`` polls ``squeue``; once the job leaves the queue,
   loads ``benchmark.json``. ``handle.stop()`` is
   ``scancel --signal=USR1 --batch`` (R2.33). ``_attach()`` rebuilds a
   handle from the ``job_id`` in ``deploy_meta.json``.
"""

from __future__ import annotations

import importlib
import secrets
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import taaf.benchmark
import taaf.deploy
import taaf.support

GPU = Literal["B200", "B300", "CPU"]

# Hardware tier → ``--gres=gpu:<name>:<count>``. B200 (``gpu:b200:8`` on
# dgx[01-06]) is verified from ``sinfo``; B300 follows the naming convention forward-compat. ``CPU`` →
# no ``--gres`` and switches the rendering to ``--cpus-per-task``/``--mem``.
_GPU_GRES_NAMES: dict[str, str | None] = {
    "B200": "b200",
    "B300": "b300",
    "CPU": None,
}

# Under the job dir: the official games' code the worker installs (``_stage_official_games``).
_OFFICIAL_GAMES_DIR = "official_games"

_DEFAULT_POLL_INTERVAL_S = 30.0
# How long a deployed step may take to be listed by squeue before the
# launcher gives up on it.
_STEP_REGISTER_TIMEOUT_S = 60.0
_STEP_REGISTER_POLL_S = 1.0
# How long a launcher gets to act on SIGTERM before SIGKILL.
_STEP_LAUNCHER_EXIT_TIMEOUT_S = 10.0
_SLURM_TEARDOWN_BUFFER = timedelta(minutes=10)


@dataclass
class TufaSlurmHandle(taaf.deploy.DeploymentHandle):
    """Handle to a Slurm job. ``deploy()`` returns one of these *before*
    the job has started running.

    Fields:

    - ``job_id``: slurm job id from ``sbatch --parsable``.
    - ``poll_interval_s``: how often ``wait()`` polls ``squeue``.
    """

    # Empty default so the ABC's job_dir (no default) stays first.
    job_id: str = ""
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S
    #: Set when the run is a step of an existing allocation
    #: (``TufaSlurmTarget.jobid``): ``job_id`` is then the allocation's.
    #: The name is unique per deployment and only finds the step before
    #: :attr:`step_id` is known.
    step_name: str = ""
    #: The step's own id (``<job>.<step>``), captured once it registers.
    step_id: str = ""
    #: False for a step the deployment only prepared
    #: (``TufaSlurmTarget.launch_step``): until the allocation starts it, its
    #: absence from the queue means it has yet to run, not that it ended.
    step_launched: bool = True
    _cached_benchmark: taaf.benchmark.Benchmark | None = field(default=None, repr=False)

    def wait(self) -> taaf.benchmark.Benchmark:
        """Block until the job leaves the queue, then load
        ``benchmark.pkl``."""
        while not self.is_done:
            time.sleep(self.poll_interval_s)
        return self._load_benchmark()

    def stop(self) -> None:
        """Request graceful shutdown via ``scancel --signal=USR1 --batch``
        (R2.33). The worker's asyncio handler cancels the solver task,
        routing through ``Benchmark.run``'s teardown.

        ``--batch`` signals the batch step and nothing else, which —
        because the work block ends in ``exec python -u`` (see
        :func:`_render_sbatch_script`) — is exactly the one process
        holding the handler.

        Emphatically **not** ``--full``: that also signals every child of
        the batch step, and SIGUSR1's default disposition is Terminate.
        A solver's subprocesses (per-game workers, an inference server and
        its engine) install no handler, so they die on the spot and the
        graceful teardown then runs against a job that has already lost
        its parts. Bare ``scancel --signal`` is the other wrong answer: it
        skips the batch step entirely, so the non-image path — where the
        work runs directly in the batch shell — never sees the signal.

        Idempotent. For a hard kill bypassing teardown, use bare
        ``scancel <job_id>`` (SIGTERM, which the worker doesn't handle).

        KNOWN GAP — container deployments (``image`` set) are not covered.
        There the batch step is a shell running ``srun``, not the worker,
        so ``--batch`` signals the shell (default action: Terminate) and
        the container's python never hears about it.

        That gap is not something ``--batch`` took away. Under ``--full``
        the signal did reach the container's python — it is a child of the
        batch step — but ``--full`` signals every other child too, and on
        the non-image path that was observed to kill the ULLM inference
        server outright, leaving teardown to run against a job with no
        backend. The container path is signalled by the same rule, so it
        inherits the same breakage. No path was ever graceful; what
        changed is the failure mode, from "signal arrives, everything
        underneath it is dead" to "signal never arrives".

        Covering containers means trapping USR1 in the *outer* batch shell
        and forwarding it to ``srun``, which requires ``srun`` to be
        backgrounded rather than run in the foreground. That is a
        different shell from the ``exec python -u`` inside the container
        work block, which would stay as it is. The cost is not the line
        count — it is that ``srun``'s forwarding through the container
        plugin cannot be verified off the cluster.
        """
        if self.step_name:
            # A step of someone's allocation: signal the step alone. Its
            # task is the ``exec``'d python, so no ``--batch`` is needed and
            # the allocation itself is never touched.
            step_id = self._step_id()
            if step_id is not None:
                subprocess.run(["scancel", "--signal=USR1", step_id], check=False, capture_output=True)
            return
        subprocess.run(
            ["scancel", "--signal=USR1", "--batch", self.job_id],
            check=False,
            capture_output=True,
        )

    def _step_id(self) -> str | None:
        """The id of this handle's step, or None once it has left the queue.

        Matched on :attr:`step_id` as soon as ``_start_step`` has captured
        it, and only before that on the name — which is why the name a
        deployment asks for is unique. Two deployments whose ``job_dir``
        share a basename would otherwise both resolve to whichever
        registered first, and the loser's ``stop()`` would signal it.
        """
        result = subprocess.run(
            ["squeue", "-s", "-j", self.job_id, "--noheader", "-o", "%i %j"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            step_id, _, name = line.strip().partition(" ")
            matches = step_id == self.step_id if self.step_id else name == self.step_name
            if matches:
                return step_id
        return None

    @property
    def is_done(self) -> bool:
        """True iff ``squeue`` no longer lists the job. We deliberately
        don't read sacct — the final state is already recoverable from
        ``benchmark.json`` + the stdout log, and sacct has reporting lag.
        """
        if self.step_name:
            if self._step_id() is not None:
                return False
            # A prepared step has ended once it ran -- srun opened its
            # stdout -- or once its allocation is gone without starting it.
            if not self.step_launched and not (self.job_dir / "stdout.log").exists():
                return not self._allocation_queued()
            return True
        return not self._allocation_queued()

    def _allocation_queued(self) -> bool:
        """Whether ``squeue`` still lists the job (or allocation) itself."""
        result = subprocess.run(
            ["squeue", "-j", self.job_id, "--noheader", "-o", "%T"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() != ""

    @classmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> TufaSlurmHandle:
        job_id = meta.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"deploy_meta.json in {job_dir} has no job_id; cannot reattach to Slurm job.")
        return cls(
            job_dir=job_dir,
            job_id=job_id,
            step_name=str(meta.get("step_name") or ""),
            step_id=str(meta.get("step_id") or ""),
            step_launched=bool(meta.get("step_launched", True)),
        )

    def _load_benchmark(self) -> taaf.benchmark.Benchmark:
        if self._cached_benchmark is not None:
            return self._cached_benchmark
        json_path = self.job_dir / "benchmark.json"
        if not json_path.is_file():
            raise RuntimeError(
                f"Slurm job {self.job_id} no longer in queue, but no benchmark.json in {self.job_dir}. "
                "The worker may have crashed before Benchmark.run's teardown wrote it; "
                "check stdout.log / stderr.log in the job dir."
            )
        self._cached_benchmark = taaf.benchmark.Benchmark.from_json(json_path)
        return self._cached_benchmark


@dataclass
class TufaSlurmTarget(taaf.deploy.DeploymentTarget):
    """Deploy as a Slurm job on the Tufa cluster (R2.23).

    Fields:

    - ``gpu``: hardware tier — ``B200`` / ``B300`` or ``CPU``
      (no GPU, useful for fast smoke tests). Mapped via the
      module-level ``_GPU_GRES_NAMES`` table.
    - ``time``: sbatch walltime, e.g. ``"04:00:00"``. Hard upper bound;
      the worker derives ``soft_end_time = end − 10min`` (R2.32).
    - ``gpu_count``: number of GPUs (must be 0 when ``gpu == "CPU"``).
    - ``max_runtime_s``: optional experiment budget; the worker stops
      at the earlier of this and the walltime-derived soft deadline.
    - ``cpus_per_gpu`` / ``mem_per_gpu_bytes`` (default 8 + 32 GiB) or
      ``total_cpus`` / ``total_mem_bytes``: pick one coordinate system.
      The per-gpu form renders ``--cpus-per-gpu`` / ``--mem-per-gpu``
      directly; totals are divided by ``gpu_count`` (ceiling) before
      rendering. Mixing them raises at deploy time. The CPU branch
      accepts per-gpu as a convenience.
    - ``image`` (R2.38): docker URI or ``.sqsh`` path. ``None`` ⇒ bare
      cluster python.
    - ``partition`` / ``nodelist``: passed through to sbatch verbatim.
    - ``extra_mounts``: extra ``host:container`` bind mounts (only used
      with ``image``). ``job_dir`` is always mounted.
    - ``extra_sbatch_flags``: free-form additional sbatch directives.
    - ``extra_source_repos``: additional local repos copied into
      ``job_dir/src`` alongside auto-discovered editables.
    - ``main_project``: name of the bundled repo the worker
      ``uv sync``s from. Required — there's exactly one right answer
      per downstream project and ``uv sync`` is the only command that
      honors ``[tool.uv.sources]`` path rewrites.
    - ``node_local_storage``: copy bundled sources, worker venvs, temp
      files, and rebuildable caches into Slurm node-local storage before
      ``uv sync``. This is intended for clusters such as Kander where
      shared filesystems are durable but too slow for venv creation.
    - ``uv_override_skip_packages``: package names passed as
      ``--no-install-package`` during ``uv sync`` when ``UV_OVERRIDE`` is
      set, allowing downstream launchers to install local editables from
      the override file after the locked sync.
    """

    gpu: GPU = "B200"
    time: str = "04:00:00"
    gpu_count: int = field(default=1, kw_only=True)
    max_runtime_s: float | None = field(default=None, kw_only=True)
    cpus_per_gpu: int | None = field(default=None, kw_only=True)
    mem_per_gpu_bytes: int | None = field(default=None, kw_only=True)
    total_cpus: int | None = field(default=None, kw_only=True)
    total_mem_bytes: int | None = field(default=None, kw_only=True)
    image: str | None = None
    partition: str | None = "dev"
    account: str | None = "research"
    qos: str | None = None
    nodelist: str | None = None
    extra_mounts: list[tuple[str, str]] = field(default_factory=lambda: list[tuple[str, str]]())
    extra_sbatch_flags: list[str] = field(default_factory=lambda: list[str]())
    extra_source_repos: list[Path] = field(default_factory=lambda: list[Path](), kw_only=True)
    main_project: str = field(kw_only=True)
    node_local_storage: bool = field(default=True, kw_only=True)
    uv_override_skip_packages: list[str] = field(default_factory=lambda: list[str](), kw_only=True)
    #: An allocation to run inside, by job id: the same script runs as an
    #: overlapping step of it rather than as a job of its own, so a run
    #: starts when it is deployed instead of when the queue reaches it. The
    #: allocation's resources and walltime are what the run gets; the
    #: directives above are rendered and ignored. Each run keys its
    #: node-local sources and venv by its step, so deploys never share them.
    jobid: str | None = field(default=None, kw_only=True)
    #: With ``jobid``, whether ``deploy`` starts the step itself. Off, it
    #: prepares the run and records the step it is to be, and a script of
    #: the allocation starts it with :func:`step_argv` under the recorded
    #: name -- one that splits the allocation's GPUs between runs it starts
    #: together, where a deployed step overlaps the whole allocation. The
    #: allocation may still be queued: its id is all a prepared run needs.
    launch_step: bool = field(default=True, kw_only=True)

    async def deploy(self, benchmark: taaf.benchmark.Benchmark) -> TufaSlurmHandle:
        if benchmark.job_dir is None:
            raise ValueError("TufaSlurmTarget requires benchmark.job_dir to be set (R2.34).")
        job_dir = benchmark.job_dir.resolve()
        taaf.deploy.check_job_dir_unused(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write the snapshot as a single tar instead of ~1k small files:
        # on NFS shared storage the per-file create latency makes a directory
        # snapshot cost minutes on the submit path, while one sequential tar
        # write is sub-second. The worker extracts it before ``uv sync`` (see
        # the install blocks below).
        # The worker ``uv sync``s from the bundled main_project's lock, so an
        # installed-package bundle is never read here, and the vLLM fork alone
        # would put hundreds of MB on every job's NFS write.
        taaf.deploy.snapshot_editable_sources_archive(
            job_dir / "src.tar", extra_repos=self.extra_source_repos, include_noneditable_tufa=False
        )
        _stage_official_games(job_dir)
        # Persist launcher-side git overview. The bundled snapshot
        # excludes .git, so this file is the only durable record of
        # the real per-repo state at run start.
        taaf.deploy.write_git_status(job_dir)

        # Pre-run benchmark; teardown writes the post-run JSON.
        taaf.support.atomic_pickle_dump(benchmark, job_dir / "benchmark_initial.pkl")
        # Pickle the target so the worker hands it back as
        # ``runtime_environment=`` (R12.01) — solvers introspect the
        # object instead of parsing a tag.
        taaf.support.atomic_pickle_dump(self, job_dir / "deploy_target.pkl")

        # R2.31.2 — capture preamble on the launcher. The bundled
        # snapshot isn't a git repo so the worker can't produce it.
        preamble = taaf.deploy.format_preamble(benchmark)
        (job_dir / "run_in_worker.py").write_text(_render_worker_script(job_dir, preamble))
        sbatch_script = _render_sbatch_script(self, benchmark, job_dir)
        (job_dir / "sbatch_script.sh").write_text(sbatch_script)
        if self.jobid is not None:
            return self._start_step(benchmark, job_dir) if self.launch_step else self._prepare_step(benchmark, job_dir)

        # ``--parsable`` makes sbatch print just ``<job_id>`` (or
        # ``<job_id>;<cluster>``); strip any trailing cluster qualifier.
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=sbatch_script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            # Surface sbatch's stderr in the message — ``check=True``
            # would hide it inside ``e.stderr`` where it usually goes
            # unread.
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}). "
                f"stderr:\n{result.stderr.strip()}\n"
                f"Rendered script saved at {job_dir / 'sbatch_script.sh'}."
            )
        job_id = result.stdout.strip().split(";")[0]
        if not job_id:
            raise RuntimeError(f"sbatch returned no job id; stdout={result.stdout!r} stderr={result.stderr!r}")

        taaf.deploy.write_deploy_meta(
            job_dir=job_dir,
            target=self,
            handle_class=TufaSlurmHandle,
            benchmark=benchmark,
            job_id=job_id,
        )

        return TufaSlurmHandle(job_dir=job_dir, job_id=job_id)

    def _start_step(self, benchmark: taaf.benchmark.Benchmark, job_dir: Path) -> TufaSlurmHandle:
        """Run the rendered script as a step of ``self.jobid``.

        ``--overlap`` because an interactive allocation's own shell is a
        step holding every resource; the run shares them. ``srun`` writes
        the step's stdio to the same files the batch directives name, and
        keeps its own messages in ``srun.log``. The launcher waits for the
        step to register before returning: a handle whose step is not yet
        listed would read as finished.

        The step is asked for under a name no other deployment will use,
        and the id it registers under is captured and persisted — the
        queue listing is the launcher's only handle on the step, so a
        name shared with a concurrent deployment would hand ``stop()``
        the wrong run.
        """
        step_name = _step_name(job_dir)
        argv = step_argv(job_dir, step_name, [f"--jobid={self.jobid}", "--overlap"])
        srun_log = open(job_dir / "srun.log", "wb")  # noqa: SIM115 -- the step outlives this call
        launcher = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=srun_log, stderr=srun_log, start_new_session=True
        )
        handle = TufaSlurmHandle(job_dir=job_dir, job_id=str(self.jobid), step_name=step_name)
        deadline = time.monotonic() + _STEP_REGISTER_TIMEOUT_S
        while (step_id := handle._step_id()) is None:
            if time.monotonic() > deadline:
                # The launch is still pending, and there is no
                # deploy_meta.json to reattach to it: left alone it would
                # start the benchmark after the caller was told the deploy
                # failed, and a retry into this job_dir would put a second
                # run on the first one's artifacts.
                _cancel_launcher(launcher)
                srun_log.close()
                raise RuntimeError(
                    f"step {step_name!r} did not appear in allocation {self.jobid} within "
                    f"{_STEP_REGISTER_TIMEOUT_S:.0f}s; see {job_dir / 'srun.log'}."
                )
            time.sleep(_STEP_REGISTER_POLL_S)
        handle.step_id = step_id
        taaf.deploy.write_deploy_meta(
            job_dir=job_dir,
            target=self,
            handle_class=TufaSlurmHandle,
            benchmark=benchmark,
            job_id=str(self.jobid),
            step_name=step_name,
            step_id=step_id,
        )
        return handle

    def _prepare_step(self, benchmark: taaf.benchmark.Benchmark, job_dir: Path) -> TufaSlurmHandle:
        """Record the run as a step of ``self.jobid`` for the allocation to
        start (``launch_step`` off): the name it is to run under, unique as
        ``_start_step``'s is, and that it has not been launched."""
        step_name = _step_name(job_dir)
        taaf.deploy.write_deploy_meta(
            job_dir=job_dir,
            target=self,
            handle_class=TufaSlurmHandle,
            benchmark=benchmark,
            job_id=str(self.jobid),
            step_name=step_name,
            step_launched=False,
        )
        return TufaSlurmHandle(job_dir=job_dir, job_id=str(self.jobid), step_name=step_name, step_launched=False)


def _step_name(job_dir: Path) -> str:
    """A step name no other deployment will use: ``job_dir``'s, which two
    runs under different parents can share, and a random suffix."""
    return f"{job_dir.name}-{secrets.token_hex(4)}"


def step_argv(job_dir: Path, step_name: str, flags: list[str] | tuple[str, ...] = ()) -> list[str]:
    """``srun`` running a deployed run's script as the step ``step_name``,
    with its stdio in the files the batch directives would have named;
    ``flags`` place the step -- the allocation it joins, its share of it."""
    return [
        "srun",
        *flags,
        "--ntasks=1",
        f"--job-name={step_name}",
        f"--output={job_dir / 'stdout.log'}",
        f"--error={job_dir / 'stderr.log'}",
        "bash",
        str(job_dir / "sbatch_script.sh"),
    ]


def _cancel_launcher(launcher: subprocess.Popen[bytes]) -> None:
    """Terminate a still-pending ``srun`` and reap it.

    SIGTERM asks ``srun`` to cancel the step it is waiting to create;
    SIGKILL is the fallback for one that will not go, and the ``wait``
    keeps the caller from raising while a launch is still in flight.
    """
    if launcher.poll() is not None:
        return
    launcher.terminate()
    try:
        launcher.wait(timeout=_STEP_LAUNCHER_EXIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        launcher.kill()
        launcher.wait()


# --- Script rendering -----------------------------------------------------


def _render_worker_script(job_dir: Path, preamble: str) -> str:
    """The Python entry-point ``sbatch`` invokes.

    ``preamble`` is captured on the launcher (where the real git
    checkout lives) and embedded as a literal so the worker can print
    it before the run — the bundled source snapshot isn't a git repo.

    Soft deadline is derived from ``SLURM_JOB_END_TIME`` rather than
    deploy()'s wall clock so queue waits don't eat into it.

    The body sits inside ``if __name__ == "__main__":`` so spawned
    subprocesses (``multiprocessing.get_context('spawn')`` — vllm
    engine-core, BFS controllers) don't re-run the benchmark when
    Python's spawn machinery rebuilds the child's ``__main__`` via
    ``runpy.run_path()``.
    """
    return f'''"""TAAF worker entry — generated by TufaSlurmTarget.deploy(). Do not edit."""

import asyncio
import os
import pickle
import signal
from pathlib import Path

import taaf.benchmark  # noqa: F401
import taaf.deploy_slurm

JOB_DIR = Path({str(job_dir)!r})


async def _amain(bm, soft_end, target):
    # R2.33: route SIGUSR1 (from scancel --signal=USR1) into
    # Benchmark.run's teardown by cancelling the live solver task.
    # SIGTERM (default scancel) stays as a hard kill via the default
    # python handler.
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, bm.request_stop)
    try:
        await bm.run(soft_end_time=soft_end, runtime_environment=target)
    except asyncio.CancelledError:
        # Teardown already wrote artifacts; swallow so the worker
        # exits 0 and sacct shows COMPLETED rather than FAILED for a
        # graceful stop.
        pass


if __name__ == "__main__":
    with open(JOB_DIR / "deploy_target.pkl", "rb") as f:
        target = pickle.load(f)

    SOFT_END = taaf.deploy_slurm.worker_soft_end_time(
        slurm_end_epoch=os.environ.get("SLURM_JOB_END_TIME"),
        max_runtime_s=getattr(target, "max_runtime_s", None),
    )

    print({preamble!r})
    print(f"deploy.slurm: job_dir       = {{JOB_DIR}}")
    print(f"deploy.slurm: SLURM_JOB_ID  = {{os.environ.get('SLURM_JOB_ID', '?')}}")
    print(f"deploy.slurm: soft_end_time = {{SOFT_END}}")
    print("---")

    with open(JOB_DIR / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = JOB_DIR
    asyncio.run(_amain(bm, SOFT_END, target))

    # Bypass interpreter shutdown's threadpool-join: any stuck
    # background thread (vLLM NCCL, C-level recv, ...) would hang the
    # slurm job in RUNNING long after the run finished. Benchmark.run's
    # finally already wrote everything; os._exit lets the OS reclaim.
    os._exit(0)
'''


def worker_soft_end_time(
    *,
    slurm_end_epoch: str | None,
    max_runtime_s: float | None,
    now: datetime | None = None,
) -> datetime | None:
    now = now or datetime.now()

    experiment_deadline = (
        now + timedelta(seconds=float(max_runtime_s)) if max_runtime_s is not None and max_runtime_s > 0 else None
    )

    slurm_deadline = None
    if slurm_end_epoch:
        slurm_end = datetime.fromtimestamp(int(slurm_end_epoch))
        remaining = slurm_end - now
        buffer = min(_SLURM_TEARDOWN_BUFFER, remaining / 2)
        slurm_deadline = slurm_end - buffer

    deadlines = [deadline for deadline in (experiment_deadline, slurm_deadline) if deadline is not None]
    return min(deadlines, default=None)


def _resolve_resources(target: TufaSlurmTarget) -> tuple[int, int]:
    """Resolve target's CPU/memory fields into sbatch values, returning
    ``(cpus, mem_mib)``. GPU jobs get the per-gpu pair; CPU jobs get
    the totals. Raises ``ValueError`` on mixed / partial coordinate
    systems. Called at deploy time (not construction) so callers can
    mutate the target between ``__init__`` and ``deploy()``.
    """
    per_gpu_set = (target.cpus_per_gpu is not None) or (target.mem_per_gpu_bytes is not None)
    totals_set = (target.total_cpus is not None) or (target.total_mem_bytes is not None)
    if per_gpu_set and totals_set:
        raise ValueError(
            "TufaSlurmTarget: cannot mix per-gpu (cpus_per_gpu / mem_per_gpu_bytes) "
            "with totals (total_cpus / total_mem_bytes) — pick one coordinate system."
        )
    if per_gpu_set and (target.cpus_per_gpu is None or target.mem_per_gpu_bytes is None):
        raise ValueError(
            f"TufaSlurmTarget: cpus_per_gpu and mem_per_gpu_bytes must be set together "
            f"(got cpus_per_gpu={target.cpus_per_gpu!r}, mem_per_gpu_bytes={target.mem_per_gpu_bytes!r})."
        )
    if totals_set and (target.total_cpus is None or target.total_mem_bytes is None):
        raise ValueError(
            f"TufaSlurmTarget: total_cpus and total_mem_bytes must be set together "
            f"(got total_cpus={target.total_cpus!r}, total_mem_bytes={target.total_mem_bytes!r})."
        )

    # Normalize to a per-gpu pair. Defaults (8 CPUs + 32 GiB) match the
    # single-GPU case and scale to multi-GPU. ``None`` here means "use
    # totals instead" — handled below.
    per_gpu_cpus: int | None
    per_gpu_mem_bytes: int | None
    if per_gpu_set:
        assert target.cpus_per_gpu is not None and target.mem_per_gpu_bytes is not None
        per_gpu_cpus = target.cpus_per_gpu
        per_gpu_mem_bytes = target.mem_per_gpu_bytes
    elif not totals_set:
        per_gpu_cpus = 8
        per_gpu_mem_bytes = 32 * 1024**3
    else:
        per_gpu_cpus = None
        per_gpu_mem_bytes = None

    if target.gpu == "CPU":
        # Accept per-gpu as totals here so callers don't have to switch
        # coordinate systems just to smoke-test on CPU.
        if per_gpu_cpus is not None and per_gpu_mem_bytes is not None:
            cpus_total = per_gpu_cpus
            mem_bytes_total = per_gpu_mem_bytes
        else:
            assert target.total_cpus is not None and target.total_mem_bytes is not None
            cpus_total = target.total_cpus
            mem_bytes_total = target.total_mem_bytes
        return cpus_total, max(1, mem_bytes_total // (1024 * 1024))

    # GPU branch.
    if target.gpu_count < 1:
        raise ValueError("gpu_count must be at least 1 for GPU Slurm jobs.")
    if per_gpu_cpus is not None and per_gpu_mem_bytes is not None:
        return per_gpu_cpus, max(1, per_gpu_mem_bytes // (1024 * 1024))
    # Totals → divide by gpu_count (ceiling) so the allocation ≥ ask.
    assert target.total_cpus is not None and target.total_mem_bytes is not None
    cpus_pg = -(-target.total_cpus // target.gpu_count)
    mem_pg_mib = max(1, -(-target.total_mem_bytes // (target.gpu_count * 1024 * 1024)))
    return cpus_pg, mem_pg_mib


def _render_sbatch_script(
    target: TufaSlurmTarget,
    benchmark: taaf.benchmark.Benchmark,
    job_dir: Path,
) -> str:
    """Build the sbatch script that runs the benchmark on a worker node.

    Layout:
    1. ``#SBATCH`` directives (walltime / CPU / mem / partition / nodelist
       / gres — all from the target).
    2. A bash work block that creates the per-run ``.venv`` (R2.37),
       installs the bundled tufalabs repos editable, then invokes
       ``run_in_worker.py``.
    3. The work block is wrapped in ``srun --container-image=... bash -c
       '...'`` when ``image`` is set (R2.38), otherwise run directly on
       the allocation.
    """
    sb: list[str] = ["#!/bin/bash"]
    sb.append(f"#SBATCH --job-name={shlex.quote(benchmark.label or 'taaf')}")
    sb.append(f"#SBATCH --time={target.time}")
    sb.append(f"#SBATCH --output={job_dir}/stdout.log")
    sb.append(f"#SBATCH --error={job_dir}/stderr.log")
    # The Tufa cluster's submit plugin requires --ntasks=1 and the
    # per-gpu CPU/memory forms for GPU jobs (it rejects
    # --cpus-per-task / --mem); CPU-only jobs use the standard forms.
    sb.append("#SBATCH --ntasks=1")
    cpus_val, mem_mib = _resolve_resources(target)
    if target.gpu == "CPU":
        if target.gpu_count != 0:
            raise ValueError("gpu_count must be 0 for CPU Slurm jobs.")
        sb.append(f"#SBATCH --cpus-per-task={cpus_val}")
        sb.append(f"#SBATCH --mem={mem_mib}M")
    else:
        sb.append(f"#SBATCH --cpus-per-gpu={cpus_val}")
        sb.append(f"#SBATCH --mem-per-gpu={mem_mib}M")
    if target.partition:
        sb.append(f"#SBATCH --partition={target.partition}")
    if target.account:
        sb.append(f"#SBATCH --account={target.account}")
    if target.qos:
        sb.append(f"#SBATCH --qos={target.qos}")
    if target.nodelist:
        sb.append(f"#SBATCH --nodelist={target.nodelist}")
    gres_name = _GPU_GRES_NAMES[target.gpu]
    if gres_name is not None:
        sb.append(f"#SBATCH --gres=gpu:{gres_name}:{target.gpu_count}")
    for flag in target.extra_sbatch_flags:
        sb.append(f"#SBATCH {flag}")

    install_block = _render_install_block(target, job_dir)

    work = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(job_dir))}",
            "",
            "# Slurm propagates the submitter's env; a launcher in",
            "# Jupyter / VS Code ships",
            "# MPLBACKEND=module://matplotlib_inline.backend_inline into",
            "# the worker, which breaks matplotlib import. Pin to Agg.",
            "export MPLBACKEND=Agg",
            "",
            "# Iterating a set of strings is ordered by their hashes, which",
            "# CPython randomises per process. Any such order that reaches an",
            "# agent's prompt makes a run unreproducible -- including by",
            "# itself, since a child gets its own seed. Pinned here rather",
            "# than in Python: the seed is read at interpreter startup, so a",
            "# process cannot set its own, only its children's.",
            "export PYTHONHASHSEED=0",
            "",
            "# Slurm propagates only the PATH the launcher had. If that",
            "# was a non-login shell, ~/.local/bin is missing here.",
            'export PATH="$HOME/.local/bin:$PATH"',
            "",
            *_render_uv_bootstrap_block(target),
            "",
            "# R2.37: per-run venv from the bundled sources.",
            *install_block,
            "",
            *_render_uv_binary_sanitize_block(),
            "",
            "# ``-u`` for unbuffered stdio (otherwise the solver loop",
            "# looks wedged from outside).",
            "#",
            "# ``exec`` is critical for R2.33 graceful stop: without it",
            "# the shell stays as python's parent and ``scancel",
            "# --signal=USR1 --batch`` delivers SIGUSR1 to the shell",
            "# (default action: Terminate). Shell exits, python gets",
            "# reparented to slurmstepd and never sees the signal. With",
            "# exec, python takes over the shell's PID, so the batch step",
            "# slurm signals IS the process holding the handler -- which",
            "# is why --batch can stay narrow and leave children alone.",
            "exec python -u run_in_worker.py",
        ]
    )

    sb.append("")
    if target.image:
        mounts = [f"{job_dir}:{job_dir}"] + [f"{s}:{d}" for s, d in target.extra_mounts]
        # Under ``jobid`` the outer step already holds the allocation's
        # resources, and a step's ``--overlap`` does not carry to the steps
        # it starts: without its own, this one waits for resources that
        # will not free up until the run it is supposed to be has ended.
        overlap = "--overlap " if target.jobid is not None else ""
        sb.append(
            "srun "
            f"{overlap}"
            f"--container-image={shlex.quote(target.image)} "
            f"--container-mounts={','.join(shlex.quote(m) for m in mounts)} "
            f"--container-workdir={shlex.quote(str(job_dir))} "
            f"bash -c {shlex.quote(work)}"
        )
    else:
        sb.append(work)

    return "\n".join(sb) + "\n"


def _render_uv_bootstrap_block(target: TufaSlurmTarget) -> list[str]:
    # Bootstrap uv inside a pyxis container (R2.38) where the host's
    # ~/.local/bin isn't mounted and the base image (e.g.
    # nvcr.io/nvidia/pytorch) doesn't ship uv. Compute nodes have outbound
    # network, so curl-install uv when it's missing rather than resolving a
    # pre-staged copy.
    del target  # signature kept for call-site symmetry; no per-target config.
    return [
        "if ! command -v uv >/dev/null 2>&1; then",
        "    curl -LsSf https://astral.sh/uv/install.sh | sh",
        '    export PATH="$HOME/.local/bin:$PATH"',
        "fi",
    ]


def _render_uv_binary_sanitize_block() -> list[str]:
    # ``--export=ALL`` and ``/etc/kander/job-env.sh`` both hand the worker a
    # ``UV`` naming an absolute host path (e.g. ``/opt/tufa/uv/<ver>/bin/uv``).
    # Under R2.38 that path isn't mounted in the image, and anything the
    # solver shells out to that resolves uv as ``UV ?= uv`` inherits the dead
    # binary. Bare ``uv`` on PATH is always live here — the bootstrap block
    # guarantees it.
    return [
        'if [ -n "${UV:-}" ] && ! command -v "${UV}" >/dev/null 2>&1; then',
        "    unset UV",
        "fi",
    ]


def _render_uv_sync_args_block(target: TufaSlurmTarget, *, indent: str = "    ") -> list[str]:
    lines = [f"{indent}uv_sync_args=()"]
    # A uv-workspace main_project needs --all-packages: plain `uv sync`
    # installs only the root project, and a virtual workspace root would
    # install nothing. Detected on the worker from the bundled pyproject.
    lines.append(f"{indent}if grep -qF '[tool.uv.workspace]' pyproject.toml 2>/dev/null; then")
    lines.append(f"{indent}    uv_sync_args+=(--all-packages)")
    lines.append(f"{indent}fi")
    packages = [package for package in target.uv_override_skip_packages if package]
    if not packages:
        return lines
    lines.append(f'{indent}if [ -n "${{UV_OVERRIDE:-}}" ]; then')
    lines.append(f"{indent}    uv_sync_args+=(")
    for package in packages:
        lines.append(f"{indent}        --no-install-package {shlex.quote(package)}")
    lines.append(f"{indent}    )")
    lines.append(f"{indent}fi")
    return lines


def _render_install_block(target: TufaSlurmTarget, job_dir: Path) -> list[str]:
    """Bash lines building the worker's per-run venv (R2.37) from the
    bundled sources under ``$JOB_DIR/src/``.

    Always ``uv sync`` from ``src/<target.main_project>``'s pyproject —
    it's the only uv command that honours ``[tool.uv.sources]`` path
    rewrites, and degenerates to a plain locked install when none are
    declared. ``UV_OVERRIDE`` lets a launcher point at a requirements
    file whose entries are then installed ``--no-deps`` afterward.

    The sync runs inside a ``flock``ed subshell on ``$UV_CACHE_DIR``:
    in no-image mode the cache lives on (often-NFS) ``$HOME/.cache/uv``
    where concurrent jobs would race; in image mode the cache is
    container-local and the lock is a no-op.

    Factored out so tests can bash-execute it against fixture sources
    rather than only string-asserting the rendered output.
    """
    if target.node_local_storage:
        return _render_node_local_install_block(target, job_dir)

    src_dir_q = shlex.quote(str(job_dir / "src"))
    src_tar_q = shlex.quote(str(job_dir / "src.tar"))
    main_project_q = shlex.quote(target.main_project)
    return [
        'UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"',
        'mkdir -p "$(dirname "${UV_CACHE_DIR}")"',
        f'export UV_PROJECT_ENVIRONMENT="{job_dir}/.venv"',
        # deploy() ships the snapshot as src.tar (one NFS write instead of
        # ~1k small-file creates). Extract it into src/ unless a directory
        # snapshot is already present (backward-compatible with older jobs).
        f"if [ -f {src_tar_q} ] && [ ! -d {src_dir_q}/{main_project_q} ]; then",
        f"    mkdir -p {src_dir_q}",
        f"    tar -C {src_dir_q} -xf {src_tar_q}",
        "fi",
        "(",
        "    flock 9",
        f"    cd {shlex.quote(str(job_dir / 'src' / target.main_project))}",
        *_render_uv_sync_args_block(target),
        '    uv sync "${uv_sync_args[@]}"',
        '    if [ -n "${UV_OVERRIDE:-}" ]; then',
        '        if [ ! -f "${UV_OVERRIDE}" ]; then',
        '            echo "UV_OVERRIDE points to a missing file: ${UV_OVERRIDE}" >&2',
        "            exit 1",
        "        fi",
        '        uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"',
        "    fi",
        ') 9>"${UV_CACHE_DIR}.lock" || { echo "TAAF worker dependency install failed; aborting worker startup." >&2; exit 1; }',
        'source "$UV_PROJECT_ENVIRONMENT/bin/activate"',
        _render_official_games_install(job_dir),
    ]


def _render_official_games_install(job_dir: Path) -> str:
    """The script line installing the official games' code that came with the job.

    The worker's fresh install lacks that code (``_stage_official_games``). Only
    games that need it fail without it, so a failure here does not stop the job.
    """
    staged = shlex.quote(str(job_dir / _OFFICIAL_GAMES_DIR))
    return f"RE_ARC_OFFICIAL_GAMES_DIR={staged} python -m re_arc.official_games --optional || true"


def _stage_official_games(job_dir: Path) -> None:
    """Put the launcher's official games' code where the worker's install finds it.

    ``re_arc`` ships these games' metadata but not their code, and the worker
    installs ``re_arc`` afresh, so a job carries what its launcher fetched and
    needs neither network nor Kaggle credentials on the node. A launcher that
    has not fetched them stages nothing, and only games that need them fail.
    """
    try:
        official_games = importlib.import_module("re_arc.official_games")
    except ImportError:
        return
    for source in official_games.official_sources():
        if source.is_file():
            staged = job_dir / _OFFICIAL_GAMES_DIR / source.relative_to(official_games.ENVIRONMENTS_DIR)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)


def _render_managed_python_heal_block(*, indent: str = "    ") -> list[str]:
    """Script lines removing a node-cached managed interpreter that no longer
    runs, so the sync after them reinstalls it.

    The node-local cache keeps uv's managed interpreters across jobs, and a
    cleanup of ``/cache`` has been seen to take ``bin/``, ``lib/`` and
    ``include/`` from one and leave ``share/``. uv finds the directory, takes
    the interpreter as installed, and dies writing ``EXTERNALLY-MANAGED`` into
    the missing ``lib/`` -- in a second, on every job that lands on that node,
    until someone deletes it. A directory younger than ten minutes is left
    alone: it may be another job's install in progress, which is possible
    whenever the sync lock's wait times out.
    """
    lines = [
        'if [ -d "${UV_PYTHON_INSTALL_DIR:-}" ]; then',
        '    for _taaf_py in "${UV_PYTHON_INSTALL_DIR}"/cpython-*; do',
        '        [ -d "${_taaf_py}" ] && [ ! -L "${_taaf_py}" ] || continue',
        '        [ -n "$(find "${_taaf_py}" -maxdepth 0 -mmin +10)" ] || continue',
        '        if ! "${_taaf_py}/bin/python3" -c "" >/dev/null 2>&1; then',
        '            echo "TAAF: removing a managed interpreter that no longer runs: ${_taaf_py}" >&2',
        '            rm -rf "${_taaf_py}"',
        "        fi",
        "    done",
        "    # The version aliases uv keeps beside it, now pointing at nothing.",
        '    find "${UV_PYTHON_INSTALL_DIR}" -maxdepth 1 -name "cpython-*" -xtype l -delete 2>/dev/null || true',
        "fi",
    ]
    return [f"{indent}{line}" for line in lines]


def _render_node_local_install_block(target: TufaSlurmTarget, job_dir: Path) -> list[str]:
    resolved_job_dir = job_dir.resolve()
    shared_src = resolved_job_dir / "src"
    shared_src_tar = resolved_job_dir / "src.tar"
    main_project = target.main_project
    storage_name = main_project
    shared_src_q = shlex.quote(str(shared_src))
    shared_src_tar_q = shlex.quote(str(shared_src_tar))
    main_project_q = shlex.quote(main_project)
    storage_name_q = shlex.quote(storage_name)
    shared_src_uri_q = shlex.quote(shared_src.as_uri())

    lines = [
        "# TAAF node-local storage: keep uv cache, worker venv, temp, and editable src off shared storage.",
        "if [ -f /etc/kander/job-env.sh ]; then",
        "    . /etc/kander/job-env.sh",
        "fi",
        'taaf_user="${USER:-${LOGNAME:-taaf}}"',
        'taaf_uid="$(id -u 2>/dev/null || printf \'%s\' "${UID:-0}")"',
        # A step of an allocation is keyed by job AND step: the allocation
        # outlives the run, and the next deploy into it must not find this
        # one's sources and venv already in place and skip its own.
        'taaf_job_id="${SLURM_JOB_ID:-$$}.${SLURM_STEP_ID:-0}"'
        if target.jobid is not None
        else 'taaf_job_id="${SLURM_JOB_ID:-$$}"',
        f"taaf_storage_name={storage_name_q}",
        # --export=ALL leaks the submitter's shell env; a login-node
        # UV_CACHE_DIR/TMPDIR would win the ${VAR:-...} fallbacks below and be
        # mkdir'd on unwritable node-local storage. Clear them so they resolve
        # locally; the override surface (TAAF_NODE_LOCAL_ROOT / *_CACHE_ROOT /
        # SCRATCH_DIR / SLURM_TMPDIR) is left intact.
        "unset UV_CACHE_DIR TMPDIR TEMPDIR NODE_LOCAL_STORAGE_ROOT "
        "TAAF_LOCAL_JOB_ROOT TAAF_LOCAL_SRC_ROOT TAAF_LOCAL_VENV_ROOT "
        "UV_PYTHON_INSTALL_DIR MPLCONFIGDIR XDG_CACHE_HOME TRITON_CACHE_DIR",
        'if [ -n "${TAAF_NODE_LOCAL_ROOT:-}" ]; then',
        '    taaf_node_root="${TAAF_NODE_LOCAL_ROOT}"',
        'elif [ -n "${SCRATCH_DIR:-}" ] && [ -d "${SCRATCH_DIR}" ] && [ -w "${SCRATCH_DIR}" ]; then',
        '    taaf_node_root="${SCRATCH_DIR}"',
        'elif [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ] && [ -w "${SLURM_TMPDIR}" ]; then',
        '    taaf_node_root="${SLURM_TMPDIR}"',
        'elif [ -n "${SLURM_JOB_ID:-}" ] && mkdir -p "/scratch/jobs/${SLURM_JOB_ID}" 2>/dev/null && [ -w "/scratch/jobs/${SLURM_JOB_ID}" ]; then',
        '    taaf_node_root="/scratch/jobs/${SLURM_JOB_ID}"',
        "else",
        '    taaf_node_root="/tmp/${taaf_user}/${taaf_storage_name}-${taaf_job_id}"',
        '    mkdir -p "${taaf_node_root}"',
        "fi",
        # Keyed by uid alone, like job-env.sh's HF_HOME: the contents are
        # content-addressed wheels and versioned interpreters that every
        # project shares, and a per-project root costs each new project a cold
        # download of what a sibling already holds.
        'if [ -n "${TAAF_NODE_CACHE_ROOT:-}" ]; then',
        '    taaf_cache_root="${TAAF_NODE_CACHE_ROOT}"',
        'elif mkdir -p "/cache/users/${taaf_uid}" 2>/dev/null && [ -w "/cache/users/${taaf_uid}" ]; then',
        '    taaf_cache_root="/cache/users/${taaf_uid}"',
        "else",
        '    taaf_cache_root="${taaf_node_root}/cache"',
        '    mkdir -p "${taaf_cache_root}"',
        "fi",
        'export TAAF_NODE_LOCAL_ROOT="${TAAF_NODE_LOCAL_ROOT:-${taaf_node_root}}"',
        'export NODE_LOCAL_STORAGE_ROOT="${NODE_LOCAL_STORAGE_ROOT:-${taaf_node_root}}"',
        'export TAAF_NODE_CACHE_ROOT="${TAAF_NODE_CACHE_ROOT:-${taaf_cache_root}}"',
        'export TAAF_LOCAL_JOB_ROOT="${TAAF_LOCAL_JOB_ROOT:-${taaf_node_root}/${taaf_storage_name}-${taaf_job_id}}"',
        'export TAAF_LOCAL_SRC_ROOT="${TAAF_LOCAL_SRC_ROOT:-${TAAF_LOCAL_JOB_ROOT}/src}"',
        'export TAAF_LOCAL_VENV_ROOT="${TAAF_LOCAL_VENV_ROOT:-${TAAF_LOCAL_JOB_ROOT}/venvs}"',
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-${TAAF_NODE_CACHE_ROOT}/uv}"',
        'export UV_LINK_MODE="${UV_LINK_MODE:-copy}"',
        # The uv-managed CPython is the venv's base interpreter -- its stdlib is
        # read on every import. Default ($HOME/.local/share/uv/python) is shared
        # storage, so a node-local venv still pays per-import NFS latency and cold
        # kernels time out under concurrent startup. Install it node-local (and
        # force managed so uv never falls back to a shared system interpreter).
        'export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${TAAF_NODE_CACHE_ROOT}/python}"',
        "export UV_PYTHON_PREFERENCE=only-managed",
        # Same trap for $HOME-defaulting caches read/written hot at startup:
        # matplotlib's font cache and the XDG cache catch-all (fontconfig, ...).
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-${TAAF_NODE_CACHE_ROOT}/matplotlib}"',
        'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TAAF_NODE_CACHE_ROOT}/xdg}"',
        # Triton is the same trap once more, and the costliest: it JIT-compiles
        # kernels during engine warmup and caches them under $HOME/.triton.
        # torch's inductor cache needs no such line because it defaults under
        # tempfile.gettempdir(), which TMPDIR below already sends node-local --
        # Triton does not follow TMPDIR, so it lands on shared storage and is
        # both read and written hot while the model warms up. Per node rather
        # than per job: the artifacts are expensive and identical across jobs on
        # one node (same GPU arch, same pinned libraries), while a cache shared
        # BETWEEN nodes is the thing to avoid.
        'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TAAF_NODE_CACHE_ROOT}/triton}"',
        'export TMPDIR="${TMPDIR:-${TAAF_LOCAL_JOB_ROOT}/tmp}"',
        'export TEMPDIR="${TEMPDIR:-${TMPDIR}}"',
    ]

    lines.extend(
        [
            'mkdir -p "${TAAF_LOCAL_JOB_ROOT}" "${TAAF_LOCAL_SRC_ROOT}" "${TAAF_LOCAL_VENV_ROOT}" "${UV_CACHE_DIR}" "${TAAF_NODE_CACHE_ROOT}" "${TMPDIR}" "${UV_PYTHON_INSTALL_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"',
            f'if [ ! -d "${{TAAF_LOCAL_SRC_ROOT}}"/{main_project_q} ]; then',
            '    rm -rf "${TAAF_LOCAL_SRC_ROOT}"',
            '    mkdir -p "${TAAF_LOCAL_SRC_ROOT}"',
            # deploy() ships the snapshot as a single src.tar; extracting it
            # node-locally is one sequential read + local writes, versus ~1k
            # small-file reads off NFS via rsync/cp. Fall back to the
            # directory snapshot for backward compatibility.
            f"    if [ -f {shared_src_tar_q} ]; then",
            f'        tar -C "${{TAAF_LOCAL_SRC_ROOT}}" -xf {shared_src_tar_q}',
            "    elif command -v rsync >/dev/null 2>&1; then",
            f'        rsync -a --delete {shared_src_q}/ "${{TAAF_LOCAL_SRC_ROOT}}/"',
            "    else",
            f'        cp -a {shared_src_q}/. "${{TAAF_LOCAL_SRC_ROOT}}/"',
            "    fi",
            "fi",
            'if [ -n "${UV_OVERRIDE:-}" ]; then',
            '    if [ ! -f "${UV_OVERRIDE}" ]; then',
            '        echo "UV_OVERRIDE points to a missing file: ${UV_OVERRIDE}" >&2',
            "        exit 1",
            "    fi",
            '    taaf_local_override="${TAAF_LOCAL_JOB_ROOT}/deployment-overrides.txt"',
            f"    taaf_shared_src_uri={shared_src_uri_q}",
            '    taaf_local_src_uri="file://${TAAF_LOCAL_SRC_ROOT}"',
            "    while IFS= read -r line; do",
            '        printf "%s\\n" "${line//$taaf_shared_src_uri/$taaf_local_src_uri}"',
            '    done < "${UV_OVERRIDE}" > "${taaf_local_override}"',
            '    export UV_OVERRIDE="${taaf_local_override}"',
            "fi",
            f'export UV_PROJECT_ENVIRONMENT="${{TAAF_LOCAL_VENV_ROOT}}"/{main_project_q}.venv',
            # The lock path follows the cache dir, so jobs from different
            # projects on one node serialize their syncs: they write the same
            # cache, and a warm one keeps each sync short enough to wait on.
            # The wait is bounded and logged: syncing unlocked is safe (the
            # venv is per-job and uv's cache locks internally), and beats
            # burning the walltime behind one wedged sync.
            "(",
            '    exec {_taaf_sync_lock_fd}>"${UV_CACHE_DIR}.lock"',
            '    echo "waiting for uv sync lock ${UV_CACHE_DIR}.lock"',
            '    flock -w 900 "${_taaf_sync_lock_fd}" || echo "uv sync lock wait timed out; proceeding unlocked" >&2',
            *_render_managed_python_heal_block(),
            f'    cd "${{TAAF_LOCAL_SRC_ROOT}}"/{main_project_q}',
            *_render_uv_sync_args_block(target),
            '    uv sync "${uv_sync_args[@]}"',
            '    if [ -n "${UV_OVERRIDE:-}" ]; then',
            '        uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"',
            "    fi",
            ') || { echo "TAAF worker dependency install failed; aborting worker startup." >&2; exit 1; }',
            'source "$UV_PROJECT_ENVIRONMENT/bin/activate"',
            _render_official_games_install(job_dir),
        ]
    )
    return lines
