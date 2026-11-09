"""A worked sweep: the random reference agent on the official 25.

Ten SLURM jobs of five arms each — 50 configurations over 1,250 plays. The
shape is the point: **arms are cheap, jobs are not**, so a per-arm knob is
answered inside one job while a run-level knob needs as many jobs as you
want observations of it.

Two knobs, one per batch:

- ``max_moves`` (per arm, log-uniform 1..1000) — how long a game may run
  before the agent gives up. Per arm because the agent reads it off its own
  per-game copy, and log-uniform because the question is a ratio. Each job
  draws its own five, so ten jobs give fifty distinct arms.
- ``click_on_color`` (run level) — whether a CLICK targets a colour present
  in the frame. It is an ordinary agent field, not an engine setting, so it
  *could* be swept per arm; it is here as the run-level example, and ten
  jobs are what make it measurable at all.

CPU-only: no LLM, no GPU, and the games come from ``re_arc``, so the jobs
ask the cluster for cores and nothing else.

Notebook: ``packages/sweeper/notebooks/example_analysis.ipynb``.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import taaf.benchmark
import taaf.deploy_slurm
import taaf.standard_benchmarks

from platypus import AgentSolver, RunConfig
from reference_agents import RandomAgent
from sweeper.agent import SweepAgent
from sweeper.jobs import check_reusable, write_design_stamp
from sweeper.jobs import finished_job_dirs as _finished
from sweeper.jobs import job_status as _job_status
from sweeper.knobs import Choice, Design, Field, Knob, LogUniformInt
from sweeper.seeds import arm_seed_block, play_seed

#: Arms per job, i.e. passes per benchmark.
ARMS_PER_RUN = 5

#: Where each job's arm seed block starts (``sweeper.seeds``).
ARM_SEED_BASE = 1000

#: Games x passes in one job, which is what the play-seed stride must clear.
ENTRIES_PER_JOB = 25 * ARMS_PER_RUN

#: What this experiment varies. The `missing` values are the agent's own
#: defaults, so a job recorded before either knob existed would still pool.
DESIGN = Design(
    run={
        "click_on_color": Knob(
            draw=Choice((False, True)),
            apply=Field("inner.click_on_color"),
            missing=False,
        )
    },
    arm={
        "max_moves": Knob(
            draw=LogUniformInt(1, 1000),
            apply=Field("inner.max_moves"),
            missing=100,
        )
    },
)


def arm_seeds_for(run_seed: int) -> tuple[int, ...]:
    """This job's arm seeds."""
    return arm_seed_block(run_seed, ARMS_PER_RUN, base=ARM_SEED_BASE)


def job_dir_for(runs_root: Path | str, run_seed: int) -> Path:
    """Where this job's results live.

    Named after the seeds that produce it, which is what makes a resubmit
    idempotent: the same seeds address the same directory, and a finished one
    is reused rather than replayed.
    """
    arm_seeds = arm_seeds_for(run_seed)
    return Path(runs_root) / f"rs{run_seed}_arms{arm_seeds[0]}to{arm_seeds[-1]}"


def detect_main_project() -> str:
    """The bundled repo directory the SLURM worker ``uv sync``s from.

    Must equal this checkout's DIRECTORY name: the source snapshot names each
    repo after its root directory, and a git worktree is named differently
    from the repo it came from — so a hardcoded value is right in exactly one
    checkout and wrong in every other one.
    """
    root = pathlib.Path(__file__).resolve()
    for candidate in root.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate.name
    raise RuntimeError(f"no repo root (pyproject.toml + .git) above {root}")


def build_agent(run_seed: int, *, max_concurrent: int = 16, announce: bool = True) -> SweepAgent:
    """The sweep agent for one job.

    Subprocess hosting, the default, so the design makes a real trip through
    a real pickle into a real child — which is where a knob built from a
    closure fails.

    Concurrency buys less here than it looks: the engine stays in the PARENT
    (only the agent and the move loop run in the child), so the games' engine
    steps are serialised however many children there are. The children do
    write their own diagnostics in parallel, which is worth roughly a
    doubling over in-process hosting; past that, more children mostly wait on
    the parent, so the job asks for cores to match rather than to saturate.
    """
    return SweepAgent(
        inner=RandomAgent(),
        design=DESIGN,
        run_seed=run_seed,
        arm_seeds=arm_seeds_for(run_seed),
        announce=announce,
        # Derived from the job's seed rather than equal to it, and checked
        # against this job's entry count so two jobs cannot share randomness.
        run_config=RunConfig(
            max_concurrent=max_concurrent,
            rng_seed=play_seed(run_seed, entries=ENTRIES_PER_JOB),
        ),
    )


def build_benchmark(run_seed: int, job_dir: Path) -> taaf.benchmark.Benchmark:
    """The official 25 at one pass per arm, into ``job_dir``."""
    solver = AgentSolver(agent=build_agent(run_seed))
    benchmark = taaf.standard_benchmarks.make_benchmark_official_25(solver=solver)
    benchmark.label = f"sweep_random_rs{run_seed}"
    benchmark.n_passes = ARMS_PER_RUN
    benchmark.job_dir = job_dir
    return benchmark


def job_status(runs_root: Path | str, run_seed: int) -> str:
    """``finished`` / ``in flight`` / ``not submitted`` for one job."""
    return _job_status(job_dir_for(runs_root, run_seed))


async def deploy_job(
    runs_root: Path | str,
    run_seed: int,
    *,
    time: str = "01:00:00",
    total_cpus: int = 16,
    mem_gib: int = 32,
) -> Path:
    """Submit one job to SLURM unless it is already finished or in flight.

    Fire and forget: the handle is not returned, because a notebook that
    reattaches by job dir survives a kernel restart and a handle does not.
    """
    job_dir = job_dir_for(runs_root, run_seed)
    agent = build_agent(run_seed, announce=False)
    # Before the skip, not after: a dir holding another design's results would
    # otherwise be reported "finished" and the design meant to run never runs.
    check_reusable(job_dir, agent)
    status = job_status(runs_root, run_seed)
    if status != "not submitted":
        print(f"[example] {job_dir.name}: {status}")
        return job_dir

    write_design_stamp(job_dir, agent)
    target = taaf.deploy_slurm.TufaSlurmTarget(
        gpu="CPU",  # a random agent has nothing to put on a card
        gpu_count=0,
        total_cpus=total_cpus,
        total_mem_bytes=mem_gib * 1024**3,
        time=time,
        main_project=detect_main_project(),
    )
    handle = await build_benchmark(run_seed, job_dir).deploy(target)
    print(f"[example] {job_dir.name}: submitted as {getattr(handle, 'job_id', handle)}")
    return job_dir


def finished_job_dirs(runs_root: Path | str, run_seeds: tuple[int, ...]) -> list[Path]:
    """The job dirs that have results for this design, for the collector."""
    return _finished([(job_dir_for(runs_root, seed), build_agent(seed, announce=False)) for seed in run_seeds])


__all__ = [
    "ARMS_PER_RUN",
    "ARM_SEED_BASE",
    "DESIGN",
    "ENTRIES_PER_JOB",
    "arm_seeds_for",
    "build_agent",
    "build_benchmark",
    "deploy_job",
    "detect_main_project",
    "finished_job_dirs",
    "job_dir_for",
    "job_status",
]
