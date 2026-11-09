"""What a job dir says about the job that owns it.

A sweep is many jobs, submitted over time and often from a different process
than the one that reads them, so everything here is answered from the
directory rather than from a handle: a notebook that was restarted, or a
different machine entirely, can still follow a sweep it did not submit.

The design stamp is the load-bearing part. A job dir is addressed by its
seeds, but seeds only identify an experiment given a fixed knob set — edit
the design and the same seeds mean something else, so reuse has to be a
refusal rather than a silent pool of two experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import platypus.viewer_data
from sweeper.agent import SweepCore

#: Written beside the results by whoever launched the job.
DESIGN_STAMP = "sweep_design.txt"

#: What TAAF writes as a benchmark runs, and when a deploy targets a dir.
RESULTS_FILE = "benchmark.json"
DEPLOY_FILE = "deploy_meta.json"

JobStatus = Literal["finished", "in flight", "not submitted"]


def design_stamp_path(job_dir: Path | str) -> Path:
    """Where a job dir records the design it was launched with."""
    return Path(job_dir) / DESIGN_STAMP


def write_design_stamp(job_dir: Path | str, agent: SweepCore) -> None:
    """Record the design a job is being launched with.

    Write it at LAUNCH, not at completion: a deployed job finishes on a
    compute node, so the launching process never sees it end.
    """
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    design_stamp_path(job_dir).write_text(agent.describe())


def is_finished(job_dir: Path | str) -> bool:
    """Whether a FINISHED benchmark has been written here.

    Not the file's existence: TAAF saves periodically while a run is in
    flight, so ``benchmark.json`` appears minutes after submission with every
    lane still playing. Reading that as "finished" is worse than an error —
    the collector pools a half-played job, the arms that had not got going
    yet score low, and the sweep reports a knob effect that is really a
    snapshot time.

    The rule itself is ``platypus.viewer_data.benchmark_finished``, which the
    run listing reads too; one predicate rather than two that drift, and it
    memoises so a watch loop polling this does not re-parse a multi-MB file
    every time.
    """
    return platypus.viewer_data.benchmark_finished(job_dir)


def design_moved(job_dir: Path | str, agent: SweepCore) -> bool:
    """Whether this dir's results answer a different design than ``agent``'s."""
    stamp = design_stamp_path(job_dir)
    return not stamp.is_file() or stamp.read_text() != agent.describe()


def check_reusable(job_dir: Path | str, agent: SweepCore) -> None:
    """Refuse to submit into a dir already holding another design's results.

    A SUBMIT-side check, not a read-side one. The job dir is addressed by its
    seeds, which name an experiment only given a fixed knob set — so a
    launcher that finds results here and skips would report the job done
    while the design it meant to run never ran. Raising is the only outcome
    that cannot be mistaken for success.

    Reading those same results is always fine, and deliberately not gated on
    this: every row carries the values its own job was configured with, and
    `missing` exists so jobs from different knob sets pool in one table.

    Gated on SUBMISSION evidence rather than on results, because the hole is
    the in-flight window: the stamp is written at launch, so a design edited
    while jobs are still running would otherwise pass here and be skipped as
    "in flight" — the compute spent on the old design, and nothing saying so
    until collection.
    """
    submitted = is_finished(job_dir) or (Path(job_dir) / DEPLOY_FILE).is_file()
    if submitted and design_moved(job_dir, agent):
        raise RuntimeError(
            f"{job_dir} holds a different design than the one submitting "
            f"(compare {design_stamp_path(job_dir)}). Point the runs root elsewhere so both "
            "experiments keep their results, or delete the directory to replay it. Analysis of "
            "what is already there is unaffected — collect_rows reads each job's own design."
        )


def job_status(job_dir: Path | str) -> JobStatus:
    """``finished`` / ``in flight`` / ``not submitted``, read off the dir."""
    if is_finished(job_dir):
        return "finished"
    return "in flight" if (Path(job_dir) / DEPLOY_FILE).exists() else "not submitted"


def finished_job_dirs(jobs: list[tuple[Path, SweepCore]]) -> list[Path]:
    """The subset with results, for the collector.

    Lets an analysis run on a partial sweep — it simply has fewer
    observations — rather than waiting for the slowest job.

    A job whose design has since moved is INCLUDED, and said out loud. Its
    rows describe themselves: `collect_rows` reads the design off the job's
    own solver, so the values are the ones that played. What the reader needs
    to know is that the pool spans more than one design, which is a fact
    about the table rather than a reason to refuse it.
    """
    finished = [(job_dir, agent) for job_dir, agent in jobs if is_finished(job_dir)]
    moved = [job_dir for job_dir, agent in finished if design_moved(job_dir, agent)]
    if moved:
        print(
            f"\n{len(moved)} of {len(finished)} finished job(s) ran a design that has since changed. "
            "Their rows carry the values that played, so they pool — but a knob's meaning may not be "
            "the same across them:",
            flush=True,
        )
        for job_dir in moved:
            print(f"  {job_dir}", flush=True)
    return [job_dir for job_dir, _ in finished]


__all__ = [
    "DEPLOY_FILE",
    "DESIGN_STAMP",
    "RESULTS_FILE",
    "JobStatus",
    "design_stamp_path",
    "finished_job_dirs",
    "is_finished",
    "job_status",
    "write_design_stamp",
]
