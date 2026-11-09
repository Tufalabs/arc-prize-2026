"""The two seeding rules, and what a job dir is allowed to be reused for.

Both are rules whose violation is silent: the sweep runs, every number looks
ordinary, and the experiment answers a different question than the one asked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_sweep_agent import DESIGN, ScriptedAgent

from sweeper.agent import SweepAgent
from sweeper.jobs import (
    DESIGN_STAMP,
    RESULTS_FILE,
    check_reusable,
    finished_job_dirs,
    is_finished,
    job_status,
    write_design_stamp,
)
from sweeper.seeds import arm_seed_block, play_seed


def test_no_two_jobs_share_an_arm() -> None:
    """Reusing one block across jobs would replay five design points ten
    times instead of drawing fifty — a measurement of the noise floor wearing
    the shape of a sweep."""
    blocks = [arm_seed_block(run_seed, 5, base=1000) for run_seed in range(10)]
    seeds = [seed for block in blocks for seed in block]

    assert len(set(seeds)) == len(seeds) == 50
    assert blocks[0] == (1000, 1001, 1002, 1003, 1004)


def test_no_two_jobs_share_a_play_seed() -> None:
    """A game's seed is the job's base plus its position in the run's list, so
    consecutive bases must clear a whole job's worth of entries."""
    entries = 250
    ranges = [range(play_seed(run_seed), play_seed(run_seed) + entries) for run_seed in range(10)]
    seen: set[int] = set()
    for span in ranges:
        assert not seen & set(span)
        seen |= set(span)


def test_a_stride_that_cannot_separate_two_jobs_is_refused() -> None:
    """The failure this replaces is invisible: no error, and no result file
    records the seeds a run used."""
    with pytest.raises(ValueError, match="stride above 500"):
        play_seed(3, stride=100, entries=500)


def _results(job_dir: Path, *, finished: bool) -> None:
    """A results file as TAAF writes it — mid-run, or after the final save.

    ``end_time`` is stamped just before that last save, so it is the only
    thing in the file that distinguishes the two.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    end = '"2026-01-01T00:00:00"' if finished else "null"
    (job_dir / RESULTS_FILE).write_text(f'{{"end_time": {end}}}')


def _agent(run_seed: int = 1) -> SweepAgent:
    return SweepAgent(inner=ScriptedAgent(), design=DESIGN, run_seed=run_seed, arm_seeds=(11, 12), announce=False)


def test_submitting_over_another_design_is_refused(tmp_path: Path) -> None:
    """A job dir is addressed by its seeds, which name an experiment only
    given a fixed knob set. A launcher that found these results and skipped
    would report the job done while the design it meant to run never ran."""
    agent = _agent()
    write_design_stamp(tmp_path, agent)
    _results(tmp_path, finished=True)

    check_reusable(tmp_path, agent)  # same design: nothing to refuse
    with pytest.raises(RuntimeError, match="different design"):
        check_reusable(tmp_path, _agent(run_seed=2))


def test_results_stay_readable_after_the_design_changes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Reading is not gated on the design still matching. Every row carries
    the values its own job ran, so a moved design changes what the pool
    MEANS, not whether it can be read — and refusing would strand finished
    work behind an edit to a file that cannot reach into the past."""
    agent = _agent()
    write_design_stamp(tmp_path, agent)
    _results(tmp_path, finished=True)
    moved = _agent(run_seed=2)

    assert is_finished(tmp_path)
    assert job_status(tmp_path) == "finished"
    assert finished_job_dirs([(tmp_path, moved)]) == [tmp_path]
    assert "design that has since changed" in capsys.readouterr().out, "silently is the one way not to do it"


def test_a_job_without_results_is_in_flight_only_once_deployed(tmp_path: Path) -> None:
    """Status is read off the dir, so a restarted notebook — or another
    machine — can follow a sweep it did not submit."""
    agent = _agent()
    assert job_status(tmp_path) == "not submitted"

    write_design_stamp(tmp_path, agent)
    (tmp_path / "deploy_meta.json").write_text("{}")
    assert job_status(tmp_path) == "in flight"

    # A running job writes results too — periodically, from its first save
    # on. Reading that as finished would hand the collector a half-played
    # job, whose late-starting arms score low for no reason of their own.
    _results(tmp_path, finished=False)
    assert job_status(tmp_path) == "in flight"

    _results(tmp_path, finished=True)
    assert job_status(tmp_path) == "finished"


def test_an_analysis_can_run_on_the_finished_part_of_a_sweep(tmp_path: Path) -> None:
    """Waiting for the slowest job is not required; the rest is real data."""
    done, pending = tmp_path / "done", tmp_path / "pending"
    agent = _agent()
    write_design_stamp(done, agent)
    _results(done, finished=True)
    write_design_stamp(pending, agent)
    _results(pending, finished=False)

    assert finished_job_dirs([(done, agent), (pending, agent)]) == [done]
    assert (done / DESIGN_STAMP).read_text() == agent.describe()
