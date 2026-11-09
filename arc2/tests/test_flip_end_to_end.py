"""The example agent against the shipped ARC-AGI-2 data, end to end.

This is the check the whole harness is arranged around: a constant
left-right flip solves exactly one task in `arc2-train` (`67a3c6ac`) and none
of `arc2-eval`, measured rather than assumed, so the expected outcome of a
run is known exactly before it starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import taaf.datasets
import taaf.game
import taaf.solver_examples
import taaf.standard_benchmarks
import taaf.submission
from taaf.standard_benchmarks import FLIP_TASK_ID


def _flip_solves(spec: taaf.game.GameSpec) -> bool:
    """Whether a constant fliplr is correct on every pair of ``spec``."""
    return all(p.answer is not None and np.array_equal(np.fliplr(p.question.data), p.answer.data) for p in spec.pairs)


def test_exactly_one_shipped_task_is_a_constant_flip():
    """The measurement the example agent's assertion rests on. If a dataset
    rebuild ever changed this, every expectation below would silently mean
    something else."""
    train = taaf.datasets.load("arc2-train")
    ev = taaf.datasets.load("arc2-eval")
    assert [s.game_id for s in train if _flip_solves(s)] == [FLIP_TASK_ID]
    assert [s.game_id for s in ev if _flip_solves(s)] == []


def test_the_flip_task_has_the_shape_we_expect():
    """Three demonstrations and one pair to answer, all square."""
    (spec,) = [s for s in taaf.datasets.load("arc2-train") if s.game_id == FLIP_TASK_ID]
    assert len(spec.given) == 3
    assert len(spec.to_answer) == 1
    assert all(p.question.shape[0] == p.question.shape[1] for p in spec.pairs)


async def test_flip_demo_scores_exactly_one_task(tmp_path: Path):
    """The end-to-end run: 100 on the flippable task, 0 on the rest, one
    task solved."""
    bm = taaf.standard_benchmarks.make_benchmark_flip_demo(taaf.solver_examples.SolverFlipLR(report_tokens=100))
    bm.job_dir = tmp_path
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()

    by_id = {r.game_id: r for r in bm.game_runs}
    assert by_id[FLIP_TASK_ID].final_score == 100.0
    assert by_id[FLIP_TASK_ID].state == "won"
    assert sum(r.final_score == 100.0 for r in bm.game_runs) == 1
    assert all(r.final_score == 0.0 for gid, r in by_id.items() if gid != FLIP_TASK_ID)

    # Every game answered, and the submission covers all of them.
    assert all(r.state in ("won", "submitted") for r in bm.game_runs)
    submission = taaf.submission.read(tmp_path / "submission.json")
    assert set(submission) == set(by_id)
    assert (tmp_path / "answers" / f"g{FLIP_TASK_ID}_p0.html").is_file()


async def test_simulated_submission_answers_identically_but_scores_nothing(tmp_path: Path):
    """The dress rehearsal over the same tasks: same submission, no score,
    no diagnostics — the state a competition rerun is actually in."""
    scored = taaf.standard_benchmarks.make_benchmark_flip_demo(taaf.solver_examples.SolverFlipLR())
    scored.job_dir = tmp_path / "scored"
    scored.submission_path = tmp_path / "scored" / "submission.json"
    await scored.run()

    sim = taaf.standard_benchmarks.make_benchmark_flip_demo(taaf.solver_examples.SolverFlipLR())
    sim.job_dir = tmp_path / "sim"
    sim.simulate_submission = True
    sim.submission_path = tmp_path / "sim" / "submission.json"
    await sim.run()

    assert all(r.final_score is None for r in sim.game_runs)
    assert not (tmp_path / "sim" / "benchmark.json").exists()
    assert taaf.submission.read(tmp_path / "scored" / "submission.json") == taaf.submission.read(
        tmp_path / "sim" / "submission.json"
    )


@pytest.mark.local
async def test_full_eval_set_runs_and_scores_zero(tmp_path: Path):
    """120 tasks, 167 test outputs, and a flip solves none of them. Cheap
    enough to be worth running for real: it is the only test that exercises
    the whole eval set's shapes through the scorer and the submission."""
    bm = taaf.standard_benchmarks.make_benchmark_named(
        "arc2-eval", taaf.solver_examples.SolverFlipLR(), weighting="output"
    )
    bm.job_dir = tmp_path
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()
    assert len(bm.game_runs) == 120
    assert all(r.final_score == 0.0 for r in bm.game_runs)
    submission = taaf.submission.read(tmp_path / "submission.json")
    assert sum(len(v) for v in submission.values()) == 167
