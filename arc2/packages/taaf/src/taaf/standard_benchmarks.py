"""Benchmark builders over the named datasets.

The weighting choice is the interesting part: which of the two official
metrics a run reports is decided here, and only here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import taaf.benchmark
import taaf.datasets
import taaf.game
import taaf.solver

#: The two ways to weight tasks, and what each reproduces.
#:
#: ``"task"``  — every task counts once. This is ``arc-agi-benchmarking``'s
#:               ``total_score / total_tasks``, the arcprize.org number.
#: ``"output"`` — a task counts once per pair it must answer. This is the
#:               Kaggle metric, "the average across all task test outputs".
#:
#: They differ on any multi-test task, which is 37.5% of the public
#: evaluation set — so this is a choice, never a default to drift into.
Weighting = Literal["task", "output"]


def weights_for(specs: Sequence[taaf.game.GameSpec], weighting: Weighting) -> list[float] | None:
    """Per-task weights for a weighting scheme; ``None`` means equal."""
    if weighting == "task":
        return None
    if weighting == "output":
        return [float(len(spec.to_answer)) for spec in specs]
    raise ValueError(f"unknown weighting {weighting!r}; use 'task' or 'output'")  # type: ignore[unreachable]


def make_benchmark(
    specs: Sequence[taaf.game.GameSpec],
    solver: taaf.solver.Solver | None = None,
    *,
    label: str,
    n_passes: int = 1,
    weighting: Weighting = "task",
    withhold_answers: bool = False,
) -> taaf.benchmark.Benchmark:
    """A benchmark over ``specs``.

    ``withhold_answers`` drops the ground truth from every pair the solver
    must answer, putting the run in the state a competition rerun is actually
    in. It is the data half of ``simulate_submission`` — see
    ``Benchmark.run``'s ``minimal_diagnostics`` for the other half.
    """
    used = [spec.withhold_answers() for spec in specs] if withhold_answers else list(specs)
    return taaf.benchmark.Benchmark(
        label=label,
        games=taaf.game.games_from_specs(used),
        solver=solver,
        n_passes=n_passes,
        game_weights=weights_for(used, weighting),
    )


def make_benchmark_named(
    names: str | Sequence[str],
    solver: taaf.solver.Solver | None = None,
    *,
    label: str | None = None,
    limit: int | None = None,
    ids: Sequence[str] | None = None,
    n_passes: int = 1,
    weighting: Weighting = "task",
    withhold_answers: bool = False,
) -> taaf.benchmark.Benchmark:
    """A benchmark over one named dataset, or several merged, optionally narrowed.

    Several names are concatenated in order through :func:`taaf.datasets.combine`,
    so a task two sets share plays once; ``ids`` and ``limit`` then narrow the
    merged list. The label defaults to the names joined by ``+``.
    """
    names = (names,) if isinstance(names, str) else tuple(names)
    specs = taaf.datasets.combine(*names)
    if ids is not None or limit is not None:
        specs = taaf.datasets.subset(specs, ids=ids, limit=limit)
    return make_benchmark(
        specs,
        solver,
        label=label or "+".join(names),
        n_passes=n_passes,
        weighting=weighting,
        withhold_answers=withhold_answers,
    )


#: The one task in ARC-AGI-2's public training set that a constant
#: left-right flip solves. Measured, not assumed: ``tests/test_flip_end_to_end.py``
#: re-measures it against the shipped data.
FLIP_TASK_ID = "67a3c6ac"


def make_benchmark_easy(
    solver: taaf.solver.Solver | None = None,
    *,
    label: str = "easy",
    n_passes: int = 1,
    weighting: Weighting = "task",
    withhold_answers: bool = False,
) -> taaf.benchmark.Benchmark:
    """The curated easy set (``arc2-train-easy``), for exercising the harness.

    A capable solver should score 100 here. Anything less is worth reading as
    a harness defect first and a solver failure second — which is the whole
    point, and the opposite of how a score on a real benchmark should be
    read.
    """
    return make_benchmark_named(
        "arc2-train-easy",
        solver,
        label=label,
        n_passes=n_passes,
        weighting=weighting,
        withhold_answers=withhold_answers,
    )


def make_benchmark_flip_demo(
    solver: taaf.solver.Solver | None = None,
    *,
    n_other: int = 7,
    label: str = "flip-demo",
    n_passes: int = 1,
    weighting: Weighting = "task",
    withhold_answers: bool = False,
) -> taaf.benchmark.Benchmark:
    """The example agent's end-to-end benchmark: the one flippable task plus
    a few that are not.

    Small enough to run anywhere, and its expected outcome is exact — one
    task at 100, the rest at 0 — which is what makes it a check rather than a
    demo.
    """
    specs = taaf.datasets.load("arc2-train")
    others = [s for s in specs if s.game_id != FLIP_TASK_ID][:n_other]
    flip = [s for s in specs if s.game_id == FLIP_TASK_ID]
    if not flip:
        raise LookupError(f"{FLIP_TASK_ID} not in arc2-train — the packed dataset is not what we expect")
    return make_benchmark(
        flip + others, solver, label=label, n_passes=n_passes, weighting=weighting, withhold_answers=withhold_answers
    )
