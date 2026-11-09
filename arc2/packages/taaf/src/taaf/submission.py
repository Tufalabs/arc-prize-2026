"""The Kaggle submission file — on Kaggle, the deliverable.

Two properties earn their complexity from the environment a scoring rerun
runs in — no cancel signal, no graceful stop, a hard cutoff, and no logs
returned:

* the file is **seeded for every task before the run starts**, so a kill at
  any moment leaves something scorable rather than nothing;
* every write is **atomic**, because a kill landing mid-write would leave
  truncated non-JSON — strictly worse than the placeholder it replaced.

Whether a killed run's file is scored at all is still unknown; ``roadmap.md``
records the experiment that decides it and the deadline policy chosen until
then.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import taaf.support
from taaf.game import N_ATTEMPTS, GameRun, GameSpec, Submission
from taaf.grid import Grid

SUBMISSION_NAME = "submission.json"

#: What an unanswered pair submits. A 1x1 grid is the cheapest legal answer
#: and is essentially never right, so a placeholder cannot flatter a score.
PLACEHOLDER = Grid(np.zeros((1, 1), dtype=np.int8))


def placeholder_submission(spec: GameSpec) -> Submission:
    """A submission of placeholders, shaped for ``spec``."""
    return Submission(attempts=tuple((PLACEHOLDER,) * N_ATTEMPTS for _ in spec.to_answer))


def build(specs: Sequence[GameSpec], runs: Sequence[GameRun]) -> dict[str, list[dict[str, list[list[int]]]]]:
    """The submission for ``specs``, filled from whichever runs have answered.

    A run's *standing* answer — its last, since a solver may replace an
    earlier one. Every task in ``specs`` appears, answered or not: a
    missing id scores 0 at best and may invalidate the submission outright.
    """
    answered = {run.game_id: run.history[-1].submission for run in runs if run.history}
    return {spec.game_id: answered.get(spec.game_id, placeholder_submission(spec)).to_json_list() for spec in specs}


def validate(submission: dict[str, list[dict[str, list[list[int]]]]], specs: Sequence[GameSpec]) -> None:
    """Check the submission covers every task with well-formed grids.

    Runs before each write. Failing here is recoverable; failing in the
    scoring rerun is a wasted submission we would never see a log for.
    """
    by_id = {spec.game_id: spec for spec in specs}
    missing = set(by_id) - set(submission)
    if missing:
        raise ValueError(f"submission is missing {len(missing)} task(s): {', '.join(sorted(missing))}")
    for game_id, groups in submission.items():
        spec = by_id.get(game_id)
        if spec is None:
            raise ValueError(f"submission carries unknown task {game_id!r}")
        if len(groups) != len(spec.to_answer):
            raise ValueError(f"{game_id}: {len(groups)} answer group(s) for {len(spec.to_answer)} pair(s)")
        for i, group in enumerate(groups):
            expected = {f"attempt_{k + 1}" for k in range(N_ATTEMPTS)}
            if set(group) != expected:
                raise ValueError(f"{game_id} pair {i}: keys {sorted(group)}, expected {sorted(expected)}")
            for key, rows in group.items():
                # Grid's own validation is the shape/palette check; reuse it
                # rather than restating the rules where they could drift.
                try:
                    Grid.from_lists(rows)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{game_id} pair {i} {key}: {e}") from e


def write(path: Path, submission: dict[str, list[dict[str, list[list[int]]]]], specs: Sequence[GameSpec]) -> None:
    """Validate and atomically write the submission."""
    validate(submission, specs)
    taaf.support.atomic_json_dump(submission, path)


def read(path: Path) -> dict[str, list[dict[str, list[list[int]]]]]:
    """Read a submission back, for comparing two runs' answers."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]
