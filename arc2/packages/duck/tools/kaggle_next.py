"""The one Kaggle traffic run queued on this branch, launched as committed:

    git pull && uv run python packages/duck/tools/kaggle_next.py

No flags. What the run is lives in ``RUN`` below, committed at the branch's
tip, so whoever launches it -- with their own Kaggle credentials -- launches
exactly what was committed. Every run goes to the one notebook ``SLUG`` under
the launching account, each launch a new version of it, and is read out from
there. ``RUN`` and ``RUN_ID`` are None while nothing is queued, and the file
then refuses.

Each run's source bundle is a dataset of its own, ``<SLUG>-src-<RUN_ID>``: a
kernel version attaches a dataset's current version, which for one shared
dataset could still be the previous run's, and a new dataset has no earlier
version to attach.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

# The notebook every run goes to (under the launching account).
SLUG = "arc2-traffic-4xl4"
# The queued run's own id: its source dataset is <SLUG>-src-<RUN_ID>. A new
# id for every run queued, short (Kaggle caps a dataset's slug); None with
# RUN, so a run queued without one of its own refuses rather than attach the
# last run's dataset.
RUN_ID: str | None = None
# What the queued run is for, printed at launch; set with RUN.
PURPOSE = ""
# The queued run: `traffic_launch.py kaggle` arguments, without --slug.
RUN: list[str] | None = None


def main() -> None:
    """Launch the queued run, or refuse when none is queued, it has no id of
    its own, or flags are given."""
    if len(sys.argv) > 1:
        raise SystemExit(f"kaggle_next.py takes no arguments (got {sys.argv[1:]}); the run is RUN in the file")
    if RUN is None:
        raise SystemExit("no Kaggle run is queued on this branch (RUN is None)")
    if RUN_ID is None:
        raise SystemExit("the queued run has no id of its own (RUN_ID is None); set a new one beside RUN")
    launcher = pathlib.Path(__file__).with_name("traffic_launch.py")
    sys.argv = [str(launcher), "kaggle", *RUN, "--slug", SLUG, "--run-id", RUN_ID]
    print(f"launching to {SLUG}: {PURPOSE}\n  {' '.join(sys.argv[1:])}", flush=True)
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
