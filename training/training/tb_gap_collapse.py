"""Collapse queue-wait gaps in a run's TensorBoard event files.

A run spanning several SLURM segments accumulates hours of queue wait
between segments; TensorBoard's RELATIVE axis counts that dead time, so
resumed runs read slower than ones that scheduled luckily. This rewrites
a run's event files with inter-event gaps above ``gap_threshold`` seconds
replaced by the run's median inter-event gap.

Used two ways: the trainer calls :func:`collapse_run_dir` on its own
tensorboard directory at RESUME (before the new segment's writer opens),
so raw logs self-heal; ``scripts/rebase_tensorboard_active_time.py``
applies it to a whole index for the parallel active-time view.

Files are swapped individually, never the directory: a serving
TensorBoard holds the old files open, and on NFS deleting an open file
leaves a ghost ``.nfs*`` entry that fails a directory-level replace.
Unlinking works even while open; the ghosts are not tfevents-named so
loaders ignore them.

The swap is crash-ordered: the staged replacements (their writer-generated
names are unique — fresh timestamp/pid) are installed into the output
directory FIRST, and the superseded originals are unlinked only after the
whole replacement set is in place. A manifest written into the staging dir
before any install lets the next invocation finish an interrupted swap, so
no crash point loses the run's event history.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_GAP_THRESHOLD_SECONDS = 2400.0  # > any real iteration gap, < any queue wait

_SUPERSEDED_MANIFEST_NAME = "superseded.json"


def _install_staged_events(staging: Path, output_dir: Path) -> set[str]:
    """Move every staged event file into place; returns the installed names."""
    installed: set[str] = set()
    for name in sorted(os.listdir(staging)):
        if "tfevents" not in name:
            continue
        os.replace(staging / name, output_dir / name)
        installed.add(name)
    return installed


def _recover_interrupted_swap(staging: Path, output_dir: Path) -> None:
    """Finish (or discard) the swap a crashed previous invocation left.

    The manifest is written into the staging dir before any replacement is
    installed, so: no manifest -> nothing was installed, the originals are
    intact and the partial staging output is safely discarded; manifest
    present -> the staged replacement set is complete, so finish the plan
    (install whatever is still staged, then unlink the superseded originals
    the manifest lists).
    """
    if not staging.is_dir():
        return
    manifest = staging / _SUPERSEDED_MANIFEST_NAME
    if manifest.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        installed = _install_staged_events(staging, output_dir)
        for name in json.loads(manifest.read_text(encoding="utf-8")):
            name = str(name)
            if name in installed:
                continue
            try:
                os.unlink(output_dir / name)
            except FileNotFoundError:
                pass
        log.warning(
            "finished an interrupted event-file swap in %s (recovered from %s)",
            output_dir,
            staging,
        )
    shutil.rmtree(staging, ignore_errors=True)


def collapse_run_dir(
    run_dir: str | Path,
    *,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD_SECONDS,
    output_dir: str | Path | None = None,
) -> int:
    """Rewrite ``run_dir``'s events with large gaps collapsed.

    Returns the number of gaps collapsed. ``output_dir`` defaults to
    ``run_dir`` (in-place, file-swapped). A run with no event files is a
    no-op. Call only when no writer holds the directory open (segment
    start, or an offline index pass). Crash-safe and idempotent: an
    interrupted swap is completed from its staged manifest before any
    events are read.
    """
    from tensorboard.backend.event_processing.event_file_loader import (
        EventFileLoader,
    )
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    if not run_dir.is_dir():
        return 0
    staging = output_dir.with_name(output_dir.name + ".gap-staging")
    # Before reading anything: a crash mid-swap may have left the output dir
    # holding originals AND replacements; finish that swap first.
    _recover_interrupted_swap(staging, output_dir)
    events = []
    source_names = []
    for name in sorted(os.listdir(run_dir)):
        if "tfevents" not in name:
            continue
        source_names.append(name)
        for event in EventFileLoader(str(run_dir / name)).Load():
            # Only summary-bearing events matter for plots; file-version /
            # metadata events carry file-creation wall_times (including the
            # rewrite's own "now"), which would read as phantom gaps. The
            # writer stamps every rewritten file with a fresh version header.
            if event.HasField("summary"):
                events.append(event)
    if not events:
        return 0
    events.sort(key=lambda e: e.wall_time)
    deltas = [
        b.wall_time - a.wall_time
        for a, b in zip(events, events[1:])
        if 0 < b.wall_time - a.wall_time <= gap_threshold
    ]
    if (
        not any(
            b.wall_time - a.wall_time > gap_threshold
            for a, b in zip(events, events[1:])
        )
        and output_dir == run_dir
    ):
        return 0  # nothing to collapse; leave the files untouched
    typical = statistics.median(deltas) if deltas else 1.0

    # EventFileWriter appends once per event; keep those small writes on
    # node-local TMPDIR, then copy closed files to the durable staging dir.
    # The manifest must only appear after every copy has finished so a crash
    # during the copy leaves the originals authoritative and intact.
    with tempfile.TemporaryDirectory(prefix="tensorboard-gap-") as local_dir:
        writer = EventFileWriter(local_dir)
        adjusted = events[0].wall_time
        previous = events[0].wall_time
        collapsed = 0
        try:
            for event in events:
                delta = event.wall_time - previous
                if delta > gap_threshold:
                    collapsed += 1
                    delta = typical
                adjusted += max(0.0, delta)
                previous = event.wall_time
                event.wall_time = adjusted
                writer.add_event(event)
        finally:
            writer.close()
        staging.mkdir(parents=True)
        for path in Path(local_dir).iterdir():
            if "tfevents" in path.name:
                shutil.copyfile(path, staging / path.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    superseded = (
        source_names
        if output_dir == run_dir
        else [name for name in os.listdir(output_dir) if "tfevents" in name]
    )
    # Crash-ordered swap: record the plan in the staging dir, install every
    # replacement, and only then unlink the superseded originals — a crash at
    # any point leaves either the intact original set (no manifest yet) or a
    # state _recover_interrupted_swap finishes on the next invocation.
    manifest_tmp = staging / (_SUPERSEDED_MANIFEST_NAME + ".tmp")
    manifest_tmp.write_text(json.dumps(sorted(superseded)), encoding="utf-8")
    os.replace(manifest_tmp, staging / _SUPERSEDED_MANIFEST_NAME)
    installed = _install_staged_events(staging, output_dir)
    for name in superseded:
        if name in installed:
            continue
        try:
            os.unlink(output_dir / name)
        except FileNotFoundError:
            pass
    shutil.rmtree(staging, ignore_errors=True)
    if collapsed:
        log.info(
            "collapsed %d queue-wait gap(s) in %s (threshold %.0fs)",
            collapsed,
            output_dir,
            gap_threshold,
        )
    return collapsed
