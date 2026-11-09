"""Shared helpers for the compare-runs scripts.

The entry points map serve URLs onto the filesystem and decide whether a
directory is a run dir and whether it finished. Those rules live in one
place because `compare_runs.py` refuses to clear a directory that looks like
run data — a second copy of the test that drifted would delete real runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_ROOT = os.environ.get("ARC_RESULTS_ROOT", "/data/shared/research/arc_2_results")


def url_to_fs_path(url: str, root: str) -> Path:
    """Map a localhost serve_diagnostics URL (or a filesystem path) onto root.

    A filesystem path is taken as given; anything else is a serve path and is
    resolved against ``root``. Absoluteness is what decides, NOT existence:
    the output directory is normally created by the run, so an existence
    test sends every absolute ``out`` down the URL branch and silently
    writes to ``<root>/<the absolute path>`` — a nested duplicate tree that
    looks plausible in the log and is nowhere the caller asked for.
    """
    parts = urlsplit(url)
    if parts.scheme or parts.netloc:  # a real URL
        return Path(root.rstrip("/")) / parts.path.lstrip("/")
    path = Path(url)
    if path.is_absolute() or path.exists():
        return path
    return Path(root.rstrip("/")) / url.lstrip("/")


def is_run_dir(d: Path) -> bool:
    """A directory holding a benchmark snapshot in either layout."""
    return (d / "benchmark.json").exists() or (d / "benchmark.pkl").exists()


def is_finished(d: Path) -> bool:
    """Cheap finished check: benchmark.json with end_time set. Legacy
    pkl-only runs are assumed finished (the renderer refuses any that
    aren't).
    """
    j = d / "benchmark.json"
    if j.exists():
        try:
            with open(j) as f:
                return bool(json.load(f).get("end_time"))
        except Exception:
            return False
    return (d / "benchmark.pkl").exists()


def holds_run_data(d: Path) -> bool:
    """Whether any run dir lives at or under ``d``.

    The guard before a destructive clear. It must use the same test as
    :func:`is_run_dir`, or a tree of legacy pkl-only runs — a layout these
    scripts accept as input — passes the guard and is deleted.
    """
    if is_run_dir(d):
        return True
    return any(p.name in ("benchmark.json", "benchmark.pkl") for p in d.rglob("benchmark.*"))
