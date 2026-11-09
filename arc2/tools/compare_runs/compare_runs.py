#!/usr/bin/env python3
"""Generate a TAAF cross-run comparison over the finished run folders
directly under a parent directory.

Usage:
    python compare_runs.py <parent_url_or_path> [out_url_or_path] \
        [--root R] [--only SUB ...] [--exclude SUB ...] [--dry-run]

`parent` and `out` may be serve_diagnostics localhost URLs
(http://localhost:8765/shared/...), absolute filesystem paths, or paths
relative to the serve root; URLs and relative paths map onto `--root`
(default /data/shared/research/arc_2_results, env ARC_RESULTS_ROOT). `out`
defaults to `<parent>/compare`.

Discovers the immediate child run dirs (those with a benchmark.json /
legacy benchmark.pkl), applies --only then --exclude name filters, splits
finished vs in-flight, clears `out`, and writes the comparison HTML from
the finished ones via taaf.diagnostics.regenerate_comparison_diagnostics.
In-flight runs are skipped (the comparison renderer refuses them) and
reported, as are name-filtered ones. Run it from this repo's venv
(`uv run python tools/compare_runs/compare_runs.py ...`) so `taaf` imports.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from _common import DEFAULT_ROOT, holds_run_data, is_finished, is_run_dir, url_to_fs_path


def main() -> int:
    """Select the finished children of the parent, clear the output, write the comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", help="parent dir / URL holding run subfolders")
    ap.add_argument("out", nargs="?", default=None, help="output dir / URL (default <parent>/compare)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="localhost serve root for URL mapping")
    ap.add_argument(
        "--only", nargs="*", default=[], metavar="SUB", help="keep only run dirs whose name contains one of these"
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[], metavar="SUB", help="drop run dirs whose name contains one of these"
    )
    ap.add_argument("--dry-run", action="store_true", help="list selection, do not write")
    args = ap.parse_args()

    parent = url_to_fs_path(args.parent, args.root)
    if not parent.is_dir():
        print(f"ERROR: parent is not a directory: {parent}", file=sys.stderr)
        return 2
    out = url_to_fs_path(args.out, args.root) if args.out else parent / "compare"

    finished: list[Path] = []
    in_flight: list[Path] = []
    filtered: list[tuple[Path, str]] = []
    for d in sorted(p for p in parent.iterdir() if p.is_dir()):
        if d.resolve() == out.resolve() or not is_run_dir(d):
            continue
        if args.only and not any(s in d.name for s in args.only):
            filtered.append((d, "not in --only"))
            continue
        if any(s in d.name for s in args.exclude):
            filtered.append((d, "matched --exclude"))
            continue
        (finished if is_finished(d) else in_flight).append(d)

    print(f"parent : {parent}")
    print(f"out    : {out}")
    print(f"finished ({len(finished)}):")
    for d in finished:
        print(f"  + {d.name}")
    print(f"in-flight, skipped ({len(in_flight)}):")
    for d in in_flight:
        print(f"  - {d.name}")
    print(f"name-filtered, skipped ({len(filtered)}):")
    for d, why in filtered:
        print(f"  x {d.name}  ({why})")

    if args.dry_run:
        return 0
    if not finished:
        print("\nNo finished run dirs selected -> nothing to compare.")
        return 1
    if len(finished) < 2:
        # The renderer refuses fewer than two unconditionally, so there is
        # nothing to write. Stop before the clear below: proceeding would
        # delete an existing comparison and then raise.
        print(
            f"\nERROR: only {len(finished)} finished run selected; the comparison renderer "
            f"needs at least 2. Leaving {out} untouched.",
            file=sys.stderr,
        )
        return 1

    # Safety: never rmtree a dir that holds real runs (e.g. out == parent,
    # or out accidentally points at a run tree). Legacy pkl-only runs count
    # -- this script accepts them as input.
    if out.exists() and holds_run_data(out):
        print(
            f"\nERROR: refusing to clear {out}: it holds benchmark.json / benchmark.pkl "
            f"(looks like a run tree, not a comparison output dir).",
            file=sys.stderr,
        )
        return 2

    # And never rmtree a directory this tool did not write. The check above
    # is negative -- it names the one shape that must not be deleted -- so
    # anything it does not recognize (a mistyped path, a directory of
    # analysis by hand) reaches the clear below and is gone. This one is
    # positive: a comparison is exactly what has an index.html at its root,
    # which is also what the write below produces, so an out dir either
    # holds a previous comparison to replace or is empty.
    if out.exists() and any(out.iterdir()) and not (out / "index.html").is_file():
        print(
            f"\nERROR: refusing to clear {out}: it is not empty and has no index.html, so it is "
            f"not a comparison this tool wrote. Point --out at a new directory, or delete that one "
            f"yourself if it really is disposable.",
            file=sys.stderr,
        )
        return 2

    import taaf.diagnostics  # lazy: only needed for the write path

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    taaf.diagnostics.regenerate_comparison_diagnostics([str(d) for d in finished], str(out))
    print(f"\nWrote comparison -> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
