"""Zip an existing job dir's span streams and block store, the way a run on
scratch leaves them (``platypus.diagnostics``, "Scratch and archive"), and
delete the loose copies: a 15-arm sweep job's ~40k files become one.

    uv run python tools/archive_job_diagnostics.py JOB_DIR [JOB_DIR ...]
    uv run python tools/archive_job_diagnostics.py --dry-run JOB_DIR

What moves is what a scratch run archives -- ``games/<run>/spans.jsonl``,
``run_spans.jsonl`` and ``diag_blocks/`` -- into the job dir's next part,
``diagnostics-NNNN.zip``, under the same job-dir-relative names, so every
platypus reader finds them as before. The rest of a game dir (capture,
references) stays where it is.

Nothing is deleted until the archive checks out without the loose files:
every new member's CRC and size, every stream parsed, and every block a
stream references resolved from the job's parts alone. A failure leaves the
loose files as they were and the part beside them, where the readers prefer
the loose copy.

Resumable: a job an interrupted run left half-deleted already has its part,
and a loose file whose member there matches it (size and CRC) is only
deleted; anything else goes into a new part.

A job without ``end_time`` in ``benchmark.json`` is refused: it may still be
writing. ``--force`` archives one that was killed before it could stamp it.
NFS makes both halves latency-bound (a read and an unlink per small file), so
both run on a thread pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from platypus.diagnostics import (
    ARCHIVE_PART_GLOB,
    BLOCKS_DIR,
    BlockStore,
    DiagnosticsArchive,
    archive_parts,
    blocks_dir,
    parse_events,
    run_stream_path,
)

#: Files read ahead of the zip writer; bounds what is held in memory.
READ_AHEAD = 512


def loose_files(job_dir: Path) -> list[str]:
    """The job-dir-relative paths of every loose stream and block, sorted."""
    found: list[str] = []
    games = job_dir / "games"
    if games.is_dir():
        found += [f"games/{d.name}/spans.jsonl" for d in os.scandir(games) if Path(d.path, "spans.jsonl").is_file()]
    if run_stream_path(job_dir).is_file():
        found.append(run_stream_path(job_dir).name)
    blocks = blocks_dir(job_dir)
    if blocks.is_dir():
        for entry in os.scandir(blocks):
            if entry.is_dir():
                found += [
                    f"{BLOCKS_DIR}/{entry.name}/{b.name}"
                    for b in os.scandir(entry.path)
                    if not b.name.startswith(".tmp-")
                ]
            elif entry.name.startswith("pack-"):
                found.append(f"{BLOCKS_DIR}/{entry.name}")
    return sorted(found)


def finished(job_dir: Path) -> bool:
    """Whether the job stamped its end: nothing writes to it any more."""
    try:
        return bool(json.loads((job_dir / "benchmark.json").read_text()).get("end_time"))
    except (OSError, ValueError):
        return False


def archived_members(job_dir: Path) -> dict[str, tuple[int, int]]:
    """(size, CRC) of every member of the job's parts, the latest part winning."""
    members: dict[str, tuple[int, int]] = {}
    for part in archive_parts(job_dir):
        with zipfile.ZipFile(part) as zf:
            for info in zf.infolist():
                members[info.filename] = (info.file_size, info.CRC)
    return members


def already_archived(job_dir: Path, relpaths: list[str], workers: int) -> set[str]:
    """The loose files an earlier part already holds byte for byte (size and CRC)."""
    members = archived_members(job_dir)
    candidates = [r for r in relpaths if r in members and members[r][0] == (job_dir / r).stat().st_size]

    def same(relpath: str) -> bool:
        return zlib.crc32((job_dir / relpath).read_bytes()) == members[relpath][1]

    with ThreadPoolExecutor(workers) as pool:
        return {r for r, ok in zip(candidates, pool.map(same, candidates), strict=True) if ok}


def write_part(job_dir: Path, relpaths: list[str], workers: int) -> Path:
    """Write the next part holding ``relpaths``, read on ``workers`` threads."""
    part = job_dir / ARCHIVE_PART_GLOB.replace("*", f"{len(archive_parts(job_dir)) + 1:04d}")
    tmp = part.with_name(f".{part.name}.tmp")

    def read(relpath: str) -> tuple[str, bytes, float]:
        path = job_dir / relpath
        return relpath, path.read_bytes(), path.stat().st_mtime

    with ThreadPoolExecutor(workers) as pool, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for start in range(0, len(relpaths), READ_AHEAD):
            for relpath, data, mtime in pool.map(read, relpaths[start : start + READ_AHEAD]):
                info = zipfile.ZipInfo(relpath, date_time=time.localtime(mtime)[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, data, compresslevel=1)
    os.replace(tmp, part)
    return part


def verify(job_dir: Path, part: Path | None, relpaths: list[str]) -> str | None:
    """Why the job's parts cannot stand in for the loose files, or None when
    they can.

    The new part's members are checked against their CRCs, and every loose
    file must be in some part at its size; then every stream in the parts is
    parsed and every block it references resolved from the parts alone. Not
    through the readers: they prefer a loose file, so they would check the
    loose copies instead.

    What is asked of the parts is what the job dir could answer BEFORE they
    were written -- nothing is lost, rather than everything resolves. A block
    a stream references can be missing for reasons of its own: a Kaggle pull
    that fetched the streams without the block store has none of them, and
    demanding them of the archive would refuse a job whose loose copy is no
    better."""
    if part is not None:
        with zipfile.ZipFile(part) as zf:
            bad = zf.testzip()
            if bad is not None:
                return f"CRC mismatch in {bad}"
    members = archived_members(job_dir)
    for relpath in relpaths:
        if relpath not in members:
            return f"{relpath} is in no part"
        if members[relpath][0] != (job_dir / relpath).stat().st_size:
            return f"{relpath}: {members[relpath][0]} bytes archived, {(job_dir / relpath).stat().st_size} on disk"
    archive = DiagnosticsArchive(archive_parts(job_dir))
    archived = BlockStore(job_dir / ".no-loose-blocks", archive=archive)  # the parts alone
    # The loose copies are still here (nothing is deleted before this passes),
    # so this store is the job dir as a reader finds it today.
    loose = BlockStore(blocks_dir(job_dir), archive=archive)
    streams = [name for name in archive.names() if name.endswith(".jsonl")]
    n_blocks = missing_already = 0
    for name in streams:
        for event in parse_events(archive.read(name) or b""):
            if event.get("e") != "artifact":
                continue
            try:
                archived.raw_bytes(event["block"])
            except FileNotFoundError:
                try:
                    loose.raw_bytes(event["block"])
                except FileNotFoundError:
                    missing_already += 1
                    continue
                return f"block {event['block']} resolves in the job dir but not from its parts"
            n_blocks += 1
    already = f", {missing_already} the job dir never had" if missing_already else ""
    print(f"  verified: CRCs, sizes, {len(streams)} streams, {n_blocks} block references{already}")
    return None


def delete_loose(job_dir: Path, relpaths: list[str], workers: int) -> None:
    """Unlink every archived file, then the directories that emptied."""

    def unlink(relpath: str) -> None:
        (job_dir / relpath).unlink()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(unlink, relpaths))
    prune_empty_dirs(job_dir)


def prune_empty_dirs(job_dir: Path) -> int:
    """Remove what is left of the stream and block trees where it holds nothing.

    Bottom-up over the whole tree rather than over the parents of what this run
    deleted: a job an earlier run was killed in has directories it emptied and
    never came back to. A game dir that still holds capture or reference files
    stays, and so does the tree above it."""
    removed = 0
    for root in (job_dir / "games", blocks_dir(job_dir)):
        if not root.is_dir():
            continue
        for dirpath, _, _ in os.walk(root, topdown=False):
            try:
                Path(dirpath).rmdir()
                removed += 1
            except OSError:
                pass  # not empty (a game dir's capture, say) or already gone
    return removed


def archive_job(job_dir: Path, *, force: bool, dry_run: bool, workers: int) -> bool:
    """Archive one job dir; whether it ended archived (or had nothing to do)."""
    print(f"{job_dir}:")
    if not finished(job_dir) and not force:
        print("  refused: no end_time in benchmark.json (still running? --force for a job that was killed)")
        return False
    t0 = time.monotonic()
    relpaths = loose_files(job_dir)
    if not relpaths:
        # Already archived. Its trees can still hold the empty directories an
        # interrupted run left behind, which cost a metadata round trip each to
        # list or delete.
        pruned = prune_empty_dirs(job_dir) if archive_parts(job_dir) and not dry_run else 0
        print(f"  nothing loose to archive{f'; pruned {pruned:,} empty directories' if pruned else ''}")
        return True
    size = sum((job_dir / r).stat().st_size for r in relpaths)
    print(f"  {len(relpaths):,} loose files, {size / 2**20:,.0f} MiB ({time.monotonic() - t0:.0f}s to list)")
    if dry_run:
        return True
    t1 = time.monotonic()
    done = already_archived(job_dir, relpaths, workers) if archive_parts(job_dir) else set[str]()
    if done:
        print(f"  {len(done):,} already in a part, byte for byte ({time.monotonic() - t1:.0f}s to check)")
    new = [r for r in relpaths if r not in done]
    part = None
    if new:
        t1 = time.monotonic()
        part = write_part(job_dir, new, workers)
        print(f"  wrote {part.name}: {part.stat().st_size / 2**20:,.0f} MiB in {time.monotonic() - t1:.0f}s")
    t2 = time.monotonic()
    problem = verify(job_dir, part, relpaths)
    if problem is not None:
        print(f"  NOT deleting anything: {problem}")
        return False
    print(f"  ({time.monotonic() - t2:.0f}s to verify)")
    t3 = time.monotonic()
    delete_loose(job_dir, relpaths, workers)
    print(f"  deleted the loose copies in {time.monotonic() - t3:.0f}s; {time.monotonic() - t0:.0f}s in all")
    return True


def main() -> None:
    """Archive every job dir named on the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("job_dirs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count what would move, change nothing")
    parser.add_argument("--force", action="store_true", help="archive a job without end_time (killed, not running)")
    parser.add_argument("--workers", type=int, default=32, help="threads for the NFS reads and unlinks")
    args = parser.parse_args()
    ok = [archive_job(d, force=args.force, dry_run=args.dry_run, workers=args.workers) for d in args.job_dirs]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
