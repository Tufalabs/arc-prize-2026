"""Packed-dataset cache + node-local trace staging for the SFT pipeline.

Building a dataset from raw TAAF traces is NFS-hostile: `taaf_trace_to_records`
eagerly opens every referenced media file (`sft/dataset.py:_resolve_image`) --
~1.4k tiny PNGs per trace, one network round-trip each -- and `sft.train`
builds twice per job (the --check-data pass, then the train pass). On a
contended filer that measured ~300 opens/min: a 480-trace build would take
a day per pass.

Two moves fix it, per Kander storage best practices:

- `load_or_build` packs the built dataset once with `Dataset.save_to_disk`
  into a durable cache dir on /data -- a few large arrow files with the media
  bytes embedded, the layout NFS is good at -- keyed by the selected trace
  dirs + requested mode. Every later build (the train pass of the same job,
  and future jobs on the same selection) loads the pack in seconds and never
  touches the raw traces.
- On a cache miss, `stage_trace_dirs` first copies the raw traces to
  node-local scratch with a bounded thread pool. Concurrency is what beats
  NFS per-file latency (archive formats don't: tar reads the source one file
  at a time too), and the build then does its many small opens against the
  local SSD.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_EX, flock
from pathlib import Path

from sft.taaf_adapter import build_dataset_from_trace_dirs

# Bump to invalidate existing packs when the record schema built by
# sft/taaf_adapter.py changes shape. v3: empty-completion guard covers
# token-less steps + JSON-null output_text; policy is part of the key.
CACHE_FORMAT_VERSION = 3

_META_NAME = "sft_pack.json"
_STAGE_MARKER = ".staged"


def _effective_empty_completion_policy() -> str:
    policy = os.environ.get("SFT_ON_EMPTY_COMPLETION", "error").strip().lower()
    if policy not in {"error", "skip", "keep"}:
        raise ValueError(
            "SFT_ON_EMPTY_COMPLETION must be one of ['error', 'keep', 'skip'], "
            f"got {policy!r}"
        )
    return policy


def cache_key(trace_dirs: list[str | Path], mode: str) -> str:
    """Content key for one selection: resolved trace dirs + requested mode +
    every knob that shapes the built records (a pack built under `skip` must
    not be silently reused by a run configured `error`).

    Trace folders are immutable once their `_SUCCESS` marker exists, so the
    sorted absolute paths identify the inputs; no per-file hashing needed.
    """
    payload = json.dumps(
        {
            "version": CACHE_FORMAT_VERSION,
            "mode": mode,
            "on_empty_completion": _effective_empty_completion_policy(),
            "dirs": sorted(str(Path(d).resolve()) for d in trace_dirs),
        }
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def stage_trace_dirs(
    trace_dirs: list[str | Path], stage_root: str | Path, *, workers: int | None = None
) -> list[Path]:
    """Copy trace dirs to node-local storage with a bounded thread pool.

    Returns the staged paths (same order as the input). Each dest is the trace
    basename suffixed with a hash of the resolved source path: basenames are
    only unique within one source, and sequential invocations (train then eval)
    share the stage root -- a bare-name dest would let the second split hit the
    first's `.staged` marker and silently reuse its data. Copies only what the
    pipeline reads -- steps.jsonl, _SUCCESS, trace.json, media/* -- skipping
    bulky extras like intermediate_states.pkl. Each file is one pool task so
    many NFS opens are in flight at once. A trace with a `.staged` marker is
    skipped, making the second build pass of a job (and retries) free.
    """
    if workers is None:
        try:
            workers = int(os.environ.get("SFT_STAGE_WORKERS", "32"))
        except ValueError as exc:
            raise ValueError("SFT_STAGE_WORKERS must be a positive integer") from exc
    if workers <= 0:
        raise ValueError(f"stage workers must be positive, got {workers}")

    stage_root = Path(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    # DDP ranks or two retrying processes can reach the same node-local stage.
    # Serialize the marker check + copy + marker publication: without the lock,
    # one process could observe another's marker while its copyfile calls were
    # still truncating the same destinations. flock is released on process exit,
    # so a killed stager cannot leave a permanent lock.
    with (stage_root / ".stage.lock").open("a+") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        return _stage_trace_dirs_locked(trace_dirs, stage_root, workers=workers)


def _stage_trace_dirs_locked(
    trace_dirs: list[str | Path], stage_root: Path, *, workers: int
) -> list[Path]:
    """Implementation of :func:`stage_trace_dirs` while its process lock is held."""

    staged: list[Path] = []
    jobs: list[tuple[Path, Path]] = []
    pending_markers: list[Path] = []
    pending_sources: dict[Path, Path] = {}
    for src in (Path(d) for d in trace_dirs):
        resolved_src = src.resolve()
        digest = hashlib.sha256(str(resolved_src).encode()).hexdigest()[:16]
        dest = stage_root / f"{src.name}-{digest}"
        staged.append(dest)
        if (dest / _STAGE_MARKER).exists():
            continue
        previous_source = pending_sources.get(dest)
        if previous_source is not None:
            if previous_source != resolved_src:
                raise RuntimeError(
                    f"trace stage destination collision: {previous_source} and {resolved_src} -> {dest}"
                )
            # Preserve duplicate input entries in the returned path list (and
            # therefore intentional dataset weighting), but copy the source once.
            continue
        pending_sources[dest] = resolved_src
        if not src.is_dir():
            raise FileNotFoundError(f"trace directory does not exist: {src}")
        missing = [name for name in ("steps.jsonl", "_SUCCESS") if not (src / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"cannot stage incomplete trace {src}: missing {', '.join(missing)}"
            )
        # A killed previous attempt has no marker. Rebuild it from scratch so
        # deleted/renamed source files cannot survive in a stale destination.
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "media").mkdir(parents=True, exist_ok=True)
        for name in ("steps.jsonl", "_SUCCESS", "trace.json"):
            if (src / name).exists():
                jobs.append((src / name, dest / name))
        media = src / "media"
        if media.is_dir():
            with os.scandir(media) as entries:
                jobs.extend(
                    (Path(entry.path), dest / "media" / entry.name)
                    for entry in entries
                    if entry.is_file()
                )
        pending_markers.append(dest / _STAGE_MARKER)

    if not pending_markers:
        return staged

    start = time.monotonic()
    print(
        f"staging {len(pending_markers)} trace dir(s), {len(jobs)} file(s) "
        f"-> {stage_root} ({workers} workers)"
    )
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda job: shutil.copyfile(job[0], job[1]), jobs):
            done += 1
            if done % 20000 == 0:
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f"  staged {done}/{len(jobs)} file(s) ({rate:.0f}/s)")
    for marker in pending_markers:
        marker.write_text("", encoding="utf-8")
    print(f"staged {len(jobs)} file(s) in {time.monotonic() - start:.0f}s")
    return staged


def load_or_build(
    name: str,
    trace_dirs: list[str | Path],
    *,
    mode: str,
    cache_dir: str | Path | None = None,
    stage_dir: str | Path | None = None,
):
    """(datasets.Dataset, resolved_mode) for a selection, via the pack cache.

    `cache_dir=None` builds from the raw traces every time (previous
    behavior). Otherwise: load the pack on a hit; on a miss, optionally stage
    the raw traces to `stage_dir` first, build from the (staged) copies, and
    save the pack under `cache_dir/<name>-<key>` for every run after this one.
    """
    dirs = [Path(d) for d in trace_dirs]
    if cache_dir is None:
        return build_dataset_from_trace_dirs(dirs, mode=mode)

    cache_dir = Path(cache_dir)
    packed = cache_dir / f"{name}-{cache_key(dirs, mode)}"
    meta_path = packed / _META_NAME
    if meta_path.exists():
        from datasets import load_from_disk

        meta = json.loads(meta_path.read_text())
        print(
            f"{name}: loading packed dataset {packed} "
            f"(mode={meta['resolved_mode']}, rows={meta['rows']})"
        )
        return load_from_disk(str(packed)), meta["resolved_mode"]

    if stage_dir is not None:
        dirs = stage_trace_dirs(dirs, stage_dir)

    # Create (and fail on an unwritable) cache dir BEFORE the expensive build.
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Stream the build to Arrow: peak RSS stays near one trace instead of the
    # whole selection (which OOMs on full-run builds). The Arrow workspace
    # should be LOCAL disk (SFT_BUILD_WORK_DIR, e.g. $SCRATCH_DIR in jobs):
    # num_proc shard writers renaming/rmtree-ing on NFS hit the lingering
    # .nfs-file race ("Directory not empty"). Only the finished pack is
    # streamed to the durable cache dir (large sequential writes, NFS-fine).
    work_root = Path(os.environ.get("SFT_BUILD_WORK_DIR") or cache_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    build_dir = work_root / f".{packed.name}.build.{os.getpid()}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    dataset, resolved = build_dataset_from_trace_dirs(dirs, mode=mode, work_dir=build_dir)

    tmp = cache_dir / f".{packed.name}.tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    dataset.save_to_disk(str(tmp))
    (tmp / _META_NAME).write_text(
        json.dumps(
            {
                "version": CACHE_FORMAT_VERSION,
                "requested_mode": mode,
                "resolved_mode": resolved,
                "rows": len(dataset),
                "trace_dirs": sorted(str(d) for d in trace_dirs),
            },
            indent=2,
        )
    )
    rows = len(dataset)
    try:
        tmp.rename(packed)
        print(f"{name}: packed dataset -> {packed} ({rows} rows)")
    except OSError:
        # Only a completed pack at the target proves a concurrent run won; any
        # other OSError (perms, NFS) must surface now, not as a confusing
        # load_from_disk failure below.
        if not (packed / _META_NAME).exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        shutil.rmtree(tmp)
    # Return the packed copy (memory-mapped) so the streaming build dir can be
    # reclaimed -- the built dataset maps Arrow files inside build_dir.
    from datasets import load_from_disk

    dataset = load_from_disk(str(packed))
    shutil.rmtree(build_dir, ignore_errors=True)
    return dataset, resolved
