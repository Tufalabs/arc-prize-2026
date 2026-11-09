"""Read a checkpoint into the page cache a bounded window ahead of the engine's loader.

On a Kaggle kernel the weights sit on a network mount, and the engine's loader
reads them one shard at a time: it maps a shard, copies its tensors out, and
moves on (``safetensors_weights_iterator``, shards in natural filename order,
the ones the index names). Faulting each in over the network is slow, and the
engine's own prefetch (``safetensors_load_strategy="prefetch"``) reads every
shard at once with no bound: the checkpoint (100.5 GiB) is larger than the
RAM left once the PLE table is pinned (69 GiB on the RTX box, 59 on the four
L4s, at load), so pages read early are evicted before the loader reaches
them, and it starts only once the engine loads, after the install (265 s into
an RTX kernel, 547 s into a four-L4 one, where the load then takes 220 and
335-390 s).

This reader starts from the kernel's first setup command, so it reads while
the engine is still being installed, in the loader's order, and reads at most
a window of bytes past the shard the loader has open. It finds the loader by
the shards each process maps (``/proc/*/maps``; a process counts while it
keeps mapping, the slowest one sets the position), and drops the pages of the shards
the loader has passed (``POSIX_FADV_DONTNEED``) so that read-ahead is not the
first thing evicted. When the loader has finished, every shard's pages are
dropped: the weights then live on the card and in pinned memory. It writes a
report of every shard's read and of the loader's progress beside the run's
outputs.

Started detached (``ullm.kaggle.kaggle_prefetch_command``); it imports nothing
beyond the standard library, since it runs before the engine is installed. A
cell running it sets the engine's own strategy to ``lazy``, so the two do not
read the same bytes twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import sys
import threading
import time
from collections.abc import Sequence

GIB = 1 << 30
INDEX = "model.safetensors.index.json"


def _natural_sort_key(filepath: str) -> list[int | str]:
    """vLLM's own key (``weight_utils._natural_sort_key``)."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(filepath))]


def resolve_checkpoint_dir(path: str) -> pathlib.Path | None:
    """A configured checkpoint path as the kernel mounted it: the path itself,
    or the mount TAAF's input map gives for it (``vllm_factory``'s rule).
    Inputs are mounted before the notebook runs, so absence is final."""
    if os.path.isdir(path):
        return pathlib.Path(path)
    mounts = json.loads(os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "{}") or "{}")
    for ref, mounted in mounts.items():
        if path.rstrip("/").endswith("/" + ref) and os.path.isdir(str(mounted)):
            return pathlib.Path(str(mounted))
    base = os.path.basename(path.rstrip("/"))
    for mounted in mounts.values():
        if os.path.basename(str(mounted).rstrip("/")) == base and os.path.isdir(str(mounted)):
            return pathlib.Path(str(mounted))
    return None


def loader_order(directory: pathlib.Path) -> list[pathlib.Path]:
    """The shards the loader reads, in its order: the ones the index names
    (all of them without an index), natural-sorted by file name."""
    shards = [p for p in directory.glob("*.safetensors") if p.is_file()]
    index = directory / INDEX
    if index.is_file():
        named = set(json.loads(index.read_text())["weight_map"].values())
        shards = [p for p in shards if p.name in named]
    return sorted(shards, key=lambda p: _natural_sort_key(str(p)))


class Shard:
    """One checkpoint file in the loader's order, and when it was read, reached, passed and dropped."""

    def __init__(self, index: int, path: pathlib.Path) -> None:
        self.index = index
        self.path = path
        self.names = {str(path), os.path.realpath(path)}
        self.size = path.stat().st_size
        self.read_start: float | None = None
        self.read_end: float | None = None
        self.reached: float | None = None
        self.passed: float | None = None
        self.dropped: float | None = None

    def report(self, t0: float) -> dict[str, object]:
        """This shard's times, in seconds from ``t0``."""

        def rel(t: float | None) -> float | None:
            """``t`` in seconds from ``t0``, rounded, or None."""
            return None if t is None else round(t - t0, 1)

        return {
            "shard": str(self.path),
            "gib": round(self.size / GIB, 3),
            "read_start": rel(self.read_start),
            "read_end": rel(self.read_end),
            "loader_reached": rel(self.reached),
            "loader_passed": rel(self.passed),
            "dropped": rel(self.dropped),
        }


def _read(path: pathlib.Path, size: int, threads: int, block: int) -> None:
    """Read a whole file into the page cache, ``threads`` ranges at a time."""
    fd = os.open(path, os.O_RDONLY)

    def read_at(offset: int) -> int:
        """Read one block at ``offset``; its length."""
        return len(os.pread(fd, block, offset))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(read_at, range(0, size, block)):
                pass
    finally:
        os.close(fd)


def _drop(path: pathlib.Path) -> None:
    """Drop a file's pages from the page cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _mapped_shards(by_name: dict[str, Shard]) -> dict[int, int]:
    """Per process, the furthest shard it has mapped right now."""
    furthest: dict[int, int] = {}
    me = os.getpid()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            with open(f"/proc/{entry.name}/maps") as maps:
                for line in maps:
                    # address perms offset dev inode [path]
                    fields = line.rstrip("\n").split(None, 5)
                    shard = by_name.get(fields[5].strip()) if len(fields) == 6 else None
                    if shard is not None:
                        pid = int(entry.name)
                        furthest[pid] = max(furthest.get(pid, -1), shard.index)
        except OSError:
            continue
    return furthest


class Prefetcher:
    """The window ahead of the loader, and the loader's progress through it."""

    def __init__(self, shards: list[Shard], *, window: int, threads: int, block: int, active_s: float) -> None:
        self.shards = shards
        self.by_name = {name: s for s in shards for name in s.names}
        self.window = window
        self.threads = threads
        self.block = block
        self.active_s = active_s
        self.t0 = time.monotonic()
        # pid -> (furthest shard it has mapped, when it first mapped that one)
        self.loaders: dict[int, tuple[int, float]] = {}
        self.last_mapping = 0.0
        # The slowest active loader's shard, never moving back; 0 before any
        # loader shows.
        self.position = 0
        self.lock = threading.Lock()

    def watch(self, poll_s: float, stop: threading.Event) -> None:
        """Track the loaders. A process counts while it keeps mapping shards
        it had not mapped before, so one that opened a shard once (a header
        read) and moved on does not hold the position back, and the engine's
        lazy load, which keeps every shard it read mapped for the whole run,
        does not read as a loader that never finishes."""
        while not stop.is_set():
            now = time.monotonic()
            mapped = _mapped_shards(self.by_name)
            with self.lock:
                for pid, index in mapped.items():
                    if index > self.loaders.get(pid, (-1, now))[0]:
                        self.loaders[pid] = (index, now)
                        self.last_mapping = now
                active = [index for index, seen in self.loaders.values() if now - seen <= self.active_s]
                if active:
                    self.position = max(self.position, min(active))
                furthest = max((index for index, _ in self.loaders.values()), default=-1)
                for shard in self.shards:
                    if shard.reached is None and shard.index <= furthest:
                        shard.reached = now
                    if shard.passed is None and shard.index < self.position:
                        shard.passed = now
            stop.wait(poll_s)

    def loader_done(self, quiet_s: float) -> bool:
        """The loaders reached the last shard and have mapped none for a while."""
        with self.lock:
            return self.position >= self.shards[-1].index and time.monotonic() - self.last_mapping > quiet_s

    def next_to_read(self) -> Shard | None:
        """The next unread shard: the loader's own, or one ending within the
        window past it."""
        with self.lock:
            start = self.position
        ahead = 0
        for shard in self.shards[start:]:
            if shard.index > start:
                ahead += shard.size
            if shard.read_start is None:
                return shard if ahead <= self.window else None
        return None

    def drop_passed(self, *, everything: bool = False) -> None:
        """Drop the pages of every shard the loader has passed (all of them, at the end)."""
        for shard in self.shards:
            if shard.dropped is None and (everything or shard.passed is not None):
                try:
                    _drop(shard.path)
                    shard.dropped = time.monotonic()
                except OSError:
                    pass

    def run(self, *, poll_s: float, deadline_s: float, quiet_s: float) -> str:
        """Read ahead until the loader is done or the deadline passes; returns which."""
        stop = threading.Event()
        watcher = threading.Thread(target=self.watch, args=(poll_s, stop), daemon=True)
        watcher.start()
        outcome = "deadline"
        try:
            while time.monotonic() - self.t0 < deadline_s:
                if self.loader_done(quiet_s):
                    outcome = "loader finished"
                    break
                self.drop_passed()
                shard = self.next_to_read()
                if shard is None:
                    time.sleep(poll_s)
                    continue
                shard.read_start = time.monotonic()
                try:
                    _read(shard.path, shard.size, self.threads, self.block)
                except OSError as exc:
                    print(f"[prefetch] reading {shard.path.name} failed: {exc!r}", flush=True)
                shard.read_end = time.monotonic()
                print(
                    f"[prefetch] {shard.index + 1}/{len(self.shards)} {shard.path.name}: "
                    f"{shard.size / GIB:.2f} GiB in {shard.read_end - shard.read_start:.1f} s "
                    f"(loader at {self.position + 1})",
                    flush=True,
                )
        finally:
            stop.set()
            watcher.join(timeout=5)
            self.drop_passed(everything=outcome == "loader finished")
        return outcome


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve the checkpoints, read them ahead of the loader, write the report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("checkpoints", nargs="+", help="checkpoint directories, in the order the engine loads them")
    # About 40% of the RAM the four-L4 box has left at load (59 GiB): two of
    # a 10.7 GB shard past the loader's own. Not tuned.
    parser.add_argument("--window-gib", type=float, default=24.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--block-mib", type=int, default=16)
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--active-s", type=float, default=60.0)
    parser.add_argument("--quiet-s", type=float, default=30.0)
    parser.add_argument("--deadline-s", type=float, default=5400.0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    shards: list[Shard] = []
    for configured in args.checkpoints:
        directory = resolve_checkpoint_dir(configured)
        if directory is None:
            print(f"[prefetch] {configured} not mounted; skipped", flush=True)
            continue
        for path in loader_order(directory):
            shards.append(Shard(len(shards), path))
    if not shards:
        print("[prefetch] no shards to read", flush=True)
        return
    total = sum(s.size for s in shards)
    print(
        f"[prefetch] {len(shards)} shards, {total / GIB:.1f} GiB, window {args.window_gib:g} GiB, "
        f"{args.threads} threads",
        flush=True,
    )
    prefetcher = Prefetcher(
        shards,
        window=int(args.window_gib * GIB),
        threads=args.threads,
        block=args.block_mib << 20,
        active_s=args.active_s,
    )
    outcome = prefetcher.run(poll_s=args.poll_s, deadline_s=args.deadline_s, quiet_s=args.quiet_s)
    report = {
        "outcome": outcome,
        "elapsed_s": round(time.monotonic() - prefetcher.t0, 1),
        "window_gib": args.window_gib,
        "threads": args.threads,
        "read_gib": round(sum(s.size for s in shards if s.read_end is not None) / GIB, 2),
        "loader_pids": sorted(prefetcher.loaders),
        "shards": [s.report(prefetcher.t0) for s in shards],
    }
    path = args.report or os.path.join(
        os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working"), "shard_prefetch.json"
    )
    try:
        pathlib.Path(path).write_text(json.dumps(report, indent=1))
    except OSError as exc:
        print(f"[prefetch] report not written: {exc!r}", flush=True)
    print(f"[prefetch] {outcome} after {report['elapsed_s']} s; {report['read_gib']} GiB read", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
