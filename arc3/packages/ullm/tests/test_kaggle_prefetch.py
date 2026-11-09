"""``ullm.kaggle_prefetch``: the reader a Kaggle kernel starts first, which
keeps a bounded window of checkpoint shards in the page cache ahead of the
engine's loader.

The loader here is a child process that maps the shards as the engine's
loader does, and the prefetcher finds it the way it does on a kernel, through
``/proc/*/maps``.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest
import ullm.kaggle_prefetch as kaggle_prefetch

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="finds the loader through /proc")


def _checkpoint(root: pathlib.Path, name: str, shards: int, *, index: bool = True) -> pathlib.Path:
    """A checkpoint directory of ``shards`` one-MiB files named as the Hub
    names them, with the index naming every one when ``index``."""
    directory = root / name
    directory.mkdir()
    weight_map = {}
    for i in range(1, shards + 1):
        file = f"model-{i:05d}-of-{shards:05d}.safetensors"
        (directory / file).write_bytes(os.urandom(1 << 20))
        weight_map[f"tensor{i}"] = file
    if index:
        (directory / kaggle_prefetch.INDEX).write_text(json.dumps({"weight_map": weight_map}))
    return directory


def test_the_shards_come_in_the_loader_s_order(tmp_path: pathlib.Path) -> None:
    """Natural order by file name, as vLLM sorts them, and only the files the
    index names; without an index, every shard."""
    target = _checkpoint(tmp_path, "target", 12)
    (target / "stray-1.safetensors").write_bytes(b"not in the index")
    assert [p.name for p in kaggle_prefetch.loader_order(target)] == [
        f"model-{i:05d}-of-00012.safetensors" for i in range(1, 13)
    ]

    plain = tmp_path / "plain"
    plain.mkdir()
    for name in ("model-10.safetensors", "model-9.safetensors"):
        (plain / name).write_bytes(b"x")
    assert [p.name for p in kaggle_prefetch.loader_order(plain)] == ["model-9.safetensors", "model-10.safetensors"]


def test_a_checkpoint_resolves_to_its_mount(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cell's path where it exists, else the mount TAAF's input map gives
    for the dataset it names, else the mount of the same name; a checkpoint
    with none is skipped."""
    mounted = tmp_path / "input" / "e307"
    mounted.mkdir(parents=True)
    monkeypatch.setenv("TAAF_KAGGLE_INPUT_PATHS", json.dumps({"owner/e307": str(mounted)}))
    assert kaggle_prefetch.resolve_checkpoint_dir(str(mounted)) == mounted
    assert kaggle_prefetch.resolve_checkpoint_dir("/kaggle/input/datasets/owner/e307") == mounted
    assert kaggle_prefetch.resolve_checkpoint_dir("/elsewhere/e307/") == mounted
    assert kaggle_prefetch.resolve_checkpoint_dir("/kaggle/input/datasets/owner/other") is None


# The engine's loader, as the prefetcher sees it: each shard mapped in turn.
# ``keep`` holds every mapping until it exits, as vLLM's lazy load does for the
# whole run; ``close`` drops each as it moves on. It is paced by marks the
# test's prefetcher leaves (``_marking``), not by a clock: a shard is mapped
# once it has been read, and left once the prefetcher has seen it mapped. On a
# busy node a scan of ``/proc`` can take longer than any fixed pace, and a
# loader that outruns it passes shards the window never reached.
LOADER = textwrap.dedent(
    """
    import mmap, os, pathlib, sys, time
    keep = sys.argv[1] == "keep"
    marks = pathlib.Path(sys.argv[2])
    held = []

    def wait_for(mark):
        deadline = time.monotonic() + 30
        while not (marks / mark).exists() and time.monotonic() < deadline:
            time.sleep(0.01)

    for index, path in enumerate(sys.argv[3:]):
        wait_for(f"read-{pathlib.Path(path).name}")
        with open(path, "rb") as fh:
            m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        wait_for(f"seen-{os.getpid()}-{index}")
        if keep:
            held.append(m)
        else:
            m.close()
    time.sleep(60)
    """
)


def _marking(monkeypatch: pytest.MonkeyPatch, marks: pathlib.Path) -> None:
    """The prefetcher leaves a mark in ``marks`` for every shard it has read
    and for every process's furthest shard each time it scans the mappings."""
    read = kaggle_prefetch._read  # pyright: ignore[reportPrivateUsage]
    mapped = kaggle_prefetch._mapped_shards  # pyright: ignore[reportPrivateUsage]

    def read_and_mark(path: pathlib.Path, size: int, threads: int, block: int) -> None:
        read(path, size, threads, block)
        (marks / f"read-{path.name}").touch()

    def mapped_and_mark(by_name: dict[str, kaggle_prefetch.Shard]) -> dict[int, int]:
        furthest = mapped(by_name)
        for pid, index in furthest.items():
            (marks / f"seen-{pid}-{index}").touch()
        return furthest

    monkeypatch.setattr(kaggle_prefetch, "_read", read_and_mark)
    monkeypatch.setattr(kaggle_prefetch, "_mapped_shards", mapped_and_mark)


def _wait_until_mapped(pid: int, path: pathlib.Path) -> None:
    """Until process ``pid`` maps ``path``: the prefetcher reads its first
    window as it starts, so a process meant to be there from the start has to
    be, however slowly it came up."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if str(path.resolve()) in pathlib.Path(f"/proc/{pid}/maps").read_text():
                return
        except OSError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"process {pid} never mapped {path}")


# A process that maps the first shard once (a header read) and keeps it
# mapped: it must not hold the window back.
HEADER_READER = textwrap.dedent(
    """
    import mmap, sys, time
    with open(sys.argv[1], "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    time.sleep(60)
    """
)


@pytest.mark.parametrize("keeps_mappings", [False, True], ids=["closes", "keeps"])
def test_the_window_follows_the_loader_and_ends_with_it(
    tmp_path: pathlib.Path, keeps_mappings: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shard is read at most the window ahead of the loader, every
    shard's pages are dropped and the report written once the loader stops
    mapping new shards -- also when it keeps all of them mapped, as the
    engine's lazy load does for the whole run: progress is a newly mapped
    shard, not a mapping held."""
    target = _checkpoint(tmp_path, "target", 6)
    drafter = _checkpoint(tmp_path, "drafter", 1, index=False)
    order = kaggle_prefetch.loader_order(target) + kaggle_prefetch.loader_order(drafter)
    report = tmp_path / "shard_prefetch.json"
    marks = tmp_path / "marks"
    marks.mkdir()
    _marking(monkeypatch, marks)
    header = subprocess.Popen([sys.executable, "-c", HEADER_READER, str(order[0])])
    loader = subprocess.Popen(
        [sys.executable, "-c", LOADER, "keep" if keeps_mappings else "close", str(marks), *map(str, order)]
    )
    try:
        _wait_until_mapped(header.pid, order[0])
        started = time.monotonic()
        kaggle_prefetch.main(
            [
                str(target),
                str(drafter),
                str(tmp_path / "not-mounted"),
                # Two of the one-MiB shards past the loader's own.
                *("--window-gib", str(2 / 1024)),
                *("--poll-s", "0.05", "--quiet-s", "1", "--active-s", "1.5", "--deadline-s", "60"),
                *("--report", str(report)),
            ]
        )
        elapsed = time.monotonic() - started
    finally:
        for child in (loader, header):
            child.kill()
            child.wait()

    result = json.loads(report.read_text())
    assert result["outcome"] == "loader finished"
    assert elapsed < 30, "the loader's mappings held the prefetcher to its deadline"
    assert loader.pid in result["loader_pids"]
    shards = result["shards"]
    assert [pathlib.Path(s["shard"]).name for s in shards] == [p.name for p in order]
    for s in shards:
        assert s["read_end"] is not None and s["loader_reached"] is not None and s["dropped"] is not None
    # The window: a shard is read once the loader is within two shards of it
    # (the report rounds to a tenth of a second).
    for i in range(2, len(shards)):
        assert shards[i]["read_start"] >= shards[i - 2]["loader_reached"] - 0.15, i
