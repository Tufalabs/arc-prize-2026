"""Diagnostics written to scratch and archived into the job dir
(platypus.diagnostics): what a periodic part takes, what waits for the last,
and every reader resolving a file wherever it ended up."""

from __future__ import annotations

import collections
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

import platypus.diagnostics
from platypus.diagnostics import (
    BlockStore,
    DiagnosticsArchive,
    DiagnosticsArchiver,
    archive_parts,
    blocks_dir,
    job_archive,
    job_block_store,
    job_file_version,
    read_job_file,
    register_scratch,
)


def _dead_pid() -> int:
    """The pid of a process that has exited."""
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def _store(scratch: Path, pid: int) -> BlockStore:
    """A block store under ``scratch`` whose packs are named for ``pid``."""
    store = BlockStore(blocks_dir(scratch))
    store._pack_path = blocks_dir(scratch) / f"pack-{pid}.bin"
    store._index_path = blocks_dir(scratch) / f"pack-{pid}.idx"
    return store


def _stream(scratch: Path, relpath: str, text: str = '{"e":"open"}\n') -> None:
    path = scratch / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_a_periodic_part_takes_what_is_final_and_every_pack_as_it_stands(tmp_path: Path) -> None:
    """Mid-run, a finished game's stream and a loose block go in, with the
    run stream and every pack as they stand, a live process's included; an
    unfinished game's stream waits for the last part."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    _stream(scratch, "games/done/spans.jsonl")
    _stream(scratch, "games/playing/spans.jsonl")
    _stream(scratch, "run_spans.jsonl")
    dead = _store(scratch, _dead_pid())
    dead.store({"small": 1}, "json")
    loose = dead.store({"large": "x" * 8192}, "json")
    live = _store(scratch, os.getpid())
    live.store({"small": 2}, "json")

    archiver = DiagnosticsArchiver(scratch, job_dir)
    part = archiver.archive(["done"])
    assert part is not None and part.name == "diagnostics-0001.zip"
    archive = job_archive(job_dir)
    assert archive is not None
    names = set(archive.names())
    assert "games/done/spans.jsonl" in names
    assert f"diag_blocks/{loose[:2]}/{loose}" in names
    assert {f"diag_blocks/{dead._pack_path.name}", f"diag_blocks/{dead._index_path.name}"} <= names
    assert {f"diag_blocks/{live._pack_path.name}", f"diag_blocks/{live._index_path.name}"} <= names
    assert "run_spans.jsonl" in names, "a running job's viewer needs the run's overview"
    assert "games/playing/spans.jsonl" not in names

    assert archiver.archive(["done"]) is None, "nothing new, no part"
    _stream(scratch, "run_spans.jsonl", '{"e":"open"}\n{"e":"close"}\n')  # the run went on
    assert archiver.archive(["done"]) is not None, "a grown run stream makes a part of its own"

    last = archiver.finish()
    assert last is not None and last.name == "diagnostics-0003.zip"
    assert not scratch.exists(), "the scratch dir goes once everything is archived"
    archive = job_archive(job_dir)
    assert archive is not None
    names = set(archive.names())
    assert "games/playing/spans.jsonl" in names
    assert archive.read("run_spans.jsonl") == b'{"e":"open"}\n{"e":"close"}\n', "the latest snapshot wins"


def test_a_finished_games_block_in_a_live_pack_reads_from_the_part(tmp_path: Path) -> None:
    """Dedup reaches across packs: a finished game's stream can reference a
    small block another, still-running process packed. The part that carries
    the stream carries that pack as it stood, so the block resolves from the
    archive alone; and a pack the live process appends to goes in again."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    live = _store(scratch, os.getpid())
    shared = live.store({"system prompt": "x" * 100}, "json")
    finished = _store(scratch, 4242)
    assert finished.store({"system prompt": "x" * 100}, "json") == shared, "found in the live pack, not packed twice"
    _stream(scratch, "games/done/spans.jsonl", json.dumps({"e": "artifact", "key": shared}) + "\n")

    archiver = DiagnosticsArchiver(scratch, job_dir)
    assert archiver.archive(["done"]) is not None
    store = BlockStore(blocks_dir(job_dir), archive=job_archive(job_dir))
    assert store.load(shared, "json") == {"system prompt": "x" * 100}

    later = live.store({"later": 1}, "json")
    assert archiver.archive([]) is not None, "the grown pack makes a part of its own"
    store = BlockStore(blocks_dir(job_dir), archive=job_archive(job_dir))
    assert store.load(later, "json") == {"later": 1}
    assert store.load(shared, "json") == {"system prompt": "x" * 100}


def test_open_parts_are_bounded_and_an_evicted_part_reopens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A viewer over the results tree reads many jobs over its life; the
    parts it has read are not held open one descriptor each for ever."""
    monkeypatch.setattr(platypus.diagnostics, "_OPEN_PARTS_MAX", 3)
    opened: collections.OrderedDict[tuple[Path, int], zipfile.ZipFile] = collections.OrderedDict()
    monkeypatch.setattr(platypus.diagnostics, "_OPEN_PARTS", opened)
    archives: list[DiagnosticsArchive] = []
    for i in range(6):
        scratch, job_dir = tmp_path / f"scratch{i}", tmp_path / f"job{i}"
        job_dir.mkdir()
        _stream(scratch, "run_spans.jsonl", f'{{"job":{i}}}\n')
        DiagnosticsArchiver(scratch, job_dir).finish()
        archive = job_archive(job_dir)
        assert archive is not None
        archives.append(archive)
    assert len(opened) == 3
    assert archives[0].read("run_spans.jsonl") == b'{"job":0}\n', "an evicted part is reopened on demand"
    assert len(opened) == 3


def test_blocks_read_back_from_the_archive_loose_and_packed(tmp_path: Path) -> None:
    """With the scratch dir gone, a large block (its own member) and a small
    one (inside an archived pack) both resolve through the job's store."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    writer = _store(scratch, _dead_pid())
    small = writer.store({"s": "x" * 100}, "json")
    large = writer.store({"s": "x" * 8192}, "json")
    DiagnosticsArchiver(scratch, job_dir).finish()

    store = job_block_store(job_dir)
    assert store.load(small, "json") == {"s": "x" * 100}
    assert store.load(large, "json") == {"s": "x" * 8192}
    assert store.size(small) == len(store.raw_bytes(small))
    assert not blocks_dir(job_dir).exists(), "nothing loose was written into the job dir"


def test_a_changed_pack_goes_in_again_and_the_later_part_wins(tmp_path: Path) -> None:
    """A pid a later process reuses appends to an archived pack; the next
    part carries the whole pack again, and a reader finds both blocks."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    pid = _dead_pid()
    first = _store(scratch, pid).store({"n": 1}, "json")
    archiver = DiagnosticsArchiver(scratch, job_dir)
    assert archiver.archive() is not None
    second = _store(scratch, pid).store({"n": 2}, "json")  # the reused pid appends
    assert archiver.finish() is not None

    assert len(archive_parts(job_dir)) == 2
    store = job_block_store(job_dir)
    assert store.load(first, "json") == {"n": 1}
    assert store.load(second, "json") == {"n": 2}


def test_a_job_store_is_shared_until_a_new_part_lands(tmp_path: Path) -> None:
    """Readers share one store per job, index parsed once; a part that
    lands mid-run gives the next reader a store that sees it."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    pid = _dead_pid()
    first = _store(scratch, pid).store({"n": 1}, "json")
    archiver = DiagnosticsArchiver(scratch, job_dir)
    assert archiver.archive() is not None
    store = job_block_store(job_dir)
    assert job_block_store(job_dir) is store and store._index is not None
    second = _store(scratch, pid).store({"n": 2}, "json")
    assert archiver.finish() is not None
    later = job_block_store(job_dir)
    assert later is not store
    assert later.load(first, "json") == {"n": 1} and later.load(second, "json") == {"n": 2}


def test_a_store_keeps_its_decompressed_packs_under_a_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A viewer's store lives as long as the viewer, so the archived packs it
    decompresses are kept by bytes: reading from more packs than the budget
    holds evicts the least recently read, and a block in an evicted pack
    still loads."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    # One block a pack, each a little over a KiB; the budget holds two packs.
    keys = [_store(scratch, _dead_pid()).store({"n": n, "pad": "x" * 1000}, "json") for n in range(3)]
    assert DiagnosticsArchiver(scratch, job_dir).finish() is not None
    monkeypatch.setattr(platypus.diagnostics, "ARCHIVED_PACK_CACHE_BYTES", 2500)
    store = job_block_store(job_dir)
    packs = [store._locate(key)[0] for key in keys]
    assert len(set(packs)) == 3 and all(isinstance(pack, str) for pack in packs), "three archived packs"

    assert [store.load(key, "json")["n"] for key in keys] == [0, 1, 2]
    assert list(store._archived_packs) == packs[1:], "the least recently read pack went"
    assert store._archived_pack_bytes <= 2500

    assert store.load(keys[0], "json")["n"] == 0, "an evicted pack is read again"
    assert list(store._archived_packs) == [packs[2], packs[0]]


def test_a_job_file_version_names_where_the_file_is_read_from(tmp_path: Path) -> None:
    """Loose, the file's own size and mtime; archived, the part holding it;
    nowhere, None — each without reading the file."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    _stream(scratch, "games/done/spans.jsonl")
    assert DiagnosticsArchiver(scratch, job_dir).archive(["done"]) is not None
    (part,) = archive_parts(job_dir)
    relpath = "games/done/spans.jsonl"
    assert job_file_version(job_dir, relpath) == (str(part), part.stat().st_size, 0)
    _stream(job_dir, relpath, '{"e":"open"}\n{"e":"close"}\n')
    loose = job_dir / relpath
    assert job_file_version(job_dir, relpath) == (str(loose), loose.stat().st_size, loose.stat().st_mtime_ns)
    assert job_file_version(job_dir, "games/absent/spans.jsonl") is None


def test_monitor_files_go_in_once_final_and_the_stall_log_whenever_it_grows(tmp_path: Path) -> None:
    """A closed pressure segment goes into one part; the live one waits for
    the last; the stall log goes in again after it grows; and every one
    resolves from the archive once the scratch dir is gone."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    _stream(scratch, "monitor/pressure-0001.jsonl", '{"t":1}\n')
    _stream(scratch, "monitor/pressure-0002.jsonl.part", '{"t":2}\n')
    _stream(scratch, "monitor/loop_stalls.log", "stall 1\n")

    archiver = DiagnosticsArchiver(scratch, job_dir)
    assert archiver.archive() is not None
    archive = job_archive(job_dir)
    assert archive is not None
    assert archive.names("monitor/") == ["monitor/loop_stalls.log", "monitor/pressure-0001.jsonl"]

    assert archiver.archive() is None, "nothing final has changed"
    _stream(scratch, "monitor/loop_stalls.log", "stall 1\nstall 2\n")
    assert archiver.archive() is not None, "a grown stall log makes a part of its own"

    (scratch / "monitor/pressure-0002.jsonl.part").rename(scratch / "monitor/pressure-0002.jsonl")
    assert archiver.finish() is not None
    assert platypus.diagnostics.monitor_files(job_dir) == [
        "monitor/loop_stalls.log",
        "monitor/pressure-0001.jsonl",
        "monitor/pressure-0002.jsonl",
    ]
    assert read_job_file(job_dir, "monitor/loop_stalls.log") == b"stall 1\nstall 2\n"


def test_a_half_written_block_is_not_archived(tmp_path: Path) -> None:
    """A loose block is written to a temp name and renamed; the temp file is
    never final."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    half = blocks_dir(scratch) / "ab" / ".tmp-1-abcdef"
    half.parent.mkdir(parents=True)
    half.write_bytes(b"partial")
    assert DiagnosticsArchiver(scratch, job_dir).archive() is None


def test_a_file_resolves_loose_then_on_scratch_then_archived(tmp_path: Path) -> None:
    """Loose in the job dir wins (an old or unarchived run), then this
    process's scratch for the job (written, not yet archived), then the
    archive."""
    scratch, job_dir = tmp_path / "scratch", tmp_path / "job"
    job_dir.mkdir()
    relpath = "games/r/spans.jsonl"
    _stream(scratch, relpath, "archived\n")
    DiagnosticsArchiver(scratch, job_dir).archive(["r"])
    assert read_job_file(job_dir, relpath) == b"archived\n"

    _stream(scratch, relpath, "on scratch\n")
    register_scratch(job_dir, scratch)
    try:
        assert read_job_file(job_dir, relpath) == b"on scratch\n"
        _stream(job_dir, relpath, "loose\n")
        assert read_job_file(job_dir, relpath) == b"loose\n"
    finally:
        register_scratch(job_dir, None)
    assert read_job_file(job_dir, "games/missing/spans.jsonl") is None
