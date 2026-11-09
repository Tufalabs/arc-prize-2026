"""The host monitors (``platypus.host_monitor``): a blocked event loop is
reported with where it blocked, an idle one never is; an expected hold
is counted and only reported past its limit; the kernel's pressure,
reclaim, network-filesystem and disk counters read from their files and turn
into intervals; the sampler rolls segments and leaves none half-written; and
the readers work on this machine's own files.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
from pathlib import Path

import pytest

import platypus.host_monitor
from platypus.diagnostics import monitor_dir
from platypus.host_monitor import (
    STALLS_LOG,
    LoopStallWatchdog,
    PressureReader,
    PressureSampler,
    interval_record,
    read_diskstats,
    read_nfs_ops,
    read_pressure,
    read_psi,
)

_MOUNTSTATS = """\
device sysfs mounted on /sys with fstype sysfs
device server:/tank/shared mounted on /data/shared with fstype nfs4 statvers=1.1
\topts:\trw,vers=4.2
\tper-op statistics
\t        NULL: 1 1 0 44 24 0 0 0 0
\t        READ: {read} {read} 0 5461 1813 808 2795 {read_ms} 49
\t       WRITE: {write} {write} 0 6932 2333 4060 6141 {write_ms} 143
\t     GETATTR: 3235 3235 0 657 801 506 1135 1686 55
device tmpfs mounted on /run with fstype tmpfs
"""


def _block_the_loop(seconds: float) -> None:
    """Hold the event loop the way a blocking call made on it would."""
    time.sleep(seconds)


async def test_a_blocked_loop_is_reported_where_it_blocked(tmp_path: Path, capsys: object) -> None:
    """A call that holds the loop past the threshold opens one stall, whose
    snapshots name the blocking function on the loop thread with its kernel
    state; the stall closes when the loop runs again, with one stdout line."""
    watchdog = LoopStallWatchdog(tmp_path, threshold_s=0.3, repeat_s=0.2, tick_s=0.02, poll_s=0.02)
    watchdog.start()
    await asyncio.sleep(0.1)
    _block_the_loop(1.0)
    await asyncio.sleep(0.2)
    worst = watchdog.take_max_lag()
    await watchdog.stop()

    assert watchdog.stalls == 1
    assert 0.9 <= watchdog.stalled_s <= 1.5
    assert worst >= 0.9, "the pressure log sees the lag too"
    log = (tmp_path / STALLS_LOG).read_text()
    assert log.count("=== stall 1: loop silent") >= 2, "repeated while the stall lasts"
    loop_thread = log.split("\n--- ")[1]  # the loop thread comes first
    assert "(event loop)" in loop_thread and "_block_the_loop" in loop_thread
    assert "state=S" in loop_thread, "asleep in the kernel, not spinning"
    assert "=== stall 1 ended after" in log
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "[stall] host event loop blocked" in out and "_block_the_loop" in out


def _hold_the_loop_on_purpose(seconds: float) -> None:
    """A hold the watchdog is told to expect, like TAAF's periodic save."""
    time.sleep(seconds)


async def test_an_expected_hold_is_counted_not_reported(tmp_path: Path, capsys: object) -> None:
    """A stall whose snapshots all find the loop inside expected code goes
    in the count and nowhere else: no log, no stdout line."""
    watchdog = LoopStallWatchdog(
        tmp_path,
        threshold_s=0.3,
        repeat_s=0.2,
        tick_s=0.02,
        poll_s=0.02,
        expected=frozenset({_hold_the_loop_on_purpose.__code__}),
    )
    watchdog.start()
    await asyncio.sleep(0.1)
    _hold_the_loop_on_purpose(1.0)
    await asyncio.sleep(0.2)
    await watchdog.stop()
    assert (watchdog.stalls, watchdog.expected_holds) == (0, 1)
    assert 0.9 <= watchdog.expected_held_s <= 1.5
    assert not (tmp_path / STALLS_LOG).exists()
    assert "[stall]" not in capsys.readouterr().out  # type: ignore[attr-defined]


async def test_an_expected_hold_past_its_limit_is_reported_whole(tmp_path: Path, capsys: object) -> None:
    """Held past ``expected_limit_s``, an expected hold is a stall like any
    other, and the log carries its snapshots from the start."""
    watchdog = LoopStallWatchdog(
        tmp_path,
        threshold_s=0.3,
        repeat_s=0.2,
        tick_s=0.02,
        poll_s=0.02,
        expected=frozenset({_hold_the_loop_on_purpose.__code__}),
        expected_limit_s=0.8,
    )
    watchdog.start()
    await asyncio.sleep(0.1)
    _hold_the_loop_on_purpose(1.5)
    await asyncio.sleep(0.2)
    await watchdog.stop()
    assert (watchdog.stalls, watchdog.expected_holds) == (1, 0)
    log = (tmp_path / STALLS_LOG).read_text()
    assert "=== stall 1: loop silent 0." in log, "the snapshots from before the limit are kept"
    assert "_hold_the_loop_on_purpose" in log and "=== stall 1 ended after" in log
    assert "[stall] host event loop blocked" in capsys.readouterr().out  # type: ignore[attr-defined]


async def test_an_idle_loop_opens_no_stall(tmp_path: Path, capsys: object) -> None:
    """A loop that yields opens nothing and prints nothing."""
    watchdog = LoopStallWatchdog(tmp_path, threshold_s=0.3, tick_s=0.02, poll_s=0.02)
    watchdog.start()
    for _ in range(20):
        await asyncio.sleep(0.02)
    await watchdog.stop()
    assert watchdog.stalls == 0
    assert watchdog.take_max_lag() < 0.3
    assert not (tmp_path / STALLS_LOG).exists()
    assert "[stall]" not in capsys.readouterr().out  # type: ignore[attr-defined]


def test_pressure_files_parse(tmp_path: Path) -> None:
    """A pressure file's two lines, a cgroup's with no ``full`` line
    (``cpu.pressure`` on older kernels), and an absent file."""
    both = tmp_path / "io"
    both.write_text(
        "some avg10=32.36 avg60=41.68 avg300=49.37 total=1774693047616\n"
        "full avg10=32.32 avg60=41.59 avg300=49.28 total=1770991745534\n"
    )
    assert read_psi(both) == {
        "some": {"avg10": 32.36, "avg60": 41.68, "avg300": 49.37, "total": 1774693047616.0},
        "full": {"avg10": 32.32, "avg60": 41.59, "avg300": 49.28, "total": 1770991745534.0},
    }
    some = tmp_path / "cpu"
    some.write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=12\n")
    assert read_psi(some) == {"some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 12.0}}
    assert read_psi(tmp_path / "absent") is None


def test_only_network_filesystem_mounts_and_named_operations_are_read(tmp_path: Path) -> None:
    """``mountstats``: the NFS mount's named operations as (count, execute
    ms); other filesystems and unnamed operations are skipped."""
    path = tmp_path / "mountstats"
    path.write_text(_MOUNTSTATS.format(read=10, read_ms=30, write=4, write_ms=4000))
    assert read_nfs_ops(path) == {"/data/shared": {"READ": (10, 30), "WRITE": (4, 4000), "GETATTR": (3235, 1686)}}


def test_whole_disks_only(tmp_path: Path) -> None:
    """``diskstats`` keeps the devices named, and never loop devices."""
    path = tmp_path / "diskstats"
    path.write_text(
        "   7       0 loop0 11 0 28 0 0 0 0 0 0 1 0 0 0 0 0 0 0\n"
        " 259       0 nvme0n1 100 0 2048 50 200 0 4096 80 3 900 1000 0 0 0 0 0 0\n"
        " 259       1 nvme0n1p1 100 0 2048 50 200 0 4096 80 3 900 1000 0 0 0 0 0 0\n"
    )
    assert read_diskstats(path, {"nvme0n1", "loop0"}) == {"nvme0n1": (2048, 4096, 3, 900)}


def _fake_system(root: Path, *, io_total: int, write: int, write_ms: int, scan: int, busy_ms: int, high: int) -> None:
    """A /proc, cgroup and /sys/block with the counters that change."""
    proc, cgroup, block = root / "proc", root / "cgroup", root / "block"
    for directory in (proc / "pressure", proc / "self", cgroup, block / "nvme0n1"):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("cpu", "memory"):
        (proc / "pressure" / name).write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        (cgroup / f"{name}.pressure").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    (proc / "pressure" / "io").write_text(
        f"some avg10=1.00 avg60=0.00 avg300=0.00 total={io_total}\n"
        f"full avg10=1.00 avg60=0.00 avg300=0.00 total={io_total // 2}\n"
    )
    (cgroup / "io.pressure").write_text(f"some avg10=1.00 avg60=0.00 avg300=0.00 total={io_total}\n")
    (cgroup / "memory.events").write_text(f"low 0\nhigh {high}\nmax 0\noom 0\noom_kill 0\n")
    (proc / "vmstat").write_text(
        f"nr_dirty 512\nnr_writeback 256\npgscan_direct {scan}\npgsteal_direct {scan}\n"
        f"allocstall_normal {scan}\nallocstall_movable {scan}\npgmajfault 7\npswpin 0\npswpout 0\n"
    )
    (proc / "stat").write_text("cpu 1 2 3\nprocs_running 5\nprocs_blocked 3\n")
    (proc / "self" / "mountstats").write_text(_MOUNTSTATS.format(read=10, read_ms=30, write=write, write_ms=write_ms))
    (proc / "diskstats").write_text(f" 259 0 nvme0n1 100 0 2048 50 200 0 4096 80 3 {busy_ms} 1000 0 0 0 0 0 0\n")


def _reader(root: Path) -> PressureReader:
    """A reader over a fake system."""
    return PressureReader(proc=root / "proc", cgroup=root / "cgroup", sys_block=root / "block")


def test_two_readings_become_an_interval(tmp_path: Path) -> None:
    """Pressure as the share of the interval stalled, counters as their
    change, gauges as they stand, only active mounts and disks, memory.high
    breaches from the cgroup's events."""
    _fake_system(tmp_path, io_total=0, write=4, write_ms=400, scan=100, busy_ms=900, high=2)
    reader = _reader(tmp_path)
    before = reader.raw()
    _fake_system(tmp_path, io_total=5_000_000, write=14, write_ms=10_400, scan=160, busy_ms=5900, high=7)
    after = reader.raw()
    before["mono"], after["mono"] = 100.0, 110.0

    record = interval_record(before, after)
    assert record["dt"] == 10.0
    assert record["node"]["io"] == {"some": 50.0, "full": 25.0}
    assert record["node"]["cpu"] == {"some": 0.0}
    assert record["cg"]["io"] == {"some": 50.0}
    assert record["cg"]["memory_events"] == {"high": 5, "max": 0, "oom_kill": 0}
    assert record["vm"]["pgscan_direct"] == 60 and record["vm"]["allocstall"] == 120
    assert record["vm"]["pgmajfault"] == 0
    assert record["procs"] == {"running": 5, "blocked": 3}
    assert record["nfs"] == {"/data/shared": {"WRITE": [10, 1000.0]}}, "a mean of one second per write"
    assert record["disk"] == {"nvme0n1": {"util": 50.0, "rd_mib": 0.0, "wr_mib": 0.0, "inflight": 3}}


def test_the_sampler_rolls_segments_and_leaves_none_half_written(tmp_path: Path) -> None:
    """Samples land in numbered segments; stopping writes the last interval
    and closes the live segment, so every file left is final and the reader
    returns every record in order."""
    _fake_system(tmp_path, io_total=0, write=4, write_ms=400, scan=100, busy_ms=900, high=2)
    out_dir = monitor_dir(tmp_path / "job")
    lags = iter(range(1000))
    sampler = PressureSampler(
        out_dir, reader=_reader(tmp_path), interval_s=0.05, segment_s=0.12, lag=lambda: float(next(lags))
    )
    sampler.start()
    time.sleep(0.5)
    sampler.stop()

    names = sorted(path.name for path in out_dir.iterdir())
    assert len(names) >= 2 and all(name.startswith("pressure-") and name.endswith(".jsonl") for name in names)
    records = list(read_pressure(tmp_path / "job"))
    assert len(records) >= 5
    assert [record["loop_lag_s"] for record in records] == list(range(len(records)))
    assert all({"t", "dt", "node", "cg", "vm", "nfs", "disk"} <= set(record) for record in records)
    assert all(json.loads(line) for name in names for line in (out_dir / name).read_text().splitlines())


def test_this_machines_counters_read(tmp_path: Path) -> None:
    """On the live system nothing raises, whatever the kernel exposes, and
    two readings make an interval."""
    reader = PressureReader.for_this_process()
    assert reader.sample() is None, "the first call takes the baseline"
    record = reader.sample()
    assert record is not None and record["dt"] >= 0
    assert set(reader.instant()) == {"node", "cg", "procs"}


async def test_hold_loop_holds_the_event_loop_it_runs_on(capsys: pytest.CaptureFixture[str]) -> None:
    """A loop hold blocks every other task on the loop for its length, start to
    start every interval, and says so as each begins."""
    ticks: list[float] = []

    async def tick() -> None:
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    holder = asyncio.create_task(platypus.host_monitor.hold_loop(0.2, 0.3))
    await asyncio.sleep(0.8)
    for task in (holder, ticker):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    gaps = [later - earlier for earlier, later in itertools.pairwise(ticks)]
    assert max(gaps) >= 0.2
    assert "[loop hold] holding the host's event loop 0 s" in capsys.readouterr().out
