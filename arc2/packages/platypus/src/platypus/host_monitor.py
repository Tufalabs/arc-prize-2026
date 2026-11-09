"""Host-process monitors: the stalls the engine cannot see.

Every request a game sends, and every status line the run prints, passes
through the host process's event loop. When that loop blocks, the engine
drains the requests it holds and sits idle, the status lines go silent, and
nothing in the run says why: from vLLM's side there is no problem, only no
work. Two monitors run on threads of their own, so they keep working while
the loop does not.

- :class:`LoopStallWatchdog` ticks on the loop and watches the tick from a
  thread. A tick older than ``threshold_s`` opens a stall, and every
  ``repeat_s`` while it lasts the thread writes a snapshot: each thread's
  Python stack with its kernel state and wait channel, and the pressure
  readings of the moment. Once the loop runs again, one ``[stall]`` line on
  stdout says how long it was blocked and where.
- :class:`PressureSampler` writes, every ``interval_s``, what the kernel
  reports about contention: pressure stall information for the job's cgroup
  and for the node, reclaim and writeback counters, per-operation latency on
  network-filesystem mounts, local disk utilisation, and the loop's worst
  lag since the last sample.

Both write under the diagnostics root's ``monitor/`` directory
(:func:`platypus.diagnostics.monitor_dir`): node-local scratch on SLURM, so a
slow shared filesystem cannot block the monitor meant to catch it, and stdout
is written only once a stall is over. The pressure log rolls into a new
segment every ``segment_s``; a closed segment is final, and the archiver
takes it once. Every reading tolerates a file it cannot read: kernels,
cgroup versions and containers differ in what they expose, and a missing
reading is recorded as absent rather than ending the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import platypus.diagnostics
import platypus.memory

#: The stall log, appended to by the watchdog's thread.
STALLS_LOG = "loop_stalls.log"

_PSI_RESOURCES = ("cpu", "memory", "io")
# Cumulative counters, reported as their change over the interval.
_VMSTAT_COUNTERS = ("pgscan_direct", "pgsteal_direct", "pgmajfault", "pswpin", "pswpout")
# Instantaneous, in pages.
_VMSTAT_GAUGES = ("nr_dirty", "nr_writeback")
# The NFS operations whose latency says whether the mount is what stalls.
_NFS_OPS = ("READ", "WRITE", "COMMIT", "GETATTR", "LOOKUP", "ACCESS", "OPEN", "CLOSE", "SETATTR", "CREATE", "REMOVE")
_MEMORY_EVENTS = ("high", "max", "oom_kill")
# x86-64 system call numbers a blocked loop is likely to be sitting in,
# named so a snapshot reads without a table; any other number stays a number.
_SYSCALL_NAMES_X86_64 = {
    0: "read",
    1: "write",
    3: "close",
    7: "poll",
    17: "pread64",
    18: "pwrite64",
    19: "readv",
    20: "writev",
    23: "select",
    35: "nanosleep",
    74: "fsync",
    75: "fdatasync",
    82: "rename",
    87: "unlink",
    202: "futex",
    230: "clock_nanosleep",
    232: "epoll_wait",
    257: "openat",
    262: "newfstatat",
    270: "pselect6",
    281: "epoll_pwait",
}
_PAGE_KIB = os.sysconf("SC_PAGE_SIZE") // 1024


# ---------------------------------------------------------------------------
# Readers. Pure functions of a file, so the tests hand them fixtures.
# ---------------------------------------------------------------------------


def read_psi(path: Path) -> dict[str, dict[str, float]] | None:
    """A ``*.pressure`` file as ``{"some"|"full": {"avg10", "avg60",
    "avg300", "total"}}``; ``total`` is cumulative stalled microseconds.
    None when the file is absent or unreadable."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return None
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        values: dict[str, float] = {}
        for item in rest.split():
            key, _, value = item.partition("=")
            with contextlib.suppress(ValueError):
                values[key] = float(value)
        if kind in ("some", "full") and values:
            out[kind] = values
    return out or None


def read_key_values(path: Path) -> dict[str, int]:
    """A ``key value`` per line file (``/proc/vmstat``, ``memory.events``,
    the ``procs_*`` lines of ``/proc/stat``), integers only; empty when
    unreadable."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            with contextlib.suppress(ValueError):
                out[parts[0]] = int(parts[1])
    return out


def read_nfs_ops(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Per network-filesystem mount in a ``mountstats`` file, each named
    operation's cumulative ``(ops, execute_ms)``: the count, and the time
    from queueing to reply summed over them. Empty when unreadable."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return {}
    mounts: dict[str, dict[str, tuple[int, int]]] = {}
    current: dict[str, tuple[int, int]] | None = None
    for line in text.splitlines():
        if line.startswith("device "):
            # device <what> mounted on <where> with fstype <type> ...
            words = line.split()
            try:
                where = words[words.index("on") + 1]
                fstype = words[words.index("fstype") + 1]
            except (ValueError, IndexError):
                current = None
                continue
            current = mounts.setdefault(where, {}) if fstype.startswith("nfs") else None
            continue
        if current is None:
            continue
        name, sep, rest = line.strip().partition(":")
        if not sep or name not in _NFS_OPS:
            continue
        fields = rest.split()
        # ops, transmissions, timeouts, bytes sent, bytes received,
        # queue ms, rtt ms, execute ms[, errors]
        if len(fields) >= 8:
            with contextlib.suppress(ValueError):
                current[name] = (int(fields[0]), int(fields[7]))
    return mounts


def read_diskstats(path: Path, devices: set[str] | None = None) -> dict[str, tuple[int, int, int, int]]:
    """Per block device in a ``diskstats`` file, ``(sectors read, sectors
    written, in flight, ms doing I/O)``. ``devices`` keeps only those names
    (whole disks, where ``/sys/block`` lists them); loop and ram devices are
    never kept. Empty when unreadable."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[int, int, int, int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        name = fields[2]
        if name.startswith(("loop", "ram")) or (devices is not None and name not in devices):
            continue
        with contextlib.suppress(ValueError):
            out[name] = (int(fields[5]), int(fields[9]), int(fields[11]), int(fields[12]))
    return out


def thread_kernel_state(native_id: int, task_root: Path = Path("/proc/self/task")) -> dict[str, str]:
    """A thread's scheduler state (``R`` running, ``S`` sleeping, ``D``
    uninterruptible -- disk, network filesystem, reclaim), the kernel
    function it waits in, and its current system call, each where
    readable."""
    task = task_root / str(native_id)
    out: dict[str, str] = {}
    try:
        stat = (task / "stat").read_text()
        # pid (comm) STATE ...; comm may hold spaces and parentheses.
        out["state"] = stat[stat.rindex(")") + 2 :].split()[0]
    except (OSError, ValueError, IndexError):
        pass
    for name in ("wchan", "syscall"):
        with contextlib.suppress(OSError, ValueError):
            value = (task / name).read_text().strip()
            if value:
                out[name] = value.split()[0]
    number = out.get("syscall", "")
    if number.isdigit() and platform.machine() == "x86_64":
        out["syscall"] = _SYSCALL_NAMES_X86_64.get(int(number), number)
    return out


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------


@dataclass
class PressureReader:
    """Reads the node's and the job's contention counters and turns two
    readings into the interval between them.

    The roots are parameters so a test can point them at fixtures; the
    defaults are the live system, with the job's cgroup found the way the
    memory report finds it."""

    proc: Path = Path("/proc")
    cgroup: Path | None = None
    sys_block: Path = Path("/sys/block")
    _previous: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @classmethod
    def for_this_process(cls) -> PressureReader:
        """A reader over the live system and this job's cgroup."""
        return cls(cgroup=platypus.memory.job_cgroup_dir())

    def raw(self) -> dict[str, Any]:
        """Every counter as it stands, cumulative."""
        try:
            devices: set[str] | None = {entry.name for entry in os.scandir(self.sys_block)}
        except OSError:
            devices = None
        vmstat = read_key_values(self.proc / "vmstat")
        return {
            "t": time.time(),
            "mono": time.monotonic(),
            "node": {name: read_psi(self.proc / "pressure" / name) for name in _PSI_RESOURCES},
            "cg": {name: read_psi(self.cgroup / f"{name}.pressure") for name in _PSI_RESOURCES}
            if self.cgroup is not None
            else {},
            "cg_events": read_key_values(self.cgroup / "memory.events") if self.cgroup is not None else {},
            "vm": vmstat,
            "procs": {
                key.removeprefix("procs_"): value
                for key, value in read_key_values(self.proc / "stat").items()
                if key in ("procs_running", "procs_blocked")
            },
            "nfs": read_nfs_ops(self.proc / "self" / "mountstats"),
            "disk": read_diskstats(self.proc / "diskstats", devices),
        }

    def instant(self) -> dict[str, Any]:
        """The kernel's own ten-second averages, no interval needed: what a
        stall snapshot records alongside the stacks."""
        raw = self.raw()

        def avg10(block: Mapping[str, dict[str, dict[str, float]] | None]) -> dict[str, Any]:
            return {
                name: {kind: psi[kind].get("avg10") for kind in psi} for name, psi in block.items() if psi is not None
            }

        return {"node": avg10(raw["node"]), "cg": avg10(raw["cg"]), "procs": raw["procs"]}

    def sample(self) -> dict[str, Any] | None:
        """The interval since the previous call, as one record; None on the
        first call, which only takes the baseline."""
        current = self.raw()
        previous, self._previous = self._previous, current
        if previous is None:
            return None
        return interval_record(previous, current)


def _psi_share(
    before: dict[str, dict[str, float]] | None, after: dict[str, dict[str, float]] | None, dt: float
) -> dict[str, float] | None:
    """Share of the interval (percent) in which tasks stalled on a resource:
    ``some`` -- at least one did, ``full`` -- every non-idle one did."""
    if before is None or after is None or dt <= 0:
        return None
    out: dict[str, float] = {}
    for kind in ("some", "full"):
        if kind in before and kind in after and "total" in before[kind] and "total" in after[kind]:
            out[kind] = round((after[kind]["total"] - before[kind]["total"]) / (dt * 1e6) * 100, 2)
    return out or None


def interval_record(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Two raw readings as the interval between them: pressure as the share
    of the interval stalled, counters as their change, gauges as they stand,
    and only the mounts and disks that saw any activity."""
    dt = after["mono"] - before["mono"]
    record: dict[str, Any] = {"t": round(after["t"], 3), "dt": round(dt, 3)}
    record["node"] = {
        name: share
        for name in _PSI_RESOURCES
        if (share := _psi_share(before["node"].get(name), after["node"].get(name), dt)) is not None
    }
    record["cg"] = {
        name: share
        for name in _PSI_RESOURCES
        if (share := _psi_share(before["cg"].get(name), after["cg"].get(name), dt)) is not None
    }
    events = {
        name: after["cg_events"][name] - before["cg_events"][name]
        for name in _MEMORY_EVENTS
        if name in after["cg_events"] and name in before["cg_events"]
    }
    if events:
        record["cg"]["memory_events"] = events
    vm_before, vm_after = before["vm"], after["vm"]
    vm: dict[str, int] = {
        name: vm_after[name] - vm_before[name] for name in _VMSTAT_COUNTERS if name in vm_after and name in vm_before
    }
    allocstall = [name for name in vm_after if name.startswith("allocstall") and name in vm_before]
    if allocstall:
        vm["allocstall"] = sum(vm_after[name] - vm_before[name] for name in allocstall)
    for name in _VMSTAT_GAUGES:
        if name in vm_after:
            vm[f"{name}_mib"] = vm_after[name] * _PAGE_KIB // 1024
    record["vm"] = vm
    record["procs"] = dict(after["procs"])
    nfs: dict[str, dict[str, list[float]]] = {}
    for mount, ops in after["nfs"].items():
        earlier = before["nfs"].get(mount, {})
        active: dict[str, list[float]] = {}
        for op, (count, execute_ms) in ops.items():
            count0, execute0 = earlier.get(op, (count, execute_ms))
            if count > count0:
                # [operations, mean milliseconds from queueing to reply]
                active[op] = [count - count0, round((execute_ms - execute0) / (count - count0), 2)]
        if active:
            nfs[mount] = active
    record["nfs"] = nfs
    disks: dict[str, dict[str, float]] = {}
    for name, (read, written, inflight, busy_ms) in after["disk"].items():
        if name not in before["disk"] or dt <= 0:
            continue
        read0, written0, _, busy0 = before["disk"][name]
        if busy_ms > busy0 or inflight:
            disks[name] = {
                "util": round((busy_ms - busy0) / (dt * 1000) * 100, 1),
                "rd_mib": round((read - read0) * 512 / 2**20, 1),
                "wr_mib": round((written - written0) * 512 / 2**20, 1),
                "inflight": inflight,
            }
    record["disk"] = disks
    return record


@dataclass
class PressureSampler:
    """Writes one :func:`interval_record` a line every ``interval_s``, from
    a thread, into ``pressure-NNNN.jsonl`` segments that roll every
    ``segment_s``. ``lag`` reports -- and resets -- the loop's worst lag
    since the last sample, when a watchdog is there to say."""

    out_dir: Path
    reader: PressureReader = field(default_factory=PressureReader.for_this_process)
    interval_s: float = 10.0
    segment_s: float = 1800.0
    lag: Callable[[], float] | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _segment: int = field(default=0, init=False, repr=False)
    _segment_started: float = field(default=0.0, init=False, repr=False)

    def start(self) -> None:
        """Take the baseline and start sampling."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.reader.sample()
        self._open_segment()
        self._thread = threading.Thread(target=self._run, name="platypus-pressure", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Write the interval since the last sample, close the segment and
        return once the thread has."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _live_path(self) -> Path:
        """The segment being written."""
        return self.out_dir / f"pressure-{self._segment:04d}.jsonl{platypus.diagnostics.MONITOR_LIVE_SUFFIX}"

    def _open_segment(self) -> None:
        """Start the next segment."""
        self._segment += 1
        self._segment_started = time.monotonic()

    def _close_segment(self) -> None:
        """Make the live segment final, if anything was written to it."""
        live = self._live_path()
        if live.exists():
            os.replace(live, live.with_name(live.name.removesuffix(platypus.diagnostics.MONITOR_LIVE_SUFFIX)))

    def _write_sample(self) -> None:
        """One record onto the live segment."""
        record = self.reader.sample()
        if record is None:
            return
        if self.lag is not None:
            record["loop_lag_s"] = round(self.lag(), 2)
        with open(self._live_path(), "a") as out:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _run(self) -> None:
        """Sample until stopped; never fatal to the run."""
        while True:
            stopping = self._stop.wait(self.interval_s)
            try:
                self._write_sample()
                if stopping:
                    self._close_segment()
                    return
                if time.monotonic() - self._segment_started >= self.segment_s:
                    self._close_segment()
                    self._open_segment()
            except Exception:
                traceback.print_exc()
                if stopping:
                    return


# ---------------------------------------------------------------------------
# The loop watchdog
# ---------------------------------------------------------------------------


def _short(filename: str) -> str:
    """A source path cut to its last three components."""
    return "/".join(Path(filename).parts[-3:])


def _stack_lines(frame: FrameType | None) -> list[str]:
    """A thread's Python stack, innermost call last."""
    if frame is None:
        return []
    return [f"{_short(entry.filename)}:{entry.lineno} {entry.name}" for entry in traceback.extract_stack(frame)]


@dataclass
class _Snapshot:
    """What one look at a stalled process found."""

    #: How long the loop had been silent, and when the look was taken.
    lag: float
    wall: str
    #: The pressure line and every thread's stack, for the log.
    body: str
    #: The loop thread's innermost frame, kernel state and wait channel.
    where: str
    #: Whether the loop was inside one of the watchdog's expected holds.
    expected: bool

    def text(self, number: int) -> str:
        """The snapshot as the log records it, under its stall's number."""
        return f"=== stall {number}: loop silent {self.lag:.1f}s at {self.wall} ===\n{self.body}"


@dataclass
class LoopStallWatchdog:
    """Notices the host's event loop not running and records what every
    thread was doing instead.

    The loop proves it runs by stamping a tick every ``tick_s``; a thread
    checks the stamp every ``poll_s``. It also keeps the worst lag it has
    seen since last asked, for the pressure log's continuous record of
    stalls too short to open one.

    ``expected`` names code that holds the loop on purpose (a function's
    ``__code__``). A stall whose every snapshot finds the loop inside one is
    counted in ``expected_holds`` and neither logged nor printed, unless it
    lasts ``expected_limit_s``: a hold that long is a stall whatever it is
    inside, and a node freezing during one looks exactly like this."""

    out_dir: Path
    threshold_s: float = 10.0
    repeat_s: float = 15.0
    tick_s: float = 1.0
    poll_s: float = 1.0
    reader: PressureReader | None = None
    expected: frozenset[CodeType] = frozenset()
    expected_limit_s: float = 180.0
    _beat: float = field(default=0.0, init=False, repr=False)
    _max_lag: float = field(default=0.0, init=False, repr=False)
    _loop_ident: int | None = field(default=None, init=False, repr=False)
    _loop_native: int | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    stalls: int = field(default=0, init=False)
    stalled_s: float = field(default=0.0, init=False)
    expected_holds: int = field(default=0, init=False)
    expected_held_s: float = field(default=0.0, init=False)

    def start(self) -> None:
        """Start ticking on the running loop and watching from a thread."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._loop_ident = threading.get_ident()
        self._loop_native = threading.get_native_id()
        self._beat = time.monotonic()
        self._task = asyncio.get_running_loop().create_task(self._tick(), name="platypus-loop-tick")
        self._thread = threading.Thread(target=self._watch, name="platypus-loop-watchdog", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        """Stop both halves; a stall still open is closed as it stands."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def take_max_lag(self) -> float:
        """The worst lag seen since the last call, and reset it."""
        with self._lock:
            lag, self._max_lag = self._max_lag, 0.0
        return lag

    async def _tick(self) -> None:
        """Stamp the tick while the loop runs."""
        while True:
            self._beat = time.monotonic()
            await asyncio.sleep(self.tick_s)

    def _watch(self) -> None:
        """Open, snapshot and close stalls as the tick goes stale and
        resumes; never fatal to the run."""
        stall_from: float | None = None
        next_snapshot = 0.0
        wheres: list[str] = []
        # Snapshots of a stall not reported yet: an expected hold's are kept
        # back, and dropped with it if it ends as one.
        held: list[_Snapshot] = []
        reported = False
        while True:
            stopping = self._stop.wait(self.poll_s)
            try:
                now = time.monotonic()
                beat = self._beat
                lag = now - beat
                with self._lock:
                    self._max_lag = max(self._max_lag, lag)
                if stall_from is None and lag > self.threshold_s and not stopping:
                    stall_from, next_snapshot, wheres, held, reported = beat, now, [], [], False
                if stall_from is not None and not stopping and lag > self.threshold_s and now >= next_snapshot:
                    snapshot = self._snapshot(lag)
                    wheres.append(snapshot.where)
                    held.append(snapshot)
                    next_snapshot = now + self.repeat_s
                    if not reported and (not snapshot.expected or lag >= self.expected_limit_s):
                        reported = True
                        self.stalls += 1
                    if reported:
                        self._append("".join(item.text(self.stalls) for item in held))
                        held = []
                if stall_from is not None and (lag <= self.threshold_s or stopping):
                    end = beat if lag <= self.threshold_s else now
                    if reported:
                        self._close(stall_from, end, wheres, still_open=stopping)
                    else:
                        self.expected_holds += 1
                        self.expected_held_s += max(0.0, end - stall_from)
                    stall_from, wheres, held, reported = None, [], [], False
                if stopping:
                    return
            except Exception:
                traceback.print_exc()
                if stopping:
                    return

    def _snapshot(self, lag: float) -> _Snapshot:
        """Every thread's stack and kernel state, loop thread first."""
        frames = sys._current_frames()  # pyright: ignore[reportPrivateUsage] - the documented way to see other threads
        threads = {thread.ident: thread for thread in threading.enumerate()}
        order = sorted(frames, key=lambda ident: ident != self._loop_ident)
        wall = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = []
        if self.reader is not None:
            with contextlib.suppress(Exception):
                lines.append("pressure (avg10 %): " + json.dumps(self.reader.instant(), separators=(",", ":")))
        where = "?"
        for ident in order:
            thread = threads.get(ident)
            native = thread.native_id if thread is not None else None
            if ident == self._loop_ident:
                native = self._loop_native
            state = thread_kernel_state(native) if native is not None else {}
            name = thread.name if thread is not None else str(ident)
            tag = " (event loop)" if ident == self._loop_ident else ""
            described = " ".join(f"{key}={value}" for key, value in state.items())
            lines.append(f"--- {name}{tag} tid={native} {described}")
            stack = _stack_lines(frames[ident])
            lines.extend(f"    {entry}" for entry in stack)
            if ident == self._loop_ident:
                top = stack[-1] if stack else "?"
                where = f"{top} (state {state.get('state', '?')}, wchan {state.get('wchan', '?')})"
        expected = False
        frame = frames.get(self._loop_ident) if self._loop_ident is not None else None
        while frame is not None and not expected:
            expected = frame.f_code in self.expected
            frame = frame.f_back
        return _Snapshot(lag=lag, wall=wall, body="\n".join(lines) + "\n", where=where, expected=expected)

    def _close(self, start: float, end: float, wheres: list[str], *, still_open: bool) -> None:
        """Record a stall's end and say so on stdout."""
        duration = max(0.0, end - start)
        self.stalled_s += duration
        began = datetime.datetime.now() - datetime.timedelta(seconds=time.monotonic() - start)
        common = max(set(wheres), key=wheres.count) if wheres else "?"
        status = "still blocked at the run's end" if still_open else "ended"
        self._append(f"=== stall {self.stalls} {status} after {duration:.1f}s ===\n")
        print(
            f"[stall] host event loop blocked {duration:.1f}s from {began:%H:%M:%S}: {common}"
            f" ({len(wheres)} snapshot{'s' if len(wheres) != 1 else ''} in"
            f" {platypus.diagnostics.MONITOR_DIR}/{STALLS_LOG})",
            flush=True,
        )

    def _append(self, text: str) -> None:
        """Onto the stall log."""
        with open(self.out_dir / STALLS_LOG, "a") as out:
            out.write(text)


async def hold_loop(hold_s: float, every_s: float) -> None:
    """Hold the event loop this runs on for ``hold_s`` every ``every_s``,
    start to start, the first ``every_s`` in (``RunConfig.loop_hold_s``): a
    host stall on demand, said on stdout as each begins, which the stall
    watchdog then records as any other."""
    await asyncio.sleep(every_s)
    while True:
        print(f"[loop hold] holding the host's event loop {hold_s:.0f} s (RunConfig.loop_hold_s)", flush=True)
        began = time.monotonic()
        time.sleep(hold_s)  # noqa: ASYNC251 — the point: blocking, on the loop
        await asyncio.sleep(max(every_s - (time.monotonic() - began), 0.0))


@contextlib.asynccontextmanager
async def host_monitors(root: Path, *, expected: Collection[CodeType] = ()) -> AsyncGenerator[LoopStallWatchdog]:
    """Run both monitors, writing under ``root``'s monitor dir, for the
    block; ``expected`` is the code that holds the loop on purpose
    (:class:`LoopStallWatchdog`). Must be entered on the loop it is to watch."""
    out_dir = platypus.diagnostics.monitor_dir(root)
    reader = PressureReader.for_this_process()
    # One reader for both: a snapshot's reading keeps no state, so it cannot
    # disturb the sampler's intervals.
    watchdog = LoopStallWatchdog(out_dir, reader=reader, expected=frozenset(expected))
    sampler = PressureSampler(out_dir, reader=reader, lag=watchdog.take_max_lag)
    watchdog.start()
    sampler.start()
    try:
        yield watchdog
    finally:
        await asyncio.to_thread(sampler.stop)
        await watchdog.stop()


def read_pressure(job_dir: str | Path) -> Iterator[dict[str, Any]]:
    """A job's pressure records in order, wherever its segments are: loose
    in the job dir, on this process's scratch, or archived."""
    segments = [name for name in platypus.diagnostics.monitor_files(job_dir) if Path(name).name.startswith("pressure-")]
    for name in segments:
        raw = platypus.diagnostics.read_job_file(job_dir, name)
        for line in (raw or b"").decode("utf-8", "replace").splitlines():
            with contextlib.suppress(ValueError):
                yield json.loads(line)
