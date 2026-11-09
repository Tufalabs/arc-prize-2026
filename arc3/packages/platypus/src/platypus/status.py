"""What every hosted game is doing right now, printed periodically.

Four coarse states belong to infrastructure and are known from the host
task's own lifecycle, without any cooperation from the agent:

    waiting for slot  ->  running  ->  finished
                            ^  |
                            |  v
                          paused

A game is ``paused`` when it has offered its slot back and not yet been
re-granted one (``platypus.priority``). Kept apart from ``waiting for
slot`` although both are waiting for the same thing: one has a live
process, a conversation and a kernel behind it and the other is a game
the run has not touched yet, and a report that cannot tell them apart
cannot say what a run is holding in memory.

An agent may refine ``running`` into whatever it is actually doing by
writing a short label through ``ctx.status``. Labels are published **by
the process doing the work**, at the moment it starts waiting or
working — never inferred host-side from a semaphore counter, which can
only say how many games are blocked somewhere, not which game is blocked
on what.

Publication is push-on-change, so the registry always holds each game's
last declared state together with when it entered it. A game wedged
inside a kernel call or an LLM request therefore keeps reporting that
state, which is exactly what a stuck run should look like; the
time-in-state is what makes it legible.

The registry lives in the parent process. In-process games write to it
directly; subprocess games push their label over the existing control
pipe (``platypus.subproc``), which the parent applies as it services
them.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import platypus.memory

# The coarse states. "running" is the one an agent may refine.
WAITING_FOR_SLOT = "waiting for slot"  # never started
RUNNING = "running"
PAUSED = "paused"  # started, gave its slot back, waiting for another
FINISHED = "finished"


@dataclass
class GameStatus:
    """One hosted game's current state and how long it has held it."""

    game_id: str
    coarse: str = WAITING_FOR_SLOT
    detail: Optional[str] = None
    since: float = field(default_factory=time.monotonic)

    @property
    def label(self) -> str:
        """What to report: the agent's refinement while running, else the
        coarse state."""
        if self.coarse == RUNNING and self.detail is not None:
            return self.detail
        return self.coarse


class StatusRegistry:
    """Every hosted game's status, owned by the parent process.

    Plain dict, no locking: every write reaches it on the parent's event
    loop — directly for in-process games, through the subprocess message
    loop for the rest.
    """

    def __init__(self) -> None:
        """An empty registry; games register themselves as they start."""
        self._games: dict[str, GameStatus] = {}

    def register(self, key: str, game_id: str) -> None:
        """Announce a game, before it takes a concurrency slot — one queued
        behind the cap should read as queued, not as missing."""
        self._games[key] = GameStatus(game_id=game_id)

    def set_coarse(self, key: str, coarse: str) -> None:
        """Move a game between the infra states. Clears any agent
        refinement: it described the previous state and would otherwise
        outlive it (a finished game must never still read 'running
        kernel')."""
        status = self._games.get(key)
        if status is None:
            return
        if status.coarse != coarse:
            status.coarse, status.detail, status.since = coarse, None, time.monotonic()

    def detail(self, key: str) -> Optional[str]:
        """What ``key`` last said it was doing, or ``None``.

        For a caller that has to move a game between coarse states and put
        its refinement back afterwards: the agent's own handle publishes
        on change, so it will not re-send a label the registry dropped
        underneath it.
        """
        status = self._games.get(key)
        return status.detail if status is not None else None

    def set_detail(self, key: str, detail: str) -> None:
        """Record what a game says it is doing right now."""
        status = self._games.get(key)
        if status is None:
            return
        if status.detail != detail:
            status.detail, status.since = detail, time.monotonic()

    def snapshot(self) -> list[GameStatus]:
        """A copy of every registered game's state, safe to read while games write."""
        return list(self._games.values())


# ---------------------------------------------------------------------------
# The agent-facing handle
# ---------------------------------------------------------------------------


class StatusHandle:
    """``ctx.status``: the agent's (and its tools') way of saying what it
    is doing.

    Used as a context manager, so a label describes exactly the span of
    work it wraps and the previous one is restored on the way out::

        with ctx.status("waiting for kernel slot"):
            await semaphore.acquire()

    Nests. Unclaimed time reports as ``base`` — ``"running"`` by default,
    which agents that publish sub-states set to something of their own
    (the duck uses ``"other"``, so its report accounts for all of its
    time). Consecutive identical labels publish once, so wrapping a hot
    path costs a comparison.

    Never stored on the agent (it belongs to ``ctx``, A1); handed to
    tools per use, the way every other ``ctx`` service is.
    """

    def __init__(self, publish: Callable[[str], None], base: str = RUNNING):
        """A handle publishing through ``publish``, with ``base`` for unclaimed time."""
        self._publish = publish
        self._base = base
        self._stack: list[str] = []
        self._sticky: Optional[str] = None  # set(); outranked by any open scope
        self._last: Optional[str] = None

    @property
    def base(self) -> str:
        """What unclaimed time reports as."""
        return self._base

    def set_base(self, label: str) -> None:
        """Set what unclaimed time reports as."""
        self._base = label
        self._flush()

    def set(self, label: str) -> None:
        """Publish a label that lasts until the next change — for work
        whose extent is not a block (a state machine's phase). Prefer the
        context-manager form where the work has a scope.

        Kept beside the scope stack rather than in it: every open scope
        still owes a pop, so replacing the stack here would make an
        enclosing scope's exit pop something it never pushed."""
        self._sticky = label
        self._flush()

    @contextlib.contextmanager
    def __call__(self, label: str) -> Generator[None, None, None]:
        """Publish ``label`` for the duration of the block."""
        self._stack.append(label)
        self._flush()
        try:
            yield
        finally:
            self._stack.pop()
            self._flush()

    def _flush(self) -> None:
        # An open scope wins: it describes work happening right now, while a
        # sticky label describes the phase that work sits inside.
        """Publish the current label, if it changed — push-on-change means a
        wedged process keeps reporting the state it is wedged in."""
        label = self._stack[-1] if self._stack else (self._sticky or self._base)
        if label != self._last:
            self._last = label
            self._publish(label)


class _NullStatusHandle(StatusHandle):
    """The handle when status reporting is off: every operation is a
    no-op, so instrumented code never checks whether reporting is on."""

    def __init__(self) -> None:  # deliberately no super().__init__
        """A handle that publishes nothing."""
        pass

    def set_base(self, label: str) -> None:
        """No-op: reporting is off."""
        pass

    def set(self, label: str) -> None:
        """No-op: reporting is off."""
        pass

    @contextlib.contextmanager
    def __call__(self, label: str) -> Generator[None, None, None]:
        """No-op scope: reporting is off."""
        yield


NULL_STATUS: StatusHandle = _NullStatusHandle()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _read_memory() -> tuple[platypus.memory.TreeMemory, Optional[platypus.memory.EnforcedLimit]]:
    """Both memory figures in one off-loop call, so the tree total and the
    ceiling it gets compared against are measured at the same moment."""
    return platypus.memory.read_tree_memory(), platypus.memory.read_enforced_limit()


def _duration(seconds: float) -> str:
    """A short human duration for the periodic line."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def _tally(labels: list[str]) -> str:
    """``label: n`` counts, most populated first, ties alphabetical — so
    a line read minute after minute keeps a stable order."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{count} {label}" for label, count in ordered)


def format_status_lines(games: list[GameStatus], now: float) -> list[str]:
    """The report: one line of coarse counts, and — when any game is
    running — one of what those games are doing, with the game that has
    held its state longest named.

    That last figure is the point of the whole report: a run where one
    game has been ``waiting for LLM response`` for forty minutes looks
    identical to a healthy one in every other number here.
    """
    if not games:
        return []
    lines = [f"{len(games)} games: {_tally([status.coarse for status in games])}"]
    running = [status for status in games if status.coarse == RUNNING]
    if running:
        line = f"running: {_tally([status.label for status in running])}"
        oldest = max(running, key=lambda status: now - status.since)
        line += f" | longest: {oldest.game_id} {oldest.label} {_duration(now - oldest.since)}"
        lines.append(line)
    return lines


@dataclass
class StatusMonitor:
    """The run's periodic report, printed every ``interval_s`` until
    stopped: what each hosted game is doing, and what the process tree
    costs in memory.

    Shaped like ``ullm.engine_monitor.EngineStatusMonitor`` on purpose —
    all of these lines land in the same stdout on the same cadence, and a
    tick failure is visible but never fatal to the run.
    """

    registry: StatusRegistry = field(kw_only=True)
    interval_s: float = 60.0
    label: str = "games"
    # One line per reporter, on the game line's tick. Machine-wide occupancy
    # is what no per-game status can give: a game knows only its own state,
    # so "how many slots exist and how many are held" — across every game and
    # every process on this box — has to be read off the pool itself. The
    # per-game labels say which games are queueing; these say what for.
    reporters: list[Callable[[], str]] = field(default_factory=lambda: list[Callable[[], str]]())
    # Memory costs a page-table walk per process, so it rides this slow
    # cadence rather than getting a loop of its own.
    report_memory: bool = True
    _task: Optional["asyncio.Task[None]"] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Begin printing the periodic report."""
        assert self._task is None, "monitor already started"
        self._task = asyncio.create_task(self._loop(), name="platypus-status")

    async def stop(self) -> None:
        """Stop the periodic report and await its task."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        """Print one status line per interval until cancelled."""
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                for line in format_status_lines(self.registry.snapshot(), time.monotonic()):
                    print(f"[{self.label}] {line}", flush=True)
                for reporter in self.reporters:
                    try:
                        print(f"[pools] {reporter()}", flush=True)
                    except Exception as e:
                        # A pool that cannot report is not a reason to lose the
                        # rest of the tick, or the run.
                        print(f"[pools] reporter failed: {e!r}", flush=True)
                if self.report_memory:
                    # Off the event loop: this is a page-table walk per
                    # process, and the loop it would block is the one
                    # servicing every game subprocess's control pipe.
                    tree, limit = await asyncio.to_thread(_read_memory)
                    for line in platypus.memory.format_memory_lines(tree, limit):
                        print(f"[mem] {line}", flush=True)
        except asyncio.CancelledError:
            raise
        except BaseException:
            print(f"[{self.label}] status monitor died:", flush=True)
            traceback.print_exc()


@contextlib.contextmanager
def status_monitor(
    registry: StatusRegistry,
    interval_s: Optional[float],
    *,
    report_memory: bool = True,
    reporters: Optional[list[Callable[[], str]]] = None,
) -> Generator[None, None, None]:
    """Run a :class:`StatusMonitor` for the block, or nothing at all when
    ``interval_s`` is None. Stopping is fire-and-forget: the monitor holds
    no resources, and awaiting its cancellation in a synchronous exit path
    is not worth a second code path in the solver's teardown."""
    if interval_s is None:
        yield
        return
    monitor = StatusMonitor(
        registry=registry, interval_s=interval_s, report_memory=report_memory, reporters=reporters or []
    )
    monitor.start()
    try:
        yield
    finally:
        task = monitor._task
        if task is not None:
            task.cancel()
