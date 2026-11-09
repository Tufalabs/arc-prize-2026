"""The engine stall watchdog: end a run whose engine has gone quiet.

A stuck engine is the one failure the harness could not see. Requests park
on it, the per-request inactivity bound is an hour, and a submission runs
out its clock producing nothing. The watchdog reads what the engine status
line prints from -- the engine's own generated-token counter and its
request gauges -- and ends the run the way the soft deadline does
(``taaf.solver.Solver.stop_run``: games cancelled, diagnostics written, a
normal exit, the reason on the benchmark) when the counter has not moved
for ``timeout_s`` while the engine still reports requests, or when the
engine has died outright.

Demand is read from the engine's gauges rather than the host's request
tally: a frozen engine keeps reporting the requests it froze with, and the
traffic replay drives the engine without the host's proxy.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import ullm.engine_monitor

GENERATION_TOKENS = "vllm:generation_tokens"
REQUESTS_RUNNING = ullm.engine_monitor.REQUESTS_RUNNING
REQUESTS_WAITING = "vllm:num_requests_waiting"


@dataclass
class EngineStallWatchdog:
    """Fires ``stop`` once, with a reason, when the engine stalls or dies.

    ``metrics`` is the backend's own snapshot (``Backend.engine_metrics``);
    one that cannot be read counts as no progress. ``dead`` says the engine
    has died outright, which ends the run at the next tick rather than after
    the timeout. The quiet clock runs only while the engine owes tokens: it
    restarts when tokens arrive, and while nothing is asked of the engine.
    """

    metrics: Callable[[], Awaitable[ullm.engine_monitor.EngineMetrics | None]]
    dead: Callable[[], bool]
    stop: Callable[[str], None]
    timeout_s: float = 600.0
    interval_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _tokens: float | None = field(default=None, init=False, repr=False)
    _quiet_since: float | None = field(default=None, init=False, repr=False)
    _fired: str | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @property
    def fired(self) -> str | None:
        """The reason the watchdog ended the run, once it has."""
        return self._fired

    async def check(self) -> str | None:
        """One observation: the reason once the run is stopped, else None."""
        if self._fired is not None:
            return self._fired
        now = self.clock()
        if self.dead():
            return self._fire("the engine died")
        try:
            snapshot = await self.metrics()
        except Exception as error:  # an unreadable engine is a quiet one
            snapshot = None
            gauges = f"metrics unreadable: {type(error).__name__}: {error}"
        else:
            gauges = "no metrics" if snapshot is None else _gauges(snapshot)
        tokens = None if snapshot is None else snapshot.scalars.get(GENERATION_TOKENS)
        demand = (
            None
            if snapshot is None
            else snapshot.scalars.get(REQUESTS_RUNNING, 0.0) + snapshot.scalars.get(REQUESTS_WAITING, 0.0)
        )
        moved = tokens is not None and tokens != self._tokens
        if tokens is not None:
            self._tokens = tokens
        if moved or demand == 0.0 or self._quiet_since is None:
            self._quiet_since = now
            return None
        quiet_for = now - self._quiet_since
        if quiet_for < self.timeout_s:
            return None
        return self._fire(f"no tokens generated for {quiet_for:.0f} s ({gauges})")

    def _fire(self, reason: str) -> str:
        """Record the reason, say so, and end the run."""
        self._fired = reason
        print(f"[watchdog] {reason}; stopping the run", flush=True)
        self.stop(reason)
        return reason

    def start(self) -> None:
        """Watch on ``interval_s`` until the run ends or the watchdog fires."""
        assert self._task is None, "watchdog already started"
        self._task = asyncio.create_task(self._loop(), name="engine-stall-watchdog")

    async def stop_watching(self) -> None:
        """Stop watching and wait for the loop to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        """Look every ``interval_s``. Never fatal: a broken observation must
        not take the run down, and a watchdog that died says so where the
        engine line is read."""
        try:
            while self._fired is None:
                await asyncio.sleep(self.interval_s)
                await self.check()
        except asyncio.CancelledError:
            raise
        except BaseException:
            print("[watchdog] the stall watchdog died:", flush=True)
            traceback.print_exc()


def _gauges(snapshot: ullm.engine_monitor.EngineMetrics) -> str:
    """The engine's request gauges and its counter, as the reason names them."""
    running = int(snapshot.scalars.get(REQUESTS_RUNNING, 0.0))
    waiting = int(snapshot.scalars.get(REQUESTS_WAITING, 0.0))
    generated = int(snapshot.scalars.get(GENERATION_TOKENS, 0.0))
    return f"running={running} waiting={waiting} generated={generated}"
