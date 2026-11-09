"""The engine stall watchdog (``platypus.watchdog``): it ends the run when
the engine's generated-token counter stops moving while the engine still
reports requests, or when the engine has died; it holds its peace while
tokens flow or nothing is asked of the engine, and it fires once.
"""

from __future__ import annotations

import ullm.engine_monitor

from platypus.watchdog import GENERATION_TOKENS, REQUESTS_RUNNING, REQUESTS_WAITING, EngineStallWatchdog


class _Engine:
    """A fake engine: its metrics as the watchdog reads them, a clock the
    test advances, and what the watchdog did about it."""

    def __init__(self) -> None:
        self.tokens = 0.0
        self.running = 0.0
        self.waiting = 0.0
        self.now = 1000.0
        self.dead = False
        self.unreadable = False
        self.stops: list[str] = []
        self.watchdog = EngineStallWatchdog(
            metrics=self._metrics,
            dead=lambda: self.dead,
            stop=self.stops.append,
            timeout_s=600.0,
            clock=lambda: self.now,
        )

    async def _metrics(self) -> ullm.engine_monitor.EngineMetrics | None:
        if self.unreadable:
            raise ConnectionError("the engine's registry is gone")
        snapshot = ullm.engine_monitor.EngineMetrics()
        snapshot.scalars.update(
            {GENERATION_TOKENS: self.tokens, REQUESTS_RUNNING: self.running, REQUESTS_WAITING: self.waiting}
        )
        return snapshot

    async def tick(self, seconds: float, *, tokens: float = 0.0) -> str | None:
        """Advance the clock, generate ``tokens``, and let the watchdog look."""
        self.now += seconds
        self.tokens += tokens
        return await self.watchdog.check()


async def test_tokens_flowing_never_fire() -> None:
    engine = _Engine()
    engine.running = 40
    for _ in range(20):
        assert await engine.tick(60, tokens=500) is None
    assert engine.stops == []


async def test_a_frozen_engine_with_requests_fires_once_after_the_timeout() -> None:
    engine = _Engine()
    engine.running, engine.waiting = 44, 14
    assert await engine.tick(0, tokens=100) is None
    for _ in range(9):
        assert await engine.tick(60) is None
    reason = await engine.tick(60)
    assert reason is not None
    assert reason.startswith("no tokens generated for 600 s")
    assert "running=44 waiting=14 generated=100" in reason
    assert engine.stops == [reason]
    # Later looks repeat the verdict and stop nothing twice.
    assert await engine.tick(60) == reason
    assert engine.stops == [reason]


async def test_an_idle_engine_is_not_a_stall_and_the_clock_starts_with_demand() -> None:
    engine = _Engine()
    for _ in range(30):
        assert await engine.tick(60) is None
    # Requests arrive and the engine never answers: the clock counts from
    # the demand, not from the idle half hour before it.
    engine.running = 3
    for _ in range(9):
        assert await engine.tick(60) is None
    assert await engine.tick(60) is not None
    assert len(engine.stops) == 1


async def test_tokens_arriving_restart_the_clock() -> None:
    engine = _Engine()
    engine.running = 10
    assert await engine.tick(0, tokens=1) is None
    for _ in range(8):
        assert await engine.tick(60) is None
    assert await engine.tick(60, tokens=1) is None  # a token at 540 s
    for _ in range(9):
        assert await engine.tick(60) is None
    assert await engine.tick(60) is not None


async def test_metrics_that_cannot_be_read_count_as_silence() -> None:
    engine = _Engine()
    engine.running = 5
    assert await engine.tick(0, tokens=10) is None
    engine.unreadable = True
    for _ in range(9):
        assert await engine.tick(60) is None
    reason = await engine.tick(60)
    assert reason is not None and "metrics unreadable: ConnectionError" in reason


async def test_a_dead_engine_fires_at_the_next_look() -> None:
    engine = _Engine()
    engine.running = 5
    assert await engine.tick(0, tokens=10) is None
    engine.dead = True
    assert await engine.tick(1) == "the engine died"
    assert engine.stops == ["the engine died"]
