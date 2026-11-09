"""The per-instance floor on the gap between consecutive calls (``Backend.min_request_gap_s``)."""

from __future__ import annotations

import time

import ullm.backend


async def test_the_floor_waits_only_for_what_the_caller_did_not_already_spend() -> None:
    """No wait before the first call, none when the caller was away longer
    than the floor, and only the remainder otherwise."""
    backend = ullm.backend.DummyBackend(min_request_gap_s=0.3)

    t0 = time.monotonic()
    await backend._pace()  # never called: no floor yet
    first = time.monotonic() - t0

    backend._last_response_ended = time.monotonic() - 1.0  # away longer than the floor
    t0 = time.monotonic()
    await backend._pace()
    away = time.monotonic() - t0

    backend._last_response_ended = time.monotonic() - 0.1  # back after a tenth: waits the rest
    t0 = time.monotonic()
    await backend._pace()
    remainder = time.monotonic() - t0

    assert first < 0.05 and away < 0.05
    assert 0.15 <= remainder < 0.3


async def test_no_floor_by_default() -> None:
    """A backend built without a floor never waits, whatever the stamp says."""
    backend = ullm.backend.DummyBackend()
    backend._last_response_ended = time.monotonic()
    t0 = time.monotonic()
    await backend._pace()
    assert time.monotonic() - t0 < 0.05
