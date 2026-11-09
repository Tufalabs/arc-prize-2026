"""The permits a search holds: the kernels' CPU permit, and a search slot.

A search child occupies a core by construction; so does a REPL cell running
the model's numpy. Capped separately, a node could sit at both limits at once,
so a search contends on the kernel pool's own permit. It also holds a slot of
its own, because each child may take its whole memory cap and a machine's
memory bounds how many run at once.
"""

from __future__ import annotations

import asyncio
from typing import Iterator

import pytest
import ullm.tools.python as python_tool

from seal import search_process


@pytest.fixture
def fresh_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[python_tool.KernelPool]:
    """A pool of this test's own, standing in for the module default.

    Named per test so it never resolves the counter a real run uses, and
    swapped in for ``default_kernel_pool`` because that is what the REPL and
    the search each reach for.
    """
    pool = python_tool.KernelPool(name_prefix="test_cpu_permit")
    monkeypatch.setattr(python_tool, "default_kernel_pool", lambda: pool)
    monkeypatch.setattr(search_process, "default_kernel_pool", lambda: pool)
    yield pool


def test_the_search_and_the_repl_hold_the_same_cpu_permit(fresh_pool: python_tool.KernelPool) -> None:
    """Same object, so there is one counter and one cap — not two that agree."""
    assert search_process.cpu_semaphore() is fresh_pool.cpu_semaphore()


def test_searches_are_capped_by_a_slot_of_their_own() -> None:
    """Not the CPU permit: the kernels' cap is sized for cores, and searches at
    their memory cap could exhaust a machine's memory well before its cores."""
    slots = search_process.search_semaphore()
    assert slots.limit == search_process.SEARCH_SLOTS
    assert slots.name != search_process.cpu_semaphore().name


async def test_a_search_waits_for_a_slot_before_it_takes_a_core(
    fresh_pool: python_tool.KernelPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Queued for memory, a search holds no core: with every slot taken, the
    CPU permit stays free for the kernels."""
    monkeypatch.setattr(search_process, "SEARCH_SLOTS", 1)
    slots = search_process.SharedSemaphore("test_search_slots", 1)
    monkeypatch.setattr(search_process, "search_semaphore", lambda: slots)
    cpu = fresh_pool.cpu_semaphore()
    await slots.acquire()
    waiting = asyncio.ensure_future(
        search_process.run_search(None, None, [], seed=0, hard_timeout=1.0)  # type: ignore[arg-type]
    )
    await asyncio.sleep(0.05)
    assert not waiting.done()
    free = cpu.limit
    for _ in range(free):
        await cpu.acquire()
    for _ in range(free):
        cpu.release()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    slots.release()
