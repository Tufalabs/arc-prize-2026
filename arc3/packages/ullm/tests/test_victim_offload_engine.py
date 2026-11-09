"""Victim-cache KV offload driven through a real vLLM engine (gpu tier).

The policy unit tests in `test_victim_offload.py` reach the selection logic
without an engine. This covers what only an engine can show: that the
connector is accepted by vLLM's factory, that it writes nothing while the GPU
pool has headroom, that it starts writing once the pool is under pressure, and
that what it wrote can be read back as an external prefix-cache hit.

The engine comes from the `victim_offload_engine` fixture in `conftest.py`,
which is where this package keeps its engine fixtures.
"""

from __future__ import annotations

import asyncio
import typing
import uuid

import pytest
import ullm.engine_monitor as engine_monitor

# Long enough to span several blocks, so one prompt occupies enough of the
# small pool to push earlier ones out — and short enough to stay inside
# VICTIM_MAX_MODEL_LEN, which these synthetic words reach at about five
# tokens each rather than the one they look like.
PROMPT_WORDS = 96
# Prompts held resident at once. Spilling is driven by how few blocks are FREE,
# which sequential requests never affect — each frees its own before the next
# arrives, however many pass through. Only simultaneous residency oversubscribes
# the pool, and this is several times what it can hold.
CONCURRENT_PROMPTS = 32
# Tokens each request generates. Residency, not throughput, is what fills the
# pool: a request that retires in two steps frees its blocks before the next
# wave needs them, so the pool churns at part capacity however many requests
# pass through. Long generations keep them all alive at once.
GENERATED_TOKENS = 128


def make_prompt(seed: int) -> str:
    """A distinct prompt of roughly `PROMPT_WORDS` words, stable across runs.

    Distinct prefixes are the point: shared ones would hit in the GPU cache and
    never reach the connector.
    """
    return " ".join(f"w{seed}x{i}" for i in range(PROMPT_WORDS))


async def run_prompt(engine: typing.Any, prompt: str, max_tokens: int = GENERATED_TOKENS) -> None:
    """Generate `max_tokens` tokens for `prompt`, draining the stream."""
    import vllm

    params = vllm.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    async for _ in engine.generate(prompt, params, str(uuid.uuid4())):
        pass


async def offload_counters() -> tuple[float, float, float]:
    """Host-tier bytes stored, lookups made against it, and hits, so far.

    Queries separate a lookup that never ran from one that ran and missed —
    two different bugs that read identically from the hit counter alone.
    """
    metrics = await engine_monitor.inprocess_source()
    return (
        metrics.scalars.get("vllm:kv_offload_store_bytes", 0.0),
        metrics.scalars.get("vllm:external_prefix_cache_queries", 0.0),
        metrics.scalars.get("vllm:external_prefix_cache_hits", 0.0),
    )


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="module")
async def test_nothing_is_written_while_the_pool_has_headroom(
    victim_offload_engine: typing.Any,
) -> None:
    """One prompt into an empty pool must not touch the host tier.

    The stock connector writes here — it stored 100 MB/s at 22% occupancy on a
    real run — and every such byte is a resident block being mirrored.
    """
    await run_prompt(victim_offload_engine, make_prompt(0))

    stored, _, _ = await offload_counters()
    assert stored == 0.0


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="module")
async def test_a_prompt_finishing_under_pressure_is_read_back_later(
    victim_offload_engine: typing.Any,
) -> None:
    """The connector's contract: what finishes under pressure is preserved.

    Long decodes keep the pool below the watermark while a short prompt
    finishes among them, so its blocks are stored while the engine still
    holds them. A wave of distinct prompts then evicts everything, and the
    short prompt's second run must find its prefix on the host. Content that
    finishes during HEADROOM carries no such promise — the pool is its cache
    then, and the head-spill at the cliff is only a best effort.
    """
    # Enough long-decoding occupiers to keep free blocks under the watermark
    # for several seconds; their growing decodes oversubscribe the pool.
    occupiers = [
        asyncio.create_task(run_prompt(victim_offload_engine, make_prompt(seed), max_tokens=256))
        for seed in range(100, 112)
    ]
    # Let them prefill and start decoding before the visitor arrives.
    await asyncio.sleep(2)
    revisited = make_prompt(1)
    await run_prompt(victim_offload_engine, revisited, max_tokens=8)
    stored_after_hold, _, hits_before = await offload_counters()
    await asyncio.gather(*occupiers)

    assert stored_after_hold > 0.0, "nothing was stored while the pool was under pressure"

    # Distinct prompts cycle the pool so nothing of the visitor survives on GPU.
    await asyncio.gather(
        *(run_prompt(victim_offload_engine, make_prompt(seed), max_tokens=8) for seed in range(200, 232))
    )

    await run_prompt(victim_offload_engine, revisited, max_tokens=8)

    _, queries, hits_after = await offload_counters()
    assert hits_after > hits_before, (
        f"the held prefix did not come back (queries={queries}, hits={hits_after}: "
        "zero queries means the lookup never ran; queries without hits mean the "
        "host did not have the keys)"
    )


def make_shared_prefix_prompt(seed: int) -> str:
    """A prompt whose leading half is common to every seed.

    Concurrent requests over one prefix each prefill it independently, before
    any of them has cached it, so the pool ends up holding several blocks that
    carry the same hash — a state distinct prompts cannot produce.
    """
    shared = " ".join(f"shared{i}" for i in range(PROMPT_WORDS // 2))
    return shared + " " + " ".join(f"s{seed}t{i}" for i in range(PROMPT_WORDS // 2))


@pytest.mark.gpu
@pytest.mark.asyncio(loop_scope="module")
async def test_blocks_sharing_a_hash_do_not_kill_the_engine(
    victim_offload_engine: typing.Any,
) -> None:
    """Spill a pool holding duplicate blocks, then keep using the engine.

    The pool does not de-duplicate, so one offload key can have several blocks
    behind it; a spill that offers every one of them writes more sources than
    the destination spec has slots. What that corrupts surfaces later, as
    vLLM's own assertion that a matched prefix's blocks still carry their
    hashes — so the check is that the engine survives and still serves.
    """
    # First wave shares one prefix, which is what creates the duplicates.
    await asyncio.gather(
        *(run_prompt(victim_offload_engine, make_shared_prefix_prompt(seed)) for seed in range(CONCURRENT_PROMPTS))
    )
    # Second wave pushes the pool under the watermark so those duplicates are
    # spilled, and re-matches the shared prefix while they are in flight.
    await asyncio.gather(
        *(run_prompt(victim_offload_engine, make_shared_prefix_prompt(seed)) for seed in range(CONCURRENT_PROMPTS))
    )

    stored, _, _ = await offload_counters()
    assert stored > 0.0, "the duplicate-carrying pool never spilled"

    # A dead EngineCore surfaces on the next request, not at the crash.
    await run_prompt(victim_offload_engine, make_shared_prefix_prompt(9999))
