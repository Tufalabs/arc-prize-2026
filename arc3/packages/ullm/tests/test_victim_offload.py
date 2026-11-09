"""Victim-cache KV offload policy (`ullm.backends.victim_offload`).

Covers what separates a victim cache from the stock mirroring connector:
the band's spill order and bounds, the finish hold, hash-mismatch handling,
the two eviction classes and their reclassification, a boundary aging as
one, checkpoint classification at lookup and finish, the shared and
kept-checkpoint stores, the keep-young pass, the departure and zero-lookup
classifiers, the trace, the fill's answer and load and its clearing, and
the version guard (`kv_offload.md`, "How it was tested").

The GPU block pool is real rather than mocked — it is pure Python — so these
exercise the same structures the engine hands the connector. The scheduler is
built without its spec, which needs a full engine config and reaches none of
the code under test.
"""

from __future__ import annotations

import collections
import io
import json
import pathlib
import types
import typing

import pytest
import ullm.backends.victim_offload as victim_offload
import ullm.engine_monitor
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.common as offload_common
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics as offload_metrics
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as offload_sched
import vllm.v1.core.block_pool
import vllm.v1.core.kv_cache_utils
import vllm.v1.kv_offload.base as offload_base
import vllm.v1.outputs
import vllm.v1.request

# One group, matching the single-group shape the ordering tests care about;
# the group id is packed into the block hash the same way either way.
GROUP_IDX = 0
HASH_BLOCK_SIZE = 16


def make_pool(num_blocks: int) -> vllm.v1.core.block_pool.BlockPool:
    """A caching block pool with `num_blocks` blocks, as the engine builds it."""
    return vllm.v1.core.block_pool.BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=HASH_BLOCK_SIZE,
    )


def give_hash(block: typing.Any, seed: int, group: int = GROUP_IDX) -> None:
    """Mark a block as full and cached, which is what makes it spillable."""
    raw = vllm.v1.core.kv_cache_utils.BlockHash(seed.to_bytes(32, "big"))
    block._block_hash = vllm.v1.core.kv_cache_utils.BlockHashWithGroupId(raw + group.to_bytes(4, "big", signed=False))


def key_of(block: typing.Any) -> offload_base.OffloadKey:
    """The offload key the connector derives for a hashed block."""
    return offload_base.make_offload_key(
        vllm.v1.core.kv_cache_utils.get_block_hash(block.block_hash),
        vllm.v1.core.kv_cache_utils.get_group_id(block.block_hash),
    )


def make_scheduler(
    pool: vllm.v1.core.block_pool.BlockPool | None,
    watermark: float = victim_offload.DEFAULT_SPILL_WATERMARK,
    max_spill: int = victim_offload.DEFAULT_MAX_SPILL_BLOCKS,
    band: int = victim_offload.DEFAULT_SPILL_BAND_BLOCKS,
    num_groups: int = 1,
    single_state_groups: typing.Collection[int] = (),
) -> victim_offload.VictimOffloadingScheduler:
    """A scheduler carrying only the state the spill policy reads."""
    scheduler = object.__new__(victim_offload.VictimOffloadingScheduler)
    scheduler._gpu_block_pool = pool
    scheduler._spill_watermark = watermark
    scheduler._max_spill_blocks = max_spill
    scheduler._spill_band_blocks = band
    scheduler._spill_disabled_warned = False
    scheduler._spilled = {}
    scheduler._victim_job_ids = set()
    scheduler._jobs = {}
    scheduler._job_counter = 0
    scheduler._stale_job_threshold = 0
    scheduler._demote = None
    # The drop path reports to the audit; a bare proxy keeps its own sets
    # and forwards nothing, which is all these scheduler-only tests need.
    scheduler._lookup_audit = victim_offload._LookupAudit(types.SimpleNamespace(), lambda: None)
    scheduler._drain_hit_depths = None
    scheduler._drain_hit_records = None
    scheduler._spill_holds = {}
    scheduler._hit_depth_policy = None
    scheduler._retention = None
    scheduler._finish_patterns_logged = 0
    scheduler._hit_depth_run_max = 0
    scheduler._pending_finish_jobs = {}
    scheduler._held_finish_jobs = set()
    scheduler._held_this_step = frozenset()
    scheduler._pressure_until = 0.0
    scheduler._recent_block_roles = {}
    scheduler._current_batch_load_jobs = {}
    # Stock request_finished registers finished jobs here so a later
    # block reuse can flush them.
    scheduler._block_id_to_pending_jobs = {}
    scheduler._req_status = {}
    scheduler._fills = {}
    scheduler._fills_loaded = 0
    scheduler._fill_mismatch_warned = False
    scheduler._chunks_being_loaded = None
    scheduler._blocks_held = 0
    scheduler._passed_skipped = 0
    scheduler._dead_skipped = 0
    scheduler._dead_keys = collections.OrderedDict()
    scheduler._demote_dead = None
    scheduler._park = False
    scheduler._gpu_park = False
    scheduler._gpu_park_reserve = victim_offload.DEFAULT_GPU_PARK_RESERVE
    scheduler._admission_watermark = 0.0
    scheduler._park_needs = {}
    scheduler._kv_share_at = 0.0
    scheduler._tail_pins = {}
    scheduler._connector_stats = offload_metrics.OffloadingConnectorStats()
    scheduler._parked = {}
    scheduler._park_owner = {}
    scheduler._park_jobs = {}
    scheduler._park_retire = []
    scheduler._park_started = set()
    scheduler._park_deferred = {}
    scheduler._park_active = {}
    scheduler._park_snapshot_at = 0.0
    scheduler._tail_keys = {}
    scheduler._tail_step_keys = set()
    scheduler._park_counts = collections.Counter()
    scheduler._park_log = None
    scheduler._band_present = 0
    scheduler._shared_stored = 0
    scheduler._checkpoints_stored = 0
    scheduler._reage_pending = []
    scheduler._reaged = 0
    scheduler._kept_last_step = 0
    scheduler._kept_missing_last_step = 0
    scheduler._checkpoints_gone = 0
    scheduler._step = 0
    scheduler._reaged_at = {}
    scheduler._queue_snapshot = {}
    scheduler._departures = collections.Counter()
    scheduler._blocks_spilled = 0
    scheduler._blocks_dropped = 0
    scheduler._jobs_with_drops = 0
    scheduler._last_tally_at = 0.0
    scheduler._trace = victim_offload._Trace(None)
    scheduler._trace_started = False
    scheduler._engine_log = False
    scheduler._sizing = None
    scheduler._trace_lookups = {}
    scheduler._finished_turns = collections.OrderedDict()
    scheduler._kept_map = {}
    scheduler._kept_map_prev = {}
    scheduler._kept_blocks = []
    scheduler._kept_stored = 0
    scheduler._queue_entered = {}
    scheduler._last_sample_at = 0.0
    scheduler._sample_counters = collections.Counter()
    scheduler._victim_req_context = offload_base.ReqContext(req_id=victim_offload.VICTIM_REQ_ID)
    # Only the fields the store paths read: the spill path counts the groups,
    # and the finish path takes every hash at factor 1.
    scheduler.config = offload_sched.SchedulerOffloadConfig(
        kv_group_configs=typing.cast(
            typing.Any,
            tuple(
                types.SimpleNamespace(
                    hashes_per_chunk=1,
                    tokens_per_chunk=HASH_BLOCK_SIZE,
                    # 1 is how the stock scheduler marks a Mamba/GDN group;
                    # None is full attention.
                    sliding_window_size_in_chunks=1 if g in single_state_groups else None,
                )
                for g in range(num_groups)
            ),
        ),
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
        tokens_per_hash=HASH_BLOCK_SIZE,
        supports_partial_tail=False,
    )
    return scheduler


def free_blocks(pool: vllm.v1.core.block_pool.BlockPool, count: int) -> list[typing.Any]:
    """Allocate `count` blocks, hash them, and release them in order.

    Returns them in the order they were freed, which is the order the pool will
    evict them in.
    """
    blocks = pool.get_new_blocks(count)
    for seed, block in enumerate(blocks, start=1):
        give_hash(block, seed)
    pool.free_blocks(blocks)
    return blocks


def test_no_spill_while_the_eviction_frontier_is_unhashed() -> None:
    """A pool whose free-queue front is fresh blocks must produce no store job.

    This is the whole point of the connector: the stock one wrote 100 MB/s at
    22% occupancy, and anything written here is a resident block being
    mirrored. The cached blocks sit behind the fresh ones, beyond the band.
    """
    pool = make_pool(1024)
    free_blocks(pool, 8)
    scheduler = make_scheduler(pool, band=256)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())

    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) == {}


def test_the_band_spills_cached_blocks_at_the_front_whatever_the_headroom() -> None:
    """The pool allocates from the queue front whether the queue is short or
    long, so cached blocks within the band are about to die even when the
    free fraction says the pool is comfortable."""
    pool = make_pool(128)
    free_blocks(pool, 8)
    scheduler = make_scheduler(pool, watermark=0.15, band=256)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    assert pool.get_num_free_blocks() > int(pool.num_gpu_blocks * 0.15)
    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) != {}
    assert len(manager.offered[0]) == 8


def test_unbound_pool_spills_nothing_and_warns_once() -> None:
    """Without a bound pool there is no victim set, and it is worth saying so.

    A topology that never binds would otherwise offload silently forever,
    looking identical to a run whose cache never filled.
    """
    scheduler = make_scheduler(None)

    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) == {}
    assert scheduler._spill_disabled_warned
    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) == {}


def test_victims_come_out_in_the_pool_s_own_eviction_order() -> None:
    """Spill order must be the free queue's order, head first.

    Head-first spilling is what makes the two tiers concatenate into one LRU
    stack: inserted at the CPU tier's recent end in this order, the last block
    spilled ends up on top.
    """
    pool = make_pool(64)
    freed = free_blocks(pool, 10)
    scheduler = make_scheduler(pool)

    victims = scheduler._collect_victims(pool, band=64)

    assert [b.block_id for b in victims] == [b.block_id for b in freed]


def test_unhashed_blocks_are_never_spilled() -> None:
    """A block with no hash can never be matched, so a copy could never load.

    The pool puts these at the very front of the free queue, so a policy that
    took the head blindly would spill nothing else.
    """
    pool = make_pool(64)
    hashed = pool.get_new_blocks(4)
    for seed, block in enumerate(hashed, start=1):
        give_hash(block, seed)
    unhashed = pool.get_new_blocks(3)
    pool.free_blocks(unhashed)
    pool.free_blocks(hashed)
    scheduler = make_scheduler(pool)

    victims = scheduler._collect_victims(pool, band=64)

    assert {b.block_id for b in victims}.isdisjoint({b.block_id for b in unhashed})
    assert [b.block_id for b in victims] == [b.block_id for b in hashed]


def test_the_band_bounds_the_walk_in_queue_entries() -> None:
    """Unhashed entries count toward the band: a front of fresh blocks is
    not walked through to the cached content behind it."""
    pool = make_pool(11)  # block 0 is the null block; the other ten are all allocated below
    hashed = pool.get_new_blocks(4)
    for seed, block in enumerate(hashed, start=1):
        give_hash(block, seed)
    unhashed = pool.get_new_blocks(6)
    pool.free_blocks(unhashed)
    pool.free_blocks(hashed)
    scheduler = make_scheduler(pool)

    assert scheduler._collect_victims(pool, band=6) == []
    assert len(scheduler._collect_victims(pool, band=8)) == 2


def test_new_stores_stop_at_the_spill_cap_but_present_keys_are_all_touched() -> None:
    """The per-step cap bounds how much PCIe traffic one step can queue; it
    does not bound the touches, which cost nothing and keep the host's order
    honest for the whole band."""
    pool = make_pool(64)
    freed = free_blocks(pool, 20)
    scheduler = make_scheduler(pool, max_spill=5)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert manager.offered[0] == [key_of(b) for b in freed[:5]]


class RecordingManager:
    """Offload manager that records what the spill policy asks of it.

    Keys in `present` are reported as already held, which is what makes the
    policy touch rather than store them.
    """

    def __init__(self, present: typing.Collection[offload_base.OffloadKey] = ()) -> None:
        """Start with nothing recorded and `present` already on the host."""
        self.present = set(present)
        self.offered: list[list[offload_base.OffloadKey]] = []
        self.touched: list[list[offload_base.OffloadKey]] = []
        self.completed: list[tuple[list[offload_base.OffloadKey], bool]] = []

    def on_new_request(self, req_context: typing.Any) -> None:
        """Accepted and ignored, as the base bookkeeping calls it."""

    def on_request_finished(self, req_context: typing.Any) -> None:
        """Accepted and ignored, as the base bookkeeping calls it."""

    def prepare_store(
        self, keys: typing.Any, req_context: typing.Any, source: str = "engine"
    ) -> offload_base.PrepareStoreOutput:
        """Record the offer; allocate a slot for every key not already held."""
        keys = list(keys)
        self.offered.append(keys)
        to_store = [k for k in keys if k not in self.present]
        return offload_base.PrepareStoreOutput(
            keys_to_store=to_store, store_spec=typing.cast(typing.Any, object()), evicted_keys=[]
        )

    def prepare_load(self, keys: typing.Any, req_context: typing.Any) -> typing.Any:
        """Record the load and hand back a stand-in source spec."""
        self.loaded = list(keys)
        return object()

    def touch(self, keys: typing.Any, req_context: typing.Any) -> None:
        """Record a touch in the order given."""
        self.touched.append(list(keys))

    def lookup(self, key: typing.Any, req_context: typing.Any) -> offload_base.LookupResult:
        """Held keys hit; everything else misses."""
        return offload_base.LookupResult.HIT if key in self.present else offload_base.LookupResult.MISS

    def complete_store(self, keys: typing.Any, req_context: typing.Any, success: bool = True) -> None:
        """Record a completion in the order given."""
        self.completed.append((list(keys), success))

    def reset_cache(self) -> None:
        """Forget everything held, as the stock manager's reset does."""
        self.present.clear()


def worker_completed(job_id: int) -> vllm.v1.outputs.KVConnectorOutput:
    """A worker report saying `job_id` finished on one worker."""
    meta = offload_common.OffloadingWorkerMetadata(completed_jobs={job_id: 1})
    return vllm.v1.outputs.KVConnectorOutput(kv_connector_worker_meta=meta)


@pytest.mark.parametrize(("band", "expect_offered"), [(0, False), (64, True)])
def test_the_band_gates_whether_victims_are_offered(band: int, expect_offered: bool) -> None:
    """At 0 the connector never spills; at the pool size every step may.

    The two ends are what a sweep will reach for.
    """
    pool = make_pool(64)
    free_blocks(pool, 4)
    scheduler = make_scheduler(pool, band=band)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    jobs = scheduler._build_store_jobs(typing.cast(typing.Any, None))
    assert bool(manager.offered) is expect_offered
    assert bool(jobs) is expect_offered


def test_two_blocks_with_one_hash_yield_one_store() -> None:
    """The pool does not de-duplicate, so two blocks can carry the same hash.

    A store spec pairs one source block with one destination slot, so emitting
    both against a single key writes more sources than there are slots. Games
    sharing a long common prefix produce these routinely.
    """
    pool = make_pool(64)
    blocks = pool.get_new_blocks(4)
    for block in blocks:
        give_hash(block, 7)
    pool.free_blocks(blocks)
    scheduler = make_scheduler(pool, watermark=1.0)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert manager.offered, "the watermark should have opened"
    offered = manager.offered[0]
    assert len(offered) == len(set(offered)) == 1


def test_the_spill_leaves_the_pool_untouched() -> None:
    """Spilling must not change the pool's free accounting or block state.

    The connector reads the pool and copies out of it; every earlier attempt to
    also hold blocks against reuse ended up fighting an invariant of vLLM's,
    because an unreferenced block is the pool's to give away at any moment.
    """
    pool = make_pool(64)
    freed = free_blocks(pool, 6)
    scheduler = make_scheduler(pool, watermark=1.0)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    before_free = pool.get_num_free_blocks()
    before_refs = [b.ref_cnt for b in freed]

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert manager.offered, "the watermark should have opened"
    assert pool.get_num_free_blocks() == before_free
    assert [b.ref_cnt for b in freed] == before_refs
    assert [b.block_id for b in scheduler._collect_victims(pool, band=64)] == [b.block_id for b in freed]


def test_keys_are_offered_group_major_to_match_the_source_blocks() -> None:
    """The worker pairs sources with slots by position, group by group.

    So the order keys are offered to the manager — which is the order slots are
    allocated in — must be the order the source block ids are listed in, or a
    group-1 block lands in a group-0 slot. Victims arrive interleaved across
    groups whenever one request's blocks follow another's in the free queue.
    """
    pool = make_pool(64)
    blocks = pool.get_new_blocks(4)
    for seed, (block, group) in enumerate(zip(blocks, (1, 0, 1, 0)), start=1):
        give_hash(block, seed, group)
    pool.free_blocks(blocks)
    scheduler = make_scheduler(pool, watermark=1.0, num_groups=2)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    (job,) = scheduler._build_store_jobs(typing.cast(typing.Any, None)).values()
    src = typing.cast(typing.Any, job.src_spec)

    victims = scheduler._collect_victims(pool, band=64)
    expect = [b for g in (0, 1) for b in victims if vllm.v1.core.kv_cache_utils.get_group_id(b.block_hash) == g]
    assert manager.offered[0] == [key_of(b) for b in expect]
    assert list(src.block_ids) == [b.block_id for b in expect]
    assert list(src.group_sizes) == [2, 2]


def test_a_victim_the_host_already_holds_is_touched_not_stored() -> None:
    """A block reaching the eviction front again is about to stop being a
    duplicate of what the GPU has: that is the use a victim cache orders
    by. Left untouched it would keep the position of its first spill however
    often it bounced between the tiers since."""
    pool = make_pool(64)
    freed = free_blocks(pool, 4)
    held = {key_of(freed[1]), key_of(freed[2])}
    scheduler = make_scheduler(pool, watermark=1.0)
    manager = RecordingManager(present=held)
    scheduler.manager = typing.cast(typing.Any, manager)

    (job,) = scheduler._build_store_jobs(typing.cast(typing.Any, None)).values()

    assert set(typing.cast(typing.Any, job.src_spec).block_ids) == {freed[0].block_id, freed[3].block_id}
    # Newest-first: touch makes the first key it sees the most recent, and the
    # free queue hands victims out oldest first.
    assert manager.touched == [[key_of(freed[2]), key_of(freed[1])]]
    assert scheduler._band_present == 2


def test_a_block_beyond_the_band_is_neither_stored_nor_touched() -> None:
    """Deeper in the queue a block is one a prefix hit may take back before
    it leaves the GPU; the band does not look there."""
    pool = make_pool(7)  # block 0 is the null block; the queue is exactly the six freed below
    freed = free_blocks(pool, 6)
    scheduler = make_scheduler(pool, band=3)
    manager = RecordingManager(present={key_of(freed[4])})
    scheduler.manager = typing.cast(typing.Any, manager)

    (job,) = scheduler._build_store_jobs(typing.cast(typing.Any, None)).values()

    assert set(typing.cast(typing.Any, job.src_spec).block_ids) == {b.block_id for b in freed[:3]}
    assert manager.touched == []


def test_completion_hands_keys_back_in_spill_order() -> None:
    """The host tier's LRU order for a batch is the order its keys complete in.

    The job tracks its keys as a set; handing that back would make the order of
    a whole batch arbitrary, and a hit that loses one block in the middle of it
    is truncated to everything before that block.
    """
    pool = make_pool(64)
    freed = free_blocks(pool, 3)
    scheduler = make_scheduler(pool, watermark=1.0)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    (job_id,) = scheduler._build_store_jobs(typing.cast(typing.Any, None))

    scheduler.update_connector_output(worker_completed(job_id))

    assert manager.completed == [([key_of(b) for b in freed], True)]
    assert job_id not in scheduler._jobs


def test_a_block_reused_mid_copy_is_dropped_and_the_rest_kept() -> None:
    """A victim is the pool's to reuse at any moment; that copy must not be trusted.

    The block no longer carries the hash it was read under, so what reached
    the host for it was some mixture of two blocks. The other blocks in the
    job copied fine, and under the churn of a cliff nearly every job has one
    reuse in it — so they are kept, in spill order.
    """
    # A pool with no never-used blocks left, so the next allocation has to
    # take a cached one — the head of the queue, which was spilled first.
    pool = make_pool(4)
    freed = free_blocks(pool, 3)
    scheduler = make_scheduler(pool, watermark=1.0)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    (job_id,) = scheduler._build_store_jobs(typing.cast(typing.Any, None))
    keys = [key_of(b) for b in freed]

    # The pool hands the head block out to a new request, which resets it.
    reused = pool.get_new_blocks(1)[0]
    assert reused.block_id == freed[0].block_id
    scheduler.update_connector_output(worker_completed(job_id))

    assert manager.completed == [([keys[1], keys[2]], True), ([keys[0]], False)]
    assert (scheduler._blocks_spilled, scheduler._blocks_dropped, scheduler._jobs_with_drops) == (3, 1, 1)


def test_the_victim_policy_is_registered_with_the_cpu_manager() -> None:
    """Importing the connector is what makes `eviction_policy="victim_lru"` valid.

    The manager resolves the name through the policy factory at construction,
    and a name it does not know is a hard error at engine start. The factory
    loads lazily, so RESOLVING is what proves the registration is usable
    rather than merely present.
    """
    import vllm.v1.kv_offload.cpu.policies.factory as policy_factory

    resolved = policy_factory.CachePolicyFactory.get_cache_policy_cls(victim_offload.VICTIM_POLICY)

    assert resolved is victim_offload.VictimLRUPolicy


def test_demote_sends_a_key_to_the_eviction_end() -> None:
    """A demoted key must be the next one `evict` takes, ahead of older keys."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    keys = [offload_base.OffloadKey(bytes([i]) * 36) for i in range(4)]
    for i, key in enumerate(keys):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0  # a stored, readable block; the constructor marks "not ready"
        policy.insert(key, block)

    policy.demote([keys[3]])

    evicted = policy.evict(2, protected=set())
    assert evicted is not None
    assert [k for k, _ in evicted] == [keys[3], keys[0]]


def test_an_asked_block_records_its_depth_from_the_recent_end() -> None:
    """The reuse-depth CDF is the tier-sizing curve, so the depth must be the
    position at the moment of the ask: newest key depth 0, oldest key depth
    counted over what remains evictable."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    keys = [offload_base.OffloadKey(bytes([i]) * 36) for i in range(4)]
    for i, key in enumerate(keys):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)

    policy.mark_non_evictable(keys[3])  # newest
    policy.mark_non_evictable(keys[0])  # oldest of the 3 still evictable

    assert policy.drain_hit_depths() == [0, 2]
    assert policy.drain_hit_depths() == []  # drained


def test_only_completed_load_jobs_are_collected_for_demotion() -> None:
    """Stores, still-pending loads, and the connector's own jobs are not loads that finished."""
    scheduler = make_scheduler(make_pool(8))
    k = [offload_base.OffloadKey(bytes([i]) * 36) for i in range(4)]
    scheduler._jobs = {
        1: offload_sched.TransferJobStatus(req_id="r", pending_count=1, keys={k[0]}, is_store=False),
        2: offload_sched.TransferJobStatus(req_id="r", pending_count=2, keys={k[1]}, is_store=False),
        3: offload_sched.TransferJobStatus(req_id="r", pending_count=1, keys={k[2]}, is_store=True),
        4: offload_sched.TransferJobStatus(req_id="v", pending_count=1, keys={k[3]}, is_store=False),
    }
    scheduler._victim_job_ids = {4}
    meta = offload_common.OffloadingWorkerMetadata(completed_jobs={1: 1, 2: 1, 3: 1, 4: 1})

    assert scheduler._completed_load_keys(meta) == [k[0]]


class FakeRequest:
    """The two attributes of a vLLM Request that the finish path reads."""

    def __init__(self, request_id: str, block_hashes: list[bytes], num_computed_tokens: int | None = None) -> None:
        """A finished request with the given hash chain; every chunk of it
        computed unless ``num_computed_tokens`` says less."""
        self.request_id = request_id
        self.block_hashes = block_hashes
        self.num_computed_tokens = (
            len(block_hashes) * HASH_BLOCK_SIZE if num_computed_tokens is None else num_computed_tokens
        )
        self.kv_transfer_params: dict[str, typing.Any] | None = None

    def is_finished(self) -> bool:
        """A fake finishing request has finished, as the stock reset asks."""
        return True


def _no_op_method(_self: object) -> None:
    """A method stand-in for the fake request status."""


def tracked(scheduler: typing.Any, request: FakeRequest) -> typing.Any:
    """Register `request` the way _update_req_states would have."""
    status = type(
        "ReqStatus",
        (),
        {
            "req": request,
            "transfer_jobs": set(),
            "req_context": offload_base.ReqContext(req_id=request.request_id),
            # Read by the stock request_finished's lookup-delay metric; None
            # means "no deferred lookup was outstanding", which is this fake's
            # situation and keeps the metric a no-op.
            "deferred_lookup_start_time": None,
            # Stock request_finished flushes the last block's keys before the
            # victim path runs. The finish-hold builds its own keys from
            # request.block_hashes, so a no-op keeps the fake honest.
            "update_offload_keys": _no_op_method,
        },
    )()
    scheduler._req_status = {request.request_id: status}
    return status


def test_a_finish_above_the_watermark_holds_nothing() -> None:
    """With headroom the blocks join the free queue as ordinary cache."""
    pool = make_pool(128)
    scheduler = make_scheduler(pool, watermark=0.15)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    request = FakeRequest("r1", [bytes([9]) * 32])
    tracked(scheduler, request)

    held, _ = scheduler.request_finished_with_blocks(request, ([5],))

    assert held is False
    assert scheduler._pending_finish_jobs == {}
    assert manager.offered == []


def test_a_finish_under_pressure_stores_and_holds() -> None:
    """Under the watermark the delta is stored while the blocks are still held.

    Keys pair with block ids positionally, the padding sentinel is skipped,
    already-hosted keys are touched instead, and the job rides the request id
    so the worker can report the hold released.
    """
    pool = make_pool(8)
    free_blocks(pool, 7)
    blocks = pool.get_new_blocks(7)  # pool exhausted: unequivocally under any watermark
    scheduler = make_scheduler(pool, watermark=0.15, num_groups=2)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    # The request's blocks, hashed as the engine hashes a computed chunk:
    # group 0 holds chunks 0 and 2 (1 is the sentinel), group 1 all three.
    g0_a, g0_c, g1_a, g1_b, g1_c = blocks[:5]
    for block, seed, group in ((g0_a, 1, 0), (g0_c, 3, 0), (g1_a, 1, 1), (g1_b, 2, 1), (g1_c, 3, 1)):
        give_hash(block, seed, group=group)
    already = offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.BlockHash(hashes[0]), 0)
    manager = RecordingManager(present={already})
    scheduler.manager = typing.cast(typing.Any, manager)
    request = FakeRequest("r2", hashes)
    status = tracked(scheduler, request)

    held, _ = scheduler.request_finished_with_blocks(
        request, ([g0_a.block_id, 0, g0_c.block_id], [g1_a.block_id, g1_b.block_id, g1_c.block_id])
    )

    assert held is True
    assert manager.touched == [[already]]
    ((job_id, job),) = scheduler._pending_finish_jobs.items()
    assert job.req_id == "r2"
    src = typing.cast(typing.Any, job.src_spec)
    # group 0 drops the sentinel and the already-hosted key; group 1 keeps all.
    assert list(src.block_ids) == [g0_c.block_id, g1_a.block_id, g1_b.block_id, g1_c.block_id]
    assert list(src.group_sizes) == [1, 3]
    assert job_id in status.transfer_jobs
    assert list(scheduler._jobs[job_id].keys)[0] == offload_base.make_offload_key(
        vllm.v1.core.kv_cache_utils.BlockHash(hashes[2]), 0
    )
    # The next metadata build carries the job out exactly once.
    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) == {job_id: job}
    assert scheduler._pending_finish_jobs == {}


def make_worker() -> typing.Any:
    """A worker half carrying only the held-store state and a scripted backend."""
    worker = object.__new__(victim_offload.VictimOffloadingWorker)
    worker._held_jobs = {}
    worker._held_pending = {}
    worker._load_jobs = {}
    worker._unsubmitted_store_jobs = []
    # Rank 0 of a replicated layout writes stores; non-writers only ack. This
    # fake is the writer, which is the path get_finished's hold reporting sits on.
    worker._is_store_writer = True
    worker._connector_worker_meta = offload_common.OffloadingWorkerMetadata()
    return worker


def test_the_worker_reports_a_held_request_once_all_its_stores_land() -> None:
    """finished_sending is what lets the engine free the held blocks.

    The scheduler names the held stores; nothing else is reported. Job 4 is
    the shape that matters: the engine's own boundary-state store, carrying a
    LIVE request's id, which reporting would free out from under a request
    that is still running. A held request with two jobs is reported exactly
    once, after the second lands.
    """
    worker = make_worker()
    spec = offload_base.GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    meta = victim_offload.HeldStoreMetadata(
        load_jobs={},
        store_jobs={
            1: offload_common.TransferJob(req_id="r9", src_spec=spec, dst_spec=spec),
            2: offload_common.TransferJob(req_id="r9", src_spec=spec, dst_spec=spec),
            3: offload_common.TransferJob(req_id=victim_offload.VICTIM_REQ_ID, src_spec=spec, dst_spec=spec),
            4: offload_common.TransferJob(req_id="still-running", src_spec=spec, dst_spec=spec),
        },
        jobs_to_flush=set(),
        held_job_ids=frozenset({1, 2}),
    )
    worker.prepare_store_kv(meta)
    assert worker._held_jobs == {1: "r9", 2: "r9"}

    class Backend:
        """Scripted transfer backend: one batch of results per drain."""

        def __init__(self) -> None:
            self.batches = [
                [offload_base.TransferResult(job_id=1, success=True)],
                [
                    offload_base.TransferResult(job_id=2, success=True),
                    offload_base.TransferResult(job_id=3, success=True),
                ],
            ]

        def get_finished(self) -> list[typing.Any]:
            return self.batches.pop(0) if self.batches else []

    worker.worker = Backend()
    assert worker.get_finished(set()) == (set(), set())
    assert worker.get_finished(set()) == ({"r9"}, set())
    assert worker._held_jobs == {} and worker._held_pending == {}


def test_pressure_is_sticky_through_a_preemption_spike() -> None:
    """A finish during a free-count spike must still hold.

    A preemption storm frees whole contexts into the queue, so the
    instantaneous count reads headroom at maximal pressure; a finish gated on
    it alone skips exactly the store the hold exists for — measured as the one
    revisited prompt in the engine test never being stored.
    """
    pool = make_pool(64)
    freed = free_blocks(pool, 60)
    scheduler = make_scheduler(pool, watermark=0.15)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    # A step observes the watermark crossed (nearly all blocks allocated)...
    taken = pool.get_new_blocks(60)
    scheduler._build_store_jobs(typing.cast(typing.Any, None))
    assert scheduler._pressure_until > 0
    # ...then a preemption frees them all, spiking the free count back up.
    for block, old in zip(taken, freed):
        give_hash(block, old.block_id + 100)
    pool.free_blocks(taken)
    assert not scheduler._under_watermark(pool)

    give_hash(pool.blocks[9], 7)  # the block the request's one chunk names
    request = FakeRequest("r7", [(7).to_bytes(32, "big")])
    tracked(scheduler, request)
    held, _ = scheduler.request_finished_with_blocks(request, ([9],))

    assert held is True


def test_a_lookup_miss_is_classified_by_the_key_s_history() -> None:
    """The same MISS means three different things; the audit tells them apart.

    Never stored points at key derivation, evicted at the host policy, dropped
    at the validate path — each a different fix, and the engine reports all
    three as one number.
    """
    manager = RecordingManager()
    audit = victim_offload._LookupAudit(manager, lambda: None)
    ctx = offload_base.ReqContext(req_id="r")
    a, b, c, d = (offload_base.OffloadKey(bytes([i]) * 36) for i in range(4))

    audit.prepare_store([a, b, d], ctx)
    manager.present.update({a, b, d})
    assert audit.lookup(a, ctx) is offload_base.LookupResult.HIT

    assert audit.lookup(c, ctx) is offload_base.LookupResult.MISS  # never offered
    manager.present.discard(b)
    audit.evicted.add(b)  # what prepare_store's evicted_keys would have reported
    assert audit.lookup(b, ctx) is offload_base.LookupResult.MISS
    manager.present.discard(d)
    audit.note_dropped([d])
    assert audit.lookup(d, ctx) is offload_base.LookupResult.MISS

    outcomes = {k: v for k, v in audit.counts.items() if not k.startswith(("g", "store_"))}
    assert outcomes == {"hit": 1, "miss_never_stored": 1, "miss_evicted": 1, "miss_dropped": 1}
    # The same outcomes again, keyed by the group each key's tail encodes
    # (store offers are also counted by group, so select the outcome keys).
    outcome_keys = ("_hit", "_miss_", "_retry", "_other")
    by_group = {k: v for k, v in audit.counts.items() if k.startswith("g") and any(s in k for s in outcome_keys)}
    assert sum(by_group.values()) == 4

    # A re-store puts a key back and clears its earlier fate.
    audit.prepare_store([b], ctx)
    manager.present.add(b)
    assert audit.lookup(b, ctx) is offload_base.LookupResult.HIT
    assert b not in audit.evicted


def stored_policy(capacity: int, count: int) -> tuple[victim_offload.VictimLRUPolicy, list[offload_base.OffloadKey]]:
    """A policy holding `count` stored, readable, evictable keys in insertion order."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=capacity)
    keys = [offload_base.OffloadKey(bytes([i]) * 36) for i in range(count)]
    for i, key in enumerate(keys):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)
    return policy, keys


def test_passed_checkpoints_are_evicted_before_any_current_key() -> None:
    """A passed key goes ahead of every current key regardless of age, and
    passed keys keep their own LRU order among themselves."""
    policy, keys = stored_policy(8, 5)

    policy.mark_passed([keys[3], keys[2]])  # keys[3] classified first: older in its class

    evicted = policy.evict(3, protected=set())
    assert evicted is not None
    assert [k for k, _ in evicted] == [keys[3], keys[2], keys[0]]
    assert (policy.evicted_passed, policy.evicted_current) == (2, 1)


def test_a_key_classified_before_it_is_stored_lands_in_its_class() -> None:
    """The finish that classifies a checkpoint can precede its spill."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    key = offload_base.OffloadKey(bytes([7]) * 36)
    policy.mark_passed([key])
    block = policies_base.BlockStatus(0)
    block.ref_cnt = 0
    policy.insert(key, block)

    assert list(policy.passed_blocks) == [key]
    assert key not in policy.evictable_blocks


def test_a_passed_key_s_depth_counts_every_current_key_ahead_of_it() -> None:
    """Depth is the capacity a tier needs for the key to survive: a current
    key needs only the current keys newer than it; a passed key outlives
    nothing current."""
    policy, keys = stored_policy(8, 4)
    policy.mark_passed([keys[1]])

    policy.mark_non_evictable(keys[0])  # oldest current, two current keys newer
    policy.mark_non_evictable(keys[1])  # passed: behind the two remaining current keys

    assert policy.drain_hit_depths() == [2, 2]


def test_a_finish_classifies_the_request_s_single_state_checkpoints() -> None:
    """At each finish the first and last two full boundaries are current and
    the rest passed, whatever blocks the engine still holds; the attention
    group is left alone, and a boundary two turns back is passed."""
    pool = make_pool(128)  # headroom: no hold, classification alone
    scheduler = make_scheduler(pool, watermark=0.15, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    scheduler._retention = policy
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    hashes = [bytes([i]) * 32 for i in (1, 2, 3, 4)]

    def key(i: int, g: int) -> offload_base.OffloadKey:
        return offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.BlockHash(hashes[i]), g)

    request = FakeRequest("r3", hashes[:3])
    tracked(scheduler, request)
    scheduler.request_finished_with_blocks(request, ([11, 0, 13], [21, 22, 23]))
    assert set(policy._passed_keys) == set()
    assert set(policy._current_keys) == {key(0, 0), key(1, 0), key(2, 0)}

    request = FakeRequest("r3b", hashes)
    tracked(scheduler, request)
    scheduler.request_finished_with_blocks(request, ([11, 0, 0, 14], [21, 22, 23, 24]))
    assert set(policy._passed_keys) == {key(1, 0)}
    assert set(policy._current_keys) == {key(0, 0), key(2, 0), key(3, 0)}
    assert not any(k in policy._passed_keys or k in policy._current_keys for k in (key(1, 1), key(2, 1)))

    # The engine may have moved its held block past the last full boundary
    # already (a partial block carries the live state); the last full
    # boundary is still the one the next lookup asks for.
    request = FakeRequest("r3c", hashes)
    tracked(scheduler, request)
    scheduler.request_finished_with_blocks(request, ([11, 0, 0, 0, 15], [21, 22, 23, 24, 25]))
    assert set(policy._passed_keys) == {key(1, 0)}


def test_a_lookup_never_demotes_the_checkpoint_it_is_about_to_ask_for() -> None:
    """A continuation's chain runs two chunks past what was computed. The
    lookup-time classification must not read those two as the current
    boundaries and make the previous turn's newest checkpoint — the one the
    tier holds and the scan needs — passed; that demotion waits for the
    finish, which knows the new checkpoint exists."""
    pool = make_pool(128)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    scheduler._retention = policy
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    hashes = [bytes([i]) * 32 for i in (1, 2, 3, 4, 5)]

    def key(i: int, g: int) -> offload_base.OffloadKey:
        return offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.BlockHash(hashes[i]), g)

    # The previous turn finished with three full chunks: 2 is its newest checkpoint.
    previous = FakeRequest("t1", hashes[:3])
    tracked(scheduler, previous)
    scheduler.request_finished_with_blocks(previous, ([11, 0, 13], [21, 22, 23]))
    assert policy.is_current(key(2, 0)) and not policy.is_passed(key(2, 0))

    # The continuation arrives with two new chunks; its lookup classifies the chain.
    continuation = FakeRequest("t2", hashes)
    tracked(scheduler, continuation)
    scheduler._classify_checkpoints(continuation, None)
    assert policy.is_current(key(2, 0)), "the checkpoint the scan needs was demoted at lookup"
    assert not policy.is_passed(key(2, 0))
    assert policy.is_current(key(4, 0)) and policy.is_current(key(3, 0)) and policy.is_current(key(0, 0))

    # Its finish, with the new checkpoint held at 4, is where 2 becomes passed.
    scheduler.request_finished_with_blocks(continuation, ([11, 0, 0, 0, 15], [21, 22, 23, 24, 25]))
    assert policy.is_passed(key(1, 0)) and policy.is_passed(key(2, 0))
    assert set(policy._current_keys) == {key(0, 0), key(3, 0), key(4, 0)}


def test_a_local_prefix_hit_demotes_the_request_s_hosted_copies() -> None:
    """The chunks a GPU prefix hit covers are resident from here on, so
    their host copies go to the eviction end, exactly as a loaded block's
    do; the chunks past the hit are left where they are."""
    pool = make_pool(64)
    scheduler = make_scheduler(pool, num_groups=2)
    demoted: list[list[offload_base.OffloadKey]] = []
    scheduler._demote = lambda keys: demoted.append(list(keys))
    keys = {g: [offload_base.OffloadKey(bytes([g, i]) * 18) for i in range(4)] for g in range(2)}
    scheduler._req_status["r"] = typing.cast(
        typing.Any, types.SimpleNamespace(group_states=[types.SimpleNamespace(offload_keys=keys[g]) for g in range(2)])
    )

    scheduler._demote(scheduler._local_hit_keys("r", 2 * HASH_BLOCK_SIZE + 5))

    assert demoted == [keys[0][:2] + keys[1][:2]]
    assert scheduler._local_hit_keys("unknown", 64) == []


def test_the_passed_registry_forgets_unstored_keys_only() -> None:
    """A stored passed key must never drift back to the current class when
    the registry is pruned; only classifications of keys the tier does not
    hold are forgotten, oldest first."""
    policy, keys = stored_policy(8, 2)
    policy._registry_limit = 3
    unstored = [offload_base.OffloadKey(bytes([9, i]) * 18) for i in range(4)]
    policy.mark_passed([keys[0], keys[1]])
    policy.mark_passed(unstored)

    assert keys[0] in policy._passed_keys and keys[1] in policy._passed_keys
    assert list(policy.passed_blocks) == [keys[0], keys[1]]
    assert list(policy.evictable_blocks) == []
    assert [k for k in policy._passed_keys if k in unstored] == unstored[3:]


def test_the_band_neither_stores_nor_touches_a_passed_checkpoint() -> None:
    """A copy of a passed checkpoint can never become a hit, and a
    recompute re-creates every one of a context's checkpoints, so the band
    must let them die on the GPU unrecorded."""
    pool = make_pool(64)
    freed = free_blocks(pool, 4)
    scheduler = make_scheduler(pool)
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    policy.mark_passed([key_of(freed[1]), key_of(freed[2])])
    scheduler._retention = policy
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert manager.offered == [[key_of(freed[0]), key_of(freed[3])]]
    assert manager.touched == []
    assert scheduler._passed_skipped == 2


def test_a_lookup_classifies_the_whole_chain_before_the_turn_runs() -> None:
    """With no block ids yet, a lookup promotes the first and last two
    boundaries of the chain to current and demotes nothing: the chain may run
    past what was computed, and which boundaries exist is the finish's
    knowledge. The band's skip list is what earlier finishes marked."""
    pool = make_pool(64)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    scheduler._retention = policy
    hashes = [bytes([i]) * 32 for i in (1, 2, 3, 4)]

    def key(i: int, g: int) -> offload_base.OffloadKey:
        return offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.BlockHash(hashes[i]), g)

    policy.mark_passed([key(1, 0)])  # an earlier finish's verdict stands
    scheduler._classify_checkpoints(FakeRequest("r4", hashes), None)

    assert set(policy._passed_keys) == {key(1, 0)}
    assert set(policy._current_keys) == {key(0, 0), key(2, 0), key(3, 0)}
    assert scheduler._finish_patterns_logged == 0


def test_departures_are_counted_by_what_they_cost() -> None:
    """A block taken back by a prefix hit costs nothing; one evicted from
    inside the band without a host copy is a lost race; one evicted from
    beyond the band was never looked at."""
    pool = make_pool(7)  # block 0 is the null block; the queue is exactly the six freed below
    freed = free_blocks(pool, 6)
    scheduler = make_scheduler(pool, band=3)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())

    scheduler._account_departures(pool)  # snapshot
    taken = pool.get_new_blocks(4)  # the four oldest: freed[0..3], evicted
    assert [b.block_id for b in taken] == [b.block_id for b in freed[:4]]
    # give_hash marks a block cached without registering it in the pool's hash
    # map; a prefix hit needs the registration, so it is added here.
    pool.cached_block_hash_to_block.insert(freed[5].block_hash, freed[5])
    pool.touch(typing.cast(typing.Any, [freed[5]]))  # freed[5] taken back by a prefix hit
    scheduler._account_departures(pool)

    assert dict(scheduler._departures) == {
        "uncopied": 3,
        "unseen": 1,
        "reused": 1,
        "uncopied_g0": 3,
        "unseen_g0": 1,
        "reused_g0": 1,
    }


@pytest.mark.parametrize("version", ["0.29.0", "0.29.1.dev8+g902f43ad3", "0.29.0rc4"])
def test_any_build_on_the_supported_line_passes_the_version_check(version: str) -> None:
    victim_offload._check_vllm_line(version)


@pytest.mark.parametrize("version", ["0.25.1", "0.27.1", "1.0.0", "garbage"])
def test_another_line_is_refused_at_import(version: str) -> None:
    """The connector reads stock internals that move between lines; an
    engine on another line must fail at startup, not mid-run."""
    with pytest.raises(ImportError):
        victim_offload._check_vllm_line(version)


def test_a_passed_key_marked_current_rejoins_the_main_order_at_the_recent_end() -> None:
    """A boundary is passed at one finish and current at the next (a
    compaction's head, say); it must leave the passed class, newest."""
    policy, keys = stored_policy(8, 3)
    policy.mark_passed([keys[0]])
    policy.mark_current([keys[0]])

    assert list(policy.passed_blocks) == []
    assert list(policy.evictable_blocks) == [keys[1], keys[2], keys[0]]
    assert policy.is_current(keys[0]) and not policy.is_passed(keys[0])


def test_a_current_key_marked_passed_is_the_first_to_go() -> None:
    """Last turn's latest boundary is passed at this turn's finish."""
    policy, keys = stored_policy(8, 3)
    policy.mark_current([keys[2]])
    policy.mark_passed([keys[2]])

    evicted = policy.evict(1, protected=set())
    assert evicted is not None and [k for k, _ in evicted] == [keys[2]]
    assert (policy.evicted_passed, policy.evicted_current) == (1, 0)


def test_depth_counts_the_keys_that_would_survive_the_asked_one() -> None:
    """Depth is the capacity a tier needs for the key to survive: a main
    key needs only the main keys newer than it; a passed key outlives
    nothing in the main order."""
    policy, keys = stored_policy(8, 4)
    policy.mark_passed([keys[1]])

    policy.mark_non_evictable(keys[0])  # oldest main, two main keys newer
    policy.mark_non_evictable(keys[1])  # passed: behind the two remaining main keys

    assert policy.drain_hit_depths() == [2, 2]


def boundary_keys(seed: int) -> tuple[offload_base.OffloadKey, offload_base.OffloadKey]:
    """The checkpoint (group 0) and attention (group 3) keys of one boundary."""
    block_hash = bytes([seed]) * 32
    return offload_base.make_offload_key(block_hash, 0), offload_base.make_offload_key(block_hash, 3)


def test_a_use_of_a_boundary_s_attention_chunk_is_a_use_of_its_checkpoint() -> None:
    """The engine frees a boundary's checkpoint a turn before its attention,
    so key by key the checkpoint is always the older; the two carry the same
    hash, and a touch of the attention must age the checkpoint with it, or
    LRU would take the checkpoint first and leave the attention useless."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    checkpoint, attention = boundary_keys(5)
    _, other = boundary_keys(6)
    for i, key in enumerate((checkpoint, other, attention)):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)
    policy.touch([other], req_context=None)

    policy.touch([attention], req_context=None)  # the context's blocks pass the band

    evicted = policy.evict(3, protected=set())
    assert evicted is not None
    # `other` is now the oldest; the checkpoint is as young as its twin and
    # goes right after it, not before.
    assert [k for k, _ in evicted] == [other, attention, checkpoint]


def test_a_context_is_evicted_whole_and_oldest_first() -> None:
    """Checkpoints are stored before attention (the engine frees them
    first); with a boundary aging as one, the older context goes in full
    before any key of the younger, attention before its checkpoint."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    checkpoint_a, attention_a = boundary_keys(1)
    checkpoint_b, attention_b = boundary_keys(2)
    for i, key in enumerate((checkpoint_a, checkpoint_b, attention_a, attention_b)):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)

    evicted = policy.evict(2, protected=set())

    assert evicted is not None and [k for k, _ in evicted] == [attention_a, checkpoint_a]
    assert list(policy.evictable_blocks) == [attention_b, checkpoint_b]


def test_a_passed_sibling_is_not_refreshed_by_its_attention() -> None:
    """A checkpoint the context has moved past stays first to go however
    often the attention at its boundary is used."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    checkpoint, attention = boundary_keys(7)
    _, other = boundary_keys(8)
    for i, key in enumerate((checkpoint, other, attention)):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)
    policy.mark_passed([checkpoint])
    policy.touch([attention], req_context=None)

    evicted = policy.evict(1, protected=set())
    assert evicted is not None and [k for k, _ in evicted] == [checkpoint]


def test_a_demoted_sibling_no_thread_needs_stays_first_to_go() -> None:
    """Under a sibling filter (park mode), a use of a boundary's attention
    refreshes the checkpoint a thread needs and leaves a dead one demoted."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    dead, attention_a = boundary_keys(1)
    live, attention_b = boundary_keys(2)
    for i, key in enumerate((dead, live, attention_a, attention_b)):
        block = policies_base.BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(key, block)
    policy.sibling_filter = lambda key: key == live
    policy.demote([dead, live])
    policy.touch([attention_a, attention_b], req_context=None)

    evicted = policy.evict(1, protected=set())
    assert evicted is not None and [k for k, _ in evicted] == [dead]
    assert list(policy.evictable_blocks) == [attention_b, live, attention_a]


def test_a_finish_stores_the_request_s_shared_blocks_the_host_lacks() -> None:
    """A block shared with other requests never reaches the free queue while
    a sharer runs, so the band cannot copy it, and a lookup that starts at
    it fails there. It is stored at finish, once; a block only this request
    holds is left to the band, and one the host has is not offered again."""
    pool = make_pool(128)  # headroom: no hold, only the shared store
    scheduler = make_scheduler(pool, watermark=0.15)
    shared, own, hosted = pool.get_new_blocks(3)
    for seed, block in enumerate((shared, own, hosted), start=1):
        give_hash(block, seed)
    pool.touch([shared, hosted])  # a second sharer of both
    manager = RecordingManager(present={key_of(hosted)})
    scheduler.manager = typing.cast(typing.Any, manager)
    hashes = [
        vllm.v1.core.kv_cache_utils.get_block_hash(typing.cast(typing.Any, b.block_hash)) for b in (shared, own, hosted)
    ]
    request = FakeRequest("r5", [bytes(h) for h in hashes])
    tracked(scheduler, request)

    held, _ = scheduler.request_finished_with_blocks(request, ([shared.block_id, own.block_id, hosted.block_id],))

    assert held is False
    # The recording manager has no policy to pre-filter with, so the hosted
    # key is offered and declined; the one this request alone holds is not.
    assert manager.offered == [[key_of(shared), key_of(hosted)]]
    ((job_id, job),) = scheduler._pending_finish_jobs.items()
    assert list(typing.cast(typing.Any, job.src_spec).block_ids) == [shared.block_id]
    assert job_id in scheduler._victim_job_ids and scheduler._shared_stored == 1
    # Emitted with the next step's store jobs, exactly once.
    assert scheduler._build_store_jobs(typing.cast(typing.Any, None)) == {job_id: job}


def test_a_finish_stores_the_last_two_checkpoints_and_re_ages_them_on_the_gpu() -> None:
    """The engine frees a context's checkpoints a turn before its attention,
    so a pool that evicts block by block takes them first and leaves an
    attention prefix with no state — a hit the connector cannot serve. At
    finish the last two checkpoints are stored regardless of the band, and
    at the next step their GPU blocks move to the tail of the free queue,
    behind the attention the engine has freed since."""
    pool = make_pool(8)  # block 0 is the null block
    scheduler = make_scheduler(pool, watermark=0.15, single_state_groups=(0,))
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    older, ckpt_a, ckpt_b, live = pool.get_new_blocks(4)
    # The request's chain, in give_hash's encoding so the blocks match it.
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    for block, seed in ((ckpt_a, 2), (ckpt_b, 3)):
        give_hash(block, seed)
        pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, block.block_hash), block)
    pool.free_blocks([ckpt_a, ckpt_b])  # freed mid-turn, as the engine does
    give_hash(older, 9)  # hashed, so the pool queues it at the tail like any cached block
    pool.free_blocks([older])  # attention freed after: the checkpoints are older
    request = FakeRequest("r6", hashes)
    tracked(scheduler, request)

    scheduler.request_finished_with_blocks(request, ([0, 0, 0, live.block_id],))

    (job,) = scheduler._pending_finish_jobs.values()
    assert list(typing.cast(typing.Any, job.src_spec).block_ids) == [ckpt_a.block_id, ckpt_b.block_id]
    assert scheduler._checkpoints_stored == 2
    # The next step re-ages: the checkpoints now sit at the tail, after `older`.
    scheduler._build_store_jobs(typing.cast(typing.Any, None))
    tail = typing.cast(typing.Any, pool.free_block_queue.fake_free_list_tail.prev_free_block)
    assert tail is ckpt_b and tail.prev_free_block is ckpt_a and ckpt_a.prev_free_block is older
    assert scheduler._reaged == 2 and scheduler._reage_pending == []


def test_a_zero_lookup_is_classified_and_a_divergent_hit_is_named() -> None:
    """A zero says nothing by itself. Nothing askable and a miss at the
    chain's last full chunk are benign; a walk that failed earlier is
    counted with its start; and a start past chunk 0 whose recurrent state
    at the boundary is not on the GPU is the divergent hit — with a second
    count when the tier holds that state and was never asked for it."""
    pool = make_pool(64)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    manager = RecordingManager()
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3, 4)]
    request = FakeRequest("r7", hashes)
    c = scheduler._lookup_audit.counts

    scheduler._classify_zero_lookup(request, 4 * HASH_BLOCK_SIZE)  # the GPU has every full chunk
    scheduler._classify_zero_lookup(request, 3 * HASH_BLOCK_SIZE)  # only the straddling chunk left
    scheduler._classify_zero_lookup(request, 0)  # a walk from the head that found nothing
    assert (c["lookup_zero_nothing_askable"], c["lookup_zero_at_straddle"], c["lookup_zero_walk_failed"]) == (1, 1, 1)
    # Both starts past chunk 0 had no recurrent state at their boundary on
    # the GPU: those are the divergent hit itself (Kaggle's case), the
    # "nothing askable" one above all. The walk from chunk 0 is not.
    assert c["lookup_walk_failed_start_0"] == 1 and c["lookup_divergent"] == 2
    assert (c["lookup_divergent_nothing_askable"], c["lookup_divergent_at_straddle"]) == (1, 1)
    assert c["lookup_divergent_tier_has_ckpt"] == 0

    # A walk from chunk 2: the attention prefix is local, but the recurrent
    # state at boundary 2 (index 1) is not in the GPU's cache — and it is
    # in the tier.
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    block = policies_base.BlockStatus(0)
    block.ref_cnt = 0
    policy.insert(offload_base.make_offload_key(hashes[1], 0), block)
    scheduler._classify_zero_lookup(request, 2 * HASH_BLOCK_SIZE)
    assert c["lookup_walk_failed_start_2-4"] == 1
    assert (c["lookup_divergent"], c["lookup_divergent_tier_has_ckpt"]) == (3, 1)

    # With that state cached on the GPU the hit is not divergent.
    gpu_block = pool.get_new_blocks(1)[0]
    give_hash(gpu_block, 2, group=0)
    pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, gpu_block.block_hash), gpu_block)
    scheduler._classify_zero_lookup(request, 2 * HASH_BLOCK_SIZE)
    assert c["lookup_divergent"] == 3


def test_a_current_checkpoint_leaving_the_gpu_soon_after_its_re_age_is_counted() -> None:
    """The two departures that would say the re-aging is not doing its job."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    scheduler._retention = policy
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    block = pool.get_new_blocks(1)[0]
    give_hash(block, 2, group=0)
    pool.free_blocks([block])
    policy.mark_current([key_of(block)])

    scheduler._account_departures(pool)  # snapshot: the block is free and current
    scheduler._reaged_at[block.block_id] = scheduler._step  # re-aged this step
    # A freed hashed block queues at the tail; the pool reaches it once every
    # never-used block ahead of it has been handed out.
    taken = pool.get_new_blocks(pool.get_num_free_blocks())
    assert taken[-1] is block
    scheduler._account_departures(pool)

    assert scheduler._departures["current_ckpt"] == 1
    assert scheduler._departures["reaged_recently"] == 1


def test_a_known_request_keeps_its_newest_checkpoint_young_every_step() -> None:
    """The checkpoint a continuation scans to is freed at the turn's first
    step and needed at its last. Kept at the tail every step it cannot age
    out under the turn; the probe walks back when the newest boundary has
    no cached block; and a request with none on the GPU is counted, since
    that is the continuation that will recompute its prompt."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, single_state_groups=(0,))
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    ckpt_1, ckpt_2, later = pool.get_new_blocks(3)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    for block, seed in ((ckpt_1, 2), (ckpt_2, 3)):
        give_hash(block, seed)
        pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, block.block_hash), block)
    pool.free_blocks([ckpt_1, ckpt_2])  # freed mid-turn, as the engine does
    give_hash(later, 9)
    pool.free_blocks([later])  # freed after: the checkpoints are the older blocks
    tracked(scheduler, FakeRequest("r8", hashes))
    queue = pool.free_block_queue

    scheduler._keep_checkpoints_young(pool)
    tail = typing.cast(typing.Any, queue.fake_free_list_tail.prev_free_block)
    # The newest only: the one before it stays where it was.
    assert tail is ckpt_2 and tail.prev_free_block is later and later.prev_free_block is ckpt_1
    assert (scheduler._kept_last_step, scheduler._kept_missing_last_step) == (1, 0)

    # The pool takes the newest: the probe walks back a chunk.
    queue.remove(ckpt_2)
    pool._maybe_evict_cached_block(ckpt_2)
    scheduler._keep_checkpoints_young(pool)
    assert queue.fake_free_list_tail.prev_free_block is ckpt_1
    assert (scheduler._kept_last_step, scheduler._kept_missing_last_step) == (1, 0)

    queue.remove(ckpt_1)
    pool._maybe_evict_cached_block(ckpt_1)
    scheduler._keep_checkpoints_young(pool)
    assert (scheduler._kept_last_step, scheduler._kept_missing_last_step) == (0, 1)


def test_a_finish_that_finds_no_checkpoint_on_the_gpu_is_counted() -> None:
    """A finishing request whose checkpoints have all been evicted is a
    continuation that will recompute its prompt: counted, and no job."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, single_state_groups=(0,))
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    live = pool.get_new_blocks(1)[0]
    request = FakeRequest("r9", [seed.to_bytes(32, "big") for seed in (1, 2, 3)])
    tracked(scheduler, request)

    scheduler.request_finished_with_blocks(request, ([0, 0, 0, live.block_id],))

    assert scheduler._checkpoints_gone == 1 and scheduler._pending_finish_jobs == {}


def _pressured_scheduler(num_blocks: int = 8, **kwargs: typing.Any) -> tuple[typing.Any, typing.Any, list[typing.Any]]:
    """A pool exhausted below any watermark, its blocks in hand, and a scheduler on it."""
    pool = make_pool(num_blocks)
    free_blocks(pool, num_blocks - 1)
    blocks = pool.get_new_blocks(num_blocks - 1)
    scheduler = make_scheduler(pool, watermark=0.15, **kwargs)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    return pool, scheduler, blocks


def test_the_hold_stores_only_chunks_every_token_of_which_was_computed() -> None:
    """A response ending exactly on a block boundary has a hash for a chunk
    whose last token was sampled and never went through the model; the hold
    offers the chunks before it and not that one. A request aborted
    mid-prefill offers nothing."""
    _, scheduler, blocks = _pressured_scheduler()
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2)]
    first, last = blocks[:2]
    give_hash(first, 1)
    give_hash(last, 2)  # hashed as if complete: the chain says so, the model did not
    request = FakeRequest("rb", hashes, num_computed_tokens=2 * HASH_BLOCK_SIZE - 1)
    tracked(scheduler, request)

    held, _ = scheduler.request_finished_with_blocks(request, ([first.block_id, last.block_id],))

    assert held is True
    (job,) = scheduler._pending_finish_jobs.values()
    assert list(job.src_spec.block_ids) == [first.block_id]

    aborted = FakeRequest("ra", hashes, num_computed_tokens=0)
    tracked(scheduler, aborted)
    scheduler._pending_finish_jobs.clear()
    held, _ = scheduler.request_finished_with_blocks(aborted, ([first.block_id, last.block_id],))
    assert held is False and scheduler._pending_finish_jobs == {}


def test_the_hold_checks_each_source_against_the_hash_the_gpu_holds() -> None:
    """A block that does not carry the chain's hash under this group is not
    the content the key would name — unhashed because the engine never
    completed it, or hashed for something else — and is passed over."""
    _, scheduler, blocks = _pressured_scheduler()
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    good, unhashed, other = blocks[:3]
    give_hash(good, 1)
    unhashed._block_hash = None
    give_hash(other, 99)
    request = FakeRequest("rh", hashes)
    tracked(scheduler, request)

    held, _ = scheduler.request_finished_with_blocks(request, ([good.block_id, unhashed.block_id, other.block_id],))

    assert held is True
    (job,) = scheduler._pending_finish_jobs.values()
    assert list(job.src_spec.block_ids) == [good.block_id]
    assert scheduler._lookup_audit.counts["hold_unverified"] == 2


def test_the_fill_refuses_a_windowed_group_wider_than_one_chunk() -> None:
    """The fill loads one block per group, which is a whole state only for a
    single-state group; a sliding-window attention group whose window spans
    several chunks is refused rather than served one block of its window."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    manager = RecordingManager()
    manager._policy = victim_offload.VictimLRUPolicy(cache_capacity=8)  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    request = FakeRequest("rw", [seed.to_bytes(32, "big") for seed in (1, 2, 3, 4)])
    tracked(scheduler, request)
    scheduler.config.kv_group_configs[0].sliding_window_size_in_chunks = 4  # type: ignore[attr-defined]

    assert scheduler.fill_divergent_hit(request, 4 * HASH_BLOCK_SIZE, (0,)) is False
    assert scheduler._lookup_audit.counts["fill_refused_group"] == 1 and not scheduler._fills


def test_the_connector_refuses_to_start_on_a_group_shape_it_has_not_run() -> None:
    """A sliding-window attention group wider than a chunk, or an eagle/MTP
    draft group, fails at construction with the group named; full attention
    and single-state groups pass."""

    def cfg(g: int = 0, **kw: typing.Any) -> types.SimpleNamespace:
        fields: dict[str, typing.Any] = {"group_idx": g, "is_eagle_group": False}
        fields.update(kw)
        return types.SimpleNamespace(**fields)

    victim_offload._refuse_unsupported_groups(
        [cfg(sliding_window_size_in_chunks=None), cfg(g=1, sliding_window_size_in_chunks=1)]
    )
    with pytest.raises(ValueError, match="group 1 is a sliding-window group of 4 chunks"):
        victim_offload._refuse_unsupported_groups(
            [cfg(sliding_window_size_in_chunks=None), cfg(g=1, sliding_window_size_in_chunks=4)]
        )
    with pytest.raises(ValueError, match="group 2 is an eagle/MTP draft group"):
        victim_offload._refuse_unsupported_groups([cfg(g=2, sliding_window_size_in_chunks=None, is_eagle_group=True)])


def test_a_cache_reset_clears_the_victim_bookkeeping_and_ignores_stale_completions() -> None:
    """Stock's reset discards every job and raises the stale-job threshold;
    the victim path's spills, job ids and finish-holds go with them, and a
    worker completion for a job from before the reset is dropped rather
    than looked up."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._events_tracker = typing.cast(
        typing.Any, types.SimpleNamespace(reset=lambda: None)
    )  # stock's reset clears it
    scheduler._job_counter = 8
    scheduler._jobs[7] = offload_sched.TransferJobStatus(
        req_id=victim_offload.VICTIM_REQ_ID, pending_count=1, keys=set(), is_store=True
    )
    scheduler._victim_job_ids = {7}
    scheduler._spilled[7] = []
    scheduler._pending_finish_jobs[7] = typing.cast(typing.Any, object())

    scheduler.reset_cache()

    assert scheduler._jobs == {} and scheduler._victim_job_ids == set()
    assert scheduler._spilled == {} and scheduler._pending_finish_jobs == {}
    assert scheduler._stale_job_threshold == 8
    scheduler._victim_job_ids = {7}  # even a stray id from before the reset
    scheduler.update_connector_output(worker_completed(7))  # no KeyError
    assert scheduler._victim_job_ids == set()


def test_a_cache_reset_waits_for_a_held_store_to_reach_the_worker() -> None:
    """A finishing request held for its store has its blocks kept by the
    engine until the worker reports the request sent, which the worker can
    do only for a job it was handed. A reset before the hold's first
    dispatch is refused, naming the request; once the next metadata build
    has carried the hold out, the reset goes through."""
    _, scheduler, blocks = _pressured_scheduler()
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2)]
    first, second = blocks[:2]
    give_hash(first, 1)
    give_hash(second, 2)
    request = FakeRequest("held-req", hashes)
    status = tracked(scheduler, request)
    status.finished_signaled = True  # the stock reset asks; the finish path here does not set it
    held, _ = scheduler.request_finished_with_blocks(request, ([first.block_id, second.block_id],))
    assert held is True and scheduler._pending_finish_jobs
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._events_tracker = typing.cast(typing.Any, types.SimpleNamespace(reset=lambda: None))
    scheduler._job_counter = 8

    with pytest.raises(RuntimeError, match="held-req"):
        scheduler.reset_cache()
    assert scheduler._pending_finish_jobs  # nothing was discarded

    scheduler._spill_band_blocks = 0
    dispatched = scheduler._build_store_jobs(typing.cast(typing.Any, None))
    assert len(dispatched) == 1 and scheduler._pending_finish_jobs == {}
    scheduler.reset_cache()
    assert scheduler._jobs == {} and scheduler._victim_job_ids == set()


def read_trace(path: pathlib.Path) -> list[dict[str, typing.Any]]:
    """The events a trace file holds, in order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_the_trace_writes_events_and_a_bad_path_turns_it_off(tmp_path: pathlib.Path) -> None:
    """A trace is one JSON object per event with its time and kind; a path
    that cannot be opened leaves it off, and writing to it is a no-op."""
    path = tmp_path / "t.jsonl"
    trace = victim_offload._Trace(str(path))
    trace.write("start", pool=8)
    trace.write("evict", g=1, kind="hosted")
    trace.close()
    events = read_trace(path)
    assert [e["ev"] for e in events] == ["start", "evict"]
    assert events[0]["pool"] == 8 and events[1]["kind"] == "hosted" and events[1]["t"] > 0

    off = victim_offload._Trace(str(tmp_path / "missing" / "t.jsonl"))
    assert not off.enabled and off.disabled_reason
    off.write("start")  # dropped, not raised


def test_the_tally_is_a_trace_event_and_the_engine_log_is_silent_by_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a minute the counters go to the trace as one ``tally`` event —
    the spill tally, the departures, every lookup-audit counter, the keys
    sampled since the last tally, the asked-depth percentiles — whether or
    not the engine log gets the same as lines; by default it does not."""
    lines: list[str] = []

    def info(fmt: str, *args: typing.Any) -> None:
        lines.append(fmt % args)

    def warning(fmt: str, *args: typing.Any) -> None:
        pass

    monkeypatch.setattr(victim_offload, "logger", types.SimpleNamespace(info=info, warning=warning))
    pool = make_pool(8)
    scheduler = make_scheduler(pool)
    scheduler._trace = victim_offload._Trace(str(tmp_path / "t.jsonl"))
    scheduler._blocks_spilled = 3
    scheduler._departures["hosted"] = 2
    scheduler._lookup_audit.counts["hit"] = 4
    scheduler._lookup_audit.sample_asked[1] = ["k1"]
    scheduler._drain_hit_depths = lambda: [9, 1, 5]

    scheduler._maybe_log_tally()
    assert lines == []
    scheduler._maybe_log_tally()  # inside the interval: nothing more
    scheduler._last_tally_at = 0.0
    scheduler._engine_log = True
    scheduler._maybe_log_tally()
    assert any(line.startswith("victim offload: spilled=3 ") for line in lines)
    assert any(line.startswith("victim offload: asked depth blocks") for line in lines)
    scheduler._trace.close()

    tallies = [e for e in read_trace(tmp_path / "t.jsonl") if e["ev"] == "tally"]
    assert len(tallies) == 2
    first, second = tallies
    assert first["spill"]["spilled"] == 3 and first["departed"] == {"hosted": 2}
    assert first["lookups"]["hit"] == 4
    assert first["samples"] == {"1": {"asked": ["k1"], "stored": []}}
    assert first["depth"] == {
        "n": 3,
        "p50": 5,
        "p90": 9,
        "p99": 9,
        "max": 9,
        "run_max": 9,
        "evictable": 0,
        "passed": 0,
        "evicted_passed": 0,
        "evicted_current": 0,
    }
    # Samples are per interval; the depth line keeps the run's maximum.
    assert second["samples"] == {} and second["depth"]["run_max"] == 9


def test_the_first_finish_patterns_are_traced(tmp_path: pathlib.Path) -> None:
    """The engine's per-group block pattern at a finish goes to the trace
    for the first few finishes, whether or not the engine log is on."""
    pool = make_pool(128)  # headroom: no hold, the pattern alone
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    scheduler._retention = victim_offload.VictimLRUPolicy(cache_capacity=8)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    scheduler._trace = victim_offload._Trace(str(tmp_path / "t.jsonl"))
    request = FakeRequest("r1", [seed.to_bytes(32, "big") for seed in (1, 2, 3)])
    tracked(scheduler, request)

    scheduler.request_finished_with_blocks(request, ([0, 0, 5], [21, 22, 23]))
    scheduler._trace.close()

    patterns = [e for e in read_trace(tmp_path / "t.jsonl") if e["ev"] == "pattern"]
    # One per recurrent group: the attention group carries no checkpoints.
    assert len(patterns) == 1
    assert {k: patterns[0][k] for k in ("req", "g", "full_blocks", "block_ids", "nonzero_at")} == {
        "req": "r1",
        "g": 0,
        "full_blocks": 3,
        "block_ids": 3,
        "nonzero_at": [2],
    }
    assert scheduler._finish_patterns_logged == 1


def test_a_lookup_joins_its_schedule_and_the_turn_before_it(tmp_path: pathlib.Path) -> None:
    """One trace line per request arrival: what the scheduler reported as
    the attention hit, each recurrent group's newest boundary still on the
    GPU below it, the tier's answer, the hit the scheduler settled on once
    the request was scheduled — and, through the chain, the turn of the
    same context that finished before it."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    scheduler._trace = victim_offload._Trace(str(tmp_path / "t.jsonl"))
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    ckpt = pool.get_new_blocks(1)[0]
    give_hash(ckpt, 2, group=0)  # boundary 2's state (chunk index 1), cached
    pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, ckpt.block_hash), ckpt)
    pool.free_blocks([ckpt])

    previous = FakeRequest("turn-1", hashes)
    previous.num_tokens = 3 * HASH_BLOCK_SIZE + 5  # type: ignore[attr-defined]
    previous.arrival_time = 0.0  # type: ignore[attr-defined]
    tracked(scheduler, previous)
    scheduler.request_finished_with_blocks(previous, ([0, 0, 0], [0, 0, 0]))

    continuation = FakeRequest("turn-2", hashes + [(4).to_bytes(32, "big")])
    continuation.num_tokens = 4 * HASH_BLOCK_SIZE + 7  # type: ignore[attr-defined]
    # The GPU reports three attention chunks; the recurrent state at that
    # boundary is gone, boundary 2's is the newest left.
    scheduler._build_store_jobs(typing.cast(typing.Any, None))  # a step: the queue is walked, the trace starts
    scheduler._trace_lookup(continuation, 3 * HASH_BLOCK_SIZE, 0, {"kind": "at_straddle", "divergent": True})
    assert "turn-2" in scheduler._trace_lookups
    scheduler._trace_schedule(
        typing.cast(
            typing.Any,
            types.SimpleNamespace(
                scheduled_new_reqs=[types.SimpleNamespace(req_id="turn-2", num_computed_tokens=2 * HASH_BLOCK_SIZE)],
                scheduled_cached_reqs=types.SimpleNamespace(req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]),
                preempted_req_ids=None,
                finished_req_ids=set(),
                total_num_scheduled_tokens=0,
            ),
        )
    )
    scheduler._trace.close()

    by_kind: dict[str, list[dict[str, typing.Any]]] = collections.defaultdict(list)
    for event in read_trace(tmp_path / "t.jsonl"):
        by_kind[event["ev"]].append(event)
    # The finish stored the checkpoint it found (its own event); the step
    # after it spilled the same block from the band, since the recording
    # manager holds nothing.
    assert set(by_kind) == {"start", "store", "finish", "lookup", "step"}
    assert [s["origin"] for s in by_kind["store"]] == ["finish", "band"]
    finish, lookup, step = by_kind["finish"][0], by_kind["lookup"][0], by_kind["step"][0]
    assert finish["req"] == "turn-1" and finish["chunks"] == 3
    assert finish["turn_s"] > 0 and finish["ck_g0"] == [1]
    assert lookup["req"] == "turn-2"
    assert (lookup["att_hit"], lookup["rec_g0"], lookup["ext"], lookup["kind"]) == (3, 2, 0, "at_straddle")
    assert lookup["divergent"] is True
    assert (lookup["computed"], lookup["recomputed"]) == (2 * HASH_BLOCK_SIZE, 2 * HASH_BLOCK_SIZE + 7)
    assert lookup["prev_req"] == "turn-1" and lookup["new_chunks"] == 1 and lookup["gap_s"] >= 0
    assert step["free"] == pool.get_num_free_blocks() and step["q"]["unk"] == 1


def test_an_evicted_kept_checkpoint_is_traced_with_its_request(tmp_path: pathlib.Path) -> None:
    """A departure of a block the keep-young pass moved the step before
    names the request and chunk it was kept for, and how long it sat."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, single_state_groups=(0,))
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    scheduler._trace = victim_offload._Trace(str(tmp_path / "t.jsonl"))
    block = pool.get_new_blocks(1)[0]
    give_hash(block, 2, group=0)
    pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, block.block_hash), block)
    pool.free_blocks([block])
    tracked(scheduler, FakeRequest("r", [seed.to_bytes(32, "big") for seed in (1, 2)]))

    scheduler._keep_checkpoints_young(pool)
    scheduler._account_departures(pool)  # snapshot with the block queued
    scheduler._keep_checkpoints_young(pool)  # this step's map; the previous one names it too
    taken = pool.get_new_blocks(pool.get_num_free_blocks())
    assert taken[-1] is block
    scheduler._account_departures(pool)
    scheduler._trace.close()

    (evict,) = read_trace(tmp_path / "t.jsonl")
    assert evict["ev"] == "evict" and (evict["req"], evict["chunk"], evict["g"]) == ("r", 1, 0)
    assert evict["kind"] == "uncopied" and evict["age"] >= 0  # inside the band, no host copy


def test_the_job_dir_reaches_the_connector_config() -> None:
    """The factory hands the run's job directory to the connector through
    the engine config, which is all the engine subprocess sees."""
    import ullm.backends.config_local as config_local
    import vllm

    engine_args = vllm.AsyncEngineArgs(model="x")
    config_local.attach_job_dir(engine_args, pathlib.Path("/run/job"))  # no connector: nothing to tell
    assert engine_args.kv_transfer_config is None
    config_local.enable_cpu_kv_offload(engine_args, cpu_gib=1.0, victim=True)
    config_local.attach_job_dir(engine_args, pathlib.Path("/run/job"))
    assert engine_args.kv_transfer_config is not None
    assert engine_args.kv_transfer_config.kv_connector_extra_config["job_dir"] == "/run/job"


def test_a_divergent_hit_is_filled_from_the_tier() -> None:
    """The fork asks whether the tier holds the lagging group's checkpoint
    at the attention boundary; yes when it is there and ready, and the
    allocation that follows becomes a load into the boundary block the
    engine allocated fresh — nothing for the attention group."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    pool = make_pool(8)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    manager = RecordingManager()
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    request = FakeRequest("r10", hashes)
    status = tracked(scheduler, request)
    c = scheduler._lookup_audit.counts

    # Nothing in the tier: refused. Then the checkpoint at boundary 2 (chunk
    # index 1) is there but its copy is still landing: refused. Ready: yes.
    assert scheduler.fill_divergent_hit(request, 2 * HASH_BLOCK_SIZE, (0,)) is False
    block = policies_base.BlockStatus(0)
    block.ref_cnt = -1  # a store in flight
    key = offload_base.make_offload_key(hashes[1], 0)
    policy.insert(key, block)
    assert scheduler.fill_divergent_hit(request, 2 * HASH_BLOCK_SIZE, (0,)) is False
    block.ref_cnt = 0
    assert scheduler.fill_divergent_hit(request, 2 * HASH_BLOCK_SIZE, (0,)) is True
    assert (c["fill_refused_absent"], c["fill_refused_pending"], c["fill_accepted"]) == (1, 1, 1)
    # The attention group is never fillable.
    assert scheduler.fill_divergent_hit(request, 2 * HASH_BLOCK_SIZE, (1,)) is False

    # The engine's allocation: the lagging group has a null for the skipped
    # chunk and a fresh block at the boundary; the attention group its two
    # hashed hit blocks.
    null = pool.null_block
    fresh, att_a, att_b = pool.get_new_blocks(3)
    give_hash(att_a, 1, group=1)
    give_hash(att_b, 2, group=1)
    scheduler.update_state_after_alloc(request, types.SimpleNamespace(blocks=([null, fresh], [att_a, att_b])), 0)

    assert manager.loaded == [key]
    (job,) = scheduler._current_batch_load_jobs.values()
    dst = typing.cast(typing.Any, job.dst_spec)
    assert (list(dst.block_ids), list(dst.group_sizes), list(dst.block_indices)) == ([fresh.block_id], [1, 0], [1, 0])
    assert job.req_id == "r10" and status.transfer_jobs and not scheduler._fills
    assert scheduler._fills_loaded == 1 and all(not j.is_store for j in scheduler._jobs.values())


def test_a_stale_fill_does_not_load_into_a_hashed_block() -> None:
    """A fill accepted at one lookup stands only for the allocation that
    follows it. When the scheduler allocated by another path, the group's
    last block is a hashed hit block, not a fresh one: nothing is loaded,
    and the mismatch is counted rather than asserted."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    pool = make_pool(8)
    scheduler = make_scheduler(pool, num_groups=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    manager = RecordingManager()
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2, 3)]
    request = FakeRequest("r11", hashes)
    tracked(scheduler, request)
    block = policies_base.BlockStatus(0)
    block.ref_cnt = 0
    policy.insert(offload_base.make_offload_key(hashes[1], 0), block)
    assert scheduler.fill_divergent_hit(request, 2 * HASH_BLOCK_SIZE, (0,)) is True

    hit = pool.get_new_blocks(1)[0]
    give_hash(hit, 2, group=0)  # the group's own hit block, hashed: no fill was allocated
    scheduler.update_state_after_alloc(request, types.SimpleNamespace(blocks=([pool.null_block, hit], [])), 0)

    assert not hasattr(manager, "loaded") and scheduler._current_batch_load_jobs == {}
    assert scheduler._lookup_audit.counts["fill_shape_mismatch"] == 1 and not scheduler._fills


def test_a_kept_checkpoint_the_tier_lacks_is_stored_ahead_of_the_band() -> None:
    """The block keep-young moves each step is the one the fill loads. It is
    offered to the tier that same step, before the band, and not touched
    when the tier already has it."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    pool = make_pool(8)
    scheduler = make_scheduler(pool, band=2, single_state_groups=(0,))
    policy = victim_offload.VictimLRUPolicy(cache_capacity=8)
    manager = RecordingManager()
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    # Every block allocated, so the queue holds only what is freed below.
    ckpt, older_a, older_b, *_held = pool.get_new_blocks(7)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2)]
    give_hash(ckpt, 2, group=0)
    pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, ckpt.block_hash), ckpt)
    pool.free_blocks([ckpt])  # freed at the crossing, as the engine does
    for block, seed in ((older_a, 7), (older_b, 8)):
        give_hash(block, seed)
    pool.free_blocks([older_a, older_b])  # freed after; once ckpt is re-aged they are the front
    tracked(scheduler, FakeRequest("r12", hashes))

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    # Offered first, ahead of the band's two, in one job.
    assert manager.offered[0] == [key_of(ckpt), key_of(older_a), key_of(older_b)]
    assert scheduler._kept_stored == 1 and manager.touched == []

    # Already in the tier: neither stored again nor touched.
    status = policies_base.BlockStatus(0)
    status.ref_cnt = 0
    policy.insert(key_of(ckpt), status)
    manager.offered.clear()
    scheduler._build_store_jobs(typing.cast(typing.Any, None))
    assert all(key_of(ckpt) not in keys for keys in manager.offered) and manager.touched == []
    assert scheduler._kept_stored == 1


def test_the_connector_says_it_can_complete_a_divergent_hit() -> None:
    """The scheduler asks a connector to fill a divergent hit only when the
    connector declares it can; this one fills through fill_divergent_hit,
    so an undeclared connector would never be asked."""
    connector = object.__new__(victim_offload.VictimOffloadingConnector)
    assert connector.supports_divergent_local_hybrid_hits is True


def test_the_audit_counts_the_connector_and_the_engine_stores_apart() -> None:
    """An offer that says it is the connector's is counted as such; one that
    arrives without saying is the engine's own boundary-state store."""
    keys = [offload_base.make_offload_key(bytes([i]) * 32, 0) for i in range(3)]

    def prepare_store(keys: typing.Any, ctx: typing.Any) -> typing.Any:
        return types.SimpleNamespace(keys_to_store=list(keys), evicted_keys=[])

    manager = types.SimpleNamespace(prepare_store=prepare_store)
    audit = victim_offload._LookupAudit(manager, lambda: None)
    audit.prepare_store(keys[:2], None)
    audit.prepare_store(keys[2:], None, source="connector")
    assert audit.counts["store_new_engine"] == 2
    assert audit.counts["store_new_connector"] == 1
    assert audit.counts["store_new"] == 3


def test_only_a_finish_store_the_engine_holds_blocks_for_is_named_held() -> None:
    """A finishing request's shared blocks and checkpoints are stored under the
    victim sentinel, which names no request the engine has heard of. Naming one
    of those held would report it back as sent, and the engine asserts on a
    request id it does not have -- so the held set is the finish-stores whose
    blocks the engine is actually holding, and nothing else."""
    scheduler = make_scheduler(None, single_state_groups=(0,))
    spec = offload_base.GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    held_job, sentinel_job = 11, 12
    scheduler._pending_finish_jobs = {
        held_job: offload_common.TransferJob(req_id="r10", src_spec=spec, dst_spec=spec),
        sentinel_job: offload_common.TransferJob(req_id=victim_offload.VICTIM_REQ_ID, src_spec=spec, dst_spec=spec),
    }
    # Only the first was recorded as one the engine holds blocks for.
    scheduler._held_finish_jobs = {held_job}

    dispatched = scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert set(dispatched) == {held_job, sentinel_job}
    assert scheduler._held_this_step == frozenset({held_job})
    assert scheduler._held_finish_jobs == set()


def test_the_engine_s_own_boundary_store_is_always_declined() -> None:
    """The engine offers every retained recurrent boundary to the connector,
    keyed as this connector's finish-time store keys them, so taking both
    would store the same block twice. This connector's store is the one that
    runs, so the stock builder is never consulted -- and the request the
    handoff names is registered but its fake carries none of the state the
    stock builder reads, so an empty result is proof it was not reached.
    """
    scheduler = make_scheduler(None)
    tracked(scheduler, FakeRequest("r", [bytes([1]) * 32]))
    output = typing.cast(
        typing.Any,
        types.SimpleNamespace(
            kv_connector_block_state=types.SimpleNamespace(boundary_state_offloads={"r": [(0, 1, 16)]})
        ),
    )
    assert scheduler._build_partial_tail_store_jobs(output) == {}


def test_the_band_neither_stores_nor_touches_a_dead_block() -> None:
    """A block the engine sent to the eviction end as nobody's live context
    (vLLM's KVCacheBlock.dead) can never become a hit either: the band lets
    it die on the GPU, and its departure reads as dead, not a lost race."""
    pool = make_pool(64)
    freed = free_blocks(pool, 4)
    if not hasattr(freed[0], "dead"):
        pytest.skip("the pinned vLLM has no KVCacheBlock.dead")
    scheduler = make_scheduler(pool)
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    for block in (freed[1], freed[2]):
        block.dead = True

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    assert manager.offered == [[key_of(freed[0]), key_of(freed[3])]]
    assert manager.touched == []
    assert scheduler._dead_skipped == 2
    assert list(scheduler._dead_keys) == [key_of(freed[1]), key_of(freed[2])]


class _ParkThreads:
    """The engine's context threads as park mode reads them: the finishing
    request's decode tails, and dead keys to drain."""

    def __init__(self, tails: list[typing.Any]) -> None:
        self.tails = tails

    def request_decode_tails(self, request_id: str) -> list[tuple[typing.Any, typing.Any]]:
        return [(block, block.block_hash) for block in self.tails]

    def take_dead_keys(self) -> list[typing.Any]:
        return []


def test_park_stores_a_finished_thread_whole_and_uncaches_it_on_the_gpu() -> None:
    """At a finish, park mode offers the whole thread: every full attention
    chunk, the recurrent checkpoint at the last full boundary, and at the
    decode tail the attention block and the recurrent state, keyed at the
    tail's boundary. The idle tail block is held until the copy lands; then
    every block of the thread is uncached on the GPU."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    unit, per_chunk = HASH_BLOCK_SIZE, 4
    chunk = unit * per_chunk
    hashes = [vllm.v1.core.kv_cache_utils.BlockHash(i.to_bytes(32, "big")) for i in range(1, 13)]
    a0, a1, a2, r1, tail = pool.get_new_blocks(5)

    def key_hash(index: int, group: int) -> typing.Any:
        return vllm.v1.core.kv_cache_utils.BlockHashWithGroupId(hashes[index] + group.to_bytes(4, "big"))

    a0._block_hash = key_hash(3, 1)  # attention chunk 0
    a1._block_hash = key_hash(7, 1)  # attention chunk 1
    r1._block_hash = key_hash(7, 0)  # the recurrent state at 2 chunks
    tail._block_hash = key_hash(9, 0)  # the decode tail at 2 chunks + 2 units
    pool.free_blocks([tail])  # keyed and idle, as a decode tail is
    typing.cast(typing.Any, pool).context_threads = _ParkThreads([tail])
    scheduler = make_scheduler(pool)
    scheduler._park = True
    log = io.StringIO()
    scheduler._park_log = log
    scheduler.config = offload_sched.SchedulerOffloadConfig(
        kv_group_configs=typing.cast(
            typing.Any,
            (
                types.SimpleNamespace(
                    hashes_per_chunk=per_chunk, tokens_per_chunk=chunk, sliding_window_size_in_chunks=1
                ),
                types.SimpleNamespace(
                    hashes_per_chunk=per_chunk, tokens_per_chunk=chunk, sliding_window_size_in_chunks=None
                ),
            ),
        ),
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
        tokens_per_hash=unit,
        supports_partial_tail=True,
    )
    manager = RecordingManager()
    scheduler.manager = typing.cast(typing.Any, manager)
    request = types.SimpleNamespace(
        request_id="game#r1",
        block_hashes=hashes,
        num_tokens=2 * chunk + 3 * unit,
        num_computed_tokens=2 * chunk + 3 * unit,
        sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": "game/conv1"}),
    )
    req_status = types.SimpleNamespace(req_context=offload_base.ReqContext(req_id="game#r1"), transfer_jobs=set())
    groups = ([0, r1.block_id, 0], [a0.block_id, a1.block_id, a2.block_id])

    assert scheduler._park_store(request, groups, pool, req_status, "game/conv1")

    def key(index: int, group: int) -> typing.Any:
        return offload_base.make_offload_key(hashes[index], group)

    assert manager.offered == [[key(7, 0), key(9, 0), key(3, 1), key(7, 1), key(9, 1)]]
    (job_id,) = scheduler._pending_finish_jobs
    assert tail.ref_cnt == 1  # held until the copy lands
    assert scheduler._parked["game/conv1"]["boundary"] == 2 * chunk + 2 * unit

    scheduler._jobs.pop(job_id)  # the copy lands
    scheduler._park_settle([])
    assert tail.ref_cnt == 0
    pool.free_blocks([a0, a1, a2, r1])  # the engine frees the request
    scheduler._park_step(pool)
    assert all(block.block_hash is None for block in (a0, a1, r1, tail))
    text = log.getvalue()
    assert "PARK ctx=conv1" in text and "STORED ctx=conv1" in text and "GPU-UNCACHE" in text
    assert "SNAPSHOT running 0" in text


def _park_config(unit: int, per_chunk: int) -> typing.Any:
    """A recurrent group (0) and an attention group (1), one chunk a block."""
    chunk = unit * per_chunk
    return offload_sched.SchedulerOffloadConfig(
        kv_group_configs=typing.cast(
            typing.Any,
            (
                types.SimpleNamespace(
                    hashes_per_chunk=per_chunk, tokens_per_chunk=chunk, sliding_window_size_in_chunks=1
                ),
                types.SimpleNamespace(
                    hashes_per_chunk=per_chunk, tokens_per_chunk=chunk, sliding_window_size_in_chunks=None
                ),
            ),
        ),
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
        tokens_per_hash=unit,
        supports_partial_tail=True,
    )


def test_park_mode_still_keeps_and_stores_a_running_request_s_newest_checkpoint() -> None:
    """Park mode has no band, but the rest of the step stands: a known
    request's newest checkpoint, which the engine frees the step after its
    boundary is crossed and its park needs, is kept young and stored."""
    pool = make_pool(8)
    scheduler = make_scheduler(pool, band=2, single_state_groups=(0,))
    scheduler._park = True
    manager = RecordingManager()
    manager._policy = victim_offload.VictimLRUPolicy(cache_capacity=8)  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    ckpt, older_a, older_b, *_held = pool.get_new_blocks(7)
    hashes = [seed.to_bytes(32, "big") for seed in (1, 2)]
    give_hash(ckpt, 2, group=0)
    pool.cached_block_hash_to_block.insert(typing.cast(typing.Any, ckpt.block_hash), ckpt)
    pool.free_blocks([ckpt])
    for block, seed in ((older_a, 7), (older_b, 8)):
        give_hash(block, seed)
    pool.free_blocks([older_a, older_b])
    tracked(scheduler, FakeRequest("r12", hashes))

    scheduler._build_store_jobs(typing.cast(typing.Any, None))

    # The checkpoint alone: the band's two are not park mode's to store. It
    # is held while its copy is in flight, then goes back to the young end.
    assert manager.offered == [[key_of(ckpt)]]
    (job_id,) = scheduler._spill_holds
    assert ckpt.ref_cnt == 1
    scheduler._release_spill_hold(job_id)
    assert ckpt.ref_cnt == 0
    assert pool.free_block_queue.fake_free_list_tail.prev_free_block is ckpt


def test_a_park_counts_a_checkpoint_only_the_tier_still_holds() -> None:
    """A tail restore loads no recurrent checkpoint, so by the next finish
    the GPU often has none at the last full boundary. When the tier holds it
    (the keep-young pass stored it), the park counts it as the thread's."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    unit, per_chunk = HASH_BLOCK_SIZE, 4
    chunk = unit * per_chunk
    hashes = [vllm.v1.core.kv_cache_utils.BlockHash(i.to_bytes(32, "big")) for i in range(1, 13)]
    a0, a1, a2, tail = pool.get_new_blocks(4)

    def key_hash(index: int, group: int) -> typing.Any:
        return vllm.v1.core.kv_cache_utils.BlockHashWithGroupId(hashes[index] + group.to_bytes(4, "big"))

    a0._block_hash = key_hash(3, 1)
    a1._block_hash = key_hash(7, 1)
    tail._block_hash = key_hash(9, 0)
    pool.free_blocks([tail])
    typing.cast(typing.Any, pool).context_threads = _ParkThreads([tail])
    scheduler = make_scheduler(pool)
    scheduler._park = True
    scheduler.config = _park_config(unit, per_chunk)
    checkpoint = offload_base.make_offload_key(hashes[7], 0)
    policy = victim_offload.VictimLRUPolicy(cache_capacity=32)
    status = policies_base.BlockStatus(0)
    status.ref_cnt = 0
    policy.insert(checkpoint, status)
    manager = RecordingManager(present=[checkpoint])
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    request = types.SimpleNamespace(
        request_id="game#r2",
        block_hashes=hashes,
        num_tokens=2 * chunk + 3 * unit,
        num_computed_tokens=2 * chunk + 3 * unit,
        sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": "game/conv1"}),
    )
    req_status = types.SimpleNamespace(req_context=offload_base.ReqContext(req_id="game#r2"), transfer_jobs=set())
    # No recurrent block at the last full boundary on the GPU.
    groups = ([0, 0, 0], [a0.block_id, a1.block_id, a2.block_id])

    assert scheduler._park_store(request, groups, pool, req_status, "game/conv1")

    assert checkpoint in manager.offered[0] and [checkpoint] in manager.touched
    record = scheduler._parked["game/conv1"]
    assert checkpoint in record["keys"] and record["checkpoints"] == {checkpoint}


def test_park_never_demotes_a_key_a_parked_thread_or_a_live_checkpoint_needs() -> None:
    """A key a parked thread holds (a common prompt head) stays, whoever else
    is demoting it; so does a running thread's checkpoint at its last full
    boundary, which the GPU need not keep. Anything else is demoted."""
    scheduler = make_scheduler(None)
    scheduler._park = True
    head, ckpt, tail = (offload_base.make_offload_key(bytes([i]) * 32, 0) for i in (1, 2, 3))
    scheduler._parked = {
        "g/a": {"keys": [head, ckpt, tail], "checkpoints": {ckpt}, "state": "running"},
        "g/b": {"keys": [head], "checkpoints": set(), "state": "parked"},
    }
    scheduler._park_claim("g/a", [head, ckpt, tail], held=True)
    scheduler._park_claim("g/b", [head], held=True)

    assert scheduler._park_unneeded([head, ckpt, tail]) == [tail]
    scheduler._parked["g/b"]["state"] = "running"
    assert scheduler._park_unneeded([head, ckpt, tail]) == [head, tail]
    scheduler._park_claim("g/a", [ckpt], held=False)
    assert scheduler._park_unneeded([ckpt]) == [ckpt]
    scheduler._park = False
    assert scheduler._park_unneeded([head, ckpt, tail]) == [head, ckpt, tail]


def test_a_lookup_waits_for_its_own_thread_s_park_copy() -> None:
    """While a conversation's park copy is in flight its next request's
    lookup is deferred; another conversation's is not, and the wait ends
    when the copy lands or the cap runs out."""
    scheduler = make_scheduler(None)
    scheduler._park = True

    def request(rid: str, context: str) -> typing.Any:
        return types.SimpleNamespace(
            request_id=rid, sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": context})
        )

    mine, other = request("g#1", "g/c1"), request("h#1", "h/c9")
    assert not scheduler._park_in_flight(mine)
    scheduler._park_jobs[7] = {"context": "g/c1"}
    assert scheduler._park_in_flight(mine) and scheduler._park_in_flight(mine)
    assert not scheduler._park_in_flight(other)
    scheduler._park_deferred["g#1"] -= victim_offload.PARK_DEFER_MAX_S + 1
    assert not scheduler._park_in_flight(mine)
    assert scheduler._park_counts["defer_timeout"] == 1
    scheduler._park_jobs.clear()
    assert not scheduler._park_in_flight(mine) and "g#1" not in scheduler._park_deferred


def test_a_park_without_a_decode_tail_keeps_its_boundary_on_the_hash_chain() -> None:
    """With no decode tail the park falls back to the computed count, which
    runs ahead of the hash chain by the drafts in flight; a boundary past the
    chain has no key, so the count is clamped to the chain."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    unit, per_chunk = HASH_BLOCK_SIZE, 4
    chunk = unit * per_chunk
    hashes = [vllm.v1.core.kv_cache_utils.BlockHash(i.to_bytes(32, "big")) for i in range(1, 8)]  # 7 units
    typing.cast(typing.Any, pool).context_threads = _ParkThreads([])
    scheduler = make_scheduler(pool)
    scheduler._park = True
    scheduler.config = _park_config(unit, per_chunk)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    request = types.SimpleNamespace(
        request_id="game#r3",
        block_hashes=hashes,
        num_tokens=7 * unit,
        num_computed_tokens=8 * unit + 2,  # drafts in flight past a block boundary
        sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": "game/conv3"}),
    )
    req_status = types.SimpleNamespace(req_context=offload_base.ReqContext(req_id="game#r3"), transfer_jobs=set())

    scheduler._park_store(request, ([0, 0], [0, 0]), pool, req_status, "game/conv3")

    assert scheduler._parked["game/conv3"]["boundary"] == chunk


def test_a_thread_stays_parked_until_its_request_is_scheduled() -> None:
    """A request's first lookup does not end its thread's protection: until
    the request is scheduled, a key its restore will load (a head shared
    with sibling passes) is never demoted by a sibling's load."""
    scheduler = make_scheduler(None)
    scheduler._park = True
    head = offload_base.make_offload_key(bytes([1]) * 32, 0)
    scheduler._parked = {"g/a": {"keys": [head], "checkpoints": set(), "state": "parked", "boundary": 64}}
    scheduler._park_claim("g/a", [head], held=True)
    request = types.SimpleNamespace(
        request_id="g#2",
        num_tokens=100,
        sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": "g/a"}),
    )

    scheduler._park_start(request, 0, None)
    assert scheduler._parked["g/a"]["state"] == "parked" and scheduler._park_unneeded([head]) == []
    scheduler._park_running(request)
    assert scheduler._parked["g/a"]["state"] == "running" and scheduler._park_unneeded([head]) == [head]


class _ThreadsByRequest:
    """The engine's context threads for several requests: each one's decode
    tails, and the dead keys the engine reports, drained once."""

    def __init__(self) -> None:
        self.tails: dict[str, list[typing.Any]] = {}
        self.dead: list[tuple[typing.Any, str, str]] = []

    def request_decode_tails(self, request_id: str) -> list[tuple[typing.Any, typing.Any]]:
        return [(block, block.block_hash) for block in self.tails.get(request_id, [])]

    def take_dead_keys(self) -> list[tuple[typing.Any, str, str]]:
        dead, self.dead = self.dead, []
        return dead


def test_a_cell_whose_engine_parks_no_thread_starts_without_the_gpu_park() -> None:
    """Outside park mode (no context threads or decode tails, the 27B cells)
    the GPU park is off rather than refused, so those engines start without
    it; asked for there, it is refused, and turned off in park mode, it stays off."""
    assert victim_offload._gpu_park_from({}, park=False) is False
    assert victim_offload._gpu_park_from({"gpu_park": False}, park=True) is False
    with pytest.raises(ValueError, match="needs park mode"):
        victim_offload._gpu_park_from({"gpu_park": True}, park=False)


def _gpu_park_scheduler(pool: typing.Any) -> tuple[typing.Any, _ThreadsByRequest, io.StringIO]:
    """A park-mode scheduler parking on the GPU over ``pool``: a recurrent group
    and an attention group, four units a chunk."""
    threads = _ThreadsByRequest()
    pool.context_threads = threads
    scheduler = make_scheduler(pool)
    scheduler._park = True
    scheduler._gpu_park = True
    log = io.StringIO()
    scheduler._park_log = log
    scheduler.config = _park_config(HASH_BLOCK_SIZE, 4)
    scheduler.manager = typing.cast(typing.Any, RecordingManager())
    return scheduler, threads, log


def _finished_thread(
    pool: typing.Any, threads: _ThreadsByRequest, game: int
) -> tuple[typing.Any, typing.Any, list[typing.Any]]:
    """A game's finished request on ``pool``: two attention chunks, the
    recurrent state at their end, and a decode tail two units on, keyed and
    idle as the engine leaves it. Returns the request, its block ids by group,
    and every block of its thread."""
    unit, per_chunk = HASH_BLOCK_SIZE, 4
    chunk = unit * per_chunk
    hashes = [vllm.v1.core.kv_cache_utils.BlockHash((game * 100 + i).to_bytes(32, "big")) for i in range(1, 13)]
    a0, a1, a2, r1, tail = pool.get_new_blocks(5)

    def key_hash(index: int, group: int) -> typing.Any:
        return vllm.v1.core.kv_cache_utils.BlockHashWithGroupId(hashes[index] + group.to_bytes(4, "big"))

    a0._block_hash = key_hash(3, 1)
    a1._block_hash = key_hash(7, 1)
    r1._block_hash = key_hash(7, 0)
    tail._block_hash = key_hash(9, 0)
    pool.free_blocks([tail])
    request = types.SimpleNamespace(
        request_id=f"game{game}#r1",
        block_hashes=hashes,
        num_tokens=2 * chunk + 3 * unit,
        num_computed_tokens=2 * chunk + 3 * unit,
        sampling_params=types.SimpleNamespace(extra_args={"prefix_cache_context": f"game{game}/conv{game}"}),
    )
    threads.tails[request.request_id] = [tail]
    groups = ([0, r1.block_id, 0], [a0.block_id, a1.block_id, a2.block_id])
    return request, groups, [a0, a1, a2, r1, tail]


def _engine_frees(pool: typing.Any, blocks: list[typing.Any]) -> None:
    """The engine frees a finished request's blocks: all but its idle tail."""
    pool.free_blocks(blocks[:4])


def test_gpu_park_keeps_a_finished_thread_pinned_until_its_next_request_runs_on_it() -> None:
    """Parking on the GPU, a finish copies nothing: the thread's blocks stay
    pinned after the engine frees the request, and the next request of the
    conversation, once scheduled on its hit, takes them over."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)

    scheduler._park_pin(request, groups, pool, "game1/conv1")
    _engine_frees(pool, blocks)

    assert scheduler.manager.offered == []
    assert all(block.ref_cnt == 1 for block in blocks)
    follow_up = types.SimpleNamespace(request_id="game1#r2")
    scheduler._park_active[follow_up.request_id] = "game1/conv1"
    pool.touch(blocks)  # its hit, allocated
    scheduler._park_running(follow_up)
    assert all(block.ref_cnt == 1 for block in blocks)
    assert scheduler._parked["game1/conv1"]["state"] == "running"
    assert "pins" not in scheduler._parked["game1/conv1"]
    assert "PARK-GPU ctx=conv1" in log.getvalue() and "UNPIN ctx=conv1 5 blocks" in log.getvalue()


def test_a_pool_short_of_room_spills_the_oldest_gpu_parked_thread_whole() -> None:
    """When the free blocks fall short of what admission needs, the thread
    parked on the GPU longest is copied whole to the tier, its pins holding it
    while the copy is in flight; once it lands the blocks come free and are
    uncached. A younger thread stays pinned."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    first, first_groups, first_blocks = _finished_thread(pool, threads, 1)
    second, second_groups, second_blocks = _finished_thread(pool, threads, 2)
    scheduler._park_pin(first, first_groups, pool, "game1/conv1")
    scheduler._park_pin(second, second_groups, pool, "game2/conv2")
    _engine_frees(pool, first_blocks)
    _engine_frees(pool, second_blocks)
    free = pool.get_num_free_blocks()
    scheduler._park_needs["game3#r1"] = free + 1  # a request waiting to be admitted

    jobs = scheduler._park_spill_step(pool)

    (job_id,) = jobs
    assert scheduler._park_jobs[job_id]["context"] == "game1/conv1"
    assert len(scheduler.manager.offered) == 1
    assert all(block.ref_cnt == 1 for block in first_blocks)  # held while the copy flies
    assert "pins" in scheduler._parked["game2/conv2"]
    assert scheduler._park_spill_step(pool) == {}  # the copy in flight already makes the room

    scheduler._jobs.pop(job_id)  # the copy lands
    scheduler._park_settle([])
    scheduler._park_step(pool)
    assert pool.get_num_free_blocks() == free + len(first_blocks)
    assert all(block.block_hash is None for block in first_blocks)
    assert all(block.ref_cnt == 1 for block in second_blocks)
    assert "SPILL ctx=conv1" in log.getvalue() and "STORED ctx=conv1" in log.getvalue()


def test_a_gpu_parked_thread_whose_next_request_arrived_is_not_spilled() -> None:
    """A thread whose conversation has a request in the engine is about to run
    where it is: the spill passes over it to the next oldest."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, _ = _gpu_park_scheduler(pool)
    first, first_groups, first_blocks = _finished_thread(pool, threads, 1)
    second, second_groups, second_blocks = _finished_thread(pool, threads, 2)
    scheduler._park_pin(first, first_groups, pool, "game1/conv1")
    scheduler._park_pin(second, second_groups, pool, "game2/conv2")
    _engine_frees(pool, first_blocks)
    _engine_frees(pool, second_blocks)
    scheduler._park_active["game1#r2"] = "game1/conv1"
    scheduler._park_needs["game1#r2"] = pool.get_num_free_blocks() + 1

    (job_id,) = scheduler._park_spill_step(pool)

    assert scheduler._park_jobs[job_id]["context"] == "game2/conv2"
    assert "pins" in scheduler._parked["game1/conv1"]


def test_a_step_that_admits_nothing_spills_threads_whose_requests_wait() -> None:
    """Requests waiting on threads parked on the GPU can hold, between them,
    the room each needs: none is admitted, so none runs on its thread and
    lets go of it. A step that scheduled nothing spills them, oldest first,
    until the room is coming; one that scheduled work leaves them where they
    are about to run."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    first, first_groups, first_blocks = _finished_thread(pool, threads, 1)
    second, second_groups, second_blocks = _finished_thread(pool, threads, 2)
    scheduler._park_pin(first, first_groups, pool, "game1/conv1")
    scheduler._park_pin(second, second_groups, pool, "game2/conv2")
    _engine_frees(pool, first_blocks)
    _engine_frees(pool, second_blocks)
    scheduler._park_active["game1#r2"] = "game1/conv1"
    scheduler._park_active["game2#r2"] = "game2/conv2"
    scheduler._park_needs["game1#r2"] = 1
    scheduler._park_needs["game2#r2"] = pool.get_num_free_blocks()

    assert scheduler._park_spill_step(pool, idle=False) == {}
    (job_id,) = scheduler._park_spill_step(pool, idle=True)

    assert scheduler._park_jobs[job_id]["context"] == "game1/conv1"
    assert "pins" in scheduler._parked["game2/conv2"]
    assert "engine idle" in log.getvalue()


def test_the_engine_is_idle_only_when_no_request_moves_on_by_itself() -> None:
    """Waiting requests alone free nothing; a running one, or one whose load
    is in flight, will. A step that schedules nothing does not make the engine
    idle: under pipeline stages every running request can be in flight in the
    other micro-batch."""
    scheduler = make_scheduler(make_pool(8))
    state = vllm.v1.request.RequestStatus

    def tracked(*states: vllm.v1.request.RequestStatus) -> dict[str, typing.Any]:
        return {f"r{i}": types.SimpleNamespace(req=types.SimpleNamespace(status=s)) for i, s in enumerate(states)}

    scheduler._req_status = tracked(state.WAITING, state.PREEMPTED)
    assert scheduler._engine_idle()
    scheduler._req_status = tracked(state.WAITING, state.WAITING_FOR_REMOTE_KVS)
    assert not scheduler._engine_idle()
    scheduler._req_status = tracked(state.WAITING, state.RUNNING)
    assert not scheduler._engine_idle()


def test_a_gpu_parked_thread_that_dies_goes_to_the_eviction_end_still_cached() -> None:
    """A conversation the engine reports closed (or rewritten) leaves nothing
    pinned: its blocks come free at the eviction end of the free queue, the
    first reused, but still cached, as the tier demotes a dead thread's keys.
    The close comes ahead of the request that replaces the thread, whose
    prompt can still extend it."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)
    scheduler._park_pin(request, groups, pool, "game1/conv1")
    _engine_frees(pool, blocks)
    free = pool.get_num_free_blocks()
    hashes = [block.block_hash for block in blocks]
    threads.dead.append((blocks[0].block_hash, "game1/conv1", "close"))

    scheduler._park_step(pool)

    assert "game1/conv1" not in scheduler._parked
    assert pool.get_num_free_blocks() == free + len(blocks)
    assert [block.block_hash for block in blocks] == hashes
    first: list[typing.Any] = []
    block: typing.Any = pool.free_block_queue.fake_free_list_head.next_free_block
    for _ in blocks:
        first.append(block)
        block = block.next_free_block
    assert {id(block) for block in first} == {id(block) for block in blocks}
    assert "UNPIN ctx=conv1 5 blocks; thread dead (close)" in log.getvalue()


def test_gpu_park_publishes_the_pool_s_shares_requests_and_parked_threads_hold() -> None:
    """Parking on the GPU, the engine line's two shares of the pool: the
    blocks requests hold, and the blocks only parked threads hold. A block a
    request shares with a parked thread counts as the request's."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, _ = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)
    scheduler._park_pin(request, groups, pool, "game1/conv1")
    _engine_frees(pool, blocks)
    running = pool.get_new_blocks(3)  # a running request's own blocks
    pool.touch(blocks[:1])  # and the parked thread's first block, shared with it

    scheduler._publish_kv_shares(pool)

    total = pool.num_gpu_blocks - 1
    shares = scheduler._connector_stats.data["data"]
    assert shares[ullm.engine_monitor.KV_REQUESTS_SHARE][()] == pytest.approx((len(running) + 1) / total)
    assert shares[ullm.engine_monitor.KV_GPU_PARKED_SHARE][()] == pytest.approx((len(blocks) - 1) / total)


class _RecordedMetric:
    """A Prometheus metric stand-in: records what each label set was set to."""

    def __init__(self, name: str, documentation: str, labelnames: list[str], **_: typing.Any) -> None:
        self.name = name
        self.values: dict[tuple[str, ...], float] = {}
        self._bound: tuple[str, ...] = ()

    def labels(self, *labelvalues: typing.Any) -> "_RecordedMetric":
        bound = _RecordedMetric(self.name, "", [])
        bound.values = self.values
        bound._bound = tuple(str(value) for value in labelvalues)
        return bound

    def set(self, value: float) -> None:
        self.values[self._bound] = value

    def inc(self, value: float = 1.0) -> None:
        self.values[self._bound] = self.values.get(self._bound, 0.0) + value

    def observe(self, value: float) -> None:
        self.values[self._bound] = value


def test_the_frontend_defines_and_records_the_pool_s_shares() -> None:
    """The frontend builds the connector's metrics through the connector, and
    refuses a stats key it has no definition for: the two shares are defined
    there and reach their gauges."""
    import prometheus_client

    config = types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config={}))
    kinds = {
        prometheus_client.Gauge: _RecordedMetric,
        prometheus_client.Counter: _RecordedMetric,
        prometheus_client.Histogram: _RecordedMetric,
    }
    metrics = victim_offload.VictimOffloadingConnector.build_prom_metrics(
        typing.cast(typing.Any, config), typing.cast(typing.Any, kinds), ["model_name", "engine"], {0: ["m", "0"]}
    )
    stats = offload_metrics.OffloadingConnectorStats()
    stats.set_gauge(ullm.engine_monitor.KV_REQUESTS_SHARE, 0.5)
    stats.set_gauge(ullm.engine_monitor.KV_GPU_PARKED_SHARE, 0.25)

    metrics.observe(stats.data, engine_idx=0)

    defs = typing.cast(typing.Any, metrics)._offloading_metric_defs
    assert defs[ullm.engine_monitor.KV_REQUESTS_SHARE].values == {("m", "0"): 0.5}
    assert defs[ullm.engine_monitor.KV_GPU_PARKED_SHARE].values == {("m", "0"): 0.25}


def _keyed_idle_block(pool: typing.Any, seed: int) -> typing.Any:
    """A block keyed as a recurrent group's decode tail and left idle in the
    free queue, as the engine leaves one."""
    (block,) = pool.get_new_blocks(1)
    block._block_hash = vllm.v1.core.kv_cache_utils.BlockHashWithGroupId(
        vllm.v1.core.kv_cache_utils.BlockHash(seed.to_bytes(32, "big")) + (0).to_bytes(4, "big")
    )
    pool.free_blocks([block])
    return block


def test_gpu_park_pins_a_running_request_s_newest_tail_in_place_of_a_copy() -> None:
    """Parking on the GPU, a running request's newest decode tail is held on
    the GPU rather than copied to the tier, and the tail it supersedes is let
    go of."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, _ = _gpu_park_scheduler(pool)
    scheduler._req_status = {"g#r": types.SimpleNamespace(req=types.SimpleNamespace(request_id="g#r"))}
    first, second = _keyed_idle_block(pool, 901), _keyed_idle_block(pool, 902)

    threads.tails["g#r"] = [first]
    scheduler._keep_tails(pool)
    assert first.ref_cnt == 1
    threads.tails["g#r"] = [second]
    scheduler._keep_tails(pool)

    assert (first.ref_cnt, second.ref_cnt) == (0, 1)
    assert scheduler._kept_blocks == [] and scheduler.manager.offered == []


def test_a_short_pool_lets_go_of_running_tails_before_it_spills_a_parked_thread() -> None:
    """Short of room, the spill step first lets go of the running requests'
    pinned tails, which copies nothing; only if that is not enough does a
    thread parked on the GPU spill."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)
    scheduler._park_pin(request, groups, pool, "game1/conv1")
    _engine_frees(pool, blocks)
    scheduler._req_status = {"g#r": types.SimpleNamespace(req=types.SimpleNamespace(request_id="g#r"))}
    tail = _keyed_idle_block(pool, 903)
    threads.tails["g#r"] = [tail]
    scheduler._keep_tails(pool)
    assert tail.ref_cnt == 1
    scheduler._park_needs["w#1"] = pool.get_num_free_blocks() + 1  # one tail's room short

    assert scheduler._park_spill_step(pool) == {}
    assert tail.ref_cnt == 0 and "pins" in scheduler._parked["game1/conv1"]
    assert "TAILS-RELEASED 1 tail blocks" in log.getvalue()

    scheduler._park_needs["w#1"] = pool.get_num_free_blocks() + 1  # short again, no tail left
    (job_id,) = scheduler._park_spill_step(pool)
    assert scheduler._park_jobs[job_id]["context"] == "game1/conv1"


def test_a_finish_hands_its_pinned_tail_to_the_parked_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """At a finish the parked thread pins the tail its request held pinned,
    and the request's own pin goes: the tail is held throughout, once."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, _ = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)
    tail = blocks[4]
    scheduler._req_status = {
        request.request_id: types.SimpleNamespace(
            req=request, transfer_jobs=set(), req_context=offload_base.ReqContext(req_id=request.request_id)
        )
    }
    scheduler._keep_tails(pool)
    assert tail.ref_cnt == 1

    def stock_finish(self: typing.Any, request: typing.Any) -> tuple[bool, None]:
        """The stock bookkeeping at a finish, which holds nothing here."""
        return False, None

    monkeypatch.setattr(offload_sched.OffloadingConnectorScheduler, "request_finished", stock_finish)

    scheduler._park_finish(request, groups)
    _engine_frees(pool, blocks)

    assert tail.ref_cnt == 1
    assert scheduler._tail_pins == {}
    assert tail.block_id in {block_id for block_id, _ in scheduler._parked["game1/conv1"]["pins"]}


def _tier_holding(scheduler: typing.Any, keys: list[typing.Any]) -> tuple[typing.Any, typing.Any]:
    """Give ``scheduler`` a tier policy that holds ``keys``, idle, and a
    recording manager that reports them present. Returns both."""
    import vllm.v1.kv_offload.cpu.policies.base as policies_base

    policy = victim_offload.VictimLRUPolicy(cache_capacity=32)
    for key in keys:
        status = policies_base.BlockStatus(0)
        status.ref_cnt = 0
        policy.insert(key, status)
    manager = RecordingManager(present=keys)
    manager._policy = policy  # type: ignore[attr-defined]
    scheduler.manager = typing.cast(typing.Any, manager)
    return policy, manager


def _resumed_past_its_checkpoint(
    pool: typing.Any, threads: _ThreadsByRequest, scheduler: typing.Any
) -> tuple[typing.Any, tuple[list[int], ...], list[typing.Any], typing.Any, typing.Any, typing.Any]:
    """A game's finished request that resumed at its thread's decode tail, so
    it holds no recurrent state at the last full boundary, which the tier
    alone still has. Returns the request, its block ids by group, the
    thread's GPU blocks, the checkpoint's key, and the tier's policy and
    manager."""
    request, groups, blocks = _finished_thread(pool, threads, 1)
    checkpoint = offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.get_block_hash(blocks[3].block_hash), 0)
    policy, manager = _tier_holding(scheduler, [checkpoint])
    pool.free_blocks([blocks[3]])  # the engine reused its block long ago
    blocks[3]._block_hash = None
    return request, ([0, 0, 0], groups[1]), [b for i, b in enumerate(blocks) if i != 3], checkpoint, policy, manager


def test_a_gpu_park_holds_and_touches_the_checkpoint_only_the_tier_has() -> None:
    """A request resumed at its thread's decode tail holds no recurrent state
    at the last full boundary. Its park on the GPU pins the rest and holds
    the tier's checkpoint as the thread's, touched as a park touches what the
    tier has, and the next request taking the thread over touches it again,
    as a tier resume's load does: nothing else of the thread uses it, so
    untouched it ages out of the tier while the thread lives."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    request, groups, blocks, checkpoint, _, manager = _resumed_past_its_checkpoint(pool, threads, scheduler)

    scheduler._park_pin(request, groups, pool, "game1/conv1")

    record = scheduler._parked["game1/conv1"]
    assert record["keys"] == [checkpoint] and len(record["pins"]) == len(blocks)
    assert manager.touched == [[checkpoint]] and manager.offered == []
    assert "PARK-GPU ctx=conv1" in log.getvalue() and "hosted=1" in log.getvalue()
    follow_up = types.SimpleNamespace(request_id="game1#r2")
    scheduler._park_active[follow_up.request_id] = "game1/conv1"
    scheduler._park_running(follow_up)
    assert manager.touched == [[checkpoint], [checkpoint]]


def test_a_park_leaves_out_a_host_only_key_the_tier_lost_since_its_plan() -> None:
    """A key no GPU block backs is offered only while the tier still has it:
    one the tier evicted between the plan and the copy (a spill comes long
    after its plan) is left out, so every key the tier takes as new has a
    block to copy from, and the thread parks without it."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, log = _gpu_park_scheduler(pool)
    request, groups, _, checkpoint, policy, manager = _resumed_past_its_checkpoint(pool, threads, scheduler)
    req_status = types.SimpleNamespace(
        req_context=offload_base.ReqContext(req_id=request.request_id), transfer_jobs=set()
    )
    plan = scheduler._park_plan(request, groups, pool)
    assert plan["hosted"] == [checkpoint]
    policy.remove(checkpoint)  # the tier evicts it before the copy
    manager.present.clear()

    assert scheduler._park_issue(plan, "game1/conv1", request.request_id, pool, req_status=req_status) == "queued"

    (offered,) = manager.offered
    assert checkpoint not in offered and checkpoint not in scheduler._parked["game1/conv1"]["keys"]
    (job,) = scheduler._pending_finish_jobs.values()
    assert len(job.src_spec.block_ids) == len(offered)
    assert "lost=1" in log.getvalue()


def test_a_gpu_park_holds_the_tier_s_checkpoint_when_the_gpu_has_one_too() -> None:
    """The tier's copy of the checkpoint is the thread's even when the GPU
    has the checkpoint at the park: the next request, resumed at the tail,
    does not hold the GPU's, which the pool then reuses, so held and touched
    the tier's is the copy that stays."""
    pool = make_pool(32)
    if not hasattr(pool, "context_threads"):
        pytest.skip("the pinned vLLM has no context threads")
    scheduler, threads, _ = _gpu_park_scheduler(pool)
    request, groups, blocks = _finished_thread(pool, threads, 1)
    checkpoint = offload_base.make_offload_key(vllm.v1.core.kv_cache_utils.get_block_hash(blocks[3].block_hash), 0)
    _, manager = _tier_holding(scheduler, [checkpoint])

    scheduler._park_pin(request, groups, pool, "game1/conv1")

    record = scheduler._parked["game1/conv1"]
    assert blocks[3].block_id in {block_id for block_id, _ in record["pins"]}  # the GPU's, pinned
    assert record["keys"] == [checkpoint] and manager.touched == [[checkpoint]]
