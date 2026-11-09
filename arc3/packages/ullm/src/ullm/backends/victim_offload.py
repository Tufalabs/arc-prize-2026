"""CPU KV offload that mirrors the GPU pool's eviction front.

vLLM's stock ``OffloadingConnector`` stores every block as soon as it is
computed, so a host tier fills with content the GPU still holds. This
variant copies a block only as it nears the front of the GPU free queue —
the position the pool allocates from, whatever the pool's occupancy — so
the tier holds what the GPU is about to lose and nothing it will keep.
Two things are layered on that admission rule:

- Eviction classes for a recurrent model's checkpoints. A single-state KV
  group (Mamba, GDN) writes a checkpoint at every aligned boundary and the
  lookup scans it from the end for the first survivor, so the tier ranks
  a context's latest boundaries last to go, the boundaries it has moved
  past first to go (and never copies them), and everything else by age.
  See ``VictimLRUPolicy``.
- A hash check on every copy. A source block is the pool's to reuse at
  any moment, so each copy is kept only if the block still carries the
  hash it was read under when the copy lands. Nothing a hash did not
  vouch for is served.

Selected through ``KVTransferConfig.kv_connector_module_path``; see
``config_local.enable_cpu_kv_offload``. Everything below is scheduler-side
except a small hold-bookkeeping override in the worker: what changes is
which blocks are handed to it, not how they travel. Instrumentation goes
to a trace in the run's job directory (``kv_offload.md``, Instrumentation)
and, with ``engine_log`` set, to the engine log through vLLM's logger: a
spill and departure tally, a lookup classifier, and the reuse-depth line
that sizes the tier, once a minute each; a departure walk of the free
queue each step, bounded by the pool.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import time
import typing

import ullm.engine_monitor
import vllm.config
import vllm.distributed.kv_transfer.kv_connector.v1.base as _connector_base
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.common as _offload_common
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.config as _offload_config
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics as _offload_metrics
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as _offload_sched
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker as _offload_worker
import vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector as _offload_conn
import vllm.logger
import vllm.v1.core.kv_cache_utils as _kv_cache_utils
import vllm.v1.kv_cache_interface
import vllm.v1.kv_offload.base as _offload_base
import vllm.v1.kv_offload.cpu.policies.factory as _policy_factory
import vllm.v1.kv_offload.cpu.policies.lru as _lru_policy
import vllm.v1.kv_offload.factory as _offload_factory
import vllm.v1.request as _request

if typing.TYPE_CHECKING:
    import vllm.v1.core.block_pool
    import vllm.v1.core.sched.output
    import vllm.v1.outputs

# The vLLM line this connector is written against. It subclasses the stock
# offloading scheduler and worker and reads their private state — job
# tables, the flush registry, the free queue's list, the manager's policy —
# every one of which moves between lines.
# Checked when the module is imported, which happens only when this
# connector is selected, so an engine on another line fails at startup with
# a message rather than partway into a run with a missing attribute.
SUPPORTED_VLLM_LINE = (0, 29)


def _check_vllm_line(version: str) -> None:
    """Raise unless `version` (any build of it: release, dev, fork) is on
    the supported line."""
    parts = version.split(".")
    try:
        line = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError) as exc:
        raise ImportError(f"victim offload: cannot read a vLLM line from version {version!r}") from exc
    if line != SUPPORTED_VLLM_LINE:
        raise ImportError(
            f"victim offload: written against vLLM {SUPPORTED_VLLM_LINE[0]}.{SUPPORTED_VLLM_LINE[1]}.x, "
            f"this engine is {version}; the stock internals it subclasses move between lines"
        )


_check_vllm_line(vllm.__version__)

# Under vLLM's own logger hierarchy on purpose: the "vllm" logger carries the
# handler and does not propagate, so a logger named for this module would send
# INFO to the root logger, which drops it — the tally would never be seen.
logger = vllm.logger.init_logger("vllm.ullm.victim_offload")

# Reserved request id for stores that belong to no request. Victim blocks are
# unreferenced by definition, so their transfers cannot be attributed to a
# running request the way the stock per-request store path attributes its own.
VICTIM_REQ_ID = "__victim_spill__"

# Fraction of the GPU pool that must be free before a finishing request's
# blocks are held for a copy. Under it a freed block is reused within a step,
# before any copy could land, so the copy has to happen while the request
# still owns the blocks. Above it the freed blocks join the queue as cache
# and the head-spill reaches them in their turn.
DEFAULT_SPILL_WATERMARK = 0.15

# How far into the free queue the head-spill looks, in queue entries. The
# pool allocates from the head of that queue whether it is short or long,
# and in a warm run every entry in it is cached content, so eviction is not
# a state the pool enters but the front of the queue at all times. What the
# tier must hold is whatever is about to leave that front, and reaching the
# front is the one event that makes a host copy worth something — the block
# stops being a duplicate of what the GPU has. So the band is the eviction
# front and no deeper: under chunked prefill the pool evicts at most a
# couple of blocks a step from the head, so this many entries is seconds of
# lead for the copy, while a band that spans a deep queue stores and
# refreshes blocks that a prefix hit will take back before they ever leave
# (at a 1,330-block pool, half of them), and makes the tier's recency the
# GPU's churn. A never-full run has unhashed blocks at the front, which the
# walk skips, so it still spills nothing.
DEFAULT_SPILL_BAND_BLOCKS = 128

# New stores issued per scheduler step. Bounds the PCIe burst one step can
# queue, so a band that fills with fresh victims drains over several steps
# rather than stalling one on a single large transfer.
DEFAULT_MAX_SPILL_BLOCKS = 128

# Name under which the host tier's eviction policy below is registered with
# vLLM's CPU offloading manager. Selecting it is what gives the connector a
# demote, which stock LRU does not have.
VICTIM_POLICY = "victim_lru"

# How long after the last observed watermark crossing the system still counts
# as pressured. The instantaneous free-block count is a bad signal in exactly
# the state that matters: a preemption storm frees whole contexts into the
# queue, so the count spikes to "headroom" at maximal pressure and an
# instantaneous gate would skip the very finishes the hold exists for. A
# false positive costs one held copy of a context that was next to be evicted
# anyway; a false negative costs recomputing it.
PRESSURE_STICKY_S = 30.0

# Whether a block's host copy is sent to the eviction end of its class when
# the block lands on the GPU (a load, or a local prefix hit). Exclusive
# ordering: the tier then holds what the pool does not, at the cost of
# copying the block again when the GPU gives it up — a copy per cycle,
# which is the load rate. On, because the alternative is a tier that spends
# a pool's worth of slots on duplicates and is capacity-bound as soon as
# the pool is large against it (kv_offload.md, "The demote"). The
# re-copies evict the demoted copies first, not keys still wanted.
DEFAULT_DEMOTE_ON_USE = True

# Whether the instrumentation also goes to the engine log. The trace in the
# job directory carries everything the log lines do — the tally once a
# minute as one event — so the lines exist for watching a run live in the
# engine's output, and off by default the engine log stays the engine's.
# Leave it off on Kaggle: the lines were ~98% of what the engine process
# wrote, and a kernel that stops draining that pipe stalls the engine on
# the tally's `logger.info` mid-step (ARC2's twelve-hour run of 2026-09-19
# froze that way for its last 2h10m).
DEFAULT_ENGINE_LOG = False

# How many KV cache groups a boundary's keys are looked for in when one of
# them is used: the attention block and the checkpoints at a boundary carry
# the same block hash with different group indices, and a use of any is a
# use of all (see VictimLRUPolicy). A hybrid model has a handful of groups;
# the probe is a dictionary lookup per group.
MAX_KV_GROUPS = 8

# A departure within this many steps of a re-age is counted as the
# re-aging failing to hold the block. Steps run at tens a second; a block
# the pool takes back this soon after being put at the tail of the queue
# did not leave by age.
REAGE_RECENT_STEPS = 200

# Dead band victims' keys remembered for the departure accounting: enough for
# every block of a large pool to have been dead once.
DEAD_KEYS_LIMIT = 1 << 16

# How far back from a request's newest full chunk its cached checkpoint is
# looked for, in chunks. The continuation's lookup scans from the end for
# the newest boundary the GPU still has, so that is the block worth keeping;
# the engine hashes every boundary it completes and a turn rarely crosses
# more than a couple before it finishes, so a short probe finds it and a
# longer one would only find passed ones.
KEEP_YOUNG_PROBE_CHUNKS = 4

# The scheduler-side trace: one JSON object per event, appended to this
# file in the run's job directory when the engine was given one.
TRACE_FILENAME = "kv_offload_trace.jsonl"
# Park mode's log in the job directory: one line per host transfer and per
# decision behind one, each with the tier's state at that moment. On by
# default wherever the run has a job directory (2026-09-28); the
# connector's ``park_log: False`` turns it off, which platypus's solver sets
# under TAAF's minimal diagnostics (a Kaggle submission).
PARK_LOG_FILENAME = "kv_park.log"
DEFAULT_PARK_LOG = True
# Which of a used boundary's other keys park mode moves to the newest end
# with it: "needed", those a thread still needs (``_park_needed``), or "all"
# (the tier's own rule, and the band's). Each park supersedes its thread's
# previous recurrent checkpoints and demotes them, but they share hashes with
# the thread's attention chunks, which stay live: under "all" every resume and
# re-park pulls them back young, and on the RTX they were over half of what
# sat newer than the deepest parked thread (Kaggle E/F, 2026-09-28), which
# the tier then dropped in a burst of parks. The connector's
# ``sibling_refresh`` config chooses.
DEFAULT_SIBLING_REFRESH = "needed"
SIBLING_REFRESH_MODES = ("needed", "all")
# How often park mode writes its per-conversation snapshot: which are
# running, and the host tier's contents in eviction order.
PARK_SNAPSHOT_S = 10.0
#: Park on the GPU (``gpu_park``, park mode only): a finished thread stays on
#: the GPU, pinned, and spills whole to the tier, oldest first, only when the
#: pool needs the room (``kv_offload.md``, "Parking on the GPU"). On wherever
#: threads park, the one place it acts (2026-10-18).
DEFAULT_GPU_PARK = True
#: Free blocks the GPU park keeps beyond the engine's admission watermark, as
#: a share of the pool: the running requests grow into them while a spill's
#: copy is in flight, which they cannot do into pinned blocks.
DEFAULT_GPU_PARK_RESERVE = 0.02
#: How often, parking on the GPU, the pool's shares are set for the engine
#: line (``ullm.engine_monitor.KV_REQUESTS_SHARE``): its line is a minute's.
KV_SHARE_INTERVAL_S = 1.0
#: The states of a request that holds GPU blocks and moves on with them by
#: itself: it runs, or waits for a load already in flight. A request in flight
#: in a pipeline stage's other micro-batch still runs.
_PROGRESSING = frozenset({_request.RequestStatus.RUNNING, _request.RequestStatus.WAITING_FOR_REMOTE_KVS})
# Longest park mode holds a request's lookup back while its thread's park
# copy is in flight (1-4 steps, normally): past it the lookup goes ahead, so
# a copy that never lands cannot stall the request.
PARK_DEFER_MAX_S = 5.0

# Cadence of the trace's per-step sample. The collapse the trace exists to
# see unfolds over a couple of minutes at tens of steps a second: a sample
# a second resolves it at a hundredth of the lines.
TRACE_SAMPLE_INTERVAL_S = 1.0

# How far back the trace probes: from a lookup's attention hit for each
# recurrent group's newest cached boundary, and from a chain's end for the
# finish of the turn before. A turn adds at most a couple of chunks, and
# further back is passed history.
TRACE_PROBE_CHUNKS = 8

# Finishes remembered for linking a lookup to the turn before it. Hundreds
# of requests can pass between a finish and its continuation.
TRACE_FINISH_MEMORY = 1024

# How many finishes have their per-group block pattern written to the log,
# so a run shows which blocks the engine still holds for a finished request
# in each group; the classification does not depend on it, but reading a
# run does.
FINISH_PATTERNS_TO_LOG = 6

# How often the running spill tally is written to the log. Prometheus carries
# bytes moved but not blocks dropped, and dropped blocks are what separate a
# tier that is cold from one that is being fed copies it then throws away.
TALLY_INTERVAL_S = 60.0


def _gpu_park_from(extra: typing.Mapping[str, typing.Any], park: bool) -> bool:
    """Whether finished threads park on the GPU: as ``gpu_park`` asks, else
    wherever park mode is, without which no thread parks, so a cell whose
    engine parks none starts without it. Asked for without park mode, refused."""
    gpu_park = bool(extra.get("gpu_park", DEFAULT_GPU_PARK and park))
    if gpu_park and not park:
        raise ValueError("victim offload: gpu_park parks threads, and needs park mode")
    return gpu_park


def _refuse_unsupported_groups(configs: typing.Iterable[typing.Any]) -> None:
    """Refuse a KV cache group shape this connector has never run.

    Everything recurrent-specific here — the checkpoint classes, the
    keep-young pass, the kept-checkpoint store, the fill — reads a windowed
    group as a single-state one: one block carries the whole state at a
    boundary. That is Mamba/GDN, whose window is one chunk. Ordinary
    sliding-window attention with a wider window, and the volatile trailing
    chunk of an eagle/MTP draft group, are other shapes; running them would
    serve a partial window as a whole one. Refused at start so the run never
    begins, rather than at the fill where it would go quiet.
    """
    for config in configs:
        window = getattr(config, "sliding_window_size_in_chunks", None)
        group_idx = getattr(config, "group_idx", "?")
        if getattr(config, "is_eagle_group", False):
            raise ValueError(
                f"victim offload: KV cache group {group_idx} is an eagle/MTP draft group; "
                "the connector's checkpoint handling assumes every windowed group is a single-state "
                "(Mamba/GDN) group, and this shape has not been run"
            )
        if window is not None and window != 1:
            raise ValueError(
                f"victim offload: KV cache group {group_idx} is a sliding-window group of {window} chunks; "
                "the connector's checkpoint handling and its fill assume every windowed group is a "
                "single-state (Mamba/GDN) group of one chunk, and this shape has not been run"
            )


def _request_context(request: typing.Any) -> str | None:
    """The client context a request names (vLLM's prefix_cache_context)."""
    extra = getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}
    context = extra.get("prefix_cache_context")
    return None if context is None else str(context)


def _boundary_of(hashes: list[typing.Any], block_hash: typing.Any, per_hash: int) -> int | None:
    """The token boundary a key's block hash closes in the request's chain,
    searched from the end; None when it is not in the chain."""
    target = _kv_cache_utils.get_block_hash(block_hash)
    for index in range(len(hashes) - 1, -1, -1):
        if hashes[index] == target:
            return (index + 1) * per_hash
    return None


def _verified_row(
    pool: typing.Any, block_ids: list[int], index: int, block_hash: typing.Any, group_idx: int
) -> tuple[int, _offload_base.OffloadKey] | None:
    """The (block id, host key) of a request's chunk, when its GPU block
    still carries that chunk's hash under this group; None otherwise."""
    if index >= len(block_ids) or not block_ids[index]:
        return None
    gpu_hash = pool.blocks[block_ids[index]].block_hash
    if (
        gpu_hash is None
        or _kv_cache_utils.get_block_hash(gpu_hash) != block_hash
        or _kv_cache_utils.get_group_id(gpu_hash) != group_idx
    ):
        return None
    return block_ids[index], _offload_base.make_offload_key(block_hash, group_idx)


def _short_ctx(context: str | None) -> str:
    """A context's conversation id, short enough for a log line."""
    return "--------" if context is None else context.rsplit("/", 1)[-1][:8]


def _short_key(key: typing.Any) -> str:
    """A host key, short enough for a log line."""
    return bytes(key)[:6].hex()


def _short_req(request_id: str) -> str:
    """A request's own id, short enough for a log line: the part after the
    run id ullm prefixes, which every request of a game shares."""
    return request_id.rsplit("#", 1)[-1][:8]


class VictimLRUPolicy(_lru_policy.LRUCachePolicy):
    """Host-tier LRU in which a boundary ages as one, with passed
    checkpoints ahead of everything, and a DEMOTE.

    A single-state group (Mamba, GDN) writes a checkpoint at every aligned
    boundary of a request, the engine frees each one as the request moves
    past it, and a lookup for such a group scans from the end and takes
    the first surviving checkpoint. Two things follow for the tier.

    PASSED checkpoints — boundaries the request has moved past — are worth
    a hit only when the latest is already gone, and the connector stops
    copying them once it knows; the few that arrive before it knows are
    evicted first, in their own LRU order.

    Everything else is one LRU in which A BOUNDARY AGES AS ONE. The engine
    frees a boundary's checkpoint mid-turn, as the state carries forward,
    and the attention block at that boundary at finish; so the checkpoint
    enters the free queue a turn earlier, leaves the GPU earlier, and block
    by block is always the older of the two. Plain LRU would take a
    context's checkpoints before its attention, and attention without its
    checkpoint is recomputed anyway. The attention block and the
    checkpoints at a boundary carry the same block hash — only the group
    index differs — so a use of any of them is taken as a use of all: a
    context's checkpoints are as young as its attention and die right
    after it, never before, and a context nothing touches ages to the
    front whole. No class, share or constant is needed for that, and any
    such bound inverts at some concurrency (``kv_offload.md``, Retention).

    DEMOTE moves a key to the eviction end of its class, for a copy of a
    block that is now on the GPU. Whether the connector uses it is its
    decision: exclusive ordering saves slots when the tier is small against
    the pool and costs a re-copy per cycle when it is not.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        # LRU depth of every block asked back from the tier, measured at the
        # moment of the ask as the number of keys that would survive it: the
        # capacity a tier needs for that ask to hit. Under LRU inclusion the
        # CDF this accumulates is the tier-sizing curve, so one run at a
        # generous size prices every smaller size at once.
        self.hit_depths: list[int] = []
        # What last moved each stored key in its order, and when (monotonic):
        # "landed" (a store or load completed: the newest end), "touched",
        # "sibling" (a use of its boundary's other keys), "demoted" (the
        # eviction end). Each ask is recorded with its depth and that move,
        # for the connector to attribute to the load that made it.
        self.moved: dict[_offload_base.OffloadKey, tuple[str, float]] = {}
        self.hit_records: list[tuple[_offload_base.OffloadKey, int, str, float]] = []
        # Which siblings a use of their boundary refreshes; None, all of them.
        # Park mode passes whether a thread still needs the key.
        self.sibling_filter: typing.Callable[[_offload_base.OffloadKey], bool] | None = None
        # Evictable passed keys, in LRU order; `evictable_blocks` keeps the
        # rest. A key is in one at most.
        self.passed_blocks: collections.OrderedDict[_offload_base.OffloadKey, None] = collections.OrderedDict()
        # Which keys are passed, and which are a context's current
        # boundaries, whether or not they are stored right now: the
        # classification can precede the spill. The current registry serves
        # the audit only — a miss on a current key is the one that costs a
        # context. Entries for stored keys are kept as long as the key is;
        # entries for keys that are not stored are bounded, oldest first.
        self._passed_keys: collections.OrderedDict[_offload_base.OffloadKey, None] = collections.OrderedDict()
        self._current_keys: collections.OrderedDict[_offload_base.OffloadKey, None] = collections.OrderedDict()
        self._registry_limit = max(4 * cache_capacity, 1024)
        self.evicted_passed: int = 0
        self.evicted_current: int = 0

    def _class_of(self, key: _offload_base.OffloadKey) -> "collections.OrderedDict[_offload_base.OffloadKey, None]":
        """The eviction order a key belongs in, by its classification."""
        return self.passed_blocks if key in self._passed_keys else self.evictable_blocks

    def _reclassify(
        self,
        keys: typing.Iterable[_offload_base.OffloadKey],
        registry: "collections.OrderedDict[_offload_base.OffloadKey, None]",
        other: "collections.OrderedDict[_offload_base.OffloadKey, None]",
    ) -> None:
        """Move `keys` into `registry`, out of `other`, and put whichever of
        them the tier holds into the matching eviction order, newest end."""
        for key in keys:
            if key in registry:
                registry.move_to_end(key)
                continue
            other.pop(key, None)
            registry[key] = None
            for order in (self.evictable_blocks, self.passed_blocks):
                if key in order:
                    del order[key]
                    self._class_of(key)[key] = None
                    break
        if len(registry) > self._registry_limit:
            self._prune(registry)

    def _prune(self, registry: "collections.OrderedDict[_offload_base.OffloadKey, None]") -> None:
        """Forget the oldest classifications of keys that are not stored,
        down to the bound. Stored keys keep theirs."""
        excess = len(registry) - self._registry_limit
        stale = []
        for key in registry:
            if excess <= 0:
                break
            if key not in self.blocks:
                stale.append(key)
                excess -= 1
        for key in stale:
            del registry[key]

    def mark_passed(self, keys: typing.Iterable[_offload_base.OffloadKey]) -> None:
        """Classify keys as passed checkpoints: evicted first."""
        self._reclassify(keys, self._passed_keys, self._current_keys)

    def mark_current(self, keys: typing.Iterable[_offload_base.OffloadKey]) -> None:
        """Classify keys as a context's current boundaries: ordinary LRU
        keys, and the ones the audit counts misses on."""
        self._reclassify(keys, self._current_keys, self._passed_keys)

    def is_passed(self, key: _offload_base.OffloadKey) -> bool:
        """Whether the connector has classified this key as a passed checkpoint."""
        return key in self._passed_keys

    def is_current(self, key: _offload_base.OffloadKey) -> bool:
        """Whether the connector has classified this key as a current boundary."""
        return key in self._current_keys

    def _refresh_siblings(self, key: _offload_base.OffloadKey) -> None:
        """A use of a boundary's key is a use of the boundary: the stored
        keys with the same hash in the other groups move to the newest end
        of the main order, after the key itself, so a checkpoint outlives
        the attention it belongs with. Only those ``sibling_filter`` passes,
        when set: a demoted key stays where it was put."""
        block_hash = _offload_base.get_offload_block_hash(key)
        group_idx = _offload_base.get_offload_group_idx(key)
        for other in range(MAX_KV_GROUPS):
            if other == group_idx:
                continue
            sibling = _offload_base.make_offload_key(block_hash, other)
            if sibling in self.evictable_blocks and (self.sibling_filter is None or self.sibling_filter(sibling)):
                self.evictable_blocks.move_to_end(sibling)
                self.moved[sibling] = ("sibling", time.monotonic())

    def demote(self, keys: typing.Iterable[_offload_base.OffloadKey]) -> None:
        """Move each evictable key to the eviction end of its class."""
        for key in keys:
            order = self._class_of(key)
            if key in order:
                order.move_to_end(key, last=False)
                self.moved[key] = ("demoted", time.monotonic())

    def insert(self, key: _offload_base.OffloadKey, block: typing.Any) -> None:
        """Take a newly stored block into its class, not the single order
        stock LRU keeps — the key may have been classified before it landed."""
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self._class_of(key)[key] = None
            self.moved[key] = ("landed", time.monotonic())
            self._refresh_siblings(key)

    def remove(self, key: _offload_base.OffloadKey) -> None:
        """Drop a key from the tier, whichever class held it."""
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)
        self.passed_blocks.pop(key, None)
        self.moved.pop(key, None)

    def touch(self, keys: typing.Iterable[_offload_base.OffloadKey], req_context: typing.Any) -> None:
        """Record a use: newest end OF THE KEY'S OWN CLASS, so a use never
        promotes a passed checkpoint past the attention it serves; and a
        use of the boundary's other keys with it."""
        for key in reversed(list(keys)):
            order = self._class_of(key)
            if key in order:
                order.move_to_end(key)
                self.moved[key] = ("touched", time.monotonic())
                self._refresh_siblings(key)

    def clear(self) -> None:
        """Empty the tier, both eviction orders and both registries."""
        super().clear()
        self.passed_blocks.clear()
        self._passed_keys.clear()
        self._current_keys.clear()

    def evict(
        self, n: int, protected: set[_offload_base.OffloadKey]
    ) -> list[tuple[_offload_base.OffloadKey, typing.Any]] | None:
        """Take `n` keys to evict, or None if that many cannot be freed:
        passed checkpoints first, then the main order, oldest first."""
        if n == 0:
            return []
        candidates: list[tuple[_offload_base.OffloadKey, typing.Any]] = []
        from_passed = 0
        for order in (self.passed_blocks, self.evictable_blocks):
            for key in order:
                if key in protected:
                    continue
                block = self.blocks[key]
                assert block.ref_cnt == 0
                candidates.append((key, block))
                if order is self.passed_blocks:
                    from_passed += 1
                if len(candidates) == n:
                    break
            if len(candidates) == n:
                break
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            self.evictable_blocks.pop(key, None)
            self.passed_blocks.pop(key, None)
            self.moved.pop(key, None)
            del self.blocks[key]
        self.evicted_passed += from_passed
        self.evicted_current += len(candidates) - from_passed
        return candidates

    def mark_evictable(self, key: _offload_base.OffloadKey) -> None:
        """A store or load has completed: the key is free to be evicted, at
        the newest end of its class, and its boundary's other keys with it."""
        self._class_of(key)[key] = None
        self.moved[key] = ("landed", time.monotonic())
        self._refresh_siblings(key)

    def mark_non_evictable(self, key: _offload_base.OffloadKey) -> None:
        """Record the key's depth: the manager calls this exactly when a
        stored block is asked back (prepare_load), while the key still sits
        at its true position. Load completion would be too late — the block
        is re-marked evictable at the recent end there. A passed key's depth
        counts every main-order key ahead of it, since all of them survive
        it."""
        order = self._class_of(key)
        base = len(self.evictable_blocks) if order is self.passed_blocks else 0
        for depth, candidate in enumerate(reversed(order)):
            if candidate == key:
                self.hit_depths.append(base + depth)
                move, at = self.moved.get(key, ("unknown", time.monotonic()))
                self.hit_records.append((key, base + depth, move, time.monotonic() - at))
                break
        del order[key]

    def drain_hit_depths(self) -> list[int]:
        """Return the depths recorded since the last drain, clearing them."""
        depths, self.hit_depths = self.hit_depths, []
        return depths

    def drain_hit_records(self) -> list[tuple[_offload_base.OffloadKey, int, str, float]]:
        """Return the asks recorded since the last drain (key, depth, what
        last moved the key, seconds since), clearing them."""
        records, self.hit_records = self.hit_records, []
        return records


# Registered at import: the module is loaded by vLLM's connector factory before
# any connector — and so any manager — is constructed. Registration is
# idempotent here because the factory rejects a repeated name, and a module can
# be imported twice (connector factory, then a test importing it directly).
if VICTIM_POLICY not in _policy_factory.CachePolicyFactory._registry:
    _policy_factory.CachePolicyFactory.register_cache_policy(VICTIM_POLICY, __name__, VictimLRUPolicy.__name__)


class _LookupAudit:
    """Manager proxy that classifies every host-tier lookup miss.

    Three misses are indistinguishable to the engine and mean opposite things:
    a key never stored (the spill path derives a different key than the lookup
    path), a key stored and then evicted by the host policy (capacity, or the
    demote putting it at the front of the eviction line too early), and a key
    stored and then dropped on validation (the pool reused the block mid-copy).
    Only the manager sees all three, so the classification wraps it. Every
    other call passes straight through.
    """

    def __init__(self, manager: typing.Any, gpu_pool: typing.Callable[[], typing.Any]) -> None:
        self._manager = manager
        # Late-bound: the pool arrives through bind_gpu_block_pool after
        # construction, and a miss on a key the pool still holds is a
        # different fault from one on a key the tier lost.
        self._gpu_pool = gpu_pool
        self.stored: set[_offload_base.OffloadKey] = set()
        self.evicted: set[_offload_base.OffloadKey] = set()
        self.dropped: set[_offload_base.OffloadKey] = set()
        self.counts: collections.Counter[str] = collections.Counter()
        # A few raw keys per group from each side of the mismatch, for the
        # log: what the lookup asks for against what the spill stored. Read
        # side by side they say whether the hash or the group index differs.
        self.sample_asked: dict[int, list[str]] = collections.defaultdict(list)
        self.sample_stored: dict[int, list[str]] = collections.defaultdict(list)

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._manager, name)

    def lookup(self, key: _offload_base.OffloadKey, req_context: typing.Any) -> typing.Any:
        """Forward the lookup, counting the outcome by cause and by group.

        A miss is only actionable once it says WHY: never stored, stored and
        evicted, dropped as a raced copy, or stored under another group's
        index. Misses on current keys are counted apart, since those are
        the ones that cost a whole context.
        """
        result = self._manager.lookup(key, req_context)
        if result is _offload_base.LookupResult.HIT:
            outcome = "hit"
        elif result is _offload_base.LookupResult.HIT_PENDING:
            outcome = "hit_pending"
        elif result is _offload_base.LookupResult.MISS:
            if key in self.dropped:
                outcome = "miss_dropped"
            elif key in self.evicted:
                outcome = "miss_evicted"
            elif key in self.stored:
                outcome = "miss_stored_absent"
            else:
                outcome = "miss_never_stored"
                gid = _offload_base.get_offload_group_idx(key)
                raw = _offload_base.get_offload_block_hash(key)
                if len(self.sample_asked[gid]) < 6:
                    self.sample_asked[gid].append(raw[:6].hex())
                # Same hash stored under another group's index: a numbering
                # mismatch rather than a hashing one.
                if any(
                    _offload_base.make_offload_key(raw, other) in self.stored
                    for other in range(MAX_KV_GROUPS)
                    if other != gid
                ):
                    self.counts[f"g{gid}_miss_hash_stored_elsewhere"] += 1
            if self._gpu_resident(key):
                self.counts["miss_gpu_resident"] += 1
                self.counts[f"g{_offload_base.get_offload_group_idx(key)}_miss_gpu_resident"] += 1
            # A miss on a key the connector meant to keep is the one that
            # costs a context; a miss on a passed key is the walk going by.
            is_current = getattr(getattr(self._manager, "_policy", None), "is_current", None)
            if is_current is not None and is_current(key):
                self.counts[f"g{_offload_base.get_offload_group_idx(key)}_current_{outcome}"] += 1
        elif result is _offload_base.LookupResult.RETRY:
            outcome = "retry"
        else:
            outcome = "other"
        self.counts[outcome] += 1
        # Also by KV cache group, read off the key's own tail: on a hybrid
        # model one group's chunk spans a hundred times the tokens of the
        # other's, and a host LRU that evicts by count treats them alike.
        # Whether the misses sit in the large-chunk group is the question.
        self.counts[f"g{_offload_base.get_offload_group_idx(key)}_{outcome}"] += 1
        return result

    def _gpu_resident(self, key: _offload_base.OffloadKey) -> bool:
        """Whether the GPU prefix cache still holds this key's block."""
        pool = self._gpu_pool()
        if pool is None:
            return False
        gid = _offload_base.get_offload_group_idx(key)
        bh = _kv_cache_utils.make_block_hash_with_group_id(
            _kv_cache_utils.BlockHash(_offload_base.get_offload_block_hash(key)), gid
        )
        return pool.cached_block_hash_to_block.get_one_block(bh) is not None

    def prepare_store(self, keys: typing.Any, req_context: typing.Any, source: str = "engine") -> typing.Any:
        """Forward the offer, recording what was offered, newly stored and
        evicted to make room — the other half of the lookup classifier.

        ``source`` names who offered: this connector's own store paths say
        so, and an offer that arrives without saying is the engine's own
        boundary-state store, which reaches the manager through the stock
        scheduler. Counted apart because the two store the same keys, and
        which of them is running is a setting that has to be visible in the
        counters rather than inferred from a hit rate."""
        out = self._manager.prepare_store(keys, req_context)
        self.counts["store_offered"] += len(keys)
        self.counts[f"store_offered_{source}"] += len(keys)
        # Offers and new stores by group: whether one group's chunks reach the
        # tier only through finish-holds, never through the head-spill, is
        # visible here and nowhere else.
        for key in keys:
            self.counts[f"g{_offload_base.get_offload_group_idx(key)}_store_offered"] += 1
        if out is None:
            self.counts["store_refused"] += 1
        if out is not None:
            self.counts["store_new"] += len(out.keys_to_store)
            self.counts[f"store_new_{source}"] += len(out.keys_to_store)
            self.counts["store_evicted"] += len(out.evicted_keys)
            for key in out.keys_to_store:
                gid = _offload_base.get_offload_group_idx(key)
                self.counts[f"g{gid}_store_new"] += 1
                if len(self.sample_stored[gid]) < 6:
                    self.sample_stored[gid].append(_offload_base.get_offload_block_hash(key)[:6].hex())
            for key in out.evicted_keys:
                self.counts[f"g{_offload_base.get_offload_group_idx(key)}_store_evicted"] += 1
            # A re-store clears the key's earlier fate: it is back in the tier.
            self.stored.update(out.keys_to_store)
            self.evicted.difference_update(out.keys_to_store)
            self.dropped.difference_update(out.keys_to_store)
            self.evicted.update(out.evicted_keys)
        return out

    def note_dropped(self, keys: typing.Iterable[_offload_base.OffloadKey]) -> None:
        """Record keys the validate path removed after a mid-copy reuse."""
        self.dropped.update(keys)


class _Trace:
    """Append-only JSON-lines record of what the scheduler side observes and
    does: one object per event, each with its wall time and kind.

    Off when built without a path. A path that cannot be opened, or a write
    that fails, turns it off with a warning rather than touching the engine.
    Buffered, flushed at most once a second, so a run killed mid-way still
    leaves everything but its last second.
    """

    def __init__(self, path: str | None) -> None:
        """Open `path` for appending, or stay off when it is None."""
        self.path = path
        self.disabled_reason: str | None = None
        self._file: typing.IO[str] | None = None
        self._last_flush = 0.0
        if path is None:
            return
        try:
            self._file = open(path, "a", encoding="utf-8")
        except OSError as exc:
            self.disabled_reason = str(exc)
            logger.warning("victim offload: trace off, cannot open %s: %s", path, exc)

    @property
    def enabled(self) -> bool:
        """Whether writes reach a file."""
        return self._file is not None

    def write(self, ev: str, **fields: typing.Any) -> None:
        """Append one event."""
        if self._file is None:
            return
        try:
            record = {"t": round(time.time(), 3), "ev": ev, **fields}
            self._file.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            now = time.monotonic()
            if now - self._last_flush >= 1.0:
                self._file.flush()
                self._last_flush = now
        except (OSError, TypeError, ValueError) as exc:
            self.disabled_reason = str(exc)
            logger.warning("victim offload: trace off after a failed write: %s", exc)
            self.close()

    def flush(self) -> None:
        """Push buffered events to the file."""
        if self._file is not None:
            try:
                self._file.flush()
            except OSError:
                self.close()

    def close(self) -> None:
        """Flush and close; further writes are dropped."""
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        except OSError:
            pass
        self._file = None


@dataclasses.dataclass
class HeldStoreMetadata(_offload_common.OffloadingConnectorMetadata):
    """The step's stores, saying which of them the engine is holding blocks for.

    The worker has to report a held store's request back through
    ``finished_sending`` or the engine never frees it, and it cannot tell a
    held store from any other by looking: the engine's own boundary-state
    stores carry a live request's id exactly as this connector's finish-time
    stores carry a finishing one's. So the scheduler, which knows, says.
    """

    held_job_ids: frozenset[int] = frozenset()


class VictimOffloadingScheduler(_offload_sched.OffloadingConnectorScheduler):
    """Scheduler half: spills on imminent eviction instead of on computation.

    Two departures from the stock scheduler (``kv_offload.md``, Admission):
    the store trigger moves from "block was computed" to "block is near the
    head of the GPU free queue", and a block loaded back to GPU is not
    re-promoted on the CPU side.
    """

    def __init__(
        self,
        spec: _offload_base.OffloadingSpec,
        vllm_config: "vllm.config.VllmConfig",
        kv_cache_config: "vllm.v1.kv_cache_interface.KVCacheConfig",
    ) -> None:
        """Wrap the stock scheduler, adding the victim-spill state."""
        super().__init__(spec, vllm_config, kv_cache_config)
        # Handed over by bind_gpu_block_pool once the engine's scheduler is
        # built; until then there is nothing to spill from.
        self._gpu_block_pool: "vllm.v1.core.block_pool.BlockPool | None" = None
        # What each in-flight store believed it was copying: block id and the
        # hash the block carried when the job was issued. Checked on completion,
        # because nothing stops the pool reusing an unreferenced block meanwhile.
        self._spilled: dict[int, list[tuple[int, typing.Any, _offload_base.OffloadKey]]] = {}
        # Job ids owned by the victim path, so their completions can be handled
        # here rather than by the base class's request-scoped handler.
        self._victim_job_ids: set[int] = set()
        # Park mode: the GPU blocks each band store reads, held from issue
        # until the copy lands (``_release_spill_hold``). A band store in park
        # mode copies a running request's newest checkpoint and decode tail,
        # idle blocks in the free queue; at a tight admission watermark the
        # queue is a few dozen blocks, reused within a step or two, and an
        # unheld copy loses the race (59% dropped on the RTX at 0.03).
        self._spill_holds: dict[int, list[int]] = {}
        # Store jobs created at request finish, waiting for the next
        # build_connector_meta to carry them to the worker. Their blocks stay
        # allocated until the worker reports the request sent, which is what
        # makes these stores safe where head-spills are not: at full occupancy
        # a freed block is reused within a step, before any copy can land.
        self._pending_finish_jobs: dict[int, _offload_common.TransferJob] = {}
        # Finish-stores whose BLOCKS the engine is holding, awaiting dispatch.
        # Not every pending finish job is one: a finishing request's shared
        # blocks and checkpoints are stored under the victim sentinel, which
        # the engine knows nothing about and must never be reported as sent.
        self._held_finish_jobs: set[int] = set()
        # Those of them the step being built dispatches, named for the worker.
        self._held_this_step: frozenset[int] = frozenset()
        # Monotonic deadline until which the system counts as pressured; moved
        # forward every step that observes the watermark crossed.
        self._pressure_until: float = 0.0
        # Ring of recently issued transfers, kept only to give the crash
        # handler below something to say: block id -> what last touched it.
        self._recent_block_roles: dict[int, str] = {}
        extra = spec.extra_config
        self._spill_watermark: float = float(extra.get("spill_watermark", DEFAULT_SPILL_WATERMARK))
        self._max_spill_blocks: int = int(extra.get("max_spill_blocks", DEFAULT_MAX_SPILL_BLOCKS))
        self._spill_band_blocks: int = int(extra.get("spill_band_blocks", DEFAULT_SPILL_BAND_BLOCKS))
        # Park mode (with vLLM's prefix_cache_context_threads and the decode
        # tail): the GPU holds running requests only. A finishing request's
        # thread -- its blocks up to its decode tail -- is stored whole while
        # the engine holds them, and uncached on the GPU once the copy has
        # landed; a starting request loads its thread back, and the host
        # copy is demoted. No band, no checkpoint classes. On whenever the
        # engine names each request's thread and keeps decode tails, the two
        # things it is built on; ``park: False`` keeps the band.
        cache_config = vllm_config.cache_config
        park_default = bool(
            getattr(cache_config, "prefix_cache_context_threads", False)
            and getattr(cache_config, "prefix_cache_decode_tail", False)
        )
        self._park: bool = bool(extra.get("park", park_default))
        # Parking on the GPU: threads stay pinned there until the pool needs
        # the room, measured against the engine's own admission watermark.
        self._gpu_park: bool = _gpu_park_from(extra, self._park)
        self._gpu_park_reserve: float = float(extra.get("gpu_park_reserve", DEFAULT_GPU_PARK_RESERVE))
        self._admission_watermark: float = float(getattr(vllm_config.scheduler_config, "watermark", 0.0) or 0.0)
        # Requests looked up and not yet scheduled, by id: the GPU blocks their
        # admission takes, which the spill step makes room for.
        self._park_needs: dict[str, int] = {}
        self._kv_share_at: float = 0.0
        # Parking on the GPU: each running request's newest decode tail, held
        # on the GPU (block id -> the hash it was pinned under) in place of a
        # copy to the tier (`_pin_tails`).
        self._tail_pins: dict[str, dict[int, typing.Any]] = {}
        # Per context: its thread's host keys, the boundary it resumes at,
        # the request that parked it, and whether it is parked or running.
        self._parked: dict[str, dict[str, typing.Any]] = {}
        # The contexts whose thread holds a host key: more than one for a
        # common prompt head. A key a parked thread holds is never demoted.
        self._park_owner: dict[_offload_base.OffloadKey, set[str]] = {}
        # Park stores in flight: job id -> what to release and uncache when
        # it lands; and GPU blocks to uncache at the next step.
        self._park_jobs: dict[int, dict[str, typing.Any]] = {}
        self._park_retire: list[tuple[int, typing.Any]] = []
        self._park_started: set[str] = set()
        # Requests whose lookup waits for their thread's park copy to land,
        # by id, with when the wait began.
        self._park_deferred: dict[str, float] = {}
        # Requests running now, by id, with their context.
        self._park_active: dict[str, str] = {}
        self._park_snapshot_at: float = 0.0
        # Per request: the host keys of its newest decode tail, copied to the
        # tier as the engine keys it (``_keep_tails``), and this step's tail
        # blocks, stored ahead of the rest.
        self._tail_keys: dict[str, list[_offload_base.OffloadKey]] = {}
        self._tail_step_keys: set[_offload_base.OffloadKey] = set()
        self._park_counts: collections.Counter[str] = collections.Counter()
        self._park_log: typing.TextIO | None = None
        self._victim_req_context = _offload_base.ReqContext(req_id=VICTIM_REQ_ID)
        # A victim's offload key is derived from the GPU block's own hash, which
        # matches what the request path computes only when one GPU block is one
        # offloaded block. Above factor 1 the request path takes every Nth hash
        # and the keys would differ, so every store would be unreachable.
        if self.config.blocks_per_chunk != 1:
            raise ValueError(
                "victim offload requires one GPU block per offloaded block "
                f"(blocks_per_chunk={self.config.blocks_per_chunk}); unset "
                "kv_connector_extra_config['block_size'] to get it"
            )
        _refuse_unsupported_groups(self.config.kv_group_configs)
        self._spill_disabled_warned: bool = False
        # Every manager call goes through the audit; it forwards what it does
        # not classify, so the policy lookup below still reaches the manager.
        self._lookup_audit = _LookupAudit(self.manager, lambda: self._gpu_block_pool)
        self.manager = typing.cast(typing.Any, self._lookup_audit)
        # The host policy's demote, when the tier was configured with one. A
        # tier under stock LRU still works, it just drifts inclusive under
        # churn — said once rather than silently.
        policy = getattr(self.manager, "_policy", None)
        self._demote: typing.Callable[[typing.Iterable[_offload_base.OffloadKey]], None] | None = getattr(
            policy, "demote", None
        )
        # Same policy's reuse-depth recorder; None under a stock LRU tier.
        self._drain_hit_depths: typing.Callable[[], list[int]] | None = getattr(policy, "drain_hit_depths", None)
        # Each ask with what last moved its key: attributed to the resume that
        # made it in the park log (``_park_loading``), dropped at the tally
        # otherwise.
        self._drain_hit_records: typing.Callable[[], list[typing.Any]] | None = getattr(
            policy, "drain_hit_records", None
        )
        self._hit_depth_policy = policy
        # Park mode refreshes only the siblings a thread still needs
        # (DEFAULT_SIBLING_REFRESH), so dead checkpoints stay demoted.
        sibling_refresh = extra.get("sibling_refresh", DEFAULT_SIBLING_REFRESH)
        if sibling_refresh not in SIBLING_REFRESH_MODES:
            raise ValueError(f"sibling_refresh must be one of {SIBLING_REFRESH_MODES}, got {sibling_refresh!r}")
        if self._park and sibling_refresh == "needed" and isinstance(policy, VictimLRUPolicy):
            policy.sibling_filter = self._park_needed
        # Same policy's checkpoint classes; None under a stock LRU tier, which
        # then keeps every checkpoint at full price.
        self._retention = policy if hasattr(policy, "mark_current") else None
        if not bool(extra.get("demote_on_use", DEFAULT_DEMOTE_ON_USE)):
            self._demote = None
        self._finish_patterns_logged: int = 0
        # Deepest ask seen over the whole run: with per-interval percentiles
        # this is the one number the intervals cannot reconstruct.
        self._hit_depth_run_max: int = 0
        if self._demote is None and bool(extra.get("demote_on_use", DEFAULT_DEMOTE_ON_USE)):
            logger.warning(
                "victim offload: host tier policy %s has no demote; set eviction_policy=%r",
                type(policy).__name__,
                VICTIM_POLICY,
            )
        # Running totals, reported on a timer: blocks whose copy was issued,
        # and blocks whose copy landed after the pool had already reused them.
        self._blocks_spilled: int = 0
        self._blocks_dropped: int = 0
        self._jobs_with_drops: int = 0
        self._blocks_held: int = 0
        self._passed_skipped: int = 0
        # Band victims the engine marked dead (vLLM's KVCacheBlock.dead), and
        # their keys, newest last and bounded, so a departure of one reads as
        # "dead" rather than a lost race. A key leaves the set when a live
        # block carries it again.
        self._dead_skipped: int = 0
        self._dead_keys: collections.OrderedDict[_offload_base.OffloadKey, None] = collections.OrderedDict()
        self._demote_dead: typing.Callable[[typing.Iterable[_offload_base.OffloadKey]], None] | None = getattr(
            policy, "demote", None
        )
        self._band_present: int = 0
        self._shared_stored: int = 0
        self._checkpoints_stored: int = 0
        # Checkpoint blocks of finished requests, by hash with group, to move
        # to the tail of the GPU's free queue at the next step.
        self._reage_pending: list[typing.Any] = []
        self._reaged: int = 0
        # Gauges from the last step's keep-young pass: checkpoints of known
        # requests moved to the tail, and known requests whose newest
        # checkpoint is not on the GPU at all — the continuation that will
        # miss. And, cumulative, the finishes that found none.
        self._kept_last_step: int = 0
        self._kept_missing_last_step: int = 0
        self._checkpoints_gone: int = 0
        # Step counter and the step each block was last re-aged at, for the
        # departure accounting to say whether re-aged blocks leave anyway.
        self._step: int = 0
        self._reaged_at: dict[int, int] = {}
        # The free queue as of the previous step, block id -> (hash, position):
        # a block gone from it since is a departure, and where it stood and
        # whether the host had it says whether the departure cost anything.
        self._queue_snapshot: dict[int, tuple[typing.Any, int]] = {}
        self._departures: collections.Counter[str] = collections.Counter()
        # Zero so the first completion writes a line at once: the first minute
        # of a cliff is the one the tally exists for.
        self._last_tally_at: float = 0.0
        # Divergent-hit fills accepted and not yet allocated: request id ->
        # (boundary in tokens, lagging group ids, the tier keys to load).
        self._fills: dict[str, tuple[int, tuple[int, ...], list[_offload_base.OffloadKey]]] = {}
        self._fills_loaded: int = 0
        self._fill_mismatch_warned: bool = False
        # The scheduler-side trace, in the run's job directory when the
        # engine was given one (ullm.factory.BackendFactory.job_dir).
        job_dir = extra.get("job_dir")
        self._trace = _Trace(os.path.join(str(job_dir), TRACE_FILENAME) if job_dir else None)
        if self._park and job_dir and bool(extra.get("park_log", DEFAULT_PARK_LOG)):
            self._park_log = open(os.path.join(str(job_dir), PARK_LOG_FILENAME), "a", buffering=1)
        self._engine_log: bool = bool(extra.get("engine_log", DEFAULT_ENGINE_LOG))
        # The connector's slot arithmetic, set by the connector that builds
        # this scheduler and carried in the trace's start event.
        self._sizing: dict[str, typing.Any] | None = None
        self._trace_started: bool = False
        # Lookups awaiting their schedule, so one line carries both what
        # was asked and what the scheduler settled on.
        self._trace_lookups: dict[str, dict[str, typing.Any]] = {}
        # Recent finishes by the chain hash of their newest full chunk, so a
        # continuation's lookup can be linked to the turn before it.
        self._finished_turns: collections.OrderedDict[bytes, dict[str, typing.Any]] = collections.OrderedDict()
        # The blocks the keep-young pass moved, this step and the step
        # before: a departure of one is the pass failing to hold it.
        self._kept_map: dict[int, tuple[str, int, int]] = {}
        self._kept_map_prev: dict[int, tuple[str, int, int]] = {}
        # The blocks the keep-young pass moved this step, offered to the
        # tier ahead of the band: a running request's newest checkpoint is
        # what the fill needs, and on a pool that drains its queue within a
        # step the band would never see it before it left.
        self._kept_blocks: list[typing.Any] = []
        self._kept_stored: int = 0
        # When each hashed block was first seen in the free queue.
        self._queue_entered: dict[int, float] = {}
        self._last_sample_at: float = 0.0
        self._sample_counters: collections.Counter[str] = collections.Counter()
        if self._trace.enabled:
            if self._engine_log:
                logger.info("victim offload: trace at %s", self._trace.path)
        elif job_dir:
            logger.warning("victim offload: trace off: %s", self._trace.disabled_reason)
        elif self._engine_log:
            logger.info("victim offload: trace off (no job_dir in the connector config)")

    def get_num_new_matched_tokens(self, request: typing.Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        """Stock lookup, with the outcome tallied per call: how many lookups
        yield nothing is the number the engine's ext_hit averages away."""
        # The chain is known before the turn runs, so the boundaries the
        # turn will move past are classified now: the band then never copies
        # them. The finish reclassifies once the last boundary has moved.
        if not self._park:
            self._classify_checkpoints(request, None)
        # A fill accepted at an earlier lookup of this request stands only for
        # the allocation that follows it; when the scheduler could not
        # allocate that step it looks up again, and the fork asks again if
        # the hit is still divergent. Anything older is stale.
        self._fills.pop(request.request_id, None)
        if self._park and self._park_in_flight(request):
            return None, False
        ext, load_async = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if self._park:
            self._park_start(request, num_computed_tokens, ext)
        # The blocks of a local hit are on the GPU from here on, so their
        # host copies are worth what a loaded block's copy is: nothing until
        # the GPU gives the block up again. Same treatment.
        if num_computed_tokens > 0 and self._demote is not None:
            self._demote(self._park_unneeded(self._local_hit_keys(request.request_id, num_computed_tokens)))
        c = self._lookup_audit.counts
        c["lookup_calls"] += 1
        c["lookup_local_tokens"] += num_computed_tokens
        zero: dict[str, typing.Any] | None = None
        if ext is None:
            c["lookup_deferred"] += 1
        elif ext == 0:
            c["lookup_zero"] += 1
            zero = self._classify_zero_lookup(request, num_computed_tokens)
        else:
            c["lookup_ext_tokens"] += ext
            c["lookup_served"] += 1
        if self._trace.enabled:
            self._trace_lookup(request, num_computed_tokens, ext, zero)
        return ext, load_async

    def _classify_zero_lookup(self, request: typing.Any, num_computed_tokens: int) -> dict[str, typing.Any] | None:
        """Say why a lookup yielded nothing, since the zero itself does not.

        Three kinds: nothing askable (the GPU's hit already covers every
        full chunk — benign), a walk that failed at the chain's last full
        chunk (the one the previous turn never finished computing, which no
        cache can hold — benign), and a walk that failed earlier, bucketed
        by where it started. A start past chunk 0 is also checked for the
        DIVERGENT hit: the GPU holds the attention prefix but not the
        recurrent state at its boundary, which the protocol cannot serve —
        and whether the tier holds that state, which is the failure by name.
        Returns the kind and the two flags, for the trace.
        """
        c = self._lookup_audit.counts
        configs = self.config.kv_group_configs
        if not configs:
            return None
        tokens_per_chunk = configs[0].tokens_per_chunk
        hashes = list(request.block_hashes)
        n_full = len(hashes[configs[0].hashes_per_chunk - 1 :: configs[0].hashes_per_chunk])
        start = num_computed_tokens // tokens_per_chunk
        if n_full - start < 1:
            kind = "nothing_askable"
        elif start == n_full - 1:
            kind = "at_straddle"
        else:
            kind = "walk_failed"
            bucket = (
                "0" if start == 0 else "1" if start == 1 else "2-4" if start <= 4 else "5-15" if start <= 15 else "16+"
            )
            c[f"lookup_walk_failed_start_{bucket}"] += 1
        c[f"lookup_zero_{kind}"] += 1
        info: dict[str, typing.Any] = {"kind": kind, "divergent": False, "tier_ckpt": False}
        # The divergence check covers every start past chunk 0 — the
        # "nothing askable" case above all, since an attention prefix the
        # GPU holds in full with no recurrent state at its end is exactly
        # the hit the scheduler reconciles away.
        pool = self._gpu_block_pool
        if start == 0 or pool is None:
            return info
        policy = getattr(self.manager, "_policy", None)
        for group_idx, group_config in enumerate(configs):
            if not group_config.hashes_per_chunk:
                continue  # a transient group the tier passes over
            if group_config.sliding_window_size_in_chunks is None:
                continue
            group_hashes = hashes[group_config.hashes_per_chunk - 1 :: group_config.hashes_per_chunk]
            if start - 1 >= len(group_hashes):
                continue
            boundary_hash = group_hashes[start - 1]
            with_group = _kv_cache_utils.make_block_hash_with_group_id(
                _kv_cache_utils.BlockHash(boundary_hash), group_idx
            )
            if pool.cached_block_hash_to_block.get_one_block(with_group) is not None:
                continue
            c["lookup_divergent"] += 1
            c[f"lookup_divergent_{kind}"] += 1
            info["divergent"] = True
            key = _offload_base.make_offload_key(boundary_hash, group_idx)
            if policy is not None and policy.get(key) is not None:
                c["lookup_divergent_tier_has_ckpt"] += 1
                info["tier_ckpt"] = True
            break
        return info

    def _local_hit_keys(self, req_id: str, num_computed_tokens: int) -> list[_offload_base.OffloadKey]:
        """Offload keys of the chunks a request's GPU prefix hit covers."""
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return []
        keys: list[_offload_base.OffloadKey] = []
        for group_config, group_state in zip(self.config.kv_group_configs, req_status.group_states):
            if not group_config.hashes_per_chunk:
                continue  # a transient group the tier passes over
            keys.extend(group_state.offload_keys[: num_computed_tokens // group_config.tokens_per_chunk])
        return keys

    def bind_gpu_block_pool(self, gpu_block_pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Take the GPU pool the engine offers every connector.

        The stock offloading connector ignores this; the free queue it exposes
        is the victim set this connector spills from.
        """
        self._gpu_block_pool = gpu_block_pool

    def _touch(self, req_status: _offload_sched.RequestOffloadState) -> None:
        """Leave CPU-side LRU order alone on a lookup.

        The stock scheduler promotes a block to the CPU tier's most-recent end
        from the lookup path — that is, exactly as the block is being loaded
        onto the GPU, after which it would hold top-of-stack space in both
        tiers. Skipping the promotion lets a GPU-resident block age out of the
        CPU tier on its own. Ordering only: a block being loaded is protected
        from eviction by ``prepare_load``, not by this.
        """
        return

    def _offer_store(self, keys: list[_offload_base.OffloadKey], req_context: typing.Any) -> typing.Any:
        """This connector's own offer to the tier, said to be so: the audit
        counts it apart from the engine's boundary-state store, which
        reaches the same manager without saying."""
        return typing.cast(typing.Any, self.manager).prepare_store(keys, req_context, source="connector")

    def _collect_victims(self, pool: "vllm.v1.core.block_pool.BlockPool", band: int) -> list[typing.Any]:
        """The hashed blocks among the first `band` entries of the free
        queue, nearest-to-die first.

        Unhashed blocks are skipped: they sit at the very front of the queue
        precisely because they can never match in prefix caching, so a copy
        of one could never be loaded. They still count toward the band, so a
        pool whose front is fresh blocks yields nothing rather than a walk to
        the cached content behind them.
        """
        queue = pool.free_block_queue
        victims: list[typing.Any] = []
        block = queue.fake_free_list_head.next_free_block
        # Walking the linked list rather than get_all_free_blocks(), which
        # materializes the whole pool when only the front is wanted.
        seen = 0
        while block is not None and block is not queue.fake_free_list_tail and seen < band:
            seen += 1
            if block.block_hash is not None:
                victims.append(block)
            block = block.next_free_block
        return victims

    # --- park mode -----------------------------------------------------

    def _park_say(self, event: str, text: str) -> None:
        """One line of the park log: the event, its detail, and the tier's
        state at that moment."""
        if self._park_log is None:
            return
        self._park_log.write(f"{time.strftime('%H:%M:%S')} step={self._step} {event} {text} | {self._park_state()}\n")

    def _park_state(self) -> str:
        """The host tier as park mode sees it: slots held of its capacity,
        the parked threads and how much of them it still holds, the running
        threads' demoted copies, and every eviction so far by what it cost;
        and parking on the GPU, the threads parked there."""
        policy = getattr(self.manager, "_policy", None)
        held: typing.Any = getattr(policy, "blocks", {})
        parked = running = parked_slots = parked_held = running_held = 0
        gpu = pinned = 0
        for record in self._parked.values():
            if record.get("pins"):
                gpu += 1
                pinned += len(record["pins"])
                continue
            present = sum(1 for key in record["keys"] if key in held)
            if record["state"] == "parked":
                parked += 1
                parked_slots += len(record["keys"])
                parked_held += present
            else:
                running += 1
                running_held += present
        c = self._park_counts
        return (
            f"tier {len(held)}/{getattr(policy, 'cache_capacity', '?')} slots; "
            f"parked {parked} threads {parked_held}/{parked_slots} slots held; "
            f"running {running} threads {running_held} demoted slots held; "
            f"evicted parked={c['evicted_parked']} running={c['evicted_running']} other={c['evicted_other']}"
            + (self._gpu_park_state(gpu, pinned) if self._gpu_park else "")
        )

    def _gpu_park_state(self, gpu: int, pinned: int) -> str:
        """The GPU's side of the state line: threads parked there, the blocks
        they pin, the running requests' pinned tails, spills in flight, and the
        pool's free blocks."""
        pool = self._gpu_block_pool
        spilling = sum(1 for info in self._park_jobs.values() if info.get("spill"))
        free = pool.get_num_free_blocks() if pool is not None else 0
        tails = sum(len(held) for held in self._tail_pins.values())
        return (
            f"; gpu-parked {gpu} threads {pinned} blocks pinned, {tails} tail blocks pinned, "
            f"{spilling} spilling, {free} free"
        )

    def _park_evicted(self, keys: typing.Iterable[_offload_base.OffloadKey]) -> None:
        """Classify the tier's evictions by whose they were. A parked
        thread's is the loss park mode exists to prevent: said loudly."""
        for key in keys:
            owners = sorted(self._park_owner.pop(key, ()))
            parked = [context for context in owners if self._parked.get(context, {}).get("state") == "parked"]
            if parked:
                self._park_counts["evicted_parked"] += 1
                self._park_say(
                    "EVICTED-PARKED",
                    f"ctx={','.join(_short_ctx(context) for context in parked)} key={_short_key(key)} "
                    "the tier dropped a parked thread's block",
                )
                self._park_snapshot()
            elif any(context in self._parked for context in owners):
                self._park_counts["evicted_running"] += 1
            else:
                self._park_counts["evicted_other"] += 1

    def _park_needed(self, key: _offload_base.OffloadKey) -> bool:
        """Whether a parked thread holds the key, or a running one holds it
        as the recurrent state at its last full boundary."""
        for context in self._park_owner.get(key, ()):
            record = self._parked.get(context)
            if record is not None and (record["state"] == "parked" or key in record.get("checkpoints", ())):
                return True
        return False

    def _park_unneeded(self, keys: list[_offload_base.OffloadKey]) -> list[_offload_base.OffloadKey]:
        """The keys no thread needs from the tier; every key outside park
        mode. A restore walks its thread from the head, so demoting a key a
        parked thread shares with the one being demoted (a common prompt
        head) would put that restore first in line for eviction. And a
        running thread's recurrent state at its last full boundary is one
        the GPU need not keep (a state loaded from the tier is, most
        often, not there by the finish), while its next park needs it: the
        tier's copy is the one that stays."""
        if not self._park:
            return keys
        return [key for key in keys if not self._park_needed(key)]

    def _park_claim(self, context: str, keys: typing.Iterable[_offload_base.OffloadKey], held: bool) -> None:
        """Record whether the context's thread holds these keys."""
        for key in keys:
            owners = self._park_owner.setdefault(key, set())
            if held:
                owners.add(context)
            else:
                owners.discard(context)
                if not owners:
                    del self._park_owner[key]

    def _park_demote(self, keys: list[_offload_base.OffloadKey], why: str) -> None:
        """Demote host copies nobody parked needs, saying why."""
        policy = getattr(self.manager, "_policy", None)
        unneeded = self._park_unneeded(keys)
        present = [key for key in unneeded if policy is not None and policy.get(key) is not None]
        if present and self._demote_dead is not None:
            self._demote_dead(present)
        self._park_counts["demoted"] += len(present)
        self._park_counts["demote_kept_for_parked"] += len(keys) - len(unneeded)
        if keys:
            kept = f", {len(keys) - len(unneeded)} kept for threads that need them" if len(unneeded) < len(keys) else ""
            self._park_say("DEMOTE", f"{len(present)} of {len(keys)} keys held{kept}; {why}")

    def _park_snapshot(self) -> None:
        """Two lines naming every conversation by its short id: which are
        running, and what the host tier holds in eviction order (the first
        is evicted first). Each tier entry is a conversation's slots by
        state: P parked (its live thread, which must stay), D demoted (a copy
        of a running thread's), X neither (superseded, dead, or unowned)."""
        if self._park_log is None:
            return
        self._park_snapshot_at = time.time()
        running = sorted({_short_ctx(context) for context in self._park_active.values()})
        stamp = f"{time.strftime('%H:%M:%S')} step={self._step}"
        self._park_log.write(f"{stamp} SNAPSHOT running {len(running)}: {' '.join(running)}\n")
        if self._gpu_park:
            on_gpu = sorted(
                (record["parked_at"], context) for context, record in self._parked.items() if record.get("pins")
            )
            names = " ".join(f"{_short_ctx(context)}:{len(self._parked[context]['pins'])}" for _, context in on_gpu)
            self._park_log.write(f"{stamp} SNAPSHOT gpu-parked {len(on_gpu)}, spilled first: {names}\n")
        policy = getattr(self.manager, "_policy", None)
        orders = [getattr(policy, "passed_blocks", {}), getattr(policy, "evictable_blocks", {})]
        running_contexts = set(self._park_active.values())
        entries: dict[str, collections.Counter[str]] = {}
        slots = 0
        for order in orders:
            for key in list(order):
                slots += 1
                # A key several threads hold is counted once: under a parked
                # holder if there is one, since that thread needs it.
                owners = sorted(self._park_owner.get(key, ()))
                parked = [owner for owner in owners if self._parked.get(owner, {}).get("state") == "parked"]
                running_owners = [owner for owner in owners if owner in running_contexts]
                if parked:
                    state, context = "P", parked[0]
                elif running_owners:
                    state, context = "D", running_owners[0]
                else:
                    state, context = "X", owners[0] if owners else None
                name = _short_ctx(context) if context is not None else "--------"
                entries.setdefault(name, collections.Counter())[state] += 1
        rendered = " ".join(
            f"{name}:" + "+".join(f"{state}{count}" for state, count in sorted(counts.items()))
            for name, counts in entries.items()
        )
        self._park_log.write(
            f"{stamp} SNAPSHOT tier, evicted first ({len(entries)} conversations, {slots} evictable slots): {rendered}\n"
        )

    def _park_step(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Each step: demote the dead threads' keys the engine reported, and
        uncache the GPU blocks of threads whose park store has landed and
        whose request has let them go."""
        if time.time() - self._park_snapshot_at >= PARK_SNAPSHOT_S:
            self._park_snapshot()
        threads = getattr(pool, "context_threads", None)
        if threads is not None:
            dead: dict[tuple[str, str], list[_offload_base.OffloadKey]] = collections.defaultdict(list)
            for block_hash, context, cause in threads.take_dead_keys():
                key = _offload_base.make_offload_key(
                    _kv_cache_utils.get_block_hash(block_hash), _kv_cache_utils.get_group_id(block_hash)
                )
                dead[(context, cause)].append(key)
            for (context, cause), keys in dead.items():
                if cause in ("epoch", "close"):
                    record = self._parked.pop(context, None)
                    if record is not None:
                        self._park_claim(context, record["keys"], held=False)
                        keys = list(dict.fromkeys([*keys, *record["keys"]]))
                        self._park_unpin(context, record, pool, dead=True, why=f"thread dead ({cause})")
                self._park_demote(keys, f"dead thread ctx={_short_ctx(context)} cause={cause}")
        if self._park_retire:
            retired = skipped = 0
            still: list[tuple[int, typing.Any]] = []
            for block_id, block_hash in self._park_retire:
                block = pool.blocks[block_id]
                if block_hash is None or block.block_hash != block_hash:
                    continue  # uncached or reused already
                if block.ref_cnt > 0:
                    # Still referenced: by the finished request until the
                    # engine frees it, or by a running request sharing it,
                    # which keeps it. Looked at again next step.
                    still.append((block_id, block_hash))
                    skipped += 1
                    continue
                pool.retire_cached_block(block, block_hash)
                retired += 1
            self._park_retire = still
            self._park_counts["gpu_uncached"] += retired
            if retired:
                self._park_say(
                    "GPU-UNCACHE",
                    f"{retired} blocks of parked threads uncached on the GPU ({skipped} still referenced)",
                )

    def _park_in_flight(self, request: typing.Any) -> bool:
        """Whether the request's lookup waits for its thread's park copy.
        Until the copy lands the new tail is neither ready in the tier nor
        idle on the GPU (a partial-block hit needs an idle block), so a
        lookup would serve the previous park's tail, still in the tier as a
        demoted copy. Asked again each step, for up to PARK_DEFER_MAX_S."""
        context = _request_context(request)
        rid = request.request_id
        if context is None or not any(info["context"] == context for info in self._park_jobs.values()):
            self._park_deferred.pop(rid, None)
            return False
        now = time.monotonic()
        since = self._park_deferred.get(rid)
        if since is None:
            self._park_deferred[rid] = now
            self._park_counts["deferred_in_flight"] += 1
            self._park_say(
                "DEFER", f"ctx={_short_ctx(context)} req={_short_req(rid)} reason=its thread's park copy is in flight"
            )
            return True
        if now - since <= PARK_DEFER_MAX_S:
            return True
        if since > 0:
            # Said once: the lookup goes ahead from here on.
            self._park_deferred[rid] = 0.0
            self._park_counts["defer_timeout"] += 1
            self._park_say(
                "DEFER-TIMEOUT",
                f"ctx={_short_ctx(context)} req={_short_req(rid)} the park copy is still in flight after "
                f"{PARK_DEFER_MAX_S:.0f} s; looking up anyway",
            )
        return False

    def _park_start(self, request: typing.Any, num_computed_tokens: int, ext: int | None) -> None:
        """A request of a context arrives. Its thread stays parked until the
        request is scheduled (``_park_running``): a lookup can wait steps,
        for a sibling's load of the keys they share or a copy in flight, and
        until then the keys it will load are a parked thread's, which no
        sibling's load or demotion may send to the eviction end."""
        if request.request_id in self._park_started:
            return
        context = _request_context(request)
        if context is None:
            return
        self._park_started.add(request.request_id)
        self._park_active[request.request_id] = context
        if self._gpu_park:
            self._park_needs[request.request_id] = self._admission_blocks(request, num_computed_tokens)
        record = self._parked.get(context)
        state = record["state"] if record is not None else "none"
        self._park_counts["starts"] += 1
        self._park_say(
            "START",
            f"ctx={_short_ctx(context)} req={_short_req(request.request_id)} prompt={request.num_tokens} gpu_hit={num_computed_tokens} "
            f"tier_hit={ext} parked_at={record['boundary'] if record else None} was={state}",
        )

    def _park_running(self, request: typing.Any) -> None:
        """The request is scheduled: its thread runs from here on, so its host
        copy is demoted once the load (if any) has read it, and a thread parked
        on the GPU is let go of by its pins, the part only the tier holds
        touched."""
        self._park_needs.pop(request.request_id, None)
        context = self._park_active.get(request.request_id)
        record = self._parked.get(context) if context is not None else None
        if record is not None and record["state"] == "parked":
            record["state"] = "running"
            if record.get("pins") and self._gpu_block_pool is not None:
                # Allocated by now: the request holds the blocks it hit.
                assert context is not None
                self._touch_hosted(record["keys"])
                self._park_unpin(
                    context,
                    record,
                    self._gpu_block_pool,
                    dead=False,
                    why=f"req={_short_req(request.request_id)} runs on it",
                )

    def _park_loading(self, request: typing.Any, num_external_tokens: int) -> None:
        """A load job was just issued for a starting request: say what it
        reads."""
        req_status = self._req_status.get(request.request_id)
        keys = 0
        job_keys: set[typing.Any] = set()
        if req_status is not None:
            for job_id in req_status.transfer_jobs:
                job = self._jobs.get(job_id)
                if job is not None and not job.is_store:
                    keys = len(job.keys)
                    job_keys = set(job.keys)
        self._park_counts["loads"] += 1
        self._park_counts["load_slots"] += keys
        # The asks this load made: how deep each key sat, and what last moved
        # it there and how long before; beside how long the thread was parked.
        records = self._drain_hit_records() if self._drain_hit_records is not None else []
        mine = [record for record in records if record[0] in job_keys]
        context = _request_context(request)
        parked = self._parked.get(context) if context is not None else None
        parked_for = time.monotonic() - parked["parked_at"] if parked is not None and "parked_at" in parked else None
        asks = ""
        if mine:
            depths = sorted(record[1] for record in mine)
            ages = sorted(record[3] for record in mine)
            moves = collections.Counter(record[2] for record in mine)
            asks = (
                f" depth={depths[0]}/{depths[len(depths) // 2]}/{depths[-1]}"
                f" moved={','.join(f'{move}:{count}' for move, count in moves.most_common())}"
                f" since={ages[len(ages) // 2]:.1f}s"
            )
            if self._trace.enabled:
                self._trace.write(
                    "ask",
                    step=self._step,
                    n=len(mine),
                    depth=depths,
                    moved=dict(moves),
                    since=round(ages[len(ages) // 2], 2),
                    parked_for=round(parked_for, 2) if parked_for is not None else None,
                    other=len(records) - len(mine),
                )
        self._park_say(
            "LOAD",
            f"ctx={_short_ctx(context)} req={_short_req(request.request_id)} tokens={num_external_tokens} slots={keys}"
            + asks
            + (f" parked_for={parked_for:.1f}s" if parked_for is not None else "")
            + " reason=resume",
        )

    def _park_settle(self, loaded: list[_offload_base.OffloadKey]) -> None:
        """After the base settled the step's completions: release and queue
        for uncaching what each landed park store held, and demote what the
        step loaded (its thread runs on the GPU now)."""
        pool = self._gpu_block_pool
        for job_id in [job_id for job_id in self._park_jobs if job_id not in self._jobs]:
            info = self._park_jobs.pop(job_id)
            if pool is not None:
                for block_id, _ in info["held"]:
                    pool.free_blocks([pool.blocks[block_id]])
            self._park_retire.extend(info["retire"])
            self._park_counts["stored"] += 1
            self._park_counts["stored_slots"] += info["slots"]
            if info.get("spill") and info["context"] not in self._parked:
                self._park_demote(list(info["keys"]), f"ctx={_short_ctx(info['context'])} died while it spilled")
            self._park_say(
                "STORED",
                f"ctx={_short_ctx(info['context'])} req={_short_req(info['req'])} slots={info['slots']} job={job_id}",
            )
        if loaded:
            self._park_counts["loaded_slots"] += len(loaded)
            kept = len(loaded) - len(self._park_unneeded(loaded))
            self._park_say(
                "LOADED",
                f"{len(loaded)} slots landed on the GPU; their host copies demoted"
                + (f", {kept} kept for threads that need them" if kept else ""),
            )

    def _park_finish(
        self, request: typing.Any, block_id_groups: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, typing.Any] | None]:
        """Park mode at a request's finish: store its thread and hold its
        blocks until the copy is sent."""
        self._fills.pop(request.request_id, None)
        self._park_started.discard(request.request_id)
        self._park_deferred.pop(request.request_id, None)
        self._park_active.pop(request.request_id, None)
        context = _request_context(request)
        pool = self._gpu_block_pool
        req_status = self._req_status.get(request.request_id)
        held = False
        if context is None:
            self._park_counts["unparked"] += 1
            self._park_say("SKIP", f"req={_short_req(request.request_id)} reason=no context tag (not a game's thread)")
        elif pool is None or req_status is None:
            self._park_counts["unparked"] += 1
            self._park_say(
                "SKIP",
                f"ctx={_short_ctx(context)} req={_short_req(request.request_id)} reason=no pool or request state",
            )
        elif not all(self._jobs[job_id].is_store for job_id in req_status.transfer_jobs):
            self._park_counts["unparked"] += 1
            self._park_say(
                "SKIP",
                f"ctx={_short_ctx(context)} req={_short_req(request.request_id)} reason=load in flight (aborted mid-load)",
            )
        elif self._gpu_park:
            self._park_pin(request, block_id_groups, pool, context)
        else:
            held = self._park_store(request, block_id_groups, pool, req_status, context)
        self._park_needs.pop(request.request_id, None)
        if pool is not None:
            # After the park: a tail it took is pinned by the thread now.
            self._release_tail_pins(request.request_id, pool)
        # Its tail keys are the parked thread's now, if the park took them;
        # otherwise nobody's.
        self._park_demote_quiet(self._tail_keys.pop(request.request_id, None))
        finished, params = super().request_finished(request)
        return held or finished, params

    def _park_store(
        self,
        request: typing.Any,
        block_id_groups: tuple[list[int], ...],
        pool: "vllm.v1.core.block_pool.BlockPool",
        req_status: typing.Any,
        context: str,
    ) -> bool:
        """Offer the request's thread to the tier (`_park_plan`), its blocks held
        until the copy lands. True if a copy was queued."""
        plan = self._park_plan(request, block_id_groups, pool)
        return self._park_issue(plan, context, request.request_id, pool, req_status=req_status) == "queued"

    def _park_plan(
        self,
        request: typing.Any,
        block_id_groups: tuple[list[int], ...],
        pool: "vllm.v1.core.block_pool.BlockPool",
    ) -> dict[str, typing.Any]:
        """A finishing request's thread, block by block: every full attention
        chunk, each recurrent group's checkpoint at the last full boundary, and
        at the decode tail the attention block and every recurrent state, keyed
        at the tail's boundary (the attention block's KV below it is final).
        Each row is a GPU block and the tier key it goes under, group by group;
        beside them the keys only the tier still has, and which of the blocks
        are idle ones the request does not hold."""
        rid = request.request_id
        configs = self.config.kv_group_configs
        hashes = list(request.block_hashes)
        per_hash = self.config.tokens_per_hash
        policy = getattr(self.manager, "_policy", None)
        threads = getattr(pool, "context_threads", None)
        tails: dict[int, tuple[typing.Any, typing.Any]] = {}
        for block, key in threads.request_decode_tails(rid) if threads is not None else []:
            tails[_kv_cache_utils.get_group_id(key)] = (block, key)
        recurrent = [
            g for g, cfg in enumerate(configs) if cfg.hashes_per_chunk and cfg.sliding_window_size_in_chunks is not None
        ]
        chunk = next((cfg.tokens_per_chunk for cfg in configs if cfg.hashes_per_chunk), 0)
        boundary: int | None = None
        # Recurrent tails the GPU has reused but the tier holds (``_keep_tails``
        # copied them as the engine keyed them): taken from there.
        hosted_tails: set[int] = set()
        if recurrent and all(g in tails for g in recurrent):
            bounds = {_boundary_of(hashes, tails[g][1], per_hash) for g in recurrent}
            for g in recurrent:
                block, with_group = tails[g]
                if block.block_hash == with_group:
                    continue
                tail_key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(with_group), g)
                if policy is None or policy.get(tail_key) is None:
                    break
                hosted_tails.add(g)
            else:
                if len(bounds) == 1 and None not in bounds:
                    boundary = bounds.pop()
            if boundary is None:
                hosted_tails.clear()
        # The computed count runs ahead of the chain by the drafts in flight,
        # and a boundary past the chain has no hash to key it by.
        computed = min(int(getattr(request, "num_computed_tokens", 0)), len(hashes) * per_hash)
        end = boundary if boundary is not None else (computed // chunk) * chunk if chunk else 0
        full = end // chunk if chunk else 0
        partial = boundary is not None and chunk and boundary % chunk != 0
        per_group: list[list[tuple[int, _offload_base.OffloadKey]]] = []
        extra: list[typing.Any] = []  # idle blocks the request does not hold
        hosted: list[_offload_base.OffloadKey] = []  # thread keys only the tier still has
        checkpoints: list[_offload_base.OffloadKey] = []  # the recurrent states at the last full boundary
        notes: list[str] = []
        for g, cfg in enumerate(configs):
            rows: list[tuple[int, _offload_base.OffloadKey]] = []
            per_group.append(rows)
            if not cfg.hashes_per_chunk:
                continue  # a transient group the tier passes over
            block_ids = block_id_groups[g] if g < len(block_id_groups) else []
            hpc = cfg.hashes_per_chunk
            if cfg.sliding_window_size_in_chunks is None:
                for i in range(full):
                    row = _verified_row(pool, block_ids, i, hashes[(i + 1) * hpc - 1], g)
                    if row is None:
                        notes.append(f"g{g} chunk {i} unverified")
                        break
                    rows.append(row)
                if partial and len(rows) == full and full < len(block_ids) and block_ids[full]:
                    assert boundary is not None
                    rows.append((block_ids[full], _offload_base.make_offload_key(hashes[boundary // per_hash - 1], g)))
            else:
                if full > 0:
                    block_hash = hashes[full * hpc - 1]
                    row = _verified_row(pool, block_ids, full - 1, block_hash, g)
                    if row is None:
                        cached = pool.cached_block_hash_to_block.get_one_block(
                            _kv_cache_utils.make_block_hash_with_group_id(block_hash, g)
                        )
                        checkpoint_key = _offload_base.make_offload_key(block_hash, g)
                        if cached is not None and not cached.is_null:
                            row = (cached.block_id, checkpoint_key)
                            extra.append(cached)
                        elif policy is not None and policy.get(checkpoint_key) is not None:
                            # Gone from the GPU, but the keep-young pass stored
                            # it while it lived. A boundary the decode
                            # completed lives only that way: the engine keys
                            # it as a decode tail, until the next tail
                            # replaces it.
                            hosted.append(checkpoint_key)
                            checkpoints.append(checkpoint_key)
                            notes.append(f"g{g} checkpoint at {full * chunk} on the host only")
                        else:
                            notes.append(f"g{g} no checkpoint at {full * chunk}")
                    if row is not None:
                        rows.append(row)
                        checkpoints.append(row[1])
                if partial:
                    assert boundary is not None
                    tail_block, _ = tails[g]
                    tail_key = _offload_base.make_offload_key(hashes[boundary // per_hash - 1], g)
                    if g in hosted_tails:
                        hosted.append(tail_key)
                        notes.append(f"g{g} tail at {boundary} on the host only")
                    else:
                        rows.append((tail_block.block_id, tail_key))
                        extra.append(tail_block)
        flat = [row for rows in per_group for row in rows]
        return {
            "rows": per_group,
            # The tier-held keys go last: they are present, so they take no
            # slot, and the new keys keep the group-major order the copy pairs by.
            "keys": [key for _, key in flat] + hosted,
            "hosted": hosted,
            "checkpoints": checkpoints,
            "idle": {block.block_id for block in extra},
            "end": end,
            "partial": bool(partial),
            "hosted_tails": bool(hosted_tails),
            "notes": notes,
            "tokens": request.num_tokens,
            "computed": computed,
        }

    def _park_issue(
        self,
        plan: dict[str, typing.Any],
        context: str,
        rid: str,
        pool: "vllm.v1.core.block_pool.BlockPool",
        req_status: typing.Any = None,
        pins: list[tuple[int, typing.Any]] | None = None,
        into: dict[int, _offload_common.TransferJob] | None = None,
        frees: int = 0,
        why: str = "finish",
    ) -> str:
        """Copy a thread to the tier as `_park_plan` laid it out. What the tier
        already has is touched, so the thread ages as one; the rest is copied,
        its GPU blocks held until the copy lands and then uncached. At a finish
        (``req_status``) the engine holds the request's blocks and this holds
        the idle ones; a thread spilled from the GPU (``pins``) is held by its
        pins, and its job goes out in ``into``, under the victim sentinel, as
        no request of the engine's holds it; ``frees`` is how many GPU blocks
        its landing frees, which the spill step counts as coming. "queued" when
        a copy is in flight, "present" when the tier already had all of it,
        "refused" when the tier could not take it."""
        # A key no GPU block backs (one only the tier held at the plan) is
        # offered only while the tier still has it: then it is present, and
        # this offer's evictions pass over it, so every key the tier takes as
        # new has a block to copy from. One the tier has lost since (a spill
        # comes long after its plan) is left out, and the thread parks
        # without it.
        sourced = {key for rows in plan["rows"] for _, key in rows}
        policy = getattr(self.manager, "_policy", None)
        lost = [key for key in plan["keys"] if key not in sourced and policy is not None and policy.get(key) is None]
        keys = [key for key in plan["keys"] if key not in lost] if lost else plan["keys"]
        self._park_counts["host_keys_lost"] += len(lost)
        spill = pins is not None
        req_context = self._victim_req_context if spill else req_status.req_context
        store_output = self._offer_store(keys, req_context) if keys else None
        if keys and store_output is None:
            self._park_counts["store_refused"] += 1
            self._park_say(
                "PARK-REFUSED",
                f"ctx={_short_ctx(context)} req={_short_req(rid)} the tier could not take {len(keys)} slots",
            )
            return "refused"
        self._park_evicted(getattr(store_output, "evicted_keys", None) or [])
        new = set(store_output.keys_to_store) if store_output is not None else set()
        present = [key for key in keys if key not in new]
        if present:
            self.manager.touch(list(reversed(present)), req_context)
        previous = self._parked.get(context)
        self._parked[context] = {
            "keys": keys,
            "checkpoints": set(plan["checkpoints"]),
            "boundary": plan["end"],
            "req": rid,
            "state": "parked",
            # A spilled thread has been parked since its finish.
            "parked_at": previous["parked_at"] if spill and previous is not None else time.monotonic(),
        }
        self._park_claim(context, keys, held=True)
        if plan["hosted_tails"]:
            self._park_counts["tail_from_host"] += 1
        if previous is not None:
            kept = set(keys)
            stale = [key for key in previous["keys"] if key not in kept]
            self._park_claim(context, stale, held=False)
            self._park_demote(stale, f"superseded by ctx={_short_ctx(context)}'s next park")
        flat = [row for rows in plan["rows"] for row in rows]
        gpu = [(block_id, pool.blocks[block_id].block_hash) for block_id, _ in flat]
        partial = plan["partial"]
        event = "SPILL" if spill else "PARK"
        detail = (
            f"ctx={_short_ctx(context)} req={_short_req(rid)} tokens={plan['tokens']} computed={plan['computed']} "
            f"resume_at={plan['end']} tail={('host' if plan['hosted_tails'] else 'yes') if partial else 'no'} "
            f"slots={len(keys)} new={len(new)} present={len(present)}"
            + (f" lost={len(lost)}" if lost else "")
            + (f" notes={';'.join(plan['notes'])}" if plan["notes"] else "")
        )
        if not new:
            self._park_retire.extend(gpu)
            if pins:
                pool.free_blocks([pool.blocks[block_id] for block_id, _ in pins])
            self._park_counts["parked_nothing_new"] += 1
            self._park_say(event, detail + f" reason={why} (host already had all of it)")
            return "present"
        src_block_ids: list[int] = []
        group_sizes: list[int] = []
        final_keys: list[_offload_base.OffloadKey] = []
        hold: list[typing.Any] = []
        for rows in plan["rows"]:
            count = 0
            for block_id, key in rows:
                if key not in new:
                    continue
                src_block_ids.append(block_id)
                final_keys.append(key)
                count += 1
                if not spill and block_id in plan["idle"]:
                    hold.append(pool.blocks[block_id])
            group_sizes.append(count)
        assert store_output is not None  # new keys come from an accepted store
        if hold:
            # Idle blocks (decode tails, a checkpoint the request moved past)
            # are not in the engine's hold on the request: held here instead.
            pool.touch(hold)
        job_id = self._generate_job_id()
        src_spec = _offload_base.GPULoadStoreSpec(
            src_block_ids,
            group_sizes=group_sizes,
            block_indices=[0] * len(plan["rows"]),
        )
        if spill:
            assert into is not None and pins is not None
            self._jobs[job_id] = _offload_sched.TransferJobStatus(
                req_id=VICTIM_REQ_ID,
                pending_count=self.config.num_workers,
                keys=set(final_keys),
                is_store=True,
            )
            self._victim_job_ids.add(job_id)
            # Checked on landing like any victim store, though a pinned block
            # cannot have been reused meanwhile.
            self._spilled[job_id] = [
                (block_id, pool.blocks[block_id].block_hash, key) for block_id, key in zip(src_block_ids, final_keys)
            ]
            into[job_id] = _offload_common.TransferJob(
                req_id=VICTIM_REQ_ID, src_spec=src_spec, dst_spec=store_output.store_spec
            )
            held = pins
        else:
            self._jobs[job_id] = _offload_sched.TransferJobStatus(
                req_id=rid,
                pending_count=self.config.num_workers,
                keys=typing.cast("set[_offload_base.OffloadKey]", final_keys),
                is_store=True,
            )
            req_status.transfer_jobs.add(job_id)
            self._held_finish_jobs.add(job_id)
            self._pending_finish_jobs[job_id] = _offload_common.TransferJob(
                req_id=rid, src_spec=src_spec, dst_spec=store_output.store_spec
            )
            held = [(block.block_id, block.block_hash) for block in hold]
        self._park_jobs[job_id] = {
            "context": context,
            "req": rid,
            "slots": len(final_keys),
            "held": held,
            "retire": gpu,
            "spill": spill,
            "keys": final_keys,
            "frees": frees,
        }
        self._park_counts["spills" if spill else "parks"] += 1
        self._park_say(event, detail + f" job={job_id} reason={why}")
        return "queued"

    def _park_pin(
        self,
        request: typing.Any,
        block_id_groups: tuple[list[int], ...],
        pool: "vllm.v1.core.block_pool.BlockPool",
        context: str,
    ) -> None:
        """Park on the GPU (``gpu_park``): the thread stays where it is, every
        block of it pinned so the free queue cannot reuse it, until its next
        request takes it over (`_park_running`), it dies (`_park_step`), or the
        pool needs the room and it spills to the tier, oldest first
        (`_park_spill_step`). Nothing of it is copied; what the tier holds of
        it is its recurrent checkpoints, held there as a running thread's
        are, and touched, as a park touches what the tier has: a request
        resumed at the tail holds none, so the GPU's goes once the next
        request takes the thread over, and the tier's is often the only one.
        A follow-up that cannot resume at the tail resumes at the checkpoint,
        its attention hit on the GPU and the state loaded from the tier
        (``fill_divergent_hit``), and a spill copies the rest."""
        plan = self._park_plan(request, block_id_groups, pool)
        blocks = [pool.blocks[block_id] for rows in plan["rows"] for block_id, _ in rows]
        rid = request.request_id
        hosted = plan["hosted"]
        previous = self._parked.get(context)
        if not blocks and not hosted and previous is not None:
            # A request that ended before it computed anything (aborted while
            # it waited) leaves its conversation's thread where it was.
            self._park_say("SKIP", f"ctx={_short_ctx(context)} req={_short_req(rid)} reason=nothing computed")
            return
        pool.touch(blocks)
        if previous is not None:
            # Pinned by a request that never ran (aborted while it waited):
            # this pin takes over.
            self._park_unpin(context, previous, pool, dead=False, why=f"superseded by req={_short_req(rid)}")
        # The tier's checkpoints are the thread's whether or not the GPU has
        # them too: the next request resumed at the tail does not hold the
        # GPU's, which goes soon after it takes the thread over.
        policy = getattr(self.manager, "_policy", None)
        held = list(
            dict.fromkeys(
                [*hosted, *(key for key in plan["checkpoints"] if policy is not None and policy.get(key) is not None)]
            )
        )
        self._parked[context] = {
            "keys": held,
            "checkpoints": set(plan["checkpoints"]),
            "boundary": plan["end"],
            "req": rid,
            "state": "parked",
            "parked_at": time.monotonic(),
            "plan": plan,
            "pins": [(block.block_id, block.block_hash) for block in blocks],
        }
        self._park_claim(context, held, held=True)
        self._touch_hosted(held)
        if previous is not None:
            kept = set(held)
            stale = [key for key in previous["keys"] if key not in kept]
            self._park_claim(context, stale, held=False)
            self._park_demote(stale, f"superseded by ctx={_short_ctx(context)}'s park on the GPU")
        self._park_counts["gpu_parks"] += 1
        self._park_say(
            "PARK-GPU",
            f"ctx={_short_ctx(context)} req={_short_req(rid)} tokens={plan['tokens']} resume_at={plan['end']} "
            f"tail={'yes' if plan['partial'] else 'no'} pinned={len(blocks)} hosted={len(hosted)}"
            + (f" notes={';'.join(plan['notes'])}" if plan["notes"] else ""),
        )

    def _touch_hosted(self, keys: list[_offload_base.OffloadKey]) -> None:
        """Keep the tier's part of a thread parked on the GPU young: at its
        park, and when its next request takes it over, where a thread parked
        in the tier is touched by its park and its resume's load. Nothing
        else of the thread uses those keys, so untouched they age out while
        it lives."""
        if keys:
            self.manager.touch(list(reversed(keys)), self._victim_req_context)

    def _park_unpin(
        self,
        context: str,
        record: dict[str, typing.Any],
        pool: "vllm.v1.core.block_pool.BlockPool",
        dead: bool,
        why: str,
    ) -> None:
        """Let go of a thread's GPU pins: to the request that took the thread
        over, or, ``dead``, to the eviction end of the free queue, still
        cached, as the engine sends a dead thread's idle blocks and the tier
        demotes its keys. A thread is declared dead ahead of the request that
        replaces it (a close, a new epoch), and that request's prompt can
        still extend it, so it is the first thing reused, not gone."""
        pins = record.pop("pins", None)
        record.pop("plan", None)
        if not pins:
            return
        blocks = [pool.blocks[block_id] for block_id, _ in pins]
        pool.free_blocks(blocks)
        if dead:
            idle = [
                block
                for block, (_, block_hash) in zip(blocks, pins)
                if block.ref_cnt == 0 and block_hash is not None and block.block_hash == block_hash
            ]
            for block in idle:
                pool.free_block_queue.remove(block)
                block.dead = True
            if idle:
                pool.free_block_queue.prepend_n(idle)
        self._park_counts["unpinned"] += 1
        self._park_say("UNPIN", f"ctx={_short_ctx(context)} {len(pins)} blocks; {why}")

    def _gpu_park_target(self, pool: "vllm.v1.core.block_pool.BlockPool") -> int:
        """The free blocks parking on the GPU keeps: the engine's admission
        watermark, the reserve the running requests grow into while a spill's
        copy is in flight, and the blocks of the requests waiting to be
        admitted."""
        return int(pool.num_gpu_blocks * (self._admission_watermark + self._gpu_park_reserve)) + sum(
            self._park_needs.values()
        )

    def _pin_tails(
        self,
        rid: str,
        live: dict[int, typing.Any],
        pool: "vllm.v1.core.block_pool.BlockPool",
        target: int,
    ) -> None:
        """Parking on the GPU: hold a running request's newest decode tail
        (``live``, block id -> hash) on the GPU while the pool has room above
        what admission needs, in place of a copy to the tier, so its finish
        parks it there; let go of the tail it supersedes. Without the room it
        is kept young in the free queue, as a copied one is."""
        held = self._tail_pins.setdefault(rid, {})
        gone = [block_id for block_id in held if block_id not in live]
        if gone:
            pool.free_blocks([pool.blocks[block_id] for block_id in gone])
            for block_id in gone:
                del held[block_id]
        queue = pool.free_block_queue
        for block_id, with_group in live.items():
            if block_id in held:
                continue
            block = pool.blocks[block_id]
            if pool.get_num_free_blocks() > target:
                pool.touch([block])
                held[block_id] = with_group
            elif block.ref_cnt == 0:
                queue.remove(block)
                queue.append(block)
        if not held:
            del self._tail_pins[rid]

    def _release_tail_pins(self, rid: str, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Let go of a request's pinned decode tail: to its parked thread at
        its finish, or to the free queue when the pool is short."""
        held = self._tail_pins.pop(rid, None)
        if held:
            pool.free_blocks([pool.blocks[block_id] for block_id in held])

    def _park_spill_step(
        self, pool: "vllm.v1.core.block_pool.BlockPool", idle: bool = False
    ) -> dict[int, _offload_common.TransferJob]:
        """Spill threads parked on the GPU to the tier, oldest first, while the
        pool's free blocks and those in-flight spills will free fall short of
        what it needs (`_gpu_park_target`); the running requests' pinned
        tails are let go of first. A thread whose next request has arrived is
        about to run where it is, so it is spilled only while the engine is
        idle (`_engine_idle`): then the arrived threads' pins can be all that
        keeps the waiting requests out, and only a running request lets go of
        a thread's pins."""
        jobs: dict[int, _offload_common.TransferJob] = {}
        free = pool.get_num_free_blocks()
        incoming = sum(info["frees"] for info in self._park_jobs.values() if info.get("spill"))
        waiting = sum(self._park_needs.values())
        target = self._gpu_park_target(pool)
        if free + incoming >= target:
            return jobs
        if self._tail_pins:
            # Running requests' tails go first: letting go of one copies
            # nothing, and it stays cached at the free queue's young end, the
            # last a block is taken from.
            released = sum(len(held) for held in self._tail_pins.values())
            for rid in list(self._tail_pins):
                self._release_tail_pins(rid, pool)
            self._park_counts["tails_released"] += released
            self._park_say(
                "TAILS-RELEASED", f"{released} tail blocks; pool short: {free} free + {incoming} coming < {target}"
            )
            free = pool.get_num_free_blocks()
            if free + incoming >= target:
                return jobs
        arrived = set(self._park_active.values())
        on_gpu = sorted(
            (record["parked_at"], context) for context, record in self._parked.items() if record.get("pins")
        )
        candidates = [context for _, context in on_gpu if context not in arrived]
        if idle:
            candidates += [context for _, context in on_gpu if context in arrived]
        for context in candidates:
            if free + incoming >= target:
                break
            record = self._parked[context]
            pins = record["pins"]
            # Only blocks nothing else holds come free when the pins go.
            frees = sum(1 for block_id, _ in pins if pool.blocks[block_id].ref_cnt == 1)
            plan = record.pop("plan")
            record.pop("pins")
            why = f"pool short: {free} free + {incoming} coming < {target} (waiting {waiting})" + (
                ", engine idle, its request waiting" if context in arrived else ""
            )
            outcome = self._park_issue(plan, context, record["req"], pool, pins=pins, into=jobs, frees=frees, why=why)
            if outcome == "refused":
                # The tier cannot take it; holding it on the GPU would starve
                # the engine instead, so it is dropped, to be recomputed.
                pool.free_blocks([pool.blocks[block_id] for block_id, _ in pins])
                self._park_retire.extend(pins)
                self._parked.pop(context, None)
                self._park_claim(context, record["keys"], held=False)
                self._park_counts["spill_dropped"] += 1
                self._park_say("SPILL-DROPPED", f"ctx={_short_ctx(context)} {len(pins)} blocks; the tier refused it")
            # In flight, dropped, or unpinned already when the host had it all.
            incoming += frees
        return jobs

    def _publish_kv_shares(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Set the pool's two shares for the engine line, at most every
        `KV_SHARE_INTERVAL_S`: the blocks requests hold, and the blocks only
        threads parked on the GPU hold (a block a request shares with a
        parked thread is the request's)."""
        now = time.monotonic()
        if now - self._kv_share_at < KV_SHARE_INTERVAL_S:
            return
        self._kv_share_at = now
        pins = collections.Counter(
            block_id for record in self._parked.values() for block_id, _ in record.get("pins", ())
        )
        pins.update(block_id for info in self._park_jobs.values() if info.get("spill") for block_id, _ in info["held"])
        parked = sum(1 for block_id, count in pins.items() if pool.blocks[block_id].ref_cnt <= count)
        total = max(pool.num_gpu_blocks - 1, 1)  # less the null block
        used = total - pool.get_num_free_blocks()
        self._connector_stats.set_gauge(ullm.engine_monitor.KV_REQUESTS_SHARE, max(used - parked, 0) / total)
        self._connector_stats.set_gauge(ullm.engine_monitor.KV_GPU_PARKED_SHARE, parked / total)

    def _engine_idle(self) -> bool:
        """Whether no request the engine tracks moves on by itself (none is in
        `_PROGRESSING`), so nothing but the spill step frees pool blocks."""
        return not any(status.req.status in _PROGRESSING for status in self._req_status.values())

    def _admission_blocks(self, request: typing.Any, num_computed_tokens: int) -> int:
        """About the GPU blocks admitting a request takes: its attention past
        the GPU's hit, chunk by chunk, and a state block for each recurrent
        group."""
        total = 0
        for cfg in self.config.kv_group_configs:
            if not cfg.hashes_per_chunk or not cfg.tokens_per_chunk:
                continue  # a transient group the tier passes over
            if cfg.sliding_window_size_in_chunks is None:
                chunks = -(-int(request.num_tokens) // cfg.tokens_per_chunk)
                total += max(chunks - num_computed_tokens // cfg.tokens_per_chunk, 0)
            else:
                total += 1
        return total

    def _build_store_jobs(
        self, scheduler_output: "vllm.v1.core.sched.output.SchedulerOutput"
    ) -> dict[int, _offload_common.TransferJob]:
        """Spill imminent victims, in place of the stock per-request store.

        Mirrors the eviction front of the free queue: whatever hashed block
        sits within the band and is not yet on the host is stored,
        nearest-to-die first, up to the per-step cap; whatever is already
        there is touched, since reaching the front is when a host copy stops
        being a duplicate. The band is the front and no deeper: a block
        further back is one a prefix hit may take back before it leaves the
        GPU, and refreshing its copy would make the tier's recency the GPU's
        churn. A run whose front is unhashed blocks produces no job at all,
        which is what distinguishes this connector from the stock one on a
        run that never fills its cache.
        """
        jobs = self._pending_finish_jobs
        self._held_this_step = frozenset(self._held_finish_jobs)
        self._held_finish_jobs = set()
        self._pending_finish_jobs = {}
        for job_id, load in self._current_batch_load_jobs.items():
            dst = getattr(load, "dst_spec", None)
            for bid in getattr(dst, "block_ids", []) if dst is not None else []:
                self._note_block(int(bid), f"load-dst:{job_id}")

        pool = self._gpu_block_pool
        if pool is None:
            # The engine binds the pool before the first schedule, so this is a
            # topology that never called bind_gpu_block_pool rather than a
            # transient state — say so once instead of silently never spilling.
            if not self._spill_disabled_warned:
                logger.warning("victim offload: no GPU block pool bound; nothing will be offloaded")
                self._spill_disabled_warned = True
            return jobs

        if self._trace.enabled and not self._trace_started:
            self._trace_started = True
            self._trace.write(
                "start",
                pool=pool.num_gpu_blocks,
                band=self._spill_band_blocks,
                cap=self._max_spill_blocks,
                probe=KEEP_YOUNG_PROBE_CHUNKS,
                tier=getattr(self._hit_depth_policy, "cache_capacity", None),
                groups=[
                    {
                        "g": g,
                        "single_state": cfg.sliding_window_size_in_chunks is not None,
                        "hashes_per_chunk": cfg.hashes_per_chunk,
                        "tokens_per_chunk": cfg.tokens_per_chunk,
                    }
                    for g, cfg in enumerate(self.config.kv_group_configs)
                ],
                vllm=vllm.__version__,
                pid=os.getpid(),
                sizing=self._sizing,
            )

        if self._park:
            # No band: a thread leaves the GPU at its request's finish. The
            # rest of the step stands. A running request's newest checkpoint
            # is part of the thread its finish parks, and the engine frees it
            # the step after its boundary is crossed; a boundary the decode
            # completes it keys only as a decode tail, until the next tail
            # replaces it. The keep-young pass keeps the checkpoint on the GPU
            # until the finish, and stores either to the tier while it lives.
            self._park_step(pool)
            if self._gpu_park:
                jobs.update(self._park_spill_step(pool, idle=self._engine_idle()))
                self._publish_kv_shares(pool)

        # The watermark gates only the finish-hold, not the spill: it says
        # when a finishing request must be held for its copy.
        if self._under_watermark(pool):
            self._pressure_until = time.monotonic() + PRESSURE_STICKY_S

        self._reage_checkpoints(pool)
        self._keep_checkpoints_young(pool)
        if self._park:
            self._keep_tails(pool)
        self._account_departures(pool)
        victims = self._collect_victims(pool, 0 if self._park else self._spill_band_blocks)
        # The keep-young pass's blocks go first: a running request's newest
        # checkpoint is the block the fill loads, and on a pool whose queue
        # drains within a step the band never sees it before it leaves.
        # Ahead of the band, not touched when present: reaching the front is
        # what makes a copy worth refreshing, being kept young is not.
        kept_ids = {block.block_id for block in self._kept_blocks}
        victims = self._kept_blocks + [block for block in victims if block.block_id not in kept_ids]
        if not victims:
            return jobs

        # One offload key per victim, from the block's own hash. The pool does
        # not de-duplicate blocks, so two can carry the same hash and hence the
        # same key; a key gets one slot, so only its first claimant goes. Keys
        # the host already holds are split off before the offer so the cap
        # counts new copies only: the band is walked every step, and most of
        # it is already there.
        policy = getattr(self.manager, "_policy", None)
        candidates: list[tuple[typing.Any, _offload_base.OffloadKey]] = []
        present: list[_offload_base.OffloadKey] = []
        claimed: set[_offload_base.OffloadKey] = set()
        for block in victims:
            group_idx = _kv_cache_utils.get_group_id(block.block_hash)
            key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(block.block_hash), group_idx)
            if key in claimed:
                continue
            claimed.add(key)
            # The engine sent this block to the eviction end as nobody's live
            # context (vLLM's KVCacheBlock.dead: superseded, or a dead variant
            # of a context's thread). As with a passed checkpoint, a copy could
            # never turn into a hit: not stored, and a copy the host already
            # holds goes to the eviction end of its class. Not a lasting
            # class: the same content cached again as a live block is stored
            # as usual.
            if getattr(block, "dead", False):
                self._dead_skipped += 1
                self._dead_keys[key] = None
                self._dead_keys.move_to_end(key)
                if len(self._dead_keys) > DEAD_KEYS_LIMIT:
                    self._dead_keys.popitem(last=False)
                if self._demote_dead is not None and policy is not None and policy.get(key) is not None:
                    self._demote_dead(self._park_unneeded([key]))
                continue
            self._dead_keys.pop(key, None)
            # A passed checkpoint is evicted ahead of everything else by
            # construction, so a copy of one is a slot transit and a PCIe
            # burst that can never turn into a hit. Neither stored nor
            # touched. A recompute re-creates a context's every checkpoint,
            # so without this each miss would pay for ninety copies of them.
            if self._retention is not None and self._retention.is_passed(key) and key not in self._tail_step_keys:
                self._passed_skipped += 1
                continue
            if policy is not None and policy.get(key) is not None:
                if block.block_id not in kept_ids:
                    present.append(key)
            elif len(candidates) < self._max_spill_blocks:
                candidates.append((block, key))
                if block.block_id in kept_ids:
                    self._kept_stored += 1

        # Gathered GROUP-MAJOR before anything is offered to the manager: the
        # worker pairs source blocks with destination slots by position, group
        # by group, so the key order the manager allocates slots in has to be
        # the order the source block ids are listed in.
        num_groups = len(self.config.kv_group_configs)
        by_group: list[list[tuple[typing.Any, _offload_base.OffloadKey]]] = [[] for _ in range(num_groups)]
        for block, key in candidates:
            by_group[_kv_cache_utils.get_group_id(block.block_hash)].append((block, key))
        ordered = [(block, key) for group in by_group for block, key in group]

        keys_to_store: set[_offload_base.OffloadKey] = set()
        store_output = None
        if ordered:
            store_output = self._offer_store([key for _, key in ordered], self._victim_req_context)
            if store_output is not None:
                keys_to_store = set(store_output.keys_to_store)
                if self._park:
                    self._park_evicted(getattr(store_output, "evicted_keys", None) or [])
                    self._park_counts["tail_slots_stored"] += len(keys_to_store & self._tail_step_keys)
            # What the manager declined is there after all (a copy in flight).
            present.extend(key for _, key in ordered if key not in keys_to_store)

        # A victim the host already holds has reached the eviction front
        # AGAIN: the block is about to stop being a duplicate, which is the
        # use a victim cache orders by. Its copy moves to the recent end, or
        # it would keep the position of its first spill however often it
        # bounced since. Newest-first, because touch makes the first key it
        # is given the most recent and victims come nearest-to-die first.
        self._band_present += len(present)
        if present:
            self.manager.touch(list(reversed(present)), self._victim_req_context)
        if not keys_to_store or store_output is None:
            return jobs

        # What the job copies, in the order the slots were allocated. The hash
        # is kept because the pool is free to reuse any of these blocks before
        # the copy lands, and completion checks each one is still what was read.
        spilled: list[tuple[int, typing.Any, _offload_base.OffloadKey]] = []
        src_block_ids: list[int] = []
        group_sizes: list[int] = []
        for group in by_group:
            count = 0
            for block, key in group:
                if key not in keys_to_store:
                    continue
                spilled.append((block.block_id, block.block_hash, key))
                src_block_ids.append(block.block_id)
                count += 1
            group_sizes.append(count)

        # block_indices is only consulted when an offloaded block spans several
        # GPU blocks, which __init__ has already ruled out.
        src_spec = _offload_base.GPULoadStoreSpec(
            src_block_ids,
            group_sizes=group_sizes,
            block_indices=[0] * num_groups,
        )

        job_id = self._generate_job_id()
        self._victim_job_ids.add(job_id)
        self._spilled[job_id] = spilled
        self._blocks_spilled += len(spilled)
        self._sample_counters["spilled"] += len(spilled)
        if self._park:
            pool.touch([pool.blocks[block_id] for block_id in src_block_ids])
            self._spill_holds[job_id] = src_block_ids
        if self._trace.enabled:
            self._trace.write("store", step=self._step, origin="band", blocks=len(spilled), groups=group_sizes)
        self._jobs[job_id] = _offload_sched.TransferJobStatus(
            req_id=VICTIM_REQ_ID,
            pending_count=self.config.num_workers,
            keys=keys_to_store,
            is_store=True,
        )
        logger.debug(
            "victim offload: spilling %d blocks (job %d), %d free of %d",
            len(src_block_ids),
            job_id,
            pool.get_num_free_blocks(),
            pool.num_gpu_blocks,
        )
        jobs[job_id] = _offload_common.TransferJob(
            req_id=VICTIM_REQ_ID, src_spec=src_spec, dst_spec=store_output.store_spec
        )
        return jobs

    def _reage_checkpoints(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Move the checkpoint blocks finished requests left for re-aging to
        the tail of the GPU's free queue, now that the engine has freed the
        attention they belong with: a context's checkpoints are then evicted
        after its attention, never before, and an intact attention prefix
        keeps the state at its boundary."""
        if not self._reage_pending:
            return
        queue = pool.free_block_queue
        for with_group in self._reage_pending:
            block = pool.cached_block_hash_to_block.get_one_block(with_group)
            # Only a free block sits in the queue; one in use is ordered by
            # its own release later.
            if block is None or block.ref_cnt != 0 or block.block_hash != with_group:
                continue
            queue.remove(block)
            queue.append(block)
            self._reaged += 1
            self._reaged_at[block.block_id] = self._step
        self._reage_pending = []

    def _recent_checkpoints(
        self,
        pool: "vllm.v1.core.block_pool.BlockPool",
        hashes: typing.Sequence[bytes],
        group_idx: int,
        factor: int,
        upto_chunk: int,
        want: int,
        probe: int = KEEP_YOUNG_PROBE_CHUNKS,
    ) -> list[tuple[int, bytes, typing.Any, typing.Any]]:
        """The newest `want` checkpoints of one single-state group that the
        GPU still caches, probing back from `upto_chunk` over at most
        `probe` chunks. Oldest first; each is the chunk index, the chain
        hash, the hash with group, and the block."""
        found: list[tuple[int, bytes, typing.Any, typing.Any]] = []
        for chunk in range(upto_chunk - 1, max(upto_chunk - probe, 0) - 1, -1):
            index = (chunk + 1) * factor - 1
            if index >= len(hashes):
                continue
            with_group = _kv_cache_utils.make_block_hash_with_group_id(
                _kv_cache_utils.BlockHash(hashes[index]), group_idx
            )
            block = pool.cached_block_hash_to_block.get_one_block(with_group)
            if block is None or block.block_hash != with_group:
                continue
            found.append((chunk, hashes[index], with_group, block))
            if len(found) == want:
                break
        found.reverse()
        return found

    def _keep_checkpoints_young(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Move the newest cached checkpoint of every request the connector
        knows — waiting, running, or just finished — to the tail of the
        GPU's free queue, every step.

        The engine frees a checkpoint the step after its boundary is
        crossed and needs it again at the next turn's lookup, so on a pool
        that evicts block by block it has to survive a whole turn as a free
        block, ordered by an age the request's own blocks never accumulate.
        Its residence in the queue is the queue's length over the eviction
        rate, which shrinks as the live set grows and crosses the turn
        length well before anything is preempted; from then on every turn
        that outlasts it recomputes its prompt, and each recomputation
        shortens the residence further. Re-aging at finish is a turn too
        late for that. Kept young from the start, the block is as safe as
        the attention it belongs with.
        """
        queue = pool.free_block_queue
        kept = missing = 0
        kept_map: dict[int, tuple[str, int, int]] = {}
        kept_blocks: list[typing.Any] = []
        for status in self._req_status.values():
            hashes = status.req.block_hashes
            req_id = status.req.request_id
            for group_idx, group_config in enumerate(self.config.kv_group_configs):
                if not group_config.hashes_per_chunk:
                    continue  # a transient group the tier passes over
                if group_config.sliding_window_size_in_chunks is None:
                    continue
                factor = group_config.hashes_per_chunk
                upto = len(hashes) // factor
                if upto == 0:
                    continue
                found = self._recent_checkpoints(pool, hashes, group_idx, factor, upto, want=1)
                if not found:
                    missing += 1
                    continue
                chunk, _, _, block = found[0]
                # A block with a holder is not in the queue and needs no help.
                if block.ref_cnt != 0:
                    continue
                queue.remove(block)
                queue.append(block)
                kept += 1
                kept_map[block.block_id] = (req_id, group_idx, chunk)
                kept_blocks.append(block)
        self._kept_last_step = kept
        self._kept_missing_last_step = missing
        self._kept_map_prev, self._kept_map = self._kept_map, kept_map
        self._kept_blocks = kept_blocks

    def _keep_tails(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Park mode: copy every request's newest decode tail to the tier as
        the engine keys it, keeping its block young until the copy lands;
        parking on the GPU, pin it there instead (`_pin_tails`).

        The engine keys a tail once the decode passes its boundary and leaves
        it idle in the free queue while the request decodes on; the park at
        the finish, up to a unit later, finds it on the GPU only if the queue
        has not reused it by then. A pool at its admission line turns its
        queue over in under a second, faster than that: 75% of the parks on
        the pool-bound RTX carried no tail. A tail the tier holds is one the
        park takes from there. The tail a newer one supersedes is demoted: a
        follow-up extends the whole reply, so only the newest is worth a
        slot."""
        threads = getattr(pool, "context_threads", None)
        step_keys: set[_offload_base.OffloadKey] = set()
        if threads is None:
            self._tail_step_keys = step_keys
            return
        queue = pool.free_block_queue
        policy = getattr(self.manager, "_policy", None)
        kept_ids = {block.block_id for block in self._kept_blocks}
        target = self._gpu_park_target(pool) if self._gpu_park else 0
        for status in self._req_status.values():
            rid = status.req.request_id
            keys: list[_offload_base.OffloadKey] = []
            live: dict[int, typing.Any] = {}
            for block, with_group in threads.request_decode_tails(rid):
                key = _offload_base.make_offload_key(
                    _kv_cache_utils.get_block_hash(with_group), _kv_cache_utils.get_group_id(with_group)
                )
                keys.append(key)
                if block.block_hash != with_group:
                    continue  # reused already: the tier has it or never will
                if self._gpu_park:
                    live[block.block_id] = with_group
                    continue
                if block.ref_cnt == 0:
                    queue.remove(block)
                    queue.append(block)
                if block.block_id not in kept_ids and (policy is None or policy.get(key) is None):
                    self._kept_blocks.append(block)
                    kept_ids.add(block.block_id)
                    step_keys.add(key)
            if self._gpu_park:
                self._pin_tails(rid, live, pool, target)
            previous = self._tail_keys.get(rid)
            if keys and previous != keys:
                self._tail_keys[rid] = keys
                if previous:
                    self._park_demote_quiet([key for key in previous if key not in keys])
        self._tail_step_keys = step_keys

    def _park_demote_quiet(self, keys: list[_offload_base.OffloadKey] | None) -> None:
        """Demote host copies nobody parked needs, unlogged: superseded decode
        tails, many a second."""
        if not keys:
            return
        policy = getattr(self.manager, "_policy", None)
        present = [key for key in self._park_unneeded(keys) if policy is not None and policy.get(key) is not None]
        if present and self._demote_dead is not None:
            self._demote_dead(present)
        self._park_counts["tail_demoted"] += len(present)

    def _park_order_gauge(self) -> dict[str, typing.Any] | None:
        """The tier's main order as park mode needs it: how deep its deepest
        parked key sits (the size the tier needs for every parked thread to
        survive), and what the keys newer than it are, by whose they are and
        what last moved them there."""
        policy = self._hit_depth_policy
        order = getattr(policy, "evictable_blocks", None)
        if order is None:
            return None
        moved = getattr(policy, "moved", {})
        tails = {key for keys in self._tail_keys.values() for key in keys}
        kinds: list[str] = []
        deepest = -1
        for index, key in enumerate(reversed(order)):  # newest first
            owners = self._park_owner.get(key, ())
            records = [self._parked[context] for context in owners if context in self._parked]
            if any(record["state"] == "parked" for record in records):
                state = "parked"
                deepest = index
            elif any(key in record.get("checkpoints", ()) for record in records):
                state = "checkpoint"
            elif key in tails:
                state = "tail"
            elif records:
                state = "running-copy"
            else:
                state = "other"
            kinds.append(f"{state}/{moved.get(key, ('unknown', 0.0))[0]}")
        newer = collections.Counter(kinds[:deepest]) if deepest > 0 else collections.Counter()
        return {"size": len(kinds), "deepest_parked": deepest, "newer": dict(newer.most_common())}

    def _account_departures(self, pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Count the hashed blocks that left the free queue since the last
        step, by what their leaving cost.

        `reused`: taken back by a prefix hit, content still on the GPU.
        `hosted`: evicted, but the host holds it. `passed`: evicted, a
        checkpoint the band skips on purpose. `dead`: evicted, a block the
        engine marked dead, which the band skips too. `uncopied`: evicted from
        inside the band without the host having it — the copy lost the race.
        `unseen`: evicted from beyond the band, which never looked at it.
        The whole queue is walked, which is bounded by the pool.
        """
        queue = pool.free_block_queue
        trace = self._trace.enabled
        now = time.time()
        current: dict[int, tuple[typing.Any, int]] = {}
        block = queue.fake_free_list_head.next_free_block
        position = 0
        while block is not None and block is not queue.fake_free_list_tail:
            if block.block_hash is not None:
                current[block.block_id] = (block.block_hash, position)
            position += 1
            block = block.next_free_block
        policy = getattr(self.manager, "_policy", None)
        for block_id, (block_hash, position) in self._queue_snapshot.items():
            if current.get(block_id, (None, 0))[0] == block_hash:
                continue
            group_idx = _kv_cache_utils.get_group_id(block_hash)
            if pool.cached_block_hash_to_block.get_one_block(block_hash) is not None:
                kind = "reused"
            else:
                key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(block_hash), group_idx)
                if policy is not None and policy.get(key) is not None:
                    kind = "hosted"
                elif self._retention is not None and self._retention.is_passed(key):
                    kind = "passed"
                elif key in self._dead_keys:
                    kind = "dead"
                elif position < self._spill_band_blocks:
                    kind = "uncopied"
                else:
                    kind = "unseen"
            # Totals and per group: whether the recurrent groups' blocks
            # leave the pool ahead of the attention group's is the order
            # that produces the divergent hit.
            self._departures[kind] += 1
            self._departures[f"{kind}_g{group_idx}"] += 1
            self._sample_counters[kind] += 1
            if kind == "reused":
                if trace:
                    self._queue_entered.pop(block_id, None)
            else:
                # A CURRENT checkpoint leaving the GPU is the event that
                # makes a hit divergent; one leaving within a few steps of
                # being re-aged says the re-aging is not holding it.
                key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(block_hash), group_idx)
                if self._retention is not None and self._retention.is_current(key):
                    self._departures["current_ckpt"] += 1
                reaged_step = self._reaged_at.pop(block_id, None)
                if reaged_step is not None and reaged_step > self._step - REAGE_RECENT_STEPS:
                    self._departures["reaged_recently"] += 1
                if trace:
                    self._trace_evict(block_id, block_hash, group_idx, kind, position, now)
        if trace:
            for block_id in current:
                if block_id not in self._queue_snapshot:
                    self._queue_entered[block_id] = now
        self._step += 1
        self._queue_snapshot = current

    def _note_block(self, block_id: int, role: str) -> None:
        """Remember what last touched a block, for the crash report only."""
        if len(self._recent_block_roles) > 4096:
            self._recent_block_roles.clear()
        self._recent_block_roles[block_id] = role

    def fill_divergent_hit(
        self, request: typing.Any, num_computed_tokens: int, lagging_group_ids: tuple[int, ...]
    ) -> bool:
        """The fork's hook for the divergent hit: the GPU holds the attention
        prefix up to `num_computed_tokens` but not every recurrent group's
        state at that boundary. Yes when the tier holds each lagging group's
        checkpoint there — the block the finish-time store put in it for
        exactly this — and it is ready to load; the load itself is built
        when the scheduler allocates the boundary blocks. No for a group
        without a single state (nothing to fill), a request with a job in
        flight (the load must be its only one), or a key the tier lacks.
        """
        c = self._lookup_audit.counts
        c["fill_asked"] += 1
        req_status = self._req_status.get(request.request_id)
        policy = getattr(self.manager, "_policy", None)
        configs = self.config.kv_group_configs
        if req_status is None or policy is None or not configs or req_status.transfer_jobs:
            c["fill_refused_state"] += 1
            return False
        hashes = request.block_hashes
        keys: list[_offload_base.OffloadKey] = []
        for group_idx in sorted(lagging_group_ids):
            group_config = configs[group_idx]
            # One block per group is what the fill loads, so only a group
            # whose whole state at a boundary is one block: a single-state
            # group's window is one chunk. Full attention has no window; a
            # wider window would need the whole window loaded, and is refused.
            if group_config.sliding_window_size_in_chunks != 1:
                c["fill_refused_group"] += 1
                return False
            chunk = num_computed_tokens // group_config.tokens_per_chunk - 1
            index = (chunk + 1) * group_config.hashes_per_chunk - 1
            if chunk < 0 or index >= len(hashes):
                c["fill_refused_chain"] += 1
                return False
            key = _offload_base.make_offload_key(hashes[index], group_idx)
            block = policy.get(key)
            if block is None or not block.is_ready:
                c["fill_refused_absent" if block is None else "fill_refused_pending"] += 1
                return False
            keys.append(key)
        self._fills[request.request_id] = (num_computed_tokens, tuple(sorted(lagging_group_ids)), keys)
        c["fill_accepted"] += 1
        pending = self._trace_lookups.get(request.request_id)
        if pending is not None:
            pending["fill"] = list(self._fills[request.request_id][1])
        return True

    def _load_fill(self, request: typing.Any, blocks: typing.Any) -> None:
        """Turn an accepted fill into the load job: one destination block per
        lagging group — the last block of the group, which the engine
        allocated fresh for the boundary — and the tier's copy of that
        group's checkpoint as the source. Registered on the request like
        any load, so the stock completion path marks it received."""
        num_computed_tokens, group_ids, keys = self._fills.pop(request.request_id)
        req_status = self._req_status[request.request_id]
        dst_block_ids: list[int] = []
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_idx in range(len(self.config.kv_group_configs)):
            if group_idx not in group_ids:
                group_sizes.append(0)
                block_indices.append(0)
                continue
            group_blocks = blocks.blocks[group_idx]
            block = group_blocks[-1] if group_blocks else None
            # The engine allocates the boundary block fresh, unhashed. Any
            # other shape means the scheduler did not take the fill path for
            # this allocation, so there is nothing to load into: say so once
            # per run and leave the request to the path it did take.
            if block is None or block.is_null or block.block_hash is not None:
                self._lookup_audit.counts["fill_shape_mismatch"] += 1
                if not self._fill_mismatch_warned:
                    self._fill_mismatch_warned = True
                    logger.warning(
                        "victim offload: fill for %s found no fresh boundary block in group %d (%s); not loading",
                        request.request_id[:8],
                        group_idx,
                        block,
                    )
                return
            dst_block_ids.append(block.block_id)
            group_sizes.append(1)
            block_indices.append(len(group_blocks) - 1)
            self._note_block(block.block_id, f"fill-dst:{request.request_id[:8]}")
        src_spec = self.manager.prepare_load(keys, req_status.req_context)
        dst_spec = _offload_base.GPULoadStoreSpec(dst_block_ids, group_sizes=group_sizes, block_indices=block_indices)
        job_id = self._generate_job_id()
        self._current_batch_load_jobs[job_id] = _offload_common.TransferJob(
            req_id=request.request_id, src_spec=src_spec, dst_spec=dst_spec
        )
        req_status.transfer_jobs.add(job_id)
        self._jobs[job_id] = _offload_sched.TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys),
            is_store=False,
        )
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys)
        self._fills_loaded += 1
        self._sample_counters["filled"] += 1
        if self._trace.enabled:
            self._trace.write(
                "fill",
                step=self._step,
                req=request.request_id,
                boundary=num_computed_tokens // self.config.kv_group_configs[0].tokens_per_chunk,
                groups=list(group_ids),
                blocks=dst_block_ids,
            )

    def update_state_after_alloc(self, request: typing.Any, blocks: typing.Any, num_external_tokens: int) -> None:
        """Stock behaviour, plus the divergent-hit fill's load and a
        post-mortem when the stock invariant fails.

        A fill arrives here with no external tokens (the hit is local; only
        the lagging groups' boundary blocks are loaded), which is the case
        the stock path returns from at once.

        The invariant — every block inside a matched prefix still carries its
        hash — can fail under heavy external-load traffic with this connector
        mutating no pool state at all. The bare assert says nothing about
        which block or what last touched it; this does, before re-raising.
        """
        if self._park:
            self._park_running(request)
        if self._park and num_external_tokens > 0:
            super().update_state_after_alloc(request, blocks, num_external_tokens)
            self._park_loading(request, num_external_tokens)
            return
        if num_external_tokens == 0 and request.request_id in self._fills:
            self._load_fill(request, blocks)
            return
        try:
            super().update_state_after_alloc(request, blocks, num_external_tokens)
        except AssertionError:
            try:
                for group_idx, group_blocks in enumerate(blocks.blocks):
                    bad = [
                        (i, b.block_id, self._recent_block_roles.get(b.block_id, "?"))
                        for i, b in enumerate(group_blocks)
                        if not b.is_null and b.block_hash is None
                    ]
                    logger.error(
                        "victim offload: alloc-invariant post-mortem req=%s group=%d computed=%s external=%d unhashed=%s",
                        request.request_id[:8],
                        group_idx,
                        getattr(request, "num_computed_tokens", "?"),
                        num_external_tokens,
                        bad[:8],
                    )
            except Exception:
                logger.exception("alloc-invariant post-mortem itself failed")
            raise

    def _under_watermark(self, pool: "vllm.v1.core.block_pool.BlockPool") -> bool:
        """Whether the pool is low enough on free blocks that eviction is near."""
        return pool.get_num_free_blocks() <= int(pool.num_gpu_blocks * self._spill_watermark)

    def _pressured(self, pool: "vllm.v1.core.block_pool.BlockPool") -> bool:
        """Under the watermark now, or recently enough that it still counts."""
        return self._under_watermark(pool) or time.monotonic() < self._pressure_until

    def request_finished_with_blocks(
        self, request: typing.Any, block_id_groups: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, typing.Any] | None]:
        """Store what a finishing request holds that the band will never see.

        Two cases. A block the request SHARES with others (a common prompt
        prefix) never reaches the free queue while any sharer runs — on a
        host that never preempts, never at all — so the band, which mirrors
        the eviction front, cannot copy it; and a lookup that starts at it
        fails there before anything else is consulted. Those are stored at
        every finish, once, while the sharers keep them allocated.

        And under the watermark, a freed block is reused within a step —
        before any copy can land — so the request's remaining blocks are
        stored while the engine holds them, until the worker reports the
        request sent. That hold is the `True` in the return value.
        """
        if self._park:
            return self._park_finish(request, block_id_groups)
        self._classify_checkpoints(request, block_id_groups)
        self._fills.pop(request.request_id, None)
        held = False
        pool = self._gpu_block_pool
        req_status = self._req_status.get(request.request_id)
        summary: dict[str, typing.Any] = {}
        if pool is not None:
            summary = self._store_at_finish(request, block_id_groups, pool)
        if (
            pool is not None
            and self._pressured(pool)
            and req_status is not None
            # The base invariant: a request carries one load or only stores.
            # A load in flight at finish (an abort mid-load) is left alone.
            and all(self._jobs[jid].is_store for jid in req_status.transfer_jobs)
        ):
            held = self._build_finish_store(request, block_id_groups, req_status)
        if self._trace.enabled:
            self._trace_finish(request, summary, held)
        # Base bookkeeping runs either way; with the job already registered on
        # the request it keeps req_status alive for the completion.
        finished, params = super().request_finished(request)
        return held or finished, params

    def _classify_checkpoints(self, request: typing.Any, block_id_groups: tuple[list[int], ...] | None) -> None:
        """Tell the host policy which of a request's single-state checkpoints
        are passed.

        At every lookup and every finish, pressured or not: the class of a
        key is a fact about the request's history, and a checkpoint that was
        current at the last turn's finish is passed at this one. At lookup
        the whole chain is read (no block ids yet); at finish the chain is
        cut to the blocks the request held. Current means the last two full
        boundaries and the first. The last is the one the next lookup of
        this context asks for, since the lookup scans from the end; the one
        before it is the fallback for when the last was completed during
        decode, which the engine never caches, so the tier can never hold
        it and the scan must land one boundary earlier. The first is what a
        compaction returns to. Everything between is passed, whatever block
        the engine still holds for it: which blocks it keeps at finish is its
        own bookkeeping (the state is carried forward a block at a time), not
        a statement about which boundary is worth a slot.

        A lookup only promotes; only a finish demotes. A continuation's
        chain runs past what has been computed — a turn's reply and the
        next observation can add two full chunks — and the last two
        boundaries of THAT chain have no checkpoint anywhere yet. Reading
        them as current would make the previous turn's newest checkpoint
        passed at the very lookup that asks for it, and passed keys are the
        first the tier evicts, so a lookup that waits on a pending store
        would come back to find its boundary gone and the whole context
        recomputing. The finish knows which boundaries exist, so the
        demotion waits for it.
        """
        if self._retention is None:
            return
        hashes = list(request.block_hashes)
        passed: list[_offload_base.OffloadKey] = []
        current: list[_offload_base.OffloadKey] = []
        for group_idx, group_config in enumerate(self.config.kv_group_configs):
            if not group_config.hashes_per_chunk:
                continue  # a transient group the tier passes over
            if group_config.sliding_window_size_in_chunks is None:
                continue
            factor = group_config.hashes_per_chunk
            group_hashes = hashes[factor - 1 :: factor]
            block_ids: list[int] = []
            if block_id_groups is not None:
                block_ids = block_id_groups[group_idx] if group_idx < len(block_id_groups) else []
                group_hashes = group_hashes[: len(block_ids)]
            last = len(group_hashes) - 1
            for i, block_hash in enumerate(group_hashes):
                key = _offload_base.make_offload_key(block_hash, group_idx)
                (current if i in (0, last - 1, last) else passed).append(key)
            if block_id_groups is not None and self._finish_patterns_logged < FINISH_PATTERNS_TO_LOG:
                self._finish_patterns_logged += 1
                nonzero_at = [i for i, b in enumerate(block_ids) if b != 0][:12]
                self._trace.write(
                    "pattern",
                    step=self._step,
                    req=request.request_id[:8],
                    g=group_idx,
                    full_blocks=len(group_hashes),
                    block_ids=len(block_ids),
                    nonzero_at=nonzero_at,
                )
                if self._engine_log:
                    logger.info(
                        "victim offload: finish pattern req=%s group=%d full_blocks=%d block_ids=%d nonzero_at=%s",
                        request.request_id[:8],
                        group_idx,
                        len(group_hashes),
                        len(block_ids),
                        nonzero_at,
                    )
        if passed and block_id_groups is not None:
            self._retention.mark_passed(passed)
        if current:
            self._retention.mark_current(current)

    def _store_at_finish(
        self, request: typing.Any, block_id_groups: tuple[list[int], ...], pool: "vllm.v1.core.block_pool.BlockPool"
    ) -> dict[str, typing.Any]:
        """Queue a store of what the finishing request holds that the band
        cannot be relied on for, and re-age its checkpoints on the GPU.

        Two kinds of block. Its SHARED blocks (a common prompt prefix, held
        by other requests too) never reach the free queue while a sharer
        runs, so the band never sees them. And its LAST TWO recurrent
        checkpoints: the engine freed those a turn before the attention they
        belong with, so on a pool that evicts block by block they go first,
        leaving an attention prefix on the GPU with no state at its boundary
        — the divergent hit, which the connector protocol cannot serve
        (it is only asked for chunks beyond the attention hit, and the first
        of those is the one the previous turn never finished computing).
        Storing the checkpoints now makes a partially evicted context
        servable: asked from where the attention ends, the walk finds the
        evicted chunks and the checkpoint within their range. Re-aging their
        GPU blocks at the next step, after the engine has freed the
        attention, carries the keep-young pass (`_keep_checkpoints_young`)
        over the gap between this request leaving the connector's books and
        its continuation arriving.

        A victim job like the band's, validated by hash at completion, since
        the pool may reuse any of these blocks before the copy lands.
        Returns, for the trace, the checkpoint chunks found per group and
        what was stored.
        """
        hashes = list(request.block_hashes)
        policy = getattr(self.manager, "_policy", None)
        num_groups = len(self.config.kv_group_configs)
        by_group: list[list[tuple[typing.Any, _offload_base.OffloadKey]]] = [[] for _ in range(num_groups)]
        claimed: set[_offload_base.OffloadKey] = set()
        checkpoint_keys: set[_offload_base.OffloadKey] = set()
        summary: dict[str, typing.Any] = {"ckpt": {}, "shared": 0, "stored": 0}

        def offer(block: typing.Any, block_hash: bytes, group_idx: int) -> None:
            key = _offload_base.make_offload_key(block_hash, group_idx)
            if key in claimed or (self._retention is not None and self._retention.is_passed(key)):
                return
            if policy is not None and policy.get(key) is not None:
                return
            claimed.add(key)
            by_group[group_idx].append((block, key))

        for group_idx, group_config in enumerate(self.config.kv_group_configs):
            if not group_config.hashes_per_chunk:
                continue  # a transient group the tier passes over
            factor = group_config.hashes_per_chunk
            block_ids = block_id_groups[group_idx] if group_idx < len(block_id_groups) else []
            group_hashes = hashes[factor - 1 :: factor][: len(block_ids)]
            for i, block_hash in enumerate(group_hashes):
                block_id = block_ids[i]
                # 0 is the padding sentinel; an id past the pool is a fake's.
                if block_id == 0 or block_id >= len(pool.blocks):
                    continue
                block = pool.blocks[block_id]
                # Shared means a reference beyond this request's own.
                if block.ref_cnt <= 1 or block.block_hash is None:
                    continue
                offer(block, block_hash, group_idx)
            if group_config.sliding_window_size_in_chunks is None:
                continue
            # The newest two checkpoints the GPU still has: already freed by
            # the engine, so found by hash rather than by the request's ids.
            # None at all is a continuation that will recompute its prompt.
            found = self._recent_checkpoints(pool, hashes, group_idx, factor, len(group_hashes), want=2)
            if group_hashes and not found:
                self._checkpoints_gone += 1
            summary["ckpt"][group_idx] = [chunk for chunk, _, _, _ in found]
            for _, chain_hash, with_group, block in found:
                # Re-aged whoever stores them: keep-young is about the GPU
                # copy outliving the attention it belongs with.
                self._reage_pending.append(with_group)
                checkpoint_keys.add(_offload_base.make_offload_key(chain_hash, group_idx))
                offer(block, chain_hash, group_idx)
        ordered = [(block, key) for group in by_group for block, key in group]
        if not ordered:
            return summary
        store_output = self._offer_store([key for _, key in ordered], self._victim_req_context)
        if store_output is None:
            return summary
        keys_to_store = set(store_output.keys_to_store)
        if not keys_to_store:
            return summary
        spilled: list[tuple[int, typing.Any, _offload_base.OffloadKey]] = []
        src_block_ids: list[int] = []
        group_sizes: list[int] = []
        for group in by_group:
            count = 0
            for block, key in group:
                if key not in keys_to_store:
                    continue
                spilled.append((block.block_id, block.block_hash, key))
                src_block_ids.append(block.block_id)
                count += 1
                if key in checkpoint_keys:
                    self._checkpoints_stored += 1
                    summary["stored"] += 1
                else:
                    self._shared_stored += 1
                    summary["shared"] += 1
            group_sizes.append(count)
        job_id = self._generate_job_id()
        self._victim_job_ids.add(job_id)
        self._spilled[job_id] = spilled
        self._blocks_spilled += len(spilled)
        self._sample_counters["spilled"] += len(spilled)
        if self._trace.enabled:
            self._trace.write("store", step=self._step, origin="finish", blocks=len(spilled), groups=group_sizes)
        self._jobs[job_id] = _offload_sched.TransferJobStatus(
            req_id=VICTIM_REQ_ID,
            pending_count=self.config.num_workers,
            keys=keys_to_store,
            is_store=True,
        )
        self._pending_finish_jobs[job_id] = _offload_common.TransferJob(
            req_id=VICTIM_REQ_ID,
            src_spec=_offload_base.GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=[0] * num_groups
            ),
            dst_spec=store_output.store_spec,
        )
        return summary

    def _build_finish_store(
        self, request: typing.Any, block_id_groups: tuple[list[int], ...], req_status: typing.Any
    ) -> bool:
        """Queue a store of the request's not-yet-hosted blocks; True if any.

        Keys come from the request's own hash chain and pair positionally with
        its block ids — one offloaded block per GPU block, which __init__
        guarantees. Group-major, because the worker pairs sources with slots
        in that order.

        The chain covers the token sequence, not the computed KV: a response
        that ends exactly on a block boundary has a hash for a chunk whose
        last token was sampled and never went through the model, and a
        request aborted mid-prefill has hashes for chunks never computed at
        all. So a chunk is offered only when every one of its tokens is
        computed AND the GPU block still carries that chunk's hash under this
        group — the engine hashes a block only once it is complete, so an
        unhashed or re-hashed block is not the content the key names. Held
        blocks are not reused before the copy lands, so the check at the
        offer is the check at completion.
        """
        hashes = list(request.block_hashes)
        pool = self._gpu_block_pool
        computed = int(getattr(request, "num_computed_tokens", 0))
        per_group: list[list[tuple[int, _offload_base.OffloadKey]]] = []
        claimed: set[_offload_base.OffloadKey] = set()
        # Why a chunk the request holds was not offered. Every arm of the loop
        # below except the offer itself lands in one of these, so the census
        # accounts for the whole chain: what the hold does not store
        # is otherwise invisible, and a store that quietly stops is what a
        # version change breaks first.
        census: collections.Counter[str] = collections.Counter()
        for group_idx, group_config in enumerate(self.config.kv_group_configs):
            if not group_config.hashes_per_chunk:
                per_group.append([])  # a transient group the tier passes over, in its place
                continue
            factor = group_config.hashes_per_chunk
            block_ids = block_id_groups[group_idx] if group_idx < len(block_id_groups) else []
            group_hashes = hashes[factor - 1 :: factor]
            rows: list[tuple[int, _offload_base.OffloadKey]] = []
            for i, block_hash in enumerate(group_hashes):
                if i >= len(block_ids) or (i + 1) * group_config.tokens_per_chunk > computed:
                    census["short_ids" if i >= len(block_ids) else "short_computed"] += len(group_hashes) - i
                    break
                block_id = block_ids[i]
                # 0 is the padding sentinel for sliding-window/SSM positions.
                if block_id == 0:
                    census["null_id"] += 1
                    continue
                if pool is None or block_id >= len(pool.blocks):
                    census["off_pool"] += 1
                    continue
                gpu_hash = pool.blocks[block_id].block_hash
                if (
                    gpu_hash is None
                    or _kv_cache_utils.get_block_hash(gpu_hash) != block_hash
                    or _kv_cache_utils.get_group_id(gpu_hash) != group_idx
                ):
                    census["unverified"] += 1
                    continue
                key = _offload_base.make_offload_key(block_hash, group_idx)
                if key in claimed:
                    census["claimed"] += 1
                    continue
                claimed.add(key)
                rows.append((block_id, key))
            per_group.append(rows)
        flat = [row for rows in per_group for row in rows]
        census["offered"] = len(flat)
        # Each becomes a hold_<reason> counter; "unverified" is the one the
        # trace already reported, and the census is now its only source.
        for reason, count in census.items():
            self._lookup_audit.counts[f"hold_{reason}"] += count
        if not flat:
            return False

        store_output = self._offer_store([key for _, key in flat], req_status.req_context)
        if store_output is None:
            return False
        keys_to_store = set(store_output.keys_to_store)
        # Blocks the host already has are about to be reused, which is a use.
        present = [key for _, key in flat if key not in keys_to_store]
        if present:
            self.manager.touch(list(reversed(present)), req_status.req_context)
        if not keys_to_store:
            return False

        src_block_ids: list[int] = []
        group_sizes: list[int] = []
        final_keys: list[_offload_base.OffloadKey] = []
        for rows in per_group:
            count = 0
            for block_id, key in rows:
                if key not in keys_to_store:
                    continue
                src_block_ids.append(block_id)
                final_keys.append(key)
                count += 1
            group_sizes.append(count)

        job_id = self._generate_job_id()
        # An ordered container where the base expects a set: complete_store
        # marks keys evictable in iteration order, and that order is the host
        # LRU order for the batch.
        self._jobs[job_id] = _offload_sched.TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=typing.cast("set[_offload_base.OffloadKey]", final_keys),
            is_store=True,
        )
        req_status.transfer_jobs.add(job_id)
        self._held_finish_jobs.add(job_id)
        self._pending_finish_jobs[job_id] = _offload_common.TransferJob(
            req_id=request.request_id,
            src_spec=_offload_base.GPULoadStoreSpec(
                src_block_ids,
                group_sizes=group_sizes,
                block_indices=[0] * len(self.config.kv_group_configs),
            ),
            dst_spec=store_output.store_spec,
        )
        for bid in src_block_ids:
            self._note_block(bid, f"held-src:{job_id}")
        self._blocks_held += len(final_keys)
        self._sample_counters["held"] += len(final_keys)
        if self._trace.enabled:
            self._trace.write("store", step=self._step, origin="hold", blocks=len(final_keys))
        return True

    def update_connector_output(self, connector_output: "vllm.v1.outputs.KVConnectorOutput") -> None:
        """Settle victim stores here, then let the base class settle its own.

        The base class resolves a completed job through the request that owns
        it; a victim store owns no request, so its completions are taken out of
        the batch before they reach it.
        """
        meta = connector_output.kv_connector_worker_meta
        loaded = self._completed_load_keys(meta) if isinstance(meta, _offload_common.OffloadingWorkerMetadata) else []
        if isinstance(meta, _offload_common.OffloadingWorkerMetadata) and (self._victim_job_ids):
            for job_id in list(meta.completed_jobs):
                if job_id not in self._victim_job_ids:
                    continue
                # A completion from before a cache reset names a job the reset
                # discarded; the stock handler drops it by the threshold, and
                # so does this one.
                if job_id < self._stale_job_threshold or job_id not in self._jobs:
                    meta.completed_jobs.pop(job_id)
                    self._victim_job_ids.discard(job_id)
                    self._spilled.pop(job_id, None)
                    self._release_spill_hold(job_id)
                    continue
                count = meta.completed_jobs.pop(job_id)
                job_status = self._jobs[job_id]
                job_status.pending_count -= count
                if job_status.pending_count > 0:
                    continue
                # A victim is unreferenced, so the pool was free to hand it out
                # and overwrite it while the copy was in flight. A block that no
                # longer carries the hash it was read for was not the block that
                # was read, so the whole job's content is suspect and the
                # manager drops what it was holding for it.
                spilled = self._spilled.pop(job_id, [])
                pool = self._gpu_block_pool
                # A block the pool reused before the copy landed no longer
                # carries the hash it was read under, and what reached the
                # host for it is some mixture of two blocks. Each block is
                # judged on its own: the rest of the job copied fine, and
                # discarding them with it would throw away a whole batch for
                # one reuse — under the churn of a cliff, most batches. This
                # check is the only thing that stands between a raced copy
                # and a served prefix: the engine's flush-on-reuse, which
                # waits for a registered store before overwriting its source,
                # is not relied on, since copies it vouched for have differed
                # from what was read (kv_offload.md, Integrity); nothing a
                # hash did not vouch for is kept.
                intact = [
                    key
                    for block_id, block_hash, key in spilled
                    if pool is not None and pool.blocks[block_id].block_hash == block_hash
                ]
                reused = [key for _, _, key in spilled if key not in set(intact)]
                if intact:
                    # In spill order: the manager marks each evictable as it
                    # goes, so this order is the host tier's LRU order for
                    # the batch, and the job's own set would give none.
                    self.manager.complete_store(intact, self._victim_req_context, success=True)
                if reused:
                    self.manager.complete_store(reused, self._victim_req_context, success=False)
                    self._lookup_audit.note_dropped(reused)
                    self._blocks_dropped += len(reused)
                    self._jobs_with_drops += 1
                    self._sample_counters["dropped"] += len(reused)
                    if self._trace.enabled:
                        self._trace.write("drop", step=self._step, blocks=len(spilled), dropped=len(reused))
                self._release_spill_hold(job_id)
                del self._jobs[job_id]
                self._victim_job_ids.discard(job_id)
        super().update_connector_output(connector_output)
        if self._park:
            self._park_settle(loaded)
        # After the base class has marked them evictable — at the recent end,
        # as a use — the blocks now on the GPU go to the eviction end instead.
        if loaded and self._demote is not None:
            self._demote(self._park_unneeded(loaded))
        self._maybe_log_tally()

    def _release_spill_hold(self, job_id: int) -> None:
        """Let go of the blocks a band store held while its copy was in
        flight: back to the free queue's young end."""
        block_ids = self._spill_holds.pop(job_id, None)
        pool = self._gpu_block_pool
        if block_ids and pool is not None:
            pool.free_blocks([pool.blocks[block_id] for block_id in block_ids])

    def _completed_load_keys(self, meta: _offload_common.OffloadingWorkerMetadata) -> list[_offload_base.OffloadKey]:
        """Keys of the load jobs this report brings to completion.

        Read before the base class settles them, since it forgets a job as it
        completes it.
        """
        keys: list[_offload_base.OffloadKey] = []
        for job_id, count in meta.completed_jobs.items():
            if job_id in self._victim_job_ids or job_id < self._stale_job_threshold:
                continue
            job = self._jobs.get(job_id)
            if job is not None and not job.is_store and job.pending_count - count <= 0:
                keys.extend(job.keys)
        return keys

    def _build_partial_tail_store_jobs(
        self, scheduler_output: "vllm.v1.core.sched.output.SchedulerOutput"
    ) -> dict[int, _offload_common.TransferJob]:
        """Decline the engine's own boundary-state stores.

        The stock scheduler turns every recurrent boundary the engine hands it
        into a store job, keyed exactly as this connector's finish-time
        checkpoint store keys them, so running both would put the same block
        in the tier twice. Which of the two runs was measured, and this one
        wins on three counts (`kv_offload.md`, "The engine's own boundary
        store"): its copies are checked against the hash the block was read
        under, where the engine's rest on the fence this connector does not
        trust; it stores at finish rather than as the block is cached, so the
        copy has less of the tier's turnover to survive; and it stores the
        boundaries a continuation resumes from rather than every boundary
        computed.
        """
        return {}

    def build_connector_meta(
        self, scheduler_output: "vllm.v1.core.sched.output.SchedulerOutput"
    ) -> _connector_base.KVConnectorMetadata:
        """Stock metadata, then the trace's view of the step: which lookups
        were scheduled and at what hit, what was resumed or preempted, and
        once a second the state of the pool."""
        # Named by _build_store_jobs while super() runs; empty if it does not.
        self._held_this_step = frozenset()
        meta = super().build_connector_meta(scheduler_output)
        if self._trace.enabled:
            self._trace_schedule(scheduler_output)
        assert isinstance(meta, _offload_common.OffloadingConnectorMetadata)
        return HeldStoreMetadata(
            load_jobs=meta.load_jobs,
            store_jobs=meta.store_jobs,
            jobs_to_flush=meta.jobs_to_flush,
            held_job_ids=self._held_this_step & frozenset(meta.store_jobs),
        )

    def _class_name(self, key: _offload_base.OffloadKey, group_idx: int) -> str:
        """The trace's class of a block: attention, a current or passed
        checkpoint, or a recurrent block of no known class."""
        configs = self.config.kv_group_configs
        if group_idx < len(configs) and configs[group_idx].sliding_window_size_in_chunks is None:
            return "att"
        if self._retention is not None:
            if self._retention.is_current(key):
                return "cur"
            if self._retention.is_passed(key):
                return "pas"
        return "unk"

    def _trace_lookup(
        self, request: typing.Any, num_computed_tokens: int, ext: int | None, zero: dict[str, typing.Any] | None
    ) -> None:
        """Record a lookup: the chain, the attention hit the scheduler
        reported, each recurrent group's newest boundary still cached on the
        GPU at or below it, the tier's answer, and — when the previous turn
        of this context finished recently — which request that was, how long
        it ran and how long ago it finished. Held until the request is
        scheduled, when the settled hit joins it."""
        configs = self.config.kv_group_configs
        if not configs:
            return
        factor = configs[0].hashes_per_chunk
        hashes = request.block_hashes
        n_full = len(hashes) // factor
        start = num_computed_tokens // configs[0].tokens_per_chunk
        previous = self._trace_lookups.get(request.request_id)
        fields: dict[str, typing.Any] = {
            "step": self._step,
            "req": request.request_id,
            "tokens": getattr(request, "num_tokens", None),
            "chunks": n_full,
            "att_hit": start,
            "ext": ext,
            "attempts": (previous["attempts"] + 1) if previous else 1,
        }
        pool = self._gpu_block_pool
        if pool is not None:
            for group_idx, group_config in enumerate(configs):
                if not group_config.hashes_per_chunk:
                    continue  # a transient group the tier passes over
                if group_config.sliding_window_size_in_chunks is None:
                    continue
                found = self._recent_checkpoints(
                    pool, hashes, group_idx, group_config.hashes_per_chunk, start, want=1, probe=TRACE_PROBE_CHUNKS
                )
                fields[f"rec_g{group_idx}"] = found[0][0] + 1 if found else -1
        if zero is not None:
            fields.update(zero)
        now = time.time()
        for chunk in range(n_full - 1, max(n_full - TRACE_PROBE_CHUNKS, 0) - 1, -1):
            turn = self._finished_turns.get(bytes(hashes[(chunk + 1) * factor - 1]))
            if turn is None:
                continue
            fields["prev_req"] = turn["req"]
            fields["prev_turn_s"] = turn["turn_s"]
            fields["gap_s"] = round(now - turn["t"], 2)
            fields["new_chunks"] = n_full - turn["n_full"]
            break
        self._trace_lookups[request.request_id] = fields

    def _trace_schedule(self, scheduler_output: "vllm.v1.core.sched.output.SchedulerOutput") -> None:
        """Write the lookups the scheduler settled this step, with the hit it
        settled on; resumptions and preemptions; and the sample when due."""
        for new in scheduler_output.scheduled_new_reqs:
            fields = self._trace_lookups.pop(new.req_id, None) or {"req": new.req_id}
            fields["computed"] = new.num_computed_tokens
            if fields.get("tokens") is not None:
                fields["recomputed"] = fields["tokens"] - new.num_computed_tokens
            fields["sched_step"] = self._step
            self._trace.write("lookup", **fields)
        cached = scheduler_output.scheduled_cached_reqs
        resumed = getattr(cached, "resumed_req_ids", None) or ()
        for i, req_id in enumerate(cached.req_ids):
            if req_id in resumed:
                self._trace.write("resume", step=self._step, req=req_id, computed=cached.num_computed_tokens[i])
        for req_id in scheduler_output.preempted_req_ids or ():
            self._trace.write("preempt", step=self._step, req=req_id)
        now = time.monotonic()
        if now - self._last_sample_at >= TRACE_SAMPLE_INTERVAL_S:
            self._last_sample_at = now
            self._trace_sample(scheduler_output)

    def _trace_sample(self, scheduler_output: "vllm.v1.core.sched.output.SchedulerOutput") -> None:
        """One line on the pool: free and used blocks, what the free queue
        and its front band are made of by class, the requests known, the
        keep-young gauges, this step's schedule, and everything that moved
        since the previous sample."""
        pool = self._gpu_block_pool
        if pool is None:
            return
        whole: collections.Counter[str] = collections.Counter()
        band: collections.Counter[str] = collections.Counter()
        for block_hash, position in self._queue_snapshot.values():
            group_idx = _kv_cache_utils.get_group_id(block_hash)
            key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(block_hash), group_idx)
            cls = self._class_name(key, group_idx)
            whole[cls] += 1
            if position < self._spill_band_blocks:
                band[cls] += 1
        free = pool.get_num_free_blocks()
        preempted = scheduler_output.preempted_req_ids
        self._trace.write(
            "step",
            step=self._step,
            free=free,
            used=pool.num_gpu_blocks - free,
            known=len(self._req_status),
            kept=self._kept_last_step,
            missing=self._kept_missing_last_step,
            gone=self._checkpoints_gone,
            new=len(scheduler_output.scheduled_new_reqs),
            fin=len(scheduler_output.finished_req_ids),
            pre=len(preempted) if preempted else 0,
            tok=scheduler_output.total_num_scheduled_tokens,
            moved=dict(self._sample_counters),
            q={**whole, "unh": free - len(self._queue_snapshot)},
            band=dict(band),
        )
        self._sample_counters.clear()
        self._trace.flush()

    def _trace_evict(
        self, block_id: int, block_hash: typing.Any, group_idx: int, kind: str, position: int, now: float
    ) -> None:
        """Record a hashed block leaving the free queue unreused: its group,
        what its leaving cost, its class, where it stood, how long it had
        been queued, and — when the keep-young pass had moved it the step
        before — whose checkpoint it was."""
        key = _offload_base.make_offload_key(_kv_cache_utils.get_block_hash(block_hash), group_idx)
        fields: dict[str, typing.Any] = {
            "step": self._step,
            "g": group_idx,
            "kind": kind,
            "cls": self._class_name(key, group_idx),
            "pos": position,
        }
        entered = self._queue_entered.pop(block_id, None)
        if entered is not None:
            fields["age"] = round(now - entered, 1)
        kept = self._kept_map_prev.get(block_id)
        if kept is not None:
            fields["req"] = kept[0]
            fields["chunk"] = kept[2]
        self._trace.write("evict", **fields)

    def _trace_finish(self, request: typing.Any, summary: dict[str, typing.Any], held: bool) -> None:
        """Record a finish: the turn's length in tokens and seconds, the
        checkpoint chunks still on the GPU per recurrent group, what was
        stored, whether the request was held for its copy. The chain's
        newest full chunk is remembered so the continuation can be linked."""
        configs = self.config.kv_group_configs
        factor = configs[0].hashes_per_chunk if configs else 1
        hashes = request.block_hashes
        n_full = len(hashes) // factor
        now = time.time()
        fields: dict[str, typing.Any] = {
            "step": self._step,
            "req": request.request_id,
            "tokens": getattr(request, "num_tokens", None),
            "out": getattr(request, "num_output_tokens", None),
            "chunks": n_full,
            "held": held,
            "shared": summary.get("shared", 0),
            "stored": summary.get("stored", 0),
        }
        arrival = getattr(request, "arrival_time", None)
        if arrival is not None:
            fields["turn_s"] = round(now - arrival, 2)
        for group_idx, chunks in summary.get("ckpt", {}).items():
            fields[f"ck_g{group_idx}"] = chunks
        self._trace.write("finish", **fields)
        self._trace_lookups.pop(request.request_id, None)
        if n_full > 0:
            self._finished_turns[bytes(hashes[n_full * factor - 1])] = {
                "req": request.request_id,
                "t": now,
                "turn_s": fields.get("turn_s"),
                "n_full": n_full,
            }
            while len(self._finished_turns) > TRACE_FINISH_MEMORY:
                self._finished_turns.popitem(last=False)

    def _maybe_log_tally(self) -> None:
        """Once per TALLY_INTERVAL_S: the running counters as one `tally`
        trace event, and the same as engine-log lines when `engine_log` is on.

        The event carries the spill tally, every departure class, every
        lookup-audit counter, the keys asked and stored since the last
        tally, and the asked-depth percentiles with the policy's gauges —
        the whole of what the log lines say, so a run read from its job
        directory alone lacks nothing.
        """
        now = time.monotonic()
        if now - self._last_tally_at < TALLY_INTERVAL_S:
            return
        self._last_tally_at = now
        spill = {
            "spilled": self._blocks_spilled,
            "dropped": self._blocks_dropped,
            "jobs_with_drops": self._jobs_with_drops,
            "held": self._blocks_held,
            "shared": self._shared_stored,
            "checkpoints": self._checkpoints_stored,
            "reaged": self._reaged,
            "kept": self._kept_last_step,
            "missing": self._kept_missing_last_step,
            "gone": self._checkpoints_gone,
            "kept_stored": self._kept_stored,
            "filled": self._fills_loaded,
            "in_flight": len(self._victim_job_ids),
            "passed_skipped": self._passed_skipped,
            "dead_skipped": self._dead_skipped,
            "band_present": self._band_present,
        }
        audit = getattr(self, "_lookup_audit", None)
        lookups: dict[str, int] = dict(audit.counts) if audit is not None else {}
        samples: dict[int, dict[str, list[str]]] = {}
        if audit is not None:
            for gid in sorted(set(audit.sample_asked) | set(audit.sample_stored)):
                samples[gid] = {
                    "asked": list(audit.sample_asked.get(gid, [])),
                    "stored": list(audit.sample_stored.get(gid, [])),
                }
        depth: dict[str, typing.Any] | None = None
        if self._drain_hit_records is not None:
            self._drain_hit_records()
        depths = sorted(self._drain_hit_depths()) if self._drain_hit_depths is not None else []
        if depths:
            self._hit_depth_run_max = max(self._hit_depth_run_max, depths[-1])
            policy = self._hit_depth_policy
            depth = {
                "n": len(depths),
                "p50": depths[len(depths) // 2],
                "p90": depths[(len(depths) * 9) // 10],
                "p99": depths[(len(depths) * 99) // 100],
                "max": depths[-1],
                "run_max": self._hit_depth_run_max,
                "evictable": len(getattr(policy, "evictable_blocks", ())),
                "passed": len(getattr(policy, "passed_blocks", ())),
                "evicted_passed": int(getattr(policy, "evicted_passed", 0)),
                "evicted_current": int(getattr(policy, "evicted_current", 0)),
            }
        order_gauge = self._park_order_gauge() if self._park else None
        if order_gauge is not None:
            if depth is not None:
                depth["order"] = order_gauge
            self._park_say(
                "TIER-ORDER",
                f"deepest parked key at {order_gauge['deepest_parked']} of {order_gauge['size']}; newer than it: "
                + ",".join(f"{kind}:{count}" for kind, count in order_gauge["newer"].items()),
            )
        self._trace.write(
            "tally",
            step=self._step,
            spill=spill,
            departed=dict(self._departures),
            lookups=lookups,
            samples=samples,
            depth=depth,
        )
        if audit is not None:
            # Per interval: the first keys asked and stored SINCE the last
            # tally, not since the run began — a run's first minute is not
            # what a reader of minute forty wants to see.
            audit.sample_asked.clear()
            audit.sample_stored.clear()
        if not self._engine_log:
            return
        logger.info(
            "victim offload: %s departed %s",
            " ".join(f"{k}={v}" for k, v in spill.items()),
            " ".join(f"{k}={v}" for k, v in sorted(self._departures.items())),
        )
        if lookups:
            c = collections.Counter(lookups)
            logger.info(
                "victim offload: lookups hit=%d pending=%d miss_never_stored=%d miss_evicted=%d "
                "miss_dropped=%d miss_stored_absent=%d",
                c["hit"],
                c["hit_pending"],
                c["miss_never_stored"],
                c["miss_evicted"],
                c["miss_dropped"],
                c["miss_stored_absent"],
            )
            per_group = " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k.startswith("g") and k[1].isdigit())
            logger.info("victim offload: lookups by group %s", per_group)
            rest = " ".join(
                f"{k}={v}"
                for k, v in sorted(c.items())
                if k.startswith(("lookup_", "store_", "miss_gpu", "retry", "other", "fill_"))
            )
            logger.info("victim offload: lookup calls and stores %s", rest)
            for gid, sample in samples.items():
                logger.info(
                    "victim offload: g%d sample keys asked=%s stored=%s", gid, sample["asked"], sample["stored"]
                )
        if depth is not None:
            logger.info("victim offload: asked depth blocks %s", " ".join(f"{k}={v}" for k, v in depth.items()))

    def reset_cache(self) -> None:
        """Reset the tier as stock does, and the victim bookkeeping with it.

        Stock discards every job and raises the stale-job threshold so the
        worker's later completions are ignored; the spills, victim job ids
        and finish-holds those jobs were tracked under go with them, or a
        completion would index a job the reset already forgot.

        Refused while a held store has not yet reached the worker. The
        engine keeps a held request's blocks until the worker reports the
        request sent, and the worker reports that only for a job it was
        handed; a hold discarded before its first metadata dispatch would
        leave the request's blocks held for good. The hold goes out at the
        next `build_connector_meta`, so the caller retries a step later.
        """
        undispatched = sorted(
            {job.req_id for job_id, job in self._pending_finish_jobs.items() if job_id not in self._victim_job_ids}
        )
        if undispatched:
            raise RuntimeError(
                "victim offload: cache reset refused while held finish-stores await dispatch for "
                f"{undispatched}; retry after the next scheduler step has carried them to the worker"
            )
        super().reset_cache()
        for job_id in list(self._spill_holds):
            self._release_spill_hold(job_id)
        self._spilled.clear()
        self._victim_job_ids.clear()
        self._pending_finish_jobs.clear()
        self._held_finish_jobs.clear()
        self._fills.clear()
        # Park mode: the threads the tier held are gone with it. A park store
        # the reset discarded still holds the idle blocks it read from.
        pool = self._gpu_block_pool
        for info in self._park_jobs.values():
            if pool is not None:
                for block_id, _ in info["held"]:
                    pool.free_blocks([pool.blocks[block_id]])
        self._park_jobs.clear()
        self._park_retire = []
        if pool is not None:
            for record in self._parked.values():
                for block_id, _ in record.get("pins", ()):
                    pool.free_blocks([pool.blocks[block_id]])
            for rid in list(self._tail_pins):
                self._release_tail_pins(rid, pool)
        self._park_needs.clear()
        self._parked.clear()
        self._park_owner.clear()
        self._tail_keys.clear()
        self._tail_step_keys = set()
        if self._park:
            self._park_say("RESET", "cache reset: every parked thread dropped")

    def shutdown(self) -> None:
        """Drop in-flight spill bookkeeping before the stock teardown."""
        self._trace.close()
        if self._park_log is not None:
            self._park_say("SHUTDOWN", f"counts {dict(self._park_counts)}")
            self._park_log.close()
            self._park_log = None
        self._spilled.clear()
        self._victim_job_ids.clear()
        super().shutdown()


class VictimOffloadingWorker(_offload_worker.OffloadingConnectorWorker):
    """Worker half that reports held finish-stores back as sent requests.

    The engine keeps a finishing request's blocks allocated until the request
    id comes back through ``finished_sending``; the stock worker never emits
    it for stores, because stock never holds. Which stores are held is stated
    by the scheduler in the metadata rather than read off the job: the
    engine's own boundary-state stores carry a LIVE request's id, and
    reporting one of those as sent frees the blocks of a request that is
    still running.
    """

    def __init__(
        self,
        spec: _offload_base.OffloadingSpec,
        vllm_config: "vllm.config.VllmConfig",
        kv_cache_config: "vllm.v1.kv_cache_interface.KVCacheConfig",
    ) -> None:
        """Wrap the stock worker, adding held-store bookkeeping."""
        super().__init__(spec, vllm_config, kv_cache_config)
        # job id -> request id for stores whose blocks the engine is holding.
        self._held_jobs: dict[int, str] = {}
        # request id -> its still-pending held job ids.
        self._held_pending: dict[str, set[int]] = {}

    def prepare_store_kv(self, metadata: _offload_common.OffloadingConnectorMetadata) -> None:
        """Record which incoming stores the engine is holding blocks for."""
        held = metadata.held_job_ids if isinstance(metadata, HeldStoreMetadata) else frozenset()
        for job_id in held:
            entry = metadata.store_jobs.get(job_id)
            # The sentinel names no request the engine has ever heard of, so
            # reporting one back as sent is an assertion in its scheduler.
            if entry is None or entry.req_id == VICTIM_REQ_ID:
                continue
            self._held_jobs[job_id] = entry.req_id
            self._held_pending.setdefault(entry.req_id, set()).add(job_id)
        super().prepare_store_kv(metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Emit a held request as sent once every one of its stores landed.

        The stock handler has just marked completions into the worker meta and
        the meta is drained later in the step, so the completed ids are still
        visible here.
        """
        sending, recving = super().get_finished(finished_req_ids)
        completed = self._connector_worker_meta.completed_jobs
        sent: set[str] = set()
        for job_id in [j for j in self._held_jobs if j in completed]:
            req_id = self._held_jobs.pop(job_id)
            pending = self._held_pending.get(req_id)
            if pending is None:
                continue
            pending.discard(job_id)
            if not pending:
                del self._held_pending[req_id]
                sent.add(req_id)
        return (set(sending) | sent) if sent else sending, recving


class _VictimOffloadPromMetrics(_offload_metrics.OffloadPromMetrics):
    """The stock offloading metrics, and the pool's two shares the scheduler
    publishes parking on the GPU (`VictimOffloadingScheduler._publish_kv_shares`):
    the frontend refuses a stats key it has no definition for."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Define the stock metrics, then the two shares."""
        super().__init__(*args, **kwargs)
        shares = {
            ullm.engine_monitor.KV_REQUESTS_SHARE: "Share of the GPU KV pool the requests hold.",
            ullm.engine_monitor.KV_GPU_PARKED_SHARE: "Share of the GPU KV pool only threads parked on the GPU hold.",
        }
        for name, documentation in shares.items():
            metadata = _offload_base.OffloadingGaugeMetadata(documentation=documentation)
            self._offloading_metric_metadata[name] = metadata
            self._offloading_metric_defs[name] = self._create_metric(name, metadata)


class VictimOffloadingConnector(_offload_conn.OffloadingConnector):
    """``OffloadingConnector`` with the victim-cache scheduler and worker.

    The worker differs from stock only in reporting held finish-stores back
    as sent requests: what this changes is which blocks the scheduler hands
    it, not how they travel.
    """

    def __init__(
        self,
        vllm_config: vllm.config.VllmConfig,
        role: _connector_base.KVConnectorRole,
        kv_cache_config: vllm.v1.kv_cache_interface.KVCacheConfig,
    ) -> None:
        """Build the stock connector, then swap in the victim scheduler.

        The replaced scheduler's spec is discarded with it. A spec is inert
        until asked for a manager or a worker — it only sizes the host tier at
        construction — so the one built here costs nothing beyond the object.
        """
        super().__init__(vllm_config, role, kv_cache_config)
        # Held under its own name as well: the base class types its scheduler
        # slot as the stock class, which carries no pool binding.
        self.victim_scheduler: VictimOffloadingScheduler | None = None
        # Built the way the stock connector builds it, once: a spec is inert
        # until asked for a manager or a worker, and only one half is live in
        # any given process (scheduler or worker, never both).
        offloading_config = _offload_config.build_offloading_config(vllm_config, kv_cache_config)
        spec = _offload_factory.OffloadingSpecFactory.create_spec(offloading_config)
        # The tier's real capacity is slots, not bytes: one key holds one
        # group's chunk, and a slot sized for more than that is capacity lost.
        # Stated once at init so a run's log carries the arithmetic.
        tensors = list(getattr(kv_cache_config, "kv_cache_tensors", []) or [])
        sizing: dict[str, typing.Any] = {
            "host_slots": getattr(spec, "num_blocks", None),
            "bytes_per_slot": getattr(spec, "kv_bytes_per_chunk", None),
            "gpu_blocks": getattr(kv_cache_config, "num_blocks", None),
            "tensors": len(tensors),
            "sum_bytes": sum(int(getattr(x, "size", 0)) for x in tensors),
            "groups": len(getattr(kv_cache_config, "kv_cache_groups", []) or []),
            "tokens_per_block": list(getattr(spec, "tokens_per_block", []) or []),
        }
        if bool(dict(spec.extra_config or {}).get("engine_log", DEFAULT_ENGINE_LOG)):
            logger.info("victim offload sizing: %s", " ".join(f"{k}={v}" for k, v in sizing.items()))
        if self.connector_scheduler is not None:
            self.victim_scheduler = VictimOffloadingScheduler(spec, vllm_config, kv_cache_config)
            self.victim_scheduler._sizing = sizing
            self.connector_scheduler = self.victim_scheduler
        if self.connector_worker is not None:
            self.connector_worker = VictimOffloadingWorker(spec, vllm_config, kv_cache_config)

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """The scheduler only asks a connector to complete a divergent hit
        when the connector says it can; this one can, through
        ``fill_divergent_hit``, so without this the fill is never reached."""
        return True

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: vllm.config.VllmConfig,
        metric_types: dict[typing.Any, typing.Any],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> _offload_metrics.OffloadPromMetrics:
        """The stock offloading metrics with the pool's shares beside them."""
        return _VictimOffloadPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    def bind_gpu_block_pool(self, gpu_block_pool: "vllm.v1.core.block_pool.BlockPool") -> None:
        """Pass the engine's GPU pool through to the scheduler half."""
        if self.victim_scheduler is not None:
            self.victim_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def request_finished(self, request: typing.Any, block_ids: list[int]) -> tuple[bool, dict[str, typing.Any] | None]:
        """Route a single-group finish through the block-holding path."""
        assert self.victim_scheduler is not None
        return self.victim_scheduler.request_finished_with_blocks(request, (block_ids,))

    def request_finished_all_groups(
        self, request: typing.Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, typing.Any] | None]:
        """Route a multi-group finish through the block-holding path.

        The stock wrapper drops the block ids on the floor; they are exactly
        what a finish-time store needs.
        """
        assert self.victim_scheduler is not None
        return self.victim_scheduler.request_finished_with_blocks(request, block_ids)
