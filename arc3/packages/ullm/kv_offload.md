# CPU KV offload

*Register: as-built — updates with the code.*

A host-memory tier behind the GPU KV cache, for a run whose games
outnumber the contexts the pool holds. `victim_offload.py` is the
connector, the `cpu_kv_offload*` fields on the engine factories select it,
and the duck turns it on for both served columns (`duck.presets.configure`,
the P-00 and P-EV layers). It runs on the arc-prize-2026-vllm fork the
workspace pins (`tufa/v0.29.0`); a stock 0.29 engine runs it too,
without the fill described below, and needs dense recurrent retention
asked of it either way (below).

On Flash-Next the connector runs in **park mode** ("Park mode", below): a
finished request's whole thread is copied to the tier and loaded back when
the conversation continues. Everything up to that section describes the
band, the mode the connector takes when the engine does not name threads
(`park: False`).

## What the tier can and cannot buy

A host tier cannot hold a decoding sequence: attention reads from GPU
memory, so the GPU pool alone bounds how many contexts are **resident**.
What a tier buys is the context of a game that is *between turns* — the
prompt it will be served on when it comes back. So the tier is worth
exactly as much as it holds that the pool does not, and a tier that
duplicates the pool buys nothing at any size.

That is the whole argument for copying a block only as the pool is
about to lose it, rather than as it is computed: the host then holds
what the GPU is about to lose and nothing it will keep, so `pool + tier`
distinct bytes serve the run instead of `max(pool, tier)`.

The workload: 40 games on Kaggle against a pool of 1,240 blocks that
holds fewer than half of them at the window, 80 on a B200 against 3,006.
Games cycle through the pool; every cycle-in is a load from the tier or
a re-prefill, and a re-prefill is what a collapse is made of.

## What the model shape forces

Measured on Qwen3.8-27B-FP8 (hybrid); the numbers move with the model,
not with this code:

- Four KV cache groups: three gated-delta-net (recurrent, single-state)
  groups in lockstep and one full-attention group. The hybrid allocator
  makes one attention page equal one recurrent state page, which sets
  the block at **1,568 tokens** for every group and a tier slot at
  **49 MiB** (one group's layers for one boundary).
- A recurrent group writes a **checkpoint at every aligned boundary**,
  and the engine frees each one as the request moves past it. A
  ~46k-token context is 30 attention chunks and ~90 checkpoints; stored
  naively, three quarters of the tier is checkpoints.
- A lookup **scans a recurrent group from the end** and takes the first
  surviving checkpoint; the served boundary is the minimum across
  groups. Of those ~90 only the newest is normally asked for, and losing
  it costs the whole context, where losing an attention chunk costs only
  the tail past it.
- A boundary the request completes *during decode* is never cached by
  the engine, so the tier can never hold it and the scan lands one
  boundary earlier.
- In align mode a prefill chunk must end on a recurrent block boundary,
  so a cold request advances **one 1,568-token block per step** and
  chunked prefill runs one partial prefill at a time. Prefill is the
  throughput: a full recompute of a 45k context is ~29 steps that each
  run at chunk length and starve the decodes behind them. The tier's
  load rate is not the constraint (~0.6 GB/s serves ~18k prompt tok/s,
  four times what the card can prefill).

## The regime the design assumes

The connector serves a queue, not a storm. Three engine settings make
that the engine's behaviour, and the design below is built for it.

**An admission reserve.** vLLM's `watermark` — the fraction of the pool
kept free when admitting a waiting request — is 0.0 unless set:
admission fills the pool to the last block, every later decode step that
needs one preempts, in storms, and each storm reloads 20–30 contexts
through the tier. With a reserve the engine holds admissions at the
line, running requests grow into the reserve, and the rest queue with
their contexts in the tier and come back through it. It applies only to
admission — the GPU pool alone still bounds the running set.

The reserve is also the free queue's length, and that is what sizes it
(`ullm.backends.deployment.ENGINE_ADMISSION_WATERMARK`). Absorbing the running set's
growth between admissions needs little: the engine re-admits every step,
so the overshoot past the line is bounded by the running requests that
cross a block boundary together, a hundredth of the pool over nine hours
on Kaggle. But a freed block stays on the GPU only as long as the queue
takes to turn over — free blocks over departures, seconds at the line —
and that residence is what the keep-young pass has to hold a finishing
request's newest checkpoint through to its continuation. Measured on
Kaggle's pool at 40 games, a tenth against seven hundredths run side by
side: the free queue 124 against 87 blocks, a finishing request finding
its checkpoint still on the GPU ten times more often (114 against 1,395
finishes without one), continuation losses 0.6% against 1.0%, and the
generation floor 607 against 443 tok/s. A tenth is the default; the
smaller reserve bought about 37 resident blocks and cost more than that.

**A step budget of several blocks.** `max_num_batched_tokens` of little
more than one block admits a single block-sized prefill per step, and
while any prefill is in flight every step runs at chunk length with
nothing admitted behind it. Five blocks per step
(`max_num_batched_tokens`, the P-00 layer of `duck.presets.configure`) lets
several prefills advance together and keeps admission flowing through a
recompute wave.

**A scheduler that does not stall on a cold head**, from the fork (below).

Measured on a B200 with the pool pinned to Kaggle's 1,240 blocks at 40
games, generated tokens per second over minutes 10–55 went from 382
(no tier, storms) to 784–834 with zero preemptions, the tier serving
93–97% of lookups at 1–2.5 GB/s between turns, and no malformed
outputs. On Kaggle itself the nine-hour run at these settings held the
generation rate above 300 tok/s through its pool-full hour with zero
preemptions and no missing checkpoints.

## The design

Three separable mechanisms. Admission works on any model; retention is
where the recurrent-model knowledge lives; integrity guards the copy.

### Admission — mirror the eviction front

Each scheduler step the connector walks the first `spill_band_blocks`
entries of the GPU free queue and stores any hashed block the host does
not hold, nearest-to-die first, up to `max_spill_blocks` new copies.
Blocks the host already holds are left alone: a block in the free queue
is one the GPU still has, and at a deep queue half of them are taken
back by a prefix hit before they leave it, so refreshing a host copy for
standing there would make the tier's recency the GPU's churn and its
between-turn contexts the oldest content.

The free queue is the prefix cache, and the pool allocates from its head
whether the queue is short or long, so in a warm run *every* allocation
destroys the oldest cached block. The band therefore tracks the eviction
frontier at all times rather than gating on how free the pool looks —
a preemption storm frees whole contexts, the pool reads 40% free, and
the refill would evict half of it with no copy taken. Unhashed blocks
count toward the band without being copied, so a run whose pool never
fills — the front is fresh blocks — writes nothing at all.

Three things the band cannot see, all handled at a request's finish or
in the step after it:

- **Shared blocks.** A block the request shares with others — the
  common prompt prefix — never reaches the free queue while any sharer
  runs, and on an engine that never preempts that is never. It matters
  more than its size: a lookup walks the attention group from wherever
  the GPU's own hit ends, and when the recurrent state at the head is
  gone that hit is zero, so the walk starts at the shared chunk, misses
  it, and returns before the recurrent groups are consulted — every
  evicted context the tier does hold goes unasked. So at every finish
  the request's shared, hashed blocks the host lacks are stored, once,
  while the sharers keep them allocated.
- **The newest checkpoint, kept young.** The engine frees a checkpoint
  the step after its boundary is crossed, typically a turn's first step,
  and the next turn's lookup needs it back — so it spends the whole turn
  as a free block, ordered by an age the request's own blocks never
  accumulate. On a pool that evicts block by block (one whose live set
  just fits, so cycling requests take each other's freed blocks from the
  head and nothing is preempted), its residence in the queue is the
  queue's length over the eviction rate, which shrinks as the live set
  grows and crosses the turn length well before anything is preempted.
  So for as long as the connector knows a request — waiting, running,
  or just finished — its newest checkpoint per recurrent group is moved
  to the tail of the free queue every step, and the same pass offers
  that block to the tier when the tier lacks it, ahead of the band. The
  block is then as safe as the attention it belongs with, for a
  dictionary lookup and two list moves per known request and group per
  step. At finish the newest two are also stored and re-aged once more
  after the engine has freed the attention.
- **A finishing request's other blocks under pressure** are reused
  within a step, before any copy could land. Those are stored while the
  engine still holds them, with the engine told to hold the request
  until the worker reports the copy sent. Only chunks every token of
  which went through the model are offered, and each source block must
  carry the chunk's hash under its group on the GPU: the request's hash
  chain covers the token sequence, so a response ending exactly on a
  block boundary has a hash for a chunk whose last token was never
  computed, and a request aborted mid-prefill has hashes for chunks never
  computed at all. `spill_watermark` and `PRESSURE_STICKY_S` gate only
  this hold; the stickiness is because the instantaneous free count is
  misleading precisely here — a storm frees whole contexts, so the count
  spikes to headroom at maximum pressure.

**The fill.** A checkpoint that is lost anyway — a pool the live set has
outgrown, a queue that turns over within seconds — leaves an attention
prefix on the GPU with no recurrent state at its boundary: the
*divergent* hit of the hybrid lookup. The stock protocol cannot serve
it (the fork section says why), and the cost is a full recompute with
the checkpoint sitting in the tier unasked. With the fork, the
scheduler asks the connector (`fill_divergent_hit`) whether it holds
each lagging group's checkpoint at the boundary of the attention hit;
this connector says yes when the tier has it and it is ready — the
block the kept-checkpoint store put there for exactly this — and the
scheduler keeps the attention hit, allocates one fresh block per
lagging group at the boundary's index, and holds the request for the
load like any async external hit; the connector turns the allocation
into a load of those blocks alone. A full recompute becomes a partial
one of the straddling chunk. A fill stands only for the allocation
that follows it: it is cleared at the next lookup, and a load whose
allocation does not match the fill's shape is counted and dropped, not
served. It loads one block per group, which is a whole state only for
a single-state group, so it answers only for groups whose window is one
chunk.

### Retention — passed first, then a boundary ages as one

One host policy (`VictimLRUPolicy`, registered as `victim_lru`), with
two eviction classes:

1. **Passed checkpoints** — recurrent boundaries a context has moved
   past. Useful only as a fallback when the newest is already gone. The
   band does not copy them at all; the few that arrive before their
   classification go first, in their own LRU order.
2. **Everything else**, by age — attention chunks and each context's
   current boundaries: its last two (the last may be the uncacheable
   decode-completed one) and its first, which a compaction returns to.

Classification is by **boundary index in the request's own hash chain**,
at every lookup (where the chain is known before the turn runs, so the
band already knows what to skip) and at every finish. It is deliberately
not read off which blocks the engine still holds at finish: that is the
engine's carry-forward bookkeeping and says nothing about which boundary
is worth a slot. A lookup only promotes to current; only a finish demotes
to passed. A continuation's chain runs past what has been computed, so
its last two boundaries have no checkpoint yet; demoting at lookup made
the previous turn's newest checkpoint passed at the moment the scan asked
for it, and on a tier at capacity it was evicted while the lookup waited
on a pending store — a full recompute for two thirds of the continuations
that added two chunks. The finish knows which boundaries exist.

The thing that comes back is a context, and a context is worth something
only whole: attention without the checkpoint at its boundary is
recomputed anyway, and a checkpoint without its attention is never asked
for. So the right eviction is the least recently seen context, all of
it, which plain LRU over blocks gives *provided every block of a context
carries the same age*. Left alone it does not: the engine frees a
boundary's checkpoint mid-turn and the attention block at that boundary
at finish, so the checkpoint is always the older of the two and LRU
would take every context's checkpoints before its attention. The
attention block and the checkpoints at a boundary carry the same block
hash with different group indices, so **a use of any of them is taken
as a use of all**: a use of a boundary's key moves its siblings in the
other groups to the newest end after it. A context's checkpoints are
then as young as its attention and die right after it, never before; a
context nothing uses ages to the front whole; and no protected class,
share or constant is involved.

**What a use is.** A host copy is used when it is *served* (a load),
when it is *stored*, and when its block *reaches the eviction front* of
the GPU's free queue — the moment the copy stops being a duplicate of
what the GPU has, which is the band's touch. Not a GPU prefix hit on the
block, not the block sitting deeper in the queue, not the context
finishing: each of those refreshes a copy of something the GPU still
holds.

**The demote.** A served block's host copy goes to the eviction end, so
the tier holds what the pool does not, at the cost of copying the block
again when the GPU gives it up. It is on because the tier has to work
where it is *smaller* than the working set — 40 games on Kaggle is ~450
slots of exclusive need against ~1,700 inclusive, in a 1,253-slot tier
— and what it costs where the tier has room is the drop rate, which is
the copy race below. Off is a knob (`demote_on_use`) for a tier several
times the pool.

### Integrity — every copy is checked

A spilled block is unreferenced, so the pool may hand it out and
overwrite it while the copy is in flight. Each block of a completed
store is kept only if it still carries the hash it was read under; the
rest of the job is kept, since under churn most batches have one such
reuse and discarding the batch would throw away good copies.

vLLM offers a fence for this — registering a store's source blocks makes
the engine submit and wait for the copy before reallocating one — and
this connector does **not** rely on it: copies the fence vouched for
have differed from what was read, for a reason not established. Nothing
a hash did not vouch for is served. The price is a recompute for the
copies a run drops; the alternative is a served prefix of wrong KV,
which the model turns into fluent nonsense.

A cache reset (`reset_prefix_cache` with the connector) discards the
victim path's in-flight bookkeeping with the stock jobs, and a worker
completion for a job from before the reset is dropped by the stale-job
threshold rather than looked up. It is refused while a finish-hold has
not yet reached the worker: the engine keeps a held request's blocks
until the worker reports the request sent, which the worker can do only
for a job it was handed, so the reset waits one scheduler step for the
hold to go out rather than leave those blocks held for good.

## Park mode

The band copies what the pool is about to lose. It fails when an idle
conversation's blocks lose the race to the free queue: a turn writes about
fourteen blocks, thirteen of them dead copies, all at the young end, so an
idle live thread is evicted after one pass of that stream. Park mode keeps
threads instead. It is on when the engine names each conversation's thread
(`prefix_cache_context_threads`) and keeps decode tails
(`prefix_cache_decode_tail`), which ullm's Flash-Next engine layer turns on.

**A thread's life.** A thread is one conversation's history on the engine,
named `<run_id>/<conversation_id>` by the client (ullm's
`Conversation.conversation_id`) and tagged with its history epoch.
- **A finish parks it.** The connector copies to the tier every full
  attention chunk, each recurrent group's state at C (the last full block
  boundary below the tail), and at the decode tail the attention block and
  every recurrent state, keyed at the tail's boundary. It holds the source
  blocks until the copy lands, then uncaches them on the GPU. The GPU keeps
  running requests only.
- **A start loads it back** by hash, within a unit of the previous reply's
  end. The host copy is demoted, not removed: the next park copies only
  what is new, and a demoted copy is the first to go. A lookup whose
  thread's park copy is still in flight waits for it (`PARK_DEFER_MAX_S`).
- **A dead thread is demoted:** replaced by a newer request of its
  conversation, ended by a new history epoch (a compaction), or closed by the
  client. The host closes a game's threads when it ends; the duck closes its
  conversation when its history is rewritten
  (`Backend.close_conversation`), so a game that gives up its slot at a
  compaction does not leave its dead thread parked. A close that arrives
  while the last request's park copy is in flight, which is the usual case
  since the reply leaves before the copy lands, reaches that thread when the
  engine frees it.

**The decode tail.** A hybrid hit needs every recurrent group's state at its
end, and the engine keeps states only at block boundaries and at a prompt's
last unit. Without a tail, a continuation re-prefills the previous reply: 551
tokens a continuation at k=0 and unit 64, 33 with the tail (ARC2, B200). The
engine keys a tail at the last unit boundary its decode passed; under
drafting it relies on RecoverSSM, whose commit writes the state at the
boundary a verify window crosses. It needs `prefix_match_unit`: without one,
the engine turns the tail off and says so. A tail is keyed and then left idle
in the free queue, and a pool at its admission line reuses it within a
second: 75% of the parks on the pool-bound RTX carried none. So the connector
copies each request's newest tail as the engine keys it, keeps its block
young until the copy lands, and demotes the tail it supersedes. A park takes
a tail the tier holds when the GPU has reused its block. Parking on the GPU,
the tail is pinned there instead of copied (below).

**What the tier protects.**
- Keys a parked thread holds are never demoted, including a prompt head it
  shares with others. Nor is a running thread's C: a tail restore never
  loads it, so the tier's copy is the one the thread's next park needs.
- A use of a boundary refreshes only the siblings a thread still needs
  (`sibling_refresh`). Under the tier's own rule every stored sibling moves
  young. Each park supersedes its thread's previous checkpoints, which share
  hashes with its live attention chunks, so every resume dragged them back:
  55% of what sat newer than the deepest parked thread on the RTX, 29-34% on
  the B200. A burst of parks then evicted a parked thread.
- The engine's admission watermark stays 0.10. At 0.03 the RTX preempted
  in bursts of up to 39 a minute, and a preempted request is not parked.

**Sizing.** A slot is one block of one group (1,792 tokens at k=3). A parked
thread holds its full chunks plus 3-7 slots: about 60 slots, ~1.5 GiB, at
96k tokens. A running thread holds 6, its protected checkpoints and newest
tail. So the tier needs the parked threads (idle, or waiting for capacity)
at their size, plus six slots a running request. Before the sibling filter,
the tier capacity a resume load needed sat at p99 1,300 of the RTX's 2,000
slots (Kaggle E and F, 35 slots) and 2,768-2,858 of the B200's 3,740 (the
long runs), with maxima near the end in both. With the filter, the RTX's
p99 fell to 558 (max 964), and no parked thread was evicted in nine hours
(Kaggle G, E's config and seed), which is what took the RTX cell to 40
slots.

**Reading it: `kv_park.log`** in the job directory, on wherever the run has
one, and off under TAAF's minimal diagnostics (a submission). It has one line
per transfer or decision, each with the tier's state: `PARK`, `STORED`,
`START` (its GPU and tier hits against where it parked), `LOAD` (`depth` is
each key's distance from the tier's newest end when asked, the capacity it
needed; `moved`, what last moved it there), `DEMOTE` (why), `GPU-UNCACHE`,
and `EVICTED-PARKED`, the line to watch. Every 10 s, `SNAPSHOT` lists the
running threads and the tier's contents in eviction order (parked P, a
running thread's copy D, neither X), and `TIER-ORDER` gives the deepest
parked key and what is newer than it, by owner and last move.

**Limits.**
- A thread keeps recurrent state only at its tail and at C. A retry that
  perturbs the last tool result (ullm's `perturb_retry_prompt`) changes the
  prompt before both, so that thread's cache is lost: a few times in a
  9-hour run.
- A host stall (a synchronous trace write to a network disk, 12-21 s) parks
  every finishing thread at once, and they can outgrow the tier: in seal
  sweep 4, 90-110 games parked at once past an 80 GiB tier.
- A parked thread leaves the GPU even when the GPU has room for it, unless
  it parks on the GPU (below).

## Parking on the GPU

`gpu_park` (park mode only; on by default wherever park mode is,
`DEFAULT_GPU_PARK`) keeps a finished thread where it is. Park mode's demand
on the tier is every thread outside the engine, so anything that slows
requests on their way in -- a host stall, a renderer burst -- moves threads
from the pool to the tier one for one, and the tier overflowing then evicts
parked threads that re-prefill whole. A stall drains the engine, which frees
pool room at the same moment more threads park: kept on the GPU, they absorb
it there.

- **A finish pins the thread.** The blocks park mode would copy -- every full
  attention chunk, each recurrent group's state at C, the decode tail -- are
  pinned on the GPU (a reference the free queue cannot evict), and nothing
  goes to the tier. One place at a time: the GPU holds the youngest parked
  threads, the tier the older ones.
- **C is the tier's, held and touched.** The tier has C from when it was
  computed (the newest checkpoint is stored as the decode crosses its
  boundary), and the park holds the tier's C as the thread's whether or not
  the GPU has it too, as park mode holds a running thread's: a request
  resumed at the tail holds no C, so the GPU's goes soon after the next
  request takes the thread over, and the tier's is then the only one. The
  park touches it, as a park touches what the tier already has, and so does
  the next request's take-over, as a tier resume's load refreshes it:
  nothing else of the thread uses it, and untouched it ages out of a tier
  that turns over in about a minute. A follow-up whose prompt ends at the
  tail's boundary cannot use the tail (a hit stops a token short of the
  prompt) and resumes at C: the attention hit on the GPU, C loaded from
  the tier (the divergent hit, below).
- **Its next request takes it over.** The lookup hits it on the GPU; once
  the request is scheduled it holds the blocks itself and the pins go.
- **The pool short of room spills the oldest.** Each step, while the free
  blocks and those in-flight spills will free fall short of the engine's
  admission watermark, a reserve the running requests grow into while a copy
  is in flight (`DEFAULT_GPU_PARK_RESERVE`), and the blocks the requests
  waiting to be admitted take, the thread parked longest is copied whole to
  the tier, as a park copies it, and held by its pins until the copy lands;
  then its blocks come free and are uncached. A thread whose next request has
  arrived is about to run where it is and goes last, only while the engine is
  idle, no request running or waiting on a load in flight: waiting requests
  can then hold between them, as pins, the room each needs, and only a
  running request lets go of a thread's. A spill the tier refuses drops the
  thread, to be recomputed, rather than starve the engine of blocks.
- **Every key a copy takes as new has a GPU block behind it.** The copy
  worker asserts that its source and destination counts match, so a key no
  GPU block backs (C, held by the tier when the park was planned) is
  offered only while the tier still has it, and left out otherwise: a
  spill comes long after its plan, and an overflowing tier may have
  evicted it meanwhile.
- **A dead thread lets go**: its pins are released and its blocks go to the
  eviction end of the free queue, still cached, as the engine sends a dead
  thread's idle blocks and the tier demotes its keys. A close comes ahead of
  the request that replaces the thread, and that request's prompt can still
  extend it (a declared rewrite that removed nothing served).
- **A running request's decode tail is pinned, not copied.** Copied, a new
  tail every unit a request decodes is a few GB/s to the tier, nearly all
  of it superseded seconds later. Pinned while the pool has room above what
  admission needs, the tail is on the GPU when its request finishes, and the
  parked thread takes the pin over. Short of room, the spill step lets go of
  every tail pin first: that copies nothing, and the tail stays cached at the
  free queue's young end, as a copied one is kept, until the room returns.

Pinned blocks count as used, so the engine's KV share reads parked threads
as occupied. The engine line splits it, `kv_cache=X%(running:R%,parked:P%)`:
the share the requests hold (one they share with a parked thread is theirs)
and the share only parked threads hold, which the connector publishes once a
second (`ullm.engine_monitor.KV_REQUESTS_SHARE`). The park log's state line
counts them too (`gpu-parked N threads B blocks pinned`, the spills in
flight, the pool's free blocks), and a `SNAPSHOT gpu-parked` line names
them, the next to spill first. Its events
are `PARK-GPU` (with whether the thread kept its tail), `UNPIN`, `SPILL`
(and `STORED` when it lands), `SPILL-DROPPED` and `TAILS-RELEASED`; on any
park, `lost=N` counts keys the tier lost since the plan.

## The fork: what changed in arc-prize-2026-vllm, and why

Two scheduler changes, both behind conditions a stock connector never
meets, so an engine running any other connector behaves as upstream.

**A hook for the divergent hit** (`fill_divergent_hit`). The stock
hybrid lookup reports the attention hit as the local prefix when it runs
deeper than a recurrent group's, flags the hit as divergent, and — if
the connector returns no external tokens — reconciles to the boundary
every group agrees on, which is the head when the checkpoint is gone.
The connector protocol has no way to say otherwise: `get_num_new_matched_tokens`
is asked for chunks *beyond* the local hit, and the first of those is
the chunk the previous turn never finished, so the walk returns zero and
the recurrent groups are never consulted; a recurrent state is not a
token range the protocol can express. The hook adds one question on the
connector base, answered by default with no; `MultiConnector` forwards
it; the KV cache manager exposes per-group hits and can allocate one
fresh block per lagging group at the boundary's index; the scheduler,
on yes, keeps the attention hit and treats the request as an async
external load of those blocks. Nothing else in the lookup path moves.

**A cold head no longer blocks admission.** Because a prefill chunk must
end on a recurrent block boundary, a cold request needs a whole block's
worth of step budget to start at all. The stock waiting loop stops at
the first request it cannot schedule, so once the budget left in a step
is under a block, a cold request at the head of the queue blocks every
warm request behind it — a fill's straddling chunk, a continuation that
needs a few hundred tokens — for that step and every step a prefill is
in flight. The fork sets the budget-blocked head aside for the step
(`step_skipped_waiting`, shown as `deferred` in the engine line) and
keeps admitting behind it. With the five-block budget this is what keeps
admission flowing during a compaction wave, when a cohort of cold
prefills arrives at once.

Both carry scheduler tests driven through the real scheduler and KV
cache manager. The fork also carries `zero_recurrent_state` (the duck's
blank), which is independent of this connector.

### The engine's own boundary store

The engine offers every retained recurrent boundary block to a connector
as the block is cached, and the stock scheduler turns each into a store
job keyed exactly as this connector's finish-time checkpoint store keys
them. Both cannot run: the same block would go to the tier twice. This
connector's store is the one that does, on three counts.

**Integrity.** A copy from this connector is kept only if the source
still carries the hash it was read under (above). The engine's store
rests on the fence instead, which this connector does not trust for the
same reason it does not trust it anywhere else — and recurrent state is
where a wrong copy is least recoverable, since it is not a prefix the
model reads but the state it decodes from.

**Residence.** This connector stores at finish, immediately before the
gap a continuation must survive. The engine stores as the block is
cached, earlier in the turn, so its copy has more of the tier's turnover
to outlive for the same hit.

**Volume.** The engine offers every boundary computed; this connector
stores the ones a continuation resumes from. Measured on the pinned
pool, that is a difference of about fifty times the store traffic for
the same job.

The connector declines the engine's jobs in `_build_partial_tail_store_jobs`.
Taking them instead was measured, and fills a few points more of the
divergent hits — the engine stores every boundary, so a resume further
back still finds one. That is a coverage difference, and coverage is a
parameter of this connector's own store rather than a reason to serve
copies nothing checked.

## Retention the engine must be asked for

The engine caches only the recurrent boundaries it can prove reachable —
the latest replay boundary and any shared-prefix junction — unless asked
for dense retention (`prefix_cache_retention_interval`). A continuation
resumes at the boundary its own PREVIOUS turn ended on, which is
neither, so under the engine's own default the state beneath its
attention prefix is never cached at all: every continuation arrives as a
divergent hit, and no connector can repair that, because what the engine
never caches it never offers to one. `enable_cpu_kv_offload` therefore
asks for dense retention.

## Finding the next one

The engine's own defaults are the port's first suspect, and no counter
inside this connector can see them: every one of them measures a
consequence. The 0.29 regression above was a retention default that moved,
and three rounds of instrumentation downstream of it found nothing, because
what they measured — stores, holds, fills — was all effect. What found it
was diffing the engine configuration the two runs actually recorded: 26
arguments differed and one was a new knob.
`tools/port_check_config_diff.py` does that from two job directories.
Run it before instrumenting anything.

## Defaults

| Setting | Value | Why |
|---|---|---|
| `cpu_kv_offload` (duck P-00) | on, both served columns | Each runs more games than its pool holds contexts; the tier is what turns the tail's cycle-ins into loads. |
| `DEFAULT_CPU_KV_OFFLOAD_GIB` | 64 | Claimed eagerly in `/dev/shm` and counted by the job's cgroup, so the job asks for it up front. The duck sizes its own on Flash-Next: 96 GiB on the B200, 50 GiB on the RTX (its P-EV layer). |
| `DEFAULT_CPU_KV_OFFLOAD_VICTIM` | on | The stock mirror duplicates the pool and serves nothing back under pressure. `victim=False` restores it. |
| `DEFAULT_SPILL_BAND_BLOCKS` | 128 | The eviction front and no deeper: the head evicts a couple of blocks a step, so this is seconds of lead for a copy, while a band spanning a deep queue refreshes blocks a prefix hit takes back. |
| `DEFAULT_MAX_SPILL_BLOCKS` | 128 | Bounds one step's PCIe burst; store traffic is a small fraction of load traffic, so it does not bind. |
| `DEFAULT_DEMOTE_ON_USE` | on | The tier is smaller than the working set (above). |
| `DEFAULT_SPILL_WATERMARK` | 0.15 | Finish-hold only. |
| `DEFAULT_ENGINE_LOG` | off | The instrumentation goes to the trace; the engine-log lines are for watching a run live and are a knob (`engine_log`) that no preset sets. The lines were ~98% of what the engine process wrote, and a Kaggle kernel that stopped draining that pipe stalled the engine on the tally's `logger.info` mid-step for the last 2 h 10 min of a twelve-hour submission (2026-09-19). |
| `ENGINE_ADMISSION_WATERMARK` (`ullm.backends.deployment`, U-00) | see the constant | The reserve that makes the engine queue rather than preempt, sized from the measured overshoot past the line. |
| `park` | on when the engine names threads and keeps decode tails | Whole threads parked and loaded back ("Park mode"); `False` keeps the band. |
| `DEFAULT_GPU_PARK` (`gpu_park`) | on, where park mode is | Parked threads stay pinned on the GPU until the pool needs the room ("Parking on the GPU"); on by default (2026-10-18) ahead of the run that measures it (`roadmap.md`). |
| `DEFAULT_GPU_PARK_RESERVE` (`gpu_park_reserve`) | see the constant | The free blocks kept beyond the admission watermark, which running requests grow into while a spill's copy is in flight. |
| `DEFAULT_PARK_LOG` | on | `kv_park.log` is what shows the tier's health in a run; TAAF's minimal diagnostics turn it off through `BackendFactory.use_minimal_diagnostics`. |
| `DEFAULT_SIBLING_REFRESH` | `"needed"` | Siblings a thread still needs move young with a use; dead checkpoints stay demoted ("Park mode"). `"all"` is the tier's own rule. |
| `max_num_batched_tokens` (duck P-00) | five blocks | Several prefills per step; admission keeps flowing. |

**Sizing the tier.** The reuse-depth line (below) is the curve: under
LRU inclusion a hit at depth *d* would also hit in any tier at least *d*
deep, so one run at a generous size prices every smaller one. Kaggle's
pool-full hour needed a p99 depth of ~410 slots at 40 games; the tier
beyond that is headroom for the queued tail as concurrency rises.

## Alternatives, and why not

The stock `OffloadingConnector` stores every block as it is computed.
The host then fills with what the GPU still holds, the tier's recency is
the GPU's churn, and under pressure it serves nothing back: measured
at a pool a third of the live set, the cached fraction was that of the
pool alone.

Four nearby designs were measured and rejected: a **protected class**
for a context's latest checkpoints with a share of the tier as its
bound — the bound needs a constant, and the constant inverts at a
concurrency nobody has measured, turning the class evicted last into the
class evicted first; **refreshing host copies on GPU prefix hits and at
finish** — each refreshes a copy of something the GPU still holds and
makes the between-turn contexts the tier's oldest content; **trusting the
engine's fence** on a spilled block's copy instead of the hash — it
served copies that differed from what was read; and **re-aging the
checkpoint at finish only** — a turn too late on a pool that evicts
block by block, so runs with and without it collapse at the same point.
The mechanisms above need no bound in their place: a boundary ages as
one, a use is a serve, a store or the eviction front, every copy is
hash-checked, and the checkpoint is kept young for as long as the
request is known.

At the engine level, vLLM's zero watermark and a one-block step budget
make the tier serve preemption storms instead of a queue; the reserve
and the budget are what make it a queue, and the fill and the step-skip
are what make the queue's recompute waves cheap.

## How it was tested

**Cheap tier** — `tests/test_victim_offload.py`, against a real
`BlockPool` and a recording manager: spill order and the band's bounds,
the finish hold, hash-mismatch handling, the two eviction classes and
their reclassification, a boundary aging as one, checkpoint
classification at lookup and finish, the shared and kept-checkpoint
stores, the keep-young pass and its probe, the departure and zero-lookup
classifiers, the trace, the fill's answer and load and its clearing, and
the version guard. The fork's side has its own scheduler tests
(`tests/v1/core/test_scheduler.py`: a fill driven to completion, and a
budget-blocked cold head that no longer stalls the queue).

**GPU tier** — `tests/test_victim_offload_engine.py`, through a real
vLLM engine: nothing is written while the pool has headroom, a prompt
finishing under pressure is read back later, and blocks sharing a hash
do not kill the engine.

**Whole-system measurement.** Unit tests cannot show whether the tier
holds the *right* blocks, so the configuration is measured on duck runs
sized so the pool cannot hold the live set. The metric is the cached
fraction of prompt tokens per ten-minute window, read from the run's
spans — a whole-run average hides a cliff — and generated tokens per
second per minute, whose troughs are what a collapse costs.

The retention design was closed against a **tier-less control** at a
pool that does fit: does a small pool plus a host tier serve like a pool
twice its size? At `gpu_memory_utilization=0.30` (473 blocks against a
630–890 block live set), 24 games, the official-110 benchmark, 75
minutes:

| arm | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 min |
|---|---|---|---|---|---|---|---|---|
| 50 GiB tier, pool 0.30 | 63.6 | 87.7 | 93.4 | 94.3 | 93.7 | 95.5 | 93.6 | 94.6 |
| replicate | 64.9 | 89.1 | 93.7 | 94.7 | 93.2 | 95.4 | 94.7 | 96.5 |
| tier-less control, pool 0.49 | 94.5 | 80.2 | 83.2 | 92.2 | 93.4 | 93.4 | 94.0 | |

Yes, from the point the tier fills; per call the residual uncached
tokens are the same ~2.5% of genuinely new content as the control.

The regime (reserve, budget, fill, step-skip) was closed on **Kaggle's
pool on SLURM**: `num_gpu_blocks_override=1240` puts a B200 in Kaggle's
pool, which is a function of the pool in blocks, at 40 games and a 64 GiB
tier (speed does not carry over — the B200 grows contexts about twice as
fast). Generated tokens per second over minutes 10–55: 382 with no tier,
620 with the tier and fill at a zero watermark, 784–834 with the reserve,
at zero preemptions, the tier serving 93–97%, no malformed outputs.
Kaggle's own nine-hour run at the shipped settings confirmed the regime
on the card it is for. The SLURM defaults (80 games) bring the B200's
pool to 85% at its peak, with the tier serving up to 28% of lookups in
the minutes the pool churns hardest, but never to the admission line, so
the pinned pool remains the vehicle for anything that must exercise the
reserve or the tier there.

Run dirs under `/data/shared/research/arc_3_results/shared/`; each holds
`git_status.txt`, the spans, and the trace. Scores are not a comparable
quantity over 75 minutes and are not read from these runs.

## Instrumentation

Everything the connector observes goes to one place: the trace in the
run's job directory. The same is available as engine-log lines through
vLLM's own logger (the `vllm` logger owns the handler and does not
propagate) when `engine_log` is set in the connector's settings; off by
default, the engine log carries the engine's own lines and this
connector's warnings only.

**The trace.** When the engine was started with a job directory —
TAAF's, which the solver sets on the backend factory before `start()`
and `config_local.attach_job_dir` carries into the connector's config —
the scheduler side appends one JSON object per event to
`kv_offload_trace.jsonl` there. It grows with activity, about 5–7 MB an
hour on a busy engine (evictions dominate), so it is always written when
there is a directory to write it to. `start`: pool, band, cap, groups,
and the sizing arithmetic (host slots, bytes per slot, GPU blocks).
`tally`, once a minute: the spill tally (spilled, dropped, held, in
flight, checkpoints skipped; the keep-young gauges — checkpoints moved to
the tail in the last step, known requests with none on the GPU, `gone`
finishes that found no checkpoint, `kept_stored` and `filled`); every
hashed block that left the free queue classified by what its leaving
cost; every lookup-audit counter (hits and misses by cause, per group,
misses on a context's current boundaries counted separately, every zero
lookup classified as nothing askable, a miss at the straddling chunk, or
a failed walk by start chunk, the divergent hit counted by name and
again when the tier holds the state it was never asked for, the fill's
asks and refusals by reason, store counters); the keys asked and stored
since the last tally; and the asked-depth percentiles with the policy's
gauges. `pattern`: the engine's per-group block pattern at the first
finishes.
`lookup`, one per request arrival, written when the request is
scheduled so it carries both what was asked (chain length, the attention
hit, each recurrent group's newest cached boundary, the tier's answer, a
zero's kind) and what the scheduler settled on (`computed`, hence
`recomputed`), plus the turn of the same context that finished before it.
`finish`: tokens, `turn_s`, the checkpoint chunks still on the GPU per
group, what was stored, whether it was held. `evict`, one per hashed
block leaving the free queue unreused: group, cost, class, position,
age, whose kept checkpoint it was. `store`, `drop`, `fill`, `resume`,
`preempt`. And `step`, once a second: free and used blocks, the queue and
its front band by class, known requests, the keep-young gauges, the
schedule, and everything that moved since the last sample.
`tools/victim_offload_trace.py` summarises a trace by time window. The
trace is what a Kaggle run hands back: the engine log says that a
collapse happened, the trace says which request lost what.

**Reuse depth is the sizing curve.** For every block asked back, the
policy records how many keys would have to survive for that key to still
be there — the tier capacity that ask needed.

Cost: a walk of the free queue each step (bounded by the pool) and a
dictionary lookup per key per lookup. `tools/victim_offload_decompose.py`
splits a run's uncached prompt tokens into what a tier can and cannot
save.

## Limits and open questions

- **The tier's capacity is the workload's.** Queued in and out of a
  64 GiB tier, 40 contexts at the Kaggle window are near its size in the
  pool-full phase, and the full recomputes that remain there are chunks
  or checkpoints the tier had evicted. Concurrency past that on a tier
  that does not grow converts loads back into recomputes.
- **Why a fenced copy can be wrong** is not established. The clean fix
  is engine-side: hold a spilled block until its copy lands.
- **Dead contexts** age out by LRU alone; a signal from the agent side
  (compaction, game over) would let them go first.
- **The compaction wave.** Games that start together compact together,
  and a cohort of cold prefills halves the generation rate for a few
  minutes each time. It is not a cache event — no preemptions, no tier
  misses — but it is the floor of every run; a staggered start is the
  harness-side answer.
- **Refused and untested configurations.** The connector refuses to
  start on a sliding-window attention group wider than one chunk or on
  an eagle/MTP draft group: everything recurrent-specific in it reads a
  windowed group as a single-state one, and those shapes have not been
  run. Untested but allowed: TP > 1 (stock designs for it and the worker
  override defers to its store-writer gate, but every run here was
  TP=1), and pure-attention models, where the checkpoint classes stay
  empty and the policy degenerates to LRU plus the band.
- **Coupling.** The connector subclasses vLLM's offloading scheduler and
  worker and reads their private state — job tables, the free queue's
  list, the manager's policy. The module refuses to import off
  `SUPPORTED_VLLM_LINE`, which fires only when this connector is
  selected. The fill is the one explicit hook; moving the remaining
  reach-ins to hooks is the next step. Upstream's own recurrent
  boundary-state stores sit in the same path, and are declined ("The
  engine's own boundary store").
