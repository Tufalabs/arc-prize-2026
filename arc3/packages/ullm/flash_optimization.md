# Serving Flash-Next: what was measured, what it means, what ships

*Register: living.*

Qwen3.8-Flash-Next (the e307 NVFP4 prune) on two boxes: the cluster's B200
(`slurm-b200`) and Kaggle's RTX PRO 6000 (`kaggle-rtx`). This is the record
of the experiments behind both deployment cells, the reading of each, and
the open work. It is written to be re-read before changing a Flash setting:
several of the conclusions below are the opposite of the plausible first
guess, and each says what rules the guess out.

Where a value lives is `settings.md`'s business; this document says why it
is that value. Run artifacts referenced here (pulled Kaggle logs, profile
traces, launch scripts) sit under `/data/personal/$USER/` and
`/data/shared/research/arc_3_results/shared/`; they are evidence, not code.

## What ships

| setting | B200 (`slurm-b200`) | RTX (`kaggle-rtx`) | layer | why, in one line (sections below) |
|---|---|---|---|---|
| checkpoint | e307 from shared storage | the same, as Kaggle model `qwen3-8-flash-next-nvfp4-e307/Transformers/default/1` | U-EV | the prune is worth ~25 GiB of card on the RTX |
| `gpu_memory_utilization` | 0.9488 | 0.9437 | U-EV | 0.97 less the profiler's graph reservation, since the graphs are captured once, after the pool: the footprint 0.97 had ("Competitor tricks"); the B200's W4A16 MoE fits it at the cell's slots, not at 220 ("The MoE backend"). With two captures the B200 died capturing graphs at 0.98; the RTX cannot start at 0.99, dies in play at 0.98 with or without expandable segments, and ended a nine-hour submission an hour early at 0.975, for a hundredth worth ~0.7% of pool ("The utilization ceiling"). The RTX throughput stack (`FLASHNEXT_THROUGHPUT_ENV`) takes a hundredth of 0.9537 back: its CUDA graphs are ~0.65 GiB larger, and at 0.9537 its engine runs out of memory in warm-up |
| CUDA graph capture | once, after the pool (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) | the same | U-EV | engine start-up 318 to 187 s on the B200, 238 to 158 s on the RTX ("Competitor tricks") |
| MoE backend | `flashinfer_cutedsl` in its W4A16 mode (`VLLM_CUTEDSL_MOE_FP4_FORCE_A16`) | `marlin` | U-EV | Marlin's arithmetic, 4-bit weights over bf16 activations, in each card's fastest kernel for it: on the B200 the CuTe DSL kernel on the card's own tensor cores, +21% generated on the replay at Marlin's accuracy; on the RTX Marlin, the fastest backend it admits ("The MoE backend") |
| `enable_flashinfer_autotune` | False | False | U-EV | measured: no gain on Marlin; the B200's W4A16 kernel reads faster tuned on one layer, unread on the replay ("The MoE backend") |
| PLE n-gram table | host (pinned, 48 GB) | host (pinned, 48 GB) | U-E / U-EV | the gather over PCIe is under 0.1 ms a step |
| input embedding | host (pinned, 1.18 GiB) | the same | U-E0 | one row a token over UVA, shared with the drafter; its 1.18 GiB goes to the pool ("Competitor tricks") |
| KV cache | INT4 per token and head (sparse attention, the drafter's QSA layer too); the indexer's compressed-key cache FP8; the compressor rings in a pool of their own | the same | P-EV | chosen on score (2026-10-12); ARC3's until ARC2 adopts them ("INT4 KV cache", "The fixed blocks") |
| attention block | 3,120 | 3,008 | P-EV | the smallest multiple of the cell's unit that holds the bf16 recurrent page at an INT4 token's bytes ("The fixed blocks") |
| recurrent state | bf16 | bf16 | U-E0 | the state sets the page every cache group shares; bf16 halves the block (3,136 to 1,600) and the fixed pool a running request holds: +36% generated on the RTX. FP8 rows (opt-in in the fork) were read out on score ("The recurrent state in bf16") |
| step token budget (`max_num_batched_tokens`) | 8,192 | 512 | P-00 / P-EV | the RTX's largest captured CUDA graph; on the B200 four drafts' decodes alone would fill it ("512 tokens a step") |
| weight load | engine default | `lazy`, read ahead by `ullm.kaggle_prefetch` from the first setup command | U-EV | the engine's own prefetch is unbounded and starts at load; the pinned table defeats its check anyway |
| engine install | the workspace venv | the wheelhouse's recorded install replayed (`ullm.kaggle_site`), pip where the record does not match the kernel | kaggle | pip took ~200 s of every RTX boot |
| Kaggle image | -- | pinned to run H's digest | kaggle | the install record was made on it |
| FlashInfer kernels | downloaded | prebuilt wheels, `FLASHINFER_NO_DOWNLOAD=1` | U-0V / kaggle | no network on a kernel; the JIT fallback killed boots |
| `async_scheduling` | on | on | U-00 | stated so an incompatibility raises instead of silently disabling it |
| dense CUDA graph capture | on | on | U-00 | a rung per decode width; follows drafted width under MTP |
| vision tower | kept | kept (dropped only for an agent with no image) | P-0V | the duck sends the grid as an image by default; dropping it took 0.95 GiB of weights off the card and bought no pool |
| image processor ceiling, video | the largest image the duck renders; no video | the same | P-00, derived | the start-up profile's peak activation 4.86 to 1.02 GiB ("Competitor tricks") |
| `max_concurrent` | 100 | 80 | P-EV | at or just above what the drafted pool runs under park mode, so requests wait for capacity most minutes without outgrowing the tier: 100 read pool-bound in 43% of minutes where 120 lanes overflowed the 96 GiB tier (2026-09-28); on the RTX, with the INT4 caches, the rings' pool and 512-token steps, a little above what the pool then runs (2026-10-12; "Concurrency and the pool"). "Speculative decoding" has the history |
| slot ranking (Kaggle) | -- | each game's whole spend (`least_game_tokens`) | P-0V | the games share the machine evenly (2026-10-12; `settings.md`, P-0V, says why) |
| `start_gap_s` | 12 s | 30 s | P-EV | the opening fill spreads over 20 minutes, so games reach their compactions apart rather than in one wave |
| completion tokens per level | 175k | no cap (the submission's clock bounds it) | P-EV / P-0V | B200 games play about the tokens a Kaggle game does, so what is tuned there holds here; eyeballed from the Kaggle runs |
| host KV tier | 96 GiB (the seal's 64 GiB, `packages/seal/evidence.md`), victim connector in park mode | 50 GiB, the same | P-00 / P-EV | the tier regime (`kv_offload.md`), sized on the RTX from its reuse depth ("Host memory"); its engine-log copy stays at the connector's default, off, since writing it stalled a Kaggle engine (`kv_offload.md`, defaults) |
| speculative decoding | MTP, k=4 | MTP, k=3 | U-E0 / P-EV | the ARC-tuned NVFP4 head on the cell's MoE backend, GDN RecoverSSM, drafts sampled and verified probabilistically; +51% generated on the RTX replay; k=3 on the RTX because the official games inflate acceptance ("Speculative decoding") |
| `prefix_match_unit` | 80 | 64 | U-E0 / P-EV | a hit may end every unit rather than at a block; each cell's attention block is a multiple of its unit ("Cache hits inside a block", "The fixed blocks") |
| decode tail, context threads | on | on | U-E0 | each conversation's thread named by the client and a state kept at the last unit its decode passed: what puts the host tier in park mode (`kv_offload.md`, "Park mode") |
| allocator | engine default | expandable segments, the cumem allocator holding the KV cache | U-EV | allocator failures 10 to 0 on the replay at no cost; one rank, so no custom all-reduce to conflict with ("Expandable segments and the indexer bound") |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 512 (engine default) | 64 | U-EV | the indexer's prefill scoring buffer chunked; free beside expandable segments ("Expandable segments and the indexer bound") |
| engine memory log | off | off | U-00 | bring-up instrumentation (`memory_log_s`); off by default, turned on per run |
| engine profile | off (a factory option) | off (a factory option) | U-00 | `EngineProfile`; on for experiments |

Drafting's readings and decisions are under "Speculative decoding".

**A third cell: the four L4s (`kaggle-4xl4`).** ARC2's submission box, which
ARC3 never runs on: two pipeline stages of two tensor ranks at 0.904
(0.955's equivalent under one graph capture, not yet booted), Marlin, the
table on the host, the bf16 state at a 3,300-token step and 60 slots. Its
record -- the bring-up, the topology, the images ARC2 sends, and ARC2's B200
readings -- is the addendum, `flash_optimization_arc2.md`.

## Reading a run

These habits carry every comparison here, and a comparison without them was
wrong more than once.

- **Bucket by the running count.** A step's cost rises with the number of
  requests decoding, and a run's running count moves through the hour (it
  falls once the pool saturates). An hour's average tokens/s mixes phases;
  compare `[engine]` lines at matched `running=`, and split pool-only
  minutes from tier-active ones (`ext_hit` 0 vs >90%).
- **Scores say "not worse", not "better".** Two identical Flash-Next runs
  once landed 6.76 and 9.00. An arm whose score does not collapse has not
  broken the model; a ranking between arms needs full-length runs.
- **Throughput is read on the traffic replay** (`traffic_benchmark.md`):
  recorded traffic against the live engine, fifteen minutes after a
  ten-minute warm-up, counted only while the pool binds. The standard error
  of that mean is ~1.3%; the per-minute scatter (5-6%) is not a reading's
  noise. Every RTX reading below is one run on one workload file.
- **A pair starts together.** On a shared node a neighbour's start moves a
  replay: a duck sweep job starts all its game processes at once and stalls
  a replay's host side for the minutes after, which read as a 5.8%
  regression once (the first B200 pair of "vLLM v0.31.0 picks", whose table
  holds the rerun). Run the two arms as the two steps of one two-GPU job, so
  they share a node, a start and their neighbours
  (`arc_3_results/shared/tmp/v031_pair/pair.sbatch`, which runs the
  launcher's own job scripts; their node-local roots stay short, since the
  engine's Unix socket paths must fit in 108 characters). Check a step in
  the minute rates against the node's job starts.
- **A slow stretch can come from the host.** Two B200 nodes once stepped
  25-30% slower for forty minutes with GPU kernel time unchanged, the GPU
  waiting on the host, and the same configurations read normal an hour
  later; the cause was not found (2026-10-11). The host tier's offload trace
  (`kv_offload_trace.jsonl`, its `step` events) gives a run's step rate in
  10 s bins, so any reading can be checked for such a stretch.

`packages/ullm/tools/engine_profile_read.py` reads an engine profile window
(below); a comparison script for pulled Kaggle runs by running count lives
with the pulls (`kaggle_pulls/compare_flash.py`).

## Bring-up on the RTX card

The first boots on Kaggle died, in a few distinct ways, before the cell
served at all.

**OOM at boot was FlashInfer compiling, not the tier.** Twice the tier was
blamed (60 GiB, then 16) and twice wrong: the time of death moved by a
second and the connector never allocated. The killer was `cicc` -- NVIDIA's
compiler front end, several instances at several GiB each. FlashInfer
downloads its cubins from NVIDIA everywhere else; a Kaggle kernel has no
network at any point, so it built them instead, inside the cgroup. The fix
is the wheelhouse: `flashinfer-cubin` and `flashinfer-jit-cache` at the
pinned version, plus `FLASHINFER_NO_DOWNLOAD=1` so a missing kernel fails
loudly instead of compiling. The same wheelhouse carries
`nvidia-cutlass-dsl` for the CuteDSL paths.

**Seeing it took instrumentation inside the engine.** A vLLM plugin never
fired on Kaggle (plugins are discovered through installed entry points, and
the kernel runs the fork from source), so the memory log lives in the fork
(`vllm/utils/memory_log.py`): cgroup split, top processes, phase markers,
started by the factory's `memory_log_s` and stopped once start-up ends. It
also corrected a misreading worth remembering: `file` in cgroup accounting
includes `shmem`, and tmpfs is unreclaimable without swap, so "page cache"
is not all reclaimable.

**The utilization ceiling is 0.97, tested twice.** 0.99 cannot start (the card
reports 93.88 GiB free of 94.97). 0.98 starts and dies in play: first at 50
slots on cutlass, the CUTLASS MoE runner asking for workspace the profiler
never saw (`getWorkspaceInfo`, 814 MiB against 402 free); then again on the
shipped cell, Marlin at 40 slots (`platypus-flash-util98-nb`, 2026-09-13), this
time as plain allocator exhaustion rather than a kernel's hidden workspace: a
90 MiB request with 12 MiB free, PyTorch holding 91.71 GiB, ~2.4 GiB of the
process outside PyTorch (CUDA context and library workspaces), and another
process on the card holding 550 MiB -- about 1 GiB past the 93.07 GiB the
profiler planned for at 0.98. The runtime footprint outgrows the start-up
profile by that much, and 0.97 leaves room for it. 0.97 serves. Profiler figures at 0.97: 48.70 GiB
weights + non-torch, 3.05 GiB "peak activation", 40.37 GiB of KV, 1,466,001
tokens (~16 full 90k contexts).

**A hundredth above 0.97 is not worth its risk.** With expandable segments,
on the replay at 100 slots, 0.98 died again and 0.975 played its window with
no allocator failure, for 2,650,112 KV tokens against 0.97's 2,640,554:
+0.36%, about 0.7% of pool per hundredth rather than the ~2.5% a notebook's
arithmetic suggests, because `gpu_memory_utilization` bounds vLLM's whole
footprint, weights and activations included, so most of a hundredth never
reaches the pool. That is also how 0.98 can fail while 0.975 gains so
little. The nine-hour submission at 0.975 (2026-09-21) scored but ended
after eight of its nine hours, cause unread (open item 9); a gain that small
buys no death, so the cell stays at 0.97.

**The B200's ceiling is 0.97 too** (2026-09-13, the shipped cell at 150 slots
over 200 games, one 10-minute boot each). The cell had carried 0.90 from the
source checkpoint's runs, unmeasured. 0.98 starts and sizes a 120.23 GiB pool
(7,059,563 tokens), then dies in the CUDA graph capture that runs after the
pool is allocated: a 2 MiB request with 1.81 MiB free, the process holding
178.10 GiB of the card's 178.35 -- 3.3 GiB past the 174.8 GiB that 0.98
allows, where the profiling capture before the pool took 2.87 GiB. 0.97
captures, sizes 118.44 GiB (6,955,223 tokens, 12% over 0.90's 6,191,642) and
played ten minutes at 150 slots without an allocation failure, the pool at 48%
when it stopped. The saturated phase, where
the RTX's 0.98 died, is not yet seen at 0.97 on this box; the concurrency and
the 96 GiB tier were sized against the 0.90 pool, whose queue the larger pool
shortens.

Two facts about that profiler line, both learned by getting them wrong:

- **Graph memory is inside the budget.** vLLM subtracts a CUDA-graph
  *estimate* before sizing the pool, and the reported "peak activation"
  already includes it. The estimate ran 1.59 GiB against 0.77 measured, so
  ~0.8 GiB is reserved and never used. The log line's `--kv-cache-memory`
  suggestion adds the measured graphs on top of the estimate again, so its
  "fit" figure undersizes the pool -- do not take it.
- **Dropping the vision tower freed 0.95 GiB of weights and the pool did not
  move.** The graph estimate for the denser capture absorbed it. Dropping
  it also turns off the 4 GiB shared-memory processor cache (host RAM),
  though the cgroup's tmpfs figure did not visibly move. It is kept on Kaggle
  now that the duck sends the grid as an image by default: a Kaggle trial with it
  (2026-09-26) booted to a 2,845,701-token pool and generated as fast as the
  run without it, minute for minute over its first minutes.

**The e307 prune is what makes the card viable.** The unpruned 512-expert
checkpoint takes 74.3 GiB of the card against e307's 49.0, leaving a KV pool
too small to serve the concurrency, and boots about 2.5 minutes slower.

**Host memory.** The pinned PLE table (47.68 GiB) and the tier (in
`/dev/shm`) are both unreclaimable, in a 175 GiB cgroup that the game
processes grow in too. The tier's size is a balance of its robustness
against that memory, read off the tier's reuse-depth line
(`kv_offload.md`, "Sizing the tier"): on a nine-hour run at 80 slots the
asks at the highest pressure landed within ~1,400 slots (~29 GiB), the
deepest asks came from games returning after a long absence at low
pressure, and eviction misses were 0.01% of lookups. 50 GiB (~2,430 slots)
covers the pressured asks with room for settings that queue more in the
tier.

**The boot** is ~466 s of weight load (Marlin adds ~58 s of repacking) and
~115 s of engine init, ~12.5 minutes to the pool line.

## The MoE backend

**Why each box names its own.** `flashinfer_trtllm` checks the 10.x
capability family; the RTX PRO 6000 is 12.0, so it refuses. `auto` is not an
answer: it walks a priority list, logs rejections only at debug, and on the
B200 once demoted to a CUTLASS path that died initialising its grouped GEMM
-- a demotion reads like a choice. So both cells name their backend, and each
states its own (the builder states none).

**The public notebook was not evidence.** A Kaggle notebook serving the
unpruned checkpoint on this card has a guard accepting only
`flashinfer_b12x`. But b12x is not in `AVAILABLE_BACKENDS` -- nothing reaches
it unless named -- and that notebook leaves the variable unset, so it runs
whatever `auto` picks, which on 12.x is FlashInfer CUTLASS. Reading the guard
as a recommendation cost a boot.

**b12x: two backends, one runs, and it is slower.** `flashinfer_b12x`,
FlashInfer's CuTe DSL kernel, compiles once reachable (the CuteDSL path
export was the missing piece) and then faults with `cudaErrorIllegalAddress`
inside its own `run`, on both e307 and the unpruned 512-expert checkpoint,
so not an expert-count artefact; the same fault class is open upstream three
ways (vllm#50189, flashinfer#3383, vllm#47365). vLLM's own
`moe_backend="b12x"` is a separate implementation over the standalone
`b12x` package (vllm#52018): its kernels compute, and its one fault -- an
illegal address in CUDA graph capture -- was ours to fix. It indexes its
expert tables with the routing ids it is handed and does not bound them,
and capture runs the model over buffers no router has written; one
in-place `remainder_` on the serving path clears it (the fork's
`TUFA.md`). Measured on the replay at 80 slots it is slower than
Marlin here: -3.2% with FP4 activations (1,397.3 against 1,443.3
generated). The profile said
so in advance: Marlin's expert reads already sit within 5% of a pure memory
read ("The RTX card"), so decode has no headroom for any kernel, and W4A4's
arithmetic case is prefill's alone; the one published third-party comparison
on this card also prefers Marlin to W4A4 CUTLASS. Dropped. The launcher's
arms and a wheelhouse variant carrying the package keep a retest at one run
(open item 11).

**Marlin beat cutlass on the card.** Overnight 2026-09-12/13, each one hour
at 40 slots against the cutlass baseline:

| run | vs | tokens/s per running request | score at each 10-min report |
|---|---|---|---|
| cutlass baseline (v14) | the earlier 50-slot run | same band | 0.21 0.87 1.39 1.66 |
| **Marlin** | cutlass | **+10% pool-only, +4% tier-active** | 0.43 1.16 2.05 2.46 |
| FlashInfer autotune on Marlin | Marlin | -1 to -3% (noise) | 0.43 1.38 2.45 2.64 |
| Inductor autotune on Marlin | Marlin | died at compile | -- |

Marlin runs the experts as W4A16 (4-bit weights, bf16 activations), where
the native FP4 kernels are W4A4. At decode each expert sees a token or two
per step, too little for FP4 arithmetic to pay for quantizing its input, and
Marlin skips that. It also leaves the grouped CUTLASS FP4 path that has an
open corrupt-output report on this card family (flashinfer#2723). Cost: a
minute of repacking at load.

**Is Marlin more faithful?** On paper, yes: same 4-bit weights, no
activation rounding. Every FlashInfer W4A4 backend (TRT-LLM, cutlass,
CuteDSL, b12x) additionally quantizes expert inputs with one activation
scale shared across experts -- vLLM collapses the calibrated per-expert
scales to their max (`amax_for_moe_activation_quant`) -- which coarsens the
block scales of quieter experts. Unmeasured; see open items for how. What
Marlin does to scores and game length is in "Checkpoint, active experts and
MoE backend: scores on the B200".

**Marlin's arithmetic on both boxes** (2026-09-13; on the B200 in
the CuTe DSL kernel, 2026-10-15, below). The RTX decides it: Marlin is
the fastest backend the card admits (above), and throughput on the submission
box trumps everything else. The B200 follows so both boxes run the same
expert arithmetic: the B200 is where hyperparameters are tuned and Kaggle is
where they are scored, and with one arithmetic the MoE half of that gap
closes (the kernels, attention, graphs and the pool still differ). The B200 score is not
part of the decision. It was checked, and is in "Checkpoint, active experts
and MoE backend: scores on the B200":

- **Not worse.** Over twenty seeds each on the official 25 games, 30.79
  against TRT-LLM's 28.19 (Welch p = 0.034).
- **Not cheaper per point.** Games play longer on it (175k tokens a game
  against 166k) and generation is slower (1,956-2,032 tok/s in the first hour
  against 2,532-2,760), so the higher score is not at constant token cost. The
  faithfulness argument above -- no shared activation scale on the experts'
  inputs -- fits both halves, and is still unmeasured.

Costs: the repacking at load, and a drafter's bf16 head needs its own
backend, since Marlin serves no unquantized experts. The B200's FP8 slot
count and tier ("FP8 KV cache") were measured on TRT-LLM; the next long run
on Marlin is where they are read again.

**W4A16 on the B200's own tensor cores** (2026-10-15). FlashInfer's
CuTe DSL NVFP4 MoE (`flashinfer_cutedsl`) has a W4A16 mode,
`CuteDslMoEWrapper(quant_mode="w4a16")`, that the fork's
`VLLM_CUTEDSL_MOE_FP4_FORCE_A16` turns vLLM's backend to: Marlin's arithmetic,
the 4-bit weights decoded to bf16 and no activation scale, but on the 10.x
family's own tensor-core instructions, fed by its tensor memory accelerator
and pipelined across warps, where Marlin issues the warp-level ones older
cards run too, at a fraction of the B200's tensor-core rate once a decode step
carries tens of tokens an expert. On the RTX Marlin's expert reads already sit
at the card's memory speed, so no kernel has room there. Without the variable the backend is W4A4,
which the cell refuses to run. Read on the traffic replay at 220 lanes on the
205-expert checkpoint (`...-arc3-reap307-dials-joint205-u800`, 108-268
experts a layer), the seal's column, against Marlin; one run each, noise ~2%:

| | Marlin | W4A16 |
|---|---|---|
| generated tok/s, settled reading | 5,524.4 | 6,685.1 (+21.0%) |
| generated tok/s, pool-bound minutes | 6,389 | 7,208 (+12.8%) |
| decode step, start to start (batch-queue trace) | 98.6 ms | 80.6 ms |
| MoE kernels in a decode-only step (profile) | 38.1 ms | 19.5 ms |
| draft acceptance past the gate | 67.8% | 67.5% |

One layer alone (`tools/bench_moe_w4a16.py`): 0.609 to 0.326 ms at 1,080
tokens, 4.10 to 1.56 at 9,272; 0.248 at 1,080 autotuned, which the cell
does not do, unread on the replay. Compiling a new expert count takes ~0.9 s.

**Its accuracy is Marlin's.** One real layer (layer 10, 196 experts) through
each kernel as vLLM prepares it, against an exact fp32 dequantization of the
same weights (`tools/moe_layer_reference_check.py`): relative error 4.71e-3
for Marlin, 4.62e-3 for W4A16, at 216 to 9,272 tokens alike, the two apart by
about sqrt(2) of either: independent rounding of the same arithmetic. End to
end (`tools/moe_logprob_check.py`, 32 prompts of this repo's text, 98,272
positions, the drafter off): W4A16 against Marlin a mean |gap| of 0.172 nats
in the actual token's log-prob and 89.8% of top-1 tokens shared, where a
second Marlin engine against the first reads 0.166 and 90.3% (the engine is
not deterministic across starts) and W4A4, the backend's activation-quantized
mode, 0.244 and 86.0%. That is inference evidence, eager prompt prefills
with the drafter off, for play; a training rollout's sampled log-probs
against its learner are a different question, the trainer's to qualify when
it repins (its parity gate). For that the fork keeps two switches: the
experts summed in a fixed order rather than the epilogue's BF16 atomics
(`VLLM_CUTEDSL_MOE_W4A16_FUSED_FINALIZE=0`), and the B200 on the GDN and QSA
shapes every other card runs (`VLLM_SM100_KERNEL_SHAPES=0`).

**It costs CUDA-graph memory.** Its graphs take 7.57 GiB at 220 lanes against
Marlin's 6.39, all outside the budget, so at 220 lanes the cell's 0.9488 fell
0.12 GiB short and 0.94 captured and played. The ladder grows with the slots
(Marlin's 5.38 GiB at 180, ~25 MiB a slot), so at the cell's own 85-100 its
graphs come to less than Marlin's took at 220, which 0.9488 held: the cell
keeps it, unbooted at those slots. A harness running more slots states its own
budget: ARC2's 0.9174 sits ~3.6 GiB below where Marlin's capture failed at
230 sequences, which W4A16's ~1.2 GiB more there fits inside. ARC2 runs 180
for now, while sweep work tunes its slots; also unbooted.

Read on the 205-expert checkpoint; the cell serves e307, 307 experts in every
layer, whose fewer tokens an expert leave the kernel less to gain. The RTX
cannot run it (10.x only); the boxes run the same arithmetic in
different kernels.

## Kernel-level settings

- **FlashInfer autotune**: off, measured (table above). The original reason
  -- a 27B A/B on the B200 measuring it 20-25% slower -- was about monolithic
  TRT-LLM kernels with little to choose between; on Marlin FlashInfer only
  has the non-MoE layers left to tune, and it found nothing. The B200's
  W4A16 MoE gives it the experts to tune again (open item 31).
- **Inductor `triton.autotune_at_compile_time`**: nothing to set, since the
  Qwen4Exp model runs without torch.compile (upstream #55272). Compiled, it
  must stay off: its autotune block allocates `95.37 GiB` -- exactly
  20,000,000 n-gram rows x 2,560 dims x 2 bytes, an example buffer the shape
  of the whole PLE table, generated on the card to benchmark a kernel that
  reads it. That is larger than the RTX card and fails the same way on the
  B200, wherever the served table lives, and vLLM compiles through
  `standalone_compile`, which turns this autotune on by default.
- **`linear_backend`**: nothing to choose. ModelOpt quantized only the routed
  experts (its ignore list excludes attention, linear attention, router gates,
  shared experts, hyper-connections, the PLE, embeddings and the LM head), so
  no quantized-linear backend has a layer to serve. The setting reaches only
  the bf16 layers, whose one alternative to cuBLAS is `flashinfer_cutedsl`'s
  bf16 GEMM -- 10.x only, untested on the B200.
- **Dense CUDA graph capture** (`dense_cuda_capture`, on everywhere): a rung
  for every decode width up to `max_num_seqs`, the ladder kept above for mixed
  batches. 83 graphs at 40 slots on the RTX, 0.77 GiB. No measurable
  throughput gain, no loss. Under a drafter the rungs are the multiples of
  `1 + num_speculative_tokens`, since vLLM dispatches a drafted decode batch
  to a FULL graph only at those widths.
- **Two kinds of graph.** `FULL_AND_PIECEWISE` captures PIECEWISE graphs at
  every size (the model cut at attention and the other custom ops, which run
  eagerly between the pieces) for mixed and prefill steps, and FULL graphs of
  the whole forward pass for uniform decode batches up to `max_num_seqs`. A
  decode-only step is one FULL replay, with no CPU-side kernel launches.
- **INT4 KV cache**: per token and head on the sparse-attention layers, on
  both boxes ("INT4 KV cache" below).
- **GDN kernels**: vLLM's defaults on both boxes. The RTX's two
  alternatives -- FlashInfer's prefill behind a gate the fork can open, and
  the Triton decode -- both measured as washes ("The GDN kernels on the
  RTX").

## The GDN kernels on the RTX

Two phases, two kernels, both measured on the replay at 80 slots, and
neither worth a change (the table under "The RTX column on the traffic
replay").

- **Prefill.** vLLM admits FlashInfer's GDN prefill on 9.0 and 10.x only
  (`_resolve_gdn_prefill_backend`), so the 12.0 card runs Triton/FLA. The
  stop is arbitrary rather than a support limit -- the kernel serves this
  model on SM120 elsewhere -- and the fork branch `sm120-gdn-prefill`
  admits 12.0. With it: 1,439.1 against 1,443.3 generated (-0.3%), the
  uncached prompt rate unmoved, as it must be, since the kernel changes how
  fast prefill computes and not how much of it there is. The branch stays
  unpinned: correct, and worth nothing here. The kernel is in no prebuilt
  jit-cache and still compiled without a visible stall.
- **Decode.** vLLM's fused CUDA kernel (`VLLM_GDN_DECODE_KERNEL=cuda`)
  against its Triton kernel: 1,449.6 against 1,443.3 (+0.4%), inside the
  error; GDN is ~4.4 of ~34 ms a decode step here, so a few percent was the
  ceiling either way. Triton is the general path the fused kernel
  specialises, not a fringe one: vLLM falls back to it whenever the fused
  kernel's support predicate fails. The four L4s read it at +1.5%, also
  under the bar (the addendum).

## Where decode time goes

### Engine profiling

`EngineFactory.profile = EngineProfile()` (either factory) arms vLLM's torch
profiler and opens windows over the running engine: when KV usage passes
`kv_cache_above` with requests running -- never before `not_before_s` (5
minutes) after the engine is ready, since a small pool can cross its threshold
on the first prompt wave while kernels still compile -- `windows` of them,
`spacing_s` apart, each `iterations` steps. Off by default (`profile=None`). Outside a window it costs a check per step;
each window stalls the engine ~15-25 s while its trace is written. Traces
land in `engine_profile/` under the job directory and are packed into
`engine_profile.tar` beside it after each window and at shutdown.

The reader attributes every GPU kernel to the step that launched it (by the
launch's correlation id; a step owns everything launched until the next
step, which picks up sampling and a drafter's passes) and splits decode-only
steps from mixed ones. Kernels replayed from CUDA graphs appear individually.

What a trace cannot say is whether a kernel waits on memory or computes: its
duration includes both. The handle is scaling -- the same kernel at a
different token count. Flat means a fixed amount of work per step.

### The B200 (TRT-LLM, tier on, windows at KV 88-90%)

Decode-only kernel time, ms per step:

| family | 93 requests | 65 requests | ratio (0.70 if proportional) |
|---|---|---|---|
| MoE | 5.75 | 5.53 | 0.96 |
| dense bf16 GEMMs | 5.55 | 5.24 | 0.94 |
| GDN linear attention | 3.66 | 2.65 | 0.72 |
| QSA attention | 1.61 | 1.31 | 0.81 |
| hyper-connections | 1.21 | 1.09 | 0.90 |
| elementwise / glue | 1.03 | 1.03 | 1.00 |
| total | 19.3 | 17.3 | 0.90 |

GPU busy 92-94%, the PLE gather under 0.1 ms. About 60% of decode kernel
time is flat in the batch. GDN scales with requests. The flat part has two
different causes:

- **MoE** is consistent with moving expert weights: the union of experts a
  step touches is 1-(1-10/307)^n, 88% at 65 requests and 95% at 93.
- **The bf16 group is not byte-bound.** It reads about a sixth of the bytes
  MoE does and takes as long: ~724 small kernel calls a step at ~9 us each,
  whose cost is per call (or per cuBLASLt tile band -- the kernel names
  carry tile sizes) rather than per token.

### The RTX card (Marlin, tier on, windows at KV >85%)

| family | 38 requests | 27 requests | ratio (0.71 if proportional) |
|---|---|---|---|
| MoE (Marlin) | 16.73 | 15.30 | 0.91 |
| dense bf16 GEMMs | 8.70 | 8.24 | 0.95 |
| GDN linear attention | 4.40 | 2.90 | 0.66 |
| QSA attention | 2.18 | 1.92 | 0.88 |
| total | 33.9 | 30.2 | 0.89 |

GPU busy 98%. Here MoE is half the step and **matches a pure memory read to
within ~5%**: uniform routing touches 71% / 59% of the ~43.6 GB of experts,
which at the card's ~1.79 TB/s is 17.3 / 14.4 ms predicted against 16.7 /
15.3 measured. The bf16 group is again flat at about twice its byte cost
(~700 calls a step); GDN again scales with requests.

### What that means

- **Decode leaves the arithmetic idle.** A decode token costs ~5.8B
  multiply-adds (routed experts ~2.36B, ~40%; the bf16 layers ~3.4B, ~60%;
  the model card's "6B activated"). At ~93 requests a B200 step is ~540
  billion multiply-adds in ~19 ms of kernel time, a few percent of what the
  card's arithmetic could do. A matmul stops waiting on bytes only when its
  tokens-per-weight-byte exceeds the card's compute-to-bandwidth ratio: about
  2,000 tokens per call for a bf16 projection, and -- because each routed
  expert sees 10/307 of the batch -- tens of thousands of tokens per step for
  the experts. Decode never gets there; prefill chunks do.
- **So faster arithmetic buys little** (better GEMMs, FP8 activations,
  autotuning), and **more tokens per step buys a lot**: concurrency, then
  speculative decoding.
- **The two cards differ in what binds.** On the B200 experts are ~30% of the
  step and drafting is nearly free. On the RTX experts are ~50% and bounded by
  memory speed, so fewer active experts would pay there (k=6 at 38 requests
  touches 53% instead of 71%, MoE ~16.7 -> ~12.5 ms) and drafting costs more
  (38 x 4 tokens touches ~99% of experts).

Three things a two-point fit of tokens/s against running count cannot tell
apart, which the profiles settle: expert reads are ~50% of the RTX step and
~30% of the B200's, not the ~60% a fixed-plus-per-sequence fit attributes to
weight streaming; the FLOP count is per layer, so the experts are ~2.36B
multiply-adds per token across the 48 layers, not tens of millions; and the
bf16 group is per-call-bound rather than byte-bound, so FP8 weights on
attention are a pool lever (~+8%), not a speed lever.

## Concurrency and the pool

With the tier on, concurrency is set above what the pool holds, and the
running count once the pool saturates is what it actually holds. Measured
while requests waited for capacity (KV >= 85%):

| box | running at worst | median |
|---|---|---|
| RTX, float32 state, 40 or 50 slots, four 1-hour runs | 25-27 | ~30-32 |
| RTX, bf16 state, 80 slots, the traffic replay | ~60 (late in the window) | ~66 |
| RTX, bf16 state, INT4 and the rings' pool, 512-token steps, 80 slots, the traffic replay | 63 | ~67 |
| B200 e307, float32 state, 100 slots, five runs up to ~2 h | 65-70 | ~75-78 |

**Without the tier, oversubscription thrashes.** Contexts that do not fit
cannot be parked on the host; the KV cache thrashes and throughput drops
~20x. So a tier-less arm sits well under the floor, not at it -- and runs
longer than an hour grow contexts further, toward the engine's own floor
(~16 full 90k contexts on the RTX). Chosen without a drafter: **20 on the RTX,
50 on the B200**. A drafter shrinks the pool further (below), and the floors
scale with it: 50 slots thrashed on the B200 with MTP, and the RTX drafter arm
runs **15**. Read `preempt=` before raising any of them.

## FP8 KV cache

Superseded on both cells by INT4 ("INT4 KV cache"), which builds on what
follows. The sparse-attention layers' K/V are stored as e4m3 with a float32 scale per
token and head, computed as each row is written (`fp8_per_token_head`); the
checkpoint has no calibrated KV scales, so a per-tensor format would clip.
The fork carries the kernels (its `TUFA.md`). The indexer's caches and the
query stay bf16. Compute is unchanged: attention reads are 1-1.6 ms of a 17-19 ms decode
step on the B200, so the gain is capacity.

**The page and the block.** A cell is 544 bytes against bf16's 1,024 (the
scales, padded so a token's bytes stay a multiple of the compressed-key
cache's): 53% of bf16. Every cache group shares one page size, so the
attention block doubles, 1,568 -> 3,136 tokens. The hybrid alignment sizes the
page without the scales, so the Mamba pages are padded up to it by the
unifier; seeing the scales would make the block ~6% smaller. The recurrent
state's dtype sets that page, so the bf16 state takes the block to 1,600
("The recurrent state in bf16").

**Pool.** B200, 0.90: 3,948,788 -> 6,381,352 tokens at a 50-slot cap
(+62%), 6.19M at 150 (the dense graph rungs take more). At the admission line
the pool runs 106-130 requests against bf16's 65-86 at 100 slots, ~1.45x
rather than 2x: a running request's Mamba state blocks do not shrink, and its
partial last block wastes twice the tokens.

**What the bigger block costs.**

- **Recompute on every call.** The partial tail a prefix hit cannot cover
  averages half a block, ~780 tokens more. Matched quiet minutes at 50 slots:
  uncached tokens per generated token 1.3-1.4 against bf16's 1.0-1.2, about
  two points of prefix hit rate, and a slower ramp at the start while prompts
  are short. The shipped column's own traffic
  (`duck_premerge_slurm_20260913_215356`) recomputed a mean 1,582 tokens of
  shared prefix on each call extending the previous one, 20% of its uncached
  input. Two levers since: the bf16 state halves the block, and the RTX cell
  lets a hit end every 64 tokens ("Cache hits inside a block").
- **Prefill per step.** In `align` mode a chunk ends on a block boundary, so
  the 8,192-token budget fits two blocks instead of five: 6,272 tokens and two
  concurrent prefills per step. Kept as is (2026-09-13); it bites
  when many returns are recomputed.

**The tier holds the queue.** Per context the tier cost roughly halves (half
the slots: checkpoints halve with the block, attention with FP8; each slot
12% larger). But what it must hold is the waiting set, slots minus what the
pool runs, and that is a difference: at 180 slots it was 51-68 contexts
against bf16's 1-17 at 100. Measured on the B200, 25 games x 8 passes:

| slots / tier | first ~40 min of play | outcome |
|---|---|---|
| 200 / 64 GiB | tier loads fell to <0.2 GB/s against 1.4 GB/s of stores; 72-83% uncached | ~60 running, stopped |
| 180 / 96 GiB | asked depth p50 1,712 -> 2,307 of 2,377 slots in six minutes | running 116 -> 76, stopped |
| 150 / 96 GiB | depth p50 at most 1,310 of 2,377; p99 reached 2,000+ in 22 of 107 intervals | held: 82 full-pool minutes, uncached median 8.6% (worst 23%), 91% of hits from the tier, 0.7% evicted misses, no preemptions |

The hardest stretch is the first compaction wave, when the games that started
together compact together; it disperses over a run. 150 on 96 GiB is close to
the limit for the slowest-returning contexts, not comfortably under it. The
RTX column held an hour at 70 slots, and the Kaggle runs sat more than ten
over capacity there, so it shipped at 60 on the float32 state 
(2026-09-14): a short standing queue rather than a long one, which on Kaggle
also keeps the slot ranking choosing. The bf16 pool runs ~66, and 80 keeps
the same short queue (waits 0-15 on the replay), so it ships at 80. The
B200's 150 was the float32 pool's; the bf16 pool holds about half as many
again, so it runs 200 with 32 GiB more asked of the job, unread 
(2026-09-21; open item 23).

**The harness at this concurrency.** More than ~100 games connecting to the
backend host at once overflowed the Unix socket's listen backlog and died
with `EINVAL` at their first call; the backlog is now deep and the handshake
retried. Game starts are also paced (`RunConfig.start_gap_s`), so a run's
first grants no longer spawn every game's process in one burst.

**Not measured:** faithfulness. It goes through the teacher-forced logprob
comparison (open items). Two identical Flash runs scoring 6.76 and 9.00 is
the scale of noise a score comparison has to clear.

## INT4 KV cache

The sparse-attention layers' main cache is INT4 on both Flash-Next cells
(2026-10-12, on score), the drafter's QSA layer included
(`duck.presets._flashnext_kv_caches`). The write is vLLM's per-token-head
INT4 cache update: a randomized Hadamard rotation, asymmetric 4-bit values two
to a byte, the zero point in the float32 scale's low mantissa bits. The sparse
kernel unpacks the gathered nibbles, takes the zero point off and folds the
scales into the score and probability columns as on the FP8 path; the query is
rotated before the kernel and the output rotated back after it, the gate
applied then. Compute stays bf16, and it is Triton, so any card runs it. The
kernels are the fork's (its `TUFA.md`).

**The page and the block.** A cell is 288 bytes against FP8's 544 at two KV
heads (128 data bytes a side plus the scale, padded to the compressed-key
cache's multiple). The fork counts an INT4 token as the attention group
stores it, the cell plus the compressed-key row, so the block is the smallest
that holds the recurrent page ("The fixed blocks").

**Pool and kernel time (B200).** 13,719,961 tokens against FP8's 7,955,196
on the same 116.82 GiB (1.72x). INT4 runs its own split-dot kernel (each
packed row loaded once, split into its nibbles and dotted against the query's
even and odd elements, the zero points taken off per column): 0.31 ms a layer
against FP8's 0.30 at the duck's decode shape (225 rows, 2,048 selected
tokens). With the rotations as one Triton kernel each, a decode step at 50
requests costs +1.1 ms of kernel time over FP8's 42.4 ms (+2.7% of the
step's wall), the attention kernel +0.57 of it. A first kernel that unpacked
per element took 6.6 ms a layer and played at a third of FP8's generation
rate. Unsaturated, at 50 lanes, no option costs what a reading can see:
INT4 3,611.0 generated tok/s, the FP8 state 3,591.7, both 3,631.0, against
the baselines' 3,548.7 and 3,652.7; a larger pool shows only where requests
wait for capacity.

**Play and acceptance.** A one-hour B200 smoke of the shipped duck, 2 x the
official 25, scored 38.14 against FP8's 40.28 (per game -2.1, se 3.7, higher
in 10 and lower in 12): no difference a smoke can see, and neither run was
pool-bound. Draft acceptance fell from 2.40 to 2.28 a draft, which is the
drafter's own INT4 cache: a drafter cache in FP8
(`speculative_config["kv_cache_dtype"]`, which composes with no fork code)
gave 2.38 back. Seal sweep 2 read INT4 in on its own play. On the RTX, where
the pool binds, INT4 alone read +31.4% generated ("The RTX column on the
traffic replay").

**The FP8 indexer cache** (`attention_config.indexer_kv_dtype="fp8"`,
upstream #54890 in the fork). Beside an FP8 main cache it was worth ~2.8% of
the pool and was declined for the score (2026-09-26, without a reading of its
own); beside INT4 the bf16 indexer is a larger share of a token, and it reads
+5.1% of the pool and +5.6% generated on the RTX. In since 2026-10-12, on
score; its own accuracy cost is unread.

## The fixed blocks

**What a request holds whatever its length** (the engine's KV-blocks trace,
`VLLM_KV_BLOCKS_TRACE`, on a B200 replay at k=3, unit 64, 50 running): its
running state, one block in each of the three GDN groups; exactly its
context's blocks in the attention group; one QSA compressor ring (13 layers of
8 rows); and ~0.94 blocks no request lists, the decode tails' landing blocks
(RecoverSSM writes the next unit boundary's state into a block taken for the
step that may cross it) and a few copy-on-write sources. About five fixed
blocks a request, each costing the widest group's bytes.
`tools/kv_capacity.py` takes its fixed blocks from this trace, not from vLLM's
align-mode reservation.

Two fixes, both free, no score effect and no recompute traded 
(2026-10-11):

- **An INT4 token counted as stored** (the fork, always on). The alignment
  counted the packed cell without the compressed-key row, so every recurrent
  page was padded to the undercount and a GDN block sat 13% empty. Counted as
  stored, a cell's block is the smallest multiple of its unit that holds the
  bf16 recurrent page (the pages and a token's bytes: `tools/traffic_launch.py`,
  `_quantized_cache_block`): 3,120 at four drafts (unit 80), 3,008 at three
  (unit 64). The fixed blocks shrink 12.5%, ~1.5% of the pool at 60 running.
- **The rings in a pool of their own** (`transient_kv_pool`, on in both
  cells). A ring is a few kilobytes and took a whole shared block, 16 MB on
  the RTX: ~1 block a request back, ~2.4% of the pool at 60 running.

On the RTX the two read +1.8% (the table in "The RTX column on the traffic
replay"), the decoding batch 6.5% larger. A decode-only step
costs 27.5 ms + 0.50 ms a request on both runs' line, so the fixes cost no
step time and a request adds about half the average request's share; prefill
takes the rest, a step that prefills 1-1.5 requests costing 98-127 ms
against 58 decode-only, in a fifth to a quarter of steps.

## 512 tokens a step

The RTX cell steps at most 512 tokens, the largest CUDA graph the card
captures: above it a step runs eagerly, and at 600-1,000 tokens (a full
decode batch plus a prefill chunk) the GPU idled ~50 ms a step on the
kernels' launches. +2.7% on the traffic replay (the same table), from two
effects:

- **The pool grows 7.3%.** The engine sizes its activation reserve for the
  largest step, so a 512-token budget frees ~2.8 GiB for KV; ~+2% of the
  reading. A 1,024 budget would free nearly as much.
- **Prefill ~8% cheaper a token** (117 against 127 us of wall a token): every
  mixed step replays graphs, though a prompt spreads over more of them, each
  with its fixed cost; ~+1%.

A full decode batch (80 x 4 tokens) leaves ~190 tokens a step for prefill,
~3x the replay's per-turn prefill. The decode gate's prefill runs ~37% slower,
which bounds a backlog, not a game's start: real games start from a short,
cached prompt. The B200 keeps 8,192: four drafts make a decode five tokens,
so its 100 slots' decodes alone would fill 500 of 512. A step budget below a
recurrent block also needs the next section's admission change wherever a
host tier is in play.

## Admission on a spent budget

At 512 tokens a step with a host tier, the scheduler stopped admitting. Its
waiting loop ran only while the step had budget left, and below a step budget
smaller than a recurrent block a running prefill's chunk spends the whole
budget every step, so the loop looked up ~18 waiting requests per thousand
steps (744 at 3,300 tokens a step). A load from the tier needs no budget, but
it needs a lookup: parked threads waited unread until the tier evicted them,
and each miss re-prefilled its whole context, holding the budget for 50-150
steps. The fork admits budget-free loads on a spent budget and sets requests
that need compute aside for the step (`TUFA.md`, "Admission on a spent step
budget").

The RTX cell meets the same condition, 512 tokens a step
beside a 50 GiB tier; its default reading ("The RTX column on the traffic
replay") ran before the change and did not collapse, so the tier's pressure
decides when it bites.

## The RTX column on the traffic replay

The knob pass of 2026-09-20/21, every row one run on
`workload_120lanes_20260920.b64` (`traffic_benchmark.md`); the readings are
the settled fifteen minutes, the rest the engine lines over the same
stretch, and each row's archive directory holds its provenance.

| arm | lanes | gen / uncached tok/s | running | capacity waits (min / med / max) | hit, tier + pool | allocator failures | archive |
|---|---|---|---|---|---|---|---|
| the column as it shipped (float32 state, 0.97, 60 slots) | 60 | 1,041.0 / 2,976.8 | 52.6 | 0 / 1 / 10 | 88.9% | 3 | `20260920_1301_rtx_baseline` |
| bf16 state | 80 | **1,418.4** / 2,159.9 | 66.4 | 1 / 7 / 15 | 94.4% | 10 | `20260920_1407_rtx_bf16_80lanes` |
| + expandable segments, the 64 MB bound (**the baseline from here**) | 80 | 1,443.3 / 2,015.4 | 66.5 | 0 / 4 / 14 | 95.0% | **0** | `20260920_1501_rtx_bf16_expandable_indexer64` |
| + b12x, FP4 activations | 80 | 1,397.3 / 1,990.4 | 66.4 | 0 / 8 / 16 | 94.9% | 0 | `20260920_1845_rtx_b12x_fp4` |
| + FlashInfer GDN prefill | 80 | 1,439.1 / 2,029.9 | 66.4 | 0 / 8 / 17 | 94.9% | 0 | `20260920_2039_rtx_gdn_prefill_flashinfer` |
| + Triton GDN decode | 80 | 1,449.6 / 2,032.5 | 65.9 | 0 / 8 / 15 | 95.0% | 0 | `20260920_2141_rtx_gdn_decode_triton` |
| utilization 0.975 (a stability test at 100 lanes; not comparable) | 100 | 1,508.2 / 2,011.7 | 71.3 | 13 / 19 / 33 | 94.4% | 0 | `20260920_2315_rtx_util0975_100lanes` |
| + `prefix_match_unit` 64 | 80 | 1,457.1 / **1,745.2** | 66.6 | 0 / 6 / 13 | 95.7% | 0 | `20260921_1212_rtx_prefix_unit64` |

The first row's window was 20 minutes read from 300 s, before the settle was
lengthened; read from its own flat stretch it gives 1,062, so its figure is
~2% low. It was not repeated: the next run changed the window, the
lanes and the state together, and the state's gain clears the confounders.

**The baseline in context.** 1,041 generated tok/s at 52.6 running against
the four L4s' 552 at ~52 (the addendum): the RTX is 1.9x the L4 column at
the same running count. The tier is the workhorse on every Kaggle column:
the pool serves 16.4% of prompt tokens here and the host tier 86% of the
rest, the four L4s' pool 2.5-5.3% with the tier at ~90%, where the B200
serves 87% from its pool alone because its pool holds the working set. So
"frees pool" shows on a deployment box as tier relief and less recompute --
2.86 uncached tokens per generated token at the baseline against the
workload's own floor of 1.19 -- rather than as a high pool hit rate.

**The quantized caches, 2026-10-11/12.** A second pass on
`workload_long100_20260929.b64` with the decode gate, read after a 300 s
settle, every row one Kaggle run and pool-bound throughout; the RTX cell then
shipped FP8 KV and 40 slots. The FP8 state's rows ran at 0.9437, since its
70-sequence cap ran out of GPU memory at 0.9537 as play started.

| arm | lanes | utilization | gen / uncached tok/s | running | archive |
|---|---|---|---|---|---|
| the cell as it shipped (FP8 KV, k=3, unit 64, 50 GiB tier) | 48 | 0.9537 | 1,975.1 / 654.5 | 33.3 | `20261011_1545_rtx_kvstate_base48` |
| INT4 | 64 | 0.9537 | 2,595.8 / 843.8 | 53.4 | `20261011_1628_rtx_kvstate_int4_64` |
| + the FP8 recurrent state | 70 | 0.9437 | 2,761.5 / 922.2 | 56 | `20261011_1745_rtx_kvstate_combined_70_u9437` |
| the same at k=2 | 76 | 0.9437 | 2,559.0 / 791.2 | 58.4 | `20261011_1907_rtx_kvstate_combined_k2_76_u9437` |
| + the FP8 indexer cache (k=3) | 72 | 0.9437 | 2,914.9 / 948.6 | 59.3 | `20261011_2049_rtx_kvstate_combined_idx_72_u9437` |
| + the fixed-block fixes | 76 | 0.9437 | 2,966.6 / 990.7 | 65.5 | `20261011_2248_rtx_kvstate_fixedblocks_76_u9437` |
| + 512 tokens a step | 76 | 0.9437 | 3,047.5 / 1,006.9 | 68.9 | `20261012_1245_rtx_kvstate_b512_76_u9437` |
| **the cell as it ships**: the above less the FP8 state, 80 slots | 80 | 0.9537 | **2,919.8** / 885.8 | 67.1 | `20261012_1603_rtx_defaults1012_80` |

The shipped cell reads +47.8% on the one it replaced, utilization held
through the start of play, no OOM, the pool 5,153,639 tokens; 4.2% under the
same stack with the FP8 state at a hundredth less utilization. k stays 3: at
k=2 a draft is accepted 1.58-1.67 times against 2.14-2.29 at three, ~2.6
tokens a step a request against ~3.2, and the cheaper step does not make it
up (a request's step cost barely depends on its drafted tokens, "The fixed
blocks"). FP8 dense weights are not pursued: their ~+4% does not outweigh
their cost to the score (2026-10-11).

## The recurrent state in bf16

The checkpoint's `mamba_ssm_dtype` is float32, which every cell followed
through `mamba_ssm_cache_dtype="auto"`. The state sets the page every cache
group shares -- vLLM sizes the attention block "to ensure that attention
page size is >= mamba page size" -- so a bf16 state halves the block, 3,136
to 1,600 on both Kaggle cards, and with it the fixed state blocks a running
request holds and the half block its partial last block wastes. It is a
setting, not fork work, and the fused GDN decode kernel takes a bf16 state
on any card from 8.0. The state is rounded at every token and carried the
whole context, so it went through a score check first: ARC2's sweep, on
the same checkpoint family and mechanism, held (2026-09-20); on
ARC3 the B200 held too, the demo duck on the 25 official games at 37.2 over
4 passes against 37.0 over 6 for `main`'s float32 state (2026-09-21, 200
slots against 150), and the RTX's reading is the nine-hour run's (open item
9).

**+36% generated on the RTX** at 80 slots (1,418.4 against 1,041.0), and
**+24% on the four L4s** at 70 (the addendum). Three things moved:

| | float32, 60 slots | bf16, 80 slots |
|---|---|---|
| generated tok/s | 1,041.0 | **1,418.4** |
| running | 52.6 | 66.4 |
| generated per running request | 19.8 tok/s | 21.4 tok/s |
| uncached per generated token | 2.86 | **1.52** |
| vLLM's KV cache, tokens | 2,352,397 | 2,635,093 (+12%) |

**The lanes explain none of it**. The float32 run was pool-bound,
not slot-bound: a queue in 8 of its 14 scored minutes, 53.2 of its 60 slots
running, the pool never below 78.6%; twenty more slots would have queued.
The rise in the running set is pool arithmetic: the pool grew x1.120 and a
request's footprint shrank x0.882 (44,218 to 38,981 pool tokens per running
request), 1.120 / 0.882 = 1.270 against a measured 67.6 / 53.2 = 1.271. The
extra lanes only let the larger pool be filled; without them bf16 would have
been slot-bound near 57 running and under-read itself. Recompute per
generated token nearly halved, the block halving as predicted plus eviction
relief; and each running request got faster, 19.8 to 21.4 tok/s, where more
concurrency alone would slow it -- the step spends less on recompute and
more on decode.

**Int8 checkpoints are no second lever** (2026-09-19). SGLang keeps
the recurrent states its prefix cache holds as int8, but vLLM's align mode
keeps only the live state per running request: passed checkpoints are freed
pages the allocator reclaims first, and the tier never copies them. So int8
would shrink no running request, only the tier's latest-boundary states,
and the tier does not bind. It matters only with many kept checkpoints.

**FP8 rows, read out on score.** The fork's `mamba_ssm_cache_dtype="fp8_e4m3"`
(opt-in, Qwen4Exp only; its `TUFA.md`) stores each state row as e4m3 values
with a float32 scale, the row's maximum, rounding stochastically at every
write so that an update much smaller than a step lands on average. Each GDN
page, and with it each request's fixed state blocks, is ~40% smaller
(`tools/traffic_launch.py`, `FP8_STATE_SAVING`). After the fork's fixes
(an integer hash for the rounding, rows 16-byte aligned, the commit replaying
to its stops) it costs +1.35 ms of kernel time a B200 step (+3.6% of the
wall), and on the RTX it reads +6.4% generated over INT4 alone at a hundredth
less utilization (the table above). Seal sweep 2 read it out on its own play
(2026-10-12). Its first B200 smoke, before the fixes, scored 25.22
against 40.28 in an hour played 31% slower a request, so its accuracy cost
has not been separated from its speed.

## Expandable segments and the indexer bound

Two memory arms taken together on the bf16 run, for robustness rather than
speed: expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
with `enable_cumem_allocator` holding the KV cache, which vLLM demands
beside any KV connector) against the fragmentation that ends a long run,
and the indexer's 64 MB scoring bound (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`;
the addendum, "Memory in play"). Together: 1,443.3 against 1,418.4
generated, +1.8%, inside the error, with the running set and the pool
unmoved -- free, not a gain, and no gain was possible: the bf16 run's ten
allocator failures all fell inside the warm-up the reading leaves out,
expandable segments change how memory is mapped rather than any rate, and
the bound only splits a long prompt's scoring into more launches (the four
L4s read +1.6% for taking it off, inside their spread). What they bought is
**allocator failures 10 to 0**, and the risk they remove sits at a run's
start, where the L4 column's engine died on a failed retry. Both ship on the
RTX. The RTX runs one rank, so there is no custom all-reduce for
expandable-segment memory to conflict with; on the four L4s there is, and
the arm is not viable there (the addendum).

## Cache hits inside a block: `prefix_match_unit`

The block is baked in: every cache group shares one page size, the
attention block grows until its page covers a recurrent state page, and
FP8 KV doubles it ("The page and the block"), so a hit lands on a block
boundary and a call recomputes half a block on average. Upstream's partial
hybrid hits (#46384, in the fork) decouple the match from the page:
`cache_config.prefix_match_unit` hashes every that many tokens, and a hit
may end inside a block wherever a recurrent state is stored, which align
mode keeps at a prompt's end. Every group's block must divide by the unit
or the engine refuses to start: 64 divides both the float32 block (3,136 =
64 x 49) and the bf16 one (1,600), 56 the 27Bs' 1,568.

**64 on the RTX** (the table above): 1,457.1 against 1,443.3 generated
(+1.0%, inside the error) and uncached prompt 1,745.2 against 2,015.4
(-13.4%), the running set unchanged. **Off on the four L4s**, where under
the bf16 state it cost 3.9% generated: each running request held ~2.5 more
blocks, so 10% fewer ran (the addendum, "Hits inside a block"). The one
configuration difference is the layout (two stages of two ranks against
one card); four-way tensor parallelism halves the block
to 800, which 64 does not divide. The cause is unexplained (the addendum's
open items).

A coarser unit is no compromise. The cost is one extra block per GDN group
for every hit that ends inside a block -- the boundary state is registered
in a copy block and copied again on use -- and that block is the same size
whatever the unit. The unit only sets how often a hit ends mid-block
(1 - unit/1600 of them, for prompts spread evenly), while the saving per hit
is about (1600 - unit)/2 tokens: both scale with 1600 - unit, so 400 or 800
land between off and 64 on both axes. Decision (2026-09-21): 64 on
the RTX cell, off on ARC2's, and neither B200 column sets it, unmeasured
there. **With drafting the B200 cell keys too (2026-09-26)**: read
on the B200 at 16 and 50 lanes, a unit cut uncached prompt 18% and bought
+0.6% generated without a drafter, +8% with a k=1 drafter. The B200 keys
every 80 tokens and the RTX every 64, chosen when the unit had to divide the
block the alignment derived (1,840 at four drafts, which 64 does not divide);
the INT4 cells choose their blocks as multiples of the unit ("The fixed
blocks"), so that no longer binds either cell.

## Speculative decoding

Flash-Next has one MTP layer (512 experts, bf16, excluded from quantization),
which vLLM runs autoregressively for each draft token.

**What ships (2026-09-29).** Drafting is on in both cells, beside the host
tier. Park mode (`kv_offload.md`) is what let the two run together; the
first arms below (2026-09-13) ran without a tier and were turned down for
it.

- **The head:** the MTP layer fine-tuned on ARC2 and ARC3 play, NVFP4
  experts on Marlin, and a 16k-id draft vocabulary scattered into the
  target's (fork `4b1cb6c9`), which covers 99.3-99.8% of generated tokens.
  Drafts are sampled and verified against their probabilities.
- **GDN RecoverSSM:** a verify runs from the step's starting state and
  recovers the accepted one, so no recurrent group holds a state block per
  draft token (+17.6% pool on the B200). Its commit also writes the state at
  the unit boundary a verify window crosses, which the decode tail needs.
- **The prefix block a drafter's lookup drops is kept** (upstream #53388): a
  continuation shares the token its cached row depends on.
- **k=4 on the B200** (2026-09-28): +9% generated over k=3 on an
  underfilled 100-lane replay, and more at low concurrency, which is the tail
  of every run and most test runs.
- **k=3 on the RTX** (2026-09-29). Over two 9-hour runs on the
  official 110, four drafts read 2-3% ahead (1,916 against 1,875 generated
  tok/s in minutes free of stalls; the same score 34 minutes sooner). But
  acceptance on the official games is inflated, because they are easier than
  the private set, and lower acceptance erodes four drafts' edge first.
- **The unit and the block.** A cell's attention block must be a multiple of
  its unit; the INT4 cells choose it as one ("The fixed blocks").
- **The QSA ring bounds k.** The drafter's QSA key-state ring holds a
  group's committed keys plus every drafted row, in whole groups of 4 (8
  slots up to k=4, 12 for 5-8, 16 for 9-12, 20 for 13-16), and it must
  divide the attention block (`qsa_cache.py` asserts it). The block grows
  with k, always a multiple of 16 (on the B200 1,600 plain, 1,792 at k=3,
  2,000 at k=7, 2,112 at k=9), so k <= 4 and 9-12 always run; on the B200
  k=5-8 cannot (k=7 was refused) and k=13 can. Rounding the block up to a
  multiple of the ring where the platform sizes it (`platforms/interface.py`)
  would free every k, at a few tokens a block.
- **What it bought on the RTX:** the stack read 2,216 generated tok/s on the
  traffic replay against the undrafted column's 1,463 (+51%), and 2,317 once
  park mode's tail copies were in (2026-09-28).
- **Kept opt-in and off:** FP8 dense layers (+4.0% on the RTX) and the FP8
  indexer cache (~+2.8% pool). Both reduce the target's accuracy, a trade
  declined for the score (2026-09-26). The draft top-k/top-p mask is
  opt-in too (`flash_optimization_arc2.md`, open item 23).

**The first arms (2026-09-13).** Everything from here to the section's end
is that history.

**What a speculative arm changes, and why** (decided 2026-09-13):

- **No host tier.** The victim connector and a drafter cannot both hold the
  KV (open items).
- **No admission watermark and no request pacing** (`min_request_gap_s`).
  Both exist for the tier: the watermark reserves room for running requests
  to grow while the overflow waits in the tier, and pacing lands a
  continuation after its blocks have left for it. Without a tier neither has
  a job. (`settings.md` files the watermark as U-00; it becomes a function of
  whether the tier is on when this ships.)
- **Concurrency under the floor, scaled for MTP's pool cost:** 15 (RTX). The
  B200's 50 was derived from the drafter-less floor and thrashed (below).
- **`draft_sample_method="probabilistic"`**, per `settings.md`.
- **Dense capture off** in the experiments.

**MTP costs pool.** On the B200 the pool fell from 3.95M tokens to 2.89M
(-27%) with MTP on -- more than the drafter's bf16 weights explain; the MTP
layer's own attention KV is stored per token. It is also what the first B200
arm got wrong: at 50 slots the k=3 run reached 95-99% KV with preemptions, and
generation fell from ~2,000 to ~750 tok/s, prefix hits to 4-9%, within
~15 minutes of play -- the thrash the tier-less floor warns about. Both its
profile windows had already opened, so the read below stands; the throughput
half of that run does not.

**The B200 arms (jobs 2588868 / 2588872: 50 slots, tier-less, 5 s pacing
still on, windows at KV 25%).**

Decode-only steps, drafter and sampling included, ms per step:

| family | no drafter, 42-43 req | k=3, window 1 (41 req) | k=3, window 2 (46 req) |
|---|---|---|---|
| MoE | 5.42 / 5.34 | 4.84 | 7.06 |
| dense bf16 GEMMs | 5.22 / 5.23 | 6.40 | 6.93 |
| GDN | 1.84 / 1.87 | 3.30 | 4.68 |
| QSA attention (+ MTP's) | 0.94 / 1.09 | 1.50 | 2.53 |
| sampling | 0.84 / 0.86 | 1.34 | 1.80 |
| step | 16.8 / 17.0 | 20.8 | 27.3 |

Acceptance held at 55-56% throughout (1.67 accepted tokens per 3-token draft;
73% / 54% / 41% by position). So a decode step carrying ~2.7 tokens per
request cost 1.2-1.6x a plain one: per accepted token, decode kernel time
fell from ~0.40 to ~0.22 ms. MoE and the bf16 group stay nearly flat at 4x
the tokens (MoE moves up a TRT-LLM tile band by ~180 tokens), GDN and
attention scale with verify tokens -- the premise holds.

**End to end it bought far less.** At matched running counts (37-42) before
the pool gave out, generation ran 2,450-2,600 tok/s with the drafter against
~2,050-2,230 without: **about +15-20%**, not the ~1.7-2.3x the decode steps
suggest. Where it went:

- **Prefill.** Faster games feed more new prompt tokens per second (7,900-9,100
  against 2,000-7,100), and the drafter's smaller pool holds fewer prefixes
  (prefix hits 86-90% against 93-97%), so more of each prompt is recomputed.
  Mixed steps under the drafter are expensive: in window 2, 45 of 100 steps
  carried 331k prefill tokens at ~194 ms of kernel time each and took 86% of
  the window's wall time. The 100-step windows sample that mix badly (window 2
  of the plain arm had 2 mixed steps); the per-minute engine lines are the
  measure of it.
- **Then the pool.** At 50 slots the drafter's pool (-27%) filled and thrashed
  (above).
- **Not the host.** Once past the early-play window, GPU idle was 14% in both
  arms; window 1's 58% idle was a prefill-heavy start, not drafting overhead.

Also: a drafter run's traces are larger (32-42 MB against ~15), and closing
window 2 stalled the engine for a full minute (a `gen=0` engine line), so the
profiler costs more under a drafter.

What follows: on the B200 at this concurrency, prefill and prefix retention --
not decode -- bound what drafting can deliver.

The RTX k=3 run (15 slots, Marlin) first failed at load: `moe_backend` is one
setting, it reaches the MTP layer's unquantized experts too, and Marlin serves
no bf16 weights. vLLM's `speculative_config.moe_backend` exists for exactly
this, but only the V1 proposer applied it; the V2 draft loader now does too
(fork ff6fab89), and the drafter's experts run on triton.

**The RTX arm (Kaggle, 15 slots, Marlin target, triton drafter, profiled).**
Decode-only steps carry 15 x 4 = 60 tokens:

| family, ms/step | k=3, 15 req (two windows) | no drafter, tier on: 38 / 27 req |
|---|---|---|
| MoE (Marlin target + triton drafter) | 19.4 / 21.2 | 16.7 / 15.3 |
| dense bf16 GEMMs | 12.8 / 12.7 | 8.7 / 8.2 |
| GDN | 4.8 / 4.8 | 4.4 / 2.9 |
| QSA attention | 1.5 / 2.2 | 2.2 / 1.9 |
| kernel total | 41.3 / 43.8 | 33.9 / 30.2 |

Acceptance 54% (1.62 per draft; 72% / 52% / 38% by position). MoE stays half
the step: 60 tokens touch ~86% of the target's experts, and the drafter's own
bf16 experts add ~3.5 ms over its three passes -- on this memory-bound card the
drafter buys its tokens near full price. Prefill runs ~12,100 tok/s with the
drafter, and mixed steps took 70-82% of each window's wall time. End to end,
**804 tok/s at 15 running** (`gen=` counts emitted tokens, after rejection),
against ~786 at 27-29 and ~893 at 36-38 running for the plain tier-on column:
the drafter matches the tier's saturated phase with half the requests, and
beats what a tier-less plain run could hold, but not the tier.

**Why prefix hits collapse under a drafter.** With MTP on, the engine's prefix
hit rate fell from >90% to ~60-80%, and far more new prompt tokens came in. The
per-call spans (`llm/response` events: cached and uncached input tokens) of the
two B200 arms separate the causes. For consecutive calls in a game, genuinely new
content is this prompt minus the previous prompt and reply; anything uncached
beyond that is recompute ("excess"):

| per call | no drafter | k=3, before the pool thrashed | k=3, whole run |
|---|---|---|---|
| cache hit rate | 95.2-96.6% | 83.3% | 74.3% |
| median excess | 819 | 2,538 | 2,668 |
| clean hit (<200 excess) | 12% | 0% | 0% |
| a partial block (200-1,400) | 77% | 0% | 0% |
| one to two blocks (1,400-3,300) | 11% | 88% | 79% |
| full recompute (>=80% of prompt) | 0% | 0% | 7% |
| uncached tokens that were genuinely new | 32-36% | 14% | 8% |

- **A block per hit.** vLLM drops the last matched block of every prefix hit
  under an EAGLE/MTP drafter ("drop the last matched block to force recompute
  the last block to get the required hidden states for eagle drafting head",
  `single_type_kv_cache_manager.py`), and the aligned split backs off one block
  to match (`scheduler._mamba_block_aligned_split`, `use_eagle`). On an
  attention model that is 16 tokens. Here a block is 1,568 -- forced up by the
  GDN state alignment -- and the recurrent state exists only at block
  boundaries, so the target cannot recompute just a short tail: it redoes the
  block. The median excess rises by ~1,700 tokens per call, and nearly every
  call pays it.
- **Eviction on top, once the pool is tight.** Without the tier an evicted
  prefix is gone; in the thrash 7% of calls recomputed almost the whole prompt.
  On the RTX (tier off, a ~1.07M-token pool, 15 games at 20-40k-token prompts)
  this is the likely extra step down toward ~60%.
- **Not lost decode checkpoints.** A verify step can cross a block boundary, but
  `postprocess_mamba` (`vllm/v1/worker/mamba_utils.py`) copies the intermediate
  speculative state at the boundary into its cache block, so generated tokens
  keep their checkpoints.

That per-call recompute is the prefill that bounded the B200 gain. Ways around
it are open item 2.

**Decision (2026-09-13), since superseded ("What ships" above): speculative
decoding stays off on both cells.** On the B200 it bought ~+15-20% generation
before prefill and pool bound it; on the RTX it draws level with the tier it
has to give up. What would change that is running it *with* the tier, a
drafter that accepts more, and a pool that holds more -- open items 2, 1 and
6-7.

Earlier cluster measurement for reference: two draft tokens, 73% / 54-57%
accepted by position with the whole head and a sampled draft; 1320-1440
tok/s MTP-only against 2400-2600 tier-only at 100 slots.

## Checkpoint, active experts and MoE backend: scores on the B200

One change each from the `slurm-b200` cell as it shipped then (e307, TRT-LLM, 10
routed experts, 100 slots), scored on the official 25 games. Every arm ran
from `cuda-home-path` at 6452e00c (off this branch at 6f7dcd64, since
merged; vLLM 1ab35488), launched by
`arc3-experiments/flash_arms/launch.py`. Base, e256 and Marlin ran a
4-pass run and two 8-pass replicas, merged into 20 seeds each; the other arms
ran 4 seeds. The unpruned checkpoints ran 70 slots to fit their smaller pool
(concurrency does not move the score). Comparison:
`/data/shared/research/arc_3_results/shared/flash_arms/compare_merged/`.

| arm | change from baseline | seeds | mean score | Welch p vs baseline | tokens per game | first-hour gen tok/s | tokens generated in the first hour (4-seed run) |
|---|---|---|---|---|---|---|---|
| **baseline** | e307 (60% of experts) | 20 | 28.19 | -- | 166k | 2,532-2,760 | 9.77M |
| e256 | 50% of experts | 20 | 24.42 | 0.0006 (baseline better) | 166k | 2,452-2,688 | 9.68M |
| Marlin | `moe_backend="marlin"` | 20 | 30.79 | 0.0337 (Marlin better) | 175k | 1,956-2,032 | 7.04M |
| k7 | 7 routed experts per token instead of 10 | 4 | 19.11 | 0.0054 (baseline better) | 163k | 2,800 | 10.08M |
| unpruned, RadixArk | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (the prune's source), 70 slots | 4 | 25.27 | 0.0253 (baseline better) | 161k | 2,194 | 7.90M |
| unpruned, NVIDIA | [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), 70 slots | 4 | 24.62 | 0.2707 | 167k | 2,099 | 7.55M |
| unpruned, Primitive | [`primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8), 70 slots, triton GDN decode kernel | 4 | 26.29 | 0.5058 | 165k | 2,030 | 7.31M |
| FP8 KV | `kv_cache_dtype="fp8_per_token_head"`, 150 slots and a 96 GiB tier for the doubled pool (a separate run, below) | 8 | 29.43 | 0.3889 | 165k | 3,148 | 11.33M (the 8-pass run) |

First-hour throughput is the 6th ten-minute report's generated tokens over
the time since the games started (the report's own figure includes engine
start); the ranges span an arm's runs. Tokens generated in the first hour
are that rate over the hour after the engine came up, from the 4-seed run
alone, so every arm is read from one run of the same shape.

The FP8 KV arm is one 8-pass run from `fp8-kv` at 3e10d8ba (vLLM
b469bbde, which carries the FP8 KV kernels), `duck_fp8kv_long_20260913_151748`
under `platypus_demo/`, added to the same comparison. Beside the cache format,
its harness differs from the other arms only in scheduling and logging (the
host tier's `engine_log`, a pause between game starts), none of it model-visible.

Conclusions:

- **Strong suggestion that 60% of experts beats not pruning**; e307 stays the
  baseline. 50% does hurt, but may still be the best balance of performance
  against throughput.
- **Removing active experts strongly hurts performance** (k7).
- **Marlin scores higher on the B200, but not at constant token cost**: the
  games actually play longer with it. Not why it ships (that is the RTX,
  "The MoE backend"); the check that it is not worse.
- **The two other checkpoints tried** (NVIDIA's and Primitive's, linked above)
  **do not seem to affect performance much** (null result).
- **FP8 KV is baseline.**

## Competitor tricks (2026-10-01)

What the public Kaggle notebooks do that we did not, taken in three classes: safe
items go in as code; isolation items go in as code once a benchmark of their
own qualifies them; sweep items are options, off by default, for a score sweep
after the merge.

**The benchmark.** The B200 traffic replay of `workload_long100_20260929.b64`
at 140 lanes, shipped sampling, a 1,800 s window read from 900 s, the park
log, profile windows after the reading, dgx06; run-to-run noise about ±2%.
The cell's own 100 lanes are not pool-bound under park mode (98-100 running,
never a capacity wait); at 180 the pool thrashes (nine times the uncached
prompt for a third of the generation); at 140 every reading minute waits for
capacity at KV 88-91%. Lanes past the workload's 100 replay its lanes again,
nonced.

| run | what changed | generated tok/s | uncached tok/s | accepted a draft; by position |
|---|---|---|---|---|
| baseline | `main` 6f20219e | 4,989.9 | 1,587.0 | 2.84; 86 / 75 / 65 / 58% |
| phase 2 | T2, T3, T13, S8, S2's variable (still two captures) | 5,024.2 | 1,617.5 | 2.86; 87 / 75 / 66 / 58% |
| phase 2b | one capture (the fork's gate), S5's shards | 4,914.2 | 1,568.5 | 2.82; 86 / 74 / 64 / 57% |
| phase 2c | every safe item | 4,953.3 | 1,592.3 | 2.83; 86 / 75 / 65 / 57% |
| T5 | the drafter's QSA indices shared across MTP steps | 4,890.5 | 1,545.7 | 2.85; 87 / 75 / 65 / 58% |
| T12 | upstream #58114 | **5,106.1** | 1,593.2 | 2.80; 85 / 73 / 64 / 57% |
| T8 (option) | relaxed draft acceptance at 0.3 | **5,528.1** | 1,712.4 | 3.28; 93 / 86 / 78 / 71% |
| T1 (option) | AutoRound experts | 5,078.0 | 1,610.1 | 2.80; 85 / 74 / 64 / 57% |
| T7 (option) | the QAD checkpoint | 4,692.9 | 1,510.0 | 2.83; 86 / 74 / 65 / 58% |

Run directories `traffic/replay_140lanes_*_20261001_*_ct_*`; every reading
waited for capacity in all its minutes, with no preemption and no OOM.

The baseline's profile (two windows of 100 steps, alike within 1%):
decode-only steps take 63.2-63.4 ms of kernel time at 95-97 requests -- the
MoE 42.3%, GDN 16.8%, attention 14.4%, the dense bf16 GEMMs 11.4%, sampling
5.1%, elementwise 4.2%, the hyper-connections 2.8% -- and mixed steps 87-88
ms (30-34% of steps); a step takes 82-84 ms of wall time, the GPU idling
14-15% of the span on the host, and the tier's offload copies 1.05-1.14 s
device-to-host a window on its own stream, outside the steps.

**Memory and start-up.** Phase 2 on: the model takes 49.61 GiB of the card
(50.8; T3), the start-up profile's peak activation 1.02 GiB (4.86; T2), and
the pool 116.7 GiB = 7,946,458 tokens (+1.1%). With one capture (S2) the
closing accounting puts the graphs at 4.33 GiB outside the 0.9488 budget, so
~173.5 GiB is in use, the old 0.97 budget's 173.0 and half a GiB: no replay
met an OOM. Engine start-up falls from 318 to 187 s. A cold target load read
268 s on S5's re-sharded copy against 431 s, which mixes S5, S8 and the
mount's variance.

S2 also showed what the profiler's reservation is, for the L4 work. One list
of sizes (`cudagraph_capture_sizes`) feeds the target's piecewise graphs, its
FULL graphs (one per drafted decode width) and the drafter's; the profile
samples only two FULL graphs. Its reservation (0.92-1.12 GiB a card on the
four L4s, 1.55 GiB on the RTX) is 3.5-4.5 times the graph pool the real
capture measures (0.20-0.28 GiB; ~0), but not slack: vLLM's closing
accounting puts the L4 cards 0.35-0.42 GiB over their budget and the RTX on
it. The rest is most likely the driver's memory for the captured graphs,
counted as non-torch, which would grow with the number of graphs and their
nodes rather than the token ceiling; unmeasured.

**Isolation items.** T12 qualified: +3.1% at unchanged acceptance, and its
profiles say why: the GPU waits on the host 8-9% of the span instead of
13-15%, and steps take 73-75 ms instead of 82-88. T5 gained nothing and
stays off.

**The sweep options.** T8 lifts acceptance by 17% and generation by 8.3%:
only decode steps gain, and the pool still bounds the running set; the price
is a sampled distribution sharpened toward the draft. T1's AutoRound experts
leave 2.9 GiB more of the card (46.73 GiB against 49.61) and 2.1% more pool
but generate the same, at the same acceptance: the drafter, fine-tuned on
the NVFP4 target's hidden states, drafts as well for it. T7's MXFP8 attention
and shared experts decode 8.1% slower than the bf16 layers they replace at
this batch (FlashInfer's cute-dsl MXFP8 kernel, and Marlin for the layers
below its 128-wide minimum), with 2.6% more pool and the same acceptance.
The score sweep decides all three, beside T9 and T10 (FP8 dense layers, the
FP8 indexer cache), options declined for the score on 2026-09-26. The arms
are `relaxed_accept`, `autoround` and `qad` (`tools/traffic_launch.py`).

**On the RTX.** The branch at `57e06002` played the one-hour demo (40 slots,
k=3; `kaggle_pulls/ct_rtx_57e06002_20261001`) against run H, `main` on the
same cell:

| from kernel start | run H | competitor-tricks |
|---|---|---|
| install done | 200 s (pip) | ~80 s (the record replayed in 70.4 s) |
| engine process starts | 254 s | 145 s |
| target weights loaded | 473 s, after 208 s of loading | 290 s, after 133 s |
| drafter weights | 23.2 s | 0.55 s (prefetched) |
| engine ready, games start | 744 s | 457 s |

One capture took engine initialisation from 238 to 158 s, its graphs 2.01
GiB outside the 0.9537 budget: ~92.6 GiB in use, where run H's 0.97 budget
put it. The pool held 2,614,631 tokens (+3.4%). From games start, over the
run's 2,555 s of play, TAAF counted 5.6% more generated tokens than run H at
the same point (the engine's lines +1.7%) at the same acceptance (2.02 a
draft); score and uncached tokens do not compare, since the two runs shuffled
their games with different seeds. As on the B200, the items together move
generation within noise: the gain on the RTX is the 287 s it starts sooner,
about a tenth of a one-hour budget, and the public notebooks (531-609 s to
serving) now start later than ours. The prefetch read the 103 GiB at 550-650
MB/s a shard, one to two shards ahead of the loader; its report never landed
on that run, because the lazy load keeps every shard mapped and the
prefetcher took that for a loader still at work. It now ends when the loader
stops mapping new shards.

The nine-hour run, at `ceff4f3b` (the branch and a launcher file), played
run H's 110 games in run H's order (seed 20260930;
`kaggle_pulls/ct_rtx_9h_ceff4f3b_20261001`): its games started at 457 s
again, and it scored a mean of 67.56 against run H's 67.87, winning 65 games
against 60 (median 81.19 against 79.92). The install replayed in 72.1 s, and
the prefetch's report landed ("loader finished", the target's weights loaded
in 124 s).

**Parked and descoped.** S5, the re-sharded checkpoint
(`tools/checkpoints/reshard_checkpoint.py`; the copy stands on `/data`), is
parked: a Kaggle copy does not fit the account's 200 GB of private storage
beside the e307 dataset. The copy leaves out the target's 4.86 GiB MTP block
(`--drop-mtp`), which every cell's own drafter makes dead weight, ~5% of
every load; a B200 replay served on it with the same pool. S3 (capture changes), T6 and S7 (a quantized or
smaller PLE table) were descoped; T4, T11 and S6 wait for card-specific work.

## vLLM v0.31.0 picks (2026-10-05)

Upstream's v0.31.0, read for Flash-Next, offered no throughput win and four
small picks (the fork's `TUFA.md`, "Above #11"): #57105 (the QSA indexer's
prefill workspace at its worst case, against fragmentation), #57104 (a
scheduler deadlock with a KV connector and MTP under KV pressure, the
conditions park mode's asynchronous loads at a full pool make), #57102 and
#58762 (the token-to-request mapping and the GDN metadata built once per
step rather than once per KV cache group, adapted to the fork's per-group
RecoverSSM commit). At capacity they are neutral, so what they bring is the
two fixes:

| cell | before | after | accepted a draft | waits for capacity |
|---|---|---|---|---|
| B200, 140 lanes, the pair started together | 5,152 | 5,164 (+0.2%) | 2.83 both | 27/34/40 and 27/34/44 |

Generated tok/s; the pools are identical to the token, nothing preempted.
The pair in the
table ran under #57104's conditions and neither stalled, but
whether its new path fired is not logged. The readings predate the fork's
completion of #57104's margin across KV cache groups (`TUFA.md`, "Above
#11"), which reserves for an in-flight load the blocks its promotion step
needs in each group, as the allocation counts them, where upstream's count
held a fixed few. Unmeasured. Run directories
`traffic/replay_140lanes_*_20261005_*_v031_*2` (the B200 pair).

## Open items

Serving and measurement:

1. ~~**SFT on the MTP head.**~~ Shipped ("Speculative decoding"): the head
   fine-tuned on ARC2 and ARC3 play (the drafter fork's
   `packages/drafter/README.md`), with a 16,384-id draft vocabulary, sized
   from how hard our generated tokens concentrate (12.8M generated tokens of
   recorded duck play, 25,356 distinct ids):

   | draft vocabulary | share of generated tokens it can propose |
   |---|---|
   | top 256 | 70.1% |
   | top 1,024 | 86.3% |
   | top 4,096 | 96.5% |
   | top 8,192 | 98.9% |
   | top 16,384 | 99.85% |
   | top 32,768 | 100.000% |

   The table is fitted to this prompt format and task family: a prompt
   change moves it, and rebuilding it is a pass over the recording's
   `output_ids`. Open from it: pruning the head by its own routing, which
   needs the drafter's router bound too.
2. ~~**What a drafter needs before it can ship.**~~ Both bounds are gone:
   the block drop is off (upstream #53388) and park mode runs the drafter
   beside the tier ("What ships"; `kv_offload.md`, "Park mode").
3. **Fewer active experts.** Costs accuracy; pays through the union of experts
   touched, so on the RTX (MoE ~50%, memory-bound) and at low concurrency, not
   on the B200 at 100 slots or for prefill.
4. ~~**FP8 weights on attention, linear attention, LM head, shared
   experts.**~~ Built for the attention and linear-attention projections and
   the shared expert, as online FP8 at load (fork `8ed31d3a`, opt-in): +4.0%
   generated on the RTX, declined for the score ("What ships"). The LM head
   and the hyper-connection mixes were left out.
5. **Fewer, larger kernels.** The bf16 group is ~700 calls a step at a few
   microseconds each; fusing projections would attack its per-call cost.
   Unexplored.
6. **The drafter's pool cost.** On the B200 a k=3 drafter with the bf16 head
   and no RecoverSSM costs 26.7% of the pool's tokens, of which ~6 points are
   its weights, activation and graphs and ~21 the MTP layer's own attention
   KV on every token (28.9 -> 37.1 KiB per token). The NVFP4 head and
   RecoverSSM take back its experts' weight and the verify's per-draft
   recurrent states ("What ships"). The drafter's QSA layer takes the
   target's INT4 cache, which costs acceptance (2.40 to 2.28 a draft); its
   own cache in FP8 (`speculative_config.kv_cache_dtype`, which the V2 draft
   loader applies) gave 2.38 back in a diagnostic and is not shipped. Open:
   whether that acceptance is worth the drafter's share of the pool ("INT4
   KV cache").
7. ~~**FP8 KV cache: what is left.**~~ Superseded by INT4 on both cells
   ("INT4 KV cache"), whose token the alignment counts as stored. The
   compressed-key cache is FP8 since 2026-10-12, on score; its own accuracy
   cost is unread.
8. **The tier size.** Sized on the RTX at 80 slots (50 GiB, "Host memory");
   the B200's is not. A change of concurrency moves the depth the size is
   read from, under park mode the loads' depth in `kv_park.log`
   (`kv_offload.md`, "Park mode"). The concurrency is set (100 / 80); when
   it is revisited, note that on Kaggle a higher value also makes the slot
   scheduler (which prioritises the most promising games) less effective --
   more slots, less choosing -- besides what it does to throughput and the
   tier.
9. **The nine-hour run at 0.975, unread.** It played on 2026-09-21
   (`arc3-rtx-9h`: the bf16 state, 80 slots, 0.975, expandable
   segments, the 64 MB bound, the unit off), scored, and ended after eight of
   its nine hours; its log is with a colleague, and the cell went back to
   0.97 without waiting for it. To read off it: what ended it -- an
   allocator death would name 0.975, anything else is a crash of its own --
   ARC3's score on the bf16 state on the RTX, and the RTX cell's `max_concurrent`:
   80 was set on the replay's fifteen minutes (a queue of 0-15 over ~66
   running), and over the hours the run's running count, its capacity queue,
   its deferrals through the compaction waves and the tier's load say
   whether 80 is the short standing queue the cell is sized for, as ARC2's
   twelve-hour run sized the four-L4 cell ("Concurrency and the pool").
10. **Which MoE backend is more faithful, measured.** Teacher-forced next-token
    distributions from each backend, compared by per-token KL against the
    unquantized model over a fixed set of real duck transcripts. The reference
    must be the *pruned* model in bf16 (producing a bf16 e307 comes first if
    none exists), or the prune's own effect folds into the comparison. The
    hard part is its cost: bf16 Flash-Next does not fit one card, so tensor
    parallelism across a B200 node, or a layer-at-a-time offloaded forward run
    once, its top-k logprobs stored and every backend scored against the file
    on its own card. Every candidate runs on a B200, TRT-LLM included.
11. ~~**b12x.**~~ Measured and dropped ("The MoE backend"). A retest is one
    run: the wheelhouse variant carrying the standalone package
    (`arc3-utils-0290-b12x`) named in `ullm.kaggle` with its
    requirement, plus the launcher's `b12x_fp4` arm. Owed from the exercise:
    the fork PR for the routing-id bound. An upstream report against the
    direct backend is drafted and parked (2026-09-21): nobody has
    filed one, and `main` still hands the ids through unbounded.
    `flashinfer_b12x` stays faulty
    upstream; our surroundings are unusual (the PLE table over UVA,
    `mamba_cache_mode="align"`, prefix caching, the fork), so that fault is
    established here, not in the kernel.
12. ~~**Inductor compile-time autotune.**~~ Moot: the Qwen4Exp model runs
    without torch.compile since upstream #55272.
13. **The profile window's stall, and its trigger.** Closing a window stalls
    the engine -- 68 s seen on Kaggle, most likely vLLM building and printing
    its per-op table, which the reader does not need
    (`torch_profiler_dump_cuda_time_total=False` would likely remove it).
    Accepted while profiling is opt-in. A trigger on average context depth
    (KV tokens in use / running) instead of pool fraction would compare
    windows across pool sizes; the pool fraction lined up only for arms
    sharing a pool. Also unmeasured: how much CUPTI tracing inflates the
    many-small-kernel groups.
14. **The PLE table on the card at 100 slots.** Offload is measured at 32
    slots (no cost) and unmeasured at the column's load, where prefill gathers
    16 rows per token over PCIe at 60-110k prompt tokens per second. One
    45-minute run before the pool line settles it.
15. **Numerical continuation.** The prefix-cache row proves a hit was served,
    not that the restored recurrent state is right; the evidence today is
    games scoring. (b) first: one live game with log-probs in its traces,
    every step's exact input and output ids re-scored on a cold engine with
    prompt log-probs, two cold re-scorings of one step as the noise floor --
    one to two days, and it measures the states the column actually serves.
    (a) only if (b) finds a divergence: a forced-state harness (cold, GPU hit,
    tier eviction and restore, around Mamba boundaries and through tool turns
    and compaction), three to five days on Flash-Next itself.
16. **The boot.** ~12.5 minutes to the pool line on the RTX against a public
    reference's ~7.5; whether that is the budget's problem or the column's.
17. ~~**The GDN kernels on the RTX.**~~ Both measured, washes ("The GDN
    kernels on the RTX"); the SM120 prefill gate stays on a fork branch,
    unpinned.
18. ~~**The recurrent state's precision.**~~ The bf16 state ships on every
    cell ("The recurrent state in bf16"); int8 checkpoints are dropped, since
    they would shrink no running request.
19. ~~**Cache hits inside a block: `prefix_match_unit`.**~~ 80 on the B200
    and the four L4s, 64 on the RTX ("Cache hits inside a block"). Open: why the
    unit costs pool on the four L4s, which the layout alone does not
    reproduce (the addendum's open items).

Qualification and structure:

20. **A score against the 27B incumbent** at equal wall-clock on the same set.
    Serving is not qualification. The submission-emulation run is done
    (`run_as_submission=True`, the arena's bundled env-files): TAAF's
    keepalive sent one RESET during the engine boot and none once play began.
    Submission mode turns TAAF's soft deadline off, so a budget does not end
    such a run; stop it by hand.
21. **Engine keys for pruned checkpoints.** `settings.md` keys by `(engine,
    environment)`, and an engine is a checkpoint. Several prunes differ only in
    `num_experts` and path: each its own engine key, or the engine as the base
    with the checkpoint a field. Decide before a second prune is deployed.
22. ~~**MTP in a deployment cell.**~~ Both cells draft, beside the tier
    ("What ships"); the watermark stays U-00 at 0.10 (`kv_offload.md`,
    "Park mode").
23. ~~**The B200 column at 200 slots.**~~ 100 under park mode, read on the
    B200 ("What ships").

Drafting:

24. **k per step, and RecoverSSM near bandwidth.**
    - **Dynamic k** (upstream #32374, in the fork):
      `speculative_config.num_speculative_tokens_per_batch_size` picks k per
      step from the batch size, so a run's tail, with fewer games left,
      drafts deeper without anything triggering it. The GDN and PLE paths
      take shorter windows by construction and the verify stops at each
      request's window, but the MTP speculator varying its passes per step
      has not run on Qwen4Exp. Everything is sized for the maximum k, which
      under RecoverSSM costs the bulk little (a maximum of 7, ~1% of pool),
      and the QSA ring bounds which k a cell can serve ("Speculative
      decoding"). To read: a replay whose lanes thin out, or a Kaggle run's
      tail, under a schedule such as `[(1, 8, 7), (9, 80, 3)]` against a
      fixed k, per box.
    - **The verify and commit near bandwidth.** At k=1 and 41 requests the
      verify reads each request's state once a layer in 1.47 ms (~1.6 TB/s)
      and the commit reads and writes it in 1.08-1.14 ms (~4.3 TB/s); at
      ~7 TB/s the pair would be ~1.0 ms. The verify is latency-bound: two
      dependent reductions over K a position. A lazy commit, folded into the
      next step's verify, is not taken: every reader of the state between
      steps would need a committed one, the likeliest place for silent
      corruption. A CUDA variant of the fused kernel writing the record
      needs a JIT-built extension or a full build, untried.
25. **The competitor-tricks sweep options.** T1 (`autoround`), T7 (`qad`)
    and T8 (`relaxed_accept`) are launcher arms, off by default, beside T9
    and T10; their score sweep comes after the merge ("Competitor
    tricks"). On Kaggle, T1 and T7 need their checkpoints as datasets, which
    the account's private storage (200 GB) does not hold beside e307's; S5's
    re-sharded copy waits on the same. T4, T11 and S6 wait for card-specific
    work, and the four L4s' 0.904 for a boot on that box.

Upstream:

26. **What v0.31.0 leaves to check** ("vLLM v0.31.0 picks"). Not picked,
    each a question about our cells first:
    - #57355, a max-load throughput cliff when `max_num_seqs` is not a
      multiple of 8: three of the four Flash-Next cells are not (100 on
      ARC3's B200, 230 on ARC2's, 60 on its four L4s).
    - #58434, #58784 and #58400, one-token prompt tails under drafting
      (padded into spec-decode rows on hybrids, draft slots never proposed
      rejected, FULL decode graphs for them): whether our runs have such
      tails.
    - #55390, MTP draft KV cache groups annotated positionally on the hybrid
      grouping path: only if ullm's connector reads the annotation. #58368,
      prompt-tail hits for Mamba with MTP, corrects an eagle lowering of the
      tail that the fork's `_cache_partial_tail_block` does not have.
    - #57270, MRV2's profile routing dummy tokens to the MoE experts: a
      safer profile and a slightly smaller pool.
    - #58582, parallel Triton warmup at start-up, for item 16's boot; it sits
      on upstream's warmup rework (#56683 and its series, ~1,000 lines).

The quantized caches:

27. **INT4's last milliseconds** (its cost over FP8 is "INT4 KV cache"'s, not
    chased; 2026-10-11). The rotations (0.47 ms, 64 launches a step):
    rotate the query in the INT4 attention kernel's query load and the output
    with its gate in the split-merge epilogue, and K and V in one launch or
    inside the packing kernel. The attention kernel: retune the split-dot
    kernel's launch per shape, and the nibble unpack.
28. **The FP8 state, if it is revisited** ("The recurrent state in bf16").
    Its accuracy cost apart from its speed, first. Then the commit's +1.6 ms
    (alignment hints on the state rows, whose addresses the commit loads, so
    the compiler cannot widen its stores; round to nearest if an accuracy
    reading allows it, which costs fp32's), prefill's gather and scatter
    (0.26 ms a mixed step), and a copy kernel of +0.16 ms in both FP8 runs,
    unattributed.
29. **Prefill on the RTX.** ~10 ms of a ~68 ms average step ("The fixed
    blocks"); unexamined there: the GDN chunked prefill and the indexer's
    prefill scoring. A run below the pool's line would also give the
    decode-only step's curve over 30-50 requests, which the decode-only step's
    line was fitted without.
30. **The decode tails' landing blocks**, next after the two fixes in "The
    fixed blocks" (2026-10-11); not started. A landing buffer, not a
    copy the tier keeps: the connector copies each tail to the tier as the
    engine keys it, and the keyed block then idles in the free queue. They
    held ~2.3% of the used pool on that section's trace, before the fixes;
    unread on the INT4 cells, whose blocks are larger.

The B200 after W4A16 ("The MoE backend"; 2026-10-15):

31. **W4A16 where the cells run it.** Read on the 205-expert checkpoint at
    220 lanes; the cell serves e307, 307 experts in every layer, and its own
    85-100 slots on 0.9488 and ARC2's 180 on 0.9174 are unbooted. The first
    B200 run of each reads its speed and its graphs' fit. FlashInfer's
    autotune on the kernel reads faster on one layer, unread on the replay.
32. **The prefill-carrying steps are half the B200 replay's time.** At 220
    lanes, steps that carry prefill take 51-57% of the reading's wall time,
    ~200 ms each for ~1,400-1,900 prefill tokens beside the ~1,000 decode
    tokens: about 120 ms over a decode step (the batch-queue trace,
    unprofiled), far off what such a prefill computes. They launch ~2,900
    kernels outside CUDA graphs where a decode step launches ~115 and five
    graphs, and the profiles show the GPU idle in up to ~40% of them when the
    prefill is small; the profiler's stack and shape recording inflates
    exactly that host side, so that is an upper bound. A profile without
    `--profile-shapes` says whether the worker's host side of a mixed step is
    the lever.
33. **The engine is not deterministic across starts.** Two Marlin engines on
    the same prompts disagree on the actual token's log-prob and on a tenth
    of top-1 tokens ("The MoE backend" has the reading). Source unknown; the
    INT4 rotation is seeded, so not that. It is the floor any numerics
    comparison on this engine reads against.
34. **Further on the B200's decode step.** The GDN verify is latency-bound
    even tiled (~1.1 TB/s); a chunked, tensor-core verify is the next step
    and changes the numerics. The router GEMM falls to an unaligned CUTLASS
    kernel, since most per-layer expert counts are not multiples of 8. The
    top-k sampling kernel takes ~3 ms a step.
35. **The traffic replay's two artifacts on the seal's column.** The decode
    gate parks every lane at once, ~4,800 slots against the 64 GiB tier's
    2,786, so about half the lanes recompute their history on their first
    call; and in the window's last minutes a few dozen continued calls
    report no cached tokens though the park log records a full hit at their
    start. Both are common to every arm, but a benchmark that wants a clean
    settled reading needs them gone; until then the pool-bound minutes and
    the decode step's cycle are the measures.
36. **The RTX's and the L4s' own GDN and QSA shapes.** The B200's were read
    on a B200 alone and apply on the 10.x family only; the other cards run
    the base's kernels, untuned at their own shapes.
