# Serving Flash-Next for ARC2: the four L4s, images, and ARC2's readings

*Register: living.*

An addendum to `flash_optimization.md`, which stays the record of
Flash-Next itself; read that first. This file carries only what ARC2 adds:
its submission box, Kaggle's four L4s (`kaggle-4xl4`), which ARC3 never runs
on; the images ARC2 sends and no ARC3 column does; and what the cluster's
B200 (`slurm-b200`) measured on ARC2 tasks. Where a figure below repeats one
from the original, it is only to compare against it.

The harness is arc2-platypus (`duck.presets` holds ARC2's P layers); the
engine cell is ullm's, so it lives here. Where a value lives is each repo's
`settings.md`; this document says why it is that value.

Evidence, not code:
- Kaggle runs: `arc2-demo-duck`, one version per run, pulled to
  `/data/shared/research/arc_2_results/shared/kaggle_pulls/flash_4xl4/v<N>/`.
- B200 runs: `/data/shared/research/arc_2_results/shared/platypus_demo/`, with
  comparisons beside them.
- Launch notebooks, the live-log probes and the analysis scripts (per-call
  cache hits, pass@2, TAAF throughput, opening-prompt lengths, the
  multimodal input probe): `/data/personal/$USER/flash_arc2/`.
- Traffic replays (`traffic_benchmark.md`): the Kaggle versions above, with
  their B200 recordings and smokes under
  `/data/shared/research/arc_2_results/shared/traffic/`.

## What ships

The ARC2 cells, where they differ from or add to the original's table:

| setting | B200 (`slurm-b200`) | four L4s (`kaggle-4xl4`) | layer | why, in one line (sections below) |
|---|---|---|---|---|
| checkpoint | e307 | e307 (the RTX cell's Kaggle dataset) | U-EV | on ARC2 too, e307 beats the unpruned checkpoint and e256; e358 is indistinguishable ("Qualification on the B200") |
| topology | one card | **two pipeline stages of two tensor ranks, split 27/21** | U-EV | the pool of four stages with ~20% more generation; half EP's per-token cost ("The topology"); 27 layers first because the second stage holds the head, the sampling and the drafter ("Drafting on the four L4s") |
| `gpu_memory_utilization` | 0.9174 | **0.935** | U-EV, P-EV | budgets with one graph capture, outside the budget; on the L4s an absolute margin: drafting with both graph kinds on the cell's ladder, 0.92 ran out of memory in serving where 0.91 played; with FULL decode graphs alone the cards served at 0.92 with more free than that, and a ladder to 0.945 put the cell at 0.935 ("Memory in play on the four L4s") |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | engine default (512) | **64** | U-EV | the indexer's prefill scoring buffer grows with the context and the start-up profile reserves none of it ("Memory in play on the four L4s") |
| MoE backend | `marlin` | `marlin` | U-EV | the only NVFP4 MoE backend the L4 (capability 8.9) admits |
| CUDA graph capture | dense | **off; FULL decode graphs alone, on a ladder the cell names** | U-0V, U-EV | captured per card under tensor ranks; the ladder is the request counts the traffic decodes, 12-20 and 28 exact, ~0.4 GiB a card under the engine's ("Drafting on the four L4s"); a step carrying a prefill runs eagerly, since the PIECEWISE graphs held about half the graph memory and bought little ("Memory in play on the four L4s") |
| speculative decoding | MTP head, four drafts | MTP head, four drafts, from its own dataset | U-E0, U-EV | the engine's; on the L4s drafting three alone added 12%, four against three ~2-3% at equal memory ("Drafting on the four L4s") |
| driver work queues | engine default (8) | **`CUDA_DEVICE_MAX_CONNECTIONS=32`** | U-0V | a graph's branch waited behind a pipeline stage's spinning receive at 8: +14.5% ("Drafting on the four L4s") |
| weight loading | engine default | **the engine's prefetch** | U-EV | the kernel's bounded reader drops shards four loaders have yet to read (535 s of load against 252) |
| kernel compile jobs | engine default | `MAX_JOBS=4` | U-0V | a `cicc` per job in a cgroup the model already fills; insurance, since nothing on this cell compiles with nvcc |
| throughput stack (`FLASHNEXT_THROUGHPUT_ENV`) | off | **its portable part** (`FLASHNEXT_THROUGHPUT_PORTABLE_ENV`) | U-EV | the fused commit needs one pipeline stage and the CUDA images capability 12.0, so the rest cannot run here ("The throughput stack on the four L4s") |
| vision tower | kept | kept | -- | ARC2 sends images; ARC3's P-0V drop is not called |
| image processor `max_pixels` | derived: 73,728 at scale 4 | the same | P-00 | the start-up profile reserves one image of the ceiling; the checkpoint's 16.7 MP cost the four L4s their pool |
| recurrent state | bf16 | bf16 | U-E0 | the engine's (the original, "The recurrent state in bf16"); on the four L4s +24% generated, the block 3,136 to 1,600 undrafted, 1,840 at four drafts ("The four-L4 cell on the traffic replay") |
| `max_num_batched_tokens` | 8,192 | **512** | P-EV | a prefill advances in chunks of the budget beside the decodes, and the smaller workspace goes to the pool (arc2-platypus `settings.md`, P-EV; "The step budget") |
| `prefix_match_unit` | 80 | **80** | U-E0 | the engine's; undrafted on the four L4s it cost pool (-3.9%), drafted it pays (+4.3% at 64 at three drafts), and 80 divides the 1,840 block ("Hits inside a block") |
| `max_concurrent` | 200 | 60 | P-EV | B200: ARC3's figure, scaled to the bf16 pool and unread; L4s: 50 read from the twelve-hour float32 run (the pool held 32-49), scaled by the pool the bf16 state gains ("The four-L4 cell on the traffic replay") |
| host KV tier | 96 GiB | 60 GiB | P-EV | the largest measured safe on Kaggle's host, beside the ~48 GB pinned PLE table |
| `max_tokens_per_game` | 150k | none | P-EV | B200: the score curve's knee, traded for experiment turnaround; Kaggle: every cap costs score there, and the ranking handles long games ("What a per-game cap costs") |
| reasoning effort, opening | `xhigh`, no served opening | the same | P-E0 | ARC3's reading; ARC2's +15.4 points for the opening was the 27B's |
| image prompts to the in-process engine | untyped, placeholders collapsed | the same | ullm `vllmx` | a typed token prompt drops its images silently ("Images and the prefix cache") |

The 27B cells step 8,192 tokens and set no cache-hit unit. Two fork defaults act only on the four L4s, where tensor ranks and pipeline stages meet: each rank's own top-k logits gathered instead of the vocabulary, and decodes balanced between the micro-batches ("Drafting on the four L4s").

## Reading an ARC2 run

The original's two habits hold (bucket by running count; scores say "not
worse"). Two more, both learned here:

- **TAAF's counts lead, and a deadline leaves a tail.** Compare generated
  tokens from `benchmark.json` over the play window (first engine line with a
  request running to the last); the summary's tokens/s divides by the job's
  wall clock, boot included; engine lines are the cross-check. A run ended by
  its budget under-counts in TAAF by the calls still in flight at the cancel:
  the host's vLLM backend charges a cancelled call from its last streamed
  chunk (`Backend._usage_from_partial`), the child's `ProxyBackend` does not.
  The harness's reconciliation line prints both (`host backend consumed
  generated=...; games reported generated=...`). The tail grows with running
  count and reply length: host/TAAF was 1.22x at 30 slots (version 30), 1.52x
  at 60 on four stages (version 37), 1.17x at 60 on 2x2 (version 38), 1.00x on
  a finished B200 run. Quote both counts for a deadline-ended comparison.
- **Cache hits, per call, from the spans.** A call that extends the previous
  one should hit `floor((prev_input + prev_output) / B) * B` tokens, `B` the
  attention block (3,136 on one card, on stages and on 2x2; 1,568 over four
  tensor ranks; 1,568 for the 27B). Classify each extending call as exact, one
  block short, **zero**, or not extending; the uncached tokens a zero hit
  leaves is the figure that found the image bug. With `prefix_match_unit` set,
  a hit may end past the formula, so read "hits past the formula" too.
  Scripts: `cache_per_call.py`, `cache_formula.py`.
- **Capture the session log while the run is alive.** A version whose engine
  dies can end in ERROR with no output files at all, and Kaggle serves a
  session's log only while the session runs: versions 39 and 40 left nothing
  but what was captured. `kaggle_pulls/flash_4xl4/tools/kaggle_log_capture.py`
  appends the whole log to a file as it arrives. A crashed engine also dumps
  the step that killed it (`dump_input.py`: the scheduler output, with every
  request's token count and any encoder inputs), which is how the step that
  ran out of memory is read.
- **A four-L4 reading comes with its gaps.** Beside the settled rate, read
  where the step goes: each rank's compute, communication-only and idle time
  in the profile (one window only: vLLM's profiler crashes a stage's worker
  opening a second window under pipeline stages; and a window's trace write
  must fit the engine's 300 s worker timeout,
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, which run E's 400-step window outran,
  voiding the run); stage 1's wait in its
  activations receive against stage 0's gap between steps, which says
  which stage sets the step; the micro-batches' request counts in
  alternation from the batch-queue trace (`VLLM_BATCH_QUEUE_TRACE`); who
  holds the pool from the block trace (`VLLM_KV_BLOCKS_TRACE`); the
  environment probe, which prints the variables each engine process
  actually received; and the machine log (clocks, links, each card's memory
  every 15 s) beside the memory log (torch's serving peaks). A kernel run in
  the editor has no output the API serves: its files are the
  `run_bundle.tar.gz` the teardown packs. Scripts:
  `kaggle_pulls/flash_4xl4/tools/` (`run_readout.py`, `round_trip.py`,
  `stage1_tail.py`, `mb_balance.py`, `graph_usage.py`, `bq_compare.py`).
- **Throughput is read on the traffic replay** (`traffic_benchmark.md`), not
  on an hour of play. ARC2's readings below used the earlier workload format,
  which prefilled every lane's history ahead of a 20-minute window read from
  300 s; the format now opens cold, so the next reading here is a fresh
  baseline (open items).

## The four-L4 cell on the traffic replay

The knob pass of 2026-09-19, every row one Kaggle run on
`workload_100lanes_20260919.b64` (100 lanes, `arc2-eval` x 3, cut from B200
job 2706110): the settled fifteen minutes, the rest the engine lines over
the same stretch. Allocator failures are PyTorch's
`memory allocation failed with OOM` warnings over the whole run, each
survived by a cache flush and retry.

| Kaggle version | configuration | vLLM's KV cache, tokens | gen / uncached tok/s | running | capacity waits (min / med / max) | allocator failures |
|---|---|---|---|---|---|---|
| v2 | the cell as it shipped (float32 state, step 3,236, 50 lanes) | 1,363,148 | 449.4 / 215.5 | 44.0 | 2 / 5 / 10 | 4 |
| v5 | without the 64 MB indexer bound | 1,363,148 | 456.7 / 243.1 | 43.7 | 2 / 6 / 10 | **52** |
| v6 | bf16 state, 70 lanes | 1,591,214 (+16.7%) | **555.2** / 175.4 | 51.4 | 13 / 17 / 22 | 2 |
| v7 | bf16 state, step 3,300 (**the cell from here**) | 1,588,592 | 552.4 / 175.6 | 52.1 | 13 / 16 / 22 | 10 |
| v8 | + Triton GDN decode | 1,588,592 | 560.5 / 178.0 | 52.1 | 14 / 16 / 21 | 10 |
| v9 | + `prefix_match_unit` 64 | 1,588,592 | 531.0 / 149.0 | 46.9 | 17 / 22 / 26 | 12 |

Three runs are prose rather than rows: v3 (expandable segments) died
capturing graphs and v4 (expandable segments over NCCL) read -38%, both
under "Memory in play"; v10 (utilization 0.96) ran out of memory in its
warm-up, there too.

**The bf16 state: +24% generated at 70 lanes** against the float32 cell at
50 (555.2 against 449.4), running ~51 against ~44, uncached prompt -19%,
vLLM's KV cache +17%; the block is 1,600 on the L4s as on the RTX. The
lanes were raised with the state because the float32 pool was already
bound at 50 (a queue in every scored minute), and a knob that frees pool
needs lanes to fill it with (`traffic_benchmark.md`); a 70-lane float32
baseline is unread. The step at 3,300 (v7) read the same as 3,236 (v6) and
is the cell's for the reason under "The step budget". The Triton decode
(v8, +1.5%) and the unit (v9, -3.9%) did not earn a place, and 0.96 failed,
so the cell is v7's configuration at 60 slots: 50 scaled by the KV cache
the state gains (x1.165) and the running count (x1.18).

**The bf16 cell over twelve hours.** `arc2-demo-duck-7cc732`
(2026-09-20/21, the two-take duck, `arc2-eval` x 2, 240 runs): 11 h 45 min,
the engine at 491 tok/s mean over 625 minutes with a queue in 95% of them,
476 tok/s over the job's wall clock, three allocator moments (two at
start-up, one at 00:35, all 64 MiB indexer requests, all survived), no
stall. Its predecessor (`...-5f53ea`) froze for its last 2 h 10 min on the
host tier's engine-log writes (`kv_offload.md`, defaults), which is why the
tier's log is off on Kaggle. Its score is under "Qualification".

## Drafting on the four L4s

Drafting was measured run by run on the 09-29 workload (`workload_long230_20260929_skip1.b64`, 60 lanes,
the decode gate, settle 300 s, window 1,200 s; `traffic_benchmark.md`), each
row one Kaggle kernel (readouts in `kaggle_pulls/<kernel>_run<X>`), each
change measured on its own:

| run | change | gen tok/s | pool | tokens a draft |
|---|---|---|---|---|
| B | undrafted, the 09-29 workload | 535.5 | | |
| C | MTP, three drafts, 24/24 | 599.3 | | 2.2 |
| D | 26/22 | 588.2 | 737 | 2.2 |
| F | `CUDA_DEVICE_MAX_CONNECTIONS=32` | 673.8 | 737 | 2.2 |
| G | `prefix_match_unit` 64 | 702.5 | 737 | 2.21 |
| H | competitor-tricks merged (one capture at 0.889, input embedding on the host) | 707.0 | 736 | 2.27 |
| I | 27/21; the engine's prefetch | 719.9 | 765 | 2.28 |
| J | local top-k logits, decode-balanced micro-batches | 758.2 | 765 | 2.27 |
| K | the trimmed graph ladder, 0.919 | 794.5 | 868 | 2.28 |
| L | four drafts, unit 80, 0.91 | 803.4 | 816 x 1,840 | 2.72 |
| M | 0.92 | out of memory after 23 min (~828 before) | 849 x 1,840 | 2.74 |
| N | the default agent, the cell as it ships | 813.3 | 816 x 1,840 | 2.71 |

Pools are blocks of 1,792 tokens at three drafts. Run E was a diagnostic.

**What drafting under stages took.** Five upstream picks and three fixes of
ours in the fork (its `TUFA.md`, "Drafting under pipeline stages"): the MTP
head's input path, RecoverSSM's deferred commit on the first stage, and the
host tier's shared region sized by the larger stage's block. Undrafted, the
L4 stages already overlapped (both computing 84.7% of the time).

**The step that jumped above 64 tokens: the driver's work queues.** At three
drafts a micro-batch's step took ~35 ms longer above 16 requests (64 tokens),
so a pool admitting more bought nothing. Stage 0's trace: a 68-token step is
one graph replay that stalls at the same node every time, a layer's
shared-expert branch (an auxiliary stream) waiting while the only kernel
running is the pipeline's token receive, three NCCL broadcasts spinning
until stage 1 broadcasts. The branch starts the moment the third broadcast
launches: in-order launch in a hardware queue the two share, at the
driver's default eight queues for ~30 streams; no data dependency exists.
Above 64 tokens cuBLAS picks other GEMMs, a different graph whose branch
lands behind the receive. `CUDA_DEVICE_MAX_CONNECTIONS=32` (the maximum)
separates them: +14.5% (run F against D). Every engine on the box gets it;
no other box runs pipeline stages.

**The round trip.** With two micro-batches in flight, each waiting on its own
sampled tokens and drafts, the step is (stage 0 + stage 1 + hand-offs) / 2,
and moving work between the stages keeps the sum: 26/22 to 27/21 cut stage
1's busy time a step from ~72 to ~64 ms without moving the step at equal
requests. The ~28 ms a cycle of hand-offs were three things (run I, both
stages' traces on one clock): stage 1's tail after its forward (22.8 ms: its
all-gathers 6.3, ~4.5 of them the logits', then sampling and the drafter's
three steps);
stage 0's work before each replay (the GDN commit 2.2 ms); and the
micro-batches' imbalance, 14 against 17-19 requests, each stage waiting on
the other every other step, since a request decodes in the micro-batch of
the step its prefill ended. The fork's local top-k logits and decode
balancing (its `TUFA.md`, "The pipeline's round trip") took those
all-gathers to 1.0 ms and the micro-batches to 16/17: +5.3% (run J). Stage 0 then sets
the step, never waiting, while stage 1 waits ~5 ms for its activations;
its step is MoE 35 ms, dense GEMMs 15 and the pair's all-reduce 6 of ~65, a
pass over the stage's weights that more tokens a step amortize.

**The stage split.** Stage 1 holds the lm head, the sampling and the
drafter, so at 24/24 it was the busier stage and the smaller pool: its block
is a page larger (7 against 6 at three drafts, 7.66 against 6.57 MiB). Moving
layers to stage 0 grows the pool until stage 0's room falls below stage 1's;
with the input embedding on the host (competitor-tricks T3) that point is
27/21. Layer 26 is GDN, so stage 0's block stays as it was; a twenty-eighth
would move attention layer 27, give stage 0 a seventh page and make it the
smaller pool (~687 blocks). 27/21: +3.9% pool, +1.8% generated (run I).

**The graph ladder.** The engine captures two kinds: FULL graphs (the whole
forward) for uniform decode batches, at the ladder's sizes rounded up to the
verify width (four tokens a request at three drafts, five at four), up to
`max_num_seqs` requests; and PIECEWISE graphs (the fork's breakable graphs,
broken at attention, which runs eagerly) for any batch, at the ladder's
sizes, above whose top a batch runs eagerly. The drafter's later positions
take FULL graphs at the sizes up to 60, read as request counts; one list
feeds all three. On run J's traffic the engine's ladder captured 31 FULL, 49
PIECEWISE and 10 draft graphs and used 4, 27 and 2: a micro-batch holds half
the ~32 running requests, so decode never passed 72 tokens, and odd request
counts padded by one (eight-token steps). The cell's ladder is every request
count from 12 to 20 and 28, coarse elsewhere, nothing above 256 tokens:
padding gone, ~0.4 GiB a card less (0.79 and 0.89 GiB against 1.22 and
1.29), and the batches above 256 tokens it leaves eager (~2%) show no step
penalty, their GPU work hiding the launches. The tail down to one game pads
at most a request; growth past 20 a micro-batch pads to 24, 28 or 32.

**Four drafts or three.** Four drafts accept 2.72 tokens a draft against
2.28 (by position 85/72/62/54%), so a request advances 3.72 tokens a verify
against 3.28 (+13.4%), for a step 8-11% longer at equal requests (five
tokens a request against four); the block grows to 1,840. At 0.91 against
run K's 0.919 its pool was 3.5% smaller in tokens and it still read +1.1%;
at equal memory four drafts lead by ~2-3%. Five to eight drafts do not
work; nine was not tried here.

## The throughput stack on the four L4s

The RTX cell serves the kernel stack whole (the original's settings,
`FLASHNEXT_THROUGHPUT_ENV`). This cell serves the part of it that runs here,
`FLASHNEXT_THROUGHPUT_PORTABLE_ENV`: the sampler's top-k fold and the
compact draft sampling it feeds (`XKB_SAMPLER_FOLD=2`,
`XKB_SPARSE_SAMPLER`), the copy-free and sliced GDN paths (`XKB_GDN_NOCOPY`,
`XKB_GDN_MIX=2`), the fused hyper-connection up-mix (`XKB_HC_FUSED`), the
drafter's prefill on last-token rows (`XKB_DRAFT_PRUNE=2`), the dense GDN
projection in [K, N] (`XKB_DENSE_KN`) and the shared experts' stream up to
1,024 tokens. Each takes its tensor rank's shard, and the sampling and the
drafter run on the last stage, where the engine puts them. The rest is out of
reach:

- **The fused commit** (`GDN_FUSED_COMMIT`) folds a request's RecoverSSM
  commit into its next verify: the largest kernel-time saving in the RTX
  replay's profiles (`arc_3_results/shared/traffic/decode_gate_comparison_1793461436`).
  The GDN builder refuses it under pipeline stages.
- **The CUDA images** (MoE, router, hyper-connections, L2 prefetch, the
  fused verify) are built for capability 12.0 alone, and the L4 is 8.9.

The fold serves a batch whose every row samples with a temperature and a
top-k alone, which is play's profile; one row with a ban (the summary's
tool-call token), a penalty, a nucleus or a min-p sends its batch down the
stock path. A replay engages it only with the play profile
(`--sampling preset`).

## Bring-up on the four L4s

A card is 22.03 GiB usable. Versions 25-28 ran the checkpoint over four
tensor ranks.

**The first boot died capturing dense CUDA graphs** (version 25, 0.85). Per
card, before any activation or KV:

| what | per card | where it comes from |
|---|---|---|
| weights after load | **15.91 GiB** | `Model loading took` on every rank |
| a quarter of the non-PLE, non-MTP body | 11.97 | experts 37.94, GDN 3.89, embeddings + LM head 2.37, attention 1.74, shared experts 0.44, other 0.60, vision 0.84 GiB (the PLE's 47.75 is on the host) |
| Marlin's padding of the expert shards | ~1.9 | `moe_intermediate_size` 640, a rank's slice 160, padded to 192 (`round_up(N, 64)`): +20% on 9.49 GiB of experts |
| replicated small tensors, workspaces, allocator reserve | ~2 | hyper-connections and norms per rank |
| left for activation, graphs and KV at 0.85 | **2.8** | |

It died at 91 of 110 graphs in the V2 runner's capture profiling, before the
pool was sized. The engine's own ladder (version 26) captured in 0.84 GiB and
then sized `-0.9 GiB` of KV: the start-up profile had reserved activation for
one image of the checkpoint's 16,384-token ceiling.

**Three changes made it serve** (version 27):
- **The image ceiling is derived**, the largest image the duck renders at its
  scale and gridlines, sides rounded up to 64: 73,728 pixels at scale 4. Past
  the ceiling the processor shrinks an image rather than refusing it, so a
  stated ceiling would blur a sweep at a larger scale without a word. The
  encoder profile became 43 items of 72 tokens where it was one of 16,384. It
  reaches every column, the 27B ones included.
- **A one-block step budget**, 3,136 (later found to starve prefill on
  stages; "The step budget").
- **0.95**: this fork's V2 runner profiles graph capture inside the budget (a
  boot reports 0.95 as "0.90 without CUDA graph memory profiling"), so the
  reserve the 27B cells' 0.85 keeps outside it has no job on this pin.

Pool **325,109 tokens**, 22% of the RTX card's; 10 of 10 easy games won with
images flowing (`mm_hit` in the engine line).

**Expert parallelism** (version 28, `enable_expert_parallel`) gives each card
whole experts, removing Marlin's padding: 14.26 GiB of weights per card, pool
**589,824** at 0.95 and **701,620** at 0.97 (version 30).

**The host is the next wall.** The cgroup sits at 100% of 173 GiB from ten
games on (125 GiB of shm: the 60 GiB tier, the pinned table, the processor
cache; 144 GiB unreclaimable). It has held every run since; a full run adds
~55 MB per game process, and the page cache the weights were prefetched into
is what reclaim takes.

Smaller facts from bring-up:
- **Nothing on this cell compiles with nvcc.** The wheelhouse's generic
  FlashInfer modules are fat binaries carrying sm_89; the one FlashInfer
  kernel this model reaches on the L4 is top-k/top-p sampling. The QSA
  indexer's fused path, GDN and the PLE are the fork's Triton kernels,
  compiled at first use (nine "JIT compilation during inference" warnings,
  latency spikes on the first calls). Marlin is in the vLLM wheel for sm_89.
- **All-reduce runs on NCCL over PCIe**: custom all-reduce refuses more than
  two PCIe cards, FlashInfer's refuses four.
- **ECC cannot be turned off from a kernel**: it runs as root without
  `CAP_SYS_ADMIN` (`nvidia-smi -e 0` and `-r` answer "Insufficient
  Permissions").
- **`kaggle kernels push` ignores the metadata's `machine_shape`**; the
  accelerator goes on the command line, `--accelerator NvidiaL4`, or the
  version lands on a P100.

## Pipeline stages

Four changes in the fork (its `TUFA.md`) let Flash-Next pipeline at all:

- **The PLE layer on the first stage.** The model refused any pipeline
  because later stages receive no input ids. The checkpoint has one PLE
  layer (index 2) holding the whole table, so the refusal is now a PLE layer
  placed after the first stage.
- **Stage-local KV tensors.** A stage allocated tensors for the whole
  model's uniform spec inside a group it holds no layer of, and died in
  `allocate_kv_cache`.
- **The table born on the host.** The offloader moves a parameter only after
  its module exists, so a stage of one card first materialised the whole
  47.69 GiB table on a 22 GiB card (version 31). The table is now created
  in pinned host memory, by upstream's Engram offload (#54371).
- **Stage-local embedding, head and vision tower.** Every stage built all
  three (3.2 GiB) where one stage needs each; now upstream's `PPMissingLayer`
  and meta-device patterns place them. Measured under the emulation below:
  pool 1,033,284 -> **1,288,874 tokens (+24.7%)**.

**Emulating an L4 on a B200.** `VLLM_DEVICE_MEMORY_LIMIT_GIB` (fork) caps the
caching allocator per worker, and `gpu_memory_utilization` is scaled to the
same budget (0.97 of 22.03 GiB is 0.1198 of a B200). Needed because vLLM
never caps allocation at the utilization, it only sizes the pool by it. It
reproduced version 31's table death at the same line, and its pools land
within 3% of Kaggle's. **It caps PyTorch alone**: on the L4 the same 22.03
GiB also holds the process's non-PyTorch memory (~0.5 GiB: context, NCCL,
workspaces) and other processes' contexts (~0.3), so a run that plays under
the emulation can still die on the card, as version 32 did.

**Utilization on a small card is an absolute margin.** Version 32 (four
stages, 0.97) sized a 1,218,969-token pool and died at the first play step,
stage 0 asking 38 MiB with 9 MiB free. Its own profiler line closes it: the
budget of 21.37 GiB was weights 14.68, non-torch 0.52, activation 0.54, KV
5.64. At the OOM the first real prefill ran ~0.35 GiB past the dummy profile
(the GDN chunk kernel) and fragmentation added 0.10, ~0.45 GiB into the 0.66
GiB that 0.97 leaves outside the budget, of which other contexts on the card
held ~0.3. The same 0.97 leaves 2.85 GiB on the RTX card. Played since: 0.83
(version 33, 552,686 tokens), 0.94 (version 34, 1,076,974), 0.95 (version 37,
stages), 0.96 (version 38, 2x2, an hour without an OOM). Nothing above 0.96
was tried on 2x2. Each hundredth is ~110k tokens.

**Layer splits other than twelve a stage lose.** `11,13,12,12` and
`11,13,13,11` both sized 1,177,463 tokens, below the even split: the hybrid
manager groups layers across the model in stripes, and a stage's bytes per
block follow its largest group, so a stage off the GDN-GDN-GDN-attention
period pays a whole extra group slot (~30% per cached token for one extra
layer). Both then crashed at KV init: the host tier's `/dev/shm` region is
sized by each worker from its own tensors, and with unequal stages one worker
waits for a size the other never writes. After stage-local placement the
middle stages of four carry ~2.2 GiB the pool cannot use.

**No numerical parity check, by decision (2026-09-14).** The one path
a pipeline runs that a single card never does is the stage boundary, where
the hyper-connections' pending combine is materialised. A mistake there
perturbs a residual rather than collapsing the output, and the easy set would
not see it. The check, if wanted, is in the open items.

## The topology: tensor ranks with EP, four stages, or 2x2

What each layout costs a card, and what it buys:

- **Four tensor ranks** pad the experts (above) and store the attention share
  of every cached token twice (2 KV heads over 4 ranks). With EP the padding
  goes; the replication stays: ~9.2 KiB a cached token per card.
- **Four stages** store each token once per stage (~4.85 KiB a card) and pay
  a full per-card runtime on every stage. A stage generates for a quarter of
  the running set per micro-batch.
- **2x2** splits the experts at 320 rows (no padding) and the 2 KV heads one
  per rank, holds half the running set per micro-batch, and pays two
  all-reduces a layer over the pair's PCIe. Its per-token cost measured
  higher than expected, ~5.4 KiB a card (untraced; open items).

Under the emulation at 0.94 on the stage-local pin, 2x2 sized 1,190,570
tokens against four stages' 1,288,874 (-7.6%). On Kaggle, 2x2 at 0.96 held
within 1% of four stages at 0.95. The hour-long Kaggle reads (`arc2-eval` x 2,
240 game runs, deadline-ended, so both counts):

| | v30 EP, 30 slots, 0.97 | v36 stages, 40, 0.94 | v37 stages, 60, 0.95 | **v38 2x2, 60, 0.96** |
|---|---|---|---|---|
| image fixes, stage-local placement | neither | neither | both | both |
| KV pool | 701,620 | 1,068,236 | 1,365,333 | **1,376,256** |
| play window | 31.5 min | 31.5 min | 33.5 min | 33.5 min |
| TAAF generated, tok/s over play | 536,845, 284 | 406,843, 215 | 542,839, 270 | **852,471, 424** |
| host generated, tok/s over play | 654,823, 347 | 595,727, 315 | 826,341, 411 | **993,503, 494** |
| TAAF uncached input | 1,113,397 | 1,306,009 | 695,742 | 888,622 |
| cached share (engine lines) | 72.8% | 51.3% | 71.4% | 77.1% |
| mean running | 28.1 | 38.9 | 55.0 | 53.5 |
| capacity-wait lines | 9 of 32 | 2 of 32 | 16 of 34 | 19 of 34 |
| preemptions | 0 | 0 | 0 | 0 |
| won / mean score | 18 / 7.50 | 14 / 6.67 | 10 / 4.17 | 14 / 6.25 |

Engine generation by ten-minute phase, version 37 against 38: 376/435,
432/537, 404/488, 394/414 tok/s. 2x2 generates ~20% more than four stages by
the host count at a pool within 1%, so the gain is per-step speed; it also
reaches capacity sooner (~17 minutes against ~23). Once at capacity the
GPU's own hit rate falls to ~0-5% and the host tier serves ~86-88% of
prompt tokens. Scores over an hour are the "not worse" kind.

**EP closed without a rerun (2026-09-15).** Version 30 predates the
image fixes (whole-hit misses were 51% of its uncached input) and ran 30
slots. The fixes would lift its prefill, but the larger part of 2x2's lead
(+42% host, +49% TAAF) is structural: at ~9.2 against ~5.4 KiB a cached token
its pool is about half, so more slots cannot be held. Closed to save Kaggle
quota.

**Four stages at a one-block budget starved** (versions 33-35); see "The
step budget". Version 35's reading is of that, not of stages.

**A full submission, extrapolated.** The 27B's twelve-hour Kaggle run
(`arc2-eval120-w65536-t22000`, pulled under `kaggle_pulls/`) ran its
first 33.5 minutes at 1.38x its 11.56-hour engine rate (347 against 252
tok/s). Applied to 2x2's 487 engine tok/s over the play window: ~350 tok/s,
~14.7 M tokens over a submission, ~1.4x the 27B's 10.5 M (four stages: ~1.2x).
The factor may not transfer: Flash binds capacity inside its first half hour,
and its later hours are unmeasured.

## Images and the prefix cache

**Flash lost whole cache hits on ARC2, and ARC3 did not.** Engine lines put
the cached share at 95-96% for the 27B (B200 and Kaggle) but 87% for Flash
on the B200, 73% on EP and 51% on stages. Per call:

| run | extending calls: exact | one block short | **zero** | not extending | uncached a zero hit left |
|---|---|---|---|---|---|
| Flash ARC3, B200 (`duck_premerge_slurm_20260913_215356`) | 19,416 | 7 | **0** | 785 | 0% |
| 27B ARC2, B200 (the register A/B's inject arm) | 6,398 | 11 | **0** | 76 | 0% |
| Flash ARC2 e307, B200 (`duck_slurm_20260914_133217`) | 3,647 | 25 | **253** | 150 | 21% |
| Flash ARC2 EP, Kaggle version 30 | 139 | 2 | **46** | 20 | 51% |
| Flash ARC2 stages, Kaggle version 36 | 75 | 2 | **57** | 22 | 60% |

A miss was all or nothing, the prompt's own blocks included, and landed
mostly on the call after a game's first (B200 192 of 200). Same checkpoint,
pin, block, step budget and cache mode as the ARC3 run; the difference was
that ARC2 sends images.

**The cause was ullm: every continued call dropped its images.** vLLM takes
a prompt dict with a `type` as input it has already processed, and a `token`
input has no images: `multi_modal_data` beside it is ignored without a word
(vLLM's own input processor on a B200: `mm_features=None`; untyped, 3 images).
vllmx sent every continued and injected call as `{"type": "token", ...}` with
the images beside it. A game's first call, a fresh text render, hashed its
blocks with the images and every later call without, so the second call
missed whole. The model read bare `<|image_pad|>` tokens on every call but a
fresh render's: task images on the first call only, `show()` images never.
The 27B's inject arm sent even its first call as tokens, so its cache was
consistent and it most likely never saw an image; every Flash image result
before the fix saw them on first calls only. The OpenAI-server backend refuses
token continuation with images and is unaffected. `exact_history` could not
see it: the ids were right, the images were what went missing.

**The fixes, and what separated them** (B200, `arc2-eval` x 2, 150 slots):

| run | zero hits | reading |
|---|---|---|
| images off (`duck_slurm_imagesoff_20260914_190745`, 50 min) | 0 of 1,982 extending | cached share 88.7% |
| the fork's multimodal hasher fix alone (`duck_slurm_mmhashfix_20260914_191547`, 50 min) | 218 against 1,990 exact, 26% of uncached | a real bug, not this cause |
| both fixes (`duck_slurm_mmprompt_20260914_200902`, finished) | **0** | below |

- **ullm (arc3 `3a4d9f39`):** an id prompt with images goes to the
  in-process engine untyped, each image's expanded placeholder run collapsed
  to the one token vLLM expands again. ullm keeps building, counting and
  witnessing the expanded ids, and vLLM's echoed `prompt_token_ids` must equal
  them or the call raises. Untyped with expanded ids double-expands.
- **The fork (`be61aab9`):** `get_request_block_hasher` hashed a prompt tail
  completed by decode from its last image only (`curr_mm_idx=-1`); it now
  finds the images a block overlaps. Kept, being a clear bug: with images
  served on continued calls, a tail holding several images is where it bites.

Both fixes against the e307 run before them
(`flash_mmprompt_vs_e307/compare/`):

| | e307, before | both fixes |
|---|---|---|
| per-run mean score | 72.29 | 74.65 |
| pass@2 per test output | 81.44 (136/167) | 82.04 (137/167) |
| won / submitted wrong / gave up | 182 / 42 / 16 | 185 / 32 / 22, 1 crashed |
| TAAF generated | 10.24 M | 10.41 M |
| TAAF uncached input | 16.23 M | **10.90 M (-33%)** |
| zero-hit calls | 253 | **0** |
| generated tok/s over play | 2,313 (73.8 min) | 2,228 (77.9 min) |

Paired over the 120 tasks: better on 26, worse on 20, +2.4 points, p = 0.38.
Seeing every image has not measurably moved the score; it removed a third
of the prefill, which the L4s pay for and the B200 barely does. The crash is
unexplained (open items).

**On Kaggle a start-up transient remains, accepted (2026-09-15).**
Version 37: 15 of 137 extending calls missed whole (the deadline-cancelled
calls, recorded as empty responses, excluded), 11 of them the call after a
game's first, all within ~5-13 minutes of play (11 of 13 there, 0 of 14
before, 0 of 17 after); version 38: 14 of 196, 14% of uncached tokens. Not
the pipeline: four B200 stages at full memory (`duck_slurm_pp4_zerohit_noemu`)
missed 0 of 1,491, and the L4 emulation at capacity from its third minute
(`duck_slurm_pp4_zerohit_emu`) 4 of 1,455. What differs is the host tier: in
that window Kaggle's wrote 17-46 MB/s and served 0-24% of prompt tokens, the
emulation's 200-275 MB/s and 65-90%; the misses stop when Kaggle's tier
starts serving (~16 minutes). Unconfirmed reading: a finished call's state
blocks leave the GPU's free queue before its follow-up arrives, and the host
tier does not hold them yet. ~20% of uncached tokens in a 45-minute run,
negligible over twelve hours.

## Hits inside a block: `prefix_match_unit`

The block is baked in: the hybrid allocator keeps one page size for every
group, the attention block grows until its page covers a recurrent state
page, and FP8 KV doubles it (the original's "The page and the block"), so a
hit lands on a 3,136-token boundary and a call recomputes half a block on
average. Upstream's partial hybrid hits (#46384, in the fork) decouple the
match from the page: `cache_config.prefix_match_unit` hashes every that many
tokens, and a hit may end inside a block wherever a recurrent state is
stored. States still sit on block boundaries plus the prompt-end tail that
align mode keeps, so it recovers the previous prompt's tail past its last
boundary, not a completion's. Every group's block must divide by the unit or
the engine refuses to start: 3,136 = 64 x 49 and 1,568 = 32 x 49, so 64 for
Flash-Next and 56 for the 27Bs (1,568 = 56 x 28). Hashing cost grows as the
unit shrinks; one U-00 value small enough for every engine's block (16) was
judged too fine, so each served cell states its own beside the step budget.

A/B under the four-L4 emulation (four stages, 0.95, the 3,236 step, 60 slots,
`arc2-eval` x 2, 45 minutes; nothing else changed), both at capacity from
their third minute with the host tier serving ~90% of prompt tokens:

| | unit = block (`duck_slurm_pp4_zerohit_emu`) | unit = 64 (`duck_slurm_pp4_pmu64_emu`) |
|---|---|---|
| TAAF generated / 39.2 min play | 4,555,477 (1,935 tok/s) | 4,934,230 (2,096 tok/s, **+8%**) |
| TAAF uncached input | 4,653,350 | 4,296,545 |
| uncached per generated token | 1.02 | **0.87 (-15%)** |
| engine cached share | 88.8% | 90.6% |
| extending calls hitting past the block formula | 0 of 1,469 | 678 of 1,565 |
| shared prefix recomputed per extending call | mean 1,664, median 1,616 | **mean 1,157, median 741** |
| whole-hit misses | 4 | 10 |

Partial hits work with the victim connector, and not only while the GPU
holds history: most of the 678 were taken with the host tier carrying the
cache, so its finish-time store keys the boundaries a partial lookup asks
for. Whole-hit misses 4 -> 10 is noise at that count or the finer lookup,
unread.

**On with the drafter.** Drafting, the decode tails a unit enables carry a
verify window's boundary state (RecoverSSM writes it), and on the four L4s at
three drafts unit 64 read +4.3% (run G against F, "Drafting on the four
L4s"), where undrafted it had cost pool (below). The cells take the
engine's 80, which divides the 1,840-token block four drafts set.

**Off undrafted.** On the four L4s under the bf16 state (the
replay, v9 against v7) the unit cost 3.9% generated for 15% less uncached
prompt: each running request held ~2.5 more blocks (23.3 against 20.8 at
the same 89.9% fill), so 10% fewer ran, with deferrals up to 5 in the
scored minutes against every other run's 1, and on a decode-bound box the
recompute it saves does not pay for that. The RTX reads the opposite and
takes 64 (the original, "Cache hits inside a block"). The cause is
unexplained (open items). The OOMs that
first took the unit off were not its doing: the cell ran out without it
too, and the 64 MB bound and 0.955 are what ended them ("Memory in play on
the four L4s"). The 27B's 56 had booted on its one-card block of 1,568
(arc2-platypus's duck-engine column run once with the 27B: pool 4,078,250
tokens, the easy set 10/10); its hit rate and its four-L4 block were never
read.

## The step budget

In `align` mode a prefill chunk ends on a block boundary, and this fork's
split sets a request aside (`waiting=N(deferred:N)`) when the step's remaining
budget is under a block. So the budget counts blocks, not tokens:

- **One block, 3,136** (the stages of versions 32-35; under EP's 1,568-token
  block the same budget was two): any step carrying a decode admits no
  prefill. Version 33 ran 0-127 prompt tok/s with deferrals climbing; version
  35 held 40 running for ten minutes, then fell to 12 with deferrals at 27
  and prompt traffic at 0 most minutes.
- **One block and a hundred, 3,236** (versions 36-38): 40 running
  throughout, no deferral past start-up; ~15 MB more activation per card.
- **Two blocks and a hundred, 6,372**: meant to clear new prompts and
  restored history faster, since uncached input ran ~3x generated tokens on
  version 36. It cost 11% of the pool at start-up (1,223,338 tokens against
  1,376,256) and ran out of memory 12 seconds into play (below). Version 38's
  engine lines also show it would have bought little: after start-up its
  waits were for capacity, with at most one request deferred for the budget.

The cell shipped 3,236 on the float32 state and ships **3,300** on the bf16
one: two of its 1,600-token blocks and the decodes, the prefill per step
3,236 bought with one 3,136-token block. 3,236 fits one bf16 block beside
the decodes and read the same on the replay (v6 against v7), but halves what
a cohort of cold prefills -- a compaction wave -- advances per step; 1,700
(one block and the decodes, the start-up profile's reserve back to the
pool) is untried. The B200 cell's 8,192 holds several blocks and the
decodes. Drafting four, the block is 1,840: 3,300 then holds one block, the
decodes' five tokens a request and most of a second block, and the drafted
replays played at it until run V moved to 512 on the fork's scheduler fix.
ARC2 now steps 512 tokens on the four L4s (arc2-platypus `settings.md`,
P-EV): the scheduler admits a prefill's chunks beside the decodes on a spent
step budget, and the smaller workspace goes to the pool.

## Memory in play on the four L4s

The 2x2 cell at 0.96 on Kaggle (60 slots, `arc2-eval` x 2, 2026-09-14 and
15) played an hour at the 3,236 step (version 38) and ran out of memory in
play three times over: at a two-block step (6,372) within seconds, with and
without the unit; at 3,236 with the unit seven minutes in (version 40); and
at 3,236 without it 50 minutes into a twelve-hour run (`driessmit1` version
3). The first two died on both first-stage ranks allocating an Inductor
buffer of the step's tokens x 10,240 in bf16 (62-92 MiB asked, 27-67 free,
358-477 MiB reserved but unusable), which puts PyTorch's ceiling on this
card at **21.09 GiB** (allocated, reserved-unused and free together) of the
budget's 21.15 at 0.96.

**The 12-hour run ran out somewhere else, and names a mechanism.** Version 3
of the shipped cell died ~50 minutes into play, at the capacity line (42
running, 16 waiting), in the sparse-attention indexer's prefill scoring
(`qsa_indexer.py`, `_prefill_logits`): 126 MiB asked, 75 free, the process at
21.76 GiB with PyTorch holding 20.66 allocated and 352 MiB reserved and
unused. The fatal step was ordinary, one 3,136-token prefill block and 23
decodes with no encoder inputs, but its longest context was **41,760
tokens**. The indexer scores a prefill's queries against the whole compressed
context in one float32 buffer of queries x cdiv(context/4, 64) x 64 columns,
chunked only above `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 by default):
3,136 x 10,496 x 4 bytes is exactly the 126 MiB asked, and at a 65k context
it is ~200 MiB. The start-up profile runs a short context and reserves none
of it. The caching allocator reuses a freed block for any request up to its
size, so as contexts set new highs through a run, each new largest buffer
must come from the card, not the cache: the 352 MiB it held was in smaller
pieces, the card had 75 free. So the failure waits for contexts to grow,
which is why the hour-long runs, stopped after ~34 minutes of play, missed
it.

**The cell now bounds it** (2026-09-15): `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
64 in the four-L4 Flash cell's engine environment, so each piece stays near
the step's other buffers for at most four chunks a call at this step budget,
and `gpu_memory_utilization` 0.955, ~0.11 GiB a card more margin for ~55k
tokens of pool. Unmeasured on Kaggle. A lower bound buys little once it is
below the step's other temporaries (the Inductor buffers of tokens x 10,240
bf16, ~62 MiB here) and costs a pair of kernel launches per extra chunk in
every sparse-attention layer.

- **Two blocks per step do not fit at 0.96.** A step carrying two prefills
  and the vision encoder on the first stage outgrows what the start-up
  profile keeps, in the first seconds of play.
- **The unit was suspected and cleared.** Version 40's death with it on
  matched version 3's without it in every particular but the buffer, and
  the fragmentation alone (358-477 MiB reserved and unusable) was larger
  than either failed allocation; on the bounded cell the unit played a
  full replay window (v9) with no allocator death. What it costs is pool,
  not activation ("Hits inside a block").
- **Above 0.955 there is no margin as the allocator stands.** 0.96 on the
  bf16 cell (v10) ran out inside the warm-up's long prefills: 34 allocator
  failures on the first stage in ~1.5 minutes (64 MiB indexer chunks), then
  64 MiB asked with 39 MiB free and 326 MiB reserved but unusable, for 1.65%
  of KV cache (1,614,807 tokens).
- **Expandable segments would remove exactly that fragmentation and cannot
  run here.** vLLM's custom all-reduce shares its graph buffers between the
  tensor ranks over CUDA IPC, which expandable-segment memory does not
  support: the L4's workers died capturing graphs
  (`custom_all_reduce.cuh:164 'invalid argument'`, v3), and a B200 smoke
  had missed it because a B200 all-reduces through FlashInfer first. NCCL
  in custom all-reduce's place (v4) cost 38% -- 277.2 against 449.4
  generated at the same running count -- over the pairs' PCIe. What would
  revive it is fork work keeping custom all-reduce's shared buffers out of
  expandable segments; the allocator's other knobs without virtual memory
  (`max_split_size_mb`, `garbage_collection_threshold`) are the other route
  to the margin. On the RTX, one rank, it ships (the original).
- **The 64 MB bound costs nothing measurable.** Without it (v5) 456.7
  against 449.4 generated, inside the spread, and the allocator failed 52
  times in 20 minutes against the baseline's 2-4, the buffer growing with
  the context (62-180 MiB asked); a failed retry is what killed version 3.
  It stays.

**Why the B200 emulation missed all of it.** It was not an L4 in two ways:
- Its allocator cap was the card's 22.03 GiB, where PyTorch's ceiling on the
  L4 is 21.09: the cap limits PyTorch alone, and on the L4 ~0.74 GiB of the
  process's own memory and ~0.19 GiB of other contexts sit outside it.
- The B200 (Blackwell, SM 10.0, CUDA 13) runs FlashInfer's GDN prefill
  kernel. The L4 (SM 8.9) falls back to Triton/FLA (`Using Triton/FLA GDN
  prefill kernel` in its log), the kernel whose working set outgrew the
  start-up profile in version 32.

A faithful emulation caps each worker at 21.09 GiB, sets
`additional_config={"gdn_prefill_backend": "triton"}`, and matches the pool.

**The cap alone did not reproduce it** (2026-09-15, job 2623078,
`platypus_demo/duck_slurm_emu2x2_unit64_cap2109_20260915_161757`). Version
40's cell on four B200s: 2x2, the 3,236 step, the unit at 64, 60 slots, the
utilization scaled to 0.96 of an L4, each worker capped at 21.09 GiB (the log
confirms the allocator fraction). It played its whole 35-minute budget
without an OOM: the pool at capacity (~89%), the host tier serving 91-93% of
hits, 4.04 M tokens generated at ~2,400 tok/s (about four times Kaggle's
hour), 72 of 240 runs won. It was unfaithful in two ways, and either could
account for the missing failure:
- **The pool was 8% smaller**: 1,267,029 tokens against Kaggle's 1,376,256.
  The B200's start-up profile left less of the scaled budget to KV, so each
  card had ~0.6 GiB more room under the cap (109k tokens at ~5.4 KiB), about
  the overrun that killed version 40. The utilization has to be raised until
  the pools match, roughly +0.005 of a B200.
- **It ran FlashInfer's GDN prefill kernel** (above).

What it does say: with ~0.6 GiB of slack, the unit at 3,236 does not run the
card out, so whatever the unit costs in play is smaller than that or tied to
the Triton kernel.

**The drafted cell's margin** (runs K-N, one graph capture, the cell's
ladder). 0.92 ran out of memory after 23 minutes of serving, on both of
stage 1's ranks: 58 MiB asked for an eager prefill's hyper-connection
linear, 46 MiB free, torch at 20.43 GiB allocated with 373 MiB reserved but
in pieces too small (280 MiB of it in the graphs' private pools), the
process at 21.86 GiB, 1.07 of it outside torch (0.74 undrafted, above). The
fork's memory log, on through serving, showed the peak climbing as rarer
batch shapes arrived: stage 1's peak allocated 20,611 MiB a minute in,
20,800 at three, 20,837 at seven, then flat for sixteen minutes until the
failing step asked past ~20,978. At 0.91 (run N) stage 1's serving peak was
20,548 MiB under 21,290 reserved, the card 36-56 MiB free; stage 0 peaked
at 20,542 under 20,986, ~200 MiB free. So 0.91 is 0.22 GiB of KV inside the edge, which a 25-minute
window cannot prove for twelve hours (item 25). The memory log is off by
default; a run that reads the peaks turns it on in its engine environment
(`VLLM_MEMORY_LOG_S=15`, `VLLM_MEMORY_LOG_SERVING=1`).

**FULL decode graphs alone, at 0.935** (2026-10-28; runs km1b to km7,
the 230-game replay at ARC2's 512-token steps with the rings' own pool). The
PIECEWISE graphs held about half of the cards' graph memory and bought little
here: run X's 512-token one moved generation +0.3%, the mixed steps' launches
sitting off the critical path. Captured as FULL decode graphs alone
(`FULL_DECODE_ONLY`), a step carrying a prefill runs eagerly, and at 0.92
(km2) the cards served with more memory free than km1b's 0.91 with both
graph kinds, the pool holding more requests at the same step cost. A ladder
above it on the same replay, 0.925 to 0.945 (km3 to km7, arc2-platypus
`kaggle-models-ladder`; km3 at 0.925 held, its second stage at km1b's
free memory in serving), put the cell at 0.935.

## What a running request holds

The twelve-hour pass@2 run (`arc2-demo-duck` version 5,
scriptVersionId 350395308: the shipped 2x2 cell at 50 slots, FP8 KV, `align`
mode, the 60 GiB tier), pulled to `kaggle_pulls/flash_4xl4/v5_pass2_12h/`.
Its connector trace opens on the pool: 624 blocks of 3,136 tokens, shared by
six cache groups -- four holding a single recurrent state each, the attention
group, and one of 4-token chunks that are never hashed, unidentified. Over its
690 engine lines it ran 40.3 requests with 8.8 waiting for capacity, at 88.4%
of the pool: the pool, not the slot count, bounded it.

- **Each running request holds ~4 blocks whatever its context.** Regressing
  used blocks (kv% x 624) per engine minute on the live requests the trace
  reconstructs: used ~ 0.98 x (context blocks, rounded up) + 4.1 x running.
  The 3.9 +/- 0.4 blocks beyond the context are the four single-state groups'
  one block each, and rounding the context up adds half a block on average:
  ~14k tokens of pool per running request, ~29% of the pool at 40 running.
  The engine line's own proxy agrees, kv x pool / running ~ 0.96 x measured
  context + 15.0k. The regression's R^2 is only 0.51, because the pool sat at
  88-90% all run.
- **The engine's "GPU KV cache size" is not the pool.** vLLM reports
  `max_concurrency x max_model_len` (`get_kv_cache_capacity`): 624 blocks over
  the 30 that one request of the full 65,536-token window takes across the six
  groups (21 of them attention), times 65,536 -- 1,363,148 tokens, "20.80x" on
  the same line. The attention pool is 624 x 3,136 = 1,956,864 tokens. The
  pools quoted in this addendum are read off that line (version 38's 1,376,256
  is exactly 21 windows), so they compare between runs of one topology, not
  as token capacity.
- **A decode step has a fixed cost, so a larger running set is cheaper per
  token.** The engine's seconds per second over the same run, fitted as
  `a0*P + w*G/running + (b0 + b1*c)*G + gamma*L` (P uncached prompt and G
  generated tok/s, c the mean context, L tier loads in MB/s): w ~31 ms a step
  (95%: 19-40), b0 ~1.0 ms per generated token plus b1 ~0.09 ms per 10k of
  context, and a0 ~0.26 ms per uncached prompt token. At a 30k context a
  generated token costs 2.05 ms at 40 running, 1.90 at 50 and 1.80 at 60.
  Pool freed per request pays twice: more requests fit, and each of their
  tokens is cheaper.

The fit is `kaggle_scheduler/engine_cost_model.ipynb` in arc2-experiments
(branch `duck-sweep6`), over the same trace reconstruction; the block
regression is not in it. The bf16 state shrinks the fixed blocks (the
original, "The recurrent state in bf16"): on the replay at the 1,600-token
block a running request holds 17.0 blocks, context included, and 19.0 with
the unit on.

## What a per-game cap costs

The twelve-hour run (`arc2-demo-duck` version 4, the shipped cell,
11 h 45 m, 240 games of `arc2-eval` x 2) is the first full reading of it. It
finished **236 of 240** games -- 194 won, 42 answered wrong, 4 cancelled at
the stop -- for a mean score of **76.67** and 13.90 M generated tokens,
57,936 a game. The games reported 13,904,704 tokens against the host's
13,915,311, 1.0008x: a run that finishes leaves no in-flight tail, so the two
counts agree and either can be read.

Per game, from every submission rescored at the tokens it was made at
(`benchmark.json` keeps each answer's grids and its own token count; the
scripts are `kaggle_pulls/flash_4xl4/tools/peak_score_tokens.py` and
`cap_curve.py`, the run's own `benchmark.json` beside them in
`kaggle_pulls/flash_4xl4/v4_12h/`):

| cap | mean score | against no cap | tokens | freed | games cut |
|---|---|---|---|---|---|
| 40k | 56.81 | -19.86 | 6.53 M | 53% | 88 |
| 80k | 68.33 | -8.33 | 9.03 M | 35% | 43 |
| 130k | 72.92 | -3.75 | 10.71 M | 23% | 27 |
| 160k | 74.38 | -2.29 | 11.40 M | 18% | 21 |
| 200k | 75.62 | -1.04 | 12.13 M | 13% | 16 |
| none | 76.67 | -- | 13.90 M | -- | -- |

- **Every cap costs score here**, because this run did not use its clock: the
  queue drained after ~6.5 hours and the engine idled through the tail, so
  the tokens a cap frees had nothing to buy. A cap earns its keep only where
  the clock binds -- a fuller submission, or pass@2, which asks the same
  twelve hours to play every task twice (arc2-platypus's `roadmap.md`).
- **The tail is nearly flat.** A game's median is 24,722 tokens and a scoring
  game reaches its best score at a median of 23,730, so past ~150k each point
  costs several times a whole median game. That is the trade the `slurm-b200`
  cell takes at 150k: the cluster is where experiments wait on the tail.
- **34% of all tokens were spent after the game's best score was reached**,
  and **42 games (28% of tokens) never scored at all** -- 220k to 300k tokens
  each for nothing. That is where a cap takes its tokens most cheaply.
- **Long multi-test games are the clearest case.** `8b7bacbf` pass 1 held 50
  (one of two test inputs) at 92.5k tokens and spent 201.7k more; `b6f77b65`
  pass 1, 97.4k then 189.8k more. But the two longest games of all reached
  their score only at the very end (`20a9e565` at 399.8k of 399.8k), so a cap
  cuts both kinds.
- **One game's score fell and recovered**: `8b7bacbf` pass 1 answered 50 at
  92.5k, replaced it with a wrong answer twice (0 at 204.9k and at 272.5k),
  and came back to 50 at 294.2k. It is why a 250k cap scores *below* a 200k
  one. The harness keeps what the model last wrote, not the best it ever
  wrote, which is a duck-level question rather than a serving one.

## Qualification on the B200: ARC2 scores

`arc2-eval` x 2 passes (240 game runs), 130k tokens a game, 150 slots, the
column as it ships otherwise. Pass@2 follows Kaggle's rule: per test output,
best of the two passes, a run that gave up counting every output wrong
(`pass_at_2.py`).

**Flash-Next against the 27B** (`flash_vs_27b_eval120/compare/`):

| | Flash-Next e307 | 27B, register A/B inject arm (80 slots) |
|---|---|---|
| per-run mean score | 72.29 | 51.67 |
| pass@2 per test output | 81.44 (136/167) | 63.47 (106/167) |
| pass@2 per task | 78.33 (94/120) | 65.00 (78/120) |
| gave up | 16 | 82 |
| generated tokens, job | 10.24 M in 1 h 29 min | 9.73 M in 2 h 10 min |

Two caveats stand. The 27B arm most likely never saw an image (its first call
went as tokens too, "Images and the prefix cache"), so the gap is not
like-for-like. And the public evaluation set may be in this checkpoint's
training data; only the hidden set answers that.

**The checkpoint ladder** (`flash_checkpoints_eval120/compare/`; the
prunes on the current pins, e307 on the flash-kaggle line):

| checkpoint | weights on one B200 | pool at 0.97 | per-run mean | pass@2 per output | tokens a run | TAAF tok/s over play | paired vs e307 over 120 tasks |
|---|---|---|---|---|---|---|---|
| unpruned (512), 120 slots | 74.33 GiB | 5,065,932 | 61.94 | 73.65 | 31.4k | 2,111 | e307 better on 38, worse on 10; p = 0.0004 |
| e358 (70%) | 55.38 | 6,066,449 | 73.68 | 83.83 | 44.4k | 2,223 | e358 better on 24, worse on 20; p = 0.60 |
| **e307 (60%)** | 48.98 | 6,411,605 | 72.29 | 81.44 | 42.7k | 2,313 | -- |
| e256 (50%) | 42.70 | 6,754,577 | 65.76 | 77.84 | 35.9k | 2,316 | e307 better on 30, worse on 15; p = 0.03 |

~0.124 GiB of weights per expert column. The order is ARC3's (e307 > e256 >
unpruned), with e256 above the unpruned checkpoint here. The unpruned
checkpoint plays shorter games and submits wrong more often.

**Why e256 is not faster on the B200.** At matched running (100-139) e256
generates 3,582 tok/s against e307's 3,346 (+7%), and at 140-150 they tie: a
step at 150 touches nearly every expert, e256 reads ~83% of e307's expert
bytes, and MoE is ~30% of a B200 step. Its shorter games also start more
often, feeding 17% more prompt per generated token. On the L4s, where MoE is
half the step and memory-bound, a prune pays in decode as well as pool.
Expected from the weights alone, before stage-local placement: e256 on four
stages at 0.94 ~1.42 M tokens against e307's 1.08 M.

**e307 ships (2026-09-15).** Every prune here was cut by an
expert-routing histogram taken on ARC3 traffic; which experts ARC2's puzzles
route to is unmeasured, and that is where an ARC2 prune would start (open
items).

**The bf16 cell's first submission-length run, like for like.**
`arc2-demo-duck-7cc732` (the two-take duck, `arc2-eval` x 2, the
bf16 state at 3,300 and 60 slots) against `arc2-demo-duck`
version 5 (two-take, the float32 cell at 3,236 and 50 slots): mean score
**81.04 against 78.40**, 177 won against 141, 46 cancelled at the clock
against 89. One run each, so the score reads as not worse; the cancellations
are where the throughput shows.

## Open items

Serving on the four L4s:

1. ~~**Memory in play on the four L4s.**~~ The instrument is now the traffic
   replay on the cell itself (`traffic_benchmark.md`), which read every
   candidate above on Kaggle; the B200 emulation is retired. What the replay
   does not read is memory over a submission's length, which the twelve-hour
   runs of the bf16 cell have now covered once ("The four-L4 cell on the
   traffic replay").
2. ~~**`max_concurrent` on 2x2, from the 12-hour run.**~~ Read 
   (2026-09-16): **50**. Over minutes 0.5-6.5 h of the twelve-hour run the pool
   was what bound throughout -- it held 32-49 requests (p10 34, median 38, p90
   45) at its ~89% line while 11-24 games queued for room, and running plus
   waiting sat at the slot count, so every slot had a request in the engine
   and games idle between calls were rare. 60 was never reached after the
   first half hour, so the extra slots only lengthened the queue. 50 leaves a
   standing queue at every minute of that window (the pool's maximum was 49)
   and hands the slot ranking a smaller resident set to choose from, which is
   what a slot buys here: slots gate the engine only, every game's process
   runs from the start, so this is not a host-memory figure. 45 would have
   starved the pool in 12% of those minutes. What remains open is the other
   direction: the pool's own capacity per token.
3. ~~**The B200's per-game cap, from the 12-hour run.**~~ Read 
   (2026-09-16): **150k on the cluster, still none on Kaggle** ("What a
   per-game cap costs"). The cluster's is a trade, not a score reading -- past
   the knee a point costs several median games, and an experiment waits on
   that tail. Kaggle keeps every point it can pay for, and a cap returns there
   with pass@2.
4. **2x2's per-token cost.** ~5.4 KiB a cached token per card against four
   stages' 4.85: some part of the cache does not halve over two tensor ranks.
   Candidates: the QSA indexer's caches, or a state the grouping replicates.
5. ~~**The layer split.**~~ Read with the drafter: 27/21, and the host
   tier's shared region sized by the larger stage's block ("Drafting on the
   four L4s"). Splits on the four-layer period (`8,16,16,8`,
   `8,12,16,12`; `20,28` or `28,20` on 2x2), read against the stage extras
   per card; a stage's bytes per block following its own layers rather than
   the model's striped groups; the host tier's shared region under unequal
   stages, which refuses any uneven split today; compute balance beside
   memory.
6. ~~**Whether stages overlap their micro-batches.**~~ They do on the L4s;
   drafted, the step is their round trip ("Drafting on the four L4s"). vLLM's pipeline scheduler
   should keep one batch per stage in flight; run one at a time, every stage
   idles most of the step. The engine log or a profile would say.
7. **The start-up whole-hit misses on Kaggle.** Accepted; confirming the
   reading takes the gap between a call and its follow-up, and block removal
   events.
8. ~~**Utilization above 0.955 on 2x2.**~~ 0.96 ran out in the warm-up
   (v10, "Memory in play"); no margin until the allocator changes (item 9).
9. **Expandable segments, blocked by custom all-reduce.** Measured: the arm
   dies capturing graphs, and NCCL in custom all-reduce's place costs 38%
   ("Memory in play"). Every OOM in play had more memory reserved and
   unusable (326-477 MiB) than it asked for, which is the fragmentation
   expandable segments removes, and each hundredth of utilization it would
   let the cell take back is ~110k tokens of pool. What would revive it is
   fork work keeping custom all-reduce's IPC buffers out of expandable
   segments; the allocator's knobs without virtual memory are the other
   route.
10. ~~**The indexer's scoring bound, measured.**~~ Free (v5, "Memory in
    play"); it stays at 64.

Correctness and measurement:

11. **An engine-side check that images are consumed.** The echo check proves
   the ids; the image bug was images silently ignored beside right ids. A
   check that the engine's multimodal features match the images sent would
   catch the next such drop.
12. **The child's in-flight tail.** `ProxyBackend` does not implement
   `_usage_from_partial`, so TAAF under-counts a deadline-ended run.
13. **A `SubprocessDiedError` after a win** (game `b5ca7ac4`, pass 0, in the
    both-fixes run): scored 100, then exited 0 without finishing the
    protocol ("Task was destroyed but it is pending" from
    `ProxyBackend._read_loop`), with no deadline cancel.
14. **Two upstream vLLM reports:** the multimodal block hasher for tails
    completed by decode (the fork's fix, with its test), and typed prompt
    inputs accepting `multi_modal_data` without a word.
15. **Numerical parity at the stage boundary**, open by decision: prompt
    log-probs over real duck prompts, one card against 2x2, against the floor
    of two single-card runs; a boundary bug drifts from the first token.

Qualification:

16. **An ARC2 prune.** An expert-routing histogram on ARC2 traces, then a
    prune at the ratios above. e358 and e307 are indistinguishable on the
    B200; e256 costs ~6.5 points but on the L4s gives ~30% more pool and a
    faster decode.
17. **The 27B, like for like.** Its B200 arm most likely saw no images, and
    its four-L4 cells carry a 0.85 set for a reserve this pin's runner keeps
    inside the budget. Standing note (2026-09-14): if Flash-Next does
    not handily beat the 27B on the submission box, the 27B gets the same
    optimization pass.

Carrying it to ARC3:

18. ~~**What reaches ARC3, checked on ARC3.**~~ Carried and read on the
    RTX's own replay (the original, "The RTX column on the traffic replay"):
    the bf16 state, the 64 MB bound, expandable segments (viable on one
    rank) and the unit, which the RTX takes at 64. The full-duration ARC3
    check is the nine-hour run, unread (the original, item 9). Still open
    for the RTX from this list: a harder prune (MoE is half its decode step
    and memory-bound, so e256 would pay in speed there), a larger step
    budget, and whether its host tier fills as slowly as the L4s' at start.
19. ~~**Why the unit costs pool on the four L4s.**~~ Moot: drafted, the unit
    pays ("Hits inside a block"). ~2.5 more blocks a running
    request under the bf16 state (v9), where the RTX holds none more (the
    original, "Cache hits inside a block"). What remains is ARC2's traffic
    (prompt-length distribution, images) or the L4 itself. Also
    a single pair of readings so far.
20. ~~**A fresh replay baseline on the current workload format.**~~ Runs B-N
    ("Drafting on the four L4s"). The runs
    above prefilled every lane's history ahead of a 20-minute window read
    from 300 s; the format now opens cold with the settle as warm-up, so the
    next reading here starts a new board.
21. **The 70-lane start stall**, a benchmark artifact of the old format
    (v6 and v7, read from the connector trace): the warm-up left 54
    histories as freed cached blocks in lane order, the window's first calls
    were admitted in the same order, and each admission's allocations evicted
    the next lanes' histories faster than the tier's 128-block band copied
    them -- 11 lanes' histories lost, the last first token at 290 s against
    the baseline's 105. Not the step budget (3,300 repeated it to the
    minute), and no block was lost after the first minute of any run. What
    could carry into production is a cohort of cold prefills (a compaction
    wave) at one block a step, which 3,300 halves against 3,236.
22. **The B200 column at 200 slots**, unread (the original, item 23).

Drafting:

23. **Drafts under the request's own top-k/top-p, at a fraction of the
    cost.** The fork's opt-in `speculative_config.draft_top_k_top_p` (fork
    `6485a0d9`, off) samples drafts from the head's distribution cut the way
    the target's is, which is what ARC2's shipped sampling (top_p 0.8899,
    top_k 194) makes worth having. On the B200 at k=3 it read +3% acceptance
    (2.22 -> 2.29 a draft) but -5.7% generated (3,287 -> 3,099 tok/s). Three
    top-k/top-p passes over the full 248k vocabulary each step cost +0.7 ms in
    `_topk_topp_kernel`, about +2 ms of wall time. The head produces only
    16,384 ids and scatters them into the target vocabulary (`d2t`), the rest
    at -inf, so the same cut applied to its compact logits before the scatter
    is ~1/15 the work. That needs a compact-logits hook in the MTP model.
    ARC3's cells are touched too: their top_p is 1.0, but top_k 194 still
    cuts. Waits on top_p's sweep (2026-10-03): worked on only if
    top_p stays.

    **How to test the fix:**
    - **Equivalence, on the CPU.** For random compact logits, a `d2t`
      mapping and top-k/top-p pairs (ARC2's 194/0.8899, top_k alone, top_p
      alone, neither), the compact cut followed by the scatter must keep the
      same ids with the same renormalized probabilities as the full-vocabulary
      cut. Under a fixed seed it must draw the same drafts.
    - **Speed and acceptance, on the traffic replay.** A B200 before/after
      pair at the column's shipped sampling and k, on the same workload
      (`workload_long230_20260929.b64`), at a lane count that keeps the pool
      bound (`traffic_benchmark.md`, "When a number counts"). Arms: the mask
      off, the full-vocabulary mask, and the compact cut. Read acceptance per
      draft (the engine lines' `accept=`), generated tok/s, and
      `_topk_topp_kernel`'s time in the replay's profile. The fix passes if
      the compact arm keeps the full mask's acceptance, its kernel time falls
      to a small fraction of 0.7 ms, and it generates more than the mask-off
      arm.

24. **FP8 on the four L4s** (`fp8_dense`, `indexer_fp8`): the last
    experiment on the drafted cell, not for the defaults 
    (2026-10-03). Stage 0 sets the step and its dense GEMMs are 15 ms of ~65.
25. **0.935 over a submission's length.** The memory log's serving peaks from
    a twelve-hour run with it on ("Memory in play on the four L4s"): whether they keep
    climbing past the 25-minute window's, and how close stage 1 comes.
26. **What each process holds outside torch** on the drafted cell: 1.07 GiB
    against the undrafted 0.74, untraced (communicators the pipeline creates
    at first use, workspaces).
27. **The drafter's memory on stage 1**: its own copy of the input
    embedding (0.59 GiB a rank; the target's is on the host) and its MoE
    unpruned at 512 experts (0.66 GiB a rank against ~0.40 at 307). Stage 0
    sets the pool now, so stage 1's savings reach the pool only through the
    split.

The quantized caches:

29. **INT4 on the four L4s**, a judgement call still to make 
    (2026-10-14). ARC3's Flash-Next cells ship the INT4 main cache, the FP8
    indexer cache and the rings' own pool (the original, "INT4 KV cache" and
    "The fixed blocks"); this cell already ships `indexer_fp8`, and ullm's
    engine block, which it follows, keeps FP8. Read here once, on older
    code: traffic run S (INT4) 922.7 generated tok/s against run O's 861.2,
    running 37 to 52 on a pool 1.75x, draft acceptance 2.83 to 2.53 (the
    drafter's own INT4 cache, as on the B200). Owed before adopting: a
    reading on the current pin, the block for the cell's stages, and the
    launcher's arms and block table, which arc2-platypus
    `flash-int4-l4` carries. The admission change already acts here
    wherever the tier is on (the original, "Admission on a spent budget").
