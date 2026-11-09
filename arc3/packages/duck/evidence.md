# Duck evidence

*Register: living.*

Where the duck's measurements are kept — the reading that justifies a
current default, and what it does and does not settle. Not the detail:
that stays recoverable from the job dir, where `git_status.txt` names the
commit and `games/` holds the spans.

Runs live under `/data/shared/research/arc_3_results/shared/`.

**A default with no entry here is a judgement, not a measurement.** Most
of them are. An entry earns its place by being the thing that would have
to be re-measured before the default moved, so read the absence of a
reading as an open question rather than as a settled small effect.

## The model, and the effort level

`qwen38_sweep_20260815_074942` — the official 25 across reasoning-effort
arms. `xhigh` 6.25, `medium` 5.10, and the previous model's default 1.64
and flat past ~40k tokens per game; `compare_duck_official25/` in the run
dir renders the curves.

This is what put Qwen3.8-27B-FP8 behind both presets with `xhigh` set on
the preset rather than the class. The effort level is a level only that
family's template has, which is why the class default stays
provider-neutral — a hand-built duck reaches whatever backend it is paired
with.

What it does not settle: the arms moved the model and the effort level
together against the older model's baseline, so the effort curve is
established *for this model*, not as a general claim about effort.

## Hysteretic context compaction

`20260805`, official 25 x 10 passes. Two calibrated configurations, both
score-neutral against the flat 32768-window predecessor (p > 0.28) while
taking **+42-79% more actions per episode** at 90%+ prefix-cache hit
rates, against 50-74% for the flat window: window 65536 with target 32768,
and window 32768 with target 16384.

The mechanism is the point rather than the score. Edge-pinned trimming
re-crosses the budget every round and invalidates the KV prefix each time;
the sawtooth pays for one compaction and keeps the prefix stable between
them. The reading says the duck gets more play out of the same budget, not
that it plays better per action.

The ratio between target and prompt budget has been carried forward
through later window changes rather than re-measured at each one.

This and the eviction chunk below are eviction's readings: the duck has
summarized by default since sweeps 12 and 13 ("Sweeps 12 and 13", below),
and `compact_to_tokens` and `evict_turns` steer `prune` alone.

## The eviction chunk, and why the budget was oversized

Measured on `ft09`, 60 passes per arm. At a 21500-token budget the duck
scored 50 level-wins / mean 5.14; at 25000, **95 / 10.35**.

The diagnosis matters more than the numbers, because it points at a knob
that does not exist yet. 21500 was over-pruning *generally* rather than
losing rare long contexts — level-wins showed no enrichment on >21.5k
turns (20.5% vs 21.1% base) — and its eviction cycle landed far below the
intended one (median 11.1k against 15.9k), because the ~2-block eviction
chunk is a large fraction of a small budget. At 25000 the cycle
reconstructs (landing median 14.9k) and window truncations are fewest.

So the budget was raised to compensate for a chunk sized in *blocks*.
Sizing eviction by a minimum number of **tokens** instead would let the
budget be honest; that is a live roadmap item, and this reading is its
evidence.

KV cost was not the constraint: 23.5% mean against 20.4%, zero
preemptions, prefix-hit and generated tok/s unchanged.

**This entry is historical for its absolute numbers.** It was taken
before the current window and reply-cap coordinates, so 25000 is not a
value anything sets today. What survives is the chunk-vs-budget mechanism.

## The analyzer temperature

`duck_sweep6` — 50 jobs x 4 arms x the official 25, 200 configurations drawn
at random from the duck's knob space, read at a common 120k generated tokens
per game. Temperature 0.6, down from sweep 3's best-arm draw of 0.8246.

**Read the shape, not the slope, because the shape is what argues against
going further.** Within [0.3, 0.6] the raw slope is −0.78 (p = 0.92) — flat.
Within [0.6, 1.0] it is −5.23. The whole penalty lives above 0.6, so 0.6 is
where the damage stops rather than a point on a line, and nothing here
recommends going lower. Predicted gain +1.5 points budget-corrected, +1.0
terminal.

It does not disturb the rest of the sampling profile: no interaction with
`top_p`, `top_k` or either penalty (six tested, smallest raw p 0.29), which
is why the other coordinates stayed at their sweep-3 values until the
Flash-Next sweeps moved two of them (below).

**This is the weakest of that sweep's five recommendations and is taken as
such.** One knob of twenty-two, on 69 arms. On a Benjamini-Hochberg screen
it reaches q = 0.056, which **narrowly misses the conventional 5% FDR
threshold** and clears only a 10% one; without budget correction it weakens
to q = 0.310.

**What it replaces had no evidence at all, and that is the comparison that
matters.** 0.8246 was a random draw in sweep 3's best arm. That arm won on
its whole configuration, and nothing in it measured temperature — so the
choice here is not between a settled value and a marginal one, but between
an unmeasured coordinate and a marginally-measured one. Read the q values
as the strength of the case for moving, not as a bar the incumbent had
already cleared.

What carries it beyond that is the flat-topped shape rather than the
significance: a knob whose optimum is a plateau is the kind that survives a
hidden set, and 0.6 is a round number at the edge of that plateau rather
than a fitted peak. A targeted sweep — the four knobs sweep 6 settled
pinned, temperature drawn densely — would settle it cheaply, and has not
been run.

## The brevity line

`20260805` A/B: mean generation fell from ~370 to ~210 tokens per action
at a score within noise.

Steering, not enforcement — there is no numeric budget in the line,
because the model cannot measure its own token count, and the hard ceiling
stays the sampling profile's `max_tokens`. A later joint fit found the
line's apparent effect on score to be borrowed from knobs it happened to
correlate with, so treat the token reduction as the established result and
the score as unresolved. This is the full line's reading; its one-sentence
`half` form ships since sweeps 12 and 13 (below).

## Malformed-output retries — a judgement on weak evidence

The newline perturbation on a malformed-output retry was briefly removed,
on the argument that it existed for a kernel fault this model does not hit
and that retained bytes should stay fixed for the causal transcript, for
exact-token reconciliation, and for KV reuse. It was restored. **That is a
judgement call on suggestive evidence, taken deliberately, and this entry
exists mostly to say so.**

The signal: across three comparisons of 2200+ game lanes each, the
malformed failures that end a game split into a flat background of
reasoned-but-toolless responses (5, 3, 3) and **empty** responses — a
single token, no text — which read 1 → 3 → 7, with the 7 in the comparison
that ran without perturbation.

**This is not a controlled result.** Only one of the three had
perturbation off, so commit, configuration and elapsed time are all
confounded with the treatment. The malformed-attempt and recovery
denominators are not pinned down. And no identical failed request is shown
recovering *because of* the suffix rather than because of the bumped seed
that rides with it.

What carries the decision is the mechanism plus the asymmetry: an empty
response is the input-deterministic collapse a one-byte suffix change is
most likely to break, a seed bump alone re-enters the same state, and the
three costs are small and bounded. A clean test would settle it — shadow-
retry each naturally malformed prompt at the same seed, with and without
the suffix — and it is on ullm's roadmap.

`design.md` records the costs next to the mechanism, along with the
multimodal first-round case where the edit lands deeper than the tail.

## The file tools, first flag-on run

`duck_file_tools/file_tools_20260824_174253` — the official 25, one pass,
`file_tools="full"` and nothing else off the preset. **Not a comparison:
there is no control arm, so the scores settle nothing.** What it settles
is that the surface works and how much it is reached for.

Nine of 25 games wrote a file; ten files in total, eight of them `.py`,
sized 52–126 lines with 2–9 functions each. Every one of the modules
written was later `import`ed by a snippet, which is the behaviour the
import widening exists for. Zero file-tool calls failed.

Reached for rarely: 21 file-tool calls against 1427 `python` and 187
`memory` — 1.3% of all calls. `search_files` and `undo_edit` were never
called once. The surface costs about 1,020 prompt tokens on every request
(≈643 for the five schemas, ≈377 for the prompt lines), so on this run it
bought 21 calls for roughly 1.6M tokens of prompt.

Read it as the weakest arm rather than as a verdict: `persistence` was the
default `"always"`, where the kernel already keeps every helper for the
whole game and files add only classes, the stdlib and editability. The
arms worth measuring are `"level"` and `"round"`, where files are the only
thing that survives.

No file defined a class, so the one affordance a snippet genuinely cannot
have went unused. One module imported `os` and persisted its own JSON,
which is the import widening being used to its full extent rather than
just for helpers.

## The per-game wall net

`presets.GAME_RUNTIME_CAP_S`, applied through `duck_run_config()` — the
class default and both served presets, `None` on Kaggle. A hit produces
`gave_up`, scored as it stands; the deadline cancels the in-flight move
rather than waiting it out.

**What it is for is the ending, not the detection.** A sweep lane
(`tn36-ef4dde99`, job rs40) ran 9.9 h, 24,536 actions, 77,579 tokens, still
on level 0, and died at walltime as `cancelled` — scoring nothing and taking
its whole arm out of the analysis. The token allowance cannot see such a
game: actions issued from inside the kernel generate nothing, so 90k per
level never retires it. The net ends the same game scored.

**Cost on the shipped duck: nil, measured twice.**

| run | lanes | past 3 h | score lost |
|---|---|---|---|
| `baseline_main_a9bd8298` (main defaults) | 100 | 2 | 0 |
| `duck_sweep6_baseline` (this branch's defaults) | 149 | 1 | 0 |

Median game 84.7 min, p90 126.7, max 220.2. Both over-3 h games had
completed their last level well before the line (2.91 h and 1.59 h) and
spent the remainder failing the next, which the scorer does not credit —
computed with TAAF's own `ARC_SCORER.partial`, checked to zero error against
100 recorded `final_score` values.

**But the margin was one minute.** The nearest miss is `ar25-0c556536` at
179.0 min, and it is a win.

**Cost on swept configurations: small, concentrated, and a lower bound.**
Over 3,541 finished sweep-6 lanes (mean lane score 9.40), 112 ran past 3 h
and 10 would have lost score — 118 points, 0.033 per lane, 0.35% of the
mean. Nearly all of it is one game:

    lp85-305b61c3 p0  3.6h  levels 5->8   6.6 -> 38.2  (-31.6)
    lp85-305b61c3 p0  3.4h  levels 6->8  13.5 -> 40.6  (-27.1)
    lp85-305b61c3 p1  3.2h  levels 7->8  70.4 -> 96.0  (-25.6)
    lp85-305b61c3 p2  3.8h  levels 6->8  52.2 -> 76.3  (-24.1)

A lower bound because those are finished lanes, so the longest-running are
excluded by construction. Live at the time of measurement, job rs74 had ten
games at 3.8-7.3 h sitting at levels 1-7 and generating 22k-49k tokens/hour
— productive, not stalled.

**No threshold isolates a stalled game.** Two attempts, both negative.
Longest gap without generating a token: healthy lanes reach 54.8 min, while
the cancelled runaways' longest gap is 26.7 min — backwards. Sustained
throughput (fewest tokens in any 60-minute window): the five lowest lanes in
the sweep are all non-cancelled, `tn36` among them at 763 tokens/hour,
`gave_up`, 14,871 actions; a 5,000 tok/h floor fires on 26 healthy lanes and
catches 3 of 16 cancelled. Action-spamming is a continuum and most such
games finish on the token budget — the "runaways" are just the ones still
running when walltime arrived. So any threshold, wall-clock or throughput,
cuts into a distribution rather than isolating a defect.

**Six hours** is therefore chosen for its premium rather than its catch:
clear of `lp85`'s 3.2-3.8 h scoring finishes and of the 179-minute win,
while still ending the 9.9 h case. Premium is approximately zero even in
swept configurations.

**Nine hours since 2026-09-27**. Six was chosen on runs with fewer
games at once, and at full concurrency a game's wall stretches with how
loaded its engine is. Sweep 12's second round ran 200 lanes on a KV-bound
engine, with the sweep's own net at 9 hours:

- 9.4% of its bfloat16 games played past 6 hours, against none of the first
  round's 125 lanes (whose longest ran 4.0 h);
- 15% of the 145 games that ran past 6 hours scored more afterwards: 27
  levels and 7 wins came after the mark (`tn36` 13 → 53, `r11l` 71 → 100),
  their mean 29.96 → 32.04.

Over the round that is small, about +0.15 a game, but it lands on particular
arms, so a six-hour cut would read the engine's load as a property of the
arm. Nine still ends the 9.9 h case.

What this does not settle: nobody has computed the cost-versus-cap curve
that would let the number be read off evidence rather than judgement. And
Kaggle carries none of it either way (`max_runtime_s_per_game=None`), so
submission score is untouched.

## Engine and serving

- **FlashInfer autotune off.** A 2-seed, 25-game A/B under a hard
  per-game cap measured autotune ~17% slower on this workload
  (cf. flashinfer#3935). The same A/B put the then-current engine upgrade
  at ≈ −5-6% generated tokens against its predecessor, and Model Runner V2
  at neutral. **These were taken on vLLM 0.25.1 and the engine is now
  0.27.1** — the flags stay at the conservative values they justified, but
  recalibrate before changing them.
- **Multimodal processor cache in shared memory.** The default
  per-process `lru` mirror evicts an item the EngineCore later expects
  and crashes the request (vllm#31404); `shm` is one ref-counted cache
  with no such split brain. It is also not free to skip: cacheless, the
  per-request image re-preprocess measured ~17% of the saturated single
  host thread on a thinking-off 12h arm (py-spy, `qwen38_sweep` 20260817
  A/B), because it runs on the process that feeds the engine.
  **This matters far less now that no image rides the prompt by
  default** — the setting stays because a configuration that turns the
  image back on must not crash. The cache itself is off everywhere, the
  renderer's workers processing every image instead (`settings.md`, U-00
  and U-0V; the seal's `evidence.md`, "The engine columns").
- **gpu_memory_utilization: 0.98 local, 0.95 on Kaggle.** vLLM allocates
  the KV pool before its FlashInfer GDN prefill warm-up, which needs
  device workspace of its own; the smaller Kaggle card needs the wider
  0.95 margin, the B200 runs at 0.98 — the values the sweeps ran on. The
  headroom also pairs with the scheduler shape
  (`max_num_batched_tokens` and `max_num_seqs`): a wider token budget grows
  the activation workspace out of the same headroom, and the GDN warm-up
  state scales with the sequence cap.
- **`max_num_batched_tokens=8192` pinned; `max_num_seqs` pinned to the
  column's own `max_concurrent`** (80 on SLURM, 40 on Kaggle). That the cap
  is pinned at all is the measured part: unset, it is whatever the topology
  resolves — 128 for an in-process vLLM 0.25.1 engine, which is the
  configuration the sweeps ran, and 1024 under its API server on an 80 GiB+
  card. Measured on a B200 at 0.98: the server topology left unset OOMs in
  the GDN warm-up (2.21 GiB of per-sequence state on top of the KV pool),
  and the CUDA-graph capture range follows the sequence cap
  (`min(2 * max_num_seqs, 512)`: 256 graphs at 128, 512 under the server),
  so pinning the token budget alone does not save it. The harness admits
  `max_concurrent` games and each holds one generation at a time, so that
  IS the engine's demand; both columns sit below the 128 the sweeps ran at,
  which is the safe direction, since the warm-up state scales with the cap. At the in-process figure of 2048 the
  KV-offload traces showed what the budget does against the model's
  1,568-token recurrent block: one block-sized prefill per step and no
  admission behind it while a prefill runs, so any burst of cold prefills
  drags the whole engine to chunk length (ullm's kv_offload.md, "Prefill
  is the throughput"). Five
  blocks per step is the server's figure; its activation workspace comes
  out of the KV pool's headroom, so the profiled pool is the number to
  read after changing it. Unpinned, `serving="direct"` and `"server"`
  were also two different schedulers in every A/B.
- **`watermark` set** (`ullm.backends.deployment.ENGINE_ADMISSION_WATERMARK`), vLLM's own being
  zero: the reserve that makes the engine queue admissions rather than
  preempt running requests once the live set outgrows the pool. The
  reasoning and the sizing are in ullm's `kv_offload.md`, "The regime
  the design assumes": the reserve is also the free queue's length, and
  so the residence a freed checkpoint has on the GPU.

## The Kaggle weights are content-pinned, verified once

Dataset version 1 of `qwen3-8-27b-fp8-hf-017b9c7a` holds only
the 81 files of `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` — 30,890,049,597 bytes, checked
once against Hugging Face's official LFS hashes and Git blobs.

That check is what buys the shortcut: the public dataset version is
immutable, so a run mounts it directly instead of rescanning 31 GB before
every engine start. The verification is not repeated, so it is only as good
as the immutability guarantee — a *new* dataset version would need its own
check.

## The Kaggle host KV tier

Measured on vLLM 0.27.1: a 59.3 GiB GPU KV pool holding 1,691,648 tokens.
A completed 25-game Kaggle validation allocated the full 64 GiB host tier,
ran 48m04s with **zero preemptions**, and produced 1,126,476 generated
tokens at 390.49 tok/s — 0.9977x the completed 32 GiB run's
generated-token throughput, processing 2.13% fewer uncached input tokens,
though no offload reads occurred in either run. The cgroup peaked at
137.6 GiB of its 175 GiB cap, most of it reclaimable page cache.

**64 GiB is the largest measured-safe tier, not a chosen one.** vLLM
eagerly mmap-populates through `/dev/shm`; 96 GiB needed a 103.07 GB
mapping and failed during population. So the ceiling is Kaggle's
shared-memory and cgroup headroom, not the size of the gap the tier
covers.

That gap moved with the window. The same pool holds 25.81 full sequences
at a 65536-token window but **22.94 at 73728** — 2.58 GiB per sequence
against 2.30 — so the shortfall against the 28-game live cap grew from
about 2.2 sequences (~5 GiB) to about 5.1 (~13 GiB). Still comfortably
inside the tier, but the headline numbers above were taken at the narrower
window and the sequence count is the one to re-derive, not to quote.

Both counts are worst-case bounds, not expected pressure. KV blocks are
allocated as tokens actually exist, and neither the prompt budget nor the
compaction target moved with the window — so a sequence's realized
footprint follows the same sawtooth as before and touches the widened room
only while a reply is running past the old 8192 cap, which sweep 3
measured on under 4% of rounds, at up to ~0.29 GiB each. The widened gap
costs something the narrower one did not only when several such rounds are
in flight at once.

Total game count does not multiply the live process set **while slots are
granted in queue order**: `RunConfig.max_concurrent` queues games before
their agent copy and child process exist, and a game that holds a slot
holds it until it finishes.

Under `slot_policy="least_level_tokens"` it does. A game that has never
played outranks any game that has spent something, so a freed slot goes
to an unplayed game until the queue is exhausted — and parking releases a
slot, never a subprocess, so every game in the run ends up resident.
Measured at 15 slots over 50 games: 103 live children and 43 GiB of anon
against the queue-order arm's 33 and 19 GiB, or roughly 0.8-1.3 GiB per
resident game depending how far in it is. `roadmap.md` carries the guard
this needs before a 110-game queue.

## The per-level budget, and what replaced it on Kaggle

`max_tokens_per_level` at 90,000 came from experience rather than a
reading, and the whale's method has since been run on duck traces to
settle it (`arc3-experiments`, `duck_scheduling/FINDINGS.md`, on
`pure_duck_0251_long_20260821_234037/r0` — 100 plays recorded under a
250k cap). Simulated at the submission's operating point, 110 games
queued into 28 slots with 12M generated tokens at 9 h:

| arm | @ 9 h | @ 15 h |
|---|---|---|
| fill-slots + uncap, 90k (this default) | 11.27 | 17.10 |
| round-robin | 11.59 | 16.47 |
| fill-slots + uncap, 60k (best flat cap) | 13.28 | 15.76 |
| priority: least level-tokens, no cap | **13.73** | **17.48** |

Two readings, and the second is why the Kaggle preset now ranks slots
(`run_config.slot_policy`) instead of tuning the cap. The shipped 90k
scores *below* giving every game an equal share — as a triage rule it is
worse than not triaging. And every cap that beats it does so by
forfeiting score it would have earned later: 60k leads at the deadline
and trails by 1.7 six hours past it, because a cap frees a slot by
killing a game. Deprioritising does the same job without the forfeit,
which is why the priority arm is the only one above round-robin at both
columns.

The cap stays at 90k everywhere else: it is a submission's deadline that
makes idle capacity worthless, and off a deadline the same switch would
make a game's spend depend on where in the run it was scheduled.

One Kaggle run at 9 h with the cap lifted and the ranking on scored
**4.67**, against four runs of the 90k-capped configuration at 3.50,
4.58, 2.85 and 3.04. Above all four, but n=1 against a baseline whose
own spread is 1.7 wide — consistent with the simulation, not
independent confirmation of it.

**That run validates the operating point the simulation was built on**,
which is the more durable thing it produced. The whole sweep is
calibrated to 12M generated tokens at 9 h; the run generated
**11,568,087** over 8.80 h of play, with all 110 games started. So the
token axis every conclusion here is read against is the one a submission
actually spends, to within 4%. Read the per-action `history` for that
number: `final_generated_tokens` is only the tail attributed at finish,
which for a cancelled game is the spend since its last executed action
and an order of magnitude smaller.

## zero_recurrent, and where 1568 comes from

`zero_recurrent_at = 1568` is the accident's own position, measured, not
tuned. On the connector-on capture
(`logprob_repro_20260901_180641/conn_on_lp5_3ff142b`, 2 h, 100 runs, R5
traces with per-token logprobs): every game-start over-claim resumed at
exactly `attn_hit=1568, mamba_hit=0` — one recurrent block, because the
shared system prompt covers the first boundary and the attention cache
hits it whole. Offline replay against that capture puts the blanked arm's
first-token |d| at 0.084 nats vs the unblanked 5.60 (n=6 per stratum;
matched no-over-claim control floor 0.154), and a position sweep has its
minimum exactly at 1568 with 1567/1569 both worse — the blank lands
where the accident's did, to the noise floor.

The in-game read-out is gen/tool-call before the first compaction
(`zr_ingame_*`, connector off, 25 games): blank-on ~700–860 = the
connector-ON band; blank-off and asked-at-0 land in the connector-OFF
band (991–1082). Reference bands from duck_sweep9, images off.

**The score on the fork, measured.** Official 25, 4 passes, 12 h
wallclock, connector off, direct serving, all lanes played out:

| run | blank | mean score |
|---|---|---|
| `zr_ingame_20260902_110855/zr1568_27f70ae` | on, every call, at 1568 | 16.09 |
| `zr_ingame_20260902_085055/zr1568_everycall_935dd0c` | on, every call, at 1568 | 17.58 |
| `zr_ingame_20260902_085810/control_935dd0c` | off | 10.48 |

The blanked runs sit where the 0.25.1 accident put the duck and the
control where consistent serving put it, so the mechanism carries its
points onto 0.27.1. The two blanked readings differ by 1.5 on identical
configuration, which is the size of run-to-run noise at 4 passes; read
either against the control, not against each other.

**Asking on every call costs nothing at the engine.** Compared at equal
engine-window counts, the blanked arm processes more total tokens per
second than a never-asking control (3825 vs 3157) at a higher prefix
hit rate; its lower generated-tokens-per-second is the effect itself
(shorter responses shift the mix toward prefill). A per-armed-request
device sync on an earlier host-side implementation did cost throughput
(3541 total tok/s), which is why the blank is decided inside the kernel.

## What the blank was actually worth: the register

The position sweep (`zr_sweep_20260902_181704` and its follow-ups,
notebook in arc3-experiments `zr_position_sweep/`) was run to ask
which position scores best. It answered a different question. Position
does not order the scores; what orders them is how many lanes end up
writing in a particular reasoning register — a telegraphic voice, "We
need solve game. Need inspect current frame.", against the ruminating
default one.

Read by a fitted classifier over closed-class words alone
(`ullm.reasoning_register`, AUC 0.991 on held-out games; a second fit for
the opening block, because `let's` is a BLANKED marker over a whole game
and a CONTROL marker at the opening):

- Across runs, score tracks the FRACTION of lanes that open in the
  telegraphic register, r = +0.88. Within a cell, a lane in it is worth
  +9.8 points (p = 1.4e-21).
- The register is decided at the opening block, per lane, and then holds:
  a lane that starts in it stays in it.
- What decides it is the sampler. One game, 100 concurrent lanes,
  identical prompts, a cache salt per lane so no lane can read another's
  blocks (`zr_onegame_20260904`): the lanes split 44/56 between the two
  registers rather than landing together. The blank does not create the
  register; it raises the rate at which lanes fall into it.

So the blank is one route to the register, not the thing itself, and a
position that perturbs the state without moving the register is worth
nothing.

**Reaching it from the front instead.** Asking does not work: the
register offered as an example in the system prompt, and again as the
last line of the user prompt with its own tokens immediately before the
generation prompt, produced it in 0 of 100 lanes each. Serving it does.
Text served as the first response's own opening ids
(`DuckAgent.inject_text_on_first_prompt`) puts a lane in the register in
100 of 100. This is not asking: they are the model's own tokens in its
own voice, the same continuation the sampler already makes unprompted on
some fraction of lanes.

Two words are enough (`zr_inject_length_20260904`, one game, 100 lanes
each, opening scored with the injected text stripped off):

| served | tokens | opening register |
|---|---|---|
| `We` | 1 | 0.10 |
| `We need` | 2 | 0.96 |
| `We need solve` | 3 | 0.99 |
| `We need solve game.` | 5 | 1.00 |
| `We need solve game. Need inspect current frame.` | 10 | 1.00 |

What ships is set under "How short the injections can be" below, which
reads this threshold across a whole sweep rather than one game.

**Serving it once does not hold a game, and compaction is why.** The
20-minute single-game runs above read the opening and the tail of a short
game, and reported the register holding. Over a full game it does not.
Scoring every reasoning block and bucketing by how many compactions
preceded it (20 finished games per arm, median first compaction 22% in,
6-7 compactions per game):

| compactions elapsed | injection only | blank on every call |
|---|---|---|
| 0 | 0.947 | 0.955 |
| 1 | 0.891 | 0.933 |
| 2 | 0.854 | 0.886 |
| 3 | 0.635 | 0.879 |
| 4 | 0.464 | 0.874 |
| 5 | 0.263 | 0.815 |
| 6 | 0.241 | 0.676 |
| 7 | 0.224 | 0.481 |

By the fifth compaction 82% of the injected arm's blocks are back in the
control register against 9% of the blanked arm's.

Why the blank holds longer is NOT established, and the obvious story --
that it is re-asked on every call and so re-creates the condition -- does
not survive its own numbers. The blank's measured effect is at a tail of
~1500 tokens (the game-start stratum; the accident's post-compaction
stratum ran 12k-32k). In these runs the tail past position 1568 is a
median of 26k tokens BEFORE the first compaction and 42k after, with only
5% of pre-compaction calls under 3k. The blank is therefore inert on
nearly every call of both arms, and is as much an opening-only
intervention as the injection is. What separates the two arms after the
third compaction is unexplained.

**Repetition is what sustains it, not the words.** Serving two tokens
(`We need`, `DuckAgent.inject_text_on_later_prompts`) on every call after
the opening, scored with the injected tokens stripped off first
(`control_inject04w_later02w`, 51 lanes in flight against the same
opening without it):

| compactions elapsed | opening only | opening + every call |
|---|---|---|
| 0 | 0.948 | 0.953 |
| 2 | 0.834 | 0.919 |
| 3 | 0.648 | 0.908 |
| 5 | 0.297 | 0.860 |
| 7 | 0.222 | 0.769 |

Verbose-block share at the fifth compaction: 78% against 4%. That is a
better hold than the blank manages, on a stock engine, for two tokens a
call. It still declines, slowly, and nothing here says why.

What this does NOT measure is the cost of the constraint. Every response
is made to open on the same two tokens, overriding the varied openers a
concise lane writes for itself ("Let's try", "No. Maybe the", "Timed out
after"). Forcing a VERBOSE opener (`Let me`) on the same schedule is the
control that separates the constraint from the register, and has not been
run.

## What the register is worth: the score

Everything above measures the register. This is what it scores, on the
official 25 at 4 passes each (`zr_inject_full_20260904`, against
`shared/baseline_main_38e4852e`; comparison `compare_register_arms`).
Official ARC-AGI3 score per game, each arm scored where it ended:

| arm | score | SEM | vs baseline |
|---|---|---|---|
| main as it shipped: blank at 1568 on every call, no injection | 16.09 | 0.74 | — |
| **open one sentence + `We need` every call** | **26.26** | 1.51 | **p = 0.003** |
| open one sentence only | 18.90 | 1.89 | p = 0.24 |
| open three sentences only | 17.61 | 1.65 | p = 0.45 |
| open one sentence + blank at 1568 | 18.36 | 2.63 | p = 0.46 |

p from Welch over per-pass means; the paired t and paired permutation
tests over per-game differences agree on the shipped arm (0.009, 0.003)
and are the ones to read for the rest, being paired on game.

The baseline is main's own shipped configuration at the time, which was
the BLANKED one: its saved solver carries `zero_recurrent=True` at 1568,
and the run is `zr_ingame_zr1568_27f70ae` at revision 38e4852e — the same
16.09 the table above reports for blank-on. So the 10.2 points are over
the blanked main. The unblanked control in that earlier table (10.48) is
a different revision and not an arm here; against it the gap is wider.

Three readings decide the two defaults, and each matches the register
evidence above:

- **The schedule is the effect.** Opening once is worth 2.8 points and
  does not separate from baseline (p = 0.24); adding `We need` to every
  call is worth 10.2 and does (p = 0.003). The two injected arms differ
  from each other at p = 0.024. That is the compaction result scored: an
  opening left in history stops being served once compaction drops it.
- **More words at the opening buy nothing.** Three sentences (17.61)
  scores no better than one (18.90) — the threshold really is two words,
  and the rest of the sentence is not doing work.
- **The blank adds nothing on top of the opening.** One sentence with the
  blank (18.36) scores as one sentence without it (18.90). Since the
  tokens reach the register on a stock engine, the fork's blank is
  redundant where the injection already went, which is why
  `zero_recurrent` ships off.

It is not bought with tokens. The shipped arm spends more per game, and
still leads at a matched budget: 25.31 against the same 16.09 on the
budget-matched reading, and its score-vs-tokens curve is above every
other arm from roughly 100k tokens per game onward. The blunter scorers
move the same way — games fully beaten 0.10 → 0.22, levels completed
2.89 → 3.57, flat-weight score 22.26 → 32.97.

What this does NOT establish is the cost of the constraint, which the
paragraph above states: no arm here serves a VERBOSE opener on the same
every-call schedule, so "the register held" and "every response opens
alike" are not yet separated.

## How short the injections can be

Both injection strings shipped at lengths chosen for safety margin rather
than measured at the margin. Sweep 10 drew three openings and three
per-call reminders across its 100 arms, which makes the register itself
readable per arm — the classifier over the runs' own reasoning blocks
(`platypus.reasoning_corpus`), not a separate experiment.

**How short the opening can be.** Scored with the opening model on the
first reasoning block of all 2,500 lanes, so the per-call reminder — which
cannot reach a game's first response — is a placebo in the same read:

| `inject_text_on_first_prompt` | tokens | opening register | lanes that never entered it |
|---|---|---|---|
| `We need` | 2 | 0.921 | **23 / 750 = 3.1%** |
| `We need solve` | 3 | 0.935 | 0 / 825 |
| `We need solve game.` | 5 | 0.939 | 0 / 925 |

Two words is where "Reaching it from the front instead" put the threshold,
and this says the threshold is where induction becomes *unreliable* rather
than where it starts working. The 23 misses are spread over 8 separate
arms, every one of them at two words and with `zero_recurrent` off — the
shipped pairing, where the injection is the only thing inducing the
register and nothing catches it when it fails. Three words is
indistinguishable from the sentence (Fisher p = 1.00 on misses, −0.004 on
the mean, p = 0.29), so **the default is three words**: the shortest
opening measured to work in every lane.

The placebo does not move (ANOVA p = 0.30 across the per-call values), and
`zero_recurrent` does what it should — 0.948 with the blank against 0.915
without — which is the check that the classifier is reading the register
and not the arm.

**How short the per-call reminder can be.** A different job, so a
different reading: the reminder re-induces the register with the register
already in a context that compaction is eating away at. Scored per block with
the shipped model over 900 lanes (9 of the 25 games, every arm), read on
the **last 20 blocks** of each lane, median 198 blocks per lane. In the
shipped configuration, `zero_recurrent` off, where the reminder is the
only mechanism holding it:

| `inject_text_on_later_prompts` | tokens | last-20 register | tail blocks out of register |
|---|---|---|---|
| `""` | 0 | **0.446** | **57.2%** |
| `We` | 1 | **0.935** | 2.8% |
| `We need` | 2 | 0.915 | 5.7% |

One token holds it as well as two — nominally better, +0.021 at p = 0.30,
and the same ordering across all 100 arms (+0.017, p = 0.12) — so **the
default is one token**. The opening's threshold does not transfer: `We`
alone induces the register in 10% of openings from cold, and sustains it
in 97% of tail blocks once it is already there.

The empty arm is the mechanism behind its score: without a reminder the
register is gone from more than half the tail blocks by the end of a game,
which is what "an opening left in history stops being served once
compaction drops it" looks like measured rather than inferred.

What this does not settle is the shape of the decay between those two
columns — the reading is a tail average, not a curve against compactions
elapsed, so "one token is enough" is established for games of this length
and not as a limit.

## Sweep 10: what moved, and what did not

`duck_sweep10`, run seeds 98–102 and 105–108 — 9 jobs, 100 arms, the
official 25 games per arm, 2,500 lanes over 24 knobs drawn at random. The
full report, with every reading and its caveats, is
not released.

This was the first sweep on a harness that reaches the register through
served tokens rather than a blanked state, so its job was to re-ask the
decisions the blank-era sweeps took. **Almost nothing moved.** No knob
clears a Benjamini–Hochberg screen at the arm level (smallest q = 0.155),
and `null_control` — drawn and never applied — comes back at +0.09
(p = 0.91), which is the floor every row should be read against. Three
defaults moved anyway, none of them on a score:

- **`expose_undo` on.** The joint fit over the whole benchmark says
  −2.28 (p = 0.028), which is noise, and the knob's own mechanism shows
  it: only 6 of the 25 games have an UNDO to expose, and splitting the
  same 72 arms gives −0.07 (p = 0.975) on those six and **−2.98
  (p = 0.013) on the 19 where the flag provably does nothing.** The whole
  apparent penalty sits on the placebo. Every one of the sweep's 19,794
  UNDOs falls inside those six games and inside flag-on arms, and the one
  channel that could have made the flag model-visible elsewhere — a
  different refusal when the model asks for UNDO anyway — never fired in
  any of the nine jobs. Where UNDO exists the model already reaches for it
  in 93% of lanes and spends 6% of its actions on it — used, and used
  badly: 60% of the calls that contain an UNDO run an apply/un-apply loop,
  and those loops hold 88% of all UNDOs; 27% of the segments that undo
  spend more UNDOs than half the forward moves since the level began or
  the last RESET — past the point where a RESET would have cost less —
  and those segments hold 66% of all UNDOs. The model is never told that
  an UNDO costs an action, which is the first thing the prompt owes it
  (`roadmap.md`).
- **`file_tools` on.** +0.13 raw (p = 0.86), +0.29 adjusted for *none*
  (p = 0.785). Sweep 6 measured this flag at **−5.39** and that number
  does not transfer: sweep 6 scored under the 0.25.1 serving bug its own
  first finding identified as worth ~44% of score, that bug made the
  prompt's exact bytes matter, and turning the suite on rewrites 14% of
  the system prompt starting 9% of the way in. Its coefficient is the
  tools plus a large prompt perturbation, read under a regime sensitive to
  prompt perturbation, and the two cannot be separated from that data.
  What the suite is used for, over the 1,050 lanes of the 42 flag-on
  arms: 89% of lanes touch it, at 2.2% of all tool calls — `view_file` in
  77% of lanes, `create_file` in 63%, `str_replace` in 39%, `search_files`
  in 4% (45 calls), and `undo_edit` in none at all. Whether that is five
  tools being too many to choose between or two being the wrong two is
  what `expose_tool_search_files` and `expose_tool_undo_edit` exist to
  answer (`roadmap.md`).
- **`numpy_scalar_repr` to `legacy`.** +0.49 raw for `np2` (p = 0.66),
  +0.25 adjusted (p = 0.895), on 13 arms — the draw was 80/20, so the
  sweep was never going to resolve it. Moved on cost instead: `np.str_('B')`
  for a one-character cell and `np.int64(0)` for a one-digit one is about
  nine wasted characters per scalar in everything the model prints back to
  itself, worst where `frame_api="numeric"` is trying hardest to be terse.

**What the sweep did not settle.** With 72 arms and a between-arm SD of
about 3.6 points it resolves effects of roughly two points and larger, so
each of the three above could be worth a point in either direction and
this sweep would not see it. They are tie-breaks decided on grounds
outside the score — the affordance argument for the first two, character
cost for the third — not measurements.

The strongest raw signal in the whole design is `frame_api`: ergonomic
21.03 against numeric 19.06, p = 0.0074 (q = 0.155). It points at the
value already shipped, so it moves nothing, but it agrees with sweep 6
condemning the numeric surface independently.

## Sweep 11: the prompt, and how its result got here

`duck_sweep11`, run seeds 814–845 — 32 jobs, 256 arms, the official 25
games per arm, 6,400 lanes over 20 knobs drawn at random. The full
report is
not released.

**How the shipped prompt is the swept prompt.** The sweep drew against
`ConfigurablePrompts`, a surface with one switch per claim whose defaults
rendered `DuckPrompts` byte for byte, and its six winning switches were
shipped by the procedure `design.md` describes under "There is one
implementation": set them as that class's defaults, then move
`DuckPrompts` until the equivalence test is green again. The prompt that
ships is the prompt the arms played, and the test is what says so. The
surface survives at the tag `sweep11-configurable-prompts` for the next
prompt sweep to branch from.

Two consequences worth naming. The sweep pinned four render modes —
`frame_api="ergonomic"`, `persistence="always"`, the file suite on, no
images — and refused anything else, so every result below is measured at
those and at no others; the remaining mode combinations are pinned by the
prompt goldens, not by evidence. And two of the winners cut the same
claim: with `dedup` dropping its copies and the checklist asking only on
the first move of a level, "compare `frames[-2]` with `frames[-1]`"
narrows from four sites to one, the `python` tool description, which the
model reads every turn. That combination was itself played — the levels
were drawn independently, so about a third of arms rendered it.

**What moved.** Two readings were taken, at a common token budget and at
the terminal score; a knob is quoted at the one it enters on.

- **`level_summary_entry` on.** +0.87 at budget (p = 0.037) and +1.47
  terminal (p = 0.003) — the only change that reads on both. One
  consolidated journal entry is asked for on the move after a level
  transition. It is also the knob most coupled to a setting it cannot
  see: the journal is a hard token ceiling that evicts silently, and this
  spends against it once per level.
- **`turn_checklist` to `first_level_move_only`.** +1.35 terminal
  (p = 0.026). The text is what `on` renders; the change is asking for it
  on the first move of a level instead of every turn.
- **`cost_model` explicit.** +1.06 terminal (p = 0.032). Names the three
  costs — tokens per level, actions above a baseline, code runtime free.
  It completes 0.12 fewer levels and spends 36% fewer actions, and scores
  better on the levels it does finish: the agent optimising the thing it
  is actually scored on, which is also a better surface for a trainer to
  improve on. Its token claim is true of the shipped configuration and of
  every configuration with a per-level allowance; under
  `max_tokens_per_level=None` the prompt would overstate it, and no
  render fact exists to condition on.
- **`expose_tool_search_files` off.** +0.82 at budget (p = 0.046),
  agreeing with sweep 10's usage count of a tool the model reached for in
  4% of lanes.
- **`expose_tool_undo_edit` off.** Not significant. Withdrawn on sweep
  10's count of zero calls against a schema paid for on every turn.
- **`dedup` deduplicated, `action_economy` concise, `mouse_optional`
  `when_present`.** None significant. Tie-breaks: the first two shorten
  the prompt without losing a claim, and the third spends the coordinate
  convention only on games that offer MOUSE.

**What did not move.** `explain_undo` stays off. On the six games whose
engine offers ACTION7 — the only ones where it can act — it reads −1.21
(p = 0.10), and flat on the other nineteen as a placebo should. Sweep
10's finding that the model spends two thirds of its UNDOs past the point
a RESET would have cost less still stands; this wording is not the fix
for it. An agent-side rule refusing an UNDO after another action in the
same move was drawn beside it, did not earn its place either, and is not
carried here: neither the sweep surface nor an unused enforcement path
belongs on the mainline. Both are recoverable from
`sweep11-configurable-prompts` if a later sweep wants to draw them again.

**What the sweep did not settle.** The F statistic over the whole design
sees nothing, and `null_control` moves about as much as the smaller
effects claimed above. At 256 arms this design resolves roughly a point;
five of the eight changes are inside that. They are tie-breaks on prompt
length and on affordance, decided on grounds outside the score, and are
labelled as such rather than as measurements.

## The Flash-Next column: budget, split, tier regime

`flash_smoke/2026-09-10*` — 32- and 100-slot smokes of the pruned
Qwen3.8-Flash-Next (e307) on the 0.29 line, in an allocation on dgx03.
The column serves with the PLE table on the host, the host KV tier on,
and the shipped 8192-token budget; here is what each of those rests on.

**The 8192 budget, and the split it exposed.** At 8192 the startup
matrix's prefix-cache row failed on this model where 2048 passed, tier on
or off. The engine core sets `cache_config.block_size` to the minimum
over KV groups, on Flash the QSA ring's 4 tokens, and the scheduler's
align-mode chunk split aligned prefill chunks to that instead of the
1568-token Mamba block. A whole-prompt chunk then ended past the last
Mamba boundary and left every full block null; at 2048 chunks ended
mid-block, and every prefix hit before the fix restored a recurrent
state short of its boundary. Fixed on the fork (the split takes the
Mamba groups' own block size); the 27B was never affected. The 32-slot
reference after the fix, no request gap: 1851 tok/s mean (1939 median)
at 30 in flight, 95% hit, 1518 tokens per action — against 1666 at 29 on
the old split at 2048, which is the budget and the split together.

**The PLE table on the host.** No measurable cost at 32 slots; residency
96 → 48 GiB, pool 62 → 109 GiB. On by default.

**The tier's regime.** 100 slots with the tier: before the pool line
2750–3200 tok/s at 78–92 in flight; at the line (89% pool, the watermark)
2400–2600 tok/s with 2.2–2.7 GB/s each way, a 97% external hit rate, a
queue of 3–4 on capacity and no preemptions, held for ten minutes at
both budgets. The regime the column is for.

**FP8 KV: shipped since** (`packages/ullm/flash_optimization.md`, "FP8 KV
cache"). At this smoke's pin the sparse-attention layer still refused any
KV dtype but bf16 at construction, so the figures above are bf16-cache
figures.

What it does not settle: correctness of tier-restored recurrent states
beyond games scoring in 20-minute runs, and a score against tier-off at
the same concurrency. The 4x25 runs at the closing configurations are
that.

## The Flash-Next column's concurrency and cache-hit unit

The duck's P-EV values for Flash-Next -- 200 slots on the B200, 80 on the
RTX (91 and 36 since the prompt budget went to 125,000, scaled with the mean
context; sweeps 12 and 13, below), and `prefix_match_unit` 64 on the RTX --
rest on the traffic replay
(`packages/ullm/traffic_benchmark.md`), and the readings are the Flash
record's: the RTX's bf16 pool runs ~66 and 80 keeps a short queue
("Concurrency and the pool"); the unit reads -13% uncached prompt at an
unchanged running set there ("Cache hits inside a block"); the B200's 200
is the float32 pool's tier-bound 150 scaled to the bf16 pool, a judgement
until a long run reads it (its open item 23).

## Speculative decoding on Flash-Next

The same smokes, 32 slots, no request gap, samples past the ramp in the
28–31 in-flight bucket. Two draft tokens (`num_speculative_tokens=2`);
one and three are not measured. Not shipped on the B200 column; measured
for Kaggle.

| arm | gen tok/s | per request | acceptance | per draft | drafter on card |
|---|---|---|---|---|---|
| e307, no MTP | 1851 | 61.9 | — | — | — |
| e307, head sliced to 307 experts | 2306 | 79.2 | 59.8% | 1.19 | 3.6 GiB |
| e307, the source's whole 512-expert head | 2528 | 81.2 | 62.6% | 1.25 | 5.5 GiB |
| unpruned source, its own head | 2003 | 64.1 | 64.6% | 1.29 | in a 78 GiB pool |

A quarter to a third more generated tokens per second at the same
occupancy, the same tokens per action (drafting changes speed, not
output), and acceptance that does not move with occupancy (59–62% from
17 to 94 in flight). The head matched to its own unpruned target accepts
five points more than the sliced head under the pruned one; the whole
head under the pruned target sits between, so the prune costs the head
about two points and the slicing another three. The unpruned arm's lower
throughput is its pool: 78 GiB fills at 32 slots and the hit rate falls
to 73%.

**With the tier: refused.** The victim connector refuses the eagle
shape, and vLLM, with no group annotated as the drafter's, flags every
group volatile, which its own comment says kills Mamba reuse in the stock
connector too. At 100 slots without the tier, MTP holds 1320–1440 tok/s
at the pool line against 2400–2600 with the tier and no MTP: at high
concurrency the tier is worth twice what drafting is. Supporting both
means annotating the drafter's group on the fork and teaching the victim
connector the upstream connector's three eagle rules.

**The closing 4x25s (10–11 Sep, `flash_smoke/20260910_2209*`).** Two
arms of the official 25 over four passes, 12-hour wall, the table on the
host, the whole 512-expert head from its own directory for the MTP arm:

| arm | wall | won of 100 | mean score | gen tok/s past ramp | median tokens per game |
|---|---|---|---|---|---|
| 100 slots, the tier | 2h25 | 21 | 28.45 | 2012 (134 samples) | 154,540 |
| 32 slots, MTP 2 drafts | 3h15 | 19 | 29.09 | 1558 (182 samples) | 157,592 |

The same games, the same tokens per game, the same score within a run's
noise; the tier arm plays them a third faster on the wall because it
keeps the card full, where 32 slots with the 5-second hold drift to
16–23 in flight. Acceptance held at 63.3% over the whole run, 73% on the
first draft position and 54% on the second, so a second token earns more
than half its cost and a third is not ruled out by these numbers.

**The draft is sampled, not argmaxed (11 Sep,
`flash_smoke/20260911_080049_c32_nogap_mtp2_prob`).** vLLM's default
proposes the head's argmax and verifies it as one-hot, which caps
acceptance at the target's top-1 probability under the request's
temperature however good the head is. With `draft_sample_method=
"probabilistic"` the head samples and hands over its probabilities, and
acceptance becomes the overlap of head and target: 65.0–66.1% (1.30–1.32
per draft, 74–75% / 56–57% by position) against 62.6% (1.25, 73% / 54%)
greedy, at 2545–2804 tok/s against 2528, same slots, same head, same
tokens. The output distribution is the target's under both. Not applied
by the repo (a speculative config is the experiment's own); set it
wherever a drafter is configured. The remaining gap is the head's own
disagreement with the target in this domain, a quarter of its mass at
position one, which is what a domain-trained head would move.

What it does not settle: the draft count beyond two, and the drafter's
cost on Kaggle's card, where 5.5 GiB is not five percent of the pool.

## The e307 plan and padded rows

The plan's histogram (5 Sep, the source model at 32 slots, norms on) was
captured before the instrument masked CUDA-graph padding, so its rows
include padded ones, routed from stale copies of recent tokens.
`platypus_demo/flash_padmask_20260911_134422` (11 Sep, the pruned model
on the demo wiring at 100 slots, 19-minute budget) captured with the mask:
1.46% of 4.4M rows were padding. Over the 307 survivors, mapped through
the checkpoint's kept lists, the two histograms' mass shares agree at
Spearman 0.95 per layer (median over 48, min 0.86); a further cut of 61
experts per layer by either would drop the same experts 80% of the time;
and the old ranking's coldest 61 survivors carry 2.5% of the masked run's
mass, against 19.9% for a uniform share. The padding share is well under
the difference between two runs of different shape, so the plan stands;
the ratio sweep measures the rest with more than one capture.

## Sweeps 12 and 13: the Flash-Next defaults

`duck_sweep12` and `duck_sweep13`, the first duck sweeps on Qwen3.8-Flash-Next.
Sweep 12 had 12 jobs with 77 arms read; sweep 13 had 14 jobs and 280 arms, in
four batches. Each arm played every game once: the official 25 less `ar25`,
and ten re-arc-3 games tagged arc-like and hard. The full report, with every
reading and its caveats, is
not released.

The defaults they moved (2026-09-26 and 2026-09-27), read on the
sweep's score at the Kaggle cut:

- **Summarize** in place of eviction: summarize with the journal read +3.48
  over prune with it in sweep 12 (p = 0.008).
- **The journal off**, for one place for memory rather than two. Both sweeps
  leaned towards keeping it, neither significantly: −0.66 in sweep 13
  (p = 0.23) and −2.24 in sweep 12 (p = 0.16).
- **The grid as an image on every prompt**: +4.50 over none in sweep 13's
  first three batches (p < 0.001), and +3.04 in sweep 12 (p = 0.035). Level
  start read −1.57 against it (p = 0.011).
- **No presence penalty**: −2.48 a unit where the image travels (p < 0.001),
  and flat without. Per unit, it made a third more requests a game, at
  replies a quarter shorter.
- **No nucleus** (`top_p` 1.0): nothing read either way. Neither penalty nor
  nucleus is carried into a speculative draft, which vLLM samples at
  temperature alone, so either one caps a drafter's acceptance, and trainers
  dislike both.
- **A 125,000-token prompt budget** (a 158,280-token window). Read at one
  per-level cap common to every arm, the budget gained +0.76 a 10k
  (p = 0.013) and was still rising at 180k. The Kaggle cut, priced at the
  box's throughput before speculative decoding, reads +0.11. Each cell's
  `max_concurrent` moved by the old budget over the new, so the cache its
  games hold stays where it was: context per generated token runs a steady
  0.59 of the budget (median over sweep 13's arms, 0.56-0.68), so a game at
  125k holds about twice what it held at 56,832.
- **On weak or no evidence, as judgement calls:**
  - the image at 8 pixels a cell (+1.02, p = 0.047);
  - the colour-name legend;
  - brevity at half (+1.31, p = 0.040);
  - thorough summaries;
  - the file-and-object listing after a summary;
  - the retest and winnable lines;
  - the Python session cleared at each level;
  - the wide sandbox, which every arm read played.

**The summary's room under the engine window** (`MIN_SUMMARY_ROOM_TOKENS`,
8,192; 2026-09-23). Sweep 12's 10,794 summaries wrote a median 0.26
of their target without thinking and 0.36 with it; none without thinking
reached 3.7k tokens, and p99 with thinking was 8.9k. So a configuration
reserving the whole cap, twice or six times the target, reserved window no
summary used, and 8,192 covers every summary without thinking and about the
p99 with it.

`top_k` 194 stays for now. `compaction_ratio` is moot under thorough
summaries: the request states no size, and the ratio only caps a summary's
length. Over sweep 13's four batches a thorough summary averaged 17% of that
cap, and 1 of 10,934 came within 5% of it. `memory_context_tokens` sizes the
journal's tail, which nothing reads with the journal off.
