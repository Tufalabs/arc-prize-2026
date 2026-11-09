# Where a setting lives

*Register: as-built.*

The harness/trainer boundary is a decision in `roadmap.md`.
Reply caps are agent defaults, stamped on each request's sampling copy.
Engine derivations are agent controls: a supplied factory without engine arguments
requires an explicit opt-out for each enabled derivation.

Every setting that determines an agent belongs to exactly one of eight
buckets. The bucket says which package owns it and which table states it, so
"where does this value go" has an answer before the argument starts.

Ported from arc3-platypus's `settings.md`. The rule is theirs; the
placements below are this repo's, and every row where the two repos hold a
different value says why.

## The two questions

**Is it task-free?** A setting that would be true for anything served by this
engine on this box belongs to `ullm`; one that exists because ARC2 is the task
belongs to `platypus` or `duck`. The test is a real consumer: this repo uses
`ullm` without ARC3's `platypus`, so anything both tasks need is on the
`ullm` side, and ullm's own tables are the ones this repo consumes
(`ullm.backends.deployment`, at the pin in `pyproject.toml`).

The coordinates themselves follow that split. `ullm` describes boxes and
checkpoints; each task then says which of them it runs on, and refuses the
rest — `kaggle-rtx` is a box `ullm` can serve and ARC2 never does, as
`kaggle-4xl4` is one ARC3 never does.

**Which coordinate decides it?** A deployment is an **engine** (which
checkpoint) and an **environment** (which box, which deploy target). A setting
is decided by neither, by one, or by both.

|  | neither | engine | environment | both |
|---|---|---|---|---|
| **ullm** (task-free) | U-00 | U-E0 | U-0V | U-EV |
| **platypus / duck** (ARC2) | P-00 | P-E0 | P-0V | P-EV |

## Layering

Buckets are precedence, not exclusive ownership. A setting has a **home** —
the most general bucket that can state it — and any more specific bucket may
override it there.

On the task side that reads literally: `duck.presets.configure()` walks
the layers down the page, assigning onto the agent — what ARC2 asks
everywhere, then the engine's, then the environment's, then the cell's. A
later layer overwriting an earlier one is the mechanism, and each chain
refuses a coordinate it does not cover.

General → specific:

    U-00 → U-E0 / U-0V → U-EV → P-00 → P-E0 / P-0V → P-EV

The task layer may override the task-free layer on the same setting, because
ARC2 knows things about its own workload that a checkpoint or a box cannot —
the sampling profile overriding the model card's is exactly that.

A setting nothing here overrides has its home and no more: a row that
exists only to restate a default from a higher layer is a second copy, free
to drift.

## Derived, not stated

Three engine settings are computed from others at
`DuckAgent.prepare_for_backend()`, the last moment before the backend is built,
so that an override made after construction is seen.

| Derived | From | Why they are one number |
|---|---|---|
| `max_model_len` | `context_window_tokens`, or `engine_ceiling_tokens` under `summarize` | The agent may build a prompt up to its window and the engine must accept exactly that; the reply reserve is inside the window already. A summary request goes past the window by the round that overflowed it, the instruction and the summary, and by how much is not the agent's to know, so under `summarize` the engine keeps the window it was built with — the checkpoint's native context, which the build records on the agent — and the derivation refuses if the window, the reply reserve and the summary's cap do not fit under it. |
| `max_num_seqs` | `run_config.max_concurrent` | The harness admits that many games and each holds one generation at a time. |
| the image processor's `max_pixels`, and no video (`limit_mm_per_prompt["video"]` 0) | `image_scale`, `image_gridlines` | The engine profiles one image of the processor's ceiling at start-up and keeps that activation out of the pool; the checkpoints' own ceilings are tens of megapixels and the agent's largest render is under one. Past the ceiling the processor shrinks an image rather than refusing it, so the ceiling follows the render rules (`duck.images.largest_image_pixels`). The duck sends no video, so the same settle refuses it and the profile sizes for images alone (upstream's `packages/ullm/flash_optimization.md`, "Competitor tricks"). |

Each engine argument has a flag on `DuckAgent` (`derive_*`). A value someone
chose explicitly is refused rather than overwritten: the factory records what
it was built with and updates that as the agent settles onto it, so a caller's
number can always be told from one nobody picked.

`gpu_memory_utilization` is deliberately **not** in this graph. It is measured
per cell and left a little conservative, and neither derives from
`max_num_seqs` nor drives it.

**The deployment snapshot** (`tests/references/deployment_snapshot.json`) is
what holds every column to these tables: it freezes each column's whole
configuration as resolved at `prepare_for_backend`, and a diff is the list of
what a change did to every column (`AGENTS.md`).

---

# The buckets

The task-free buckets are ullm's, stated in ullm's `settings.md` and its
`deployment.build()`; what follows names only the rows this repo depends on
or differs on. The ARC2 buckets are stated in full.

## U-00 · task-free, neither coordinate

vLLM defaults wrong for everyone: `generation_config="vllm"`, no image
processor cache (`mm_processor_cache_gb=0`: the cache evicts oldest first, a
hit not refreshing an image, so a run whose images in use outgrow it loses
its hits for good; the renderer's workers process every image instead, U-0V),
`mm_processor_cache_type="shm"` for a cell that turns it back on, the
admission `watermark` (0.10), the V2 model runner (`UNIVERSAL_ENGINE_ENV`),
CUDA graphs on and prefix caching on, the engine-startup timeouts. All of it
reaches this repo through `deployment.build()`; nothing here restates one,
and the two this repo used to set by hand — the runner and the multimodal
cache — are these.

Beside them, three this repo's runs depend on. `prefix_cache_retention_interval=None`
caches every recurrent state block: the engine's default keeps only the
boundaries it can prove reachable, and a continuation resumes at its own
previous end, which is not one, so every continuation would recompute state.
`async_scheduling` is stated rather than inferred, since unset vLLM disables
it with a warning when something is incompatible. `dense_cuda_capture` (a
factory field, on) captures a graph for every decode width up to
`max_num_seqs` instead of padding decode batches up to the default ladder;
the four-L4 box turns it off (U-0V). The host tier's shipped size and its
victim connector are U-00 too (whether it is on is P-00, its size P-EV), and
so is its park mode: the connector keeps whole conversations on the host
between turns wherever the engine names each one's thread and keeps decode
tails (U-E0), with `kv_park.log` in the job directory unless TAAF's minimal
diagnostics turn it off (upstream's `packages/ullm/kv_offload.md`, "Park
mode"); its parking on the GPU (`gpu_park`), where a parked thread stays
pinned on the GPU until the pool needs the room, is on by default wherever
park mode is (2026-10-18; the same file, "Parking on the GPU"). Two
instruments stay off: `profile`, torch-profiler windows once the pool fills,
and `memory_log_s`, the fork's start-up memory sampler.

## U-E0 · task-free, engine only

The checkpoint's facts: `Qwen/Qwen3.8-27B-FP8` at its pinned revision and
`vrfai/Qwen3.6-27B-FP8`, their numerics, parsers and template options; and
Flash-Next's, the e307 NVFP4 prune (ARC2's cells serve their own, P-EV) with
its per-token-head FP8 KV cache and its recurrent state in bf16 (ullm's row,
placed with the KV format: +24% generated on the four L4s), whose record is
upstream's `packages/ullm/flash_optimization.md`, with ARC2's four L4s, images
and B200 readings in its addendum, `flash_optimization_arc2.md`. For
Flash-Next also drafting with its MTP head, four tokens a step (fine-tuned on
ARC2 and ARC3 play, NVFP4 experts, a 16k-id draft vocabulary, GDN RecoverSSM),
and conversations kept live between turns: a cache hit every 80 tokens, a
decode tail and each conversation's thread named, which put the host tier in
park mode (upstream's `flash_optimization.md`, "Speculative decoding", and
`kv_offload.md`, "Park mode"). The training checkpoint enters here too, on the
ullm side, as a builder of its own.

## U-0V · task-free, environment only

The box: `tensor_parallel_size` (1 on `slurm-b200`, **4 on `kaggle-4xl4`** —
27B at FP8 does not fit one L4), offline once the engine is up on Kaggle,
which wheelhouse a submission installs, a kernel's compile parallelism
bounded to four jobs, **dense CUDA graph capture off on `kaggle-4xl4`**:
every rung is captured once per card under tensor parallelism, on cards
whose headroom capture memory already set, and **the driver's hardware work
queues at 32 on `kaggle-4xl4`** (`CUDA_DEVICE_MAX_CONNECTIONS`), for any model
there: a pipeline stage's token receive spins on a side stream, and at the
default eight queues a CUDA graph's branch can wait behind it (+14.5% on the
drafted Flash-Next cell). Pinned rather than probed, since the
launcher that builds the args usually has no GPU. **The renderer's workers,
4 on `slurm-b200` and 2 on `kaggle-4xl4`** (`renderer_num_workers`), process
every image of every request before it reaches the engine, with no image
cache to skip any (U-00), sized to the CPUs the box spares its tasks.

Kaggle has no network for the kernel at any point, so the wheelhouse carries
FlashInfer's prebuilt kernels (`flashinfer-cubin`, `flashinfer-jit-cache`,
pinned to the `flashinfer-python` version, since a drift misses and compiles)
with `FLASHINFER_NO_DOWNLOAD` beside them. Without cubins or nvcc vLLM
silently demotes the MoE backend; with nvcc and no cubins it builds them, a
`cicc` of several GiB per job, which is what the four-job bound is for.
`TORCHINDUCTOR_COMPILE_THREADS` is not ours: vLLM sets it to 1 at import.

## U-EV · task-free, both coordinates

What only the two together decide, memory above all. This repo's cells,
`(qwen3.8-27b, kaggle-4xl4)` and `(qwen3.6-27b, kaggle-4xl4)`, are stated in
ullm's pair block with their own figure: `gpu_memory_utilization` **0.85**,
not the RTX cells' 0.95, because the V2 model runner sizes the KV pool
without reserving for CUDA graph capture, and at 0.95 the pool takes 13.16
GiB per card and the engine dies partway through its 35 piecewise graphs.
The mount paths and dataset sources are the RTX cells', the same published
snapshots; Kaggle keeps models apart from datasets (`kaggle_model_sources`),
and the Flash-Next cells mount e307 and its head as models.
`(qwen3.8-flash-next, kaggle-4xl4)` serves two pipeline stages of two tensor
ranks split 27/21 (the second holds the lm head, the sampling and the
drafter), drafting four with the NVFP4 head from its Kaggle model, at 0.935
with FULL decode graphs alone (`FULL_DECODE_ONLY`: a step carrying a prefill
runs eagerly; with both graph kinds 0.92 ran out of memory in serving, and
the PIECEWISE graphs held about half the graph memory for little; a ladder
from 0.92 to 0.945 put the cell at 0.935, 2026-10-28), with the
sparse-attention indexer's prefill scoring buffer chunked at 64 MiB
(`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, 512 by default; upstream's
`flash_optimization_arc2.md`, "Memory in play on the four L4s"), with the
Flash-Next throughput stack's portable part (`FLASHNEXT_THROUGHPUT_PORTABLE_ENV`:
the stack's fused commit needs one pipeline stage and its CUDA images
capability 12.0; upstream's `flash_optimization_arc2.md`, "The throughput
stack on the four L4s"): the RTX Flash cell's mount, Marlin (the only
NVFP4 MoE backend the L4's capability admits; the MoE backend pin is a pair
setting, since a backend's support check is a capability family), the PLE
table on the host, and the engine's own prefetch (the RTX cell's bounded
reader drops shards the four loaders have yet to read), and its own CUDA
graph ladder (the request counts its traffic decodes, ~0.4 GiB a card under
the engine's). Above the 27B cells' 0.85,
measured on its own: its graphs and ~1 GiB a process outside torch fit what
0.935 leaves, one capture after the pool is sized (upstream's
`flash_optimization_arc2.md`, "Memory in play on the four L4s");
the 27B figure predates that and is owed a re-reading. A drafting cell names
where its drafter is mounted, and `build()` refuses one that does not:
`(qwen3.8-flash-next, slurm-b200)` names the fine-tuned head beside its
checkpoint, and the four L4s its dataset mount. ARC2 replaces both checkpoint
and head on its Flash-Next cells (P-EV).

## P-00 · ARC2, neither coordinate
*The P-00 block of `duck.presets.configure()`, and the class defaults of
`DuckAgent`, `RunConfig` and `LLMConfig`.*

What ARC2 asks of any engine anywhere.

| Setting | What / why |
|---|---|
| `DuckAgent.use_dihedral_and_order_permutation` (off) | Off: an experiment arm, not yet measured. The field comment describes what each take is shown. |
| `reply_cap_tokens` | Agent default: 24576. Each request stamps this cap onto its profile copy; the context budget reserves the same number, and the window moved with it so the prompt budget stays at the 48,640 the sweeps read. Sweep 7's reply-cap shape: the score rises 0.13 points per 1k over 12–32k, nearly all of it below ~22k, while tasks an hour fall throughout (`packages/duck/evidence.md`). `None` uncaps generation and reserves 512 tokens. Stored play profiles must leave `max_tokens` unset; a stated cap is rejected. |
| `compaction_strategy` = `summarize` | The live summary rather than pruning (sweep 6: +11.4 over prune, about 11 points fewer give-ups; `packages/duck/evidence.md`). Set here rather than as the class default because a summary needs a real reply reserve, which every column's cap gives it and an agent built uncapped does not have. |
| `token_native_context`, `exact_history` | Every column serves vLLM, which counts exactly and continues by the sampled ids. |
| `bound_chars_per_token` (1), `usage_window_tokens` (unset) | Accounting for a backend that cannot count (a hosted API teacher, `tools/teachers/`): the rendered request's characters charged per text token, a tokenizer's calibration in place of the provable bound of 1, and the billed total the window invariant holds responses to, the API's own window where the agent's is a student's. No column sets them, since every column counts by tokens. |
| the host KV tier on | Every column runs more games than its pool holds contexts. Its engine-log copy stays at the connector's default, off: writing it stalled the engine for the last two hours of a twelve-hour Kaggle run (ullm's `kv_offload.md`, defaults); the same events land in the job directory's trace. |
| `min_request_gap_s` = 0 (off), `min_request_gap_running_fraction` = 0.5 | Off by default: under the tier's park mode, which every engine that keeps conversations live runs, the tier itself holds a lookup until its conversation's copy lands, so a floor only slows the games. Set (a tier running its band wants 5 s), it is the connector's copy-in-flight window, applied only while at least half the slots are held: the gap protects a pool at its admission line, and once most games have finished it only slows the games left. Decided as a game's call starts and again before each of the duck's LLM calls, since an ARC2 game is one call. Upstream files both under the environment; no column overrides them on either side, so their home is here. |
| `start_gap_s` = 0.5 (a `RunConfig` default, which P-0V raises on Kaggle) | The floor between consecutive game starts. A run's first grants all spawn their child process together, and a couple of hundred spawning at once broke upstream's host; half a second apart is nothing against calls that run minutes. |
| `verify_backend` (off) | The backend matrix against the started engine before the first game, for a run that wants its engine proven before it plays; a configuration the engine refuses otherwise kills every game on its first call with the run exiting COMPLETED. It claims the image row exactly when the agent's `needs_images()` says its requests carry pictures (the duck: every `images` mode but `off`; a game in takes: any take's). |
| `take_handicap_tokens` = 150k | How far behind its spend a game ranks for each take before the one it is playing (`platypus.multi_take`): its second take, and within a `TieredTakeAgent` category, its later ones. Read by `least_tokens` alone, so it bites only where that policy runs — but the value is not the environment's, so its home is here. Chosen by simulation (`take_gap` in `arc2-experiments`, reported in its `results/2026-09-16_take_gap.html`) on the Kaggle column's figures, over the Flash-Next baseline (`arc2-eval` × 6 at 150k a game) with its 22 tasks that reach 150k replayed to 400k. At the twelve-hour deadline two takes gain 4.0 points over one at 150k; 100k scores within noise of it (+0.04 ± 0.02), a larger gap loses a little (200k and never, −0.08) and a smaller one more (50k or less, −0.25 to −0.29). Away from the deadline the loss is lopsided: too small a gap takes slots from first takes (no gap is 12 points behind one take at five hours), while too large a one only delays second takes. So the value errs high. |
| the analyzer sampling profile | Temperature 0.8 for every phase, `top_p` 1.0, `top_k` 194, no presence penalty. Sweep 9 read the play's reasoning flat to 0.9 and cheaper warmer, the other phases level; `top_p` 1.0 makes drafting and training on the duck's own play easier at no cost found (sweep 8, +0.15 against 0.889, p = 0.81; `packages/duck/evidence.md`, "Duck sweeps 8 and 9"). `top_k` is upstream's sweep-3 coordinate, the presence penalty off from sweep 5. |
| `auto_correct_tool_arguments` (on) | Whether tools may repair malformed argument names, JSON or escapes; independent of `malformed_retries`. Upstream's prompt and tool surface. |
| `compact_before_request_only` (off) | Defer retention to the next complete pre-call trim. |
| `require_exact_context` (off) | Require exact prompt measurements and enforce the compaction target. |
| `context_observer` (`None`) | Optional request-context observer; its contract is `duck.context_observer.ContextObserver`. |
| `perturb_retry_prompt` (on) | Perturb the prompt when retrying malformed output. |
| `preserve_failed_draws` (off) | Retain failed responses; requires zero malformed retries. See the [Duck retention contract](packages/duck/README.md). |
| `malformed_retries` (3) | Number of retries after rejected malformed replies, before the nudge (`malformed_nudges`): over sweeps 6–8 a draw after four refusals in a row never succeeded (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). |
| `classify_tool_calls` (off) | Classify tool-call failures for response accounting. |
| `response_observer` (`None`) | Optional response callback; its contract is on `DuckAgent.response_observer`. |
| `file_tools`, `expose_tool_*`, `add_files_and_repl_to_summary`, `prompts.thorough_summary` | The file tools over the kernel's working directory, the listing that closes a summary's rebuilt conversation, and the thorough summary request. All on, as upstream ships them: the file tools +1.27 in sweep 9 for 8.7k more effective tokens, a tie throughput-adjusted that the score breaks; the listing and the thorough request judgment calls, level or slightly ahead in sweeps 8 and 9 (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). `expose_tool_*` is off in both repos: neither tool was reached for. |
| `show_changed_answers` (on) | After a `python` call that changes `test_out`, each changed answer as an image beside its test input, in a user message of its own, and a system prompt line saying so. On, a judgment call: a tie throughput-adjusted in both sweeps, and the model uses the images as its last check (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). Needs the task images and is refused under `MANDATORY`, so an arm without either turns it off. |
| `audit_turn`, `prompts.audit_groundwork` (on) | The audit turn where the game would stop with its answer standing, and what it builds on earlier in the game. The turn adds 2.99 Kaggle points to one take and 2.01 to two on the eval run; the groundwork is a judgment call, +0.4 in sweep 9 and +0.6 in sweep 8, neither significant (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). The turn is refused under `MANDATORY`, so an arm under it turns the turn off. |
| `prompts.generalization_audit_line` (off) | The standing system-prompt audit. Off beside the turn: with it on the play does part of the audit's work for 5.6k more effective tokens and the audit gains 1.1 less, the score level (+0.71 throughput-adjusted off, p = 0.03; `packages/duck/evidence.md`, "Duck sweeps 8 and 9"). |
| `prompts.audit_confidence_line` (on) | The audit turn's last question: how confident the model is, from 0 to 100, that its answer is right, read back as `audit_confidence`. Asking costs nothing measurable on the audit's gain (+0.07, p = 0.86), and every sweep 9 arm but four jobs' asked it. Nothing in the harness reads the number yet. |
| `audit_handicap_tokens` (0; 10M on `MultiTakeDuck`'s second take) | Tokens the game asks the `least_tokens` ranking to put behind its spend from its audit's first compaction on, where it offers its slot with the prefix gone. At 10M the rest of the second take's audit waits for capacity no game that has not deferred wants; the take-gap simulation read it +0.07 at 12 h over auditing the first take only (2026-10-05). Inert without the audit turn. |
| `malformed_nudges` (1) | How many times a game is nudged, rather than crashed, when a play reply is refused on all `malformed_retries` re-samples for ending with no text and no tool call (or for degenerate repetition). The refused draws are dropped and the nudge says the reply was discarded; past the count the game crashes. One: sweep 9's games crashed 14 of 21,735 without it and 1 of 22,365 with it, at no score or token cost (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). A trainer that retains failed draws (`preserve_failed_draws`) sets it to 0. |
| `audit_profile`, `summary_profile` (`None`) | The sampling profiles of the game's other phases, by name in `llm_config.sampling_profiles`: the audit turn and every block after it, and every summary in either phase. A summary keeps the live request's seed and injection and takes the profile's temperature, `top_p`, `after_reasoning` and `extra_params`. Unset, both phases sample on `profile`. Unset: sweep 9 read the four phases' temperatures level, and every phase samples at the play's 0.8. |
| `intermediate_summary_prompt_lengths` (empty) | Summaries that leave the history whole: each time the full conversation, as the request is measured, passes one of these token counts, a turn of its own asks for a summary. The request has no size, no thorough mode and no candidate source. The summary stays in the history, and the next message says to go on; nothing is compacted. A compaction re-arms every length above what it leaves, runs alone where both fall due, and once an intermediate summary is in the history its request says the earlier summaries go too. Lengths passed together get one summary. Summarize arm only. Empty: sweep 9 read −1.72 on the score (p < 0.001) for 8k fewer effective tokens, a tie throughput-adjusted (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). |
| every other `DuckAgent` field | The task's own knobs — images, text grids, the check oracle, attempts, the window and target — have no upstream bucket at all. What ships for each is its default, and `packages/duck/evidence.md` says which of them has a reading behind it. |

## P-E0 · ARC2, engine only
*The P-E0 block of `duck.presets.configure()`.*

| Setting | What / why |
|---|---|
| `reasoning_effort` = `"xhigh"` | The swept level, and one only the 3.8 template renders; 3.6 keeps the class default. Flash-Next ships that template byte for byte and takes the same level. |
| `inject_text_*`, emptied for Flash-Next | Upstream's reading: its runs were played without a served opening, and the register the opening sets halves what this model reasons per action. The opening's +15.4 points here were measured on the 27B. |
| `compaction_forbidden_token_ids` = `(248058,)` | **This repo's row.** The `<tool_call>` id of the tokenizer all three checkpoints share, banned while the model writes a summary. |
| `attention_config.indexer_kv_dtype` = `"fp8"` for Flash-Next | The sparse-attention indexer's compressed keys in plain fp8 rather than bf16, on both cells: +5.9% generated and +7.1% pool on the four L4s' traffic replay, and no score cost in sweeps 8 and 9 (−0.10 in sweep 9, n.s.; `packages/duck/evidence.md`, "Duck sweeps 8 and 9"). ARC2's because the score risk is read on ARC2's tasks; the weights are untouched. The 27Bs have no such indexer. |
| `zero_recurrent`, `zero_recurrent_at` | Needs a sampling parameter only the fork declares. Supported, off. |

## P-0V · ARC2, environment only
*The P-0V block of `duck.presets.configure()`.*

How a run is scheduled where it runs.

| Setting | What / why |
|---|---|
| `slot_policy` | `queue_order` on SLURM, `least_tokens` on Kaggle: ranking pays only where there are more games than slots and a deadline. Upstream's `least_level_tokens` and its `scheduler_level_weighting` are not carried — every task here is one level, so the dial cannot do anything. |
| `slot_hold_s` | `None` on every column (2026-10-16): a slot changes hands at free points alone, compactions, take boundaries and finishes (`RunConfig.slot_hold_s` says why). |
| the per-game wall net | `GAME_RUNTIME_CAP_S` = **8 h** (upstream 6 h; 2026-10-06: clear of the wall clock a game needs to reach its token budget at the SLURM column's concurrency and 500k cap, a few tens of tokens a second a game, and inside a `dev` job's twelve hours), off on Kaggle where the submission's own clock is the budget. A game played in takes (`platypus.multi_take`) carries it per take instead — `max_runtime_s_per_take`, moved there from this net at construction — and has no game-level net: 8 h a take on SLURM, none on Kaggle. |
| `max_tokens_per_game` | 65k default, **70k on Kaggle as a backstop** where upstream takes the cap off under the ranking: the 27B's twelve-hour runs here fired it 18 and 29 times of 240 (`packages/duck/evidence.md`). Flash-Next's cells (P-EV) state their own: 500k on the B200 and none on Kaggle. The B200's is past what most games spend and short of the million the sweeps let a game run to: on the 1,200-game eval run at this column's settings 8 games went past it, their tokens past it 1.4% of the run's (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). The wall net above leaves a long game the hours it takes to reach it. Kaggle keeps the score and the ranking handles long games; playing in takes brings no cap back, since the ranking's handicaps between takes (`take_handicap_tokens`, and `TieredTakeAgent`'s categories) are what balance them. |
| `shuffle_games` | On for a submission, so the same notebook submitted repeatedly varies its play order. |
| `play_priority` (empty) | Which games start first, by `game_id`, highest first, applied stably after any shuffle. A sweep job sets it to each task's length so its long tasks do not start late and leave a tail. No column sets it. |
| the Kaggle wheelhouse, `KAGGLE_WHEELS` | `arc2-site-0290` in place of ullm's: the same wheels, with the record of installing them on an ARC2 kernel. ullm's records its install after the ARC3 competition's `arc-agi` install, which an ARC2 kernel never runs (its competition mount is data only), so an ARC2 kernel refuses that record and installs with pip (`vllm_fork.md`, "The install is replayed, not run"). |
| `start_gap_s` 10 s on Kaggle | A cohort started together builds its contexts together; starting games apart flattens the KV peak of that launch wave. The B200 cells keep 0.5 s: ARC2's short tasks spread their compactions themselves. |
| `backend_inactivity_timeout_s`, `accept_malformed_output`, `replay_on_link_loss` | Transport policy: how long a silent request may park a game, and whether a trainer keeps a malformed draw. |
| `engine_stall_timeout_s` = 600 | The run's own watchdog over the engine (`platypus.watchdog`): once the engine has generated nothing for this long while it still reports requests, or has died, the run ends the way the soft deadline does, the reason on the benchmark. How a run ends where it runs, so its home is here; `None` turns it off, as Kaggle does (2026-11-01, for the final submission): a stall it ended would forfeit the rest of the clock had the engine come back, and a dead engine adds no score whether the run stops at once or at its soft deadline. |
| the served-history duck's grid presentation | An image is continued by tokens by an engine and not by a chat server, so the `server` columns build text-only ducks (`images="off"`) — the column's choice, not a rule the agent enforces (`roadmap.md`). **Open**: whether the training duck also wants `text_grids`. |
| `heartbeat_*`, `status_interval_s`, `engine_status_interval_s`, `report_memory`, `label` | Observability. No heartbeat here. |
| `loop_hold_s`, `loop_hold_every_s` | Off. A test facility: the host's event loop held on purpose, to see what a host stall does to a run, the KV tier's parking above all (upstream's `packages/ullm/kv_offload.md`, "Parking on the GPU"). |
| serving topology and its client wiring | In-process engine, in-process HTTP server, or someone else's server. Flash-Next is not served over the in-process HTTP server: its engine expects the client to name each conversation's thread for the host tier's park mode, that client names none, and the build refuses the pair (ullm's `roadmap.md`, "Park mode over the server topology"). |
| `process_mode`, `subprocess_memory_limit_bytes`, `on_agent_error` | Process hosting. Correctly here, though constant in practice. |
| `rng_seed`, `rng_seed_per_game`, `shuffle_seed` | Reproducibility. No column pins a seed. |

## P-EV · ARC2, both coordinates
*The P-EV block of `duck.presets.configure()`.*

| Setting | What / why |
|---|---|
| `max_num_batched_tokens`, in every served cell | It follows the engine's recurrent block on the pair, so each cell states its own. The budget is how many blocks of prefill advance per step (in `align` mode a chunk ends on a block boundary): run high, since this task trades latency for prefill throughput, at activation out of the pool's headroom. 8192 for the 27Bs (five 1,568-token blocks) and for Flash-Next on `slurm-b200` (several blocks and the decodes); 512 for Flash-Next on `kaggle-4xl4` (2026-10-28), under the drafted cell's 1,840-token block: a prefill advances in chunks of the budget, which the fork's scheduler admits beside the decodes. The step's smaller activation workspace goes to the pool (5.96 GiB a card against 5.37 at 3,300), and the drafted replay at 512 (run Z, FP8 KV) generated 2.3–2.7% more than those at 3,300 past what that pool alone gives (arc2-experiments `arm_budget/box.py`, the box arm sweep 2 was read on). Supplied engines own their prefill budget. No cell here sets `prefix_match_unit`: ullm's engine layer keys every Flash-Next cell every 80 tokens, the four L4s included, for the decode tails park mode needs, although on the four L4s under the bf16 state a unit (64) cost 3.9% on the traffic replay, its prompt-end state copies taking pool from the running set; the four L4s under park mode are unmeasured. |
| `transient_kv_pool`, Flash-Next on `kaggle-4xl4` | The compressor rings in a block pool of their own (2026-10-28): a ring holds a few kilobytes but takes a whole shared block for every running request, about a twentieth of a request's pool at the cell's reference context (35.9 requests in the box's pool against 34.1). Four rings a sequence in each ring group, and the V2 model runner, which every Flash-Next cell runs. The B200 cell keeps its rings in the shared pool: unmeasured there. |
| `max_concurrent` | 80 on `slurm-b200`, 40 on `kaggle-4xl4` for the 27Bs; 180 on `slurm-b200` and 64 on `kaggle-4xl4` for Flash-Next, over two stages of two tensor ranks there. Sized to saturate the engine, and above what the pool holds so the host tier is in play. The 27Bs' are upstream's, unmeasured here (`roadmap.md`); the Flash four-L4 one is 64 (2026-10-28), set with the cell's utilization at 0.935: the drafted cell's twelve-hour eval runs (2026-10-04 and 06) read the pool at 30-37 requests, since the drafter keeps its own KV for every token, where 50 kept a standing queue of about 8-10 at the pool-bound p10 and parked ten fewer threads in the host tier; the pool has grown since (512 tokens a step, the rings' own pool, FULL decode graphs alone at a higher utilization), and on the 230-game replay at 50 the running set reached the slots with a few waiting (run km2), so more slots give the pool its queue back. The B200's is 180 for now (2026-10-16): the value is being tuned separately, in sweep work. At 230 under park mode at 150k tokens a game, the long run of 2026-09-28 (arc2-eval x 10, 3 h 39 min) waited for capacity in 69% of its steady minutes; at that many sequences the engine ran at utilization 0.95, since 0.97 ran out of memory capturing CUDA graphs. With the engine layer capturing once (upstream's `packages/ullm/flash_optimization.md`, "Competitor tricks"), the same footprint is 0.95's boot-line equivalent at 230 sequences, 0.9174, which 180's smaller graphs leave more room under. `notebooks/demo.ipynb` asks 32 GiB more with it. |
| `max_num_seqs` | Derived from it, so it arrives here by following. |
| Flash-Next's checkpoint and drafter | ARC2's REAP-307 (`e307_arc2`, `presets.E307_ARC2_*`) in place of ullm's e307, on both cells: our prune of the unpruned model on its own ARC-2 play, in e307's format, so only the paths move. +1.24 over e307 in sweep 9 (p = 0.006) at the same tokens (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"). ARC2's own, since ARC3 reads its cuts separately. On the B200 from shared storage. On Kaggle both are Kaggle models (`presets.E307_ARC2_KAGGLE`, `ARC23_FP8_DRAFTER_KAGGLE`): the checkpoint whole, under the name our ARC2 upload is to take beside his ARC3 one (2026-10-28; until it is up a kernel cannot mount it), and the drafter from the account's own model of the head, byte for byte. The drafter, in place of ullm's v1 head and of REAP-307's own (2026-10-29), is our combined ARC2/ARC3 MTP head trained with its KV in FP8, the format ARC2 serves its drafter in (`presets.ARC23_FP8_DRAFTER`): on his replay check it led the INT4-trained head on ARC2 in both drafter KV formats and drew level with the v1 head (`arc_3_results/shared/drafter-final-20261029`); against REAP-307's own head it reads level in Kaggle real play. |
| the CPU tier's size | On Kaggle, claimed against the cgroup the harness shares: 60 GiB for the 27Bs, 45 GiB for Flash-Next, whose pinned PLE table (~48 GB) shares that host. The drafted cell's `kv_park.log` at 60 GiB (1,953 slots) read the deepest key a resume needed at p99 ~1,150 slots and parked threads at most 861, with no parked thread evicted, while the unreclaimable memory sat at 95-99% of the 173 GiB cgroup through the pool-bound hours; 45 GiB keeps headroom over that depth and gives the host 15 GiB back. ullm's shipped size on SLURM, and 96 GiB there for Flash-Next, whose concurrency leaves a larger queue outside the pool. |

---

## Where a check goes

The same split decides validation. A rule about **one field** — an unknown
mode, a negative margin — is checked at construction *and* at
`prepare_for_backend`, since post-construction assignment is how settings are
normally applied. A rule **spanning** the agent, its config, the factory or the
engine actually installed is checked only at `prepare_for_backend`: at
construction it would see values that had not arrived yet, and after
construction nothing would run it. `build()` calls that seam once itself, so an
interactive caller still fails immediately.

A **field derived from another field** follows the one-field rule: written at
construction and again at `prepare_for_backend`, idempotently. `__post_init__`
runs once, on the class defaults, before any layer or override has been
assigned — so a derivation that lives there alone is taken from the default
and never again.

## Judgement calls worth knowing

- **`watermark` is U-00.** A pool kept ~90% full is a general good thing, not
  a property of our workload.
- **Sampling stays P-00**, not per-engine, on the reading that one model is
  the default model rather than one of several kept configured in parallel.
- **`max_model_len` and `max_num_seqs` are derived**, so they live in
  platypus even though they are engine arguments.
- **`gpu_memory_utilization` is set per cell and left conservative.** The
  Kaggle figure is coupled to the V2 model runner; the runner is not a knob
  here for that reason.
- **The Anthropic duck is outside the tables.** A hosted model is a
  checkpoint nobody deploys anywhere, so engine and environment collapse; it
  is constructed directly (`tests/test_duck_reference.py`), as upstream
  constructs its API ducks.
- **A field no deployment sets is still a knob** — for a notebook or a
  sweep. What ships for it is its default, and that default is placed like
  any other setting.
- **`KIMI_STUDENT` is not a layer.** It holds P-00 values a caller asks for
  by name at `DuckAgent.build`, over whichever coordinates it names: playing
  as the Kimi teacher traces were played is a choice about the agent, not a
  fact of a checkpoint or a box, since the same checkpoint plays either way.
  A student checkpoint that ships as an engine of its own would state them in
  its P-E0 layer.
