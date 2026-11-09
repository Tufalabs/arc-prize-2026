# Model checklist: getting an LLM into shape on our stack

*Register: living.*

This is what to do, and what to check, before a model is doing its best for
ARC2 and ARC3, on the cluster and on Kaggle. It applies to any model; a new
checkpoint of the same family runs through it too. The procedure lives here.
Each model's readings go into that model's serving record (for Flash-Next,
`flash_optimization.md` and its ARC2 addendum, `flash_optimization_arc2.md`),
and the values chosen go into `settings.md`.

## The cells

A reading counts for the cell it was taken on:

| harness | cluster | Kaggle |
|---|---|---|
| ARC3 | `slurm-b200` | `kaggle-rtx` (one RTX PRO 6000) |
| ARC2 | `slurm-b200` | `kaggle-4xl4` (four L4s) |

- **The B200 comes first.** A configuration reaches Kaggle only after it has
  passed there, so Kaggle's quota is spent on configurations that already work.
- **Throughput has one instrument.** Throughput on any cell is read on the
  traffic replay (`traffic_benchmark.md`), in an interactive session on
  Kaggle.
- **Play needs a full-length run.** Checks that need real play (capacity,
  uncached tokens, surviving the deadline) need a run of submission length,
  on every cell.

How to read a run is `flash_optimization.md`, "Reading a run", and the
ARC2 addendum's "Reading an ARC2 run":
- bucket by the running count;
- scores say "not worse", never "better";
- TAAF's token counts lead.

## Actions, in order

Each step depends on the ones before it: the agent's parameters move the
traffic, the data depends on both, and the drafter and the engine's numbers
depend on the data and on the pool.

1. **Bring it up on every cell: it fits and it survives.**
   - Put the started engine through ullm's backend matrix
     (`ullm.backend_matrix`), switched on for a run by the agent's
     `LLMConfig.verify_backend`. Before any game it checks two-turn memory,
     a tool call, the per-turn cap, a prefix-cache hit and an image. It
     catches a configuration the engine cannot serve, which otherwise kills
     every game on its first turn while the run still exits COMPLETED.
   - Find the memory-utilization ceiling.
   - Allocator and kernel buffer bounds (expandable segments, the indexer's
     chunking).
   - Kernels available offline on Kaggle.
   - Graph capture.
   - Host memory under the job's cap.
   - A run of submission length that ends by its clock, not by a crash.

   The Flash-Next record is `flash_optimization.md`, "Bring-up on the RTX
   card", and the addendum's "Bring-up on the four L4s".
2. **Compare the public variants**: quantizations and conversions of the same
   weights, from each provider. For each, read:
   - the score, not worse on the B200 over full-length runs;
   - the pool at the cell's memory utilization;
   - throughput on the replay.

   Example: `flash_optimization.md`, "Checkpoint, active experts and MoE
   backend".
3. **Prune experts, if the model is a mixture of experts.**
   - Take the routing histogram (`EngineFactory.expert_log_dir`) on each
     harness's traffic.
   - Cut a ladder of keep ratios and score each as in step 2.
   - Prune by column, never by active experts: fewer experts per token cost
     score badly.
4. **Set the pool levers on each cell.** These change how much fits:
   - KV cache format;
   - recurrent-state dtype;
   - `prefix_match_unit`;
   - MoE backend.

   Read each on the replay. A clear saving of pool ships even if a reading
   taken while the pool was not binding looks slightly negative.
5. **Re-sweep the agent's design parameters on the chosen checkpoint.** The
   agent's defaults were tuned on another model: the prompt budget, the
   compaction settings, the per-level token cap, the sampling. They also
   shape the traffic (context lengths, reply lengths, how often histories are
   rewritten), so they must settle before any data is recorded (the sweeper,
   `packages/sweeper/README.md`). Conclusions need the throughput-corrected
   reading, not the raw score.
6. **Record the traffic and build the fine-tuning set** from the swept agent
   on the final checkpoint, one recording per harness.
   - The traffic recording is ids-only traces (`traffic_benchmark.md`, "What
     a replay is").
   - The drafter's set is built from the target's own states over that
     traffic.
   - The two may come from one run later; nothing needs that yet.
7. **Fine-tune the drafter** on that set (the `drafter-sft` branch,
   `packages/drafter/README.md`). Then quantize it and read its acceptance
   per draft position in real play.
8. **Tune what depends on the pool and the drafter, last.**
   - Concurrency (`max_concurrent`): set so that the engine always has
     requests waiting for capacity (check 4).
   - The drafter's k (check 6).
   - The host tier's size, from its reuse-depth curve (`kv_offload.md`,
     "Instrumentation"; check 5).

## Checks

Run every check on every cell. Rerun them after any change to the engine,
the harness or the agent. Most are judgment calls against the evidence, not
thresholds.

1. **Quality is not worse.**
   - The score on the B200 over full-length runs, at matched token cost,
     against the configuration it replaces. Pruning, quantization, the KV
     format, active experts and the drafter's sampling have each moved it.
   - The Kaggle submission reads not worse than the last one.
2. **The model speaks our protocol.**
   - The backend matrix passes (`LLMConfig.verify_backend`, action 1).
   - The tool-call and reasoning parsers and the chat template parse what
     the model writes.
   - Exact history holds: rewrites are declared, and nothing drifts.
   - Malformed and degenerate replies stay rare.
   - TAAF's token counts reconcile with the host's.
3. **Profiling on the traffic replay shows no clear addressable
   bottleneck.**
   - Use `EngineFactory.profile` with `tools/engine_profile_read.py`.
   - Look at GPU busy time, gaps on the host, and the share of mixed steps.
   - Check each kernel family against its cost model: expert reads against
     the bytes the step touches, the recurrent layers scaling with requests.
   - What counts as addressable is judgment. `flash_optimization.md`,
     "Where decode time goes", is the worked example.
4. **In play, almost all engine lines are bound by capacity, but no deeper
   than necessary.**
   - Requests wait for capacity (`waiting=N(capacity:N)`) in nearly every
     minute of play.
   - It must not go deeper:
     - no preemptions;
     - admissions held back by the step budget (`deferred`) near zero;
     - the host tier restores what it is asked for;
     - uncached tokens (check 7) do not grow with the queue.

   Over-subscription shows up as recompute before it shows up anywhere else.
5. **The host tier is deep enough.**
   - Every `tally` event of the tier's trace (`kv_offload_trace.jsonl` in
     the job directory) carries `depth`: for each block asked back, how many
     slots the tier needed to still hold it (`p50`, `p90`, `p99`, `max`, and
     `run_max` over the whole run).
   - The asked-depth percentiles should sit well inside the tier's slots
     (the trace's `start` event).
   - `evicted_current`, evictions of blocks a context was still current on,
     should stay small.
   - When p99 closes on the capacity, either the tier is too small or dead
     content is crowding it. The uncached split (check 7) tells which.
   - Under park mode, `kv_park.log` (`kv_offload.md`, "Park mode") reads it
     over the whole run, where a tally covers only its interval. Each `LOAD`
     needs its deepest key's depth, and those should stay well inside the
     slots. No `EVICTED-PARKED` should appear. In `TIER-ORDER`, the `other`
     share of what is newer than the deepest parked key is the dead content.
6. **The drafter's k is optimal.** Compare k values on the replay while the
   pool binds, or model it: acceptance by position against what a verify
   step costs at that width.
7. **Almost all uncached tokens are genuinely new content.**
   - For each continuation, the new content is the prompt minus the previous
     prompt and reply. Anything uncached beyond that is recompute: the
     previous reply re-prefilled, or earlier history lost.
   - Split it from the per-call spans (`llm/response`: cached and uncached
     input tokens).
   - `platypus.llm.check_continuation_prefill` prints a `[recompute]` line
     for every continuation more than `RECOMPUTE_WARN_TOKENS` over. When every
     live context fits, it should be silent.
   - With a host tier, `tools/victim_offload_decompose.py` splits the
     uncached tokens into what a tier can and cannot save.
8. **It runs the distance on Kaggle.**
   - A submission-length run ends by its clock.
   - No engine death, allocator failure or host-memory kill.
   - Capture the session log while it runs (the addendum, "Reading an ARC2
     run").
