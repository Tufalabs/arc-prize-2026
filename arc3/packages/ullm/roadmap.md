# ullm roadmap

*Register: living.*

Fork work still open. (The full fork enumeration this distills was
`requirements/ullm_work.md`, now deleted — git history has it; the
completed items are simply the code.)

- **Injected traces at the consumer.** A trace with
  `extra.response_injected` is faithful in ids only; the loader on the
  training side has to refuse it for any text-retokenising method. That
  check lives in ARC3-Training, not here — until it exists the marker is
  recorded but unenforced. This is not a corner: the duck injects on every
  call by default, so every traced duck run is a marked trace. Trace
  collection is off in every preset, so nothing produces them silently, but
  the next `traces=True` run yields traces a text-retokenising SFT cannot
  use, and whoever asks for traces needs to know.
- **Typed boundaries on the served-history chain.** A fresh render has
  more than one legitimate source — the agent rewrote history
  (`history_epoch`), a rollback removed a served answer (the manager
  declares it today by bumping the same epoch), a backend has no faithful
  extension (a budget-stopped answer the template re-serialised, an image
  in the delta) — and the design admits one: the epoch. Everything else is
  an error under `exact_history` or a silent fresh render without it. The
  shape that removes the corner cases: a boundary recorded on the step by
  whoever caused it, with its kind and reason, so `exact_history` means
  "no *undeclared* break" and the collector enforces nesting except across
  declared boundaries (ARC3-Training's `segment_start`, made first-class
  here). Its first commit moves the malformed-retry perturbation off the
  conversation (a per-attempt request property applied at render time),
  which also retires the rollback-declares-a-rewrite rule.
- **Continuation gaps** (`continuation.py`). Three prompts still render
  afresh on every call: an image in the delta over the server topology
  (`/v1/completions` has no id+image route; the in-process engine
  continues it), `VLLMHarmony`
  (harmony renders to ids natively, so a witness is a small separate
  delta function), and `send_back_reasoning != "all"` (by definition a
  re-render). Each is a fresh render, never drift; `exact_history` refuses
  them at the first call rather than passing them silently.
- **Image processor refactor.** Use vLLM's registered processor for image
  preparation while preserving exact history and serving/training parity.
  See the [proposal](image_refactor.md) for the design, implementation
  stages, and validation TODOs.
- **Turn-runner inversion, manager half.** `Backend.respond()` is done;
  still deferred: the conversation as a dumb picklable value the agent
  owns, per-use runners constructed over `(backend, conversation,
  tools, profile)`, and deciding what survives of
  `Processor`/`ProcessorMulti`. Redesign tool lifecycle at the same
  moment — `clear_state_on`'s semantics are inconsistent across tool
  subclasses (the TODO at `tool.py`), and per-use runners may not need
  it at all.
- **Public API freeze line** (U1): populate `ullm/__init__` to name
  exactly what agents may import; everything else explicitly
  churn-allowed.
- **Token-based tool-output truncation.** A tool result costs tokens in
  the context window, but truncation happens in characters:
  `PythonTool.max_output_chars` at this layer, and the duck — whose
  knob speaks tokens (`tool_output_tokens`) — enforces it as
  `tokens * 4` characters, an estimate that badly undercounts
  digit-dense output like boards (~1-2 chars/token). With
  `Backend.token_count` now giving exact counts, the knob can mean what
  it says: find the (head+tail) cut point in token space, client-side
  or via a counting callback so the tool itself stays token-ignorant
  (U9). Model-visible truncation points move, so the duck side rides
  the reference-regeneration ritual.
- **A killed kernel's children** (`tools/python.py`). Every kill path
  (cell timeout, corrupted protocol, `reset`) kills the kernel process
  alone, so a process its cells started outlives it. On the timeout path
  that hangs the game: after the SIGKILL, `_execute` awaits the pending
  `readline`, and a surviving child holding the kernel's pipe means it
  never returns. `os.system` reaches that path whenever its cell times
  out, as C's `system()` ignores SIGINT while it waits. Seen in seal sweep
  4 (rs15211, 2026-10-17): a model ran its compiled simulation through
  `os.system`, the cell timed out, and the game hung 4.5 hours, its job's
  four GPUs held, until the orphan was killed by hand; the other paths
  leave only an orphan using a CPU until the job ends. Two narrow fixes
  that keep the kernel's spawn as it is: kill the kernel's descendants
  (`/proc/<pid>/task/*/children`) before the kernel, on every kill path;
  and bound the post-kill wait for the pipe, abandoning the reader thread
  and its fds to whatever still holds it, so a child that escaped cannot
  hang the game. A new session for the kernel (`start_new_session`) would
  do the first, but changes how every kernel runs, not only a killed one.
  Not now (2026-10-17).
- **Retry seeds from the caller's RNG stream** (U8). Malformed-retry
  seed bumps are still manager-internal. The invisible-newline
  perturbation is kept on by default (the flag stays for callers that
  need byte stability): it was introduced for a gpt-oss NaN-kernel
  fault, and sweep data is *suggestive but confounded* that it also
  recovers empty single-token responses on Qwen — only one sweep ran
  without it, so commit and time are confounded with the treatment (see
  U11 / `packages/duck/evidence.md`). The controlled experiment that would
  settle it is a same-seed shadow retry of each naturally malformed
  prompt, with and without the suffix. What is still open on the design
  is *where* it writes — a perturbation confined to a fresh continuation
  element would keep the retained-byte invariant (and so the duck's exact
  reconciliation, which today skips a recovered retry rather than
  counting a prompt it never sent) and the recovery both.
- **Public export of `_bypass_debugger_subprocess_patch`** — a compute
  facility that spawns subprocesses wants it; today it is private to
  `tools/python.py`.
- **`ullm.kaggle.KAGGLE_PILLOW_REQUIREMENT` has no in-repo consumer.**
  Kept because ullm is consumed outside this workspace; retire it there
  if nothing else turns out to read it.
- **Backend error-translation dedup** — the context-length →
  `PromptTooLongError` string matching is re-implemented in four
  backends; one helper with per-backend match lists.

- **Proxy delta protocol for shipped conversations.** Every count and
  request pickles the whole conversation in the game subprocess and
  re-hydrates it on the host loop — O(context), GIL-bound at both ends,
  several times per duck move. `_run_count` memoizes byte-identical
  count payloads, which covers repeats but not each move's first ship;
  shipping only appended elements (host keeps per-game conversation
  state, invalidated on eviction/prune/reconnect) would fix both ends.
- **Host phase propagation over the proxy protocol.** The proxy
  client's status labels claim only its own serialization; the host's
  rendering/parsing for a request lands inside the client's "waiting
  for LLM response" scope, since the host backend has no per-game
  status hook. STATUS frames per req_id (host publishes, client maps
  into its scope) would let the status line separate host CPU from
  engine time in the solver's real topology; needs a proxy-topology
  test.
- **Counting's `_prepare_input` still renders on the event loop.**
  `_token_count` offloads only the tokenizer/processor work; the
  conversation → template render it needs first runs on the loop and
  grows with context. It shares code with the request path, so moving
  it off-loop means making the render loop-state-free first.

- **`FileTool.linked_tool` is an `Any` reaching a private attribute**
  (`linked_tool._tmpdir`), and `linked_file_tools` returns
  `tuple[Any, Any, Any]`, so a caller's attribute assignments are unchecked.
  The only consumers are ullm's tests: either promote a small protocol
  with a public `working_directory` or retire the helper.

- **The proxy watchdog covers generation only.** `_run_request` bounds
  each step with `default_inactivity_timeout_s`; `_run_count` has no
  equivalent, so a wrapped backend that never answers a COUNT parks the
  caller with nothing to surface it. It has not bitten because a count
  is local tokenizer work on both trace-capable backends rather than an
  engine request, so the drop mode the watchdog exists for cannot reach
  it — the gap is that this is a property of today's backends, not of
  the transport.

- **Answer sampling before it ships** (`after_reasoning`, `design.md`).
  No deployment sets it. A split turn goes through the engine's waiting
  queue twice, which a capacity-bound Kaggle engine would feel, so if the
  split ships to Kaggle it moves into the vLLM fork's sampler as one
  request (2026-09-30). Calls after a split turn miss the cache
  more: in the B200 check (answer at 0.3, 2026-09-30) `[recompute]` printed
  on 2.6% of calls against 0.08–0.37% in three B200 runs without the split,
  each re-prefill under one 1,840-token block and mostly after short
  replies. Likely, not proven: the answer request resumes from the
  reasoning's decode tail, context threads mark that state dead when the
  answer frees, and an answer too short to cross an 80-token unit leaves
  none of its own. A traced call is refused, since a trace step records one
  sampling for the whole reply. And a caller whose profile also serves calls
  rendered without thinking samples those wholly at the answer's settings:
  the duck's summaries copy the play profile.

## The KV tier and the engine fork

- **Parking on the GPU, measured** (`gpu_park`; on by default wherever park
  mode is, ahead of this run: 2026-10-18). Owed: a run that measures
  it against park mode alone, by parked-thread evictions and whole
  re-prefills in stalls, preemptions and the spill rate. Until then park
  mode without it is covered by `tests/test_victim_offload.py` alone: no GPU
  column turns it off. What it answers and how it works: `kv_offload.md`,
  "Park mode" (its limits) and "Parking on the GPU".
- ~~**Retries that keep the cache.**~~ Accepted as it is 
  (2026-09-29): no work planned. A malformed-output retry perturbs the last
  tool result and restores it afterwards (`perturb_retry_prompt`). That
  changes the prompt before the thread's tail and, for a reply longer than a
  block, before its last checkpoint too, so the thread is re-prefilled in
  full. With a shorter reply the checkpoint would still serve, but the
  duck's thread close discards it. The cost is about 0.5% of a 9-hour Kaggle
  run's work (five retries, three full re-prefills). Turning the
  perturbation off (a retry would vary only its seed) and limiting the close
  to compactions were the two levers.
- **Park mode over the server topology** (2026-09-30: refused for
  now). `deployment.build` refuses `serving="server"` for an engine that
  expects its client to name each conversation's thread (Flash-Next).
  `OpenAIServer` names none and sends no closes, so park mode would skip
  every finish, and with no band the tier would hold almost nothing, said
  only in the park log. Serving it there takes three things. `OpenAIServer`
  puts `prefix_cache_context` and `prefix_cache_epoch` into `vllm_xargs`
  (the fork's completions and chat requests already copy those into
  `extra_args`). The fork's API server gets a close route beside the dev
  `reset_prefix_cache` one, and `OpenAIServer.close_context` calls it. Then
  the refusal goes, and a B200 Flash-Next server column comes back to
  `DUCK_ENGINE_COLUMNS`.
- **What the fill does not cover.** A divergent hit is filled from the
  checkpoint the finish-time store put in the tier, and it stores the
  newest two. About two fifths of the asks on a pinned-pool hour find
  nothing, and the counters do not say whether those resumed further back
  than two checkpoints or were stored and evicted before the ask -- which
  want opposite fixes, a larger `want` or nothing at all. Classifying the
  refusal against the audit's own three fates (never stored / evicted /
  dropped) is a handful of lines and settles it.
- **Stop deriving the KV cache size by hand.** 0.29 profiles CUDA-graph
  memory into the sizing instead of leaving it to be absorbed after it,
  which is why the same `gpu_memory_utilization` yields fewer blocks than
  on 0.27.1. Two things fall out of that, neither taken yet. The worker
  prints a full breakdown once profiling is done -- consumed, peak
  activation, graph memory, and the two ceilings a KV reservation could
  take, one against the card and one against what the utilisation asked
  for -- which is the number to read instead of raising utilisation until
  something dies. And `VLLM_ENABLE_STARTUP_PLAN=1` persists a profiled
  size under the cache root and skips profiling on a later boot whose
  fingerprint matches, the fingerprint covering the vLLM version, the
  config, the device and the torch/CUDA build, with an apply-time gate
  that re-profiles when free memory is below the recorded baseline. It
  cannot help Kaggle, where every kernel is a fresh container, and it
  invalidates on every fork commit because the version string carries the
  commit count -- so its value is repeated local starts on a settled
  build, not the runs that matter. `kv_cache_memory_bytes` bypasses
  utilisation entirely; it is only valid while free memory on that host
  holds, so it is the less robust of the two for Kaggle.
- **Per-token-head FP8 KV for the 27B.** Its cells set `fp8_e4m3`, vLLM's
  per-tensor mode, whose one scale per layer comes from the checkpoint; ours
  carries none, so every layer stores keys and values at scale 1.0 and
  anything past e4m3's range clips. `fp8_per_token_head` computes a scale per
  token and head at write time and needs no calibration; it costs 8 bytes per
  token-head of inline scales. Flash-Next's sparse-attention cache ships in
  that format (`flash_optimization.md`, FP8 KV cache); the 27B's
  FlashAttention path would need vLLM's Triton attention backend,
  which is the one that reads it. Measure by the teacher-forced logprob
  comparison, not by scores.
- **The pinned-pool run as a tool.** Anything that must exercise the
  admission reserve or the tier on a B200 needs Kaggle's pool, not the
  card's: `num_gpu_blocks_override=1240`, 40 games, a 64 GiB tier, the
  presets' budget and reserve. That launcher lives outside the repo
  today; it belongs beside the notebook, or the notebook takes the
  override.
- **Staggered game starts.** Games that start together compact
  together, and each cohort of cold prefills halves generation for a
  few minutes — the floor of every run, and the reason Kaggle
  concurrency is held at 40 for now. A start stagger on `RunConfig`
  (one grant every N seconds until the slots fill) is harness-only and
  model-invisible; raise Kaggle concurrency after it.
- **A dead-context signal to the tier.** Compaction and game over leave
  contexts the tier keeps by age alone; a signal from the agent side
  lets them go first (kv_offload.md, limits).
- **Hold a spilled block until its copy lands.** Copies the engine's
  fence vouched for have differed from what was read, so every copy is
  hash-checked and the dropped ones recompute; the engine-side hold is
  the fix that removes the race rather than detecting it.
- **Reach-ins to hooks.** The fill is the fork's one explicit connector
  hook; the free queue's list, the job tables and the manager's policy
  are still read as private state, which is what a line change breaks
  (0.29 moved all of them).
- **The victim walk's bound on a small-page model.** The band walks the
  free queue each step, bounded by the free-block count the reserve
  holds — ~120 blocks on the hybrid model's 1,568-token pages, but tens
  of thousands on a model with 16-token pages. A cap on blocks examined
  is the wrong bound: an untouched pool keeps its never-allocated,
  unhashed blocks at the head, so such a cap starves the case it exists
  for. Bound by blocks *skipped*, or not at all.
- **`eviction_policy="arc"` is available and untested.** A re-prefill
  storm is a scan, which is LRU's worst case.
- **The tier's resident cost is superlinear in `cpu_bytes_to_use`**:
  64 GiB measured 68.8 GB resident, 96 GiB 132.8 GB. Unexplained, and it
  bounds how large the CPU half of the stack can be asked to go; suspect
  a per-KV-cache-group rather than total budget on a hybrid model.
- **Prefill that the engine's counters cannot see.** Prompt and
  generation counters credit a prefill only on its first pass, so a
  preempted request's replay and a divergent recompute read as
  near-zero prompt throughput while they drag every step to chunk
  length; the trace's per-step samples show it, the engine line does
  not. A per-step prompt-token counter on the engine line, or an
  upstream fix to `prefill_stats`.

## The traffic benchmark

`traffic_benchmark.md` is the instrument; what it still owes:

- **The settled rule.** The workload cutter places the snapshot where 90%
  of lanes are past their first handover, which a recording with long
  first sequences never satisfies at a usable horizon (ARC3's cut is
  placed by hand). It wants a measure that survives long sequences --
  dispersion of context length, say -- and the option to drop the lanes
  that cannot hold the horizon, since 120 are recorded and 60-80 replayed.
- **Between-run noise.** Every reading is one run; the within-window drift
  is systematic and should repeat, but one repeat of a configuration would
  calibrate every comparison in the records.
- **Deferred against capacity.** The replay queues partly on the step
  budget where production queues on the pool; whether that biases a
  comparison between arms that move prefill is unread.
- **ARC2's readings of 2026-09-19 have no archive entries**: the tool
  arrived there after them, so they are traced by Kaggle version alone.
