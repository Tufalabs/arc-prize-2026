# Roadmap

*Register: living — direction beyond the current round; entries carry
what has to become true before they can land.*

Package-scoped roadmaps: `packages/duck/roadmap.md`,
`packages/seal/roadmap.md`, `packages/ullm/roadmap.md`.

## Infrastructure

- **Checkpointing (P9)** — level checkpoints and branching from them are
  built (`platypus.checkpoints`: `checkpointing="level"`, `CheckpointStart`).
  Still open: `"move"` cadence, and I3 crash/timeout recovery, gated on the
  TAAF-level design (`checkpointing_notes.md`).
- **Duck data parallelism with per-game affinity** — the original duck's
  serving topology, which `tensor_parallel_size` does not reproduce.
  Unbuilt here; carried out of the duck-compaction round as roadmap
  material.
- **Compute facility + `_host_setup` / `ctx.shared`** — the
  heavy-non-LLM-compute needs deferred from this round (contract §4,
  I11). Nothing in-tree exercises the compute half today, so it is
  specified rather than built.
- **Retire `uncap_when_queue_drained`** — `ctx.slots.drained` answers
  "would giving this game up free a slot anything can use?", which under
  a ranking policy is the wrong lever: capacity is freed by yielding, for
  nothing, rather than by ending a game. The uncap would then essentially
  never fire, because a parked game is nearly always waiting — which is
  fine, since a ranked run wants no cap in the first place (the Kaggle
  duck now carries none). The duck is the only agent that still has the
  flag, defaults it to `False` and sets it in no preset. Remove it once
  nothing reads it.
- **Which deadline a submission actually runs to** — every scheduling
  conclusion on this branch is calibrated to 12M generated tokens at 9 h,
  and `taaf.deploy_kaggle.KaggleTarget.max_runtime_s` defaults to 12 h
  (a real submission with `run_as_submission=True` ignores it entirely
  and runs to the competition's own limit). The ranking is the only arm
  that leads both at 9 h and at 15 h, so a longer deadline moves in its
  favour rather than against it — but the operating point the caps and
  the rate were derived from is the 9 h one, and nothing currently
  records which number a submission is really playing against. Settle it
  before the next calibration, not after.
- **A residency ceiling for the slot scheduler** — `slot_policy`
  ranks a game that has never played ahead of any game that has spent
  anything, so a freed slot goes to an unplayed game until the queue is
  exhausted and every game in the run ends up resident. Parking releases
  a slot, never a subprocess. Measured at 15 slots over 50 games: 103
  live children and 43 GiB of anon against the queue-order arm's 33 and
  19 GiB, or roughly 0.8-1.3 GiB per resident game depending how far in
  it is. A submission queues 110, and one has now been run: all 110 games
  resident for a full 8.8 h on the Kaggle box, completing without dying.
  So this is a headroom question, not a known failure — what is missing is
  any margin we can name. The shape of a guard is a second cap on live
  children, where a freed slot goes to an already-started parked game
  rather than starting a new one once the ceiling is hit; the simulator
  cannot price it, because lanes on a token axis have no notion of a
  process. The per-game figures above were taken while a child still
  carried the engine spec and its imports, which has since come off
  (`AgentSolver._child_agent`); the floor under each resident game fell
  by roughly 0.55 GiB, and the rest is the item below.
- **Retained frames are the per-game term that is left** — with the
  import floor gone, what a resident game costs is what it accumulates,
  and every frame is held three times: the engine's ndarray in
  `Observation.state.raw`, the `taaf.game.Frame` copies on the
  observation (`Frame.__post_init__` copies), and the duck's base64
  `TimelineEntry`. Nothing evicts any of it. Measured on the
  `38e4852e` baseline over 5.7 h: the median child grows +208 MB and
  p90 +900 MB, the tail being the long games — so on a long run this
  exceeds the import floor that was removed, and at 110 resident games
  it is the headroom question above. Three copies serve three readers
  (engine replay, the contract's frame view, the kernel payload); which
  one should own the bytes is the design question. The kernel half —
  the whole timeline reserialized on every snippet and action reply —
  is `packages/duck/roadmap.md`, "Incremental Python-runtime state sync".
- **Carrying a hard level forward is a null result** — the rank key
  resets to zero on a level win, so a game that ground 200k tokens
  through level 0 arrives at level 1 looking like one that cleared it in
  5k. Swept in the simulator as
  `rank = level_tokens + c * (total - level_tokens)`: c=0.2 gains 0.146
  at the deadline and *loses* 0.05 by 15 h, and c=0.35 loses 0.79 — a
  spike on the edge of a cliff, which the calibration rule in AGENTS.md
  says not to ship. Recorded so the idea is not re-derived from scratch;
  the cheap next step if anyone wants the 0.2 bump is the
  strip-the-top-scorers check.
- **Count malformed-retry tokens** — a discarded attempt's generation is
  real spend, but `ResponseManager` rolls its element out of the turn, so
  `platypus.llm._record_turn` (which sums `turn_response_summaries`) never
  sees it. Counting it in `Backend.total_usage` alone makes the span leg
  of the three-way reconciliation disagree and raises on a clean run, so
  the backend deliberately skips it today. Fix is to emit the rolled-back
  attempt as an `llm/response` event and add its tokens to the span total,
  so both legs count it; then I1 can say "every generated token".
- **Usage on cancel is approximate across the proxy** — the per-game
  client counts from the last streamed partial (throttled to one per
  `stream_interval`, 3 s), while the host counts exactly at the
  cancellation instant, so host-vs-client stays a warning rather than an
  equality for cancelled games. Exact fix: the host replies to the
  client's existing `ABORT` frame with a final `USAGE` frame.
- **Usage on cancel for HTTP backends** — `Backend.respond` now counts a
  cancelled call's tokens from the last streamed chunk
  (`_usage_from_partial`), which both vLLM backends can supply. Backends
  whose wire format carries usage only in a final chunk return `None`
  and still lose those tokens (I1 says so, and a test pins it); closing
  it means asking servers for incremental usage
  (`stream_options={"include_usage": true}` on OpenAI-compatible
  endpoints) and deriving the cached split from what they report.
- **Cache-aware scheduling** (I10) — slots are now granted by rank
  (`platypus.priority`), but nothing in the ranking knows about the KV
  cache: the agent decides where a swap is free (it yields where its
  prefix is already gone) and the host only decides whether to take the
  offer. A scheduler that ranked on cache residency itself is the
  unbuilt part; revisit if throughput parity fails or the GPU count
  grows. Prefix-cache hit rate is already a logged metric.
- **Cache completed-game viewer payloads.** Building a replay currently
  rereads every diagnostic block for that game from the shared run tree.
  A completed game's payload is immutable, so cache it by run, game and
  adapter schema version; active games remain live and uncached.
- **Better game ordering.** Play non-CLICK games first, and/or games
  with fewer levels first. Today's order is fixed (or uniformly random,
  per the Kaggle presets); starting the cheap games means finished games
  — and their signal — arrive earlier, and the expensive tail (CLICK's
  action space, long level ladders) is not what everything else queues
  behind at the start. Both sort keys are known before any game starts,
  so this is a comparator in the solver, not a scheduler — and it composes
  with `shuffle_games` by shuffling first and stable-sorting second, which
  keeps ties in shuffled order: Kaggle's repeated submissions still vary
  within a cost class, without any bucketing machinery.
- **Multi-GPU serving** — one backend per GPU, games assigned to the
  least-loaded backend, one gating semaphore, engine startup in
  parallel; layers on the backend-of-record decision
  (`packages/duck/roadmap.md`). Design fork recorded: subprocess-per-GPU
  servers (the ARC3-Inference pattern, process-isolated
  `CUDA_VISIBLE_DEVICES`) vs in-process multi-engine.
- **Inter-game budget allocation** (I8/U10) — translating a run's
  wall-clock deadline into per-game token allowances; budgets stay
  agent-internal until then.
- **Timeout-outcome injectability** — contract §8 item 9, explicitly
  open: timeouts as values-at-seams so agent timeout-reaction paths are
  testable without real hangs; the wiring depth through ullm facilities
  is the unsettled part.
- **Eviction headroom as a token floor, not an item count** — the
  duck's `evict_turns` sizes headroom by count; turns and rounds vary
  wildly in size, so the right knob is a minimum number of TOKENS to
  evict once eviction fires, unifying the block and round paths. The
  ft09 measurement in `packages/duck/evidence.md` ("The eviction chunk,
  and why the budget was oversized") is the evidence: the chunky ~2-block
  slash lands the cycle well below its budget, so the budget has to be
  oversized to compensate. Round-level (within-turn) eviction is the
  other half of the same fix — it keeps the bound tight without the
  chunk.
- **TufaSlurmTarget resource defaults** for engine-carrying runs — the
  32G TAAF default OOMs a 27B FP8 load; runs set 64G/8cpu per GPU by
  hand today.
- **Shared flashinfer JIT workspace** keyed by venv identity — a
  per-run workspace means a cold recompile every launch.
- **Review coverage the first pass did not reach.** The adversarial review
  (PR #24) swept nine dimensions; a second round targeting its own gaps was
  cut off and produced nothing, so absence of findings in these areas means
  absence of looking, not absence of problems: a systematic `requirements.md`
  ledger sweep (A#/I#/U# against code and tests — the two high-severity
  findings that round produced were both stumbled into rather than swept
  for); `packages/reference-agents/` entire; ullm's concurrency internals
  (`shared_semaphore.py`, the kernel pool, `proxy.py` under concurrent
  in-flight calls, `engine_monitor.py`); `platypus/llm.py` and `live.py`; and whether the root integration tests
  would actually have caught the defects that round found.
- **Centralize the Kaggle offline stack.** The duck's preset and the
  prompted-random reference agent's each spell out the same wheels payload,
  dataset mount and 0.25.1 engine levers, and install `--no-index` from one
  shared `arc3-utils` dataset. They have already drifted apart once — the
  reference agent held a vLLM 0.22.0 pin the rebuilt dataset no longer
  carried — so the pins want one home rather than a convention.
- **`test_gpu_node.sh` and its Makefile target.** `make test-gpu-srun`
  invokes the script by path, so it survived the `tools/` cleanup that
  retired the other scripts there. Whether the GPU tier should keep
  dispatching through it is unsettled; retiring the script means retiring
  the target with it.
- **The rendered-prompt golden.** Capture/replay cannot catch a prompt defect
  that was already present when the reference was recorded — both swallowed
  bullet boundaries this round found replay green, by construction.
  `tests/test_prompt_boundaries.py` covers the mechanical half; the
  readable half is `packages/duck/tests/prompt_golden/`, rendered per
  configuration and regenerated by `PLATYPUS_REGENERATE_PROMPT_GOLDEN=1`.
  What is left is the same treatment for any agent that does not have it.

- **No public way to read a lane's score-vs-tokens curve once.** TAAF
  exposes `lane_score_at_budget` but not the curve behind it, so the
  sweeper's collector builds each lane's curve twice — once at an infinite
  budget for the terminal score, once at the arm's ceiling. Offline and
  therefore cheap, but the fix belongs upstream: a primitive that returns
  the curve would let both reads share one build. Reaching into TAAF
  internals to save it locally is the wrong trade.

- **Capture/replay cannot cover a multi-pass run.** `llm_config.replay_dir`
  is keyed by `game_id` and reference collection refuses duplicate ids, so
  `n_passes > 1` has no reference story until the keying extends to
  `(game_id, ordinal)`. That is what a sweep run is, so the sweeper has no
  determinism test of the kind every other agent gets; its own tests cover
  the mechanism instead (`packages/sweeper/tests/`).

- **A run-level knob has nowhere to reach a started backend.** The sweeper
  applies its run batch at construction, which covers engine specs and
  `run_config`. A knob that must touch something only alive after `setup()`
  would need a hook that does not exist. None is wanted so far, so this is
  recorded rather than built.

- **A sweep that wraps a built agent sends its engine spec into every game
  child.** `LLMAgentSolver._child_agent` blanks the wrapper's
  `llm_config.backend_factory` alone, and the wrapped agent holds an
  `llm_config` of its own. The convention keeps the factory on the wrapper
  (`LLMSweepAgent`), so nothing bites while the inner agent is constructed
  bare; an inner agent made by its `build` carries one, the child unpickles
  it, and `platypus.subproc`'s vllm assertion ends the run at its first
  game. ARC2 hit it and fixed both halves (arc2-platypus `e2b4917`): the
  child's copy drops the inner agent's factory too, and every arm, knobs
  and derivations applied, is put to the inner agent's own
  `prepare_for_backend` at construction, so an arm wider than the engine
  the sweep states is refused before submission rather than in play. Not
  carried here: ARC2 does it through `LLMAgent.without_engine`, which this
  repo does not have.

- **A game child can exit without its last message.** Seen on ARC2, in
  this repo's subprocess code: exit code 0, the parent's liveness watchdog
  finds the child gone and the inbox empty, and `SubprocessDiedError`
  stamps the run `crashed` with its score standing but the tokens after
  its last action never reaching TAAF. It tracks I/O pressure: 14% of
  games on three ARC2 sweep jobs that shared a node while their span
  streams went to NFS, 0-1.3% once the streams moved to node-local scratch
  (`design.md`, "Scratch and archive"), clustered in a run's tail. Cause
  not found, and not looked for here; ARC2's sweep analysis reads a lane's
  tokens from its span stream instead.

- **The last pyright relaxation.** Everything is checked now except ullm's
  unknown-type family, relaxed per-directory: closing it is ~100 findings
  across the package's dynamic surfaces. Bounded and not urgent, but worth
  keeping visible — the exclusions that sat next to it turned out to be
  deferring work that had already been done.
- **Status reporting residue** — mostly closed. Pool-level reporting
  landed (`[pools]`, the agent's `status_reporters`), so an agent that
  wires its own facilities into `ctx.status` can say a game is parked on a
  slot instead of reading bare `running`. What is left: the cgroup reader is
  exercised on real v2 only, with v1 and the `job_<id>` ancestor walk
  covered by fixtures.
- **Partial outputs are parsed for nobody.** `_process_partial_output`
  runs per engine tick and re-reads the whole accumulated output, while
  its only consumers (`live_snapshot`, the peek helpers) pull on demand
  and `_latest_partial` is written but never read. Currently bounded by
  `Backend.partial_interval_s`; the decided end state is to show completed
  rounds only and delete the path — the override, the base hook call, the
  interval knob and the dead field.
- **The host ceiling at high concurrency.** At 60 concurrent games ~37
  requests reach the engine while ~54 games wait, with search slots free
  and the engine
  at ~40% of a core — i.e. the host, not the GPU, bounds concurrency.
  **Measured before two changes that move the arithmetic**, so re-measure
  before acting on it: the mm token check went from 39.8 ms to 0.017 ms a
  call, and the search loop to 1.28x speed and 0.52x peak memory. Both took
  host CPU out of the very path this entry is about.
  Whether 60 is the right default depends on closing that, and the
  measurements that looked like progress on 2026-07-29 were confounded by
  run phase (modeler vs search), so it is open.
- **`default_socket_path` ignores `TMPDIR`** (hardcodes `/tmp`), so a
  proxy-mode factory on a node with a small `/tmp` tmpfs writes its socket
  somewhere the job did not choose. Unused today — platypus owns the
  socket for its own games — and one line to fix when it matters.
- **Heartbeat vs a real arena** — the mechanism has only ever been
  exercised against the toy game, never a real watchdog arena.
- **`minimal_diagnostics` is a `Benchmark.run()` argument, not a setting.**
  Every other thing a submission does differently is a P-0V setting on the
  agent (`settings.md`); this one is passed at run time and so sits in no
  table. Pull it onto `RunConfig`, where the deploy plumbing already lives,
  so a column states it.
- **Flash-Next serving and optimization** — its open items live in
  `packages/ullm/flash_optimization.md`, beside the experiments behind them;
  ARC2's four-L4 cell and image serving in `flash_optimization_arc2.md`.
- **Distilling the pruned checkpoint (QAD)** (2026-10-01: out of
  scope for now). Quantization-aware distillation trains the quantized
  model end to end against the BF16 original's output distribution, instead
  of rounding it once. Applied to our prune, the unpruned BF16 model would
  teach a pruned NVFP4 student. What it could buy: a deeper prune that holds
  its score, where a plain one does not (e256 scored 24.42 against e307's
  28.19, and e307 was not worse than unpruned), and e307 recovering what
  NVFP4's rounding costs. That it works on this model: local-inference-lab's
  QAD NVFP4 (`qad-step-4000`; ~300M tokens, days on 16 RTX PRO 6000s) reads
  AA-LCR 79.4 against 77.5 for their own one-shot NVFP4, and exact arithmetic
  78.7 against 77.2. What has
  to become true: a BF16 teacher pass, which needs a B200 node's tensor
  parallelism or a stored top-k file as in `flash_optimization.md` open item
  10; a trainer for the quantized student (the drafter SFT trained the MTP
  head alone); and a score run against e307 on the B200. local-inference-lab's
  own release, cut to e307's experts, is a sweep option meanwhile (T7, arm
  `qad`): 8.1% slower on the B200 for its MXFP8 layers
  (`packages/ullm/flash_optimization.md`, "Competitor tricks").
- **A sweep's reading of a score in TAAF** (2026-09-23, "at some
  point"). Duck sweeps read a game at `10 * sqrt(arc score)`, with the settled
  games dropped and the frontier ones weighted up, and that lives only in
  post-processing (`arc3-experiments` `sweep12/scoring.py`). In TAAF it
  would reach a run's own reports -- `summary.txt`, the diagnostics page, the
  log's running mean -- which show the official score only. The one score
  adjustment already in TAAF is the re-arc-3 baselines' multiplier (R11.15).
- **The per-game wall net as a platypus default** (2026-09-23, "down
  the line"). It is the duck's alone today: `duck.presets.GAME_RUNTIME_CAP_S`
  (nine hours, `packages/duck/evidence.md`, "The per-game wall net") through
  `duck_run_config`, while `RunConfig.max_runtime_s_per_game` defaults to
  `None`, so every other agent runs without one.

## Requirement states — partial and dormant only

Fully active requirements are code and tests; this table tracks only
what is not (IDs from `requirements.md`).

| Req | State | Condition / target |
|---|---|---|
| A4 kernel/checkpoint cadence | enforced by the pickle at `"level"`; the duck meets it | — |
| I2 checkpointing | `"off"` and `"level"`; `"move"` rejected | P9 |
| I3 recovery | dormant, sketch-only | P9 + TAAF design |
| I4 branching | from level checkpoints (`CheckpointStart`, replaying the source's actions); no config or LoRA delta at the branch point yet | a way to change a restored agent's fields |
| I10 scheduling | heartbeat active; scheduling dormant | cache-aware scheduling above |
| I11 fan-out | deepcopy + per-game seeds only | `_host_setup` / `ctx.shared`, next round |
| I14 nothing O(N²) | honored by construction | audit never run |
| U1 stable surface | partial: chat + usage + turn-runner per use | the public-API freeze line (`packages/ullm/roadmap.md`) |
| U7 LoRA | adapter-identity logging done | live validation once a permanent training server exists |
| U8 sampling profiles | partial | malformed-retry seed bumps still manager-internal (`packages/ullm/roadmap.md`) |

## The engine serving dependence

The 0.25.1 accident is a deliberate mechanism, available on the fork
(its `TUFA.md`) through the public sampling API
(`DuckAgent.zero_recurrent`, off by default); what it bought was a
reasoning register, which the shipped duck reaches by serving the
register's opening as tokens instead (`packages/duck/evidence.md`, "What
the blank was actually worth"). What remains standing:

- **Kaggle**: the fork ships there — the wheels install the stock
  engine and `ullm.vllm_overlay` replaces it with the pinned revision
  (`vllm_fork.md`, "Kaggle"). The `kaggle-rtx` column's engine levers —
  the host tier, the concurrency, the step budget, the admission
  reserve — were retuned on this line and run for nine hours.
- **The constraint's own cost.** The shipped schedule serves the same two
  tokens on every call, so every response opens alike; its score
  (`packages/duck/evidence.md`, "What the register is worth") does not
  separate "the register held" from that constraint. The control is the
  same schedule serving a VERBOSE opener (`Let me`), and has not been run.
- **Mechanism**: whether the persistent in-game effect flows through the
  recurrent state generation starts from, the re-read tail's attention
  KV, or the model's own earlier terse responses in the history is not
  settled; the offline replay separates the first two by tail length
  (the first-token effect vanishes for long tails), and a
  fixed-context response-length replay would settle the third.
- Re-derive the config optimum under the deliberate mechanism: sweep-3
  conclusions were optimized under the accidental one.
- Firm up the blank's own margin: on the fork, blank on
  scored 16.09 and 17.58 against a control at 10.48 (4 passes each,
  `evidence.md`), directionally clear and about two pooled SEMs apart,
  with the two blanked readings 1.5 apart on identical configuration.
  More passes on both arms before any conclusion beyond "the mechanism
  is real and worth points".

## Next round (sketch, not specced)

One solver codebase expressing duck-like, world-model-like and hybrid
solvers as configurations of shared modules — perception / world model /
proposer / gate / learn policy as orthogonal axes rather than an enum of
modes.
