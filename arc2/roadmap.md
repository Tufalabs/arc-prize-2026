# Roadmap

*Register: living.*

Direction and open work. How things are today is the code, the tests, and
the package READMEs.

## Headline: descend the ladder

Bring the ARC-AGI3 stack down to ARC-AGI2, one layer at a time, adapting
each to a setting where a task is a handful of input/output grid pairs
rather than an interactive game — and forking rather than depending
(`AGENTS.md`, *Forking, not depending*).

1. **Orchestration** (from `tufa-arc-agi-framework`, branch
   `public-budget-curve`) — the benchmark/task/solver split, the
   deploy targets, and the score-vs-token pooled-lane budget curve. The
   curve's construction assumes only independent lanes with a cost and a
   score, so `(task, pass)` substitutes for `(game, pass)` unchanged; the
   move-loop-shaped parts of the framework do not survive contact with a
   one-shot task.
2. ~~**Agent contract and hosting**~~ — built, `packages/platypus`. An
   agent is asked once per task and answers through `ctx.submit` as often as
   it likes; the package README has the contract and what enforces it.
3. **The agent** (from the same branch, `packages/duck`) — the model
   writes Python in a kernel and the kernel calls back into the agent. What
   changes most is that ARC2 lets a candidate be **checked against the
   training pairs before it is submitted**, which ARC3 never offered; that
   is a new lever rather than a port.

## The ARC2 shape

Built: `packages/taaf/README.md` holds the data model, the `Game` lifecycle,
the scoring and weighting decisions, and the lane's step curve. What the
curve still needs watching for is a per-task token cap that binds for most
tasks — the finishing costs then pile up at the cap, the knots collapse, and
the right end of the curve measures the cap rather than the agent. Keep the
cap loose offline, and report how many tasks hit it beside the plot.

A solver may answer more than once. `execute_action` replaces the standing
answer and the submission file follows it immediately; `finish_game` scores
whatever stood. A task cut off mid-work therefore lands its best so far
rather than a placeholder, and a lane is a step curve rather than a single
point — diagnostics scores every recorded answer at the cost it stood for, so
a budget reading says what the run would really have scored there.

Two properties hold that together and neither is optional. **The solver
never sees a score**: `execute_action` returns nothing, because handing back
correctness for an answer the solver can still replace is an oracle it could
iterate against. And **scores are computed outside the play**, from the
recorded submissions and the answer key — the curve's intermediate steps in
diagnostics, and the reading of a game still in play — which is how the
curve gets its resolution and a report in progress its scores without the
solver ever seeing one.

The consequence to keep in view: lane scores are not monotonic.
`packages/taaf/README.md` says what a dip means.

## Decision: the harness/trainer boundary

Platypus exposes behaviour-named, individually settable controls and public
observation hooks. It carries no training profile. The trainer composes these
controls and observes the public lifecycle without overriding private methods.
Environments name machines; supplied backends work in every environment.

The boundary lets callers compose inference and rollout behavior without a
parallel agent implementation. Dataset recipes, reward coefficients, engine
ownership and training schedules belong to the trainer. Setting ownership is
mapped in `settings.md`; callback contracts live on the public hooks.

## Decision: ullm is a dependency, not a fork

`packages/ullm` is the one layer that stays upstream. It imports nothing
from its host by design, and uv can take it alone out of the arc3-platypus
workspace:

    ullm = { path = "../arc3/packages/ullm", editable = true }

That resolves without taaf, platypus, duck, whale or the scene decomposer,
and installs under its own import name — so forking later is a one-line
change with no import churn.

Two obligations ride along.

- ~~**Re-pin when `pure-duck-0251` lands.**~~ Landed. A bump stays a
  deliberate act rather than a chore: the rendering ullm does decides what
  the model sees, and a bump has already changed that once — see
  `packages/duck/README.md`, "What the backend sends back".
- ~~**Re-pin when arc3-platypus `flash-arc2` lands.**~~ Landed (#94); the
  ullm pin includes its merge commit on `main`, and the vLLM pin is the fork's merge
  of `flash-arc2-pp` (arc-prize-2026-vllm #7).
- ~~**Re-pin when upstream's `image-continuation` (PR #64) lands.**~~
  Landed; the pin is its merge commit on `main`.
- ~~**Re-pin when arc3-platypus `flash-traffic` (#98) lands.**~~ Landed;
  the pin is its merge commit on `main`, `871be67c`, which carries the bf16
  recurrent state in ullm's Flash-Next engine (the duck's own line came out
  with it); the vLLM pin is the fork's #8 merge, the pin ARC3 runs, and
  `taaf`'s base commit is TAAF #56's merge.
- ~~**The first lock takes a one-time `[tool.uv] constraint-dependencies`
  block**~~ — written, with ullm pinned at `45f4c3a3`. It pins ullm's
  direct dependencies to the versions arc3-platypus has locked. Not
  because those are canonical — they are a snapshot of whenever it last
  re-locked, and `uv sync --upgrade` there would land where a fresh
  resolve lands — but because they are the combination ullm's own CI runs
  green against, and a dependent cannot fix what breaks. Resolving fresh
  jumps `openai` 2→3, `anthropic` 0→1 and `mcp` 1→2, none of which ullm's
  backends have been run against. `uv lock` is conservative, so the block
  is scaffolding rather than standing config: the lock holds the versions
  once written.

`protobuf` is the one thing the block cannot cover — arc3-platypus locks it
at two versions under different markers, so a single `==` constraint would
break a resolution branch. It wants a look when the serving stack first
runs.

## Upstream provenance

What this repo is tied to, so a catch-up knows where to read from. A
catch-up branch moves these in the same PR that takes the commits.

| Layer | Upstream | Base commit |
|---|---|---|
| `platypus`, `duck`, `sweeper` (forked) | arc3-platypus `main` | `871be67c` (2026-09-22, the #98 merge) |
| `taaf` (forked) | tufa-arc-agi-framework `main` | `25e8733` (2026-09-22, the #56 merge) |
| `ullm` (pinned) | arc3-platypus, `packages/ullm` | the `rev` in `pyproject.toml` |

A base commit is the last upstream commit whose applicable changes are in;
what was read and deliberately not taken is in the catch-up PR that moved
the row.

## Decision: the sweeper is a fork

`packages/sweeper` is forked from arc3-platypus (at `9f26d98`), not pinned
the way `ullm` is. The criterion the `ullm` decision passed is the one it
fails: it depends on `platypus` under the same import name, so pulled by
git it would resolve against our fork and die on its first import, and its
rule about where a knob's value is *read* is a statement about the host.
The merge commit records what differs from upstream. The real sweeps live
in `arc2-experiments`,
which depends on this workspace by editable path; the package keeps the
mechanism and a worked example.

- **The per-game cap (65k) follows the reading budget by scaling, not by
  measurement.** `DuckAgent.max_tokens_per_game` is sized so an arm's mean
  spend can reach the budget a sweep reads at, and the measurement behind
  it was made against the old 40k reading
  (`packages/duck/evidence.md`, "max_tokens_per_game"). Re-deriving it at
  45k is the open item, and the register A/B is a reason to expect the
  answer to differ by arm: one ceiling does not sit the same distance
  above a concise arm and a verbose one.

- **`DEFAULT_BUDGET_TOKENS` (45k) is derived, not solved.** It charges
  uncached input at 1/9.4 of a generated token and is the charged
  equivalent of the 40k generation-only reading sweep 4 was analysed at,
  for lanes running ~1:1 (`packages/duck/evidence.md` has the assumption).
  Solving it on a real charged curve — the budget at which the mean score
  is where 40k generation-only put it — is the refinement, once a sweep has
  been read this way.
- **A knob must target something that exists, and sweep 4's
  `repetition_penalty` does not.** The sweeper refuses a knob whose path
  names a missing attribute or mapping key, because setting it would create
  a stray nothing reads. Sweep 4's design aims that knob at
  `extra_params[repetition_penalty]`, which the duck's play profile does not
  declare; the engine read the key anyway, so the sweep's readings stand,
  but when arc2-experiments re-pins past this, `DuckSweep.build_agent`
  has to declare `extra_params["repetition_penalty"] = 1.0` on the profile
  before wrapping, or the sweep refuses to construct. Every other knob in
  that design resolves.

## Next: the hidden-set submission

The endgame is a real submission over the competition's hidden set (240
tasks, 259 test outputs, 17 multi-test — measured from
`sample_submission.json`). The prerequisites are in place: Flash-Next, the
default engine, serves on the four L4s as two pipeline stages of two tensor
ranks, and a submission stops itself at `submission_stop_runtime_s` rather
than being killed and scored as an error.

- **Flash-Next's serving record and open items** are upstream's
  `packages/ullm/flash_optimization_arc2.md`, the ARC2 addendum to
  `flash_optimization.md`: the four-L4 cell and why it is 2x2 at 0.955 with
  the bf16 state at a 3,300-token step and 60 slots, what the traffic replay
  read there (the state +24%; the unit, expandable segments and 0.96 off),
  the image prompts and their two fixes, and ARC2's B200 scores across
  checkpoints. ARC2's own prune ships, our REAP-307 with its own
  drafter (`settings.md`, P-EV). The cell has played
  two twelve-hour Kaggle runs: one stalled on the tier's engine log, now
  off, and the other scored 81.04 against the float32 cell's 78.40, like for
  like.
- **Read it against the B200 figures, not the port's number.** Flash-Next's,
  with every image served: per-run mean 74.65, pass@2 per test output 82.04
  over `arc2-eval` x 2 (the addendum's "Images and the prefix cache"). For
  the 27B, `arc2-eval` at per-run mean 52.36 / best-of-two 63.61 was
  measured with malformed history from round two of every game
  (`packages/duck/README.md`, "What the backend sends back"); its
  single-pass re-baseline at the current pin is job 2313489
  (`/data/personal/$USER/arc2_results/eval_120/`).
- **Two takes per task: built.** One
  play's two attempts share its reading of the puzzle and its mistakes, while
  the competition scores an output right if *either* attempt is, which pays
  for independence: on the B200, pass@2 per test output reads 82.04 against
  a per-run mean of 74.65. `MultiTakeDuck` (`platypus.multi_take`,
  `duck.multi_take`) plays every task in two independent takes and composes
  the two attempts from them. The demo's Kaggle column plays its ten-take
  variant, `TieredTakeDuck`, which queues further takes by how far they agree
  and votes their answer (`packages/platypus/README.md`, "Up to ten takes");
  SLURM keeps one take. What is left:
  - **A different second take.** Both takes are the same duck today. A second
    take on another prompt or sampling is set on `takes[1]`.
  - **A later take's images are not checked against the engine's
    `max_pixels`.** Only the first take settles the ceiling, and the duck's
    coherence check covers the context window alone. Moot while the takes
    are identical; a second take on other image settings needs it.
- Run it as a real submission, not an emulation — the rerun path is the
  one we get no logs back from, so everything before it is the rehearsal.
  The hidden set is far less multi-test than `arc2-eval` (7% vs 37.5%),
  so per-run mean and best-of-two diverge much less there.
- Decide what a disappointing number would mean before seeing it.
- **The slot scheduler is in, on Kaggle.** `platypus.priority` grants a
  freed slot to the game that has spent the fewest tokens
  (`slot_policy="least_tokens"`, the Kaggle preset alone); the duck offers
  its slot back at every round, a free offer (a trim just fired) is taken
  at once and any other after `slot_hold_s` — which no shipped deployment
  takes: Kaggle sets it to `None`, since a non-free swap spends a resident
  context against a deadline that does not move, so a swap happens at a
  compaction or a finish and nowhere else, and SLURM keeps `queue_order`,
  where a running game outranks every waiter. Upstream's level weighting is
  not carried: every task here is one level, so it priced every token at 1.
  Residency is
  deliberately uncapped — upstream's open item, a second cap on started
  games, is not built here: the twelve-hour runs at 240 entries on 40 slots
  are the measurement (`packages/duck/evidence.md`, "The narrowed context
  window"), and a ceiling that makes the parked set fit the tier is the
  shape of the fix if one is ever needed. The per-game cap stays as a
  backstop above the SLURM figure (`KAGGLE_MAX_TOKENS_PER_GAME`), because
  the ranking frees a slot by deprioritising, not by ending.

## Kaggle — open questions

- ~~**Is a timed-out run's `submission.json` scored, or is the run just
  failed?**~~ **Answered by the first real submission (2026-08): the run
  is failed** — it errored out at the platform limit and nothing was
  scored. So a submission must stop itself, and it now does:
  `KaggleTarget.submission_stop_runtime_s` bounds a submission's TOTAL
  runtime from notebook start. The run stops gracefully AT that line —
  the same cancel-and-teardown as an ordinary run, measured at well
  under a minute — because a measured hard `os._exit(0)` gets the
  notebook recorded as ERROR even at exit code 0 (kernel death is
  failure to the executor, whatever the code; fired at 30.12 min with a
  complete submission.json and still ERROR), and an ERROR run's
  submission is not scored. For the same reason there is NO hard-exit
  fallback behind the graceful stop: it can only ever destroy a run
  that would have completed, and a hang ends in the platform kill's
  identical ERROR anyway. Applies only when the run is a submission,
  emulated or real; an ordinary run keeps the `max_runtime_s - 600 s`
  soft stop.
- ~~**Submission mode needs a soft cutoff** so a proposal survives the
  kill.~~ **Closed by construction**: `submission.json` is rewritten on every
  `execute_action`, so a hard kill finds the standing answers rather than
  nothing, and nothing has to reach a terminal path first. A soft cutoff
  would still be worth having for its own reasons — a graceful stop scores
  every game and writes its diagnostics — but it is no longer what stands
  between a kill and losing in-flight work. (The question this left open —
  whether a killed run's file is read at all — has since been answered
  above: it is not.)
- ~~Does `arc-agi_test_challenges.json` exist in an interactive run?~~
  **Yes** — the competition mount carries all six files
  (`arc-agi_{training,evaluation,test}_challenges.json`, the two solutions
  files, and `sample_submission.json`). So forcing
  `KAGGLE_IS_COMPETITION_RERUN` in an ordinary run exercises the rerun's own
  file branch, which is otherwise the one path that runs only where we get
  no logs back. Wire that into the Kaggle rehearsal.
- ~~**Why did a twelve-hour run's engine stop generating two hours before
  the clock?**~~ **The host tier's engine log.** Its lines were ~98% of what
  the engine process wrote; the kernel stopped draining that pipe and the
  tally's `logger.info` blocked inside the scheduler, freezing the engine
  mid-step (2026-09-19). Off on every column now, ullm's default (its
  `kv_offload.md`); the rerun played its full twelve hours. A SLURM run
  whose engine goes quiet now ends itself ten minutes in, diagnostics
  written; Kaggle's plays on to its soft deadline (`settings.md`,
  `engine_stall_timeout_s`).

## Measured on the competition mount

From `sample_submission.json`, which is the organizer's own statement of
what a submission must contain:

- **240 tasks, 259 test outputs, 17 multi-test.** The 240 confirms the
  reported split (120 semi-private + 120 private) first-hand.
- **The hidden set is far less multi-test than the public one**: 7% against
  `arc2-eval`'s 37.5%. The two official metrics therefore diverge much less
  on the hidden set than the public evaluation set suggests, and any
  weighting experiment calibrated on `arc2-eval` overstates how much the
  choice matters where it is scored. Worth confirming against a second
  source before leaning on it.
- The sample's shape is exactly what `taaf.submission` emits: one dict per
  test input, in order, keyed `attempt_1` / `attempt_2`.

## The duck — deferred by decision

Each of these was decided against **for the porting round**, with a reason,
rather than missed. `packages/duck/README.md` says what the agent does now.

- **A "finish now" prompt.** Measured during the port, Haiku over four
  multi-question `arc2-eval` tasks: three of the four ended `gave_up`
  having written *nothing* into `test_out`. The per-call harvest and the
  placeholder fill are both inert on those games — there was never
  anything to harvest. The model works until it is confident and the
  budget runs out first.

  Telling the model up front to write a provisional answer early was
  **rejected**: it spends tokens on every game, including the large
  majority that finish comfortably, to buy something only the minority
  that run out ever uses. The shape that fits is infra-side — when the
  remaining budget drops below a threshold, send one turn asking for the
  best answer the model currently has and take what comes back. Nothing
  is spent until it is needed, and the model hears about the deadline
  exactly when the deadline is real.

  It needs a threshold worth defending, and it is the same problem as the
  Kaggle stop above: both are "the clock is nearly up, salvage what
  exists". Until it lands, a game that runs out of budget scores zero.
- **Namespace helpers in the kernel: segmentation, and friends.**
  Upstream's `segmentation.py` reads as ARC3 machinery and is not — it is
  stdlib-only connected-component segmentation of a grid layer
  (components, object hashes, outer contours), written self-contained
  precisely so it can be spliced into a sandbox bootstrap where project
  packages are not importable. So it is a grid primitive, and the reason
  it is out is not that it would not work here.

  It was held back because a namespace furnished in advance answers the
  round's first question dishonestly: does the model reach for the kernel
  at all? It does — so the objection has expired, and this is now an
  ordinary "is it worth the tokens" question. When it comes, prefer
  splicing upstream's file verbatim over rewriting it. `ascii_grid()` has
  already gone the other way and lives in the kernel prelude.
- **The Kaggle memory figure is coupled to the V2 model runner, and the
  runner is not a knob.** ullm pins V2 for every environment
  (`UNIVERSAL_ENGINE_ENV`), and `gpu_memory_utilization=0.85` on the four-L4
  cell is the reserve V2 does not make for CUDA graph capture. V1 would
  account for capture itself and let utilization rise to the ceiling, which
  is the reading that would argue for a per-cell runner setting on the ullm
  side; nothing here has measured it, and there is no runner toggle on the
  column to try it with (`settings.md`).
- **Sweep the compaction constants together, and price them against
  generation.** Done by sweeps 5–7 (`arc2-experiments`, `duck_sweep7/`):
  the reply cap and the prompt budget were drawn as coordinates and read at
  terminal score with the throughput priced in (tasks an hour on the Kaggle
  box). The cap moved to 24,576 and the window with it (`reply_cap_tokens`,
  `context_window_tokens`); the budget stayed at 48,640, which sat on the flat
  part of its shape; and the live summary replaced pruning, so
  `compact_to_tokens` and `evict_turns` no longer bite on a shipped column.

### Eviction can drop the task itself

Every analyzer re-entry — a nudge, an answer-repair prompt — opens a new
user-led block, and eviction removes whole blocks from the oldest end,
keeping only the most recent. So once a game has re-entered and then
compacts, the FIRST thing to go is the opening prompt: the legend, the
example-pair pictures, the question pictures. From then on the model's
opening prompt is a repair note, the puzzle exists only as arrays in the
kernel, and the pictures D1 gave it to *see* with are gone. Seen directly:
under a compacting window the trace's later steps carry no media at all.

Not a bug in the trim — it does what the original did — but a design hole
the original never had, because an ARC3 prompt re-sends the frame every
move and ours sends the puzzle once. **Fixed behind `keep_task_prompt`
(default off): the first block's turn is evicted like any other, its
prompt stays as the head, and later evictions take the block behind it.**
The pictures then stay in the KV-cached prefix. Sweeps 5 and 6 drew the flag
and resolved nothing (`packages/duck/evidence.md`), and the shipped columns
no longer prune: under `summarize` the rebuilt history opens on the task
prompt by construction. The flag stays off, prune's alone.

## The duck after sweeps 8 and 9 — open

The audit turn, the file tools and the rest of sweeps 8 and 9's settings ship
(`packages/duck/README.md`; `packages/duck/evidence.md`, "Duck sweeps 8 and 9"). What they leave open:

- **The audit's confidence only breaks the ten-take vote's ties.** It is a
  weak signal: it ranks tasks by difficulty, and puts a task's won game above a
  lost one 59% of the time. The vote reads it on a tie
  (`packages/platypus/README.md`, "Up to ten takes"); it may later decide
  whether a game gets a further take.
- **The audit settles a doubt the training way.** In the long run's traces it
  finds what is new in the test, then decides by preference or by a check
  narrower than its own question (`dbff022c`, `581f7754`, `88e364bc`,
  `800d221b`, `c7f57c3e`); its misses in the smoke did the same. Having the
  model compute the other reading's output, rather than judge which reading
  is more natural, is the change that would follow.
- **Draft acceptance is not priced.** It falls about 2% of a decode step's
  tokens from a play temperature of 0.6 to 0.8, which the throughput-adjusted
  score behind the call leaves out.
- **ARC3's AutoRound cut of our prune** is read on
  ARC3 first; AutoRound over e307 was a tie here once its 9% was priced in.

## The Kimi student — open

`DuckAgent.build(kimi_student=True)` plays the duck as the Kimi teacher
traces were played (`tools/teachers/README.md`). What a student run still
needs:

- **A reading of its sampling.** It starts on the column's (temperature 0.8,
  top_k 194, no nucleus; 2026-10-29); the teacher's (temperature 1,
  top_p 0.95, the only values Kimi's API takes), which the traces were drawn
  at, may serve it better. A higher temperature lowers draft
  acceptance (above), and a nucleus caps it, since vLLM drafts at
  temperature alone.
- **Its checkpoint.** The Flash-Next cells serve the ARC2 prune
  (`arc2-reap307`) in NVFP4 beside the combined ARC2/ARC3 drafter head,
  trained against it and ARC3's joint205; a student needs its checkpoint in
  the format the cell serves, and a drafter for it or drafting off.
- **ARC2's own teacher calibration.** The teacher entry runs ARC3's
  accounting factors (3.7 characters a Flash-Next token, 1.054 Flash-Next
  tokens a billed reply token); grids as text tokenize differently (a token a
  digit for Flash-Next, up to three for Kimi). The trace sets in
  `tools/teachers/README.md` are what to calibrate from before the next ARC2
  teacher run.

## Possible additions from upstream's sweeps 12 and 13

Pieces upstream's duck took on after its sweeps 12 and 13. The listing of
files and Python objects after a summary, the thorough summary request, the
summary kept from an unclosed thinking block and sampling with `top_p` 1.0
came over with sweeps 8 and 9 (`packages/duck/README.md`). One remains, ported
only for a reason of its own here (2026-09-27).

- **The summary's cap clamped to the engine window.** Upstream lowers each
  summary request's `max_tokens` to what the engine window has left over
  its measured prompt, and asks only that 8,192 tokens remain
  (`MIN_SUMMARY_ROOM_TOKENS`). This duck reserves the whole cap, twice or
  six times the target. The clamp is what let upstream draw prompt budgets
  to 180k: its summaries wrote a median 0.26 of the target without
  thinking and 0.36 with it, so the reserved cap was room no summary used.
  It has no score reading of its own.

## Watch items

- **The competitor-tricks sweep options, and the four L4s' budget.** The
  launcher's `relaxed_accept`, `autoround`, `qad` and `fp8_dense` arms
  (`packages/duck/tools/traffic_launch.py`) are upstream's T8, T1, T7 and
  its FP8 dense GEMMs; their B200 readings are in upstream's
  `packages/ullm/flash_optimization.md`, "Competitor tricks". Sweeps 8 and 9
  read them for score (`packages/duck/evidence.md`, "Duck sweeps 8 and 9"):
  relaxed acceptance played longer games for no score, QAD and the FP8
  dense GEMMs cost points, AutoRound tied once its speed was priced, and
  `indexer_fp8` ships. ullm's four-L4 cell now
  captures CUDA graphs once at 0.904, 0.955's equivalent, which no boot on
  that box has checked; `spec-decode-4xl4` owns that cell.
- **Summaries written just before a park may never be read.** `before_round`
  writes and installs the summary, then offers the slot with `compacted: True`,
  which the host always takes. On `slurm-b200` (`queue_order`) the yield hands
  the slot straight back; on `kaggle-4xl4` (`least_tokens`, `slot_hold_s=None`)
  a compaction is the one point a slot changes hands, and a compacting game is
  a high spender that has just added the summary to its spend, so it parks
  behind any cheaper waiter and near the deadline may never resume. To
  measure on a Kaggle 12-hour run: installs not followed by an ordinary
  request of the same game before the run ended, and their tokens against
  the run's total. The fix, if the count warrants it: after a calibrated
  point in the run's walltime, offer the slot *before* writing the summary
  (marked free) — before that point a parked game still has time to resume
  and writing first saves it re-prefilling its history; after it the decode
  is what to save, so nothing cheaper waiting hands the slot straight back and
  the summary still runs on the hot cache, and otherwise the game parks with
  its summary unwritten.

- **The kernel is escapable, and the answer channel is losable.** In sweep 5's
  first round 15 of 4,917 games reached outside the puzzle, 13 of them under
  `check_mode=mandatory`; none touched task data or answers, two crashed on
  forged host calls, one won through a forged channel. Every case began with
  the model losing its own answer channel — under `MANDATORY` a deletable
  global, `check`, shadowed by a stray `def check(fn): pass` or removed by
  `del check` — and reverse-engineering the harness to get it back, which is
  all inside one Python process: the trusted `host_call` in `check.__closure__`
  (and via `gc.get_objects()`), `host_call_token` in the worker's `main()`
  frame through `sys._current_frames()`, `_make_trusted_host_call` in
  `sys.modules['__main__']`, the callback names by `import duck.python_tool`,
  and the sources through `subprocess` over the job's venv. Not a
  score-integrity hole: a forged `duck_check_success` submits grids that
  `test_out` already allows in `ENABLED` mode, and a lookup `fn` passes
  `check` anyway. Fixes in order of value per effort: (1) restore the REPL
  surface before every model cell — re-inject the built `check`, `show`,
  `ascii_grid` and the arrays in a private pre-execution, the way the answer
  read-back already runs after each one; (2) forged or malformed host calls
  refused as a tool error the model sees rather than `crash_game`; (3) worker
  hygiene upstream in ullm — `del host_call_token` after building the closure,
  the maker out of the module dict, the worker not reachable as `__main__`;
  (4) accept that in-process trust is behavioural, and say so beside
  `sandbox.py`'s header; (5) close the cheap read gaps — the bundled main
  project's `src/` in `blocked_dirs`, the source tripwire in `kernel_code.py`
  and the worker; (6) a per-game diagnostic marker for code touching
  `sys._current_frames`, `gc.get_objects`, `__closure__`, `subprocess` or a
  host-callback name, so sweeps can condition on such lanes. The ARC3 duck
  has the same holes. Fix (1) is in for `check` under the file tools, which
  ship on: a private call before every cell puts it back and says so; `show`,
  `ascii_grid` and the arrays stay the model's to rebind, on purpose. And a
  read gap the patched `open` and `np.load` leave: a cell can read the answer
  key (`taaf/data/arc2-eval.npz`) through `os.open` or a subprocess, and list
  directories outside its own. Kaggle has no answers on disk; Landlock
  confinement, as the seal has, is not taken.

- **A checkpoint loads from NFS in 32 s to 27 minutes.** Cached on the node,
  e307 loads in 32 s; on dgx06 with five jobs loading three checkpoints at
  once, 7–27 minutes, most likely its prefetched pages evicted before the
  loader read them. The nodes have a 28 TB local `/raid`: staging checkpoints
  there, or fewer distinct checkpoints on a node, would bound it.

- **A game child can exit without its last message.** Exit code 0, the
  parent's liveness watchdog finds the child gone and the inbox empty, and
  `SubprocessDiedError` stamps the run `crashed` with its answers and score
  standing but the tokens after its last answer never reaching TAAF. It
  tracks I/O pressure: 14% of games on the three sweep-7 jobs that shared a
  node while their span streams went to NFS, 0–1.3% once the streams moved
  to node-local scratch, and clustered in a run's tail. Cause not found; the
  sweep analysis reads a lane's tokens from its span stream instead.

- **The slot counts were set on the earlier defaults' traffic.** The B200's
  200 is the float32 cell's 150 scaled by the pool the bf16 state gains, and
  the four L4s' 60 was read on a replay of a workload recorded under `prune`
  (`settings.md`, P-EV; `packages/duck/evidence.md`, "The Flash-Next cells on
  the traffic replay"). Under `summarize` a game holds a longer context for
  longer: sweep 7's later jobs ran 250 games in flight on the bf16 state at
  66–90% of the B200's pool. Owed: a long B200 run's reading of 200, and a
  workload recorded under the shipped defaults, before the replay speaks
  for the deployment again.

- **The recurrent-state blank is in, off, and unmeasured here.** The GPU
  columns run the vLLM 0.29.0 fork (`vllm_fork.md`), on which
  `DuckAgent.zero_recurrent` can zero a request's recurrent state at
  `zero_recurrent_at`; upstream scored the blank at 16–17.6 against a 10.5
  control on ARC-AGI-3, at the position its shared system prompt spans.
  Three things are open before it goes on. Where the blank should land on
  THIS prompt layout: the shared prefix — system prompt plus tool schema as
  the chat template renders them — is 1,371 tokens on Qwen3.8's tokenizer,
  so 1568 falls 197 tokens into the first user turn, i.e. into the task
  prompt: what the recurrent layers would forget is the legend and the start
  of the training pairs, not the tool interface upstream's blank forgets. The
  candidates are shipping 1568 and measuring, padding or reordering the
  shared prefix to span the boundary, or measuring our own position off a
  logprob replay the way upstream did. Where the 0.25.1 + offload accident actually blanked on
  these prompts, which was never measured and is the treatment every score
  before this branch was produced under. And the regression itself: the
  offload tier is off (`packages/duck/evidence.md`), so runs on the fork
  play the no-blank treatment until the position is settled.
  Upstream's later reading of what the blank was worth: not a position but a
  reasoning REGISTER, reachable by serving its opening as the response's
  own first tokens on a stock engine (`packages/duck/evidence.md`, "The
  reasoning register"); the blank added nothing on top of the served
  opening there, and ships off upstream too.

- **The host KV tier, the scheduler shape and the concurrency are
  upstream's numbers, unmeasured here.** The SLURM column runs 80 games
  with the victim tier behind the pool, the Kaggle column 40 at a 60 GiB
  tier (the P-EV layer of `presets.configure`), both on a step budget of
  five recurrent blocks (8192, P-00) and ullm's 10% admission reserve
  (`deployment.ENGINE_ADMISSION_WATERMARK`); every one of those figures is
  what arc3-platypus measured on its own games (ullm's `kv_offload.md`,
  "How it was tested"), and ARC2 tasks are shorter and compact less. Two
  things to read off the first runs here. The prefix-cache misses both
  arms of the register A/B showed at 80 (heavier at 80 than at 40, never
  diagnosed) have a candidate cause in upstream's reading — a 2048-token
  step budget admits one block-sized prefill per step and nothing behind
  it, and a zero watermark preempts in storms once the live set outgrows
  the pool — so the cached fraction of prompt tokens per window is the
  first number to compare against those runs. And the Kaggle sizing was
  taken on one card at `gpu_memory_utilization=0.95`; here it is four L4s
  at 0.85 with the V2 runner's capture headroom to protect, and the wider
  step budget's activation workspace comes out of that same headroom, so
  the profiled pool at engine start is the number to read before trusting
  the 40-game figure there.

- **The ranking prices generated tokens only.** `least_tokens` ranks on
  `ctx.backend.total_usage.completion_tokens`, so the uncached prompt a
  game forces the engine to prefill costs it nothing in the ranking that
  decides who keeps a slot. The two do not track each other: across the
  twelve-hour Kaggle run uncached input per generated token spanned 0.42
  to 1.87 from the tenth to the ninetieth percentile and reached 9.92,
  with one game spending 466,629 uncached input against 70,000
  generated, and the run's totals were 12.4M uncached against 9.7M
  generated. Prefill was about a sixth of engine time there, so the
  unpriced half is not a rounding error — a game with bulky tool output
  reads as cheap and keeps winning slots against a lean one that
  generates steadily. `TokenMeter` already computes the number
  (`prompt_tokens - cached_tokens`, off the same private backend), so
  what changes is the hint the duck offers and how `_rank_key` weighs
  it. The engine cost model fitted to that run prices an uncached prompt
  token at about 0.15 of a generated one at its mean context, which is
  the coefficient to start from; it is not a constant, and rises as
  contexts shrink, because prefill stays flat while decode gets cheaper.

- ~~**The backend host's listen backlog is widened by hand.**~~ **Closed
  by the ullm pin**: the host now listens with a 4,096 backlog and the
  client retries its connect-and-hello on the reconnect schedule, so a
  run's slots are no longer bounded by asyncio's default queue of 100;
  the local widening and its storm test went with it, and
  `RunConfig.start_gap_s` (P-00) paces the spawn burst a run past ~100
  slots meets.

- **Decision: what a backend supports is the backend's to know.** An image
  is continued by tokens by an engine, not by a chat server, so the columns
  that serve images are built on an engine and `serving="server"` builds with `images="off"` (`settings.md`, P-0V).
  The agent does not check that. Reading the factory's `mode` to guess
  whether the topology can keep the promise asks a capability question of
  the wrong object: it would refuse a real engine reached over ullm's engine
  proxy, and every future transport that can carry an image. A backend that
  cannot keep the promise fails, which is the answer from the thing that
  knows. A training run that wants the duck's images takes an engine —
  in this process, or over the engine proxy. Still open: whether the
  served-history duck should see `text_grids` or the kernel arrays alone.
- **The register opening ships on, measured once.**
  `DuckAgent.inject_text_on_first_prompt` / `inject_text_on_later_prompts`
  serve `We need solve` and `We need`, worth +15.4 points at a matched
  budget over `arc2-eval` at 2 passes, while spending fewer tokens
  (`packages/duck/evidence.md`, "The reasoning register, measured here").
  What is open: two passes is a thin ±, the words and the schedule were
  taken from upstream rather than swept here, and the cost of the
  constraint is unmeasured — no arm serves a VERBOSE opener (`Let me`) on
  the same schedule, which is what separates "the register held" from
  "every response opens alike". Upstream owes the same control.
  `platypus.reasoning_corpus` and the sweeper's `register_games` column
  read the register off a run; the opening-block classifier does NOT
  transfer to ARC2 and should not be used on it.

- **Two crash signatures from the first real sweep, neither the
  configuration's.** Eleven lanes died in ullm's proxy client at child exit
  (`ProxyBackend._read_loop` destroyed while pending; nine in one job) — an
  upstream teardown race, one lane at a time. One lane died in the
  mandatory-check host callback: `duck_check_success callback payload is
  not a dictionary` (`python_tool._receive_checked_answer`), a kernel-side
  payload the host refused rather than a model error. Both are under the
  collector's crash threshold at 120 games an arm; both want a root cause.

- **`reasoning_tokens` is always 0.** ullm does not populate it, whatever the
  model actually thinks. Nothing computes with it — the budget counts
  completion tokens, and `platypus.llm` only stamps it onto the `llm/turn`
  span — so it is a display field reading zero rather than a defect in what
  is served. Read it as evidence and it says the model is not reasoning,
  which is not what it means.

- ~~**A node's shared uv cache is not healed by TAAF's deploy.**~~ It is:
  the deploy script removes a managed interpreter that no longer runs, under
  the sync lock and before the sync, as `tools/test_gpu_node.sh` does
  (`packages/taaf/README.md`, "Deploying").

- **No test boots a preset's engine.** The `gpu` tier drives
  `PromptedFlipAgent` against a real engine, but configures it inline on a
  2B model — upstream's shape, and the reason a column is ~2 minutes.
  `DuckAgent.build()`'s columns are asserted only at config level, so a
  column whose config vLLM refuses passes the whole gate and fails on
  the box. Only the demo notebook exercises them, and only when someone runs
  it. Upstream has the same gap and closed it for the duck alone
  (`test_duck_engine_startup.py`); worth copying when a preset matters
  enough.
- **Kaggle's machine shape is silently downgraded, never refused.** The name
  is `NvidiaL4` — not `NvidiaL4x4`, and not TAAF's `NvidiaRtxPro6000`
  default, which was the ARC-AGI-3 shape. Getting it wrong lands a P100,
  whose compute capability is below what PyTorch itself supports, and the
  first sign is the engine refusing the model's quantization after the
  kernel has already installed everything. The demo passes it explicitly.
- **The status report is blind during model load.** `LLMAgentSolver` builds
  the engine before the base solver opens the monitor, so the longest
  silence in any real run is the one stretch `[games]`/`[mem]` cannot see.
  ullm's own `[engine]` line covers it on a separate cadence. Moving the
  monitor earlier is a two-line change if the silence ever costs a
  diagnosis.

- **Depending on `ullm` means depending on vLLM.** It is a direct dependency
  of `ullm` on Linux, so `uv sync` pulls the whole CUDA stack — a CPU-only
  run installs ~10 GB of wheels it never loads, and a SLURM worker spends
  minutes on it before playing anything. The lever if it bites is asking
  `ullm` upstream to move `vllm` behind an extra; nothing here can fix it.
- ~~**Node uv caches can poison a SLURM run.**~~ **Closed**: the gpu
  dispatcher heals them (`tools/test_gpu_node.sh`), the way upstream's has
  since 2026-08-01. It is not the transient the first sighting suggested —
  a second one persisted on one node for two days, failing every job that
  landed there, because nothing ever asks the installer to try again. Two
  shapes, both handled under the sync lock: a sync that fails (retry once,
  then wipe python + uv cache + venv and rebuild) and a sync that succeeds
  onto an interpreter that will not start, which uv cannot see because an
  install DIRECTORY satisfies it.

- **`benchmark.json` grows with how chatty a solver is.** Every answer's
  grids are serialized into `GameRun.history`, so a solver that replaces its
  answer often multiplies the file. At 240 tasks and ~10 answers each that is
  tens of MB, which is fine; at a hundred answers each it is not, and the
  periodic save would start stalling the run. The lever if it bites is
  storing a repeated submission by back-reference — the *cost* changed, which
  is what the curve needs, but the grids did not. Measure before building it.
- **Writing the submission per answer is O(tasks) each time**, tens of
  milliseconds for a 240-task set. That is what buys a hard kill finding the
  standing answers; worth re-measuring if the answer rate climbs.

- **`TEST_ARGS` does not reach the srun tiers.** `test-gpu-srun` and
  `test-duck-engine-srun` append it to `tools/test_gpu_node.sh`, whose
  first positional argument means "node side, run this test id" — so
  `make test-duck-engine-srun TEST_ARGS='-k x'` builds a venv on the
  login node and runs nothing. The Makefile comment says the opposite.
  Selection on those tiers is the mark expression (`ARC2_GPU_MARK`); a
  single-column re-run wants either `-k` threaded through the collect
  step or the comment corrected.

- **What continuation buys is unmeasured.** Exact history is on for the
  GPU presets; the nesting guarantee on traces is the prize, and the
  prefill saving is expected to be small because prefix caching already
  covered the repeated head. The compacting B200 column prints per-step
  uncached input; a full pass has the tokens/sec. Measure before claiming.
- **Summarize compaction under exact history has run on an engine.** Sweep
  6's extension (2026-09-18) played 50 arms of the live-request summary, then
  `summarize_alternative`, on B200 engines under exact history: about 5,900
  summaries, their requests continuing the served ids (91% of their input
  cached), no drift after the fix to the install's declaration (the stub had
  hidden it; its tests now hold the rule). That version is now `summarize`.
  The `b200` tier's summarize column still runs it once per boot; there is no
  composed v0.6 profile here, and a rerun states its coordinates as fields on
  `build()`.
- **The sweeper's `check_enabled` target is dead.** `check_mode` (a
  `CheckMode` enum) replaced the boolean, and `setattr` on a dataclass
  instance creates an attribute rather than refusing one, so a sweep
  column still aimed at `check_enabled` would vary while configuring
  nothing — the failure the sweeper's arm-target check exists to catch.
  Retarget it at `check_mode`; three-level if `MANDATORY` is swept.
- **The rendered-prompt golden is the narrower version of a closed gap.**
  `test_backend_history_render.py` pins the message dicts the backend hands
  the chat template, not the template's output, because the template needs
  the model's tokenizer and the cheap tier has no HF cache. A golden over
  the rendered request wants a tier that does.

## Repo core — open items

- **The `gpu` and `api` tolerances can go.** Each tier target tolerates
  pytest's exit 5 (nothing matched) and says so; both tiers now carry tests,
  so the tolerance is no longer protecting anything and an accidentally
  empty selection would read as a pass. The `local` tier keeps it.
- ~~**The gpu tier runs where it is invoked.**~~ Built:
  `tools/test_gpu_node.sh` stages to shared storage and fans out one `srun`
  per gpu test, since the checkout lives on login-node-local `/scratch` and
  a second vLLM engine in one process wedges startup.
- **Cheap-tier parallelism is unmeasured.** Worker count is capped at 12
  against CPU count and the xdist distribution mode is the default; both
  are guesses until the suite has enough shape to time.
- ~~**A deploy from a git worktree names the wrong bundled repo.**~~
  **Closed**: `TufaSlurmTarget.main_project` now defaults to
  `taaf.deploy.detect_bundled_repo_name()`, the same derivation
  `snapshot_editable_sources` uses to name what it bundles, so the two
  cannot disagree. Requiring the caller to pass it was the older contract
  and it made the failure worse: a literal is right until the checkout is
  renamed or a worktree checks the same repo out under a branch name.
- **No calibration discipline written down yet.** Tuning a default against
  benchmark runs needs its own rules (how much of the score sits in how few
  tasks, what a train/test split does and does not divide). Write it when
  the first default is tuned, from the ARC2 evaluation set's own
  statistics.
