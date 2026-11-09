# Design

*Register: as-built — updates with the code. `requirements.md` is the
normative contract this implements; this document says how. Alternatives
that were genuinely close are recorded inline with the reason for the
choice, so a later reader can reopen one without re-deriving the
argument. Symbol references are `module:Symbol`; detail beyond what is
here lives in module and class docstrings.*

## The three layers

- **platypus** (`packages/platypus/`) — the infrastructure layer: the
  agent contract's value types and the hosting machinery. The core is
  LLM-free; the LLM layer plugs into two narrow seams (below).
- **ullm** (`packages/ullm/`) — the LLM layer: backends, conversation
  model, managers, tools, capture/replay, trace production. Its own
  design doc: `packages/ullm/design.md`.
- **agents** (`packages/duck/`, `packages/reference-agents/`) — pure
  state plus configuration on the platypus contract; each documents
  itself.

Two class *pairs* carry the split: `Agent` / `AgentSolver` are LLM-free
(`agent.py`, `solver.py`); `LLMAgent` / `LLMAgentSolver` extend them
with backend lifecycle, per-game clients, capture and traces (`llm.py`,
`llm_solver.py`). The solver classes hold **no configuration**: all of
it lives on the agent (`run_config: RunConfig`, and for LLM agents
`llm_config: LLMConfig`), read off the template once at setup — one
fully-configured agent instance describes a whole experiment, and
per-(game, pass) copies carry the config inertly.

## Contract types (`contract.py`)

The types deliberately do **not** replace the TAAF/arcengine data model
— they wrap it where the contract adds semantics, and they exist to keep
agents off taaf/arcengine entirely (I1: the solver is the only code that
touches the engine).

- `Action` — engine-true name (`ACTION1..ACTION7`, `RESET`) plus
  optional `(row, col)`. It is not `arcengine.ActionInput` on purpose:
  `ActionInput` carries a raw `{'x','y'}` dict (the recurring
  x/y-vs-row/col bug source) and validates nothing at construction.
  Agent code speaks (row, col); the one defined bijection
  `{'x': col, 'y': row}` lives in `to_action_input` /
  `from_action_input` and nowhere else, and a misshapen action — unknown
  name, coordinates on an action that takes none or missing on one that
  does — fails at the agent. Coordinate values are the engine's to
  reject: their bound is the game's grid size.
- `Observation` — a delivery envelope over `taaf.GameState` (`outcome`,
  `level`, `index`, `caused_by`; raw engine data stays reachable via
  `obs.state.raw`). Its `frame` / `animation_frames` are the
  *partitioned* view, not always `state.frame` — see the move loop.
- `Move` — the action together with **every** observation it produced
  (usually one; two at a level transition); `move.observation` is the
  settled one the next move starts from.
- `RunConfig` — hosting knobs (concurrency, process mode, memory limit,
  error policy, seeds, wall budget, heartbeat, status cadence, Kaggle
  deploy hooks — see the field comments). `LLMConfig` lives in `llm.py`
  (backend factory, sampling profiles, capture / replay / traces).

## The move loop (`solver.py:run_move_loop`)

One `do_move` call per observation; the loop is shared **verbatim** by
both process modes. The framework-owned `Agent.do_move` wrapper converts
a raised `GiveUpError` into a `GiveUp`, validates the return type, and
rejects an action not in `obs.available_actions`
(`ContractViolationError`).

- **Level-win splitting (the whale's rule).** One engine state at a
  level transition yields two observations: the final frame becomes the
  new level's initial observation; everything before it (animation +
  win screen) belongs to the `level_won` observation. Observations are
  never fused; a `level_won` observation's `do_move` is delivery-only
  (returned action discarded, `GiveUp` honored); `game_won` ends the
  game with no call.
- **In-move execution.** `ctx.execute` is a `_MoveExecutor`, armed only
  while a decision `do_move` runs. Whether the agent executes in-move
  or returns the action, the **same recording path** stamps the move
  span's `infra/action` event and appends the `Move` — history and
  diagnostics cannot diverge between the two styles. Returning `None`
  without having executed at least one action is a contract violation.
- **Wall budget.** `max_runtime_s_per_game` is checked between moves
  *and* enforced inside the move: `asyncio.wait_for` cancels the
  in-flight `do_move` at the deadline, so one unbounded move (the duck
  runs unbounded LLM rounds) cannot overrun the cap. The cancellation
  lands in the innermost await; actions executed before the cut are
  recorded, and the game finishes `gave_up`.
- **The replay shadow** (`RunConfig.replay_shadow`, off). A RESET, and the
  actions after it that repeat a path already played on the level from its
  opening board, are answered from the record (`platypus.replay_shadow`):
  the agent sees the same frames, and the game is sent nothing until the
  agent leaves the recorded paths, when the deferred RESET and the replayed
  path go first, each checked against what was served. So the game is
  charged only for actions that could teach the agent something. It rests on
  a RESET restoring the level's opening board; a catch-up the game does not
  reproduce ends the game on the cluster, a determinism defect to find, and
  on Kaggle switches the shadow off for that game, whose score still counts
  (`replay_shadow_mismatch`). Served actions carry `shadow=True` on their
  `infra/action` event and catch-ups get `infra/catch_up` events. Off: it
  buys little and games break its premise (`settings.md`, P-0V, has the
  reading).
- **Stuck on a level** (`RunConfig.max_attempt_repeats`,
  `max_actions_per_level`, off). `platypus.attempts` counts a level's
  attempts as they are played -- the actions from the level's start or a
  RESET up to the next RESET -- each distinct attempt on its own, in
  constant time an action. The executor counts every agent action, served
  ones included, and gives the game up once one attempt reaches the repeat
  limit or the level the action limit, ending the move there; an action that
  wins its level ends nothing. Off: they save compute, not score
  (`settings.md`, P-0V, has the reading).
- **Malformed-output exhaustion.** Rejected replies propagate as an error when
  Duck exhausts its retry budget, and the game finishes `crashed`. A training
  client can instead accept malformed outputs with zero retries and continue
  the rollout with every generated reply retained.
- **Token attribution (I1).** The meter is an opaque object threaded
  through the loop: `meter.take()` runs immediately before each
  `game.execute_action(...)`, so tokens spent since the previous action
  attach to the next one — a batch's LLM cost lands on its **first**
  action, which is when the tokens were physically spent. Give-up and
  cancelled-move tails go to `finish_game` (R11.03).
- **Diagnostics shape.** The whole game is an `infra/game` span; each
  game span starts with a `game_metadata` JSON artifact carrying its
  reported level count and scoring baselines, so live readers need not
  wait for final benchmark bookkeeping. Each
  `do_move` an `infra/move` span stamped with what happened, carrying
  `frame_before` / `frame_at` artifacts (content-addressed, so repeated
  frames cost nothing) and one `infra/action` event per executed
  action (multi-action moves also store per-action `frame_after_<k>`
  artifacts). Agent and `llm/turn` spans nest under the move span via
  the ambient current span. An agent crash unwinds through the open
  spans, closing them with outcome `error`.

## Process topology (`subproc.py`)

Subprocess-per-game by default (`process_mode="subprocess"`);
`"inprocess"` is the single-process debug mode with identical agent
code. The taaf `Game` (arcade session, `GameRun` bookkeeping) stays in
the parent; the agent and the move loop run in a spawned child. Actions
flow child→parent and `GameState`s parent→child over two
one-directional pipes; status labels ride the same control pipe
(a few bytes each), while diagnostics are written by the child directly
— bulk never rides the pipe. Anything going wrong — child death,
memory-limit breach, pickle failure — is a loud error; **there is
deliberately no fallback to in-process hosting**.

The hardening follows aaf's solver subprocess, each item earned:
**spawn** (a fresh interpreter — no inherited CUDA contexts, threads,
or half-initialized module state); a **liveness watchdog** (a
SIGKILLed/OOMed child may never deliver pipe EOF; without the periodic
alive-check the parent would hang forever holding a concurrency slot);
a **cancellation relay** (SIGTERM → the child cancels its move-loop
task, so I5 semantics hold inside the child; the parent's send-pipe
close unblocks a synchronous `recv`; grace period, then SIGKILL);
**stderr capture** to a file whose tail is embedded in the death report
(the only way to see why an OOM/segfault child died); a **stdout relay**
(the child's prints reach the parent over its pipe and print there,
and its descriptor 1 goes to the stderr file: in a notebook the
inherited descriptor is a pipe one ipykernel thread drains, and every
process writing to it blocks once that thread stops); **RLIMIT_AS**
from `subprocess_memory_limit_bytes`, where failure to *apply* the
limit is itself fatal — never a silent uncapped mode; and **no engine
in the child** — a spawned child re-runs the agent module's imports and
rebuilds the agent's pickled fields in its own heap, so both are
per-game costs, and the engine spec (`llm_config.backend_factory`) is
dropped before the agent crosses (`AgentSolver._child_agent`, the LLM
solver's override) while `_child_main` asserts vllm is absent. The reap is
synchronous and lives in a `finally` so a second cancellation cannot
abort it and leave an unjoined orphan; each rung it escalates through is
announced on stdout, and a child still alive after SIGKILL and a join
(uninterruptible kernel wait, which no signal ends) is abandoned as an
orphan rather than closed — `close()` raises on a live process, and a
raise out of a game's `finally` cancels every sibling in the fan-out.
Abandoning detaches it from multiprocessing's bookkeeping (the
active-children set, and the parent's end of the resource tracker's
pipe), both of which the interpreter would otherwise wait on at exit.
Exceptions crossing the pipe are
round-tripped through pickle; unpicklable ones are substituted with a
`RuntimeError` carrying the traceback text.

Run-level fan-out (`AgentSolver._fan_out`): `platypus.priority.PrioritySlots`
caps concurrency; one named task per (game, pass). It grants a freed slot
to the best-ranked waiter rather than the longest-waiting one — ranked at
the grant by `run_config.slot_policy`, whose default is the run's own play
order and therefore FIFO. Rank never preempts, so a slot changes hands
only where a game finishes or offers its own back (`ctx.yield_slot`, §4). On any failure every
sibling is cancelled and awaited to the end of its cleanup before the
error propagates — `asyncio.gather` alone would return while sibling
subprocesses were still unreaped. Only cancellation finishes
still-playing games; on a real error they stay `playing` for the
benchmark teardown to mark crashed.

## Seeding (I11/I13)

`AgentSolver._game_seed` is the one place a per-game seed is decided:
`rng_seed_per_game[game_id]` if set, else `rng_seed + game_index`, else
OS entropy — but **always a concrete integer**, recorded per run. A seed
nobody can observe makes an interesting run unreplayable, so even the
unseeded case draws one explicitly; and because the value is explicit, a
run can adopt seeds it did not generate — which is what lets a reference
replay without depending on the game's position in the run's list.
`game_index` is the position in the list the AGENT is handed, so a
heartbeat game (taken off the front) shifts every derived seed by one:
`rng_seed_per_game` is what pins seeds across that flag. `ctx.rng` seeded from it is
the agent's only sanctioned randomness source (A2); per-call LLM
sampling seeds are drawn from it by `resolve_sampling` (U8 — profiles
must not pin `seed`). Hash-order determinism is opt-in at the reference
seam: `platypus.reference.require_pinned_hash_seed` raises unless the
interpreter started with `PYTHONHASHSEED` pinned — consumed by replay
paths whose comparisons are hash-order-sensitive (the committed-reference
replay, regeneration and viewer refresh in `tests/reference_support.py`);
ordinary runs don't pin.

## Level checkpoints (`checkpoints.py`, `checkpoint_games.py`)

`RunConfig.checkpointing="level"` snapshots every (game, pass) as each new
level starts, and a game that is a `CheckpointStart` starts from such a
snapshot. The purpose is branched play — many games from exactly one state a
run reached — not crash recovery (I3 stays open), so nothing of the source
run is carried beyond the agent, the engine state and the agent's history: a
branch is a new gameplay with its own traces and diagnostics.

- **When.** Between moves, after a move that leaves the game playing on a
  new level's first observation, and before the next `do_move` — so the
  snapshot is exactly what that call would start from. A move that plays on
  past the level's first observation leaves no such moment, and the level
  gets no checkpoint (an `infra/checkpoint` event says it was skipped). The
  pickle is the A4 check: an agent still holding a live kernel there fails
  the game loudly rather than storing a partial snapshot.
- **What.** `agent.pkl` (the whole per-game agent), `rng.json` (`ctx.rng`'s
  state) and `manifest.json` (the level's first frame and observation index,
  and every action since the level the game started on — readable without
  unpickling), written under a temporary name and renamed. Recorded as an
  event on the `infra/game` span rather than a span of its own, so a run's
  diagnostics tree is the same with checkpoints on as off.
- **The environment is replayed.** A branch game (`CheckpointGameAPI` for the
  arcade) replays the manifest's actions into a fresh engine inside
  `_start_game`, before TAAF's run bookkeeping starts: that is its
  fast-forward to `start_at_level`, so TAAF treats it as any level window
  and none of the prefix counts as its play. Replaying the same actions into
  the same engine reaches the source's state by construction, including
  state that carries across levels; each game checks the level, board and
  observation index it arrived at against the manifest as it starts, which
  catches an engine that does not replay deterministically. The alternative,
  TAAF's level jump, resets the target level instead, and loses that
  carried state (sc25 in the branch study) where no board check can see it.
- **Branching.** The game's per-(game, pass) agent is the checkpoint's,
  unpickled afresh for every pass, instead of a copy of the template; the
  branching run's own `RunConfig` replaces the one it was pickled with. The
  replay also rebuilds the source's moves (`partition`, as the loop builds
  them), which seed the branch's history: `ctx.history` continues the
  source's, its first observation is the one the source settled on, and a
  checkpoint the branch takes records the whole prefix. `branch_rng="fresh"`
  seeds `ctx.rng` like any game, so the passes of one checkpoint draw apart;
  `"resume"` continues the source's stream, which is what a test that a
  branch reproduces its source needs.

## The LLM layer seams (`llm.py`, `llm_solver.py`)

There is deliberately no LLM code in the base solver — the LLM layer
plugs into two seams that must live in the shared hosting path because
both process modes use it: `_game_llm_backend()` (build this game's
client) and the token meter.

- **One host backend, always behind a socket.** The config holds a
  picklable *factory*; the solver starts a **copy** host-side (the
  template's own field stays an unstarted spec, so the host's per-game
  agent copies keep pickling — the copy a *child* receives has the spec
  dropped, `_child_agent` above) and always serves the result over a Unix socket
  (`ullm.backends.proxy`), even when the backend would be picklable
  itself — the host-side backend is where run-wide accounting lives.
  Every game gets its own `ProxyBackend` client, in-process and
  subprocess alike, so **per-game accounting is structural**. The
  client carries the solver-stamped `run_id`, so trace attribution is
  per call, on the wire, with no turn-manager plumbing.
- **Replay mode is the exception**: with `llm_config.replay_dir` set
  there is no live backend and no socket — each game gets a
  `ReplayBackend` over its recorded stream, and the recorded usage is
  what the games report.
- **One context construction site.** `build_llm_context` is shared by
  the in-process host and the subprocess child, so both modes build the
  identical surface (I9). It also wires status: the backend's
  `status_hook` reports "waiting for LLM response" from inside the
  backend where the agent cannot see it, and the agent's unclaimed time
  is named `"other"` rather than reading back as a coarse `running`.
- **Turn running.** Agents call `new_turn_manager(ctx, profile=...,
  system_prompt=...)`, which resolves the named sampling profile
  (unknown names and pinned seeds fail loudly) and returns a
  `DiagnosticTurnManager` — a `TurnManager` that opens a live
  `llm/turn` span. Each manager keeps an append-only view of the
  responses and tool results its own turn completed, so overlapping
  managers on one backend cannot claim each other's usage and responses
  removed before close still count. Context-local forwarding attributes a
  nested plain `TurnManager` to its active instrumented parent, while a
  nested instrumented manager owns a child span. Incomplete calls remain
  subject to I1's backend capability carve-outs. It records one
  `llm/response` event per observed response and one `tool/result` event
  per result visible to the model.
  Removed rounds carry their reason on those events and preserve their
  content plus availability metadata in a separate `discarded_rounds`
  artifact; `conversation` remains the surviving prompt state. It also
  records one `kernel/exec` event per drained tool execution. Discard
  serialization is deferred to turn close, so diagnostics I/O never
  runs in the compaction path. With the null stream it is exactly a
  `TurnManager`.
- **Capture per game.** In capture mode the solver attaches a
  `CaptureSink` to each game's *client*, pointed at
  `games/<run_id>/capture/`. The sink is identity-blind (a directory
  plus a call counter); ownership is expressed by where the directory
  points. Recording happens in whichever process holds the client —
  the child, in subprocess mode.
- **Traces (U5).** A fail-loud capability gate at setup: if
  `llm_config.traces` and the host backend cannot produce R5-grade
  records, setup raises — never a silent skip. The backend *reports*
  (one record per successful call, tagged with `run_id`); the host-side
  `TraceCollector` *owns* step numbering, media splitting, payload
  hashes, action linkage and the taaf boundary. Action linkage is
  snapshot-based — each step links to the first action it authorized,
  matching the I1 batch-tokens-to-first rule — and the media split is
  proven exactly invertible per step (hash pair recorded; a mismatch is
  fatal). The proof only means anything because every image part is
  rewritten to name its file: a split that left some parts alone would
  re-inline to itself and the hashes would agree without the media being
  recoverable at all.
- **Three-way token reconciliation.** At the end of a clean run:
  per-game span-stream totals == per-game TAAF `ActionRecord` sums, and
  the TAAF sum across games == what the host backend consumed. A
  mismatch is a hard error when every game finished cleanly; otherwise
  it is printed (a crashed child legitimately loses its unreported
  tail).
- **Machine-wide kernel caps.** In subprocess mode the LLM solver
  enables ullm's multiprocess semaphores before the first child spawns
  (the manager address travels via an inherited env var), so kernel
  pool caps are genuinely machine-wide (I2b).

## Diagnostics (`diagnostics.py`)

One append-only JSONL **spine** per (game, pass), written as events
happen by whichever process hosts the game, plus a parent-written
run-level stream, plus a shared content-addressed **block store** for
anything larger than a spine line. In memory only the open-span stack —
never an accumulating tree. The streams are line-buffered and
tail-readable in flight; a span with no close event reads as in-flight
(live run) or aborted (dead run), which is exactly the aborted-span
signature I12 wants.

Design points worth knowing:

- **Inline values are deliberately restricted**: JSON scalars pass,
  long strings auto-block, anything structured raises and points at
  `artifact()` — the spine stays scannable and nothing large rides a
  line. Codecs: `json`, `bytes`, `png-grid` (indexed PNG, ~30x smaller
  than a pickled grid), `pickle`.
- **The ambient current span is a `ContextVar`** — it propagates across
  awaits within a task and copies into child tasks, which is exactly
  the "attach at whatever span is current" semantics I7 asks for, with
  per-game isolation for free when games share one process.
- **Self + inclusive token totals** roll up at span close; the close
  event carries both.
- **The null stream** (no job_dir, or `minimal_diagnostics`) makes
  every operation a cheap no-op, so instrumented code never branches on
  "is diagnostics on".
- **Small blocks pack.** Payloads ≤4 KiB append to per-process pack
  files (measured: 76% of blocks under 4 KiB; one file each turned a
  25-game arm into ~14k files of NFS metadata). Per-process naming
  means concurrent children never coordinate; loose files dedup across
  processes, packs exactly within one store and best-effort beyond it
  (writers consult the on-disk indexes, but that view can be stale) — a
  bounded space waste traded for never locking.
- **Scratch and archive.** The streams and the block store are written
  where `run_config.diagnostics_scratch` points: node-local disk under
  TAAF's Slurm preamble (`TAAF_LOCAL_JOB_ROOT`). A stream appends a line
  per event and a large block is a file of its own, which a network
  filesystem makes slow to write, and slow to delete by the tens of
  thousands a job leaves (measured on ARC2's sweep jobs, the same code:
  three jobs sharing a node had 65–86 of each job's 150 slots held by
  games blocked on NFS writes of their streams). The solver archives
  them into the job dir as numbered zip parts, `diagnostics-NNNN.zip`:
  finished games' streams and loose blocks every
  `diagnostics_archive_interval_s`, the rest at the run's end,
  after which the scratch copy goes. The run stream and every pack go
  into every part as they stand (a finished game's stream may reference
  a small block a still-running process packed), so a viewer of a
  running job has the run's overview and every finished game whole. Members keep their job-dir-relative paths, and
  the readers (`read_game_events`, `read_run_events`, `job_block_store`,
  `game_run_ids`) find a file wherever it is: loose in the job dir, on
  this process's scratch, or in a part. With no scratch (off Slurm, the
  tests) the streams are written into the job dir.
  A game on scratch cannot be watched live from another node, and a
  hard-killed job loses what no part had taken yet; a run that is to be
  watched as it plays sets `diagnostics_scratch = ""`, and its viewer
  otherwise has the games that finished before the last part.
  `tools/archive_job_diagnostics.py` zips an existing job dir the same
  way, verified from the part alone before any loose file is deleted.

## Viewer (`chunks.py`, `viewer.py`, `live.py`)

- **One server, two front ends.** The reserved `/_replay/` prefix
  (its adapter at `/_api/`, JSON and the prompts' images as PNG, both
  outside the file space — no run layout
  produces underscore-prefixed names) hosts the ARC3 Replay Viewer,
  backed directly by `benchmark.json`, per-game span streams, and the
  shared block store — agent-agnostic, since everything it reads is
  infra-owned surface. Its reasoning, level, and action lists navigate
  the associated board frames directly; transcript cards link to their
  action's frame via an opt-in host-tool convention (a tool result
  reporting `executed`/`action_num`) and fall back to the move's frame
  without it. Run discovery accepts a run itself or nested
  experiment/seed layouts; it is depth-bounded, prunes payload
  directories, and caches results so browser traffic does not become an
  NFS metadata walk. Run summaries stay benchmark-only; full
  conversation content loads for the selected game's replay.
  Generated-token and response totals include every completed backend
  response in a move, including reasoning and tool calls; empty diagnostic
  shells left by cancellation are retained but not counted. The messages a
  compaction carried into a rebuilt context are shown on its first request,
  marked retained, and not counted either. The panel shows
  hosted-game elapsed time and the final ARC-AGI score, with no synthesized
  Return metric. The canonical diagnostics paths below remain the export and
  detailed span-tree surface.
- **Producer-owned rendering.** A chunk renders one span node; the
  kind→class registry is filled by the modules that emit the spans,
  and unregistered kinds fall back to `GenericChunk` (header + kv +
  artifacts + children) — new span shapes always show up, and a
  bespoke chunk is a progressive enhancement, never a prerequisite.
  The stream never carries HTML, and it stays engine-true (`ACTION1`);
  UP/DOWN/MOUSE naming is presentation only.
- **Producers announce themselves through an entry point.** A viewer
  runs standalone over a directory of runs and imports no agent on its
  own, so a producer declares its chunk module under the
  `platypus.chunks` entry-point group and `load_producer_chunks()`
  imports them on the render and serve paths. A producer whose module
  raises is not swallowed: rendering with silently missing renderers
  looks exactly like working fallback.
- **The viewer never deserializes a pickle.** The codec travels in the
  stream, so honoring it would execute whatever a run directory holds,
  and an agent's richest artifacts are objects built by *model-written*
  code, rendered from a shared results tree. A producer that wants a
  rich view of such an object renders it where it is still alive and
  stores the markup; pickle artifacts are for offline repro bundles,
  loaded deliberately.
- **URL space = file space.** The server renders viewer-scheme paths
  fresh from the JSONL streams (live viewing — never stale from disk)
  and serves anything else as static files; `export_viewer`
  materializes the same canonical paths with the same page builders for
  serverless archival. TAAF's central diagnostics page links into this
  scheme with plain relative hrefs, and the links work in both worlds.
- **Live turn streaming is pure job-dir files.** While a viewer's live
  page is open it touches a `live_request` flag; the game process
  writes 1 Hz snapshots (and the conversation, only when it grew) only
  while that flag is fresh. Files, not sockets, deliberately: a Unix
  socket is bound to the host whose process created it, while files
  over NFS let a login-node viewer reach a run on any compute node —
  and nothing is written when nobody watches.
- **Deterministic rendering.** `ViewContext.deterministic` suppresses
  wall-clock-derived output so the rendered HTML is reproducible —
  which is what lets a slice of it live in a replay reference as a
  golden of the diagnostics→viewer pipeline.

## Reference mode (`reference.py`)

A reference is **one flat JSONL file per (game, pass)** — no archive —
so a deliberate behavior change reviews as a line diff. Typed lines:
`info` (run_id, game_id, and the per-game seed — replayable input, not
just provenance), `block`, `record` (LLM calls in order), `expected`
(actions, per-action tokens, final outcome, optional `viewer_html`
golden), `solver_pkl` (API keys scrubbed), `git_status`.

The **solver writes** the per-game file at game end from the capture
streams; test code only **collects** finished files — after the
clean-state guard, which refuses crashed/cancelled generation runs (a
crashed game would still replay bit-exact, so it must be rejected at
collection, not left to validation). Regeneration is api-gated and
validates every new reference **by an immediate replay** (I13) before
it is accepted. `load_references` splits files back into the capture
layout `replay_dir` expects; `load_reference_seeds` feeds
`rng_seed_per_game` — the one place reference provenance becomes
load-bearing, because a reference produced elsewhere carries
generation-time inputs that can only be adopted, not reconstructed.
The `viewer_html` golden must come from the original run, never a
replay — a golden taken from replay would make replay-vs-replay always
agree, blind to the very divergence it must catch.

## Run operations (`heartbeat.py`, `watchdog.py`, `status.py`, `memory.py`)

- **Heartbeat.** With `heartbeat_game` set, infra takes `games[0]` away
  before the fan-out and plays it in the host process with one random
  move every `heartbeat_interval_s` — so an environment watchdog that
  kills a run after a silent stretch (Kaggle's arena) sees activity
  while every real game is still in its expensive first turn. Nothing
  reaches agent code: no context field, no hook, just a shorter game
  list, which is what makes it work for agents not yet written. The
  start is idempotent and the LLM solver starts it *before* the engine
  build (the silence most likely to trip a watchdog); `stop()` finishes
  the game so it lands `gave_up` rather than being stamped `crashed`.
  The cost is honest and bounded: one game scores as a random walk.
- **Stall watchdog.** The LLM solver watches the engine from the moment
  the proxy is up: once a minute it reads the backend's own engine
  metrics -- the generated-token counter and the request gauges the
  engine line prints from -- and when the counter has not moved for
  `engine_stall_timeout_s` while the engine still reports requests, or
  the in-process engine has died, it ends the run through TAAF's
  `stop_run` (R2.64): the soft deadline on demand, so games land
  `cancelled`, diagnostics are written, the reason is on the benchmark and
  the job exits normally. The quiet clock counts from the demand, so an
  idle engine is never a stall. Demand is read from the engine rather than
  the proxy because a frozen engine keeps reporting the requests it froze
  with, and the traffic replay bypasses the proxy. What it replaces is the
  rest of the budget: the per-request bound is an hour, and a frozen
  engine otherwise runs a submission's clock out producing nothing.
- **Host monitors** (`platypus.host_monitor`). Every request a game
  sends and every status line passes through the host's event loop; when
  that loop blocks, the engine drains what it holds and idles, and the
  status lines go silent -- the engine watchdog sees no demand, so no
  stall. Two monitors on threads of their own run wherever diagnostics
  do (off under `minimal_diagnostics`): a loop watchdog that, once the
  loop's one-second tick is ten seconds stale, snapshots every thread's
  Python stack with its kernel state, wait channel and system call every
  fifteen seconds, and prints one `[stall]` line when the loop runs
  again -- except inside TAAF's periodic save, which renders the run's
  page on the loop on purpose and which the engine generates through on
  the requests it holds: that hold is counted, and reported only past
  three minutes, when it is a stall whatever it is inside; and a sampler
  writing, every ten seconds, pressure stall information for the job's
  cgroup and the node, reclaim and writeback counters, per-operation
  latency on network-filesystem mounts, local disk utilisation and the
  loop's worst lag. Both write to `monitor/`
  under the diagnostics root -- node-local scratch on Slurm, so the
  filesystem under suspicion cannot block the monitor meant to catch it
  -- and the archiver carries them into the parts. The thresholds are
  constants: the question they answer is which resource a stall waits
  on, not how often to act on one.
- **Status.** Three infra-owned coarse states (waiting for slot /
  running / finished) known from the host task's own lifecycle; agents
  refine only `running` through `ctx.status`. Labels are published **by
  the process doing the work** at the moment it starts waiting — never
  inferred host-side from a semaphore counter, which can say how many
  games are blocked but not which game is blocked on what. Push-on-
  change means a wedged child keeps reporting the state it is wedged
  in; the periodic line names the longest-held state, which is the one
  figure that distinguishes a wedged run from a healthy one.
- **Memory.** The process tree is reported by **PSS, not RSS**: every
  game child maps the same interpreter and wheel shared objects —
  file-backed pages shared through the page cache; a spawned child
  shares no heap — so summed RSS reads a 32-way fan-out as hundreds of gigabytes on a
  machine holding tens; PSS divides each shared page by its mappers and
  the tree total counts it exactly once — the number a cgroup or SLURM
  limit actually judges. Under a cgroup the report also splits what the
  job is judged against into anon + page cache + kernel, because cache
  is reclaimed before any OOM kill and `current` alone cannot tell a
  real emergency from a full cache. The tree is split by role (TAAF,
  vLLM, games, REPLs, other) so the growing part is named, and the
  cgroup line carries the OOM killer's count, which tells a process
  killed for memory from one that crashed.
