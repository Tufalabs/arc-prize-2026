# Checkpointing — considerations (deliberately unresolved)

*Register: living — design input for P9; consumed and deleted when P9
resolves it.*

**Built since:** the per-game snapshot primitive at level boundaries, and
starting a game from one by replaying its action prefix, as the shared
primitive below has it (use case 2 without its lineage, retention or config
deltas) — `platypus.checkpoints`, described in `design.md`, "Level
checkpoints". Its on-disk layout (`agent.pkl`, `rng.json`, `manifest.json`
with the action prefix) supersedes the working proposal's below. Everything
else here is still open.

Two use cases hide under "checkpointing", with different persistence
requirements and different resume mechanisms. Much of this must be defined
at the TAAF level before platypus can finalize its side; the agent
contract (I2–I4) intentionally specifies only the per-game snapshot
primitive and leaves the orchestration semantics open.

## The shared primitive

Both use cases build on the same per-game unit, which platypus's contract
already defines: **snapshot = agent pickle (tools + scratch dirs ride
along) + infra per-game state (RNG, history position)**, taken at a
move/level boundary where kernel state is empty by contract, plus
**environment recreation by replaying the recorded action prefix** into a
fresh local engine (deterministic; online arcades excluded). Everything
below diverges.

## Use case 1: whole-run crash/timeout recovery

Goal: a benchmark run cut off by SLURM's 12 h walltime (or a crash)
continues in a follow-up job instead of being lost.

- **Scope is the run, not the game**: besides per-game snapshots it needs
  run-level state — which (game, pass) copies are finished vs in-flight,
  their `GameRun` histories/scores/token counts, diagnostics streams,
  capture streams for games whose traces aren't finalized yet (R5 traces
  are written at game end; an in-flight game's trace must be
  reconstructible after resume or its capture stream must persist) — and
  the agent template itself, so `_host_setup` can be re-run on resume.
- **Persistence policy: latest-only.** Recovery needs the most recent
  snapshot per game; older ones are garbage. Cadence = the configured
  move/level cadence.
- **Two flavors, very different difficulty**: *planned suspend* (soft-end
  reached → snapshot every game at its next move boundary → exit cleanly)
  gives a consistent cut and is probably the primary mechanism for the
  12 h case; *crash recovery* falls back to the last periodic snapshots
  (per-game staleness varies) and must tolerate a torn run-level picture.
- **Resume mechanism**: a new job on possibly different nodes that loads
  run state, skips finished games, reopens in-flight games (env replay +
  agent restore), and continues *the same logical run* — same run
  identities, appended histories, continued diagnostics.
- **TAAF collisions**: R2.13 explicitly disclaims resume today
  (periodic JSON "won't enable resuming"); `Benchmark.from_json`
  reconstructs everything *except the game list* (R2.14); scorecards
  restart with the new engine, breaking R11.12 reconciliation; deploy
  needs a requeue/reattach story (`deploy_meta.json` pointing at the
  successor job). All of these are TAAF requirements changes.

## Use case 2: branching while training

Goal: fan out N continuations from a chosen point of a recorded run —
fresh sampling, a new LoRA, a config delta — e.g. to focus play on
high-variance levels.

- **Scope is one (game, pass) at one boundary**; run-level state is
  irrelevant.
- **Persistence policy: a retained library, not latest-only.** Branch
  points must be enumerable after the run ("the snapshot at the start of
  level 3 of game X, pass Y"), so snapshots need addressing + metadata +
  retention beyond the run's lifetime — and a storage budget (snapshots ×
  games × levels; the workdir-bloat guard matters here). Level boundaries
  are the likely default retention set.
- **Resume mechanism**: not "continue the run" but "**construct a new
  run** from a snapshot": a new `GameRun` with a new `run_id` (trace
  identity must not collide) carrying **lineage** (parent run_id + branch
  point), env replay of the prefix, then live play under the modified
  config/adapter.
- **Semantics questions**: how does a branched run score (prefix levels
  are inherited — akin to `start_at_level` windows, R11.07, which already
  skip scorecard reconciliation)? Do prefix actions appear in its history
  (probably yes, marked inherited; their tokens belong to the parent)?
  What config deltas are legal at a branch point (LoRA/sampling clearly;
  changing the agent's *behavior* config mid-lineage muddies training
  attribution)?

## Why TAAF must move first

The per-game snapshot format is platypus's. Everything that gives
snapshots *meaning across runs* is TAAF's: run identity and lineage
(run_id, parent links — also the R5 trace join key), whether a resumed
run is "the same run" (R2.13 reversal, benchmark resume API, game-list
reconstruction), scoring semantics for partial/branched runs (extend
R11.07), scorecard reconciliation exceptions, and deploy-level
requeue/reattach. Defining platypus-side resume before these exist would
bake in guesses.

## Open questions (parking list)

1. Is planned suspend-at-soft-end a first-class TAAF operation (probably
   yes — it's the cheap 80% of the recovery value)?
2. Same run_id after resume, or successor-with-parent-link? (Trace
   folders and diagnostics keying hang on this.)
3. Snapshot addressing/retention API for the branch library; who prunes?
4. Capture-stream persistence for in-flight games vs re-deriving traces
   by replay on resume.
5. Branched-run scoring and comparability rules; lineage in diagnostics.
6. Engine determinism guarantees for prefix replay across versions (pin
   the re-arc commit as part of snapshot identity, as the duck's
   significance gate already does for runs).
7. Online-arcade runs: explicitly unrecoverable and unbranchable, or
   partial support (resume-from-scratch replays)?

---

# Working proposal (2026-07-21 discussion — NOT settled; loop back before P9)

Recorded so the P5 formats can avoid foreclosing it. We don't think this
is fully right yet.

## Multi-part runs (supersedes the two-flavor recovery framing above)

A long run = a chain of SLURM jobs ("parts") under one run root:

- Part N runs in `job_dir/part<N>/` (the run root is a real directory —
  consumers point at it; the chain is `ls`-discoverable).
- Each part ends with a **tidy shutdown** before walltime: snapshot every
  in-flight game at its next move boundary, sync trace mirrors, write the
  handoff manifest, submit part N+1, exit. A `handoff_complete` marker is
  written last (atomic); part N+1 refuses to start without it.
- **Job-level crash = that part is lost, full stop.** No periodic
  snapshots, no torn-state reconstruction — planned suspend is the only
  resume mechanism. Bounded loss: the previous part's handoff still
  stands, so relaunching the dead part is a manual one-liner, not a
  recovery feature.
- **Game-level crash is unaffected**: the game ends terminal-`crashed`
  under `on_agent_error`, appears in the handoff manifest's terminal
  bucket (state + note + tokens + trace-so-far), and later parts simply
  never resume it.
- Handoff manifest buckets: **terminal** (won/gave_up/crashed —
  consolidate, never resume), **in-flight** (snapshot + trace mirror),
  **not-yet-started** (concurrency cap never reached them — next part
  starts them fresh, no snapshot).
- Tidy shutdown needs its own deadline: a game that can't reach a move
  boundary by soft-end-minus-buffer is cancelled (I5) and marked crashed
  — one game lost beats an untidy part.
- Termination: all (game, pass) terminal → skip successor, write a root
  `done` marker, run consolidation in the same job. `max_parts` as a
  backstop.
- The **last part consolidates**: merged benchmark.json, all trace
  bundles, merged diagnostics at the root; snapshots/mirrors/periodic
  JSON deleted from part dirs; per-part logs + manifests KEPT (small,
  forensic).
- Successor submission: from inside part N at tidy shutdown (planned, so
  it reliably runs; verify compute nodes can sbatch). The
  upfront-`--dependency=afterany` alternative is more robust to shutdown
  bugs but pre-books slots and needs cancel logic — start simple.
- Resume writes ONLY into its own part dir; the predecessor is read-only
  after death. Same `GameRun.run_id`s continue (same logical run);
  attempt lineage lives in the manifests.

## What encapsulates a snapshot

One directory per (game, pass) boundary — not a single opaque file, not
the working directory:

```
part<N>/snapshots/<game_id>/p<pass>/<boundary>/   # boundary = m000123 | L03
  manifest.json   # readable WITHOUT unpickling: run_id, boundary, agent
                  # class + config digest, code identity (workspace shas +
                  # re-arc pin), trace step count, sizes
  agent.pkl       # tool scratch-dir contents ride along (A4: kernels empty)
  infra.pkl       # RNG state, move index, token-meter counters
  prefix.json     # executed ActionInputs + final-frame digest (env replay
                  # recipe, self-contained for post-run branching)
```

Plus, pointed into by the manifest: the game's **durable trace mirror**
(below). Rationale: the resume planner and the UC2 branch library both
need enumeration/validation without unpickling; whole-workdir snapshots
couple unrelated lifecycles and can't express per-game state. Restore
refuses (loudly) on code-identity mismatch. Latest-only per (game, pass)
for handoff; retained level boundaries are the UC2 branch library
(orthogonal — the recovery path no longer needs cadences at all).
Resume needs NO LLM replay: env replays the action prefix, post-boundary
work is discarded, the agent pickle already reflects pre-boundary calls.

## Streams: capture point, sinks, and the two-home placement rule

**Placement is a fixed property of each stream, never a run mode.** Every
environment provides exactly two locations — *local tmp* and *job_dir* —
and the deploy layer decides what they physically are (node-local disk +
NFS on SLURM; both local for login-node runs; /tmp + /kaggle/working on
Kaggle). Stream code never knows the difference. This RETIRES TAAF's
caller-facing `trace_dir` knob. What varies per run kind is only which
sinks are enabled, never where a sink writes.

**Capture point** (always on, per LLM call, in the game process at the
backend-client seam — U3): assembles one record (request held in
memory during flight, joined with response/usage/token ids at
completion), dispatches to enabled sinks, retains nothing.

- **Trace sink** (always on): TAAF's progressive per-run_id writer →
  **local tmp** (the heavy stream: media, token ids). Durable only via
  (a) per-game bundle export to job_dir **at game finish** (TAAF change:
  today only at teardown) and (b) the incremental **trace mirror**
  `part<N>/trace_wal/<run_id>/` synced at handoff (append-only files →
  copy only what's new; manifest records the consistent step count).
  Resume copies the mirror back to tmp and the writer continues
  appending — traces are NEVER reconstructed.
- **Diagnostics sink** (always on): append-only JSONL event stream per
  (game, pass) written DIRECTLY BY THE CHILD (bulk never rides the
  parent control pipe) + a parent-written run-level stream + a
  content-addressed block store (U2; write-if-absent tmp+rename) — all
  → **job_dir**, as events happen, line-buffered, no fsync. Diagnostics
  is the one stream that must never live on vanishing tmp: its value
  peaks when a part dies untidily (forensics) and it feeds in-flight
  viewing. In memory only: the open-span stack (O(depth)) and the event
  being written — never an accumulating tree, no diagnostics state on
  core objects (the sandbox 1.9TB pickle-bloat lesson).
- **Replay-reference sink** (opt-in, reference generation only —
  "capture" in the narrow sense): strict-format records → **job_dir**.
  Reference runs are concurrency-1 and short; capture-across-parts is
  unsupported by declaration.

Flagged trade (revisit if measured): diagnostics blocks on durable NFS
under a 64-way multimodal run; fallback = per-part block store on tmp,
flushed at handoff (a job crash then loses image payloads, text events
survive).

## P5 must / TAAF asks

- P5: diagnostics/reference record formats keyed by (run_id, move_index,
  call ordinal); per-(game,pass) files; parts-aware paths (part dirs) so
  P9 is not a format migration.
- TAAF: suspend as a first-class operation (snapshot-then-exit, distinct
  from request_stop); per-game trace-bundle export at finish; the
  two-home rule replacing `trace_dir`; scorecard/R11.12 reconciliation
  across parts; run-root manifest chain as the run's identity.
