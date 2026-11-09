# platypus

*Register: living.*

The agent contract and the machinery that hosts one. TAAF says what a game
is and scores it; platypus says what an *agent* is and runs it — per-task
process isolation, hierarchical diagnostics, the LLM backend topology,
capture and replay.

## The contract

```python
class MyAgent(Agent):
    async def _do_move(self, ctx: MoveContext, view: GameView) -> GiveUp | None:
        ctx.submit(my_answer(view))  # as often as you like
        return None  # returning ends the game
```

`_do_move` is called **once** per game. `ctx.submit` is the only way an
agent answers, and it may be called any number of times while that call
runs: each answer replaces the standing one, is recorded with what it cost,
and reaches `submission.json` before anything can kill the run. Returning
ends the game with whatever stood; returning `GiveUp` ends it early and says
why, which frees the concurrency slot **without withdrawing the answer**.

`ctx.yield_slot(hints)` offers this game's concurrency slot back to the run
and returns when the host re-grants it — immediately, unless something
outranks the game. It is an **offer, never a limit**: the game continues
either way, no yield can end it, and a run whose host has no scheduler
returns from it having done nothing. The agent chooses *where* — only it
knows where a swap is free, because a compaction has already lost the KV
prefix — and the host chooses *whether*, because how long one game may hold
a slot is a clock, and agents do not read clocks. The `hints` are what the
host ranks on under `run_config.slot_policy` (the duck reports `tokens` and
`compacted`), so they must be live at the moment of the call. `ctx.slots`
is the other half: it says whether anything is waiting for the slot a
give-up would free.

All configuration lives on the agent — `run_config`, and `llm_config` for an
`LLMAgent`. The solver classes hold none, so one fully-configured agent
instance describes a whole experiment.

## Takes

A `MultiTakeAgent` (`multi_take.py`) plays its game in two **takes** — two
agents in turn, inside the one game process — and submits an answer composed
from both; its class docstring states the rule. Every test output is scored on two
attempts, so the second is worth most as a genuinely different grid, and an
independent second take is where one comes from. A take is not a TAAF pass:
a pass is a replicate reported on its own, and takes are combined into one
answer.

To the solver it is an ordinary agent. It owes platypus what any agent does,
passed through to the take that is playing, except the answer channel: a
take's submissions stop at the wrapper, and what goes up is the composition.
Four parts of the contract exist for it:

- **Provenance on submit.** `ctx.submit(submission, produced=…)` says which
  grids the agent made itself rather than filled in — a placeholder, a copy
  of its one grid. The composition reads it; the solver only checks its
  shape, and TAAF never sees it.
- **Take hints.** A take's slot offers carry `take_index`, and the one
  between takes carries `take_boundary`, which the host treats as a free swap
  like a compaction: nothing is resident there. `least_tokens` ranks a game
  `RunConfig.take_handicap_tokens` behind its spend for each take before the
  one it is playing, so first answers are bought before second ones.
- **`LLMAgent.without_engine`**, so an agent that holds agents drops every
  engine spec on the way into the child, not only its own.
- **`LLMAgent.needs_images()`**, so a host-side check of the backend against
  what the games will send (`verify_backend`'s image row) asks the agent
  rather than a field of one kind of agent; the wrapper answers for its takes.

The first take's `run_config` and `llm_config` *are* the wrapper's, so the
deployment the first take was built for is what the solver reads. The second
take plays on the engine the first one sized: it points at the same factory
and prepares with `settle_engine=False`, writing nothing onto the engine, and is
refused if it does not fit. Every take plays on the wrapper's `run_config`,
since hosting settings are the game's.
Limits move from the game to the take: the wall net is the wrapper's own
clock, stopped while the game is parked, and a take that counts its token
budget off the game's one meter needs its subclass to carry the spend across
(`duck.multi_take`). A take that gives up, crashes under `crash_game` or
outruns its net hands on to the next; the game gives up only when every take
did. A take's own cleanup (`on_finish`) failing is contained as the host
contains an agent's, so it never costs the next take its play.

**Every finished take prints a `[take]` block on stdout**: how it ended, what
it cost, what it produced per question, and where each grid of the standing
answer came from, each with a short content hash so identical grids read as
identical. A submission carries grids and nothing else, so that block is the
run's record of provenance.

### Up to ten takes

`TieredTakeAgent` (same module) is the same machinery over ten takes, and is
not configurable. Before each take the game is put in a category from how far
its ended takes agree (`category`): its first take, its second, a further
take while some test output has no grid 1 that two takes produced first, a
further take while some has none that three did — the first two takes
agreeing there counting as three — and from then on a category a take. Its
slot offers carry the category as `CATEGORY_HANDICAP_TOKENS` a step on top of
any handicap the take asks for (`HANDICAP_HINT`), far past any game's spend,
so `least_tokens` serves a lower category before anything in a higher one and
ranks within one as it ranks any game; the solver is unchanged. A game may
play all ten takes, and the ranking decides which get further ones before the
clock runs out.

The answer: takes 1 and 2 count from their first submission, a later take
once its play is over, and so does everything the categories read. Up to two
counted answers compose as two takes do; three or more are voted on (`vote`):
the grid most takes produced first, then the one after it, ranked by first
productions, then second ones. If both counts tie, the grid the most
confident take produced, in the place that counted, takes precedence: each
take's own confidence in its answer (`LLMAgent.answer_confidence`, the duck's
audit's), read as the take ends, and a take that stated none ranks below any
that did. Then take 1's first produced grid, followed by its second produced
grid. A hash orders
the remaining tied grids, keeping their order stable from one answer to the
next. Placeholders and copied fillers never earn take 1's preference. This
rule applies whether augmentation is enabled or not: take 1 is the baseline
configuration and retains the original puzzle layout when later takes are
augmented. Its preference breaks ties only; it never overrides either vote
count.

The queue's category rule is the pass-budget simulation's (`pass_budget` in
`arc2-experiments`: the agreed-pair ranking under variant A), which
rescheduled a run of ten passes a task on 50 slots against the token budget
charged, idle slots included.

## The agent never learns whether it was right

An answer's result is its score, so anything that carried one back would be
an oracle the agent could iterate against. Three places this is enforced,
because it would otherwise leak through each of them:

- **`ctx.submit` returns nothing.** So does TAAF's `execute_action` beneath
  it; the score is recorded at `finish_game`, and a game in play is read
  only on the host's `GameRun`, which the agent's context does not carry.
- **`GameOutcome.state` is not `GameRun.state`.** TAAF distinguishes `won`
  from `submitted`, and "won" means at least one pair came back right — the
  same oracle wearing a different hat. What `on_finish` gets is
  `finished` / `gave_up` / `cancelled` / `crashed`: how the agent's own call
  ended, knowable without scoring anything.
- **The context carries no handle on the live game.** Agents use TAAF's
  *value* types (`Grid`, `Pair`, `GameView`, `Submission`) and never its
  hosting types (`Game`, `GameRun`, `Benchmark`, `Solver`).

What the ban does not cost: LLM-written code inside a kernel can call a
`submit(grids)` host-call, because submitting returns nothing. Checking a
candidate against the *training* pairs before submitting is the agent's own
lever, and a better one.

## Package map

| module | what it is |
|---|---|
| `agent.py`, `context.py`, `contract.py` | the contract: the ABC, what arrives per call, and the configuration |
| `solver.py` | `AgentSolver` — fan-out under a concurrency cap, per-task seeds, the wall budget, the error policy |
| `subproc.py` | subprocess-per-game hosting and its hardening |
| `diagnostics.py` | append-only span streams, the content-addressed block store, the `png-grid` codec, and the zip parts a run's scratch copy of both is archived into |
| `priority.py` | `PrioritySlots` — the run's concurrency, granted by rank rather than by arrival |
| `watchdog.py` | the engine stall watchdog: a run whose engine goes quiet ends itself through TAAF's `stop_run` |
| `host_monitor.py` | the host monitors: a watchdog on the host's event loop, and a sampler of the node's and the job's pressure |
| `status.py`, `slots.py`, `memory.py` | what each game is doing, what is still queued, and what the process tree is using |
| `viewer.py`, `chunks.py`, `live.py` | HTML over the span streams, served live or exported static |
| `viewer_data.py`, `viewer_app.html` | the replay viewer the server hosts at `/_replay/`: one puzzle's transcript context by context, beside the puzzle and the answer standing at the selected point |
| `reasoning_corpus.py` | a game's reasoning text read back out of its span stream, and the register classifier over it |
| `llm.py`, `llm_solver.py` | `LLMAgent` and the backend topology: one host backend behind a Unix socket, one proxy client per game |
| `tracing.py`, `reference.py` | R5 training traces, and capture/replay references |
| `traffic.py` | a recorded run's traces read back as the calls `ullm.traffic` builds a workload from |
| `multi_take.py` | `MultiTakeAgent`: a game played in two takes, and the answer composed from both; `TieredTakeAgent`: up to ten, queued by how far they agree, and the answer they vote to |
| `reference_agents.py` | `FlipAgent` and `PromptedFlipAgent` |

## Hosting

Subprocess-per-game by default; `process_mode="inprocess"` is the debug mode
and runs identical agent code. The child gets everything at spawn and
**nothing about the game comes back** — the parent signals with SIGTERM and
answers exactly two things, neither of them about the game: the re-grant of
a slot the child offered back (`ctx.yield_slot`, a request the child blocks
in while parked) and the cancellation raised into a child parked in that
wait. Nothing an answer about the game could carry is something the agent
may see, so a mis-shaped answer has to fail in the child, and `ctx.submit`
validates against the task before sending; TAAF's own check is the backstop.

Run-level fan-out: `platypus.priority.PrioritySlots` caps concurrency and
grants a freed slot to the best-ranked waiter (`run_config.slot_policy`; the
default is the run's own play order, which is FIFO). Rank never preempts —
a slot changes hands only where a game finishes or offers its own back, and
an offer that is not free (no compaction or take boundary behind it) is
taken only once the game has held its slot for `slot_hold_s`, or never
where that is `None`, the default.
Under `least_tokens` an unplayed
game outranks any that has spent, so a freed slot starts a new game until
the queue is exhausted: every game in the run ends up resident, running or
parked, and parking releases a slot, never a process. There is deliberately
no ceiling on that residency yet (`roadmap.md`). The scheduler prints every
start, pause and resume, `[slots]` crossings of the admitted set, and an
`infra/schedule` event on the run stream with the order and the hints it
ranked on.

Diagnostics — each game's span stream and the run's block store — are written
where `run_config.diagnostics_scratch` points: node-local disk under TAAF's
Slurm preamble (`TAAF_LOCAL_JOB_ROOT`). A stream appends a line per event and
a large block is a file of its own, which a network filesystem makes slow to
write, and slow to delete by the tens of thousands a job leaves: three sweep
jobs sharing a node had 65-86 of each job's 150 slots held by games blocked on
NFS writes of their streams and ran 1.3x slower than alone, and a 15-arm job
left ~42k files, 38.6k of them blocks. The solver
archives them into the job dir as numbered zip parts, `diagnostics-NNNN.zip`:
finished games' streams and loose blocks every
`diagnostics_archive_interval_s`, the rest at the run's end, after which the
scratch copy goes. The run stream and every pack go into every part as they
stand (a finished game's stream may reference a small block a still-running
process packed), so a viewer of a running job has the run's overview and
every finished game whole. Members keep their job-dir-relative paths, and the readers
(`read_game_events`, `read_run_events`, `job_block_store`, `game_run_ids`)
find a file wherever it is: loose in the job dir, on this process's scratch,
or in a part. With no scratch (off Slurm, the tests) the streams are written
into the job dir. A game on scratch cannot be watched live
from another node, and a hard-killed job loses what no part had taken yet; a run
that is to be watched as it plays sets `diagnostics_scratch = ""`, and its
viewer otherwise has the games that finished before the last part.

The hardening is worth keeping intact: a spawn context per child, a liveness
watchdog (a SIGKILLed child may never deliver pipe EOF), stderr capture
whose tail lands in the death report, a stdout relay (the child's prints
reach the parent over its pipe and print there, and its descriptor 1 goes
to the stderr file, since in a notebook the inherited descriptor is a pipe
that blocks every writer once ipykernel stops draining it), RLIMIT_AS where failing to *apply* the
limit is itself fatal, a synchronous reap in a `finally` so a second
cancellation cannot leave an orphan, and **no engine in the child**: a
spawned child re-runs the agent module's imports and rebuilds the agent's
pickled fields in its own heap, so the engine spec
(`llm_config.backend_factory`) is dropped before the agent crosses
(`LLMAgent.without_engine`, an agent's own, because an agent holding
agents has more than one to drop) and `_child_main` asserts vllm is absent.

`LLMConfig.exact_history` turns token continuation from a best effort into
a promise: the host backend is probed at setup and refused if it cannot
extend served ids (a hosted API, replay, a stub), and the trace collector
then refuses any step whose prompt does not extend the previous prompt +
sampled ids within one history epoch. Agents declare their own rewrites
through `Conversation.mark_history_rewritten`; what a rewrite is, and
where the promise cannot hold at all, is ullm's "Token continuation".

### The stall watchdog

A stuck engine otherwise costs the rest of the budget: requests park on it,
the per-request inactivity bound is an hour, and a submission runs its
clock out producing nothing (the twelve-hour run of 2026-09-19 froze two
hours before its deadline with every game waiting). `LLMAgentSolver`
watches the engine from the moment the proxy is up (`platypus.watchdog`):
once a minute it reads the backend's own engine metrics, and when the
generated-token counter has not moved for `RunConfig.engine_stall_timeout_s`
(600 s) while the engine still reports requests, or the in-process engine
has died, it ends the run through taaf's `stop_run` -- the soft deadline on
demand -- so games land `cancelled`, diagnostics are written, the reason is
on the benchmark (`stop_reason`, and the summary's `stopped:` line) and the
job exits normally. Idle is not a stall: the quiet clock counts from the
demand. Read live on a B200 (job 2736506, 2026-09-21): the engine's core
SIGSTOPped four minutes into play, the watchdog fired at 603 s of silence
with 17 requests running, teardown force-killed the frozen core twenty
seconds later, and the run ended with 18 games cancelled beside the 2 won,
exit code 0.

### The host monitors

Every request a game sends and every status line passes through the
host's event loop. When that loop blocks, the engine drains what it holds
and idles, and the status lines go silent: the stall watchdog sees no
demand, so no stall. Two monitors on threads of their own
(`platypus.host_monitor`) run wherever diagnostics do, and are off under
`minimal_diagnostics`.

- **A loop watchdog.** Once the loop's one-second tick is ten seconds
  stale, it snapshots every thread's Python stack with its kernel state,
  wait channel and system call, every fifteen seconds, and prints one
  `[stall]` line when the loop runs again. TAAF's periodic save is the
  exception: it renders the run's page on the loop on purpose, and the
  engine generates through it on the requests it holds. That hold is
  counted, and reported only past three minutes, when it is a stall
  whatever it is inside.
- **A sampler.** Every ten seconds it writes pressure stall information
  for the job's cgroup and the node, reclaim and writeback counters,
  per-operation latency on network-filesystem mounts, local disk
  utilisation, and the loop's worst lag.

Both write to `monitor/` under the diagnostics root, and the archiver
carries them into the parts. On Slurm that root is node-local scratch, so
the filesystem under suspicion cannot block the monitor meant to catch it.
The thresholds are constants: the question they answer is which resource a
stall waits on, not how often to act on one.

## The reference agents

`FlipAgent` submits an all-black grid, then the mirror, twice over, settling
on the mirror. Free and deterministic, so what it should score is known
before it runs — which is what makes it the end-to-end check for a deploy
path. It spends no tokens.

`PromptedFlipAgent` is the same player with one trivial arithmetic turn
through the real turn-runner before each answer, hard-failing unless the
right answer comes back. That makes it the check on the whole LLM layer at
once — backend, socket, sampling profile, capture, token accounting — and it
is what the root-level integration tests drive. `ullm`'s
`ArithmeticStubBackend` answers the sums, so those run on the cheap tier.

The `gpu` tier drives the same agent against a real engine, but configures
it inline on a small VL model rather than through a column — the seam check
is arithmetic, so model quality is irrelevant and a 2B model keeps a column
at ~2 minutes. The duck's own columns (`DuckAgent.build`, `settings.md`) are
covered at config level by the cheap tier and booted by the `b200` tier
(`tests/test_duck_engine_startup.py`); the reference agents carry no engine
of their own any more, so nothing but the duck's columns deploys for real.

## Depending on ullm

`ullm` is the one layer this repo does not fork (`roadmap.md`). It is pinned
to an arc3-platypus revision by git subdirectory, with a
`constraint-dependencies` block holding its direct dependencies at the
combination its own CI runs green against.
