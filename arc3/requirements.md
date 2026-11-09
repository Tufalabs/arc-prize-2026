# Agent contract and layer requirements

*Register: normative — the source of truth for behavior. Change
deliberately; edits change shared truth and are called out explicitly
in the PR.*

What an agent author writes, what they get, and the requirements this
places on the infrastructure layer and on ullm. Numbered items
(A#/I#/U#) are requirements; pseudocode is illustrative — the code
settles exact signatures, but the *capabilities* shown are required. §8
is the running decision ledger (item 9 still open). Which requirements
are not yet fully active is tracked in `roadmap.md`;
`agent_authoring.md` is the practical manual built on this contract.

## 1. The shape of an agent

Two class *pairs*. `Agent` / `AgentSolver` are LLM-free: `Agent` is the
abstract per-game player, `AgentSolver` (a `taaf.Solver`) is generic
hosting machinery — subprocesses, scheduling, checkpointing, diagnostics.
`LLMAgent` / `LLMAgentSolver` extend them with everything LLM: backend
lifecycle, capture wiring, and enabling the machine-wide kernel-pool
semaphores before any game child spawns (I2b). The random reference agent
runs as a plain `Agent` under `AgentSolver` with no ullm involvement at
all.

All configuration lives **on the agent**, in layered config dataclasses:

- the agent's own behavior fields;
- `run_config: RunConfig` — hosting knobs read by `AgentSolver` (label,
  concurrency, subprocess memory, checkpointing cadence, error policy,
  RNG seed, per-game wall budget — `max_runtime_s_per_game`, enforced
  by infra — checked between moves and cancelling an in-flight move at
  the deadline — game finishes `gave_up` scored as-is; agents
  still never read clocks — and Kaggle deploy assets — kernel/dataset
  sources and setup commands, forwarded to TAAF's duck-typed solver
  attributes);
- `llm_config: LLMConfig` (LLMAgent only) — a picklable backend factory,
  named sampling profiles, and the capture / replay / trace switches.

The solver classes hold **no configuration**; they read it off the agent
template. One fully-configured agent instance therefore describes a whole
experiment, and agent constructors are the single home for environment
presets — no per-agent solver factories:

```python
solver = LLMAgentSolver(DuckAgent.build(environment="kaggle-rtx"))  # preset, or explicit:
solver = LLMAgentSolver(DuckAgent(
    history_turns=30,
    run_config=RunConfig(max_concurrent=32),  # checkpointing defaults to "off" this round
    llm_config=LLMConfig(backend=VllmSpec(model="...", lora=None),   # picklable spec
                  sampling_profiles={"play": Sampling(temperature=0.6, top_p=0.95)}),
))
```

```python
@dataclass
class DuckAgent(LLMAgent):
    # -- configuration --
    history_turns: int = 30
    run_config: RunConfig = field(default_factory=RunConfig)

    def supported_checkpointing(self) -> frozenset[str]:  # resolved at run start
        return frozenset({"move", "level", "off"})
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    # -- state: evolves across moves; must remain picklable --
    notes: dict[str, str] = field(default_factory=dict)
    conversation: Conversation = field(default_factory=Conversation)
    repl: PythonTool = field(default_factory=PythonTool)  # kernel + scratch dir; picklable when idle

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp: ...
    async def _on_finish(self, ctx: LLMMoveContext, result: GameResult) -> None: ...  # optional
    async def _host_setup(self, shared: dict) -> None: ...  # optional; once, host-side
```

Subclasses implement the underscored hooks; the public `do_move` /
`on_finish` are framework-owned (the TAAF `_setup` / `_execute_action`
pattern) — e.g. `do_move` converts a raised `GiveUpError` into a `GiveUp`
result and validates the returned action.

The agent instance is **pure state plus configuration** — where state
includes ullm tool objects (`repl` above), which pickle like any other
state (their live internals are managed by ullm's pickling hooks).
Shared services — LLM client, RNG, diagnostics — arrive per call via the
context and are never stored (A1/§4); kernel-pool caps travel as tool
configuration instead (I2b/U9).

Notes:

- Run-level knobs (`run_config.max_concurrent`, memory, backend) ride along
  inertly on the per-(game, pass) agent copies; the solver reads them from
  the **template** once at setup, never from per-game copies. The config
  objects are plain mutable dataclasses — compose, tweak, then run;
  edits after setup have no effect on a running solver.
- Validation is fail-fast: at solver init/setup, `LLMAgentSolver` requires
  an `LLMAgent` and `run_config.checkpointing` is checked against
  `supported_checkpointing`; an unknown sampling-profile name fails on
  first use (there is no declaration mechanism).
- `run_config.label` (default: the agent class name) becomes the TAAF solver
  label.

## 2. The move loop

Terminology: a **move** is one `do_move` call, ending in one or more
executed actions (or a give-up). A **turn** always means one LLM round
(user prompt → final response); a single move may contain many turns, or
none.

One `do_move` call is one **move**: the agent receives the current
observation, does any amount of LLM/kernel/search work, and acts — either
by returning **exactly one action** to execute (or giving up), or by
executing actions itself during the call through **`ctx.execute`** and
returning `None`. (Delivery-only calls — level-win observations, see the
semantics below — are the one exception: `do_move` runs, but the returned
action is discarded, not executed.) Move boundaries (the moments between
`do_move` calls) are the harness's clean points: snapshots, recovery,
branching, and cancellation all operate there — recovery never loses more
than one move's in-flight work.

```python
async def _do_move(self, ctx, obs) -> Action | GiveUp | None:
    if self.plan and not self._invalidated(obs):
        return self.plan.pop(0)      # multi-action sequences as agent state
    ...                              # LLM / kernel / search work
    return action                    # executed after return; result arrives next move
    # — or, in-move:
    observations = await ctx.execute(action)   # executes NOW; real results
    return None                                # ≥1 execute required
```

Multi-action behavior is the agent's business, in either style:

- **Queue style**: sequences live in instance state, drained one returned
  action per move. Model-run code may *request* several actions in one
  snippet, but results become visible only on the next move.
- **In-move style (`ctx.execute`)**: only callable while a decision
  `do_move` runs (delivery-only calls and `on_finish` get a refusing
  executor). Each call executes immediately against the live game,
  records a full `Move` (same partitioning, history, spans, and token
  attribution as a returned action — tokens spent since the previous
  action attach to the next one, so a batch's LLM cost lands on its
  first action), and returns the resulting observations to the caller.
  Observations returned this way are **not** re-delivered through
  `do_move` (no delivery-only calls for their level-win splits — the
  agent already has them); the next decision `do_move` starts from the
  latest settled observation, which the agent has also already seen.
  Returning `None` without having executed at least one action is a
  contract violation. This is what makes model-visible act-inspect-act
  (the duck's in-snippet `submit_action()`) possible — see §8.

Data types: platypus does **not** reinvent frames, engine actions, or game
state — it builds on TAAF/arcengine (`taaf.game.Frame`, `taaf.GameState`,
`arcengine.ActionInput`), the same substrate the duck uses today. New
types exist only where the contract adds semantics, and they *wrap* rather
than mirror: `Observation` is a delivery envelope over `taaf.GameState`
(adding `outcome`, `level`, `index`, `caused_by`; raw engine
data stays reachable via `obs.state.raw`), `Action` is engine-true —
the engine action (`ACTION1..ACTION7`, `RESET`) plus optional `row`/`col`
data, with the one defined `{'x': col, 'y': row}` bijection to
`ActionInput`; model-facing names (UP/SPACE/MOUSE, …) are prompt-building,
owned by the agent. `Move` records the action together with **every**
observation it produced (usually one; two at a level transition), so
split observations are part of the canonical history.

Semantics:

- `obs` is the observation the move starts from — the result of the
  previous move's action. **Every observation is delivered through its own
  `do_move` call** — one uniform channel, observations never fused. On
  a level transition the engine state splits in two and `do_move` runs
  twice: first with the level-win observation — a delivery-only call whose
  returned action is **discarded** (it is not a decision point; return any
  action — RESET is always legal — though a `GiveUp` is still honored) —
  then with the new level's initial observation, whose returned action
  executes as normal. A game-win observation ends the game and gets no
  `do_move` call (`on_finish` sees it via the history).
  Observations are never fused. Partition rule: the engine returns one
  `GameState` per action; on a level transition its final frame becomes
  the new level's initial observation, and all preceding frames
  (animation and win screen) belong to the level-win observation — the
  whale's rule. `Observation.level` is the level the observation
  *belongs to*: the completed level for a `level_won` observation, the
  new level for the following initial one (TAAF's `levels_completed` has
  already incremented on the win frame). A loss is an ordinary observation
  (`outcome="lost"`); resetting is the agent's decision — typically one
  LLM-free move returning RESET (the whale's lost-state shortcut).
  Infra never executes an action that no `do_move` returned: every action
  in history, spans, and traces has an agent decision attached.
- There is no pass and no yield **at the move**: every move ends in an
  action or a give-up. How much work (LLM calls included) happens inside
  one `do_move` is entirely the agent's business, bounded by its own token
  budgeting — moves are async, so a long move doesn't starve other games.
  (The old duck's 60 s yield was a workaround for its thread scheduling;
  nothing replaces it.) Not to be confused with `ctx.yield_slot` (§4),
  which is a real yield of a different thing: it offers the game's
  *concurrency slot* back to the run and returns. It ends no move, skips
  no decision and cannot refuse the agent anything — the move it was
  called from resumes exactly where it left off.
- Give-up: return `GiveUp(reason)`, or raise `GiveUpError(reason)` from any
  depth — the public `do_move` converts it. The game is finished as
  `gave_up`; `on_finish` runs. An agent that cannot play a class of games
  (e.g. CLICK) gives up on its first move — recorded as `gave_up`, which
  is then what those games' scores mean for that agent.
- Uncaught exceptions from `do_move` are handled per the solver's
  `on_agent_error` policy: `"crash_game"` (contain — this game ends
  `crashed`, other games continue) or `"raise"` (propagate — the run dies
  and a debugger lands on the original raise site; the whale's
  `raise_errors` debugging pattern). There is no infra-side
  forfeit-on-error: an agent that wants to give up on internal failure
  catches its own exception and returns `GiveUp` — give-up stats stay
  meaningful. (A retry-the-move policy was dropped — its interaction with
  strict replay is messy, and an agent can retry internally if it wants
  to.)

## 3. Agent obligations

- **A1** Everything on the instance must pickle. This is not only a
  checkpointing rule — infra pickles an agent at three moments, and only
  the last is conditional: deploy pickles the solver with the agent
  template; the default subprocess mode pickles each per-game copy into
  its child at game start (before any kernel exists); and checkpointing,
  when enabled, pickles at move/level boundaries — the one part
  `supported_checkpointing()` gates (A4: kernel lifetime is what decides
  which cadences an agent can declare; an agent supporting only `"off"`
  still pickles at the first two moments). Objects may manage live
  internals through `__getstate__`/`__setstate__` — ullm tools do exactly
  this (shedding kernel process and host wiring when idle; ullm's
  responsibility, not the agent author's). Live shared runtime resources —
  clients, instrumentation — are never instance state: they arrive on
  `ctx` per call, and `ctx` itself must not be stored. Cross-game
  resources the agent itself needs — a cap on concurrent searches, say —
  go on `ctx.shared`, published once by `_host_setup` as name-resolving
  handles (§4). The ullm kernel pool is neither: its cap is plain named
  configuration the tool carries (I2b/U9), so it pickles with the agent
  like any other field and resolves to the machine-wide counter wherever
  it lands.
- **A2** Deterministic given (observations, config, LLM outputs, timeout
  events): all randomness from `ctx.rng`, no wall-clock reads (none is
  provided), no filesystem/network/env access except through its tools and `ctx`
  facilities (read-only bundled package data — e.g. shipped per-game hint
  files like the whale's energy hints — is fine). Pacing
  decisions are made in tokens and moves. Time enters agent code only as
  **bounds on execution** — kernel timeouts, search deadlines — requested
  through tool/`ctx` facilities with limits set in config; the agent may react
  to a *timed-out outcome* (the whale's timeout-triggered speculation is
  the pattern), but never branches on elapsed time it measured itself.
  LLM-*written* code inside kernels is exempt by necessity — we cannot
  force determinism on it (it may consult timing or unseeded RNG). This is
  accepted: U9 pins the environment best-effort, and reference validation
  (I13) catches the consequences.
- **A3** Unstarted agents (templates) are cheaply deepcopyable.
- **A4** Under a declared checkpoint cadence, the instance must be
  picklable at that cadence's boundaries — end of move under `"move"`,
  end of level under `"level"` — so snapshots are exact by construction.
  Whatever cannot pickle must not exist at a boundary; a live Python
  kernel is one example of a reason it might not (ullm's pickle hook
  fails loudly on one, and the snapshot path is the pickle path, so a
  violation can never yield a silent snapshot — U9). Which cadences an
  agent can honor is declared by `supported_checkpointing()` — a
  **method resolved at run start**, not a class constant, because the
  answer can depend on configuration (the duck's `persistence`); infra
  validates `run_config.checkpointing` against it at setup, and a
  combination the agent cannot honor is a hard error there rather than
  a divergence discovered later. Honoring the cadence during play is
  the agent's job — not policed beyond the loud pickle; replay
  validation (I13) is the behavioral backstop. Anything durable lives
  in instance state or a tool's scratch directory, referenced by
  **relative path only** (§4/U9 — a scratch dir's physical path changes
  on restore or branch, so an absolute path in agent state is a
  dangling pointer).
- **A5** Queued plans are agent state and agent responsibility: deciding
  whether an observation invalidates the queue (level transition, loss,
  surprising result) is the agent's job — infra never manages queues.
- **A6** Behavior-affecting parameters are constructor-settable fields, so
  experiments differ by construction, not code edits. `run_config` /
  `llm_config` are the only *structured* config objects; beyond them there
  is no enforced config/state separation and no config-dump mechanism —
  the pickled/repr'd template is the run's config record, and agents
  organize the rest as they like (often on child objects). Compressed-
  sensing sweeps use dedicated DoE setup, not an auto-sweep over a knob
  registry.

## 4. Context surface

Split like the class pairs: base context for every agent, LLM additions
for `LLMAgent`.

```python
class MoveContext:            # all agents
    history: Sequence[Move]   # canonical append-only (action, observation) record
    shared: Mapping[str, Any] # agent-populated shared resources (see _host_setup)
    game: GameInfo            # game id, hint, action space, level count, pass index (§4)
    rng: Rng                  # the only randomness source; state owned by infra
    diag: SpanLogger          # hierarchical diagnostics: spans + key-values + artifacts
    slots: SlotView           # games wanting a slot and not holding one (§4)
    status: StatusHandle      # refine the run report's coarse "running" (§4)
    yield_slot: Yield         # offer this game's concurrency slot back (§4)
    execute: Execute | None   # in-move action execution (§2); armed on decision calls

class LLMMoveContext(MoveContext):   # LLMAgent
    backend: ullm.Backend     # shared LLM backend; capture/instrumentation sit at this seam
```

`ctx.game.pass_index` is which pass copy of the game this is (TAAF R2.2,
0-based). Passes remain independent plays — no copy's play may depend on
what another copy saw or did — but an agent may read the index to tell the
copies apart, which is what lets one run cover several configurations by
giving each pass its own (`packages/sweeper/`). Infra takes it from TAAF's
stamped `GameRun.pass_index`, never from the entry's position in the run's
list: `shuffle_games` and the heartbeat both permute that list, so a
position encodes play order rather than pass. A consequence worth stating:
per-pass aggregate statistics (TAAF's per-pass means, σ, SEM, head-to-head
t-tests) assume every pass ran the same configuration, so a run that varies
configuration by pass invalidates them for itself.

`ctx.slots` reports how many games want a concurrency slot and do not
hold one — `waiting` / `running` / `limit`, and `drained` for the common
question. It exists because whether abandoning a game is worth anything
depends on whether something can take the slot: once nothing can, a
give-up frees capacity with nothing to spend it on, and the machine idles
for the rest of the run.

**`waiting` counts two populations, and `drained` means neither has
anyone in it**: games the run has not started, and games that started and
parked (§4's `ctx.yield_slot`). Both can take a freed slot, which is the
question being asked, so one parked game is enough to make `drained`
false. Where the difference between them matters — what a run is holding
in memory, since a parked game owns a process and an unstarted one does
not — the per-game status report distinguishes them; folding it in here
would only make this answer wrong.

Three properties are normative. It is **read on access**, not pushed, so
the answer is current at the moment asked — a single move can occupy a
game for many minutes, and any pushed number would be that stale.
Not knowing is **never** reported as drained: the numbers are `None`
until something has published them, so an agent that spends more when
nothing is queued must have positively read that nothing is queued.
And infra owns the numbers — an agent reads, it does not write.

The transport is job-dir files (`platypus.slots`), for the reason
`platypus.live` gives: games run in their own processes, the control pipe
is reply-only, and a socket is bound to the host that created it. A run
whose solver publishes no tally is indistinguishable from one without the
feature.

`ctx.yield_slot(hints)` offers this game's concurrency slot back to the
run and returns when the host re-grants it — immediately, unless
something outranks the game. It is the other half of `ctx.slots`: that
one asks whether the queue is worth giving up a whole game for, this one
hands the machine over for a while without forfeiting anything. Infra
acquires each game's slot before it starts, so the offer is the only
thing an agent does with it, and declining to yield is always safe.

Three properties are normative. It is an **offer, never a limit**: the
game continues either way, no yield can end it, and a run whose host has
no scheduler returns from it having done nothing. The agent chooses
**where** — the point of the primitive is that only the agent knows
where a swap is free (a compaction has already lost the KV prefix) — and
the host chooses **whether**, because the competing consideration is how
long one game may hold a slot, which is a clock and therefore infra's
(A2). And the `hints` an agent reports are the numbers the host ranks on
(the duck sends its level and the tokens spent on it), so they must be
live at the moment of the call rather than as of the last accounting
boundary.

The ullm kernel pool is deliberately **not** here. It is a plain
named-config value on the tool, with a module-level ullm default
(I2b/U9): copies of a tool are equivalent because the name is the
identity, so nothing has to travel per call or be re-attached after a
restore. Semaphores the AGENT owns are a different case and do ride the
context — as `ctx.shared` entries published by `_host_setup`, below.

(Python tools are *not* on the context — they are agent-held state. The
context carries the shared resources tools attach to per call; the
attach/detach mechanics live on the ullm side.)

Usage requirements:

- Agents drive LLM turns by constructing ullm's **turn-runner machinery
  per use over `ctx.backend`** (there is no bespoke `chat()` client). The
  manager-layer defenses ride along: malformed-output retry/recovery,
  prompt-too-long handling, per-turn completion cap, truncated-tool-call
  detection — each gets an explicit keep/adapt/drop decision in the
  fork (the closing audit's discipline — the round's branch notes). Calls normally select a **named sampling profile** from the agent's
  `llm_config.sampling_profiles` — the turn-runner surface offers no
  per-call override knobs, so needing variation usually means defining
  another profile, which keeps every call's resolved parameters in
  configuration and in the capture record. This is a configuration
  convenience, not a wall: sampling parameters are ordinary values and
  an agent is free to construct its own — it then owes by hand what the
  profile path gives for free (behavior-affecting values as
  constructor-settable config, A6; determinism under replay). The hard
  rule either way is seeds: per-call sampling seeds (including
  malformed-retry seed bumps) are derived from the per-game RNG —
  agents never pick seeds (`resolve_sampling` rejects a profile that
  pins one), and retry variation falls out naturally.
- Kernel executions support **host-calls** (LLM-written code invoking
  agent-provided callbacks). The duck's `submit_action(...)` reaches one:
  it validates the requested moves, executes them immediately through
  `ctx.execute` (§2, in-move style), and returns the real result plus
  refreshed runtime state to the snippet.
- Python tools (kernel + scratch directory) are **agent state**: ullm
  `PythonTool` instances held as fields, picklable whenever the kernel is
  not live — which A4's cadence rule guarantees at every checkpoint
  boundary. Pickle/deepcopy preserve the scratch dir *contents*, never its
  path: a restore or branch gets a fresh directory (necessarily — several
  branches of one checkpoint can coexist on a node). An agent may hold any
  number of tools/kernels; infra tracks none of them individually —
  snapshotting the agent snapshots them all.
- `tool.is_fresh` tells the agent whether its bootstrap must (re)run —
  first use, post-boundary kernel, and post-restore kernel are one code
  path. Kernel cwd = the tool's directory, so **relative paths** keep
  working across restores; file tools scope to a linked Python tool's
  directory automatically (the whale's manual `path_mapping` re-pointing
  is the anti-pattern this kills).
- Heavy non-LLM compute (search over a world model, expensive analysis)
  runs through a `ctx` compute facility that applies configured time and
  memory bounds and returns a timed-out/over-limit outcome instead of
  hanging — the platypus home for the whale's BFS duration/memory caps and
  hard-kill isolation. Limits are config; a limit *firing* is an observable
  outcome, not a clock read.
- **Agent-defined shared resources**: `ctx.shared` is a read-only dict the
  agent's `_host_setup(shared)` hook filled **once, on the host process**,
  before the per-(game, pass) fan-out; infra pickles it to every game
  subprocess and mounts it on each context. This is where agent-specific
  cross-game resources live — e.g. a whale-mode cap on concurrent BFS
  searches. Entries must pickle *and remain functional across processes*
  (name-resolving handles like today's `SharedSemaphore`, which carries
  only name+limit and resolves to a machine-wide counter — the sanctioned
  primitive, to be provided ullm/infra-side; never bare locks). `ctx.shared`
  is not agent state and not snapshotted: it is rebuilt by `_host_setup` at
  every run start, including resume.
- `ctx.diag` emits hierarchical diagnostics (I7 has the full picture).
  Infra already wraps each `do_move` call in a span of its own; anything
  the agent opens — `ctx.diag.span("myagent/hypothesize")` — nests under
  whatever span is currently open, so the agent's structure lands inside
  the right move with no wiring. Names are arbitrary and nesting is
  free-form (the `producer/kind` shape is only the convention viewer
  chunk registration keys on); LLM calls, kernel executions, and actions
  are auto-spanned by instrumentation without agent involvement,
  attaching the same way. `ctx.diag` is **write-only** — span timings
  are not readable back (they would be a clock).
- Budgets are **agent-internal** (for now): per-level / per-phase token
  caps are ordinary agent config fields, spend is tracked by the agent from
  per-call usage, and running out is an agent decision (degrade, then
  `GiveUp`). Inter-game budget allocation — dividing the run's wall-clock
  deadline into per-game token allowances — is a future topic; until then
  the only run-level stops are TAAF's soft-end cancellation (I5) and the
  stall watchdog, which ends a run whose engine has gone quiet through that
  same path (`design.md`, "Run operations").

## 5. Requirements on the infrastructure layer

- **I1** Drive the move loop: deliver observations complete and unmodified
  (move splitting, never fused; animation frames and `available_actions`
  exactly as the engine reports them — what the model sees is agent-side
  perception, not an infra filter), execute the
  action each `do_move` returns against
  the TAAF `Game`, and report per-action token counts to TAAF (each move's
  tokens attach to its action; a give-up or cancelled move's tokens go to
  `finish_game`). **A cancelled call's generated tokens count**: the GPU
  produced them, so a total that omitted them would report less spend than
  the run paid for. Two carve-outs, both deliberate. A backend whose
  stream cannot report usage until its final chunk loses them
  (`Backend._usage_from_partial` returns `None`). And a malformed reply's
  discarded attempt is NOT counted, because `ResponseManager` rolls its
  element out of the turn and the diagnostics span stream therefore never
  sees it — counting it in the backend total alone would break the
  three-way reconciliation below on an otherwise clean run. Closing that
  means recording the rolled-back attempt in the spans too (roadmap).
  Token counts come
  from the LLM layer's capture records —
  backend-reported usage; agents never self-report — and diagnostics
  cross-check the totals (the three-way reconciliation: span sums vs
  TAAF `ActionRecord`s vs backend counts). Agents never touch TAAF or
  the engine.
- **I2** Checkpointing per `run_config.checkpointing` (`"move"` / `"level"` /
  `"off"`), validated against the agent's `supported_checkpointing` at setup.
  The default is `"off"`: a snapshot per level per game is storage a run
  takes on only when it wants branch points. `"level"` is implemented;
  `"move"` is rejected at setup.
  Snapshots capture the agent instance (agent-held tools pickle their
  scratch-dir contents with it — §4) plus infra-owned per-game state (RNG
  state, history position) — taken at move boundaries (`"move"`) or level
  boundaries (`"level"`), where nothing unpicklable exists by contract
  (A4), so restores are exact.
- **I2b** Infra does **not** manage agent tools — no discovery, no
  lifecycle enforcement. Its only tool-adjacent duties are resource-shaped:
  make kernel caps genuinely machine-wide (enable cross-process semaphores
  before spawning game children — caps are run protection, same category
  as subprocess memory limits; the pool itself is a plain named-config
  value with a module-level ullm default, so nothing travels on `ctx`)
  and guard against
  scratch-dir bloat at snapshot time (it performs the pickling; snapshots
  multiply dir contents by checkpoint count). Keeping kernel state within
  checkpoint intervals is the agent's obligation (A4) — not policed by
  infra, because violations surface on their own as
  replay-vs-snapshot divergence in reference validation (I13). There is no
  reattachment step: pool config is a plain value on the tool, and
  per-move bindings are per-use bound views (ullm-side mechanics, U9).
> Checkpointing beyond the per-game snapshot primitive — whole-run
> recovery vs training-time branching, their different persistence and
> resume mechanisms, and the TAAF-level prerequisites — is deliberately
> unresolved; see `checkpointing_notes.md`. I3/I4 below sketch intent
> only.

- **I3** Crash/timeout recovery for local games: reconstruct the
  environment by replaying the recorded action prefix up to the last
  snapshot, restore the agent, continue in a new job. Actions executed
  after the last snapshot are discarded (recovery granularity = the
  configured cadence). Online arcades are explicitly out of recovery scope.
- **I4** Branching: start a new run from (run, move index) with a config /
  LoRA delta applied from that point.
- **I5** Cancellation (soft-end, killswitch, playtime cap) is asyncio task
  cancellation of the in-flight `do_move`: it lands at the agent's next
  await (LLM call, kernel execute, compute facility — long CPU work is
  cancellable precisely because it sits behind those awaits; agents must
  not swallow `CancelledError`). The interrupted move produced no action,
  so its tokens are reported through TAAF's `finish_game` — R11.03 exists
  for exactly this (give-up and cancelled-mid-reasoning tokens). The game
  finishes with the correct TAAF terminal state; `on_finish` runs briefly
  shielded (no LLM calls — it exists for clearing resources and final
  bookkeeping), and also runs best-effort after a crash. With checkpointing enabled (P9+) infra
  may restore the last boundary snapshot before persisting; with `"off"`
  the agent object simply keeps its mid-move mutations.
- **I6** Own the canonical history: append-only, structurally shared, O(1)
  exposure per move (agents never rebuild or re-inject history; the whale's
  full-history re-injection is the anti-pattern).
- **I7** Emit the hierarchical diagnostic stream: append-only, written as
  events happen, live-viewable, token/time attribution on every span
  (span close carries self AND inclusive generated/uncached totals).
  Infra owns only the outer levels it drives — run → game → move, plus
  the action-execution event. Below the move, structure is **free-form**:
  agents open arbitrarily named, arbitrarily nested spans via `ctx.diag`,
  and auto-instrumented llm-turn spans / kernel-execution events (emitted
  by infra's turn-manager wrapper and drained from the tools — ullm holds
  no span knowledge, only introspection helpers) attach at whatever span
  is current. Unrecognized span kinds render via a generic fallback, so
  new span shapes require no viewer or infra changes; producers ship
  chunk classes co-located with their emitters when the generic view
  isn't enough. Respects TAAF's `minimal_diagnostics` flag
  (Kaggle submission): the stream is reduced to the minimum TAAF's
  submission rules allow.
- **I8** Budgets are agent-internal for now — infra enforces none, and
  inter-game allocation (wall-clock→token conversion at the run boundary)
  is a future topic. Infra owns time-based limits for non-LLM work only:
  kernel execution timeouts, search deadlines via the compute facility
  (§4), subprocess hard-kills.
- **I9** Process model: subprocess-per-game by default, single-process
  debug mode on request (`run_config.process_mode`); agent code identical
  in both. A child that dies or misbehaves is a loud error for that game
  (or the run, under `on_agent_error="raise"`) — never a silent fallback
  to in-process hosting. Backends reached
  from game processes through the LLM layer (HTTP or proxy — an LLM-layer
  detail).
- **I10** Scheduling: keep the backend saturated across games without
  thrashing the prefix cache; cap concurrency; optional heartbeat game
  (infra itself plays one sacrificed (game, pass) copy with random moves —
  it never goes through the agent template). Slots are granted by rank
  rather than by arrival (`platypus.priority`,
  `run_config.slot_policy`) — a run with more games than slots and a
  deadline wants the machine on the games worth spending on, which is
  not the same question as who queued first. Rank decides the *next*
  grant and never preempts, so a slot changes hands only where a game
  offers it (§4's `ctx.yield_slot`) or finishes. It is computed **at the
  grant, from what each game last reported**, never from a periodically
  published order: comparing is free and only swapping costs, so ranking
  on stale numbers buys nothing and loses precisely the events that
  change a game's standing most. What the ranking charges for a token is
  `run_config.scheduler_level_weighting`, a continuous discount by what
  the token's level is worth to the score (0 = raw spend, 1 = full
  price); which setting is right is empirical, so it is a dial rather
  than a pair of named policies. The default policy is the run's own
  play order, which is FIFO — a plain concurrency cap.
- **I11** Deepcopy the agent template per (game, pass); stamp per-game RNG
  seeds from `run_config.rng_seed` (`None`, the default, = OS entropy; a
  fixed value gives per-game seed = `rng_seed` + game index — reference
  mode fixes it). Before the
  fan-out, run the template's `_host_setup(shared)` once on the host and
  mount the resulting dict on every game's context (§4). (`_host_setup`
  runs inside `run_games` — TAAF's `setup()` is synchronous.)
- **I12** Apply the `on_agent_error` policy. Diagnostic stamping stays at
  the level infra owns: the **move span** gets its outcome (`gave_up` +
  the `GiveUp.reason` string, or `error` + traceback) — never anything
  deeper. Rich give-up context is the agent's to log at the decision site
  via `ctx.diag`. On an unwinding exception, the stream writer closes any
  agent spans left open, marked *aborted* — preserving where in the
  agent's own structure the failure happened.
- **I13** Reference mode for integration tests: fixed seeds, replay
  backend; bit-reproducible end to end at any concurrency (per-game
  record streams + per-game RNG make scheduling irrelevant to every
  compared field; concurrency 1 is a debugging convenience, not a
  requirement). Besides the
  LLM, two nondeterminism seams remain — timeout events and LLM-written
  kernel code (A2/U9) — so reference generation **validates each new
  reference by an immediate replay**: a reference that hit an execution
  limit or fails to replay bit-exact is rejected and regenerated.
  Reproducibility of references is thus a checked property at capture
  time, not an assumed guarantee.
- **I14** Nothing O(N²) in move count: history exposure, prompt-support
  structures, diagnostics. Snapshots are the accepted exception in write
  cost — each is a plain pickle, linear in current state size — with
  totals bounded by retention policy, not dedup (U2).

## 6. Requirements on ullm

("R5" below = section 5 of TAAF's requirements document, the
trace-collection-for-training spec: standardized training traces with
normalized messages, exact token ids, and provider-validation evidence,
written one folder per game run. Status: TAAF's `docs/requirement_status.md`.)

- **U1** A stable agent-facing surface — the part of ullm frozen into the
  contract: the `Backend` protocol (a shared `ctx` resource; per-process
  live clients built from picklable specs — reversing today's
  Backend-pickles-to-`None` behavior), immutable message/content types,
  tool definitions, responses exposing text / tool calls / reasoning
  content / usage with exact token ids / optional logprobs, **exact token
  counting** as a backend capability (requires a tokenizer reachable from
  game processes — the chars/3 heuristic is banned as the budgeting
  mechanism), and the
  turn-runner utilities agents construct over the backend per use,
  carrying the manager-layer defenses (see §4). Everything else in ullm
  may churn.
- **U2** Content blocks (text, image, tool results) are immutable value
  objects with content identity. The replay-reference and diagnostics
  sinks (U3) store each block once and reference it; their storage grows
  linearly with unique content, not quadratically with moves. The trace
  sink follows the TAAF R5 format, which permits this only partially
  today: media is by-reference (image dedup works within spec), but
  normalized messages are mandated inline per step — full text dedup
  would need a later R5 format extension (U5); for now traces are simply
  written without it — quadratic text growth there is a property of the
  format, accepted. Snapshots deliberately do **not** route
  through this layer: they are plain agent pickles (§4) — pickle's
  memoization already shares repeated block objects within one snapshot,
  and retention policies bound growth across snapshots.
- **U3** One capture point on the client; three sinks (R5 trace storage,
  replay reference recording, diagnostics), all deriving from the same
  per-call record so token accounting agrees by construction. Their
  *storage* is fully independent — the diagnostics block store and
  capture's files never reference each other (content stored twice when
  both are on; capture is rare, and neither sink depends on the other's
  lifecycle).
- **U4** A replay backend implementing the same protocol: strict matching
  against recorded traffic, hard error on unseen input with a readable
  diff against the nearest recorded request. Defense paths must be
  deterministic under replay: malformed-output retries draw seeds from the
  RNG stream and their traffic is part of the recorded reference.
- **U5** R5 trace writing incl. provider-validation evidence (payload and
  token-id reconstruction proofs), since that evidence can only be produced
  next to the backend code. The R5 PR has merged (`taaf.trace`); traces
  are written to the merged format without text dedup (U2); a
  block-reference extension stays a possible later upstream proposal.
  **Steps of one run nest**: after the first, a step's prompt token ids
  extend the previous step's prompt + sampled output ids, unless the agent
  rewrote history in between (compaction, eviction, a retry rollback).
  The tokens the model sampled are the truth of what happened; re-deriving
  them from parsed messages is lossy, and a step that fails to nest for
  any other reason is the backend re-tokenizing the model's own words —
  which a training consumer sees as a broken episode. The collector stamps
  every step (`extra.nests_previous`) and every trace (`extra.nesting_breaks`)
  so the property is measured on every traced run, and a traced game in
  which no rewrite happened must record zero breaks. The rewrite is the
  agent's to declare: `Conversation.mark_history_rewritten` bumps
  `history_epoch`, which every step records. A capable backend always
  extends the served ids by tokens (`ullm.continuation`); with
  `LLMConfig.exact_history` a call
  that cannot be continued without a declared rewrite raises, and the
  collector refuses a non-nesting step within an epoch (the cases where
  the promise cannot hold are listed in `packages/ullm/design.md`, "Token
  continuation"). A continued step's
  `messages` are then a view of its ids, not their source
  (`extra.continued`): a consumer re-tokenising messages gets the canonical
  encoding, the ids are what was served.
  Remote multimodal engines carry lossless image bytes alongside the exact
  expanded token IDs. Ordered pixel identities, processor identity, image
  layout and processed tensor/position hashes must agree at the worker,
  engine and training boundaries. Token-count compaction evidence retains
  the corresponding images. Training may opt into retaining malformed
  sampled output with explicit tool-schema feedback and no replay; ordinary
  clients retain their existing repair and retry behavior.
- **U6** Backends as peers behind the protocol: vLLM/OpenAI-compatible and
  native Anthropic at minimum. Live clients built per process from
  picklable specs; in-process engines behind a proxy are acceptable
  implementations. Server/engine lifecycle is Python-native ullm code.
- **U7** LoRA adapter identity — rescoped from the original "runtime
  selection and load/unload" goal: training serves adapters server-side,
  and platypus only consumes and *verifies* them. The backend logs the
  served adapter identity into the trace identity block (OpenAI path:
  `/v1/models` + runtime-export sha, with a per-step hot-swap drift
  guard). Direct vllmx traces name the base in the capability block and
  record the selected adapter on each adapter-bearing step; every engine
  output must match the adapter admitted for that request, even if the
  host changes its selection while the request is in flight. Engine owners
  retain responsibility for loading, draining and retiring adapters.
  Fail-loud stands: a silently wrong adapter must
  surface, never pass unnoticed.
- **U8** Named sampling profiles resolved from the agent's LLM config; the resolved
  parameters of every call appear in the capture record. Per-call seeds
  derived from the caller-provided RNG stream, not chosen by agents.
- **U9** Python tools (kernel + scratch directory). The requirements;
  the as-built mechanics live in `packages/ullm/design.md`:
  - **Pooled, machine-wide caps.** Kernels draw slots from a shared
    pool whose identity is a plain name + limit — config, never live
    wiring — so copies are equivalent and nothing re-attaches.
  - **Snapshot-compatible.** Picklable when idle, with
    contents-preserving (path-changing) copy semantics; **pickling with
    a live kernel fails loudly**, so an A4 violation can never yield a
    silent snapshot. `reset()` is the sanctioned one-call cadence op
    (~40 ms full teardown); `is_fresh` signals that bootstrap must
    (re)run. Path healing at pickle/restore is best-effort mitigation —
    A4's relative-path rule stands.
  - **Per-use bindings.** Host functions are supplied per use via a
    bound view, never mutated onto a long-lived tool. Everything in a
    binding is MODEL-REACHABLE: validate host-side, hold no secrets. Bootstrap
    deletes direct names for the trusted dispatcher, but callable traversal is
    not a security boundary: unknown dispatcher targets and invalid callback
    arguments must be refused as recoverable model errors. A registered callback
    raising after validation remains a fatal harness failure.
  - **Host-call payloads are model-authored, so untrusted.** The payload of a
    host call rides out through the model's own cell — even a well-formed call
    carries whatever object the model handed the adapter — so by default it is
    decoded with a restricted unpickler that reconstructs builtin containers
    only; a payload that would run code (any class with a `__reduce__` /
    `__setstate__`, `builtins.exec`) is refused before it runs, recoverably.
    Every frame-derived failure — bad token, undecodable payload, unknown name
    — is a recoverable refusal logged for visibility, never a fatal error;
    fatal is reserved for a *resolved* host function raising (our own code).
    Full unpickling is a per-owner waiver (`host_payload_trusted`) an owner may
    set ONLY when its host functions genuinely need model-built objects rather
    than plain data, and only until a structural codec replaces the object
    transfer; a data-only surface must never set it. No owner sets it today,
    so the restricted unpickler is what every host call actually meets.
  - **Scoped file tools.** Read/write/edit tools auto-scope to a linked
    tool's directory (kills manual path re-mapping).
  - **Bounded output and time.** Output truncated by characters at the
    tool (tokens are the client's business); execution timeouts with
    limits from config, enforcement encapsulated at the facility seam —
    the timed-out outcome ideally injectable by a test double (open,
    §8 item 9).
  - **Best-effort determinism.** Fixed seeds, hash randomization off,
    no network. LLM-written code may still consult timing or unseeded
    entropy — accepted, with prompt guidance discouraging it and
    reference validation (I13) as the backstop.
- **U10** (reserved) Budget *enforcement* at the call layer. Weakened, not
  met: an agent can now read an inter-game fact and act on it per call
  (`ctx.slots.drained` retiring a per-turn cap), so the "revisit with
  inter-game allocation" trigger has fired without ullm gaining a budget of
  its own. What stays reserved is ullm deciding anything about spend; it
  carries the caller's ceiling and reports usage (U1), and the policy is the
  agent's.
- **U11** Reasoning-content round-tripping (assistant reasoning preserved
  into subsequent requests where the backend supports it) — a known
  score-relevant lever for the duck. Once Duck retains a user, assistant, or
  tool block, its content stays byte-identical in later requests. When enabled,
  a cutoff notice is a new user block, never a rewrite of generated output;
  when disabled, the ordinary continuation nudge is still a new block. Explicit
  context compaction may remove old blocks or rounds and add a compaction
  notice, but does not edit surviving generated content.

  **Malformed-output retry is the second exception, and a deliberate one.** A
  retry may append invisible whitespace to the last perturbable content before
  re-requesting; the edit is reverted before the manager returns, so it is
  visible only to the retry's own request. It is scoped to that request, never
  to generated content, and never to a turn that succeeded. It costs what the
  invariant otherwise buys — for the retried request only, the billed input
  exceeds the pre-call prediction by the appended bytes (the Duck's safety
  margin absorbs the excess and its usage check names the retry count
  rather than treating it as an accounting failure), the recorded
  transcript differs from what that request carried by the same suffix,
  and the KV prefix is invalidated from the edit point. The evidence for the
  benefit is suggestive, not controlled — empty single-token responses, the
  failure a suffix change is most likely to break, read higher in the one
  comparison that ran without perturbation, but treatment is confounded with
  commit and time — so this is a deliberate choice on a plausible mechanism
  and a small downside, not a measured win. See
  `packages/duck/evidence.md` ("Malformed-output retries") for the numbers
  and the caveats, and the duck's `design.md` for the multimodal first-round
  case, where the edit lands deeper than the tail.
- In the Duck, bounded persistent memory is appended
  through one native `memory` tool with a required `text` argument. Its snapshot
  occupies a stable tail of the system prompt and is replaced only when explicit
  conversation compaction fires. It contains only entries whose immutable memory
  tool calls are no longer retained; newer writes remain in those calls, whose
  paired results acknowledge only the host timestamp. Each active entry is sent
  through one mechanism. Ordinary state prompts do not repeat the journal.
  Prompt text, actual tools and counted
  tool schemas must resolve from the same configuration.

## 7. Worked example: the duck as an agent (sketch)

Illustrative only; ~the duck's `analyze()` loop expressed in the contract.

```python
@dataclass
class DuckAgent(LLMAgent):
    # run_config / llm_config fields omitted — see §1
    context_budget_tokens: int = 32768
    memory: list[MemoryEntry] = field(default_factory=list) # bounded append-only journal
    memory_snapshot: str = ""                              # system tail, refreshed on compaction
    conversation: Conversation = field(default_factory=Conversation)
    repl: PythonTool = field(default_factory=PythonTool)     # picklable-when-idle tool

    async def _do_move(self, ctx, obs):
        executed = 0
        self.conversation.append(user_message(
            describe(obs, ctx.history), frame_image(obs)))
        while not executed:
            memory_tool = bounded_memory_tool(self.memory, obs, last_action)
            resp = await run_llm_turn(   # ullm turn-runner over the shared backend
                ctx.backend,
                self.conversation.evicted(self.context_budget_tokens),
                profile="play", tools=[PYTHON_TOOL, memory_tool])
            self.conversation.append(resp.message)           # reasoning preserved (U11)
            if not resp.tool_calls:
                self.conversation.append(nudge_emit_tool_call(resp.stop_reason, resp.usage))
                continue

            async def action_host_call(requested):           # submit_action(...) lands here
                nonlocal executed
                for action in parse_actions(requested, obs.available_actions):
                    observations = await ctx.execute(action) # executes NOW (§2, in-move)
                    executed += 1
                    if terminal(observations):               # win/loss ends the batch
                        break
                return {"result": last_result(observations), "state": runtime_state()}

            with ctx.diag.span("python"):
                out = await self.repl.execute(
                    ctx, resp.tool_calls[0].code,
                    host_calls={"action": action_host_call})
            self.conversation.append(tool_result(out))
        return None                                          # acted in-move (≥1 execute)
```

Notably absent, because infra owns them: subprocess plumbing,
snapshotting (the instance is a plain pickle — memoization and retention
policy keep it cheap, U2), history bookkeeping, span timing, trace/replay
capture, TAAF reporting.

## 8. Decision ledger

1. **(decided, then revisited as predicted)** Originally: one action per
   move, returned from `do_move`, with multi-action sequences as an
   agent-side queue and mid-snippet action results denied. That decision
   named itself the first suspect if duck score parity fell short — it
   was: the taps comparison measured queue mode at +2.15x generated
   tokens and −12% score, so the contract gained **`ctx.execute`** (§2,
   in-move style) and the duck restored the original act-inspect-act
   semantics. Queue style stayed a configurable duck mode until sweep 3,
   whose recommendations retired it: the duck now acts only in-move,
   through `ctx.execute`. (Returning one action per move remains the
   contract's other style — the whale's — per §2.)
2. **(decided)** No clock in `MoveContext` — agents cannot *read* time; token/
   move-based pacing only. Time-bounded execution (kernel timeouts, search
   deadlines) is still available via tool/`ctx` services with config-set
   limits; a limit firing is an outcome the agent reacts to. Next to the
   LLM, the acknowledged nondeterminism seams are timeout events and
   LLM-written kernel code (uncontrollable by design — accepted);
   references are validated by immediate replay (I13).
3. **(decided)** No pass, no yield: `do_move` returns an action or gives
   up. Work inside a move is bounded by the agent's own budgeting. This is
   about the MOVE, and stands: `ctx.yield_slot` (§4, added later) yields
   the game's concurrency slot, not its turn — it ends no move and skips
   no decision, so the loop still has an action or a give-up out of every
   `do_move`.
   Accepted cost:
   a crash mid-move loses that move's in-flight work (a long learn phase
   is atomic); if that ever matters, recovery can replay the interrupted
   move's captured LLM traffic — an infra-side option needing no contract
   change.
4. **(decided)** Sampling seeds are never agent-chosen; derived from the
   per-game RNG.
5. **(decided, later clarified)** The turn-runner surface offers no
   per-call sampling-override knobs — variation is expressed as another
   named profile, keeping resolved parameters in config and capture.
   Clarified: this shapes the sanctioned surface, it does not forbid an
   agent constructing sampling parameters directly (they are ordinary
   values); the rule that binds either path is U8's — seeds come from
   the per-game RNG, never from the agent.
6. **(decided for this round)** Recovery is local-games-only (online
   arcade runs are not recoverable). Consequence worth naming: the Kaggle
   competition benchmark (TAAF R2.57) is online, so exactly where a
   walltime cut hurts most is where it is unrecoverable. Partial online
   support is reopened as `checkpointing_notes.md` Q7.
7. **(decided)** There is no `on_restore`: checkpoint cadence is
   `run_config.checkpointing` (`"move"` / `"level"` / `"off"`), nothing
   unpicklable exists at a boundary (A4), so restores are exact by
   construction and nothing needs rebuilding. Agents declare
   `supported_checkpointing`; an agent that genuinely needs state to
   live across moves (a level-long kernel, say) simply doesn't support
   `"move"`.
8. **(decided)** Action visibility and auto-reset are agent-side: infra
   delivers engine-true `available_actions` and executes only what
   `do_move` returns. Hiding RESET/UNDO from the model means filtering the
   advertised names and rejecting model-authored submissions at the agent
   boundary; resetting after a loss is an LLM-free agent move. No action ever
   exists without an agent decision (keeps history/spans/traces uniform).
9. **(open)** Timeout-outcome injectability (U9): timeouts as
   values-at-seams makes them simulatable/preventable in tests, but the
   implementation cost of wiring this through the ullm facilities (Python
   tool, compute facility, backend timeouts) is unclear — needs further
   consideration before it becomes a hard requirement.
10. **(decided)** Python tools are **agent-held state**, not context
   services: an agent may hold any number of `PythonTool`s (kernel +
   scratch dir), picklable when idle — live internals managed by ullm's
   pickle hooks — with contents-preserving, path-changing copy semantics,
   required because several branches of one checkpoint cannot share a
   directory. Hence the relative-paths-only rule (A4). Tools store no live
   wiring: instrumentation comes from `ctx` per call and the kernel pool is
   a name, so restores need no reattachment step at all.
11. **(decided)** What a backend supports is the backend's to know, and to
    declare. A capability guessed from a field that does not mean capability
    — a factory's `mode` reading `"direct"` — refuses a real engine reached
    over `EngineProxy`, and every later transport able to carry an image by
    ids, so the duck asks no such question. It asks the factory's own
    declaration instead: `BackendFactory.continues_images`, whether the
    backend it builds can continue a served history that carries an image,
    by tokens. `InprocessServerFactory` declares it cannot (its
    `OpenAIServer` continues over /v1/completions, which takes no image);
    every other factory can. Under `LLMConfig.exact_history` the duck
    refuses an image asked of a factory that cannot at
    `prepare_for_backend`, before the engine starts, where U5 would
    otherwise fail the first image request — on the submission box, where
    that is found out last. The server-topology column sends the board as
    text (`duck.presets.configure`), and an image asked of it after the
    build meets that refusal.

12. **(decided)** Platypus carries no training profile. It exposes
    behaviour-named, individually settable controls and public observation
    hooks; the trainer composes these controls and observes the public
    lifecycle without overriding private methods. Environments name machines,
    and supplied backends work in every environment. Setting placement lives
    in [settings.md](settings.md).
