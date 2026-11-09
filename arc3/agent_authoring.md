# Writing an agent

*Register: as-built — updates with the code.*

The practical manual for agent authors: the patterns to follow and the
gotchas that will bite you. The normative source remains
`requirements.md`; this document tells you how to *use* the machinery. Worked examples, simplest first:
`packages/reference-agents/src/reference_agents/random_agent.py`,
`prompted_random_agent.py`, and the full-scale
`packages/duck/src/duck/agent.py`.

## The shape of an agent

Subclass `platypus.Agent` (LLM-free) or `platypus.LLMAgent`, as a
dataclass. Implement `_do_move(ctx, obs) -> Action | GiveUp` and
optionally `_on_finish(ctx, result)`. Configuration and per-game state
are both plain dataclass fields; presets (a named backend + config
combination) are classmethods that construct an instance
(`DuckAgent.build`, over an engine/environment coordinate — `settings.md`).

`build()` constructs on the class defaults and *assigns* afterwards — the
column's layers, then the caller's overrides — so `__post_init__` sees only
defaults. Put a rule about one field there; put a field that follows from
another field, and any rule spanning fields, in `prepare_for_backend`, which
runs after every assignment (the solver calls it before the backend is built,
and `build()` calls it once itself). `settings.md`, "Where a check goes", has
the reasoning.

Rules that everything else depends on (A1, "pure state"):

- **Everything on the instance must pickle**, and the template must
  deepcopy cheaply. You are a *template*: the solver deepcopies you per
  (game, pass) and, in the default subprocess mode, pickles the copy
  into a child process. Live resources (backends, sockets, running
  kernels) never live on fields.
- **Shared services arrive per call on the context** — `ctx.backend`,
  `ctx.rng`, `ctx.diag`, `ctx.history`. Never store the ctx or anything
  from it on `self`.
- **No environment reading, no clocks.** Config is fields; time budgets
  are infra's (`RunConfig.max_runtime_s_per_game`). Reading `os.environ`
  or `time` for *behavior* breaks reproducibility (A2) and replay.

## The move loop you live inside

- One `do_move` call per observation. `obs.outcome` tells you what kind:
  `"playing"` (decide), `"lost"` (decide — usually RESET),
  `"level_won"` (delivery-only: your returned action is DISCARDED; a
  `GiveUp` is still honored), `"game_won"` (never delivered).
- Return an `Action` whose `name` is in `obs.available_actions`, or
  `GiveUp(reason)` (or raise `GiveUpError` from any depth). Actions
  speak engine names (`ACTION1`…) and (row, col) — never x/y.
- Everything runs on ONE event loop shared with every other game in the
  process. Long CPU-bound work inside `_do_move` must hit an `await` at
  least every ~5 s or you stall cancellation and sibling games.
- `on_finish` runs on every terminal path (crash and cancellation
  included, best-effort): cleanup and bookkeeping only, **no LLM
  calls** — the backend may already be gone.
- Randomness: `ctx.rng` only. Anything else (including `random.random()`
  or unseeded numpy) breaks determinism and reference replay.
- `ctx.history` holds the game's moves, newest last. The newest is held
  as it was recorded and costs nothing to read; every older one is held
  compressed and rebuilt on each read, as an equal copy rather than the
  object you were handed. Read what you need of the past once, rather
  than walking the whole history every move.

## LLM calls

- **Always get your turn runner from
  `self.new_turn_manager(ctx, profile=..., system_prompt=...)`**
  (or, if you need TurnManager knobs it doesn't expose, construct
  `platypus.llm.DiagnosticTurnManager` yourself and pass
  `diag=ctx.diag` — the duck does). Then append a `UserPrompt` and
  `await .run()`. What rides along: malformed-output retries,
  prompt-too-long recovery, truncated-tool-call skipping, per-turn token
  caps — and the diagnostics llm/turn span.
- **A raw `ullm.manager.TurnManager` (or driving `ctx.backend.respond`
  directly) produces NO diagnostics record.** Tokens still reach TAAF
  (the meter reads the backend), and precisely because of that the
  per-game reconciliation check hard-errors on a clean run: span totals
  below TAAF-reported is exactly that mismatch. It fails loudly, but fix
  it by using the instrumented manager, not by silencing the check.
- Sampling: select a **named profile** (`resolve_sampling` /
  `new_turn_manager(profile=...)`) from `llm_config.sampling_profiles`.
  Profiles must not pin `seed` (per-call seeds come from `ctx.rng`).
  Gotcha: `GenericSamplingParameters` defaults `max_tokens=30000`; a
  context-budgeting agent usually wants `max_tokens=None` (the duck's
  reply reserve falls back to 512 then).
- Conversations are yours: per-use fresh (`new_turn_manager`) or
  persistent across moves (the duck keeps `self.messages` — a plain
  list field, so it pickles). If persistent, YOU own trimming/eviction
  (`TurnManager.before_round` is the hook to reshape just before each
  call) — and every rewrite of history the model was already sent
  (evicting or editing elements before the last answer, changing the
  system prompt) must call `conversation.mark_history_rewritten()`.
  Appending, and dropping elements after the last answer, need no mark.
  Under `LLMConfig.exact_history` an undeclared rewrite fails the next
  call (the duck keeps its own `history_epoch` because it re-hosts its
  element list in a fresh `Conversation` per block).
- Multimodal: build frame images with `platypus.frame_image(frame,
  scale=...)` — the one shared frame→pixels mapping, pixel-verified
  under replay.

## Diagnostics (`ctx.diag`)

Free-form spans below infra's move span; nesting is automatic via the
ambient current span, so structure costs one `with` statement:

```python
with ctx.diag.span("myagent/plan") as plan:
    plan.kv(candidates=n, chosen=best_name)
    plan.artifact("frame", obs.frame.data, codec="png-grid")
    plan.set_outcome("chose " + best_name)      # an exception still wins
```

- Spans are for *structure*, never for correctness of the LLM record
  (see above — that's the turn manager's job). LLM/kernel activity
  attaches to whatever span is current automatically.
- `kv` accepts JSON scalars (numpy scalars coerced); long strings
  auto-block; dicts/lists/arrays are **rejected loudly** — use
  `artifact(name, payload, codec=...)` with `json` / `png-grid` (2-D
  small-int grids) / `pickle` (repro bundles; code-coupled) / `bytes`.
- Mark no-decision moves trivial so the viewer can fold them:
  `platypus.diagnostics.current_span().kv(trivial=True)` inside
  `_do_move` (the ambient span there is infra's move span). Add what the
  move acted on where there is anything to say. A
  move that spent LLM tokens is never trivial, whatever you set.
- Everything is a no-op on the null stream (no job_dir /
  `minimal_diagnostics`) — never branch on "is diagnostics on".
- Don't put secrets or huge payloads in kv/artifacts; streams are
  plain files in the job_dir.
- Rendering: unregistered span kinds get the generic fallback
  (header + kv + artifacts + children). When that isn't enough, ship a
  chunk class next to your emitter (`@platypus.chunks.chunk("myagent/plan")`,
  override `header_extra`/`body_html`) — a presentation upgrade, never a
  prerequisite.

## Python kernels and tools

- Hold a `PythonTool` (or subclass) as an agent field; pass it in
  `new_turn_manager(tools=[...])`. It draws kernel slots from the shared
  machine-wide pool automatically (semaphore-named caps; don't build
  your own pool unless you need private caps).
- **A4 discipline: a live kernel cannot pickle.** If you declare a
  checkpoint cadence (`supported_checkpointing()`), your agent must be
  picklable at that cadence's boundaries — call `tool.reset()` (kills
  the kernel, keeps or wipes the tmpdir) at the boundary. Every boundary
  falls between two moves, so a reset owed there is paid at the end of the
  move before it, not at the top of the next: the duck resets after every
  move under `persistence="move"`, and after a level-winning move under
  `"level"` (`DuckAgent._do_move`); `tool.is_fresh`
  tells you the next execute starts a clean kernel (your re-bootstrap
  signal). Longer-lived kernel modes narrow the cadences you can declare
  — the duck's `persistence` knob does exactly that. (Independent of
  cadence, the agent is pickled once at game start in subprocess mode —
  before any kernel exists, so this costs you nothing.)
- Host functions (`host_calls` from kernel code into your process): use
  `tool.bound(host_functions={...})` per use — and remember the trust
  contract: anything registered is model-reachable; validate arguments
  host-side; never close over secrets. A host-function exception is
  FATAL by design (it propagates out of the turn; read
  `pending_host_exception` if you wrap things).
- Files + kernel sharing one directory: `tool.linked_file_tools(...)`
  gives Read/Write/Edit tools scoped to the kernel's tmpdir.
- Tool results land in the conversation; kernel executions are
  auto-reported into diagnostics (timings, error flags) — nothing to do.

## Gotchas that have bitten real code

- **Never rebind a bare package name inside a function.** An import is
  an assignment, and any assignment in a function body makes that name
  local to the whole function — so a lazy `import ullm.something`
  inside a function shadows the module-level `ullm`, and a `ullm.x`
  read earlier in the same function raises `UnboundLocalError` far from
  the cause. Lazy function-body imports are legitimate (cycle
  breaking); just bind a different name:
  `import ullm.backends.proxy as _proxy` (or the `from` equivalent).
  At module level the normal `import A` / use `A.B` style stands.
- **ullm dataclasses freeze after `__post_init__`** (`uls.BaseClass`):
  you cannot add new attributes to backends, tools, or managers at
  runtime. Extend by subclassing with new fields. On the platypus side,
  the value types (`Action`, `Observation`, `Move`, …) are
  `@dataclass(frozen=True)` — immutable by design: they validate at
  construction and are shared by reference in history. Agents and their
  config objects (`RunConfig`, `LLMConfig`) are ordinary mutable
  dataclasses — tweak fields freely while composing an experiment; the
  solver reads config at run start, so later edits have no effect.
- Track your own token spend from each response's `token_usage`, never
  from backend counters — backends copy as detached configuration
  clones with fresh accounting (they ride your config through the
  per-game fan-out and subprocess pickles), and reporting to TAAF is
  infra's job regardless.
- The delivery-only `level_won` call really does discard your action;
  don't advance internal state as if it executed (and clear any queued
  plan — the board changed).
- A crash in `_do_move` ends the GAME (policy `crash_game`; the whole
  run under `"raise"`). Both a crash and a `GiveUp` keep the score
  earned so far — the difference is what the terminal state *means*:
  `gave_up` is a deliberate decision and keeps give-up stats
  meaningful, `crashed` says something broke and wants investigating.
  Catch your own recoverable errors and `GiveUp` when the failure is
  yours to absorb.
- Agent behavior in tests: seeded runs must produce identical engine-side
  records (states, levels, action sequences) across `process_mode`
  values; if yours don't, you've read state outside `ctx` (see
  `tests/test_process_modes.py` for the harness).

## Seeing what your agent did

- `make viewer` (or `uv run python -m platypus.viewer <runs-dir>`) —
  game pages, per-move chunk trees, conversations with pre-history,
  frame rows, context-pressure column; in-flight runs render live, and
  `g_<run_id>_live.html` streams the running turn's partial output on
  demand.
- `tools/demo_diagnostics_run.py` generates a clickable offline demo
  run in seconds — useful to see what good instrumentation looks like.
- Capture/replay references (`platypus.reference`): fixed-seed runs with
  `llm_config.capture=True` produce committed, bit-exact-replayable
  reference files — the regression harness for deliberate behavior
  changes. Determinism rules above are what make this work.
