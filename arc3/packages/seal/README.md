# seal

*Register: living.*

The seal agent: the duck's control loop over the whale's model-facing
surface, built to ask one question the duck could not — **does a world model
help, holding the surface fixed?**

The duck could not be made world-model-ready by neutral steps. Every piece of
scaffolding a world model needs (files the model writes, integer frames, a
checked model of the game) costs tokens up front and pays only once a world
model exists to use it, so measured against a tuned duck each piece lost. The
seal moves the comparison inside one agent: its own configuration without a
world model is the control. **The seal may play worse than the duck. That is
allowed;** the bar is that it plays without collapsing.

What it takes from each:

- **from the duck:** the move loop, token budgets, hysteretic compaction,
  in-snippet acting, the file suite, the images, the platypus contract;
- **from the whale:** a real kernel (full stdlib, real builtins, source
  tripwires, normalised tracebacks), `history: list[Turn]` as a live object
  graph, the `runtime.game` surface, the world-model contract and its search.

## What it owes the duck

Sameness. With the world model and the frame delta off, the seal is the duck
apart from the differences decided for it: the kernel and its API, per-level
`history`, and the prompt lines that describe them. A duck change that is not
one of those is ported here, as a three-way merge per file
(`git diff <merge>^1 <merge> -- packages/duck`, renamed). Removed rather than
switched off: the file-suite switches (the suite is always on), `frame_api`
and `numpy_scalar_repr` (the kernel that took them is gone), and the duck's
kernel.

`SealAgent.build(duck_like=True)` (`DUCK_LIKE` in `agent.py`) is the seal
played as the duck: no world model, no frame delta, no side turns or
reminder, the duck's window, timeout and eviction chunk, the board in the
duck's letters without row numbers, and UNDO offered. What still differs is
the seal by construction.

Deployments: `SealAgent.build(engine=…, environment=…)`, as the duck's, with
Flash-Next the default engine, on a checkpoint of the seal's own, joint205
(`presets.JOINT205_CHECKPOINT`), with slots of its own. The window is longer
than the duck's, so the 27Bs' columns run a share of the duck's slots
(`presets.WINDOW_SLOT_SHARE`); the other engine settings are the duck's
(`presets.py`, `settings.md`). `SealAgent.build(checkpoint=…)` plays another
checkpoint directory in joint205's place on the B200 column, beside the same
drafter (an SFT checkpoint, say); the engine
arguments are built around it, so an offline launcher (`HF_HUB_OFFLINE`)
needs no Hub cache. Run via `platypus.LLMAgentSolver(SealAgent.build())`
under a taaf benchmark.

## Reading it

`agent.py` is the agent, written to be read top to bottom: its module
docstring carries the reading order and what the seal asks the harness for.

- `prompts.py` — every fixed string the model reads, on `SealPrompts`; what
  the agent decides arrives per render as a `PromptModes`
- the kernel: `repl.py` (`GameREPL`, the REPL over ullm's Python tool),
  `harness/` (the code the kernel runs: `game.py` is `runtime.game`, the
  model's surface; `_repl_state.py` the kernel's private state;
  `test.py` the world-model validation), `model.py` (`Turn`, `Frame`,
  `Action`), `segmentation.py`, `errors.py` (what a traceback shows the
  model), `landlock.py` (the filesystem confinement), `module_names.py`
- the search: `search.py` (the host's chain and its report),
  `search_process.py` (the child process and its slots), `planner.py`,
  `py_model.py`, `move_lister.py`, `trace.py`, `rng.py`
- the frame delta: `frame_delta.py`, over the `framediff` package
- images: `board_image.py` (every board the model sees, axis labels
  included), `diff_image.py`, `palette.py`, `palette_legend.py`
- `presets.py` — what ARC3 asks of a deployment, as layers over ullm's
  engine/environment coordinates; `state.py`, `workspace.py`,
  `file_suite.py`, `memory_tool.py`, `context_observer.py` and
  `action_names.py` as the duck's
- `seeds/` — the example world models written at a game's start; data, not
  code: their bytes are what the model reads
- `tools/` — `level_tokens.py` (completion tokens per level from a run, the
  count the per-level cap charges) and `framediff_corpus.py` (a run's
  decompositions and animations, for searching them again)

The root `tests/test_seal_*` and `tests/references/seal_*` are the seal's
scripted references and integration tests; the package's own suite and its
prompt goldens are under `tests/`.

## Docs here

- `design.md` — how the seal is put together, and the reasons that span
  more than one place
- `evidence.md` — the measurements a current default rests on, and what each
  does and does not settle
- `roadmap.md` — what we intend to change, and decisions still open
