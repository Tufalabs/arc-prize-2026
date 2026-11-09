# Duck roadmap

*Register: living.*

Open work on the duck: changes we intend to make, and decisions still to
take. How the duck works today is `design.md`; the measurements behind
current defaults are `evidence.md`.

- **World-model tooling, and the preparation still owed.** The duck is to
  gain the whale's world-model flow as an OPTION — it may write a
  `world_model.py`, run a scaffolding check, and have a search plan from a
  model that passes, while still acting directly whether or not it wrote
  one. Confidence output is expected to be left out; segmentation stays
  and may later become interactive (sprite-shaped, converging with
  arc3-scene-decomposition); scene decomposition probably follows later.
  Nothing is taken as a dependency on another agent's package for any of
  it — what is shared gets duplicated, so each agent's references stay
  independent.

  Still owed before the tooling itself, each landing behind a flag whose
  default is today's behaviour, so the default reference does not move:

  - **A kernel the model can keep files in.** The restricted namespace
    (`SAFE_BUILTINS` / `SAFE_MODULES`) gives way to a whale-style full
    kernel — real builtins, the tmpdir as cwd and on `sys.path`,
    accident-prevention path rules and normalised tracebacks rather than
    an allowlist. Chosen over widening the allowlist because file tools
    force file-and-import semantics either way; the cost is that shape's
    weaker containment and determinism posture,
    which the allowlist gave for free.
  - ~~**File editing tools**~~ and ~~**file lifetime decoupled from
    kernel lifetime**~~ — both landed behind `file_tools`
    (`design.md`, "The working directory"). Duplicated into
    `packages/duck/` rather than built on ullm's `linked_file_tools`
    seam: for a prompt surface the code is the bytes, and a shared
    implementation would let one agent's wording change force another's
    reference regeneration. Files now outlive every reset unconditionally.
    What the first flag-on run showed is in `evidence.md`, and sweep 10
    promoted the flag to on. Still open:
    - ~~`search_files` and `undo_edit` are barely reached for~~ — sweep
      11 withdrew both by default (`evidence.md`, "Sweep 11"); the gates
      `expose_tool_search_files` and `expose_tool_undo_edit` stay so a
      sweep can offer one again.
    - The arms that matter are `file_tools` x `persistence` at
      `"level"` and `"round"` — the default `"always"` keeps every
      helper in the kernel anyway, so it is the arm where files can
      show least.
    - **The disambiguation clause**, owed whenever a sweep offers
      `undo_edit` beside `expose_undo` again: the shipped game no longer
      shows the model `UNDO` and `undo_edit` together, since sweep 11
      withdrew `undo_edit`. `PromptModes` already carries the file-tool
      names, so the field the sentence needs exists; the sentence does not.
      Grouped with the other UNDO prompt work, which also owes the model
      the fact that an `UNDO` costs an action (`evidence.md`).
  - ~~**Colors as indices**~~ — landed as `frame_api="numeric"`: an
    `int8` `.array`, hex `.ascii`, integer segment colors, and a palette
    legend image where an image already travels. It exists because on the
    character surface `array == 5` matched nothing and raised nothing, so
    a model reasoning in color indices drew a confident wrong conclusion.
    Sweep 6 read it at -4.27 and did not adopt it, but condemned that rung
    rather than the idea: it moves two things at once, colors as indices
    and `.ascii` as hex. It stays in the repo, off, as the arm to split.
    Open from it:
    - **The palette legend has no preset path.** `DuckAgent` refuses
      `multimodal_context` together with `llm_config.exact_history`, and
      both duck presets set the promise — an image in the prompt cannot
      be continued by tokens, so the two are exclusive by physics. The
      scripted and non-default references both exercise the image, and the
      sweep pins `multimodal` off, so nothing is lost today; an arm that
      wants the legend has to turn `exact_history` off explicitly.
    - **Segmentation is not optional.** Scoped and not built: the numeric
      arm makes "segmentation costs tokens the model may not need" a real
      hypothesis, since a model handed integer arrays already writes its
      own connected-component passes (`np.where`, `argwhere`, `nonzero`
      dominate the numeric run's snippets). The cost is prompt wording —
      four sites tell the model to use `.segment()` as its primary
      structural view and would need a positive replacement, not a
      deletion.
  - **No quota on the working directory.** A single file is bounded (the
    file tools refuse to read past 8 MB), but nothing bounds the total, and
    nothing reclaims it within a game — numpy can write there without the
    file tools at all. Accepted: the process memory limit takes only that
    game down. Revisit if a run ever hits it.
  - **The multimodal image path has exactly one reference pin** (the
    most-values-different case), so anything near it is thinly covered.
  - **A bad-names check becomes a need** once duck
    references record a real kernel's output — the kernel swap above is
    what makes the duck's internals reachable to `help()` and `dir()`.
  - **A timeline the scaffolding can replay against.** What the runtime
    surface shows grows toward what a world-model check needs (per-step
    outcome and available actions), rather than adopting a world-model
    agent's presentation wholesale — the duck's kernel-centric, image-off
    surface is the measured default.
  - **A prompt rewrite** as a second `Prompts` implementation, leaving
    room for world-model instructions and making prompt-content
    experiments cheap.

  Then one sweeper campaign co-sweeps the new flags with the existing
  ones, and each is promoted to default, frozen (flag and old form
  deleted), or dropped on the verdict.

- **The journal's layout is the agent's, its wording the prompt
  surface's.** The entry joiner and the newline between entries are still
  composed in `agent.py` while every string around them is a
  `DuckPrompts` field. Deliberate — layout is arguably structure — but it
  is the one seam where "every fixed string the model reads lives on the
  prompt surface" is a half-truth.

- **The prompt golden writes every rendering per config, including the
  identical ones.** 69 of the 96 non-default files are byte-identical
  copies of `default/`, and a new config costs all sixteen whether or
  not its flag changes them. Writing only what differs, plus a manifest
  naming the rest, would take 112 files to about 43 and make a variant
  directory show exactly what its flag does — which is the golden's
  whole job.

- **Backend of record: direct vs server.** The `serving="server"` path
  reached score parity after the reasoning-history fix (3-pass A/B:
  direct 1.32 vs server 1.40, p=0.87; pre-fix 1.26 vs 0.69). Still
  wanted before deciding: a fixed-prompt throughput microbench, and a
  `top_k` profile on the gpu-matrix `server` column as a regression
  guard. Default stays `"direct"`; multi-GPU serving (root roadmap)
  layers on whichever wins. Flash-Next cannot take the server path until
  park mode can run over it (`packages/ullm/roadmap.md`, "Park mode over
  the server topology").
- **`<tool_call>` markup recovery on the vllmx column.** `OpenAIServer`
  salvages parser-missed calls from markup; vllmx — the default
  serving column — does not. First suspect if Qwen tool-calling
  reliability drops (`design.md`, "The LLM surface").
- **Trim-and-retry on a context-length error.** `PromptTooLongError`
  becomes a synthetic response and the duck nudges, where force-trimming
  with an extra margin and retrying the same call would recover the turn.
  Needs a retry hook at the ResponseManager level (`design.md`, "The LLM
  surface").
- **Empty-response crash class.** Reasoning-only output exhausts
  malformed retries at the backend before the duck's no-tool-call nudge
  can fire; a corrective user prompt at that point would recover it.
  Measured ~1/75 games on the serving A/B.
- **The retry perturbation edits the wrong element on a multimodal first
  round.** The finder selects the most recent `StringContent`, which a
  `ListContent(text, image)` prompt is not, so the edit lands in the
  previous turn's tool result and re-prefills the current prompt and its
  image. Confining the perturbation to a fresh continuation element keeps
  the recovery and all three of the invariants it currently costs
  (`design.md`, "The LLM surface"; `evidence.md`, "Malformed-output
  retries").
- **`max_llm_calls_per_move` is named for the wrong scope.** It bounds one
  analyzer ENTRY, and the outer loop re-enters with a fresh state prompt
  and a fresh allowance until the model acts, so it is a re-grounding
  cadence rather than a spend cap. Renaming it is a config-surface change
  and wants its own PR.
- **The system prompt still says the memory snapshot can refresh from "a
  configured history cap."** No such cap exists since the sweep-3 flag
  retirement; the honest clause is token-pressure compaction only. The
  fix is one line in `prompts.memory_runtime_addendum` but changes
  model-visible bytes, so it re-records every duck reference — batch it
  with the next deliberate model-visible change rather than spending a
  re-record round on a clause the model cannot act on.
- **Frame-diff ergonomics in the Python runtime.** Add a lazy frame-to-frame
  diff API covering changed cells, bounding boxes, color transitions,
  moved/appeared/disappeared components, and animation summaries. Keep results
  symbolic and compact so the API cannot become a raw-color or full-board dump
  path. This is model-visible API work: specify it first, then cover it with
  prompt goldens and references before implementation.
- **Incremental Python-runtime state sync.** The host currently sends and the
  persistent kernel rebuilds the full frame timeline on every `get_state` and
  live action reply. That makes a long game quadratic in serialized history and
  discards each `FrameView`'s lazy array/segmentation caches between snippets.
  Track the installed entry count per kernel, append only newer timeline entries,
  and preserve existing frame identities; specify reset/resync behavior and cover
  live actions plus every persistence mode before changing this model-visible path.
- **Count cadence: ~2.5 exact counts per move, one always redundant.**
  The trim's fired-check (or, when it fired, the eviction loop's last
  check) already counts the exact element list the reconciliation count
  re-counts; a trim evicting N blocks takes N+1 full-prompt counts.
  Deduplicating changes how counting and calling interleave, which the
  replay references pin — so it rides the reference-regeneration
  ritual, ideally bundled with other reference-moving work.
- **`max_tokens_per_level` is a judgement, not a measurement.** 90,000 comes
  from experience with the duck; what would settle it is a simulation over
  duck traces: run candidate allowances against recorded plays, on the
  wall-clock axis rather than the spend axis, and strip-test the winner so
  one game cannot carry it.
  The first material to read is under
  `/data/shared/research/arc_3_results/shared/duck_token_budget_*/`: the official
  25 at 8 passes, once at 70k and once at 90k, plus a 10k arm that stresses the
  give-up path. Their `budget` events carry per-turn spend, the level it ran on
  and whether the cap was enforced, which is what a simulation needs.
- **The engine pin is load-bearing and undocumented as such.** `uv.lock`
  constrains `vllm==0.25.1` while `evidence.md` ("Engine and serving")
  still says the engine is now 0.27.1, so the two disagree about what runs.
  Sweep 6 makes the pin deliberate rather than incidental: the host KV tier
  is written and never read, and what it actually does is trigger a 0.25.1
  serving bug that breaks rumination attractors, worth about 44% of the
  score. That dependence dies silently three ways — an engine bump (>=
  0.27.0 rewrote the connector lookup), real cache pressure, or any
  upstream rewrite of that path — so it needs writing down as a deliberate
  dependence on a bug, and the long-term aim stays what `design.md`
  records: reproduce the anti-rumination effect as an explicit mechanism
  and retire the dependence.

- **The wall net has no cost-versus-cap curve.** Nine hours
  (`duck.presets.GAME_RUNTIME_CAP_S`), off on Kaggle, chosen for its premium
  rather than its catch (`evidence.md`, "The per-game wall net"). Nobody has
  computed the curve that would let the number be read off evidence instead
  of judgement, and the swept-configuration cost is a lower bound — it counts
  finished lanes, so the longest-running are excluded by construction.
  Related: no threshold isolates a stalled game, so if a wall cap is ever
  wanted for detection rather than for ending a game scored, that is a
  different mechanism and not a retune of this one.

- **A swept wall cap would confound the sweep.** `LLMSweepAgent` builds its
  own `RunConfig`, so the cap has never reached swept runs; making the
  wrapper inherit the duck's run config would propagate it. For this field
  specifically that is worth reversing — a wall cap correlated with
  `python_timeout_s`, which the sweep draws, is a confound in the
  measurement rather than only a cost. (Experiments-side, in
  `arc3-experiments`.)

- **A stuck level solved by search, and the solution handed back to the
  model** (2026-09-23). When a game is stuck on a level, the duck
  stops calling the LLM for it and plays breadth-first search over the
  game's state space instead: RESET to the level's start, replay a prefix,
  try the next action. The level scores 0 — the search spends far more
  actions than any baseline allows, RESETs included — but what it finds is
  worth more than the level: the model is then told how the level is
  beaten, as the concise set of moves along the path found, not the
  search's full sequence with its RESETs and dead ends, in the hope that it
  learns the mechanic and beats the next level itself. Open:
  - **Detection.** Undecided. Sweep 12's second round showed one shape of
    stuck: games playing 11,000-16,000 actions on a level from inside the
    kernel while the engine sat idle — they were not going to score.
    A level's token budget spent without progress is another candidate.
  - **Whether the search fits.** The state space has to be hashable from
    the frame, and the branching factor small enough: a click game offers
    every cell of the grid as an action, so the search likely needs its
    actions pruned (to objects, say, via the segmentation the kernel
    already carries). A cap on the search's own spend, in actions or wall
    time, before it gives up and the game ends as it would have.
  - **What the model is told, and when.** The move list, where in the
    conversation, and whether the summary compaction keeps it.

  Related: the world-model tooling's search plan above, which asks the
  model for a model of the game rather than searching the real one.

- **An "unstuck" prompt every X tokens in a level** (2026-09-25).
  Once a level has spent X tokens without being won, and every X after,
  the duck sends a dedicated prompt: step back, list the assumptions the
  current approach rests on, and re-test the ones not directly observed —
  above all actions written off as inert. Nothing does this today: the
  followup nudge fires only on a reply without a tool call, and
  `winnable_line` / `retest_actions_line` are static lines in the game
  overview and at level start. Motivation: on the sweep 13 critical levels
  losing passes wrote actions into their notes as inert after a few probes
  and never tried them again, and several concluded a level was
  unsolvable and idled out its budget. Open: X (a fraction of the level
  cap?), the wording, whether it survives summary compaction, and whether
  it is the same mechanism as the reflect-without-compacting knob in the
  sweep 13 notes (a reflection every X tokens with the history kept).
- **The prompt budget above 125k** (2026-09-27). Sweeps 12 and 13
  moved it from 56,832 to 125k and stopped there. Read at one per-level cap
  it was still rising at 180k (`evidence.md`, "Sweeps 12 and 13"), but only
  43 of sweep 13's 244 image-on arms played above 100k (the first two
  batches, where the presence penalty varied too), and
  nothing above 180k was played. Needs more investigation at high values,
  with a throughput model recalibrated from new Kaggle runs: the one the cut
  score uses (`arc3-experiments` `sweep12/kaggle_cut.py`) was fitted
  before speculative decoding, which buys far more tokens at any mean
  context and flattens the curve, so it prices a larger budget's context
  away. Sweep 13 also lowered each arm's per-level cap as its budget rose
  (`level_cap_for`), which reads a higher budget as worse at its terminal
  score.
