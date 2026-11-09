# Seal roadmap

*Register: living — open work on the seal; entries carry what has to
become true before they can land. What the seal is today is `design.md`,
and what its defaults rest on `evidence.md`.*

- **A structural way to model what cannot be predicted.** Some games have
  parts no rule predicts ahead of time: tr87's dial glyphs (a fixed but
  unguessable cycle per dial), an AI ally or opponent that moves on its
  own. Today the model can only learn such a part one transition at a
  time, after the fact: `do_move` raises `Unknown` for the unseen state,
  the move plays, nothing else runs in that python call, and the next
  call's fresh load folds the observed turn back in from `history` (tr87
  wrote a `learn_from_history()` at module level for exactly this). One
  new transition costs one python call — one LLM turn — so tr87 walked
  its dials with some 25 turns of single presses; batches only paid off
  once every edge on the path was known. Directions, none chosen:
  - an observed part of the state: the model declares which state fields
    the game supplies, and after each real move they are read from the
    actual frame (an `observe(state, frame)` alongside `do_move`) before
    the frame is compared, so a batch runs on through them;
  - an `Unknown` that does not stop the batch, for moves the model marks
    as unpredictable but safe to keep playing through;
  - a claim over a set of outcomes rather than one, checked as "the
    frame is one of these".
  Each changes what a passing model promises, and so what the search can
  rely on. Before it lands: a count of how many games show the pattern
  (single presses in a row under a passing model, each followed by a
  history-derived table edit).
- **UNDO under a world model.** UNDO is offered without a world model
  (`expose_undo`) and refused beside one: games keep UNDO by different
  rules, and a model that records its own moves in `state` to follow its
  game's rule makes states path-dependent, which defeats the search's dedup.
  A version where the harness keeps the undo chain and hands `do_move` the
  state it goes back to is archived as `seal-kernel-undo-world-model`.
  How the UNDO games keep it, read from duck sweep 11's runs (an UNDO counted
  as matched within 64 cells of the frame a rule predicts):

  | game | UNDOs | every move pushes | only moves that changed the board | one step deep |
  |---|---|---|---|---|
  | su15 | 299 | 99% | 99% | 92% |
  | sk48 | 419 | 90% | 96% | 82% |
  | ar25 | 40 | 52% | 85% | 45% |
  | sb26 | 191 | 78% | 78% | 100% |
  | lf52 | 961 | 75% | 67% | 89% |
  | bp35 | 218 | 41% | 38% | 75% |

  anchor_drift and paint_snakes push on every move by their code, and neither
  refunds energy. A chain where every move pushes is exact for su15,
  anchor_drift and paint_snakes only; a game stepping back once (sb26, likely
  lf52 and bp35) or skipping inert moves (ar25) needs its rule known. Before
  it lands: a way to know a game's UNDO rule (or a contract where the model
  states it), and a search that dedups states under it.
- **History across levels.** `history` restarts at every level, where the
  duck's frames run the whole game: the largest structural difference
  between a duck-like seal and the duck, beside the kernel. It was chosen
  for the world model, which is tested on this level's turns, and a seal
  without one does not need it. Directions, none chosen: a game-wide
  `history` with the level on each turn (the model filters), or a switch;
  and with either, a mode where the world model has to cover every level
  played so far, tested on all their turns. Before it lands: what moves
  with it, since the world model's test, the search's trace and the
  kernel's state payload are all built on one level's turns.
- **What the level reflection could add.** `level_reflection` keeps nothing
  of its own past a summary: what matters is the model's to carry, and its
  notes files often do (seal sweep 2's smokes). Variations, from the whale's
  level end (`whale`, `d5bd7a6a`) and from the reflection's design:
  - pin the latest reflection, or a cumulative catalogue of mechanics,
    through summaries;
  - revert a reflection's rewrite of `world_model.py` when its test fails,
    as the whale's review round does;
  - check whether the win screen shown is a celebration frame rather than
    the board at the moment of winning, which the whale shows instead.

  Before any lands: the sweep's price on `level_reflection` itself.
- **Simulate a cell's actions before they run.** With a passing world
  model the harness can play the cell's chain of `submit_action` calls
  through the model first, and warn when it leads nowhere useful (no
  win, no `Unknown` boundary reached, back where it started). The cell's
  code would run against the model's predicted frames rather than the
  game's, so actions it chooses from what the board shows are simulated
  too — a forked kernel, or the cell re-run with the host adapters
  swapped for the model. Two
  shapes, not yet chosen between: always on, around every acting cell;
  or a `submit_action_simulation(...)` the model calls when it wants
  the check. Before it lands: a definition of "useful" — probably the
  search's own targets, so the warning and the search agree.
- **One rule for when the model's files run.** The kernel decides what code a
  world-model load runs in two ways: at the start of every cell it re-runs the
  model's own modules from disk, and within a cell it re-runs one only when
  that module's source hash has changed (`_model_module_digests`). The
  search's process imports everything fresh. Anything a module reads at
  import other than its own source -- a data file, a module that only another
  module imports -- can change mid-cell without the kernel seeing it, and the
  kernel and the search then run different models.

  Seal sweep 2's main2 (`seal_rs3201`, r11l, game `3baddc9d`) ended on it.
  The model's `world_model.py` forwarded to its own `model2.py`, which read a
  rule, `'strict'` or `'A'`, from `legality.txt` at import, so the model
  could test both against the history by rewriting the file. One cell wrote
  `'A'` and called `test_world_model()`: `model2.py` had not changed, so the
  kernel kept the module it ran at the cell's start, and the test passed
  under `'strict'` while the model took it for rule A. After the call the
  search, importing fresh, read `'A'` and found a win; the kernel replayed
  that path to fill `suggested_history` under `'strict'`, the fifth click
  (`MOUSE(52, 37)`, legal under A only) gave a different board, and the
  `SearchMismatch` ended the game at level 1 of 6. Two defects in one: the
  model's own test silently ran stale state, and the kernel and the search
  disagreed. The only search mismatch in sweep 2's main batches.

  Direction: every world-model load in the kernel -- `test_world_model()`,
  the check on each move, the replay after a search -- re-runs the model's
  modules from disk first, as a cell start does, with no "unchanged, so
  reuse" path; the kernel then behaves as the search's fresh process does by
  construction. Kept: re-running into the same module object, so names bound
  in earlier cells follow it, and the previous version put back when the
  file raises. The alternative, every world-model evaluation in the search's
  fresh process, is consistent too but pays a process start per check.
  Before it lands: the cost of re-running module-level work on every load,
  measured on sweep traces (a model may extract a level's statics at import,
  as this one did), and a scripted reference reproducing the case.
- **What the confinement leaves open.** The model can still signal or kill
  other processes running as the run's user (the worker, the engine, other
  kernels): Landlock scopes signals from ABI 6 (Linux 6.12), and the nodes run
  6.8 (ABI 4). Everything of ours already imported stays inspectable in
  memory (names, docstrings, bytecode), the kernel imports `platypus` and
  `taaf` through the harness's own imports, and the game packages' file names
  are listable from the REPL, their contents not: listings are refused only
  outside the readable roots, because `importlib.metadata` lists every
  `sys.path` entry. A temporary file's path names the job's directory, which
  the output's path rewriting does not shorten. Before more lands: a kernel
  newer than 6.12 on the nodes, for signals; the rest is hardening, not play.
- **Other state that can split the REPL from the search.** The environment is
  refused to the model's code, but a cell can still change builtins,
  attributes on shared modules or `sys.path`, which a world model may read and
  the search's fresh process will not see. A disagreement ends the game, by
  decision, since each so far was a harness bug; skipping the search with a
  notice instead is not built.
- **The planner's unused targets.** `BFSPlanner` keeps the whale's fallbacks
  past a win and a boundary (`mode_max` 1-3: untried action types, unvisited
  states, untried state-action pairs), with the `tried_action_signatures` they
  read and the `min_mode` floor carried between searches. The seal searches
  for a win or a boundary only (`search.make_planner`) and sets neither,
  though their comments describe an agent that does; `test_seal_planner.py`
  is all that runs them. Either they go, with those tests, or the seal finds
  them a use: an untried action type is what the strategy audit lists by
  hand. Before it lands: which of the two.
- **The frame delta's worth, measured.** No run differs from another by the
  delta alone. Owed: the seal with `settled_delta` off beside a matching run
  with it on, same launcher and budget; the pair also says whether a second
  baseline beside the cheapest is worth showing when the render and the
  previous frame answer different questions.
- **The reminder's other triggers.** The strategy audit is the long level's
  reminder; a claim of impossibility and a run of idle turns are not
  triggers yet. Before they land: a way to recognise either from the trace
  without asking the model.
- **A level retry**: the game back to the level's start (the engine's RESET)
  and the agent to a copy taken there, so the attempt restarts with the
  conversation, journal and files it had on arrival and carries nothing the
  failed attempt concluded. The numbers bound it (`evidence.md`,
  "`winnable_line`"): a restart inside the same budget loses to continuing,
  and a second full budget helps about as much as a bigger cap, so a retry
  pays only where an attempt is stuck in a way more tokens will not fix.
  Before it lands: its trigger, its budget, and whether the game's action
  count resets with it. It is also the direct test of whether a level's
  outcome is random from its start.
- **Static board elements**, stated to the model as a fact: the colours or
  regions that never changed over the level's history. They often carry the
  win condition or a mechanic's key (a target preview, a legend, a glyph a
  switch repeats); a model can compute them, and stating them points at them.
- **The animation limits.** The automatic blocks run well under framediff's
  character limit on sp80; smaller limits, or one of its own for the
  automatic block, are the first thing to try, with the cut beside them.
- **An animation read by a subagent**, which gets the concise description,
  reads the frames with tool calls and returns a summary, so the frames never
  take the main thread's attention; and a world model that predicts
  animations. Large, and no trace asks for either yet.
- **A level checkpoint that carries the workspace.** The kernel dies at a
  level and the files do not, so a checkpoint at a level today loses the
  model's files.
- **A bound on the model's files.** ullm's kernel caps one file at its
  `RLIMIT_FSIZE`, sized for the disk, and nothing bounds the working
  directory, or `world_model.py` below that cap. A rebuilt conversation shows
  `world_model.py` whole, so a file past the window ends the game at
  compaction ("context overflow remained after compaction even without the
  kept tool call"): seal sweep 4's shipped-seal batch, rs16600's tu93 on pass
  4, where `s.replace(old, new)` with an empty slice for `old` wrote `new`
  between every character of the file -- one game in some four thousand
  compactions, too rare to pay for yet (2026-10-21). The same write,
  repeated or spread over files, fills the disk a deployment shares. Before
  it lands: a per-file cap for seal's kernel that a rebuilt conversation can
  hold, and a total for the working directory checked where the
  flat-directory sweep runs, both sized from the files real runs keep (the
  compaction listing records each file's size).
- **Playing the Kimi student.** `SealAgent.build(kimi_student=True)` sets
  what the teacher traces were played with (`tools/teachers/README.md`);
  two things a student run still needs. Its checkpoint: the B200 column
  serves joint205 in NVFP4 beside the combined ARC2/ARC3 drafter head, its
  MoE backend takes NVFP4 weights only, and a drafter is valid only for the
  checkpoints it was trained against (joint205 and ARC2's REAP-307). And a
  reading of its sampling: it starts on the column's (2026-10-29),
  and the teacher's (temperature 1, top_p 0.95, no top_k, the only values
  Kimi's API takes), which the traces were drawn at, may serve it better.
  Before it lands: the checkpoint the SFT produces, in the format the cell
  serves (`build(checkpoint=…)` plays one on the B200 column,), and a drafter for it
  or drafting off; then the sampling read on a student run.
