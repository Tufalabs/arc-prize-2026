# Seal design

*Register: as-built — updates with the code.*

How the seal is put together, and the reasons that span more than one place.
Where a setting lives, and its value, is `settings.md`'s; the measurements a
default rests on are `evidence.md`'s. What the seal is, and what it owes the
duck, is the README.

## Levels

### What a level change resets

| resets at a level | survives the level |
|---|---|
| the kernel namespace (`persistence="level"`) | the conversation |
| `history`, always per level, never per game | the workspace files |
| | the journal, when it is kept (`journal`) |

Levels of one game share their mechanics, not their trajectory. A world model
is a claim about the game and has to outlive the level that taught it, which
is why the files survive; a turn list is a claim about one level's board and
misleads the moment the board is replaced, which is why `history` does not.
The conversation surviving is what keeps the duck's compaction load-bearing
here: nothing else bounds the prompt. A new level's prompt says where
`history` starts and that the previous level's turns are gone.

### The winning turn

`history` switches at the kernel's level reset, at the end of the move that
won, not when the winning move returns. For the rest of that move the kernel
holds the level just won (`SealAgent._kernel_level_entries`), its winning turn
ending on the win screen: `turn_payload` cuts the animation to the old level's
frames while the host's own timeline runs on to the new board. The rest of the
winning call, and the reflection turn, read the win there.

The per-move check reports nothing on a winning move: the turn keeps the
model's state when the model predicted the win and `None` otherwise, which is
how the reflection says a win was missed. `test_world_model()` never checks a
winning frame, as it never checks a lost one. A batch sent after the move ended
its level or game is refused before any route's own rule.

### State delivery

- **The kernel holds the live `history`.** The duck rebuilds every frame
  object from a payload per call, which makes any slot the model writes into
  meaningless. The seal keeps the objects alive. A live kernel cannot be
  pickled, and here the three lifetimes coincide: the kernel dies at a level,
  `history` is per level, and `persistence="level"` puts
  `supported_checkpointing` at `{"level", "off"}`.
- **Appends ride the action reply, not an injection.** The seal acts inside
  the model's snippet, where the kernel is blocked on a host call and cannot
  be pushed to; the action reply carries the refreshed state and the kernel
  appends the new turns. The host sends the level's whole turn list and the
  kernel extends the prefix it holds, so a rebuild happens exactly when there
  is no prefix to extend: a new level, or a kernel that died.
- **What a turn carries:** its action, settled frame, animation frames and
  `outcome` (`playing`, `won`, `lost`). `available_actions` is one
  game-scoped global, not per turn. `Turn.state` and `Turn.predicted_frame`
  are the world model's (below).
- **A lost kernel is announced.** A call whose kernel was lost, killed at its
  limit or dead, gets the reset notice in its output, keyed on whether the
  model ran a call since the last reset (`GameREPL.session_used`) rather than
  on a live kernel.

## The kernel

Real Python, where the duck's is a sandbox. All builtins and the full stdlib;
`open()` refuses our source and confines writes to the working directory, and
the source guard refuses our packages' source, framediff's included. The
model's surface is `runtime.game` (private state in `runtime._repl_state`):

- **Frames** are the whale's `Frame`, a board: `.grid` (int8 indices),
  `.diff` and `.diff_sparse` against the frame before, `.color_counts` (a dict
  from each colour present to its count). When a frame was observed is its
  turn's (`Turn.step`, `Turn.level`). `ascii_grid(grid)` renders, with row
  numbers by default.
- **`segment(grid)`** returns `Node`s with attributes, array pixels and
  `bbox`, half-open `(row0, row1, col0, col1)` so that
  `grid[row0:row1, col0:col1]` is the box; `node['pixels']` answers with the
  attribute form.
- **Errors:** the model's frames keep their file and line and ours collapse
  to `<harness>`; output is plain; an attribute guess renders `help(Class)`;
  a refused batch ends by naming the game's actions.
- **The environment is the libraries' alone.** The model's code, cells and
  working-directory modules alike, can neither read nor write it
  (`_guard_environment`): a world model reading a variable its cell had set
  made the REPL and the search's process disagree.

The traps the kernel's files encode: `OPENBLAS_NUM_THREADS=1` is set before
numpy; `publish()` writes into the cell's `globals()`, not `__main__`;
`errors.for_model` runs twice per cell, so `surface_hint` is idempotent; the
prepend has no trailing newline; and a syntax error is the kernel's to report.

### The model's modules

- **One loader for every file of the model's**, `world_model.py` included
  (`_WorkspaceLoader`). A file runs into the module it ran into before, so an
  alias bound in an earlier call, or a module's own `import world_model`, is
  the current version; a file that raises is put back as it was.
- **When a file runs:** the first time a call imports it (at the call's start
  for `world_model`), and in a world-model check again if it was edited
  since, in place. Module-level variables start over when it runs; the prompt
  says to keep what must last in a snippet variable or a file.
- **Nothing is evicted in the kernel**, so the modules the cells hold stay in
  the refresh after every action.
- **One set of preloaded names** (`runtime.game._preloaded_names`). A snippet,
  a helper module and `world_model.py` all start from it; helpers and
  snippets add the moving state and `submit_action`, `test_world_model` and
  `world_model`, while `world_model.py` gets `history` and
  `available_actions` only, since the replay runs the file. `world_model` is
  bound in every cell, re-imported from disk, with a warning and the previous
  version kept when the file does not load.

### Moves are Actions

A move is an `Action` everywhere the model handles one: what
`submit_action([Action('UP')] * 3)` takes (strings and dicts still read the
same), its `executed_actions` and `last_action_result['action']`, a history
turn's action (a click carrying its coordinates), and every printed value,
journal header and gate message: `Action('UP')`,
`Action('MOUSE', row=4, col=7)`. The name is upper-cased, and a click without
coordinates, or coordinates on anything else, raises where it is written.
`available_actions` stays a list of names: it lists kinds of action. The model
steps a plan through `do_move` and submits it in the same cell, which is why
the two are one type, printed the way it is written. The one exception is the
user prompt's executed-actions line, which names moves bare, a click with its
coordinates and the colour it hit, and past ten moves the first and last five.

### Segmentation, a switch

`SealAgent.segmentation` (on): off, `segment`, `Segmentation` and `Node` are
preloaded nowhere the model's code runs and leave `runtime.game`, `help()` and
the prompt, and the checklist's first step reads "Orient on the newest frame."
The search's process still preloads them.

### Confinement

`landlock.py`, `SealAgent.confine_filesystem` (on, off on Kaggle). Once the
harness is imported the kernel asks Linux's Landlock to hold it, and every
process it starts, to reads in the system's directories, Python's installation
and the venv, and writes in the working directory and the kernel's HOME.
Our packages, the engine and the games are unreadable even inside the venv:
Landlock only grants, so the venv is granted entry by entry around them, and
the directories on the way down stay listable because the import system lists
them. Temporary files go to `HOME/tmp`. The source guard stays on top: it is
what explains a refusal to the model, and the confinement is what makes it
hold for `os.open`, shell-outs and C. Models had shelled out past the guard to
read the games' own metadata and TAAF's source. A machine without Landlock
cannot start a kernel with it on, and a submission needs neither of the things
it guards. The search's process is started by the agent, not the kernel, and
stays unconfined.

## The world model

The model may decline: a region of the board it cannot render, or a move whose
effect it has not seen. Each relaxation below loosens the whale's
all-or-nothing world model in one place without loosening what is claimed.

### The contract

`world_model.py` at the workspace root, a workspace file so that it survives
levels. Module-level code may read the full `history` and
`available_actions`. It defines:

- `initial_state() -> dict`, seeding `outcome='playing'` and a numeric
  `energy`;
- `do_move(state, action)`, which mutates the state in place, writes `outcome`
  on a transition, and raises `Unknown(reason)` for a move whose effect it has
  never observed;
- `render(state)`, the int8 board, where the wildcard −1 always matches, which
  is what makes "decline to render this region" expressible;
- optionally `distance_to_win(state)`, lower being closer; the search orders by
  it, and drops it with a note when it raises.

RESET is intercepted, so `do_move` never sees it; acting on a lost state is a
contract violation; and `Unknown` on a move already in `history` fails the
test, since an observed effect must be modelled and the wildcard, not
`Unknown`, is the flexible out. A prediction is a claim, `Unknown` or an
error: no confidence is reported, by decision.

### Testing it

`test_world_model()` loads the files fresh, replays the level's history,
checks every outcome and frame, and attaches the predicted state chain as
`history[i].state` (overwritten on every call, `None` past a failure) and each
render as `history[i].predicted_frame`; a falsy `TestResult` is a pass. It is
how the model gets a past turn's state, which is why the prompt says not to
write a replay loop over `history`. `Unknown` is sticky: the declined move is
in `history` by the next call, where declining it fails the replay, so its
transition has to be modelled before the next move.

**Matching rules.** One set decides whether a render matches its frame
(`harness.test.MatchRules`, applied by `match_frame`) in all three checks that
hold a render to a frame: `test_world_model()`, the check on every move, and
the host's test in the search chain, which has to read them too, since a loose
pass in the REPL held strictly on the host would end the game as a
disagreement. The model sets them from the REPL: `set_strict_energy(bool)`
wherever a world model is, `set_pixel_tolerance(n)` only under
`expose_pixel_tolerance`. The agent keeps them for the game, so a level's new
kernel keeps them, with `strict_energy` and `pixel_tolerance` their starting
values.

- **Energy is loose by default:** frames pass if some energy value makes them
  pass, whole steps staying integers where the model's energy is one, so a
  render that indexes with energy still gets the loose pass. Strict energy is
  the whale's exact check. The prompt says a bar is rarely worth modelling
  exactly and wildcards serve until energy shows a role.
- **The pixel tolerance** counts the cells a render misses outside its
  wildcards; a loose pass is a pass. Where it is offered, a failure that
  misses by at most `SUGGEST_TOLERANCE_PIXELS` suggests raising it, after
  checking those pixels are not the only sign of a change.

### The check on every move

One wrapper runs around every executed move, not around `submit_action`. On a
cell's first move it runs the validation silently in the kernel, every time: a
model can change between cells with `world_model.py` untouched (an edited
sibling module), so nothing carries across cells. Within a cell validation is
remembered only while it holds; a failed check, a stop or a level boundary
leaves the cell unvalidated, so the next `submit_action` checks again.

A passing model arms the check: each executed move is stepped through the
model and its frame compared, and a mismatch interrupts the batch at that move
(`WorldModelStop`). `Unknown` stops the next move, not its own: the declined
move is played, the model having made no claim to hold it to, and nothing after
it runs in the cell. Stepped batches live on the host in one resumable loop
(`_Batch`, `_run_batch`) that parks after every executed move while the kernel
answers continue or stop, so a stepped batch reads exactly like a whole one.
Each checked move's turn carries the model's state after it, so
`history[-1].state` stays current.

`use_world_model` decides what the check does without a passing model:

- **`optional`** (the default): the batch runs unchecked, and a stop is final
  for the python call. The system prompt says what a model gives and costs and
  which games favour one; the user prompt says what each batch ran under and,
  after a stop, offers fixing the model or playing on without it. The mode is
  a run setting, not an opt-out the model declares. `world_model_default`
  adds one line to the system prompt's section and to every level's first
  prompt: attempting a model is the default, dropped only on being convinced
  it does not help on the game. Off as shipped; it is how the Kimi teacher
  traces were played, so `KIMI_STUDENT` turns it on.
- **`mandatory`:** a move runs only as exploration, the seal's premoves, spent
  by the model rather than drawn at random. A game starts with
  `exploration_moves_new_game` and each level replaces what is left with
  `exploration_moves_new_level`; while budget is left and no model has passed
  on the level, the model makes single moves, one per `submit_action` and one
  per python call (counted on the host, `GameREPL.cells_started`, which
  nothing the model runs can move). The first passing model ends the level's
  exploration for good.
- **`none`:** no world-model surface at all: no prompt text, no
  `test_world_model`, no check.

### The example world models

`seeds/world_model_keyboard.py` for a game without MOUSE and
`world_model_mouse.py` for one with it are written once a game, before the
first analyzer turn (none under `none`): complete, working world models of a
different, simpler game, with descriptive comments and no instructions, which
the prompt carries instead. Both are real: `tests/test_seal_seeds.py` holds the
two TAAF games they model and replays a played sequence of each through
`validate_world_model`. The checklist's step that writes the model says to read
the example first, since models overwrote it unread.

### UNDO

Offered without a world model only (`expose_undo`, off; `DUCK_LIKE` turns it
on), named among the game's actions and played like any move, and refused
beside a world model at `prepare_for_backend`. Games keep UNDO by different
rules, and a model that records its own moves in `state` to follow its game's
rule makes states path-dependent, which defeats the search's dedup (the
roadmap has what it would take).

## The search

After a python call whose last `test_world_model()` passed and which played no
move (`use_search`, on). The silent check before a move never triggers one, nor
does a failing test. None of it counts against the call's time limit, and its
harness cells are invisible to the model.

1. **Test again in the REPL**, under the call's limit. A failure means the
   model changed its files after the pass, maybe on purpose: no search, and a
   short neutral note. There is no search after a call that timed out either.
2. **Test in the search's process.** A spawned child (`search_process.py`)
   reads the working directory, loads `PyWorldModel` from it as
   `test_world_model()` loads the file, and tests it over the same history and
   matching rules. A failure ends the game: two replays of the same files
   disagree, which is a harness fault. None of the model's code runs in the
   agent's process, whose game a synchronous load could stall and a large
   one could take out of memory.
3. **Search.** `BFSPlanner` runs in the same child over that model, from the
   level's current state replayed under it. The targets are a win or a boundary: a move the model answers with `Unknown`,
   one whose code raises (reported with its traceback), or in a MOUSE game a
   state that does not render. The repetition check is off, since it rejects
   wins a real game can need. A found path binds `suggested_actions` and
   `suggested_history` (the level's history plus a turn per claimed move, with
   its predicted board and state; a boundary's is one turn shorter),
   overwriting whatever the model had bound to those names; both are cleared
   on the next move and before the next search.
4. **Replay in the REPL.** `suggested_actions` stepped through the kernel's
   own model must reproduce `suggested_history`'s frames; a mismatch ends the
   game.
5. **Tell the model**, in the tool result: the target and the path's length;
   whether the space was searched completely or why not (time, memory,
   repetition, invisible edges); where it started (the level and the history
   length the path starts from); and which kind of nothing, when
   nothing was found. A MOUSE search tries one click per monochromatic region,
   capped per colour and overall, so completeness is claimed for those clicks
   only, and the report says whether the cap bound (`MoveLister.most_regions`,
   `boards_over_cap`).

A disagreement between the REPL and the search's process ends the game rather
than skipping the search: each one so far has been a harness bug, and the
crash is what found it. The child stops itself at `search_seconds` and at
`search_memory_bytes`, which cover its load, test and replay of the model as
well as the search: the planner gets what the rest left of the time, its own
memory check sits under the cap, and `RLIMIT_AS` is behind it, set before the
model's code first runs to what the child reserved after its imports plus the
budget (OpenBLAS alone reserves gigabytes of address space it never
touches). It runs with `OPENBLAS_NUM_THREADS=1`, a search being
single-threaded, and holds a slot of a machine-wide semaphore
(`SEARCH_SLOTS`), which bounds the searches' memory on Kaggle, before the
kernel pool's CPU permit, which is safe to share because the search runs
outside any python call. The search's duration stays off its span, since
wall-clock in a span makes every recorded search unreproducible, and searches
in tests stop on `max_expansions`, so scripted references replay bit-exact.
`seal.search` is imported inside the chain: the move lister's scipy would
otherwise load into every game's child process.

The prompt says the search runs after a passing test, and that `Unknown`
marks a search target: an honest decline is what sends the search somewhere
worth going.

## The frame delta

A frame the model did not predict or predicted wrongly, and a render its own
`test_world_model()` fails on, are described to it as a short block of numpy
from a frame it already knew (`frame_delta.py` over `framediff`,
`packages/framediff/design.md`). framediff chooses: `Decomposer.explain` takes
the baselines as named sources with handicaps and returns the cheapest one's
block, flagged past its budget. The seal names the baselines, sets the
handicaps and budgets, runs the block and words it.

- **Baselines**, in priority order: the mispredicted render, the frame before,
  and the level's opening frame (from the third turn on). The cheapest wins,
  with a handicap on the previous frame's cost and a larger one on the level
  start's (`frame_delta.HANDICAPS`): a light preference for the more
  informative comparison.
- **Budgets:** a block is shown under `decomposer.max_bits`, and under
  `max_bits_understood` once a world model has passed on the level; past the
  budget the text says there is no concise description.
- **What the model reads:** that the frame is most easily understood as a
  delta from that baseline, that this is visual decomposition without
  understanding the game, and that the code is exact (it was run against
  these frames and its assert holds). A frame predicted right gets one line,
  qualified when the render held wildcards. A level's first frame gets
  nothing, having no prediction and no earlier frame of its own level; nor
  does the frame a win ends on, unless the reflection turn follows.
- **Repeats:** a frame equal to an earlier frame of its level is named as that
  repeat (`history[k].frame`, or the level's start). After a RESET or an UNDO
  with no render to compare (`frame_delta.RESTORING`) the line replaces the
  block, since a game restoring a board is no move's effect.
- **The block runs before it is shown**, in the kernel, in a namespace of its
  own (`np`, two preloaded helpers, `history`, `current_frame`); a block that
  does not reach its frame ends the game (`FrameDeltaError`), a harness check.
  The helpers are `from_ascii_grid(text)`, advertised, which reads back what
  `ascii_grid` prints in every layout, and `bresenham_line`, preloaded and
  named nowhere, whose block opens with a comment pointing at its `help()`.
  framediff holds the one implementation of each; the seal validates their
  arguments and re-raises refusals from its own frame.
- **Where it runs:** after every python call one harness cell
  (`frame_delta_after_call`) returns the call's deltas and the newest frame's;
  the host puts each on a span and keeps the text for the next prompt.
- **Time:** a framediff call searches for at most `frame_delta.TIME_LIMIT_S`
  and keeps the best it found when the clock runs out. After a call, a timeout
  reads as no concise description (it ends the game only under
  `crash_on_framediff_timeout`, off); inside the model's call the search stops
  `CALL_MARGIN_S` before the call's limit and never ends the game. A harness
  cell that fails anyway, a hang or an import that cannot map, reads as no
  description, and the kernel is restarted and the model told; a call that
  timed out takes the kernel's stray child processes with it.
- **Records:** every decomposition, shown or not, is a `seal/frame_delta` span
  with its event, reason, baseline, cost, budget, text, code, `timed_out` and
  `duration_s`, and its frames as a `grids` artifact (packed int8, so a
  render's wildcards survive), which `tools/framediff_corpus.py` reads back.
- framediff's options are on: a move's distance priced, objects moved alone,
  an animation's runs written as loops. Settled frames search with lookahead,
  animations plain greedy.

### Animations

- **Three switches:** `settled_delta`; `animation_description` (`none`;
  `available`, offered and pointed at on the prompt's first animated move but
  nothing described unasked; `auto`, the longest new animation of a call
  described automatically); and `animation_marks`, which marks every animated
  move on its executed-actions label with the kernel's frame count. The
  settled delta comes first, then the animation block, and a world model checks
  the settled frame only.
- **`describe_animation(i, stride=None)`** prints the block that rebuilds
  `history[i].animation_frames`. Without a stride framediff tries strides 1, 2,
  3, 4 and 6 until the block fits `framediff.animation.MAX_CHARS`, and the
  text says when the stride is above 1 and how to get every frame; a model's
  own stride is used as is, the block cut at `ANIMATION_CUT_CHARS`. A requested
  description, like `test_world_model()`'s report, is kept whole and does not
  count against the tool's output cap.
- **Per frame** the cheapest script wins among the three frames before it, an
  older frame it repeats exactly, and the frame before the move. A game's
  opening animation starts from its own first frame.

### What else the model is shown of a move

- **The diff image** (`diff_image`, off): after the board, an image of the
  cells the call's last move changed, in their new colour, the rest navy,
  captioned with which move against what. Where framediff described the frame
  and the model was shown that block, against the frame it described it from,
  so the image and the block agree; otherwise a checked move against its render
  when the render missed, a win or a loss against the frame before, and any
  other move against the frame before. No image after a RESET, after an UNDO
  back to an earlier frame of the level, or when nothing changed; the win's
  only on the reflection's prompt. The kernel decides in the after-call cell
  and the host draws (`diff_image.py`). Refused unless the board image rides
  every prompt.
- **Axis labels** (`axis_labels`, off): `board_image.py` draws every board image
  the model sees, so a labelled board and its diff image line up; the whale's
  labels, a margin with a label and tick every 10 cells in DejaVu from
  `seal.fonts`, which ships in the wheel. Refused without images or below an
  upscale of 4, and the engine's pixel ceiling includes the margin.
- **What each move in a batch changed** (`move_changes`, on): the
  executed-actions report lists a batch's moves one per line, each with the
  pixels and colours it changed or `(no change)`, since the settled delta
  explains only a batch's last move.

## Side turns and notices

- **Level reflection** (`level_reflection`: `none`, `verbose`, `concise`). A
  turn after a move that wins a level, the game's last excepted, with python
  and the file tools and no `submit_action`; a reply without a tool call ends
  it. It opens on the winning move's report, its animation block and the win
  screen's delta, the win screen as the image, and a missed win in one line;
  `verbose` asks what was proven, hypothesized and learned. It is charged to
  the new level, and nothing compacts inside it: its compaction limit is
  raised by `reflection_extra_window_tokens`, past which it is cut with a
  note, and a summary due waits for the next level's first call. Refused with
  the journal on.
- **Strategy audit** (`strategy_audit`, `audit_level_tokens`,
  `audit_extra_window_tokens`). The same kind of side turn once a level's
  spend, less the reflection's, passes a threshold, checked before every model
  call: advice, the actions not yet tried and the colours no click has hit, on
  this level and in the game (RESET and UNDO left out), and the level's
  opening board and the current one as images. It triggers on tokens because
  Kaggle has no per-level cap to take a fraction of, and the untried actions
  stand in for search targets below a win.
- **The replay reminder** (`replay_reminder`). `history` keeps the turns
  before a RESET, and the turn after a RESET that closed an attempt repeating
  an earlier one on the level, move for move from its start, names that
  attempt, says it cost score for nothing and that the level is winnable.
  Counted by `platypus.attempts`, in constant time an action.
- **The irregular RESET notice** (always, with a world model). The turn after
  a RESET whose settled board differs from the level's first says the RESET led
  to a different frame, that the difference is cosmetic, and that its cells
  are for wildcards, since the replay rebuilds every RESET from
  `initial_state()`. Dropping the world model there instead was tried and
  taken out.
- **The suggestion to move** (`suggest_moves_after_tokens`, off). Once that
  many completion tokens have passed on the game's meter since the last move,
  summary, or reflection or audit end, every tool result of a playing turn
  ends with how many have and a suggestion to experiment rather than think.
  It is added through ullm's `after_tool_result`, so a refused call's result
  carries it as a run one's does; a call that moved has just reset the count.
  Not in a side turn, which cannot move. The meter starts again at zero
  wherever the agent is restored, and the mark with it.
- **`winnable_line`** (on): the game overview says every level can be won,
  and that an impossibility proof means an assumption is wrong, so the
  unobserved ones are what to test. A pass cannot give up, and a wrong reading
  of the rules otherwise makes a level look provably unwinnable and the pass
  idles out its budget.
- **`notes_guidance`** (`none`, `encourage`, `discourage`): a line on notes
  files after the solving guidance's file lines. Refused with the journal on.
- **The chronological restart** (`chronological_restart`, on). A summary run
  mid-turn rebuilds the history in the order things happened: the prompt and
  the kept call, each where it happened, then the summary and the closing
  message, under a preamble saying the game is already under way. Otherwise
  the turn's opening prompt follows the summary, and a model read it as news.
- **The closing message** after a summary lists the model's files and the
  objects its `python` session holds, and says the session is intact, or
  that it was reset when no kernel is alive by then: at a level's start, a
  reset the more common here than on the duck. A listing that fails is said
  to be unavailable, with whether the kernel went with it and the error's
  last line, rather than ending the game.
- **A plan on the python tool** (`plan_argument`: `none`, `first`,
  `second`): an optional `plan`, cut at `PLAN_MAX_CHARS` and kept on every move
  the call plays as `history[i].plan`. `first` puts it ahead of `code` in the
  schema, since a plan written after the code can only describe what the code
  does. A call giving only a plan is refused for its missing code
  (`repl.plan_only`), where ullm would otherwise run the plan as code.
- **Malformed replies.** A play reply refused on every re-sample (no text and
  no tool call, or degenerate repetition) is dropped and the model nudged, up
  to `malformed_nudges` a game, rather than the game ending; a summary with
  neither final text nor reasoning is reseeded once without thinking. A
  summary can sample apart from play (`summary_profile`), since it opens past
  `</think>` and would otherwise be sampled wholly at the answer's settings.

## The prompt

The duck's shipped prompt with the kernel's lines: the board legend and colour
names, the ascii layout, the game's action set named once, `open()` and module
injection, `history`, `Turn`, `Frame` and `Node`, `diff_sparse` and
`segment()`, read-only records, the trailing-expression echo.
`test_seal_prompt_names_the_runtime` asks a real kernel that every global the
prompt names is bound.

Every world-model sentence is spent only when a world model is in use. Under
one, the duck's lines it makes wrong or redundant are cut: the timer-bar
warnings (the bar is modelled), re-grounding after a score increase (a new
level is marked), the HUD-only check after every action, and "short" sequences
(the check stops a batch at its first misprediction). A duck-like seal keeps
them. Host refusals are `SealPrompts` fields; the kernel's own messages
(`WorldModelStop`, the `TestResult` text) stay in the harness.

A line is changed where it is unclear. Where a model ignores something the
prompt already says, the error it gets when it trips says it instead: a
refused batch names the game's actions, and a `history` misuse is answered
with how to use `history`.
