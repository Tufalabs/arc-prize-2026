# Design

*Register: as-built — updates with the code.*

How the duck is put together on the platypus contract. `agent.py` is
written to be read top to bottom and its module docstring carries the
reading order, so this file does not restate the code: it carries the
mechanisms whose reasons span more than one place — why a game is bounded
in tokens, how the context behaves across a whole game, what the
game-over guard is actually protecting against.

Measurements that justify a current default live in `evidence.md`. Open
items live in `roadmap.md`. The contract requirements referenced by tag
(A#/I#/U#) are in the root `requirements.md`.

## The configuration surface

Every model-facing mode is a dataclass field, and the fields target
separate mechanisms rather than one bundled "mode": `persistence` changes
the Python session's lifetime, `frame_api` changes the whole frame schema
as one unit, `completion_cutoff_notice` changes only the continuation
message's wording, and `expose_undo` controls whether UNDO is seen and
submittable.

A mode earns its field by being a choice worth making. Where a sweep finds
one half of a switch never better than the other, the losing half leaves
rather than staying as an option that can only be chosen wrongly.

Every combination resolves its system prompt, tool description, actual
tool list, accepted actions and measured tool schema from the same
settings — there is no combination that leaves one of those describing a
different agent than the others. One-at-a-time settings compose: each
replaces only the claims owned by that feature.

**These switches are not a security boundary.** They control the
supported model-facing interface. U9 deliberately permits traversal
through a preloaded callable's globals, so a determined snippet can reach
implementation helpers, and the sandbox's import allowlist bounds what a
snippet may name without promising that an already-trusted module object
is unreachable by reflection. What *is* guaranteed is independent of every
switch: the canonical state path and every parent callback validate their
arguments before touching game state.

## The prompt surface

Every fixed string the model reads is a field on the prompt implementation
the agent carries (`DuckAgent.prompts`, a `DuckPrompts` unless replaced),
and the methods compose those fields at render time. An experiment or a
sweep arm therefore overrides one fragment by construction —
`DuckAgent(prompts=DuckPrompts(visual_game_addendum=...))` — and the
override lands everywhere that fragment appears. `Prompts` is the protocol
behind it: a different way of conducting the conversation is another
implementation, a different choice of words is a field.

**What the agent decides is not on that surface.** The runtime the prompt
describes — which globals the kernel installs, how long its session lives,
what the tool truncates at — arrives per render as a `PromptModes`, built
in one place (`DuckAgent._prompt_modes()`) from the agent's own fields, and
carrying no defaults of its own. A mode field on the prompt surface would
be a second copy of a shipped default: settable, so a sweep arm could set
it and render against a runtime that does not exist, and copied, so it
could disagree with the agent once a default moves. Neither is
representable.

**A string the agent reads back is owned with its reader.** The pruned-note
marker and the journal acknowledgement's index are written into
model-visible text and later scanned for — the note so that repeated
pruning rewrites it in place instead of stacking copies, the
acknowledgement so the agent can tell which journal entries the
conversation still carries. The predicate and the parser
(`is_rounds_pruned_note`, `parse_memory_append_index`) live on the same
surface as the text and derive what they look for from the same fields, so
a rewording moves both halves together.

**There is one implementation.** `DuckPrompts` is the prompt, and it
carries no switches: a claim it can make conditionally is conditional on a
render fact in `PromptModes`, never on a field of its own.

Prompt sweeps do not run against it. They run against a second surface
with one switch per claim, byte-identical to `DuckPrompts` at its defaults
and tied to it by a test, so each change is priced on its own; the winners
are then shipped by setting them as that class's defaults, which breaks
the test on exactly the renderings that moved, and moving `DuckPrompts`
until it passes again. That is what makes the shipped prompt the swept one
rather than a transcription of a report. The surface sweep 11 used is at
the tag `sweep11-configurable-prompts`; it is not carried on the mainline,
where it would be a second copy of every string, free to drift.

`game_action_names` is the game's action set, frozen at the first
observation, and it reaches a surface through `PromptModes` like any other
render input. It is deliberately not `current_valid_actions`: the engine
narrows the legal list after a loss, and a system prompt tracking that
would rewrite its own cached prefix every time the game refused an action.
What varies per turn stays where the kernel already exposes it, as
`valid_actions`.

**Fixed fragments are filled; overrides are not.** Composition applies the
`.format()` filling to the text this package phrases, and appends
user-settable fragments afterwards — an override is prose that may
legitimately contain a literal brace.

## Acting

`submit_action(...)` is a preloaded Python global inside the kernel — not
a native tool and not a module. The parent binding behind it is named
`action` behind a fixed-target adapter, and the bootstrap deletes the
direct trusted-dispatcher name from the namespace. Because callable
traversal stays model-reachable (U9), every parent binding validates its
own signature and payload; the raw dispatcher and its validation cannot be
switched off by any mode. The kernel accepts the common accidental
`import submit_action` as a compatibility alias to the same callable,
rather than spending a tool round on a sandbox error.

A call executes immediately through `ctx.execute`, returns the compact
result, refreshes the runtime globals, and early-stops the batch on a
terminal event — so one snippet can act, look at what happened, and act
again. The analyzer turn ends after the first acting call.

Three details of that path are worth knowing before changing it:

- The terminal no-op guard is per **move**, not per snippet. A move that
  has already hit a terminal event will not re-submit within the same
  move even across separate tool calls.
- Validation is shape-up-front: an unknown action name rejects the whole
  batch before anything executes, and `MOUSE` row/col clamp to 0-63
  before the first action runs, rather than per item at execution time.
- A later tool call in the same response, after one that executed
  actions, is answered with an explicit `[not executed: ...]` error
  ToolResult. The exchange stays in history; it is not dropped.

The action result never reproduces elapsed or remaining wall-clock
seconds. Agents may not read a clock (A2), and the budget the duck
actually runs under is denominated in tokens.

### The game-over guard is a guard, not advice

The duck does not auto-RESET past a loss. The turn's current frame *is*
the lost board, `valid_actions` reports `["RESET"]`, and the user prompt's
act-and-grounding lines are replaced by the game-over lines. After the
model's RESET, play continues with no further game-over prompting.

Non-RESET actions in that state are refused in-snippet with a
`ValueError`. **This is load-bearing, not a nudge**: arcengine returns an
empty frame for any non-RESET action after GAME_OVER, and TAAF raises on
an empty frame — so without the refusal the game crashes. The in-snippet
`stop_detail` tells the model to RESET on its next turn, promising
nothing on its behalf.

## The bookkeeping spine

What the game did is held in two shapes, and the boundary between them is
deliberate.

Host-side it is typed (`state.py`): the aligned timeline of
`TimelineEntry`, the batch being accumulated, the summary the next user
prompt reads, the journal's entries. That is internal state — checked by
pyright, free to gain a field, and carrying exactly what its readers use.

What crosses to the model stays a plain dict, built at the serialization
seam: the kernel's state payload and `last_action_result`. Their key order
and key presence are model-visible — a snippet can print either, and
`last_action_result` grows keys only when a batch stopped short — so they
are wire format rather than internal state, and `FramePayload.payload()` is
the single place the payload's key order is decided.

## The analyzer loop, and what bounds it

A move runs an *analyzer entry*: the duck builds a state prompt, the model
takes tool-call rounds, and the entry ends as soon as the model acts. If
the model finishes an entry without acting, the outer loop re-enters with
a fresh state prompt.

`max_llm_calls_per_move` bounds **one entry**, not the whole move.
Exceeding it does not end anything — the loop re-enters, which rebuilds
the state prompt and grants a fresh allowance. That has a second effect
that matters more than the cap: a new entry opens a new user-led block,
and whole-block eviction has nothing to act on until one exists. The
field name and the accounting therefore disagree; the rename is a
config-surface change held for its own PR (`roadmap.md`).

A response that stops on `max_tokens` without calling a tool gets a new
user message naming the completion-token count it hit. The truncated
response itself is retained unchanged — the notice is never spliced into
generated content. `completion_cutoff_notice=False` sends the ordinary
nudge instead, and still never rewrites the response.

## Run budgets: tokens, not seconds

A game is bounded by `max_tokens_per_level` — completion tokens allowed
per level, indexed by the level the game is on, with the last entry
applying to every level past the end.

**Seconds are not the game's to spend.** Games are not all concurrent, so
a game played while the run winds down has the engine largely to itself
and gets several times the work done that the same game gets at full
concurrency. Under a wall clock, what a game achieves depends on what its
neighbours were doing — which makes two runs incomparable and inflates the
spread across passes for reasons that have nothing to do with the agent.
A token budget is the agent's own to spend and does not move with the
schedule.

Enforcement runs at three depths, because a move can run many model calls
and a game many moves: an entry the level cannot pay for is refused before
the kernel is set up, the remaining allowance becomes the turn's ceiling,
and ullm lowers each round's `max_tokens` against that ceiling. Exhaustion
raises `GiveUpError` out of the analyzer loop, so the game finishes
`gave_up`, scored as it stands.

**A wall net sits under the token budget, and only under it.**
`RunConfig.max_runtime_s_per_game` defaults to `duck.presets.GAME_RUNTIME_CAP_S`
(nine hours) on every preset but Kaggle's, which passes `None`. It is a
backstop, not a budget: the token allowance is what a game is meant to
spend, and the net exists for the failure that allowance cannot see — a
game that spends its hours issuing actions from inside the kernel generates
nothing, so only the job's walltime retires it.

What the net changes is how such a game ENDS. Walltime cancels it, scoring
nothing and taking its whole pass out of the analysis; the net ends the same
game `gave_up`, scored as it stands. It does not identify a stalled game,
and no threshold does — action-spamming is a continuum, most such games
finish on the token budget, and throughput floors fire on healthy games
while missing runaways. So the number is chosen for what it costs rather
than for what it catches: nine hours clears the scoring finishes that full
concurrency pushes past six and still ends a 9.9-hour runaway (`evidence.md`,
"The per-game wall net").

Off on Kaggle because the arena's own clock already bounds the submission
there, and cutting a game would forfeit levels it was still being scored on.

`uncap_when_queue_drained` retires the cap once nothing is queued behind
the game, for a deployment that wants a cap that stops binding when the
queue can no longer use the slot it frees. No preset sets it: the Kaggle
preset carries no per-level cap at all (`max_tokens_per_level=None`) and
ranks its slots instead, and off a deadline lifting a cap makes a game's
spend depend on where in the run it was scheduled, which is noise in any
comparison between runs.

`max_tokens_per_game` is a separate opt-in comparison control, `None` by
default. It and the per-level counter both read the backend's per-game
completion-token meter, so their currencies agree when both are on: game
spend continues across level changes while the level counter resets. A
successful round discarded because a queue-drain lift raced its cap is
charged to both, even though its response is no longer retained. The game
cap never retires on a drain, so the smaller remaining allowance binds the
next turn.

### A turn that wins its level mid-flight

Such a turn is recorded against the level it **ran under**, and the level
that follows is not billed for it. In live mode `submit_action(...)`
executes inside the turn, so the win lands before the turn returns and the
counter has already reset. Billing the new level would start every level
after the first pre-spent by its predecessor, biasing against exactly the
deep levels a per-level allowance exists to protect.

The generation ran under the old level's cap, so that is where the
`budget` event puts it: level, counter and allowance are all read from the
turn's start, so one record describes one level rather than two, and
`level_crossed` marks the boundary. The live counter keeps the reset's
zero — the level it names is finished and nothing consults it again. TAAF's
per-action token records are unaffected either way; infra meters at the
action, not at turn end.

There is no run-level token budget here. TAAF's soft end is the run-level
stop.

## The context budget

The budget for eviction is `context_window_tokens` minus the reply reserve
minus a 512-token safety margin, floored at 1024. The reply reserve is the
play profile's `max_tokens` (512 when unset), which is why raising the
reply cap narrows the prompt budget by the same amount unless the window
moves with it — the two are one coordinate system, and the engine's
`max_model_len` is derived from the window at `prepare_for_backend`, so a
prompt the agent considers legal is one the engine will accept. The cap is
`reply_cap_tokens` on the agent (default 32768). Each call stamps that cap
onto a copy of its profile, so changing it moves both the generation limit
and the context reserve without modifying the stored distribution.

**Token-native mode** (`token_native_context`, declared by every vLLM
preset) measures with the backend's own exact weight instead:
`Backend.token_count` answers the id-ledger length on a continued call —
the very list `respond` then sends to `/v1/completions`, weighed once and
never re-tokenized — and the server's own render after a declared
rewrite. The trigger is then ledger + reply reserve + safety margin with
nothing else in it, so the growth band between compactions is maximal by
construction, and every continued call is checked against the engine's
account (`check_continuation_served` in ullm: echoed ids and billed
`prompt_tokens` must equal the sent ids — drift is not merely detected
but unrepresentable, since no served content is ever re-tokenized).
Each fired trim logs `compaction_spacing_tokens` and
`compaction_spacing_rounds` — how far and over how many LLM calls the
prompt grew since the previous trim's post-trim reading — so the
sawtooth's period is visible per game. Cloud backends are never asked to
count (their recorded streams are pinned); off-mode configs keep the
bill+bound arithmetic below as the general contract.

Counting otherwise runs on engine truth plus a provable bound — no
estimation anywhere in the decision path (`_decision_prompt_tokens`): the newest
engine-billed response pins everything up to and including its own
generation at exactly `usage.prompt_tokens + usage.completion_tokens`
(completion included because the duck re-sends reasoning for all turns),
and everything appended since — tool results, the next user prompt — is
charged the raw character count of its rendered text (a text token always
spans at least one character, so a string's token count never exceeds its
length) plus flat per-message template framing
(`per_message_overhead_tokens`), with prompt images charged an engine-side
per-image bound (`_image_bound_tokens`; `image_tokens_override` pins it).
Cold start, and the rounds between a trim that rewrote sent history and
the next response, bound the whole prompt the same way — the first billed
usage after either replaces the bound with engine truth. The alternative,
client-side *prediction* of the prompt, has no exact form: re-tokenizing
served text is unstable (`Backend.token_count` trips exact-history
validation on an answer whose text does not re-tokenize to its served
ids — an RL policy samples one eventually), and no text ruler can see
engine-side costs at all — vision tokens above all, a gap that runs
to thousands of tokens on a multimodal window. A bound can only fire the
trigger early; early costs a slightly deeper sawtooth, late hands an
over-window history downstream.

Post-turn, every billed usage is held against the window invariant
(`_check_engine_usage`): `prompt_tokens + completion_tokens` above
`context_window_tokens` raises, failing the one game loudly at the moment
of the overrun instead of shipping an oversized sequence into training. A
billed prompt above its recorded pre-call bound — the framing or image
constants falling short of what no text carries — is reported, not
raised: it is load-bearing for at most one round and the safety margin
(`safety_margin_tokens`) absorbs it. No estimating code remains: the
`context_after` diagnostic reads the same decision ruler the
thresholds are in.

### The sawtooth

`compact_to_tokens` is what makes the context cheap. The *trigger* for a
trim is the budget, but the trim keeps evicting down to the target, so the
context grows, compacts once, and grows again, with a stable prompt prefix
between compactions. A trim that instead stopped as soon as the prompt was
under budget would leave it at the budget edge — the next round crosses
again and re-trims, invalidating the KV prefix every round.

The target must be positive and sit below the budget. A reply reserve below
it or lower, no single response can re-cross the budget, so the sawtooth is
guaranteed at least a round; closer than that it is shallower and a
worst-case reply may undo it — slower than ideal rather than invalid, so it
is a choice the caller may make. At or above the budget there is nothing
left to choose: the target is its own trigger, no trim can reach it, and
`_compact_target_tokens` raises. A non-positive target raises for the
mirror-image reason — it names the edge-pinning the sawtooth exists to
remove.

Eviction drops whole user-led blocks. When dropping blocks cannot reach
the budget — a single turn that grew too large on its own — the trim
prunes that turn's oldest rounds instead, and tells the model. Without
pruning, the over-long prompt would be rejected by the engine and the duck
would recover through the nudge path, which opens a new block by
discarding the whole turn: everything established since the last state
prompt, with durable memory the only surviving thread.

Pruning *guarantees* no overrun to the extent the decision ruler does:
the billed part of the prompt is engine truth, and the unbilled
remainder's text cannot out-token its own characters, so an overrun
requires the flat framing or image constants to fall short — a gap the
safety margin absorbs and `_check_engine_usage` reports. The window
invariant raise is the backstop behind both.

`evict_turns` sizes the eviction chunk and the prune overshoot: the fine
control on the same sawtooth `compact_to_tokens` sets coarsely.

Every trim that fires — a block evicted, a round pruned, the memory
snapshot refreshed into the system prompt — is a rewrite of history the
model was already sent, and the duck declares it by bumping
`history_epoch`, which each block's fresh `Conversation` carries
(`ullm.continuation`): the served-token chain restarts there, by the
agent's word rather than by drift. The local and training-server presets
promise exact history (`LLMConfig.exact_history`); an image in the prompt
is continued by ids by an engine and not by a chat server, so whether a
backend can keep that promise is the backend's to answer — it raises on a
call it cannot continue (U5). The duck states the promise and does not
inspect the topology, which is why the presets that make it are built on
an engine.

## Summarize

`compaction_strategy="summarize"`, the default, compacts a full context with a
summary the model writes, in place of eviction's dropped blocks. It is ARC2's
mechanism (arc2-platypus `duck-sweep5`), ported as it was, with its prompt
and the rebuilt conversation settled for ARC3; `tests/test_context_summarization.py`
is ARC2's suite, forked.

**The trigger is the budget eviction uses**, measured the same way. What
fires is not a trim but one extra request.

**The summary rides the live request.** It uses the same system prompt, tools
and reasoning effort, and the whole history, with the summary request as its
last user prompt. It extends the ids the engine already served, so its
prefill is cached, and it never drops history to fit. The engine's window
holds it instead: under summarize a column settles the engine at its native
ceiling (`engine_ceiling_tokens`), past the agent's own window.

- **Thinking closed.** Thinking is closed by an injected end of thinking
  (`_SUMMARY_THINK_CLOSE`), unless `summary_thinking` lets the model reason
  first.
- **The cap.** The summary may run to twice its target (six times with
  `summary_thinking`). The target is `context_window_tokens // compaction_ratio`.
- **The engine window's room.** Each request's cap is lowered to what the
  engine window has left over its measured prompt. `_require_summary_fits`
  asks only that at least `MIN_SUMMARY_ROOM_TOKENS` (8,192) remain past the
  window and the reply reserve, where ARC2 reserves the whole cap. The
  whole cap reserved window no summary used (`evidence.md`, "Sweeps 12 and
  13"); the clamp is what let sweep 13 draw prompt budgets to 180k.
- **The size stated.** Under `thorough_summary` (the default) the request
  asks for a thorough summary and states no size. From the second summary
  on, it asks the model to carry over what still holds of the previous one.
- **Tool calls banned.** `compaction_forbidden_token_ids` bans tokens only
  while a summary is written; every column bans the tool-call token.

**A summary is validated before anything moves.** An empty reply, a tool call
or tool-call markup is not a summary: its tool call never runs, and the
request is retried once with a new seed. A second failure fails the move
with the durable history untouched. A summary left in the reasoning, with
the thinking never closed, is accepted: the request keeps an empty reply
(`keep_empty_reply`) so that it can.

**Retained failed draws.** A trainer's ledger holds one draw per request, so
`preserve_failed_draws=True` (which requires `malformed_retries=0`) makes a
failed draw terminal instead: an invalid or malformed summary gives the game
up without a retry or any rewrite of live history, and a summary request that
drifts from the served ids propagates rather than being re-sent. An ordinary
turn that fails keeps its unsent turn in history. Draws stay observable
(`response_observer`, before any tool runs) and billed; a malformed one is
observed and billed in full only under the backend's `accept_malformed_output`,
since ullm rejects a parser failure before its response callback and usage
meter see it.

**The rebuilt history**, in order:

1. the summary;
2. the user prompt that started the current turn, which is the current
   status, noting how many of the turn's tool calls the summary replaced;
3. the most recent tool call anywhere in the history, with its results, in
   its proper place: after that prompt when this turn made it, before it
   when an earlier turn did (`summary_tail`: as served, a copy without its
   reasoning, or none);
4. a closing message, over a listing of the model's files and Python objects
   under `add_files_and_repl_to_summary` (the default);
5. the live turn, emptied, which the next request continues.

If that request is still over the budget, the kept tool call goes. The
rebuild is declared as two rewrites: one before the install first counts
it, one after, so the next ordinary call renders it afresh. A summary
request that no longer extends the served ids is sent once more as a
declared re-render, and a second drift is not papered over.

**Budgets and the journal.** Budgets are per level, as for any call. The
summarize path lowers each call against the level's live spend, and the
turn's charge runs in a `finally` so a summary-only failure is still billed.
`_SUMMARY_REQUEST_BACKSTOP` (512) bounds one move's calls. When the journal
is on (it ships off), it is kept beside the summary, and its snapshot is
refreshed inside the install's declared rewrite.

**What the port changed from ARC2**, each for an absent or different subject:

- the rebuild keeps the live block's own prompt, the status, rather than a
  task prompt;
- the kept tail is the most recent tool call, not the latest response;
- rounds leave the live turn through the turn manager's pruned-round
  channel, as eviction's do;
- budgets are per level, and the backstop counts one move, ARC2's one-move
  game being the unit it was written for;
- a request's cap is clamped to the engine window, where ARC2 reserves the
  whole cap.

## Durable memory

The `memory` tool appends one immutable `text` entry to a host-backed
append-only journal. It is deliberately not also a Python global: one
concept, one model-facing interface.

The host tags each entry with its journal index, response step, game step,
level, and last action. The journal retains about `memory_context_tokens`
of recent entries by evicting oldest whole entries. In token-native mode
that is an exact ceiling of true tokens: the rendered journal is measured
by the backend (vLLM `/tokenize`, recorded in the replay stream like
every count that drives eviction). Elsewhere no backend can tokenize and
the long-standing nominal 3 chars/token calibration stands -- a retention
preference, never a budget measurement, with the retained text
budget-covered by the character bound like all prompt text. A single
over-limit entry is rejected rather than silently cut. It lives in pickled agent
state, so it survives conversation eviction, kernel resets, moves, losses
and level transitions.

A frozen snapshot sits at the end of the system prompt. New entry text
appears once, in the immutable tool-call arguments, while the tool result
returns only the host timestamp. Compaction refreshes the snapshot with
only those entries whose `memory` tool calls are no longer retained, so no
entry is ever carried by both mechanisms at once.

## The working directory

Behind `file_tools`, the model gets up to five tools — view, search,
replace, create, undo — over the kernel's own working directory; search and
undo each sit behind a gate of their own (`expose_tool_search_files`,
`expose_tool_undo_edit`), both closed by default since sweep 11
(`evidence.md`), and the prompt names exactly the set built. That the
directory is the kernel's is the whole point: a file written there is a
module a snippet can `import`, so work survives as something runnable
rather than as text to re-read.

**Two durable channels, good at opposite things.** The journal is always
in the prompt, small, and host-backed; files are large, runnable, and
invisible until something reads them. So the journal carries what the
model KNOWS and files carry what it BUILT. The coupling that is easy to
miss: conversation eviction eventually drops the tool calls that created
a file, and nothing else advertises it, so the journal is also the index
of what exists on disk.

**Files outlive every reset, at every persistence cadence.** Kernel state
is scratch; a file the model wrote is work. A reset that also wiped the
directory would delete that work silently, so `keep_tmpdir_contents` is
not a knob here. The reset notice therefore has to distinguish the two,
and does so only when the tools are attached: the flag-off wording still
names files among what a reset takes, and is left that way deliberately
so that turning the flag on is the only thing that changes what an
unrelated run reads.

**One access rule, in one object.** `Workspace` resolves every path for
every tool: inside the directory by realpath (so a symlink a snippet
leaves cannot redirect a read), one level deep, and any file name —
containment is the rule, the name is not, because a snippet already
writes here directly through numpy and a workspace that listed those
files while refusing to open them would describe a directory the model
does not have. A resolved path never reaches model-visible text: an
`OSError`'s filename IS the resolved path, so every handler reports the
path the model wrote instead.

Undo reaches exactly one step, and the log clears when an action is
executed against the game — the board has moved, so the file state that
produced it is history rather than something to take back.

**A module from the workspace runs with real builtins.** `__import__`
executes it with its own globals, so it reaches the full standard library
and can define classes, neither of which a snippet can do. That is a
deliberate widening rather than an oversight: the restricted namespace
was never a security boundary (see "The configuration surface"), and what
it buys is a model that stays on the intended surface — a helper the
model itself wrote is on it. The consequence to hold on to is that the
import allowlist is advisory for any model that thinks to write a file.

## The Python runtime

The kernel is ullm's `PythonTool`, drawing slots from a machine-wide pool,
running the snippet under a restricted namespace with the SAFE_BUILTINS /
SAFE_MODULES allowlists, `<python_tool>` traceback sanitization, a
`result` slot and stdout capture. It is agent *state*, not a context
service: it pickles with the agent, and how long its kernel lives is what
`supported_checkpointing()` reports (A1/A4/U9).

**One wall clock covers the whole call, host-call frames included.**
ullm's own per-frame timeout resets on every host-call frame, so a model
loop calling `submit_action()` each iteration — spinning on the terminal
guard's no-op reply after a level win, for instance — would otherwise run
until the per-game token budget stopped it, measured at 17.5 minutes. `PythonTool._send_and_read` holds a single deadline over the
whole call, with `host_reply_grace` bounding reply delivery. On the wall
the kernel is killed and the tool result is a timeout error; actions
already executed stand and surface in the next turn's summary.

`python_timeout_s` is honoured as configured, not capped; only a value
below 1 s raises, as a configuration error rather than a silent repair.
Because the value bounds the whole-call deadline above, raising it widens
the window a `submit_action()` loop can run in, proportionally and per
call.

**And it is unpriced offline.** Wall-clock time costs nothing on a run
bounded by tokens with generous walltime, but on a deadline-bound run a
slow call holds its game's concurrency slot and displaces games still
queued. Any offline comparison of this knob will therefore favour larger
values than a submission would.

## The LLM surface

The provider path is ullm. Per-call sampling seeds are stamped from the
per-game RNG (U8) — the duck never touches `ctx.rng` itself. Reasoning
round-trips through the backend's native mechanism (U11,
`send_back_reasoning`) rather than an echoed message field; on the
Qwen/vLLM targets this is plumbing rather than behavior, since vllmx
re-embeds past reasoning through the same chat template and
`preserve_thinking` path.

Malformed-output retries (seed bump plus a newline perturbation) come from
ullm's `ResponseManager`. Keeping the perturbation is a deliberate call on
weak evidence — see `evidence.md`, *Malformed-output retries* — and it
costs three bounded things: a retried turn's billed input exceeds the
pre-call bound by the appended newlines, the transcript records bytes
differing from the retry's by that suffix, and the KV prefix is
invalidated from the edit point.

**The billed-vs-predicted gap is absorbed, not reconstructed.** A
recovered response carries `malformed_retries > 0`; its extra newline
tokens land inside the safety margin, and `_check_engine_usage` names the
retry count if the bill exceeds the recorded bound. The perturbation is
rare and a perturbed retry can never crash the run on accounting alone —
only the window invariant raises. Rebuilding the exact sent prompt would
mean reaching across the ullm boundary for a rare event; the
fresh-continuation-element perturbation in `roadmap.md` removes the gap at
the source instead.

One case is worse than the tail. The perturbation finder selects the most
recent `StringContent`, so a **multimodal first-round retry** — whose
current prompt is `ListContent(text, image)` and therefore unselectable —
reaches back to the previous turn's tool result, putting the current
prompt and its image after the edit and re-prefilling both. Later rounds
of the same turn have a fresh tool result at the tail and do not. A
perturbation confined to a fresh continuation element would keep all three
properties and the recovery; until then this is a deliberate trade
(`roadmap.md`).

`PromptTooLongError` becomes a synthetic response rather than a
trim-and-retry, so the duck sees a no-tool-call response and nudges. It is
only reachable when the bound's flat constants undershoot near the cap; the
retry hook that would close it is a roadmap item.

## The engine serving dependence

The shipped engine is the arc-prize-2026-vllm fork of vLLM with
``cpu_kv_offload`` ON (the victim tier) on SLURM and Kaggle alike, each
running more games than its pool holds contexts. The fork can zero the
recurrent state at the system-prompt boundary on every call
(``SamplingParams.zero_recurrent_state``,
see ``vllm_fork.md`` and ``DuckAgent.zero_recurrent``), which flattens the
next-token distribution at high-entropy positions (total-variation
distance ~0.9 from unblanked serving) and breaks rumination attractors —
replies conclude and act instead of circling. What that buys is a
reasoning register, and the shipped duck reaches the same register
without the blank, by serving the register's opening as the response's
own first tokens (``DuckAgent.inject_text_on_first_prompt`` and
``inject_text_on_later_prompts``, which need only an engine that takes an
id prompt); the blank is off by default (``evidence.md``, "What the blank
was actually worth"). The serving-dependence below is therefore about
what the blank needs when it is asked for.

The blank was FOUND, not designed. vLLM 0.25.1 with the CPU KV offload
connector present served a hybrid model's warm prefix-cache hit at the
full-attention group's page boundary (``max(per_group_hits)`` on the
connector-aware path — written for NIXL P/D disaggregation, where the
peer ships the Mamba state for that boundary). The offload connector
shipped nothing, so the request resumed with valid attention KV one page
past the last materialized Mamba state: attention memory intact,
recurrent "working memory" blank. That accident had three silent
killers — an engine version bump, real cache pressure (an actually-hit
CPU tier serves loads consistently), and any upstream rewrite of the
connector scheduling path — and 0.27.0's rewritten connector lookup was
the first of them. Making the state a sampling parameter is what removed
the dependence on all three; on the shipped configuration the accident
cannot occur, and the host KV tier the presets turn on is a tier and
nothing else.

The pin lives in this workspace's ROOT, and uv reads ``[tool.uv]`` only
from a root: a downstream workspace consuming these packages as members
resolves vllm within ullm's ``>=0.25.1,<0.28`` range by its own rules and
gets no fork. That is why the duck refuses to construct with
``zero_recurrent=True`` on an engine whose ``SamplingParams`` lacks the
field.
`verify_serving_bug` remains as a probe of that accidental serving path (preset knob, OFF by default and covered by no
tests — its verdict is a statistical read of boot-varying engine
behavior, so a default-on check and CI assertions about it were flaky by
construction; enable it manually when validating a serving stack): at
engine start
it prefill-lays a real duck-text corpus, samples the warm serving path at
many prefix lengths (the harm is striped: it exists only where a prefix
crosses a gap between recurrent-state snapshots, and stripe layout is
chunking-specific — single-length probes read "clean" inside masked
stripes), and requires some length's draws to be improbable under a
recomputed clean distribution. The behavior itself, not a version sniff;
absence is concluded only when every usable length reads consistent. The
characterization lives in arc3-experiments:
`engine_probe/FINDINGS.md` (on that repo's main) is the full writeup —
mechanism, evidence, and the one-line upstream fix — with the probe suite
beside it. That plan's end state is now shipped — the blank is the
explicit mechanism — so the probe's only remaining job is to say whether
a given engine still carries the accident.
