# duck

*Register: living.*

The solver this repo is for. By default the model is shown the puzzle as **pictures**,
handed it as **numpy arrays in a stateful Python kernel**, and answers by
assigning into a variable in that kernel. It is a `platypus.LLMAgent`, so
TAAF scores it and platypus hosts it; what is duck's own is the prompt, the
kernel, and the answer channel.

```python
from duck import DuckAgent
from platypus import LLMAgentSolver
import taaf.standard_benchmarks as sb

agent = DuckAgent.build()  # or build(environment="kaggle-4xl4"); see settings.md
benchmark = sb.make_benchmark_easy(LLMAgentSolver(agent=agent))
```

## Grid presentation is feature-gated

A grid can reach the model as a picture, exact digit rows, and a numpy array
in the kernel. Pictures make structure visible at a glance; digit rows make
values readable without code; the kernel remains the authoritative copy for
exact work. These channels are independent experiment arms rather than one
bundled presentation.

`images` controls image delivery: `"off"` sends a text-only task and omits
`show`; `"task"` sends task images; `"task+show"` also binds
`show(grid, label="")`; and `"task+show+check"` additionally attaches an
image beside each renderable failed-check output. Failed-check digit rows are
always complete. `text_grids=True` adds exact task digit rows while retaining
whatever image mode was selected. The two channels are independent, so
`images="off"` with the rows off is a legal arm rather than a broken one:
the prompt names each array and its shape, and the cells are reachable only
by opening the kernel. The rows are off by default: with
throughput priced at its Kaggle value they cost more than they buy
(`evidence.md`, "The calls at the Kaggle weight").

`ascii_row_numbers` decides how *every* ASCII grid is laid out: off is one
character per cell, on (the default) prefixes each row with its index and a
tab — the shape every other numbered text a model reads has. One flag, one
renderer (`kernel_code.grid_text`), and it reaches all three surfaces at
once: the task prompt's digit rows, the failed-check feedback, and the
kernel's own `ascii_grid`, whose source and docstring are built with the
layout baked in. It is never offered to the model as a parameter, so a
sweep over it changes what every render looks like without the model
choosing; a render the model hand-writes is its own and carries whatever it
chose. The system prompt describes whichever layout is in force, and says
that a leading index is a label rather than a cell wherever the prompt
itself shows numbered rows.

Task and feedback images share `image_scale` and `image_gridlines`. The
defaults are scale 4, unlatticed: sweep 7 read the one-pixel lattice at −1.6
and the colour legend at −1.3, so both are off. `image_layout="paired"`
composes each training input/output pair into one compact image;
`"separate"` — the default — gives every grid its own text-image adjacency.
`image_legend=True` opens the prompt with the shared palette image
independently of that layout; without it the overview names the standard ARC
colours. The large dual-presentation arm uses `images="task+show"`,
`text_grids=True`, `image_layout="separate"`, `image_legend=False`,
`image_scale=16`, and `image_gridlines=True`. `show` accepts at most `max_shows` validated grids per
model call; labels are stripped and bounded to the same character budget as
ordinary Python output so the image channel cannot bypass that text cap. Its
queued grids and failed-check feedback arrive as distinct,
consecutive user messages, which is the multimodal channel supported by the
model backend; ordinary stdout stays in the Python tool result.

When enabled, the colour legend leads the prompt. It is byte-identical in
every game, so putting it first extends the shared prefix — system prompt plus
legend — and every game after the first gets it from the engine's KV cache
instead of prefilling it again.

## The prompt flags are fields

Every wording flag — `generalization_audit_line`, `no_hand_arithmetic_line`,
`nudge_mode`, `brevity`, `cost_model` and the rest — is a `DuckPrompts` field
beside the fragments it selects: pass `DuckAgent(prompts=DuckPrompts(...))`, or
the same `prompts` argument to `DuckAgent.build`. Each adds, changes or deletes
one thing, and a sweep arm reaches one as `Field("inner.prompts.<flag>")`.
Wording belongs on the prompt surface; kernel feedback and generation limits
belong on `DuckAgent`, so prompts cannot independently configure behavior the
harness does not provide. The fields and their applicability are documented on
the classes in `prompts.py` and `agent.py`. `PromptsConfigurable` is the name
the flags were reached through before they lived here: an alias of
`DuckPrompts`, kept for the callers and pickled runs that use it.

Every flag was read in sweeps 5–7, and [evidence.md](evidence.md) says which
readings moved a default and which kept one. Three stay off with a reading
of their own: the standing Python-result `budget_line` (+0.9, p = 0.08, but
Kaggle has no fixed budget to print), the shorter `brevity="half"`
instruction (with `brevity_line=True`; the second take plays it), and the
factual `cost_model="explicit"` description, whose line states the game's
allowance and so cannot serve a later take.

`completion_cutoff_notice` gates naming a reply cutoff on every surface,
including causal nudges. Custom implementations of `Prompts` must accept
`followup_nudge`'s answer problems, implement `budget_warning` and
`budget_status` (an empty status omits the standing line), and implement the
audit turn's methods, `audit_intro` through `read_audit_confidence`.

## The default answer path is an assignment

The kernel binds `train_in`, `train_out`, `test_in` and `test_out`. The model
answers by writing into `test_out[i][attempt]`:

```python
test_out[0][0] = my_solution(test_in[0])
```

The task's arrays are read-only, so a `my_solution` that writes into its input
raises on the line that writes rather than changing the test input under every
later read. The lists holding them are the model's to rebind; `check` reads
copies of its own either way.

Outside mandatory-check mode, the agent reads `test_out` back **after every
python call** and submits whatever is complete, so a run killed at a deadline
still scores what the model had reached rather than nothing. There is no
"submit" tool and no terminal move: the game ends when the model stops asking
for tools, and whatever stood is the answer. A reply the reply cap cut off is
not the model stopping: with the answer standing it gets
`DuckPrompts.answered_nudge` and the game goes on.

After a `python` call that changes `test_out`, each changed answer is shown as
an image beside its test input, in a user message of its own after any `show`
grids (`show_changed_answers`, on), and the system prompt says it will be. A
slot counts as changed where it holds a usable grid other than the one shown
last. It needs the task images and is refused under `MANDATORY`, where an
assignment answers nothing.

A play reply refused on all `malformed_retries` re-samples (three) for ending
with no text and no tool call, or for degenerate repetition, is dropped and
the game nudged once, rather than crashed (`malformed_nudges`, one): the nudge
says the reply was discarded and asks for a different approach, beside what
stands of the answer. Past the count the game crashes. The nudges are counted
because refused draws are not billed, so the game's budget would never stop a
model that loops every time.

Two consequences worth knowing. A model that never writes a readable answer
gets the **repair prompt** — what is still wrong, in the kernel's own terms —
rather than a fresh start. And questions left unanswered are filled with a
placeholder at submission time, because a submission with a missing key is
malformed and scores nothing at all.

## The audit turn

`DuckAgent.audit_turn`, on by default, audits a game once, at the point where
it would stop with its answer standing. The model is handed the puzzle again
in the arm's media. Each test input is followed by its answer slots, and a
slot that copies another, is empty or is unusable is described rather than
drawn. Then come its last reply, quoted, and the questions
(`DuckPrompts.audit_question_*`):
- what in the training data supports each part of the rule;
- what is new in the test;
- the answer, with any alternative a doubt turned up in `test_out[i][1]`;
- under `DuckPrompts.audit_confidence_line` (on), a confidence from 0 to 100
  in `\boxed{}`.

The game ends at its next stop. A summary after the audit pins the restated
puzzle in place of the task prompt, rebuilt with the answer as it then stands.
From that first compaction on, the game's slot offers ask the ranking to put
`audit_handicap_tokens` behind its spend (0; `MultiTakeDuck` sets the second
take's to 10M), so the rest of a long audit can wait for spare capacity.

Nothing before the turn differs from a game without it. So a game read up to
its `audit` event (on `duck/analyze`, carrying the submissions made and the
tokens spent up to it) is that game without the audit, and one run measures
both. The confidence, where asked, is the `audit_confidence` value on the same
span. The turn does not fire once the game budget or the summarize arm's
request backstop is spent. It is refused under `MANDATORY`, where a passing
check ends the game.

`DuckPrompts.audit_groundwork` (on) is what the audit builds on:
- the stopping line and the task prompt ask for the rule in full in the final
  reply, which is the reply the audit quotes;
- the summary request asks for the doubts and inconsistencies noticed so far.

It is a flag of its own, so the audit's flag changes nothing before the turn.
The standing `generalization_audit_line` is off beside the turn, which asks
the same at the stop: with it on, the play does part of the audit's work for
more tokens and the score stays level (`evidence.md`, "Duck sweeps 8 and 9").

## What ARC2 gives this agent that ARC3 never did

**The training pairs are a test oracle.** A candidate rule can be checked
against every example pair in the kernel with `check(fn)`. A failure raises
`CheckError`, reports exact shapes, mismatch counts and the first mismatched
cells, then returns every failed candidate output in full outside the ordinary
stdout cap. Inputs and expected outputs are not duplicated in that feedback;
they are already in the task prompt and kernel.

One `check(fn)` is allowed per Python tool call. It is the final validation
step for that call, not a candidate-search loop; a failure's detailed feedback
arrives in the next user message.

`check` evaluates immutable snapshots of the original training pairs, so a
model that mutates or rebinds the notebook-like public arrays cannot corrupt
its oracle. `check_mode` takes a `CheckMode` enum:

- `DISABLED` leaves the standard prompt and kernel namespace unchanged.
- `ENABLED` adds the oracle; a pass returns an inspectable `CheckReport` and
  leaves `test_out` alone unless `check_and_submit` is enabled (described
  above), preserving the ordinary direct-answer channel.
- `MANDATORY` makes a passing `check(fn)` the only answer channel. It applies
  `fn` to every test input and submits those outputs immediately, with no
  confirmation phase; edits to `test_out` are ignored for submission.

A mandatory submission carries one distinct hypothesis per test input. The
submission layer duplicates it into both ARC attempt slots rather than asking
the model for a second function.

Because `MANDATORY` has no `test_out` read-back, a game that never passes a
check submits nothing at all and finishes `gave_up` rather than `submitted`.
Comparisons across arms that key on run state or submission counts have to
account for that.

The default is `ENABLED`.

## The working directory

Behind `file_tools` (`"full"` by default), the model gets `view_file`,
`str_replace` and `create_file` (and, gated, `search_files` and `undo_edit`)
over the kernel's own working directory, and is told to keep its candidate in
`candidate.py`. The directory is the kernel's, so a file there is a module a
snippet imports: the candidate outlives a summary as a file rather than as
text the summary re-types, and a revision is an edit rather than a
redefinition. The tools are the seal's (arc3-platypus `seal`), whose
whale-style kernel is what this one is.

What a file the model wrote does in the kernel (`kernel_workspace`):

- `import candidate` reaches it, unless Python already has a module by that
  name; `create_file` refuses such a name rather than write a file no import
  would reach.
- It starts with the names a snippet starts with.
- The first import of it in a cell runs it from disk, as a new module. A run
  that raises leaves the last good module as it was, and a name bound in an
  earlier cell keeps its version until a cell imports the file again, which
  is why the prompt's example imports the module, not the function, in the
  call that uses it.
- Before every cell, a private kernel call restores the cwd and puts
  `check` back if a cell replaced it, saying so at the head of the cell's
  output; `show` and `ascii_grid` are the model's to redefine. After the
  cell, a directory it made is removed, since the directory is flat.

Files outlive a kernel death: ullm restarts a killed kernel in the same
directory, though not its session, which the duck rebinds without telling the
model. Each take builds its own kernel, so a second take has its own
directory.

With the flag off none of this runs.

`add_files_and_repl_to_summary` (on) closes a summary's rebuilt conversation on a
listing of the files and of the objects the model defined, with
`candidate.py` shown in full where the file tools are on and it exists, and
the summary request says so rather than asking for the candidate's source.
The closing message follows the summary: the kept tail stays in the live
turn, which the next request continues. A `candidate.py` too large to show
(`CANDIDATE_SHOWN_MAX_CHARS`) is listed, and the summary carries the
candidate as it otherwise does. A listing that fails is said to be
unavailable, with whether the kernel went with it and the error's last line,
rather than ending the game, and the closing line says when the `python`
session was reset instead of calling it intact.

## Two takes

`MultiTakeDuck.build()` takes `DuckAgent.build`'s arguments and wraps two
copies of that duck in `platypus.multi_take`, whose README section has the
contract. What is the duck's own:

- Both takes are asked to put their likeliest answer in `test_out[i][0]`
  (`DuckPrompts.likeliest_first_line`), because the composition reads
  a take's first grid as its first choice.
- At the take boundary, the second take's `max_tokens_per_game` is raised by
  what the game has spent: the duck reads the game's one backend meter, and
  would otherwise start take 2 already over budget.
- The second take's `history_epoch` opens past the first take's. Serving does
  not need it — a fresh conversation renders afresh — but the trace
  collector checks nesting per game, and would read take 2's first step as a
  continuation of take 1's last.
- The second take differs from the first in what is chosen to move *which*
  tasks it solves rather than how many. Three from sweep 7 (2026-09-21): its
  task prompt carries the digit rows the first leaves out, so the
  no-hand-arithmetic caveat takes its digit-rows wording there; the shorter
  brevity ask (`brevity_line` with `brevity="half"`); and the full latest
  response kept after a summary (`summary_tail="full"`). Each read as no
  worse than the first take's value in the sweep, none resolved. Two from
  sweep 9 (2026-10-06): no file tools, and its play profile at 0.5, which
  its audit and summaries sample on too (`SECOND_TAKE_TEMPERATURE`; the
  profiles are each take's own, the engine theirs together). Anything else
  is set on `takes[i]` afterwards.
- A take carries no per-game observer. `context_observer` and
  `response_observer` are refused on any take: each take is a copy with its
  own, and take 2 opens a conversation no history rewrite declares, so an
  observer would see half a game.

`TieredTakeDuck.build()`, the demo's Kaggle agent, takes the same arguments
and plays `platypus.multi_take`'s ten-take variant over that build: its first
take for every take, each copied, on the same run configuration and per-take
net. No take after the first changes its sampling or its prompts 
(2026-10-27). The take boundary carries the allowance and the epoch across
exactly as between two takes.

A duck's submission says which grids the model wrote
(`AnswerReading.produced`): a copy of its one grid and a placeholder for an
unanswered question are fillers. That is part of the answer, so a change in
it alone is resubmitted — a model writing a grid equal to its question's
input changes nothing but the provenance.

## Where the defaults come from

`evidence.md` holds the readings that justify a current default, and says
which defaults have no reading behind them at all. Most do not: a default
absent from that file is a judgement, and should be read as an open question
rather than as a settled small effect.

## Package map

| Module | What it is |
|---|---|
| `agent.py` | the agent: the round loop, the context budget, the answer harvest |
| `config.py` | shared configuration enums |
| `prompts.py` | every string, as configuration — `DuckPrompts` fields, not literals |
| `kernel_code.py` | the kernel prelude, the task namespace, the answer read-back |
| `python_tool.py` | the ullm `Tool` the model calls, and how results are rendered |
| `workspace.py`, `file_suite.py`, `module_names.py` | the file tools over the kernel's working directory, the one rule for what they may reach, and the names a file may not take |
| `kernel_workspace.py` | the kernel's half of the working directory: the model's files as modules, and what runs before every cell |
| `sandbox.py` | what the kernel may not reach (lifted from the whale) |
| `errors.py` | traceback normalization — the model sees its own error, not ours |
| `state.py` | the typed answer reading: what is set, what is wrong, what is missing |
| `images.py`, `palette.py`, `fonts/` | grids and the colour legend as pictures |
| `presets.py` | what ARC2 asks of a deployment, layered onto a built agent by `configure()` |
| `multi_take.py` | `MultiTakeDuck`: two takes of the duck, and what a take carries across the boundary; `TieredTakeDuck`: up to ten |
| `traffic.py` | `TrafficDuck`: the column's engine measured on recorded traffic instead of games (ullm's `traffic_benchmark.md`); its launcher and workload cutter are in `tools/` |

## Deployments

`DuckAgent.build(engine=…, environment=…)`, one call per coordinate —
`engine` selects the checkpoint (`"qwen3.8-flash-next"` by default, the e307
NVFP4 prune of Flash-Next; `"qwen3.8-27b"` for Qwen3.8-27B-FP8, and
`"qwen3.6-27b"` for the previous model),
`environment` the box (`"slurm-b200"` by default: one B200, `tp=1`;
`"kaggle-4xl4"` for the offline Kaggle stack, four L4s -- the 27Bs over four
tensor ranks, Flash-Next over two stages of two);
`serving="server"` picks the OpenAI-server path over the same engine args,
for the 27Bs only: Flash-Next is refused there (ullm's `roadmap.md`, "Park
mode over the server topology"). Everything else is a field, and the engine follows the agent at
`prepare_for_backend` — widen the window or run more games and
`max_model_len` and `max_num_seqs` move with it. `settings.md` at the repo
root says which layer of `presets.configure` owns each setting, and why the
figures that differ from upstream's do.

All of it is configuration only — no engine is built until the run starts —
so a column that vLLM would refuse still passes every local check. That gap
is what the `b200` test tier exists to close, and what the deployment
snapshot (`tests/references/deployment_snapshot.json`) freezes in between.

The served-history columns are text-only ducks: an image is continued by
tokens on the in-process engine alone, so `serving="server"` builds with `images="off"`. Supplied factories retain
their own transport capabilities. Whether the training
duck also wants `text_grids` is open (`roadmap.md`).

An API model for cheap harness checks is not a deployment and is built
directly, as `tests/test_duck_reference.py` builds the haiku duck: a
`PrebuiltBackend` over `ullm.backends.anthropic.Anthropic` in an
`LLMConfig`, with a plain `DuckAgent(llm_config=…)` around it.

## The context budget

A game is one long turn, and the conversation outgrows the window. Every ARC2
column ships `compaction_strategy="summarize"` (below; `presets.configure` sets
it beside the reply cap a summary needs a real reserve from). The class default,
`"prune"`, has three layers, cheapest first:
**measure** the next prompt, **evict** whole user-led blocks from the oldest
end, and — because a game is usually a *single* block, so there is nothing
to evict — fall back to dropping the current turn's oldest rounds.

Measurement never estimates. In token-native mode (`token_native_context`,
declared by both vLLM presets) the backend answers the request's exact
weight — the id-ledger length on a continued call, its own render after a
declared rewrite — and the trigger is that weight plus the reply reserve
plus `safety_margin_tokens`, nothing else. Elsewhere the newest
engine-billed response pins everything through its own generation at
exactly `prompt_tokens + completion_tokens`, and whatever was appended
since is charged a provable bound: text at its raw character count (a
token never spans less than one character), flat per-message framing
(`per_message_overhead_tokens`), and each image at an engine-side cost
derived from its pixel dimensions (`image_tokens_override` pins it). Cold
start, and the rounds between a trim and the next response, bound the
whole prompt the same way. A bound can only fire the trigger early — a
deeper sawtooth, never an over-window history. Post-turn, every billed
usage is held against the window: `prompt + completion` above
`context_window_tokens` fails the game loudly rather than shipping an
oversized sequence downstream; a billed prompt above its pre-call bound is
reported, since the margin absorbs it for the one round it is load-bearing.
`exact_token_counts` is inert and kept so existing constructions bind.

Eviction takes the task prompt with it once a game has re-entered —
`keep_task_prompt` keeps the pictures as the head instead (off by default,
to be swept; `roadmap.md`, "Eviction can drop the task itself").

That fallback is the normal path here, not an edge case, and it is why
compaction is recorded: each trim that prunes appears in the diagnostics as
its own `llm/turn` span named "compacted", holding the rounds the model can
no longer see. Without it the turn's token total and its transcript would
both silently lose everything pruned.

`compaction_strategy="summarize"` is the model-written alternative. At the
same pre-request boundary it asks the model for a summary targeting
`context_window_tokens // compaction_ratio` tokens, then atomically rebuilds the
live history from the pinned task, that summary, and the newest assistant
response when it still fits. The summary request is the live request extended
by the instruction: the same system prompt, tools and reasoning effort, and the
whole history, so it continues the served ids and its prefill is cached. The
summary is written without thinking, by injecting `</think>` after the
generation prompt rather than by the template's `enable_thinking=False`, which
would also drop the effort line that opens the system block and with it the
cached prefix. It may run to twice its target, and the diagnostics record its
length (`summary_tokens`).

Two knobs change that. `summary_thinking=True` lets the model reason before it
writes: the request opens on the ordinary generation prompt, nothing injected,
and the cap rises to six times the target, since the reasoning counts against it;
only the final text becomes the summary. `summary_tail` says what the rebuilt
history keeps beside the task and the summary: `full` the latest
response and the tool results after it, `without_reasoning` (the default) the same with that
response's reasoning left out — a copy of the response, since the served object
carries its witness and its bill — and `none` nothing. A tail that does not fit
the window goes regardless, and the diagnostics say which it was.

It never drops history to fit: a summary request the engine rejects raises, and
the engine window has to hold the whole request — the round that overflowed the
agent's window, the instruction and the summary on top of it, which is not a
bound the agent can derive (a tool result is capped in characters, failed-check
grids are sent uncapped). So under this strategy `prepare_for_backend` settles
`max_model_len` to the window the engine was built with (`engine_ceiling_tokens`,
the checkpoint's native context, recorded by the build), and refuses when the
agent's window, the reply reserve and the summary's cap do not fit under even
that. The summary call is held to its measured bound like any
other, but not to the agent's window invariant, which it exceeds by design.
Invalid summaries get one reseeded retry, with reasoning where the reply was
entirely empty; ordinary malformed-output handling stays with ullm's public
response manager. Every summary attempt consumes
`max_llm_calls_per_move`, and every response ullm accepts is charged to the
per-game token budget; a one-call entry can therefore end after compaction
without making an ordinary analyzer call. A game-wide 5,120-request backstop,
including summary attempts, also ends zero-token empty-response loops that a
generation-token budget cannot advance. This arm requires an explicit positive
`max_tokens` on its sampling profile. The summary request and the rebuilt
history are measured like the live turn's: the backend's exact count in
token-native mode, the same provable bound otherwise, with images at their
engine-side cost in both. Deployments that know the model's tokenizer can set
`compaction_forbidden_token_ids` to ban tool-call marker tokens while the
summary is written (the request carries the tools, so this is what keeps a
tool call out of it); the ordinary sampling profile is never modified. A
`summary_profile`, when set, samples the summary on that profile instead
(its temperature, `top_p`, `after_reasoning` and `extra_params`), keeping the
live request's seed and injection; `audit_profile` does the same for the
audit turn and every block after it. With `intermediate_summary_prompt_lengths`
the arm also asks for summaries that leave the history whole: when the request
about to go out has passed one of the lengths, the block ends before it, a
turn of its own asks for a summary (no size, no tools, the same attempts and
retry as a compaction's, the play's reply cap), the request and the summary
stay in the history as served, and the next block opens on the message to go
on. A compaction runs alone where both fall due and re-arms the lengths. The
Qwen3.8 GPU/Kaggle presets set the verified single-token marker ID
automatically, while provider-neutral agents leave the gate empty. Response
injection is served only by the in-process vLLM backend.

A caller that retains malformed replies may set `terminate_on_empty_reply`: the first retained reply
with no text and no tool call (the empty reply an `accept_malformed_output`
backend keeps as the model's turn) ends the game as a give-up, so whatever
answer stands is scored at once instead of the entry nudging the model into
another empty turn. The reply's malformed count still applies.

`preserve_failed_draws=True` requires `malformed_retries=0`. With summarize,
an invalid or malformed summary is a terminal give-up without retry or live
history mutation; accepted draws remain observable and billed. Summary history
drift propagates without re-rendering, preserving the trainer's served-id contract.
Full response observation and billing require the backend's `accept_malformed_output`:
ullm rejects parser failures before its response callback and usage meter update.

A summary is the most complete rewrite of sent history the duck makes, and it
is declared twice, both times after its request, which is not a rewrite: once
the summary is accepted, before the install first counts the rebuilt history
(counting it on the served epoch is the drift exact history refuses), and again
after the install, so the next ordinary call renders the rebuilt history afresh
rather than failing to extend the summary's served ids. A summary request that
turns out not to extend the served ids is re-sent once as a declared rewrite
and recorded as `summary_request_rerendered` when failed draws are not retained.

Every trim that removes something — a block evicted, a round pruned, the
pruned-rounds note rewritten into the opening prompt — is a rewrite of
history the model was already sent, and the duck declares it by bumping
`history_epoch`, which each analyzer entry's fresh `Conversation` carries
(`ullm.continuation`). The declaration is made **before the trim's first
edit**, because the trim re-counts the prompt after every block it drops
and each count continues the served ids: a count over a shortened history at
the old epoch is exactly the drift the promise refuses. A trim that fires
and finds nothing it may remove declares nothing — the sent prefix stands.

The GPU columns promise exact history (`LLMConfig.exact_history`): every
call after the first extends the served token ids, and an undeclared
rewrite is an error rather than a quiet cache miss. Every duck prompt
carries images, which an engine continues by tokens and a chat server does
not — so the columns that promise it are built on an engine. Whether a
backend the caller brings can keep the promise is the backend's to answer,
and it answers by failing; the agent states the promise and does not
inspect the topology.

## Run budgets

A game ends on its own when `max_tokens_per_game` runs out — and that is the
only budget for a game that is generating. The class default is 65k generated
tokens, sized so an arm's *mean* spend can reach the per-game budget a sweep
reads its arms at (`sweeper.DEFAULT_BUDGET_TOKENS`, 45k charged); the
measurement is in `evidence.md` ("max_tokens_per_game"). The columns state
their own: Flash-Next plays 500k a game on the B200 and uncapped on Kaggle
(`settings.md`, P-0V). The wall net
(`presets.GAME_RUNTIME_CAP_S`, eight hours) is underneath it for the game that
is *not*: hours spent inside the kernel on a search that never converges cost
almost no tokens, so the token budget never fires and the job's walltime ends
the game by cancelling it, unscored. The net ends it as `gave_up` instead,
scored as it stands.

It is a net, not a budget, so it sits far above where games finish: over
`arc2-eval` on one B200 at 150k a game the longest normal game was 62 minutes
and the longest win 60, against a spinner still on its first answer past 102.
Eight hours is sized for the B200 column's 500k cap (`settings.md`, P-0V, "the
per-game wall net"). Every duck `RunConfig` comes from
`presets.duck_run_config()` so the cap has one definition; **Kaggle is the one
deployment that turns it off**, because the submission's own clock already
bounds the run and a game cut short there would forfeit an answer the arena
would have scored. Kaggle keeps the token cap only as a backstop
(`presets.KAGGLE_MAX_TOKENS_PER_GAME`, 70k): with the slots ranked by spend, a
game that has spent much is deprioritised long before it is ended, which frees
the same slot without the forfeit.

## Reading a run

`make viewer` over the results tree, or `diagnostics.html` in the job dir.
Per game the useful things are the compaction segments (above), the
`kernel/exec` events, and the answer sheet.

Throughput is read on the traffic replay, not on an hour of play:
`TrafficDuck` (`traffic.py`) over a workload cut from a recording, launched
by `tools/traffic_launch.py`. The instrument and what a reading means are
ullm's `traffic_benchmark.md` (in arc3-platypus, `packages/ullm/`); the
four-L4 readings are in its `flash_optimization_arc2.md`. ARC2's current
workload is
`/data/shared/research/arc_2_results/shared/traffic/workload_long230_20260929.b64`:
230 lanes cut from the B200 long run of 2026-09-28. Its recording and its
caveats are under "Current workloads" in that document.

Two instruments guard the model-visible surface, and both are cheap:
`tests/prompt_golden/` holds every prompt the duck can render as readable
text (regenerate with `DUCK_REGENERATE_PROMPT_GOLDEN=1`, then **read the
diff**), and `tests/references/duck_haiku.*.ref.jsonl` replays a recorded
game with no model involved.

## What the backend sends back

Neither of those two can see what the *backend* does with a past answer on
the way back into the model, and a real defect lived exactly there. With a
reasoning parser the sampled text carries no opening `<think>` — the
generation prompt pre-filled it — so handing that text back as `content`
made the Qwen template render every prior round as an empty think block, the
reasoning as content, and a stray `</think>`:

    <|im_start|>assistant
    <think>

    </think>

    The colours look mirrored. Check the arrays.
    </think>

    <tool_call>...

Every round after the first, on every run before ullm `895da96`. Past
reasoning now rides the structured `reasoning_content` field the template
renders inside the think block. **Scores measured before that pin are not
comparable to scores after it.**

`tests/test_backend_history_render.py` is the instrument that was missing:
it asserts what the configured presets hand the chat template, with no
engine and no tokenizer.
