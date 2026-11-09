# arc3-platypus

*Register: living.*

ARC-AGI3 solver harness by Tufa Labs. `README.md` has the package
layout and dev commands; the doc map below says where everything else
lives.

## Doc map

| Document | What it is | Register |
|---|---|---|
| `README.md` | overview, package layout, dev commands | living |
| `requirements.md` | the agent/infra/ullm contract + decision ledger — the source of truth for behavior | normative |
| `design.md` | the as-built high-level design of the infrastructure and LLM layers | as-built |
| `settings.md` | where a setting lives: the eight buckets, and every parameter placed in one | as-built |
| `agent_authoring.md` | the practical manual for writing an agent | as-built |
| `roadmap.md` | direction beyond the current round; partial/dormant requirement states | living |
| `checkpointing_notes.md` | deliberately unresolved design input for checkpointing (P9) | living |
| `vllm_fork.md` | how platypus consumes the vLLM fork: pin + build variables, serving, Kaggle (what the fork carries is its own `TUFA.md`) | as-built |
| `<branch>_branch_notes.md` | per-branch WIP notes, present only while that branch is open | ephemeral |
| `packages/*/` | per-package docs where needed; each package README maps its own | — |

## Documentation rules

- **Implemented needs no story; unimplemented does.** Code and tests
  speak for what exists. Docs carry only what code cannot say:
  requirements not yet met, deliberate not-yets, reproduced-bug
  decisions, contracts, and rationale. No document describes the order
  things were built — git history is the archive, and nothing is kept
  "for the record".
- **One home per fact.** The layout lives in `README.md`, requirement
  states in `roadmap.md`, test tiers here, where a setting belongs in
  `settings.md`; everything else links instead of restating.
- **Register header.** The first line under every title names the
  document's kind and update rule: `normative` (change deliberately,
  call it out in the PR) / `as-built` (updates with the code) /
  `living` / `ephemeral` (dies at a merge) / `reference @ <pin>` (not
  maintained).
- **AGENTS.md discipline.** AGENTS.mds carry guidelines and gotchas,
  never code description; package AGENTS.mds never restate root content
  (both are in context at once); about one screen each.
- **Decision homes.** Contract-level decisions → `requirements.md` §8;
  design-level → inline alternatives in the relevant design.md;
  branch-level → the branch's notes file, promoted at merge. A new
  configuration value needs a bucket in `settings.md` before it needs a
  line of code — which package owns it follows from that, not the reverse.
- Reference files by path and symbol name, never by line number.

## Way of working

The mainline is `main`. A long-running integration branch takes that
role while it is open, and says so in its own branch notes — which is
where to look if the two ever disagree.

- Work happens on per-feature branches that merge into the mainline.
  A branch MAY keep `<branch>_branch_notes.md` at the repo root (branch
  name with underscores): an optional development tool, not a
  requirement — WIP notes only (scope, decisions taken, open items, a
  running log). Long-running branches tend to want one; a short branch
  whose PR description carries the context does not. You edit only your
  own branch's notes file.
- Review happens via a draft PR into the mainline, opened
  once the branch has substance; a branch's notes (when it keeps them)
  stay in the diff during review, and review outcomes that change
  decisions are logged there — otherwise the PR description carries
  that context.
- **Branch-notes exit checklist.** After approval, the final commit
  before merge deletes the branch's notes file where one exists, having
  first swept it: decisions → `requirements.md` / the relevant
  design.md; lessons and gotchas that generalize → AGENTS.mds,
  package-specific ones → package docs; a measurement that justifies a
  current default → the package's evidence file (e.g.
  `packages/duck/evidence.md`); open items → the relevant roadmap.
  Everything else dies with the file, on purpose. The checklist applies
  at every merge, integration branches included — the sweep obligations
  hold for a notes-less branch too, just sourced from wherever its
  decisions accumulated.
- File ownership across parallel branches is assigned explicitly; if
  it's unclear whether a file is yours to edit, ask.
- PRs always merge with a **merge commit** (never squash or
  fast-forward) — branches keep their internal history. After a merge,
  other branches rebase onto / fast-forward to the mainline.
- `uv.lock` conflicts are never hand-merged: take either side, rerun
  `uv lock`.
- Edits to `requirements.md` change shared truth — call them out
  explicitly in the PR description.

## Coding guidelines

- Every function and class gets at least a short docstring. Enforced by
  `tests/test_docstring_gate.py`, which also fixes the exceptions:
  **tests** (they document per module — the module docstring names what
  the file covers and which requirement), **dunder methods**, and
  **functions nested inside functions**. Everything else is in, ullm's
  own code included — there is no grandfather list, so a failure is
  always a missing docstring. A module whose docstrings `help()` renders
  to a model needs an exception of its own, since writing one there is a
  prompt change rather than documentation; none ships today.
- Properties, arguments and return values that are not obvious get
  explained. Obvious ones do not — restating a name in prose is noise.
- Be concise. A comment earns its place by saying **why** the code is
  the way it is, where that would otherwise be unclear; code that
  already reads clearly needs none.
- **Never write the history.** A comment describes the code as it
  stands and the constraint it is under — not what it used to do, not
  the bug that prompted it, not "we tried X and it broke". `git log`
  and the PR hold that; a comment holding it goes stale silently and
  reads as an apology. State the rule, not the scar:

      # BAD:  we used to sort this later, which made replays diverge
      # GOOD: emitted sorted — the order reaches the model

  **There is no longer a standing exception.** Both ports have
  decoupled, so "the original does X" is history in every package. Their
  remaining parity comments are being retired opportunistically rather
  than in one sweep — dozens survive in `packages/duck/src`, and each
  goes when the code around it is next edited, restated as the rule it
  justifies. A new one is not allowed.

- **A top-level import in an agent module is a per-game memory cost,
  and so is every object on the pickled agent.** Games run in spawned
  children (I9) that unpickle the agent, so the module's imports and
  whatever its fields drag in run again in every child's own heap —
  ~110 of them in a submission. The engine is the host's alone:
  `platypus.subproc` asserts a child never has vllm and puts the cost on
  the assert. Import what only the host needs inside the function that
  needs it (`duck.agent._installed_vllm`), and keep host-only objects off
  the agent a child receives (`LLMAgentSolver._child_agent`).

- **No `nest_asyncio`.** It patches asyncio for the whole process: its
  loop drops cancelled timers only from the front of the queue, so a
  timeout per streamed partial holds every cancelled timer until its
  deadline (millions in the host process), and it swaps asyncio's C
  `Task` and `Future` for the pure-Python ones. Nothing here runs an event
  loop inside a running one; code that seems to need it should be async
  instead.

- ullm carries its own package conventions that **override these where
  they differ** (see its AGENTS.md — the fork keeps upstream ullm's
  style for consistency with its existing code).

## Tests and gates

Member-internal suites live in each member's `tests/`
(`packages/platypus/tests`, `packages/ullm/tests`, ...); the root
`tests/` holds cross-member integration and the committed reference
replays. `make prepare` (format + lint + pyright strict + tests) is the
gate, `make prepare-ci` is what CI runs. Expensive test tiers are
opt-in and gated twice (pytest marks + runtime guards): `make
test-gpu-srun` dispatches the GPU-marked tests to a SLURM node (one job
per backend column), `make test-api` runs the paid-API tests (requires
the `ULLM_API_TESTS=1` opt-in it sets), and `make test-full` runs every
tier in its appropriate location.

A package's own `tools/` directory is covered by format, lint and pyright
strict — `ALL_CODE` and `pyrightconfig.json` both reach it — but **not** by
`tests/test_docstring_gate.py`, which globs `packages/*/src` plus the root
`tools/`. `nbstripout` is installed as a git
filter, so a committed notebook keeps none of its outputs: a conclusion that
lives only in a rendered cell does not survive the commit.

**Gates in one repo run one at a time.** Every gate target runs
`install-dev` first, which ends in `nbstripout --install`, and that writes the
`.git/config` all of the repo's worktrees share. Two gates started together in
worktrees of one repo race for that file's lock, and the loser fails at install
(`could not lock config file`) without running a test. Start the next once the
first is past install, or run them in turn; gates in different repos do not
collide.

**Pre-merge rule**: `make test-full` runs as the FINAL check before
every merge into the mainline — and only then; during
development stick to the cheap gate (`make prepare`) so GPU allocations
and API spend aren't burned on intermediate states.

**Known flaky**: the compacting variant inside the duck-engine column
`duck_direct_boot` (`tests/test_duck_engine_startup.py::test_duck_direct_boot_plays_and_nests`).
Whether the compacting window fires depends on how much the model
actually generated, and `_STEPS_TO_COMPACT` guards only the too-short
game — one long enough to clear that guard can still finish without ever
crossing the context window, and then the assertion has nothing to find.
It has gone red and green on the same commit, so re-run the column before
reading a failure as real.

**Timing-sensitive on the CI runner**: `packages/platypus/tests/test_memory.py::test_each_process_is_named_by_what_it_is`
and `packages/platypus/tests/test_host_monitor.py::test_the_sampler_rolls_segments_and_leaves_none_half_written`.
The runner is loaded where a workstation is not, so a fixed sleep there can
pass before a sampler thread has written anything, and a process read right
after `Popen` can still be mid-exec, with no command line yet. Both tests wait
for the condition they check, with a deadline. A red run of either is a race
first, so re-run it before reading it as real. A new test that watches another
process or thread waits the same way.

**Unused imports are ruff's check, not pyright's.** `pyrightconfig.json`
turns `reportUnusedImport` off, and says why where it does: on the GitHub
runner alone, pyright reported a used `import ullm.backends.<module>` as not
accessed, and with `make static` failing on any warning that made CI a
lottery (one head failed five times in a row). It never reproduced
locally under any pyright version, install shape or thread count, and
giving `ullm/backends/` an `__init__.py` did not change it. `F401` in
`make lint` enforces the same rule deterministically on every machine, so
nothing is lost. If a used-import warning ever appears from pyright again,
the config moved — not the code.

## Recorded expectations

Two kinds, and each catches what the other structurally cannot.

**Committed references** (`tests/references/*.jsonl`, one per game+pass —
format in `design.md`) record what a model was actually sent and replay it
byte-exact. They notice *any* change to model-visible text, and cannot
notice that the text was wrong to begin with: a defect present when the
reference was recorded is in the expected bytes and replays green forever.
They also hold only the branches one run happened to take — and a reference
scripted to *cover* branches cannot notice it has stopped covering one (it
replays green against its own recording either way), so pair it with a test
asserting each intended branch's text is present.

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_<name>'
    ULLM_API_TESTS=1 PLATYPUS_REGENERATE_REFERENCES=1 \
        make test-api TEST_ARGS='-k regenerate_<name>'   # the paid ones
    PLATYPUS_REFRESH_VIEWER_HTML=1   # viewer_html slice only; free, replays recorded traffic

A NEW reference must be added to the case list in
`tests/test_viewer_refresh.py`. That list is what the free viewer refresh
walks, so a case missing from it has no cheap path to a re-render: a change
to what the viewer draws leaves the reference stale, and the only fix left
is a full regeneration — paid, if the reference is one of those.

Know the cost before running it. Stub-driven references regenerate free;
live ones spend real API credits and need `ULLM_API_TESTS=1` — three cases
are paid (two sonnet-5, one haiku-4.5), every other one is free. The
paid ones are marked `api`, which the cheap tier deselects, so they need
`make test-api` rather than `make test`: a `-k` filter alone selects nothing
and the run passes having done nothing. Regeneration self-validates by an immediate
replay, so a green run means the new recording is sound, not that the change
was intended — and "sound" is only that it reproduces: whether the run is one
worth committing (a game played, not forfeited) is a separate assertion the
case has to make for itself.

**The deployment snapshot** (`tests/references/deployment_snapshot.json`,
and the seal's `seal_deployment_snapshot.json` beside it) freezes every
deployment's whole configuration — the agent's fields, its
`RunConfig` and `LLMConfig`, and the backend factory with its engine
arguments — resolved at `prepare_for_backend`. It is the only test that reads
engine configuration at all, so it is what notices a window, a watermark or a
KV connector moving; the replays above cannot, and `settings.md` is what it
holds the layers to. A diff is meant to be read and put in the PR, since it is
the list of what a change did to every column.

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k deployment_snapshot'

**Prompt goldens** (`packages/duck/tests/prompt_golden/`,
`packages/seal/tests/prompt_golden/`) render every
prompt, in configurations chosen rather than sampled, as readable committed
text. They exist for the case references cannot reach — whether the text is
*right* — so a diff here is meant to be read, not merely to match.

    PLATYPUS_REGENERATE_PROMPT_GOLDEN=1 make test TEST_ARGS='-k prompt_golden'

Reference tests run outside `make` need `PYTHONHASHSEED=0`; the make targets
set it.

**Moving a default is a change to the reference set, both ways.** The
`duck_nondefault` case exists to hold every field away from what ships, and
a default that moves breaks that in one of two directions. If it moves ONTO a
value the case pins, the field silently stops being exercised — asserted by
`test_every_configured_field_differs_from_the_shipped_default`, because
nothing else can notice. If it moves AWAY from a value the case never pinned,
the old default stops being covered at all, and no test can notice that: the
case cannot know which of the duck's several dozen fields it ought to hold.
So a PR that moves a default owes the nondefault case a matching flip, and
where two non-default states are mutually exclusive it owes a comment saying
which one is covered elsewhere.

## Test guidelines

- No trivial tests: never assert that a declared default equals itself,
  that a getter mirrors a field, or similar duplication of the code in
  test form. Every test must be able to catch a plausible real bug.
- **"Model-visible" does not tell you what a change costs; whether a
  committed reference records it does.** The two come apart in both
  directions, and the cheap-looking direction is the dangerous one: a
  one-character fix to a prompt string can force a live re-recording of
  every reference that quotes it, while renaming the whole module
  surface the model imports can cost nothing because no reference ever
  exercised it. Measure before planning around it — and treat
  *model-visible but unrecorded* as the case to worry about, since
  nothing will fail if it goes wrong.
- Test outcomes, not internals — preferably tied to a numbered
  requirement (A#/I#/U# in `requirements.md`).
- Test against the real producer's output, not a fabricated stand-in:
  a record hand-written with only scalars never carries the list the
  engine actually emits, and the crash then arrives in a 12 h run.

## Calibrating a default from run traces

A parameter tuned against benchmark runs is tuned on **25 official games
with most of the score in two of them** (`ft09`, `tu93`), and is scored on
hidden ones. A train/test split over that divides the noise, not the
signal, so the usual discipline does not apply and this one does:

- **Width, not height.** A parameter whose simulated benefit collapses off
  its best value will not survive the hidden set. Report the range over
  which a rule beats the baseline and prefer a flat-topped optimum to a
  taller spike — a sharp peak in a 25-game sample is one game crossing one
  threshold.
- **Re-run every conclusion with the top scorers stripped out.** A rule
  that inverts without `ft09` and `tu93` is a conclusion about those two
  games. This is the cheapest overfitting check available.
- **A policy must be justifiable by its mechanism**, not only by its
  score. "Capping level 1 cannot touch a game that has already cleared
  one" bounds the downside by construction; "cap game 7 at 40k" fits the
  sample. Prefer few parameters, and round ones.
- **The null result is a real result** — and cheaper than shipping a rule
  that overfits.
