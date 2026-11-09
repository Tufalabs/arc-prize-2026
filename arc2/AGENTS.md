# arc2-platypus

*Register: living.*

ARC-AGI2 solver monorepo by Tufa Labs. `README.md` has the package layout
and dev commands; the doc map below says where everything else lives.

## Doc map

| Document | What it is | Register |
|---|---|---|
| `README.md` | overview, package layout, dev commands | living |
| `roadmap.md` | direction, open work, partial/dormant requirement states | living |
| `settings.md` | where a setting lives: the eight buckets, and every parameter placed in one | as-built |
| `<branch>_branch_notes.md` | per-branch WIP notes, present only while that branch is open | ephemeral |
| `packages/*/evidence.md` | measurements that justify that package's defaults | living |
| `packages/*/` | per-package docs where needed; each package README maps its own | — |

Deliberately lighter than the platypus repo this descends from: **there is
no `requirements.md`.** Behavior is specified by code and tests; what cannot
be read off them — decisions, rationale, deliberate not-yets — lives in the
relevant README or `roadmap.md`. Every package carries a `README.md`; a
`design.md` appears only where a package's mechanism genuinely spans more
than one file.

## Documentation rules

- **Implemented needs no story; unimplemented does.** Code and tests speak
  for what exists. Docs carry only what code cannot say: requirements not
  yet met, deliberate not-yets, contracts, and rationale. No document
  describes the order things were built — git history is the archive, and
  nothing is kept "for the record".
- **One home per fact.** The layout lives in `README.md`, requirement
  states in `roadmap.md`, test tiers here, where a setting belongs in
  `settings.md`; everything else links instead of restating.
- **Register header.** The first line under every title names the
  document's kind and update rule: `normative` (change deliberately, call
  it out in the PR) / `as-built` (updates with the code) / `living` /
  `ephemeral` (dies at a merge) / `reference @ <pin>` (not maintained).
- **AGENTS.md discipline.** AGENTS.mds carry guidelines and gotchas, never
  code description; package AGENTS.mds never restate root content (both are
  in context at once); about one screen each.
- **Decision homes.** Cross-cutting decisions → `roadmap.md`; package-level
  → that package's `README.md` (or its `design.md`, where one exists);
  branch-level → the branch's notes file, promoted at merge. A new
  configuration value needs a bucket in `settings.md` before it needs a
  line of code — which package owns it follows from that, not the reverse.
- Reference files by path and symbol name, never by line number.

## Way of working

The mainline is `main`. A long-running integration branch takes that role
while it is open, and says so in its own branch notes — which is where to
look if the two ever disagree.

- Work happens on per-feature branches that merge into the mainline. A
  branch MAY keep `<branch>_branch_notes.md` at the repo root (branch name
  with underscores): an optional development tool, not a requirement — WIP
  notes only (scope, decisions taken, open items, a running log).
  Long-running branches tend to want one; a short branch whose PR
  description carries the context does not. You edit only your own
  branch's notes file.
- Review happens via a draft PR into the mainline, opened once the branch
  has substance; a branch's notes (when it keeps them) stay in the diff
  during review, and review outcomes that change decisions are logged there
  — otherwise the PR description carries that context.
- **Branch-notes exit checklist.** After approval, the final commit before
  merge deletes the branch's notes file where one exists, having first
  swept it: decisions → `roadmap.md` / the relevant package README; lessons
  and gotchas that generalize → AGENTS.mds, package-specific ones →
  package docs; a measurement that justifies a current default → the
  package's evidence file; open items → the relevant roadmap. Everything
  else dies with the file, on purpose. The checklist applies at every
  merge, integration branches included — the sweep obligations hold for a
  notes-less branch too, just sourced from wherever its decisions
  accumulated.
- File ownership across parallel branches is assigned explicitly; if it's
  unclear whether a file is yours to edit, ask.
- PRs always merge with a **merge commit** (never squash or fast-forward) —
  branches keep their internal history. After a merge, other branches
  rebase onto / fast-forward to the mainline.
- `uv.lock` conflicts are never hand-merged: take either side, rerun
  `uv lock`.

## Forking, not depending

Every layer that comes from the ARC-AGI3 stack is copied in and owned here.
The upstream repos keep running against ARC-AGI3 and will diverge; a shared
dependency would make every ARC2 adaptation a negotiation with a codebase
that has different requirements. Ported code carries no "the original does
X" comments: upstream is history the moment it lands (see *Never write the
history* below), and a port's provenance belongs in the PR. It carries no
upstream requirement numbers either — a citation nobody here can resolve is
worse than none, and a port that keeps them scatters hundreds.

An exception is a decision, recorded with its rationale and its obligations
in `roadmap.md` — there is one today, the LLM layer `ullm`.

## Coding guidelines

- Every function and class gets at least a short docstring. Enforced by
  `tests/test_docstring_gate.py`, which also fixes the exceptions: **tests**
  (they document per module — the module docstring names what the file
  covers), **dunder methods**, and **functions nested inside functions**. A
  failure is always a missing docstring, never a number to bump.
- Properties, arguments and return values that are not obvious get
  explained. Obvious ones do not — restating a name in prose is noise.
- Be concise. A comment earns its place by saying **why** the code is the
  way it is, where that would otherwise be unclear; code that already reads
  clearly needs none.
- **Never write the history.** A comment describes the code as it stands
  and the constraint it is under — not what it used to do, not the bug that
  prompted it, not "we tried X and it broke". `git log` and the PR hold
  that; a comment holding it goes stale silently and reads as an apology.
  State the rule, not the scar:

      # BAD:  we used to sort this later, which made replays diverge
      # GOOD: emitted sorted — the order reaches the model

- **A top-level import in an agent module is a per-game memory cost,
  and so is every object on the pickled agent.** Games run in spawned
  children that unpickle the agent, so the module's imports and whatever
  its fields drag in run again in every child's own heap — 240 of them
  in a submission. The engine is the host's alone: `platypus.subproc`
  asserts a child never has vllm and puts the cost on the assert. Import
  what only the host needs inside the function that needs it
  (`duck.agent._installed_vllm`), and keep host-only objects off the
  agent a child receives (`LLMAgent.without_engine`). `taaf` loads
  its submodules on first use for the same reason — `taaf.diagnostics`
  pulls matplotlib and scipy — and `taaf.grid` builds its colormap on
  first access; don't add an eager import to either
  (`packages/taaf/tests/test_taaf_import_cost.py` and
  `packages/duck/tests/test_import_cost.py` pin both).

## Tests and gates

Member-internal suites live in each member's `tests/`; the root `tests/`
holds cross-member integration and recorded expectations. `make prepare`
(format + lint + pyright strict + tests) is the gate, `make prepare-ci` is
what CI runs.

Expensive tiers are opt-in and gated twice (pytest marks plus runtime
guards): `make test-gpu` runs the GPU-marked tests where it is invoked
(inside an allocation), `make test-api` runs the paid-API tests (requires
the `ULLM_API_TESTS=1` opt-in it sets), and `make test-local` runs the
tests that need a machine we control. `make test-full` runs every tier.

**Pre-merge rule**: `make test-full` runs as the FINAL check before every
merge into the mainline — and only then; during development stick to the
cheap gate (`make prepare`) so allocations and API spend aren't burned on
intermediate states.

`nbstripout` is installed as a git filter, so a committed notebook keeps
none of its outputs: a conclusion that lives only in a rendered cell does
not survive the commit.

**Gates in one repo run one at a time.** Every gate target runs
`install-dev` first, which ends in `nbstripout --install`, and that writes the
`.git/config` all of the repo's worktrees share. Two gates started together in
worktrees of one repo race for that file's lock, and the loser fails at install
(`could not lock config file`) without running a test. Start the next once the
first is past install, or run them in turn; gates in different repos do not
collide.

**Unused imports are ruff's check, not pyright's.** `pyrightconfig.json`
turns `reportUnusedImport` off and says why; `F401` in `make lint` enforces
the same rule on every machine alike. A used-import warning from pyright
means the config moved, not the code.

## Test guidelines

- No trivial tests: never assert that a declared default equals itself,
  that a getter mirrors a field, or similar duplication of the code in test
  form. Every test must be able to catch a plausible real bug.
- Test outcomes, not internals.
- Recorded expectations (byte-exact replays of what a model was sent) catch
  *any* change to model-visible text and cannot catch that the text was
  wrong to begin with. They need a companion test asserting each intended
  branch's text is present, or a reference scripted to cover branches
  cannot notice it has stopped covering one.
- Committed references live in `tests/references/` and replay on the cheap
  tier. Rewriting one is never a side effect of running a tier; each rewrite
  has its own opt-in, and regeneration validates itself by an immediate
  replay:

      PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_<name>'
      PLATYPUS_REFRESH_VIEWER_HTML=1 make test TEST_ARGS='-k refresh_viewer_html'   # viewer slice only

  A green regeneration means the new recording reproduces, not that the
  change was intended — review the diff of the file as the behavioural
  claim it is.
- **The deployment snapshot** (`tests/references/deployment_snapshot.json`)
  freezes every column's whole configuration — the agent's fields, its
  `RunConfig` and `LLMConfig`, and the backend factory with its engine
  arguments — resolved at `prepare_for_backend`. It is the only test that
  reads a column's engine configuration whole rather than one value it
  names, so it is what notices a window, a
  watermark, a shard count or a KV connector moving; the replays above see
  none of that, and `settings.md` is what it holds the layers to. A diff is
  meant to be read and put in the PR, since it is the list of what a change
  did to every column.

      PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k deployment_snapshot'

- **A prompt is a string, so nothing else can see into it.** Neither lint
  nor pyright can tell that a prompt names a variable the kernel does not
  bind, and a recorded reference pins whatever was wrong when it was
  recorded. Model-visible text therefore needs a *golden* — every prompt
  the agent can render, committed as readable text, regenerated
  deliberately and reviewed as a diff (`packages/duck/tests/prompt_golden/`).
  Render it through the configured agent rather than the prompt class:
  fields the agent reads on the way past (a cutoff notice, a mode flag)
  are exactly the ones a direct call renders identically whatever they
  are set to, and a golden that pins nothing is worse than none.
