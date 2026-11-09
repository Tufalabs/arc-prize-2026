# ARC-AGI-2 solver

Tufa Labs' solver for the ARC-AGI-2 track of ARC Prize 2026. An LLM solves each
task in a persistent Python REPL and places its answer grids there. The harness
hosts the REPL, scores the answers and reports what each game cost.

- [Writeup](https://tufalabs.github.io/arc-writeups/arc2_writeup/arc2_writeup.html) and
  [supplementary material](https://tufalabs.github.io/arc-writeups/supplementary/overview.html)
- [Submission notebook](https://www.kaggle.com/code/jeroencottaar/arc2-demo-duck-xkbl4-final)

The submitted agent is `TieredTakeDuck.build(environment="kaggle-4xl4")`, with
its settings in `packages/duck/src/duck/presets.py`. This repository is a
cleaned copy of the code behind that notebook.

The LLM layer `ullm` comes from `../arc3/packages/ullm`.
Depending on it also pulls vLLM, so a sync installs the CUDA stack whether or
not a run uses it.

## Layout

uv workspace monorepo:

- `packages/` — workspace members: `taaf` (orchestration: games, datasets,
  deployment, diagnostics, the Kaggle submission), `platypus` (the agent
  contract and the machinery that hosts one), `duck` (the solver), and
  `sweeper` (parameter sweeping: one run covering several agent
  configurations — arms are passes — plus the collector that turns finished
  jobs into rows)
- `tests/` — cross-member integration tests and recorded expectations;
  member-internal suites live in each package's `tests/`
- `tools/` — demo and ops scripts; `tools/teachers/` — the distillation
  teacher played through its API, for SFT traces (`tools/teachers/README.md`)

## Documentation

- `AGENTS.md` — way of working, coding/testing guidelines, and the doc map
- `roadmap.md` — direction and open work

## Prerequisites

Linux, `make`, and [uv](https://docs.astral.sh/uv/). Python 3.12.12 is
pinned and uv installs it for you; no system Python is used.

## Development

```
make install-dev   # uv-managed venv + dev deps (all workspace members)
make prepare       # format + lint + pyright + tests (run before pushing)
make prepare-ci    # what CI runs
make test-full     # every tier incl. GPU + paid API — pre-merge only
```

The SLURM targets default to our cluster's account and partition names. Pass
your own through the Makefile's `*_SRUN` variables and `TufaSlurmTarget`.

License: [MIT-0](LICENSE). The ARC-AGI-2 tasks in `packages/taaf/src/taaf/data/`
come from [ARC Prize's dataset](https://github.com/arcprize/ARC-AGI-2) under
Apache-2.0, and the bundled fonts keep their own license.
