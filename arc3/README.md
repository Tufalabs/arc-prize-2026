# ARC-AGI-3 solver

Tufa Labs' solver for the ARC-AGI-3 track of ARC Prize 2026. An LLM plays each
game from a persistent Python REPL, where it can inspect frames, write code and
send actions. The harness hosts the REPL and the game, scores the play and
reports what each game cost.

- [Writeup](https://tufalabs.github.io/arc-writeups/arc3_writeup/arc3_writeup.html) and
  [supplementary material](https://tufalabs.github.io/arc-writeups/supplementary/overview.html)
- [Submission notebook](https://www.kaggle.com/code/jeroencottaar/tufa-labs-arc3-final-solution)

The submitted agent is `SealAgent.build(environment="kaggle-rtx")`, with its
settings in `packages/seal/src/seal/presets.py`. This repository is a cleaned
copy of the code behind that notebook.

## Layout

uv workspace monorepo:

- `packages/platypus/` — infrastructure layer: agent contract + hosting
  machinery (documented by the top-level `design.md`)
- `packages/duck/` — the duck agent: the simpler harness we released for
  ARC3 milestone 1, and the one our reinforcement learning trains in
- `packages/seal/` — the seal agent, as submitted: the duck's control loop,
  with a world model the model writes
- `packages/reference-agents/` — reference/baseline agents (random,
  prompted-random)
- `packages/sweeper/` — parameter sweeping: one run covering several agent
  configurations (arms are passes), plus the collector that turns finished
  jobs into rows
- `packages/framediff/` — what changed between two frames, as a short
  numpy edit script (documented by `packages/framediff/README.md`)
- `packages/ullm/` — LLM layer (hard fork of the standalone ullm repo)
- `tests/` — cross-member integration + committed reference replays;
  member-internal suites live in each package's `tests/`
- `notebooks/demo.ipynb` — one agent on one deploy target, as a worked
  example; nothing depends on it
- `tools/` — demo and ops scripts, and the traffic benchmark's launcher,
  workload cutter and archive (`packages/ullm/traffic_benchmark.md`);
  `tools/teachers/` — the distillation teacher played through its API, for
  SFT traces (`tools/teachers/README.md`)

## Documentation

- `AGENTS.md` — way of working, coding/testing guidelines, and the doc
  map
- `requirements.md` — the agent contract (normative)
- `settings.md` — where a setting lives: the eight buckets, and every
  parameter placed in one
- `design.md` — the as-built high-level design
- `agent_authoring.md` — how to write an agent
- `roadmap.md` — direction and open work
- per-package docs: `packages/duck`, `packages/framediff`,
  `packages/seal` and `packages/ullm` each have a README mapping their own;
  the platypus package is documented by `design.md` above

## Prerequisites

Linux, `make` and [uv](https://docs.astral.sh/uv/). uv fetches the pinned
dependencies during the sync. Python
3.12.12 is pinned and uv installs it for you; no system Python is used.

**Budget the first install.** `vllm` and a pinned CUDA toolchain are
plain dependencies, not an extra, so the first `make install-dev` pulls
several GB and takes tens of minutes on a cold cache. This is
deliberate — `ullm.backends.vllmx*` imports everywhere, and only
*running* an engine needs a GPU — but it means the initial sync wants a
disk with room and a session you can leave alone. Later syncs are fast.

## Development

```
make install-dev   # uv-managed venv + dev deps (all workspace members)
make prepare       # format + lint + pyright + tests (run before pushing)
make prepare-ci    # what CI runs
make test-full     # every tier incl. GPU (SLURM) + paid API — pre-merge only
make viewer        # diagnostics viewer + reasoning replay (two links)
```

The SLURM targets default to our cluster's account and partition names. Pass
your own through the Makefile's `*_SRUN` variables and `TufaSlurmTarget`.

The 25 public ARC-AGI-3 games' code is not in this repository. `make
install-dev` fetches it from the competition's data if your Kaggle account has
accepted the competition's rules, and a SLURM job takes its launcher's copy. A
Kaggle run installs it from the competition's own files, so a source bundle never
carries it. Without the code, the runs that play those games fail and the tests
that need them skip.

License: [MIT-0](LICENSE). The bundled fonts keep their own license.

## A first run, on a CPU

No GPU, no API key, no allocation — this plays a toy game with an
offline stub backend and leaves a browsable diagnostics run behind:

```
make install-dev
uv run python tools/demo_diagnostics_run.py --serve
```

`make viewer` prints two links into one server: the canonical
diagnostics pages, where per-move span trees, conversations, and frames
are read, and the ARC3 Replay Viewer at `/_replay/`, whose run picker
understands both direct run directories and nested experiment/seed
layouts and moves the board to the frame behind any clicked reasoning,
level, or action entry. It displays both standard Duck context rewrites
and summarisation compaction, including the summary request and reply,
whether the notes were saved, and the context used after the reset.
The viewer reads Platypus diagnostics directly and needs no trainer.

From there: `agent_authoring.md` to write an agent,
`packages/reference-agents/` for the two simplest worked examples, and
`notebooks/demo.ipynb` for the same shape at full scale (a 27B model on
SLURM or Kaggle — that one costs an allocation and says so).

Duck-specific runtime documentation and comparison controls live in
[`packages/duck/README.md`](packages/duck/README.md).
