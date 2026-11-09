# tufa-arc-agi-framework (TAAF)

ARC-AGI3 orchestration framework by Tufa Labs: games, solvers, benchmarks, SLURM and Kaggle
deployment, and diagnostics. Our ARC-AGI-3 solver runs on it.

The driver document for this repository is [`docs/requirements.md`](docs/requirements.md).

License: [MIT-0](LICENSE).

## Quick start

```bash
make install-dev      # set up .venv + dev deps
make prepare          # ruff format + lint + pyright + pytest
```

See [`notebooks/demo_workflow.ipynb`](notebooks/demo_workflow.ipynb) for a happy-path tour of `Game`, `Solver`, and `Benchmark`.

## Kaggle Random Smoke

`SolverRandom` is TAAF's built-in solver example, so its Kaggle launcher lives
in this repo. The default Make target runs the 25 official games once,
CPU-only, with no action cap, and a 10-minute notebook runtime.

```bash
make kaggle-random \
  KAGGLE_RANDOM_RUN_NAME=taaf-random-smoke \
  KAGGLE_RANDOM_KERNEL_SLUG=taaf-random-smoke \
  KAGGLE_RANDOM_DATASET_REF=<your-kaggle-user>/taaf-release-source-random-smoke
```

For a local package check without uploading to Kaggle:

```bash
make kaggle-random KAGGLE_RANDOM_DRY_RUN=true KAGGLE_RANDOM_JOB_DIR=/tmp/taaf-random-dry-run
```

The direct CLI is also available:

```bash
.venv/bin/python -m taaf.kaggle_random \
  --kernel-slug taaf-random-smoke \
  --dataset-ref <your-kaggle-user>/taaf-release-source-random-smoke
```

## re-arc dependency

`pyproject.toml` takes `arc-agi-3-local` from `../re-arc-3` by path.
