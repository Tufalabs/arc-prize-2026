# Tufa Labs: ARC Prize 2026

The code behind Tufa Labs' ARC-AGI-2 and ARC-AGI-3 entries in ARC Prize 2026.
Our [paper](https://tufalabs.github.io/arc-writeups/paper/paper.html),
[ARC2 writeup](https://tufalabs.github.io/arc-writeups/arc2_writeup/arc2_writeup.html) and
[supplementary material](https://tufalabs.github.io/arc-writeups/supplementary/overview.html)
explain what it does and what each part is worth.

| Directory | What it is |
|---|---|
| [`arc2/`](arc2/) | The ARC-AGI-2 solver and its harness |
| [`arc3/`](arc3/) | The ARC-AGI-3 solver and its harness, including `ullm`, the LLM serving layer both solvers use |
| [`tufa-arc-agi-framework/`](tufa-arc-agi-framework/) | ARC-AGI-3 orchestration: games, benchmarks, SLURM and Kaggle deployment, diagnostics |
| [`re-arc-3/`](re-arc-3/) | Our collection of ARC-AGI-3-style games and the pipeline that generates them |
| [`training/`](training/) | Reinforcement learning, supervised fine-tuning, expert pruning and drafter training |

The inference engine is a fork of vLLM, kept separately because its version
comes from upstream's tags:
[`arc-prize-2026-vllm`](https://github.com/Tufalabs/arc-prize-2026-vllm).

Each directory is its own project with its own lockfile. Work inside one:

```bash
cd arc3
make install-dev
make prepare-ci
```

The projects depend on each other by relative path, so keep them side by side.

The 25 public ARC-AGI-3 games are not in this repository. See [`re-arc-3/`](re-arc-3/)
for how to install them from the competition's data.

## Running outside our cluster

Kaggle runs need only a Kaggle account that has accepted the competition's rules. The SLURM
side was written for our own cluster, and its specifics are left in place:

- **Checkpoints and results.** The SLURM presets read models from
  `/data/shared/research/checkpoints/` (`arc3/packages/seal/src/seal/presets.py`,
  `arc3/packages/ullm/src/ullm/backends/deployment.py`, `arc2/packages/duck/src/duck/presets.py`),
  and the demo notebooks write results under `/data/shared/research/`. Point these at your own
  storage; the submitted models are published on Kaggle.
- **Scheduler settings.** The account, partition and GPU type are in the Makefiles (`*_SRUN`,
  `ARC2_GPU_*`) and the arguments of `TufaSlurmTarget`.
- **Node layout.** Job scripts and `TufaSlurmTarget` assume node-local scratch space and shared
  storage under `/data`.
- **`training/`.** Its job scripts and recipes are the most tied to our cluster. Treat them as a
  record of how our runs were launched, not as something that runs elsewhere unchanged.
  `training/platypus` and `training/arc2-platypus` are symlinks to `arc3/` and `arc2/`, where
  the scripts expect the harness. They were submodules in our original repository, and its
  environment builders still assume that layout ([details](training/docs/environments.md#building)).

## Names in older documents

Documents and comments written before these repositories were merged use the earlier names:

| Name | Here |
|---|---|
| `arc3-platypus` | [`arc3/`](arc3/) |
| `arc2-platypus` | [`arc2/`](arc2/) |
| `arc-training`, `ARC3-Training` | [`training/`](training/) |
| `ARC3-Inference` | An earlier inference code base; not released |
| `arc2-experiments`, `arc3-experiments` | Our analysis notebooks and sweep reports; not released |

Pull request and issue numbers, and commit hashes other than the vLLM pin, refer to those
repositories' histories, which are not public.

License: [MIT-0](LICENSE), except for third-party material under its own license,
which each directory notes.
