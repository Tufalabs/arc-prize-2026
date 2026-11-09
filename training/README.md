# 🚀 Training for ARC-AGI

Supervised fine-tuning and reinforcement learning for language-model agents
solving **ARC-AGI-2** tasks and playing **ARC-AGI-3** games, as used for Tufa
Labs' ARC Prize 2026 entries. Our [paper](https://tufalabs.github.io/arc-writeups/paper/paper.html) and
[supplementary material](https://tufalabs.github.io/arc-writeups/supplementary/overview.html) report the results.

License: [MIT-0](LICENSE).

Recipes and reports name the harness revisions their runs used, which are in
our private history. The `platypus` and `arc2-platypus` links here are `../arc3` and `../arc2`, and
the training code was checked against it by CPU tests only: no training run has
been repeated on it.

## Workflows

| Workflow | Use | Recipes and documentation |
| --- | --- | --- |
| ARC-2 RL | Grid reasoning with multimodal GRPO | [ARC-2 training](docs/arc2_training.md) |
| ARC-3 RL | Interactive games with PPO or DPPO | [Multimodal training](docs/arc3_multimodal_training.md) · [Training recipes](docs/dppo_training.md) · [Nine-GPU official-25 experiment](docs/recipes/official25_9gpu.md) |
| Pruning and drafter distillation | Prune Flash-Next experts, then train and export its speculative drafter | [Pruning and drafter recipe](spec_decode/README.md) |
| SFT | Learn from TAAF traces, including image-based examples | [SFT guide](sft/README.md) |

The ARC-2 guide covers the pinned harness, the overfitting and held-out
recipes, and GPU validation. The nine-GPU recipe records an ARC-3
experiment, including its warm-start checkpoint and deployment provenance.

RL captures exact tokens, logprobs and rewards through the native serving
gateway, then updates the policy with training/serving parity checks. SFT
trains on recorded traces and shares the runtime adapter export contract.
Both retain configuration, metrics and checkpoints.

See [native serving](docs/native_serving.md) and
[full Duck evaluation](docs/full_duck_eval_design.md).

Retained RL windows on job-local scratch are archived and synced in the
background after training finishes reading them. See
[rollout artifact retention](docs/rollout_artifacts.md) for CPU/NFS limits,
shutdown behavior, and extracting a window.

## Run SFT

Set the trace paths in `configs/sft.json`, then:

```bash
make sft-check SFT_CONFIG=configs/sft.json
make sft-train SFT_CONFIG=configs/sft.json
```

## Run ARC-3

On a configured cluster:

```bash
export ARC3_RUN_DIR=/data/personal/$USER/training_runs/$(date +%Y%m%d_%H%M%S)_my_run
export ARC3_CONFIG=configs/ppo_2gpu_arc3_games25_platypus.json
sbatch --account=research --export=ALL scripts/platypus_debug_smoke.sbatch
```

The launcher builds the pinned environment on the compute node and starts
serving and training. Point `ARC3_RUN_DIR` at an existing compatible run to
resume it. The [nine-GPU recipe](docs/recipes/official25_9gpu.md) documents its
separate device mapping and two-node deployment requirements.

Game sampling defaults to `sampling.game_sampling_mode="balanced"`;
weighted sampling requires explicitly selecting `"nonzero_rate"`.
Balanced runs plot `rollout/avg_arc_score_sampled` (the current window mean).
The history-based `rollout/avg_arc_score` is only plotted for weighted runs
with recorded scores. It equally averages the observed games' history means;
unsampled games are excluded, and a sampled zero score still counts.

## Development

```bash
make dev-install
make test
make tensorboard
```

`training/` contains RL and serving contracts; `sft/` contains supervised
fine-tuning; `configs/` holds experiment recipes.
