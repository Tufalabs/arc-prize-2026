# ARC-3 multimodal RL

ARC-3 can train on Duck conversations containing grid images, reasoning,
tool calls and tool responses. Enable images in an existing ARC-3 recipe:

```json
{
  "inference": {
    "environment": "arc3",
    "agent_multimodal_context": "current_grid"
  },
  "training": {
    "multimodal": true,
    "value_baseline": "group_mean",
    "advantage_clip": 0
  }
}
```

This is a configuration excerpt. Keep the model, game, sampling and training
settings from your complete recipe. Use an image-capable checkpoint and pin
`training.model_revision` to the same revision used for serving.

Image observations require the `group_mean` baseline for both PPO and DPPO.
The `hidden_critic` baseline does not yet replay image inputs during feature
extraction, so image-enabled recipes using it are rejected when loaded.
When adapting a text-only hidden-critic recipe, change `value_baseline` and
`advantage_clip` as shown above and allow at least two rollouts per sampled
game. For example, the 25-game PPO recipe needs `parallel_rollouts: 50` when
sampling all 25 games in each window. The checked-in
`configs/dppo_2gpu_arc3_multimodal_smoke.json` already uses `group_mean` and
four rollouts for one game.

`current_grid` attaches the current grid image to every ordinary Duck user
turn. `level_start` attaches one image at the start of each level. The default
empty string retains text-only observations. These settings use Duck's
existing frame renderer and image scaling.

Duck keeps its normal tools, including action batching where supported, and
their responses remain in the conversation. Reasoning and model-generated
tool calls are policy targets. Observations, image inputs and tool results
condition the policy and have no policy-loss targets, as in text-only RL.

## Serving and training

Use `scripts/platypus_debug_smoke.sbatch` with the modified recipe. It derives
the server image setting from the config, retains the vision encoder and
installs the locked FlashAttention 4 runtime. When starting the server
manually, set `SERVER_ENVIRONMENT=arc3` and
`SERVER_MULTIMODAL_CONTEXT=current_grid` (or `level_start`) as well.

The ARC-2 media path is shared: the native gateway records image bytes,
image order, token spans and processing evidence. The trainer reproduces
the pixel tensors and multimodal positions and checks them against that
evidence. Episode packing retains all images needed by the sampled history;
the existing rollout/trainer logprob parity gate remains active.

Checkpoint identity includes the image schedule and processing stack.
Resuming with a different image mode or processor is rejected. A legacy
text-only checkpoint cannot silently initialize a multimodal resume.

Image history increases context length and training memory. Start with a
small rollout concurrency and measure memory and parity before scaling up.

## Rollout augmentation

`"inference": {"augmentation": {"enabled": true}}` (off by default, text or
image runs) recolors every frame of a training rollout. One permutation of all
16 palette colors is drawn per game per training window, from
(`training.seed`, the window's schedule index, game), and shared by the game's
attempts in that window: the whole palette, since a first frame need not show
every color a later level uses. `ColorPermutedGameAPI`
(`harness_support/arc3/augmented_game.py`, built where
`harness_support/arc3/worker.py` makes each slot's game) applies it at TAAF's
`_start_game` and `_execute_action` hooks, where every frame of the run enters
the framework: all levels, animation frames, and TAAF's own level-win frame.
The Duck's text boards and its `current_grid`/`level_start` images therefore
show only recolored frames, and the palette legend still names the color of
each index shown. Actions reach the engine unchanged, and the score (levels
completed against baseline actions) is the engine's own: re-arc-3's recorded
seven-level `anchor_ice` solve, replayed recolored, scores the same and
reconciles with the engine's scorecard (`tests/test_arc3_augmentation.py`).

There are no flips or rotations: `"dihedral": true` is refused, since they
change what directional and click actions mean. Evaluation is never
augmented. Each rollout's metadata records its color map (`augmentation`),
which the trainer re-derives and checks, and the checkpoint identity records
the setting: a checkpoint resumes only under the augmentation it was trained
with, including a legacy checkpoint that has no identity at all.
