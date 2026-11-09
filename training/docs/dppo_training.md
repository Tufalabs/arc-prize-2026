# DPPO and full-game training

These recipes extract the learner improvements from PR77 without its level-start
curriculum or summary-compaction adapter. Every episode starts at level 1 and
uses the existing Duck journal compaction, gateway, and ARC-score feedback.
PPO remains available; select `training.policy_loss="dppo_binary_tv"` for DPPO.

[Dr. GRPO](dr_grpo.md) is the ARC-2 GRPO path without standard-deviation
normalization and with a fixed loss denominator.

## Two-GPU AR25 recipes

| Setting | `dppo_2gpu_arc3_ar25.json` | `dppo_2gpu_arc3_ar25_hidden_critic.json` |
| --- | --- | --- |
| GPUs | 1 rollout + 1 trainer | 1 rollout + 1 trainer |
| Collection | 25 AR25 attempts, 15 minutes each | Same |
| Context / reply cap | 32,768 / 8,192 tokens | Same |
| Reasoning / sampling | xhigh; temperature 1, top-p 1, no top-k or penalties | Same |
| Baseline | Current window's same-game mean | Hidden-state critic |
| Loss aggregation | Sequence token sum / fixed 32,768 denominator | Global trained-token mean |
| Learning rate | 1e-5 | 5e-6 |
| Critic target normalization | Not applicable | Per-game running mean and standard deviation |

Both recipes use BF16, rank-32 LoRA, zero dropout, ordinary journal compaction,
no malformed-output retries, and synchronous collection. Evaluation is disabled
in these small recipes. Launch them through the existing launcher after setting
the GPU allocation and model cache for the host.

## Nine-GPU official-25 recipe

The [nine-GPU recipe and historical provenance](recipes/official25_9gpu.md)
record the seed-61 run with four trainers, four rollout GPUs and one evaluator.
It uses 100 full-game attempts/window, one-window overlap, 25-rollout optimizer
minibatches, the hidden critic and token-mean DPPO. The recipe documents its
checkpoint-30 warm start, training/evaluation compaction differences, and the
limits of reproducing a historical deployment on current main.

## Objective and advantages

DPPO uses the sampled-token binary total-variation gate (`dppo_tv_threshold=0.15`)
and a detached importance ratio capped at 5. Its configuration enforces one pass
through a complete collection window, with rollout minibatches of 16 by default.
It disables PPO's asymmetric clipping, window rollback threshold, extra passes,
KL penalties, entropy bonus, and advantage clipping. A smaller last minibatch
uses its actual size. Microbatching and DDP do not redefine the loss denominator.

With `value_baseline="group_mean"`, every original attempt contributes one final
ARC score to its game's mean, including attempts without trainable output.
The advantage is `(score - same_game_mean) / 100`, frozen before packing or
minibatching. There must be at least two attempts with the same game, budget,
and policy in each window. There is no per-game standard-deviation division:
that would change the relative importance of games with different score variance.
This baseline does not maintain a historical EMA or train a critic.

With `value_baseline="hidden_critic"`, the critic conditions on the detached
policy hidden state at the last prompt token and the state *before* generation,
including the request's actual remaining time. GAE propagates ARC-score deltas
across ordinary compaction boundaries. Gamma and lambda default to 1, so the
finite rollout horizon has zero terminal bootstrap and no discount between
model decisions. The critic does not propagate gradients into the policy.

The optional `state_critic_target_normalization="per_game"` normalizes critic
regression targets using running moments. Output-head and optimizer-state
rescaling preserve raw value predictions when those moments change. Sparse
or unseen games retain a safe fallback scale. Values and actor residuals remain
in ARC-score units; per-game target scaling does not reweight actor rewards.
The actor then centers and scales advantages once using generated-token weights
over the complete window, before packing, filtering, minibatching, or DDP.

## Integrity and continuation

The existing trainer-owned gateway remains authoritative for sampling settings,
append-only token history, permitted journal compaction, and score reconciliation.
Malformed replies remain in that history. A final captured draw that misses the
rollout deadline is retained without inventing a malformed-output penalty.

Before updating, the trainer spot-checks sampled-token logprobs under the actual
collection adapter. For one-window overlap it temporarily restores a cached
sampler adapter, then restores current training weights without advancing the
optimizer or RNG. Checks rotate through initial, reset, and continuing contexts.
The usual sample is 128 tokens, with expanded coverage when needed to measure
rare outliers. Mean/p95 absolute error thresholds are 0.05/0.25 nats; errors above
2 nats may affect at most 0.1% of the checked tokens. A failed check stops training.
This is sampled evidence, not a guarantee about every token or a fixed overhead.

Synchronous collection waits for the latest update. Optional overlapped
collection pins one adapter for an entire window and bounds staleness to one
window. Initial rollout and eval collection use the initialized LoRA adapter.
Scheduled termination finishes active DDP protocols before cleanup. Worker TERM
signals are forwarded to rank zero, including during startup, so signaling all
ranks does not disable scheduled continuation. Each vLLM GPU worker installs the
Blackwell GDN guard after device selection and before model loading or warmup.

## Configuration migration

The shipped PPO configs explicitly select the hidden critic and token-mean loss.
The feature-only critic, critic-only warmup, optional void-token filtering, and
reward-shaping penalties are removed. The penalty fields accept zero only for
configuration compatibility. Old critic checkpoints are intentionally rejected
rather than interpreted with the new state and normalization schema; start fresh
or explicitly transfer an adapter without the old critic/optimizer state.
The old synthetic-logprob benchmark hook is rejected because fabricated scores
cannot pass the mandatory parity check. The DPPO recipes do not change vLLM's dependency pin or start any experiment.
