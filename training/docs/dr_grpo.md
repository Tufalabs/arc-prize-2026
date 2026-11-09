# Dr. GRPO

*Register: as-built; update with the estimator and aggregation contracts.*

`training.advantage_estimator="dr_grpo"` is ARC-2 GRPO with exactly two
objective changes. Everything else — the window contract, group construction,
collection schedule, sample budget, extra train cycles, gradient accumulation,
clip range, window clip-fraction guard, zero-signal filtering and the LoRA/DDP
replica path — is the ordinary GRPO code path, configured by the recipe.

## 1. No standard-deviation normalization

`training/grpo.py` takes a `normalize_std` flag, selected in
`PPOTrainer._compute_step_signals` by the estimator name. For episode i in a
group of G attempts at one puzzle, with total episode reward R_i:

    group_mean = sum(R_j for j in the group) / G
    advantage_i = R_i - group_mean          # dr_grpo
    advantage_i = (R_i - group_mean) / (group_std + 1e-8)   # grpo

The baseline includes the episode itself. A group whose rewards are all equal
contributes zero advantage, as under GRPO. The advantage is the same on every
step of the episode; `StepSignal.reward` carries R on the final step and
`return_to_go` is R throughout. There is no leave-one-out baseline,
discounting, reward-to-go calculation, advantage clipping or value model.

## 2. Fixed loss weighting: `loss_aggregation="token_sum_fixed"`

`token_sum_fixed` uses `token_mean`'s per-token weights (see
`assign_loss_weights` in `training/episode_packing.py`) and then multiplies
every accumulation-cycle denominator by

    N * training.loss_normalization_tokens / T

where N is the number of original episodes in the window and T is the window's
raw generated-token mass (`_fixed_token_sum_scale` in `training/ppo.py`). The
window's token sum is therefore divided by `N * loss_normalization_tokens`
instead of by the window's raw token mass. The zero-signal denominator scale still
applies, so filtered episodes stay in N.

Relative to `token_mean` this is **one scalar per window**: it changes how
windows of different total length weigh against each other, not how tokens
weigh against each other within a window. Subsampling gives retained samples
inverse-inclusion weights and keeps the full window's loss denominator, so the
gradient is preserved in expectation, with more sampling noise.

`loss_normalization_tokens` defaults to 32768 (shared with
`sequence_sum_fixed`); the recipe sets it explicitly. Under any other
`loss_aggregation` it resolves to null, and setting it is a validation error
because it would have no effect. It is a fixed
gradient-scale reference, not a generation cap, observed token count,
context-window length or compaction threshold. Increasing it shrinks the raw
gradient uniformly, so the learning rate and gradient clipping may need
retuning.

`dr_grpo` requires `loss_aggregation="token_sum_fixed"`; configuration loading
rejects other aggregations.

## Recipe

`configs/dr_grpo_6gpu_arc2_overfit.json` is `configs/grpo_6gpu_arc2_overfit.json`
plus exactly four settings:

| Key | Value |
| --- | --- |
| `inference.agent_terminate_on_empty_reply` | `true` |
| `training.advantage_estimator` | `"dr_grpo"` |
| `training.loss_aggregation` | `"token_sum_fixed"` |
| `training.loss_normalization_tokens` | `80000` |

It therefore keeps lagged overlapped collection, the adaptive sample budget,
extra train cycles, gradient accumulation, the 0.3 clip range and the 0.5
window clip-fraction guard from the GRPO recipe. Use a fresh run: optimizer
checkpoints bind the estimator and the recipe.

See [the Dr. GRPO paper](https://arxiv.org/abs/2503.20783) for the two
normalization changes.
