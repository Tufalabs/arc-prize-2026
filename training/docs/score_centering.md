# Score centering with truncated importance sampling

*Register: as-built; update with the objective and sampler-evidence contracts.*

`training.policy_loss="score_centered_tis"` selects score centering composed
with TIS (Marek & Ryabinin, 2026, [Eq. 12](https://arxiv.org/abs/2609.20807)).
For sampled token y, sampler head H, sampler probabilities q, current trainer
probabilities p, and the trainer's existing token advantage A:

    f(r) = min(r, C)
    train_tail = max(1 - sum_H p_v, 1e-6)
    samp_tail  = max(1 - sum_H q_v, 1e-6)
    rho = samp_tail / train_tail
    alpha = rho * f(1 / rho)
    c_v = q_v * f(p_v / q_v) - alpha * p_v
    objective = A * (sg[f(p_y / q_y)] * log(p_y)
                     - sum_H sg[c_v] * log(p_v))

The loss is the negative objective. Both weights and coefficients are detached;
only trainer log probabilities carry its gradient. Ratios are capped in log
space before exponentiation to avoid overflow. The sampled q_y remains the
rollout's sampled-token logprob, including when y is outside H. Head scores use
the same float32 normalization and admitted token bias as sampled scores.

## Configuration and aggregation

`score_centering_top_k` is a required positive integer; `tis_ratio_cap` is
finite and greater than 1, defaulting to 2.0. Both resolve to null under other
policy losses; non-null settings there are errors. Optimizer resume binds both
knobs and the policy loss. GRPO and Dr. GRPO advantages can use this loss.

Span weighting, loss aggregation, accumulation, entropy/KL terms, zero-signal
filtering and the window trust region retain their existing definitions. The
reported training clip fraction counts target tokens with p_y/q_y > C. The
post-update window trust-region probe still measures movement from the
pre-update policy with the configured PPO clip bounds.

Diagnostics under `train/policy_update/score_centering/` pool token sums across
microbatches and ranks: `mean_relative_correction` is the mean per-token ratio
of absolute correction to absolute sampled weighted-logprob term (denominator
floored at 1e-6); `mean_sampler_head_mass`, `mean_trainer_head_mass`, and
`mean_rho` describe the head and tail approximation. These exclude A and count
all target tokens actually used in forwards, including repeat cycles. The
same diagnostics are written to the training log.

## Capture and memory

The launcher passes `--score-centering-top-k` to the native server, which
raises the engine's `max_logprobs` to at least k. Runtime attestation checks
capacity, vocabulary size, configured k, and `processed_logprobs`. Training
requests k; evaluation and adapter warmup request 0. Engines without score
centering accept only 0; configured engines accept 0 or exactly k.

The native response adds `sampler_topk`: k plus two base64 strings
`ids_i32le` and `logprobs_f32le`, containing row-major little-endian int32 IDs
and float32 logprobs. vLLM may include the sampled token outside its top-k;
selecting the k largest logprobs excludes that extra entry. The ordinary
response retains sampled-only logprobs. The gateway validates dimensions,
unique in-vocabulary IDs and finite nonpositive logprobs, stores compact NumPy
arrays, and strips the extension before replying to the harness.

At 2,000,000 completion tokens and k=32, retained head arrays occupy
512,000,000 bytes (488.3 MiB). Base64 transfers about 682,666,667 bytes
(651.0 MiB), plus small per-request JSON overhead and padding. Capture,
trajectory and unpacked samples share arrays. Packing concatenates completion
rows; collation alone expands them to `[B, T, k]` int64 IDs and float32
logprobs, with zeros on non-target positions. Its head storage is 384 bytes
per padded target position at k=32. Packing can add another 488.3 MiB while original
rollouts remain live, and wire decoding has transient copies; these figures
are payload sizes, not peak process RSS.

## Recipe

`configs/grpo_6gpu_arc2_overfit_score_centering.json` is
`configs/grpo_6gpu_arc2_overfit.json` plus exactly:

| Key | Value |
| --- | --- |
| `inference.agent_terminate_on_empty_reply` | `true` |
| `training.policy_loss` | `"score_centered_tis"` |
| `training.score_centering_top_k` | `32` |
| `training.tis_ratio_cap` | `2.0` |

The pinned engine is vLLM 0.25.1 (`uv.lock`). Its
[SamplingParams API](https://docs.vllm.ai/en/v0.25.1/api/vllm/sampling_params/)
defines top-k logprobs plus the sampled token, and its
[model configuration](https://docs.vllm.ai/en/v0.25.1/api/vllm/config/model/)
defines `max_logprobs` and `processed_logprobs`. CPU fake-engine tests verify
the transport contract; real AsyncLLM processed-head values and GPU/DDP
numerics still require GPU validation before a production run.
