# Historical ARC3 harness decisions (PR #54)

This is a compact archive of the decisions and experiment results that led to
PR #54 (`feat/full-duck-agent`, based on `527390d`) in July 2026. It explains
why several architectural constraints exist; it is not a launch runbook.
`README.md`, the checked-in configs, and the implementation are authoritative
for current values. In particular, episode lengths, reward penalties, batch
sizes, and topology details have changed since this design was signed off.

## Durable design decisions

- Training and official evaluation use the same agent, environment, sandbox,
  token-continuation path, context management, and compaction behavior. Their
  game sets and decoding settings intentionally differ.
- Prompt history is append-only within a segment. At the context threshold,
  whole old turns are removed and a new segment starts. The original setting
  was 32k to 16k; current configs own the actual thresholds.
- PPO trains only post-compaction generations in a new segment. The retained
  prefix is masked because it was generated in a different context and was
  already trained in the preceding segment.
- Compaction is not terminal. Credit assignment crosses segment boundaries;
  only the true finite episode horizon is terminal.
- Rollouts are collected in synchronous fixed-policy waves. An adapter changes
  only between waves, and mixed-policy or excessively stale windows are
  rejected.
- A training window is one collection wave. This bounds optimizer work and
  keeps the sample-budget controller tied to producer cadence rather than a
  batch interval that also includes training time.
- Sample budgeting is independent of rewards and stratifies per episode.
  Inverse-inclusion weights and the original window denominator preserve the
  full-population objective in expectation after subsampling.
- The default token-mean loss gives every generated token equal weight across
  the collection window. Packing, compaction, accumulation and DDP sharding
  preserve that mass; prompt and observation tokens are excluded.
- Variance sampling starts from the pinned Duck baseline. Missing or mismatched
  baseline coverage is a launch error rather than a silent uniform fallback.
- Training excludes both `official` and `official-reimplemented` games. The
  latter exclusion prevents mechanics-level leakage into the 25-game official
  evaluation split.
- The canonical 8-GPU and 16-GPU configs share learning settings but have
  explicit, different topologies. A multi-node layout is not safely expressible
  as a simple GPU-count override.
- Training originally started from scratch because the prompt and context
  distribution changed substantially. Check current configs before assuming
  that remains the intended initialization policy.

## Intentional deviations from the Duck reference

- The persistent Python REPL and segmentation tools were retained instead of
  Duck's ephemeral ASCII frame/history surface. They expose the same board
  information without invalidating the trained sandbox contract.
- Evaluation uses deployment-like decoding while training samples the policy
  at temperature 1.0. Using sharpened evaluation decoding for PPO collection
  would make stored behavior log-probabilities off-policy.
- FP8 serving was rejected by the parity gate below. Rollout and evaluation
  servers therefore use bf16 unless a new, passing gate supersedes this record.
- Multimodal input was not part of the original shipped text-only contract.
  Later vision work required exact prompt-prefix verification, replayed image
  tensors, trainer-side multimodal forwards, and a frozen vision tower; it
  should not be enabled piecemeal.

## FP8 parity gate: failed 2026-07-09

The proposed deployment used the pre-quantized
`vrfai/Qwen3.6-27B-FP8` base for rollout and evaluation while keeping the
trainer and LoRA in bf16. The gate compared rollout-server log-probabilities
with trainer recomputation and also exercised runtime LoRA hot-loading.

Run `20260709_230313_fp8_parity_gate` produced:

| window | tokens | mean absolute diff | p95 | max | outliers over 2.0 | signed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3811 | 0.0480 | 0.256 | 4.83 | 3 | -0.0101 |
| 2 | 4698 | 0.0411 | 0.250 | 1.82 | 0 | -0.0068 |
| 3 | 3724 | 0.0480 | 0.263 | 2.95 | 3 | -0.0062 |

The bf16 baseline had mean absolute differences around 0.012 to 0.016, p95
around 0.08, no outliers, and signed mean around -0.001. W8A8 therefore showed
roughly three to four times the mean divergence, outliers in two of three
windows, and systematic negative bias. The parity guard skipped optimizer
steps in windows two and three, so a longer run would stall and then abort.
LoRA hot-loading itself succeeded and generations remained coherent; the
failure was numerical policy mismatch, not server startup.

Decision: keep bf16 servers. A weight-only W8A16 base on both serving and
training sides remained a possible follow-up, but it requires a trainer loader
that did not exist. Do not enable online FP8 quantization or set
`inference.server_model` without repeating the parity, hot-load, and generation
coherence gates.

## Unresolved questions at the time

- Whether a larger context/compaction pair is worth its roughly doubled
  trainer-memory cost.
- Whether the per-turn generation cap should increase beyond 1024 tokens.
- Whether a future quantized serving stack can match trainer policy
  probabilities closely enough for PPO.
- Whether `gae_lambda=0.95` is preferable to a full Monte Carlo return for ARC's
  sparse, delayed rewards; this requires an ablation rather than argument from
  theory alone.
