# Official-25 nine-GPU DPPO recipe

[`configs/dppo_9gpu_arc3_official25_hidden_critic_seed61.json`](../../configs/dppo_9gpu_arc3_official25_hidden_critic_seed61.json)
adds the learner settings used by
`20260914_075232_platypus_official25_dppo_production_resume30_seed61_9gpu`
to main's configuration schema. The [historical record](official25_9gpu_20260914.json)
preserves the frozen settings, deployment commits and initialization lineage.
This is a recipe port, not a claim that current main reproduces the historical
runtime or its scores exactly.

## Training settings

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen3.8-27B BF16, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPU roles | 4 DDP trainer + 4 rollout servers + 1 evaluation server |
| Games and collection | All 25 official games; 100 attempts/window; 15 minutes/attempt; start at level 1 |
| Schedule | One-window overlap, buffer size 1, one epoch, full sample budget |
| Minibatch | 25 rollouts: 4 optimizer updates per full 100-rollout window |
| Objective | `dppo_binary_tv`, TV threshold 0.15, importance-ratio cap 5, global token-mean loss |
| Baseline | Hidden critic; per-game target normalization; gamma = lambda = 1 |
| Critic normalization | Alpha 0.125, minimum 16 rollouts, minimum standard deviation 1 |
| Learning rate / seed | 5e-6 / 61 |
| LoRA | Rank 32, alpha 64, all ten recorded projection types; dropout disabled |
| Training context | 32,768 total; 8,192 reply reserve; zero margin; trim trigger 24,576; target 14,336 |
| Sampling | xhigh; temperature 1, top-p 1, top-k 0; no malformed-output retries |
| Packing | Episode packing; max sequence 32,768; max microbatch tokens 40,960 |
| Regularization | Zero entropy, reference KL and advantage clipping |
| Integrity | Existing sampled-token policy-parity checks remain enabled |

The 90,112-token context and 33,925-token compaction target discussed for the
standalone BF16 evaluation are **evaluation settings**, not the training
context. The 90,000 generated-token allowance per level belongs to that
standalone comparison; this training recipe uses a 15-minute rollout horizon.

## Initialization and historical results

The September 14 run resumed the September 13
`20260913_130049_platypus_official25_dppo_production_fresh_overlap_seed61_9gpu`
run's checkpoint 30, including trainer state and 124 optimizer steps. It did
not begin with a fresh base policy. Checkpoint 242 contains 972 persisted
optimizer steps. A fresh run with this recipe cannot reproduce that checkpoint
without the preceding training history.

The initial deployment used training `353dc0721dc32c69a36e2bf0ee240aac3dd2b21a`
and harness `977964e51b4a719837bcfc170dd5dc0a48c610dc`. The later NFS workspace
reload repair used training `295fda6775bca1d5045cc8a7c948b5e7975f7deb` and harness
`d5062d44fb92c1fd719beedd005303405898d277`. The historical record contains the
SHA256 of the inspected frozen configuration; allocation URLs, venv and log
paths are omitted from that record. All historical training fields remain.
The frozen config omitted LoRA dropout, but the actual checkpoint adapter
records zero dropout; the portable recipe makes that explicit.

Main has since integrated the learner through PR79. Its checkpoint compatibility
checks still apply: this recipe does not make historical critic/optimizer
checkpoints compatible. Do not bypass those checks or describe an adapter-only
transfer as a full-state continuation.

## Configuration mapping and launch topology

The preset uses logical trainer IDs 0–3, rollout IDs 4–7 and eval ID 8. The
historical two-node launcher used 0–3 and serving labels 8–12, then mapped the
serving role onto its own five local devices. GPU IDs here specify disjoint
roles, not a claim that a single eight-GPU host can allocate nine devices.
Use a suitable allocation/orchestrator and remap node-local device IDs and
server endpoints before launch. The historical heterogeneous launcher is not
added by this recipe PR; main's single-node smoke launcher is not a drop-in
replacement for that two-node deployment. Existing serving setup is documented
in [native serving](../native_serving.md).

Validate the configuration without allocating GPUs:

```sh
python -c 'from training.config import ExperimentConfig; ExperimentConfig.load("configs/dppo_9gpu_arc3_official25_hidden_critic_seed61.json")'
```

After the serving processes and device mappings have been configured, pass the
recipe through the existing `python -m training.cli --config ...` entrypoint.
Use a new run directory for a fresh experiment. The CLI's `--run-dir` resumes
an existing compatible `training_state.json`; initialization paths remain null
in the shipped preset so it does not reference another user's checkpoints.

Schema adaptations are explicit:

- Historical journal `agent_max_reset_prompt_tokens` maps to main's
  `agent_compact_to_tokens`. Main always starts full games at level 1; the
  removed `level_curriculum="first"`, level-data, boundary-compaction and
  summary-mode controls are not restored.
- Main uses its pinned ordinary Duck defaults for journal memory, prompt
  profile and reasoning prefill. Historical overrides are retained in the
  record, not represented as supported fields in the runnable preset.
- Main's existing evaluator receives the ninth GPU, all 25 games, 45-minute
  cycles, a 90,112-token context, target 33,925 and the historical sampling
  settings. It inherits the **8,192 reply cap and zero margin** from training.
  The historical production evaluator instead used a **32,768 reply cap and
  512 margin**, plus a separate full-main worker and token-stat reporting.
  Those evaluator extensions are not part of this learner-recipe port, so its
  inline eval scores are not interchangeable with the historical evaluator or
  the separate BF16 comparison.
- Retired `metrics.log_raw_returns=false` is omitted. No learner objective,
  data window, minibatch size or optimizer setting is changed by the port.
