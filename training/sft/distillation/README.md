# Cached-teacher router distillation

Maintained ARC implementation for training only the routers of a pruned Qwen Flash
student derived from the frozen e307 teacher. This is a dedicated SFT-package
training loop: it does not use the LoRA/QAT trainer or alter ordinary SFT behavior.

## Model-stack boundary

Reusable Qwen3.8-Flash checkpoint loading, QSA/FlexAttention, packed-document
execution, grouped BF16 experts, pointwise kernels, and checkpoint/offload
helpers live in `training.flash`. This package owns the cached-teacher objective,
router-only recipe, launch controls, telemetry, and resume gates. Its
`sft.distillation.flash_model.build_model` adapter explicitly selects and verifies the
qualified router-only trainability policy. Other trainers must call the shared
builder with their own trainability and distribution policies rather than import
the distillation recipe.

## Batching default

New runs default to length-aware batching with pool_size=128, grouping similar
whole-sequence lengths across the four ranks. Set
`"batching": {"policy": "random"}` to use the legacy random order, including when
resuming older random-order checkpoints. Resume manifests reject a batching-policy
change. Existing running jobs keep their pinned source and configuration.

## Kernel mode

`kernel_mode` selects the student's execution stack from the two presets in
`training/flash/kernel_mode.py`, shared with the RL learner
([training/FLASH_RL.md](../../training/FLASH_RL.md#kernel-mode)):

- `"reference"` (default): the stack the existing distillation results were
  produced with. These are the builder's defaults plus whatever the config sets:
  SDPA attention (or `flex_qsa` with `physical_sparse`, as the published
  e205/e257/e276 configs), native convolutions, the batched FP32 selector,
  `weighted_permutation` experts, the fused norm tail, full layer checkpoints,
  FLA's autotuner unguarded. Existing configs, which have no `kernel_mode`, build
  exactly what they built before.
- `"fast"`: the RL learner's stack: `flex_qsa` with the `group_sparse` kernel,
  FLA and compiled convolutions, the compiled selector, `triton_manual`
  hyper-connections, fused GDN norm, MLP gates, GDN QKV and QSA norm,
  `fused_permutation` experts, `small_gemm` checkpoints, replicated frozen
  routers and the Blackwell GDN guard.

Any stack field set in the config overrides that one choice (`attention_backend`,
`qsa_kernel`, `gdn_convolution`, `ple_convolution`, `expert_optimization`,
`selector_policy`, `hyper_connection`, `fuse_norm_tail`, `fuse_gdn_norm`,
`fuse_mlp_gates`, `fuse_gdn_qkv`, `fuse_qsa_norm`, `pointwise_rounding`,
`checkpoint_policy`, `compile_layers`, `compile_precision`,
`replicate_frozen_routers`, `gdn_guard`). The switch does not touch placement
and data choices: `offload_every`, `reuse_qsa_selection`,
`retain_qsa_attention`, batching and the objective stay as configured.

The loaded config, the attempt snapshot, the worker command, the W&B config
(`kernel_stack`) and the checkpoint manifest carry the resolved stack. A resume
must match it: manifests written before the switch resume only as `reference`,
and a run cannot change modes mid-way. The teacher-cache command builds the
teacher with the reference defaults regardless.

`fast` changes the student's numerics beyond rounding: MoE top-10 routing and
QSA top-512 selections flip under rounding-level differences and the flips
compound through the 48 layers. On the published e276 recipe the two stacks'
student distributions differ by KL 0.018 nats/token on the held-out probe, as
much as SDPA and FlexQSA differ within `reference`; the probe's KL to the
teacher is 0.04276 (`fast`) against 0.04224, and the first update's loss differs
by 1.3% per token. Adam's first, nearly sign-like updates turn that into a
router update with cosine 0.51 between the stacks (0.55 between SDPA and
FlexQSA), so a `fast` run is a new run, not a continuation (numbers:
training/FLASH_RL.md, "Kernel mode"). Warm `fast` updates take a third to a half
as long (49-70 s against 144 s on 4 B200, GA8; the since-retired NVFP4 expert
storage made them 68-95 s), but its kernels compile and autotune per new sequence length:
cold first updates took 3-6x as long, so `slurm/sft_distillation.sbatch` keeps a
`fast` run's Triton and
Inductor caches across segments in `<run dir>/kernel_cache.tar.zst` (unpacked at
one per-run path under `/cache/users/$UID/arc-distillation/`; the job's unit
tests keep their own cache). A `reference` run keeps the per-job cache.

## Runtime and entry points

Run from the ARC repository root. Use the qualified Python 3.12/CUDA 13 runtime;
the repository `uv.lock` and `flash-distillation` extra pin Torch 2.13.0,
Transformers 5.16.1, FLA 0.5.1, and W&B 0.30.0. The mutually exclusive harness
and SFT-training resolutions retain Torch 2.11.0 and Transformers 5.14.1. The
loader validates the required
Qwen4-Exp capabilities and the test suite checks adapter topology and numerical
contracts. The Slurm launcher installs this same root lock into job-local scratch.

```bash
python -m sft.distillation.train --config /absolute/path/base-config.json --validate-only
python -m sft.distillation.launch --config /absolute/path/base-config.json --run-dir /absolute/path/new-run
# Only when training is requested; submits one bounded segment:
python -m sft.distillation.launch --config /absolute/path/base-config.json --run-dir /absolute/path/new-run --submit
```

The launch command defaults to a dry run. Submission requires a clean committed
worktree, pins the commit and base configuration, and serializes submissions with
an intent ledger and file lock. Each job uses `git archive` of that commit in
local scratch. Keep the worktree at the pinned commit when submitting subsequent
segments; already-queued jobs depend only on the archived revision. Scheduler
defaults are configurable through `scheduler.account`, `scheduler.partition`,
and `scheduler.qos`. The default allocation is 4 B200 / 48 CPU / 512 GiB / 12 h,
`dev` / `b200-pool`.

For an existing four-GPU allocation, the direct supervisor is:

```bash
python -m sft.distillation.train --config CONFIG --result NEW_RESULT_DIR \
  --stage-dir NEW_LOCAL_SCRATCH_MODEL_DIR --gpu-telemetry
```

The supervisor validates both completed caches, dataset hashes, lineage and
held-out disjointness before launching workers. `--validate-only` does not start
CUDA, W&B, teacher computation, or write result files. Full target payload hashes
are checked against stable example IDs on use; snapshot preparation independently
validated every payload. No silent teacher-forward fallback exists.

## Data and objective

Unpacked JSONL examples contain `input_token_ids`, `output_token_ids`, and
`game_id`. A stable original row index addresses its teacher cache file even when
training order is shuffled. The frozen prefix remains ordered on disk; a seeded
permutation mixes examples during training. Rows must be divisible by the configured
world size unless masked-tail batching is enabled. Each update uses one full example
per active rank and microbatch. The default objective averages sequence means
equally; `loss_weighting="token"` normalizes by global completion tokens.

Cache format: `manifest.json`, `head.safetensors`, and one indexed safetensors
file per example with BF16 `hidden` for positions `[prompt_len-1:-1]`. The teacher
output head reconstructs full-vocabulary targets in token chunks; the teacher
backbone is not run. The student executes a full forward and backward. Only FP32
router masters receive gradients/AdamW updates; the backbone and heads stay frozen.
Defaults in the prepared run: KL(teacher||student), T=1, CE=0, LR=1e-5, clip=1,
weight decay=0. The implementation supports two- and four-rank FSDP layouts;
qualification evidence remains specific to the configuration that
was measured. The cached-teacher recipe uses the e307 teacher and its pruned
descendants, with matching tokenizer and top-10 routing.

The frozen backbone uses FSDP2; routers are replicated with manual gradient
all-reduce, and checkpoint inputs are offloaded to CPU. Model shards are copied
to local scratch with checksum/provenance validation. Keep these memory fixes.

Required configuration keys are `teacher`, `student`, `data`, `data_sha256`,
`cache`, `steps`, `lr`, `temperature`, `ce_weight`, `chunk`, `save_every`, `seed`,
`probe`, and `wandb`. Probe includes fixed data,
manifest and hashes/cache, cadence/window, diagnostic sample/pruning thresholds.
The previous prepared `base-config.json` schema is accepted unchanged. For a
segmented run, `production_total_steps` is the total budget; `steps` is the
cumulative endpoint of the current attempt. `resume` is handled by the ledger.
W&B requires an explicit entity in online mode; offline mode is supported.

## Resume, outputs, and validation

One segment is 100 updates by default. Reinvoke the guarded launch command after
successful completion to continue. It requires scheduler success, finite rank
metrics, frozen-weight and replica equality gates, and the expected final step.
An interrupted/failed attempt requires diagnosis; it is never blindly resubmitted.
No automatic scheduler or monitor is installed by this module.

Result directories contain source hashes/revision, run configuration, W&B link,
GPU telemetry and `student/` with metrics, held-out probes, router safetensors,
optimizer/RNG/probe checkpoints, and acceptance reports. Resume checks the
saved objective/data/cache/student identity and restores the previous optimizer
step axis. Final router weights are an overlay on the immutable student base,
not a standalone merged checkpoint. Acceptance proves finite updates and state
integrity, not exact numerical replay or gameplay improvement.

```bash
python -m unittest discover -s tests/distillation
```

Tests cover chunked loss/gradient parity, completion alignment, routing counts,
resume state/corruption, cache transactions, staging, probes,
W&B aggregation, cached-only dispatch, and launch guards. The compute code was
ported from the qualified e276 package (GPU smoke 2654841); packaging validation
is CPU-only and does not claim a new GPU qualification.

`teacher_cache.py` is an explicit separate cache-generation command when new
examples are needed (`python -m sft.distillation.teacher_cache --help`). It is
never invoked by the training supervisor. Frozen datasets and cached targets
remain external run artifacts; run-specific snapshot scripts and historical
packages are retained as provenance, not the maintained training implementation.

## Selective activation offloading

`offload_every` controls the checkpoint-input placement: 1 offloads every layer
(the original behavior), 2 offloads layers 1, 3, ..., 47, 4 offloads every fourth
layer, and 0 keeps all checkpoint inputs on GPU. Every layer remains checkpointed.
The policy is part of the worker checkpoint manifest and incompatible resumes
are rejected. The per-layer GPU/CPU fractions are not guaranteed byte savings.

Continuous `host-memory.jsonl` records cgroup current/peak/limit, anon/file memory
and OOM events every 5 seconds; W&B receives host current/peak/limit in GiB.
Worker `pinned-memory-rankN.jsonl` records active and allocator-owned pinned bytes
beforeforward,afterbackward,aftercleanup, also logged to W&B. Allocator-owned
bytes include cached buffers and power-of-two allocation rounding. Telemetry
never clears caches or changes allocation behavior.

### Global gradient signal-to-noise diagnostic

Every update measures all trainable routers as one concatenated gradient vector,
prior to clipping and AdamW. No per-layer gradient statistics are logged.
`train/gradient_noise/global/{raw,ema}/signal_to_noise` is a **power ratio**
(signal power / noise power), not an amplitude ratio or an optimizer update ratio.
The EMA smooths powers with alpha 0.05 before taking their ratio.

For rank sequence-mean gradients g_i and completion-token weights w_i, use
A = sum(w_i ||g_i||²), M = ||sum(w_i g_i)||², q = sum(w_i²).
Estimated batch-mean noise is N = q(A-M)/(1-q); estimated signal is S = M-N.
The trainer's local gradient scaling is undone for this diagnostic only.
This assumes independent sequence gradients with common mean/covariance conditional
on length; correlated traces or length-dependent gradients can invalidate it.
Four sequences give noisy individual estimates. This is not a token-iid estimate.
Signal/noise powers, resolution flags, effective sequence count and observation
count accompany the ratio. Unresolved signal or zero/unresolved noise yields a
null ratio in JSON and no ratio sample in W&B, never an artificial infinity.

The diagnostic adds two global norm passes and one two-scalar all-reduce per step;
it adds no backward passes and does not alter training gradients. EMA state is
checkpointed under `global_gradient_noise`. Older checkpoints start a fresh EMA;
training manifest, optimizer, RNG and frozen-weight checks remain unchanged.
Existing running processes retain their loaded code; this applies on next launch.


### Four expert-usage curves (logging schema 2)

Only these expert-usage scalar curves are sent to W&B; the old all-context and
per-layer usage curves, teacher/student usage tables, per-layer probe routing
curves and expert-persistence tables are removed. Local detailed routing JSON
and probe/checkpoint diagnostics remain available. Quality and ablation-loss
metrics are preserved, as are training, gradient SNR and resource telemetry.

- `experts/effective_expert_fraction`: mean over layers of exp(selection entropy)
  divided by expert count.
- `experts/bottom_candidate_routing_mass`: mean over layers of the routing mass
  fraction assigned to the least-selected candidates (31 in the current config).
- `experts/worst_layer_effective_expert_fraction`: minimum effective fraction
  across layers.
- `experts/pruning_candidate_stability`: mean per-layer fraction of candidates
  retained since the previous fixed probe; absent on the first probe.

All values are fractions in [0,1]. First three pool completion-prediction counts
and routing masses across each non-overlapping `expert_log_every` window (default
10 updates); the final partial window is also emitted at successful completion.
Candidate count follows the probe's bottom_k/bottom_fraction configuration, and
frequency ties break by expert ID. Stability uses the same frequency criterion
on fixed probe examples (current cadence 50), with its own probe/step axis. Probe
checkpoint history preserves comparisons across resume. Training windows restart
at each new attempt; no mixing with the previous attempt's partial window.
No-observation windows are omitted, not reported as collapsed routing.
Historical W&B runs and already-running code are unchanged.


### Actual global router update ratio

`train/router_update_ratio` is emitted every update: the L2 norm of the actual
post-AdamW minus pre-AdamW router weights, divided by their pre-update L2 norm.
All trainable router tensors form one vector (not an average of layer ratios).
Clipping and optimizer moments are reflected in the observed parameter change.
Only rank 0 snapshots its replicated router weights; the temporary copy is freed
immediately after measurement, with no extra collective or backward pass.
The rank 0 metrics JSON stores the ratio; the supervisor logs it once globally.
Zero starting weight norm is undefined (null JSON, no W&B point). Older metric
files without the field remain readable. No diagnostic checkpoint state is needed.
This applies to next launches; it does not modify an already-running process.


### Useful student compute

W&B schema 3 adds:
- `train/useful_model_flops_estimate`: sum across ranks each update.
- `train/useful_model_flops_per_second_estimate`: summed useful work / slowest
  rank accounting duration.

FLOP convention `flash-router-useful-matmul-conv-v1`: analytical estimate of
student matrix multiplication and convolution arithmetic, FMA=2. Full sequence
body, completion-only student head; frozen body forward+input gradients as
needed, trainable router weight gradients included. The first layer has no
input gradients before its router. Counts routed top-k plus shared experts,
router/projection GEMMs, hyperconnections, PLE projections, depthwise convolution,
DeltaNet recurrent-form core (6KV forward / 12KV backward per value head/token),
and QSA selected causal edges (4Hd forward / 8Hd backward per edge). QSA index
selection is forward-only and counts valid compressed-key comparisons. This is
semantic useful work, not the dense masked SDPA or chunked FLA kernel's executed
FLOPs. Pointwise/norm/softmax/top-k, optimizer, teacher head/body and all
rematerialization are excluded. Component estimates remain in local rankJSON;
no per-layer W&B series. This estimator is not an exact hardware counter or a
validated MFU measurement.

### Gradient accumulation and checkpoint transitions

`gradient_accumulation` defaults to 1; GA2 processes two sequential one-sequence
microbatches per rank, then clips and updates once (global 8 examples). Each
microbatch loss is scaled by world_size / the total completion-token count over
all ranks and microbatches. Gradients are summed locally, then averaged across
ranks once. Loss, usage counts/mass, token throughput, useful FLOPs and sampled
transfer bytes cover the whole optimizer update. Per-sequence noise moments are
collected before local gradient accumulation, so the SNR estimate sees all 8
sequences. GPU graphs are released between microbatches; only router gradients
and small routing counters are retained.

Checkpoints save `data_microbatch_cursor` separately from optimizer `step`.
Legacy GA1 checkpoints infer cursor=step; GA>1 checkpoints must have a cursor.
`data_microbatches` caps the data budget and allows one final partial accumulation
without repeating/dropping rows. The existing deterministic global microbatch
order is unchanged. Allowing a GA change requires explicit
`allow_accumulation_change=true`; every other training-manifest field must match.
Optimizer/RNG/router/frozen/probe restoration checks remain. Gradient-noise EMA
starts fresh on a GA change because the batch-noise scale changed.

For a new run/ledger, launch.py accepts `--resume-checkpoint PATH`; it records the
checkpoint SHA256 in the immutable config, validates finite routers/state, and
verifies the total update target against the remaining data budget. Worker checks
the hash again. Existing ledger resumes retain their original successful-segment
guards. This is explicit checkpoint recovery, not automatic failed-job recovery.

GA1 step 50 -> GA2 with `data_microbatches=1481` consumes the remaining 1,431
microbatches in 716 updates, ending at optimizer step 766. The final update has
one microbatch/rank; total examples stay 5,924. LR and fixed-probe/checkpoint
optimizer-step cadences are unchanged.

## Optional exact QSA FlexAttention backend

Set `attention_backend: "flex_qsa"` in a fresh-run JSON config; default `sdpa`
retains baseline behavior. Only the 12 QSA attention modules change. Supported:
complete causal BF16 sequences, no cache/dropout, TF32 off, indexer ratio 4.
The public model accepts conventional right-padded `[batch, sequence]` inputs
and converts them to a tile-aligned, document-isolated token stream. Selections
use the same batched policy as the SDPA backend. A compact bitset,
128×128 sparse tile union and exact per-query callback replace dense token masks.
DeltaNet, teacher targets, routers, optimizer and objective are unchanged.
Backend is recorded in checkpoints; changing it on resume fails.

Historical qualification on one B200 covered native/real-dimension outputs,
gradients, checkpointing and timing at lengths 13,720 and 51,016.
CUDA compilation warm-up was excluded from reported warm samples.
Synthetic benchmark sparsity is not evidence of production QSA tile skipping.

W&B `attention/rankN/*` records selection/mask build/completed forward CUDA times,
call counts including recomputation, total tile skipping and extra skipping beyond
the causal tile triangle. Checkpoint early-stop can interrupt a forward span.
`train/reference_bf16_mfu_percent` divides useful work/s by 9 PF/s (four B200 dense
BF16 reference), not executed hardware FLOPs. Full-step timing on identical
batches is the primary performance measure; preserve loss/gradient checks.

### Reuse QSA selections during checkpoint backward

For FlexQSA, set `"reuse_qsa_selection": true` in the run configuration (worker
flag `--reuse-qsa-selection`). Default is false. Each checkpoint invocation
retains its forward's int32 block IDs and reuses them during backward replay.
All differentiable attention calculations are still recomputed. The selector,
including its FP32 ranking, is unchanged by selection reuse. The current
selector policy omits near-tie fallback (see below).

Selections are scoped to the computation graph, never shared across examples,
microbatches, or optimizer updates. Ordinary backward releases the cache with
that graph; retained graphs also retain their selections. Raw cache storage is
`12 * sequence_length * 512 * 4` bytes per rank (1.168 GiB at 51,016 tokens).
CPU checkpoint-input offloading remains independent; indices stay on GPU.

The policy is recorded in checkpoint manifests. Legacy checkpoints mean false;
changing this flag on resume is rejected. Per-update attention metrics expose
`selection_computations` and `selection_reuses`: with one microbatch the expected
counts are 12/12 enabled and 24/0 disabled. `selection_original_bytes` counts the
raw payload produced by selector calls, not peak or retained allocator memory.

Historical selection-reuse qualification covered exact native QSA outputs,
gradients, three AdamW updates, two microbatches, FSDP and offload 0/3 on four GPUs.
Full-model measurements used baseline/reuse warmups followed by ABBA updates on
fixed and longest examples. All updates began with the same weights and a fresh
optimizer.

### Batched QSA selection policy

The student selector now uses `batched_fp32_top513_v1`: FP32 batched scores and
top-513 truncated to 512, without the near-tie guard or per-query reference
recomputation. This intentionally changes selections/gradients versus the former
guarded policy; it is not a claim of numerical equivalence or training quality.
The original-method dispatch for unsupported layouts remains available.

`reuse_qsa_selection=true` with `attention_backend="flex_qsa"` continues to reuse
each forward's selections during checkpoint backward. Policy is recorded in the
training/checkpoint manifest. Legacy manifests mean `guarded_fp32_top513_v1`;
resuming those checkpoints under the new policy is rejected rather than silently
changing semantics. Existing immutable job snapshots are unaffected.

### QSA attention-state retention

New FlexQSA runs with selection reuse retain attention state on GPU by default.
Set `retain_qsa_attention=false` (CLI `--no-retain-qsa-attention`) to opt out.
SDPA and FlexQSA without selection reuse remain disabled. Legacy checkpoints
missing this setting mean false; resume policy changes are rejected.

## Equal-example training objective

Set `"loss_weighting": "example"` in a new run's JSON config, or pass
`--loss-weighting example` to the worker. This averages completion KL (and any
configured CE term) within each example, then averages equally across all examples
in the optimizer update, including all ranks and actual accumulation microbatches.
Partial final updates use their actual example count. Prompts contribute no loss.

New runs default to `example` in JSON configuration, supervisor and worker CLI.
Set `loss_weighting="token"` explicitly (or `--loss-weighting token`) to retain
token weighting. Checkpoints record the policy; legacy checkpoints mean `token`,
and objective changes are rejected on resume even if accumulation changes are
allowed. Resuming a legacy run therefore requires explicit token weighting.
Existing submitted jobs keep their immutable source/config snapshots.

`train/objective_loss` reports the optimized loss. `train/loss_per_example` and
`train/loss_per_completion_token` expose both reductions; the latter keeps its
original token-weighted meaning. Gradient-noise moments use the chosen sequence
weights; example weighting yields effective_sequences equal to actual batch size.
This does not guarantee resolved SNR or independent samples. Validation/probe
metrics remain token-weighted for comparability and are not changed by this option.

### Optional pointwise optimizations

New student runs enable the qualified defaults. Configuration exposes `expert_optimization` (`none`,
`no_sentinel`, `weighted_permutation`, and the fast stack's `fused_permutation`) and `fuse_norm_tail` (boolean). Defaults are `weighted_permutation`/`true`. The expert options require ordinary top-k indices,
bias-free non-transposed grouped experts, and no expert parallelism. They
remove unused EP sentinel masks; `weighted_permutation` also fuses
weighting/permutation with eager gradient reductions. The norm option fuses
only the final multiplications/cast and retains eager statistics and backward
reductions. Norm weights must stay frozen.

Use the JSON configuration keys or worker `--expert-optimization` and
`--fuse-norm-tail` / `--no-fuse-norm-tail`. Opt out with
`expert_optimization="none", fuse_norm_tail=false`. Settings are recorded in checkpoint manifests; historical
checkpoints default to `none`/`false`, and mismatched resumes are rejected.
The options do not change the experimental status of `flex_sparse_backward`.
Qualification: step-opt commit 9cead6ce, job 2700377, exact same-backend loss,
gradients and AdamW updates on four GPUs; fixed/longest GA1 cases.
The combined GA8/equal-example configuration has not been GPU-benchmarked. No running training job is modified.

### Expert storage

Routed experts are BF16 and FSDP-sharded with the rest of the frozen backbone:
the builder expands the student's NVFP4 experts once at load. The packed NVFP4
storage (`expert_storage="nvfp4"`, with `nvfp4_distribution`) and the per-GPU
replica that required it (`model_distribution="replicated"`) were removed on
2026-10-09: BF16 computes bitwise the same and is faster in both learners
([training/FLASH_RL.md](../../training/FLASH_RL.md#expert-storage)).

Configs, attempt snapshots and checkpoint manifests written before still carry
these keys. `expert_storage` (`bf16`/`nvfp4`), `nvfp4_distribution`
(`replicated`/`fsdp`) and `model_distribution="fsdp"` load with one warning per
setting; any other value, `model_distribution="replicated"` included, is
refused. A checkpoint saved with NVFP4 storage resumes with BF16 experts: the
resume gate drops those manifest keys and checks the frozen experts by content.
Its frozen fingerprint hashed the packed experts (`gate/up/down_{packed,scale,outer}`
per layer), which the build no longer holds, and the manifest pins only the
student's config, index and lineage files, not its shard contents. So the gate
hashes the student checkpoint's packed experts again as that storage did on each
rank (whole stacks when `nvfp4_distribution="replicated"`, the rank's FSDP2
shard of the experts under `"fsdp"`) and refuses any that differ from the saved
hash, naming the tensor; the BF16 experts must have the saved layout and every
other frozen tensor must match by hash. The ranks split that work, each hashing
a quarter of the packed experts: 8.5 GiB of the e276 student's 34 GiB, 7 s once
the build has read the shards (SHA-256-bound; CPU job 2924834, which also
reproduced all 432 saved hashes of an NVFP4-era checkpoint). The resumed run's
checkpoints record the BF16 fingerprint, which later resumes compare exactly.

## Expert-LoRA distillation

To train routed experts as well as routers, add to a fresh configuration:

```json
{"expert_lora": {"rank": 8, "alpha": 16, "dropout": 0}, "train_routers": true}
```

Set `train_routers: false` for expert-only adaptation. All three routed-expert
projections (gate/up/down) receive independent low-rank updates; base weights,
shared experts, attention and the output head stay frozen. `expert_optimization` must be
`weighted_permutation` or `no_sentinel`. Nonzero dropout is unsupported.

The worker averages every trainable gradient across ranks, clips the combined
gradient and updates routers/adapters together. Adapter configuration and router
trainability are part of the resume contract; changing either is rejected. Legacy
router-only checkpoints default to no adapters and `train_routers: true`. The
checkpoint fields `routers`/`router_names` retain their historical names but include
all configured trainables. This does not authorize resuming old experimental
branches across different attention or model policies.

LoRA runs write `trainables-fp32.safetensors` rather than a misleading router-only
file. For serving, export the complete training checkpoint:

```bash
python -m sft.distillation.export --student ORIGINAL_NVFP4_STUDENT \
  --checkpoint checkpoint-stepN.pt --out NEW_MODEL_DIRECTORY --device cuda
```

The exporter checks source config/index/pruning identity and the complete
trainable census, dequantizes routed experts to BF16, merges `alpha/r * B@A`,
removes expert quantization scales and ModelOpt NVFP4 configuration, and copies
the remaining model/tokenizer files. It refuses existing destinations and only
publishes a completed export. Allow disk space for the expanded BF16 model.
Router-only exporters must not be used for expert-LoRA checkpoints.
