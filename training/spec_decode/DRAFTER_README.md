# Drafter SFT: training Flash-Next's MTP head on our own traffic

This is the historical experiment record. Its branch names and launch commands
refer to the original runs. The maintained trainer and pruning recipe now live
in arc-training; use [the current recipe](README.md) to submit new jobs.

This package trains the speculative-decoding drafter: the one-layer MTP head
that ships with the Flash-Next checkpoint. It is trained against the served
target on ARC2 and ARC3 traces. One head serves both harnesses. The code
stays on the `drafter-sft` branch. The deliverable is the trained head
(`heads/v1-final`), which vLLM loads as a drop-in drafter directory.

**Result (September 2026).** In real play with k=3, temperature 0.6 alone,
on 11 games it never trained on, the trained head `v1` accepts **2.00
tokens per draft against the shipped head's 1.81 (+11%)**. The offline
metric in this package reproduces the engine's figures to within 0.005
per draft position. Its NVFP4 version, `heads/v1-nvfp4`, quantized as the
target's experts are, accepts the same and frees 3.4 GiB of GPU memory.

Contents:
1. [Where things are](#1-where-things-are)
2. [The pipeline, step by step](#2-the-pipeline-step-by-step)
3. [Evaluation methods](#3-evaluation-methods)
4. [All runs and results](#4-all-runs-and-results)
5. [What we learned](#5-what-we-learned)
6. [Next steps](#6-next-steps)
7. [Recipe: training a drafter for a new model](#7-recipe-training-a-drafter-for-a-new-model)

## 1. Where things are

| what | where |
|---|---|
| code | arc3-platypus `drafter-sft`: `packages/drafter`, and the `record`, `play` and `replay --drafter [--drafter-moe-backend marlin]` jobs in `tools/traffic_launch.py` |
| ARC2 recording launcher | arc2-platypus `drafter-sft`: `packages/duck/tools/traffic_launch.py record --dataset --node` |
| vLLM | no change needed. The engine is the `spec-decode` pin (fork `spec-decode-upstream` 64af83c6) |
| results tree (`$D`) | `/data/shared/research/arc_3_results/shared/drafter/` |
| recordings | `arc_3_results/shared/traffic/record_*_drafter_{train,test}`, `arc_2_results/shared/traffic/record_*_drafter_{train,test}` |
| target / shipped drafter | `/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform{,-drafter}` |
| trained head | `$D/heads/v1-final`, 7.3 GB, a full drafter directory with a `drafter_provenance.json` |
| trained head, NVFP4 | `$D/heads/v1-nvfp4`, 4.1 GB, the same with its routed experts in NVFP4 (`nvfp4_provenance.json`); serve it on Marlin |

Layout of `$D`:

- `datasets/<name>/chains.jsonl`: chains, plus `split.json` for split sets.
- `caches/<name>/`: one `<chain>.safetensors` per chain (the target's
  stream and the ids), plus `manifest-II-of-NN.json` per extraction shard.
- `runs/<name>/`: a training run. It holds `args.json`, `metrics.jsonl`,
  `tb/` (TensorBoard, trainer and watcher), `checkpoints/update-NNNNNN/head.safetensors`
  with `checkpoints/index.jsonl`, `state/latest.pt` (for resuming),
  `test.jsonl` (the watcher's scores) and `DONE`.
- `heads/<name>/`: exported drafter directories.
- `evals/*.json`: `drafter.evaluate` outputs.
- `play/`: real-play runs with a drafter.
- `jobs/<stamp>_<name>/`: every job's snapshot, sbatch script, `stdout.log`
  and `stderr.log`. `run_in_worker.py` holds the exact command line.

Every GPU step runs through `python -m drafter.jobs NAME [--gpus N] [--minutes M] [--node dgxNN] -- MODULE ARGS`.
It uses TAAF's workspace snapshot and node-local venv, the same sbatch a
benchmark gets. It is fire and forget: it prints the job id and directory.

## 2. The pipeline, step by step

The modules, one per step: `chains` → `capture`/`extract` → `train` (with
`head`, `attention`, `loss`) → `watch`/`evaluate` → `export` → `online`,
replay or play. `parity` checks the port against vLLM.

### 2.1 Record traffic

Record ids-only traces of the duck playing, on the B200 column as deployed,
with no drafter. Speculative decoding leaves the output distribution
unchanged, so a drafter in the recording would not change the data.

```
# ARC3 (arc3-platypus drafter-sft)
uv run python tools/traffic_launch.py record --games official_110 --lanes 120 --passes 2 --hours 2 --label drafter_train
uv run python tools/traffic_launch.py record --games arc_like_hard --lanes 120 --passes 11 --hours 2 --label drafter_test
# ARC2 (arc2-platypus drafter-sft)
uv run python packages/duck/tools/traffic_launch.py record --dataset arc2-eval --lanes 100 --passes 3 --hours 1.5 --label drafter_train
uv run python packages/duck/tools/traffic_launch.py record --dataset arc2-train-hard --lanes 100 --passes 3 --hours 1.5 --label drafter_test
```

- **Sets.** ARC3 trains on the 25 official games and is tested on sweep
  13's 11 hard ARC-like games. ARC2 trains on arc2-eval and is tested on
  arc2-train-hard.
- **Sampling.** `record` records with the play profile as deployed.
  Deployment is moving to temperature 0.6 alone; record that way once it
  ships (see [section 6](#6-next-steps)).
- **Volume.** One B200 records 7–9M generated tokens an hour. v1 used 19.5M
  (ARC3) and 15.2M (ARC2), and that was more than needed.

### 2.2 Chains and datasets (`drafter.chains`)

A **chain** is a run of calls in one trace where each prompt extends the
previous prompt plus its reply. The chain is stored as one sequence: its
last prompt plus its reply. One forward pass over it therefore covers every
call, and each reply sits at a known span. Compaction or a fresh render
starts a new chain. Only reply tokens train. Everything else (prompts, tool
results, images) is context the head attends to.

```
python -m drafter.chains build $D/datasets/arc3-train-v1 REC --val-games 2 --seed 0   # train/ + val/, whole games held out
python -m drafter.chains build $D/datasets/arc3-test-s16 REC --sample 16 --seed 0     # 16 chains, stratified by game
```

- `--val-games N` holds whole games out into `val/`.
- `--sample N` draws round-robin by game, so every game is represented.
- **Test sets are never used in full** (2026-09-24): offline tests
  use 16-chain samples per harness. The full test recordings are kept as
  sources for online workloads.
- Refused chains: a reply that sampled the image placeholder, an image-count
  mismatch, or a missing image file. `build` prints how many.

### 2.3 Extract the target's stream (`drafter.extract`, `drafter.capture`)

The head reads the target's **pre-final-mixer multi-stream** (`[4 × 2560]`
bf16 per position). The label at the next position is the target's final
mixer plus LM head applied to the same tensor. So one extraction gives both
input and label for every position, and no logprobs are stored.

```
python -m drafter.jobs arc3-train-extract-0 -- drafter.extract $D/datasets/arc3-train-v1/train $D/caches/arc3-train-v1 --shard 0/4
```

- **The engine** is the duck's B200 column (`DuckAgent.build`) with these
  targeted changes: one sequence at a time, prefix caching off (a cached
  position computes no hidden state), no host tier, no drafter, no
  profiler. Each chain is served as a prefill with one reply token.
- **Capture** is a vLLM general plugin (`drafter.capture`, entry point
  `drafter_capture`). It is inert unless `DRAFTER_CAPTURE_DIR` is set. It
  forces the model's `_mtp_hidden_buffer`, so the copy happens inside the
  compiled forward, then copies each tagged request's rows out after every
  step. `--drafter DIR` also captures a k=1 drafter's pass-0 output, for the
  parity check.
- **Cost and size:** the engine takes 5–10 min to come up, then runs at
  about 10k tokens/s per B200. Storage is 20 KiB per token: ARC3 train is
  779 GB and ARC2 train 360 GB. Shard with `--shard I/N`. It is resumable
  per chain.

### 2.4 Train (`drafter.train`)

```
python -m drafter.jobs v1-train --gpus 4 --minutes 240 -- drafter.train $D/runs/v1 \
  --train $D/caches/arc3-train-v1:$D/datasets/arc3-train-v1/train \
  --train $D/caches/arc2-train-v1:$D/datasets/arc2-train-v1/train \
  --val $D/caches/arc3-val-v1:$D/datasets/arc3-train-v1/val \
  --val $D/caches/arc2-val-v1:$D/datasets/arc2-train-v1/val \
  --gpus 4 --accum 1 --epochs 1 --steps 3 --lr 2e-5 --save-minutes 30 --val-every 50 --minutes 210
```

**The head** (`drafter.head`) is built from Transformers' `qwen4_exp`
blocks, with vLLM's `Qwen4ExpMultiTokenPredictor.forward` ported verbatim:

- the shared `fc_hidden` and `fc_embedding` with their norms;
- one full-attention QSA layer (24 query and 2 KV heads, head dim 256,
  indexer with a 2,048-token budget);
- a 512-expert bf16 MoE (top-10) plus a shared expert, and the HC mixers.

**Draft steps.** k steps train at once:

- Step 0 runs every slot t: the target's stream at t plus the token at t+1,
  drafting t+2.
- Step j feeds the head's own step j−1 multi-stream plus the data's token
  t+1+j, at position t+j.
- Each step runs its own indexer over pass 0's keys up to the anchor plus
  the draft keys, as the served drafter does
  (`index_share_for_mtp_iteration` is unset). The draft keys are merged in
  by log-sum-exp.
- A pair (t, j) trains only where tokens t+1..t+2+j all lie inside one
  call's reply.

**What trains:** everything except the embedding and LM head (shared with
the target) and the indexer (its top-k passes no gradient). The router
trains at ×0.1 LR. That is 2.61B trainable parameters.

**Objective:** full-vocabulary forward KL(target ‖ head) at `--loss-temperature`
(default 1). It uses our `ChunkedDistillation`, in 64-token chunks
recomputed in the backward. Steps are weighted equally. Keys and values are
fake-quantized to FP8 per token and head (straight-through, `--kv-fp8`,
default on), because the served drafter's cache follows the target's FP8
KV. `--draft-weighting` weights each row instead by how often serving
drafts it; it made no difference (section 4).

**Loop:**
- One chain per GPU per microbatch, data-parallel over `--gpus` ranks
  (`torch.multiprocessing.spawn`, gradients all-reduced by hand).
- Each draft step is recomputed in the backward (`--checkpoint-steps`);
  the longest chains (~67k tokens) peak at about 142 GiB.
- fp32 master weights over bf16 compute weights. AdamW (0.9, 0.95), no
  weight decay, clip 1. Warmup is `--warmup-frac` 3%, then cosine to 0.1×.
- ARC2 and ARC3 are mixed 1:1 by trained tokens each epoch.
- Rank 0 makes every save and stop decision and broadcasts it. Ranks that
  decide on their own clocks deadlock.

**Monitoring:**
- `metrics.jsonl` and TensorBoard (`tb/`): loss, per-step KL, top-1 and
  acceptance, gradient norm, LR, the update ratio, tokens/s and memory.
  Validation runs every `--val-every` updates, per harness.
- A checkpoint is written every `--save-minutes`. Resume with `--resume`,
  which is the default: a rerun in the same directory continues.
- v1's epoch was 274 updates, about 2.5 h on 4 B200s (~33 s per update).

### 2.5 Watch and evaluate (`drafter.watch`, `drafter.evaluate`)

```
python -m drafter.jobs v1-watch --minutes 290 -- drafter.watch $D/runs/v1 \
  --set play_v1=$D/caches/play-v1-s16:$D/datasets/play-v1-s16 \
  --set arc2=$D/caches/arc2-test-s16:$D/datasets/arc2-test-s16 --poll 60 --hours 4.7
python -m drafter.jobs NAME -- drafter.evaluate $D/evals/X.json --set NAME=CACHE:DATASET ... --head shipped=base --head v1=$D/runs/v1/checkpoints/update-000274/head.safetensors
```

- **The watcher** runs beside training on 1 GPU. It scores the shipped head
  as update 0, then each checkpoint as it lands, and writes `RUN/test.jsonl`
  and `RUN/tb/test`. It exits after `DONE`.
- **`evaluate`** scores several heads on several sets in one job.
- Both report the metrics in section 3: per-step means, reach, per-draft
  reach, the served-sampler estimate and KL.

### 2.6 Export and serve (`drafter.export`, `drafter.online`, launcher)

```
python -m drafter.export $D/runs/v1/checkpoints/update-000274/head.safetensors $D/heads/v1-final
uv run python tools/traffic_launch.py replay --workload FILE --lanes 25 --no-tier --no-eagle-block-drop \
  --arm temperature_only --arm mtp3 --drafter $D/heads/v1-final --no-profile --node dgx01
uv run python tools/traffic_launch.py play --games arc_like_hard --passes 3 --hours 0.5 \
  --arm temperature_only --arm mtp3 --no-tier --no-eagle-block-drop --drafter $D/heads/v1-final --node dgx01
```

- **`export`** copies the shipped drafter directory, swaps in the trained
  `mtp.*` tensors (all 31, same names, shapes and packing), and writes a
  provenance file. No config change is needed.
- **`replay`** is the traffic benchmark with the trained head.
- **`play`** is real play on the B200 column with the drafter on and ids
  traces recorded, so the same text can be scored offline afterwards.
- `drafter.online` (forced-length calls outside the duck) was used in
  milestone 1. The replay and `play` superseded it.

```
python -m drafter.jobs v1-nvfp4-export --minutes 45 -- drafter.nvfp4 $D/heads/v1-final $D/heads/v1-nvfp4 \
  --calibrate arc3_val=$D/caches/arc3-val-v1:$D/datasets/arc3-val-s16 \
  --calibrate arc2_val=$D/caches/arc2-val-v1:$D/datasets/arc2-val-s16
uv run python tools/traffic_launch.py replay ... --drafter $D/heads/v1-nvfp4 --drafter-moe-backend marlin
```

- **`nvfp4`** quantizes an exported head's routed experts as the target's
  are, and nothing else (the module docstring has the recipe, read off the
  target's weights). It calibrates the static input scales over the given
  sets, writes the experts under the target's per-expert names, and edits
  the config's ignore list so only they are quantized. About 5 minutes.
- **An NVFP4 head needs Marlin:** the `mtp` arms put the drafter on triton,
  which its bf16 experts need; `--drafter-moe-backend marlin` overrides
  that. A deployment cell leaves `speculative_config["moe_backend"]` unset,
  and the drafter inherits the target's Marlin.
- **`evaluate` scores a head directory directly**, NVFP4 experts
  dequantized as Marlin serves them, and `--nvfp4 NAME` emulates that on
  any head in memory.

## 3. Evaluation methods

Speculative decoding leaves the output distribution unchanged, so scores
are never at stake: only speed is. Every method below measures acceptance,
in increasing order of fidelity.

**Per-step acceptance, the offline workhorse.** At each position of the
recorded text, p is the target's distribution at T=0.6 and q is the head's.
The value Σ_v min(p, q) is exactly the chance the verifier accepts a draft
sampled from q. Steps 1 and 2 are scored after the recorded tokens. It is
reported per step as `accept_t0.6`. It ranks heads correctly, but its
product over steps understates k=3 for two reasons:

- **Clustering:** acceptance clusters, and a product of means ignores that.
- **Conditioning:** serving only reaches step j after accepted drafts,
  while offline scores every position.

**Reach, exact per position.** r_j = min(1, q(x)/p(x)) on the data's
token x. The running product r₀…r_j, averaged, is exactly the chance a
draft is accepted up to step j, **provided the data was sampled from the
verifier's p**. Temperature-only text satisfies that; deployed-sampled text
only approximately. Reported as `reach_t0.6`; `reach_served` is the same
under the deployed sampler, exact only on deployed-sampled text.

**Per-draft reach, what the engine reports.** The engine drafts from where
the last verify ended: after keeping j tokens, the next draft starts j+1
on. Hard stretches take many short drafts, so per draft they weigh more
than per position (4–5 points here). `drafts_t0.6/pos{j}` weights each
anchor by the chance a draft starts there. **On temperature-only real-play
text it matches the engine's acceptance-by-position to within 0.005, for
both heads.** It is the headline metric, and the `evaluate` summary prints
it as "per draft".

**Deployed-sampling estimate.** `accept_served` scores p after the V2
sampler's processing: presence penalty over the reply so far, then
temperature, top-k and top-p, as `loss.SERVED_SAMPLING` defines them. q is
the head at temperature alone, since vLLM samples drafts that way.
`ceiling_served` is the target scored as its own drafter under that
sampling: 0.93–0.95 per token. Deployment is moving to temperature alone,
so this matters only historically.

**Parity (`drafter.parity`).** Given identical inputs, the port's pass-0
output is compared against the served drafter's (captured with
`extract --drafter`). With FP8 K/V emulated, the median relative error is
0.07–0.08, and the top-1 agreement is 95–97%. That is the bf16 numerics
floor: the port moves as much when its own input is perturbed at bf16
resolution. Run this first on any new model.

**Label parity.** The label head against the engine's own prompt logprobs:
mean |Δ| 0.007–0.010, top-1 agreement 99%.

**Online checks:**
- **Replay** (`traffic_launch replay`): the throughput benchmark. It ranks
  heads correctly (v1's gain matched real play: +0.19 tokens per draft).
  It reads about 0.17 tokens per draft higher than real play. Three causes
  are confounded and not yet separated: it replays the official games,
  where real play used the test games; it forces recorded lengths past the
  end of sequence; and it generates fresh replies on recorded prompts.
  Treat its absolute acceptance as optimistic.
- **Real play** (`traffic_launch play`): the ground truth for acceptance.
  Read the engine's per-minute `[engine]` lines, which give acceptance and
  per-position rates: weight them by generated tokens over the run. With
  33 games and nothing waiting it is not a throughput reading. Extract its
  traces (`chains build --sample 16` on the play directory, then
  `extract`) and they become a temperature-only test set on which the
  per-draft reach is exact.

**Test sets now in use:**
- `play-v1-s16` and `play-shipped-s16`: temperature-only real play of the
  11 ARC3 test games. Either run's text is a valid sample for any head.
- `arc2-test-s16`: deployed-sampled; no temperature-only ARC2 play exists
  yet.
- Validation: the held-out games of the training recordings.

## 4. All runs and results

**Milestone 1: overtrain a tiny set** (`tiny4`: 2 ARC2 and 2 ARC3 chains,
69.6k tokens):

| run | what | result |
|---|---|---|
| `tiny4-overfit-1` (2764833) | k=3, LR 1e-4, 13 passes, 9 min | KL 0.34→0.06 at step 0, acceptance 0.86→0.95 on the training chains; worse on held-out chains (0.848→0.821). The chain works end to end, and too much LR on too little data overfits |
| `tiny4-dp-test`, `-dp-test2` | 2-GPU DDP, save, resume, watcher | first try deadlocked (per-rank clocks), fixed with rank-0 broadcast; resume reproduces losses update for update |
| `long3-memory` | the three longest chains | 131–142 GiB peak with step recompute, 26–28 s per update |
| parity jobs | port against vLLM drafter | 0.25 relative error until the drafter's FP8 KV was emulated, then 0.07–0.08 (numerics floor) |

**v1** (`runs/v1`, job 2766246, 4 B200s, 274 updates, 2.5 h):
T=1, LR 2e-5, 3% warmup, cosine to 0.1×, one epoch, 1:1 mix. Data:
`arc3-train-v1` (736 chains, 40.8M tokens, 18.1M trained) and
`arc2-train-v1` (513 chains, 18.8M, 14.3M trained), each with whole games
held out for validation.

- Validation per step: 0.809/0.753/0.712 → 0.851/0.795/0.763.
- About 90% of the gain came by update 50, a fifth of the epoch.
- Training KL equals validation KL (0.25 against 0.26): the head
  underfits rather than memorizes.

**v1 served**, all at temperature 0.6 alone:

| check | shipped head | v1 |
|---|---|---|
| replay k=1, 50 lanes (acceptance, gen tok/s) | 83.7%, 2,296 | 87.9%, 2,381 (+3.7%) |
| replay k=3, 25 lanes (per draft, gen tok/s) | 1.98, 1,771 | 2.17, 1,850 (+4.5%) |
| **real play k=3, 11 test games, engine (per draft)** | **1.81** (0.774/0.589/0.445) | **2.00** (0.823/0.656/0.521) |
| offline per-draft reach on that play text | 1.78–1.82 | 2.00–2.03 |
| offline per-draft reach, ARC2 test (deployed text) | 1.88 | 2.10 |

**From training games to unseen games.** The spec-decode benchmark replays
the official games, which v1 trained on, so the number that matters for it
is how much of the gain survives on games the head never saw:

| tokens a k=3 draft, T=0.6 | shipped | v1 | gain |
|---|---|---|---|
| training batches (in-sample chains, both harnesses; offline) | 1.86 | 2.08 | +0.22 |
| held-out games of the training recordings (offline) | 1.85 | 2.04 | +0.19 |
| test games, ARC3 / ARC2 (offline) | 1.71 / 1.95 | 1.89 / 2.16 | +0.18 / +0.21 |
| replay on the official (training) games (engine) | 1.98 | 2.17 | +0.19 |
| real play on the ARC3 test games (engine) | 1.81 | 2.00 | +0.19 |

- **Expect 0–0.03 fewer accepted tokens per draft** of the gain on unseen
  games than on the training games: at most about 15% of it, inside the
  samples' noise.
- **The absolute level belongs to the games**, and moves the same for both
  heads: the hard ARC-like games sit about 0.1 below the official games.
  Always compare heads on the same games.
- The offline rows are per-step means multiplied through: they rank heads
  correctly but read k=3 a little low.

**Objective and step-size round** (2026-09-26): three runs of 0.35 epoch
(96 updates, about 1 h each), all at `--loss-temperature 0.6`. Tokens per
draft, offline per-draft reach (play samples; ARC2 test):

| run | T | serving weighting | LR | play | ARC2 test |
|---|---|---|---|---|---|
| shipped | | | | 1.78–1.82 | 1.88 |
| v1 | 1 | off | 2e-5 (full epoch) | 2.00–2.03 | 2.10 |
| `kA-t06` | 0.6 | off | 3.3e-5 | 1.88–1.91 | 1.98 |
| `kA3-t06-dw` | 0.6 | on | 3.3e-5 | 1.87–1.91 | 1.97 |
| `kB3-t06-dw-lr1e4` | 0.6 | on | 1e-4 | 1.87–1.90 | 1.97 |

All three landed at the same plateau, with about 40% of v1's gain.

- **Weighting:** no effect.
- **LR:** no effect, 1.65× to 5× v1's step.
- **Leading explanation, the T=0.6 objective:** once the head agrees with
  the target's top token (86% of positions), the tempered distributions
  nearly coincide and the gradient nearly vanishes. Each update learns from
  fewer positions, and a higher LR cannot supply the missing information.
- **Not excluded, the compressed schedule:** v1 reached about 90% of its
  gain in 50 updates, but these runs had about half of theirs by update 58.
- **Settling control, not run:** T=1 at 2e-5 on the same 0.35-epoch
  schedule.

**v1 in NVFP4** (2026-09-26, `heads/v1-nvfp4`, 4.1 GB). The routed
experts quantized as the target's are (`drafter.nvfp4`):

- **The recipe.** ModelOpt NVFP4, round to nearest from each 16-value
  block's max. Gate and up share a global scale over 128-expert blocks,
  and down has one per expert. It was verified by requantizing the target:
  every expert tensor of layers 0, 5, 20 and 47 (3,684) comes back
  byte-identical under ModelOpt 0.46, up to the sign of zero. The port
  matches ModelOpt bit for bit.
- **Input scales** were calibrated on the validation samples. 12 of the
  512 experts were never reached and take the others' maximum. Marlin
  reads none of these scales.

| check | v1 bf16 | v1 NVFP4 |
|---|---|---|
| offline per-draft reach, play samples (v1's / shipped's text) | 2.029 / 1.996 | 2.025 / 1.993 |
| offline per-draft reach, ARC2 test | 2.102 | 2.100 |
| replay k=3, 25 lanes, one node back to back: per draft (by position) | 2.175 (85.1/71.5/60.8%) | 2.182 (85.1/71.8/61.2%) |
| generated tok/s, settled | 1,856.8 | 1,871.5 (+0.8%, inside a reading's noise) |
| weights loaded / KV pool | 54.06 GiB / 114.35 GiB (5.84M tokens) | 50.70 GiB / 117.72 GiB (6.01M tokens, +2.9%) |
| drafter MoE per decode step (profile) | 0.88 ms, triton | 0.49 ms, Marlin |

- **Acceptance cost:** none measurable. Offline it is −0.003 tokens per
  draft, and the engine reads the same.
- **The file is what was measured.** The exported directory scores exactly
  what the in-memory emulation (`evaluate --nvfp4`) scored.
- **Marlin serves it.** The profile shows 102 Marlin calls per decode step:
  the target's 96 plus the drafter's 6.
- **The B200 gain is small here.** The 25-lane replay is not pool-bound
  (about 15% of the KV pool in use), so the pool gain does not show in its
  throughput. The MoE saving is about 0.4 ms of a 20–26 ms step. On the
  RTX both weigh more: its pool is about 40 GiB, and the card reads
  weights at about a quarter of the B200's rate.

## 5. What we learned

- **Train at T=1.** KL at the serving temperature trained worse, even for
  acceptance at that temperature.
- **The data is not the bottleneck.** Validation equals training, gains
  came early, and the gain transferred to unseen games and to text from a
  different sampler (+0.19 tokens per draft on temperature-only real play,
  from deployed-sampled training data).
- **Adam makes step size track LR, not gradient scale.** Rescaling the LR
  to cancel a loss scale (such as T²) does not "match" runs.
- **Keep warmup at least about 8 updates.** A 3-update warmup to full LR
  shocked steps 1 and 2 badly; the draft steps compound parameter changes.
- **Emulate the drafter's FP8 KV in training.** The served drafter's cache
  follows the target's FP8 per-token-head KV. A bf16 drafter cache would
  win back 1–3 points per step offline, but it did not fit at the column's
  window (140 GiB asked of a 114 GiB pool).
- **Sampling caps acceptance.** vLLM verifies against the processed target
  but samples drafts at temperature alone. Under top-p, top-k and presence
  penalty even a perfect drafter tops out at 0.93–0.95 per token.
  Deployment is moving to temperature alone. State the off values
  explicitly (`top_p=1.0`, `top_k=0`, `presence_penalty=0.0`) on the duck's
  own profile rather than leave them unset.
- **The replay reads acceptance about 9% higher than real play**, on
  different games and for reasons not yet separated. It still ranks
  drafters correctly.
- **Counting method matters by 4–5 points:** per position against per
  draft. Compare like with like.

## 6. Next steps

In rough order of expected value:

1. **Settle the schedule question:** T=1 at 2e-5 on the 0.35-epoch
   schedule (one 4-GPU hour). Then try 2–3 epochs at T=1 to see whether the
   plateau moves.
2. **Train the indexer.** It is frozen today and decides which 2,048
   context tokens the head sees. DSA-style indexer distillation (match its
   scores to the head's attention) is the known route.
3. **Recover the FP8-KV cost** (1–3 points per step), or push the serving
   side to allow a bf16 drafter cache once the pool has room.
4. **Record at temperature alone** once deployment switches, for training
   and ARC2 test data. It makes the reach metric exact everywhere.
5. **Settle whether replay and real play agree on the same games:** real
   play on the official games with the final head, against the replay's
   per-draft figure. This decides whether the benchmark needs a
   natural-length mode.
6. **Prune the head**, the other half of spec-decode's phase 5 (NVFP4
   is done, section 4). Collect router histograms on validation, cut
   experts (`mtp_num_experts` already lets a pruned head load), re-distill
   with this loop, and re-export with `drafter.nvfp4`. Round-to-nearest
   quantizes each expert on its own, so pruning never needs a
   requantization of the others.
7. **k>3.** Nothing in training limits k (`--steps`). Re-score with the
   per-draft metric at k=4–5 before trying it in the engine.

## 7. Recipe: training a drafter for a new model

For a new checkpoint of the same architecture (Qwen4Exp with an MTP head),
expect a working day of wall-clock time and roughly 35 B200-hours: about 7
recording (the four recordings run in parallel, 2 h), 6 extraction (in
parallel, about 1 h), 10 training (4 GPUs, 2.5 h), and the rest evaluation
(watchers, play, replays). A new architecture adds porting work in steps 2–3.

1. **Paths and config.** In `drafter/head.py`, point `TARGET` and
   `DRAFTER` at the new target and its extracted drafter directory
   (`tools/checkpoints/extract_mtp_drafter.py --src TARGET`). Check that
   `head_config` still reads the head's layout (`mtp_num_experts`,
   `mtp.layer_types`, the MTP rope theta). Check the duck's deployment cell
   serves the new target (`ullm/backends/deployment.py`), because
   extraction runs the duck's own engine.
2. **Parity gates, on a tiny set, before anything big** (the milestone-1
   flow). Budget 20 minutes per run.
   - `chains pick` or `build --sample 4` a tiny set.
   - `extract` it, and again with `--drafter` for the parity capture.
   - `parity`: the relative error should sit at the bf16 floor (about 0.08;
     find the floor by perturbing the port's input stream at bf16 resolution
     and measuring how far the port itself moves). If
     it is far above, look first at the drafter's KV dtype, the RMSNorm
     variants, rope, and the HC mixer order.
   - Label parity: the label head against prompt logprobs, mean |Δ| of
     about 0.01.
   - Overtrain the tiny set (LR 1e-4, about 10 passes): KL should fall
     several-fold.

   For a new architecture, also port the forward into `head.py`, the
   attention into `attention.py`, and the capture hook's buffer and patch
   points into `capture.py`.
3. **Record traffic**, ids-only, B200 column, no drafter, at the deployed
   sampling (temperature alone, once shipped). About 15–20M generated
   tokens per harness is plenty: ARC3 official 25 × 2 passes for 2 h at 120
   lanes; ARC2 arc2-eval × 3 passes for 1.5 h at 100 lanes. Record the test
   games too, or reuse `play` runs of the test games as the test set.
4. **Datasets:** `chains build` with `--val-games` (2 for ARC3, 6 for ARC2),
   and `--sample 16` for test sets.
5. **Extract:** training sets in 3–4 shards, validation and test samples
   in one job each. Check every manifest covers its dataset; the trainer
   prints how many chains it left out.
6. **Measure the shipped head first.** Run `play` with the shipped
   drafter on the test games (30 min, temperature alone, `mtp3`, no tier,
   no-drop). Extract a 16-chain sample of it and confirm the offline
   per-draft reach matches the engine's figures within about 0.01. That is
   the calibration gate: do not trust offline gains until it passes.
7. **Train:** the v1 settings (section 2.4) are the default. T=1, LR 2e-5,
   one epoch, 4 GPUs, a watcher beside it on the test samples. Stop early
   if the watcher has flattened; v1 had most of its gain by a fifth of the
   epoch.
8. **Export and verify online:** `export`, then `play` with the trained
   head on the test games, and compare against step 6's shipped-head run.
   Optionally run the replay benchmark for throughput, keeping in mind it
   overstates acceptance.
9. **Quantize:** `drafter.nvfp4` on the exported head, calibrated on the
   validation samples (section 2.6). `evaluate` the directory against the
   bf16 head: it should lose next to nothing. Then replay it with
   `--drafter-moe-backend marlin`, and check the engine log for the smaller
   weights and larger pool.
10. **Ship:** copy the head directory next to the target in
   `/data/shared/research/checkpoints/`, named after the target. The head
   is only valid for the target it was trained on. Point the deployment's
   `speculative_config["model"]` at it, and upload it as a Kaggle dataset
   for the Kaggle cell.
