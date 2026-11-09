# Per-task Flash-Next prunes with their own drafters

The procedure that produces an ARC-3 and an ARC-2 Flash-Next model, each with
its own experts and its own speculative-decoding head, and compares them with
main's deployment (e307 with the v1 head) on main's own long runs.

ARC-2 and ARC-3 run in parallel, each on its own data. The recorded experiments
used the deployed B200 cells (arc3-platypus #104 /
arc2-platypus #35, MTP drafting at k=4, park mode); only the target and drafter
changed. For a new comparison, use the same qualified harness revision for
both arms.

## Ownership and environments

**arc-training owns the complete pruning and drafter-training recipe.**
`spec_decode/drafter/` is the canonical implementation, not a mirror of a
harness branch. The dedicated drafter distillation trainer remains separate
from the TRL trainer in `sft/train.py`.

| Piece | Implementation in arc-training | Runtime |
|---|---|---|
| Routing replay and pruning plans | `scripts/prune_*.py`, `training/flash_*.py` | Qualified Flash runtime |
| Physical checkpoint pruning | `scripts/prune_flashnext_experts.py` | Flash runtime, CPU job |
| Calibration and gameplay evaluation | `spec_decode/tools/long_run_arc{3,2}.py` | Corresponding Platypus harness |
| Chains, training, offline evaluation, export and quantization | `spec_decode/drafter/` | Dedicated `spec_decode/pyproject.toml` + `uv.lock` |
| Target activation extraction and online evaluation | Same `drafter` package | Qualified serving interpreter, in a private job environment |

The harness supplies game play and the serving engine. It does not own the
trainer or select its implementation. Keep its qualified interpreter available
on compute nodes and set `DRAFTER_SERVING_PYTHON` to its absolute path; only
extraction/online jobs use that interpreter by default. It must include
the matching Duck/ullm packages and the speculative-decoding vLLM fork. The
launcher adds this repository's package in a private environment without
modifying the serving environment. Keep that external runtime unchanged for
the lifetime of the job; its dependencies are not copied into the job snapshot.

From arc-training:

```bash
uv sync --project spec_decode --locked
uv run --project spec_decode --no-sync python -m drafter.train --help
uv run --project spec_decode --no-sync python -m drafter.jobs NAME --gpus 4 -- \
  drafter.train OUT --train CACHE:DATASET --gpus 4 --steps 4
```

Every `drafter.jobs` submission freezes this package and its lock in
`jobs/<stamp>_<name>/source/`, records the git revision, dirty status and file hashes in
`source_manifest.json`, and checks the imported package path inside the worker.
The worker records its actual interpreter and package versions in `runtime.json`.
Pending jobs therefore keep the submitted training code even if the checkout
changes. Training uses one chain per GPU and averages gradients across replicas;
it does not pool GPU memory. `spec_decode/tools/in_job.sbatch` runs the
recipe controllers in this repository's node-local drafter environment.
Jobs and controllers reuse a node-local dependency environment keyed by the
lock and project declarations, with no editable project installed. Each job
loads its own snapshotted package through `PYTHONPATH`; controllers load their
checkout. Private serving environments are removed on exit. The old
`in_job_arc2.sbatch` and `in_job_arc3.sbatch` names are compatibility symlinks.
The root RL environment and its older vLLM pin remain separate.

`DRAFTER_README.md` preserves the original experiment's results and commands;
use the commands here for current jobs. The package's CPU tests run in the
repository test suite under `spec_decode/tests/`, and CI installs the standalone
lock and runs those tests in its own environment.

`drafter.online` and `drafter.extract --drafter` select the drafter's MoE backend
from its `config.json`: Triton for BF16 experts (including heads whose MTP block
is excluded from the target's NVFP4 quantization), Marlin for NVFP4 experts.
Both accept `--moe-backend triton|marlin` to override that selection. The
selected backend is recorded in the evaluation or extraction manifest.

## Checking current harness main

[Recorded compatibility checks](validation/main_20261006.json) cover ARC-3
main `ad812b51` and ARC-2 main `81401ce8`, both serving vLLM fork `3bda1ba09`.
Each constructed and serialized its benchmark, extracted real image-bearing
chains, and supplied a cache to one four-GPU/four-draft-step training update
that saved a checkpoint. These are short compatibility checks, not evidence
of gameplay improvement or full-context training capacity.

Install ARC-2 and ARC-3 main in separate environments: both provide packages
named `duck` and `platypus`, and ARC-2 supplies its own TAAF. Do not install one
over the other. With `TRAINING` pointing to this checkout and
`DRAFTER_TARGET` / `DRAFTER_BASE` selecting the matching model pair:

```bash
PYTHONPATH="$TRAINING/spec_decode" "$ARC3_PYTHON" \
  "$TRAINING/spec_decode/tools/check_harness.py" --harness arc3 --out arc3-main.json
PYTHONPATH="$TRAINING/spec_decode" "$ARC2_PYTHON" \
  "$TRAINING/spec_decode/tools/check_harness.py" --harness arc2 --out arc2-main.json
```

This checks actual benchmark construction and serialization, the extraction
factory, the online-evaluation factory, and the selected target. It starts no
engine. Set `DRAFTER_SERVING_PYTHON` to the corresponding interpreter when
submitting extraction jobs. The runtime must be accessible on the allocated
node; a path under one node's `/cache` requires `drafter.jobs --node` for that
node. Offline parity over captured states runs in the standalone training
environment and needs no harness interpreter.

Online checks and optional drafter capture retain the harness's deployed
speculative-decoding settings. They override the selected drafter and draft
length; online checks also use `DRAFTER_TARGET`, rather than silently evaluating
against the harness's default checkpoint.

## Steps

For each task (ARC-3 | ARC-2):

1. **Calibration rollouts.** The unpruned 512-expert model plays the task's
   calibration set on the deployed cell with ids-only traces, drafting with
   e307's head (lossless for the outputs).
   ARC-3: the official 25 and `ARC_LIKE_HARD` (11 hard re-arc-3 games), 2 passes
   each. ARC-2: arc2-eval (120) and arc2-train-hard (47), 2 passes each.

       # TRAINING is this checkout; HARNESS is the qualified ARC-3 harness.
       (cd "$HARNESS" && uv run python "$TRAINING/spec_decode/tools/long_run_arc3.py" \
           --label unpruned_cal --games official_25 --passes 2 --target "$UNPRUNED" --drafter "$E307_HEAD")

2. **Expert statistics.** Replay the traces through the unpruned model and sum,
   per layer and expert, gate x output norm, gate and routed tokens (REAP needs
   the output norm, which serving does not expose, hence a replay):

       ARC3_RUN_DIR=... sbatch scripts/flash_dev.sbatch scripts/prune_route_stats.py \
           --checkpoint $UNPRUNED --recording $RUN_DIR[:PASSES] ... --out $STATS --shard K --shards N

   Every chain replays as it was served (`training.flash_route_stats.ServedReplay`,
   which the probe uses too): its images go through the frozen vision tower,
   which loads with the first chain that carries any, and a chain whose image
   placeholders have no images is refused. There is no text mode to choose by
   mistake: the ARC-3 statistics of 2026-09-30 were replayed as text, each
   placeholder as its own embedding, when images were still behind a flag
   (`docs/flash_pruning.md`). Each rollout's statistics record the checkpoint
   and the images replayed, and a plan carries them into `pruning.json`.

   With `--follow SECONDS` a shard replays each game as its trace is sealed
   (`_SUCCESS`) while the calibration run is still playing, rescanning every
   SECONDS until the run has ended, so the prune waits only for the last games.
   Games are assigned to shards by a hash of their run id. Shards resume per
   rollout, so they also suit the preemptible `nice` partition
   (`--partition=nice --qos=nice --requeue`). The replay runs at about 420
   tokens/s per B200, so a backlog is cleared by helpers beside the running
   shards: `--shard 0 --shards 1 --claim --reverse` takes rollouts through
   exclusive `<run>.claim` files, highest run id first, so helpers never repeat
   each other and meet the forward shards.

3. **Prune.** `scripts/prune_plans.py --stats $STATS --out $PLANS --keep 307`
   (REAP: the mean gate x output norm over the tokens routed to an expert), then
   `sbatch scripts/prune_physical.sbatch $PLANS/reap_k307.json $OUT`. As Slurm
   steps behind the replay shards (`--dependency=afterok:...`),
   `scripts/prune_plan.sbatch $STATS $PLANS N $E307 ...` refuses to plan on
   fewer than the N calibration rollouts and reports each plan's overlap with
   the reference prunes. (An `sbatch --wrap` step runs under `/bin/sh`, where
   `set -o pipefail` fails: write the step as a bash script.)

4. **Evaluation and drafter data.** The pruned model's long runs on main's
   protocol (ARC-3: official 25 x 10; ARC-2: arc2-eval x 10), drafting with
   e307's head, beside e307's own run on the same code: two identical runs, so
   the comparison has 20 passes and the drafter data arrives twice as fast.
   Their traces are the drafter's training data. v1 needed about 15M (ARC-2)
   and 20M (ARC-3) generated tokens, "more than needed", and one B200 records
   7-9M an hour, so the drafter starts from the traces sealed a couple of
   hours in while the runs finish.

5. **Drafter.** Extract the target's own MTP block as the drafter base, then
   run the chain from this arc-training checkout. Wait for the base extraction
   job to succeed before submitting the chain, then wait for its training job
   before submitting the finish step:

       export DRAFTER_TARGET=$TARGET DRAFTER_BASE=$BASE
       export DRAFTER_SERVING_PYTHON=/absolute/path/to/qualified/serving/venv/bin/python
       export DRAFTER_MAX_TOKENS=65536   # ARC-3's measured training-memory compromise
       sbatch spec_decode/tools/in_job.sbatch python -m drafter.extract_base --src "$TARGET" --out "$BASE"
       sbatch spec_decode/tools/in_job.sbatch bash spec_decode/tools/drafter_chain.sh NAME 2 "$LONG_RUN_A" "$LONG_RUN_B"
       sbatch spec_decode/tools/in_job.sbatch bash spec_decode/tools/drafter_finish.sh NAME "$SHARED/...-reap307-drafter"

   `drafter_chain.sh` builds one dataset per run (`drafter.chains build`, two
   whole games held out), queues its extracts (`drafter.extract`, the shards
   and the held-out games) and then `drafter.train` behind them
   (`drafter.jobs --after`: `--gpus 4 --steps 4 --lr 2e-5 --epochs 1`).
   `drafter_finish.sh` exports the last checkpoint over the base
   (`drafter.export`), quantizes its routed experts (`drafter.nvfp4 --steps 4`,
   calibrated on the held-out games) and reduces the draft vocabulary to the
   16k ids the training chains generate most (`python -m drafter.reduce_vocab`).

   The objective is full-vocabulary forward KL(target || head) at
   T=1 over the draft steps (k=4, as the B200 cells draft), on the target's own
   stream (the pre-mixer multi-stream, extracted by a teacher-forced prefill; the
   labels are recomputed from it, so no logits are stored). T=1 beat T=0.6 in
   the original ablation (2.00-2.03 against 1.87-1.91 tokens per draft).

6. **Final evaluation.** The pruned model with its own head against e307 with
   the v1 head, main's long runs: score, generated tokens per second, drafts
   accepted.

7. **Ship.** The model and its head beside e307, where the deployment cells
   read their checkpoints:
   `/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307{,-drafter}` and
   `/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc2-reap307{,-drafter}`
   (symlinks remain at their first location, `arc_{3,2}_results/shared/`).
   The physical-pruning writer (`scripts/prune_flashnext_experts.py`)
   writes shards with the same umask-derived permissions as other output
   files (use `umask 002` for group write access). It copies only serving
   configs and tokenizer/processor files, including ModelOpt's companion
   configuration; source conversion markers, metrics and reports are omitted.

   The plan a prune is made from names its source checkpoint, whether the
   calibration's images were replayed, and the calibration recordings
   (`scripts/prune_plans.py`), and the prune tool copies the plan into the
   model's `pruning.json`.

## What the head sees in training

Two kinds of context reach the head. Training matches serving completely in the
first, and in the second only as far as a chain, or a window of it, fits a card.

- **The target's.** Extraction (`drafter.extract`) puts the whole chain through
  the target as one teacher-forced prefill on the serving engine and caches the
  target's stream at every position (`hidden`, `[tokens, 10240]` bf16, 20 KiB a
  token). Each row is the target's own reading of everything before it, as when
  serving, however the head is trained afterwards.
- **The head's own.** The head attends over the keys it computed at the earlier
  positions (QSA: the 512 best four-token blocks and its open block). Served,
  that reaches back over the whole chain; in training, over what is on the card.

Neither includes the images' own embeddings. At an image placeholder the head
embeds the placeholder token, in training as when served; the images reach it
only through the target's stream. Feeding it the image embeddings is parked;
[Image embeddings for the head](#image-embeddings-for-the-head-parked) is the
passdown.

Only the head trains, one chain per GPU per microbatch, each draft step
recomputed in the backward. A B200 holds a chain of about 85k tokens that way
(an 84k-token chain peaked at 156 GiB).

| | ARC-2 | ARC-3 |
|---|---|---|
| training chains | 752 (21.5M tokens) | 138 (14.4M tokens) |
| median / longest chain | 27.8k / 69.7k tokens | 127.0k / 142.2k tokens |
| `DRAFTER_MAX_TOKENS` | unset: every chain whole | 65,536: 114 chains as windows, 332 pieces in all |
| peak memory per B200 | 140.7 GiB | 149.8 GiB |

A chain longer than `DRAFTER_MAX_TOKENS` runs as windows of at most that many
tokens (`drafter.train.windows`). The first trains every position it holds.
Each later one opens a quarter window (16,384 tokens) before the first position
it trains and holds that stretch as context only: 1024-aligned, so the indexer's
blocks fall where they do in the whole chain, and with the positions' own
places in the chain for the rotary. Every position trains exactly once. Past a
first window, which is 42% of ARC-3's tokens, the head's attention reached back
16k to 65k tokens in training, against the whole chain when it is served.

`drafter.evaluate --min-position` measures what that costs: the same late
positions, scored by the same head over windows and over whole chains
(evaluation has no backward, so a whole chain fits a card):

    DRAFTER_MAX_TOKENS=65536 python -m drafter.jobs NAME-windows --minutes 90 -- drafter.evaluate OUT/windows.json \
        --set val=$CACHE_VAL:$DATASET_VAL --head trained=$HEAD --head base=base --steps 4 --min-position 65536
    python -m drafter.jobs NAME-whole --minutes 90 -- drafter.evaluate OUT/whole.json \
        --set val=$CACHE_VAL:$DATASET_VAL --head trained=$HEAD --head base=base --steps 4 --min-position 65536

On ARC-3's held-out games (20 chains past 65,536 tokens, 815k scored positions
from there on), per draft step
(`/data/shared/research/arc_3_results/shared/drafter/evals/arc3-lookback/`):

| acceptance at T=0.6 | step 0 | step 1 | step 2 | step 3 | accepted tokens per draft, as served |
|---|---|---|---|---|---|
| trained head over windows of 65,536, as trained | 0.864 | 0.810 | 0.778 | 0.756 | 2.540 |
| trained head over the whole chain, as served | 0.865 | 0.812 | 0.781 | 0.759 | 2.554 |
| base head over the whole chain | 0.825 | 0.773 | 0.732 | 0.703 | 2.291 |

Serving's longer reach costs the window-trained head nothing: over whole chains
it scores a little higher than over the windows it was trained on (+0.1 to +0.4
points a step, +0.5% accepted tokens), and its gain over the base head holds
there (+3.9 to +5.7 points). What a head trained over whole chains would add is
not measured, since those chains do not fit a card in training; the small gap
between the first two rows says how little the far reach moves this head.

## Image embeddings for the head (parked)

Parked on 2026-10-09: nobody has measured whether this is worth doing. This
section is the passdown for whoever picks it up.

**Where things stand.**
- **The served head never receives image embeddings.** vLLM's proposer passes
  the target's image embeddings to a drafter only if the drafter declares
  `supports_multimodal_embeddings`. `Qwen4ExpMTP` does not, and its
  `embed_input_ids` takes token ids only (fork `tufa/v0.29.0`,
  `vllm/models/qwen4_exp/nvidia/mtp.py`). The active V2 runner falls back to
  text-only inputs (`DraftModelSpeculator.load_model` in
  `vllm/v1/worker/gpu/spec_decode/speculator.py`). At an image placeholder
  (`chains.IMAGE_TOKEN_ID`, 248056) the head embeds the placeholder token
  from the target's embedding table.
- **The images still reach the head, second-hand.** `drafter.extract` serves
  each chain with its images, so the stream over an image span carries them.
  The head's keys over that span are computed from that stream.
- **Training matches serving.** `drafter.train` embeds the ids
  (`F.embedding(ids, label.embed)`), so it sees the placeholder at image
  positions, exactly as served. Every head trained so far (`v1`, `v1-nvfp4`,
  the per-task prunes' heads) is consistent with today's serving.
- **Not confirmed from a log.** The proposer warns when it drops to text-only.
  That warning did not appear in the engine log of a four-L4 traffic run
  (ARC-2 run S, 2026-10-08), so the conclusion rests on the code. Confirm the
  active path first (step 1 below).

**Why the gain is expected to be small.** Drafting happens while the reply is
generated, and every input there is a text token. Images enter only as
context: through the stream, which already carries them, and through the
head's attention over the prompt. A real image embedding changes only what
the head computes for the image positions themselves.

**The trap: changing serving alone.** Never feed real image embeddings to a
head trained on placeholders. That shows the head inputs it never saw and can
only lose acceptance. Training and serving change together, and the serving
switch stays off for every head trained the old way. The routing replay had
the converse mismatch, images replayed as placeholders, until they stopped
being behind a flag (step 2 above, `docs/flash_pruning.md`).

**What it takes.**
1. **Confirm the current path.** `drafter.capture` uses the V2 runner. On a
   running engine, read `DraftModelSpeculator.supports_mm_inputs` after
   `load_model` in `vllm/v1/worker/gpu/spec_decode/speculator.py`, or add a
   one-line log; trace the embedding merge through
   `AutoRegressiveSpeculator._run_model` in
   `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`. Keep the
   log: verification step 2 needs it.
2. **Measure before building.**
   - *Cheap check:* `drafter.evaluate` on held-out chains with the current
     head, comparing reply positions in calls with images against calls
     without. If calls with images are not clearly worse, stop. Nearly every
     ARC-2 call carries images, so use ARC-3 chains or a split matched on
     context length.
   - *Decisive experiment:* steps 3 and 4 with a short retrain, against a
     head retrained on the same schedule with placeholders. Only a clear
     per-draft gain justifies step 5.
3. **Extraction.** Also store the target's merged input embeddings at image
   positions (`[image tokens, 2560]` bf16 a chain), captured in the same pass
   as the stream so the two match exactly.
4. **Training.** At image positions, use the stored embeddings in place of the
   placeholder's in `drafter.train`; change nothing else. Record it in the
   exported head's provenance (for example `"image_embeddings": true`).
5. **Fork (serving).**
   - Separate the embedding-buffer width from the feedback-state width in
     V2's `DraftModelSpeculator` and `AutoRegressiveSpeculator.load_model`
     (the two files in step 1). Qwen4Exp's four-stream MTP feedback widens
     `self.hidden_size` to 10,240, but token/image embeddings remain 2,560
     wide. Allocate `inputs_embeds` at the embedding width and keep
     `hidden_states` at the feedback width. Otherwise `_run_model` assigns
     2,560-wide `embed_input_ids(...)` output into a 10,240-wide buffer and
     crashes as soon as multimodal support is enabled, even on text-only
     calls on a single GPU. This is a prerequisite for the interface below;
     pipeline-parallel transport in step 6 does not fix it.
   - Give `Qwen4ExpMTP` the multimodal-embedding interface:
     `supports_multimodal_embeddings`, and an `embed_input_ids(input_ids,
     multimodal_embeddings=None, is_multimodal=None)` that merges them as
     the target's `Qwen4ExpForConditionalGeneration.embed_input_ids` does.
   - Gate it on a `speculative_config` option that defaults off, so every
     existing head keeps the placeholder path. Turn it on only for a head
     whose provenance says it was trained with image embeddings.
6. **Pipeline parallel (ARC-2's four-L4 Kaggle cell).** The vision encoder
   runs on the first stage only (`gpu_model_runner.py`, under
   `is_first_rank`). The drafter sits on the last stage, which has no image
   embeddings. Two options:
   - Send the scheduled image tokens' embeddings to the last stage with the
     intermediate tensors. This is needed only on steps that prefill images.
   - Run the encoder again on the last stage. This costs its weights and
     compute on the stage that sets the KV pool size, which had 30-80 MiB
     free at utilization 0.91.

   The single-GPU cells (B200, RTX) have no such problem.

**Verification.** Run these in order:
1. **No regression with the switch off.** An existing head drafts exactly as
   before: `drafter.online` acceptance on the same chains against the
   pre-change fork.
2. **The new path is active.** With the switch on, the step-1 log shows the
   V2 speculator taking multimodal inputs. First run single-GPU text-only
   and image-bearing calls and check that the embedding buffer stays
   2,560-wide while the feedback state stays 10,240-wide, with no assignment
   error. A counter of image tokens merged into the drafter must be non-zero
   on image-bearing calls.
3. **Parity over image spans.** Capture with `drafter.extract --drafter` and
   run `drafter.parity` on image-bearing chains, with the switch on and the
   port given the stored embeddings. Expect the usual bf16 floor (median
   relative error 0.07-0.08, top-1 agreement 95-97%; `DRAFTER_README.md`,
   section 3). Positions inside image spans must be no worse than text
   positions.
4. **Offline gain.** `drafter.evaluate` per-draft acceptance of the new head
   against the placeholder-trained one from step 2, on held-out sets, split
   by calls with and without images.
5. **Gameplay.** `tools/long_run_arc{3,2}.py` with each head on the same games;
   read the engine's acceptance lines weighted by generated tokens.
6. **On the four L4s, if the last-stage route is built.**
   - A traffic replay limited by the pool, against the same harness revision
     with the switch off.
   - Compare: settled tok/s; the step interval on steps that prefill images;
     the KV pool in tokens; the last stage's lowest free GPU memory.
   - The gain must survive the transfer or encoder cost.

Scores are not at stake: speculative decoding leaves the output distribution
unchanged. Only acceptance and throughput are.
