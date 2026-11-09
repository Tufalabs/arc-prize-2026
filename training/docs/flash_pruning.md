# Flash-Next expert pruning, per task

Flash-Next (Qwen3.8-Flash-Next-NVFP4) routes 10 of 512 experts per layer. This branch prunes it
separately for ARC-3 and ARC-2: calibrate on the task, keep the experts that matter, cut the rest out
of the checkpoint, and then train a drafter (speculative-decoding head) against each pruned model.

## Pipeline

1. **Calibration rollouts** of the unpruned model on the task (ARC-3: the 25 official games; ARC-2:
   arc2-train), on main's harnesses. The rollouts' `benchmark.json` and `traces/<run>/steps.jsonl`
   are all the later steps read.
2. **Routing replay** (`scripts/prune_route_stats.py`, one B200 per shard via `scripts/flash_dev.sbatch`):
   every rollout's chains go back through the unpruned model in 8192-token chunks
   (`training.flash_route_stats.ServedReplay`; a chain's images go through the frozen vision tower),
   recording per layer and expert the sums of gate * ||output||, gate and routed tokens.
   `--verify-tokens` first checks, on a real chain, that the chunked replay is as faithful to the served
   model's reply log-probs as one forward (ARC-3: 0.0906 vs 0.0930 mean |dlogprob|; ARC-2 with images:
   0.074, against 0.245 with placeholders as text) and that per-expert aggregates agree.
3. **Plans** (`scripts/prune_plans.py`, CPU): per layer, the `k` highest-scoring experts under REAP
   (arXiv 2510.13999: mean gate * output norm over the tokens routed to an expert), mass (the same with
   frequency included) and score-weighted variants.
4. **Probe** (`scripts/prune_probe_plans.py`, optional): each plan's loss of reply log-likelihood on
   held-out winning rollouts, with the pruned experts masked from the router. Cheap, but it did not
   predict the ARC score (below).
5. **Physical prune** (`scripts/prune_physical.sbatch`, CPU): arc-training's
   `scripts/prune_flashnext_experts.py` drops the other experts and their router rows.

## First results (ARC-3, 25 games, 60 min, same protocol for every arm)

Measured before the harnesses were moved to main (older Duck, plain T=1), so absolute scores are low;
every arm ran identically. Mean ARC score:

| arm | seeds 1001-1004 | notes |
|---|---|---|
| REAP-307 | 12.17 (11.0 over 8 seeds) | above unpruned on every seed |
| e307 (mass+norm+cond+uniform: REAP on our calibration) | 6.56 | |
| unpruned 512 | 6.03 | 5.94 over 8 seeds |
| mass-307 | 5.84 | best on the probe |
| REAP-307 score-enriched, beta 1 / 3 | 7.73 / 4.63 | outcome weighting hurt |
| REAP-256 / mass-256 | 3.94 / 1.28 | too aggressive |

On arc2-eval (held out), REAP-307 calibrated on ARC-3 scores 0.57, as the unpruned model does
(e307 0.52). The re-evaluation on main's harnesses at the Flash column's deployed sampling (T=1,
top_p 0.95, top_k 20) and the ARC-2-calibrated plans follow.

## The per-task procedure

The comparison that counts is main's own: the deployed B200 cells' long runs (ARC-3: the official 25 x
10, e307 + v1 head, 56.55; ARC-2: arc2-eval x 10, 81.76). `spec_decode/README.md` has the procedure that
produces an ARC-3 and an ARC-2 model, each calibrated on its own deployed-cell play, pruned by REAP and
given its own drafter (the drafter chain, `spec_decode/drafter/`, retargeted per model), and
evaluated on those long runs.

## Results on the deployed cells (2026-09-30)

Each model was calibrated on the unpruned model's play of its task (ARC-3: the official 25 and
`ARC_LIKE_HARD`, 72 rollouts; ARC-2: arc2-eval and arc2-train-hard, 334 rollouts), pruned by REAP to 307
experts, and evaluated on main's long runs. A run's score is the mean final score of its games; the
pruned models' runs were split into shards by passes (no game nears its wall-clock cap, so the score
distribution is the same; the seeds are per game from OS entropy).

A long run's score on these cells does not depend on which head drafted it: drafting is lossless (the
target verifies every drafted token) and no game nears its time limit, so each model's runs are pooled
over both heads. On Kaggle the nine hours bind, and there a head's speed does reach the score.

| task | model | runs (passes each) | passes | score |
|---|---|---|---|---|
| ARC-2 (arc2-eval) | e307 (main) | 83.46, 81.94, 81.76 (10) | 30 | 82.39 |
| | REAP-307, ARC-2-calibrated | 83.11, 81.65, 82.25, 81.28 (10) | 40 | 82.07 |
| ARC-3 (official 25) | e307 (main) | 57.22, 56.55 (10) | 20 | 56.89 |
| | REAP-307, ARC-3-calibrated | 57.96, 58.28, 58.38, 58.93, 60.20, 60.98 (5); 58.37 (10) | 40 | 58.93 |

e307 is REAP too: its `pruning.json` names the rule `mass+norm+cond+uniform`, the historical rule
matching arc-training's `scripts/prune_plans.py` REAP criterion: the gate x output-norm sum divided
by the routing count, with the same number of experts kept in every layer. Its histogram is a
2026-09-05 calibration run recorded by the serving engine (127,632 captures). So the comparison isolates the calibration data (and one model per task): the same criterion
on each task's own deployed-cell play, which for ARC-3 includes the 25 games it is scored on.

REAP-307's kept experts overlap e307's by 0.873 (ARC-2 plan) and 0.887 (ARC-3 plan); the two task plans
share 0.93 with round 1's ARC-3 plan. On ARC-2 the pruned model ties main (40 passes: 82.07 against
82.39); on ARC-3 it beats main by 2.0 points over all 40 passes (58.93 against 56.89), every 5-pass
run above both e307 runs.

**Drafters.** Each head starts from its target's own MTP block and trains on that target's play (ARC-2:
15.5M generated tokens, 188 updates; ARC-3: 10.4M, 83 updates over 332 windows of at most 65,536
tokens). A head is measured by the tokens its target accepts per draft (k=4 on these cells):

| target | head | held-out games, offline | in play, whole long runs |
|---|---|---|---|
| REAP-307 ARC-2 | its MTP block, untrained | 2.33 | |
| | its own, trained | 2.66 | 2.42-2.43 |
| | v1 | 2.65 | 2.44 |
| e307, on ARC-2 | v1 | | 2.44-2.45 |
| REAP-307 ARC-3 | its MTP block, untrained | 2.25 | |
| | its own, trained | 2.51 | 2.35-2.38 |
| | v1 | 2.53 | 2.39-2.40 |
| e307, on ARC-3 | v1 | | 2.38 |

The held-out column is two whole games per training dataset that the task's own head never saw, scored
offline on the cached stream by the served sampler's acceptance rule, with the head before quantization
and with its full vocabulary (per draft step at T=0.6: 0.835/0.786/0.747/0.716 to
0.881/0.834/0.807/0.788 on ARC-2, 0.820/0.766/0.724/0.695 to 0.859/0.804/0.772/0.751 on ARC-3). They
are out of sample for the task's own head and not for v1 (`heads/v1-final` under
`/data/shared/research/arc_3_results/shared/drafter/`), whose training covered these games on e307's
play; v1 scores 0.881/0.832/0.804/0.783 and 0.862/0.807/0.775/0.753 on them. The in-play column is the
engine's own count over whole long runs, weighted by generation, with the shipped head (NVFP4, 16k
draft vocabulary), on games the heads were trained on, v1 included.

Both ways the task heads and v1 are within 0.03 tokens a draft of each other on these targets: level
on ARC-2, v1 slightly ahead on ARC-3. The task heads trained on 10-15M tokens of one task each, v1 on
~35M over both, and their gains over the base head per draft step (ARC-3 +0.039/+0.038/+0.048, ARC-2
+0.046/+0.048/+0.060 at steps 0-2) are the ones v1 made on its own validation games
(+0.039/+0.040/+0.049 and +0.046/+0.045/+0.054). So a head trained on a pruned
target's own play learns what v1 learned, and starting from v1's weights is unlikely to add much.

The two columns also differ in the draft vocabulary, which is not what separates them. In play every
head, v1 included, drafts from the 16k ids its training chains generate most; a position whose token
lies outside them cannot be drafted, and the draft ends there, which at k=4 costs about 5.6 times that
share of tokens in tokens a draft (the sum over the draft steps of step times the chance of reaching
it). On the training chains, which come from the games the long runs play, 0.05-0.07% of ARC-3's
generated tokens lie outside (ARC-2: 0.01-0.02%), worth under 0.005 tokens a draft. On ARC-3's held-out
games 0.23-0.91% lie outside (ARC-2: 0.03-0.04%), worth 0.01-0.05 tokens a draft: on unseen games,
Kaggle's hidden ones included, the cut is not free, and ranking the ids over more games or a larger
draft vocabulary would narrow it (`draft_vocab_provenance.json` in each head's directory).

ARC-3's chains (127k tokens at the median) do not fit a B200 in training, so its head trained on
windows of 65,536 tokens: past a first window its own attention reached back 16k to 65k tokens,
against the whole chain when served (the target's stream it reads is the whole chain's either way).
Scored on the same 815k held-out positions past 65,536, the head accepts 0.864/0.810/0.778/0.756
over windows and 0.865/0.812/0.781/0.759 over whole chains, so the longer reach of serving costs
it nothing (`spec_decode/README.md`, "What the head sees in training").

**Images in the ARC-3 calibration (found 2026-10-05).** Main's ARC-3 duck sends the grid as an image,
so every one of the 250 ARC-3 calibration chains carries images (4,807 of them; their placeholders are
3.8% of the 29.5M tokens), as every ARC-2 chain does (1.3% of 20.2M). The ARC-2 replay put the images
through the vision tower (`--multimodal`). The ARC-3 replay ran as text: each placeholder went in as
its own token embedding where the served model read image features, and the tokens after an image
were computed without it. The shipped ARC-3 plan therefore rests on statistics of a model that was not
quite the served one. The model itself was evaluated as served, with its images, on the long runs and
on Kaggle, so its scores stand; what a replay with the images would change in the plan is not yet
measured (a replay of the same 72 rollouts with their images is in
`/data/shared/research/arc_3_results/shared/prune_stats/arc3_cal_unpruned_images_20261005/`). Images are
no longer behind a flag: the replay and the probe run every chain through one path
(`training.flash_route_stats.ServedReplay`) that puts its images through the vision tower and refuses
image placeholders that have no images, and each plan records the checkpoint and whether images were
replayed. No padding of any other kind enters either set of statistics: chains replay one at a time at
their own length, and every layer's routed count is exactly 10 per token over all 406 rollouts.

**Kaggle (ARC-3, 2026-10-03).** The ARC-3 model with its own head, on main's `kaggle-rtx` cell
(arc3-platypus `kaggle-arc3-reap307`, `tools/kaggle_run.py`: the cell with the two datasets
swapped), scored 55.89 on the public leaderboard, against 50.65 for e307 with the v1 head
(2026-09-30) and 52.51 for the Seal (2026-10-01). One submission each, and we put the
leaderboard's run-to-run spread at 5-10%, so this agrees with the long runs rather than adding
to them. A first submission of the same notebook scored 19.07 because its rerun stopped after
3h32m of its nine hours, cause unknown; a nine-hour offline run of it finished normally (110
games, no out-of-memory kill, host memory flat at about 150 GiB of the 175 GiB limit).
