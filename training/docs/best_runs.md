# Best runs and lessons (ARC-AGI-3 GRPO, updated 2026-08-30)

## 25-game leaderboard (avg_arc_score_sampled across all 25 officials)

Policy (2026-08-30): all new runs train on all 25 official games; selective
*sampling* among them is allowed, set exclusions are not. Code commit =
`root_commit` of the run's frozen source snapshot (`source_snapshot.json`),
i.e. the code the run actually trains on (refreezes upgrade it mid-run).

| Run (kander dir) | Peak | Status | Code commit |
|---|---|---|---|
| `20260829_073025_platypus_games25_hidden` | **3.47** @ iter 87 (14/25 games scoring; entropy 0.87 @ iter 90) | running — THE default recipe (`configs/ppo_2gpu_arc3_games25_platypus.json`) | `fe0c4d9` |
| `20260828_094012_platypus_games25_ema8_champbase` | 2.77 @ iter 76 | dead — terse entropy collapse @ ~85 (old 0.25 floor) | `fe0c4d9` |
| `20260829_133015_platypus_games25_win20cap_hidden` | 2.71 @ iter 29; ahead of games25_hidden at matched age (15/25 scoring) | running (flagship) | `96155da` |
| `20260829_073026_platypus_games25_win20cap2048` (ema8) | 1.75 @ iter 41 | running, fading | `fe0c4d9` |
| `20260829_094737_platypus_games25_ema8_v2` | 1.65 @ iter 13 | running, fading | `fe0c4d9` |

Peak differences under ~1.7 points are within run-to-run noise. Estimator
dominates: hidden-critic arms hold the top places; ema8 arms fade. The curated
16-game diagnostic (`20260829_073045_platypus_games16_ema8`, peak 4.4 avg over
its 16) showed coverage curation triples the multi-game average — fold that
back in as *sampling weights* over the full 25, never as set exclusion.

Archived run directories live in `/data/personal/$USER/training_runs_old/` on kander;
active runs in `/data/personal/$USER/training_runs/`. Each newer run carries a
`PURPOSE.md` stating its question, baseline, and retire condition.

## Best results per setting

| Setting | Best score | Run (kander dir) | Config notes |
|---|---|---|---|
| ar25 single-game | **8.53** sampled (iter 64, ~17h active), 4 levels | `20260827_115734_..._win15_entctrl` | former default (single-game era): 15-min episodes, GRPO 32 rollouts, lr 5e-6, entropy floor |
| Pathfinder single-game | **87.4 / 100**, 4.9/5 levels (iter ~44) | clip03_win10 (10-min episodes) | near game ceiling; degraded past peak without entropy control |
| tu93 single-game | **22.22**, 4/9 levels | tu93 sweep (10-min) | all-time max on tu93 |
| 25 official games | ~1.1 avg (25-game mean), no trend yet at iter 82 | `20260828_094012_..._games25_ema8` | ema8 per-game baseline, 1 rollout/game/window |

## Established lessons

1. **Episode horizon is the strongest score lever.** 15-min vs 10-min episodes ≈ 2x
   sampled score at matched iteration (rounds/episode scale with the time budget).
2. **Entropy collapse is the universal failure mode.** Identical decay slope
   (~-0.012/iter) across configs; below entropy ~0.30 the verbosity spiral starts
   (tokens/gen up, rounds down, actions down, score down). Collapse arrived on the
   same iteration schedule (~70-95) in every unprotected run (kl01, seed1337,
   champ_lr1e5, critic). 5x stronger window-KL (kl01) did not prevent it.
3. **The entropy floor is prevention, not cure.** The strengthened floor
   (target_low 0.32, step 5e-5, cap 1e-3) held entropy at ~0.28-0.36 when applied
   mid-spiral to the champion, but score did not rebound over ~13 windows.
   All fleet configs now carry the floor from birth.
4. **Coverage principle (games25 data, 82 windows):** 9 of 25 official games have
   <25% nonzero-return rate under the current policy — sparse reward can't learn
   there. Curated 16-game set (nonzero >= 25% and positive mean return):
   ft09, ar25, vc33, sp80, lp85, sb26, re86, tu93, su15, ls20, r11l, cn04, s5i5,
   cd82, ka59, lf52.
5. **Packing-invariant crashes end chains silently.** Verbosity-collapsed policies
   emit outlier sequences (41-43k tokens > 40960 budget) that trip the fail-hard
   invariant, killing the run mid-segment (champion + critic, night of 08-28).
   Open fix decision: drop-and-count oversized samples with a fatal threshold.
6. **Run-to-run noise on ar25 is ~1.7 points** — score differences under ~2 points
   between arms are not interpretable.
