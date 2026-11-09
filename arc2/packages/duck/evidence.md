# Duck evidence

*Register: living.*

Where the duck's measurements are kept — the reading that justifies a
current default, and what it does and does not settle. Not the detail: that
stays recoverable from the job dir, where `git_status.txt` names the commit
and `games/` holds the spans.

Runs live under `/data/shared/research/arc_2_results/shared/`.

**A default with no entry here is a judgement, not a measurement.** Most of
them are. An entry earns its place by being the thing that would have to be
re-measured before the default moved, so read the absence of a reading as an
open question rather than as a settled small effect.

## Duck sweep 4 — the presentation and oracle defaults

`duck_sweep4`, 100 arms over 26 knobs, the 120-task ARC2 evaluation set once
per arm, 12,000 lanes. Every arm read at 40,000 generated tokens per game
under the official ARC scorer, so arms are compared at a spend they shared;
the per-game cap was 60,000 and 37% of games reached it. Produced at
`arc2-experiments` commit `0aa4e9b`, which pins this repo at
`973bb69` — every job's `git_status.txt` records both.

The full report, with each conclusion read both raw and adjusted, is not
released (2026-08-31 duck sweep 4 findings; source notebook in
`arc2-experiments`).

**One knob clears a Benjamini-Hochberg screen across the 26**:
`reasoning_effort=xhigh`, **+3.23 ± 1.04** (q = 0.067) — and it was already
set, on the Qwen presets rather than the class, for the reason the effort
level is provider-specific. It is confirmation, not a change.

What the sweep moved, in adjusted score points over each knob's range:

| default | reading | screen |
|---|---|---|
| `text_grids=True` | +2.21 ± 1.07 (p = 0.038) | q = 0.47 |
| `check_mode` away from `MANDATORY` | −2.50 ± 1.32 for mandatory (p = 0.058) | q = 0.82 |
| `completion_cutoff_notice=False` | −2.00 ± 1.09 for on (p = 0.068) | q = 0.47 |
| `repetition_penalty` off (1.0) | −4.44 over the drawn 1.0–1.1 (p = 0.052) | q = 0.47 |

Only the first is a change; the other three are guards on defaults that were
already right. None of the four clears the screen, so each is a consistent
direction rather than a proof, adopted where the cost of being wrong is a
prompt that is slightly longer or a knob that stays where it was.

`check_mode=ENABLED` is **not** on this list and is not a measurement: the
sweep separates `MANDATORY` from the other two and cannot separate `ENABLED`
from `DISABLED` (−0.46 ± 1.05). Offering the oracle was adopted on the
judgement that a channel the model may ignore costs nothing and may be
learned; that is the one adopted change with no reading behind it.

**Throughput, where the score is silent.** `input_per_generated` — uncached
prompt tokens per generated token — is the closest thing the sweep has to a
serving cost. The engine's host KV tier is **−0.252 ± 0.059** (p = 2×10⁻⁵)
on it at no cost in score (−0.61 ± 1.07), and `text_grids` is **+0.243 ±
0.050** (p = 1×10⁻⁶) — the same size, opposite sign. The two together leave
the ratio where it started, which is what makes the digit rows affordable.
`xhigh` is a third throughput win (−0.458, p < 10⁻¹³).

The tier was drawn at 0 or **64 GiB** on vLLM 0.25.1, and the reading is
**superseded by the engine change, not refuted**: the deployment runs the
vLLM fork (`vllm_fork.md`), on which the connector-aware
scheduler path the tier's presence used to select — and the zeroed
recurrent state it produced on warm prefix hits, the accident upstream
measured its score on — no longer exists. The tier ships **off**. Its
throughput reading stands as a fact about 0.25.1; what the tier buys on
the fork has not been measured, and this is the line to revisit if it is.

**Nothing interacts.** Every knob was tested against each of the three the
sweep settles: 75 nested F-tests, 7 under a raw p of 0.05 against 3.8
expected by chance, and nothing surviving the screen (smallest q = 0.47).
`null_control` — drawn and never applied — pairs with `check_mode` at
p = 0.080, inside the top ten, which is what fixes the ceiling on how much
of that list is signal.

**What it does not settle.** Twenty-one of the 26 knobs are indistinguishable
from `null_control` here, which bounds their effects rather than zeroing
them: a single arm's score carries a standard error of about 4.3 points, and
the sweep works by averaging ~50 arms per level, not by resolving one arm. A
two-point effect is entirely possible in most of them.

It says nothing about whether to send pictures at all — every arm here sends
them, after sweep 3 put `images="off"` about nine points behind on 13 of its
40 arms. It says nothing about compaction: roughly a tenth of lanes compact
at this budget, so all four compaction knobs are being asked a question the
data barely poses. And it prices no wall clock — the budget is in tokens, so
a slow arm is not charged for being slow.

## Duck sweeps 5, 6 and 7 — the settings

Three randomised sweeps over the duck, 2026-09-17 to 2026-09-20, each fixing
what the one before it settled and moving the ranges it left open, on the 105
tasks Flash-Next does not always solve (58 of `arc2-eval` scoring below 100
over six baseline passes, and `arc2-train-hard`). Every knob any of them drew,
with every reading and the rationale for each default kept or changed, is in
these reports, which are not released:

- Duck sweep 5
  — 60 arms at 150k a game (this repo at `91cd2b3`).
- Duck sweep 6
  — 150 arms at 150k (`91cd2b3`, `fb77138`).
- Duck sweep 7
  — 420 arms drawn, 304 read, at 300k (`d484e24` to `a209ca4`).
- Sweeps 5, 6 and 7 consolidated
  — the two tables, defaults changed and defaults kept.

Two of the reports' calls were overridden afterwards (2026-09-22), once
throughput was priced at its Kaggle value: `nudge_mode` is causal and
`text_grids` off, where the reports keep legacy and on ("The calls at the
Kaggle weight", below). The tables here carry the calls as they stand.

Every arm is read at its terminal score under the ARC scorer; sweep 7 also
prices an arm's traffic as tasks per hour on Kaggle's four L4s
(`kaggle_scheduler` in `arc2-experiments`) and fits the score with the
throughput in, `score + 0.3 × tasks per hour`. Effects come from one joint OLS
over every knob at once with HC3 errors, a type-II test per knob and
Benjamini–Hochberg *q* across the knobs; sweep 7's omnibus F(40, 263) = 5.08
against a permutation null of R² 0.13 (observed 0.44), with `null_control`
at p = 0.57. One take per game, TRT-LLM MoE kernels, a B200: not the
deployment's regime (below).

**Defaults changed**, in points of sweep 7's throughput-corrected score
against the value that shipped before, unless another sweep is named:

| default | reading | sweep |
|---|---|---|
| `presence_penalty` 1.5 → 0.0 | −10.7 over 0–1.45 (p = 0.003, q = 0.022), monotone, no token saving; the shipped end was the worst | 5 |
| `reasoning_effort` xhigh (confirmed) | medium −11.4 (q = 0.001), −8.0 at an equal spend for 8% fewer tokens | 5 |
| `repetition_penalty` 1.0 (confirmed) | −23.6 over 1.0–1.15 (p = 3e-12): it counts prompt tokens and bites as the context grows | 5 |
| `check_mode` enabled (confirmed) | mandatory −7.4 (q = 0.027), then −6.8 (p = 8e-11): reach, not accuracy — −0.3 at an equal spend | 5, 6 |
| `images` on (confirmed) | off −4.0 (q = 0.042); every picture mode +4.5 to +5.0 over off, the three not separated | 6 |
| `compaction_strategy` prune → summarize | the live-request summary +11.4 over prune (q < 0.001), at least as good as the detached one at every budget and +2.9 above 25k, ~11 points fewer give-ups, cheaper | 6 |
| `reply_cap_tokens` 16,384 → 24,576 | a knee at 10–11k; +0.13 per 1k over 12–32k (p = 0.001), nearly all below ~22k; tasks an hour −0.11 to −0.18 per 1k throughout ("The reply cap", below) | 7 |
| `generalization_audit_line` off → on | +2.5 corrected, +2.9 of score (p = 2e-9), −1.4 tasks an hour; 63% of it on the first attempt. Off again beside the audit turn (sweeps 8 and 9, below) | 5, 6, 7 |
| `no_hand_arithmetic_line` off → on | with the digit rows, +2.8 over the rows alone (p = 2e-4); every cell of rows × caveat beats the shipped one by 2.8–3.7 (p = 1e-6) | 7 |
| `text_grids` on → off | at the Kaggle weight, no rows with the caveat +1.3 over the rows with it (p = 0.10); +0.95 at the old weight (p = 0.19). The second take keeps them | 7, reweighted |
| `nudge_mode` legacy → causal | at the Kaggle weight +1.3 (p = 0.018): a flat score and +1.0 tasks an hour (p = 9e-10) | 7, reweighted |
| `image_gridlines` on → off | +1.6 (p = 0.001), 1.1 more tasks an hour; +2.8 in sweep 6 (q = 0.093) | 6, 7 |
| `image_legend` on → off | +1.3 (p = 0.007); +2.2 in sweep 6 (p = 0.05) | 6, 7 |
| `nudge_format_guidance` on → off | +0.6 (p = 0.18); nothing argued for the sentences | 7 |
| `summary_tail` full → without_reasoning | +0.7 (p = 0.37), the three tails within a point | 7 |
| `check_prints_on_pass` off → on | nothing either way in three sweeps; the design that makes sense | — |
| the recurrent state in bfloat16 (ullm's Flash-Next engine) | +0.5 [−0.4, +1.4] on the score (p = 0.24), about +9% tokens a second at matched concurrency | 7 |

**Defaults kept**, each with a reading rather than none: `prompt_budget`
48,640 sits on the flat part of a soft knee near 38k (bootstrap 26.5–44.5k;
+0.16 per 1k below the 45k midpoint, −0.01 above); `temperature` 0.6, with the
harm above ~0.78 (sweep 5) and nothing below it down to greedy (+1.5 over
0.0–0.7, p = 0.054), since moved to 0.8 (sweeps 8 and 9, below); `image_layout` separate (paired −1.2,
p = 0.014); `image_scale` 4 (three levels never separated); `ascii_row_numbers`
on (off −0.7 to −1.8, three times the same way); `check_and_submit` off (on
−2.6, p = 5e-8: auto-submit games offer a distinct second attempt in 36.8% of
games against 44.4%); `second_attempt_mode` optional (`none` −4.9 against
`audit`, p = 0.0002, ruled out; `audit` +0.35, p = 0.47); `budget_line` off
(+0.9, p = 0.08, but Kaggle has no fixed budget to print); `brevity` off,
`investigation_lines` on, `work_policy` none, `assign_early_line`,
`multi_test_line`, `test_out_structure_explicit`, `completion_cutoff_notice`,
`cost_model`, `budget_warning_tokens` off, `tool_output_tokens` 2,425,
`python_timeout_s` 90, `compaction_ratio` 16 and `summary_thinking` off: none
separated from the shipped value in any sweep (p ≥ 0.12), most with the
shipped value the fit's best. Prune's own knobs (`compact_to_tokens`,
`keep_task_prompt`, `evict_turns`) resolved nothing in sweeps 5 and 6 and no
shipped column reads them now.

**pass@2 discounts a single take's reading.** The deployment composes each
task's two attempts from two independent takes, so a knob that helps only by
improving the second answer slot is worth less deployed: over the arms that
played a task twice, the two takes' first grids were identical 62.5% of the
time, and the deployed second attempt is the second take's first grid 37.5%
of the time, the first take's second grid 19.8%, the second take's second
grid 8.9%. `second_attempt_mode` is 92% a second-slot effect (its −4.9 is
about 1.5 deployed); `check_and_submit` and the audit line are 64% and 63%
first-attempt effects, so their calls hold. The second take plays with the
digit rows the first leaves out, `brevity="half"` and `summary_tail="full"` —
each read as no worse than the first take's value and none resolved — to move
which tasks it solves rather than how many.

**What the sweeps did not read.** The deployment's own regime: Marlin's
kernels where the sweeps served TRT-LLM (ullm's record puts them level on
score, unread here), two takes with `likeliest_first_line`, Kaggle's token-
ranked scheduling with no per-game cap under a 12-hour clock, and the full
`arc2-eval` set. Two B200 comparisons read the settings against `main`'s
defaults, which also carry the float32 state at 150 slots (2026-09-22):
sweep-like (TRT-LLM, the 105 tasks, 8 passes, one take at 300k) 56.5 to 74.5,
where sweep 7's fit put the settings at 76 (72–80 for one run); Kaggle-like
(Marlin, the eval set, 4 passes, two takes at 300k and 150k) 84.7 to 91.6,
all of it on the 58 hard tasks (69.5 to 82.6; the other 62 at 99 and 100). A
game generates 9% and 12% more under the settings (70.1k against 64.1k, 109.9k
against 98.3k) at half the uncached input per generated token, so on the
four-L4 cost model their tasks an hour come out 1% and 7% below `main`'s. The
summary tail's share of the extra generation is paid back in context and
uncached input on the four L4s (−0.36 ± 0.41 tasks an hour in sweep 7) and
not on the B200.

## The calls at the Kaggle weight

Sweep 7 priced throughput into its objective at 0.3 points a task an hour, a
guess. Calibrated (2026-09-22), it is about four times that.

**What 1% of throughput is worth on Kaggle.** `take_gap/kaggle_value.ipynb`
in `arc2-experiments` replays the two Kaggle-like B200 comparison runs
as Kaggle submissions (240 tasks, 60 slots, least-tokens with the 150k gap,
twelve hours, each game as played): at the settings here, a 1% change of
throughput moves the score at the deadline by **+0.086 points** (±0.001 over
±10%; +0.082 over ±5%, about +0.11 over ±20%), with the machine at the
calibrated 475 generated tokens a second on the old defaults' traffic; at the
old defaults, by +0.11. Neither set of games finishes in twelve hours. The
same replay puts the settings here 2.7 points above the old ones on Kaggle
(85.2 against 82.5).

**Into the sweeps' units.** A sweep point is a single take on the 105 hard
tasks; a Kaggle point is two takes on all of `arc2-eval`. The same change of
defaults read +16.0 sweep-like and +5.2 Kaggle-like, so a sweep point is worth
about 0.33 of a Kaggle one, and the weight is **0.26 sweep points per 1% of
tasks an hour**: about 1.3 per task an hour at sweep 7's 20, 1.2 at sweep 6's
22 and 0.85 at sweep 5's 31. It falls as throughput rises, and one ratio of
0.33 stands for every knob, where a second-slot effect compresses more under
two takes than a first-attempt one.

**What it moves in sweep 7** (the joint fit on the same 304 arms, sweep
points):

| call | weight 0.3 | Kaggle weight |
|---|---|---|
| `generalization_audit_line` on | +2.5 (p < 0.001) | +1.0 (p = 0.05) |
| `reply_cap_tokens` 16,384 → 24,576 | +1.1 (p = 0.001) | −0.1 (p = 0.73) |
| `nudge_mode` causal | +0.2 (p = 0.63) | **+1.3 (p = 0.018)** |
| no digit rows, with the caveat | +0.95 (p = 0.19) | +1.3 (p = 0.10) |
| `prompt_budget` 48,640 → 60,000 | +0.9 (p = 0.003) | +1.3 (p < 0.001) |
| `summary_tail` without reasoning | +0.7 (p = 0.20) | +0.4 (p = 0.58) |

Sweeps 5 and 6, through sweep 7's analysis with the reply cap above 12k and
the budget at 30k or more: every call holds at the Kaggle weight, the legend
and the audit line weakening there as in sweep 7; sweep 5, left with 47 arms,
cannot resolve the throughput half.

**The calls** (2026-09-22): `nudge_mode` causal and `text_grids` off
change; the second take keeps its presentation apart by carrying the digit
rows the first now leaves out. The audit line stays on and the reply cap at 24,576: both turn tokens
into score, which a later throughput gain makes cheaper. `prompt_budget` stays
at 48,640, the trainer's figure.

## Duck sweeps 8 and 9 — the audit turn, the file tools, the sampling and the checkpoint

Two randomised sweeps, 2026-10-02 to 2026-10-05, on the same 105 tasks as
sweeps 5–7, a million tokens and a twelve-hour wall a game, the audit turn on
in every arm and read by splitting each game at its `audit` event:

- **Sweep 8** — 13 B200 jobs, 140 arms: the branch's features, `top_p`, the
  answer's own temperature, and five engine options per job.
- **Sweep 9** — 42 jobs, 420 arms (384 read, the play's reasoning at 0.8 or
  below): five temperatures, a fourth checkpoint, intermediate summaries and
  the audit's confidence question.
- **An eval run** — `arc2-eval` × 10 at one point of sweep 9's design, one
  take a play (job 2894690), and a simulation of Kaggle's four L4s over its
  plays (`take_gap/audit_arms.py` in `arc2-experiments`).

Every reading, the data and how to regenerate it are in the report on duck
sweeps 8 and 9, which is not released.

**The objective.** The calls are made on the **throughput-adjusted score**:
the score less what the arm's tokens cost in score on Kaggle, against the mean
arm, `score − 100 × 0.08 / 0.5 × ln(effective tokens / their mean)`. An arm's
generated tokens are priced on the four L4s with drafting
(`kaggle_scheduler.throughput.KAGGLE_4XL4_MTP4`, fitted on the twelve-hour
demo run), heavier where its context and uncached input are slower to serve,
lighter for AutoRound's experts (9% faster on the box) and the fp8 indexer
cache (6%). 0.08 is what 1% of the box's throughput is worth at the deadline,
in Kaggle points on `arc2-eval` (0.074–0.082 by ±5–20% on the simulation);
0.5 turns a sweep point into one, since the sweep's eval tasks hold half of
`arc2-eval`'s outputs and the other half score 98.9. The raw score breaks a
tie, and judgment calls where neither separates (2026-10-06). Draft
acceptance is read per job and not priced.

**Defaults changed**, in sweep points of sweep 9's joint fit (HC3) against
the value that shipped before, unless another source is named:

| default | reading |
|---|---|
| `audit_turn` off → on | eval run, Kaggle score: +2.99 on one take for 20.8k tokens (56.7k → 77.5k), +2.01 on two takes; simulated at 12 h, the audit on take 1 adds +0.58 (se 0.04) to two takes. Sweep 9: +5.69 an arm on the hard tasks |
| `audit_handicap_tokens` 10M on take 2 | simulated at 12 h: +0.07 (se 0.03) over the audit on take 1 only, +0.16 (se 0.02) over both audits undeferred, which trail take 1 only by 0.09 |
| the checkpoint: ullm's e307 → `e307_arc2` with its drafter (`presets`) | +1.24 [+0.36, +2.11] (p = 0.006), +1.32 adjusted, tokens level (−0.5k, n.s.), over seven jobs |
| the indexer key cache bf16 → fp8 (`presets`) | −0.10 on the score (n.s.); +5.9% generated and +7.1% pool on the four L4s' traffic replay; +0.61 adjusted (p = 0.06) |
| `temperature` 0.6 → 0.8, every phase | the play's reasoning +0.24 (p = 0.06) for 1.6k fewer effective tokens, +0.50 adjusted (p < 0.001); flat to 0.9, −3.4 at 0.9–1.0, −9.5 and worse above 1.0. The other four phases (the play's answer, the audit's reasoning and answer, summaries) within ±0.25 |
| `top_p` 0.889 → 1.0 | sweep 8, 1.0 against 0.889: +0.15 [−1.12, +1.42] (p = 0.81); drafting and training on the duck's own play are easier against the plain distribution |
| `generalization_audit_line` on → off | −0.23 (n.s.) for 5.6k fewer effective tokens, +0.71 adjusted (p = 0.03); with the line on the audit gains 1.11 less (p < 0.001): the play does part of the audit's work |
| `file_tools` none → full | +1.27 (p < 0.001) for 8.7k more effective tokens, −0.19 adjusted (n.s.): a tie the score breaks. Sweep 8 +0.27 (n.s.) |
| `add_files_and_repl_to_summary` off → on | +0.53 (p = 0.08), +0.60 adjusted (p = 0.06); sweep 8 +0.50 (n.s.). Judgment |
| `thorough_summary` off → on | +0.04, 1.6k fewer effective tokens (p = 0.01), +0.31 adjusted (n.s.); sweep 8 −0.30 (n.s.). Judgment |
| `show_changed_answers` off → on | −0.52 (p = 0.09), 2.2k fewer effective tokens, −0.15 adjusted (n.s.); sweep 8 −0.31 (n.s.). Judgment: the smoke's model used the images as its last check, and once found a bug in its own code from one |
| `audit_groundwork` off → on | +0.41, +0.46 adjusted (n.s.); sweep 8 +0.60 (n.s.). Judgment |
| `audit_confidence_line` (new) on | on the audit's gain, the one outcome it can act through: +0.07 [−0.72, +0.85] (p = 0.86). Its +1.19 on the score sits before the audit, where the question is not asked: a batch difference, the four jobs that played without it |
| `malformed_nudges` 0 → 1, `malformed_retries` 5 → 3 | no score or token effect; sweep 9's games crashed 14 of 21,735 without the nudge, 1 of 22,365 with it. Over sweeps 6–8 a draw after four refusals in a row never succeeded |
| `max_tokens_per_game` 150k → 500k on the B200 cell | the eval run: 8 of 1,200 games went past 500k, and what they spent past it was 1.4% of the run's tokens |

Together, as the additive fit adds the ten in-fit changes (the confidence line
left out): +2.75 [+0.86, +4.65] on the score, +4.11 adjusted, 8.1k fewer
effective tokens a game, against `main`'s settings with the audit already on.

**Defaults kept**, each with a reading: `prompt_budget` 48,640 (over 40–100k,
−0.05 a 10k, n.s.; 3.6k fewer generated tokens a 10k, 0.8k effective);
`intermediate_summary_prompt_lengths` empty (−1.72, p < 0.001, almost all
before the audit, for 8k fewer effective tokens: a tie adjusted); exact draft
acceptance (relaxed: +30k effective tokens a game for no score, sweep 8);
`audit_profile` and `summary_profile` unset, since every phase samples at the
play's 0.8. Sweep 8's other engine options against e307: FP8 dense GEMMs
−3.5 (p < 0.001), QAD −2.4 (p = 0.003); the QAD expert grafts level on the
score and about −2 adjusted, their games 11–17k tokens longer; AutoRound
+1.06 (p = 0.009) in sweep 9 for 5.4k more effective tokens, a tie adjusted.

**What the features do in play**, from the branch's smokes (B200, `slurm-b200`
at 150k a game, 2026-10-02):
- **The audit**, 59 audited games (job 2862070): it answers the questions in
  the kernel; every win came through attempt 2, which it filled distinctly in
  65 of 103 test outputs against 46 before; it never moved attempt 1 between
  right and wrong. Its misses check a narrower alternative than the doubt
  they named.
- **The file tools** (jobs 2862205, 2862272): `create_file` in 29 of 32 games
  and in all 60, almost always `candidate.py`; 7 of 131 and 6 of 245 file-tool
  calls failed, mostly a `str_replace` without its `path`. Exposed,
  `search_files` was never called and `undo_edit` once.
- **The listing** (job 2862272): 90 summaries, each closed by it, the
  candidate shown in the 50 whose request said so, none of those restating
  the code.
- **The changed answers** (job 2862525): 86 messages in 58 games, each straight
  after the call that changed the answer; the model's own looks at the answer
  fell (`show` naming `test_out` in 19 calls against 33).

**The second take** (2026-10-06) plays without the file tools and at
0.5. Of sweep 9's knobs, those two and the checkpoint, which both takes share,
were the ones whose effect differed from task to task more than chance (each
knob's levels permuted over arms, or jobs: the play's reasoning temperature
p = 0.004, the file tools p = 0.003, the checkpoint p < 0.001; the placebo
p = 0.27). Read as a second take's change alone, neither moved pass@2
measurably: −0.04 points with take 2 the cooler, −0.30 with it lacking the
file tools (arc2-experiments `duck_sweep9/which_tasks.py`).

**Side readings.** Most wrong answers are close: of the eval run's outputs
wrong on both attempts, 95% have an attempt of the answer's shape, the closer
one a median 14 cells (3.9%) wrong, though about 60% get more than a quarter
of the cells the task changes wrong. The audit's confidence is a weak signal:
across tasks it ranks with the win rate (Spearman 0.56) in a compressed range,
about 72 on the hardest and 84 on those always won, and within a task it puts a
won game above a lost one 59% of the time.

**What the sweeps did not read.** The duck as it ships has not played: the
additive fit is a prediction, and all five temperatures at 0.8 is a corner no
arm visited (14 arms with all five at 0.4 or more, none at 0.6). The 27B
columns take the same defaults, unmeasured there. Draft acceptance falls 0.04
tokens a draft per +0.1 of the play's reasoning temperature, about 2% of a
decode step's tokens from 0.6 to 0.8, which the adjusted score leaves out. The
Kaggle figures are a simulation through a box model fitted on one Kaggle run.

## `zero_recurrent`: the option is in, the default is off

Upstream made the accident deliberate: `SamplingParams.zero_recurrent_state`
on the fork zeroes a request's recurrent (GDN) state at an absolute prompt
position while its attention KV stays intact, and `DuckAgent.zero_recurrent`
asks for it on every call. Upstream's readings are on ARC-AGI-3 (its
`evidence.md`, "zero_recurrent, and where 1568 comes from"): blank on
16.09 and 17.58 against a control of 10.48, at the position its shared
system prompt spans. **Nothing here is measured.** The default is off —
the regression from losing the 0.25.1 accident is accepted for now — because
the position is an open question for this prompt layout: the shared prefix
is shorter than upstream's, so 1568 very likely falls inside the task prompt
rather than the system prompt, a different treatment from the one scored.
Where the accident actually blanked on THESE prompts, on 0.25.1 with the
connector, was never measured either. Both are roadmap items; until one is
answered, turning the option on is a guess.

**What an uncached input token costs is assumed, not measured.** The sweeper
now charges uncached prompt tokens into an arm's budget at 1/9.4 of a
generated token (`sweeper.collect.UNCACHED_INPUT_EXCHANGE_RATE`). The 9.4 is
upstream's, measured on Kaggle's four L4s by sweeping only prefix-cache
coverage at fixed request count, input size and output length; it is an
upper bound on the rate, so input is if anything under-charged. Taken here
on the judgement that the deployments are alike enough; what would replace
it is the same throughput test run on the B200 and on Kaggle with this
engine, and the number to re-derive from it is the 45k default budget,
which is 40k × (1 + 1/9.4) for the ~1:1 input ratio these lanes run at.

## The presentation knobs the sweep could not separate

Recorded because they moved anyway, on judgement, and a later reading should
know it was not measurement that moved them: `images="task+show"` (+0.71,
p = 0.40), `image_layout="separate"` (+0.67, p = 0.53, where the *raw* gap of
1.34 does not survive adjustment), `image_gridlines=True` (p = 0.30),
`ascii_row_numbers=True` (p = 0.75, adopted so ASCII rows and the kernel's
numpy arrays index from 0 alike), `brevity_line=False` (p = 0.56), and
`python_timeout_s=90` (p = 0.74). `image_scale` stays 4 as the cheapest of
three levels the sweep found equal (p = 0.55).

`temperature` moved to 0.6 on an ARC3 result. ARC2 is neutral on it: flat
across the 0.4–0.8 that was drawn (p = 0.73), and the previous value sat
outside that range.

Sweeps 5–7 separated two of these (above): the lattice and the legend, both
off now. The rest stayed where sweep 4 left them, on three more readings each.

## `max_tokens_per_game`: the per-game cap

The cap (65k generated tokens) is sized so an arm's *mean* spend can reach
the budget a sweep reads it at (`sweeper.DEFAULT_BUDGET_TOKENS`, 45k
charged), with margin for the draws past it, while not paying for the tail
the reading discards.

The measurement behind it was made against the earlier 40k reading: over
`arc2-eval`, eight design-drawn arms spent 50–64k a game under a 100k cap,
and the per-game ceiling at which an arm's mean landed on 40k ran 44–55k.
65k carries that margin across to the 45k reading by scaling, not by
re-measuring. The margin is not uniform across arms: under this cap on
`arc2-eval` the register A/B's concise arm averaged 40.5k a game against
the verbose arm's 45.4k, so one ceiling does not sit the same distance
above both. Re-deriving the cap at 45k is open (`roadmap.md`).

**Flash-Next on the B200 plays to 150k, the score curve's knee.** It spends
more per game than the 27B the cap was read on: across three B200
qualification runs over `arc2-eval` x 2, 52 to 55 of 240 games generated past
65k and 16 to 22 reached 130k, so the default would have cut a fifth of them;
those runs played at 130k. The figure is now traded rather than measured for
score, from the twelve-hour Kaggle run of this engine rescored at caps (the
same `arc2-eval` games this column plays, every submission scored at the
tokens it was made at): mean 68.3 at 80k, 72.9 at 130k, 74.4 at 160k, 75.6 at
200k and 76.7 uncapped, against a median game of 24.7k tokens. Past the knee
each point costs several times a median game, and on a shared cluster it
costs the experiment its turnaround. On Kaggle the cell states no cap: the
score is what counts there and the ranking handles long games, two takes
included (`settings.md`, P-0V).

## The reply cap

`reply_cap_tokens=24576` limits a single generation and supplies the context
budget's reply reserve; the window moved with it (`context_window_tokens`
73,728) so the prompt budget stays at the 48,640 the sweeps read arms at.

Sweep 7 drew the cap from 2k to 32k over 374 arms with a budget above 30k
(`reply_cap_shape.ipynb` in `arc2-experiments`). Below a knee at
10–11k (bootstrap 10–15k) the score falls about 2.5 points per 1k of cap, and
the throughput a short cap buys does not pay for it: at a cap of 3k, 37% of
replies would be cut, at 12k 7.7%, at 20k 3.8%, and the replies that run long
are the ones doing the work. On 12–32k the score rises 0.13 points per 1k
(p = 0.001), nearly all of it below the 22k midpoint (+0.18 per 1k there,
p = 0.04; +0.08 above, with an interval through zero), while tasks an hour
fall 0.11–0.18 per 1k on either side. The cap sits just past where the
evidence ends. The earlier argument for 16k — a round that must stay under
prune's compaction target — no longer applies: the shipped columns summarize.

## The narrowed context window

`context_window_tokens` 65536 and `compact_to_tokens` 22000 come from an A/B
on the four-L4 box against the values they replace (90112 and 33925), each
120 `arc2-eval` tasks over two passes in twelve hours at the same pins:
`arc2-eval120-sched` v1 and `arc2-eval120-w65536-t22000` v1.

**Score is flat and the run is faster.** Mean score 55.07 against 52.92 —
the narrow arm lower, but paired over 240 runs that is p=0.53, and p=0.55
on per-task means, so the pair resolves nothing about score either way.
Generation rose from 234 to 246 tokens/s, 9,712,923 tokens to 10,453,769.

**The spend of a cancelled game is far more even**, which is what the
narrower sawtooth was for: a game yields its slot at a compaction, so the
26,640-token sawtooth rebalances the ranking where the 39,291-token one let
a game sit. Over the games the wall clock cut, the tenth-to-ninetieth
percentile band narrows from 46,548–68,019 to 61,638–68,350 — 3.2x — and
the coefficient of variation from 0.169 to 0.064. Nothing now falls below
40k, where twelve games did.

**Residency held, uncapped.** Every started game stays resident, running
or parked, with no ceiling on how many; the 90,112 arm carried 240 entries
on 40 slots that way for its twelve hours at the flat ~55 MB per child
measured below ("The Kaggle KV budget"), so the ceiling upstream keeps as
an open item is not needed at this shape.

**Two costs.** Uncached input per generated token rose from 1.28 to 1.67:
the extra turnover is paid for in resumes, each re-prefilling its context,
and prompt does not grow one-for-one with generation, so games with bulky
tool output trim more often than the ratio alone predicts. And more games
reached the 70k cap, 18 against 29, because generation got faster — 90 runs
did not finish against 71.

What this does NOT settle is the window itself. 48640 of prompt budget is
14% below the lowest arm the sweep has drawn (56832) and 22000 is below the
lowest target it has drawn (22733), so both sit outside the measured
region, and the pair reads one point rather than a curve. The engine cost
model fitted to the first of these runs puts the context term at about half
of engine time and predicts 8.9% per generated token against the 5%
measured, the gap being the prefill rise above. Sweeps 5–7 then drew the
budget from 15k to 70k and read it against the throughput ("Duck sweeps 5, 6
and 7", above): the 48,640 sits on the flat part of its shape.

## The engine's scheduler shape and the host tier — upstream's reading

**Nothing in this section is measured here.** The per-step token budget
(8192, P-00), `deployment.ENGINE_ADMISSION_WATERMARK` (0.10), the victim host
KV tier on by default and the concurrencies behind it (80 on SLURM, 40 on
Kaggle, P-EV — `settings.md`) are arc3-platypus's as of `14aed52`; the
measurements are in ullm's `kv_offload.md`, "How it was tested", and this
is the reading that put them where they are:

- **`max_num_batched_tokens=8192`, pinned, against the flat
  `max_num_seqs=128` this was read at** (the cap now follows each column's
  concurrency — `settings.md`, "Derived, not stated"). At the
  in-process figure of 2048 the KV-offload traces showed what the budget
  does against the model's 1,568-token recurrent block: one block-sized
  prefill per step and no admission behind it while a prefill runs, so
  any burst of cold prefills drags the whole engine to chunk length
  (ullm's kv_offload.md, "Prefill is the throughput"). Five blocks per
  step is the server's figure; its activation workspace comes out of the
  KV pool's headroom, so the profiled pool is the number to read after
  changing it.
- **`watermark` set** (`deployment.ENGINE_ADMISSION_WATERMARK`), vLLM's own being
  zero: the reserve that makes the engine queue admissions rather than
  preempt running requests once the live set outgrows the pool. The
  reasoning and the sizing are in ullm's `kv_offload.md`, "The regime
  the design assumes": the reserve is also the free queue's length, and
  so the residence a freed checkpoint has on the GPU.
- **The tier, measured against a tier-less control on a pool that fits.**
  At a pool holding 473 blocks against a 630–890 block live set, a 50 GiB
  victim tier served the same cached fraction of prompt tokens as a pool
  twice its size once filled (93–96% per ten-minute window against the
  control's 92–94%).

What this repo has that bears on it. **The B200 pool at this shape is
4,104,192 tokens — 62.62 windows of the 65,536 now in force** (4,232,448
and 46.97 at the 90,112 window before it) at `gpu_memory_utilization=0.98`
with the 8192 step budget (the b200 test tier's engine logs,
`.gpu-test-logs/b200/`, gate of 2026-09-09), so the SLURM column's 80 games
run 17 past what the pool holds at a full window, which is
the regime the tier is for rather than one where it idles. And both arms of
the register A/B (`register_ab_eval120`, 80 then 40 concurrent, no tier,
the 2048 budget, the zero watermark) showed heavy prefix-cache misses,
heavier at 80 — the symptom upstream's reading predicts, never diagnosed;
the cached fraction per window on the first run at these defaults is the
comparison that settles whether the reading transfers (`roadmap.md`,
Watch items). The four-L4 pool is unread.

## The Kaggle KV budget

Measured on the four-L4 box (`arc2-eval120-duck` kernel v2, 2026-09-07,
`tp=4`, `gpu_memory_utilization=0.85`, the victim tier at 60 GiB), from
the engine log and the connector's own `kv_offload_trace.jsonl`:

- **GPU pool: 818 blocks, 1,151,744 tokens, 39.2 GiB engine-wide.** vLLM's
  per-worker line reads 9.79 GiB; the four workers each hold a quarter of
  every block, so the memory sums across cards and the token capacity
  does not: **12.78 windows of the 90,112 then in force**, once. Both
  figures are the engine's own, and they put a block at 1,408 tokens —
  not the 1,568-token recurrent block the step budget is sized to, which
  is a different quantity. The B200 pool at the same shape was 4,232,448
  tokens, 46.97 such windows, at that window; at the 65,536 now in force
  it reads 4,104,192 tokens, 62.62 windows (`.gpu-test-logs/b200/`, gate
  of 2026-09-09).
- **Host tier: 1,253 slots of 51,380,224 bytes = 60 GiB total**, 1.53×
  the pool. `bytes_per_slot` is four times the per-worker block, so the
  tier's size is engine-wide, not per worker.
- **A context costs ~25 blocks at its measured mean.** Over the register
  A/B's inject arm (908 requests, 40 games) the context ran p10 9.4k, p50
  31.4k, mean 34.3k, p90 64.3k, max 72.1k tokens: ~25 blocks of 1,408 at
  the mean, ~46 at p90. (The engine's "maximum concurrency" line is plain
  pool ÷ window and says nothing about per-request state, so any
  recurrent-state overhead is unmeasured and not counted here — the counts
  below are upper bounds.) So the pool holds **roughly 33 mean-sized
  contexts, the tier ~51, the pair ~85** (18 / 27 / 45 at p90). At 40
  concurrent the running set alone wants ~1,000 blocks against the
  818-block pool: the tier is covering the running overflow before any
  parked game gets a byte.
- **There is no more pool to be had from `gpu_memory_utilization`.**
  CUDA-graph capture takes 1.79 GiB per card *after* the pool is sized;
  0.85 leaves ~3.6 GB per card for it and 0.95 would not, which is the
  startup failure the preset comment describes, now seen on 0.27.1.
  Trimming `cudagraph_capture_sizes` is the one untried lever, worth
  perhaps a tenth of the pool.
- **Memory is not the constraint.** Per-game children hold a flat ~55 MB
  across a six-hour SLURM run (median 53 → 60 MB, two processes a game);
  the box at 40 resident sat at 129 of 173 GB, of which the 60 GiB tier
  is `/dev/shm` — filed as page cache and not reclaimable, which the
  memory line now counts (`platypus.memory`, `shmem`).

## The Flash-Next cells on the traffic replay

The duck's P-EV values for Flash-Next -- the 3,300-token step, 60 slots on
the four L4s, 200 on the B200 -- rest on the traffic replay (ullm's
`traffic_benchmark.md`), and the readings are the addendum's
(`flash_optimization_arc2.md`, "The four-L4 cell on the traffic replay"):
the bf16 state +24% generated at 70 lanes, 3,300 reading the same as
3,236, the unit -3.9%, 0.96 out of memory, the Triton decode +1.5% and not
taken. 60 is the float32 run's 50 scaled by the pool the state gains; the
B200's 200 is its tier-bound 150 scaled the same way, a judgement until a
long run reads it.

## The reasoning register, measured here

`DuckAgent.inject_text_on_first_prompt` serves `We need solve` after the
generation prompt of a game's first call and `inject_text_on_later_prompts`
serves `We need` on every call after. Both ship ON, and this is the reading
that decided them. The opening is the three-token form, one past the
two-word threshold in the length sweep below (0.99 against the full
sentence's 1.00): it is what the A/B measured, and it is what ships here.

**A/B over `arc2-eval`, 120 tasks × 2 passes per arm**, one B200 each, the
arms identical but for the injection (`zero_recurrent=False` in both):

| arm | score at a matched budget | at its own endpoint | tokens/game |
|---|---|---|---|
| control (no injection) | 36.25 ± 2.22 | 43.33 | 45,352 |
| **injected** | **51.67 ± 1.25** | 51.67 | 40,547 |

**+15.4 points at a common budget of 40,547 generated tokens per game**
(`register_ab_eval120/compare`, official ARC scorer, SEM across the two
passes). Read at a matched budget rather than at each arm's endpoint
because the arms did not spend alike — and the direction is the helpful
one: the injected arm scored higher while spending **less**, so none of the
gap is bought with tokens. Its score-vs-tokens curve is above the control's
from roughly 15k tokens per game onward.

Two passes per arm is a thin basis for the ±, and the two arms ran at
different harness concurrency (80 against 40) after the control's cache
behaviour was seen to differ — concurrency does not enter a lane's score,
but it means the runs are not interchangeable for throughput readings.

**The register itself moved, which is the mechanism doing the work.** The
classifier over reasoning text (`ullm.reasoning_register`, served text
stripped before scoring) on the 10-game demo pair: control **0.107** with
0/10 lanes in the telegraphic register, injected **0.947** with 10/10. The
text says it plainly — control opens *"Let me analyze this puzzle."*, the
injected arm continues into *"ARC puzzle. Need inspect arrays, infer
transformation."*

What is NOT established here: that these words are the best ones, that the
schedule is the right one, or the cost of the constraint — every response
is made to open alike, and no arm here serves a VERBOSE opener (`Let me`)
on the same schedule, which is the control that separates "the register
held" from "every response opens alike". The opening-block classifier does
not transfer to ARC2 and should not be used on it: it rates control
openings that literally begin *"Let's analyze"* at 0.6–0.73, where it was
fitted on ARC-AGI-3 openings for which `let's` is a CONTROL marker.

## The reasoning register — upstream's reading

**Nothing in this section is measured here.** It is arc3-platypus's
`evidence.md` as of `82175da`, kept because it is where the mechanism and
the shipped words come from, and because the ARC2 experiment above was
designed against it. Every run, game count and score below is ARC-AGI-3's;
the duck there plays long games with 6–7 compactions each, where an ARC2
task at the 45k budget rarely compacts at all — so the decay-across-
compaction result, which is most of what follows, is not something the
reading above either confirms or contradicts.

### What the blank was actually worth: the register

The position sweep (`zr_sweep_20260902_181704` and its follow-ups,
notebook in arc3-experiments `zr_position_sweep/`) was run to ask
which position scores best. It answered a different question. Position
does not order the scores; what orders them is how many lanes end up
writing in a particular reasoning register — a telegraphic voice, "We
need solve game. Need inspect current frame.", against the ruminating
default one.

Read by a fitted classifier over closed-class words alone
(`ullm.reasoning_register`, AUC 0.991 on held-out games; a second fit for
the opening block, because `let's` is a BLANKED marker over a whole game
and a CONTROL marker at the opening):

- Across runs, score tracks the FRACTION of lanes that open in the
  telegraphic register, r = +0.88. Within a cell, a lane in it is worth
  +9.8 points (p = 1.4e-21).
- The register is decided at the opening block, per lane, and then holds:
  a lane that starts in it stays in it.
- What decides it is the sampler. One game, 100 concurrent lanes,
  identical prompts, a cache salt per lane so no lane can read another's
  blocks (`zr_onegame_20260904`): the lanes split 44/56 between the two
  registers rather than landing together. The blank does not create the
  register; it raises the rate at which lanes fall into it.

So the blank is one route to the register, not the thing itself, and a
position that perturbs the state without moving the register is worth
nothing.

**Reaching it from the front instead.** Asking does not work: the
register offered as an example in the system prompt, and again as the
last line of the user prompt with its own tokens immediately before the
generation prompt, produced it in 0 of 100 lanes each. Serving it does.
Text served as the first response's own opening ids
(`DuckAgent.inject_text_on_first_prompt`) puts a lane in the register in
100 of 100. This is not asking: they are the model's own tokens in its
own voice, the same continuation the sampler already makes unprompted on
some fraction of lanes.

Two words are enough (`zr_inject_length_20260904`, one game, 100 lanes
each, opening scored with the injected text stripped off):

| served | tokens | opening register |
|---|---|---|
| `We` | 1 | 0.10 |
| `We need` | 2 | 0.96 |
| `We need solve` | 3 | 0.99 |
| `We need solve game.` | 5 | 1.00 |
| `We need solve game. Need inspect current frame.` | 10 | 1.00 |

`We need solve game.` is what ships upstream, one sentence past the
threshold; here the three-token row ships (above).

**Serving it once does not hold a game, and compaction is why.** The
20-minute single-game runs above read the opening and the tail of a short
game, and reported the register holding. Over a full game it does not.
Scoring every reasoning block and bucketing by how many compactions
preceded it (20 finished games per arm, median first compaction 22% in,
6-7 compactions per game):

| compactions elapsed | injection only | blank on every call |
|---|---|---|
| 0 | 0.947 | 0.955 |
| 1 | 0.891 | 0.933 |
| 2 | 0.854 | 0.886 |
| 3 | 0.635 | 0.879 |
| 4 | 0.464 | 0.874 |
| 5 | 0.263 | 0.815 |
| 6 | 0.241 | 0.676 |
| 7 | 0.224 | 0.481 |

By the fifth compaction 82% of the injected arm's blocks are back in the
control register against 9% of the blanked arm's.

Why the blank holds longer is NOT established, and the obvious story --
that it is re-asked on every call and so re-creates the condition -- does
not survive its own numbers. The blank's measured effect is at a tail of
~1500 tokens (the game-start stratum; the accident's post-compaction
stratum ran 12k-32k). In these runs the tail past position 1568 is a
median of 26k tokens BEFORE the first compaction and 42k after, with only
5% of pre-compaction calls under 3k. The blank is therefore inert on
nearly every call of both arms, and is as much an opening-only
intervention as the injection is. What separates the two arms after the
third compaction is unexplained.

**Repetition is what sustains it, not the words.** Serving two tokens
(`We need`, `DuckAgent.inject_text_on_later_prompts`) on every call after
the opening, scored with the injected tokens stripped off first
(`control_inject04w_later02w`, 51 lanes in flight against the same
opening without it):

| compactions elapsed | opening only | opening + every call |
|---|---|---|
| 0 | 0.948 | 0.953 |
| 2 | 0.834 | 0.919 |
| 3 | 0.648 | 0.908 |
| 5 | 0.297 | 0.860 |
| 7 | 0.222 | 0.769 |

Verbose-block share at the fifth compaction: 78% against 4%. That is a
better hold than the blank manages, on a stock engine, for two tokens a
call. It still declines, slowly, and nothing here says why.

What this does NOT measure is the cost of the constraint. Every response
is made to open on the same two tokens, overriding the varied openers a
concise lane writes for itself ("Let's try", "No. Maybe the", "Timed out
after"). Forcing a VERBOSE opener (`Let me`) on the same schedule is the
control that separates the constraint from the register, and has not been
run.

### What the register is worth: the score

Everything above measures the register. This is what it scores, on the
official 25 at 4 passes each (`zr_inject_full_20260904`, against
`shared/baseline_main_38e4852e`; comparison `compare_register_arms`).
Official ARC-AGI3 score per game, each arm scored where it ended:

| arm | score | SEM | vs baseline |
|---|---|---|---|
| main as it shipped: blank at 1568 on every call, no injection | 16.09 | 0.74 | — |
| **open one sentence + `We need` every call** | **26.26** | 1.51 | **p = 0.003** |
| open one sentence only | 18.90 | 1.89 | p = 0.24 |
| open three sentences only | 17.61 | 1.65 | p = 0.45 |
| open one sentence + blank at 1568 | 18.36 | 2.63 | p = 0.46 |

p from Welch over per-pass means; the paired t and paired permutation
tests over per-game differences agree on the shipped arm (0.009, 0.003)
and are the ones to read for the rest, being paired on game.

The baseline is main's own shipped configuration at the time, which was
the BLANKED one: its saved solver carries `zero_recurrent=True` at 1568,
and the run is `zr_ingame_zr1568_27f70ae` at revision 38e4852e — the same
16.09 the table above reports for blank-on. So the 10.2 points are over
the blanked main. The unblanked control in that earlier table (10.48) is
a different revision and not an arm here; against it the gap is wider.

Three readings decide the two defaults, and each matches the register
evidence above:

- **The schedule is the effect.** Opening once is worth 2.8 points and
  does not separate from baseline (p = 0.24); adding `We need` to every
  call is worth 10.2 and does (p = 0.003). The two injected arms differ
  from each other at p = 0.024. That is the compaction result scored: an
  opening left in history stops being served once compaction drops it.
- **More words at the opening buy nothing.** Three sentences (17.61)
  scores no better than one (18.90) — the threshold really is two words,
  and the rest of the sentence is not doing work.
- **The blank adds nothing on top of the opening.** One sentence with the
  blank (18.36) scores as one sentence without it (18.90). Since the
  tokens reach the register on a stock engine, the fork's blank is
  redundant where the injection already went, which is why
  `zero_recurrent` ships off.

It is not bought with tokens. The shipped arm spends more per game, and
still leads at a matched budget: 25.31 against the same 16.09 on the
budget-matched reading, and its score-vs-tokens curve is above every
other arm from roughly 100k tokens per game onward. The blunter scorers
move the same way — games fully beaten 0.10 → 0.22, levels completed
2.89 → 3.57, flat-weight score 22.26 → 32.97.

What this does NOT establish is the cost of the constraint, which the
paragraph above states: no arm here serves a VERBOSE opener on the same
every-call schedule, so "the register held" and "every response opens
alike" are not yet separated.
