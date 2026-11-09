# Seal evidence

*Register: living.*

The measurements a current default rests on, and what each does and does not
settle. Where a value lives is `settings.md`'s; the run directories named
here are under `/data/shared/research/arc_3_results/shared/`, and the sweeps'
designs and readings are in arc3-experiments (`seal_sweep1/`,
`seal_sweep2/`). Every historical check reads one run unless it says otherwise:
`distill/seal_flashnext_38x4_20261002_113231`, the seal at its defaults of
2026-10-02, 38 games x 4 passes, with `intermediate_states.pkl` and every
game's span stream ("the reference run").

A seal run's score is noisy: one pass over the 38 games lands outside the
range of four passes for about two games in five, and two identical
Flash-Next runs once scored 6.76 and 9.00. A smoke says "plays, does not
collapse"; a ranking needs a sweep.

## The defaults of 2026-10-12

Chosen for an interim Kaggle submission, from seal sweep 2:

- **The window, 181,184 tokens** (a prompt budget of 147,904). With the side
  turns on, the summary after one needs the window, the side turn's 40,000
  extra, the reply reserve and the summary's room under the engine's native
  262,144; 147,904 is the largest budget that fits
  (`_require_summary_fits`), chosen over trimming the extras.
- **A share of the duck's slots on every column** (`WINDOW_SLOT_SHARE`), for
  the longer contexts.
- **The replay reminder on; the level reflection and the strategy audit
  concise, the audit at 300,000 level tokens.** The B200's 175k per-level cap
  means the audit does not fire there; Kaggle has no cap.
- Unchanged and already the defaults: the optional world model, segmentation,
  the settled delta, the search, no diff image.

These are a choice for one submission, not a sweep's ranked reading; the
readings below are what each piece cost and did when it was first run.

## The defaults of 2026-10-28

Chosen for the final ARC3 submission, from seal sweep 4's drawn
arms, each read on the box replay of its own checkpoint's Kaggle card at the
deadline: the tool-output cap at the top of its drawn range, the board in
letters, the pixel tolerance offered from none, each move's changes listed,
the chronological restart, and the level reflection verbose. Each was picked
as the better-reading value, some at the edge of significance. Seal sweep 5
then played them together against the defaults before them, on joint205:
they reach each score sooner, and the two meet by the deadline. The readings
and the decision log are arc3-experiments' `seal_sweep4/final_config.md`.

## The world model: optional

Seal sweep 1's long runs (`seal_sweep1/`, 2026-09-30): the six mandatory arms
read at the bottom on the sweep score (55.9, against 60.6 for no world model
and 65.4 for optional) and on the official 25 at equal weight (64.5, against
72.4 and 72.1), and at half the others at the semifast budget. No world model
and optional do not separate on the official 25 (-0.2 at the cut, p 0.91);
optional's edge on the sweep score is the hard games' (+9.5 at the cut, p
0.016), which the sweep score weights up. So `optional` is the default, and
seal sweep 1 dropped `mandatory` from its arms from then on.

## Animations: marked, described on request

The trio runs (`seal_trio_marks`: sp80, tn36, sb26 x 30 passes, no world
model, the settled delta on): mean score unmarked 23.96, marks alone 28.25,
marks with `available` 33.74, marks with `auto` 26.71, `available` ahead of
unmarked and of `auto` on every game. Without a world model the automatic
description clicked less on sp80's level 2 (19 of 30 passes against 27) and
got past it less (14 against 20), clicking being what every pass past level 2
did. A wide sweep may still move it.

The automatic blocks ran 1,600-2,300 characters each on sp80, under the
5,000-character limit; requested descriptions took a median 0.55 s and at
most 13 s, and none of 544 cells calling `describe_animation` over four runs
was cut by the output cap. That is why a requested description is exempt from
the cap and a shorter python timeout would not bound it.

## `winnable_line`

What the seal fails at, from 25 games x 8 passes (`tools/level_tokens.py`):

- **Tokens to beat a level:** median ~36k, p90 ~109k, against the 175k cap.
  Of attempts still unbeaten at a given spend, the share beaten in the next
  25k falls from ~36% to ~15% near the cap.
- **Restarting within the budget loses to continuing:** an attempt unbeaten
  at 100k beats the level in the remaining 75k half the time, a fresh start
  given those 75k about a third. A second full budget helps about as much as
  a bigger cap.
- **Earlier levels predict the next only weakly** (within-level rank
  correlation ~0.12 against a shuffled null), though a specific lesson shows:
  on sb26 L5 the passes that left L4 with the pipe rule won three of three.
- **Declared impossible, then idled**, the costliest failure: a wrong reading
  makes a level look provably unwinnable and the pass spends its budget on
  no-op actions, there being no way to end a game.

On sb26, 30 whole games per arm (`seal_winnable_ab/`), the line beat level 5
in 16 of 28 plays that reached it against 6 of 28 (Fisher p = 0.013); full-game
wins 13 against 4, mean score 38.3 against 19.8. In control every level-5
attempt reached the counting argument, and the 22 that lost spent a median 322
turns, 12 of them in an idle loop; in treatment 20 of 28 reached it and 8 of
those still won, and the 12 losses spent a median 64 turns, most of them
experiments. Winners do not quote the line: it delays giving up more than it
teaches the rule. On by default; across the games it is a sweep's to price.

## The side turns

Smokes of the 38 games, one pass, a 12-hour wall, the side turns `verbose`:

- **The level reflection** (`seal_sweep2/seal_reflection_38x1_20261007_152033`):
  157 reflections, one per level won, none cut, one missed win; median 2.2k
  tokens and 2 calls, at most 53k; 5.7% of the run's generated tokens. 26
  games kept a notes file, first written in a reflection in 25. The levels
  after a reflection are not cheaper: paired against the reference run, later
  levels took 1.31x the tokens with the reflection's own spend taken out
  (p 0.006 per level). What it may buy, levels beaten at all, one pass cannot
  price.
- **The strategy audit** (`seal_sweep2/seal_reflection_audit_38x1_20261007_175638`,
  at 75k then every 50k): 80 audits in 30 games, none cut, median 4.2k tokens,
  5.9% of the run's. Levels still unbeaten past 75k and beaten later: 31 of 54,
  against the reference run's 53% (p 0.65); one pass rules out only a large
  effect. **Why tokens:** on the reference run, of the attempts unbeaten at a
  25k bin's start, 38%, 33% and 32% were beaten within the first three bins,
  then 19%, 19%, 14% and 17%; the rate breaks at 75k.

## Off, and why

- **The plan argument** (`seal_sweep2/seal_kernel_38x1_20261007_172334`): 18%
  of python calls and 42% of acting batches carried one, in 29 of the 38
  games; read back in 3 calls and in no reasoning. A sweep knob, off by
  default.
- **The pixel tolerance's start, at none.** Offered from 0, before seal
  sweep 4 offered it by default, it was never set, and its suggestion showed
  29 times in 9 games with no uptake; started at 4
  (`seal_kernel_tol4_38x1_20261007_194522`) it acted in 6 games, and the run's
  lower mean is noise (+0.3 in those 6, -7.3 in the 32 it never touched).
- **An inert-move guard, not built.** On the reference run 85% of moves ran
  unchecked and 5.3% of those changed nothing on the board (12.0% inside a
  four-cell margin); the most optimistic rescore of a guard blocking them moves
  the mean from 49.34 to at most 49.79. `move_changes` shows the moves that
  did change something instead.
- **The replay shadow and the stuck-on-a-level stops** are platypus's, and off
  for reasons in the root `design.md`.

## The surface stays as it is

On the reference run's 19,888 python calls, 1,574 erred (7.9%), about 125 of
them on the surface: `.action` on a level's first turn, whose action is
`None` (59 calls in 22 games, against 727 calls in all 38 games that test it
first), a frame's or an action's attribute asked of a `Turn` (26, most
answered by the hint), and a handful under ten each. `Turn.step` was used right
in 220 calls in 30 games and `submit_action`'s result read for real keys in 361.
The one addition it bought is `Node.bbox`, which models had built by hand in 637
calls. Seal sweep 2's first round (23,819 tool results) moved only unclear
wording and refusals that name the game's actions.

The environment guard's cases were rare: 38 of 95,534 tool calls over three
runs, in two games.

## Robustness

The seal after its back-merge, ports and crash fixes
(`seal_sweep2/seal_robustness_38x1_20261006_171601`, one pass): no crash
against the reference run's one, 18 won and 20 given up in 2 h 18 m, mean 56.47
against the reference run's single-pass means of 43.0-53.5. Its two kernel
restarts were a model ending a loop with `raise SystemExit`, which now ends the
cell rather than the kernel (ullm), and a combinatorial search most likely at
the kernel's address-space limit.

`malformed_retries` is 3 and `malformed_nudges` 1, as ARC2's duck ships them:
ARC2's sweeps 6-8 never saw a draw succeed after four refusals.

## The engine columns

- **INT4 KV in, the FP8 recurrent state out**, on seal sweep 2's own play
  (2026-10-12; `packages/ullm/flash_optimization.md`, "INT4 KV cache"
  and "The recurrent state in bf16").
- **The B200's host tier at 64 GiB**, where the duck's is 96: seal sweep 2's
  main jobs at 100 and 140 slots asked within ~1,100 tier slots at p99, and
  below the old size only a ~0.1% deep tail is lost, alike at any size from
  64 GiB up.
- **No image cache; the renderer's workers instead** (2026-10-17).
  Flash-Next's preprocessed images are float32 (6.0 MiB for a 512 px board,
  6.8 for a labelled one), every request resends its context's, and the
  processor cache is FIFO, a hit not refreshing an image, so past its size it
  evicts images still in use and the hit rate falls and stays down. At vLLM's
  4 GiB seal sweep 2's round1 diff-image arm fell to about a fifth of hits
  within half an hour, at about 30 running games; at 16 GiB its main1 jobs,
  100-140 games each, held 98% for an hour, then fell to 31-47% as contexts
  filled with images. In seal sweep 4 (the experiments repo's
  `seal_sweep4/sweep_design.md`) 16 GiB at 120 games spent 223-266 minutes of
  17.5 hours below 90%, down to 13%, and 32 GiB at 100 games fell as low as
  6-20% hours in. Each miss is processed on the renderer's one worker before
  its request reaches the engine, about 0.7 s a request, so at 180 games
  requests waited a median 20-35 s in front of an engine with 33-46% of its
  KV in use and none waiting. Without the cache and with four workers, a job
  at 180 games kept its requests a median 1.0 s (p90 2.0 s) in front of the
  engine through its run, in 32 GiB less shared memory. Kaggle's boxes take
  two workers, with fewer CPUs and fewer games.
- **The per-level cap on the cluster is the Flash cell's 175k**: an earlier
  seal's 90k ended most of its games.
- **joint205's columns** (2026-10-28). Seal sweep 5 read joint205
  ahead of e307 on each one's own box at the Kaggle deadline, its faster card
  the lead (the experiments repo's `seal_sweep4/final_config.md`). Its fewer
  routed experts leave the KV pool about 11.8 GiB more card: 128.69 GiB against
  e307's 116.86 on the B200 (seal sweep 3's `checkpoints1`), and on the RTX
  52.83 against 40.95 in the box model. The B200's games: the duck's 100 was
  set on e307 with the FP8 cache and a bf16 indexer cache, 1,088 bytes a
  context token a layer, where the INT4 caches take 608, as the tier's
  parked threads do; 100 x 1,088/608 x 128.69/116.86, at the window share of
  0.85, is 167, not yet run. The RTX's 80 is a judgement call; the duck's 80, set on
  the INT4 caches already, at the window share and scaled with joint205's
  pool would be about 88. On Kaggle joint205's own part is a
  private model of 51.6 GB, and the 108 tensors of e307's four PLE-only
  shards, joint205's byte for byte, come from e307's model: the account then
  holds the two and e307's drafter.

## Smaller settings

- `evict_turns` 1, where the duck has 2: inert at the duck's window, and inert
  under summarize either way.
- `completion_cutoff_notice` on: a reply cap this size is hit only when a reply
  went wrong.
- `python_timeout_s` 180: a framediff search inside the model's call stops
  short of the call's limit, and the limit went up with framediff's own time
  limit so a requested description fits.
- The after-call harness cell's limit (`_HARNESS_CELL_TIMEOUT_S`): room for one
  automatic animation and one settled delta at framediff's limit each, and
  their overruns.
- `SEARCH_SLOTS` 12 machine-wide: the searches' memory on Kaggle.
