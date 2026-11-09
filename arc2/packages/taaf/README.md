# taaf

*Register: living.*

The Tufa ARC-AGI framework, adapted to ARC-AGI2. A `Benchmark` ties a
`Solver` to a list of `Game`s and plays them; `deploy` ships the same
benchmark to an inline process, the Tufa cluster, or Kaggle.

One game is one ARC task. The solver reads the task through `query()` and
calls `execute_action` with two attempts for every pair it must answer — as
many times as it likes. Each call replaces the standing answer, so a solver
that improves on its own work leaves the better one behind and a run cut off
part-way still submits the best it reached.

**The score is recorded once, at `finish_game`, on the standing answer, and
nothing the solver calls returns one.** Not on `execute_action`: that would
hand the solver per-pair correctness for an answer it can still replace,
which is a scoring oracle. `execute_action` returns nothing at all.

Whoever watches the run can still read a game in play.
`GameRun.current_score()` is the percentage of pairs right as the run stands,
whatever its state — 0 before any answer — and `GameRun.current_result()` the
per-pair result behind it, both by the finish's own rule
(`GameRun.result_against`). The score is `None` only where nothing can be
scored: a withheld answer key, or a run in play whose spec was not recorded
(`minimal_diagnostics`, or a `benchmark.json` loaded without its sidecar).

## Deliberate benchmark divergence

This fork provides opt-in `Benchmark.raise_solver_errors` and teardown that
preserves the primary solver failure and caller cancellation. A strict caller
must distinguish infrastructure failure from a scored episode, including
backend failures recorded on the played solver copy. Cleanup and game
finalization still need to run before the error reaches the caller; ordinary
benchmarks retain best-effort behavior by default.

This is an owned fork difference rather than an upstream TAAF prerequisite.
Upstream adoption and enabling strict errors in the ARC3 trainer are separate
follow-ups; this package does not assume either has happened.

## The data model

Ours end to end — no upstream type reaches this code.

| type | what it is |
|---|---|
| `Grid` | 2-D int8 array, ARC's ten colours, 1..30 per side |
| `Pair` | `question`, `answer`, and `given` — a demonstration or a pair to answer |
| `GameSpec` | a whole task: id plus pairs, answers included where known |
| `GameView` | what a solver may see: demonstrations in full, questions only |
| `Submission` | two attempt grids per pair to answer, in order |
| `GameResult` | per-pair correctness of one answer; `None` where the truth was withheld |

`given` is a per-pair flag rather than a train/test split, so designating a
different subset as demonstrations is a dataset transform rather than a code
change.

## Scoring

Each pair is scored independently and either attempt counts, which is the
competition's own rule. A task therefore scores in `[0, 100]`, and
`number_of_levels` is 1 — the level is beaten if *any* pair came back right.
That makes `GameRun.won` mean "got at least one test output right", which is
**not** ARC-AGI-2's task-solved criterion; the `solved` scorer carries that.

Which of the two official metrics a run reports is the game weighting, and
only that: equal weights give `arc-agi-benchmarking`'s `total_score /
total_tasks`, and weighting each task by its test-output count gives
Kaggle's average across all test outputs. They differ on any multi-test task.

`GameRunState` has a state ARC3 had no need for: **`submitted`**, for a game
that answered and did not win. Folding it into `gave_up` would make give-up
statistics meaningless and mislabel every game of an unscored run.

## Datasets

`taaf.datasets` ships `arc2-train` (1000 tasks) and `arc2-eval` (120) as
packed int8 `.npz` in this package — 265 KiB, no network, no checkout, no
JSON parse. `make fetch-data` rebuilds them from an ARC-AGI-2 clone, and a
test re-runs the conversion against the committed artifact.

Only ARC-AGI-2 is shipped. Its public training set already contains 391/400
of ARC-AGI-1's training tasks and 376/400 of its evaluation tasks, and the
hidden sets a submission is scored on are difficulty-calibrated against
`arc2-eval`, which no ARC-AGI-1 split can stand in for. Multi-test tasks
are 6.9% of `arc2-train` but 37.5% of `arc2-eval`: a statistic calibrated
on training data misjudges the eval set on exactly the axis where the two
scoring metrics disagree.

The packing is one flat cell buffer plus an index, not one array per grid:
an `.npz` is a zip, and 9668 members each carrying its own header and
compression stream came out ten times larger (1.9 MB against 0.2 MB for
`arc2-train`). It also beats gzipped JSON (0.27 MB) while needing no parse.

Curated subsets are named sets too (`DERIVED_SETS`): a parent set narrowed
to a listed tuple of ids, in order, each tuple carrying how it was chosen and
what it is not. `arc2-train-easy` (`EASY_TASK_IDS`) exercises the harness;
`arc2-train-hard` (`HARD_TRAIN_TASK_IDS`) is the 47 training tasks Flash-Next
did not solve on both of two passes. `make_benchmark_named` takes one name or
several, merged in order through `combine`, so `("arc2-eval",
"arc2-train-hard")` is one benchmark labelled `arc2-eval+arc2-train-hard`.

`kaggle-mount` is another named set: the competition's own challenges file,
readable only inside a Kaggle run. It is a dataset rather than a hidden
branch so that reading it is something we can deploy on purpose — a scoring
rerun forces it and returns no logs at all.

## Submission

`taaf.submission` writes `submission.json`, which on Kaggle *is* the
deliverable. It is seeded for every task before the solver starts and
rewritten atomically on its own loop (`Benchmark.submission_interval_s`,
60 s), so a hard kill leaves something scorable rather than nothing, and
never leaves truncated non-JSON. It is
deliberately exempt from `minimal_diagnostics`, which turns every other
write off.

`Benchmark.simulate_submission` is the dress rehearsal: it withholds every
answer the solver must produce *and* forces `minimal_diagnostics`, because a
real submission has neither. Simulating only the diagnostics half is how a
scoring dependency survives to the rerun. The check that makes it worth
having is that a simulated run's submission equals the scored run's.

## Diagnostics

`diagnostics.html` carries the pooled score-vs-tokens curves, the per-pass
table and the per-game pages. Each `(game, pass)` gets a static **answer
sheet** under `answers/`: every demonstration pair, then every pair to
answer with both attempts, the truth, and a verdict.

A lane steps once per answer, at the cost the solver had reached when that
answer went in. Those intermediate scores are computed **here**, from the
recorded submissions and the run's own spec — for the report, never handed to
the solver, which is what keeps `execute_action` from becoming an oracle. When
they cannot be computed (`minimal_diagnostics` records no spec,
`simulate_submission` withholds the answers) a lane falls back to a single
step at its total.

Every report reads a game still in play at the answer it holds, not as a
zero: the summary's scores, the per-game table, the pass statistics and the
end of each lane all go through `GameRun.current_result()`, so the periodic
report of a run in progress counts the answers already standing.

A periodic save renders on the event loop, where the runs the solver keeps
changing hold still, and writes on a thread (`Benchmark._write_off_loop`): the
loop is also what serves the games, and on a network filesystem the writes are
where a save's time goes; the JSON checkpoint is rendered before the
diagnostics and written even when their render fails. A sheet is written when it is new, its run changed,
or its file is gone (`_answer_sheet_key`, indexed in `.answer_sheets.pkl`).
Writing every finished game's sheet at every save, on the loop, cost sweep
7's 15-arm jobs of 1,575 runs ~0.9 h each and one on a shared node 3.4 h of
9.9 (py-spy: 79% of a save in a sheet's open and write); a steady-state save
of that size now holds the loop 2.2 s. Upstream carries the same change for
its movie player pages. With `Benchmark.periodic_page` off, a periodic save
renders neither the page nor its sheets: it writes the JSON checkpoint and
`summary.txt` alone, and the save after finishing renders the page as ever.

**Lane scores are not monotonic.** A solver that replaces a right answer with
a wrong one steps its lane back down, and the pooled curve with it. Across a
hundred-odd tasks individual regressions wash out, so a visible dip means the
solver systematically talks itself out of correct answers — which is the
whole reason to plot it this way.

## The example solver

`SolverFlipLR` submits an all-black grid, then the mirror, twice over,
settling on the mirror. One answer would exercise the harness just as well;
several exercise what only a solver changing its mind reaches — the standing
answer, a lane that steps down, a cancellation landing between two answers.

The mirror is what decides the score. Exactly one task in
`arc2-train` (`67a3c6ac`) is a left-right flip throughout and none of
`arc2-eval` is, so `make_benchmark_flip_demo`'s outcome is known before it
runs; `tests/test_flip_end_to_end.py` re-measures that against the shipped
data. ARC-AGI-1 also carries `67a3c6ac` with different given pairs that
happen to flip too — the measurement holds for our copy only.

## Deploying

The SLURM path needs `UV_CACHE_DIR` on storage the worker can see; the
login node's scratch is not.

`TufaSlurmTarget(jobid=...)` runs the same rendered script as an
overlapping step of an allocation already held, so a run starts when it is
deployed rather than when the queue reaches it. The step is named uniquely
per deployment and its id is written to `deploy_meta.json`, so `stop()`
signals that step and leaves the allocation alone; node-local sources and
the venv are keyed by job and step, so deploys into one allocation never
share them.

A uv-managed interpreter in the node-local cache that no longer runs (its
`bin/python3` gone while the directory stays) is removed before the sync,
with uv's version aliases beside it, and reinstalled; one younger than ten
minutes is left alone as a sibling's install in progress.

A solver can end its own run early the way the soft deadline would:
`Benchmark.stop_early(reason)`, reached by the solver as `Solver.stop_run`
for the run's duration, cancels the solver task so its games land
`cancelled`, teardown writes everything, and `run` returns normally with
the reason recorded (`stop_reason`, in the JSON and the summary). The
platypus stall watchdog is its caller. The solver task is cancelled once per
run, whichever of the deadline, `request_stop` and `stop_early` asks first;
a later stop cannot interrupt the cleanup the first began.

A solver's `start_up()` brings up what is slow before it can play (an
engine, say). `Benchmark.run` calls it in the solver task, after the
placeholder `submission.json` is written and right before `run_games`, so
the soft deadline, a stop and a failure treat a start-up as they treat
play; `shut_down()` releases what it brought up if `run_games` did not.
`run(startup_deadline=...)` bounds the start-up: a thread exits the process
(status 3) the moment the deadline passes with the solver still starting,
even with the event loop blocked or its output unread. On Kaggle the target sets it, from
notebook start (`startup_deadline_s`) and from machine boot
(`startup_deadline_after_boot_s`, read from `/proc/uptime`, which also sees
a session that sat on its machine before the notebook ran), the earlier
applying; in a submission, real or simulated, the one from boot is
`submission_startup_deadline_after_boot_s`, since Kaggle hands out machines
that were up long before the session was submitted. The notebook prints the
machine's uptime at its start. The exit is
a hard one, which a submission records as an error and does not score:
unlike ARC-AGI3, where a submission that has not touched the games is
refunded, nothing is saved by it. It is there to make a start that went
wrong show as one (2026-10-27).

The source dataset holds the deployed bundle as one archive,
`taaf-kaggle-bundle.taafzip` (`DATASET_BUNDLE_ARCHIVE`), which the notebook
unpacks to a temporary directory, the bundle directory from then on: a
dataset of many files is slow to attach, and Kaggle unpacks an uploaded
`.zip` into its files, so the archive is named otherwise.

`KaggleTarget.additional_model_sources`, together with a solver's
`kaggle_model_sources`, attaches Kaggle models to the kernel. A model is
not a dataset: it mounts under
`/kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>`.
