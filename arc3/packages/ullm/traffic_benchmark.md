# The traffic benchmark: throughput from recorded traffic

*Register: living.*

Flash-Next throughput on a deployment cell, read from recorded traffic
replayed against the cell's live engine, in place of an hour of gameplay.
The benchmark is `ullm.traffic` (the workload and the replay) over
`VLLM.serve_ids`; each harness runs it through its own `TrafficDuck`
(`duck/traffic.py`), built by the same `build(engine, environment)` as the
duck, so the engine, tier, step budget, concurrency and sampling under test
are the column's. Where each reading landed and what it decided is the
Flash record (`flash_optimization.md`) and its ARC2 addendum; this document
is how the instrument works and what a number from it means. Its open
items are in `roadmap.md` here.

## What a replay is

A **recording** is a benchmark run with ids-only traces on
(`trace_content="ids"`, TAAF R5.39): every call's exact token ids and reply
length, and the slot each call ran in, which platypus's collector stamps on
the steps. A **workload** is cut from a recording at a snapshot moment: the
recording's slots become **lanes**, each holding the history its slot had
built at the snapshot and the calls it still had to make. The replay plays
the lanes closed-loop against the engine: a lane's next prompt continues
the engine's own reply where the recording shared a prefix, so the engine
serves its own history and never the recording's, and each reply is forced
to its recorded length (`min_tokens = max_tokens`, `ignore_eos`), so the
work per call is the recording's whatever the engine samples. There is no
scheduler: the recording's slot decisions are replayed as they fell, and
`max_concurrent` sets how many lanes run at once.

A lane that runs out of calls starts again, so a fast engine cannot empty
the window below the asked concurrency. It starts from its first call, its
snapshot history prefilled cold again: the extra prefill is accepted to
keep the recording's balance of request sizes, where restarting at a
lane's first sequence opening loops its last, often short, sequence. Each
fresh prompt of a restarted pass (its first, and each sequence opening)
has one token
changed just past the head every opening shares (system prompt and tool
schema), so it hits the cache as a new game would and no further, and no
two passes send the same prompt. The summary counts the restarts
(`restarts`, `restarts_by_lane`) and each call's row names its pass;
readings before restarts existed never ran a lane out, so they compare.

Between one call returning and its next, a lane waits
`TrafficDuck.request_gap_s`: none since 2026-09-28, as play waits none
(`RunConfig.min_request_gap_s`). Readings on the 09-19 and 09-20 workloads
ran 5 s, so a replay of those passes `--request-gap-s 5` to compare with
them; a reading on a later workload runs none.

Nothing is prefilled ahead of the window. The engine reaches the
recording's state by serving the lanes as it would in play, so the window
opens cold and its first `settle_s` is the warm-up the reading leaves out.
An earlier format prefilled every lane's history before the window; ARC2's
runs of 2026-09-19 used it, so ARC2's next reading is a fresh baseline.

The **reading** is generated tokens per second, with uncached prompt tokens
per second beside it, over the window past the settle: ARC3's `TrafficDuck`
runs `window_s` 1,500 and `settle_s` 600, because the RTX takes about eight
minutes to reach a steady state from a cold pool (its first baseline's
running set was still filling three minutes into a five-minute settle);
ARC2's reads fifteen of twenty minutes. It is the `[traffic] settled ...`
line of the log, with `traffic_result.json` (the ten-second samples) and
`traffic_calls.jsonl` (every call's prompt, cached and reply lengths)
beside it in the job directory.

## What the engine sees

Two things differ from play, both deliberate. The conversation is
disjointed: a forced reply length makes the model's reply diverge from the
recording's, and the next prompt splices the recorded continuation onto it,
so the model reads turns it did not write. That is the price of fixing the
work per call; an engine's cost is set by token counts and prefix
structure, not by whether the text coheres. And the shared prefix a call
recomputes follows the recording's alignment rather than the engine's own
play, which is why a knob that changes alignment (`prefix_match_unit`)
moves the uncached rate where a kernel change moves generated and uncached
together.

## Cutting a workload

    uv run python tools/traffic_workload.py RECORDING_JOB_DIR OUT.b64 \
        --horizon-tokens 60000 [--snapshot-min 30]

The snapshot is placed where every lane still holds at least the horizon
of reply tokens ahead of it. By default it sits in the middle of the span
from the recording's settling (90% of lanes past their first handover) to
the last moment every lane holds the horizon. A recording whose games all
start at once settles late, if at all, because their first sequences end
in one wave. The 09-20 ARC3 cut was placed by hand at 30 minutes
(`--snapshot-min`, which skips the guard), where 119 of 120 lanes held 60k.
Recordings since 2026-09-28 start their games apart
(`RunConfig.start_gap_s`), and the default placement holds for them. The
file is base64 of a zip, because a Kaggle payload is unpacked if it is an
archive. **Two runs compare only on the same file** -- it fixes the
prompts, the replies and the lanes -- so a new cut starts a new board.

**A cut's first calls are long** (`--skip-in-flight`). Each lane starts at
the call in flight at the snapshot, and a moment falls inside a long call
more often than a short one: on the ARC2 09-29 cut the first 60 lanes'
first replies ran a median 7,244 tokens (29 of 60 over 8k) against 1,066 for
the calls after, and a cold replay on the four L4s spent its warm-up decoding
them at ~13 tok/s a request with nothing else admitted. `--skip-in-flight`
starts each lane at the call after it: first replies a median 1,372 (9 of 60
over 8k).

A cut is validated by replaying it on the engine that recorded it, at the
recording's lane count, against the recording's own rate over the same
stretch. ARC3's reads within 5% generated and 3% uncached prompt at the
same running count (2,613 against 2,746 tok/s, 5,886 against 6,058; B200
job 2715108, archived as `20260920_1208_b200_validation`); ARC2's within
0.3% (3,090 against 3,097). Neither is pool-bound on a B200, so that is a
fidelity check, not a throughput reading.

## Current workloads

Both were cut on 2026-09-29 from the B200 long runs of 2026-09-28, with a
60k horizon at the default snapshot. A replay on them waits no request gap.
Neither has yet been validated against its recording (above).

| | ARC3 | ARC2 |
|---|---|---|
| file | `/data/shared/research/arc_3_results/shared/traffic/workload_long100_20260929.b64` | `/data/shared/research/arc_2_results/shared/traffic/workload_long230_20260929.b64` |
| recording | `/data/shared/research/arc_3_results/shared/long_runs/arc3_k4_long_20260928_190433` (job 2808095), the official 25 x 10 | `/data/shared/research/arc_2_results/shared/long_runs/arc2_k4_long_20260928_190809` (job 2808112), arc2-eval x 10 |
| lanes | 100 | 230 |
| snapshot | 158 min (settled from 63, horizon held until 254) | 75 min (settled from 35, held until 116) |
| after the snapshot | 22,501 calls, 43.6M reply tokens | 13,784 calls, 39.2M reply tokens |

The four L4s read ARC2's cut with `--skip-in-flight`,
`/data/shared/research/arc_2_results/shared/traffic/workload_long230_20260929_skip1.b64`
(their runs B-N, `flash_optimization_arc2.md`, "Drafting on the four
L4s").

The recordings are gameplay at the slurm-b200 cells' defaults: k=4,
shipped sampling, park mode, games started 12 s apart on ARC3 and 0.5 s on
ARC2. They are not `record` jobs, so they differ from the Kaggle column's
play in three ways:
- **Slot policy.** The B200 cells run queue order, where Kaggle ranks games
  (`least_level_tokens`; ARC2 `least_tokens`). The calls come out alike: a
  game gives up its slot only at a compaction, so the request it resumes
  with is the post-compaction one play makes there anyway. Only the timing
  differs, and a replay does not keep timing.
- **Caps.** ARC3's per-level cap of 175,000 ended all 136 of its 250 unwon
  games, where Kaggle play goes on to the clock. More of the recording is
  therefore early game, before the first compaction: about 17% of generated
  tokens, against about 10% in 9-hour Kaggle play. ARC2's 150,000 per-game
  cap ended 81 of 1,200 games.
- **Host stalls in ARC3's timing.** TAAF wrote each game's trace on the host
  event loop, so the recording's timing carries stalls. The calls are
  unaffected.

The prompt budget and context window are the Kaggle cells'.

Earlier workloads stay for the readings made on them:
`workload_120lanes_20260920.b64` (ARC3, before the 125,000-token prompt
budget), `workload_100lanes_20260919_v2.b64` (ARC2), and the 2026-09-28
cuts `workload_120lanes_20260928.b64` and `workload_300lanes_20260928.b64`.
The 09-28 cuts come from recordings whose games started together, so their
compactions arrive in one wave, and their lanes are short enough to restart
at low lane counts.

## Running it

From an arc3-platypus checkout, against a workload already cut:

    # The RTX column exactly as it ships.
    uv run python tools/traffic_launch.py kaggle \
        --workload /data/shared/research/arc_3_results/shared/traffic/workload_long100_20260929.b64

    # A knob: --arm names one the launcher knows (ARMS), repeated to stack;
    # --max-concurrent changes the lanes, which a knob that frees pool needs.
    uv run python tools/traffic_launch.py kaggle --workload ... --arm bf16_state --max-concurrent 80

The launcher's other jobs are `record` (a B200 run with the Kaggle
column's scheduling and ids-only traces, 120 lanes so a knob that frees
pool still has lanes to fill it with) and `replay` (a B200 replay of a
file); ARC2's launcher (`packages/duck/tools/traffic_launch.py`) adds
`smoke`, a short play on four B200s set up as the four L4s. All are fire
and forget: the job dir is printed. Kaggle's batch queue runs hours; an
interactive session is faster and leaves only what you save, so save its
console log, which is enough to archive. Workload files live in the shared
results tree, `/data/shared/research/arc_3_results/<user>/traffic/` for
ARC3 and `/data/shared/research/arc_2_results/<user>/traffic/` for ARC2;
"Current workloads" above names the ones to read on.

To compare two versions of the code (a branch against the one it would
replace), replay the same workload at the same lanes from each checkout's
own launcher, so each runs its own `TrafficDuck` and engine. An older
checkout's replay may still wait 5 s between calls; set it to none, as
above, or the two do not read alike.

**The decode gate** (`replay`'s `decode_gate`; either repo's
`TrafficDuck.decode_gate` and its launcher's `--decode-gate`; ARC3's
`--profile-only` adds a minute of play, the profiler windows and no
reading). Every lane's first prompt is prefilled on its thread by a
one-token request before the window opens, so the window opens warm rather
than on a cohort of cold prefills; the settle and the window count from the
gate's lift (`on_gate_lifted`), and the engine's profile windows are held
shut through the gate and then move past it by its length
(`EngineFactory.hold_profile_windows`, `delay_profile_windows`). With a
profile after the window, the reading is taken and written as the window
closes, before the profile slows the engine.

**A kernel run by hand.** An interactive Kaggle session serves its files to
no API, so ARC2's launcher's teardown packs `/kaggle/working` into
`run_bundle.tar.gz`, and the engine's processes print into
`engine_output.log` there as well as to the console (`engine_output_log`).
ARC2's four-L4 runs queue in one file, `packages/duck/tools/kaggle_next.py`,
no flags, so whoever launches it launches what was committed. Either
launcher's `--run-id` gives a run a source dataset of its own: a kernel
version attaches a dataset's current version, which can still be the
previous run's.

## When a number counts

A reading is void unless all three hold over the scored minutes; the
archive records each:

- **Capacity waits above zero** in every engine line
  (`waiting=N(capacity:N)`). A replay that never waits for a slot reads the
  workload's pace, not the engine's, and a faster engine reads the same.
  Lanes are set above what the pool runs for exactly this -- 80 on the RTX
  against a pool running ~66 -- and a knob that frees pool needs more lanes
  than the pool then runs, or it reads as nothing. A B200 replay's lanes
  default to its column's own concurrency (`max_concurrent` as built), so
  they track gameplay, and a lane count past the workload's replays its
  lanes again as nonced copies (`replay`). Under the host tier's park mode
  the pool holds running requests only, so lane counts sized when idle
  conversations filled it (50-60 on the B200) leave it mostly empty.
- **`deferred` near zero**: admissions the step budget held back. A run
  spending its reading at the admission line measures its own start.
- **No `memory allocation failed with OOM`** anywhere in the log.

Production runs in the same regime -- ARC2's twelve-hour run queued in 87%
of its minutes, the 80-lane RTX replays in 79-100% -- but not for the same
reason: production queues on `capacity`, the replay partly on `deferred`,
the step budget holding cold heads back rather than the pool being full.

**A drafted replay reads acceptance high.** For two heads alike it read
~0.17 tokens a k=3 draft (~9%) above real play (1.98 against 1.81 for the
shipped head, 2026-09-26), confounded between the games (official against
the test set), lengths forced past EOS, and fresh replies on recorded
prompts. It ranks drafters right; read its absolute acceptance, and a
drafter's absolute speedup, as optimistic.

## How wide a reading is

Per-minute scatter on these runs is 5.5-6.4%, and it is the wrong
statistic: the standard error of a fifteen-minute mean is **~1.3%**, so a
bar set against the scatter rejects effects the reading resolves. Between
runs the spread is wider, and one pair so far measures it: the same code
and configuration (ARC3 `main` `2571d65b`, fork `0482e9d2`, the 140-lane
B200 replay) read 4,946 generated tok/s on dgx06 and 5,152 on dgx02 on
2026-10-05, 4.2% apart
(`/data/shared/research/arc_3_results/shared/traffic/replay_140lanes_*_20261005_*_v031_before*`).
Node and neighbours both differed, so the pair bounds them together; two
arms started as the two steps of one job share both
(`flash_optimization.md`, "Reading a run"). The window drifts: contexts grow as each lane's sequence advances, so the
RTX pool holds ~75 requests at 600 s and ~60 at 1,140 s, with the queue
appearing in the second half. Every arm drifts alike, so comparisons hold,
but a reading is an average over a moving operating point. Read the running
count beside the rate.

## Archiving a run

    uv run python tools/traffic_archive.py /path/to/saved/log --label NAME \
        --kind kaggle-rtx-interactive --note "..."
    uv run python tools/traffic_archive.py --pull OWNER/SLUG --label NAME

It writes `/data/shared/research/arc_3_results/<user>/traffic/archive/<stamp>_<label>/`
with the run's own files and a `provenance.json`: the code (commit, clean
tree and the vLLM fork's pin, from the bundle's `git_status.txt` or the
log's own `git status:` block), the workload (name and the sha256 of our
copy), the configuration (the deployed agent's repr), the reading (the
settled rates, with the engine lines attributed by position -- after the
settle mark, before the window's summary -- so the warm-up cannot leak in)
and the three checks. `index.jsonl` beside the runs lists every one. A log
with no settled line is archived as a failure with no reading: a run that
fell over is provenance too. A row in a record cites its archive
directory, and a reading without one does not get a row.
