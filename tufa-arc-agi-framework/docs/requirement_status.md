# Requirements implementation status

Per-requirement walkthrough of [`requirements.md`](requirements.md) in spec
order. For each: one-line implementation pointer + a tests line. "Bonus:"
flags non-spec features carried by the codebase.

## Known gaps

One requirement is not met by the current code:

- **R2.22 — Inline isolation.** R2.22 describes Inline as still "isolated"
  (its own venv, own job-dir outputs, etc.) and the per-environment table
  marks R2.37 (per-run venv) as required. The current `InlineTarget`
  treats R2.37 as N/A and runs in the launcher's existing venv. The other
  isolation pieces (job-dir outputs, source snapshot, stdout tee, reattach
  metadata) are honored; only the venv isolation is missing.

Convention: this doc references files and symbols by name, never line
numbers. Test paths are relative to the repo root.

---

## 1. General

### R1.1 — Repo on github.com/tufalabs

Lives in `tufa-arc-agi-framework/` of this repository.

### R1.2 — Makefile

[`Makefile`](../Makefile) provides `install` / `install-dev` / `test` /
`format` / `check-format` / `lint` / `static` / `prepare` / `prepare-ci`
/ `refresh-examples` / `clean`.

### R1.3 (T) — CI under 5 minutes

`.github/workflows/ci.yml` runs `make prepare-ci` on every push and PR
(≈75 s locally). Branch-protection wiring is a one-time UI step.

### R1.4 (T) — `ruff format` + `ruff check` + `pyright`, zero errors / warnings

Configured in `pyproject.toml` (`[tool.ruff]`, line length 120) and
`pyrightconfig.json` (strict on `src/`). `make static` greps the pyright
summary for "0 warnings" so warnings fail CI, not just errors.

### R1.5 — No `vllm` / `torch` imports

Runtime deps are `arc_agi`, `arc-agi-3-local` (re-arc-3 git dep),
`arcengine`, `imageio`, `matplotlib`, `numpy`, `scipy` — none pull `vllm`
or `torch`. Verifiable via grep.

---

## 2. Running and deploying benchmarks

### R2.1 — `Benchmark` class with required properties

`Benchmark` (`taaf.benchmark`) is a `@dataclass` with the five spec
fields. `job_dir=None` disables JSON / diagnostics on disk. `start_time`
/ `end_time` / `solver_label` populated by `run()`.

**Bonus:** `game_weights: list[float] | None` parallel to `games` for
weighted cross-game aggregation in diagnostics. Weight 0 excludes from
aggregation while keeping per-game visibility. Survives JSON + pickle.
Tested in `tests/test_benchmark.py` (`test_game_weights_*`).

### R2.2 (T) — Multi-pass via copies to one solver

`Benchmark.run` deepcopies each game `n_passes` times and calls
`solver.run_games(...)` once with all copies (passes-major layout).
Game ids must be unique within a benchmark; since `game_id` is only
populated by `start_game()`, `run()` validates uniqueness right after the
first pass opens its games and raises a `ValueError`. The check and the
game-start loop live inside the same `try` as the run, so the existing
`finally` tears down any games/session opened before a failed validation
(no scorecard/environment/server leak).
Each copy's `GameRun.pass_index` is stamped in that same loop — the one
authoritative record of which pass a run belongs to, read by solvers
that configure themselves per pass and by every per-pass grouping in
diagnostics (`_pass_runs`, `weights_by_game_id`, the comparison
harmonizer). `Benchmark.from_json` backfills it via
`_backfill_pass_indices` for runs recorded before the field existed, so
nothing downstream carries a positional fallback.
Tests: `test_multi_pass_creates_n_copies`,
`test_passes_use_independent_copies`, `test_pass_index_stamped_per_pass`,
`test_from_json_backfills_missing_pass_index` in `tests/test_benchmark.py`.

### R2.11 (T) — `Benchmark.run` plays all games

`Benchmark.run` schedules the solver as a task and awaits it.
Test: `test_run_single_pass` in `tests/test_benchmark.py`.

### R2.12 (T) — `soft_end_time` cancels via `task.cancel()`

`Benchmark._cancel_at` sleeps until the deadline then cancels. The
internal `_deadline_fired` flag lets `run` distinguish deadline
cancellation (swallow) from caller cancellation (re-raise).
Test: `test_soft_end_time_cancels` in `tests/test_benchmark.py`.

### R2.13 (T) — Periodic JSON + diagnostics save

`Benchmark._periodic_save_loop` saves every `periodic_save_interval_s`
(default 600). Final save happens in `run()`'s teardown. JSON written
atomically (rename-on-replace). Movie generation is idempotent on disk.
A periodic save renders on the event loop, where the runs hold still,
and writes on a thread (`Benchmark._write_off_loop` over
`taaf.support.FileWrite` / `write_files`): the loop also serves the
games, and on a network filesystem the writes are where a save's time
goes. A cancelled save waits for its write, so teardown's own write lands
last. The JSON checkpoint is rendered before the diagnostics and written
even when their render fails
(`test_a_periodic_save_keeps_the_json_checkpoint_when_diagnostics_fail`).
`render_run_html` writes a movie's player page only when it is new,
what it is drawn from changed (`_movie_player_key`, indexed in
`.movie_players.pkl`) or its file is gone, rather than every finished
game's at every save. With `Benchmark.periodic_page` off, a periodic save
renders neither page nor movies (`_diagnostics_writes`), writing the JSON
and `summary.txt` alone; teardown's `_generate_diagnostics` renders the
page as ever.
Tests: `test_periodic_json_save`, `test_run_single_pass`,
`test_a_periodic_save_writes_off_the_event_loop_and_a_cancelled_one_waits`,
`test_the_periodic_save_writes_while_the_run_goes`,
`test_a_periodic_save_without_the_page_renders_none` in
`tests/test_benchmark.py`;
`test_a_render_writes_only_the_movie_players_that_changed` in
`tests/test_diagnostics.py`. Kaggle submission passes
`minimal_diagnostics=True`, which skips periodic/final file writes and
uses `_minimal_status_loop` for stdout-only heartbeats; covered by
`test_minimal_diagnostics_skips_file_writes_and_frame_logging`.

**Bonus: sidecar pickles.** Final teardown also writes
`intermediate_states.pkl` (per-step `GameState`s as each run's
`CompressedStates` — only references taaf/arcengine/numpy; loaded by
default on `from_json`, and a sidecar of plain lists from an older run
still loads), `games.pkl`
(unstarted-original game list — opt-in), and `solver.pkl` (pre-`setup`
solver — opt-in). This lets diagnostics-only environments regenerate
plots from the JSON without needing the user's `Game` / `Solver`
subclasses to be importable. Tests: `test_sidecar_*` in
`tests/test_benchmark.py`.

### R2.14 (T) — JSON import method

`Benchmark.from_json` rebuilds the class without `games`. The kwargs
`with_intermediate_states` / `with_games` / `with_solver` attach the
matching sidecars when present.
Test: `test_json_round_trip` in `tests/test_benchmark.py`.

### R2.15 (T) — `run` finishes gracefully on solver errors

`Benchmark.run` swallows non-`BaseException`s from the solver task and
proceeds to teardown; surviving games end in `crashed`.
Test: `test_run_swallows_solver_error` in `tests/test_benchmark.py`.

### R2.21 — `Benchmark.deploy(target)`

`Benchmark.deploy` is an async thin seam delegating to
`target.deploy(self)`. ABCs `DeploymentTarget` / `DeploymentHandle` live
in `taaf.deploy`.
Test: `test_inline_deploy_runs_to_completion` in
`tests/test_deploy_inline.py`.

### R2.22 — Inline environment

`taaf.deploy_inline.InlineTarget.deploy` runs the benchmark in-process:
snapshots sources, tees stdout to `job_dir/stdout.log`, prints preamble,
awaits `Benchmark.run` with a soft deadline, writes `deploy_meta.json`.
Cancellation routes through `asyncio.CancelledError` into `Benchmark.run`'s
teardown.
**Gap:** R2.37 (per-run venv) is treated as N/A; see the known-gaps
section above.
Tests: `test_inline_deploy_*` in `tests/test_deploy_inline.py`.

### R2.23 — Tufa cluster environment

`taaf.deploy_slurm.TufaSlurmTarget.deploy` submits via `sbatch --parsable`
and returns a `TufaSlurmHandle` without blocking. Required fields cover
hardware (`gpu` ∈ `B200`/`B300`/`CPU`, mapped via `_GPU_GRES_NAMES`)
and walltime (`time`). Resource asks via two coordinate systems
(`cpus_per_gpu`/`mem_per_gpu_bytes` or `total_cpus`/`total_mem_bytes`);
mixing raises at deploy time. Cluster submit-plugin requires
`--ntasks=1` + per-gpu resource forms — emitted unconditionally for GPU
jobs. `main_project` is mandatory (which bundled repo the worker
`uv sync`s from).
Tests: `tests/test_deploy_slurm.py` covers sbatch rendering across the
config matrix using mocked `subprocess.run`.

`TufaSlurmTarget.jobid` runs the same rendered script as an overlapping
step of an existing allocation (`_start_step`) instead of submitting it,
so a run held in an interactive allocation starts without queueing.
`--overlap` is needed twice over: once on the step itself, because the
allocation's own shell is a step holding every resource, and again on the
container `srun` a step deploy renders, because a step's overlap does not
carry to the steps it starts. The launcher's only handle on the step is
the queue listing, so the step is asked for under a name unique to the
deployment and the id it registers under is captured and persisted —
`job_dir.name` alone is shared by runs under different parents, and both
handles would then resolve to whichever registered first. A step that
never registers has its launcher cancelled before the timeout is raised;
a pending launch outliving a failed deploy would start the benchmark
with no `deploy_meta.json` naming it. Node-local storage is keyed by job
*and* step (R2.37) so deploys into one allocation cannot inherit each
other's sources and venv.
Tests: `test_deploy_into_an_allocation_starts_an_overlapping_step`,
`test_two_deployments_sharing_a_job_dir_name_keep_separate_steps`,
`test_a_step_that_never_registers_cancels_its_launcher`,
`test_a_container_step_overlaps_the_allocation_too`,
`test_a_step_keys_its_node_local_root_by_step`,
`test_a_step_handle_stops_its_step_and_leaves_the_allocation`.

With `launch_step` off, a deploy into an allocation only prepares the run:
its job dir, script and a `deploy_meta.json` naming the allocation and the
step it is to be, marked not launched. A script of the allocation starts
it with `step_argv` under that name -- one splitting the allocation's GPUs
between runs it starts together, which an overlapping step holding every
resource cannot. The allocation may still be queued when the run is
prepared. Until the step runs, its absence from the queue means it has yet
to start, so its handle is done only once srun has opened its stdout or
the allocation has ended without it.
Tests: `test_a_prepared_step_waits_for_its_allocation_to_start_it`.

### R2.24 — Kaggle environment

`taaf.deploy_kaggle.KaggleTarget.deploy` writes a source dataset bundle
(`benchmark_initial.pkl`, `deploy_target.pkl`, setup commands, and
source snapshots) plus a notebook bundle rendered from
`src/taaf/kaggle/taaf_kaggle_run.ipynb`, then creates/versions the
dataset and pushes the kernel through the Kaggle CLI unless
`dry_run=True`. The rendered notebook installs ARC runtime wheels from
the competition input, finds the source bundle, and runs the pickled
benchmark in `/kaggle/working` directly from visible notebook cells.
Before pushing the kernel, `_ensure_dataset` not only waits for the
dataset to report `ready` but also for its listing `lastUpdated` to
advance past the pre-push value (`_wait_for_dataset_version_change`) —
the kernel attaches the dataset's *current* version, and that pointer
can lag `ready`, so a kernel push can otherwise attach the previous
version of a just-pushed bundle. The CLI exposes no version number, so
`lastUpdated` is the proxy; best-effort (warns and
proceeds on timeout) so an unobservable ref can't hang the deploy. The
launcher also bundles `git_status.txt` and the runner copies it into
`/kaggle/working` before `run()`, so `diagnostics.html`'s git-status
block (R2.31.2) shows the launcher-captured status — the worker can't
regenerate it (no `.git` / dist-info), and `run()` only writes that file
when absent. Every kaggle-CLI subprocess gets a per-user `TMPDIR`
(`_kaggle_env`): the CLI keeps resumable-upload state at a fixed
`{tempdir}/.kaggle/uploads`, which collides between users on a shared
host. After the run, a final notebook cell embeds
`diagnostics.html` inline (iframe `srcdoc`, so a saved kernel shows the
full report without downloading outputs); minimal-diagnostics
submissions print a note instead.
Tests: `tests/test_deploy_kaggle.py` (`test_kaggle_deploy_*`,
`test_kaggle_notebook_template_cells_are_valid_python`,
`test_dataset_last_updated_parses_csv_skipping_warnings`,
`test_wait_for_dataset_version_change_*`,
`test_kaggle_env_redirects_tmpdir_when_shared_staging_unwritable`).

### R2.51–R2.61 — Kaggle submission environment

Implemented across `taaf.deploy_kaggle`, `Benchmark.run`, `RunSession`,
and `standard_benchmarks.make_benchmark_kaggle_official_110`:

- R2.51 / R2.59: notebook-level `RUN_AS_SUBMISSION` is ORed with
  Kaggle's `KAGGLE_IS_COMPETITION_RERUN`; the notebook stamps those flags
  onto the `KaggleTarget` passed as `runtime_environment`. It calls
  `Benchmark.run(..., minimal_diagnostics=True)`, which disables JSON /
  sidecar / HTML writes and frame logging while keeping 10-minute
  stdout status.
- R2.52 / R2.53: kernel metadata defaults to private, internet off, and
  RTX6000 (`NvidiaRtxPro6000`); callers can set notebook slug/title or
  `cpu_only=True` for public/offline smoke tests.
- R2.54 / R2.55 / R2.56: source snapshots are uploaded as a Kaggle
  dataset that holds the whole bundle as one archive,
  `DATASET_BUNDLE_ARCHIVE` (`_write_source_dataset_upload`, staged in
  `kaggle/source-dataset-upload`); both notebooks find it under
  /kaggle/input and unpack it (`_unpack_bundle`) to a temporary directory,
  the bundle directory from then on (test:
  `test_the_source_dataset_goes_up_as_one_archive_the_notebooks_unpack`),
  additional dataset refs come from the target plus optional
  solver attributes (`kaggle_dataset_sources`), and setup/package hooks
  come from target commands plus optional solver attributes
  (`kaggle_setup_commands`, `kaggle_pip_install_packages`) before the
  benchmark pickle is loaded. Setup commands receive the notebook Python
  executable, bundle/working paths, and `TAAF_KAGGLE_INPUT_PATHS` (JSON
  `{ref: mount_path}` for attached datasets and utility scripts) in
  environment variables and can persist environment overrides through
  `TAAF_KAGGLE_SETUP_ENV`. GPU notebooks also set `LIBRARY_PATH` to start
  with `/usr/local/nvidia/lib64` before setup commands run.
  Best-effort cleanup hooks (`teardown_commands` /
  `kaggle_teardown_commands`) run after the benchmark so services and
  large temporary installs can be removed before Kaggle snapshots
  outputs.
  Setup commands can also be handed an offline artifact:
  `KaggleTarget.bundle_files` plus the solver's `kaggle_bundle_files`
  are copied verbatim into the source dataset's `payload/`, readable at
  `$TAAF_KAGGLE_BUNDLE_DIR/payload/<name>`. Unlike a bundled source repo
  a payload file is never interpreted — nothing puts it on `sys.path` —
  which is what makes it usable for something the kernel installs over
  itself (arc3-platypus ships its vLLM fork's delta this way). A missing
  file, or two that collide by name in the flat `payload/`, fails on the
  launcher.
- R2.57: `make_benchmark_kaggle_official_110` builds 110 `GameAPI`
  entries by repeating the 25 official games 4 or 5 times with
  `n_passes=1`. Its `competition_clone_ids=True` path matches
  `CompetitionArcadeServer.official_110()` so the repeated games use
  unique IDs under the competition simulator.
- R2.58: true competition reruns switch to the gateway-backed
  competition Arcade and replace the benchmark game list with the live
  competition environment list.
- R2.60: the generated notebook starts with a Markdown explanation of
  what it does and how submission mode is selected.
- R2.61: `KaggleTarget.make_share_version` selects the public share
  variant. `_write_kernel_bundle` renders `taaf_kaggle_run_share.ipynb`
  (the minimal template — no `RUN_AS_SUBMISSION` emulation, no local-debug
  scaffolding; it branches only on `KAGGLE_IS_COMPETITION_RERUN`: the live
  Arcade in a real rerun, the competition's own bundled `environment_files`
  offline in an interactive Save & Run, so it stays runnable end-to-end);
  `_write_source_dataset_bundle` passes
  `snapshot_editable_sources(..., exclude_repos=("re-arc-3",))` so the
  offline env-files snapshot is left out; and `deploy` `-share`-suffixes
  the default kernel/dataset slugs (explicit slugs untouched). Dropping
  `re-arc-3` is made safe by the lazy `import re_arc` in
  `taaf.game_api` / `taaf.competition_arcade` /
  `taaf.standard_benchmarks` (only the offline env-files path needs it),
  and the share notebook's offline branch reads the competition's own
  env-files rather than `re-arc-3`. `_render_kaggle_notebook` strips
  `# noqa` from code cells of both variants. `package_for_local_debug`
  honours `make_share_version` (the local bundle reflects the share
  exclusion + share notebook), while `run_source_bundle_locally` only
  *executes* the default variant — the share notebook is Kaggle-only and
  its offline branch needs the competition env-files dataset.

Tests: `tests/test_deploy_kaggle.py`,
`test_minimal_diagnostics_skips_file_writes_and_frame_logging` in
`tests/test_benchmark.py`, and
`test_kaggle_official_110_repeats_official_games`,
`test_kaggle_official_110_can_use_competition_clone_ids` in
`tests/test_standard_benchmarks.py`, plus
`test_official_110_competition_server_exposes_unique_clone_ids` in
`tests/test_competition_arcade.py`.

### R2.62 — Public-target guard

`taaf.deploy_kaggle._assert_targets_not_public`, called in `deploy`
after the auth check and before any upload, refuses to push the default
(non-share) bundle to a public target. `target.public=True` short-circuits
to a refusal before any query. Otherwise it reads the *current* remote
visibility of both the destination notebook and source dataset:
`_resource_is_public` calls `_fetch_is_private`, which does an exact
get-by-ref via the low-level `kagglesdk` client (`get_kernel` /
`get_dataset_metadata` — the CLI/search paths can't return an exact
resource's visibility, and a paginated search could miss an existing public
ref). Kaggle reports a missing/own resource as HTTP 403/404;
`_kaggle_ref_absent` classifies that as "not created yet" (mirroring
`_dataset_exists`), so `_resource_is_public` returns `False` and the deploy
proceeds to create it private (R2.52) — first deploys aren't blocked. Any
*other* failure to read visibility raises a `RuntimeError` with a hint
(fail closed), so the guard never silently passes an indeterminate (possibly
public) target. `_kaggle_api` sets the Kaggle credentials, authenticates,
and restores the prior `os.environ` values so the process env isn't mutated
(matching the `_kaggle_env` pattern). The guard applies to every variant,
including the share variant (R2.61): TAAF pushes privately and publishing is
a manual later step, so a public share target is refused too — its bundle
excludes the offline env files, but a public target at deploy time is still
treated as an accident.
Tests: `tests/test_deploy_kaggle.py`
(`test_assert_targets_not_public_*`, `test_resource_is_public_*`); the
keyless mocked tests stub `_fetch_is_private` / `_kaggle_api`, and a
`TAAF_KAGGLE_LIVE_TESTS`-gated live test confirms the guard fires on real
public resources and stands down on private / not-yet-created ones, without
adding a Kaggle-key dependency to CI.


### R2.63 (T) — Keepalive RESETs

`Benchmark.keepalive_resets` / `keepalive_interval_s` switch it on and set
the interval; `Benchmark.run(clock_start=...)` is the first action the clock
counts from (the run's own start when omitted). `run` starts a `_Keepalive` thread as soon as the first
game has started — starting the rest is itself a stretch without moves —
and stops it at the top of teardown, before any game is finished; the
soft-deadline and graceful-stop paths halt it the moment they cancel the
solver, since a solver winding down makes no moves worth covering. A thread,
not a task: a solver's synchronous engine build blocks the event loop for
exactly the stretch this covers. It polls the total recorded moves at a tenth
of the interval (at most every minute); a change restarts the clock, and once
the interval has passed since the last move, keepalive or `clock_start` it
calls `Game.keepalive_reset()` on the games in run order until one accepts,
prints a `[taaf.keepalive]` line either way, and restarts the clock. So the
gap between engine actions is bounded by the interval plus one poll,
whenever a move lands; a stretch already longer than the interval when the
run starts is covered at once, and only once. Teardown is bounded too:
`Game.finish_game` waits a bounded time for the action lock and finishes
without it if a keepalive RESET is still in flight on that game, the only
other holder once the solver is cancelled; the RESET's return then touches
nothing but its counter.
`Game.keepalive_reset()` sends the RESET through `_execute_action` without
bookkeeping, and only when the game is `playing` and its current level has
no recorded action — there a RESET returns the board already in
`current_state` (`ONLY_RESET_LEVELS` makes it a level reset). It counts the
RESET on `GameRun.keepalive_resets`. A per-game reentrant `_action_lock`,
created in `start_game`, is held by `execute_action`, `finish_game` and
`keepalive_reset`, so a solver move cannot land between the check and the
RESET, and a RESET cannot land on a game being finished; copies get their
own lock. Both Kaggle notebooks set `bm.keepalive_resets` when running as a
submission and pass `clock_start` from `NOTEBOOK_START_EPOCH`.
Tests: `test_keepalive_resets_through_a_blocked_event_loop`,
`test_keepalive_counts_from_clock_start_and_folds_missed_ticks`,
`test_keepalive_stays_quiet_while_the_solver_moves`,
`test_keepalive_halts_when_the_deadline_fires` in
`tests/test_benchmark.py`; `test_keepalive_reset_goes_unseen_on_an_untouched_level`,
`test_keepalive_resets_round_trip_through_json` in `tests/test_game.py`;
`test_game_api_keepalive_reset_leaves_play_identical` in
`tests/test_game_api.py`.

### R2.64 (T) — A solver ends its own run

`Benchmark.stop_early(reason)` is the deadline path on demand: it records
`stop_reason` (in `to_json_dict` and the run summary's `stopped:` line),
sets `_deadline_fired` so `run` swallows the cancellation and returns
normally, halts the keepalive and cancels the solver task, whose games then
land `cancelled` through the solver's own cancellation handling; a no-op
outside the solver-await, and after the first cancel: the deadline,
`request_stop` and `stop_early` all cancel through `_cancel_solver`, which
cancels once per run, so a second stop cannot interrupt the cleanup the
first one started (`test_a_second_stop_during_cleanup_is_a_no_op`). The solver reaches it as `Solver.stop_run`, stamped
on the played copy for the run's duration as a `_StopHook`: harnesses
deep-copy and pickle their solver, so the hook copies by identity and
unpickles to `None` rather than dragging the benchmark along. Test:
`test_a_solver_ends_its_own_run_like_the_deadline_and_the_reason_is_kept`
in `tests/test_benchmark.py`.

### R2.65 (T) — The notebook's image, pinned

`KaggleTarget.docker_image` and the solver hook `kaggle_docker_image` are
resolved by `_resolve_docker_image`, which refuses two different digests,
and the result is written as `docker_image` into `kernel-metadata.json`,
which the Kaggle CLI sends with the push; unnamed, the key is left out and
Kaggle runs its current image. Tests: `test_kaggle_deploy_pins_the_image_the_solver_or_target_names`
and `test_kaggle_deploy_refuses_two_images` in `tests/test_deploy_kaggle.py`.

### R2.66 (T) — Starting the solver before the games

`Benchmark.startup_stealth` switches it on. `run` then calls
`_start_up_first` inside its try, before the game list is final: the
solver's `start_up()` runs as the solver task, so `request_stop`,
`stop_early` and a `_cancel_at` deadline task reach it as they reach play:
a cancellation the deadline or `stop_early` caused is raised as a
`RuntimeError` naming it, and `request_stop`'s re-raises `CancelledError`,
as it does in play. A start-up that held the event loop past the soft
deadline, so that its cancellation never ran, raises the same `RuntimeError`
once it returns. Only then does `run` call the `games_source` it was
given, if any, and check the games and
`game_weights` it returns (`_check_games`, which runs at the top of `run`
when there is no source), deep-copy them and open them, the keepalive
starting with the first. Any failure up to the solver task propagates
after teardown, as a `start_game` failure always has; teardown awaits
`shut_down()` after the keepalive, periodic and deadline tasks are down and
before `teardown()`. Disabled, `run` neither calls `start_up()` nor changes
its order. `run` prints a `[Benchmark.run] startup_stealth:` line as the
solver starts and once it is up. Both Kaggle notebooks set
`bm.startup_stealth` from `KaggleTarget.startup_stealth` (on by default),
print it, and in a rerun wrap `_wait_for_gateway` and
`_competition_games` in `_gateway_games`: passed as `games_source` under
stealth, called before `run` without it.
Tests in `tests/test_startup_stealth.py`:
`test_startup_stealth_opens_and_lists_no_game_before_the_solver_is_up`,
`test_without_startup_stealth_the_games_open_first_and_start_up_is_not_called`,
`test_a_start_up_that_fails_opens_no_game_and_raises`,
`test_the_soft_deadline_bounds_a_start_up_that_never_finishes`,
`test_a_start_up_that_blocks_past_the_soft_deadline_opens_no_game`,
`test_a_game_that_fails_to_open_after_start_up_still_shuts_the_solver_down`,
`test_games_source_is_checked_against_game_weights_once_it_has_listed`,
`test_startup_stealth_sends_the_gateway_nothing_until_the_solver_is_up`
(against the localhost competition gateway, with a positive control that
lists before `run`) and
`test_the_kaggle_notebooks_list_the_gateway_s_games_through_one_function`.

### R2.67 (T) — The startup deadline

`run(startup_deadline=...)`, refused without `startup_stealth`; `KaggleTarget.deploy`
refuses any of the four startup deadline fields (`_STARTUP_DEADLINE_FIELDS`) when
`startup_stealth` is off, before anything is staged. `_start_up_first`
arms a `_StartupKillswitch` before the solver task starts: a daemon thread
that waits on a disarm event until the deadline and then, under the lock
`disarm()` takes, prints a `[taaf.startup_deadline]` line from a thread of
its own, waits for it at most `_EXIT_NOTICE_TIMEOUT_S` (so output nobody
drains cannot hold the exit), and calls
`_exit_process(STARTUP_DEADLINE_EXIT_CODE)` (`os._exit`). The start-up's
`finally` disarms it before anything else, so it never outlives the
start-up, and a solver up past the deadline, which the thread has not yet
ended, raises before `games_source` is called. `KaggleTarget.startup_deadline_s`
carries the allowance (`submission_startup_deadline_s` in a submission) and `KaggleTarget.startup_deadline_after_boot_s` one
counted from machine boot, `submission_startup_deadline_after_boot_s` the one
from boot in a submission, real or simulated. Both notebooks read `/proc/uptime` in their first
cell into `MACHINE_UPTIME_S` and print it; `_startup_deadline` takes the
earlier of `NOTEBOOK_START_EPOCH` plus the first and machine boot
(`NOTEBOOK_START_EPOCH` less the uptime) plus the second, each when the target
sets it and the uptime when /proc gave one, both chosen by the run's
submission mode, and the notebooks pass it. They
write the offline dummy
`submission.parquet` before the run rather than after it, so a version the
killswitch ends still leaves one.
Tests in `tests/test_startup_stealth.py`:
`test_the_startup_killswitch_fires_on_time_with_the_event_loop_blocked`,
`test_the_startup_killswitch_ends_the_process_for_real` (a child process,
unfaked), `test_the_startup_killswitch_exits_while_nobody_drains_its_output`
(a child whose stdout pipe nobody reads),
`test_a_solver_up_before_its_startup_deadline_disarms_the_killswitch`,
`test_a_startup_deadline_already_past_fires_at_once`,
`test_a_startup_deadline_without_startup_stealth_is_refused`,
`test_a_kaggle_target_refuses_a_startup_deadline_without_startup_stealth`, the
notebook check in `test_the_kaggle_notebooks_list_the_gateway_s_games_through_one_function`,
and `test_the_kaggle_notebooks_take_the_earlier_of_the_two_startup_deadlines`
(the notebooks' own `_startup_deadline`).

### R2.31 — Deploy lifecycle (env setup / preamble / run / summary)

Sub-items per the spec:

- **R2.31.1 environment setup.** *Inline:* N/A (R2.22 explicitly says
  "without creating a new environment"). *Slurm:* the rendered sbatch
  script runs `uv sync` against the bundled `main_project` (see R2.37).
  *Kaggle:* the notebook installs ARC runtime wheels from the
  competition input, adds bundled sources to `sys.path`, then runs any
  target/solver setup commands.
- **R2.31.2 preamble (label / solver repr / git status).** All targets
  call `taaf.deploy.format_preamble`. *Inline:* printed directly inside
  the tee. *Slurm:* captured on the launcher (the bundled snapshot
  isn't a git repo) and baked into the rendered worker as a string
  literal so the worker prints it first. *Kaggle:* same launcher-side
  capture, stored in the source dataset as `preamble.txt`. The
  git-status block covers editable repos (clean/DIRTY from the checkout)
  and non-editable Tufa installs (their pinned `vcs_info` commit, shown
  as "pinned") via `_noneditable_tufa_git_lines`.
- **R2.31.3 run the benchmark.** *Inline:* `await benchmark.run(...)`
  in-process. *Slurm:* the rendered worker does `asyncio.run(_amain(...))`.
  *Kaggle:* the rendered notebook loads the pickled benchmark and calls
  `Benchmark.run` directly. `deploy` pushes the kernel with `kaggle
  kernels push`, which **runs it automatically** — there is no
  push-without-run in the Kaggle API, so launching the run is unavoidable
  (an unwanted run is cancelled from the Kaggle UI); only the competition
  leaderboard submission stays manual (R2.24).
- **R2.31.4 brief summary.** Emitted by `Benchmark.run` itself at
  teardown via `diagnostics.run_summary_text` — covers all targets.

Tests: `test_inline_deploy_tees_stdout_to_log` (Inline preamble);
`test_worker_script_embeds_launcher_side_preamble`,
`test_sbatch_script_creates_per_run_venv` (Slurm preamble + env setup),
and `test_kaggle_deploy_dry_run_writes_kernel_and_source_bundles`.

### R2.32 — Soft deadline 10 minutes before kill

Inline: the same clamped buffer as the other targets —
`buffer = min(600s, budget/2)`, `soft_end_time = now + (budget − buffer)`
(a literal 10-minute subtraction lands in the past for short budgets). Slurm:
`worker_soft_end_time` in the rendered worker reads `SLURM_JOB_END_TIME`
and subtracts a buffer clamped to half the remaining walltime (a literal
10-min subtraction lands in the past for short walltimes — verified
regression). Kaggle: the rendered notebook applies the same clamped
10-minute buffer to `KaggleTarget.max_runtime_s`, except when
`RUN_AS_SUBMISSION` is true.
Tests: `test_worker_script_clamps_soft_deadline_to_half_remaining`,
`test_worker_soft_end_uses_experiment_runtime_before_slurm_walltime`,
`test_kaggle_notebook_template_cells_are_valid_python`.

### R2.33 (T) — Graceful stop

Inline: cancelling the awaiting task propagates `CancelledError` into
`Benchmark.run`'s teardown. Slurm: `handle.stop()` is `scancel
--signal=USR1 --batch`; the worker's asyncio SIGUSR1 handler calls
`Benchmark.request_stop`. `--batch` signals the batch step alone, which
`exec python -u` in the work block has made the worker process itself.
The scope matters as much as the signal: `--full` would also signal
every child, and SIGUSR1's default disposition is Terminate, so a
solver's subprocesses (per-game workers, an inference server) are killed
before teardown can run; bare `--signal` skips the batch step, which for
the non-image rendering is where the work runs.
**Gap: container deployments (`image` set) have no graceful stop.** The
batch step is then a shell running `srun`, so `--batch` terminates the
shell without the container's python ever seeing the signal. `--full` did
deliver there — the container's python is a child of the batch step — but it
signals every other child too, which on the non-image path was observed to
kill the ULLM inference server outright; the container path is signalled by
the same rule and so was never graceful either. The failure mode changed,
the capability did not. Covering
it means trapping USR1 in the outer batch shell and forwarding to a
backgrounded `srun` — a different shell from the container work block's
`exec python -u`, which is unaffected. The cost is verification: `srun`'s
forwarding through the container plugin can't be exercised off-cluster.
Kaggle: no graceful stop by spec; `KaggleHandle.stop()` is a no-op.
Tests: `test_handle_stop_runs_scancel_with_sigusr1`,
`test_bash_exec_routes_sigusr1_to_python_directly` (slow),
`test_inline_deploy_external_cancellation_runs_teardown`.

### R2.34 (T) — Job directory

Same launcher-side shape in all targets: `deploy()` raises `ValueError` when
`benchmark.job_dir is None`, runs `taaf.deploy.check_job_dir_unused`
(refuses any directory already holding `deploy_meta.json`), then
`mkdir(parents=True, exist_ok=True)`. *Inline:* in
`InlineTarget.deploy`. *Slurm:* in `TufaSlurmTarget.deploy`. *Kaggle:*
local `job_dir` is staging only; the rendered notebook overrides
`benchmark.job_dir` to `/kaggle/working`.
Tests: `test_inline_deploy_requires_job_dir` /
`test_inline_deploy_refuses_existing_deploy_meta` and the matching
`test_deploy_requires_job_dir` /
`test_deploy_refuses_existing_deploy_meta` in
`tests/test_deploy_slurm.py`, plus
`test_kaggle_deploy_refuses_existing_deploy_meta`.

### R2.35 (T) — Editable-source snapshot

Inline and Slurm call `taaf.deploy.snapshot_editable_sources(job_dir /
"src", ...)`, which discovers every editable distribution via PEP 610
(`*.dist-info/direct_url.json` with `editable: true`) and copies each
repo's `pyproject.toml`, `README.md`, `uv.lock` (when present), and
the importable source tree(s). Per-repo failures raise `RuntimeError`
on the launcher. *Slurm* additionally accepts
`TufaSlurmTarget.extra_source_repos` for forced extra repos. *Kaggle*
uses the same snapshot helper under the source dataset's `src/` and
accepts `KaggleTarget.extra_source_repos`.

Beyond editable repos, the snapshot also bundles **non-editable
distributions installed from a `github.com/Tufalabs` VCS URL**
(`_discover_noneditable_tufa_packages`): there's no checkout, so it
copies the installed top-level package tree(s) at the resolved commit
(e.g. `re_arc` + `pipeline` for `re-arc-3`, including `re_arc`'s bundled
`environment_files/`). These carry only package code — no
`pyproject.toml` / `Makefile` — so a non-editable repo can't be the
Slurm `main_project` (`uv sync` needs a pyproject); it's enough for the
Kaggle worker's `sys.path`. Standard PyPI deps (no Tufa VCS URL) are
left to the target's own install; a same-named editable repo wins.
Editable installs that are **uv-workspace members** collapse to one
snapshot rooted at the workspace root (`_find_repo_root` walks past the
member pyproject when an ancestor's `[tool.uv.workspace]` claims it;
`_snapshot_one_repo` recurses into every non-excluded member), carrying
the root `pyproject.toml` + `uv.lock` so the worker's `uv sync
--all-packages` (see R2.37) resolves `{ workspace = true }` sources; the
Kaggle notebooks add each member's source tree to `sys.path`.

The one opt-out is an explicit exclusion list. `snapshot_editable_sources`
takes `exclude_repos` (matched case-insensitively on the repo directory
name, and applied to the editable and non-editable passes alike);
*Kaggle* resolves it in `_excluded_source_repos` from
`KaggleTarget.exclude_source_repos`, the solver's
`kaggle_exclude_source_repos`, and — for a share target — the variant's
own `re-arc-3` exclusion (R2.61), so the uploaded bundle and
`package_for_local_debug`'s cannot drift apart. This is for a repo the
worker installs by other means: the runner prepends every bundled repo
to `sys.path`, so a bundled copy would shadow the installed one.

A second opt-out drops the non-editable pass wholesale:
`snapshot_editable_sources(..., include_noneditable_tufa=False)`, which
*Slurm* passes. That worker `uv sync`s from the bundled `main_project`'s
lock and so resolves those distributions itself; bundled, they are never
read, and the vLLM fork alone is 730 MB on every job's NFS write.
*Kaggle* keeps them: its worker is offline, and the bundled tree is the
only copy it has.

No test reaches the real installed-package pass by accident: an autouse
fixture in `tests/conftest.py` empties
`_discover_noneditable_tufa_packages`, so a deploy test's bundle holds
only what the test itself stubs in. Without it every deploy test
snapshots whatever Tufa wheels the launcher's venv happens to carry —
the vLLM fork included — into its own `tmp_path`. The editable pass is
deliberately left live; `test_inline_deploy_snapshots_tufalabs_sources`
deploys for real and asserts this repo's checkout lands in the bundle.
Tests: *Inline:* `test_inline_deploy_snapshots_tufalabs_sources`,
`test_find_repo_root_resolves_workspace_member_to_workspace_root`,
`test_snapshot_bundles_whole_uv_workspace`.
*Slurm:* `test_deploy_snapshots_tufalabs_sources`,
`test_deploy_bundles_every_editable_repo_via_auto_discovery`,
`test_deploy_bundles_extra_source_repos`,
`test_slurm_src_tar_omits_installed_tufa_packages`. *Shared:*
`test_discover_editable_repos_*`,
`test_snapshot_editable_sources_raises_on_per_repo_failure`,
`test_snapshot_bundles_noneditable_tufa_package`,
`test_snapshot_can_drop_noneditable_tufa_packages`,
`test_is_tufa_url_matches_tufalabs_only`,
`test_main_project_install_block_actually_installs_bundled_siblings`
(slow, end-to-end uv-sync). *Kaggle:*
`test_kaggle_deploy_dry_run_writes_kernel_and_source_bundles`,
`test_kaggle_excludes_source_repos_from_target_and_solver`,
`test_kaggle_share_exclusion_survives_a_target_exclusion`,
`test_kaggle_bundle_files_land_in_the_payload_directory`,
`test_kaggle_bundle_file_missing_fails_the_deploy`,
`test_kaggle_bundle_files_colliding_by_name_fail`.

### R2.36 (T) — Stdout/stderr to file

*Inline:* `_tee_to_file` duplicates both stdout and stderr writes into
a single `job_dir/stdout.log` while keeping the original streams (so
the run stays interactive). The spec text "to a file" (singular) is
satisfied by one combined file.
*Slurm:* `#SBATCH --output={job_dir}/stdout.log` and
`#SBATCH --error={job_dir}/stderr.log` — two files.
*Kaggle:* the rendered notebook tees stdout/stderr to
`/kaggle/working/stdout.log` outside submission mode.
Tests: `test_inline_deploy_tees_stdout_to_log`,
`test_sbatch_script_routes_output_to_job_dir`,
`test_kaggle_notebook_template_cells_are_valid_python`.

### R2.37 (T) — Per-run venv

Slurm worker runs `uv sync` from `src/<main_project>` with
`UV_PROJECT_ENVIRONMENT=$JOB_DIR/.venv`. `uv sync` is the only command
that honours `[tool.uv.sources]` path rewrites. When the bundled
main_project pyproject declares `[tool.uv.workspace]`, the worker adds
`--all-packages` (a plain sync of a virtual workspace root installs
nothing; test `test_uv_sync_gets_all_packages_flag_for_workspace_main_project`).
Optional `UV_OVERRIDE`
points at a requirements-style file installed `--no-deps` after sync.
The whole thing runs under `flock` on the uv cache so concurrent jobs
on shared NFS don't race. Inline does not satisfy this (see known
gaps). Kaggle has no per-run venv by spec; the notebook runs in the
Kaggle runtime and uses source paths/setup commands instead.
On node-local clusters the cache/venv/temp/src are localized off shared
storage; the block first `unset`s the path vars `--export=ALL` leaks from
the launcher (`UV_CACHE_DIR`, `TMPDIR`, …) so a login-node value can't win
the `${VAR:-...}` fallback and get mkdir'd on unwritable node-local storage.
Localization also covers the base interpreter and the `$HOME`-defaulting
startup caches, not just the venv: `UV_PYTHON_INSTALL_DIR` (so the uv-managed
CPython, whose stdlib is read on every import, lives node-local rather than on
NFS — otherwise cold REPL kernels time out under concurrent startup),
`UV_PYTHON_PREFERENCE=only-managed` (uv never falls back to a shared system
interpreter), `MPLCONFIGDIR`, `XDG_CACHE_HOME`, and `TRITON_CACHE_DIR` (Triton
JIT-compiles kernels during engine warmup and caches them under `$HOME/.triton`
by default — read and written hot, on a slow NFS home, in the run's hottest
startup phase; it does not follow `TMPDIR`, unlike torch's inductor cache, so
it needs its own line). The interpreter and these
caches sit under the persistent node-local cache root (`TAAF_NODE_CACHE_ROOT`),
installed once per node and reused across jobs. A managed interpreter that no
longer runs (its `bin/python3` gone while the directory stays, as a partially
cleared cache leaves it) is removed before the sync, with uv's version aliases
beside it, and the sync reinstalls it; one modified within the last ten minutes
is left alone as a sibling job's install in progress
(`test_worker_removes_a_managed_interpreter_that_no_longer_runs`). The root is
keyed by uid alone
(`/cache/users/$uid`), matching what `job-env.sh` does for `HF_HOME`: wheels and
interpreters are shareable across projects, so a per-project root would cost
each new project a cold download of what a sibling already holds. Only the job /
src / venv roots are project-scoped. Because the `flock` guarding `uv sync`
tracks `$UV_CACHE_DIR`, jobs from different projects on one node serialize their
syncs — they write the same cache, and a warm one keeps each sync short. The
lock wait is bounded (`flock -w 900`) and logged either way: one wedged sync
must not silently consume every sibling job's walltime, and timing out into an
unlocked sync is safe because the venv is per-job and uv's cache does its own
internal locking. A
launcher wanting a different root exports `TAAF_NODE_CACHE_ROOT`, which the
worker deliberately leaves out of the `unset` list. The HF / model caches are
intentionally left to the individual solvers. Redirecting `XDG_CACHE_HOME` does
not silently capture the HF cache on Kander: the sourced `/etc/kander/job-env.sh`
exports an explicit `HF_HOME=/cache/users/$uid/huggingface` (itself node-local,
persistent per node), and an explicit `HF_HOME` takes precedence over the
`$XDG_CACHE_HOME/huggingface` default — so HuggingFace never consults the
redirected `XDG_CACHE_HOME`.
Tests: `test_sbatch_script_creates_per_run_venv`,
`test_main_project_install_block_actually_installs_bundled_siblings`,
`test_sbatch_script_node_local_storage_localizes_uv_and_sources`,
`test_node_local_storage_ignores_leaked_submitter_paths`,
`test_node_local_cache_root_is_per_user_not_per_project`.

### R2.38 — Optional container

Slurm-only per the spec table. *Inline:* N/A.
*Slurm:* `TufaSlurmTarget.image` ∈ docker URI or `.sqsh` path
(pyxis/enroot). When set, the work block runs inside
`srun --container-image=... --container-mounts=... bash -c '...'`.
`extra_mounts` adds further bind mounts; `job_dir` is always mounted.
The work block bootstraps `uv` if absent (containers usually don't
ship it), and afterwards `unset`s a `UV` that names a binary the worker
can't execute: `--export=ALL` and the sourced `/etc/kander/job-env.sh`
both point it at a host toolchain path that isn't among the mounts, and
project Makefiles resolving uv as `UV ?= uv` inherit the dead path.
Tests: `test_sbatch_script_runs_bare_without_image`,
`test_sbatch_script_wraps_in_srun_when_image_set`,
`test_sbatch_script_bootstraps_uv_when_missing`,
`test_worker_drops_uv_binary_that_does_not_resolve`,
`test_sbatch_script_sanitizes_uv_after_sourcing_cluster_job_env`.

### R2.39 — Reattach metadata

All targets call `taaf.deploy.write_deploy_meta`, which writes
`job_dir/deploy_meta.json` (`handle_class`, `job_id`, target config,
host/pid). *Inline:* `job_id=None`; written before the run starts so
even an early crash leaves a reattachable record. *Slurm:* `job_id`
is the parsed sbatch id; written after sbatch submission. *Kaggle:*
`job_id` is the Kaggle kernel id; written after bundle creation and
optional upload.
`DeploymentHandle.attach(job_dir)` reads the metadata and dispatches
to the recorded handle class — `InlineHandle._attach` eagerly loads
`benchmark.json`; `TufaSlurmHandle._attach` and
`KaggleHandle._attach` keep the load lazy.
Tests: *Inline:* `test_inline_deploy_writes_deploy_meta`,
`test_attach_reconstructs_handle_for_finished_inline_run`,
`test_attach_errors_when_no_meta`,
`test_attach_errors_when_no_benchmark_snapshot`. *Slurm:*
`test_deploy_writes_deploy_meta_with_job_id`,
`test_attach_reconstructs_slurm_handle`,
`test_attach_errors_when_meta_missing_job_id`. *Kaggle:*
`test_kaggle_deploy_dry_run_writes_kernel_and_source_bundles`.

---

## 4. Reporting and diagnostics

The implementation in `taaf.diagnostics` ships substantially more than
the spec requires; bonuses are itemised under each requirement and
revisited in the "Beyond the spec" section at the end.

### R4.01 (T) — Per-run HTML

`generate_run_html` (`render_run_html`, written) writes `{job_dir}/diagnostics.html` with the
spec-mandated score-vs-tokens plot, score-vs-wallclock plot, and
per-pass-per-game table (game / pass / score / level / actions /
tokens columns). Curves use the pooled `(game, pass)`-lane formulation
described in the spec (mean-cost-per-lane vs. mean-score-per-lane,
arithmetic-mean decomposable). `total_tokens` (the single token
primitive every total + curve terminal-x funnels through) folds in the
`GameRun.final_generated_tokens` no-move give-up/cancel tokens (R11.03),
and `_total_wallclock` likewise folds in `GameRun.final_wallclock_seconds`
(the finish time — same monotonic-since-`start_game` reference as
per-action `wallclock_seconds`), so both the token and wallclock axes show
the give-up turn everywhere with no per-consumer change; the lone explicit
consumer, `_tokens_vs_wallclock_curve`, adds the tokens at that finish
time directly. `_max_tokens_per_level` deliberately ignores them — a
no-move give-up belongs to no single level.

The cross-game score-vs-wallclock plot is not pooled: `_score_vs_job_wallclock_curve`
places each lane on the job clock (`_job_clock_offset`: `GameRun.started_at`
against `Benchmark.start_time`) and averages the lanes' scores at each moment, a
lane scoring 0 until it starts and holding its last score once it ends. With equal
game weights the curve ends at the run summary's `mean score`; it is extended to the job's end
(elapsed time while in flight). The per-game plots keep game-relative wallclock.

Level completions on the curves come from `actions_per_level`
(`_completions_by_action`: level *k* completes on the action whose index
is the cumulative count through *k*), so a curve renders without
`intermediate_states` loaded. The one lane the counts cannot place is one
that started past a level (R11.07) — its empty bucket falls back to the
states, and is refused when they are absent rather than read as a flat
zero. `uncached_input_tokens` mirrors `total_tokens` on the input side,
and `charged_tokens` / the `input_weight` keyword on
`_per_game_partial_curve` and `lane_score_at_budget` charge uncached
prompt tokens into the token axis at a caller-supplied exchange rate
(zero, the default, is generation-only). An `x_fn` with no cumulative
rule is refused rather than interpolated.

**Bonuses:** per-(game, pass) MP4 movies (idempotent on disk, so the
periodic save accretes them as game-passes finish), final-state
thumbnails in the table, initial-state thumbnails next to game ids,
per-run stats box (mean / σ across passes / SEM) at the top of each
scorer panel via `_per_run_stats_box`, per-game score-vs-tokens scatter,
per-game card grid (one pooled curve per game), per-level token-cap
what-if plot, CSS-only score-variant tab widget (Official ARC /
Weighted-1,2,2,… / Levels-beaten / Games-beaten),
total-generated-tokens-vs-job-wallclock curve, embedded R2.31.2 git
overview (from `job_dir/git_status.txt`).

`GAMES_SCORER` ("Games beaten") is all-or-nothing — 1 for a fully
completed game and 0 for anything less, so its mean is the fraction of
games beaten, which is the figure a leaderboard headline reports.
Registering it in `SCORERS` is all that a scorer needs: the tab widget
and the comparison p-matrices are rendered per entry, and
`test_comparison_pvalue_matrices_for_more_than_two_runs` derives its
expected count from `len(SCORERS)` rather than a hardcoded tally.

Tests: `test_diagnostics_smoke`,
`test_per_game_partial_curve_*`, `test_movies_skipped_when_*`,
`test_the_counts_curve_matches_the_states_curve_it_replaced`,
`test_charging_input_moves_the_budget_but_not_the_scores`,
`test_an_axis_with_no_cumulative_rule_is_refused`,
`test_a_lane_that_started_past_a_level_falls_back_to_its_states`,
`test_the_job_clock_curve_credits_a_queued_game_from_when_it_started`,
`test_score_variants_rendered_as_tabs`,
`test_score_variants_skipped_in_flight`,
`test_in_flight_run_renders_banner_but_blocks_comparison`. Kaggle
submission suppresses this path via `minimal_diagnostics=True`, covered
by `test_minimal_diagnostics_skips_file_writes_and_frame_logging`.

### R4.02 (T) — Per-run summary `.txt`

`generate_run_summary_txt` writes `{job_dir}/summary.txt`: benchmark
label, solver label, totals (games, passes, won-count, mean/median
score, total actions / tokens / wallclock), per-game block. Covered by
`test_diagnostics_smoke`.

### R4.11 (T) — Comparative analysis (2+ runs)

`generate_comparison_html` accepts `list[Benchmark]` (length ≥ 2).
Mismatched game-id sets are reconciled to the intersection in place;
mismatched `game_weights` are unified to the first benchmark's resolved
map. Both surface as a banner. Empty intersection raises rather than
leaving a stale `index.html`.
Tests: `test_comparison_empty_intersection_raises`,
`test_comparison_restricts_to_intersection`,
`test_comparison_harmonizes_weights_to_first`.

### R4.12 (T) — Comparison HTML

`generate_comparison_html` writes `{out_dir}/index.html` (overview, one
pooled curve per run for tokens, job-clock curve per run for wallclock) and
`{out_dir}/per_game/<game>.html` drill-downs (table of every (run,
pass) cell). Per-run stats table (mean / σ across passes / SEM) reports
the winner with 🏆.

**Statistical comparison.** Three p-value tests under the same null
hypothesis (mean per-game score differs):
- Welch's t-test on per-pass mean scores (`scipy.stats.ttest_ind`).
- Weighted paired t-test on per-game mean differences with Kish
  effective N (`_weighted_paired_t_test` — analytic-weight Bessel
  variance, reduces to `scipy.stats.ttest_rel` at uniform weights).
- Weighted paired sign-flip permutation test on per-game differences
  (exact enumeration up to `n=20`, Monte Carlo above).

For exactly two runs these render as a head-to-head box (winner, the
three test lines). For more than two runs the same three tests render as
symmetric NxN matrices of pairwise two-sided p-values
(`_pvalue_matrices_inner` — bold marks p < 0.05, `n.m.` zero-variance,
`n/a` too few passes/games). Each cell carries a direction arrow
(`_direction_arrow`: ← row run higher, ↑ column run higher) pointing at
the better-scoring run, since the symmetric p alone doesn't show which
side won.

**Two scoring variants (since runs may differ in tokens/game).** The
statistics render under a two-level CSS-only tab widget
(`_budget_scorer_tabs_html`): two budget tabs (full runs vs common
budget) above one score-variant tab per `SCORERS` entry, so each
`(budget, scorer)`
panel shows that scorer's stats box (via `_comparison_stats_box`) plus
the budget-independent pooled curves. The "full" variant scores each run
at its endpoint; the "common budget" variant re-scores every
`(game, pass)` lane at a shared per-game token budget. The budget
(`_min_tokens_per_game_budget`) is the smallest run's weighted-mean
tokens per lane — the rightmost x all runs' score-vs-tokens curves
share. It is applied as a parallel cutoff, not a raw per-lane cap:
`ceiling_for_budget` solves for the single ceiling at which the
weighted mean of `min(ceiling, lane_tokens)` over all lanes equals the
budget, so the average lane spends exactly the budget while cheaper lanes
stay uncapped and only the expensive ones are chopped. The per-lane
capped score (`lane_score_at_budget` / `_capped_score_fn`) is read off
the lane's own partial curve at that ceiling; a lane that finished under
the ceiling keeps its terminal score. The stat helpers
(`_run_pass_stats`, `_per_game_mean_scores`, the three tests) take an
optional `score_fn` (`_resolve_score_fn`) selecting endpoint vs capped
scoring. With a single budget mode (no tokens) the widget degrades to
the scorer-only tabs.

Also: head-to-head per-game scatter (2-run only, at endpoints), per-game
card grid linking into the drill-down pages, score-variant tab widget
shared with R4.01, total-generated-tokens-vs-job-wallclock curve (one per
run).

Tests: `test_paired_permutation_test_*`,
`test_weighted_paired_t_*`, `test_weighted_permutation_*`,
`test_comparison_pvalue_matrices_for_more_than_two_runs`,
`test_comparison_common_budget_statistics` in
`tests/test_diagnostics.py`.

**Bonus: regen helpers.** `regenerate_run_diagnostics` and
`regenerate_comparison_diagnostics` accept a mix of run dirs and live
`Benchmark` instances; the per-run helper writes the JSON + sidecar at
`out_dir` so subsequent regens are self-contained. Tested by
`test_regenerate_*`.

The comparison helper also writes `COMPARISON_MEMBERS`
(`comparison_members.json`) naming the member run dirs. A comparison is
built from a hand-picked list, so adding one run to an existing
comparison means rebuilding it from the same members plus the new one;
without the manifest the only record of the membership is folder names
appearing in the rendered HTML. `movie_stem` is public for the same
reason: it is the naming contract for `movies/<stem>.html`, and anything
building or linking those files from outside — the compare-runs skill's
seed combiner writes redirect stubs — has to agree with the renderer
exactly. Tested by `test_comparison_writes_members_manifest` and
`tests/test_compare_runs_scripts.py`.

---

## 5. Trace collection for training

Implemented in `taaf.trace` (`Trace`, `TraceStep`, `MediaRef`, plus the
module functions) with `GameRun.run_id` / `GameRun.register_trace` in
`taaf.game` and `Benchmark.trace_root` / `Benchmark.retain_traces` in
`taaf.benchmark`. The module is pure-Python — it never imports `vllm` /
`torch` / HF / GPU libraries, so it can validate the contract but not
rerun backend roundtrips (R5.62). Tests: `tests/test_trace.py`.

Naming note: the spec text says `to_json_dict()`/`from_json(...)`; the code
mirrors the established `GameRun` pattern `to_json_dict`/`from_json_dict`
for the dataclasses and uses `load_trace(path)` / `export_run_bundle` for
the file-level API (R5.34's "closest existing TAAF naming pattern").

Provider-validation note: R5.31 / R5.62 list `provider_validation` contents as
"at least …" and don't fix field names or a status vocabulary, so the code pins
concrete keys (`REQUIRED_PROVIDER_VALIDATION_KEYS`: `provider_name`,
`provider_version`, `api_kind`, `provider_model_id`, `request_args` =
request-construction arguments, `validation_method` = immutable adapter id,
`original_payload_sha256`, `reconstructed_payload_sha256`,
`input_token_reconstruction_status`, `output_token_reconstruction_status`
(R5.38), `media_reconstruction_status`) and an accepted reconstruction-status
set (`ok`, `whitespace_mismatch`, `mismatch`, `unavailable`,
`not_applicable`); `failed`, missing, and unknown values are rejected.
The full set applies to the output and media statuses only: for
`input_token_reconstruction_status`, `mismatch` / `whitespace_mismatch`
are themselves rejected (R12.02 — an input-token mismatch means the trace
is not reproducible for SFT; there is no input analogue of "`output_text`
stays authoritative").
`unavailable` is the explicit "backend did not expose it" value (R5.44).
`message_format_source` pins the OpenAI OpenAPI subset identifier to an
immutable commit so the normalized schema does not drift with upstream (R5.21
forbids pointing at a moving branch); the schema files are not vendored into
the tree — subset conformance is owned by the capture adapter.

### R5.01 (T) — Lightweight structures + reader/export helpers

`Trace` / `TraceStep` / `MediaRef` dataclasses plus `load_trace`,
`iter_trace_rows`, `export_run_bundle`. Tested by `test_write_and_load_trace_round_trip`,
`test_iter_trace_rows_denormalizes_and_skips_incomplete`,
`test_export_run_bundle_contents_and_manifest`.

### R5.02 (T) — Trace root separate from `job_dir`

`Benchmark.trace_root` (defaults to `job_dir/traces` via
`Benchmark._effective_trace_root`), threaded through `RunSession.trace_root`
onto each `GameRun.trace_root` at `Game.start_game`. Solvers register via
`GameRun.register_trace`. Tested by `test_benchmark_default_trace_root_is_job_dir_traces`.

### R5.11 (T) — `run_id` identity + join key

`GameRun.run_id` (uuid4 hex `default_factory`), serialized in
`GameRun.to_json_dict` so it lands in `benchmark.json` for every run; the
trace folder and `Trace.run_id` reuse it. Run-level fields stay on
`GameRun` and are not copied into trace metadata. Tested by
`test_game_run_has_run_id_and_round_trips`, `test_run_ids_are_unique_per_run`,
`test_benchmark_registers_and_retains_traces`.

### R5.12 (T) — One trace per played run

`register_trace` / `validate_trace` enforce `trace.run_id == game_run.run_id`;
a trace cannot span runs. Tested by `test_validate_trace_rejects_run_id_mismatch`.

### R5.21 (T) — Trace-level metadata

`Trace` carries `schema_version` (`taaf.trace.v2`; `load_trace` still reads
v1 folders), `created_at`, `run_id`, `source_kind`, `n_steps` (derived),
`total_uncached_input_tokens`, repo identity, the
model/tokenizer/processor identity block, `message_format_*` pinned to
the OpenAI OpenAPI subset identifier (`OPENAI_OPENAPI_COMMIT`), and
`content` (`full` or `ids`, R5.39; a header without the key reads as
`full`). Tested by
`test_trace_json_round_trip`, `test_load_trace_reads_v1_traces`,
`test_load_trace_rejects_unknown_schema_version`,
`test_trace_header_without_content_reads_as_full`.

### R5.22 (T) — `source_kind` open string

Validated only as a non-empty string, never against an enum. Tested by
`test_validate_trace_accepts_valid_trace` (`sft_collection`) and
`test_benchmark_registers_and_retains_traces` (`solver_demo`).

### R5.31 (T) — Step rows

`TraceStep` carries every spec field; `action` is an in-memory
`arcengine.ActionInput` serialized as `{"id", "data"}` (mirrors
`GameRun`), `action_index` is the nullable index into `GameRun.history`.
`validate_trace` cross-checks the copied action against
`GameRun.history[action_index]`, and requires a present `logprobs` list to
align one-to-one to `output_token_ids` (`None` — not an empty list — means
alternatives unavailable). Tested by `test_trace_step_json_round_trip`,
`test_validate_trace_rejects_action_mismatch`,
`test_validate_trace_rejects_action_index_out_of_range`,
`test_validate_trace_logprobs_alignment`.

### R5.32 (T) — `messages` / `input_token_ids` / `output_text` separate

Distinct `TraceStep` fields. Tested by `test_trace_step_json_round_trip`.

### R5.33 (T) — `run_id` + `step_index` authoritative

`validate_trace` requires each step's `run_id` to match and `step_index`
to be contiguous from 0. Tested by `test_validate_trace_rejects_non_contiguous_step_index`.

### R5.34 (T) — JSON-serializable via the TAAF pattern

`Trace` / `TraceStep` / `MediaRef` provide `to_json_dict` / `from_json_dict`;
`validate_trace` builds the JSON projection and `json.dumps` it (no pickle).
Tested by `test_trace_json_round_trip`, `test_trace_step_json_round_trip`.

### R5.36 (T) — Reconstruction path without pickle

`load_trace` + `iter_trace_rows` recover trace-owned data from the folder;
`run_id` joins to `GameRun` in `benchmark.json`. Tested by
`test_write_and_load_trace_round_trip`, `test_benchmark_registers_and_retains_traces`.

### R5.37 (T) — Complete `output_text` + structured `output_message`

`TraceStep.output_text` is the full detokenized completion; `validate_trace`
rejects an empty string on steps with non-empty `output_token_ids` (the
empty-completion capture-bug class), while an empty generation (no output
tokens, e.g. instant end-of-turn) stays legitimate; an ids-only trace
(R5.39) is exempt, its `output_text` being empty by definition. Full fidelity evidence
remains `output_token_reconstruction_status` (no re-tokenization in core,
R5.62). Backends that decompose responses store the unnormalized
assistant message as `TraceStep.output_message` (role must be `assistant`;
`None` for plain-text backends and solver decision steps). Tested by
`test_validate_trace_rejects_empty_output_text_with_tokens`,
`test_validate_trace_allows_empty_output_text_for_empty_generation`,
`test_validate_trace_rejects_non_assistant_output_message`,
`test_output_message_json_round_trip`.

### R5.38 (T) — Output-reconstruction evidence

`provider_validation.output_token_reconstruction_status` is required, with
exactly the accepted vocabulary (`ok`, `whitespace_mismatch`,
`mismatch`, `unavailable`, `not_applicable`). `mismatch` is consumer
evidence, not a validity gate — the trace still validates — while `failed`,
unknown, and missing values are rejected. Tested by
`test_validate_trace_rejects_missing_output_status`,
`test_validate_trace_accepts_output_mismatch_status`,
`test_validate_trace_accepts_whitespace_mismatch_status`,
`test_validate_trace_rejects_unknown_output_status`.

### R5.39 (T) — Ids-only traces

`Trace.content` is `CONTENT_FULL` (`"full"`, the default and what a header
without the key reads as) or `CONTENT_IDS` (`"ids"`). `validate_trace`
requires one of the two and, for `ids`, refuses any step carrying
`messages`, `output_text` or `output_message`, applying R5.37's empty-text
rule to `full` traces only. Media refs are validated and written the same
way for both. Tested by
`test_ids_only_trace_registers_and_loads_with_its_media`,
`test_ids_only_trace_refuses_a_step_with_portable_content`,
`test_trace_header_without_content_reads_as_full`.

### R5.41 (T) — Per-trace folder layout

`write_trace` writes `trace.json`, `steps.jsonl`, `media/`, `_SUCCESS`
under `<trace_root>/<run_id>/`. Tested by `test_write_and_load_trace_round_trip`.

### R5.42 / R5.43 (T) — `_SUCCESS` finalization

`_SUCCESS` is written last; `load_trace` raises without it and
`iter_trace_rows` / `export_run_bundle` skip folders lacking it. Tested by
`test_load_trace_requires_success_marker`,
`test_iter_trace_rows_denormalizes_and_skips_incomplete`,
`test_export_run_bundle_skips_incomplete_traces`.

### R5.44 (T) — No inline frame/animation payloads

Media are external files referenced by `MediaRef` (only `media_id`/`path`/
`role`); `validate_trace` requires relative, safe paths and that every
image path in `messages` has a matching ref. `write_trace` stores media by
content — a ref carrying bytes is renamed to `media/<sha256><ext>` (one
path per digest, the first ref's extension), the message parts citing it
follow, byte-less refs at a renamed path follow as aliases, and equal
bytes are written once per trace whatever the producer named them (an
agent attaching the same frame to every prompt otherwise writes it once
per step per step). The rename is planned from the original paths and
applied once per part, then the rewritten projection is re-validated
before `_SUCCESS`; two payloads at one original path are refused. Tested by
`test_validate_trace_rejects_media_path_not_in_refs`,
`test_validate_trace_rejects_absolute_media_path`,
`test_write_and_load_trace_round_trip` (media bytes round-trip),
`test_write_trace_stores_equal_media_bytes_once`,
`test_write_trace_shared_ref_instance_rewrites_every_citing_step`,
`test_write_trace_remaps_a_byteless_ref_aliasing_a_renamed_path`,
`test_write_trace_dedups_by_digest_across_suffixes_and_names`,
`test_write_trace_rejects_two_payloads_at_one_path`.

### R5.45 — Trace payload is source of truth

No manifest/search index is required; `iter_trace_rows` rebuilds any
dataframe by joining trace folders with `benchmark.json`. (Non-`(T)`.)

### R5.46 (T) — Scratch trace root, durable `job_dir`

`trace_root` accepts any path (e.g. `/scratch/jobs/$SLURM_JOB_ID/...`) while
`job_dir` stays durable; nothing is hardcoded. A root may reference
environment variables (`${TAAF_NODE_LOCAL_ROOT}/traces`): the symbolic path
survives launcher-side pickling and `Benchmark._effective_trace_root` expands
it on the worker, raising on an unresolved variable rather than falling back
to durable storage. A symbolic root without `retain_traces=True` raises at
`run()` start — node scratch does not outlive the job. Tested by
`test_benchmark_default_trace_root_is_job_dir_traces`,
`test_benchmark_env_trace_root_resolves_on_worker`,
`test_benchmark_unresolved_trace_root_variable_is_fatal`,
`test_benchmark_rejects_env_trace_root_without_retention`, and the
`trace_root` plumbing in `test_benchmark_registers_and_retains_traces`.

### R5.48 / R5.49 (T) — Retained bundle

`Benchmark.retain_traces` triggers `export_run_bundle` at teardown (after
`benchmark.json` is saved), writing `job_dir/retained_traces/trace_bundle.tar.zst`
with `benchmark.json` at root and `traces/<run_id>/` folders.
`retain_traces=True` together with `minimal_diagnostics=True` raises at the
start of `run()` (R5.48): minimal mode records no run-level metadata for
traces to join to, and exists for Kaggle submission where outputs cannot be
retrieved. Tested by
`test_benchmark_registers_and_retains_traces`,
`test_benchmark_rejects_retain_traces_with_minimal_diagnostics`,
`test_export_run_bundle_contents_and_manifest`.

### R5.50 / R5.51 (T) — Manifest inventory + atomic publish

`export_run_bundle` writes the bundle to a `.tmp` path then renames, and
emits `trace_manifest.json` (one row per retained run) only after the
bundle exists. Tested by `test_export_run_bundle_contents_and_manifest`.

### R5.61 (T) — `GameRun.register_trace`

Validates then writes; returns the folder path; raises (no `_SUCCESS`) on
failure. Tested by `test_register_trace_writes_and_validates`,
`test_register_trace_invalid_writes_no_success`.

### R5.62 (T) — `validate_trace`

Contract-only validation; rejects missing evidence, `failed` statuses,
empty `output_text` alongside output tokens, and misaligned `logprobs`;
no backend imports. Media fingerprints: no producer records them today
(vLLM's HTTP API exposes no preprocessing tensors) —
`media_reconstruction_status` is `unavailable` and token-level media
fidelity is enforced adapter-side via the input-token roundtrip (R5.31
annex) — but when a `provider_validation.media_fingerprints` dict IS
recorded, `_validate_media_fingerprints` checks each record's structure
(`shape` list of non-negative ints, `dtype`, and at least one of
`values` / `sha256`). Tested by the `test_validate_trace_*` cases.

### R5.63 (T) — `write_trace`

`write_trace(trace, game_run, trace_root)` writes all files + `_SUCCESS`
last, and refuses (raising, no `_SUCCESS`) when a referenced media path is
not on disk after the media pass — R5.43's "all referenced media flushed"
holds even for refs without `source_bytes`. Media with bytes are
content-addressed (R5.44), so the media pass costs one file per distinct
image rather than one per reference. Tested by
`test_write_and_load_trace_round_trip`,
`test_write_trace_rejects_missing_media_file`,
`test_write_trace_stores_equal_media_bytes_once`.

### R5.64 (T) — `load_trace`

Reads one completed folder into structured objects without importing
solver code; cross-checks the `steps.jsonl` line count against the stored
`n_steps` so a truncated file inside a `_SUCCESS` folder cannot load
silently. Tested by `test_write_and_load_trace_round_trip`,
`test_load_trace_requires_success_marker`,
`test_load_trace_rejects_truncated_steps`.

### R5.65 (T) — `iter_trace_rows`

Yields one dict per step with trace metadata denormalized (collision keys
prefixed `trace_`); no pandas/Polars/PyArrow dependency. Tested by
`test_iter_trace_rows_denormalizes_and_skips_incomplete`.

### R5.66 (T) — `export_run_bundle`

Durable bundle from `job_dir` + `trace_root`, `_SUCCESS`-only, archived
(zstd) and atomically published. Tested by
`test_export_run_bundle_contents_and_manifest`,
`test_export_run_bundle_skips_incomplete_traces`.

---

## 11. Game and abstraction

### R11.01 (T) — `GameState` class

`GameState` in `taaf.game` — `@dataclass(frozen=True)` wrapping
`arcengine.FrameDataRaw` with the nine spec properties. `game_over` is
true on `WIN` or `GAME_OVER` (ended, won or lost — per the spec text);
`won` on `WIN` only. arcengine's `GAME_OVER` is a lost attempt that
RESET recovers, so it is not terminal for the run.
Tests: `test_game_state_properties`,
`test_game_over_true_on_win_and_engine_game_over`,
`test_example_game_full_play_through`,
`test_frame_draw_returns_axes` in `tests/test_game.py`.

`taaf.diagnostics` exposes the scoring primitives an out-of-tree analysis
needs: `total_tokens`, `uncached_input_tokens`, `charged_tokens`,
`lane_score_at_budget` (with its `input_weight`), `ceiling_for_budget`,
`ceiling_for_lanes` and `weights_by_game_id`. Together they let a caller read
a run at a spend it chooses rather than at the one the run reached, charged
the way its deployment charges. The budget
pair matters together: a budget belongs to the ensemble, and
`ceiling_for_budget` is what turns a fleet-mean budget into the per-lane cutoff
that realises it — `ceiling_for_lanes` is that same solve over explicit
`(spend, weight)` lanes, for a caller whose ensemble is not a whole benchmark,
and `weights_by_game_id` is the weight resolution those lanes need.
Reimplementing any of them elsewhere would be a second thing to keep in
agreement with the scorer.

### R11.02 (T) — `GameRun` class

`GameRun` in `taaf.game` — holds spec fields and the state machine
(`not_started → playing → won/cancelled/gave_up/crashed`).
`pass_index` records which pass copy the run is (R2.2), stamped by
`Benchmark.run`; it defaults to 0, which is what a game played outside a
benchmark is.
`intermediate_states` is dropped from JSON per the spec; persisted to
the `intermediate_states.pkl` sidecar (R2.13 bonus). In memory it is a
`CompressedStates`: each state a zlib-compressed pickle (about 12x smaller
than the frames), rebuilt on access, and snapshotted at `append` so an
engine buffer written into later cannot change the record. `final_generated_tokens`
/ `final_uncached_input_tokens` hold tokens reported on a no-move final
turn (give-up / cancel) via `finish_game(...)`; `final_wallclock_seconds`
stamps the finish time (monotonic elapsed since `start_game`, like
`ActionRecord.wallclock_seconds`) so those tokens land at the right place
on the wallclock axis. `keepalive_resets` counts the benchmark's
invisible RESETs (R2.63). All round-trip through JSON.
Tests: `test_game_run_to_from_json`,
`test_intermediate_states_come_back_as_recorded`,
`test_intermediate_states_are_a_snapshot`,
`test_a_sidecar_saved_as_plain_lists_still_loads`,
`test_items_come_back_equal_and_fresh`,
`test_keep_last_serves_the_appended_object`,
`test_equality_compares_the_items_a_reader_sees` and
`test_items_stored_alike_compare_without_their_own_equality` (`tests/test_support.py`,
the generic `taaf.support.CompressedList` it is built on),
`test_keepalive_resets_round_trip_through_json`,
`test_pass_index_stamped_per_pass` (the JSON leg of `pass_index`),
`test_finish_without_winning_marks_gave_up`,
`test_finish_under_cancellation_marks_cancelled`,
`test_finish_game_records_no_move_tokens`, plus six
`test_score_formula_*` for the ARC-AGI3 score formula.

### R11.03 (T) — Abstract `Game` class

`Game` in `taaf.game` — `@dataclass` with the R11.03 surface.
`start_game` accepts an optional `RunSession` for cross-game resource
sharing (used by `GameAPI`). `execute_action` validates that the action
id is in `available_actions`, that simple actions carry no data, and
that complex actions carry in-bounds `(x, y)`. `finish_game(generated_tokens=0,
uncached_input_tokens=0)` lets the solver report tokens spent on its final
no-move turn; stored on the `GameRun` final_* fields (recorded once — the
idempotency guard skips repeat calls).
Tests: `test_example_game_full_play_through`,
`test_start_game_rejects_unpopulated_subclass`, `test_token_logging`,
`test_finish_game_records_no_move_tokens`, `test_disallowed_action_raises`.

### R11.04 — Subclass contract

`_start_game(session)` / `_execute_action(action)` / `_finish_game()`
hooks on `Game`. Public `execute_action` consumes the token-count
arguments before delegating. `ExampleGame` (R11.06) is the reference
subclass.

### R11.05 (T) — Picklable + deepcopyable until `start_game`

`Game.__getstate__` raises post-start (no opt-in). `Game.__deepcopy__`
raises post-start unless `instance.allow_deepcopy = True` is set
(default False — even subclasses that can support it require explicit
opt-in). The `_enum_reduce` patch in `taaf.game` handles arcengine's
composite-keyed `GameAction` / `GameState` enums.
Tests: `test_pickle_round_trip_before_start`, `test_deepcopy_before_start`,
`test_pickle_after_start_raises`, `test_deepcopy_after_start_raises`,
`test_arcengine_enums_pickle`.

### R11.06 (T) — Example `Game`

`ExampleGame` in `taaf.game_examples` — deterministic two-level
"press-the-sequence" game with 64×64 int8 frames.
Tests: `test_example_game_full_play_through`,
`test_sequence_solver_wins_example_game`.

### R11.07 (T) — Level window (`start_at_level` / `stop_at_level`)

Two kw-only fields on `Game`, both 0-indexed and defining the half-open
window `[start_at_level, stop_at_level)`. `stop_at_level` is enforced
generically in `Game.execute_action`: once the engine reports
`levels_completed >= stop_at_level`, the returned `GameState`'s raw
`state` is forged to `WIN` (the subclass's own engine state is left
untouched, so partial-credit scoring still runs over all levels and a
windowed run cannot reach 100%). `start_at_level` is honored per
subclass inside `_start_game`; `Game.start_game` validates the window
(`_validate_level_window`) and asserts the subclass actually produced an
initial state at the requested level, and seeds `GameRun.levels_completed`
from the initial frame. A window (or basic-info) validation failure after
`_start_game` runs the subclass's `_finish_game` teardown hook before
re-raising, so an invalid window on a started `GameAPI` releases its
scorecard / competition run slot instead of leaking it. `ExampleGame` honors it by seeding its level
counter; `GameAPI` by the engine fast-forward of R11.14.
Tests (in `tests/test_game.py`): `test_stop_at_level_forces_early_win_and_caps_score`,
`test_start_at_level_skips_levels_and_seeds_counter`,
`test_start_and_stop_window_plays_only_inner_levels`,
`test_level_window_validation`,
`test_start_at_level_unhonored_by_subclass_raises`,
`test_invalid_window_start_failure_runs_subclass_teardown`.

### R11.11 (T) — `GameAPI` wrapping env + arcade spec

`GameAPI` in `taaf.game_api` — `@dataclass` subclass of `Game` taking
`env_name` and an optional `ArcadeSpec` (frozen, picklable, hashable
description of how to build an `arc_agi.Arcade`), plus the optional
`external_game_id` override sanctioned by the spec (distinct reported
`game_id` when one environment repeats, e.g. official-110) and the
`Game` window fields of R11.07. The live `Arcade` is
built lazily via `RunSession.get_or_make(spec, ...)`; pre-start
instances are pickle-safe. `ArcadeSpec` defaults to OFFLINE mode
against the re-arc-3 bundle (`environments_dir = "__auto__"` resolved
at `_start_game` time so worker containers find their bundled
snapshot). `ArcadeSpec.competition_sim=True` instead starts a local
competition-arcade simulator in the lazy build and connects in
`COMPETITION` mode — see R11.13.
Tests: `test_game_api_construction_no_network`,
`test_game_api_full_play_through`,
`test_game_api_pickle_before_start_round_trip`,
`test_game_api_deepcopy_before_start`,
`test_game_api_session_caches_arcade`,
`test_game_api_pickle_after_start_raises`.

### R11.12 (T) — Score-vs-scorecard reconciliation

Each `GameAPI` instance creates its own scorecard in `_start_game` and
reconciles in `_finish_game`: pulls this run's score from
`arcade.close_scorecard(...).find_environment(...).runs[0].score`,
compares against `_compute_final_score()`, and fires `warnings.warn` +
stderr print on mismatch. Whole block is try/except-wrapped — must not
raise out of `Benchmark.run`'s teardown `finally`. In `COMPETITION` mode
the per-game path is replaced by one shared scorecard (a
`_CompetitionScorecard` cached on the `RunSession`, closed when the last
game finishes); per-game reconciliation is skipped there since baselines
are hidden. Reconciliation is also skipped (scorecard still closed to
release the slot) when `start_at_level` is set, since skipping levels
makes the engine scorecard non-comparable (R11.14). `stop_at_level`
keeps reconciliation — WIN is forged only on the returned state after
the real step is recorded, so the engine scores the genuine play
identically. A run with `keepalive_resets` is skipped too: the engine
counted those RESETs as actions (R2.63). So is a run with a
`baseline_multiplier` other than 1: it scores against scaled baselines
while the engine scored against the ones it reported (R11.15).
Tests: `test_game_api_score_matches_engine_scorecard`,
`test_game_api_score_matches_engine_scorecard_with_random_walk`,
`test_baseline_multiplier_scales_baselines_and_skips_reconciliation`.

### R11.13 (T) — Competition arcade

`taaf.competition_arcade.CompetitionArcadeServer` starts a localhost
`arc_agi` REST app in competition mode against the bundled offline
environment files. It exposes `/api/games/anonkey`, hides baselines via
the server API, and enforces the competition scorecard behavior that
prevents multiple runs for the same game ID. `GameAPI` shares one
competition scorecard per `RunSession` when its `ArcadeSpec` uses
`OperationMode.COMPETITION`.

`ArcadeSpec(competition_sim=True)` makes the lazy build start this
server on whichever process opens the game and connect in `COMPETITION`
mode, so a non-submission run (local or on the Kaggle worker) exercises
the submission-shaped arcade without a real leaderboard submission. The
server is started once and shared via the `RunSession`, then stopped by
`RunSession.close()` from `Benchmark.run`'s teardown.
`standard_benchmarks.make_benchmark_kaggle_official_110(competition_sim=True)`
wires the spec plus cloned game IDs in one call; the launcher notebook
(`notebooks/kaggle_workflow.ipynb`) defaults to it.
Tests: `tests/test_competition_arcade.py` (including
`test_competition_sim_arcade_spec_autostarts_and_tears_down_server`) and
`test_kaggle_official_110_competition_sim_*` in
`tests/test_standard_benchmarks.py`.

### R11.14 (T) — `GameAPI` start-at-level engine fast-forward

`GameAPI._jump_to_start_level` honors `start_at_level` (R11.07) for
local/offline games. It reaches the live `arcengine.ARCBaseGame` through
arc_agi's `LocalEnvironmentWrapper._game`, sets `game._score = target`
(arcengine's `_score` is `levels_completed`), calls
`game.set_level(target)`, resets the engine state, and re-renders via
`self.env.reset()` (a level-reset, since `ONLY_RESET_LEVELS` is already
set in `_start_game`). Mirrors re-arc-3's web-player jump
(`re_arc/cli.py:_jump_to_level_locked`); no re-arc-3 changes needed.
Online / competition wrappers expose no `_game`, so the jump raises
there. R11.12 reconciliation is skipped for `start_at_level` (but kept
for `stop_at_level`; see R11.12).
Tests: `test_game_api_start_at_level_jumps_engine`,
`test_game_api_stop_at_level_wins_and_reconciles`.

### R11.15 (T) — Baseline multiplier

`Game.baseline_multiplier` (kw-only, default 1.0, positive — checked at
the top of `start_game`, before the subclass touches its engine) scales
the `base_actions_per_level` the subclass reported, rounding to whole
actions (a positive baseline never below one), between `_start_game` and
the `GameRun` it hands them to. One application at the source: every
score — `_compute_final_score`, the `taaf.diagnostics` scorers, the
partial curves — reads the run's baselines, so they cannot disagree
about what a level was worth. The factor rides on the run
(`GameRun.baseline_multiplier`, keyword-only, in the JSON; absent in
older runs, which read as 1.0) so a reader knows what a score was
computed against. `GameAPI` skips R11.12 reconciliation when it is set
(see R11.12).
Tests: `test_baseline_multiplier_scales_what_every_score_is_read_against`,
`test_baseline_multiplier_rounds_to_whole_actions_and_survives_json`,
`test_baseline_multiplier_keeps_a_positive_baseline_positive`,
`test_baseline_multiplier_must_be_positive`,
`test_baseline_multiplier_scales_baselines_and_skips_reconciliation`.

---

## 12. Solver abstraction

### R12.01 (T) — Abstract `Solver` class

`Solver` in `taaf.solver` — `@dataclass` with the public surface:
`label`, `runtime_environment`, `job_dir`, `soft_end_time` (the last
three set by `Benchmark` before `setup()`), `setup()`, `teardown()`,
async `run_games`, and async `start_up()` / `shut_down()` (R2.66), each
once: `shut_down()` runs `_shut_down` only after a `start_up()`.
Tests: `test_solver_picklable_before_setup`,
`test_random_solver_runs_example_game`,
`test_run_propagates_runtime_environment_target_to_solver` (in
`tests/test_benchmark.py`).

### R12.02 — Subclass contract

`_setup` / `_teardown` / `_run_games` hooks, and the optional
`_start_up` / `_shut_down` (default no-op); picklability until
`setup()`. ≥-every-5s yielding and `CancelledError` handling are
documented contracts that the example solvers honour.
Tests (no `(T)` marker on R12.02 itself; covered via R12.03):
`test_random_solver_responds_to_cancel`, `test_solver_yields_promptly`.

### R12.03 (T) — Example random solver

`SolverRandom` in `taaf.solver_examples` — uniform random non-RESET
valid moves; per-game cap; per-game RNG seeded `seed + i` so multi-pass
copies diverge while the run as a whole is reproducible. Synthetic
per-action token counts (uniformly drawn) so the diagnostics tokens
plot has data. `stop_on_error` toggles per-game error isolation.

**Bonus: `click_on_color`** — opt-in flag (default off) that makes a
CLICK target a uniformly-chosen present color's pixel instead of a
uniform pixel; ported from arc-agi-framework so CLICK-heavy games give
real signal. Off by default keeps existing behavior (and the committed
diagnostics examples) unchanged. Covered by
`test_random_solver_click_on_color_targets_present_colors`.

**Bonus: `SolverSequence`** in the same module — fixed action list,
used for deterministic tests.

Tests: `test_random_solver_*`, `test_sequence_solver_*` in
`tests/test_solver.py`.

---

## Beyond the spec

The framework ships several features the spec doesn't require. Most
attach themselves to a numbered requirement and are noted there; the
larger ones, collected:

- **Diagnostics far beyond R4.01 / R4.12.** Movies, score-variant
  tabbed views, per-game cards + drill-downs, head-to-head scatter,
  three statistical tests with weighted paired math, what-if per-level
  token caps, total-generated-tokens-vs-job-wallclock summary curve,
  in-flight banners, harmonization banners, regenerate-from-disk
  helpers, on-disk per-game card cache.
- **Sidecar pickle layer** (R2.13). `intermediate_states.pkl` /
  `games.pkl` / `solver.pkl` split alongside `benchmark.json` so
  diagnostics-only callers don't need user `Game` / `Solver` classes
  importable.
- **`game_weights` on `Benchmark`** (R2.1). Optional per-game weights
  for cross-game aggregation; threads through pooled curves, t-tests,
  per-pass means, and the head-to-head means.
- **`standard_benchmarks`** module. Provides
  `make_benchmark_all_games` — a frozen team-agreed benchmark
  definition (game list + weights) — `make_benchmark_official_25`
  (the 25 official games, one pass, unweighted) and
  `make_benchmark_seal_sweep3_110` (seal sweep 3's game mix as 110
  entries, one pass). Not driven by the spec, but consumed by
  experiments.
- **`Benchmark.request_stop`** (R2.33). Lets a signal handler cancel
  the in-flight solver task cooperatively from outside the run loop;
  used by the Slurm worker's `SIGUSR1` handler.

There are no large batches of orthogonal functionality not in scope of
some numbered requirement — every concrete subsystem (game, solver,
benchmark, deploy, diagnostics) is anchored to a section of the spec.
