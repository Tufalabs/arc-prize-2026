"""``Benchmark`` orchestration. Deployment
lives in ``taaf.deploy*``."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import pickle
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import taaf.game
import taaf.solver
import taaf.support

if TYPE_CHECKING:
    import taaf.deploy  # noqa: TC004 - runtime use behind a cycle-breaking guard


#: The status the startup killswitch exits the process with.
STARTUP_DEADLINE_EXIT_CODE = 3
#: How long the killswitch waits for its exit line to be written before it
#: exits without it: output whose reader has stopped draining must not hold
#: the exit, and a healthy one takes milliseconds.
_EXIT_NOTICE_TIMEOUT_S = 1.0


def _exit_process(code: int) -> None:
    """End the process now, without unwinding: the killswitch's way out, and
    the seam a test replaces."""
    os._exit(code)


class _StartupKillswitch:
    """The startup deadline, enforced: unless disarmed first, exits the
    process the moment ``deadline`` passes. A thread, not a task: the
    start-up it bounds can block the event loop, which is exactly when it
    has to fire. Disarming and firing hold one lock, so a solver that comes
    up as the deadline passes is either let through or ended, never both."""

    def __init__(self, deadline: datetime) -> None:
        self._deadline = deadline
        self._lock = threading.Lock()
        self._disarmed = threading.Event()
        self._thread = threading.Thread(target=self._wait, name="taaf-startup-killswitch", daemon=True)

    def arm(self) -> None:
        """Say when the solver must be up, and start the thread that holds it to that."""
        print(
            f"[taaf.startup_deadline] on: the solver must be up by {self._deadline.isoformat(timespec='seconds')}, "
            f"in {self.remaining_s():.0f}s, or the process exits",
            flush=True,
        )
        self._thread.start()

    def disarm(self) -> None:
        """The solver is up: the thread exits without ending the process."""
        with self._lock:
            self._disarmed.set()

    def remaining_s(self) -> float:
        """Seconds left to the deadline; negative once it has passed."""
        now = datetime.now() if self._deadline.tzinfo is None else datetime.now(UTC)
        return (self._deadline - now).total_seconds()

    def _wait(self) -> None:
        """The thread: wait out the deadline, then end the process unless disarmed."""
        self._disarmed.wait(timeout=max(0.0, self.remaining_s()))
        with self._lock:
            if self._disarmed.is_set():
                return
            # The exit line from a thread of its own, waited for a bounded time.
            notice = threading.Thread(target=self._say_exit, name="taaf-startup-killswitch-notice", daemon=True)
            notice.start()
            notice.join(timeout=_EXIT_NOTICE_TIMEOUT_S)
            _exit_process(STARTUP_DEADLINE_EXIT_CODE)

    @staticmethod
    def _say_exit() -> None:
        """The killswitch's exit line, which blocks for as long as its output's reader does."""
        print(
            "[taaf.startup_deadline] the solver is not up by its startup deadline: exiting the process "
            f"with status {STARTUP_DEADLINE_EXIT_CODE}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()


class _StopHook:
    """``Solver.stop_run`` for one run: ``stop_early`` without the benchmark
    riding along. Harnesses deep-copy and pickle their solver (a reference
    file's provenance, a sidecar), and a bound method would drag the whole
    benchmark into every copy; this shares itself across copies and
    unpickles to ``None``, since a run read back from disk has nothing to
    stop."""

    __slots__ = ("_stop_early",)

    def __init__(self, stop_early: Callable[[str], None]) -> None:
        self._stop_early = stop_early

    def __call__(self, reason: str) -> None:
        self._stop_early(reason)

    def __copy__(self) -> _StopHook:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _StopHook:
        return self

    def __reduce__(self) -> tuple[Callable[[], None], tuple[()]]:
        return (_unpickled_stop_hook, ())


def _unpickled_stop_hook() -> None:
    """What a pickled hook becomes: nothing, the run it belonged to being gone."""
    return


@dataclass
class Benchmark:
    """Benchmark configuration + run state.

    Fields:

    - ``label``: short identifier used in plots and filenames.
    - ``games``: the games to play.
    - ``solver``: the solver instance. Must be set before ``run()``.
    - ``n_passes``: number of independent copies of each game played by
      the solver.
    - ``job_dir``: output directory. ``None`` disables JSON save and
      diagnostics generation entirely — useful for in-process tests.
    - ``game_weights``: optional per-game weights parallel to ``games``.
      ``None`` means equal weights. Weight 0 excludes a game from
      cross-game diagnostics aggregation but keeps it in per-game
      drill-downs.
    - ``game_runs``: populated by ``run()``. Ordered passes-major:
      ``game_runs[pass_idx * n_games + g]`` is pass ``pass_idx`` of game
      ``g``.
    - ``solver_label``: copied from the played solver after ``setup()``.
    - ``start_time`` / ``end_time``: local-time naive ``datetime`` from
      the most recent ``run()``. Round-trip through JSON as ISO-8601.
    - ``periodic_save_interval_s``: how often the periodic save loop
      fires. Default 600 s; tunable for tests.
    - ``trace_root``: where solver traces are written. ``None``
      defaults to ``job_dir/traces``. For trace-heavy jobs point this at
      node-local scratch while ``job_dir`` stays on durable storage;
      the path may reference environment variables the worker defines
      (e.g. ``${TAAF_NODE_LOCAL_ROOT}/traces``), resolved when the root
      is used. A symbolic root requires ``retain_traces=True`` — node
      scratch does not outlive the job.
    - ``retain_traces``: when True and ``job_dir`` is set, a self-contained
      trace bundle is exported to ``job_dir/retained_traces`` at teardown.
    - ``raise_solver_errors``: propagate solver and teardown failures after
      diagnostics, preserving the primary solver error. Defaults to False.
    - ``simulate_submission``: the dress rehearsal. Withholds every answer
      the solver must produce AND forces ``minimal_diagnostics``, because a
      real submission has neither. Simulating only the diagnostics half is
      how a scoring dependency survives to the rerun, where it cannot be
      fixed and produces no logs to read.
    - ``submission_path``: where to write ``submission.json``. Seeded for
      every game before the solver starts and rewritten after every answer,
      so a hard kill leaves something scorable; exempt from
      ``minimal_diagnostics``, which is otherwise what turns every other
      write off. Under ``n_passes > 1`` the file carries the FIRST pass's
      answers alone: a submission has one answer per task, the later
      passes are a measurement's replicates, and which of them the file
      would otherwise keep is an artefact of the order they finished in.
    - ``submission_interval_s``: how often the submission is rewritten. Its
      own cadence, and much tighter than ``periodic_save_interval_s``,
      because the two are bounded by different things: a periodic save
      regenerates diagnostics and is expensive, while a submission rewrite
      is a few hundred kilobytes written atomically. The interval is what a
      hard kill can cost, so it is sized against that rather than against
      the cost of writing.
    - ``periodic_page``: whether a periodic save also renders the diagnostics
      page and its answer sheets. Off, it writes the JSON checkpoint and
      ``summary.txt`` alone, and the page is rendered once, after finishing:
      the render holds the event loop that serves the games, for seconds a
      save on a run of hundreds of games.
    """

    label: str = ""
    games: list[taaf.game.Game] = field(default_factory=lambda: list[taaf.game.Game]())
    solver: taaf.solver.Solver | None = None
    n_passes: int = 1
    job_dir: Path | None = None
    game_weights: list[float] | None = None
    trace_root: Path | None = None
    retain_traces: bool = False
    raise_solver_errors: bool = False
    simulate_submission: bool = False
    submission_path: Path | None = None
    periodic_page: bool = True

    game_runs: list[taaf.game.GameRun] = field(default_factory=lambda: list[taaf.game.GameRun](), init=False)
    solver_label: str = field(default="", init=False)
    start_time: datetime | None = field(default=None, init=False)
    end_time: datetime | None = field(default=None, init=False)
    # Why the run ended early through ``stop_early``; None when it ran to
    # its end or its deadline.
    stop_reason: str | None = field(default=None, init=False)
    periodic_save_interval_s: float = field(default=600.0, init=True, repr=False)
    submission_interval_s: float = field(default=60.0, init=True, repr=False)

    # Internal cancellation flag — lets run() distinguish deadline-fired
    # CancelledError (swallow) from caller-cancellation (re-raise).
    _deadline_fired: bool = field(default=False, init=False, repr=False)
    # Live solver task while run() is awaiting it; lets request_stop()
    # cancel cooperatively from a signal handler. ``compare=False`` to
    # preserve dataclass equality; always None at pickle time.
    _solver_task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False, compare=False)
    # Set once a stop has cancelled the solver task: a second cancel, from
    # any path, would land inside the cleanup the first one started and
    # interrupt it, so every path after the first is a no-op.
    _solver_cancelled: bool = field(default=False, init=False, repr=False)
    # Specs of the copies actually played, in game_runs order. The submission
    # needs every task's shape, including ones no run ever answered.
    _played_specs: list[taaf.game.GameSpec] = field(
        default_factory=lambda: list[taaf.game.GameSpec](), init=False, repr=False, compare=False
    )

    async def run(
        self,
        soft_end_time: datetime | None = None,
        runtime_environment: taaf.deploy.DeploymentTarget | None = None,
        minimal_diagnostics: bool = False,
        startup_deadline: datetime | None = None,
    ) -> None:
        """Run the solver on ``n_passes`` deepcopies of each game.

        ``soft_end_time``: when given, the solver task is
        cancelled at that moment; solvers respond by calling
        ``finish_game()`` which marks each game ``cancelled``.
        Timezone-aware or naive (local clock).

        ``runtime_environment``: the ``DeploymentTarget`` this
        benchmark is running under, stamped on the solver before
        ``setup()``. ``None`` for direct calls outside a target.

        On solver exception: teardown continues, then the exception propagates
        when ``raise_solver_errors`` is True (otherwise it is swallowed). On
        caller-cancellation: teardown then re-raises ``CancelledError``.
        At teardown any game still in ``playing`` is marked ``crashed``,
        JSON + sidecars + diagnostics are written when ``job_dir`` is set.
        ``minimal_diagnostics`` is for Kaggle submission: keep stdout
        status, but skip JSON / sidecar / HTML writes and frame logging.
        Also stamped on the solver so it can trim its own diagnostics.

        ``startup_deadline``: when the solver's ``start_up()``, run in the
        solver task before ``run_games``, must have finished, measured like
        ``soft_end_time`` from the environment's clock. A thread exits the
        process (``os._exit``) the moment it passes with the start-up still
        running, so a blocked event loop cannot hold it. The placeholder
        ``submission.json`` is written by then, but on Kaggle a process that
        dies is recorded as an error, and an error is not scored.
        """
        assert self.solver is not None, "Benchmark.solver must be set before run()"
        assert self.games, "Benchmark.games must be non-empty before run()"
        if self.simulate_submission:
            minimal_diagnostics = True
        if minimal_diagnostics and self.retain_traces:
            raise ValueError(
                "retain_traces=True is incompatible with minimal_diagnostics: minimal mode "
                "records no run-level metadata (no benchmark.json, no action history), so "
                "retained traces would have nothing to join to — and the mode "
                "exists for Kaggle submission, where outputs cannot be retrieved anyway."
            )
        if self.trace_root is not None and "$" in str(self.trace_root) and not self.retain_traces:
            raise ValueError(
                f"trace_root {str(self.trace_root)!r} references an environment variable "
                "(node-local scratch) but retain_traces is False: scratch does not "
                "outlive the job, so every trace would be written and then lost. Set "
                "retain_traces=True or use a durable trace_root."
            )
        if self.game_weights is not None:
            if len(self.game_weights) != len(self.games):
                raise ValueError(
                    f"game_weights length {len(self.game_weights)} must equal games length {len(self.games)}"
                )
            for i, w in enumerate(self.game_weights):
                if w < 0:
                    raise ValueError(f"game_weights[{i}] = {w} must be >= 0")

        self.start_time = datetime.now()
        self.end_time = None
        self.stop_reason = None

        # Capture launcher-side git overview for diagnostics. Skip if a
        # deploy target already wrote it: on a Slurm worker the
        # snapshot venv has no ``.git`` and overwriting would clobber
        # the launcher's good capture.
        if self.job_dir is not None and not (self.job_dir / "git_status.txt").exists():
            from taaf import deploy

            deploy.write_git_status(self.job_dir)

        # Solvers and games are picklable only until
        # setup()/start_game(). Snapshot pristine copies onto ``self``
        # before setup runs; the played copies live in this frame and
        # never enter the pickle. Reassignment doesn't affect the
        # caller's references — they kept the originals.
        self.solver = copy.deepcopy(self.solver)
        self.solver.runtime_environment = runtime_environment
        self.solver.job_dir = self.job_dir
        self.solver.soft_end_time = soft_end_time
        self.solver.minimal_diagnostics = minimal_diagnostics
        played_solver = copy.deepcopy(self.solver)
        # On the played copy alone, and never as a bound method: the pristine
        # copy is what the sidecar pickles, and the played one is copied by
        # harnesses (``_StopHook``).
        played_solver.stop_run = _StopHook(self.stop_early)
        played_solver.setup()
        self.solver_label = played_solver.label

        self.games = [copy.deepcopy(g) for g in self.games]

        # Pass copies of each game to the single solver. The
        # passes-major layout described in the class docstring is set here.
        session = taaf.game.RunSession(
            record_specs=not minimal_diagnostics,
            withhold_answers=self.simulate_submission,
            trace_root=self._effective_trace_root(),
            on_action=self._write_submission,
        )
        to_play: list[taaf.game.Game] = []
        caller_cancelled = False
        solver_error: BaseException | None = None
        solver_task: asyncio.Task[None] | None = None
        periodic_task: asyncio.Task[None] | None = None
        submission_task: asyncio.Task[None] | None = None
        deadline_task: asyncio.Task[None] | None = None
        # Setup runs inside this try so the finally below tears down every
        # game/resource opened here — including when the duplicate-id
        # validation (or an interrupt) fires mid-setup, before the solver runs.
        try:
            for pass_idx in range(self.n_passes):
                for game in self.games:
                    game_copy = copy.deepcopy(game)
                    game_copy.start_game(session)
                    assert game_copy.game_run is not None
                    game_copy.game_run.pass_index = pass_idx
                    self.game_runs.append(game_copy.game_run)
                    to_play.append(game_copy)
                # game_id is populated only by start_game(), so uniqueness
                # can't be checked until the first pass has opened its games;
                # do it before opening any more (later passes legitimately
                # repeat the same ids).
                if pass_idx == 0 and len({r.game_id for r in self.game_runs}) != len(self.games):
                    raise ValueError("duplicate game_ids in benchmark.games are not allowed")

            self._played_specs = [g.spec for g in to_play]
            # Seeded before the solver starts: a hard kill at any moment
            # after this leaves a complete, scorable file rather than none.
            self._write_submission()

            solver_task = asyncio.create_task(self._start_up_then_play(played_solver, to_play, startup_deadline))
            self._solver_task = solver_task
            self._solver_cancelled = False
            if self.job_dir is not None:
                periodic_task = asyncio.create_task(
                    self._minimal_status_loop() if minimal_diagnostics else self._periodic_save_loop()
                )
            if self.submission_path is not None:
                submission_task = asyncio.create_task(self._submission_loop())
            if soft_end_time is not None:
                deadline_task = asyncio.create_task(self._cancel_at(soft_end_time, solver_task))

            try:
                await solver_task
            except asyncio.CancelledError:
                if not self._deadline_fired:
                    caller_cancelled = True
            except Exception as e:
                if self.raise_solver_errors:
                    solver_error = e
                print(
                    f"\n[Benchmark.run] solver task raised "
                    f"{type(e).__name__}; teardown will mark surviving "
                    f"games as crashed. Traceback:",
                    flush=True,
                )
                traceback.print_exc()
        except BaseException as e:
            if self.raise_solver_errors:
                solver_error = e
            raise
        finally:
            try:
                # Clear before teardown so a stray request_stop() can't
                # cancel a stale solver_task. Teardown is uncancellable.
                self._solver_task = None
                played_solver.stop_run = None
                if periodic_task is not None:
                    periodic_task.cancel()
                if submission_task is not None:
                    submission_task.cancel()
                if deadline_task is not None:
                    deadline_task.cancel()
                if periodic_task is not None:
                    await asyncio.gather(periodic_task, return_exceptions=True)
                if submission_task is not None:
                    await asyncio.gather(submission_task, return_exceptions=True)
                if deadline_task is not None:
                    await asyncio.gather(deadline_task, return_exceptions=True)

                # Whatever start_up() brought up that run_games did not
                # release; a no-op for a solver whose start_up() never ran.
                try:
                    await played_solver.shut_down()
                except Exception as e:
                    if self.raise_solver_errors and solver_error is None:
                        solver_error = e
                    traceback.print_exc()
                try:
                    played_solver.teardown()
                except Exception as e:
                    if self.raise_solver_errors and solver_error is None:
                        solver_error = e
                    traceback.print_exc()
                teardown_errors = getattr(played_solver, "teardown_errors", ())
                if self.raise_solver_errors and solver_error is None and teardown_errors:
                    solver_error = RuntimeError("solver teardown failed: " + "; ".join(teardown_errors))

                # ``crashed``: any game still in ``playing`` at
                # teardown. Set ``crashed`` *before* finish_game so the
                # cancelled/gave_up branch is skipped, ``_finish_game`` runs
                # (so a subclass's own teardown still runs), and the finish-line
                # prints the correct state. Keep finalizing remaining games
                # when one fails, preserving the first error in strict mode.
                for game in to_play:
                    run = game.game_run
                    if run is not None and run.state == "playing":
                        run.state = "crashed"
                        try:
                            game.finish_game()
                        except Exception as e:
                            if self.raise_solver_errors and solver_error is None:
                                solver_error = e
                            traceback.print_exc()

                # Stop any per-run resources the session started, now that every
                # game has finished.
                session.close()

                # Before the minimal_diagnostics gate below: the submission is
                # the one output submission mode exists to produce.
                self._write_submission()

                # Stamp end_time before the final save so it lands in JSON.
                self.end_time = datetime.now()
                if not minimal_diagnostics:
                    self._save_json()
                    # After _save_json so the bundle picks up the final
                    # benchmark.json. retain_traces + minimal_diagnostics
                    # is rejected at run() start: minimal mode records nothing to
                    # join traces to.
                    self._export_traces()
                    self._save_game_specs()
                    self._save_games()
                    self._save_solver()
                    self._generate_diagnostics()

                # ``from taaf import diagnostics`` (not ``import
                # taaf.diagnostics``) avoids binding ``taaf`` as a
                # function-local name, which would shadow the module-level
                # ``taaf`` for the rest of run().
                from taaf import diagnostics

                print(diagnostics.run_summary_text(self))
                if self.job_dir is not None and not minimal_diagnostics:
                    # Absolute path so the link is clickable in VSCode.
                    print(f"diagnostics: {(self.job_dir / 'diagnostics.html').absolute()}")

            except asyncio.CancelledError:
                caller_cancelled = True
                raise
            finally:
                if solver_error is not None and not caller_cancelled:
                    raise solver_error

        if caller_cancelled:
            raise asyncio.CancelledError()

    def request_stop(self) -> None:
        """Cancel the in-flight solver task, triggering ``run()``'s
        teardown. Idempotent; no-op outside ``run()``'s solver-await.

        Designed for an asyncio signal handler so external graceful-stop
        paths (e.g. SIGUSR1) reuse the soft-deadline teardown.
        Unlike the deadline path, leaves ``_deadline_fired`` False so
        ``run()`` re-raises ``CancelledError`` after teardown.
        """
        task = self._solver_task
        if task is not None:
            self._cancel_solver(task)

    def _cancel_solver(self, task: asyncio.Task[Any]) -> None:
        """Cancel the solver task once: the deadline, ``request_stop`` and
        ``stop_early`` all come here, and only the first to arrive while the
        task runs cancels it."""
        if self._solver_cancelled or task.done():
            return
        self._solver_cancelled = True
        task.cancel()

    def stop_early(self, reason: str) -> None:
        """End the run now the way the soft deadline would: the solver task
        is cancelled, the games it still holds land ``cancelled``, and
        teardown writes everything. ``reason`` is recorded on the benchmark
        (``stop_reason``, in the JSON and the summary) and printed. Idempotent;
        a no-op outside ``run()``'s solver-await.

        Unlike ``request_stop`` this is the run's own decision, so ``run()``
        returns normally afterwards rather than re-raising the cancellation.
        Reached by a solver through ``Solver.stop_run``.
        """
        task = self._solver_task
        if task is None or task.done() or self._solver_cancelled:
            return
        self.stop_reason = reason
        print(f"[Benchmark.stop_early] {reason}", flush=True)
        self._deadline_fired = True
        self._cancel_solver(task)

    async def deploy(self, target: taaf.deploy.DeploymentTarget) -> taaf.deploy.DeploymentHandle:
        """Set up and run this benchmark in ``target``. Delegates
        to ``target.deploy(self)``. ``async`` so notebook callers can
        ``await bm.deploy(target)`` directly.
        """
        return await target.deploy(self)

    async def _periodic_save_loop(self) -> None:
        """Save JSON + sidecar(s) + diagnostics every
        ``periodic_save_interval_s``, so newly-finished games
        get their answer sheets sweep by sweep. Sidecar pickles other than
        game_specs are written only at final teardown. Also prints the same
        summary text the teardown emits, so long runs leave a periodic
        progress heartbeat in stdout/stderr.

        Each save is rendered here, on the event loop, and written on a
        thread. Rendering reads the runs the solver is still changing, which
        hold still only between the loop's steps; writing is where the time
        goes on a network filesystem, and the event loop is also what serves
        the games -- a save that blocks it stalls every game in the run. The
        time printed is the time the loop was held."""
        from taaf import diagnostics

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                t0 = time.monotonic()
                writes = [self._json_write()]
                summary: str | None = None
                try:
                    # The diagnostics ride behind the JSON checkpoint, never
                    # in front of it: a render that fails (an unreadable
                    # answers directory, say) must not hold up the snapshot a
                    # crashed run is read back from.
                    summary = diagnostics.run_summary_text(self)
                    writes.extend(self._diagnostics_writes(summary))
                except Exception:
                    traceback.print_exc()
                held_s = time.monotonic() - t0
                t1 = time.monotonic()
                await self._write_off_loop(writes)
                saved = "diagnostics" if self.periodic_page else "the summary, the page left to the end,"
                print(
                    f"benchmark: regenerated {saved} in {self.job_dir} in {held_s:.2f}s on the event loop; "
                    f"wrote {len(writes)} files on a thread in {time.monotonic() - t1:.2f}s"
                )
                if summary is not None:
                    print(summary)
            except Exception:
                traceback.print_exc()

    def _diagnostics_writes(self, summary: str) -> list[taaf.support.FileWrite]:
        """A periodic save's diagnostics, rendered: the page with its answer
        sheets unless ``periodic_page`` leaves them to the end, and
        ``summary.txt``."""
        assert self.job_dir is not None, "the periodic save loop runs only with a job_dir"
        from taaf import diagnostics

        page = diagnostics.render_run_html(self, self.job_dir / "diagnostics.html") if self.periodic_page else []
        return [*page, taaf.support.FileWrite(self.job_dir / "summary.txt", summary)]

    async def _write_off_loop(self, writes: list[taaf.support.FileWrite]) -> None:
        """Write on a thread, so the games go on being served while the
        filesystem is slow.

        A save cancelled mid-write still waits for its thread: teardown
        cancels this loop and then writes the same files itself, and a thread
        left running could land its older copy after teardown's."""
        write = asyncio.ensure_future(asyncio.to_thread(taaf.support.write_files, writes))
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            await asyncio.gather(write, return_exceptions=True)
            raise

    async def _submission_loop(self) -> None:
        """Rewrite the submission every ``submission_interval_s``.

        A backstop, not the primary path: every ``execute_action`` writes the
        file already (``RunSession.on_action``), so a hard kill finds the
        standing answers rather than the last sweep's. This covers what that
        cannot — a solver that has stopped answering, and a write that failed
        and left a stale file behind.
        """
        while True:
            await asyncio.sleep(self.submission_interval_s)
            try:
                self._write_submission()
            except Exception:
                traceback.print_exc()

    async def _minimal_status_loop(self) -> None:
        """Kaggle submission heartbeat: stdout status only. The submission
        has its own, tighter loop."""
        from taaf import diagnostics

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                print(diagnostics.run_summary_text(self))
            except Exception:
                traceback.print_exc()

    async def _start_up_then_play(
        self, solver: taaf.solver.Solver, games: list[taaf.game.Game], startup_deadline: datetime | None
    ) -> None:
        """The solver task: ``start_up()``, under the startup killswitch when
        ``startup_deadline`` is set, then ``run_games``. One task, so the soft
        deadline, a stop and a failure treat a start-up as they treat play."""
        killswitch = _StartupKillswitch(startup_deadline) if startup_deadline is not None else None
        if killswitch is not None:
            killswitch.arm()
        t0 = time.monotonic()
        try:
            await solver.start_up()
        except asyncio.CancelledError:
            # What the solver contract asks of run_games on a cancel, for
            # the games it never got: they land cancelled, not crashed.
            for game in games:
                if game.game_run is not None and game.game_run.state == "playing":
                    game.finish_game()
            raise
        finally:
            if killswitch is not None:
                killswitch.disarm()
        print(f"[Benchmark.run] the solver is up after {time.monotonic() - t0:.0f}s", flush=True)
        await solver.run_games(games)

    async def _cancel_at(self, soft_end_time: datetime, target: asyncio.Task[Any]) -> None:
        """Cancel ``target`` once the soft deadline passes."""
        # Match local-naive vs UTC against ``soft_end_time``'s tzinfo.
        if soft_end_time.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(UTC)
        delay = (soft_end_time - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        self._deadline_fired = True
        self._cancel_solver(target)

    def _save_json(self) -> None:
        """Write ``benchmark.json`` atomically."""
        if self.job_dir is None:
            return
        taaf.support.write_files([self._json_write()])

    def _json_write(self) -> taaf.support.FileWrite:
        """``benchmark.json``, rendered and not yet written: the text
        ``atomic_json_dump`` would write. Serialized here rather than on the
        writing thread, since the snapshot shares objects with the live runs."""
        assert self.job_dir is not None
        text = json.dumps(self.to_json_dict(), indent=2, sort_keys=False)
        return taaf.support.FileWrite(self.job_dir / "benchmark.json", text, atomic=True)

    def _effective_trace_root(self) -> Path | None:
        """The trace root for this run: explicit ``trace_root`` if set,
        else ``job_dir/traces``, else ``None`` (no durable dir → no traces).

        An explicit root may reference environment variables (e.g.
        ``${TAAF_NODE_LOCAL_ROOT}/traces``): the benchmark is pickled at
        launch, before the scheduler assigns node scratch, so the path can
        only be named symbolically then. Resolution happens here, in the
        worker. An unresolved variable is fatal — falling back to a durable
        path would silently move the trace write load off scratch."""
        if self.trace_root is None:
            return self.job_dir / "traces" if self.job_dir is not None else None
        expanded = os.path.expandvars(str(self.trace_root))
        if "$" in expanded:
            raise RuntimeError(
                f"trace_root {str(self.trace_root)!r} still contains an unresolved variable "
                f"after expansion ({expanded!r}). The worker environment must define it "
                "(the Slurm preamble exports TAAF_NODE_LOCAL_ROOT on the compute node); "
                "refusing to guess a destination for trace writes."
            )
        return Path(expanded)

    def _export_traces(self) -> None:
        """Export a retained trace bundle at teardown when configured.
        Runs after ``_save_json`` so ``benchmark.json`` is in the bundle
        (retention under minimal diagnostics is rejected at ``run()`` start).
        Must not raise — teardown swallows the original solver exception."""
        if not self.retain_traces or self.job_dir is None:
            return
        trace_root = self._effective_trace_root()
        if trace_root is None:
            return
        from taaf import trace

        out_path = self.job_dir / "retained_traces" / trace.BUNDLE_FILENAME
        try:
            trace.export_run_bundle(self.job_dir, trace_root, out_path)
            print(f"traces: exported retained bundle to {out_path.absolute()}")
        except Exception:
            traceback.print_exc()

    # Per-payload sidecar filenames. The split keeps diagnostics-only
    # loads cheap and free of user `Game` / `Solver` imports — see
    # `from_json` for how each one is attached.
    GAME_SPECS_PKL = "game_specs.pkl"
    GAMES_PKL = "games.pkl"
    SOLVER_PKL = "solver.pkl"

    def _write_submission(self) -> None:
        """Write ``submission.json`` if one was asked for.

        Called before the solver starts, after **every** answer, on the
        backstop sweep, and at teardown. Each write is atomic, because a kill
        landing mid-write would leave truncated non-JSON — strictly worse than
        the complete placeholder file it replaced.

        Writing per answer rebuilds the whole file — O(tasks), tens of
        milliseconds for a 240-task set. That is what buys a hard kill finding
        the standing answers instead of the last sweep's.
        """
        if self.submission_path is None or not self._played_specs:
            return
        from taaf import submission

        # The first pass's answers, and its specs — the first ``len(games)``
        # in the passes-major layout ``run`` builds. ``build`` keys on
        # game_id, so a later pass's run would otherwise replace the first's
        # by list position rather than by any choice.
        first_pass_runs = [run for run in self.game_runs if run.pass_index == 0]
        first_pass_specs = self._played_specs[: len(self.games)]
        submission.write(
            self.submission_path,
            submission.build(first_pass_specs, first_pass_runs),
            first_pass_specs,
        )

    def _save_game_specs(self) -> None:
        """Save each run's task spec to its sidecar.

        Kept out of ``benchmark.json`` because a spec is bulky and already
        reproducible from the dataset; kept at all because the answer sheets
        cannot be rendered without it."""
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(
            [r.spec for r in self.game_runs],
            self.job_dir / self.GAME_SPECS_PKL,
        )

    def _load_game_specs(self) -> None:
        """Attach each run's task spec from the sidecar. Silent no-op
        when the sidecar is missing (in-flight runs, older runs)."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.GAME_SPECS_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            specs_by_run: list[taaf.game.GameSpec | None] = pickle.load(f)
        if len(specs_by_run) != len(self.game_runs):
            raise ValueError(
                f"{path}: runs length {len(specs_by_run)} != benchmark.game_runs "
                f"length {len(self.game_runs)}; sidecar is stale"
            )
        for run, spec in zip(self.game_runs, specs_by_run, strict=True):
            run.spec = spec

    def _save_games(self) -> None:
        """Save the unstarted-original ``games`` list to its sidecar."""
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(self.games, self.job_dir / self.GAMES_PKL)

    def _load_games(self) -> None:
        """Attach ``games`` from the sidecar. Silent no-op when missing.
        Requires the user's ``Game`` subclass to be importable."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.GAMES_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            self.games = pickle.load(f)

    def _save_solver(self) -> None:
        """Save the pristine pre-``setup()`` ``solver`` snapshot."""
        if self.job_dir is None or self.solver is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(self.solver, self.job_dir / self.SOLVER_PKL)

    def _load_solver(self) -> None:
        """Attach ``solver`` from the sidecar. Silent no-op when missing.
        Requires the user's ``Solver`` subclass to be importable."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.SOLVER_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            self.solver = pickle.load(f)

    def _backfill_pass_indices(self) -> None:
        """Stamp ``GameRun.pass_index`` from the passes-major layout, for
        runs loaded from a JSON written before the field existed.

        The game count comes from the distinct ids rather than
        ``len(game_runs) // n_passes``: a run that died while opening its
        games persists a partial list, and dividing by ``n_passes`` there
        assigns every surviving run to pass 0.
        """
        n_games = len({r.game_id for r in self.game_runs})
        if n_games == 0:
            return
        for i, run in enumerate(self.game_runs):
            run.pass_index = i // n_games

    def _generate_diagnostics(self) -> None:
        """Writes per-run HTML + .txt to ``job_dir``. Lazy import
        of ``taaf.diagnostics`` breaks the import cycle (diagnostics
        imports benchmark for type annotations). Prints elapsed time —
        useful for spotting when in-flight diagnostics start dominating
        the periodic-save interval on a large benchmark.
        """
        if self.job_dir is None:
            return
        from taaf import diagnostics

        t0 = time.monotonic()
        diagnostics.generate_run_html(self, self.job_dir / "diagnostics.html")
        diagnostics.generate_run_summary_txt(self, self.job_dir / "summary.txt")
        print(f"benchmark: regenerated diagnostics in {self.job_dir} in {time.monotonic() - t0:.2f}s")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the benchmark snapshot. ``games`` is intentionally
        absent — ``from_json`` rebuilds everything else, and the games are
        the one payload that needs the user's classes importable."""
        return {
            "label": self.label,
            "n_passes": self.n_passes,
            "solver_label": self.solver_label,
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "stop_reason": self.stop_reason,
            "game_weights": list(self.game_weights) if self.game_weights is not None else None,
            "periodic_save_interval_s": self.periodic_save_interval_s,
            "game_runs": [r.to_json_dict() for r in self.game_runs],
        }

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        with_game_specs: bool = True,
        with_games: bool = False,
        with_solver: bool = False,
    ) -> Benchmark:
        """Reconstruct a Benchmark from its JSON snapshot.

        Each ``with_*`` flag attaches a sibling sidecar pickle:

        - ``with_game_specs`` (default True): the task-spec sidecar,
          which the answer sheets need. Safe to leave on (taaf / numpy
          only).
        - ``with_games`` / ``with_solver``: re-run payloads. Require
          the user's ``Game`` / ``Solver`` subclasses to be importable.

        Each load is a silent no-op when the sidecar is missing.
        ``job_dir`` is set to ``path.parent`` so subsequent regen and
        sidecar loaders know where to look.
        """
        with open(path) as f:
            d: dict[str, Any] = json.load(f)
        bm = cls(
            label=d["label"],
            n_passes=d["n_passes"],
            game_weights=list(d["game_weights"]) if d.get("game_weights") is not None else None,
            periodic_save_interval_s=d.get("periodic_save_interval_s", 600.0),
        )
        bm.solver_label = d.get("solver_label", "")
        if d.get("start_time"):
            bm.start_time = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):
            bm.end_time = datetime.fromisoformat(d["end_time"])
        bm.stop_reason = d.get("stop_reason")
        bm.game_runs = [taaf.game.GameRun.from_json_dict(r) for r in d["game_runs"]]
        if any("pass_index" not in r for r in d["game_runs"]):
            # Runs recorded before pass_index existed: recover it from the
            # passes-major layout, so everything downstream can read the field
            # instead of re-deriving the layout for itself.
            bm._backfill_pass_indices()
        bm.job_dir = Path(path).parent
        if with_game_specs:
            bm._load_game_specs()
        if with_games:
            bm._load_games()
        if with_solver:
            bm._load_solver()
        return bm

    @classmethod
    def from_pickle(cls, path: Path) -> Benchmark:
        """Load a legacy monolithic ``benchmark.pkl`` (pre-sidecar runs).
        Back-compat shim only — ``from_json`` is the canonical entry
        point. Requires the user's ``Solver`` / ``Game`` subclasses to
        be importable.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected pickled {cls.__name__}, got {type(obj).__name__}")
        return obj
