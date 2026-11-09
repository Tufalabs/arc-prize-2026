"""``Benchmark`` orchestration (R2.1–R2.15). Deployment (R2.21–R2.39)
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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import taaf.game
import taaf.solver
import taaf.support

if TYPE_CHECKING:
    import taaf.deploy


class _StopHook:
    """``Solver.stop_run`` for one run (R2.64): ``stop_early`` without the
    benchmark riding along. Harnesses deep-copy and pickle their solver (a
    reference file's provenance, a sidecar), and a bound method would drag
    the whole benchmark into every copy; this shares itself across copies
    and unpickles to ``None``, since a run read back from disk has nothing
    to stop."""

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
    """Benchmark configuration + run state (R2.1).

    Fields:

    - ``label``: short identifier used in plots and filenames.
    - ``games``: the games to play.
    - ``solver``: the solver instance. Must be set before ``run()``.
    - ``n_passes``: number of independent copies of each game played by
      the solver (R2.2).
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
      fires. Default 600 s per R2.13; tunable for tests.
    - ``trace_root``: where solver traces are written (R5.02). ``None``
      defaults to ``job_dir/traces``. For trace-heavy jobs point this at
      node-local scratch while ``job_dir`` stays on durable storage (R5.46);
      the path may reference environment variables the worker defines
      (e.g. ``${TAAF_NODE_LOCAL_ROOT}/traces``), resolved when the root
      is used. A symbolic root requires ``retain_traces=True`` — node
      scratch does not outlive the job.
    - ``retain_traces``: when True and ``job_dir`` is set, a self-contained
      trace bundle is exported to ``job_dir/retained_traces`` at teardown
      (R5.48).
    - ``keepalive_resets``: when True, a RESET the solver never sees goes to
      the first game on an untouched level whenever ``keepalive_interval_s``
      passes with no move on any game and no keepalive (R2.63), so an
      environment that ends idle runs cannot end this one before the solver
      plays. Off by default; the Kaggle notebook turns it on in submission.
    - ``keepalive_interval_s``: the longest stretch without an engine action
      the keepalive allows, counted from the last move, the last keepalive
      or the environment's start. The default leaves a third of the Kaggle
      submission's 15-minute idle window as margin.
    - ``periodic_page``: whether a periodic save also renders the diagnostics
      page and its movies (R2.13). Off, it writes the JSON checkpoint and
      ``summary.txt`` alone, and the page is rendered once, after finishing:
      the render holds the event loop that serves the games, for seconds a
      save on a run of hundreds of games.
    - ``startup_stealth``: when True, ``run`` awaits the solver's
      ``start_up()`` before opening any game, and lists the games
      (``run(games_source=...)``) only after it (R2.66), so nothing reaches
      the games' environment until the solver is up. Off by default, when
      the games open first and ``start_up()`` is never called; the Kaggle
      notebooks turn it on for every run unless the target turns it off
      (``KaggleTarget.startup_stealth``).
    """

    label: str = ""
    games: list[taaf.game.Game] = field(default_factory=lambda: list[taaf.game.Game]())
    solver: taaf.solver.Solver | None = None
    n_passes: int = 1
    job_dir: Path | None = None
    game_weights: list[float] | None = None
    trace_root: Path | None = None
    retain_traces: bool = False
    keepalive_resets: bool = False
    keepalive_interval_s: float = 600.0
    periodic_page: bool = True
    startup_stealth: bool = False

    game_runs: list[taaf.game.GameRun] = field(default_factory=lambda: list[taaf.game.GameRun](), init=False)
    solver_label: str = field(default="", init=False)
    start_time: datetime | None = field(default=None, init=False)
    end_time: datetime | None = field(default=None, init=False)
    # Why the run ended early through ``stop_early`` (R2.64); None when it
    # ran to its end or its deadline.
    stop_reason: str | None = field(default=None, init=False)
    periodic_save_interval_s: float = field(default=600.0, init=True, repr=False)

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
    # Live keepalive while run() is going; halted the moment the run is told to
    # stop, since a solver winding down makes no moves and needs no keepalive.
    # Always None at pickle time.
    _keepalive: _Keepalive | None = field(default=None, init=False, repr=False, compare=False)

    async def run(
        self,
        soft_end_time: datetime | None = None,
        runtime_environment: taaf.deploy.DeploymentTarget | None = None,
        minimal_diagnostics: bool = False,
        clock_start: datetime | None = None,
        games_source: Callable[[], list[taaf.game.Game]] | None = None,
        startup_deadline: datetime | None = None,
    ) -> None:
        """Run the solver on ``n_passes`` deepcopies of each game.

        ``soft_end_time`` (R2.12): when given, the solver task is
        cancelled at that moment; solvers respond by calling
        ``finish_game()`` (R12.02) which marks each game ``cancelled``.
        Timezone-aware or naive (local clock).

        ``runtime_environment`` (R12.01): the ``DeploymentTarget`` this
        benchmark is running under, stamped on the solver before
        ``setup()``. ``None`` for direct calls outside a target.

        On solver exception (R2.15): swallowed; teardown continues. On
        caller-cancellation: teardown then re-raises ``CancelledError``.
        At teardown any game still in ``playing`` is marked ``crashed``,
        JSON + sidecars + diagnostics are written when ``job_dir`` is set.
        ``minimal_diagnostics`` is for Kaggle submission: keep stdout
        status, but skip JSON / sidecar / HTML writes and frame logging.
        Also stamped on the solver so it can trim its own diagnostics.

        ``clock_start`` (R2.63): when the environment started, which is what
        an idle limit counts from — on Kaggle, notebook start, before package
        install. Keepalive ticks are counted from it; ``None`` counts from
        this call.

        ``games_source`` (R2.66): when given, replaces ``games`` with what it
        returns, called where the games are opened: after the solver's
        ``start_up()`` under ``startup_stealth``, so listing the games is
        stealthy too. With it, ``games`` and ``game_weights`` are checked once
        it has returned. A ``start_up()`` that fails or does not finish by
        ``soft_end_time`` opens no game and raises out of ``run`` after
        teardown, rather than being swallowed as a solver error (R2.15).

        ``startup_deadline`` (R2.67): under ``startup_stealth``, when the
        solver must be up and ready to play, measured like ``soft_end_time``
        from the environment's clock. A thread exits the process
        (``os._exit``) the moment it passes with the solver still starting,
        so a blocked event loop cannot hold it; a solver up after it raises
        before any game is listed. Refused without ``startup_stealth``.
        """
        assert self.solver is not None, "Benchmark.solver must be set before run()"
        if startup_deadline is not None and not self.startup_stealth:
            raise ValueError(
                "startup_deadline needs startup_stealth: without it the games are opened before the "
                "solver starts up, so a deadline on the start-up could no longer keep them closed"
            )
        if minimal_diagnostics and self.retain_traces:
            raise ValueError(
                "retain_traces=True is incompatible with minimal_diagnostics: minimal mode "
                "records no run-level metadata (no benchmark.json, no action history), so "
                "retained traces would have nothing to join to (R5.48/R5.11) — and the mode "
                "exists for Kaggle submission, where outputs cannot be retrieved anyway."
            )
        if self.trace_root is not None and "$" in str(self.trace_root) and not self.retain_traces:
            raise ValueError(
                f"trace_root {str(self.trace_root)!r} references an environment variable "
                "(node-local scratch, R5.46) but retain_traces is False: scratch does not "
                "outlive the job, so every trace would be written and then lost. Set "
                "retain_traces=True (R5.48) or use a durable trace_root."
            )
        if games_source is None:
            self._check_games()

        started = datetime.now()
        self.start_time = started
        self.end_time = None
        self.stop_reason = None

        # Capture launcher-side git overview for diagnostics. Skip if a
        # deploy target already wrote it: on a Slurm worker the
        # snapshot venv has no ``.git`` and overwriting would clobber
        # the launcher's good capture.
        if self.job_dir is not None and not (self.job_dir / "git_status.txt").exists():
            from taaf import deploy  # noqa: PLC0415

            deploy.write_git_status(self.job_dir)

        # R12.02 / R11.05: solvers and games are picklable only until
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

        # R2.2: pass copies of each game to the single solver. The
        # passes-major layout described in the class docstring is set here.
        session = taaf.game.RunSession(
            record_intermediate_states=not minimal_diagnostics,
            trace_root=self._effective_trace_root(),
        )
        to_play: list[taaf.game.Game] = []
        keepalive: _Keepalive | None = None
        caller_cancelled = False
        solver_task: asyncio.Task[None] | None = None
        periodic_task: asyncio.Task[None] | None = None
        deadline_task: asyncio.Task[None] | None = None
        # Setup runs inside this try so the finally below tears down every
        # game/resource opened here — including when the duplicate-id
        # validation (or an interrupt) fires mid-setup, before the solver runs.
        try:
            if self.startup_stealth:
                await self._start_up_first(played_solver, soft_end_time, startup_deadline)
            if games_source is not None:
                self.games = list(games_source())
                self._check_games()
            self.games = [copy.deepcopy(g) for g in self.games]
            for pass_idx in range(self.n_passes):
                for game in self.games:
                    game_copy = copy.deepcopy(game)
                    game_copy.start_game(session)
                    assert game_copy.game_run is not None
                    game_copy.game_run.pass_index = pass_idx
                    self.game_runs.append(game_copy.game_run)
                    to_play.append(game_copy)
                    # From the first started game: starting the rest is itself
                    # a stretch without moves.
                    if self.keepalive_resets and keepalive is None:
                        keepalive = _Keepalive(to_play, clock_start or started, self.keepalive_interval_s)
                        self._keepalive = keepalive
                        keepalive.start()
                # game_id is populated only by start_game(), so uniqueness
                # can't be checked until the first pass has opened its games;
                # do it before opening any more (later passes legitimately
                # repeat the same ids).
                if pass_idx == 0 and len({r.game_id for r in self.game_runs}) != len(self.games):
                    raise ValueError("duplicate game_ids in benchmark.games are not allowed")

            solver_task = asyncio.create_task(played_solver.run_games(to_play))
            self._solver_task = solver_task
            self._solver_cancelled = False
            if self.job_dir is not None:
                periodic_task = asyncio.create_task(
                    self._minimal_status_loop() if minimal_diagnostics else self._periodic_save_loop()
                )
            if soft_end_time is not None:
                deadline_task = asyncio.create_task(self._cancel_at(soft_end_time, solver_task))

            try:
                await solver_task
            except asyncio.CancelledError:
                if not self._deadline_fired:
                    caller_cancelled = True
            except Exception as e:
                # R2.15: solver errors are not fatal. Narrowed from
                # BaseException so KeyboardInterrupt / SystemExit /
                # GeneratorExit still propagate after teardown.
                print(
                    f"\n[Benchmark.run] solver task raised "
                    f"{type(e).__name__}; teardown will mark surviving "
                    f"games as crashed. Traceback:",
                    flush=True,
                )
                traceback.print_exc()
        finally:
            # Before anything finishes a game: a keepalive RESET must not land
            # on a game teardown is closing.
            if keepalive is not None:
                keepalive.stop()
            self._keepalive = None
            # Clear before teardown so a stray request_stop() can't
            # cancel a stale solver_task. Teardown is uncancellable.
            self._solver_task = None
            played_solver.stop_run = None
            if periodic_task is not None:
                periodic_task.cancel()
            if deadline_task is not None:
                deadline_task.cancel()
            if periodic_task is not None:
                await asyncio.gather(periodic_task, return_exceptions=True)
            if deadline_task is not None:
                await asyncio.gather(deadline_task, return_exceptions=True)

            # Whatever start_up() brought up that run_games did not release
            # (R2.66); a no-op for a solver whose start_up() never ran.
            try:
                await played_solver.shut_down()
            except Exception:
                traceback.print_exc()
            try:
                played_solver.teardown()
            except Exception:
                traceback.print_exc()

            # R11.02 ``crashed``: any game still in ``playing`` at
            # teardown. Set ``crashed`` *before* finish_game so the
            # cancelled/gave_up branch is skipped, ``_finish_game`` runs
            # (R11.12 reconciliation in GameAPI), and the finish-line
            # prints the correct state. The finish-line print must not
            # raise — we're already swallowing the original solver
            # exception here, and an unhandled raise would mask it.
            for game in to_play:
                run = game.game_run
                if run is not None and run.state == "playing":
                    run.state = "crashed"
                    try:
                        game.finish_game()
                    except Exception:
                        traceback.print_exc()
                    if run.final_score is None:
                        run.final_score = run._compute_final_score()

            # Stop any per-run resources the session started (e.g. a local
            # competition-arcade server) now that every game has finished.
            session.close()

            # Stamp end_time before the final save so it lands in JSON.
            self.end_time = datetime.now()
            if not minimal_diagnostics:
                self._save_json()
                # After _save_json so the bundle picks up the final
                # benchmark.json (R5.48). retain_traces + minimal_diagnostics
                # is rejected at run() start: minimal mode records nothing to
                # join traces to.
                self._export_traces()
                self._save_intermediate_states()
                self._save_games()
                self._save_solver()
                self._generate_diagnostics()

            # ``from taaf import diagnostics`` (not ``import
            # taaf.diagnostics``) avoids binding ``taaf`` as a
            # function-local name, which would shadow the module-level
            # ``taaf`` for the rest of run().
            from taaf import diagnostics  # noqa: PLC0415

            print(diagnostics.run_summary_text(self))
            if self.job_dir is not None and not minimal_diagnostics:
                # Absolute path so the link is clickable in VSCode.
                print(f"diagnostics: {(self.job_dir / 'diagnostics.html').absolute()}")

        if caller_cancelled:
            raise asyncio.CancelledError()

    def request_stop(self) -> None:
        """Cancel the in-flight solver task, triggering ``run()``'s
        teardown. Idempotent; no-op outside ``run()``'s solver-await.

        Designed for an asyncio signal handler so external graceful-stop
        paths (R2.33, e.g. SIGUSR1) reuse the soft-deadline teardown.
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
        if self._keepalive is not None:
            self._keepalive.halt()
        task.cancel()

    def stop_early(self, reason: str) -> None:
        """End the run now the way the soft deadline would (R2.64): the solver
        task is cancelled, the games it still holds land ``cancelled``, and
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
        """Set up and run this benchmark in ``target`` (R2.21). Delegates
        to ``target.deploy(self)``. ``async`` so notebook callers can
        ``await bm.deploy(target)`` directly.
        """
        return await target.deploy(self)

    async def _periodic_save_loop(self) -> None:
        """Save JSON + sidecar(s) + diagnostics every
        ``periodic_save_interval_s`` (R2.13). Movie rendering is
        idempotent on disk, so newly-finished games accumulate movies
        sweep by sweep. Sidecar pickles other than intermediate_states
        are written only at final teardown. Also prints the same
        summary text the teardown emits, so long runs leave a periodic
        progress heartbeat in stdout/stderr.

        Each save is rendered here, on the event loop, and written on a
        thread. Rendering reads the runs the solver is still changing, which
        hold still only between the loop's steps; writing is where the time
        goes on a network filesystem, and the event loop is also what serves
        the games -- a save that blocks it stalls every game in the run. The
        time printed is the time the loop was held."""
        from taaf import diagnostics  # noqa: PLC0415

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                t0 = time.monotonic()
                writes = [self._json_write()]
                summary: str | None = None
                try:
                    # The diagnostics ride behind the JSON checkpoint, never
                    # in front of it: a render that fails (an unreadable
                    # movies directory, say) must not hold up the snapshot a
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
        """A periodic save's diagnostics, rendered: the page with its movie
        players unless ``periodic_page`` leaves them to the end, and
        ``summary.txt``."""
        assert self.job_dir is not None, "the periodic save loop runs only with a job_dir"
        from taaf import diagnostics  # noqa: PLC0415

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

    async def _minimal_status_loop(self) -> None:
        """Kaggle submission heartbeat: no writes, just stdout status."""
        from taaf import diagnostics  # noqa: PLC0415

        while True:
            await asyncio.sleep(self.periodic_save_interval_s)
            try:
                print(diagnostics.run_summary_text(self))
            except Exception:
                traceback.print_exc()

    def _check_games(self) -> None:
        """The game list ``run`` is about to play, and ``game_weights`` against it."""
        assert self.games, "Benchmark.games must be non-empty before run()"
        if self.game_weights is not None:
            if len(self.game_weights) != len(self.games):
                raise ValueError(
                    f"game_weights length {len(self.game_weights)} must equal games length {len(self.games)}"
                )
            for i, w in enumerate(self.game_weights):
                if w < 0:
                    raise ValueError(f"game_weights[{i}] = {w} must be >= 0")

    async def _start_up_first(
        self, solver: taaf.solver.Solver, soft_end_time: datetime | None, startup_deadline: datetime | None
    ) -> None:
        """Bring the solver up before any game is listed or opened (R2.66), so
        a solver that never comes up has touched nothing of the games'
        environment. Under the soft deadline, ``request_stop`` and
        ``stop_early``, as the play after it is; a failure, the deadline or a
        stop raises out of ``run`` with no game opened. Under
        ``startup_deadline`` the killswitch is armed for the start-up alone
        (R2.67)."""
        print("[Benchmark.run] startup_stealth: starting the solver before any game is opened", flush=True)
        t0 = time.monotonic()
        killswitch = _StartupKillswitch(startup_deadline) if startup_deadline is not None else None
        if killswitch is not None:
            killswitch.arm()
        task = asyncio.create_task(solver.start_up())
        self._solver_task = task
        self._solver_cancelled = False
        deadline = asyncio.create_task(self._cancel_at(soft_end_time, task)) if soft_end_time is not None else None
        try:
            await task
        except asyncio.CancelledError:
            if not self._deadline_fired:
                raise
            reason = self.stop_reason or "the soft deadline passed"
            raise RuntimeError(
                f"startup_stealth: {reason} while the solver was starting up; no game was opened"
            ) from None
        finally:
            if killswitch is not None:
                killswitch.disarm()
            self._solver_task = None
            if deadline is not None:
                deadline.cancel()
                await asyncio.gather(deadline, return_exceptions=True)
        if killswitch is not None and killswitch.remaining_s() <= 0:
            raise RuntimeError("startup_deadline: the solver came up after its startup deadline; no game was opened")
        # A start-up that held the event loop past the soft deadline kept its
        # cancellation from running, so the clock is read here too.
        if soft_end_time is not None and _now_like(soft_end_time) >= soft_end_time:
            raise RuntimeError(
                "startup_stealth: the soft deadline passed while the solver was starting up; no game was opened"
            )
        print(
            f"[Benchmark.run] startup_stealth: the solver is up after {time.monotonic() - t0:.0f}s; opening the games",
            flush=True,
        )

    async def _cancel_at(self, soft_end_time: datetime, target: asyncio.Task[Any]) -> None:
        # Match local-naive vs UTC against ``soft_end_time``'s tzinfo.
        if soft_end_time.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(timezone.utc)
        delay = (soft_end_time - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        self._deadline_fired = True
        self._cancel_solver(target)

    def _save_json(self) -> None:
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
        """The trace root for this run (R5.02): explicit ``trace_root`` if set,
        else ``job_dir/traces``, else ``None`` (no durable dir → no traces).

        An explicit root may reference environment variables (R5.46, e.g.
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
        """Export a retained trace bundle at teardown when configured (R5.48).
        Runs after ``_save_json`` so ``benchmark.json`` is in the bundle
        (retention under minimal diagnostics is rejected at ``run()`` start).
        Must not raise — teardown swallows the original solver exception."""
        if not self.retain_traces or self.job_dir is None:
            return
        trace_root = self._effective_trace_root()
        if trace_root is None:
            return
        from taaf import trace  # noqa: PLC0415

        out_path = self.job_dir / "retained_traces" / trace.BUNDLE_FILENAME
        try:
            trace.export_run_bundle(self.job_dir, trace_root, out_path)
            print(f"traces: exported retained bundle to {out_path.absolute()}")
        except Exception:
            traceback.print_exc()

    # Per-payload sidecar filenames. The split keeps diagnostics-only
    # loads cheap and free of user `Game` / `Solver` imports — see
    # `from_json` for how each one is attached.
    INTERMEDIATE_STATES_PKL = "intermediate_states.pkl"
    GAMES_PKL = "games.pkl"
    SOLVER_PKL = "solver.pkl"

    def _save_intermediate_states(self) -> None:
        """Save ``intermediate_states`` to its sidecar. Called from
        teardown only — the frames are large and the periodic loop
        doesn't need them."""
        if self.job_dir is None:
            return
        self.job_dir.mkdir(parents=True, exist_ok=True)
        taaf.support.atomic_pickle_dump(
            [r.intermediate_states for r in self.game_runs],
            self.job_dir / self.INTERMEDIATE_STATES_PKL,
        )

    def _load_intermediate_states(self) -> None:
        """Attach ``intermediate_states`` from the sidecar. Silent no-op
        when the sidecar is missing (in-flight runs, older runs)."""
        if self.job_dir is None:
            return
        path = self.job_dir / self.INTERMEDIATE_STATES_PKL
        if not path.exists():
            return
        with open(path, "rb") as f:
            # A CompressedStates per run, or a plain list from a run saved
            # before states were held compressed.
            states_by_run: list[Iterable[taaf.game.GameState]] = pickle.load(f)
        if len(states_by_run) != len(self.game_runs):
            raise ValueError(
                f"{path}: runs length {len(states_by_run)} != benchmark.game_runs "
                f"length {len(self.game_runs)}; sidecar is stale"
            )
        for run, states in zip(self.game_runs, states_by_run, strict=True):
            run.intermediate_states = taaf.game.CompressedStates(states)

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

        That derivation assumes ids are unique within a pass. Archives
        recorded before ``Benchmark.run`` asserted uniqueness repeat them,
        so the distinct count under-counts games and the stamp overruns
        ``n_passes`` — raise there rather than stamp, because every
        consumer iterates ``range(n_passes)`` and would drop the surplus
        runs from every statistic with no error. Divisibility is
        deliberately not checked: a partial list is the legitimate case
        above.
        """
        n_games = len({r.game_id for r in self.game_runs})
        if n_games == 0:
            return
        for i, run in enumerate(self.game_runs):
            run.pass_index = i // n_games
        highest = self.game_runs[-1].pass_index
        if highest >= self.n_passes:
            raise ValueError(
                f"cannot recover pass_index: {len(self.game_runs)} runs over {n_games} distinct "
                f"game ids implies pass {highest}, but n_passes is {self.n_passes}. This archive "
                f"predates unique game ids, so the passes-major layout no longer identifies them."
            )

    def _generate_diagnostics(self) -> None:
        """R2.13: writes per-run HTML + .txt to ``job_dir``. Lazy import
        of ``taaf.diagnostics`` breaks the import cycle (diagnostics
        imports benchmark for type annotations). Prints elapsed time —
        useful for spotting when in-flight diagnostics start dominating
        the periodic-save interval on a large benchmark.
        """
        if self.job_dir is None:
            return
        from taaf import diagnostics  # noqa: PLC0415

        t0 = time.monotonic()
        diagnostics.generate_run_html(self, self.job_dir / "diagnostics.html")
        diagnostics.generate_run_summary_txt(self, self.job_dir / "summary.txt")
        print(f"benchmark: regenerated diagnostics in {self.job_dir} in {time.monotonic() - t0:.2f}s")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the benchmark snapshot. ``games`` is intentionally
        absent per R2.14 ("reconstructs the full class except the game
        list")."""
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
        with_intermediate_states: bool = True,
        with_games: bool = False,
        with_solver: bool = False,
    ) -> Benchmark:
        """Reconstruct a Benchmark from its JSON snapshot (R2.14).

        Each ``with_*`` flag attaches a sibling sidecar pickle:

        - ``with_intermediate_states`` (default True): the per-step
          frames sidecar. Safe to leave on (taaf / arcengine / numpy
          only) and needed for movies + level-step curves.
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
        if with_intermediate_states:
            bm._load_intermediate_states()
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

        Backfills ``pass_index`` like :meth:`from_json`, detected on the
        instance dict rather than a missing key: an unpickled pre-field
        ``GameRun`` reads the class default 0, so the field looks present
        and every pass but 0 would come back empty.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected pickled {cls.__name__}, got {type(obj).__name__}")
        if any("pass_index" not in r.__dict__ for r in obj.game_runs):
            obj._backfill_pass_indices()
        return obj


#: The status the startup killswitch exits the process with (R2.67).
STARTUP_DEADLINE_EXIT_CODE = 3
#: How long the killswitch waits for its exit line to be written before it
#: exits without it: output whose reader has stopped draining must not hold
#: the exit, and a healthy one takes milliseconds (R2.67).
_EXIT_NOTICE_TIMEOUT_S = 1.0


def _now_like(when: datetime) -> datetime:
    """Now, local-naive or UTC as ``when`` is, so the two compare."""
    return datetime.now() if when.tzinfo is None else datetime.now(timezone.utc)


def _exit_process(code: int) -> None:
    """End the process now, without unwinding: the killswitch's way out
    (R2.67), and the seam a test replaces."""
    os._exit(code)


class _StartupKillswitch:
    """The startup deadline, enforced (R2.67): unless disarmed first, exits
    the process the moment ``deadline`` passes. A thread, not a task: the
    start-up it bounds can block the event loop, which is exactly when it
    has to fire. Disarming and firing hold one lock, so a solver that comes
    up as the deadline passes is either let through or ended, never both."""

    def __init__(self, deadline: datetime) -> None:
        self._deadline = deadline
        self._lock = threading.Lock()
        self._disarmed = threading.Event()
        self._thread = threading.Thread(target=self._wait, name="taaf-startup-killswitch", daemon=True)

    def arm(self) -> None:
        print(
            f"[taaf.startup_deadline] on: the solver must be up by {self._deadline.isoformat(timespec='seconds')}, "
            f"in {self.remaining_s():.0f}s, or the process exits",
            flush=True,
        )
        self._thread.start()

    def disarm(self) -> None:
        with self._lock:
            self._disarmed.set()

    def remaining_s(self) -> float:
        now = datetime.now() if self._deadline.tzinfo is None else datetime.now(timezone.utc)
        return (self._deadline - now).total_seconds()

    def _wait(self) -> None:
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
        print(
            "[taaf.startup_deadline] the solver is not up by its startup deadline: exiting the process "
            f"with status {STARTUP_DEADLINE_EXIT_CODE}, no game opened",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()


class _Keepalive:
    """Keepalive RESETs for ``Benchmark.run`` (R2.63).

    A thread rather than a task: the stretch it exists for is typically a
    solver's own startup, and a synchronous engine build blocks the event
    loop for all of it, so a task would never be scheduled when it is
    needed. The clock runs from the last engine action it knows of -- a
    recorded move, a keepalive it sent, or the environment's start -- and a
    RESET goes out once the interval has passed since then. Bounded from
    the action rather than on a fixed tick, so a move landing just after a
    tick cannot stretch the gap to two intervals: the idle limit counts
    from the last action, and so does this.
    """

    def __init__(self, games: list[taaf.game.Game], clock_start: datetime, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError(f"keepalive_interval_s must be positive, got {interval_s}")
        # The run's own list, still filling while games start; read by copy.
        self._games = games
        self._clock_start = clock_start
        self._interval_s = interval_s
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="taaf-keepalive", daemon=True)

    def start(self) -> None:
        print(
            f"[taaf.keepalive] on: RESET every {self._interval_s:g}s without a move, "
            f"counted from {self._clock_start:%H:%M:%S}",
            flush=True,
        )
        self._thread.start()

    def halt(self) -> None:
        """Send nothing further. Non-blocking, for callers on the event loop;
        a RESET already in flight completes."""
        self._stopped.set()

    def stop(self) -> None:
        self.halt()
        # Bounded: an in-flight RESET is one engine call, and the thread is a
        # daemon, so a hung one cannot hold teardown hostage.
        self._thread.join(timeout=30.0)

    def _elapsed(self) -> float:
        return (datetime.now(self._clock_start.tzinfo) - self._clock_start).total_seconds()

    def _moves(self) -> int:
        return sum(len(game.game_run.history) for game in list(self._games) if game.game_run is not None)

    def _loop(self) -> None:
        # A move is noticed within one poll of landing, so the gap between two
        # engine actions is at most the interval plus a poll; a tenth of the
        # interval keeps that margin, capped so a long interval still polls.
        poll_s = min(self._interval_s / 10.0, 60.0)
        seen = self._moves()
        last_action_s = 0.0  # elapsed since clock_start, which counts as an action
        while True:
            moves = self._moves()
            if moves != seen:
                seen = moves
                last_action_s = self._elapsed()
            elif self._elapsed() - last_action_s >= self._interval_s:
                self._reset_one()
                # Whether or not a game accepted: the next look is an
                # interval away, not a poll away.
                last_action_s = self._elapsed()
            if self._stopped.wait(poll_s):
                return

    def _reset_one(self) -> None:
        for game in list(self._games):
            try:
                sent = game.keepalive_reset()
            except Exception:
                print(f"[taaf.keepalive] RESET on {game.game_id} failed:", flush=True)
                traceback.print_exc()
                return
            if sent:
                print(
                    f"[taaf.keepalive] RESET on {game.game_id} at +{self._elapsed():.0f}s: "
                    f"no move in the last {self._interval_s:g}s",
                    flush=True,
                )
                return
        print(
            f"[taaf.keepalive] no move in the last {self._interval_s:g}s, but no game is on an "
            "untouched level, so no RESET would go unseen; none sent",
            flush=True,
        )
