# type: ignore
"""Tests for taaf.benchmark (R2.1, R2.2, R2.11–R2.15)."""

import asyncio
import json
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import arcengine
import deepdiff
import pytest

import taaf


def _act(action_id: int) -> arcengine.ActionInput:
    return arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id), data={})


def _winning_sequence() -> list[arcengine.ActionInput]:
    return [_act(1)] * 3 + [_act(2)] * 3


# --- Basic run --------------------------------------------------------------


async def test_run_single_pass(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="test_run_single_pass",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()
    assert len(bm.game_runs) == 1
    assert bm.game_runs[0].state == "won"
    assert bm.game_runs[0].final_score == pytest.approx(100.0)
    assert (tmp_path / "benchmark.json").exists()


# --- Multi-pass (R2.2) ------------------------------------------------------


async def test_multi_pass_creates_n_copies(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="test_multi_pass",
        games=[taaf.game_examples.ExampleGame(label="example_a"), taaf.game_examples.ExampleGame(label="example_b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=3,
        job_dir=tmp_path,
    )
    await bm.run()
    # 2 games x 3 passes = 6 GameRuns
    assert len(bm.game_runs) == 6
    assert all(r.state == "won" for r in bm.game_runs)


async def test_passes_use_independent_copies(tmp_path: Path):
    """Mutating the result of one pass must not affect the others (R2.2 deepcopy)."""
    bm = taaf.benchmark.Benchmark(
        label="test_independent",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=2,
        job_dir=tmp_path,
    )
    await bm.run()
    assert len(bm.game_runs) == 2
    # Mutate one run's history; the other must stay untouched.
    bm.game_runs[0].history.clear()
    assert len(bm.game_runs[1].history) > 0


async def test_pass_index_stamped_per_pass(tmp_path: Path):
    """Each copy carries the pass it belongs to (R2.2), in memory and in the
    JSON — the identity a per-pass-configured solver reads and every per-pass
    grouping downstream keys on."""
    bm = taaf.benchmark.Benchmark(
        label="test_pass_index",
        games=[taaf.game_examples.ExampleGame(label="example_a"), taaf.game_examples.ExampleGame(label="example_b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=3,
        job_dir=tmp_path,
    )
    await bm.run()

    assert [r.pass_index for r in bm.game_runs] == [0, 0, 1, 1, 2, 2]
    # (game_id, pass_index) identifies a copy: the pair a sweep joins on.
    pairs = [(r.game_id, r.pass_index) for r in bm.game_runs]
    assert len(set(pairs)) == len(pairs)

    # Asserted on the raw JSON, not via from_json: the loader backfills a
    # missing pass_index, so a reload would mask its absence here.
    with open(tmp_path / "benchmark.json") as f:
        d = json.load(f)
    assert [r["pass_index"] for r in d["game_runs"]] == [0, 0, 1, 1, 2, 2]


async def test_from_json_backfills_missing_pass_index(tmp_path: Path):
    """A run recorded before ``pass_index`` existed still groups by pass:
    ``from_json`` recovers it from the passes-major layout."""
    bm = taaf.benchmark.Benchmark(
        label="test_backfill",
        games=[taaf.game_examples.ExampleGame(label="example_a"), taaf.game_examples.ExampleGame(label="example_b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=3,
        job_dir=tmp_path,
    )
    await bm.run()

    path = tmp_path / "benchmark.json"
    with open(path) as f:
        d = json.load(f)
    for r in d["game_runs"]:
        del r["pass_index"]
    with open(path, "w") as f:
        json.dump(d, f)

    restored = taaf.benchmark.Benchmark.from_json(path, with_intermediate_states=False)
    assert [r.pass_index for r in restored.game_runs] == [0, 0, 1, 1, 2, 2]


async def test_from_pickle_backfills_missing_pass_index(tmp_path: Path):
    """The legacy pickle entrance backfills too. It cannot test for a
    missing key the way ``from_json`` does — an unpickled pre-field
    ``GameRun`` reads the class default 0 — so it looks at the instance
    dict instead."""
    bm = taaf.benchmark.Benchmark(
        label="test_pkl_backfill",
        games=[taaf.game_examples.ExampleGame(label="example_a"), taaf.game_examples.ExampleGame(label="example_b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=3,
        job_dir=tmp_path,
    )
    await bm.run()
    for r in bm.game_runs:
        del r.__dict__["pass_index"]
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(bm, f)

    restored = taaf.benchmark.Benchmark.from_pickle(path)
    assert [r.pass_index for r in restored.game_runs] == [0, 0, 1, 1, 2, 2]


async def test_backfill_refuses_archives_predating_unique_game_ids(tmp_path: Path):
    """Ids repeated within a pass make the distinct-id count under-count
    games, so the stamp overruns ``n_passes``. Every consumer iterates
    ``range(n_passes)``, so stamping would drop the surplus runs from every
    statistic silently — the backfill raises instead."""
    bm = taaf.benchmark.Benchmark(
        label="test_legacy_ids",
        games=[taaf.game_examples.ExampleGame(label="example_a"), taaf.game_examples.ExampleGame(label="example_b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()

    path = tmp_path / "benchmark.json"
    with open(path) as f:
        d = json.load(f)
    # One pass over two games, recorded the pre-June-2026 way: the same id
    # for both, so the distinct count says one game and two passes exist.
    for r in d["game_runs"]:
        del r["pass_index"]
        r["game_id"] = "repeated_id"
    with open(path, "w") as f:
        json.dump(d, f)

    with pytest.raises(ValueError, match="predates unique game ids"):
        taaf.benchmark.Benchmark.from_json(path, with_intermediate_states=False)


async def test_run_propagates_job_dir_and_soft_end_to_solver(tmp_path: Path):
    """job_dir, soft_end_time and minimal_diagnostics are convenience fields on Solver, set by Benchmark.run.

    The caller's solver reference stays pristine (mirroring how
    ``self.games`` is replaced by pre-start deepcopies in ``run()``);
    the fields land on ``bm.solver`` — the pre-setup snapshot that
    ends up in the ``solver.pkl`` sidecar.
    """
    deadline = datetime.now() + timedelta(hours=1)
    solver = taaf.solver_examples.SolverSequence(actions=_winning_sequence())
    bm = taaf.benchmark.Benchmark(
        label="test_propagate",
        games=[taaf.game_examples.ExampleGame()],
        solver=solver,
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run(soft_end_time=deadline, minimal_diagnostics=True)
    assert bm.solver is not solver  # deepcopied at the top of run()
    assert bm.solver.job_dir == tmp_path
    assert bm.solver.soft_end_time == deadline
    assert bm.solver.minimal_diagnostics is True
    # Direct ``Benchmark.run()`` has no deployment context (R12.01).
    assert bm.solver.runtime_environment is None
    # Caller's reference is untouched.
    assert solver.job_dir is None
    assert solver.soft_end_time is None
    assert solver.minimal_diagnostics is False


_RUNTIME_ENV_CAPTURE: dict[str, object] = {}


@dataclass
class _RuntimeEnvRecordingSolver(taaf.solver.Solver):
    """Module-level so ``Benchmark._save_solver`` can serialize it; a
    test-local class can't be pickled. Records what ``setup()`` sees in
    ``runtime_environment`` into a module-level dict (the test reads it
    out)."""

    def _setup(self) -> None:
        _RUNTIME_ENV_CAPTURE["at_setup"] = self.runtime_environment

    async def _run_games(self, games):
        for g in games:
            g.finish_game()


async def test_run_propagates_runtime_environment_target_to_solver(tmp_path: Path):
    """R12.01 — ``Benchmark.run`` plumbs the ``runtime_environment`` kwarg
    onto the solver before ``setup()`` is called. The solver sees the real
    target object (not a string mirror), so subclasses can dispatch on
    target type or read target config (``cpu_count`` / ``gpu`` / etc.)."""
    _RUNTIME_ENV_CAPTURE.clear()
    target = taaf.deploy_inline.InlineTarget()
    bm = taaf.benchmark.Benchmark(
        label="test_runtime_env_target",
        games=[taaf.game_examples.ExampleGame()],
        solver=_RuntimeEnvRecordingSolver(),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run(runtime_environment=target)
    # ``played_solver`` is a deepcopy of ``bm.solver`` so the target inside
    # setup() is an equal-but-not-identical copy — same convention as
    # ``job_dir`` / ``soft_end_time`` above.
    assert isinstance(_RUNTIME_ENV_CAPTURE["at_setup"], taaf.deploy_inline.InlineTarget)
    assert _RUNTIME_ENV_CAPTURE["at_setup"] == target
    # ``bm.solver`` is the pre-setup snapshot; the assignment is a direct
    # reference so identity holds here.
    assert bm.solver.runtime_environment is target


_TEARDOWN_CAPTURE: dict[str, int] = {}


@dataclass
class _TeardownRecordingSolver(taaf.solver.Solver):
    label: str = "TeardownRecordingSolver"
    wait_until_cancelled: bool = False

    def _teardown(self) -> None:
        _TEARDOWN_CAPTURE["calls"] = _TEARDOWN_CAPTURE.get("calls", 0) + 1

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            if self.wait_until_cancelled:
                while True:
                    await asyncio.sleep(0.01)
            for game in games:
                game.finish_game()
        except asyncio.CancelledError:
            for game in games:
                run = game.game_run
                if run is not None and run.state == "playing":
                    game.finish_game()
            raise


async def test_run_calls_solver_teardown_after_success(tmp_path: Path):
    _TEARDOWN_CAPTURE.clear()
    bm = taaf.benchmark.Benchmark(
        label="test_solver_teardown_success",
        games=[taaf.game_examples.ExampleGame()],
        solver=_TeardownRecordingSolver(),
        n_passes=1,
        job_dir=tmp_path,
    )

    await bm.run()

    assert _TEARDOWN_CAPTURE["calls"] == 1


async def test_run_calls_solver_teardown_after_external_cancel(tmp_path: Path):
    _TEARDOWN_CAPTURE.clear()
    bm = taaf.benchmark.Benchmark(
        label="test_solver_teardown_cancel",
        games=[taaf.game_examples.ExampleGame()],
        solver=_TeardownRecordingSolver(wait_until_cancelled=True),
        n_passes=1,
        job_dir=tmp_path,
    )

    run_task = asyncio.create_task(bm.run())
    for _ in range(200):
        if bm._solver_task is not None:
            break
        await asyncio.sleep(0.01)
    assert bm._solver_task is not None, "run() never set _solver_task"

    bm.request_stop()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert _TEARDOWN_CAPTURE["calls"] == 1


# --- soft_end_time cancellation (R2.12) -------------------------------------


async def test_soft_end_time_cancels(tmp_path: Path):
    """Soft end fires while solver is still running — games end as `cancelled`.

    `delay_move=0.05` (slower than the 80 ms deadline → at most 1 action) makes
    this deterministic; without it the random walk can finish the game in a few
    actions before the deadline fires.
    """
    bm = taaf.benchmark.Benchmark(
        label="test_soft_end",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverRandom(max_actions_per_game=10**9, delay_move=0.05),
        n_passes=1,
        job_dir=tmp_path,
    )
    deadline = datetime.now() + timedelta(milliseconds=80)
    await bm.run(soft_end_time=deadline)
    assert len(bm.game_runs) == 1
    assert bm.game_runs[0].state == "cancelled"


# --- request_stop external cancel (R2.33) -----------------------------------


async def test_request_stop_cancels_in_flight_solver(tmp_path: Path):
    """``Benchmark.request_stop()`` should cancel the live solver task and
    route through the same teardown path as the soft-deadline (R2.12) —
    games end as ``cancelled``, ``benchmark.json`` (+ sidecars) is
    written, and ``CancelledError`` re-raises so the caller can tell the
    run was externally stopped. This is the hook the Slurm worker's
    SIGUSR1 handler plugs into for R2.33's graceful-stop contract."""
    bm = taaf.benchmark.Benchmark(
        label="test_request_stop",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverRandom(max_actions_per_game=10**9, delay_move=0.05),
        n_passes=1,
        job_dir=tmp_path,
    )

    run_task = asyncio.create_task(bm.run())
    # Wait until ``run()`` has actually entered its solver-await; ``request_stop``
    # is a no-op before that point.
    for _ in range(200):
        if bm._solver_task is not None:
            break
        await asyncio.sleep(0.01)
    assert bm._solver_task is not None, "run() never set _solver_task"

    bm.request_stop()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert len(bm.game_runs) == 1
    assert bm.game_runs[0].state == "cancelled"
    assert (tmp_path / "benchmark.json").is_file(), "teardown should have written benchmark.json"
    assert (tmp_path / "intermediate_states.pkl").is_file(), "teardown should have written intermediate_states sidecar"
    # ``_solver_task`` cleared in the finally block so a stale reference can't be
    # cancelled after teardown.
    assert bm._solver_task is None


async def test_request_stop_noop_before_run(tmp_path: Path):
    """``request_stop`` is a safe no-op before ``run()`` has started — necessary
    because the signal handler is armed at process start and may fire before
    the solver-await is reached."""
    bm = taaf.benchmark.Benchmark(
        label="test_request_stop_noop",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverRandom(),
        n_passes=1,
        job_dir=tmp_path,
    )
    # Must not raise.
    bm.request_stop()


# --- Solver errors don't crash benchmark (R2.15) ----------------------------


@dataclass
class _ExplodingSolver(taaf.solver.Solver):
    """Module-scope so the pre-setup snapshot in ``solver.pkl`` can serialise it
    (R12.02 — Solvers must be picklable until ``setup()``)."""

    label: str = "ExplodingSolver"

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        # Take one action then raise.
        games[0].execute_action(_act(1))
        raise RuntimeError("boom")


async def test_run_swallows_solver_error(tmp_path: Path):
    bm = taaf.benchmark.Benchmark(
        label="test_swallow",
        games=[taaf.game_examples.ExampleGame()],
        solver=_ExplodingSolver(),
        n_passes=1,
        job_dir=tmp_path,
    )
    # Should NOT raise.
    await bm.run()
    assert len(bm.game_runs) == 1
    # Solver never called finish_game → benchmark teardown marks the run as crashed.
    assert bm.game_runs[0].state == "crashed"
    assert (tmp_path / "benchmark.json").exists()


# --- Periodic JSON save (R2.13) ---------------------------------------------


@dataclass
class _SlowSolver(taaf.solver.Solver):
    """Module-scope so ``solver.pkl`` can serialise the pre-setup snapshot."""

    label: str = "SlowSolver"

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            # Take a few actions, then wait long enough for a periodic save to fire.
            for _ in range(2):
                games[0].execute_action(_act(1))
                await asyncio.sleep(0.05)
            games[0].finish_game()
        except asyncio.CancelledError:
            games[0].finish_game()
            raise


async def test_periodic_json_save(tmp_path: Path):
    """With a tiny interval, the JSON file should appear before the run finishes."""
    bm = taaf.benchmark.Benchmark(
        label="test_periodic",
        games=[taaf.game_examples.ExampleGame()],
        solver=_SlowSolver(),
        n_passes=1,
        job_dir=tmp_path,
        periodic_save_interval_s=0.02,
    )
    await bm.run()
    # Guaranteed to exist post-run anyway, but we also want to verify the schema includes the periodic data.
    data = json.loads((tmp_path / "benchmark.json").read_text())
    assert data["label"] == "test_periodic"
    assert len(data["game_runs"]) == 1


async def test_a_periodic_save_writes_off_the_event_loop_and_a_cancelled_one_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R2.13: the event loop serves the games, so a save that writes on it
    stalls them all for as long as the filesystem takes. The write runs on a
    thread, and teardown -- which cancels the save loop and then writes the
    same files -- must not find an older write still landing behind it."""
    started, release = threading.Event(), threading.Event()
    writers: list[int] = []

    def slow_write(writes: list[taaf.support.FileWrite]) -> None:
        writers.append(threading.get_ident())
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(taaf.support, "write_files", slow_write)
    bm = taaf.benchmark.Benchmark(label="t", games=[taaf.game_examples.ExampleGame()], job_dir=tmp_path)
    save = asyncio.create_task(bm._write_off_loop([]))
    assert await asyncio.to_thread(started.wait, 5)
    assert writers != [threading.get_ident()], "the write ran on the event loop's thread"
    save.cancel()
    await asyncio.sleep(0.05)
    assert not save.done(), "a cancelled save returned while its write was still running"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await save


@pytest.mark.parametrize("periodic_page", [True, False])
async def test_the_periodic_save_writes_while_the_run_goes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], periodic_page: bool
):
    """R2.13 end to end: saves fire during play and land every output, the
    page too when the periodic saves leave it to the end."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.ExampleGame()],
        solver=_SlowSolver(),
        n_passes=1,
        job_dir=tmp_path,
        periodic_save_interval_s=0.02,
        periodic_page=periodic_page,
    )
    await bm.run()
    assert "on the event loop; wrote" in capsys.readouterr().out
    assert all((tmp_path / name).is_file() for name in ("benchmark.json", "diagnostics.html", "summary.txt"))


async def test_a_periodic_save_without_the_page_renders_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """R2.13: with ``periodic_page`` off, a periodic save writes the JSON
    checkpoint and the summary and renders no page or movie, which would
    hold the event loop the games are served on."""

    saves: list[list[Path]] = []

    def no_render(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> list[taaf.support.FileWrite]:
        raise AssertionError("a periodic save rendered the page")

    def record(writes: list[taaf.support.FileWrite]) -> None:
        saves.append([w.path for w in writes])

    monkeypatch.setattr(taaf.diagnostics, "render_run_html", no_render)
    monkeypatch.setattr(taaf.support, "write_files", record)
    bm = taaf.benchmark.Benchmark(
        label="t", games=[taaf.game_examples.ExampleGame()], job_dir=tmp_path, periodic_page=False
    )
    bm.periodic_save_interval_s = 0.0
    save = asyncio.create_task(bm._periodic_save_loop())
    while not saves:
        await asyncio.sleep(0.01)
    save.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save
    assert saves[0] == [tmp_path / "benchmark.json", tmp_path / "summary.txt"]


async def test_a_periodic_save_keeps_the_json_checkpoint_when_diagnostics_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R2.13: the JSON snapshot is what a run is read back from, so a
    diagnostics render that keeps failing must not stop it being saved."""

    saves: list[list[Path]] = []

    def broken_render(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> list[taaf.support.FileWrite]:
        raise PermissionError("movies")

    def record(writes: list[taaf.support.FileWrite]) -> None:
        saves.append([w.path for w in writes])

    monkeypatch.setattr(taaf.diagnostics, "render_run_html", broken_render)
    monkeypatch.setattr(taaf.support, "write_files", record)
    bm = taaf.benchmark.Benchmark(label="t", games=[taaf.game_examples.ExampleGame()], job_dir=tmp_path)
    bm.periodic_save_interval_s = 0.0
    save = asyncio.create_task(bm._periodic_save_loop())
    while not saves:
        await asyncio.sleep(0.01)
    save.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save
    assert saves[0] == [tmp_path / "benchmark.json"]


async def test_minimal_diagnostics_skips_file_writes_and_frame_logging(tmp_path: Path):
    """R2.59 — Kaggle submission mode keeps stdout status but avoids
    JSON / HTML / sidecar writes and intermediate frame storage."""
    bm = taaf.benchmark.Benchmark(
        label="test_minimal_diagnostics",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
    )

    await bm.run(minimal_diagnostics=True)

    assert len(bm.game_runs) == 1
    assert bm.game_runs[0].state == "won"
    assert bm.game_runs[0].history
    assert bm.game_runs[0].intermediate_states == []
    assert not (tmp_path / "benchmark.json").exists()
    assert not (tmp_path / "diagnostics.html").exists()
    assert not (tmp_path / "intermediate_states.pkl").exists()


# --- JSON round-trip (R2.14) ------------------------------------------------


async def test_json_round_trip(tmp_path: Path):
    """R2.14: restored Benchmark equals the original except for fields that are
    intentionally NOT round-tripped.

    `DeepDiff` covers the comprehensive equality check — adding a new field to
    `GameRun` without updating `to_json_dict` / `from_json_dict` will fail this
    test. Explicit assertions document the intentional exclusions.
    """
    bm = taaf.benchmark.Benchmark(
        label="test_round_trip",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=2,
        job_dir=tmp_path,
        # Non-default value so DeepDiff catches regressions where a
        # tunable field is silently dropped from the JSON contract.
        periodic_save_interval_s=42.5,
    )
    await bm.run()
    # Bare JSON load (no sidecar attach) to verify what the JSON itself
    # actually contains, independent of which sidecars exist on disk.
    restored = taaf.benchmark.Benchmark.from_json(
        tmp_path / "benchmark.json",
        with_intermediate_states=False,
    )

    # Intentionally NOT round-tripped:
    assert restored.games == []  # R2.14: "except the game list"
    assert restored.solver is None  # only solver_label survives in JSON
    for r in restored.game_runs:
        assert r.intermediate_states == []  # R11.02: not in JSON; opt-in sidecar attach
        assert r.started_at_monotonic is None  # in-memory monotonic stamp, not persisted

    # Everything else must match. The exclude_regex_paths list IS the contract
    # for what's intentionally absent — keep it in sync with `Benchmark.to_json_dict`
    # and `GameRun.to_json_dict`.
    diff = deepdiff.DeepDiff(
        bm,
        restored,
        exclude_regex_paths=[
            r"root\.games$",
            r"root\.solver$",
            r"root\.job_dir$",  # not in JSON; restored gets the dataclass default
            r"root\.trace_root$",  # R5.02 config, not persisted
            r"root\.retain_traces$",  # R5.48 config, not persisted
            r"root\.game_runs\[\d+\]\.intermediate_states$",
            r"root\.game_runs\[\d+\]\.started_at_monotonic$",
            r"root\.game_runs\[\d+\]\.trace_root$",  # in-memory only; not in JSON
        ],
    )
    assert not diff, f"unexpected drift between original and restored Benchmark:\n{diff}"


# --- Per-payload sidecar round-trip -----------------------------------------


async def test_sidecar_round_trip_default_attaches_intermediate_states(tmp_path: Path):
    """``from_json`` (default kwargs) attaches ``intermediate_states.pkl``
    so frames round-trip without requiring the user's ``Solver`` / ``Game``
    classes to be importable. Games + solver stay unattached by default.
    """
    import numpy as np

    bm = taaf.benchmark.Benchmark(
        label="test_sidecar",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=2,
        job_dir=tmp_path / "run",
    )
    await bm.run()

    assert (tmp_path / "run" / "intermediate_states.pkl").exists()
    assert (tmp_path / "run" / "games.pkl").exists()
    assert (tmp_path / "run" / "solver.pkl").exists()

    restored = taaf.benchmark.Benchmark.from_json(tmp_path / "run" / "benchmark.json")
    assert restored.games == [], "with_games defaults to False"
    assert restored.solver is None, "with_solver defaults to False"
    for orig_run, new_run in zip(bm.game_runs, restored.game_runs, strict=True):
        assert len(new_run.intermediate_states) == len(orig_run.intermediate_states)
        for orig_state, new_state in zip(orig_run.intermediate_states, new_run.intermediate_states, strict=True):
            assert new_state.just_won_level == orig_state.just_won_level
            assert new_state.previous_action == orig_state.previous_action
            assert new_state.levels_completed == orig_state.levels_completed
            assert new_state.available_actions == orig_state.available_actions
            assert new_state.won == orig_state.won
            for new_f, orig_f in zip(new_state.raw.frame, orig_state.raw.frame, strict=True):
                np.testing.assert_array_equal(new_f, orig_f)


async def test_sidecar_round_trip_optional_games_and_solver(tmp_path: Path):
    """``with_games`` / ``with_solver`` opt-in attaches the user-class
    payloads needed to re-run. The solver round-trips as the pristine
    pre-setup snapshot — the played copy never enters the sidecar."""
    bm = taaf.benchmark.Benchmark(
        label="test_full_sidecars",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "run",
    )
    await bm.run()

    restored = taaf.benchmark.Benchmark.from_json(
        tmp_path / "run" / "benchmark.json",
        with_games=True,
        with_solver=True,
    )
    assert restored.solver is not None
    assert isinstance(restored.solver, taaf.solver_examples.SolverSequence)
    assert not restored.solver._setup_called, "solver sidecar is the pre-setup snapshot"
    assert len(restored.games) == len(bm.games)


async def test_sidecar_intermediate_states_can_be_skipped(tmp_path: Path):
    """``with_intermediate_states=False`` skips the largest sidecar —
    useful for cheap inspection of just the headline numbers."""
    bm = taaf.benchmark.Benchmark(
        label="test_no_states",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "run",
    )
    await bm.run()

    restored = taaf.benchmark.Benchmark.from_json(
        tmp_path / "run" / "benchmark.json",
        with_intermediate_states=False,
    )
    for r in restored.game_runs:
        assert r.intermediate_states == []


async def test_sidecar_diagnostics_regenerate_from_json(tmp_path: Path):
    """The diagnostics regen pipeline produces movies straight from
    ``benchmark.json`` (auto-attaching the intermediate_states sidecar).
    This is the headline feature: regen works when the user's solver /
    game classes aren't importable, hence no need for the legacy
    ``benchmark.pkl``.
    """
    bm = taaf.benchmark.Benchmark(
        label="test_sidecar_movies",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "run",
    )
    await bm.run()

    out_dir = tmp_path / "regen"
    taaf.diagnostics.regenerate_run_diagnostics(tmp_path / "run", out_dir=out_dir)

    assert (out_dir / "diagnostics.html").stat().st_size > 0
    movies = sorted((out_dir / "movies").glob("*.mp4"))
    assert movies, "movies should regenerate from benchmark.json + sidecars alone"
    for mp4 in movies:
        assert mp4.stat().st_size > 0


# --- Legacy benchmark.pkl back-compat ---------------------------------------


async def test_legacy_benchmark_pkl_still_loads(tmp_path: Path):
    """``Benchmark.from_pickle`` remains as a back-compat shim for old
    runs that wrote a monolithic ``benchmark.pkl`` before the split
    layout. Forge one by pickling a finished Benchmark and verify
    diagnostics regen still picks it up via the loader's fallback path.
    """
    bm = taaf.benchmark.Benchmark(
        label="test_legacy",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "run",
    )
    await bm.run()

    # Simulate an old run: dump the monolithic pickle and remove the
    # new-layout snapshot + sidecars.
    import pickle

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    with open(legacy_dir / "benchmark.pkl", "wb") as f:
        pickle.dump(bm, f)

    restored = taaf.benchmark.Benchmark.from_pickle(legacy_dir / "benchmark.pkl")
    assert restored.solver is not None
    assert len(restored.games) == len(bm.games)
    for r in restored.game_runs:
        assert r.intermediate_states

    # And the diagnostics loader falls back to it when no benchmark.json exists.
    out_dir = tmp_path / "regen"
    taaf.diagnostics.regenerate_run_diagnostics(legacy_dir, out_dir=out_dir)
    assert (out_dir / "diagnostics.html").stat().st_size > 0


async def test_a_sidecar_saved_as_plain_lists_still_loads(tmp_path: Path):
    """Runs saved before states were held compressed wrote each run's states
    to ``intermediate_states.pkl`` as a plain list of ``GameState``;
    ``from_json`` loads that sidecar into the same states, held compressed."""
    bm = taaf.benchmark.Benchmark(
        label="test_plain_sidecar",
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()
    sidecar = tmp_path / taaf.benchmark.Benchmark.INTERMEDIATE_STATES_PKL
    with open(sidecar, "rb") as f:
        recorded = pickle.load(f)
    with open(sidecar, "wb") as f:
        pickle.dump([list(states) for states in recorded], f)

    restored = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json")
    states = restored.game_runs[0].intermediate_states
    assert isinstance(states, taaf.game.CompressedStates)
    assert len(states) == len(recorded[0]) > 1
    for loaded, saved in zip(states, recorded[0], strict=True):
        assert [f.data.tolist() for f in loaded.all_frames] == [f.data.tolist() for f in saved.all_frames]
        assert loaded.levels_completed == saved.levels_completed


# --- Per-game weights -------------------------------------------------------


async def test_game_weights_round_trip(tmp_path: Path):
    """``Benchmark.game_weights`` survives JSON round-trips (with and
    without sidecar attach)."""
    bm = taaf.benchmark.Benchmark(
        label="weighted",
        games=[
            taaf.game_examples.ExampleGame(label="easy"),
            taaf.game_examples.ExampleGame(label="hard"),
        ],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
        game_weights=[1.0, 3.0],
    )
    await bm.run()
    restored_json = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json")
    assert restored_json.game_weights == [1.0, 3.0]
    restored_full = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json", with_games=True, with_solver=True)
    assert restored_full.game_weights == [1.0, 3.0]


async def test_game_weights_validation():
    """Wrong-length or negative weights raise at ``run()`` time."""
    games = [taaf.game_examples.ExampleGame(label="a"), taaf.game_examples.ExampleGame(label="b")]
    bm_short = taaf.benchmark.Benchmark(
        label="short",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        game_weights=[1.0],  # only 1 weight for 2 games
    )
    with pytest.raises(ValueError, match="game_weights length"):
        await bm_short.run()

    bm_neg = taaf.benchmark.Benchmark(
        label="neg",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        game_weights=[1.0, -0.5],
    )
    with pytest.raises(ValueError, match=r"game_weights\[1\] = -0.5"):
        await bm_neg.run()


async def test_game_weights_affect_pooled_curve(tmp_path: Path):
    """The pooled curve's terminal y-value matches the weighted average
    of per-game terminal scores."""
    bm = taaf.benchmark.Benchmark(
        label="weighted-curve",
        games=[
            taaf.game_examples.ExampleGame(label="easy"),
            taaf.game_examples.ExampleGame(label="hard"),
        ],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
        game_weights=[1.0, 3.0],
    )
    await bm.run()
    weights = taaf.diagnostics.weights_by_game_id(bm)
    xs, ys = taaf.diagnostics._pass_curve(bm.game_runs, taaf.diagnostics.total_tokens, weights_by_game_id=weights)
    # Both games win cleanly (score 100). Weighted mean of [100, 100] with
    # weights [1, 3] is 100. Unweighted would also be 100 — so test the
    # weight-0 case separately below.
    assert ys[-1] == pytest.approx(100.0)


async def test_game_weights_zero_excludes_from_aggregation(tmp_path: Path):
    """Weight 0 fully excludes a game from the pooled curve and from
    cross-game scalar stats. The game still appears in
    ``benchmark.game_runs`` so per-game cards / table can render it."""
    # easy game wins (score 100), excluded-game gets weight 0 — pooled
    # curve should reflect easy game only.
    bm = taaf.benchmark.Benchmark(
        label="zero-weight",
        games=[
            taaf.game_examples.ExampleGame(label="easy"),
            taaf.game_examples.ExampleGame(label="excluded"),
        ],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path,
        game_weights=[1.0, 0.0],
    )
    await bm.run()
    assert len(bm.game_runs) == 2  # both runs exist
    means = taaf.diagnostics._per_game_mean_scores(bm)
    assert set(means.keys()) == {"easy"}  # excluded omitted
    # Run-level stats: pass-mean uses only the easy game.
    stats = taaf.diagnostics._run_pass_stats(bm)
    assert stats["mean"] == pytest.approx(100.0)


async def test_weights_banner_rendered(tmp_path: Path):
    """The yellow weights banner appears in per-run and comparison HTML
    when ``game_weights`` is set, and absent otherwise."""
    games = [
        taaf.game_examples.ExampleGame(label="a"),
        taaf.game_examples.ExampleGame(label="b"),
    ]
    bm = taaf.benchmark.Benchmark(
        label="weighted",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "weighted",
        game_weights=[1.0, 2.0],
    )
    bm_plain = taaf.benchmark.Benchmark(
        label="plain",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        job_dir=tmp_path / "plain",
    )
    await bm.run()
    await bm_plain.run()
    weighted_html = (tmp_path / "weighted" / "diagnostics.html").read_text()
    plain_html = (tmp_path / "plain" / "diagnostics.html").read_text()
    # The CSS rule `.weights-banner { ... }` always appears (CSS is shared);
    # the banner element does not, so we look for the actual div.
    assert '<div class="weights-banner">' in weighted_html
    assert "Game weights applied" in weighted_html
    assert '<div class="weights-banner">' not in plain_html

    # Comparison page banner: applies when all benchmarks share the same
    # non-trivial weights (mismatched weights are rejected by the
    # comparison precondition — see test_comparison_rejects_mismatched_weights).
    bm_a_same = taaf.benchmark.Benchmark(
        label="a_same",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        game_weights=[1.0, 2.0],
    )
    bm_b_same = taaf.benchmark.Benchmark(
        label="b_same",
        games=games,
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence()),
        n_passes=1,
        game_weights=[1.0, 2.0],
    )
    await bm_a_same.run()
    await bm_b_same.run()
    cmp_dir = tmp_path / "cmp"
    taaf.diagnostics.generate_comparison_html([bm_a_same, bm_b_same], cmp_dir)
    cmp_html = (cmp_dir / "index.html").read_text()
    assert '<div class="weights-banner">' in cmp_html
    assert "shared" in cmp_html


# --- Keepalive RESETs (R2.63) ------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
async def test_keepalive_resets_through_a_blocked_event_loop(enabled: bool):
    """The solver stalls the loop before playing; the keepalive still fires,
    on the first game, and the play that follows is untouched by it."""
    bm = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label="a"), taaf.game_examples.ExampleGame(label="b")],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence(), block_before_play_s=0.55),
        keepalive_resets=enabled,
        keepalive_interval_s=0.1,
    )
    await bm.run()
    first, second = bm.game_runs
    if enabled:
        assert first.keepalive_resets >= 3
        assert second.keepalive_resets == 0, "only the first untouched game is reset"
    else:
        assert first.keepalive_resets == 0
    assert [run.state for run in bm.game_runs] == ["won", "won"]
    assert [len(run.history) for run in bm.game_runs] == [6, 6]


async def test_keepalive_counts_from_clock_start_and_folds_missed_ticks():
    """A tick already past when the run starts fires at once, and only once."""
    bm = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverSequence(actions=_winning_sequence(), block_before_play_s=0.3),
        keepalive_resets=True,
        keepalive_interval_s=600.0,
    )
    await bm.run(clock_start=datetime.now() - timedelta(seconds=1805))
    assert bm.game_runs[0].keepalive_resets == 1
    assert bm.game_runs[0].state == "won"


async def test_keepalive_stays_quiet_while_the_solver_moves():
    bm = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame()],
        solver=taaf.solver_examples.SolverRandom(delay_move=0.005, max_actions_per_game=150, seed=3),
        keepalive_resets=True,
        keepalive_interval_s=0.25,
    )
    await bm.run()
    assert bm.game_runs[0].keepalive_resets == 0


async def test_keepalive_halts_when_the_deadline_fires():
    """A solver winding down after the deadline makes no moves; that is not a
    stretch to keep alive, so queued games get no RESET during it."""

    @dataclass
    class SlowToStop(taaf.solver_examples.SolverRandom):
        async def _run_games(self, games):
            try:
                # The second game stays queued, on its untouched first level.
                await super()._run_games(games[:1])
            except asyncio.CancelledError:
                time.sleep(0.4)  # cleanup that blocks, like an engine shutdown
                for game in games[1:]:
                    game.finish_game()
                raise

    # Long enough that random play is still mid-level when the deadline fires.
    unwinnable = [[arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2] * 8]
    bm = taaf.benchmark.Benchmark(
        games=[
            taaf.game_examples.ExampleGame(label="a", target_sequence_per_level=unwinnable),
            taaf.game_examples.ExampleGame(label="b", target_sequence_per_level=unwinnable),
        ],
        solver=SlowToStop(delay_move=0.01, max_actions_per_game=None, seed=1),
        keepalive_resets=True,
        keepalive_interval_s=0.1,
    )
    await bm.run(soft_end_time=datetime.now() + timedelta(seconds=0.3))
    assert [run.keepalive_resets for run in bm.game_runs] == [0, 0]


async def test_keepalive_counts_from_the_last_move_not_from_a_tick():
    """A move landing just after a tick must not stretch the silence to two
    intervals: the clock restarts at the move, so the next RESET is one
    interval after it, on the game still untouched (R2.63)."""

    @dataclass
    class OneMoveThenSilence(taaf.solver_examples.SolverSequence):
        async def _run_games(self, games):
            games[0].execute_action(_winning_sequence()[0])
            time.sleep(0.55)  # blocking, like an engine build after a first probe
            for game in games:
                game.finish_game()

    bm = taaf.benchmark.Benchmark(
        games=[taaf.game_examples.ExampleGame(label="a"), taaf.game_examples.ExampleGame(label="b")],
        solver=OneMoveThenSilence(actions=_winning_sequence()),
        keepalive_resets=True,
        keepalive_interval_s=0.2,
    )
    await bm.run()
    touched, untouched = bm.game_runs
    assert touched.keepalive_resets == 0, "a touched level is never reset"
    assert untouched.keepalive_resets >= 2, "one RESET per interval after the move, not one per skipped tick"


# --- a solver ends its own run (R2.64) --------------------------------------


@dataclass
class _StopsItself(taaf.solver_examples.SolverRandom):
    """The random solver, ending its own run a moment after it starts: what a
    solver's watchdog does when the engine goes quiet."""

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        assert self.stop_run is not None, "stamped for the run's duration"
        asyncio.get_running_loop().call_later(0.2, self.stop_run, "the engine went quiet")
        await super()._run_games(games)


async def test_a_solver_ends_its_own_run_like_the_deadline_and_the_reason_is_kept(tmp_path: Path):
    """``stop_run`` is the soft deadline on demand: the games still held land
    cancelled, run() returns normally, and the reason reaches the JSON and
    the summary. Outside a run the hook is gone and a late call is a no-op."""
    bm = taaf.benchmark.Benchmark(
        label="test_stop_run",
        games=[taaf.game_examples.ExampleGame()],
        solver=_StopsItself(max_actions_per_game=10**9, delay_move=0.05),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()
    assert bm.stop_reason == "the engine went quiet"
    assert [r.state for r in bm.game_runs] == ["cancelled"]
    assert taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json").stop_reason == "the engine went quiet"
    assert "stopped:   the engine went quiet" in (tmp_path / "summary.txt").read_text()
    assert bm.solver is not None and bm.solver.stop_run is None
    bm.stop_early("too late")
    assert bm.stop_reason == "the engine went quiet"


class _StopsItselfTwiceMidCleanup(_StopsItself):
    """The stopped solver's cleanup awaits, and a second stop arrives inside
    that wait: a second cancellation would interrupt the cleanup and leave
    the game ``crashed`` rather than ``cancelled``."""

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            await super()._run_games(games)
        except asyncio.CancelledError:
            assert self.stop_run is not None
            self.stop_run("again")
            await asyncio.sleep(0.05)
            raise


async def test_a_second_stop_during_cleanup_is_a_no_op(tmp_path: Path):
    """R2.64: ``stop_run`` is idempotent for real, and so is the deadline
    behind it: one cancellation per run, whichever path asks first."""
    bm = taaf.benchmark.Benchmark(
        label="test_stop_run_twice",
        games=[taaf.game_examples.ExampleGame()],
        solver=_StopsItselfTwiceMidCleanup(max_actions_per_game=10**9, delay_move=0.05),
        n_passes=1,
        job_dir=tmp_path,
    )
    await bm.run()
    assert bm.stop_reason == "the engine went quiet", "the first reason stands"
    assert [r.state for r in bm.game_runs] == ["cancelled"]
