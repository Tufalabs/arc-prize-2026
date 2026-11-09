"""Tests for taaf.benchmark — the run lifecycle, passes, weighting,
simulate_submission, and the submission hooks."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest

import taaf.benchmark
import taaf.datasets
import taaf.game_examples
import taaf.solver_examples
import taaf.standard_benchmarks
import taaf.submission
import taaf.support
from taaf.game import Game
from taaf.solver import Solver


def _bm(job_dir: Path | None = None, n_games: int = 2, **kwargs: Any) -> taaf.benchmark.Benchmark:
    """A small benchmark of flip games, which the flip solver solves."""
    games: list[Game] = [taaf.game_examples.flip_game(f"g{i}") for i in range(n_games)]
    return taaf.benchmark.Benchmark(
        label="t", games=games, solver=taaf.solver_examples.SolverFlipLR(), job_dir=job_dir, **kwargs
    )


async def test_run_plays_every_game_and_scores_it(tmp_path: Path):
    """The happy path end to end: the flip solver mirrors flip tasks."""
    bm = _bm(tmp_path)
    await bm.run()
    assert len(bm.game_runs) == 2
    assert all(r.state == "won" and r.final_score == 100.0 for r in bm.game_runs)
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "diagnostics.html").is_file()
    assert (tmp_path / "summary.txt").is_file()


async def test_passes_play_independent_copies(tmp_path: Path):
    """Passes are copies of each game handed to one solver, laid out
    passes-major so diagnostics can group by pass."""
    bm = _bm(tmp_path, n_passes=3)
    await bm.run()
    assert len(bm.game_runs) == 6
    assert [r.pass_index for r in bm.game_runs] == [0, 0, 1, 1, 2, 2]
    assert len({r.run_id for r in bm.game_runs}) == 6


async def test_duplicate_game_ids_are_refused(tmp_path: Path):
    """Diagnostics and the submission both key on game_id, so a duplicate is
    ambiguous rather than merely redundant."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game("same"), taaf.game_examples.flip_game("same")],
        solver=taaf.solver_examples.SolverFlipLR(),
        job_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="duplicate game_ids"):
        await bm.run()


class _Boom(Solver):
    """Raises instead of playing. Module-level because Benchmark pickles the
    solver to its sidecar, and a local class cannot be pickled."""

    async def _run_games(self, games: list[Game]) -> None:
        raise RuntimeError("boom")


@dataclass
class _RecordedTeardownFailure(Solver):
    """Collect a backend failure without raising it from the played copy."""

    teardown_errors: list[str] = field(default_factory=list[str])
    primary_failure: bool = False

    async def _run_games(self, games: list[Game]) -> None:
        self.teardown_errors.append("backend connection failed while closing")
        if self.primary_failure:
            raise ValueError("primary solver failure")


@pytest.mark.parametrize("strict", [False, True])
async def test_recorded_teardown_failure_is_read_from_played_solver(tmp_path: Path, strict: bool):
    bm = _bm(tmp_path, raise_solver_errors=strict)
    bm.solver = _RecordedTeardownFailure()
    if strict:
        with pytest.raises(RuntimeError, match="backend connection failed while closing"):
            await bm.run()
    else:
        await bm.run()
    # The serialized solver remains the pristine pre-setup specification.
    assert bm.solver.teardown_errors == []
    assert (tmp_path / "benchmark.json").is_file()


async def test_recorded_teardown_failure_preserves_primary_solver_error(tmp_path: Path):
    bm = _bm(tmp_path, raise_solver_errors=True)
    bm.solver = _RecordedTeardownFailure(primary_failure=True)
    with pytest.raises(ValueError, match="primary solver failure"):
        await bm.run()
    assert (tmp_path / "benchmark.json").is_file()


@dataclass
class _PeekingSolver(taaf.solver_examples.SolverFlipLR):
    """Records the submission file's contents before answering anything."""

    peek_path: Path | None = None
    seen: ClassVar[list[dict[str, Any]]] = []

    async def _run_games(self, games: list[Game]) -> None:
        assert self.peek_path is not None
        type(self).seen.append(json.loads(self.peek_path.read_text()))
        await super()._run_games(games)


async def test_a_crashing_solver_still_tears_down(tmp_path: Path):
    """A solver error is not fatal to the run: games left playing are marked
    crashed and the outputs still land."""
    bm = _bm(tmp_path)
    bm.solver = _Boom()
    await bm.run()
    assert all(r.state == "crashed" for r in bm.game_runs)
    assert (tmp_path / "benchmark.json").is_file()
    # Crashing does not make a run unscored: these answered nothing, so they
    # score 0, and one that had answered first would keep what it reached.
    assert all(r.final_score == 0.0 for r in bm.game_runs)


@pytest.mark.parametrize("secondary_failure", ["none", "teardown", "diagnostics"])
async def test_strict_solver_failure_survives_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secondary_failure: str
):
    primary = RuntimeError("trace registration failed")
    teardown_calls: list[Solver] = []
    generate_diagnostics = taaf.benchmark.Benchmark._generate_diagnostics

    async def fail(self: Solver, games: list[Game]) -> None:
        raise primary

    def teardown(self: Solver) -> None:
        teardown_calls.append(self)
        if secondary_failure == "teardown":
            raise RuntimeError("secondary teardown failure")

    def diagnostics(self: taaf.benchmark.Benchmark) -> None:
        generate_diagnostics(self)
        if secondary_failure == "diagnostics":
            raise RuntimeError("secondary diagnostics failure")

    monkeypatch.setattr(_Boom, "_run_games", fail)
    monkeypatch.setattr(_Boom, "_teardown", teardown)
    monkeypatch.setattr(taaf.benchmark.Benchmark, "_generate_diagnostics", diagnostics)
    bm = _bm(tmp_path, raise_solver_errors=True)
    bm.solver = _Boom()
    with pytest.raises(RuntimeError) as raised:
        await bm.run()
    assert raised.value is primary
    assert len(teardown_calls) == 1
    assert teardown_calls[0] is not bm.solver
    assert all(r.state == "crashed" and r.final_score == 0.0 for r in bm.game_runs)
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "diagnostics.html").is_file()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
async def test_strict_game_setup_failure_survives_solver_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
):
    primary = error_type("game setup failed")

    def fail_start(self: Game, *args: Any, **kwargs: Any) -> None:
        raise primary

    def fail_teardown(self: Solver) -> None:
        raise RuntimeError("secondary teardown failure")

    monkeypatch.setattr(Game, "start_game", fail_start)
    monkeypatch.setattr(_Boom, "_teardown", fail_teardown)
    bm = _bm(tmp_path, raise_solver_errors=True)
    bm.solver = _Boom()
    with pytest.raises(error_type) as raised:
        await bm.run()
    assert raised.value is primary
    assert (tmp_path / "benchmark.json").is_file()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("cleanup", ["solver", "game"])
async def test_cleanup_failure_is_fatal_only_in_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool, cleanup: str
):
    error = RuntimeError("cleanup failed")
    finish_game = Game.finish_game

    async def leave_games_playing(self: Solver, games: list[Game]) -> None:
        pass

    def fail_teardown(self: Solver) -> None:
        raise error

    def fail_finish(self: Game) -> None:
        finish_game(self)
        raise error

    monkeypatch.setattr(_Boom, "_run_games", leave_games_playing)
    if cleanup == "solver":
        monkeypatch.setattr(_Boom, "_teardown", fail_teardown)
    else:
        monkeypatch.setattr(Game, "finish_game", fail_finish)
    bm = _bm(tmp_path, raise_solver_errors=strict)
    bm.solver = _Boom()
    if strict:
        with pytest.raises(RuntimeError) as raised:
            await bm.run()
        assert raised.value is error
    else:
        await bm.run()
    assert all(r.state == "crashed" and r.final_score == 0.0 for r in bm.game_runs)
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "diagnostics.html").is_file()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("teardown_fails", [False, True])
async def test_caller_cancellation_remains_cancelled_after_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool, teardown_fails: bool
):
    started = asyncio.Event()
    answer = taaf.solver_examples.SolverFlipLR._answer

    async def wait_for_cancellation(self: taaf.solver_examples.SolverFlipLR, game: Game) -> None:
        started.set()
        await answer(self, game)

    def teardown(self: Solver) -> None:
        if teardown_fails:
            raise RuntimeError("teardown failed during cancellation")

    monkeypatch.setattr(taaf.solver_examples.SolverFlipLR, "_answer", wait_for_cancellation)
    monkeypatch.setattr(taaf.solver_examples.SolverFlipLR, "_teardown", teardown)
    bm = _bm(tmp_path, raise_solver_errors=strict)
    bm.solver = taaf.solver_examples.SolverFlipLR(delay_s=60.0)
    task = asyncio.create_task(bm.run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(r.state == "cancelled" for r in bm.game_runs)
    assert (tmp_path / "benchmark.json").is_file()


@pytest.mark.parametrize("strict", [False, True])
async def test_cancellation_during_cleanup_preserves_caller_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: bool
):
    started = asyncio.Event()

    async def fail(self: Solver, games: list[Game]) -> None:
        await started.wait()
        raise RuntimeError("primary solver failure")

    async def background(self: taaf.benchmark.Benchmark) -> None:
        try:
            started.set()
            await asyncio.Future[None]()
        finally:
            # Arrive while run() awaits background cleanup, after the solver failed.
            task.cancel()

    monkeypatch.setattr(_Boom, "_run_games", fail)
    monkeypatch.setattr(taaf.benchmark.Benchmark, "_minimal_status_loop", background)
    bm = _bm(tmp_path, raise_solver_errors=strict)
    bm.solver = _Boom()
    task = asyncio.create_task(bm.run(minimal_diagnostics=True))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.parametrize("strict", [False, True])
async def test_soft_end_cancels_and_games_land_cancelled(tmp_path: Path, strict: bool):
    """The run-level stop: the solver is cancelled, finishes its games, and
    they report `cancelled` rather than `crashed`."""
    from datetime import datetime, timedelta

    bm = _bm(tmp_path, n_games=4, raise_solver_errors=strict)
    bm.solver = taaf.solver_examples.SolverFlipLR(delay_s=0.5)
    await bm.run(soft_end_time=datetime.now() + timedelta(seconds=0.2))
    assert any(r.state == "cancelled" for r in bm.game_runs)


async def test_benchmark_json_round_trip(tmp_path: Path):
    """Everything except the game list comes back."""
    bm = _bm(tmp_path)
    await bm.run()
    back = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json")
    assert back.label == bm.label
    assert [r.game_id for r in back.game_runs] == [r.game_id for r in bm.game_runs]
    assert [r.final_score for r in back.game_runs] == [r.final_score for r in bm.game_runs]
    # The spec sidecar is what lets the answer sheets render after a reload.
    assert all(r.spec is not None for r in back.game_runs)


async def test_simulate_submission_withholds_answers_and_diagnostics(tmp_path: Path):
    """The dress rehearsal does both halves. Simulating only the diagnostics
    half is how a scoring dependency survives to the rerun."""
    bm = _bm(tmp_path, simulate_submission=True)
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()
    assert all(r.final_score is None and r.state == "submitted" for r in bm.game_runs)
    assert not (tmp_path / "benchmark.json").exists()
    assert not (tmp_path / "diagnostics.html").exists()
    # ...but the submission is written, because that is the one output the
    # mode exists to produce.
    assert (tmp_path / "submission.json").is_file()


async def test_simulated_and_scored_runs_answer_identically(tmp_path: Path):
    """The check that makes the rehearsal worth having: same submission,
    minus the score."""
    scored = _bm(tmp_path / "a")
    scored.submission_path = tmp_path / "a" / "submission.json"
    await scored.run()
    sim = _bm(tmp_path / "b", simulate_submission=True)
    sim.submission_path = tmp_path / "b" / "submission.json"
    await sim.run()
    assert taaf.submission.read(tmp_path / "a" / "submission.json") == taaf.submission.read(
        tmp_path / "b" / "submission.json"
    )


async def test_submission_is_seeded_before_the_solver_runs(tmp_path: Path):
    """A hard kill at any moment after the games start must leave a complete
    file — so it exists before any game has answered."""
    _PeekingSolver.seen = []
    bm = _bm(tmp_path)
    bm.solver = _PeekingSolver(peek_path=tmp_path / "submission.json")
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()
    seen = _PeekingSolver.seen
    assert set(seen[0]) == {"g0", "g1"}
    assert seen[0]["g0"][0]["attempt_1"] == taaf.submission.PLACEHOLDER.to_lists()


@dataclass
class _PassMarkingSolver(Solver):
    """Answers the mirror on the first pass and black on every later one, so
    the submission file betrays which pass it was built from."""

    async def _run_games(self, games: list[Game]) -> None:
        """One answer per game, chosen by the run's pass index."""
        for game in games:
            assert game.game_run is not None
            name = "flip" if game.game_run.pass_index == 0 else "black"
            game.execute_action(taaf.solver_examples.ANSWERS[name](game.query()))
            game.finish_game()


async def test_a_multi_pass_run_writes_the_first_pass_to_the_submission(tmp_path: Path):
    """A submission carries one answer per task; several passes produce
    several, and the file would keep whichever pass landed later in the run
    list — arbitrary but well-formed, the failure that is never noticed. So
    the file is the first pass's alone, whatever the other passes answer, and
    a Kaggle measurement run (which always writes one) is not refused."""
    bm = _bm(tmp_path, n_passes=3)
    bm.solver = _PassMarkingSolver()
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()
    assert len(bm.game_runs) == 6 and all(r.state == "won" or r.pass_index > 0 for r in bm.game_runs)
    sub = taaf.submission.read(bm.submission_path)
    assert sorted(sub) == ["g0", "g1"]
    by_pass = {(r.game_id, r.pass_index): r.history[-1].submission.to_json_list() for r in bm.game_runs}
    assert by_pass["g0", 0] != by_pass["g0", 1], "the marking solver must answer differently per pass"
    assert sub["g0"] == by_pass["g0", 0], "the file must hold the first pass's answer, not a later pass's"


async def test_weighting_selects_which_official_metric_is_reported():
    """Equal weights give arc-agi-benchmarking's number; test-output weights
    give Kaggle's. They differ on any multi-test task."""
    specs = [
        taaf.game_examples.flip_task("one", n_to_answer=1),
        taaf.game_examples.flip_task("three", n_to_answer=3),
    ]
    assert taaf.standard_benchmarks.weights_for(specs, "task") is None
    assert taaf.standard_benchmarks.weights_for(specs, "output") == [1.0, 3.0]
    with pytest.raises(ValueError, match="unknown weighting"):
        taaf.standard_benchmarks.weights_for(specs, "nonsense")  # type: ignore[arg-type]


def test_a_named_benchmark_merges_several_sets_in_order():
    """Several names play as one benchmark, in the order named, labelled by
    their names; ``ids`` narrows the merged list."""

    def ids(bm: taaf.benchmark.Benchmark) -> list[str]:
        return [game.spec_data.game_id for game in bm.games]  # type: ignore[attr-defined]

    bm = taaf.standard_benchmarks.make_benchmark_named(("arc2-eval", "arc2-train-hard"), n_passes=2)
    eval_ids = [s.game_id for s in taaf.datasets.load("arc2-eval")]
    assert ids(bm) == eval_ids + list(taaf.datasets.HARD_TRAIN_TASK_IDS)
    assert (bm.label, bm.n_passes) == ("arc2-eval+arc2-train-hard", 2)
    assert ids(taaf.standard_benchmarks.make_benchmark_easy()) == list(taaf.datasets.EASY_TASK_IDS)
    one = taaf.standard_benchmarks.make_benchmark_named(("arc2-eval", "arc2-train-hard"), ids=[eval_ids[0]])
    assert ids(one) == [eval_ids[0]]


async def test_withheld_benchmark_builder_matches_the_run_flag():
    """make_benchmark(withhold_answers=True) and simulate_submission reach
    the same state by different routes; neither may score."""
    specs = [taaf.game_examples.flip_task("a")]
    bm = taaf.standard_benchmarks.make_benchmark(
        specs, taaf.solver_examples.SolverFlipLR(), label="t", withhold_answers=True
    )
    await bm.run()
    assert all(r.final_score is None for r in bm.game_runs)


@dataclass
class _WatchingSolver(Solver):
    """Answers twice per game and reads the submission file after each, so a
    test can see what the file held at that moment."""

    peek_path: Path | None = None
    seen: ClassVar[list[list[list[int]]]] = []

    async def _run_games(self, games: list[Game]) -> None:
        """Submit the black answer, then the mirror, peeking after each."""
        assert self.peek_path is not None
        for game in games:
            view = game.query()
            for name in ("black", "flip"):
                game.execute_action(taaf.solver_examples.ANSWERS[name](view))
                _WatchingSolver.seen.append(taaf.submission.read(self.peek_path)[game.game_id][0]["attempt_1"])
                await asyncio.sleep(0)
            game.finish_game()


async def test_the_submission_follows_every_answer_without_waiting_for_a_sweep(tmp_path: Path):
    """A hard kill gives no warning, so the file cannot trail the standing
    answer by a sweep interval. Both save loops are effectively off here, so
    only the per-answer write can be what updates it."""
    _WatchingSolver.seen = []
    bm = _bm(tmp_path, n_games=1)
    bm.solver = _WatchingSolver(peek_path=tmp_path / "submission.json")
    bm.submission_path = tmp_path / "submission.json"
    bm.submission_interval_s = 3600.0
    bm.periodic_save_interval_s = 3600.0
    await bm.run()
    black, mirror = _WatchingSolver.seen
    assert black != mirror, "the file did not follow the solver's second answer"
    assert mirror == taaf.submission.read(bm.submission_path)["g0"][0]["attempt_1"]


async def test_submission_is_rewritten_on_its_own_cadence(tmp_path: Path):
    """The interval is what a hard kill costs, so the submission does not
    ride the diagnostics save loop — which is paced by a far more expensive
    write, and is off entirely in submission mode."""
    bm = _bm(tmp_path, n_games=2)
    bm.solver = taaf.solver_examples.SolverFlipLR(delay_s=0.15)
    bm.submission_path = tmp_path / "submission.json"
    bm.submission_interval_s = 0.05
    bm.periodic_save_interval_s = 3600.0  # the save loop must not be what does it

    seen: list[int] = []

    async def watch() -> None:
        """Count how many tasks are answered in the file while the run goes."""
        for _ in range(12):
            await asyncio.sleep(0.05)
            if bm.submission_path is not None and bm.submission_path.is_file():
                sub = taaf.submission.read(bm.submission_path)
                placeholder = taaf.submission.PLACEHOLDER.to_lists()
                seen.append(sum(1 for v in sub.values() if v[0]["attempt_1"] != placeholder))

    watcher = asyncio.create_task(watch())
    await bm.run()
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    # The file gained answers while the run was still going, rather than only
    # at teardown — which is the whole point under a kill with no warning.
    assert seen, "watcher never saw the submission file"
    assert max(seen) > 0


async def test_a_periodic_save_writes_off_the_event_loop_and_a_cancelled_one_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The event loop serves the games, so a save that writes on it stalls
    them all for as long as the filesystem takes. The write runs on a thread,
    and teardown -- which cancels the save loop and then writes the same files
    -- must not find an older write still landing behind it."""
    started, release = threading.Event(), threading.Event()
    writers: list[int] = []

    def slow_write(writes: list[taaf.support.FileWrite]) -> None:
        writers.append(threading.get_ident())
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(taaf.support, "write_files", slow_write)
    save = asyncio.create_task(_bm(tmp_path)._write_off_loop([]))
    assert await asyncio.to_thread(started.wait, 5)
    assert writers != [threading.get_ident()], "the write ran on the event loop's thread"
    save.cancel()
    await asyncio.sleep(0.05)
    assert not save.done(), "a cancelled save returned while its write was still running"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await save


async def test_a_periodic_save_keeps_the_json_checkpoint_when_diagnostics_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The JSON snapshot is what a run is read back from, so a diagnostics
    render that keeps failing must not stop it being saved."""
    saves: list[list[Path]] = []

    def broken_render(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> list[taaf.support.FileWrite]:
        raise PermissionError("answers")

    def record(writes: list[taaf.support.FileWrite]) -> None:
        saves.append([w.path for w in writes])

    monkeypatch.setattr(taaf.diagnostics, "render_run_html", broken_render)
    monkeypatch.setattr(taaf.support, "write_files", record)
    bm = _bm(tmp_path, n_games=1, periodic_save_interval_s=0.0)
    save = asyncio.create_task(bm._periodic_save_loop())
    while not saves:
        await asyncio.sleep(0.01)
    save.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save
    assert saves[0] == [tmp_path / "benchmark.json"]


@pytest.mark.parametrize("periodic_page", [True, False])
async def test_the_periodic_save_writes_while_the_run_goes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], periodic_page: bool
):
    """End to end: saves fire during play and land every output, the page
    too when the periodic saves leave it to the end."""
    bm = _bm(tmp_path, n_games=3, periodic_page=periodic_page)
    bm.solver = taaf.solver_examples.SolverFlipLR(delay_s=0.15)
    bm.periodic_save_interval_s = 0.05
    await bm.run()
    assert "on the event loop; wrote" in capsys.readouterr().out
    assert all((tmp_path / name).is_file() for name in ("benchmark.json", "diagnostics.html", "summary.txt"))
    assert all((tmp_path / "answers" / f"gg{i}_p0.html").is_file() for i in range(3))


async def test_a_periodic_save_without_the_page_renders_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With ``periodic_page`` off, a periodic save writes the JSON checkpoint
    and the summary and renders no page or answer sheet, which would hold the
    event loop the games are served on."""
    saves: list[list[Path]] = []

    def no_render(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> list[taaf.support.FileWrite]:
        raise AssertionError("a periodic save rendered the page")

    def record(writes: list[taaf.support.FileWrite]) -> None:
        saves.append([w.path for w in writes])

    monkeypatch.setattr(taaf.diagnostics, "render_run_html", no_render)
    monkeypatch.setattr(taaf.support, "write_files", record)
    bm = _bm(tmp_path, n_games=1, periodic_save_interval_s=0.0, periodic_page=False)
    save = asyncio.create_task(bm._periodic_save_loop())
    while not saves:
        await asyncio.sleep(0.01)
    save.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save
    assert saves[0] == [tmp_path / "benchmark.json", tmp_path / "summary.txt"]


@dataclass
class _StopsItself(taaf.solver_examples.SolverFlipLR):
    """The flip solver, ending its own run a moment after it starts: what a
    solver's watchdog does when the engine goes quiet."""

    async def _run_games(self, games: list[Game]) -> None:
        assert self.stop_run is not None, "stamped for the run's duration"
        asyncio.get_running_loop().call_later(0.2, self.stop_run, "the engine went quiet")
        await super()._run_games(games)


async def test_a_solver_ends_its_own_run_like_the_deadline_and_the_reason_is_kept(tmp_path: Path):
    """``stop_run`` is the soft deadline on demand: the games still held land
    cancelled, run() returns normally, and the reason reaches the JSON and
    the summary. Outside a run the hook is gone and a late call is a no-op."""
    bm = _bm(tmp_path, n_games=4)
    bm.solver = _StopsItself(delay_s=0.5)
    await bm.run()
    assert bm.stop_reason == "the engine went quiet"
    assert any(r.state == "cancelled" for r in bm.game_runs)
    assert taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json").stop_reason == "the engine went quiet"
    assert "stopped:   the engine went quiet" in (tmp_path / "summary.txt").read_text()
    assert bm.solver is not None and bm.solver.stop_run is None
    bm.stop_early("too late")
    assert bm.stop_reason == "the engine went quiet"


class _StopsItselfTwiceMidCleanup(_StopsItself):
    """The stopped solver's cleanup awaits, and a second stop arrives inside
    that wait: a second cancellation would interrupt the cleanup and leave
    the game ``crashed`` rather than ``cancelled``."""

    async def _run_games(self, games: list[Game]) -> None:
        try:
            await super()._run_games(games)
        except asyncio.CancelledError:
            assert self.stop_run is not None
            self.stop_run("again")
            await asyncio.sleep(0.05)
            raise


async def test_a_second_stop_during_cleanup_is_a_no_op(tmp_path: Path):
    """``stop_run`` is idempotent for real, and so is the deadline behind
    it: one cancellation per run, whichever path asks first."""
    bm = _bm(tmp_path, n_games=4)
    bm.solver = _StopsItselfTwiceMidCleanup(delay_s=0.5)
    await bm.run()
    assert bm.stop_reason == "the engine went quiet", "the first reason stands"
    assert any(r.state == "cancelled" for r in bm.game_runs)
    assert not any(r.state == "crashed" for r in bm.game_runs)
